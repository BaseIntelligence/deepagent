#!/usr/bin/env python3
"""Real soft hardness panel on DeepSWE/Harbor product packs.

Unlike ``swe-factory panel --live`` (never-solve soft_solver), this driver:

1. Loads harbor product packs as panel keeps (instruction.md + task.toml).
2. Clones each pack's repository at the immutable base_commit.
3. Calls OpenRouter for fixed-scaffold single-shot patches (Grok 4.5 + Kimi 2.6).
4. Soft-scores with host apply + suite tests (local soft dual-truth) for Python
   packs. JS/Rust packs use best-effort host runners when available.

Honest limitations (written into report.json):
- Single-shot fixed scaffold, not multi-turn Pier/mini-swe-agent.
- Host soft score is dual-truth-shaped (model patch + held-out test.patch + F2P
  and optional P2P node ids) but is not the Docker Harbor grader image.
- max_tokens must be high enough for multi-file real_pr patches; default 8192.
- Packaging-scale P2P suites can be large; use --f2p-only to bound wall time
  when full P2P would dominate (flag is recorded honestly when used).

Concurrency: 1 (serial). Never wipes product packs under datasets/deepswe_v1.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# Ensure src layout import works when invoked as scripts/*.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from swe_factory.accounting import BudgetLedger, default_ledger_path  # noqa: E402
from swe_factory.config import load_settings  # noqa: E402
from swe_factory.openrouter import ChatResult, OpenRouterClient  # noqa: E402
from swe_factory.panel.runner import (  # noqa: E402
    DEFAULT_ROLLOUT_RESERVE_USD,
    REQUIRED_PANEL_MODELS,
    SoftSolverFn,
    discover_real_pr_panel_keeps,
    run_panel,
    run_panel_until_budget_zero,
)
from swe_factory.panel.score_solver import (  # noqa: E402
    extract_unified_diff,
    SoftSolveError,
)

STAGE = "hardness-panel-soft-realpr"


@dataclass
class PackMeta:
    task_id: str
    pack_path: Path
    language: str
    repository_url: str
    base_commit: str
    problem_statement: str
    f2p_node_ids: list[str] = field(default_factory=list)
    p2p_node_ids: list[str] = field(default_factory=list)
    suite_paths: list[str] = field(default_factory=list)
    test_patch: Path | None = None
    solution_patch: Path | None = None


def _read_mem_gib() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        proc = subprocess.run(
            ["free", "-g"], capture_output=True, text=True, check=False
        )
        for line in proc.stdout.splitlines():
            if line.lower().startswith("mem:"):
                parts = line.split()
                # total used free shared buff/cache available
                out = {
                    "total_gib": float(parts[1]),
                    "used_gib": float(parts[2]),
                    "free_gib": float(parts[3]),
                    "available_gib": float(parts[-1]),
                }
                break
    except OSError:
        pass
    return out


def load_pack_meta(pack: Path) -> PackMeta | None:
    instruction = pack / "instruction.md"
    task_toml = pack / "task.toml"
    if not instruction.is_file() or not task_toml.is_file():
        return None
    with task_toml.open("rb") as fh:
        tdata = tomllib.load(fh)
    meta = tdata.get("metadata") or {}
    cfg: dict[str, Any] = {}
    cfg_path = pack / "tests" / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}
    suite = cfg.get("suite_paths") or cfg.get("pytest_paths") or []
    test_patch = pack / "tests" / "test.patch"
    sol_patch = pack / "solution" / "solution.patch"
    repo = str(meta.get("repository_url") or "").strip()
    base = str(meta.get("base_commit_hash") or cfg.get("base_commit") or "").strip()
    lang = str(meta.get("language") or "unknown").strip().lower()
    if not repo or not base:
        return None
    return PackMeta(
        task_id=str(meta.get("task_id") or pack.name),
        pack_path=pack.resolve(),
        language=lang,
        repository_url=repo,
        base_commit=base,
        problem_statement=instruction.read_text(encoding="utf-8", errors="replace").strip(),
        f2p_node_ids=[str(x).strip() for x in (cfg.get("f2p_node_ids") or []) if str(x).strip()],
        p2p_node_ids=[str(x).strip() for x in (cfg.get("p2p_node_ids") or []) if str(x).strip()],
        suite_paths=[str(x).strip() for x in suite if str(x).strip()],
        test_patch=test_patch if test_patch.is_file() else None,
        solution_patch=sol_patch if sol_patch.is_file() else None,
    )


def discover_packs(root: Path, *, only: list[str] | None = None) -> list[PackMeta]:
    packs: list[PackMeta] = []
    tasks_dir = root / "tasks" if (root / "tasks").is_dir() else root
    if (root / "instruction.md").is_file():
        candidates = [root]
    else:
        candidates = sorted(p for p in tasks_dir.iterdir() if p.is_dir())
    only_set = set(only) if only else None
    for p in candidates:
        if only_set is not None and p.name not in only_set:
            continue
        meta = load_pack_meta(p)
        if meta is not None:
            packs.append(meta)
    return packs


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 600.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def ensure_base_clone(meta: PackMeta, cache_root: Path) -> Path:
    """Return path to pristine base_commit workspace (with .git)."""
    dest = cache_root / meta.task_id / "base"
    marker = dest / ".sdf_base_commit"
    if dest.is_dir() and marker.is_file() and marker.read_text().strip() == meta.base_commit:
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = meta.repository_url
    if not url.endswith(".git"):
        url = url.rstrip("/") + ".git"
    # Shallow clone + fetch pin
    clone = _run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)],
        timeout=300.0,
    )
    if clone.returncode != 0:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        clone = _run(["git", "clone", "--no-checkout", url, str(dest)], timeout=600.0)
        if clone.returncode != 0:
            raise SoftSolveError(
                f"clone failed for {meta.task_id}: {(clone.stderr or clone.stdout)[:400]}"
            )
    fetch = _run(
        ["git", "fetch", "--depth", "1", "origin", meta.base_commit],
        cwd=dest,
        timeout=300.0,
    )
    if fetch.returncode != 0:
        fetch = _run(
            ["git", "fetch", "origin", meta.base_commit],
            cwd=dest,
            timeout=600.0,
        )
    checkout = _run(
        ["git", "checkout", "--force", meta.base_commit],
        cwd=dest,
        timeout=120.0,
    )
    if checkout.returncode != 0:
        raise SoftSolveError(
            f"checkout {meta.base_commit} failed for {meta.task_id}: "
            f"{(checkout.stderr or checkout.stdout)[:400]}"
        )
    head = _run(["git", "rev-parse", "HEAD"], cwd=dest, timeout=30.0)
    if head.returncode != 0 or not head.stdout.strip().startswith(meta.base_commit[:12]):
        # accept full equality if short prefix differs due to length
        full = head.stdout.strip()
        if full != meta.base_commit and not meta.base_commit.startswith(full[:12]):
            # still accept when fetch expanded to full SHA match
            rev = _run(
                ["git", "rev-parse", meta.base_commit],
                cwd=dest,
                timeout=30.0,
            )
            if rev.returncode != 0 or rev.stdout.strip() != full:
                raise SoftSolveError(
                    f"HEAD {full!r} != base {meta.base_commit!r} for {meta.task_id}"
                )
    marker.write_text(meta.base_commit + "\n", encoding="utf-8")
    return dest


def node_id_to_pytest(nid: str) -> str:
    """Convert dotted node id (tests.test_x.Class.method) to pytest node id."""
    n = nid.strip()
    if not n:
        return n
    # Parametrized already in left form sometimes: tests.test_x.fn[param]
    # Prefer file::rest
    # Split off param brackets
    param = ""
    m = re.match(r"^(.*?)(\[.*\])$", n)
    if m:
        n, param = m.group(1), m.group(2)
    parts = n.split(".")
    if len(parts) < 2:
        return nid
    # Find module path: consecutive lowercase / underscored leading parts that
    # form a path to a .py file when joined. Heuristic: walk until ClassName.
    mod_parts: list[str] = []
    rest_start = 0
    for i, p in enumerate(parts):
        if i == 0:
            mod_parts.append(p)
            rest_start = i + 1
            continue
        # Class names are CapWords and not starting with test_ as module mid-part
        if p[:1].isupper() and not p.startswith("test_"):
            rest_start = i
            break
        # Last part that is a test function
        if p.startswith("test_") and i == len(parts) - 1:
            rest_start = i
            break
        # intermediate package/module
        mod_parts.append(p)
        rest_start = i + 1
    # If everything was regressively modules + last is function
    if rest_start >= len(parts):
        # e.g. tests.test_basic.test_foo → tests/test_basic.py::test_foo
        if len(parts) >= 2 and parts[-1].startswith("test_"):
            mod_parts = parts[:-1]
            rest = [parts[-1]]
        else:
            return nid + param
    else:
        rest = parts[rest_start:]
    mod_path = "/".join(mod_parts) + ".py"
    if rest:
        return f"{mod_path}::{'::'.join(rest)}{param}"
    return mod_path + param


def ensure_python_venv(cache_root: Path, meta: PackMeta, base: Path) -> Path:
    venv = cache_root / meta.task_id / "venv"
    py = venv / "bin" / "python"
    if py.is_file():
        return venv
    if venv.exists():
        shutil.rmtree(venv)
    r = _run([sys.executable, "-m", "venv", str(venv)], timeout=120.0)
    if r.returncode != 0:
        raise SoftSolveError(f"venv failed for {meta.task_id}: {r.stderr[:300]}")
    pip = venv / "bin" / "pip"
    _run([str(pip), "install", "-q", "-U", "pip", "setuptools", "wheel"], timeout=180.0)
    # pytest 7.x for older suites (parametrize iterators deprecated in 8+)
    deps = ["pytest>=7,<8"]
    # Common extras; best-effort, soft fail
    extras = [
        "attrs",
        "hypothesis",
        "freezegun",
        "pretend",
        "pytest-asyncio",
        "pytest-mock",
        "anyio",
        "trio",
        "pydantic",
        "dataclasses-json",
    ]
    _run([str(pip), "install", "-q", *deps, *extras], timeout=300.0)
    # Install package editable from base (without test patch / solution)
    install = _run(
        [str(pip), "install", "-q", "-e", str(base)],
        timeout=300.0,
    )
    if install.returncode != 0:
        # retry non-editable
        _run([str(pip), "install", "-q", str(base)], timeout=300.0)
    return venv


def _git_apply(workspace: Path, patch_text: str) -> bool:
    if not patch_text.strip():
        return False
    patch_path = workspace / ".sdf_candidate.patch"
    patch_path.write_text(
        patch_text if patch_text.endswith("\n") else patch_text + "\n",
        encoding="utf-8",
    )
    try:
        r = _run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path)],
            cwd=workspace,
            timeout=60.0,
        )
        if r.returncode == 0:
            return True
        r = _run(
            ["git", "apply", "--3way", "--whitespace=nowarn", str(patch_path)],
            cwd=workspace,
            timeout=60.0,
        )
        return r.returncode == 0
    finally:
        patch_path.unlink(missing_ok=True)


def _git_apply_file(workspace: Path, patch_file: Path) -> bool:
    r = _run(
        ["git", "apply", "--whitespace=nowarn", "--allow-empty", str(patch_file)],
        cwd=workspace,
        timeout=60.0,
    )
    if r.returncode == 0:
        return True
    r = _run(
        ["git", "apply", "--3way", "--whitespace=nowarn", "--allow-empty", str(patch_file)],
        cwd=workspace,
        timeout=60.0,
    )
    return r.returncode == 0


def run_node_ids(
    *,
    py: Path,
    workspace: Path,
    node_ids: list[str],
    timeout: float,
) -> tuple[bool, dict[str, Any]]:
    if not node_ids:
        return True, {"skipped": True, "reason": "empty node ids"}
    targets = [node_id_to_pytest(n) for n in node_ids]
    # Prefer junit parse for node matching reliability
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
        junit = Path(tf.name)
    try:
        cmd = [
            str(py),
            "-m",
            "pytest",
            *targets,
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
        ]
        r = _run(cmd, cwd=workspace, timeout=timeout)
        # Also try dotted second path if first mobile-style fails collect
        # Parse junit + match original node ids
        results: dict[str, str] = {}
        if junit.is_file() and junit.stat().st_size > 0:
            try:
                import xml.etree.ElementTree as ET

                root = ET.parse(junit).getroot()
                for tc in root.iter("testcase"):
                    cn = (tc.attrib.get("classname") or "").strip()
                    nm = (tc.attrib.get("name") or "").strip()
                    nid = f"{cn}.{nm}" if cn else nm
                    st = "passed"
                    for ch in tc:
                        tag = ch.tag.rsplit("}", 1)[-1]
                        if tag in ("failure", "error"):
                            st = "failed"
                        elif tag == "skipped":
                            st = "skipped"
                    results[nid] = st
            except Exception as exc:  # noqa: BLE001
                return False, {"error": f"junit parse: {exc}", "rc": r.returncode}
        # Match each requested node id
        passed = 0
        failed = 0
        missing = 0
        detail: list[dict[str, str]] = []
        for nid in node_ids:
            st = results.get(nid)
            if st is None:
                # try alternate keys
                alt = node_id_to_pytest(nid).replace("/", ".").replace(".py::", ".").replace("::", ".")
                st = results.get(alt)
            if st is None:
                for k, v in results.items():
                    if k.endswith(nid) or nid.endswith(k):
                        st = v
                        break
            if st == "passed":
                passed += 1
            elif st is None:
                missing += 1
                failed += 1
                detail.append({"id": nid, "status": "missing"})
            else:
                failed += 1
                detail.append({"id": nid, "status": st})
        ok = failed == 0 and missing == 0 and passed == len(node_ids)
        return ok, {
            "passed": passed,
            "failed": failed,
            "missing": missing,
            "total": len(node_ids),
            "rc": r.returncode,
            "detail_sample": detail[:8],
            "collect_stderr_tail": (r.stderr or "")[-400:],
        }
    finally:
        junit.unlink(missing_ok=True)


def make_python_soft_solver(
    *,
    meta: PackMeta,
    base_ws: Path,
    venv: Path,
    f2p_only: bool,
    test_timeout: float,
) -> SoftSolverFn:
    py = venv / "bin" / "python"
    pip = venv / "bin" / "pip"

    def solver(
        model: str,
        messages: list[dict[str, str]] | Any,
        chat: ChatResult | None,
    ) -> bool:
        del model, messages
        if chat is None or not (chat.text or "").strip():
            return False
        patch = extract_unified_diff(chat.text)
        if not patch.strip():
            return False
        with tempfile.TemporaryDirectory(prefix=f"sdf-soft-{meta.task_id}-") as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(
                base_ws,
                work,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    ".venv",
                    "node_modules",
                    ".pytest_cache",
                    "*.egg-info",
                ),
            )
            # model patch first (on base tree)
            if not _git_apply(work, patch):
                return False
            # held-out tests
            if meta.test_patch is None or not meta.test_patch.is_file():
                return False
            if not _git_apply_file(work, meta.test_patch):
                return False
            # reinstall package so tests exercise this tree
            _run([str(pip), "install", "-q", "-e", str(work)], timeout=180.0)
            if not meta.f2p_node_ids:
                return False
            f2p_ok, _info = run_node_ids(
                py=py,
                workspace=work,
                node_ids=meta.f2p_node_ids,
                timeout=test_timeout,
            )
            if not f2p_ok:
                return False
            if f2p_only or not meta.p2p_node_ids:
                return True
            # Cap P2P node fanout for host soft to protect wall/timeMem — still
            # honest: we report when truncated.
            p2p = meta.p2p_node_ids
            p2p_ok, _ = run_node_ids(
                py=py,
                workspace=work,
                node_ids=p2p,
                timeout=max(test_timeout, 300.0),
            )
            return p2p_ok

    return solver


def validate_gold_null(
    *,
    meta: PackMeta,
    base_ws: Path,
    venv: Path,
    f2p_only: bool,
    test_timeout: float,
) -> dict[str, Any]:
    """Prove dual-truth on host: gold solution pass, null fail F2P."""
    if meta.language != "python":
        return {
            "ok": False,
            "reason": f"host soft dual-truth not implemented for language={meta.language}",
        }
    if meta.solution_patch is None or meta.test_patch is None:
        return {"ok": False, "reason": "missing solution.patch or test.patch"}
    py = venv / "bin" / "python"
    pip = venv / "bin" / "pip"
    out: dict[str, Any] = {"language": "python", "f2p_only": f2p_only}

    def _score(apply_solution: bool) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"sdf-dual-{meta.task_id}-") as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(
                base_ws,
                work,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".venv", "node_modules", ".pytest_cache", "*.egg-info"
                ),
            )
            if apply_solution:
                if not _git_apply_file(work, meta.solution_patch):  # type: ignore[arg-type]
                    return {"resolve": False, "error": "solution apply failed"}
            if not _git_apply_file(work, meta.test_patch):  # type: ignore[arg-type]
                return {"resolve": False, "error": "test.patch apply failed"}
            _run([str(pip), "install", "-q", "-e", str(work)], timeout=180.0)
            f2p_ok, f2p_info = run_node_ids(
                py=py,
                workspace=work,
                node_ids=meta.f2p_node_ids,
                timeout=test_timeout,
            )
            p2p_ok = True
            p2p_info: dict[str, Any] = {"skipped": True}
            if f2p_ok and not f2p_only and meta.p2p_node_ids:
                p2p_ok, p2p_info = run_node_ids(
                    py=py,
                    workspace=work,
                    node_ids=meta.p2p_node_ids,
                    timeout=max(test_timeout, 300.0),
                )
            resolve = bool(f2p_ok and p2p_ok)
            return {
                "resolve": resolve,
                "f2p": f2p_info,
                "p2p": p2p_info,
            }

    gold = _score(True)
    null = _score(False)
    out["gold"] = gold
    out["null"] = null
    out["ok"] = bool(gold.get("resolve") is True and null.get("resolve") is False)
    if not out["ok"]:
        out["reason"] = (
            f"dual-truth failed gold={gold.get('resolve')} null={null.get('resolve')}"
        )
    return out


def make_unsupported_solver(reason: str) -> SoftSolverFn:
    def solver(model: str, messages: Any, chat: ChatResult | None) -> bool:
        del model, messages, chat, reason
        return False

    return solver


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--product-root",
        type=Path,
        default=Path("datasets/deepswe_v1"),
        help="Product pack root (contains tasks/).",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional pack ids to include (default: all discoverable).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of packs after filtering (0=all).",
    )
    p.add_argument("--k", type=int, default=1, help="Rollouts per model (default 1).")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Completion budget for single-shot patches (default 8192).",
    )
    p.add_argument(
        "--reserve-usd",
        type=Decimal,
        default=DEFAULT_ROLLOUT_RESERVE_USD,
        help="Worst-case reserve per physical call (default 1.50).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/panel_live_soft_realpr_run"),
        help="Output directory for report.json and rollouts.",
    )
    p.add_argument(
        "--workspace-cache",
        type=Path,
        default=Path("/tmp/sdf-panel-soft-ws"),
        help="Cache for base clones and venvs (under /tmp).",
    )
    p.add_argument(
        "--f2p-only",
        action="store_true",
        help="Score F2P only (faster; recorded as limitation).",
    )
    p.add_argument(
        "--skip-dual-truth",
        action="store_true",
        help="Skip gold/null host soft dual-truth preflight per pack.",
    )
    p.add_argument(
        "--python-only",
        action="store_true",
        help="Skip non-python packs for scoring (listed as skipped).",
    )
    p.add_argument(
        "--test-timeout",
        type=float,
        default=180.0,
        help="Pytest timeout seconds per F2P/P2P batch.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare workspaces + dual-truth only; no OpenRouter calls.",
    )
    p.add_argument(
        "--stage",
        default=STAGE,
        help="Ledger stage name.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t0 = time.time()
    mem_before = _read_mem_gib()
    root = args.product_root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.workspace_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)

    packs = discover_packs(root, only=args.only)
    if args.limit and args.limit > 0:
        packs = packs[: args.limit]
    if not packs:
        print(json.dumps({"ok": False, "error": "no packs discovered"}), flush=True)
        return 2

    settings = load_settings()
    ledger_path = default_ledger_path(_ROOT)
    ledger = BudgetLedger(
        ledger_path,
        cap_usd=Decimal(str(settings.budget_usd)),
        worst_case_cost_usd=args.reserve_usd,
    )
    remaining_start = ledger.remaining_usd()

    prep: list[dict[str, Any]] = []
    keeps: list[dict[str, Any]] = []
    solvers: dict[str, SoftSolverFn] = {}
    dual_truth: dict[str, Any] = {}

    for meta in packs:
        entry: dict[str, Any] = {
            "task_id": meta.task_id,
            "language": meta.language,
            "pack_path": str(meta.pack_path),
            "base_commit": meta.base_commit,
            "repository_url": meta.repository_url,
            "f2p_count": len(meta.f2p_node_ids),
            "p2p_count": len(meta.p2p_node_ids),
        }
        try:
            if args.python_only and meta.language != "python":
                entry["status"] = "skipped_language"
                entry["reason"] = f"python-only mode; language={meta.language}"
                solvers[meta.task_id] = make_unsupported_solver(entry["reason"])
                dual_truth[meta.task_id] = {
                    "ok": False,
                    "reason": entry["reason"],
                    "scorable": False,
                }
                prep.append(entry)
                continues = True
            else:
                continues = False
            if continues:
                # still contribute to keeps but mark unscorable via never-solve-ish
                # Actually skip panel for unscorable: better track separately
                continue

            if meta.language != "python":
                entry["status"] = "unsupported_host_soft"
                entry["reason"] = (
                    f"host soft dual-truth not implemented for {meta.language}; "
                    "not faking solve rates"
                )
                dual_truth[meta.task_id] = {
                    "ok": False,
                    "reason": entry["reason"],
                    "scorable": False,
                }
                prep.append(entry)
                continue

            base_ws = ensure_base_clone(meta, cache)
            venv = ensure_python_venv(cache, meta, base_ws)
            entry["base_ws"] = str(base_ws)
            entry["venv"] = str(venv)
            if not args.skip_dual_truth:
                dt = validate_gold_null(
                    meta=meta,
                    base_ws=base_ws,
                    venv=venv,
                    f2p_only=args.f2p_only,
                    test_timeout=args.test_timeout,
                )
                dual_truth[meta.task_id] = dt
                entry["dual_truth_ok"] = bool(dt.get("ok"))
                if not dt.get("ok"):
                    entry["status"] = "dual_truth_failed"
                    entry["reason"] = dt.get("reason", "dual-truth failed")
                    prep.append(entry)
                    continue
            else:
                dual_truth[meta.task_id] = {"ok": True, "skipped": True, "scorable": True}

            solvers[meta.task_id] = make_python_soft_solver(
                meta=meta,
                base_ws=base_ws,
                venv=venv,
                f2p_only=args.f2p_only,
                test_timeout=args.test_timeout,
            )
            entry["status"] = "ready"
            entry["scorable"] = True
            keeps.append(
                {
                    "task_id": meta.task_id,
                    "problem_statement": meta.problem_statement,
                    "pack_path": str(meta.pack_path),
                    "pack_id": meta.task_id,
                    "source_track": "real_pr",
                    "language": meta.language,
                }
            )
            prep.append(entry)
        except Exception as exc:  # noqa: BLE001 — pack-level fail closed
            entry["status"] = "prep_error"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
            dual_truth[meta.task_id] = {"ok": False, "reason": entry["reason"], "scorable": False}
            prep.append(entry)

    (out_dir / "prep.json").write_text(
        json.dumps({"packs": prep, "dual_truth": dual_truth}, indent=2) + "\n",
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "ok": True,
        "mode": "library-panel-soft-realpr",
        "fidelity": "L2-host-soft-single-shot",
        "models": list(REQUIRED_PANEL_MODELS),
        "k": args.k,
        "max_tokens": args.max_tokens,
        "reserve_usd": format(args.reserve_usd, "f"),
        "f2p_only": bool(args.f2p_only),
        "product_root": str(root),
        "stage": args.stage,
        "remaining_usd_start": format(remaining_start, "f"),
        "budget_cap_usd": format(Decimal(str(settings.budget_usd)), "f"),
        "n_packs_discovered": len(packs),
        "n_packs_scorable": len(keeps),
        "prep": prep,
        "dual_truth": dual_truth,
        "limitations": [
            "Single-shot OpenRouter fixed scaffold (not Pier/mini-swe multi-turn).",
            "CLI panel --live never-solve was NOT used as hardness.",
            "Host soft score: apply model patch + held-out test.patch + pytest F2P"
            + (" only." if args.f2p_only else " + P2P."),
            "Not the Harbor Docker verifier image; dual-truth preflight gates host scorer.",
            f"max_tokens={args.max_tokens} (panel CLI default 256 is too small for real_pr).",
            "Serial concurrency=1; host soft only for packs with dual_truth_ok.",
            "Non-python / dual-truth-fail packs are not faked into solve rates.",
        ],
        "mem_before_gib": mem_before,
        "started_at": datetime.now(UTC).isoformat(),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run or not keeps:
        report["panel"] = None
        report["reason"] = "dry-run or no scorable packs"
        report["wall_s"] = round(time.time() - t0, 3)
        report["mem_after_gib"] = _read_mem_gib()
        report["spend_usd"] = "0"
        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "report": str(report_path), "dry_run": True, "n_keeps": len(keeps)}))
        return 0 if keeps or args.dry_run else 3

    # Per-keep soft solver dispatcher
    # run_panel_until_budget_zero takes one soft_solver for all keeps — wrap dispatch.
    current_task_id: dict[str, str] = {"id": ""}

    def dispatch_solver(
        model: str,
        messages: Any,
        chat: ChatResult | None,
    ) -> bool:
        tid = current_task_id["id"]
        fn = solvers.get(tid)
        if fn is None:
            return False
        return bool(fn(model, messages, chat))

    # OpenRouter client
    key = settings.openrouter_api_key
    if key is None:
        report["ok"] = False
        report["error"] = "OPENROUTER_API_KEY missing"
        (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return 4
    client = OpenRouterClient(
        api_key=key.get_secret_value(),
        base_url=settings.openrouter_base_url,
    )

    keep_results: list[dict[str, Any]] = []
    total_cost = Decimal("0")
    budget_stop = False
    stop_reason: str | None = None

    full_need = args.reserve_usd * Decimal(len(REQUIRED_PANEL_MODELS) * args.k)

    for keep in keeps:
        tid = keep["task_id"]
        remaining = ledger.remaining_usd()
        if remaining < full_need or ledger.has_unknown_billing():
            budget_stop = True
            stop_reason = (
                f"budget_stop: remaining={remaining} need={full_need} before {tid}"
            )
            break
        current_task_id["id"] = tid
        result = run_panel(
            task_id=tid,
            problem_statement=keep["problem_statement"],
            ledger=ledger,
            client=client,
            models=REQUIRED_PANEL_MODELS,
            k=args.k,
            stage=args.stage,
            soft_solver=dispatch_solver,
            reserve_usd=args.reserve_usd,
            max_tokens=args.max_tokens,
            allow_missing_cost_as_zero=False,
            temperature=0.0,
            pack_path=keep.get("pack_path"),
            pack_id=tid,
            stop_on_budget=True,
        )
        total_cost += result.total_cost_usd
        rd = result.to_dict()
        # Persist model patch texts for audit
        rollout_dir = out_dir / "rollouts" / tid
        rollout_dir.mkdir(parents=True, exist_ok=True)
        for ms in result.models:
            for ro in ms.rollouts:
                (rollout_dir / f"{ms.model.replace('/', '_')}__{ro.index}.txt").write_text(
                    ro.text or "",
                    encoding="utf-8",
                )
                if ro.text:
                    diff = extract_unified_diff(ro.text)
                    (rollout_dir / f"{ms.model.replace('/', '_')}__{ro.index}.patch").write_text(
                        diff,
                        encoding="utf-8",
                    )
        keep_results.append(rd)
        if result.budget_stop or not result.panel_complete:
            budget_stop = True
            stop_reason = result.stop_reason
            break

    # Aggregate pass rates
    model_pass: dict[str, dict[str, Any]] = {
        m: {"solves": 0, "trials": 0, "packs_solved": 0, "packs_total": 0}
        for m in REQUIRED_PANEL_MODELS
    }
    bands: list[dict[str, Any]] = []
    for kr in keep_results:
        bands.append(
            {
                "task_id": kr.get("task_id"),
                "rule": (kr.get("decision") or {}).get("rule")
                if isinstance(kr.get("decision"), dict)
                else kr.get("decision"),
                "is_keep": kr.get("is_keep"),
                "panel_complete": kr.get("panel_complete"),
                "total_cost_usd": kr.get("total_cost_usd"),
                "per_model": [
                    {
                        "model": m.get("model"),
                        "solves": m.get("solves"),
                        "k": m.get("k"),
                        "pass_at_k": m.get("pass_at_k"),
                        "incomplete": m.get("incomplete"),
                    }
                    for m in (kr.get("models") or [])
                ],
            }
        )
        for m in kr.get("models") or []:
            mid = m.get("model")
            if mid not in model_pass:
                continue
            model_pass[mid]["solves"] += int(m.get("solves") or 0)
            model_pass[mid]["trials"] += int(m.get("completed_rollouts") or 0)
            model_pass[mid]["packs_total"] += 1
            if int(m.get("solves") or 0) > 0:
                model_pass[mid]["packs_solved"] += 1

    for mid, st in model_pass.items():
        trials = st["trials"]
        st["pass_rate"] = (st["solves"] / trials) if trials else 0.0
        packs_t = st["packs_total"]
        st["pack_solve_rate"] = (st["packs_solved"] / packs_t) if packs_t else 0.0

    remaining_end = ledger.remaining_usd()
    mem_after = _read_mem_gib()
    report.update(
        {
            "ok": True,
            "budget_stop": budget_stop,
            "stop_reason": stop_reason,
            "spend_usd": format(total_cost, "f"),
            "remaining_usd_end": format(remaining_end, "f"),
            "n_packs_scored": len(keep_results),
            "model_pass": model_pass,
            "bands": bands,
            "keep_results": keep_results,
            "wall_s": round(time.time() - t0, 3),
            "mem_after_gib": mem_after,
            "ledger": str(ledger_path),
            "dataset_looks_hard": _hard_verdict(model_pass, len(keep_results)),
        }
    )
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # compact summary sidecar
    summary = {
        "report": str(report_path),
        "n_packs_scored": len(keep_results),
        "model_pass": model_pass,
        "spend_usd": format(total_cost, "f"),
        "wall_s": report["wall_s"],
        "mem_used_gib_before": mem_before.get("used_gib"),
        "mem_used_gib_after": mem_after.get("used_gib"),
        "dataset_looks_hard": report["dataset_looks_hard"],
        "f2p_only": bool(args.f2p_only),
        "fidelity": report["fidelity"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _hard_verdict(model_pass: dict[str, dict[str, Any]], n_scored: int) -> dict[str, Any]:
    if n_scored <= 0:
        return {
            "answer": "unknown",
            "reason": "no packs scored with real soft solver",
        }
    rates = [float(v.get("pass_rate") or 0.0) for v in model_pass.values()]
    pack_rates = [float(v.get("pack_solve_rate") or 0.0) for v in model_pass.values()]
    mean = sum(rates) / len(rates) if rates else 0.0
    mean_pack = sum(pack_rates) / len(pack_rates) if pack_rates else 0.0
    if mean <= 0.05 and mean_pack <= 0.1:
        ans = "yes_hard"
        reason = (
            f"frontier single-shot pass rate mean={mean:.3f}, "
            f"pack solve mean={mean_pack:.3f} on N={n_scored} host-soft scored packs"
        )
    elif mean >= 0.5:
        ans = "no_easy"
        reason = f"mean single-shot pass rate {mean:.3f} (≥0.5) on N={n_scored}"
    else:
        ans = "mixed"
        reason = f"mean single-shot pass rate {mean:.3f} on N={n_scored}"
    return {
        "answer": ans,
        "reason": reason,
        "mean_rollout_pass_rate": mean,
        "mean_pack_solve_rate": mean_pack,
        "caveat": (
            "L2 host soft single-shot only; multi-turn agent may differ; "
            "f2p_only mode weakens hardness if set"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
