"""Host-side prefilter before expensive Docker dual-run / oracle cert.

Skip materials early (with ledger reason codes) when:

1. ``git apply --check`` of solution.patch or test.patch fails at base SHA.
2. Collector dry-run on held-out tests cannot collect (import/suite broken).
3. Heuristic dual-run potential is empty (no held-out test defs / paths).

The prefilter is intentionally lightweight vs full dual-run:
- Uses clone cache + worktree clone@SHA (no Docker image build).
- For python: applies held-out test.patch then ``pytest --collect-only``.
- Fail-open on missing cache/repo so offline unit paths still progress.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Stable ledger reason codes (reject_ledger / under_supply honesty).
REASON_PATCH_APPLY_FAIL = "prefilter_patch_apply_fail"
REASON_COLLECT_EMPTY = "prefilter_collect_empty"
REASON_COLLECT_ERROR = "prefilter_collect_error"
REASON_NO_TEST_SURFACE = "prefilter_no_test_surface"
REASON_EMPTY_F2P_HEURISTIC = "prefilter_empty_f2p_heuristic"
REASON_SKIP_NO_CACHE = "prefilter_skip_no_cache"
REASON_SKIP_INCOMPLETE = "prefilter_skip_incomplete_identity"
REASON_OK = "prefilter_ok"
REASON_SKIP_NON_PYTHON = "prefilter_skip_non_python_collect"

# Match regular sources OR unified-diff added lines (`+def test_...`).
_TEST_DEF_RE = re.compile(
    r"^[+\s]*(?:async\s+)?def\s+(test_\w+)\s*\(",
    re.MULTILINE,
)
_TEST_FN_GO_RE = re.compile(r"^[+\s]*func\s+(Test\w+)\s*\(", re.MULTILINE)
_TEST_FN_JS_RE = re.compile(
    r"(?:it|test|describe)\s*\(\s*['\"`]([^'\"`]+)['\"`]",
    re.MULTILINE,
)
_TEST_FN_RS_RE = re.compile(
    r"#\[(?:tokio::)?test(?:\([^\)]*\))?\]\s*(?:async\s+)?fn\s+(\w+)",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class PrefilterResult:
    """Outcome of host prefilter for one material."""

    ok: bool
    reason_code: str
    detail: str = ""
    collected: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "collected": self.collected,
            "meta": dict(self.meta),
        }


def has_held_out_test_surface(
    test_patch: str,
    test_files: Sequence[str] = (),
    *,
    language: str = "python",
) -> bool:
    """True when held-out tests look executable by a suite reporter."""
    body = test_patch or ""
    lang = (language or "python").strip().lower()
    if lang == "python" and _TEST_DEF_RE.search(body):
        return True
    if lang in {"go", "golang"} and _TEST_FN_GO_RE.search(body):
        return True
    if lang in {"javascript", "js", "typescript", "ts"} and _TEST_FN_JS_RE.search(body):
        return True
    if lang in {"rust", "rs"} and _TEST_FN_RS_RE.search(body):
        return True
    # Path heuristic: something that looks like a test file was held out.
    for tf in test_files or ():
        pl = str(tf).lower().replace("\\", "/")
        if any(
            x in pl
            for x in (
                "test_",
                "_test.",
                ".test.",
                ".spec.",
                "/tests/",
                "tests/",
                "__tests__/",
                "_test.go",
                "_test.rs",
            )
        ):
            return True
    return False


def check_patch_apply_at_base(
    repo: Path,
    *,
    base_commit: str,
    solution_patch: str,
    test_patch: str,
    git_bin: str = "git",
) -> tuple[bool, str]:
    """Return (ok, detail) after ``git apply --check`` of both patches at base.

    Mutates ``repo`` worktree checkout to base (hard reset + clean).
    """
    root = Path(repo)
    if not (root / ".git").exists():
        return True, REASON_SKIP_NO_CACHE
    pin = (base_commit or "").strip().lower()
    if len(pin) != 40:
        return True, REASON_SKIP_INCOMPLETE

    def _run(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    try:
        _run([git_bin, "checkout", "-f", pin], timeout=90)
        _run([git_bin, "reset", "--hard", pin], timeout=60)
        _run([git_bin, "clean", "-fdx"], timeout=60)
        with tempfile.TemporaryDirectory(prefix="sdf-prefilter-") as tmp:
            sol = Path(tmp) / "solution.patch"
            test_p = Path(tmp) / "test.patch"
            sol.write_text(
                solution_patch if solution_patch.endswith("\n") else solution_patch + "\n",
                encoding="utf-8",
            )
            test_p.write_text(
                test_patch if test_patch.endswith("\n") else test_patch + "\n",
                encoding="utf-8",
            )
            r_sol = _run(
                [git_bin, "apply", "--check", "--whitespace=nowarn", str(sol)],
                timeout=30,
            )
            if r_sol.returncode != 0:
                err = (r_sol.stderr or r_sol.stdout or "")[:240]
                return False, f"solution.patch apply-check failed: {err}"
            r_test = _run(
                [git_bin, "apply", "--check", "--whitespace=nowarn", str(test_p)],
                timeout=30,
            )
            if r_test.returncode != 0:
                err = (r_test.stderr or r_test.stdout or "")[:240]
                return False, f"test.patch apply-check failed: {err}"
        return True, "apply_ok"
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Fail-open: prefilter must not block when git/cache is broken.
        return True, f"skip_prefilter_error:{exc}"


def collector_dry_run_python(
    repo: Path,
    *,
    test_patch: str | None = None,
    test_files: Sequence[str] = (),
    python_exe: str | None = None,
    apply_test_patch: bool = True,
) -> tuple[int, str, list[str]]:
    """Apply held-out tests (optional) and ``pytest --collect-only``.

    Returns ``(count, detail, node_ids)``. Count 0 means empty collection
    (with detail reason). Raises only OS-level failures; otherwise returns.
    """
    import sys

    root = Path(repo)
    py = python_exe or sys.executable
    git = shutil.which("git") or "git"

    patch_text = (test_patch or "").strip()
    if apply_test_patch and patch_text and (root / ".git").exists():
        with tempfile.NamedTemporaryFile(
            "w", suffix=".patch", delete=False, encoding="utf-8"
        ) as handle:
            raw = test_patch or ""
            body = raw if raw.endswith("\n") else raw + "\n"
            handle.write(body)
            patch_path = handle.name
        try:
            apply = subprocess.run(
                [git, "apply", "--whitespace=nowarn", patch_path],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if apply.returncode != 0:
                return (
                    0,
                    f"test.patch apply for collect failed: "
                    f"{(apply.stderr or apply.stdout or '')[:200]}",
                    [],
                )
        finally:
            Path(patch_path).unlink(missing_ok=True)

    import os

    env = dict(os.environ)
    path_parts: list[str] = []
    src = root / "src"
    if src.is_dir():
        path_parts.append(str(src.resolve()))
    path_parts.append(str(root.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("CI", None)

    # Soft-fix boltons-style conf hooks so collect mirrors dual-run.
    try:
        from swe_factory.producers.harbor_labeling import _rewrite_legacy_pytest_conf_hooks

        _rewrite_legacy_pytest_conf_hooks(root)
    except Exception:  # noqa: BLE001
        pass

    targets: list[str] = []
    for rel in test_files or ():
        cand = root / rel
        if cand.exists():
            targets.append(str(cand))
    if not targets:
        tests_dir = root / "tests"
        targets = [str(tests_dir if tests_dir.is_dir() else root)]

    code = r"""
import json, sys
try:
    import pytest
except Exception as exc:  # pragma: no cover
    print(json.dumps({"count": 0, "error": f"pytest_import:{exc}", "nodes": []}))
    raise SystemExit(0)

class Collector:
    def __init__(self) -> None:
        self.nodes: list[str] = []
        self.errors: list[str] = []

    def pytest_collectreport(self, report) -> None:  # type: ignore[no-untyped-def]
        if report.failed:
            self.errors.append(str(getattr(report, "longrepr", report) or "collect_fail")[:200])

    def pytest_collection_modifyitems(self, items) -> None:  # type: ignore[no-untyped-def]
        for it in items:
            nodeid = getattr(it, "nodeid", None) or str(it)
            self.nodes.append(str(nodeid))

collector = Collector()
targets = sys.argv[1:]
rc = pytest.main(
    [
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=no",
        "--disable-warnings",
        "-o",
        "filterwarnings=default",
        *targets,
    ],
    plugins=[collector],
)
print(json.dumps({
    "count": len(collector.nodes),
    "nodes": collector.nodes[:50],
    "errors": collector.errors[:10],
    "rc": int(rc),
}))
"""
    try:
        proc = subprocess.run(
            [py, "-c", code, *targets],
            cwd=str(root),
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return 0, "collect-only timed out", []
    except OSError as exc:
        return 0, f"collect-only os error: {exc}", []

    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip().startswith("{")]
    if not lines:
        tail = ((proc.stdout or "") + (proc.stderr or ""))[-300:]
        return 0, f"collect-only produced no JSON (rc={proc.returncode}): {tail}", []
    try:
        payload = __import__("json").loads(lines[-1])
    except Exception as exc:  # noqa: BLE001
        return 0, f"collect-only JSON parse failed: {exc}", []
    nodes = [str(n) for n in (payload.get("nodes") or [])]
    count = int(payload.get("count") or len(nodes) or 0)
    errors = payload.get("errors") or []
    if count <= 0:
        detail = "collect-only empty"
        if errors:
            detail = f"collect-only empty; errors={errors[:3]}"
        elif payload.get("error"):
            detail = str(payload.get("error"))
        return 0, detail, nodes
    return count, "collect_ok", nodes


def base_fail_f2p_heuristic(
    repo: Path,
    *,
    test_files: Sequence[str] = (),
    solution_patch: str = "",
    python_exe: str | None = None,
    timeout_sec: int = 120,
) -> tuple[bool, str, list[str]]:
    """Return (ok, detail, failed_node_ids).

    Assumes ``test.patch`` is already applied to ``repo``. Runs held-out tests
    at base; when zero failures are observed, F2P dual-run will be empty
    (nothing fails@base). Fail-open on suite errors/timeouts.
    """

    from swe_factory.producers.harbor_labeling import run_python_suite

    del solution_patch  # reserved for future gold-side heuristics
    del python_exe
    try:
        outcome = run_python_suite(Path(repo), test_paths=list(test_files or ()))
    except Exception as exc:  # noqa: BLE001
        return True, f"skip_f2p_heuristic_suite_error:{exc}", []
    failed = list(outcome.failed) + list(outcome.errors)
    if failed:
        return True, f"base_failures={len(failed)}", failed
    if len(outcome.passed) <= 0:
        # Suite returned no pass and no fail — likely collection/runtime soft fail.
        return True, "skip_f2p_heuristic_no_outcomes", []
    return (
        False,
        f"held-out tests all pass at base (passed={len(outcome.passed)}); empty F2P expected",
        [],
    )


def prefilter_material(
    *,
    task_id: str,
    repository_url: str,
    base_commit: str,
    language: str,
    solution_patch: str,
    test_patch: str,
    test_files: Sequence[str] = (),
    clone_cache_root: Path | str | None = None,
    materialize_repo: Callable[..., Path] | None = None,
    require_python_collect: bool = True,
    run_collect: bool = True,
) -> PrefilterResult:
    """Full host prefilter for one real_pr material before Docker spend.

    Stages:
    1. Held-out test surface presence (fast reject).
    2. Patch apply-check at base (solution + test).
    3. Python collector dry-run for potential F2P (held-out tests collect).
    """
    del task_id  # reserved for ledger callers
    lang = (language or "python").strip().lower()

    if not has_held_out_test_surface(test_patch, test_files, language=lang):
        return PrefilterResult(
            ok=False,
            reason_code=REASON_NO_TEST_SURFACE,
            detail="no held-out test defs/paths for dual-run F2P potential",
        )

    url = (repository_url or "").strip()
    pin = (base_commit or "").strip().lower()
    if not url or len(pin) != 40:
        return PrefilterResult(
            ok=True,
            reason_code=REASON_SKIP_INCOMPLETE,
            detail="incomplete identity; fail-open skip prefilter",
        )

    # Resolve / materialize a worktree for apply + collect when possible.
    repo: Path | None = None
    tmp_holder: tempfile.TemporaryDirectory[str] | None = None
    try:
        if materialize_repo is not None:
            try:
                repo = Path(materialize_repo(url=url, base_commit=pin, language=lang))
            except Exception as exc:  # noqa: BLE001
                return PrefilterResult(
                    ok=True,
                    reason_code=REASON_SKIP_NO_CACHE,
                    detail=f"materialize failed fail-open: {exc}",
                )
        else:
            name = url.rstrip("/").split("/")[-1].removesuffix(".git")
            cache_root = (
                Path(clone_cache_root) if clone_cache_root else Path("datasets/_clone_cache")
            )
            cached = cache_root / name
            if (cached / ".git").is_dir():
                tmp_holder = tempfile.TemporaryDirectory(prefix="sdf-prefilter-wt-")
                dest = Path(tmp_holder.name) / name
                shutil.copytree(
                    cached,
                    dest,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(
                        ".venv",
                        "node_modules",
                        "__pycache__",
                        ".pytest_cache",
                        "*.pyc",
                    ),
                )
                repo = dest
            else:
                return PrefilterResult(
                    ok=True,
                    reason_code=REASON_SKIP_NO_CACHE,
                    detail=f"no clone cache for {name}; fail-open",
                    meta={"cache_root": str(cache_root)},
                )

        assert repo is not None
        ok_apply, apply_detail = check_patch_apply_at_base(
            repo,
            base_commit=pin,
            solution_patch=solution_patch or "",
            test_patch=test_patch or "",
        )
        if not ok_apply:
            return PrefilterResult(
                ok=False,
                reason_code=REASON_PATCH_APPLY_FAIL,
                detail=apply_detail,
            )

        if not run_collect:
            return PrefilterResult(
                ok=True,
                reason_code=REASON_OK,
                detail=apply_detail or "apply_ok",
            )

        if lang != "python":
            # Container dual-run owns non-python collect; host apply was enough.
            return PrefilterResult(
                ok=True,
                reason_code=REASON_SKIP_NON_PYTHON,
                detail=f"apply_ok; collect deferred for language={lang}",
            )

        if not require_python_collect:
            return PrefilterResult(ok=True, reason_code=REASON_OK, detail="apply_ok")

        count, cdetail, nodes = collector_dry_run_python(
            repo,
            test_patch=test_patch,
            test_files=test_files,
            apply_test_patch=True,
        )
        if count <= 0:
            return PrefilterResult(
                ok=False,
                reason_code=REASON_COLLECT_EMPTY
                if "empty" in cdetail.lower() or count == 0
                else REASON_COLLECT_ERROR,
                detail=cdetail,
                collected=0,
                meta={"sample_nodes": nodes[:5]},
            )
        # Lightweight empty-F2P heuristic: if held-out tests already pass at
        # base after applying only test.patch, dual-run will yield empty F2P
        # (fail@base ∩ pass@gold). Skip expensive Docker dual-run/oracle.
        f2p_ok, f2p_detail, base_failed = base_fail_f2p_heuristic(
            repo,
            test_files=test_files,
            solution_patch=solution_patch or "",
        )
        if not f2p_ok:
            return PrefilterResult(
                ok=False,
                reason_code=REASON_EMPTY_F2P_HEURISTIC,
                detail=f2p_detail,
                collected=count,
                meta={"sample_nodes": nodes[:5], "base_failed": base_failed[:5]},
            )
        return PrefilterResult(
            ok=True,
            reason_code=REASON_OK,
            detail=f"{cdetail}; {f2p_detail}",
            collected=count,
            meta={"sample_nodes": nodes[:5], "base_failed": base_failed[:5]},
        )
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()


def prefilter_real_pr_material(
    material: Any,
    *,
    clone_cache_root: Path | str | None = None,
    materialize_repo: Callable[..., Path] | None = None,
    run_collect: bool = True,
) -> PrefilterResult:
    """Adapter over objects with RealPrMaterial-like attributes."""
    return prefilter_material(
        task_id=str(getattr(material, "task_id", "") or ""),
        repository_url=str(getattr(material, "repository_url", "") or ""),
        base_commit=str(getattr(material, "base_commit", "") or ""),
        language=str(getattr(material, "language", "python") or "python"),
        solution_patch=str(getattr(material, "solution_patch", "") or ""),
        test_patch=str(getattr(material, "test_patch", "") or ""),
        test_files=tuple(getattr(material, "test_files", ()) or ()),
        clone_cache_root=clone_cache_root,
        materialize_repo=materialize_repo,
        run_collect=run_collect,
    )


def summarize_prefilter_ledger(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate reject reason codes from prefilter ledger rows."""
    counts: dict[str, int] = {}
    for row in rows:
        code = str(row.get("reason_code") or row.get("code") or "unknown")
        counts[code] = counts.get(code, 0) + 1
    return {
        "total": len(rows),
        "by_reason_code": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


__all__ = [
    "REASON_COLLECT_EMPTY",
    "REASON_COLLECT_ERROR",
    "REASON_EMPTY_F2P_HEURISTIC",
    "REASON_NO_TEST_SURFACE",
    "REASON_OK",
    "REASON_PATCH_APPLY_FAIL",
    "REASON_SKIP_INCOMPLETE",
    "REASON_SKIP_NO_CACHE",
    "REASON_SKIP_NON_PYTHON",
    "PrefilterResult",
    "base_fail_f2p_heuristic",
    "check_patch_apply_at_base",
    "collector_dry_run_python",
    "has_held_out_test_surface",
    "prefilter_material",
    "prefilter_real_pr_material",
    "summarize_prefilter_ledger",
]
