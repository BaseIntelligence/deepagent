"""Dual-run F2P/P2P node-id labeling for Harbor motors and DeepSWE packs.

Hand-authored ``base_f2p_node_ids`` / ``base_p2p_node_ids`` are advisory only.
Authoritative cohorts come from broken-vs-green suite outcomes:

- **F2P**: pass on green and fail on broken
- **P2P**: pass on green and pass on broken

Held-out tests are written into both trees before execution so dual-run evidence
includes the held-out contract (VAL-HARBOR-006 / VAL-LABEL-001..006).

Public helpers also:
- persist node ids into Harbor ``tests/config.json``
- reject flaky dual-run signature disagreement
- recompute labels deterministically from fixed suite summaries
- feed multi-lang suite reporters (Py/Go/TS/JS/Rust adapters)
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

HarborSuiteLang = Literal["python", "go", "typescript", "javascript", "rust"]
NodeIdSeq = Iterable[str]

# Stable reject codes mirrored from oracle.codes for label-stage flake policy.
LABEL_FLAKE_REJECT = "FLAKE_REJECT"
LABEL_G2_FLAKE = "G2_FLAKE"
LABEL_EMPTY_F2P = "G1_EMPTY_F2P"


class HarborLabelError(RuntimeError):
    """Raised when dual-run suite labeling cannot produce valid F2P/P2P cohorts."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: Sequence[str] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_codes: tuple[str, ...] = tuple(reason_codes or ())
        self.details: dict[str, Any] = dict(details or {})


@dataclass(frozen=True, slots=True)
class SuiteOutcome:
    """Normalized node-id outcomes for one tree (green or broken)."""

    language: str
    passed: tuple[str, ...]
    failed: tuple[str, ...]
    errors: tuple[str, ...] = ()
    returncode: int = 0
    raw_tail: str = ""

    @property
    def passed_set(self) -> set[str]:
        return set(self.passed)

    @property
    def failed_set(self) -> set[str]:
        return set(self.failed)

    def outcome_signature(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Stable signature for dual-run flake detection (sorted node id sets)."""
        return (
            tuple(sorted(self.passed)),
            tuple(sorted(self.failed)),
            tuple(sorted(self.errors)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "passed": list(self.passed),
            "failed": list(self.failed),
            "errors": list(self.errors),
            "returncode": self.returncode,
        }

    @classmethod
    def from_summary(
        cls,
        *,
        language: str,
        passed: NodeIdSeq,
        failed: NodeIdSeq,
        errors: NodeIdSeq = (),
        returncode: int = 0,
    ) -> SuiteOutcome:
        """Build outcome from locked fixture summaries (VAL-LABEL-005)."""
        return cls(
            language=str(language).strip().lower(),
            passed=tuple(dict.fromkeys(str(x) for x in passed if str(x).strip())),
            failed=tuple(dict.fromkeys(str(x) for x in failed if str(x).strip())),
            errors=tuple(dict.fromkeys(str(x) for x in errors if str(x).strip())),
            returncode=int(returncode),
        )


@dataclass(frozen=True, slots=True)
class DualRunLabels:
    """Authoritative node-id cohorts from dual-run evidence."""

    f2p_node_ids: tuple[str, ...]
    p2p_node_ids: tuple[str, ...]
    green: SuiteOutcome
    broken: SuiteOutcome
    all_nodes: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    accepted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "f2p_node_ids": list(self.f2p_node_ids),
            "p2p_node_ids": list(self.p2p_node_ids),
            "all_nodes": list(self.all_nodes),
            "green": self.green.to_dict(),
            "broken": self.broken.to_dict(),
            "notes": dict(self.notes),
            "reason_codes": list(self.reason_codes),
            "accepted": self.accepted,
        }

    def config_payload(
        self,
        *,
        base_commit: str,
        grade: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Shape used by Harbor ``tests/config.json`` (VAL-LABEL-003)."""
        if not self.f2p_node_ids:
            raise HarborLabelError(
                "cannot emit config.json with empty f2p_node_ids",
                reason_codes=(LABEL_EMPTY_F2P,),
            )
        # Disjoint cast: never write overlapping F2P and P2P
        f2p = list(self.f2p_node_ids)
        p2p = [n for n in self.p2p_node_ids if n not in set(f2p)]
        payload: dict[str, Any] = {
            "base_commit": str(base_commit).strip(),
            "f2p_node_ids": f2p,
            "p2p_node_ids": p2p,
        }
        if grade is not None:
            payload["grade"] = dict(grade)
        return payload


def pytest_nodeid_to_harbor(nodeid: str) -> str:
    """Convert pytest nodeid ``tests/test_x.py::test_y`` → ``tests.test_x.test_y``."""
    if "::" in nodeid:
        file_part, name = nodeid.split("::", 1)
        mod = (
            file_part[:-3].replace("/", ".").replace("\\", ".")
            if file_part.endswith(".py")
            else file_part.replace("/", ".").replace("\\", ".")
        )
        return f"{mod}.{name.replace('::', '.')}"
    return nodeid.replace("/", ".").replace("\\", ".")


def label_cohorts_from_outcomes(
    *,
    green_passed: NodeIdSeq,
    green_failed: NodeIdSeq,
    broken_passed: NodeIdSeq,
    broken_failed: NodeIdSeq,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Pure dual-run cohort math (no subprocess).

    F2P = fail(broken) ∩ pass(green)
    P2P = pass(broken) ∩ pass(green)
    Intersection F2P ∩ P2P is always empty by construction.
    """
    g_pass = set(green_passed)
    g_fail = set(green_failed)
    b_pass = set(broken_passed)
    b_fail = set(broken_failed)

    true_f2p = tuple(sorted(b_fail & g_pass))
    true_p2p = tuple(sorted(b_pass & g_pass))
    # Nodes that never appeared on green pass are not valid labels.
    _ = g_fail  # documented: green fails are rejected by callers
    # Defensive: if a node appears in both broken pass and fail lists, prefer F2P.
    if set(true_f2p) & set(true_p2p):
        p2p_clean = tuple(n for n in true_p2p if n not in set(true_f2p))
        return true_f2p, p2p_clean
    return true_f2p, true_p2p


def labels_from_suite_outcomes(
    green: SuiteOutcome,
    broken: SuiteOutcome,
    *,
    require_nonempty_f2p: bool = True,
    require_green_clean: bool = True,
    notes: Mapping[str, Any] | None = None,
) -> DualRunLabels:
    """Pure dual-run labels from fixed suite outcomes (VAL-LABEL-001/002/005).

    Deterministic for identical outcome sets: recomputing twice yields equal
    sorted F2P/P2P lists.
    """
    if require_green_clean and green.failed:
        raise HarborLabelError(
            f"green suite must be clean for labeling; failed={list(green.failed)}"
        )
    if green.errors:
        raise HarborLabelError(f"green suite collection errors: {list(green.errors)}")
    if not green.passed:
        raise HarborLabelError("green suite produced zero passing node ids")

    f2p, p2p = label_cohorts_from_outcomes(
        green_passed=green.passed_set,
        green_failed=green.failed_set,
        broken_passed=broken.passed_set,
        broken_failed=broken.failed_set,
    )
    if require_nonempty_f2p and not f2p:
        raise HarborLabelError(
            "dual-run produced empty F2P cohort "
            f"(broken_failed={list(broken.failed)} green_passed={list(green.passed)})",
            reason_codes=(LABEL_EMPTY_F2P,),
        )
    if set(f2p) & set(p2p):
        raise HarborLabelError(f"F2P/P2P overlap after labeling: {sorted(set(f2p) & set(p2p))}")
    assert_broken_matches_labels(
        broken=broken,
        f2p_node_ids=f2p,
        p2p_node_ids=p2p,
    )
    green_label_fail = sorted((set(f2p) | set(p2p)) - green.passed_set)
    if green_label_fail:
        raise HarborLabelError(f"labels must pass on green; missing/fail={green_label_fail}")

    all_nodes = tuple(
        sorted(green.passed_set | green.failed_set | broken.passed_set | broken.failed_set)
    )
    note_blob: dict[str, Any] = {
        "method": "dual_run_broken_vs_green",
        "f2p_count": len(f2p),
        "p2p_count": len(p2p),
    }
    if notes:
        note_blob.update(dict(notes))
    return DualRunLabels(
        f2p_node_ids=f2p,
        p2p_node_ids=p2p,
        green=green,
        broken=broken,
        all_nodes=all_nodes,
        notes=note_blob,
        accepted=True,
        reason_codes=(),
    )


def detect_dual_run_flake(
    runs: Sequence[SuiteOutcome],
    *,
    phase: str = "gold",
) -> tuple[bool, list[str], dict[str, Any]]:
    """Return (is_flake, reason_codes, details) for multi-run suite outcomes.

    Flaky dual disagreement (different signatures) must never cert-keep
    (VAL-ORACLE-005 / VAL-ORCD-007 / label-stage flake policy).
    """
    if len(runs) < 2:
        return False, [], {"phase": phase, "run_count": len(runs)}
    signatures = [r.outcome_signature() for r in runs]
    details: dict[str, Any] = {
        "phase": phase,
        "run_count": len(runs),
        "signatures": [list(map(list, sig)) for sig in signatures],
    }
    if len(set(signatures)) > 1:
        return True, [LABEL_G2_FLAKE, LABEL_FLAKE_REJECT], details
    return False, [], details


def assert_no_dual_run_flake(
    runs: Sequence[SuiteOutcome],
    *,
    phase: str = "gold",
) -> None:
    """Raise :class:`HarborLabelError` with FLAKE_REJECT if signatures disagree."""
    is_flake, codes, details = detect_dual_run_flake(runs, phase=phase)
    if is_flake:
        raise HarborLabelError(
            f"flake: dual-run {phase} outcome mismatch signatures={details.get('signatures')}",
            reason_codes=codes,
            details=details,
        )


def write_tests_config_json(
    path: Path | str,
    *,
    base_commit: str,
    f2p_node_ids: NodeIdSeq,
    p2p_node_ids: NodeIdSeq = (),
    grade: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist dual-run node ids into Harbor ``tests/config.json`` (VAL-LABEL-003).

    Validates:
    - ``|f2p| ≥ 1``
    - no empty-string node ids
    - F2P ∩ P2P empty
    """
    f2p = [str(n).strip() for n in f2p_node_ids if str(n).strip()]
    p2p = [str(n).strip() for n in p2p_node_ids if str(n).strip()]
    if not f2p:
        raise HarborLabelError(
            "tests/config.json requires non-empty f2p_node_ids",
            reason_codes=(LABEL_EMPTY_F2P,),
        )
    if any(not n for n in f2p):
        raise HarborLabelError("f2p_node_ids must not contain empty strings")
    overlap = sorted(set(f2p) & set(p2p))
    if overlap:
        raise HarborLabelError(f"F2P/P2P overlap before config write: {overlap}")
    # P2P may be empty only when every green-pass fails broken (caller choice).
    payload: dict[str, Any] = {
        "base_commit": str(base_commit).strip(),
        "f2p_node_ids": list(f2p),
        "p2p_node_ids": list(p2p),
    }
    if not payload["base_commit"]:
        raise HarborLabelError("tests/config.json requires base_commit")
    if grade is not None:
        payload["grade"] = dict(grade)
    if extra:
        for key, value in extra.items():
            if key in {"base_commit", "f2p_node_ids", "p2p_node_ids"}:
                continue
            payload[key] = value
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


def labels_to_tests_config(
    labels: DualRunLabels,
    *,
    base_commit: str,
    grade: Mapping[str, Any] | None = None,
    dest: Path | str | None = None,
) -> dict[str, Any]:
    """Convert dual-run labels to config.json payload; optional write to ``dest``."""
    payload = labels.config_payload(base_commit=base_commit, grade=grade)
    if dest is not None:
        write_tests_config_json(
            dest,
            base_commit=base_commit,
            f2p_node_ids=payload["f2p_node_ids"],
            p2p_node_ids=payload["p2p_node_ids"],
            grade=grade,
        )
    return payload


def assert_held_out_verifier_only(
    *,
    agent_context: Path | str,
    test_patch_path: Path | str | None = None,
    held_out_relative_paths: Sequence[str] = (),
) -> list[str]:
    """Ensure held-out tests / test.patch are absent from agent context (VAL-LABEL-004).

    Returns list of isolation hit paths (empty = clean).
    """
    root = Path(agent_context)
    hits: list[str] = []
    if not root.is_dir():
        raise HarborLabelError(f"agent context is not a directory: {root}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        name = path.name
        if name == "test.patch":
            hits.append(rel)
            continue
        if "solution/" in rel or rel.startswith("solution"):
            hits.append(rel)
            continue
        for held in held_out_relative_paths:
            held_norm = held.lstrip("./")
            if rel == held_norm or rel.endswith("/" + held_norm):
                hits.append(rel)
    # Document verifier path expectation when provided
    if test_patch_path is not None:
        tp = Path(test_patch_path)
        if not tp.is_file() or not tp.read_text(encoding="utf-8", errors="replace").strip():
            raise HarborLabelError(
                f"verifier test.patch missing or empty: {tp}",
                details={"test_patch": str(tp)},
            )
    return hits


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_held_out(repo: Path, relative_path: str, content: str) -> Path:
    target = repo / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    body = content if content.endswith("\n") else content + "\n"
    target.write_text(body, encoding="utf-8")
    return target


_LEGACY_IGNORE_COLLECT_RE = re.compile(
    r"def\s+pytest_ignore_collect\s*\(\s*path\s*,\s*config\s*\)\s*:",
    re.MULTILINE,
)
_LEGACY_IGNORE_COLLECT_PATH_FIRST = re.compile(
    r"def\s+pytest_ignore_collect\s*\(\s*path\s*:\s*[^,\)]+\s*,\s*config",
    re.MULTILINE,
)


def _soft_relax_pytest_ini(repo: Path) -> None:
    """Neutralize brittle pytest strict-config/markers that produce green=0.

    Live repos often pin older ``--strict-markers`` without registering
    ``anyio`` / ``trio`` markers (httpcore family). Soft-rewrite worktree only.
    Keep TOML/ini syntax valid (no dangling commas in emptied list entries).
    """
    markers_needed = (
        "anyio: mark anyio backend tests",
        "trio: mark trio backend tests",
        "network: mark tests that need network",
    )

    def _strip_strict_flags(text: str) -> str:
        # Remove as list items first (preserve valid commas for neighbors).
        new = re.sub(
            r""",\s*["']--strict-(?:markers|config)["']""",
            "",
            text,
        )
        new = re.sub(
            r"""["']--strict-(?:markers|config)["']\s*,?\s*""",
            "",
            new,
        )
        new = new.replace("--strict-markers", "").replace("--strict-config", "")
        # Clean empty addopts remnants like addopts = []
        new = re.sub(r"addopts\s*=\s*\[\s*\]", "addopts = []", new)
        return new

    for name in ("pytest.ini", "tox.ini", "setup.cfg"):
        path = repo / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        new = _strip_strict_flags(text)
        if "[pytest]" in new and re.search(r"(?m)^markers\s*=", new) is None:
            new = new.replace(
                "[pytest]",
                "[pytest]\nmarkers =\n    " + "\n    ".join(markers_needed),
                1,
            )
        if new != text:
            with contextlib.suppress(OSError):
                path.write_text(new, encoding="utf-8")
    pyproject = repo / "pyproject.toml"
    if not pyproject.is_file():
        return
    try:
        text = pyproject.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if "tool.pytest" not in text and "pytest.ini_options" not in text:
        return
    new = _strip_strict_flags(text)
    if "[tool.pytest.ini_options]" in new:
        section = new.split("[tool.pytest.ini_options]", 1)[-1][:800]
        if "markers" not in section:
            new = new.replace(
                "[tool.pytest.ini_options]",
                "[tool.pytest.ini_options]\nmarkers = [\n"
                + ",\n".join(f'  "{m}"' for m in markers_needed)
                + "\n]\n",
                1,
            )
    if new != text:
        with contextlib.suppress(OSError):
            pyproject.write_text(new, encoding="utf-8")


def _rewrite_legacy_pytest_conf_hooks(repo: Path) -> int:
    """Rewrite pre-pytest-8 ``pytest_ignore_collect(path, config)`` → collection_path.

    Product dual-run uses modern pytest from the host venv. Legacy packages
    (e.g. boltons conf hooks) hard-fail collection under pytest 8+ with
    PluginValidationError when the stub still declares the dropped ``path``
    argument. Soft-rewrite conf.py/conftest.py in-place on the dual-run
    worktree only (never recorded as product gold).
    """
    rewritten = 0
    candidates: list[Path] = []
    for path in repo.rglob("conftest.py"):
        rel = str(path.relative_to(repo)).replace("\\", "/")
        if any(part in rel for part in (".git/", "node_modules/", "venv/", ".tox/")):
            continue
        candidates.append(path)
    conf_py = repo / "conf.py"
    if conf_py.is_file():
        candidates.append(conf_py)

    def _compat_hook(body: str) -> str:
        """body of pytest_ignore_collect using either py.path or pathlib."""
        # Boltons uses ``path.basename`` (attr, not call). Rewrite attr & call.
        basename_expr = (
            "(collection_path.basename() "
            "if callable(getattr(collection_path, 'basename', None)) "
            "else getattr(collection_path, 'name', ''))"
        )
        body2 = body.replace("path.basename()", basename_expr)
        body2 = body.replace("path.basename", basename_expr)
        body2 = body2.replace("path.strpath", "str(collection_path)")
        # Remaining bare `path` token → collection_path (word-boundary).
        body2 = re.sub(r"(?<![\w.])path(?![\w])", "collection_path", body2)
        body2 = body2.replace("collection_collection_path", "collection_path")
        return body2

    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "pytest_ignore_collect" not in text or "collection_path" in text:
            # Already modern or absent.
            if "def pytest_ignore_collect(collection_path" in text:
                continue
            if "pytest_ignore_collect" not in text:
                continue
        new = text
        # Function-level rewrite: parse signature + body line-by-line-ish via regex.
        pattern = re.compile(
            r"def\s+pytest_ignore_collect\s*\(\s*path(?:\s*:\s*[^,\)]+)?\s*,\s*config"
            r"(?:\s*:\s*[^)]+)?\s*\)\s*:\s*\n"
            r"((?:[ \t]+.+\n)*)",
            re.MULTILINE,
        )

        def _sub(m: re.Match[str]) -> str:
            body = m.group(1)
            return "def pytest_ignore_collect(collection_path, config):\n" + _compat_hook(body)

        new2, n = pattern.subn(_sub, new)
        if n == 0 and _LEGACY_IGNORE_COLLECT_RE.search(new):
            # Signature only when body pattern stubborn.
            new2 = _LEGACY_IGNORE_COLLECT_RE.sub(
                "def pytest_ignore_collect(collection_path, config):", new
            )
            new2 = _compat_hook(new2)
            # Avoid rewriting imports / unrelated path strings poorly eventually OK
            n = 1
        if n and new2 != text:
            try:
                path.write_text(new2, encoding="utf-8")
                rewritten += 1
            except OSError:
                continue
    return rewritten


def run_python_suite(
    repo: Path,
    *,
    test_paths: Sequence[str] | None = None,
) -> SuiteOutcome:
    """Execute pytest under ``repo/tests`` (or *test_paths*) and collect node ids.

    ``test_paths`` lets Real-PR dual-run scope the host suite to held-out
    test files so ambient unrelated base-suite failures (not part of this PR)
    do not block green labeling.
    """
    env = dict(os.environ)
    # Prefer src/ layout (click/itsdangerous/markupsafe) then flat (zipp) packages.
    # Do NOT inherit ambient PYTHONPATH (factory editable install / agent worktrees
    # can poison pytest collection on product dual-run packages).
    path_parts: list[str] = []
    src = repo / "src"
    if src.is_dir():
        path_parts.append(str(src.resolve()))
    path_parts.append(str(repo.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Disable auto-loading of site / entry-point pytest plugins from factory venv
    # so only explicit plugins (our collector + repo conftest) participate.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    # Soft-fix brittleness from ambient farm CI.
    env.pop("CI", None)

    # Legacy conftest fixups (pytest 8 dropped hookimpl arg `path` on
    # pytest_ignore_collect). Older packages (e.g. boltons) still ship the
    # pre-8 signature and hard-fail collection under modern pytest.
    _rewrite_legacy_pytest_conf_hooks(repo)
    # Soft-relax strict-markers/strict-config from older package pytest.ini.
    _soft_relax_pytest_ini(repo)

    # Resolve suite targets (held-out files when provided and present).
    # Always use absolute paths — relative paths are parsed by pytest as
    # nodeid expressions and can explode into whole-suite selection (then hang
    # in interactive termui modules if -k filters are imperfect).
    targets: list[str] = []
    for rel in test_paths or ():
        cand = (repo / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if cand.exists() and (cand.is_file() or cand.is_dir()):
            # Ignore non-test residue (coveragerc etc.) that inventory may list.
            name = cand.name.lower()
            if cand.is_file() and not (
                name.startswith("test")
                or name.endswith((".py", ".js", ".ts", ".go", ".rs"))
                or "test" in cand.as_posix().lower()
            ):
                continue
            targets.append(str(cand))
    if not targets:
        tests_dir = repo / "tests"
        targets = [str((tests_dir if tests_dir.is_dir() else repo).resolve())]
    targets = [t for t in targets if Path(t).exists()]
    if not targets:
        targets = [str((repo / "tests" if (repo / "tests").is_dir() else repo).resolve())]
    # Drop interactive/hang-prone test modules (click termui prompts etc.).
    hang_needles = (
        "termui",
        "prompt",
        "cli_runner_input",
        "test_interactive",
        "test_progress",
        # click stream lifecycle exercises pager/subprocess I/O that can hang
        # headless dual-run collectors for tens of minutes.
        "stream_lifecycle",
        "test_pager",
        "test_editor",
        "test_open_url",
    )
    filtered: list[str] = []
    for t in targets:
        tl = t.replace("\\", "/").lower()
        if any(n in tl for n in hang_needles):
            continue
        filtered.append(t)
    if filtered:
        targets = filtered
    # Isolate pytest: ignore hang packages via -p no: and short timeouts.
    env["CLICK_DISABLE"] = "1"  # noop for most pkgs; harmless
    # Isolated interpreter so ambient pytest plugins / chdir cannot poison collection.
    code = r"""
import json, sys
from pathlib import Path
import pytest

targets = sys.argv[1:]

class Collector:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.errors: list[str] = []

    def pytest_configure(self, config) -> None:  # type: ignore[no-untyped-def]
        for mark in ("anyio", "trio", "network"):
            with contextlib.suppress(Exception):
                config.addinivalue_line(
                    "markers", f"{mark}: soft-registered by sdf dual-run"
                )

    def pytest_runtest_logreport(self, report) -> None:  # type: ignore[no-untyped-def]
        if report.when == "setup" and report.failed:
            self.errors.append(report.nodeid)
            return
        if report.when != "call":
            return
        nodeid = report.nodeid
        if "::" in nodeid:
            file_part, name = nodeid.split("::", 1)
            mod = (
                file_part[:-3].replace("/", ".").replace("\\", ".")
                if file_part.endswith(".py")
                else file_part.replace("/", ".").replace("\\", ".")
            )
            node = f"{mod}.{name.replace('::', '.')}"
        else:
            node = nodeid
        if report.passed:
            self.passed.append(node)
        elif report.failed:
            self.failed.append(node)

import contextlib
collector = Collector()
rc = pytest.main(
    [
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=no",
        "--disable-warnings",
        # Override ambient / project filterwarnings=error that escalate
        # PytestRemovedIn9/10Warning into hard collection failures on older PRs.
        "-o",
        "filterwarnings=default",
        "-o",
        "addopts=",
        # Avoid blocking interactive termui/prompt tests if still collected.
        "-k",
        "not termui and not prompt and not test_progressbar",
        *targets,
    ],
    plugins=[collector],
)
print(
    json.dumps(
        {
            "rc": int(rc),
            "passed": collector.passed,
            "failed": collector.failed,
            "errors": collector.errors,
        }
    )
)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, *targets],
            cwd=str(repo),
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=240,
        )
    except subprocess.TimeoutExpired as exc:
        raise HarborLabelError(
            f"python suite timed out after 240s (targets={targets[:5]!r})"
        ) from exc
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise HarborLabelError(
            "python suite produced no JSON outcome: "
            f"rc={proc.returncode} stderr={proc.stderr[-500:]!r}"
        )
    data = json.loads(lines[-1])
    return SuiteOutcome(
        language="python",
        passed=tuple(dict.fromkeys(data.get("passed") or [])),
        failed=tuple(dict.fromkeys(data.get("failed") or [])),
        errors=tuple(dict.fromkeys(data.get("errors") or [])),
        returncode=int(data.get("rc", proc.returncode)),
        raw_tail=(proc.stdout + proc.stderr)[-1500:],
    )


def run_go_suite(repo: Path) -> SuiteOutcome:
    """Execute ``go test ./... -count=1 -v`` and collect test function names."""
    if shutil.which("go") is None:
        raise HarborLabelError("go binary not available for dual-run labeling")
    proc = subprocess.run(
        ["go", "test", "./...", "-count=1", "-v"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + "\n" + proc.stderr
    passed = tuple(dict.fromkeys(re.findall(r"--- PASS: (\w+)", out)))
    failed = tuple(dict.fromkeys(re.findall(r"--- FAIL: (\w+)", out)))
    return SuiteOutcome(
        language="go",
        passed=passed,
        failed=failed,
        errors=(),
        returncode=int(proc.returncode),
        raw_tail=out[-1500:],
    )


def ensure_local_node_bin_on_path(
    repo: Path,
    env: Mapping[str, str] | dict[str, str] | None = None,
) -> dict[str, str]:
    """Prepend ``repo/node_modules/.bin`` to PATH for dual-run JS workspaces.

    Local bins (ava, xo, jest, tape) must be callable after npm install when
    verkettung scripts invoke bare binaries. Returns a new env mapping; does
    not mutate the caller's os.environ unless passed through.
    """
    base = dict(env if env is not None else os.environ)
    bin_dir = Path(repo) / "node_modules" / ".bin"
    if bin_dir.is_dir():
        cur = base.get("PATH", "") or ""
        prefix = str(bin_dir.resolve())
        parts = [p for p in cur.split(os.pathsep) if p]
        if prefix not in parts:
            base["PATH"] = prefix + (os.pathsep + cur if cur else "")
    return base


def detect_js_test_framework(repo: Path) -> dict[str, str]:
    """Detect jest vs ava vs tape (or generic npm test) from package.json.

    Returns keys: framework, command (shell fragment for execution meta).
    """
    pkg_path = Path(repo) / "package.json"
    framework = "npm"
    command = "npm test"
    scripts: dict[str, str] = {}
    deps: dict[str, str] = {}
    if pkg_path.is_file():
        try:
            data = json.loads(pkg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            raw_scripts = data.get("scripts") or {}
            if isinstance(raw_scripts, dict):
                scripts = {str(k): str(v) for k, v in raw_scripts.items()}
            merged = {}
            for key in ("dependencies", "devDependencies", "peerDependencies"):
                blob = data.get(key) or {}
                if isinstance(blob, dict):
                    merged.update({str(k).lower(): str(v) for k, v in blob.items()})
            deps = merged
    test_script = (scripts.get("test") or "").lower()
    tests_only = (scripts.get("tests-only") or "").lower()
    combo = f"{test_script} {tests_only}"
    has = set(deps)
    nm = Path(repo) / "node_modules"

    def _has_mod(name: str) -> bool:
        return name in has or (nm / name).exists() or (nm / ".bin" / name).exists()

    # Prefer more specific runners when package declares them.
    if _has_mod("ava") or re.search(r"(^|[\s&|;])ava([\s&|;]|$)", combo):
        framework = "ava"
        # Skip lint chains (xo && ava) by invoking ava directly for dual-run node IDs.
        command = "npx --no-install ava"
    elif _has_mod("tape") or "tape" in combo or re.search(r"(^|[\s&|;])nyc\s+tape", combo):
        framework = "tape"
        # Prefer package tests-only (nyc tape ...) over full pretest/lintcompilation.
        if scripts.get("tests-only"):
            command = "npm run tests-only --silent"
        else:
            command = "npx --no-install tape 'test/**/*.js'"
    elif _has_mod("jest") or "jest" in combo:
        framework = "jest"
        command = "npm test -- --json --runInBand"
    elif _has_mod("mocha") or "mocha" in combo:
        framework = "mocha"
        command = "npm test"
    else:
        framework = "npm"
        command = "npm test"
    return {"framework": framework, "command": command}


def _ensure_js_node_modules(repo: Path, *, require_jest: bool = False) -> None:
    """npm install deps when node_modules missing (jest optional for live packs)."""
    nm = Path(repo) / "node_modules"
    if nm.is_dir() and any(nm.iterdir()):
        # Already present (possibly without jest — fine for ava/tape packs).
        if require_jest and not (nm / "jest").exists() and not (nm / ".bin" / "jest").exists():
            pass  # fall through to install
        else:
            return
    if shutil.which("npm") is None:
        raise HarborLabelError("npm not available for dual-run JS/TS labeling")
    # Prefer reusing a warmed fixture deps tree under the package if present (jest motors).
    cache = _package_root() / "fixtures" / "harbor_motors" / ".ts_deps"
    cached_nm = cache / "node_modules"
    if cached_nm.is_dir() and (cached_nm / "jest").exists() and require_jest:
        dest = repo / "node_modules"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cached_nm, dest, symlinks=True)
        return
    proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    nm = Path(repo) / "node_modules"
    if proc.returncode != 0:
        raise HarborLabelError(
            "npm install for dual-run JS/TS deps failed: "
            f"rc={proc.returncode} out={proc.stdout[-400:]!r} err={proc.stderr[-400:]!r}"
        )
    # Packages with no deps may leave node_modules absent; create placeholder so
    # PATH helpers and dual-run copy trees still have a stable layout.
    if not nm.is_dir():
        nm.mkdir(parents=True, exist_ok=True)
    if require_jest and not (nm / "jest").exists() and not (nm / ".bin" / "jest").exists():
        # Motor path only; live ava/tape packs should not require jest.
        raise HarborLabelError(
            "npm install for TS motor deps failed: jest missing after install: "
            f"out={proc.stdout[-400:]!r} err={proc.stderr[-400:]!r}"
        )
    # Warm shared cache for subsequent dual-runs (best-effort, jest motors only).
    if (repo / "node_modules" / "jest").exists():
        try:
            cache.mkdir(parents=True, exist_ok=True)
            if cached_nm.exists():
                shutil.rmtree(cached_nm)
            shutil.copytree(repo / "node_modules", cached_nm, symlinks=True)
        except OSError:
            pass


def _ensure_ts_node_modules(repo: Path) -> None:
    """Backward-compatible name: install JS/TS deps (jest required only for motors)."""
    det = detect_js_test_framework(repo)
    # Fixtures/motors always use jest; live packs may not have package.json yet.
    require_jest = det["framework"] == "jest" or not (repo / "package.json").is_file()
    if det["framework"] in {"ava", "tape", "mocha", "npm"} and not require_jest:
        _ensure_js_node_modules(repo, require_jest=False)
    else:
        _ensure_js_node_modules(repo, require_jest=True)


def _parse_js_outcome_text(
    text: str,
    *,
    language: str,
    returncode: int,
    json_path: Path | None = None,
) -> SuiteOutcome:
    """Parse jest JSON file + stdout/stderr using suite_reporters multi-framework rules."""
    passed: list[str] = []
    failed: list[str] = []
    if json_path is not None and json_path.is_file() and json_path.stat().st_size > 0:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for tr in data.get("testResults") or []:
                for assertion in tr.get("assertionResults") or []:
                    name = assertion.get("title") or assertion.get("fullName")
                    if not name:
                        continue
                    status = assertion.get("status")
                    if status == "passed":
                        passed.append(str(name))
                    elif status == "failed":
                        failed.append(str(name))
        except json.JSONDecodeError:
            pass
    if not passed and not failed:
        # Defer to suite_reporters multi-framework parse_log (ava/tape/jest).
        from swe_factory.producers.suite_reporters import parse_with_reporter

        parsed = parse_with_reporter(language, text, returncode=returncode)
        return parsed
    return SuiteOutcome(
        language=language,
        passed=tuple(dict.fromkeys(passed)),
        failed=tuple(dict.fromkeys(failed)),
        errors=(),
        returncode=int(returncode),
        raw_tail=text[-1500:],
    )


def _run_js_framework_suite(repo: Path, *, language: str) -> SuiteOutcome:
    """Run the detected JS/TS framework with local node_modules/.bin on PATH."""
    if shutil.which("node") is None:
        raise HarborLabelError("node binary not available for dual-run labeling")
    rep = Path(repo)
    det = detect_js_test_framework(rep)
    require_jest = det["framework"] == "jest"
    _ensure_js_node_modules(rep, require_jest=require_jest)
    env = ensure_local_node_bin_on_path(rep)
    framework = det["framework"]

    if framework == "jest":
        with tempfile.TemporaryDirectory(prefix="sdf-jest-") as td:
            out_json = Path(td) / "jest.json"
            proc = subprocess.run(
                [
                    "npm",
                    "test",
                    "--",
                    "--json",
                    f"--outputFile={out_json}",
                    "--runInBand",
                ],
                cwd=str(rep),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            out = proc.stdout + "\n" + proc.stderr
            return _parse_js_outcome_text(
                out, language=language, returncode=int(proc.returncode), json_path=out_json
            )

    if framework == "ava":
        # Use local ava bin; avoids xo/tsd lint recreation failures on dual-run trees.
        ava_bin = rep / "node_modules" / ".bin" / "ava"
        cmd = [str(ava_bin)] if ava_bin.is_file() else ["npx", "--no-install", "ava"]
        proc = subprocess.run(
            cmd,
            cwd=str(rep),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        out = proc.stdout + "\n" + proc.stderr
        return _parse_js_outcome_text(out, language=language, returncode=int(proc.returncode))

    if framework == "tape":
        # Prefer package tests-only (skips readme lint pretest).
        pkg = rep / "package.json"
        scripts: dict[str, str] = {}
        if pkg.is_file():
            try:
                data = json.loads(pkg.read_text(encoding="utf-8"))
                scripts = {str(k): str(v) for k, v in ((data or {}).get("scripts") or {}).items()}
            except (OSError, json.JSONDecodeError, AttributeError):
                scripts = {}
        if scripts.get("tests-only"):
            proc = subprocess.run(
                ["npm", "run", "tests-only", "--silent"],
                cwd=str(rep),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        else:
            tape_bin = rep / "node_modules" / ".bin" / "tape"
            cmd = (
                [str(tape_bin), "test/**/*.js"]
                if tape_bin.is_file()
                else ["npx", "--no-install", "tape", "test/**/*.js"]
            )
            # Expand globs via shell-ish: use npm exec with pkg tests when available
            proc = subprocess.run(
                cmd,
                cwd=str(rep),
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        out = proc.stdout + "\n" + proc.stderr
        return _parse_js_outcome_text(out, language=language, returncode=int(proc.returncode))

    # Generic npm test with PATH (mocha etc.)
    proc = subprocess.run(
        ["npm", "test"],
        cwd=str(rep),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    out = proc.stdout + "\n" + proc.stderr
    return _parse_js_outcome_text(out, language=language, returncode=int(proc.returncode))


def run_typescript_suite(repo: Path) -> SuiteOutcome:
    """Execute real TS suite (jest motors or ava/tape live packs)."""
    return _run_js_framework_suite(repo, language="typescript")


def run_javascript_suite(repo: Path) -> SuiteOutcome:
    """Execute real JS suite (jest/ava/tape) with node_modules/.bin on PATH."""
    return _run_js_framework_suite(repo, language="javascript")


def run_language_suite(repo: Path, language: HarborSuiteLang | str) -> SuiteOutcome:
    """Dispatch to multi-lang suite reporters (VAL-LABEL-006).

    Prefers :mod:`suite_reporters` adapters so dual-run and grader selectors share
    the same tool identity. Falls back to the in-module runners for Py/Go/TS.
    """
    lang = language.strip().lower()
    # Local import: suite_reporters imports this module for SuiteOutcome/runners.
    from swe_factory.producers import suite_reporters as _reporters

    normalized = _reporters.normalize_reporter_lang(lang)
    if normalized in _reporters.list_reporter_languages():
        try:
            return _reporters.run_with_reporter(repo, normalized)
        except HarborLabelError:
            raise
        except Exception as exc:
            # Known hosted langs still try legacy runners as last resort.
            if normalized not in {"python", "go", "typescript", "javascript"}:
                raise HarborLabelError(f"suite reporter failed for {language!r}: {exc}") from exc
    if lang == "python" or normalized == "python":
        return run_python_suite(repo)
    if lang == "go" or normalized == "go":
        return run_go_suite(repo)
    if lang in {"typescript", "ts", "javascript", "js"} or normalized in {
        "typescript",
        "javascript",
    }:
        return run_typescript_suite(repo)
    raise HarborLabelError(f"unsupported language for dual-run suite: {language!r}")


def assert_broken_matches_labels(
    *,
    broken: SuiteOutcome,
    f2p_node_ids: NodeIdSeq,
    p2p_node_ids: NodeIdSeq,
) -> None:
    """Assert broken suite fails exactly F2P and none of P2P."""
    f2p = set(f2p_node_ids)
    p2p = set(p2p_node_ids)
    if f2p & p2p:
        raise HarborLabelError(f"F2P/P2P overlap: {sorted(f2p & p2p)}")
    missing_f2p_fails = sorted(f2p - broken.failed_set)
    f2p_still_pass = sorted(f2p & broken.passed_set)
    p2p_failed = sorted(p2p & broken.failed_set)
    p2p_missing = sorted(p2p - broken.passed_set)
    problems: list[str] = []
    if missing_f2p_fails:
        problems.append(f"F2P not failed on broken: {missing_f2p_fails}")
    if f2p_still_pass:
        problems.append(f"F2P still pass on broken: {f2p_still_pass}")
    if p2p_failed:
        problems.append(f"P2P fail on broken: {p2p_failed}")
    if p2p_missing:
        problems.append(f"P2P not pass on broken: {p2p_missing}")
    if problems:
        raise HarborLabelError("; ".join(problems))


def compute_dual_run_labels(
    *,
    language: HarborSuiteLang | str,
    green_repo: Path,
    broken_repo: Path,
    held_out_relative_path: str | None = None,
    held_out_content: str | None = None,
    require_green_clean: bool = True,
    require_nonempty_f2p: bool = True,
) -> DualRunLabels:
    """Run green + broken suites (with held-out applied) and label F2P/P2P."""
    green = Path(green_repo)
    broken = Path(broken_repo)
    if not green.is_dir() or not broken.is_dir():
        raise HarborLabelError("green_repo and broken_repo must be directories")

    # Copy to temp workspaces so held-out writes / suite side-effects never mutate callers.
    with tempfile.TemporaryDirectory(prefix="sdf-dual-label-") as td:
        root = Path(td)
        g_ws = root / "green"
        b_ws = root / "broken"
        shutil.copytree(
            green,
            g_ws,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                "*.pyc",
                ".venv",
                "node_modules",
                ".pytest_cache",
                "*.egg-info",
            ),
        )
        shutil.copytree(
            broken,
            b_ws,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                "*.pyc",
                ".venv",
                "node_modules",
                ".pytest_cache",
                "*.egg-info",
            ),
        )
        if held_out_relative_path and held_out_content is not None:
            _write_held_out(g_ws, held_out_relative_path, held_out_content)
            _write_held_out(b_ws, held_out_relative_path, held_out_content)

        lang = language.strip().lower()
        if lang in {"typescript", "ts", "javascript", "js"}:
            # Warm deps on green, then share node_modules into broken.
            suite_lang = "javascript" if lang in {"javascript", "js"} else "typescript"
            green_out = run_language_suite(g_ws, suite_lang)
            nm = g_ws / "node_modules"
            if nm.is_dir() and not (b_ws / "node_modules").exists():
                shutil.copytree(nm, b_ws / "node_modules", symlinks=True)
            for name in ("package-lock.json",):
                src = g_ws / name
                if src.is_file():
                    shutil.copy2(src, b_ws / name)
            broken_out = run_language_suite(b_ws, suite_lang)
        elif lang in {"python", "go", "rust", "rs"}:
            mapped = "rust" if lang in {"rust", "rs"} else lang
            green_out = run_language_suite(g_ws, mapped)
            broken_out = run_language_suite(b_ws, mapped)
        else:
            raise HarborLabelError(f"unsupported language: {language!r}")

    try:
        from swe_factory.producers.suite_reporters import reporter_info

        rep_meta = reporter_info(language).to_dict()
    except Exception:
        rep_meta = {"language": language.strip().lower()}

    labels = labels_from_suite_outcomes(
        green_out,
        broken_out,
        require_nonempty_f2p=require_nonempty_f2p,
        require_green_clean=require_green_clean,
        notes={
            "held_out": held_out_relative_path,
            "reporter": rep_meta,
        },
    )
    return labels


def compute_dual_run_labels_twice(
    *,
    language: HarborSuiteLang | str,
    green_repo: Path,
    broken_repo: Path,
    held_out_relative_path: str | None = None,
    held_out_content: str | None = None,
    require_green_clean: bool = True,
    require_nonempty_f2p: bool = True,
) -> DualRunLabels:
    """Run dual-run labeling twice and flake-reject on disagreement (VAL-LABEL-005).

    On success returns the first run (identical node id sets guaranteed).
    """
    kwargs = dict(
        language=language,
        green_repo=green_repo,
        broken_repo=broken_repo,
        held_out_relative_path=held_out_relative_path,
        held_out_content=held_out_content,
        require_green_clean=require_green_clean,
        require_nonempty_f2p=require_nonempty_f2p,
    )
    first = compute_dual_run_labels(**kwargs)  # type: ignore[arg-type]
    second = compute_dual_run_labels(**kwargs)  # type: ignore[arg-type]
    if first.f2p_node_ids != second.f2p_node_ids or first.p2p_node_ids != second.p2p_node_ids:
        raise HarborLabelError(
            "flake: dual-run label recompute disagreed on node id sets",
            reason_codes=(LABEL_G2_FLAKE, LABEL_FLAKE_REJECT),
            details={
                "first_f2p": list(first.f2p_node_ids),
                "second_f2p": list(second.f2p_node_ids),
                "first_p2p": list(first.p2p_node_ids),
                "second_p2p": list(second.p2p_node_ids),
            },
        )
    # Also defend against suite outcome flake across the two broken/green runs.
    assert_no_dual_run_flake(
        [first.green, second.green],
        phase="green_recompute",
    )
    assert_no_dual_run_flake(
        [first.broken, second.broken],
        phase="broken_recompute",
    )
    return first


__all__ = [
    "LABEL_EMPTY_F2P",
    "LABEL_FLAKE_REJECT",
    "LABEL_G2_FLAKE",
    "DualRunLabels",
    "HarborLabelError",
    "HarborSuiteLang",
    "SuiteOutcome",
    "assert_broken_matches_labels",
    "assert_held_out_verifier_only",
    "assert_no_dual_run_flake",
    "compute_dual_run_labels",
    "compute_dual_run_labels_twice",
    "detect_dual_run_flake",
    "label_cohorts_from_outcomes",
    "labels_from_suite_outcomes",
    "labels_to_tests_config",
    "pytest_nodeid_to_harbor",
    "run_go_suite",
    "run_language_suite",
    "run_python_suite",
    "run_typescript_suite",
    "write_tests_config_json",
]
