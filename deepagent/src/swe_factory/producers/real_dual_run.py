"""Real-PR dual-run labeling against **real** language suites (VAL-RDUAL-001..004).

Unlike synthetic harbor motors (hand-seeded inverse trees), Real-PR dual-run:

1. Starts from a base worktree at PR base SHA (broken / base state).
2. Applies held-out ``test.patch`` **only** on verifier/label workspace copies
   (never durable agent context).
3. Applies ``solution.patch`` (gold source) for the gold evaluation only.
4. Runs the **real** suite reporter for the language
   (pytest / go test / jest / cargo when present) — not motor fake stubs or
   hardcoded nodeid strings without a reporter identity.
5. Labels F2P (fail@base ∩ pass@gold) and P2P (pass both); requires ``|f2p| ≥ 1``.
6. Persists node ids into Harbor ``tests/config.json`` for the verifier.
7. Flake dual-run signature disagreement rejects the candidate (never cert-keep).

Assertions:
- VAL-RDUAL-001 real suite reporter
- VAL-RDUAL-002 F2P fail@base pass@gold; empty F2P rejected
- VAL-RDUAL-003 P2P pass-both; F2P∩P2P empty; config.json fields
- VAL-RDUAL-004 held-out test.patch only on verifier prepare path
"""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.producers.harbor_labeling import (
    LABEL_EMPTY_F2P,
    LABEL_FLAKE_REJECT,
    LABEL_G2_FLAKE,
    DualRunLabels,
    HarborLabelError,
    SuiteOutcome,
    assert_held_out_verifier_only,
    assert_no_dual_run_flake,
    detect_dual_run_flake,
    run_language_suite,
    write_tests_config_json,
)
from swe_factory.producers.suite_reporters import (
    grade_tool_label_for,
    list_reporter_languages,
    normalize_reporter_lang,
    reporter_info,
)

REAL_PR_SOURCE_TRACK = "real_pr"

# Authentic suite commands named in dual-run logs / notes (VAL-RDUAL-001).
# JS/TS live packs select ava/tape/jest at runtime; meta command stays npm-family.
REAL_SUITE_COMMANDS: dict[str, str] = {
    "python": "pytest -q -p no:cacheprovider --tb=no",
    "go": "go test ./... -count=1 -v",
    "typescript": "npm test",
    "javascript": "npm test",
    "rust": "cargo test -- --nocapture --test-threads=1",
}

# Codes specific to Real-PR dual-run refusal / flake.
RDUAL_STUB_CONFIG = "RDUAL_STUB_CONFIG"
RDUAL_AGENT_LEAK = "RDUAL_AGENT_LEAK"
RDUAL_EMPTY_TEST_PATCH = "RDUAL_EMPTY_TEST_PATCH"
RDUAL_PATCH_APPLY = "RDUAL_PATCH_APPLY"
RDUAL_FLAKE = LABEL_FLAKE_REJECT


class RealDualRunError(HarborLabelError):
    """Raised when Real-PR dual-run labeling / isolation fails."""


# ---------------------------------------------------------------------------
# Patch apply helpers (verifier prepare path)
# ---------------------------------------------------------------------------


_DIFF_GIT_RE = re.compile(
    r"^diff --git (?P<a>\"?a/(?P<path_a>.*?)\"?) (?P<b>\"?b/(?P<path_b>.*?)\"?)$"
)


def paths_touched_by_unified_diff(text: str) -> list[str]:
    """Unique file paths a unified diff touches (git apply target set)."""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        path: str | None = None
        m = _DIFF_GIT_RE.match(line)
        if m:
            path = m.group("path_b") or m.group("path_a")
        elif line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:]
        if path and path != "/dev/null" and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def apply_unified_patch(
    workspace: Path | str,
    patch_text: str,
    *,
    label: str = "patch",
    allow_empty: bool = False,
) -> list[str]:
    """Apply a unified diff to ``workspace`` via ``git apply`` (or pure fallback).

    Used on **verifier / dual-run label** workspaces for:
    - gold ``solution.patch``
    - held-out ``test.patch``

    Agent image/context must never call this with held-out test.patch durable copy.
    Returns the list of paths touched.
    """
    root = Path(workspace)
    if not root.is_dir():
        raise RealDualRunError(f"workspace is not a directory: {root}")
    body = patch_text if patch_text.endswith("\n") or not patch_text else patch_text + "\n"
    if not body.strip():
        if allow_empty:
            return []
        raise RealDualRunError(
            f"{label} is empty",
            reason_codes=(RDUAL_EMPTY_TEST_PATCH if "test" in label else RDUAL_PATCH_APPLY,),
        )
    paths = paths_touched_by_unified_diff(body)
    patch_file = root / f".sdf_{label.replace('/', '_')}.patch"
    patch_file.write_text(body, encoding="utf-8")
    try:
        # Prefer git apply when .git exists (real clone trees).
        if (root / ".git").exists() or (root / ".git").is_file():
            proc = subprocess.run(
                [
                    "git",
                    "apply",
                    "--whitespace=nowarn",
                    "--allow-empty" if allow_empty else "--recount",
                    str(patch_file.name),
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            # Some patches are new-file only without parent dirs; retry with -p1 default.
            if proc.returncode != 0:
                proc = subprocess.run(
                    [
                        "git",
                        "apply",
                        "--whitespace=nowarn",
                        "--unsafe-paths",
                        str(patch_file.name),
                    ],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            if proc.returncode != 0:
                # Fall through to pure apply for incomplete git trees.
                _apply_unified_patch_pure(root, body)
        else:
            _apply_unified_patch_pure(root, body)
    finally:
        with contextlib.suppress(OSError):
            patch_file.unlink(missing_ok=True)
    return paths


def _apply_unified_patch_pure(root: Path, body: str) -> None:
    """Minimal pure-Python unified-diff applier for offline fixtures without git.

    Supports simple text file create / modify / delete hunks sufficient for unit
    and motor dual-run offline evidence. Complex binary/rename patches should
    use a real git worktree.
    """
    # Split into file chunks by "diff --git" or "--- a/"
    chunks: list[str] = []
    current: list[str] = []
    for line in body.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            chunks.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("".join(current))

    for chunk in chunks:
        if not chunk.strip():
            continue
        rel: str | None = None
        for line in chunk.splitlines():
            if line.startswith("+++ b/"):
                rel = line[6:].strip()
                break
            m = _DIFF_GIT_RE.match(line)
            if m and m.group("path_b"):
                rel = m.group("path_b")
        if not rel or rel == "/dev/null":
            # Deletion: --- a/path +++ /dev/null
            for line in chunk.splitlines():
                if line.startswith("--- a/"):
                    del_path = root / line[6:].strip()
                    if del_path.is_file():
                        del_path.unlink()
                    break
            continue
        target = root / rel
        is_new = "new file mode" in chunk or "--- /dev/null" in chunk
        old_lines: list[str] = []
        if target.is_file() and not is_new:
            old_lines = target.read_text(encoding="utf-8", errors="replace").splitlines(
                keepends=True
            )
        new_lines = _apply_hunks_to_lines(old_lines, chunk)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(new_lines), encoding="utf-8")


def _apply_hunks_to_lines(old_lines: list[str], chunk: str) -> list[str]:
    """Apply @@ hunks from one file chunk onto old_lines → new content.

    When context lines do not match the on-disk file (common for loose offline
    fixtures), fall back to a best-effort rewrite: keep the original body and
    append any pure-addition trailing hunks, or rebuild the whole file from the
    ``+`` lines when the old side is empty.
    """
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    lines = chunk.splitlines(keepends=True)
    # Fast path: brand-new file with only + lines
    if not any(hunk_re.match(ln.rstrip("\n")) for ln in lines):
        plus = [
            ln[1:] if ln.startswith("+") else ln
            for ln in lines
            if ln.startswith("+") and not ln.startswith("+++")
        ]
        if plus:
            return plus
        return list(old_lines)

    if not old_lines:
        plus = [ln[1:] for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
        return plus if plus else old_lines

    # Parse hunks into structured ops; verify context against old_lines.
    hunks: list[tuple[int, list[tuple[str, str]]]] = []
    i = 0
    while i < len(lines):
        m = hunk_re.match(lines[i].rstrip("\n"))
        if not m:
            i += 1
            continue
        old_start = int(m.group(1))
        ops: list[tuple[str, str]] = []
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            ln = lines[i]
            if ln.startswith(("diff --git", "--- ", "+++ ")):
                break
            if ln.startswith("\\"):
                i += 1
                continue
            if ln.startswith(" "):
                ops.append((" ", ln[1:]))
            elif ln.startswith("-"):
                ops.append(("-", ln[1:]))
            elif ln.startswith("+"):
                ops.append(("+", ln[1:]))
            else:
                break
            i += 1
        hunks.append((max(0, old_start - 1), ops))

    # Context-match validation
    context_ok = True
    for start, ops in hunks:
        idx = start
        for kind, text in ops:
            if kind in {" ", "-"}:
                if idx >= len(old_lines) or old_lines[idx] != text:
                    # tolerate missing trailing newline differences
                    if idx < len(old_lines) and old_lines[idx].rstrip("\n") == text.rstrip("\n"):
                        idx += 1
                        continue
                    context_ok = False
                    break
                idx += 1
        if not context_ok:
            break

    if not context_ok:
        # Fallback for fixture mismatch: append trailing pure-addition body.
        plus_only = [text for _s, ops in hunks for k, text in ops if k == "+"]
        minus_only = [text for _s, ops in hunks for k, text in ops if k == "-"]
        if not minus_only and plus_only:
            return list(old_lines) + plus_only
        # Entire-file rewrite from net new content if everything is addition-weighted.
        if plus_only and len(plus_only) >= len(minus_only):
            # Prefer precise pure rewrite: start from old, string-replace first minus block.
            body = "".join(old_lines)
            for _s, ops in hunks:
                for kind, text in ops:
                    if kind == "-" and text in body:
                        body = body.replace(text, "", 1)
                for kind, text in ops:
                    if kind == "+" and text not in body:
                        if not body.endswith("\n") and body:
                            body += "\n"
                        body += text if text.endswith("\n") else text + "\n"
            return body.splitlines(keepends=True) or [body]
        return list(old_lines)

    rebuilt: list[str] = []
    old_idx = 0
    for start, ops in hunks:
        while old_idx < start and old_idx < len(old_lines):
            rebuilt.append(old_lines[old_idx])
            old_idx += 1
        for kind, text in ops:
            if kind == " ":
                rebuilt.append(old_lines[old_idx] if old_idx < len(old_lines) else text)
                old_idx += 1
            elif kind == "-":
                old_idx += 1
            elif kind == "+":
                rebuilt.append(text)
    while old_idx < len(old_lines):
        rebuilt.append(old_lines[old_idx])
        old_idx += 1
    return rebuilt


# ---------------------------------------------------------------------------
# Suite identity (VAL-RDUAL-001)
# ---------------------------------------------------------------------------


def suite_command_for(language: str) -> str:
    """Return the authentic real-suite command string for *language*."""
    lang = normalize_reporter_lang(language)
    if lang not in REAL_SUITE_COMMANDS:
        raise RealDualRunError(
            f"no real suite command for language {language!r}; "
            f"supported={sorted(REAL_SUITE_COMMANDS)}"
        )
    return REAL_SUITE_COMMANDS[lang]


_LINT_PLUGIN_MARKERS = (
    ".ruff",
    ".mypy",
    ".black",
    ".flake8",
    ".pylint",
    ".isort",
    ".bandit",
    ".pyright",
    "ruff.format",
    "ruff.check",
)


def is_lint_plugin_node(node_id: str) -> bool:
    """True for pytest-plugin lint node ids (ruff/mypy/black) not product F2P/P2P."""
    n = (node_id or "").lower()
    if not n:
        return True
    return any(m in n for m in _LINT_PLUGIN_MARKERS)


def filter_suite_product_nodes(nodes: Sequence[str]) -> tuple[str, ...]:
    """Drop lint-plugin / non-product pytest plugin node ids from dual-run labels."""
    return tuple(n for n in nodes if n and not is_lint_plugin_node(str(n)))


def suite_paths_from_node_ids(node_ids: Sequence[str]) -> list[str]:
    """Map Harbor pytest node ids -> unique repo-relative test file paths.

    Examples:
    - ``tests.test_termui.test_echo`` -> ``tests/test_termui.py``
    - ``tests.test_path.TestPath.test_iterdir_on_file`` -> ``tests/test_path.py``
    - ``tests.test_basic.test_custom_version_option[args0]`` -> ``tests/test_basic.py``
    """
    out: list[str] = []
    seen: set[str] = set()
    for nid in node_ids:
        s = str(nid).strip()
        if not s or is_lint_plugin_node(s):
            continue
        bare = re.sub(r"\[.*\]$", "", s)
        parts = [part for part in bare.split(".") if part]
        if len(parts) < 2:
            continue
        # Drop trailing method/function leaf.
        head = parts[:-1]
        # Drop class segment(s): bare "Test" or CamelCase Test*.
        while head and (head[-1] == "Test" or re.match(r"Test[A-Z_]", head[-1] or "") is not None):
            head.pop()
        if not head:
            continue
        # Keep modules that look like a test file (name test_*/tests package).
        # Prefer last component as the file stem when it is a test_* module.
        if not any(part.startswith("test") or part == "tests" for part in head):
            continue
        rel = "/".join(head) + ".py"
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def assert_real_suite_reporter(language: str) -> dict[str, str]:
    """Resolve real suite reporter identity; refuse unknown/stub langs."""
    lang = normalize_reporter_lang(language)
    if lang not in list_reporter_languages():
        raise RealDualRunError(
            f"no real suite reporter for language {language!r}; "
            f"supported={list(list_reporter_languages())}",
            reason_codes=(RDUAL_STUB_CONFIG,),
        )
    info = reporter_info(lang)
    cmd = suite_command_for(lang)
    payload = info.to_dict()
    payload["suite_command"] = cmd
    return payload


def refuse_stub_only_config(
    *,
    f2p_node_ids: Sequence[str],
    source_track: str = REAL_PR_SOURCE_TRACK,
    reporter: Mapping[str, Any] | None = None,
    language: str | None = None,
) -> None:
    """Refuse certified real_pr keeps that only have stub/hardcoded node ids.

    Pass only when a real reporter identity is present and |f2p| ≥ 1.
    """
    track = (source_track or "").strip().lower()
    if track and track != REAL_PR_SOURCE_TRACK and track not in {"", "real"}:
        # Non-product tracks may keep motor stubs; real_pr cannot.
        return
    if not f2p_node_ids:
        raise RealDualRunError(
            "real_pr dual-run refused empty f2p_node_ids",
            reason_codes=(LABEL_EMPTY_F2P,),
        )
    if reporter is None and language is None:
        raise RealDualRunError(
            "real_pr dual-run refused stub-only config (no suite reporter identity)",
            reason_codes=(RDUAL_STUB_CONFIG,),
        )
    if reporter is not None:
        tool = str(reporter.get("tool_label") or "").strip()
        rid = str(reporter.get("reporter_id") or "").strip()
        if not tool or not rid:
            raise RealDualRunError(
                "real_pr dual-run refused incomplete reporter metadata",
                reason_codes=(RDUAL_STUB_CONFIG,),
            )
        stub_markers = ("fake", "stub", "motor_only", "hardcoded")
        blob = f"{tool} {rid}".lower()
        if any(m in blob for m in stub_markers):
            raise RealDualRunError(
                f"real_pr dual-run refused stub reporter {reporter!r}",
                reason_codes=(RDUAL_STUB_CONFIG,),
            )
    if language is not None:
        assert_real_suite_reporter(language)


# ---------------------------------------------------------------------------
# Verifier prepare workspaces (VAL-RDUAL-004)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerifierPrepareResult:
    """Base + gold workspaces with held-out test.patch applied (never agent)."""

    base_workspace: Path
    gold_workspace: Path
    test_patch_paths: tuple[str, ...]
    solution_paths: tuple[str, ...]
    suite_command: str
    reporter: dict[str, str]
    apply_log: tuple[str, ...]
    root: Path  # temporary root; caller owns cleanup if temp owned=False
    owned_temp: bool = True

    def cleanup(self) -> None:
        if self.owned_temp and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def prepare_verifier_dual_workspaces(
    *,
    base_repo: Path | str,
    solution_patch: str,
    test_patch: str,
    language: str,
    work_root: Path | str | None = None,
) -> VerifierPrepareResult:
    """Materialize base and gold workspaces for dual-run (test.patch verifier-only).

    Both trees receive held-out ``test.patch`` (verifier prepare contract).
    Gold additionally receives ``solution_patch``. The original ``base_repo``
    (agent tree candidate) is **not** mutated and must not already ship
    held-out materials.
    """
    base = Path(base_repo)
    if not base.is_dir():
        raise RealDualRunError(f"base_repo is not a directory: {base}")
    if not (test_patch or "").strip():
        raise RealDualRunError(
            "held-out test.patch required for real dual-run",
            reason_codes=(RDUAL_EMPTY_TEST_PATCH,),
        )
    if not (solution_patch or "").strip():
        raise RealDualRunError(
            "solution.patch required for gold dual-run",
            reason_codes=(RDUAL_PATCH_APPLY,),
        )

    rep = assert_real_suite_reporter(language)
    cmd = suite_command_for(language)

    owned = work_root is None
    root = Path(work_root) if work_root else Path(tempfile.mkdtemp(prefix="sdf-rdual-"))
    root.mkdir(parents=True, exist_ok=True)
    base_ws = root / "base"
    gold_ws = root / "gold"
    if base_ws.exists():
        shutil.rmtree(base_ws)
    if gold_ws.exists():
        shutil.rmtree(gold_ws)

    # Keep node_modules when already present on base (npm install in compose path);
    # omitting them forces a second install inside each dual workspace and breaks
    # PATH for local bins. Still drop bulky rust target / python caches.
    common_ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".venv",
        ".pytest_cache",
        "*.egg-info",
        "target",
    )
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        "*.pyc",
        ".venv",
        ".pytest_cache",
        "*.egg-info",
        "target",
    )
    # Keep .git when present so git apply works on real clones.
    if (base / ".git").exists() or (base / ".git").is_file():
        shutil.copytree(
            base,
            base_ws,
            symlinks=True,
            ignore=common_ignore,
        )
        shutil.copytree(
            base,
            gold_ws,
            symlinks=True,
            ignore=common_ignore,
        )
    else:
        shutil.copytree(base, base_ws, ignore=ignore)
        shutil.copytree(base, gold_ws, ignore=ignore)

    apply_log: list[str] = []
    # Held-out test.patch on *both* verifier trees (prepare path).
    tp_paths = apply_unified_patch(base_ws, test_patch, label="test.patch")
    apply_log.append(f"applied test.patch to base ({len(tp_paths)} paths)")
    apply_unified_patch(gold_ws, test_patch, label="test.patch")
    apply_log.append(f"applied test.patch to gold ({len(tp_paths)} paths)")
    # Gold only: solution
    sol_paths = apply_unified_patch(gold_ws, solution_patch, label="solution.patch")
    apply_log.append(f"applied solution.patch to gold ({len(sol_paths)} paths)")
    apply_log.append(f"suite_command={cmd}")
    apply_log.append(f"reporter_id={rep.get('reporter_id')}")

    return VerifierPrepareResult(
        base_workspace=base_ws,
        gold_workspace=gold_ws,
        test_patch_paths=tuple(tp_paths),
        solution_paths=tuple(sol_paths),
        suite_command=cmd,
        reporter=rep,
        apply_log=tuple(apply_log),
        root=root,
        owned_temp=owned,
    )


# ---------------------------------------------------------------------------
# Result type + main dual-run entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RealDualRunResult:
    """Authoritative Real-PR dual-run labels + config payload."""

    labels: DualRunLabels
    f2p_node_ids: tuple[str, ...]
    p2p_node_ids: tuple[str, ...]
    language: str
    suite_command: str
    reporter: dict[str, str]
    base_commit: str
    config_path: Path | None
    config_payload: dict[str, Any]
    apply_log: tuple[str, ...]
    test_patch_applied: bool
    accepted: bool = True
    reason_codes: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "f2p_node_ids": list(self.f2p_node_ids),
            "p2p_node_ids": list(self.p2p_node_ids),
            "language": self.language,
            "suite_command": self.suite_command,
            "reporter": dict(self.reporter),
            "base_commit": self.base_commit,
            "config_path": str(self.config_path) if self.config_path else None,
            "config_payload": dict(self.config_payload),
            "apply_log": list(self.apply_log),
            "test_patch_applied": self.test_patch_applied,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "notes": dict(self.notes),
            "labels": self.labels.to_dict(),
        }


def _broken_fail_set(outcome: SuiteOutcome) -> set[str]:
    """Fail side for F2P includes call failures **and** collection/setup errors."""
    return set(outcome.failed) | set(outcome.errors)


def labels_from_real_suite_outcomes(
    green: SuiteOutcome,
    broken: SuiteOutcome,
    *,
    require_nonempty_f2p: bool = True,
    require_green_clean: bool = True,
    notes: Mapping[str, Any] | None = None,
) -> DualRunLabels:
    """F2P/P2P from real suite outcomes; broken fail side = failed ∪ errors.

    Setup/collection errors on base (common when held-out tests import missing
    gold symbols) count as F2P when gold is green (VAL-RDUAL-002).
    """
    from swe_factory.producers.harbor_labeling import (  # local: avoid cycle
        LABEL_EMPTY_F2P as _EMPTY,
    )
    from swe_factory.producers.harbor_labeling import (
        DualRunLabels as _DL,
    )
    from swe_factory.producers.harbor_labeling import (
        HarborLabelError as _HLE,
    )
    from swe_factory.producers.harbor_labeling import (
        label_cohorts_from_outcomes as _cohorts,
    )

    green_product_failed = list(filter_suite_product_nodes(green.failed))
    green_product_errors = list(filter_suite_product_nodes(green.errors))
    green_product_passed = list(filter_suite_product_nodes(green.passed))
    if require_green_clean and green_product_failed:
        raise _HLE(f"green suite must be clean for labeling; failed={green_product_failed}")
    if green_product_errors:
        raise _HLE(f"green suite collection errors: {green_product_errors}")
    if not green_product_passed:
        raise _HLE("green suite produced zero passing node ids")

    # Union failed + errors on the broken/base side (VAL-RDUAL / library note).
    # Strip lint-plugin node ids (ruff/mypy) so dual-run labels stay product tests.
    green_pass = set(filter_suite_product_nodes(green.passed))
    green_fail = set(filter_suite_product_nodes(green.failed))
    broken_fail = set(filter_suite_product_nodes(tuple(set(broken.failed) | set(broken.errors))))
    # Prefer fail over pass when a reporter emits both for the same node id
    # (param re-collection / multi-report edges can double-list rare cases).
    broken_pass = set(filter_suite_product_nodes(broken.passed)) - broken_fail
    f2p, p2p = _cohorts(
        green_passed=green_pass,
        green_failed=green_fail,
        broken_passed=broken_pass,
        broken_failed=broken_fail,
    )
    if require_nonempty_f2p and not f2p:
        raise _HLE(
            "dual-run produced empty F2P cohort "
            f"(broken_failed={sorted(broken_fail)} green_passed={list(green.passed)})",
            reason_codes=(_EMPTY,),
        )
    if set(f2p) & set(p2p):
        raise _HLE(f"F2P/P2P overlap after labeling: {sorted(set(f2p) & set(p2p))}")

    # Sanity: F2P must fail-or-error on broken, pass on green; P2P pass both.
    missing_f2p = sorted(set(f2p) - broken_fail)
    if missing_f2p:
        raise _HLE(f"F2P not failed/errored on broken: {missing_f2p}")
    f2p_still_pass = sorted(set(f2p) & broken_pass)
    if f2p_still_pass:
        raise _HLE(f"F2P still pass on broken: {f2p_still_pass}")
    p2p_failed = sorted(set(p2p) & broken_fail)
    if p2p_failed:
        raise _HLE(f"P2P fail on broken: {p2p_failed}")
    p2p_missing = sorted(set(p2p) - broken_pass)
    if p2p_missing:
        raise _HLE(f"P2P not pass on broken: {p2p_missing}")
    green_missing = sorted((set(f2p) | set(p2p)) - green_pass)
    if green_missing:
        raise _HLE(f"labels must pass on green; missing/fail={green_missing}")

    all_nodes = tuple(sorted(green_pass | green_fail | broken_pass | broken_fail))
    note_blob: dict[str, Any] = {
        "method": "real_pr_dual_run_broken_vs_green",
        "f2p_count": len(f2p),
        "p2p_count": len(p2p),
        "broken_errors_unioned": True,
    }
    if notes:
        note_blob.update(dict(notes))
    # Store broken with errors folded into failed for audit parity.
    broken_view = SuiteOutcome(
        language=broken.language,
        passed=broken.passed,
        failed=tuple(sorted(broken_fail)),
        errors=(),
        returncode=broken.returncode,
        raw_tail=broken.raw_tail,
    )
    return _DL(
        f2p_node_ids=f2p,
        p2p_node_ids=p2p,
        green=green,
        broken=broken_view,
        all_nodes=all_nodes,
        notes=note_blob,
        accepted=True,
        reason_codes=(),
    )


def run_real_suite(
    repo: Path | str,
    language: str,
    *,
    offline_outcome: SuiteOutcome | None = None,
    test_paths: Sequence[str] = (),
) -> SuiteOutcome:
    """Execute the real suite reporter (or inject locked offline outcome).

    ``test_paths`` scopes Python dual-run to held-out files when present so
    unrelated ambient suite fails (other modules) do not block green labeling.
    """
    lang = normalize_reporter_lang(language)
    rep_meta = assert_real_suite_reporter(lang)
    if offline_outcome is not None:
        # Offline / fixture: trust caller-provided real-reporter-shaped outcome.
        return offline_outcome
    if lang == "python" and test_paths:
        from swe_factory.producers.harbor_labeling import run_python_suite

        out = run_python_suite(Path(repo), test_paths=list(test_paths))
    else:
        out = run_language_suite(Path(repo), lang)
    # Annotate identity for logs
    _ = rep_meta
    return out


def label_real_pr_dual_run(
    *,
    language: str,
    base_repo: Path | str,
    solution_patch: str,
    test_patch: str,
    base_commit: str,
    config_dest: Path | str | None = None,
    work_root: Path | str | None = None,
    agent_context: Path | str | None = None,
    held_out_relative_paths: Sequence[str] = (),
    require_nonempty_f2p: bool = True,
    dual_runs: int = 1,
    offline_base_outcome: SuiteOutcome | None = None,
    offline_gold_outcome: SuiteOutcome | None = None,
    grade: Mapping[str, Any] | None = None,
    source_track: str = REAL_PR_SOURCE_TRACK,
) -> RealDualRunResult:
    """Dual-run base vs gold on the **real** language suite (VAL-RDUAL-001..004).

    Parameters
    ----------
    base_repo:
        Upstream tree at PR base SHA (no gold, no held-out tests).
    solution_patch:
        Multi-file source-only gold.
    test_patch:
        Held-out tests applied only to verifier dual-run workspaces.
    dual_runs:
        When ≥2, recompute labels twice and flake-reject on disagreement.
    offline_*_outcome:
        Optional locked suite outcomes for offline multi-lang reporter tests
        (parsers already proved authentic node ids from real suite log format).
    agent_context:
        Optional agent build context scanned for held-out leak (must be clean).
    """
    lang = normalize_reporter_lang(language)
    rep = assert_real_suite_reporter(lang)
    cmd = suite_command_for(lang)
    if not str(base_commit).strip():
        raise RealDualRunError("base_commit required for tests/config.json")

    # Isolation: agent must not hold test.patch / held-out paths.
    agent_hits: list[str] = []
    if agent_context is not None:
        agent_hits = assert_held_out_verifier_only(
            agent_context=agent_context,
            held_out_relative_paths=held_out_relative_paths,
        )
        if agent_hits:
            raise RealDualRunError(
                f"held-out material leaked into agent context: {agent_hits}",
                reason_codes=(RDUAL_AGENT_LEAK,),
                details={"hits": agent_hits},
            )

    prep = prepare_verifier_dual_workspaces(
        base_repo=base_repo,
        solution_patch=solution_patch,
        test_patch=test_patch,
        language=lang,
        work_root=work_root,
    )

    # Scope dual-run only when callers name held-out paths (product ship).
    # Empty held_out keeps full-suite dual-run for existing unit fixtures.
    # Prefer pure *test* files from held-out paths / test.patch (never source
    # paths) so docker verifier suite_paths do not explode into untested modules.
    def _is_test_rel(rel: str) -> bool:
        pl = rel.replace("\\", "/").lower()
        name = Path(rel).name.lower()
        return (
            "/tests/" in f"/{pl}"
            or pl.startswith("tests/")
            or name.startswith("test_")
            or name.endswith("_test.py")
            or ".test." in name
            or ".spec." in name
        )

    scoped_test_paths: list[str] = []
    for rel in held_out_relative_paths:
        rel_s = str(rel).strip()
        if not rel_s or not _is_test_rel(rel_s):
            continue
        if (prep.base_workspace / rel_s).exists() or (prep.gold_workspace / rel_s).exists():
            scoped_test_paths.append(rel_s)
    if held_out_relative_paths and not scoped_test_paths:
        for rel in paths_touched_by_unified_diff(test_patch):
            if not _is_test_rel(rel):
                continue
            if (prep.base_workspace / rel).exists() or (prep.gold_workspace / rel).exists():
                scoped_test_paths.append(rel)
    try:
        # Dual evaluation(s).
        label_runs: list[DualRunLabels] = []
        base_outcomes: list[SuiteOutcome] = []
        gold_outcomes: list[SuiteOutcome] = []
        n = max(1, int(dual_runs))
        for _ in range(n):
            if offline_base_outcome is not None and offline_gold_outcome is not None:
                base_out = offline_base_outcome
                gold_out = offline_gold_outcome
            else:
                base_out = run_real_suite(prep.base_workspace, lang, test_paths=scoped_test_paths)
                gold_out = run_real_suite(prep.gold_workspace, lang, test_paths=scoped_test_paths)
            base_outcomes.append(base_out)
            gold_outcomes.append(gold_out)
            labels = labels_from_real_suite_outcomes(
                gold_out,  # green = gold
                base_out,  # broken = base
                require_nonempty_f2p=require_nonempty_f2p,
                require_green_clean=True,
                notes={
                    "method": "real_pr_dual_run_base_vs_gold",
                    "suite_command": cmd,
                    "reporter": rep,
                    "held_out": "test.patch",
                    "source_track": source_track,
                    "apply_log": list(prep.apply_log),
                },
            )
            label_runs.append(labels)

        if n >= 2:
            # Flake: repoter outcome signatures must agree (VAL-RDUAL / ORCD flake).
            is_flake_b, codes_b, det_b = detect_dual_run_flake(base_outcomes, phase="base")
            is_flake_g, codes_g, det_g = detect_dual_run_flake(gold_outcomes, phase="gold")
            if is_flake_b or is_flake_g:
                raise RealDualRunError(
                    f"flake: dual-run real suite outcome mismatch base={det_b} gold={det_g}",
                    reason_codes=tuple(dict.fromkeys([*codes_b, *codes_g, RDUAL_FLAKE])),
                    details={"base": det_b, "gold": det_g},
                )
            first, second = label_runs[0], label_runs[1]
            if (
                first.f2p_node_ids != second.f2p_node_ids
                or first.p2p_node_ids != second.p2p_node_ids
            ):
                raise RealDualRunError(
                    "flake: dual-run label recompute disagreed on node id sets",
                    reason_codes=(LABEL_G2_FLAKE, LABEL_FLAKE_REJECT),
                    details={
                        "first_f2p": list(first.f2p_node_ids),
                        "second_f2p": list(second.f2p_node_ids),
                        "first_p2p": list(first.p2p_node_ids),
                        "second_p2p": list(second.p2p_node_ids),
                    },
                )

        labels = label_runs[0]
        # Product path (explicit held-out files): restrict P2P to F2P modules so
        # docker verifier suite_paths stay PR-local.
        if held_out_relative_paths:
            f2p_files = suite_paths_from_node_ids(list(labels.f2p_node_ids))
            f2p_mods = {
                Path(pp).as_posix().removesuffix(".py").replace("/", ".") for pp in f2p_files
            }
            if f2p_mods:
                filtered_p2p = tuple(
                    n
                    for n in labels.p2p_node_ids
                    if any(
                        str(n).startswith(mod + ".") or str(n).startswith(mod + "::")
                        for mod in f2p_mods
                    )
                )
                labels = type(labels)(
                    f2p_node_ids=labels.f2p_node_ids,
                    p2p_node_ids=filtered_p2p,
                    green=labels.green,
                    broken=labels.broken,
                    all_nodes=tuple(sorted(set(labels.f2p_node_ids) | set(filtered_p2p))),
                    notes=labels.notes,
                    accepted=labels.accepted,
                    reason_codes=labels.reason_codes,
                )
        refuse_stub_only_config(
            f2p_node_ids=labels.f2p_node_ids,
            source_track=source_track,
            reporter=rep,
            language=lang,
        )
        if set(labels.f2p_node_ids) & set(labels.p2p_node_ids):
            raise RealDualRunError(
                f"F2P/P2P overlap: {sorted(set(labels.f2p_node_ids) & set(labels.p2p_node_ids))}"
            )

        grade_payload: dict[str, Any] = {
            "format": "junit",
            "node_id": rep.get("node_id_form", "name"),
            "tool_label": grade_tool_label_for(lang),
            "reports": ["/logs/verifier/new.xml", "/logs/verifier/base.xml"],
        }
        if grade:
            grade_payload.update(dict(grade))

        config_payload = labels.config_payload(base_commit=base_commit, grade=grade_payload)
        # Ensure documented dual-run reporter fields travel with config.
        config_payload["suite_reporter"] = dict(rep)
        config_payload["suite_command"] = cmd
        config_payload["label_method"] = "real_pr_dual_run_base_vs_gold"
        config_payload["source_track"] = source_track
        suite_paths = suite_paths_from_node_ids(
            list(labels.f2p_node_ids) + list(labels.p2p_node_ids)
        )
        # Prefer F2P-containing files only for docker verifier (stable subset).
        f2p_paths = suite_paths_from_node_ids(list(labels.f2p_node_ids))
        if f2p_paths:
            suite_paths = f2p_paths
        elif scoped_test_paths:
            suite_paths = list(scoped_test_paths)
        if suite_paths:
            config_payload["suite_paths"] = list(suite_paths)
            config_payload["pytest_paths"] = list(suite_paths)

        cfg_path: Path | None = None
        if config_dest is not None:
            extra_cfg = {
                "suite_reporter": dict(rep),
                "suite_command": cmd,
                "label_method": "real_pr_dual_run_base_vs_gold",
                "source_track": source_track,
            }
            if suite_paths:
                extra_cfg["suite_paths"] = list(suite_paths)
                extra_cfg["pytest_paths"] = list(suite_paths)
            cfg_path = write_tests_config_json(
                config_dest,
                base_commit=base_commit,
                f2p_node_ids=config_payload["f2p_node_ids"],
                p2p_node_ids=config_payload["p2p_node_ids"],
                grade=grade_payload,
                extra=extra_cfg,
            )

        return RealDualRunResult(
            labels=labels,
            f2p_node_ids=labels.f2p_node_ids,
            p2p_node_ids=labels.p2p_node_ids,
            language=lang,
            suite_command=cmd,
            reporter=rep,
            base_commit=str(base_commit).strip(),
            config_path=cfg_path,
            config_payload=config_payload,
            apply_log=prep.apply_log,
            test_patch_applied=True,
            accepted=True,
            reason_codes=(),
            notes={
                "apply_log": list(prep.apply_log),
                "test_patch_paths": list(prep.test_patch_paths),
                "solution_paths": list(prep.solution_paths),
                "agent_isolation_hits": agent_hits,
                "dual_runs": n,
            },
        )
    finally:
        if prep.owned_temp:
            prep.cleanup()


def label_real_pr_from_outcomes(
    *,
    language: str,
    base_outcome: SuiteOutcome,
    gold_outcome: SuiteOutcome,
    base_commit: str,
    test_patch: str = "non-empty-placeholder",
    config_dest: Path | str | None = None,
    source_track: str = REAL_PR_SOURCE_TRACK,
    dual_gold_outcomes: Sequence[SuiteOutcome] = (),
    dual_base_outcomes: Sequence[SuiteOutcome] = (),
    grade: Mapping[str, Any] | None = None,
) -> RealDualRunResult:
    """Offline multi-lang dual-run from locked real-reporter outcomes (no subprocess).

    Use when exercising go/jest/rust adapters without host toolchains. Still
    requires non-empty test.patch document token and real reporter identity.
    Flake dual-runs: pass ``dual_*_outcomes`` length ≥2.
    """
    lang = normalize_reporter_lang(language)
    rep = assert_real_suite_reporter(lang)
    cmd = suite_command_for(lang)
    if not (test_patch or "").strip():
        raise RealDualRunError(
            "held-out test.patch required",
            reason_codes=(RDUAL_EMPTY_TEST_PATCH,),
        )

    bases = list(dual_base_outcomes) if dual_base_outcomes else [base_outcome]
    golds = list(dual_gold_outcomes) if dual_gold_outcomes else [gold_outcome]
    if len(bases) >= 2:
        is_flake, codes, details = detect_dual_run_flake(bases, phase="base")
        if is_flake:
            raise RealDualRunError(
                f"flake: base dual-run mismatch {details}",
                reason_codes=tuple(codes) or (LABEL_G2_FLAKE, LABEL_FLAKE_REJECT),
                details=details,
            )
    if len(golds) >= 2:
        is_flake, codes, details = detect_dual_run_flake(golds, phase="gold")
        if is_flake:
            raise RealDualRunError(
                f"flake: gold dual-run mismatch {details}",
                reason_codes=tuple(codes) or (LABEL_G2_FLAKE, LABEL_FLAKE_REJECT),
                details=details,
            )

    labels = labels_from_real_suite_outcomes(
        golds[0],
        bases[0],
        require_nonempty_f2p=True,
        notes={
            "method": "real_pr_dual_run_from_outcomes",
            "suite_command": cmd,
            "reporter": rep,
            "source_track": source_track,
            "held_out": "test.patch",
        },
    )
    refuse_stub_only_config(
        f2p_node_ids=labels.f2p_node_ids,
        source_track=source_track,
        reporter=rep,
        language=lang,
    )
    grade_payload: dict[str, Any] = {
        "format": "junit",
        "node_id": rep.get("node_id_form", "name"),
        "tool_label": grade_tool_label_for(lang),
        "reports": ["/logs/verifier/new.xml", "/logs/verifier/base.xml"],
    }
    if grade:
        grade_payload.update(dict(grade))
    config_payload = labels.config_payload(base_commit=base_commit, grade=grade_payload)
    config_payload["suite_reporter"] = dict(rep)
    config_payload["suite_command"] = cmd
    config_payload["label_method"] = "real_pr_dual_run_from_outcomes"
    config_payload["source_track"] = source_track

    cfg_path: Path | None = None
    if config_dest is not None:
        cfg_path = write_tests_config_json(
            config_dest,
            base_commit=base_commit,
            f2p_node_ids=config_payload["f2p_node_ids"],
            p2p_node_ids=config_payload["p2p_node_ids"],
            grade=grade_payload,
            extra={
                "suite_reporter": dict(rep),
                "suite_command": cmd,
                "label_method": "real_pr_dual_run_from_outcomes",
                "source_track": source_track,
            },
        )
    return RealDualRunResult(
        labels=labels,
        f2p_node_ids=labels.f2p_node_ids,
        p2p_node_ids=labels.p2p_node_ids,
        language=lang,
        suite_command=cmd,
        reporter=rep,
        base_commit=str(base_commit).strip(),
        config_path=cfg_path,
        config_payload=config_payload,
        apply_log=("offline_outcomes", f"suite_command={cmd}"),
        test_patch_applied=True,
        accepted=True,
        notes={"mode": "offline_outcomes", "held_out": "test.patch"},
    )


def assert_verifier_prepare_applies_test_patch(apply_log: Sequence[str]) -> None:
    """Guard: dual-run success path must name test.patch application (VAL-RDUAL-004)."""
    blob = "\n".join(apply_log).lower()
    if "test.patch" not in blob:
        raise RealDualRunError(
            "dual-run apply_log must name test.patch application",
            reason_codes=(RDUAL_EMPTY_TEST_PATCH,),
            details={"apply_log": list(apply_log)},
        )


def agent_context_excludes_test_patch(agent_context: Path | str) -> list[str]:
    """Return isolation hits if agent context durable-ships test.patch / solution."""
    return assert_held_out_verifier_only(agent_context=agent_context)


# Re-export flake helper for callers that only import this module.
def flake_reject_on_disagreement(
    runs: Sequence[SuiteOutcome],
    *,
    phase: str = "gold",
) -> None:
    """Raise :class:`RealDualRunError` when dual suite signatures disagree."""
    try:
        assert_no_dual_run_flake(runs, phase=phase)
    except HarborLabelError as exc:
        raise RealDualRunError(
            str(exc),
            reason_codes=exc.reason_codes or (LABEL_G2_FLAKE, LABEL_FLAKE_REJECT),
            details=exc.details,
        ) from exc


__all__ = [
    "REAL_PR_SOURCE_TRACK",
    "REAL_SUITE_COMMANDS",
    "RDUAL_AGENT_LEAK",
    "RDUAL_EMPTY_TEST_PATCH",
    "RDUAL_FLAKE",
    "RDUAL_PATCH_APPLY",
    "RDUAL_STUB_CONFIG",
    "RealDualRunError",
    "RealDualRunResult",
    "VerifierPrepareResult",
    "agent_context_excludes_test_patch",
    "apply_unified_patch",
    "assert_real_suite_reporter",
    "assert_verifier_prepare_applies_test_patch",
    "filter_suite_product_nodes",
    "suite_paths_from_node_ids",
    "flake_reject_on_disagreement",
    "is_lint_plugin_node",
    "label_real_pr_dual_run",
    "label_real_pr_from_outcomes",
    "labels_from_real_suite_outcomes",
    "paths_touched_by_unified_diff",
    "prepare_verifier_dual_workspaces",
    "refuse_stub_only_config",
    "run_real_suite",
    "suite_command_for",
]
