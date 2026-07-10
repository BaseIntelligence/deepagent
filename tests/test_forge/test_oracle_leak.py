"""Unit tests for the leak audit + sanitize gate (m4-leak).

Offline coverage (no real Docker) of the gate's contract assertions, driven over
real temp-directory trees plus a programmable :class:`AgentTreeProvider` fake:

- VAL-ORACLE-014: a clean agent-facing tree (broken source, no oracle/solution
  content, no hidden-test bodies) passes the leak audit -> ``leak_audit == "clean"``.
- VAL-ORACLE-015: a planted leak (oracle/solution content or a hidden-test answer)
  is detected and either stripped by the sanitizer (post-sanitize tree clean ->
  pass listing the marker) or, when it cannot be safely removed, the gate rejects
  citing the leak.

The real DockerSandbox export path is exercised by this feature's manual
verification and the user-testing validator in real Docker.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    EnvImage,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.leak import (
    REASON_LEAK,
    AgentTreeProvider,
    LeakAudit,
    LeakError,
    LeakFinding,
    assess_leak,
    audit_agent_tree,
    build_leak_report,
    normalize_agent_tree,
    run_leak_gate,
    sanitize_leaks,
)

# A gold line long enough (>= 24 chars) to be treated as a significant snippet.
_GOLD_LINE = "return total  # the correct gold implementation line"
_ORACLE_PATCH = (
    "--- a/src/m.py\n"
    "+++ b/src/m.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def total(xs):\n"
    f"-    return sum(xs) + 1  # off-by-one fault injected by the mutation\n"
    f"+    {_GOLD_LINE}\n"
)
_HIDDEN_TEST_LINE = "assert compute_total([1, 2, 3]) == 6  # discriminating F2P check"
_HIDDEN_TEST_BODY = "import m\n\n\ndef test_total():\n    " + _HIDDEN_TEST_LINE + "\n"


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #
class FakeAgentTreeProvider:
    """An :class:`AgentTreeProvider` that yields a pre-built local tree."""

    def __init__(self, root: Path, language: str = "python") -> None:
        self._root = root
        self.language = language
        self.opened = 0

    @contextlib.asynccontextmanager
    async def open(self) -> AsyncIterator[Path]:
        self.opened += 1
        yield self._root


def _write(root: Path, rel: str, content: str) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _clean_tree(root: Path) -> Path:
    """A broken agent-facing tree: the fault is present, the gold line is NOT."""
    _write(root, "src/m.py", "def total(xs):\n    return sum(xs) + 1  # bug\n")
    _write(root, "README.md", "# Demo project\n\nCompute totals.\n")
    return root


def _hidden_test_files() -> list[OracleTestFile]:
    return [
        OracleTestFile(
            path="tests/hidden/test_total.py",
            content=_HIDDEN_TEST_BODY,
            origin="synthesized",
        )
    ]


def _candidate() -> Candidate:
    return Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("src/m.py",), symbols=("total",)),
        mutation_patch=(
            "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,2 +1,2 @@\n"
            " def total(xs):\n-    return sum(xs)\n+    return sum(xs) + 1\n"
        ),
        oracle_patch=_ORACLE_PATCH,
        difficulty_hint="medium",
        provenance=Provenance(generator="ast_mutation", seed=7, language="python"),
    )


def _env_image(*, green: bool = True) -> EnvImage:
    return EnvImage(
        repo_id="demo",
        language="python",
        image_tag="swe-forge-env-demo:abc123",
        base_image="python:3.12-slim",
        commit="0" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command="python -m pytest",
        baseline_green=green,
        baseline_exit_code=0 if green else 1,
    )


def _alt_correct_report(
    test_files: list[OracleTestFile] | None = None,
    *,
    verdict: str = "pass",
) -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict=verdict,
        reasons=[] if verdict == "pass" else ["alt_correct_overfit: ..."],
        fail_to_pass=["python -m pytest tests/hidden/test_total.py"],
        pass_to_pass=["python -m pytest"],
        test_files=test_files if test_files is not None else _hidden_test_files(),
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        differential_pass=True,
        alt_correct_accepted=True,
        provenance=Provenance(generator="ast_mutation", seed=7, language="python"),
        details={"stage": "alt_correct"},
    )


# --------------------------------------------------------------------------- #
# LeakFinding / LeakAudit value types
# --------------------------------------------------------------------------- #
def test_leak_audit_clean_when_no_findings() -> None:
    audit = LeakAudit(findings=())
    assert audit.is_clean
    assert audit.markers() == []


def test_leak_audit_not_clean_with_findings() -> None:
    finding = LeakFinding(path="x.py", marker="oracle_snippet", removable=False)
    audit = LeakAudit(findings=(finding,))
    assert not audit.is_clean
    assert audit.markers() == ["oracle_snippet: x.py"]
    assert audit.removable_findings() == ()


# --------------------------------------------------------------------------- #
# audit_agent_tree
# --------------------------------------------------------------------------- #
def test_audit_clean_tree_has_no_findings(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    audit = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    assert audit.is_clean


def test_audit_detects_oracle_snippet_embedded_in_source(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    # Plant the restored gold line into a legitimate source file (a comment leak).
    _write(tmp_path, "src/notes.py", f"# hint: {_GOLD_LINE}\nVALUE = 1\n")
    audit = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    assert not audit.is_clean
    snippet = [f for f in audit.findings if f.marker == "oracle_snippet"]
    assert snippet and snippet[0].path == "src/notes.py"
    # oracle content embedded in a source file cannot be safely removed
    assert all(not f.removable for f in snippet)


def test_audit_detects_forbidden_artifact(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "solution.patch", _ORACLE_PATCH)
    audit = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    artifact = [f for f in audit.findings if f.marker == "forbidden_artifact"]
    assert artifact and artifact[0].path == "solution.patch"
    assert artifact[0].removable


def test_audit_detects_hidden_test_file_present(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "tests/hidden/test_total.py", _HIDDEN_TEST_BODY)
    audit = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    hidden = [f for f in audit.findings if f.marker == "hidden_test_file"]
    assert hidden and hidden[0].path == "tests/hidden/test_total.py"
    assert hidden[0].removable


def test_audit_detects_hidden_test_body_in_source(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    # Plant the discriminating assertion into a shipped fixture/source file.
    _write(tmp_path, "src/fixture.py", f"DATA = '''{_HIDDEN_TEST_LINE}'''\n")
    audit = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    body = [f for f in audit.findings if f.marker == "hidden_test_body"]
    assert body and body[0].path == "src/fixture.py"
    assert not body[0].removable


def test_audit_ignores_partial_overlap_with_baseline_test(tmp_path: Path) -> None:
    """One ordinary assertion overlap is not a copied hidden test body.

    A synthesized test can exercise a public behavior already covered by the
    repository's agent-visible regression suite.  Flagging a single shared
    assertion would reject a clean tree, while a complete hidden-test body is
    still detected below.
    """
    _clean_tree(tmp_path)
    shared = "assert clamp(5, 3, 3) == 3"
    _write(tmp_path, "tests/test_mathutils.py", f"def test_existing():\n    {shared}\n")
    hidden = [
        OracleTestFile(
            path="tests/hidden/test_clamp.py",
            content=(
                "def test_clamp_equal_bounds():\n"
                f"    {shared}\n"
                "    assert clamp(1, 3, 3) == 3\n"
            ),
            origin="synthesized",
        )
    ]

    audit = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=hidden
    )

    assert not [f for f in audit.findings if f.marker == "hidden_test_body"]


# --------------------------------------------------------------------------- #
# normalize_agent_tree + sanitize_leaks
# --------------------------------------------------------------------------- #
def test_normalize_strips_build_cache_artifacts(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "src/__pycache__/m.cpython-312.pyc", "bytecode")
    _write(tmp_path, "build/leftover.txt", "junk")
    removed = normalize_agent_tree(tmp_path)
    assert not (tmp_path / "src" / "__pycache__").exists()
    assert not (tmp_path / "build").exists()
    assert removed  # reported what it stripped


def test_sanitize_leaks_removes_removable_artifact(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "solution.patch", _ORACLE_PATCH)
    _write(tmp_path, "tests/hidden/test_total.py", _HIDDEN_TEST_BODY)
    audit = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    removed = sanitize_leaks(tmp_path, audit)
    assert "solution.patch" in removed
    assert "tests/hidden/test_total.py" in removed
    after = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    assert after.is_clean


def test_sanitize_leaks_leaves_embedded_content(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "src/notes.py", f"# hint: {_GOLD_LINE}\n")
    audit = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    sanitize_leaks(tmp_path, audit)
    # the legitimate source file is NOT deleted (would break the repo)
    assert (tmp_path / "src" / "notes.py").exists()
    after = audit_agent_tree(
        tmp_path, oracle_patch=_ORACLE_PATCH, hidden_test_files=_hidden_test_files()
    )
    assert not after.is_clean


# --------------------------------------------------------------------------- #
# assess_leak
# --------------------------------------------------------------------------- #
def test_assess_clean() -> None:
    outcome = assess_leak(LeakAudit(findings=()), LeakAudit(findings=()))
    assert outcome.is_pass
    assert outcome.leak_audit == "clean"
    assert outcome.detected is False


def test_assess_sanitized_pass() -> None:
    before = LeakAudit(
        findings=(LeakFinding(path="solution.patch", marker="forbidden_artifact"),)
    )
    after = LeakAudit(findings=())
    outcome = assess_leak(before, after, removed=["solution.patch"])
    assert outcome.is_pass
    assert outcome.detected is True
    assert outcome.sanitized is True
    assert outcome.leak_audit.startswith("sanitized:")
    assert "solution.patch" in outcome.leak_audit


def test_assess_reject_when_residual_leak() -> None:
    before = LeakAudit(
        findings=(
            LeakFinding(path="src/notes.py", marker="oracle_snippet", removable=False),
        )
    )
    outcome = assess_leak(before, before)
    assert not outcome.is_pass
    assert outcome.verdict == "reject"
    assert outcome.leak_audit.startswith("leak:")
    assert "src/notes.py" in outcome.leak_audit
    assert outcome.reasons and outcome.reasons[0].startswith(REASON_LEAK)


# --------------------------------------------------------------------------- #
# build_leak_report
# --------------------------------------------------------------------------- #
def test_build_report_pass_carries_prior_fields_and_sets_leak_audit() -> None:
    outcome = assess_leak(LeakAudit(findings=()), LeakAudit(findings=()))
    prior = _alt_correct_report()
    report = build_leak_report(_candidate(), prior, outcome, env_image=_env_image())
    assert report.verdict == "pass"
    assert report.leak_audit == "clean"
    # prior gate fields carried forward unchanged
    assert report.differential_pass is True
    assert report.alt_correct_accepted is True
    assert report.flakiness_runs == 3
    assert report.mutants_total == 10
    assert report.fail_to_pass == ["python -m pytest tests/hidden/test_total.py"]
    # serializable + reproducible
    again = OracleReport.from_dict(report.to_dict())
    assert again.leak_audit == "clean"
    assert again.verdict == "pass"


def test_build_report_reject_carries_reason() -> None:
    before = LeakAudit(
        findings=(
            LeakFinding(path="src/notes.py", marker="oracle_snippet", removable=False),
        )
    )
    outcome = assess_leak(before, before)
    prior = _alt_correct_report()
    report = build_leak_report(_candidate(), prior, outcome)
    assert report.verdict == "reject"
    assert any(REASON_LEAK in r for r in report.reasons)
    assert report.leak_audit.startswith("leak:")
    # reject invariant survives (de)serialization
    assert OracleReport.from_dict(report.to_dict()).verdict == "reject"


# --------------------------------------------------------------------------- #
# run_leak_gate (fake provider; no Docker)
# --------------------------------------------------------------------------- #
async def test_run_gate_clean_passes(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    provider = FakeAgentTreeProvider(tmp_path)
    report = await run_leak_gate(
        _candidate(),
        _env_image(),
        _alt_correct_report(),
        provider=provider,  # type: ignore[arg-type]
    )
    assert report.verdict == "pass"
    assert report.leak_audit == "clean"
    assert provider.opened == 1


async def test_run_gate_planted_artifact_is_sanitized(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "solution.patch", _ORACLE_PATCH)
    provider = FakeAgentTreeProvider(tmp_path)
    report = await run_leak_gate(
        _candidate(),
        _env_image(),
        _alt_correct_report(),
        provider=provider,  # type: ignore[arg-type]
    )
    assert report.verdict == "pass"
    assert report.leak_audit.startswith("sanitized:")
    assert "solution.patch" in report.leak_audit
    # the leak artifact is gone from the shipped tree
    assert not (tmp_path / "solution.patch").exists()


async def test_run_gate_embedded_oracle_rejects(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "src/notes.py", f"# hint: {_GOLD_LINE}\n")
    provider = FakeAgentTreeProvider(tmp_path)
    report = await run_leak_gate(
        _candidate(),
        _env_image(),
        _alt_correct_report(),
        provider=provider,  # type: ignore[arg-type]
    )
    assert report.verdict == "reject"
    assert any(REASON_LEAK in r for r in report.reasons)
    assert "src/notes.py" in report.leak_audit


async def test_run_gate_planted_hidden_test_body_rejects(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    _write(tmp_path, "src/fixture.py", f"DATA = '''{_HIDDEN_TEST_LINE}'''\n")
    provider = FakeAgentTreeProvider(tmp_path)
    report = await run_leak_gate(
        _candidate(),
        _env_image(),
        _alt_correct_report(),
        provider=provider,  # type: ignore[arg-type]
    )
    assert report.verdict == "reject"
    assert any(REASON_LEAK in r for r in report.reasons)


async def test_run_gate_requires_passing_prior_report(tmp_path: Path) -> None:
    _clean_tree(tmp_path)
    provider = FakeAgentTreeProvider(tmp_path)
    with pytest.raises(LeakError):
        await run_leak_gate(
            _candidate(),
            _env_image(),
            _alt_correct_report(verdict="reject"),
            provider=provider,  # type: ignore[arg-type]
        )


async def test_run_gate_requires_green_baseline(tmp_path: Path) -> None:
    from swe_forge.forge.models import BaselineNotGreenError

    _clean_tree(tmp_path)
    provider = FakeAgentTreeProvider(tmp_path)
    with pytest.raises(BaselineNotGreenError):
        await run_leak_gate(
            _candidate(),
            _env_image(green=False),
            _alt_correct_report(),
            provider=provider,  # type: ignore[arg-type]
        )


def test_agent_tree_provider_protocol_is_runtime_checkable(tmp_path: Path) -> None:
    provider = FakeAgentTreeProvider(tmp_path)
    assert isinstance(provider, AgentTreeProvider)
