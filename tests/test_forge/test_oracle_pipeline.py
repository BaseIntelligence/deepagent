"""Unit tests for the oracle pipeline orchestration (m4-pipeline).

Offline coverage (no real Docker) of the pipeline's verdict/ordering/export
contract, driven over injected fake gate steps that return canned
:class:`OracleReport`s so the orchestration logic is exercised deterministically:

- VAL-ORACLE-016: ``verdict == pass`` only when every gate passes with consistent
  fields (empty reasons, F2P non-empty, flakiness_runs >= 3, kill ratio >=
  threshold, differential_pass, alt_correct_accepted, clean leak_audit).
- VAL-ORACLE-017: any single gate failure -> ``verdict == reject`` with a
  non-empty attributable reason.
- VAL-ORACLE-018: gate ordering honored; on an early failure the later gates do
  not run and their fields are not spuriously credited.
- VAL-ORACLE-019: a reject (or oracle-pass + calibration-drop) is never
  exportable; oracle-pass + calibration-keep is.
- VAL-ORACLE-020: the OracleReport serializes and reproduces on the same inputs.

The full Docker-backed pipeline (incl. the mutation gate live on py/js/go) is
exercised by this feature's manual verification and the user-testing validator.
"""

from __future__ import annotations

import pytest

import swe_forge.forge.oracle.pipeline as pipeline_module
from swe_forge.forge.models import (
    BaselineNotGreenError,
    Candidate,
    CandidateTarget,
    EnvImage,
    FinalMutationEvidence,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.pipeline import (
    GATE_ORDER,
    REASON_PIPELINE_INCONSISTENT,
    ExportRefusedError,
    OraclePipelineError,
    build_default_gates,
    ensure_oracle_exportable,
    is_oracle_exportable,
    orchestrate_gates,
    run_oracle_pipeline,
    verify_pass_consistency,
)
from swe_forge.forge.oracle.mutation import final_suite_fingerprint

_FIXED_TS = "2026-01-01T00:00:00+00:00"
_F2P = ["python -m pytest tests/hidden/test_total.py"]


def _teacher_gate_evidence() -> dict[str, object]:
    def call(gate: str) -> dict[str, object]:
        return {
            "gate": gate,
            "call_kind": "proposal",
            "real_teacher": True,
            "status": "success",
            "response_kind": "content",
            "model": "anthropic/test-teacher",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "cost": 0.0,
            "finish_reason": "stop",
            "requested_proposals": 1,
            "received_proposals": 1,
            "parsed_proposals": 1,
            "identical_proposals": 0,
            "invalid_proposals": 0,
            "discarded_proposals": 0,
            "execution_attempted": 1,
            "execution_completed": 1,
            "execution_errors": 0,
            "executable_proposals": 1,
            "error_type": "",
        }

    return {
        "differential": {"calls": [call("differential")]},
        "alt_correct": {"calls": [call("alt_correct")]},
    }


def _alt_correct_audit() -> dict[str, object]:
    return {
        "version": 1,
        "original_public_suite_sha256": "a" * 64,
        "gold": {
            "public": {"passed": True, "exit_code": 0},
            "filtered_p2p": {"passed": True, "exit_code": 0},
            "hidden": [{"test_id": _F2P[0], "exit_code": 0}],
        },
        "alternatives": {
            "alt_1": {
                "proposal_sha256": "b" * 64,
                "patches": [
                    {"path": "src/m.py", "content": "def total(xs): return sum(xs)\n"}
                ],
                "public": {"passed": True, "exit_code": 0},
                "filtered_p2p": {"passed": True, "exit_code": 0},
                "hidden": [{"test_id": _F2P[0], "exit_code": 0}],
            }
        },
    }


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _candidate() -> Candidate:
    return Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("src/m.py",), symbols=("total",)),
        mutation_patch=(
            "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,2 +1,2 @@\n"
            " def total(xs):\n-    return sum(xs)\n+    return sum(xs) + 1\n"
        ),
        oracle_patch=(
            "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,2 +1,2 @@\n"
            " def total(xs):\n-    return sum(xs) + 1\n+    return sum(xs)\n"
        ),
        difficulty_hint="medium",
        provenance=Provenance(
            generator="ast_mutation", seed=7, language="python", created_at=_FIXED_TS
        ),
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


def _provenance() -> Provenance:
    return Provenance(
        generator="ast_mutation", seed=7, language="python", created_at=_FIXED_TS
    )


def _pass_report(**overrides: object) -> OracleReport:
    """A fully consistent pass report (the shape the final leak gate yields)."""
    test_files = [OracleTestFile(path="tests/hidden/test_total.py", content="x = 1\n")]
    fields: dict[str, object] = {
        "language": "python",
        "generator": "ast_mutation",
        "verdict": "pass",
        "reasons": [],
        "fail_to_pass": list(_F2P),
        "pass_to_pass": ["python -m pytest"],
        "test_files": test_files,
        "flakiness_runs": 3,
        "mutants_total": 10,
        "mutants_killed": 10,
        "differential_pass": True,
        "alt_correct_accepted": True,
        "leak_audit": "clean",
        "final_mutation_evidence": FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(test_files),
            mutants_total=10,
            mutants_killed=10,
            threshold=0.8,
            tool="fake-tool",
        ),
        "provenance": _provenance(),
        "details": {
            "teacher_gates": _teacher_gate_evidence(),
            "alt_correct": {
                "public_suite_sha256": "a" * 64,
                "gold_public_suite_passed": True,
                "public_valid_alternatives": 1,
                "invalid_teacher_proposals": [],
            },
        },
        "protected_alt_correct_audit": _alt_correct_audit(),
    }
    fields.update(overrides)
    return OracleReport(**fields)  # type: ignore[arg-type]


def _reject_report(gate: str) -> OracleReport:
    """A reject report as the given gate would emit it (downstream fields default)."""
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="reject",
        reasons=[f"{gate}_failed: induced single-gate failure for {gate}"],
        provenance=_provenance(),
    )


class RecordingGate:
    """A fake :data:`GateStep` that records invocation and returns a canned report."""

    def __init__(self, name: str, report: OracleReport, calls: list[str]) -> None:
        self.name = name
        self._report = report
        self._calls = calls

    async def __call__(self, prior: OracleReport | None) -> OracleReport:
        self._calls.append(self.name)
        return self._report


def _gates(calls: list[str], *, failing: str | None = None) -> list[tuple[str, object]]:
    """Build a full ordered gate list; ``failing`` (if set) emits a reject there."""
    steps: list[tuple[str, object]] = []
    for name in GATE_ORDER:
        if name == failing:
            report = _reject_report(name)
        elif name == GATE_ORDER[-1]:
            report = _pass_report()
        else:
            report = _pass_report(leak_audit="", differential_pass=False)
        steps.append((name, RecordingGate(name, report, calls)))
    return steps


# --------------------------------------------------------------------------- #
# VAL-ORACLE-016: pass only when ALL gates pass + fields consistent
# --------------------------------------------------------------------------- #
async def test_all_gates_pass_yields_pass() -> None:
    calls: list[str] = []
    report = await orchestrate_gates(_gates(calls))
    assert report.verdict == "pass"
    assert report.reasons == []
    assert report.fail_to_pass == _F2P
    assert report.flakiness_runs >= 3
    assert report.mutants_total > 0
    assert report.mutants_killed / report.mutants_total >= 0.8
    assert report.differential_pass is True
    assert report.alt_correct_accepted is True
    assert report.leak_audit == "clean"
    # every gate ran, in order
    assert calls == list(GATE_ORDER)
    assert report.details["pipeline"]["failed_gate"] is None
    assert report.details["pipeline"]["gates_run"] == list(GATE_ORDER)


@pytest.mark.parametrize(
    "bad",
    [
        {"fail_to_pass": []},
        {"flakiness_runs": 2},
        {"mutants_total": 0, "mutants_killed": 0},
        {"mutants_total": 10, "mutants_killed": 3},
        {"final_mutation_evidence": None},
        {
            "final_mutation_evidence": FinalMutationEvidence(
                suite_fingerprint="f" * 64,
                mutants_total=10,
                mutants_killed=10,
                threshold=0.8,
                tool="fake-tool",
            )
        },
        {"differential_pass": False},
        {"alt_correct_accepted": False},
        {"leak_audit": "leak: src/x.py"},
    ],
)
async def test_inconsistent_pass_is_demoted_to_reject(bad: dict[str, object]) -> None:
    calls: list[str] = []
    steps: list[tuple[str, object]] = []
    for name in GATE_ORDER:
        report = _pass_report(**bad) if name == GATE_ORDER[-1] else _pass_report()
        steps.append((name, RecordingGate(name, report, calls)))
    result = await orchestrate_gates(steps)
    assert result.verdict == "reject"
    assert result.reasons
    assert all(r.startswith(REASON_PIPELINE_INCONSISTENT) for r in result.reasons)
    assert result.details["pipeline"]["failed_gate"] == "consistency"


def test_verify_pass_consistency_clean() -> None:
    assert verify_pass_consistency(_pass_report()) == []


def test_verify_pass_consistency_requires_protected_public_validity_audit() -> None:
    report = _pass_report()
    report.protected_alt_correct_audit = None

    assert any(
        "protected public-validity audit" in problem
        for problem in verify_pass_consistency(report)
    )
    with pytest.raises(ExportRefusedError):
        ensure_oracle_exportable(report, calibration_kept=True)


def test_custom_final_mutation_threshold_is_exportable() -> None:
    report = _pass_report()
    evidence = report.final_mutation_evidence
    assert evidence is not None
    report.final_mutation_evidence = FinalMutationEvidence(
        suite_fingerprint=evidence.suite_fingerprint,
        mutants_total=evidence.mutants_total,
        mutants_killed=evidence.mutants_killed,
        threshold=0.9,
        tool=evidence.tool,
    )

    assert verify_pass_consistency(report, kill_threshold=0.9) == []
    ensure_oracle_exportable(report, calibration_kept=True)


def test_verify_pass_consistency_rejects_stale_final_mutation_counts() -> None:
    report = _pass_report()
    report.mutants_killed = 9
    assert any(
        "replacement counts" in problem for problem in verify_pass_consistency(report)
    )


def test_verify_pass_consistency_ignores_reject() -> None:
    # a reject report is not policed by the consistency check
    assert verify_pass_consistency(_reject_report("establish")) == []


# --------------------------------------------------------------------------- #
# VAL-ORACLE-017: any single gate failure -> reject with attributable reasons
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("failing", list(GATE_ORDER))
async def test_single_gate_failure_rejects(failing: str) -> None:
    calls: list[str] = []
    report = await orchestrate_gates(_gates(calls, failing=failing))
    assert report.verdict == "reject"
    assert len(report.reasons) >= 1
    assert any(failing in r for r in report.reasons)
    assert report.details["pipeline"]["failed_gate"] == failing


# --------------------------------------------------------------------------- #
# VAL-ORACLE-018: ordering honored; later gates not credited on early failure
# --------------------------------------------------------------------------- #
async def test_early_failure_skips_later_gates() -> None:
    calls: list[str] = []
    report = await orchestrate_gates(_gates(calls, failing="establish"))
    # only the establish gate ran
    assert calls == ["establish"]
    assert report.details["pipeline"]["gates_run"] == ["establish"]
    # downstream fields are NOT spuriously true
    assert report.differential_pass is False
    assert report.alt_correct_accepted is False
    assert report.leak_audit == ""
    assert report.flakiness_runs == 0
    assert report.mutants_total == 0


async def test_failure_cites_earliest_failed_gate() -> None:
    calls: list[str] = []
    # make BOTH mutation and differential "fail"; mutation is earlier, so it wins
    steps: list[tuple[str, object]] = []
    for name in GATE_ORDER:
        if name in ("mutation", "differential"):
            report = _reject_report(name)
        elif name == GATE_ORDER[-1]:
            report = _pass_report()
        else:
            report = _pass_report(leak_audit="", differential_pass=False)
        steps.append((name, RecordingGate(name, report, calls)))
    report = await orchestrate_gates(steps)
    assert report.details["pipeline"]["failed_gate"] == "mutation"
    assert "differential" not in calls
    assert any("mutation" in r for r in report.reasons)


async def test_empty_gates_raises() -> None:
    with pytest.raises(OraclePipelineError):
        await orchestrate_gates([])


# --------------------------------------------------------------------------- #
# VAL-ORACLE-020: serializable + reproducible
# --------------------------------------------------------------------------- #
async def test_report_round_trips_through_json() -> None:
    calls: list[str] = []
    report = await orchestrate_gates(_gates(calls))
    again = OracleReport.from_dict(report.to_dict())
    assert again.verdict == "pass"
    assert again.fail_to_pass == report.fail_to_pass
    assert again.differential_pass == report.differential_pass
    assert again.leak_audit == report.leak_audit


async def test_pipeline_reproduces_on_same_inputs() -> None:
    first = await orchestrate_gates(_gates([]))
    second = await orchestrate_gates(_gates([]))
    assert first.to_dict() == second.to_dict()


async def test_reject_reproduces_on_same_inputs() -> None:
    first = await orchestrate_gates(_gates([], failing="differential"))
    second = await orchestrate_gates(_gates([], failing="differential"))
    assert first.to_dict() == second.to_dict()
    assert first.verdict == "reject"


# --------------------------------------------------------------------------- #
# run_oracle_pipeline wiring + green-baseline precondition
# --------------------------------------------------------------------------- #
async def test_run_pipeline_threads_injected_gates() -> None:
    calls: list[str] = []
    report = await run_oracle_pipeline(_candidate(), _env_image(), gates=_gates(calls))
    assert report.verdict == "pass"
    assert calls == list(GATE_ORDER)


async def test_run_pipeline_reject_through_injected_gates() -> None:
    calls: list[str] = []
    report = await run_oracle_pipeline(
        _candidate(), _env_image(), gates=_gates(calls, failing="leak")
    )
    assert report.verdict == "reject"
    assert any("leak" in r for r in report.reasons)


async def test_default_pipeline_final_remeasures_after_alt_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def stub(name: str):
        async def _run(*_args: object, **_kwargs: object) -> OracleReport:
            calls.append(name)
            return _pass_report()

        return _run

    async def final_mutation(*_args: object, **kwargs: object) -> OracleReport:
        calls.append("final_mutation")
        # The final gate deliberately has no synthesis parameter, so additions
        # cannot alter the suite after it is fingerprinted.
        assert "synthesizer" not in kwargs
        return _pass_report()

    monkeypatch.setattr(pipeline_module, "run_establish_gate", stub("establish"))
    monkeypatch.setattr(pipeline_module, "run_flakiness_gate", stub("flakiness"))
    monkeypatch.setattr(pipeline_module, "run_mutation_gate", stub("mutation"))
    monkeypatch.setattr(pipeline_module, "run_differential_gate", stub("differential"))
    monkeypatch.setattr(pipeline_module, "run_alt_correct_gate", stub("alt_correct"))
    monkeypatch.setattr(pipeline_module, "run_final_mutation_gate", final_mutation)
    monkeypatch.setattr(pipeline_module, "run_leak_gate", stub("leak"))

    report = await run_oracle_pipeline(_candidate(), _env_image())

    assert report.verdict == "pass"
    assert calls == [
        "establish",
        "flakiness",
        "mutation",
        "differential",
        "alt_correct",
        "final_mutation",
        "leak",
    ]


async def test_run_pipeline_requires_green_baseline() -> None:
    with pytest.raises(BaselineNotGreenError):
        await run_oracle_pipeline(
            _candidate(), _env_image(green=False), gates=_gates([])
        )


# --------------------------------------------------------------------------- #
# build_default_gates wiring (offline: only structure is asserted)
# --------------------------------------------------------------------------- #
def test_build_default_gates_order_and_names() -> None:
    gates = build_default_gates(_candidate(), _env_image())
    assert [name for name, _ in gates] == list(GATE_ORDER)
    assert all(callable(step) for _, step in gates)


# --------------------------------------------------------------------------- #
# VAL-ORACLE-019: reject never exports; pass needs calibration keep too
# --------------------------------------------------------------------------- #
def test_reject_is_not_exportable() -> None:
    report = _reject_report("mutation")
    assert is_oracle_exportable(report) is False
    with pytest.raises(ExportRefusedError):
        ensure_oracle_exportable(report)


def test_pass_is_oracle_exportable_but_needs_calibration_keep() -> None:
    report = _pass_report()
    assert is_oracle_exportable(report) is True
    # oracle-pass alone (calibration not yet run) satisfies the necessary cond.
    ensure_oracle_exportable(report)
    ensure_oracle_exportable(report, calibration_kept=True)
    # oracle-pass + calibration drop -> still refused
    with pytest.raises(ExportRefusedError):
        ensure_oracle_exportable(report, calibration_kept=False)


def test_reject_refused_even_with_calibration_keep() -> None:
    report = _reject_report("flakiness")
    with pytest.raises(ExportRefusedError):
        ensure_oracle_exportable(report, calibration_kept=True)
