"""Offline contract tests for the final pilot integrity recertification."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import swe_forge.forge.recertification as recertification
from swe_forge.forge.export import assemble_forge_task
from swe_forge.forge.models import (
    CalibrationReport,
    Candidate,
    CandidateTarget,
    EnvImage,
    FinalMutationEvidence,
    GeneratedSpec,
    ModelSolveRecord,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.differential import NullVariantSynthesizer
from swe_forge.forge.oracle.multifault import (
    RECOVERY_DUPLICATE_VALUE_TEST_COMMAND,
    ConstituentVerdict,
    MultiFaultCompletenessEvidence,
)
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.publication import PublicationEntry, PublishedGeneration
from swe_forge.forge.recertification import (
    CERTIFIED_RECOVERY_TASK_ID,
    RecertificationError,
    build_recertification_request,
    historical_recovery_spend_evidence,
    require_unchanged_suite_for_calibration,
    validate_certified_recovery_source,
    validate_certified_recovery_task,
)
from swe_forge.forge.recovery_accounting import RecoveryBudgetLedger
from swe_forge.forge.teacher import (
    TransportReceipt,
    Usage,
    candidate_transport_fingerprint,
)

_TASK_ID = CERTIFIED_RECOVERY_TASK_ID
_F2P = "python -m pytest tests/hidden/test_repair.py"
_P2P = "python -m pytest -q"


def _teacher_evidence(gate: str) -> dict[str, object]:
    return {
        "calls": [
            {
                "gate": gate,
                "call_kind": "proposal",
                "real_teacher": True,
                "status": "success",
                "response_kind": "content",
                "model": "anthropic/test-model",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "cost": 0.01,
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
        ]
    }


def _candidate() -> Candidate:
    first_inverse = "--- a/src/alpha.py\n+++ b/src/alpha.py\n@@ -1 +1 @@\n-x\n+y\n"
    second_inverse = "--- a/src/beta.py\n+++ b/src/beta.py\n@@ -1 +1 @@\n-x\n+y\n"
    return Candidate(
        language="python",
        generator="bug_combination",
        target=CandidateTarget(
            files=("src/alpha.py", "src/beta.py"), symbols=("alpha", "beta")
        ),
        mutation_patch=(
            "--- a/src/alpha.py\n+++ b/src/alpha.py\n@@ -1 +1 @@\n-x\n+y\n"
            "--- a/src/beta.py\n+++ b/src/beta.py\n@@ -1 +1 @@\n-x\n+y\n"
        ),
        oracle_patch=(
            "--- a/src/alpha.py\n+++ b/src/alpha.py\n@@ -1 +1 @@\n-y\n+x\n"
            "--- a/src/beta.py\n+++ b/src/beta.py\n@@ -1 +1 @@\n-y\n+x\n"
        ),
        difficulty_hint="high",
        provenance=Provenance(
            generator="bug_combination",
            seed=18,
            language="python",
            created_at="2026-07-10T00:00:00+00:00",
            details={
                "constituents": [
                    {
                        "index": 0,
                        "file": "src/alpha.py",
                        "mutation_patch": first_inverse.replace("-x\n+y", "-y\n+x"),
                        "inverse_patch": first_inverse,
                    },
                    {
                        "index": 1,
                        "file": "src/beta.py",
                        "mutation_patch": second_inverse.replace("-x\n+y", "-y\n+x"),
                        "inverse_patch": second_inverse,
                    },
                ]
            },
        ),
    )


def _oracle(candidate: Candidate) -> OracleReport:
    tests = [
        OracleTestFile(
            path="tests/hidden/test_repair.py",
            content="def test_repair():\n    assert True\n",
        )
    ]
    constituents = candidate.provenance.details["constituents"]
    assert isinstance(constituents, list)
    evidence = MultiFaultCompletenessEvidence(
        suite_fingerprint=final_suite_fingerprint(tests),
        p2p_command=_P2P,
        constituents=tuple(
            ConstituentVerdict(
                index=index,
                file=str(item["file"]),
                inverse_patch_sha256=hashlib.sha256(
                    str(item["inverse_patch"]).encode()
                ).hexdigest(),
                repaired_indices=(1 - index,),
                other_inverse_patches_applied=True,
                p2p_passed=True,
                failed_f2p_test_ids=(_F2P,),
                verdict="pass",
            )
            for index, item in enumerate(constituents)
        ),
    )
    report = OracleReport(
        language="python",
        generator="bug_combination",
        verdict="pass",
        reasons=[],
        fail_to_pass=[_F2P],
        pass_to_pass=[_P2P],
        test_files=tests,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=8,
        final_mutation_evidence=FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(tests),
            mutants_total=10,
            mutants_killed=8,
            threshold=0.8,
            tool="fake",
        ),
        multifault_evidence=evidence,
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        provenance=Provenance(
            generator="bug_combination",
            seed=18,
            language="python",
            created_at="2026-07-10T00:00:00+00:00",
        ),
        details={
            "teacher_gates": {
                "differential": _teacher_evidence("differential"),
                "alt_correct": _teacher_evidence("alt_correct"),
            },
            "alt_correct": {
                "public_suite_sha256": "a" * 64,
                "gold_public_suite_passed": True,
                "public_valid_alternatives": 1,
                "invalid_teacher_proposals": [],
            },
        },
        protected_alt_correct_audit={
            "version": 1,
            "original_public_suite_sha256": "a" * 64,
            "gold": {"public": {"passed": True, "exit_code": 0}},
            "alternatives": {
                "alt_1": {
                    "proposal_sha256": hashlib.sha256(
                        b"src/alpha.py\0def alpha(): ...\n\0"
                    ).hexdigest(),
                    "patches": [
                        {"path": "src/alpha.py", "content": "def alpha(): ...\n"}
                    ],
                    "public": {"passed": True, "exit_code": 0},
                    "hidden": [{"test_id": _F2P, "exit_code": 0}],
                }
            },
        },
    )
    gates = report.details["teacher_gates"]
    assert isinstance(gates, dict)
    receipts: list[dict[str, object]] = []
    for index, (gate, payload) in enumerate(gates.items(), start=1):
        assert isinstance(gate, str) and isinstance(payload, dict)
        calls = payload["calls"]
        assert isinstance(calls, list) and isinstance(calls[0], dict)
        call = calls[0]
        call["recovery_accounting"] = None
        receipt = TransportReceipt(
            call_id=f"{index:032x}",
            candidate_fingerprint=candidate_transport_fingerprint(candidate),
            gate=gate,
            call_kind=str(call["call_kind"]),
            model=str(call["model"]),
            usage=Usage(**call["usage"]),  # type: ignore[arg-type]
            cost=float(call["cost"]),
            receipt_secret=f"{index:064x}",
        )
        call["call_id"] = receipt.call_id
        call["receipt_commitment"] = receipt.commitment
        receipts.append(receipt.to_private_dict())
    report.protected_teacher_transport_receipts = receipts
    return report


def _calibration() -> CalibrationReport:
    report = CalibrationReport(
        language="python",
        models=[],
        k=0,
        irt_difficulty=1.2,
        irt_discrimination=4.7,
        details={
            "band_filter": {"band_high": 0.5},
            "usage_accounting": {
                "validation": {
                    "calls": 0,
                    "usage": Usage().to_dict(),
                    "cost": 0.0,
                    "per_call": [],
                },
                "rollout": {
                    "calls": 0,
                    "usage": Usage().to_dict(),
                    "cost": 0.0,
                    "per_call": [],
                },
                "aggregate": {
                    "total_calls": 0,
                    "usage": Usage().to_dict(),
                    "cost": 0.0,
                },
            },
            "recovery_accounting": [],
        },
    )
    report.set_band_verdict("keep", "genuine in-band keep")
    return report


def _task():
    candidate = _candidate()
    return assemble_forge_task(
        candidate=candidate,
        spec=GeneratedSpec(
            problem_statement="Repair both public behaviors.",
            requirements=["Both hidden checks must pass."],
            interface_block="def alpha(): ...\ndef beta(): ...",
            provenance=Provenance(
                generator="bug_combination",
                seed=18,
                language="python",
                created_at="2026-07-10T00:00:00+00:00",
            ),
        ),
        oracle_report=_oracle(candidate),
        calibration_report=_calibration(),
        env_image=EnvImage(
            repo_id="mahmoud-boltons",
            language="python",
            image_tag="swe-forge-env:certified",
            base_image="python:3.12-slim",
            commit="a" * 40,
            workspace_dir="/workspace/repo",
            install_commands=["pip install pytest"],
            baseline_test_command=_P2P,
            baseline_green=True,
            baseline_exit_code=0,
        ),
        repo_url="https://github.com/mahmoud/boltons.git",
        task_id=_TASK_ID,
    )


def _ledger(tmp_path: Path) -> RecoveryBudgetLedger:
    return RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="test-recovery",
        cap_usd=1,
        worst_case_cost_usd=0.1,
    )


def _fresh_metered_calibration(
    task: object, ledger: RecoveryBudgetLedger
) -> CalibrationReport:
    """Attach every recovery oracle/calibration call to exact ledger records."""
    calibration = task.calibration_report  # type: ignore[attr-defined]
    oracle = task.oracle_report  # type: ignore[attr-defined]
    teacher_gates = oracle.details["teacher_gates"]
    assert isinstance(teacher_gates, dict)
    for gate in ("differential", "alt_correct"):
        evidence = teacher_gates[gate]
        assert isinstance(evidence, dict)
        calls = evidence["calls"]
        assert isinstance(calls, list) and len(calls) == 1
        call = calls[0]
        assert isinstance(call, dict)
        logical_call_id = f"oracle-{gate}-proposal"
        physical_id = ledger.reserve(
            logical_call_id=logical_call_id,
            stage=f"oracle.{gate}",
            model="anthropic/test-model",
            retry=0,
        )
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        ledger.settle(
            physical_id,
            request_id=f"{gate}-request",
            usage=usage,
            cost=0.01,
            status="success",
            finish_reason="stop",
        )
        settled = next(
            item
            for item in ledger.settled_calls()
            if item["physical_call_id"] == physical_id
        )
        call["recovery_accounting"] = {
            "logical_call_id": logical_call_id,
            "physical_calls": [settled],
        }
    calibration.models = [ModelSolveRecord("mid/test", "mid", 1, 1, 1.0)]
    calibration.k = 1
    validation_usage = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    rollout_usage = Usage(prompt_tokens=2, completion_tokens=2, total_tokens=4)

    validation_id = ledger.reserve(
        logical_call_id="calibration-validation",
        stage="calibration.validation",
        model="mid/test",
        retry=0,
    )
    ledger.settle(
        validation_id,
        request_id="validation-request",
        usage=validation_usage,
        cost=0.01,
        status="success",
        finish_reason="stop",
    )
    rollout_id = ledger.reserve(
        logical_call_id="calibration-rollout",
        stage="calibration.rollout",
        model="mid/test",
        retry=0,
    )
    ledger.settle(
        rollout_id,
        request_id="rollout-request",
        usage=rollout_usage,
        cost=0.01,
        status="success",
        finish_reason="stop",
    )
    settled = {call["physical_call_id"]: call for call in ledger.settled_calls()}
    validation_evidence = {
        "logical_call_id": "calibration-validation",
        "physical_calls": [settled[validation_id]],
    }
    rollout_evidence = {
        "logical_call_id": "calibration-rollout",
        "physical_calls": [settled[rollout_id]],
    }
    calibration.details["usage_accounting"] = {
        "validation": {
            "calls": 1,
            "usage": validation_usage.to_dict(),
            "cost": 0.01,
            "per_call": [
                {
                    "model": "mid/test",
                    "valid": True,
                    "usage": validation_usage.to_dict(),
                    "cost": 0.01,
                    "recovery_accounting": validation_evidence,
                }
            ],
        },
        "rollout": {
            "calls": 1,
            "usage": rollout_usage.to_dict(),
            "cost": 0.01,
            "per_call": [
                {
                    "model": "mid/test",
                    "tier": "mid",
                    "index": 0,
                    "solved": True,
                    "usage": rollout_usage.to_dict(),
                    "cost": 0.01,
                    "recovery_accounting": [rollout_evidence],
                }
            ],
        },
        "aggregate": {
            "total_calls": 2,
            "usage": (validation_usage + rollout_usage).to_dict(),
            "cost": 0.02,
        },
    }
    calibration.details["recovery_accounting"] = [
        validation_evidence,
        rollout_evidence,
    ]
    return calibration


def test_certified_recovery_requires_exact_genuine_keep() -> None:
    task = _task()

    validate_certified_recovery_task(task)

    task.calibration_report.set_band_verdict("drop", "too easy")
    with pytest.raises(RecertificationError, match="calibration keep"):
        validate_certified_recovery_task(task)


def test_legacy_source_can_be_recertified_without_stale_teacher_evidence() -> None:
    task = _task()
    task.oracle_report.details.pop("teacher_gates")

    validate_certified_recovery_source(task)
    with pytest.raises(RecertificationError, match="teacher evidence"):
        validate_certified_recovery_task(task)


def test_historical_recovery_spend_is_lower_bound_not_budget_authority() -> None:
    evidence = historical_recovery_spend_evidence()

    assert evidence["historical_observed_lower_bound_usd"] == "21.43272865"
    assert evidence["is_exact"] is False
    assert evidence["can_prove_cap"] is False
    assert evidence["can_authorize_publication"] is False


def test_changed_final_suite_invalidates_prior_calibration() -> None:
    task = _task()
    changed = _oracle(task.candidate)
    changed.test_files.append(
        OracleTestFile(path="tests/hidden/test_new.py", content="assert True\n")
    )
    changed.final_mutation_evidence = FinalMutationEvidence(
        suite_fingerprint=final_suite_fingerprint(changed.test_files),
        mutants_total=10,
        mutants_killed=8,
        threshold=0.8,
        tool="fake",
    )

    with pytest.raises(RecertificationError, match="recalibration"):
        require_unchanged_suite_for_calibration(task.oracle_report, changed)


def test_recertification_export_preserves_the_single_valid_task_id(
    tmp_path: Path,
) -> None:
    task = _task()
    ledger = _ledger(tmp_path)
    _fresh_metered_calibration(task, ledger)
    generation = PublishedGeneration(
        generation_id="genuine",
        root=Path("/tmp/genuine"),
        tasks_dir=Path("/tmp/genuine/tasks"),
        jsonl_path=Path("/tmp/genuine/dataset.jsonl"),
        parquet_path=Path("/tmp/genuine/dataset.parquet"),
        entries=(PublicationEntry(index=0, task=task),),
    )

    request = build_recertification_request(
        generation, task.oracle_report, recovery_ledger=ledger
    )

    assert request.task_id == _TASK_ID
    assert request.calibration_report is task.calibration_report
    assert request.oracle_report is task.oracle_report


def test_recertification_refuses_publication_without_durable_ledger() -> None:
    task = _task()
    generation = PublishedGeneration(
        generation_id="genuine",
        root=Path("/tmp/genuine"),
        tasks_dir=Path("/tmp/genuine/tasks"),
        jsonl_path=Path("/tmp/genuine/dataset.jsonl"),
        parquet_path=Path("/tmp/genuine/dataset.parquet"),
        entries=(PublicationEntry(index=0, task=task),),
    )

    with pytest.raises(RecertificationError, match="durable budget ledger"):
        build_recertification_request(generation, task.oracle_report)


def test_recertification_blocks_unreconciled_ledger_calls(tmp_path: Path) -> None:
    task = _task()
    ledger = _ledger(tmp_path)
    _fresh_metered_calibration(task, ledger)
    generation = PublishedGeneration(
        generation_id="genuine",
        root=Path("/tmp/genuine"),
        tasks_dir=Path("/tmp/genuine/tasks"),
        jsonl_path=Path("/tmp/genuine/dataset.jsonl"),
        parquet_path=Path("/tmp/genuine/dataset.parquet"),
        entries=(PublicationEntry(index=0, task=task),),
    )
    physical = ledger.reserve(
        logical_call_id="unlinked",
        stage="oracle.differential",
        model="anthropic/test-model",
        retry=0,
    )
    ledger.settle(
        physical,
        request_id="provider-unlinked",
        usage=Usage(),
        cost=0,
        status="error",
        error_type="RuntimeError",
    )

    with pytest.raises(RecertificationError, match="accounting cannot authorize"):
        build_recertification_request(
            generation, task.oracle_report, recovery_ledger=ledger
        )


def test_recertification_blocks_calibration_only_oracle_accounting(
    tmp_path: Path,
) -> None:
    """Calibration ledger records cannot authorize an unlinked real oracle call."""
    task = _task()
    ledger = _ledger(tmp_path)
    _fresh_metered_calibration(task, ledger)
    teacher_gates = task.oracle_report.details["teacher_gates"]
    assert isinstance(teacher_gates, dict)
    differential = teacher_gates["differential"]
    assert isinstance(differential, dict)
    calls = differential["calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    calls[0]["recovery_accounting"] = None
    generation = PublishedGeneration(
        generation_id="genuine",
        root=Path("/tmp/genuine"),
        tasks_dir=Path("/tmp/genuine/tasks"),
        jsonl_path=Path("/tmp/genuine/dataset.jsonl"),
        parquet_path=Path("/tmp/genuine/dataset.parquet"),
        entries=(PublicationEntry(index=0, task=task),),
    )

    with pytest.raises(
        RecertificationError, match="accounting cannot authorize publication"
    ):
        build_recertification_request(
            generation, task.oracle_report, recovery_ledger=ledger
        )


def test_recertification_blocks_unknown_billing_reservation(tmp_path: Path) -> None:
    """An unresolved post-send reservation can never authorize publication."""
    task = _task()
    ledger = _ledger(tmp_path)
    _fresh_metered_calibration(task, ledger)
    generation = PublishedGeneration(
        generation_id="genuine",
        root=Path("/tmp/genuine"),
        tasks_dir=Path("/tmp/genuine/tasks"),
        jsonl_path=Path("/tmp/genuine/dataset.jsonl"),
        parquet_path=Path("/tmp/genuine/dataset.parquet"),
        entries=(PublicationEntry(index=0, task=task),),
    )
    ledger.reserve(
        logical_call_id="unknown-billing",
        stage="oracle.differential",
        model="anthropic/test-model",
        retry=0,
    )

    with pytest.raises(RecertificationError, match="unsettled physical calls"):
        build_recertification_request(
            generation, task.oracle_report, recovery_ledger=ledger
        )


def test_recertification_rejects_calibration_without_per_call_ledger_linkage(
    tmp_path: Path,
) -> None:
    task = _task()
    ledger = _ledger(tmp_path)
    _fresh_metered_calibration(task, ledger)
    generation = PublishedGeneration(
        generation_id="genuine",
        root=Path("/tmp/genuine"),
        tasks_dir=Path("/tmp/genuine/tasks"),
        jsonl_path=Path("/tmp/genuine/dataset.jsonl"),
        parquet_path=Path("/tmp/genuine/dataset.parquet"),
        entries=(PublicationEntry(index=0, task=task),),
    )
    validation_rows = task.calibration_report.details["usage_accounting"]["validation"][
        "per_call"
    ]
    assert isinstance(validation_rows, list)
    assert isinstance(validation_rows[0], dict)
    validation_rows[0]["recovery_accounting"] = None

    with pytest.raises(RecertificationError, match="no physical ledger evidence"):
        build_recertification_request(
            generation, task.oracle_report, recovery_ledger=ledger
        )


def test_recertification_requires_direct_calibration_accounting_to_match_rows(
    tmp_path: Path,
) -> None:
    task = _task()
    ledger = _ledger(tmp_path)
    _fresh_metered_calibration(task, ledger)
    generation = PublishedGeneration(
        generation_id="genuine",
        root=Path("/tmp/genuine"),
        tasks_dir=Path("/tmp/genuine/tasks"),
        jsonl_path=Path("/tmp/genuine/dataset.jsonl"),
        parquet_path=Path("/tmp/genuine/dataset.parquet"),
        entries=(PublicationEntry(index=0, task=task),),
    )
    task.calibration_report.details["recovery_accounting"] = []

    with pytest.raises(RecertificationError, match="does not match its per-call"):
        build_recertification_request(
            generation, task.oracle_report, recovery_ledger=ledger
        )


def test_recertification_reruns_teacher_gates_before_final_suite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task()
    ledger = _ledger(tmp_path)
    _fresh_metered_calibration(task, ledger)
    calls: list[str] = []
    variant_generator = object()
    alt_generator = object()

    async def _establish(candidate, env_image, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("establish")
        assert candidate is task.candidate
        assert env_image is task.env_image
        frozen = kwargs["provided_tests"]
        assert any(
            test.test_id == RECOVERY_DUPLICATE_VALUE_TEST_COMMAND for test in frozen
        )
        return task.oracle_report

    async def _flakiness(candidate, env_image, report, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("flakiness")
        assert candidate is task.candidate
        assert env_image is task.env_image
        assert report is task.oracle_report
        return report

    async def _differential(candidate, env_image, report, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("differential")
        assert candidate is task.candidate
        assert env_image is task.env_image
        assert report is not task.oracle_report
        assert kwargs["variant_generator"] is variant_generator
        assert isinstance(kwargs["synthesizer"], NullVariantSynthesizer)
        return report

    async def _alt_correct(candidate, env_image, report, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("alt_correct")
        assert candidate is task.candidate
        assert env_image is task.env_image
        assert kwargs["spec"] is task.spec
        assert kwargs["alt_generator"] is alt_generator
        return report

    async def _final_mutation(candidate, env_image, report, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("final_mutation")
        assert report is not task.oracle_report
        return report

    async def _multifault(candidate, env_image, report, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("multifault")
        assert report is not task.oracle_report
        return report

    async def _leak(candidate, env_image, report, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("leak")
        assert report is not task.oracle_report
        return report

    monkeypatch.setattr(
        recertification, "run_differential_gate", _differential, raising=False
    )
    monkeypatch.setattr(recertification, "run_establish_gate", _establish)
    monkeypatch.setattr(recertification, "run_flakiness_gate", _flakiness)
    monkeypatch.setattr(
        recertification, "run_alt_correct_gate", _alt_correct, raising=False
    )
    monkeypatch.setattr(recertification, "run_final_mutation_gate", _final_mutation)
    monkeypatch.setattr(
        recertification, "run_multifault_completeness_gate", _multifault
    )
    monkeypatch.setattr(
        recertification, "verify_recovery_duplicate_value_proof", lambda _report: []
    )
    monkeypatch.setattr(
        recertification, "verify_pass_consistency", lambda *args, **kw: []
    )
    monkeypatch.setattr(
        recertification, "verify_multifault_evidence", lambda *args, **kw: []
    )
    monkeypatch.setattr(recertification, "run_leak_gate", _leak)

    result = asyncio.run(
        recertification.recertify_final_oracle(
            task,
            recovery_ledger=_ledger(tmp_path),
            variant_generator=variant_generator,  # type: ignore[arg-type]
            alt_generator=alt_generator,  # type: ignore[arg-type]
        )
    )

    assert result is not task.oracle_report
    assert calls == [
        "establish",
        "flakiness",
        "multifault",
        "differential",
        "alt_correct",
        "final_mutation",
        "multifault",
        "leak",
    ]


def test_recertification_transactionally_supersedes_stale_same_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = _task()
    ledger = _ledger(tmp_path)
    _fresh_metered_calibration(task, ledger)
    generation = PublishedGeneration(
        generation_id="genuine",
        root=tmp_path / "source",
        tasks_dir=tmp_path / "source" / "tasks",
        jsonl_path=tmp_path / "source" / "dataset.jsonl",
        parquet_path=tmp_path / "source" / "dataset.parquet",
        entries=(PublicationEntry(index=0, task=task),),
    )
    captured: dict[str, object] = {}

    async def _recertify(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert args[0].env_image.original_public_test_command == "python -m pytest -q"
        return task.oracle_report

    async def _recalibrate(*args, **kwargs):  # type: ignore[no-untyped-def]
        return task.calibration_report

    def _export(requests, out_dir, **kwargs):  # type: ignore[no-untyped-def]
        captured["requests"] = requests
        captured["out_dir"] = out_dir
        captured.update(kwargs)
        return SimpleNamespace(kept=[object()], refused=[])

    monkeypatch.setattr(
        recertification,
        "load_certified_recovery",
        lambda source_dir: (generation, task),
    )
    monkeypatch.setattr(recertification, "recertify_final_oracle", _recertify)
    monkeypatch.setattr(recertification, "export_batch", _export)

    result = asyncio.run(
        recertification.recertify_recovery_export(
            tmp_path,
            recovery_ledger=ledger,
            recalibrator=_recalibrate,
            public_suite_command="python -m pytest -q",
        )
    )

    assert result.task_id == _TASK_ID
    assert captured["out_dir"] == tmp_path
    assert captured["overwrite"] is True
    assert captured["replace_existing"] is True
    requests = captured["requests"]
    assert isinstance(requests, list)
    assert requests[0].task_id == _TASK_ID
    assert requests[0].env_image.original_public_test_command == "python -m pytest -q"
