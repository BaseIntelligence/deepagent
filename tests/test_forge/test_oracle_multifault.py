"""Fail-closed oracle evidence for structural multi-fault candidates."""

from __future__ import annotations

import hashlib

from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    FinalMutationEvidence,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.multifault import (
    FullGoldScore,
    MultiFaultCompletenessEvidence,
    PartialRepairScore,
    RECOVERY_DUPLICATE_VALUE_TEST_COMMAND,
    RECOVERY_DUPLICATE_VALUE_TEST_NODE,
    TestStateExit,
    assess_multifault_completeness,
    normalize_constituent_inverse_patches,
    strengthen_recovery_duplicate_value_invariant,
    verify_recovery_duplicate_value_proof,
)
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.oracle.pipeline import verify_pass_consistency

P2P = "python -m pytest -q"
F2P = "python -m pytest tests/hidden/test_total.py"


def _candidate(
    *,
    generator: str = "bug_combination",
    metadata: object | None = None,
) -> Candidate:
    constituents = (
        metadata
        if metadata is not None
        else [
            {
                "index": 0,
                "file": "src/alpha.py",
                "mutation_patch": "mutation alpha",
                "inverse_patch": "inverse alpha",
                "single_fault_revert": "inverse alpha",
            },
            {
                "index": 1,
                "file": "src/beta.py",
                "mutation_patch": "mutation beta",
                "inverse_patch": "inverse beta",
                "single_fault_revert": "inverse beta",
            },
        ]
    )
    details: dict[str, object] = {}
    if metadata is not None or constituents:
        details["constituents"] = constituents
    return Candidate(
        language="python",
        generator=generator,
        target=CandidateTarget(
            files=("src/alpha.py", "src/beta.py"), symbols=("alpha", "beta")
        ),
        mutation_patch="combined mutation",
        oracle_patch="combined oracle",
        difficulty_hint="high",
        provenance=Provenance(
            generator=generator,
            seed=7,
            language="python",
            details=details,
        ),
    )


def _tests() -> list[OracleTestFile]:
    return [
        OracleTestFile(
            path="tests/hidden/test_total.py",
            content="def test_total():\n    assert True\n",
        )
    ]


class FakePartialRepairRunner:
    def __init__(
        self,
        *,
        p2p_passed: bool = True,
        failing_f2p: tuple[str, ...] = (F2P,),
        applied: bool = True,
    ) -> None:
        self.p2p_passed = p2p_passed
        self.failing_f2p = failing_f2p
        self.applied = applied
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    async def score(
        self,
        leave_broken,
        repairs,
        tests,
        *,
        p2p_command: str,
    ) -> PartialRepairScore:
        self.calls.append((leave_broken.index, tuple(item.index for item in repairs)))
        assert p2p_command == P2P
        assert tests
        return PartialRepairScore(
            other_inverse_patches_applied=self.applied,
            p2p_passed=self.p2p_passed,
            failed_f2p_test_ids=self.failing_f2p,
        )


class RecoveryProofRunner(FakePartialRepairRunner):
    """Records the recovery proof's exact gold and leave-one-broken exits."""

    async def score(self, leave_broken, repairs, tests, *, p2p_command):  # type: ignore[no-untyped-def]
        score = await super().score(
            leave_broken, repairs, tests, p2p_command=p2p_command
        )
        exits = tuple(
            TestStateExit(
                test_id=test.test_id,
                exit_code=(
                    1
                    if leave_broken.file == "boltons/dictutils.py"
                    and test.test_id == RECOVERY_DUPLICATE_VALUE_TEST_COMMAND
                    else 0
                ),
            )
            for test in tests
        )
        return PartialRepairScore(
            other_inverse_patches_applied=score.other_inverse_patches_applied,
            p2p_passed=score.p2p_passed,
            failed_f2p_test_ids=tuple(exit.test_id for exit in exits if not exit.passed)
            or score.failed_f2p_test_ids,
            test_exits=exits,
        )

    async def score_gold(self, tests, *, p2p_command):  # type: ignore[no-untyped-def]
        assert p2p_command == P2P
        return FullGoldScore(
            p2p_exit_code=0,
            test_exits=tuple(
                TestStateExit(test_id=test.test_id, exit_code=0) for test in tests
            ),
        )


async def test_each_leave_one_broken_variant_repairs_all_other_faults() -> None:
    candidate = _candidate()
    runner = FakePartialRepairRunner()

    outcome = await assess_multifault_completeness(
        candidate,
        _tests(),
        p2p_command=P2P,
        runner=runner,
    )

    assert outcome.verdict == "pass"
    assert outcome.evidence is not None
    assert [record.index for record in outcome.evidence.constituents] == [0, 1]
    assert all(record.verdict == "pass" for record in outcome.evidence.constituents)
    assert runner.calls == [(0, (1,)), (1, (0,))]


async def test_recovery_duplicate_value_node_fails_only_with_one_to_one_broken() -> (
    None
):
    """The exact duplicate-value node replaces ineffective whole-file attribution."""
    candidate = _candidate(
        metadata=[
            {
                "index": 0,
                "file": "boltons/dictutils.py",
                "mutation_patch": "mutation OneToOne",
                "inverse_patch": "inverse OneToOne",
            },
            {
                "index": 1,
                "file": "boltons/statsutils.py",
                "mutation_patch": "mutation Stats",
                "inverse_patch": "inverse Stats",
            },
        ]
    )
    candidate.target = CandidateTarget(
        files=("boltons/dictutils.py", "boltons/statsutils.py"),
        symbols=("OneToOne.__init__", "Stats.describe"),
    )
    report = strengthen_recovery_duplicate_value_invariant(
        OracleReport(
            language="python",
            generator="bug_combination",
            verdict="pass",
            fail_to_pass=[F2P],
            pass_to_pass=[P2P],
            test_files=_tests(),
            flakiness_runs=3,
            mutants_total=10,
            mutants_killed=10,
        )
    )

    outcome = await assess_multifault_completeness(
        candidate,
        report.test_files,
        p2p_command=P2P,
        runner=RecoveryProofRunner(),
        fail_to_pass=report.fail_to_pass,
    )

    assert outcome.verdict == "pass"
    assert outcome.evidence is not None
    one_to_one = outcome.evidence.constituents[0]
    assert RECOVERY_DUPLICATE_VALUE_TEST_COMMAND in one_to_one.failed_f2p_test_ids
    assert {exit.test_id: exit.exit_code for exit in one_to_one.test_exits}[
        RECOVERY_DUPLICATE_VALUE_TEST_COMMAND
    ] == 1
    assert {
        exit.test_id: exit.exit_code for exit in outcome.evidence.full_gold_test_exits
    }[RECOVERY_DUPLICATE_VALUE_TEST_COMMAND] == 0
    assert outcome.evidence.full_gold_p2p_exit_code == 0
    report.multifault_evidence = outcome.evidence
    assert (
        report.details["recovery_duplicate_value_invariant"]["test_node"]  # type: ignore[index]
        == RECOVERY_DUPLICATE_VALUE_TEST_NODE
    )
    assert verify_recovery_duplicate_value_proof(report) == []


async def test_partial_repair_accepted_by_final_hidden_suite_rejects() -> None:
    outcome = await assess_multifault_completeness(
        _candidate(),
        _tests(),
        p2p_command=P2P,
        runner=FakePartialRepairRunner(failing_f2p=()),
    )

    assert outcome.verdict == "reject"
    assert outcome.evidence is not None
    assert all(record.verdict == "reject" for record in outcome.evidence.constituents)
    assert any("final hidden F2P all passed" in reason for reason in outcome.reasons)


async def test_p2p_regression_in_partial_repair_rejects() -> None:
    outcome = await assess_multifault_completeness(
        _candidate(),
        _tests(),
        p2p_command=P2P,
        runner=FakePartialRepairRunner(p2p_passed=False),
    )

    assert outcome.verdict == "reject"
    assert any("P2P is not green" in reason for reason in outcome.reasons)


async def test_constituent_inverse_patch_application_failure_rejects() -> None:
    outcome = await assess_multifault_completeness(
        _candidate(),
        _tests(),
        p2p_command=P2P,
        runner=FakePartialRepairRunner(applied=False),
    )

    assert outcome.verdict == "reject"
    assert any("apply_failed" in reason for reason in outcome.reasons)


async def test_missing_or_malformed_constituent_metadata_rejects() -> None:
    missing = _candidate(metadata=[])
    malformed = _candidate(
        metadata=[
            {
                "index": 1,
                "file": "src/alpha.py",
                "inverse_patch": "inverse alpha",
            }
        ]
    )

    for candidate in (missing, malformed):
        outcome = await assess_multifault_completeness(
            candidate,
            _tests(),
            p2p_command=P2P,
            runner=FakePartialRepairRunner(),
        )
        assert outcome.verdict == "reject"
        assert outcome.evidence is None
        assert any("metadata" in reason for reason in outcome.reasons)


def test_normalized_metadata_requires_indexed_executable_inverse_patches() -> None:
    candidate = _candidate(
        generator="multi_file",
        metadata=[
            {
                "index": 0,
                "file": "src/alpha.py",
                "mutation_patch": "mutation alpha",
                "inverse_patch": "inverse alpha",
            },
            {
                "index": 1,
                "file": "src/beta.py",
                "mutation_patch": "mutation beta",
                "inverse_patch": "inverse beta",
            },
        ],
    )

    constituents = normalize_constituent_inverse_patches(candidate)

    assert [item.index for item in constituents] == [0, 1]
    assert all(item.inverse_patch for item in constituents)
    assert all(item.inverse_patch_sha256 for item in constituents)


def _pass_report(
    evidence: MultiFaultCompletenessEvidence | None,
) -> OracleReport:
    tests = _tests()

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

    return OracleReport(
        language="python",
        generator="bug_combination",
        verdict="pass",
        fail_to_pass=[F2P],
        pass_to_pass=[P2P],
        test_files=tests,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        final_mutation_evidence=FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(tests),
            mutants_total=10,
            mutants_killed=10,
            threshold=0.8,
            tool="fake",
        ),
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        multifault_evidence=evidence,
        details={
            "teacher_gates": {
                "differential": {"calls": [call("differential")]},
                "alt_correct": {"calls": [call("alt_correct")]},
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
                    "hidden": [{"test_id": F2P, "exit_code": 0}],
                }
            },
        },
    )


async def test_export_consistency_requires_complete_final_suite_evidence() -> None:
    outcome = await assess_multifault_completeness(
        _candidate(),
        _tests(),
        p2p_command=P2P,
        runner=FakePartialRepairRunner(),
    )
    assert outcome.evidence is not None

    assert verify_pass_consistency(_pass_report(outcome.evidence)) == []
    assert any(
        "multifault" in problem
        for problem in verify_pass_consistency(_pass_report(None))
    )

    stale = MultiFaultCompletenessEvidence(
        suite_fingerprint=hashlib.sha256(b"stale").hexdigest(),
        p2p_command=P2P,
        constituents=outcome.evidence.constituents,
    )
    assert any(
        "suite fingerprint" in problem
        for problem in verify_pass_consistency(_pass_report(stale))
    )


async def test_oracle_report_round_trip_preserves_each_constituent_verdict() -> None:
    outcome = await assess_multifault_completeness(
        _candidate(),
        _tests(),
        p2p_command=P2P,
        runner=FakePartialRepairRunner(),
    )
    assert outcome.evidence is not None

    restored = OracleReport.from_dict(_pass_report(outcome.evidence).to_dict())

    assert restored.multifault_evidence is not None
    assert [
        record.to_dict() for record in restored.multifault_evidence.constituents
    ] == [record.to_dict() for record in outcome.evidence.constituents]
