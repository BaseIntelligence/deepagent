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
    MultiFaultCompletenessEvidence,
    PartialRepairScore,
    assess_multifault_completeness,
    normalize_constituent_inverse_patches,
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
