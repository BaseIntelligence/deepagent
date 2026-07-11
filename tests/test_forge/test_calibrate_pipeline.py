"""Unit tests for the calibration pipeline finalizer (m5-cal-finalize).

Offline coverage (no real Docker, no live LLM), driven through injectable
``validator`` / ``rollout_fn`` seams, of the contract assertions this feature
owns:

- VAL-CAL-019: every LLM call in the run (each pre-flight validation AND each
  rollout) records its own token usage + cost, surfaced per-call PLUS an
  aggregate; values are non-zero for real (non-empty) calls.
- VAL-CAL-020: the finalized CalibrationReport is valid JSON with the per-model
  ``{model,tier,k,solves,pass_at_k}`` array spanning weak/mid/frontier,
  ``irt_difficulty``/``irt_discrimination``, and a terminal ``band_verdict`` +
  ``reason``; re-running on the same panel config reproduces the schema, the same
  keep/drop rule application on an equivalent solve band, and the same
  discrimination direction.
- VAL-CAL-023: a coherent CalibrationReport is produced per language (python /
  javascript / go), the band filter applied to each.

The live panel endpoint + real Docker scoring paths (incl. Docker hygiene,
VAL-CAL-022) are exercised by the worker manual verification + the user-testing
validator.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from swe_forge.forge.calibrate.filter import (
    RULE_KEEP,
    RULE_SOLVE_NONE,
    BandFilterConfig,
)
from swe_forge.forge.calibrate.pipeline import (
    CalibrationOutcome,
    assemble_calibration_report,
    build_usage_accounting,
    run_calibration,
)
from swe_forge.forge.calibrate.runner import run_panel_calibration
from swe_forge.forge.calibrate.solver import RolloutOutcome, SolveScore
from swe_forge.forge.models import (
    SUPPORTED_LANGUAGES,
    CalibrationReport,
    Candidate,
    CandidateTarget,
    EnvImage,
    FinalMutationEvidence,
    GeneratedSpec,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.calibrate.runner import CalibrationRunnerError
from swe_forge.forge.panel import ModelValidation, PanelModel
from swe_forge.forge.teacher import (
    TransportReceipt,
    Usage,
    candidate_transport_fingerprint,
)

P2P = "python -m pytest"
F2P = "python -m pytest tests/test_subtract.py"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _candidate(
    language: str = "python",
    difficulty_hint: str = "medium",
    generator: str = "ast_mutation",
) -> Candidate:
    return Candidate(
        language=language,
        generator=generator,
        target=CandidateTarget(files=("calc.py",), symbols=("subtract",)),
        mutation_patch="diff --git a/calc.py b/calc.py\n",
        oracle_patch="diff --git a/calc.py b/calc.py\n",
        difficulty_hint=difficulty_hint,
        provenance=Provenance(generator=generator, seed=7, language=language),
    )


def _env_image(language: str = "python") -> EnvImage:
    bases = {
        "python": "python:3.12-slim",
        "javascript": "node:22-slim",
        "go": "golang:1.22",
    }
    return EnvImage(
        repo_id=f"{language}-oracle",
        language=language,
        image_tag=f"swe-forge-env-{language}:abc123",
        base_image=bases[language],
        commit="0" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["install"],
        baseline_test_command=P2P,
        baseline_green=True,
        baseline_exit_code=0,
    )


def _spec(language: str = "python") -> GeneratedSpec:
    return GeneratedSpec(
        problem_statement="subtract(a, b) returns the wrong value.",
        requirements=["subtract(a, b) must return a - b."],
        interface_block="def subtract(a, b): ...",
        provenance=Provenance(generator="ast_mutation", seed=7, language=language),
    )


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


def _oracle_report(
    language: str = "python",
    generator: str = "ast_mutation",
) -> OracleReport:
    test_files = [
        OracleTestFile(path="tests/test_subtract.py", content="x", origin="provided")
    ]
    return OracleReport(
        language=language,
        generator=generator,
        verdict="pass",
        fail_to_pass=[F2P],
        pass_to_pass=[P2P],
        test_files=test_files,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        final_mutation_evidence=FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(test_files),
            mutants_total=10,
            mutants_killed=10,
            threshold=0.8,
            tool="fake",
        ),
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        details={
            "teacher_gates": _teacher_gate_evidence(),
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
                        b"calc.py\0def subtract(): ...\n\0"
                    ).hexdigest(),
                    "patches": [
                        {"path": "calc.py", "content": "def subtract(): ...\n"}
                    ],
                    "public": {"passed": True, "exit_code": 0},
                    "hidden": [{"test_id": F2P, "exit_code": 0}],
                }
            },
        },
    )


def _attach_transport_receipts(report: OracleReport, candidate: Candidate) -> None:
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


def _panel() -> list[PanelModel]:
    return [
        PanelModel(
            id="weak",
            model_string="openai/gpt-4o-mini",
            tier="weak",
            base_url="https://host",
            api_key="k",
        ),
        PanelModel(
            id="mid",
            model_string="anthropic/claude-sonnet-4-6",
            tier="mid",
            base_url="https://host",
            api_key="k",
        ),
        PanelModel(
            id="frontier",
            model_string="anthropic/claude-opus-4-8",
            tier="frontier",
            base_url="https://host",
            api_key="k",
        ),
    ]


def _outcome(model: str, *, solved: bool, tokens: int = 12) -> RolloutOutcome:
    return RolloutOutcome(
        model=model,
        patch="diff" if solved else "",
        finished=solved,
        solved=solved,
        score=SolveScore(
            solved=solved,
            applied=solved,
            empty=not solved,
            f2p_passed=solved,
            p2p_passed=solved,
        ),
        turns=3,
        usage=Usage(
            prompt_tokens=tokens, completion_tokens=tokens, total_tokens=2 * tokens
        ),
        cost=0.001,
    )


class FakeProbe:
    """Live-validator stand-in: probes each id once, billing a small usage/cost."""

    def __init__(self, invalid: tuple[str, ...] = ()) -> None:
        self.invalid = set(invalid)

    async def __call__(self, model: PanelModel) -> ModelValidation:
        if model.model_string in self.invalid:
            return ModelValidation(model=model.model_string, valid=False, error="bad")
        return ModelValidation(
            model=model.model_string,
            valid=True,
            usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            cost=0.0002,
        )


class FakeRollouts:
    """Deterministic rollout stand-in: model solves its first ``solves`` of k."""

    def __init__(self, solves_by_model: dict[str, int] | None = None) -> None:
        self.solves_by_model = solves_by_model or {}

    async def __call__(self, model: PanelModel, index: int) -> RolloutOutcome:
        solved = index < self.solves_by_model.get(model.model_string, 0)
        return _outcome(model.model_string, solved=solved)


# Separation: weak solves 0, mid solves 1, frontier solves 1 of k=4 -> in-band,
# discriminating (a keep band for the default filter).
_DISCRIMINATING = {
    "openai/gpt-4o-mini": 0,
    "anthropic/claude-sonnet-4-6": 1,
    "anthropic/claude-opus-4-8": 1,
}


async def _run(
    *,
    language: str = "python",
    difficulty_hint: str = "medium",
    solves: dict[str, int] | None = None,
    invalid: tuple[str, ...] = (),
    k: int | None = 4,
    config: BandFilterConfig | None = None,
) -> CalibrationOutcome:
    candidate = _candidate(language, difficulty_hint)
    oracle = _oracle_report(language)
    _attach_transport_receipts(oracle, candidate)
    return await run_calibration(
        candidate,
        _env_image(language),
        _spec(language),
        oracle,
        _panel(),
        k=k,
        config=config or BandFilterConfig(),
        validator=FakeProbe(invalid=invalid),
        rollout_fn=FakeRollouts(solves if solves is not None else _DISCRIMINATING),
    )


# --------------------------------------------------------------------------- #
# Usage / cost accounting (VAL-CAL-019)
# --------------------------------------------------------------------------- #
async def test_per_call_and_aggregate_usage_recorded() -> None:
    outcome = await _run()
    accounting = outcome.report.details["usage_accounting"]
    assert isinstance(accounting, dict)

    # Per-call validation entries: one per panel id, each with its own usage/cost.
    validation = accounting["validation"]
    assert validation["calls"] == 3
    assert len(validation["per_call"]) == 3
    for call in validation["per_call"]:
        assert call["usage"]["total_tokens"] == 3
        assert call["cost"] == pytest.approx(0.0002)

    # Per-call rollout entries: one per (model, rollout) = 3 models x k=4.
    rollout = accounting["rollout"]
    assert rollout["calls"] == 12
    assert len(rollout["per_call"]) == 12
    for call in rollout["per_call"]:
        assert call["usage"]["total_tokens"] > 0  # non-zero for real calls
        assert call["cost"] > 0

    # The aggregate sums per-call data WITHOUT discarding it.
    aggregate = accounting["aggregate"]
    assert aggregate["total_calls"] == 3 + 12
    assert aggregate["cost"] == pytest.approx(0.0002 * 3 + 0.001 * 12)
    summed_tokens = sum(
        c["usage"]["total_tokens"] for c in validation["per_call"]
    ) + sum(c["usage"]["total_tokens"] for c in rollout["per_call"])
    assert aggregate["usage"]["total_tokens"] == summed_tokens


async def test_aggregate_matches_run_accounting() -> None:
    outcome = await _run()
    aggregate = outcome.report.details["usage_accounting"]["aggregate"]
    # the accounting aggregate never diverges from the raw run's own totals
    assert aggregate["total_calls"] == outcome.run.total_calls
    assert aggregate["cost"] == pytest.approx(outcome.run.cost)
    assert aggregate["usage"]["total_tokens"] == outcome.run.usage.total_tokens


async def test_invalid_id_billed_for_its_probe_but_no_rollout_burst() -> None:
    outcome = await _run(invalid=("anthropic/claude-sonnet-4-6",))
    accounting = outcome.report.details["usage_accounting"]
    # the bad id is still probed once (recorded) ...
    assert accounting["validation"]["calls"] == 3
    models_in_rollouts = {c["model"] for c in accounting["rollout"]["per_call"]}
    # ... but contributes NO rollout burst
    assert "anthropic/claude-sonnet-4-6" not in models_in_rollouts
    assert accounting["rollout"]["calls"] == 2 * 4


async def test_calibration_refuses_multifault_without_final_constituent_proof() -> None:
    """No panel call may start when a multi-fault oracle proof is incomplete."""
    with pytest.raises(CalibrationRunnerError, match="multifault"):
        await run_calibration(
            _candidate(generator="bug_combination"),
            _env_image(),
            _spec(),
            _oracle_report(generator="bug_combination"),
            _panel(),
            k=1,
            validator=FakeProbe(),
            rollout_fn=FakeRollouts(_DISCRIMINATING),
        )


def test_build_usage_accounting_handles_missing_validation_usage() -> None:
    # A rejected validation may carry no usage; accounting must not crash on None.
    from swe_forge.forge.calibrate.runner import CalibrationRun

    run = CalibrationRun(
        models=[],
        validations=[ModelValidation(model="x", valid=False, error="bad")],
        k=0,
        difficulty_hint="medium",
        band="medium",
        validation_calls=1,
        rollout_calls=0,
    )
    accounting = build_usage_accounting(run)
    assert accounting["validation"]["per_call"][0]["usage"]["total_tokens"] == 0
    assert accounting["aggregate"]["total_calls"] == 1


# --------------------------------------------------------------------------- #
# Report schema + provenance (VAL-CAL-020)
# --------------------------------------------------------------------------- #
async def test_report_is_valid_json_with_all_fields() -> None:
    outcome = await _run()
    report = outcome.report
    # round-trips through JSON
    blob = json.dumps(report.to_dict())
    data = json.loads(blob)

    assert data["language"] == "python"
    assert data["band_verdict"] in {"keep", "drop"}
    assert data["reasons"] and data["reasons"][0]
    assert isinstance(data["irt_difficulty"], float)
    assert isinstance(data["irt_discrimination"], float)

    # per-model array spans all three tiers with the full schema
    tiers = {m["tier"] for m in data["models"]}
    assert tiers == {"weak", "mid", "frontier"}
    for m in data["models"]:
        assert set(m) >= {"model", "tier", "k", "solves", "pass_at_k"}
        assert 0 <= m["solves"] <= m["k"]
        assert 0.0 <= m["pass_at_k"] <= 1.0

    # reconstructs into a CalibrationReport without loss
    rebuilt = CalibrationReport.from_dict(data)
    assert rebuilt.band_verdict == report.band_verdict
    assert len(rebuilt.models) == 3


async def test_report_provenance_present() -> None:
    outcome = await _run()
    prov = outcome.report.provenance
    assert prov is not None
    assert prov.generator == "ast_mutation"
    assert prov.seed == 7
    assert prov.language == "python"
    assert prov.created_at  # timestamp populated


async def test_discriminating_matrix_keeps() -> None:
    outcome = await _run(solves=_DISCRIMINATING)
    assert outcome.report.band_verdict == "keep"
    assert outcome.report.details["band_filter"]["rule"] == RULE_KEEP
    assert outcome.report.irt_discrimination > 0.0


async def test_solve_none_drops_with_reason() -> None:
    outcome = await _run(solves={})  # nobody solves anything
    assert outcome.report.band_verdict == "drop"
    assert outcome.report.details["band_filter"]["rule"] == RULE_SOLVE_NONE
    assert outcome.report.frontier_pass_at_k() == 0.0


# --------------------------------------------------------------------------- #
# Reproducibility (VAL-CAL-020): same panel config -> same schema + rule + dir
# --------------------------------------------------------------------------- #
async def test_rerun_reproduces_schema_rule_and_discrimination_direction() -> None:
    first = await _run(solves=_DISCRIMINATING)
    second = await _run(solves=_DISCRIMINATING)

    # identical schema (same keys, same tier coverage)
    assert first.report.to_dict().keys() == second.report.to_dict().keys()
    assert {m.tier for m in first.report.models} == {
        m.tier for m in second.report.models
    }
    # identical keep/drop rule application on the equivalent solve band
    assert first.report.band_verdict == second.report.band_verdict
    assert (
        first.report.details["band_filter"]["rule"]
        == second.report.details["band_filter"]["rule"]
    )
    # preserved discrimination DIRECTION + the weak<=mid<=frontier trend
    assert first.report.irt_discrimination == pytest.approx(
        second.report.irt_discrimination
    )
    rates = first.report.tier_pass_rates()
    assert rates["weak"] <= rates["mid"] <= rates["frontier"]


async def test_equivalent_band_reproduces_keep_rule_under_stochastic_counts() -> None:
    # Two materially-equivalent (in-band, discriminating) solve matrices that
    # differ in the exact mid count still land on the same keep rule + direction.
    a = await _run(
        solves={
            "openai/gpt-4o-mini": 0,
            "anthropic/claude-sonnet-4-6": 1,
            "anthropic/claude-opus-4-8": 1,
        }
    )
    b = await _run(
        solves={
            "openai/gpt-4o-mini": 0,
            "anthropic/claude-sonnet-4-6": 0,
            "anthropic/claude-opus-4-8": 1,
        }
    )
    assert a.report.band_verdict == b.report.band_verdict == "keep"
    assert a.report.irt_discrimination > 0.0
    assert b.report.irt_discrimination > 0.0


# --------------------------------------------------------------------------- #
# Cross-language parity (VAL-CAL-023)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
async def test_cross_language_coherent_report(language: str) -> None:
    outcome = await _run(language=language, solves=_DISCRIMINATING)
    report = outcome.report
    assert report.language == language
    assert len(report.models) == 3
    assert report.band_verdict in {"keep", "drop"}
    assert report.reasons and report.reasons[0]
    assert isinstance(report.irt_difficulty, float)
    assert isinstance(report.irt_discrimination, float)
    # usage accounting present for every language
    assert report.details["usage_accounting"]["aggregate"]["total_calls"] > 0


# --------------------------------------------------------------------------- #
# assemble_calibration_report directly over a recorded run
# --------------------------------------------------------------------------- #
async def test_assemble_from_run_is_pure_and_deterministic() -> None:
    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        _panel(),
        k=4,
        validator=FakeProbe(),
        rollout_fn=FakeRollouts(_DISCRIMINATING),
    )
    r1 = assemble_calibration_report(run, language="python")
    r2 = assemble_calibration_report(run, language="python")
    assert r1.to_dict() == r2.to_dict()
    assert r1.band_verdict == "keep"


async def test_difficulty_aware_budget_used_when_k_unset() -> None:
    # A 'hard' candidate gets the hard-band k (6) from the default budget.
    outcome = await _run(difficulty_hint="hard", k=None, solves=_DISCRIMINATING)
    assert outcome.report.k == 6
    assert all(m.k == 6 for m in outcome.report.models)
