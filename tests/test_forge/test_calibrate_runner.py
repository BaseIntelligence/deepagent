"""Unit tests for the calibration panel runner (m5-runner).

Offline coverage (no real Docker, no live LLM) of the runner's contract
assertions, driven through injectable ``validator`` / ``rollout_fn`` seams:

- VAL-CAL-006: each model gets exactly ``k`` independent, concurrency-bounded
  rollouts; the in-flight cap is never exceeded and ``k`` distinct usage records
  are kept per model.
- VAL-CAL-007: the per-model record schema is ``{model, tier in
  {weak,mid,frontier}, k, solves(0<=solves<=k), pass_at_k in [0,1]}``.
- VAL-CAL-008: every id is probed once BEFORE the bulk run; an invalid id is
  excluded WITHOUT its ``k``-burst; with all ids valid the call count equals
  ``sum(1 + k)`` over the panel.
- VAL-CAL-009: a difficulty-aware budget gives the hard band a ``k`` no smaller
  than the easy band, and the per-model usage-record count matches the selected
  ``k``.

The live panel endpoint + real Docker scoring paths are exercised by the
worker/user-testing validator (see this feature's manual verification).
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

import swe_forge.forge.calibrate.runner as runner_module
from swe_forge.forge.calibrate.runner import (
    DEFAULT_BUDGET,
    CalibrationRun,
    CalibrationRunnerError,
    ModelCalibration,
    RolloutBudget,
    compute_pass_at_k,
    run_panel_calibration,
    suppress_litellm_async_warning,
)
from swe_forge.forge.calibrate.solver import RolloutOutcome, SolveScore
from swe_forge.forge.models import (
    BaselineNotGreenError,
    Candidate,
    CandidateTarget,
    EnvImage,
    GeneratedSpec,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.panel import ModelValidation, PanelModel
from swe_forge.forge.teacher import Usage

P2P = "python -m pytest"
F2P = "python -m pytest tests/test_subtract.py"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _candidate(difficulty_hint: str = "easy") -> Candidate:
    return Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("calc.py",), symbols=("subtract",)),
        mutation_patch="diff --git a/calc.py b/calc.py\n",
        oracle_patch="diff --git a/calc.py b/calc.py\n",
        difficulty_hint=difficulty_hint,
        provenance=Provenance(generator="ast_mutation", seed=1, language="python"),
    )


def _env_image(green: bool = True) -> EnvImage:
    return EnvImage(
        repo_id="py-oracle",
        language="python",
        image_tag="swe-forge-env-py-oracle:abc123",
        base_image="python:3.12-slim",
        commit="0" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command=P2P,
        baseline_green=green,
        baseline_exit_code=0 if green else 1,
    )


def _spec() -> GeneratedSpec:
    return GeneratedSpec(
        problem_statement="subtract(a, b) returns the wrong value.",
        requirements=["subtract(a, b) must return a - b."],
        interface_block="def subtract(a, b): ...",
        provenance=Provenance(generator="ast_mutation", seed=1, language="python"),
    )


def _oracle_report() -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=[F2P],
        pass_to_pass=[P2P],
        test_files=[
            OracleTestFile(
                path="tests/test_subtract.py",
                content="x",
                origin="provided",
            )
        ],
    )


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


# --------------------------------------------------------------------------- #
# Fakes (injectable seams)
# --------------------------------------------------------------------------- #
class FakeProbe:
    """A live-validator stand-in: probes each id once; flags configured ids bad.

    Tracks in-flight concurrency so the semaphore cap can be asserted. Invalid
    ids carry no usage (mirroring the real ``validate_model`` error path).
    """

    def __init__(self, invalid: tuple[str, ...] = ()) -> None:
        self.invalid = set(invalid)
        self.probed: list[str] = []
        self.current = 0
        self.max_inflight = 0

    async def __call__(self, model: PanelModel) -> ModelValidation:
        self.current += 1
        self.max_inflight = max(self.max_inflight, self.current)
        await asyncio.sleep(0.002)
        self.current -= 1
        self.probed.append(model.model_string)
        valid = model.model_string not in self.invalid
        if not valid:
            return ModelValidation(
                model=model.model_string, valid=False, error="bad id"
            )
        return ModelValidation(
            model=model.model_string,
            valid=True,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=0.0001,
        )


class FakeRollouts:
    """A rollout stand-in: records every (model, index); bounds concurrency.

    ``solves_by_model`` is the number of the model's ``k`` rollouts that solve
    (decided by index, so the count is deterministic regardless of scheduling).
    If ``probe``/``expected_probes`` are given, it asserts that EVERY validation
    finished before any rollout starts (validate-before-bulk ordering).
    """

    def __init__(
        self,
        solves_by_model: dict[str, int] | None = None,
        *,
        probe: FakeProbe | None = None,
        expected_probes: int = 0,
    ) -> None:
        self.solves_by_model = solves_by_model or {}
        self.probe = probe
        self.expected_probes = expected_probes
        self.calls: list[tuple[str, int]] = []
        self.current = 0
        self.max_inflight = 0

    async def __call__(self, model: PanelModel, index: int) -> RolloutOutcome:
        if self.probe is not None:
            assert len(self.probe.probed) == self.expected_probes, (
                "rollout started before all validations completed"
            )
        self.current += 1
        self.max_inflight = max(self.max_inflight, self.current)
        await asyncio.sleep(0.005)
        self.current -= 1
        self.calls.append((model.model_string, index))
        solved = index < self.solves_by_model.get(model.model_string, 0)
        return _outcome(model.model_string, solved=solved, tokens=10 + index)


# --------------------------------------------------------------------------- #
# compute_pass_at_k
# --------------------------------------------------------------------------- #
def test_compute_pass_at_k_zero_solves_is_zero() -> None:
    assert compute_pass_at_k(0, 5) == 0.0


def test_compute_pass_at_k_any_solve_is_positive() -> None:
    assert compute_pass_at_k(1, 5) > 0.0


def test_compute_pass_at_k_clamped_and_fraction() -> None:
    assert compute_pass_at_k(3, 6) == 0.5
    assert compute_pass_at_k(7, 5) == 1.0  # clamped
    assert compute_pass_at_k(2, 0) == 0.0  # degenerate k


# --------------------------------------------------------------------------- #
# RolloutBudget (VAL-CAL-009)
# --------------------------------------------------------------------------- #
def test_default_budget_is_nondecreasing() -> None:
    assert DEFAULT_BUDGET.easy <= DEFAULT_BUDGET.medium <= DEFAULT_BUDGET.hard


def test_budget_rejects_decreasing_bands() -> None:
    with pytest.raises(CalibrationRunnerError):
        RolloutBudget(easy=5, medium=4, hard=3)


def test_budget_rejects_negative_band() -> None:
    with pytest.raises(CalibrationRunnerError):
        RolloutBudget(easy=-1)


def test_budget_band_and_k_mapping() -> None:
    budget = RolloutBudget(easy=2, medium=4, hard=8)
    assert budget.band_for("trivial") == "easy"
    assert budget.band_for("HIGH") == "hard"
    assert budget.band_for("moderate") == "medium"
    assert budget.band_for("unknown-label") == "medium"  # default band
    assert budget.k_for("easy") == 2
    assert budget.k_for("hard") == 8
    assert budget.k_for("hard") >= budget.k_for("easy")


# --------------------------------------------------------------------------- #
# ModelCalibration schema (VAL-CAL-007)
# --------------------------------------------------------------------------- #
def test_model_calibration_schema_and_to_dict() -> None:
    rec = ModelCalibration(
        model="anthropic/claude-opus-4-8",
        tier="frontier",
        k=4,
        solves=1,
        pass_at_k=0.25,
    )
    data = rec.to_dict()
    assert data["model"] == "anthropic/claude-opus-4-8"
    assert data["tier"] == "frontier"
    assert data["k"] == 4
    assert data["solves"] == 1
    assert data["pass_at_k"] == 0.25


def test_model_calibration_rejects_bad_tier() -> None:
    with pytest.raises(CalibrationRunnerError):
        ModelCalibration(model="m", tier="superhuman", k=1, solves=0, pass_at_k=0.0)


def test_model_calibration_rejects_solves_gt_k() -> None:
    with pytest.raises(CalibrationRunnerError):
        ModelCalibration(model="m", tier="weak", k=2, solves=3, pass_at_k=1.0)


def test_model_calibration_rejects_pass_at_k_out_of_range() -> None:
    with pytest.raises(CalibrationRunnerError):
        ModelCalibration(model="m", tier="weak", k=2, solves=1, pass_at_k=1.5)


# --------------------------------------------------------------------------- #
# k independent, concurrency-bounded rollouts (VAL-CAL-006)
# --------------------------------------------------------------------------- #
async def test_runs_exactly_k_rollouts_per_model() -> None:
    panel = _panel()
    probe = FakeProbe()
    rollouts = FakeRollouts(solves_by_model={"openai/gpt-4o-mini": 1})
    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        k=3,
        validator=probe,
        rollout_fn=rollouts,
    )
    assert run.k == 3
    assert len(run.models) == 3
    for rec in run.models:
        assert rec.k == 3
        assert len(rec.rollouts) == 3  # k distinct usage records
    assert run.rollout_calls == 3 * 3
    # one rollout call per (model, index) pair, all distinct
    assert len(rollouts.calls) == 9
    assert len(set(rollouts.calls)) == 9


async def test_concurrency_cap_never_exceeded() -> None:
    panel = _panel()
    rollouts = FakeRollouts()
    await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        k=4,
        concurrency=2,
        validator=FakeProbe(),
        rollout_fn=rollouts,
    )
    assert rollouts.max_inflight <= 2
    assert rollouts.max_inflight >= 1


async def test_solves_and_pass_at_k_reflect_outcomes() -> None:
    panel = _panel()
    # weak solves 0/4, mid 1/4, frontier 2/4
    rollouts = FakeRollouts(
        solves_by_model={
            "openai/gpt-4o-mini": 0,
            "anthropic/claude-sonnet-4-6": 1,
            "anthropic/claude-opus-4-8": 2,
        }
    )
    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        k=4,
        validator=FakeProbe(),
        rollout_fn=rollouts,
    )
    by_model = {r.model: r for r in run.models}
    assert by_model["openai/gpt-4o-mini"].solves == 0
    assert by_model["openai/gpt-4o-mini"].pass_at_k == 0.0
    assert by_model["anthropic/claude-sonnet-4-6"].solves == 1
    assert by_model["anthropic/claude-sonnet-4-6"].pass_at_k == 0.25
    assert by_model["anthropic/claude-opus-4-8"].solves == 2
    assert by_model["anthropic/claude-opus-4-8"].pass_at_k == 0.5
    # every per-model record conforms to the schema ranges (VAL-CAL-007)
    for rec in run.models:
        assert rec.tier in {"weak", "mid", "frontier"}
        assert 0 <= rec.solves <= rec.k
        assert 0.0 <= rec.pass_at_k <= 1.0


async def test_per_model_usage_aggregates_k_records() -> None:
    panel = _panel()[:1]
    rollouts = FakeRollouts(solves_by_model={"openai/gpt-4o-mini": 1})
    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        k=3,
        validator=FakeProbe(),
        rollout_fn=rollouts,
    )
    rec = run.models[0]
    # tokens 10+index for index 0,1,2 -> total = (10+11+12)*2 = 66
    expected = sum((10 + i) * 2 for i in range(3))
    assert rec.usage.total_tokens == expected
    assert rec.cost == pytest.approx(0.001 * 3)


async def test_default_rollout_factory_isolates_client_per_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent recovery rollouts must not share mutable accounting state."""
    clients: list[object] = []
    solvers: list[object] = []

    def _client(self: PanelModel, **_kwargs: object) -> object:
        client = object()
        clients.append(client)
        return client

    async def _run_solver_rollout(*args: object, **kwargs: object) -> RolloutOutcome:
        solvers.append(kwargs["solver"])
        return _outcome(str(kwargs["model"]), solved=False)

    monkeypatch.setattr(PanelModel, "client", _client)
    monkeypatch.setattr(runner_module, "run_solver_rollout", _run_solver_rollout)

    rollout = runner_module._build_rollout_fn(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        adapter=None,
        docker_client=None,
        max_turns=2,
        max_tokens=32,
        command_timeout=10.0,
        recovery_ledger=object(),  # type: ignore[arg-type]
    )
    model = _panel()[0]
    await asyncio.gather(rollout(model, 0), rollout(model, 1))

    assert len(clients) == 2
    assert len({id(client) for client in clients}) == 2
    assert len({id(solver) for solver in solvers}) == 2


# --------------------------------------------------------------------------- #
# Validate-before-bulk; no k-burst for a bad id; call accounting (VAL-CAL-008)
# --------------------------------------------------------------------------- #
async def test_all_valid_call_count_equals_sum_one_plus_k() -> None:
    panel = _panel()
    probe = FakeProbe()
    rollouts = FakeRollouts(probe=probe, expected_probes=len(panel))
    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        k=3,
        validator=probe,
        rollout_fn=rollouts,
    )
    assert run.validation_calls == 3
    assert run.rollout_calls == 3 * 3
    # Sigma over validated models of (1 validation + k rollouts)
    assert run.total_calls == sum(1 + 3 for _ in panel)


async def test_invalid_id_excluded_without_k_burst() -> None:
    panel = _panel()
    probe = FakeProbe(invalid=("anthropic/claude-sonnet-4-6",))
    rollouts = FakeRollouts(probe=probe, expected_probes=len(panel))
    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        k=5,
        validator=probe,
        rollout_fn=rollouts,
    )
    # every id probed exactly once, before the bulk run
    assert sorted(probe.probed) == sorted(m.model_string for m in panel)
    # the bad id is excluded from the per-model records (never fabricated)
    models = {r.model for r in run.models}
    assert "anthropic/claude-sonnet-4-6" not in models
    assert len(run.models) == 2
    # ... yet it is recorded as a rejected validation (marked, not silently dropped)
    rejected = [v for v in run.validations if not v.valid]
    assert [v.model for v in rejected] == ["anthropic/claude-sonnet-4-6"]
    # NO k-burst for the bad id: it produced zero rollout calls
    assert all(m != "anthropic/claude-sonnet-4-6" for m, _ in rollouts.calls)
    assert run.validation_calls == 3
    assert run.rollout_calls == 2 * 5  # only the two validated models
    assert run.total_calls == 3 + 2 * 5


async def test_no_validate_runs_all_models() -> None:
    panel = _panel()
    rollouts = FakeRollouts()

    async def _never(model: PanelModel) -> ModelValidation:  # pragma: no cover
        raise AssertionError("validator must not run when validate=False")

    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        k=2,
        validate=False,
        validator=_never,
        rollout_fn=rollouts,
    )
    assert run.validation_calls == 0
    assert len(run.models) == 3
    assert run.rollout_calls == 3 * 2


async def test_no_validate_keeps_validations_consistent_with_count() -> None:
    # Under validate=False no probe is issued, so the validations array must stay
    # empty (no synthetic entries) and match validation_calls == 0.
    panel = _panel()
    rollouts = FakeRollouts()
    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        k=2,
        validate=False,
        rollout_fn=rollouts,
    )
    assert run.validation_calls == 0
    assert run.validations == []
    assert len(run.validations) == run.validation_calls
    # to_dict mirrors the same consistency.
    data = run.to_dict()
    assert data["validations"] == []
    assert data["validation_calls"] == 0


# --------------------------------------------------------------------------- #
# Difficulty-aware budget (VAL-CAL-009)
# --------------------------------------------------------------------------- #
async def test_hard_band_k_ge_easy_band_k() -> None:
    budget = RolloutBudget(easy=2, medium=3, hard=5)
    panel = _panel()[:1]

    easy_run = await run_panel_calibration(
        _candidate("easy"),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        budget=budget,
        validator=FakeProbe(),
        rollout_fn=FakeRollouts(),
    )
    hard_run = await run_panel_calibration(
        _candidate("hard"),
        _env_image(),
        _spec(),
        _oracle_report(),
        panel,
        budget=budget,
        validator=FakeProbe(),
        rollout_fn=FakeRollouts(),
    )
    assert easy_run.k == 2
    assert easy_run.band == "easy"
    assert hard_run.k == 5
    assert hard_run.band == "hard"
    assert hard_run.k >= easy_run.k
    # per-model usage-record count matches the selected k for each band
    assert all(len(r.rollouts) == 2 for r in easy_run.models)
    assert all(len(r.rollouts) == 5 for r in hard_run.models)
    assert hard_run.rollout_calls >= easy_run.rollout_calls


async def test_explicit_k_overrides_budget() -> None:
    run = await run_panel_calibration(
        _candidate("hard"),
        _env_image(),
        _spec(),
        _oracle_report(),
        _panel()[:1],
        budget=RolloutBudget(easy=2, medium=3, hard=9),
        k=1,
        validator=FakeProbe(),
        rollout_fn=FakeRollouts(),
    )
    assert run.k == 1
    assert run.models[0].k == 1


# --------------------------------------------------------------------------- #
# Preconditions / errors
# --------------------------------------------------------------------------- #
async def test_red_baseline_is_rejected() -> None:
    with pytest.raises(BaselineNotGreenError):
        await run_panel_calibration(
            _candidate(),
            _env_image(green=False),
            _spec(),
            _oracle_report(),
            _panel(),
            k=1,
            validator=FakeProbe(),
            rollout_fn=FakeRollouts(),
        )


async def test_negative_k_override_rejected() -> None:
    with pytest.raises(CalibrationRunnerError):
        await run_panel_calibration(
            _candidate(),
            _env_image(),
            _spec(),
            _oracle_report(),
            _panel(),
            k=-1,
            validator=FakeProbe(),
            rollout_fn=FakeRollouts(),
        )


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
async def test_calibration_run_to_dict_is_serializable() -> None:
    run = await run_panel_calibration(
        _candidate(),
        _env_image(),
        _spec(),
        _oracle_report(),
        _panel(),
        k=2,
        validator=FakeProbe(invalid=("openai/gpt-4o-mini",)),
        rollout_fn=FakeRollouts(),
    )
    data = run.to_dict()
    assert data["k"] == 2
    assert data["total_calls"] == data["validation_calls"] + data["rollout_calls"]
    assert isinstance(data["models"], list)
    assert isinstance(data["validations"], list)
    # the excluded id appears in validations (marked invalid), not in models
    assert {m["model"] for m in data["models"]} == {
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-8",
    }
    assert any(
        v["model"] == "openai/gpt-4o-mini" and v["valid"] is False
        for v in data["validations"]
    )
    assert isinstance(run, CalibrationRun)


# --------------------------------------------------------------------------- #
# litellm benign async-warning suppression (cosmetic-only; no error swallowing)
# --------------------------------------------------------------------------- #
def test_suppress_filters_only_the_benign_coroutine_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_litellm_async_warning():
            warnings.warn("coroutine 'x.async_log' was never awaited", RuntimeWarning)
            warnings.warn("a genuine user warning", UserWarning)
            warnings.warn("a different runtime warning", RuntimeWarning)
    messages = [str(w.message) for w in caught]
    # the exact benign litellm message is filtered out ...
    assert not any("was never awaited" in m for m in messages)
    # ... while every other warning (incl. other RuntimeWarnings) is untouched
    assert any("a genuine user warning" in m for m in messages)
    assert any("a different runtime warning" in m for m in messages)


def test_suppress_does_not_swallow_real_errors() -> None:
    with pytest.raises(ValueError, match="real error"):
        with suppress_litellm_async_warning():
            raise ValueError("real error")
