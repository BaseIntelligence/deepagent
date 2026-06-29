"""Calibration panel runner: k rollouts x N models -> per-model pass records.

Stage 4 (architecture S6) measures how hard a manufactured task is by letting a
*panel* of solver models attempt it and counting how often they succeed. This
module orchestrates that bulk run on top of the m5 solver scaffold
(:mod:`swe_forge.forge.calibrate.solver`); it does NOT re-implement rollout or
scoring.

For each panel model the runner issues exactly ``k`` independent, uncached
rollouts concurrently under a single :class:`asyncio.Semaphore` (so the global
in-flight rollout count never exceeds the configured cap), then records the
per-model ``{model, tier, k, solves, pass_at_k}`` summary (plus per-rollout usage
records so the count == ``k`` is auditable).

Governing principles:

* **Validate before bulk (cost discipline).** Every model id is probed with a
  single live pre-flight call BEFORE any rollouts. Only ids that pass proceed to
  their ``k``-burst; an invalid id is excluded WITHOUT issuing any rollouts, so a
  typo never costs a ``k``-burst of completions.
* **Difficulty-aware budget.** Harder candidates receive a larger ``k`` than
  easy ones per :class:`RolloutBudget`, concentrating rollouts where the signal
  is.
* **Clean rollout output.** litellm's async success/failure logger occasionally
  leaves an un-awaited coroutine that the interpreter reports as a benign
  ``RuntimeWarning: coroutine ... was never awaited`` when it is garbage
  collected. That is cosmetic noise on the rollout path (an m1 user-testing
  finding); :func:`suppress_litellm_async_warning` filters ONLY that exact
  message so rollout stdout/logs stay clean. It never suppresses any other
  warning and never swallows a real error (exceptions are not warnings).
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import warnings
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.calibrate.irt import pass_at_k as _irt_pass_at_k
from swe_forge.forge.calibrate.solver import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    AgenticSolver,
    RolloutOutcome,
    run_solver_rollout,
)
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    GeneratedSpec,
    OracleReport,
    require_green_baseline,
)
from swe_forge.forge.panel import (
    DEFAULT_ROLLOUT_CONCURRENCY,
    DEFAULT_VALIDATE_MAX_TOKENS,
    DEFAULT_VALIDATE_NUM_RETRIES,
    DEFAULT_VALIDATE_TIMEOUT,
    VALID_TIERS,
    ModelValidation,
    PanelModel,
    validate_model,
)
from swe_forge.forge.teacher import Usage

# The benign litellm async-logging warning message (matched as a regex prefix).
_LITELLM_ASYNC_WARNING = r"coroutine .* was never awaited"

#: Type of the live model-id validator (one probe per model, no bulk).
ValidatorFn = Callable[[PanelModel], Awaitable[ModelValidation]]
#: Type of a single rollout: (model, rollout-index) -> scored outcome.
RolloutFn = Callable[[PanelModel, int], Awaitable[RolloutOutcome]]


class CalibrationRunnerError(RuntimeError):
    """Raised when the calibration runner is configured with invalid inputs."""


@contextlib.contextmanager
def suppress_litellm_async_warning():  # type: ignore[no-untyped-def]
    """Filter ONLY litellm's benign ``coroutine ... was never awaited`` warning.

    Scopes a narrow ``ignore`` filter (RuntimeWarning whose message starts with
    ``coroutine ... was never awaited``) for the duration of the block, then
    forces a garbage collection while the filter is still active so the orphaned
    litellm logging coroutine is reclaimed (and its warning suppressed)
    deterministically. No other warning category/message is touched, and real
    errors -- which are exceptions, not warnings -- are never affected.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_LITELLM_ASYNC_WARNING,
            category=RuntimeWarning,
        )
        try:
            yield
        finally:
            gc.collect()


def compute_pass_at_k(solves: int, k: int) -> float:
    """Per-model pass@k estimator; re-exported from :mod:`...calibrate.irt`.

    The canonical implementation lives in the IRT module (the single source of
    truth for pass@k, VAL-CAL-011); the runner re-exports it so existing
    ``runner.compute_pass_at_k`` imports keep working.
    """
    return _irt_pass_at_k(solves, k)


# Difficulty labels emitted by the generators (low/medium/high) plus common
# synonyms, mapped to the three budget bands.
_DIFFICULTY_BANDS: dict[str, str] = {
    "trivial": "easy",
    "easy": "easy",
    "low": "easy",
    "medium": "medium",
    "mid": "medium",
    "moderate": "medium",
    "hard": "hard",
    "high": "hard",
    "difficult": "hard",
}


@dataclass(frozen=True)
class RolloutBudget:
    """Difficulty-aware rollout budget: more ``k`` on harder candidates.

    The three bands must be nondecreasing (``easy <= medium <= hard``) so the
    hard-band ``k`` is always at least the easy-band ``k`` (VAL-CAL-009).
    """

    easy: int = 3
    medium: int = 4
    hard: int = 6

    def __post_init__(self) -> None:
        for name in ("easy", "medium", "hard"):
            if getattr(self, name) < 0:
                raise CalibrationRunnerError(
                    f"RolloutBudget.{name} must be >= 0; got {getattr(self, name)}"
                )
        if not (self.easy <= self.medium <= self.hard):
            raise CalibrationRunnerError(
                "RolloutBudget must be nondecreasing (easy <= medium <= hard); "
                f"got easy={self.easy}, medium={self.medium}, hard={self.hard}"
            )

    def band_for(self, difficulty_hint: str) -> str:
        """Map a candidate's ``difficulty_hint`` to a band (default ``medium``)."""
        return _DIFFICULTY_BANDS.get(difficulty_hint.strip().lower(), "medium")

    def k_for(self, difficulty_hint: str) -> int:
        """Return the rollout count ``k`` for a candidate's difficulty band."""
        return int(getattr(self, self.band_for(difficulty_hint)))


#: Default difficulty-aware budget (small, for cost discipline).
DEFAULT_BUDGET = RolloutBudget()


@dataclass
class ModelCalibration:
    """Per-model calibration record: ``{model, tier, k, solves, pass_at_k}``.

    ``rollouts`` keeps the ``k`` individual scored outcomes so the usage-record
    count is auditable (one independent, uncached completion per rollout), and
    ``usage``/``cost`` aggregate this model's rollout spend.
    """

    model: str
    tier: str
    k: int
    solves: int
    pass_at_k: float
    rollouts: list[RolloutOutcome] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise CalibrationRunnerError(
                f"ModelCalibration.tier {self.tier!r} not in {VALID_TIERS}"
            )
        if self.k < 0:
            raise CalibrationRunnerError(
                f"ModelCalibration.k must be >= 0; got {self.k}"
            )
        if not (0 <= self.solves <= self.k):
            raise CalibrationRunnerError(
                f"ModelCalibration.solves must satisfy 0 <= solves <= k "
                f"({self.solves} vs k={self.k})"
            )
        if not (0.0 <= self.pass_at_k <= 1.0):
            raise CalibrationRunnerError(
                f"ModelCalibration.pass_at_k must be in [0, 1]; got {self.pass_at_k}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "tier": self.tier,
            "k": self.k,
            "solves": self.solves,
            "pass_at_k": self.pass_at_k,
            "cost": self.cost,
            "usage": self.usage.to_dict(),
            "rollouts": [r.to_dict() for r in self.rollouts],
        }


@dataclass
class CalibrationRun:
    """The full panel run: per-model records + validation + call/cost accounting.

    ``models`` holds one :class:`ModelCalibration` per VALIDATED model (invalid
    ids are excluded, never fabricated). ``validations`` records every pre-flight
    probe (one per id, including the rejected ones). ``validation_calls`` /
    ``rollout_calls`` make the call accounting auditable: with every id valid,
    ``total_calls == sum(1 + k)`` over the panel; a rejected id contributes its
    single probe but ZERO rollouts (no k-burst).
    """

    models: list[ModelCalibration]
    validations: list[ModelValidation]
    k: int
    difficulty_hint: str
    band: str
    validation_calls: int
    rollout_calls: int
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0

    @property
    def total_calls(self) -> int:
        """Observed LLM call count: validation probes + rollout completions."""
        return self.validation_calls + self.rollout_calls

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "difficulty_hint": self.difficulty_hint,
            "band": self.band,
            "validation_calls": self.validation_calls,
            "rollout_calls": self.rollout_calls,
            "total_calls": self.total_calls,
            "cost": self.cost,
            "usage": self.usage.to_dict(),
            "models": [m.to_dict() for m in self.models],
            "validations": [v.to_dict() for v in self.validations],
        }


def _sum_usage(items: Iterable[Usage]) -> Usage:
    total = Usage()
    for item in items:
        total = total + item
    return total


async def run_panel_calibration(
    candidate: Candidate,
    env_image: EnvImage,
    spec: GeneratedSpec,
    oracle_report: OracleReport,
    panel: Sequence[PanelModel],
    *,
    budget: RolloutBudget = DEFAULT_BUDGET,
    k: int | None = None,
    concurrency: int = DEFAULT_ROLLOUT_CONCURRENCY,
    validate: bool = True,
    validate_prompt: str = "ping",
    validate_max_tokens: int = DEFAULT_VALIDATE_MAX_TOKENS,
    validate_num_retries: int = DEFAULT_VALIDATE_NUM_RETRIES,
    validate_timeout: float = DEFAULT_VALIDATE_TIMEOUT,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    command_timeout: float = 600.0,
    adapter: LanguageAdapter | None = None,
    docker_client: object | None = None,
    validator: ValidatorFn | None = None,
    rollout_fn: RolloutFn | None = None,
) -> CalibrationRun:
    """Run ``k`` independent rollouts x each panel model and record pass@k.

    A green baseline is a hard precondition. ``k`` is selected from the
    difficulty-aware ``budget`` (overridable via ``k=``). Every model id is probed
    once (``validate=True``) BEFORE the bulk run; only validated ids get their
    ``k``-burst. Rollouts run concurrently under a single semaphore of size
    ``concurrency`` (never exceeded). ``validator``/``rollout_fn`` are injectable
    seams for deterministic unit testing; the defaults drive the live panel
    endpoint and the shared Docker FAIL->PASS scorer.
    """
    require_green_baseline(env_image)

    selected_k = budget.k_for(candidate.difficulty_hint) if k is None else int(k)
    if selected_k < 0:
        raise CalibrationRunnerError(f"k must be >= 0; got {selected_k}")
    band = budget.band_for(candidate.difficulty_hint)

    if rollout_fn is None and adapter is None:
        adapter = build_default_registry().get(candidate.language)

    resolved_validator = validator or _build_validator(
        prompt=validate_prompt,
        max_tokens=validate_max_tokens,
        num_retries=validate_num_retries,
        timeout=validate_timeout,
    )
    resolved_rollout = rollout_fn or _build_rollout_fn(
        candidate,
        env_image,
        spec,
        oracle_report,
        adapter=adapter,
        docker_client=docker_client,
        max_turns=max_turns,
        max_tokens=max_tokens,
        command_timeout=command_timeout,
    )

    semaphore = asyncio.Semaphore(max(1, concurrency))

    with suppress_litellm_async_warning():
        # Phase 1 - validate every id with a single probe BEFORE any rollouts.
        if validate:

            async def _bounded_validate(model: PanelModel) -> ModelValidation:
                async with semaphore:
                    return await resolved_validator(model)

            validations = list(
                await asyncio.gather(*(_bounded_validate(m) for m in panel))
            )
        else:
            validations = [
                ModelValidation(model=m.model_string, valid=True) for m in panel
            ]

        valid_models = [m for m, v in zip(panel, validations) if v.valid]

        # Phase 2 - k independent, concurrency-bounded rollouts per VALID model.
        async def _bounded_rollout(model: PanelModel, index: int) -> RolloutOutcome:
            async with semaphore:
                return await resolved_rollout(model, index)

        owners: list[str] = []
        tasks: list[Awaitable[RolloutOutcome]] = []
        for model in valid_models:
            for index in range(selected_k):
                owners.append(model.model_string)
                tasks.append(_bounded_rollout(model, index))
        outcomes = list(await asyncio.gather(*tasks)) if tasks else []

    by_model: dict[str, list[RolloutOutcome]] = {
        m.model_string: [] for m in valid_models
    }
    for owner, outcome in zip(owners, outcomes):
        by_model[owner].append(outcome)

    records: list[ModelCalibration] = []
    for model in valid_models:
        rollouts = by_model[model.model_string]
        solves = sum(1 for o in rollouts if o.solved)
        records.append(
            ModelCalibration(
                model=model.model_string,
                tier=model.tier,
                k=selected_k,
                solves=solves,
                pass_at_k=compute_pass_at_k(solves, selected_k),
                rollouts=rollouts,
                usage=_sum_usage(o.usage for o in rollouts),
                cost=sum(o.cost for o in rollouts),
            )
        )

    validation_calls = len(validations) if validate else 0
    validation_usage = _sum_usage(v.usage for v in validations if v.usage is not None)
    validation_cost = sum(v.cost for v in validations)
    rollout_usage = _sum_usage(o.usage for o in outcomes)
    rollout_cost = sum(o.cost for o in outcomes)

    return CalibrationRun(
        models=records,
        validations=validations,
        k=selected_k,
        difficulty_hint=candidate.difficulty_hint,
        band=band,
        validation_calls=validation_calls,
        rollout_calls=len(outcomes),
        usage=validation_usage + rollout_usage,
        cost=validation_cost + rollout_cost,
    )


def _build_validator(
    *, prompt: str, max_tokens: int, num_retries: int, timeout: float
) -> ValidatorFn:
    async def _validate(model: PanelModel) -> ModelValidation:
        return await validate_model(
            model.model_string,
            base_url=model.base_url,
            api_key=model.api_key,
            prompt=prompt,
            max_tokens=max_tokens,
            num_retries=num_retries,
            timeout=timeout,
        )

    return _validate


def _build_rollout_fn(
    candidate: Candidate,
    env_image: EnvImage,
    spec: GeneratedSpec,
    oracle_report: OracleReport,
    *,
    adapter: LanguageAdapter | None,
    docker_client: object | None,
    max_turns: int,
    max_tokens: int,
    command_timeout: float,
) -> RolloutFn:
    solvers: dict[str, AgenticSolver] = {}

    def _solver_for(model: PanelModel) -> AgenticSolver:
        if model.model_string not in solvers:
            solvers[model.model_string] = AgenticSolver(
                client=model.client(max_tokens=max_tokens, timeout=command_timeout),
                max_turns=max_turns,
                max_tokens=max_tokens,
            )
        return solvers[model.model_string]

    async def _rollout(model: PanelModel, index: int) -> RolloutOutcome:
        from swe_forge.execution.docker_client import DockerClient

        client = docker_client if isinstance(docker_client, DockerClient) else None
        return await run_solver_rollout(
            candidate,
            env_image,
            spec,
            oracle_report,
            model=model.model_string,
            solver=_solver_for(model),
            adapter=adapter,
            docker_client=client,
            command_timeout=command_timeout,
        )

    return _rollout


__all__ = [
    "DEFAULT_BUDGET",
    "CalibrationRun",
    "CalibrationRunnerError",
    "ModelCalibration",
    "RolloutBudget",
    "RolloutFn",
    "ValidatorFn",
    "compute_pass_at_k",
    "run_panel_calibration",
    "suppress_litellm_async_warning",
]
