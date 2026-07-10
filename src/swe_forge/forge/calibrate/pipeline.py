"""Calibration pipeline: assemble a finalized CalibrationReport from a panel run.

This is the last Stage 4 step (architecture S6, Stage 4). The panel runner
(:mod:`swe_forge.forge.calibrate.runner`) measures the per-model/per-rollout
solve matrix and bills every LLM call; the IRT fitter
(:mod:`swe_forge.forge.calibrate.irt`) turns that matrix into difficulty +
discrimination; the band filter (:mod:`swe_forge.forge.calibrate.filter`) assigns
the terminal keep/drop verdict. This module wires those three together into one
end-to-end run that produces a single, self-contained
:class:`~swe_forge.forge.models.CalibrationReport`:

* the per-model ``{model, tier, k, solves, pass_at_k}`` array spanning the panel
  tiers, ``irt_difficulty``/``irt_discrimination``, and the terminal
  ``band_verdict`` + ``reason`` (VAL-CAL-020);
* a complete **usage/cost accounting** recorded under
  ``details["usage_accounting"]`` -- per-call for every pre-flight validation AND
  every rollout, PLUS an aggregate, so no per-call data is lost behind the total
  (VAL-CAL-019);
* full provenance (generator, seed, language, tool versions, timestamp).

Solve semantics are the runner's (and therefore the solver's): a rollout counts
as a solve only when its submitted patch passes the FULL
``OracleReport.test_files[]`` hidden suite AND the P2P/regression suite stays
green -- never just the original F2P. The assembly here does not re-score; it
consumes the runner's recorded outcomes.

The teacher LLM proposes; deterministic counting disposes: the report assembly,
IRT fit, and band verdict are pure deterministic functions of the recorded solve
matrix, so re-running on the same panel config reproduces the same schema, the
same keep/drop rule application (given an equivalent solve band), and the same
discrimination direction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from swe_forge.forge.adapters import LanguageAdapter
from swe_forge.forge.calibrate.filter import (
    DEFAULT_BAND_FILTER,
    BandFilterConfig,
    apply_band_filter,
)
from swe_forge.forge.calibrate.irt import build_calibration_report
from swe_forge.forge.calibrate.runner import (
    DEFAULT_BUDGET,
    CalibrationRunnerError,
    CalibrationRun,
    RolloutBudget,
    RolloutFn,
    ValidatorFn,
    run_panel_calibration,
)
from swe_forge.forge.oracle.multifault import verify_multifault_evidence
from swe_forge.forge.oracle.mutation import DEFAULT_KILL_THRESHOLD
from swe_forge.forge.oracle.pipeline import verify_pass_consistency
from swe_forge.forge.calibrate.solver import DEFAULT_MAX_TOKENS, DEFAULT_MAX_TURNS
from swe_forge.forge.models import (
    CalibrationReport,
    Candidate,
    EnvImage,
    GeneratedSpec,
    OracleReport,
    Provenance,
)
from swe_forge.forge.panel import (
    DEFAULT_ROLLOUT_CONCURRENCY,
    DEFAULT_VALIDATE_MAX_TOKENS,
    DEFAULT_VALIDATE_NUM_RETRIES,
    DEFAULT_VALIDATE_TIMEOUT,
    PanelModel,
)
from swe_forge.forge.recovery_accounting import RecoveryBudgetLedger
from swe_forge.forge.teacher import Usage


def _sum_usage(items: Iterable[Usage]) -> Usage:
    total = Usage()
    for item in items:
        total = total + item
    return total


def _tool_versions() -> dict[str, str]:
    """Best-effort tool-version capture for provenance (LiteLLM is the LLM path)."""
    versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            versions["litellm"] = version("litellm")
        except PackageNotFoundError:  # pragma: no cover - litellm is a hard dep
            pass
    except Exception:  # pragma: no cover - importlib.metadata always present on 3.12
        pass
    return versions


def build_usage_accounting(run: CalibrationRun) -> dict[str, object]:
    """Build the per-call + aggregate usage/cost accounting for a panel run.

    Encodes VAL-CAL-019: every LLM call in the calibration run -- each pre-flight
    model-id validation AND each rollout -- contributes its own token usage and
    cost to a ``per_call`` list, and the section totals plus a final ``aggregate``
    summarize them WITHOUT discarding the per-call detail. The aggregate mirrors
    the runner's own ``usage``/``cost``/``total_calls`` so the two never diverge.
    """
    validation_per_call: list[dict[str, object]] = [
        {
            "model": v.model,
            "valid": v.valid,
            "usage": (v.usage if v.usage is not None else Usage()).to_dict(),
            "cost": v.cost,
            "recovery_accounting": (
                dict(v.recovery_accounting)
                if v.recovery_accounting is not None
                else None
            ),
        }
        for v in run.validations
    ]
    validation_usage = _sum_usage(
        v.usage for v in run.validations if v.usage is not None
    )
    validation_cost = sum(v.cost for v in run.validations)

    rollout_per_call: list[dict[str, object]] = []
    rollout_usage = Usage()
    rollout_cost = 0.0
    for record in run.models:
        for index, outcome in enumerate(record.rollouts):
            rollout_per_call.append(
                {
                    "model": record.model,
                    "tier": record.tier,
                    "index": index,
                    "solved": outcome.solved,
                    "usage": outcome.usage.to_dict(),
                    "cost": outcome.cost,
                    "recovery_accounting": [
                        dict(item) for item in outcome.recovery_accounting
                    ],
                }
            )
            rollout_usage = rollout_usage + outcome.usage
            rollout_cost += outcome.cost

    return {
        "validation": {
            "calls": run.validation_calls,
            "usage": validation_usage.to_dict(),
            "cost": validation_cost,
            "per_call": validation_per_call,
        },
        "rollout": {
            "calls": run.rollout_calls,
            "usage": rollout_usage.to_dict(),
            "cost": rollout_cost,
            "per_call": rollout_per_call,
        },
        "aggregate": {
            "total_calls": run.total_calls,
            "usage": run.usage.to_dict(),
            "cost": run.cost,
        },
    }


def build_recovery_accounting(run: CalibrationRun) -> list[dict[str, object]]:
    """Collect physical-call evidence without losing its oracle-calibration link."""
    entries: list[dict[str, object]] = []
    for validation in run.validations:
        if validation.recovery_accounting is not None:
            entries.append(dict(validation.recovery_accounting))
    for record in run.models:
        for outcome in record.rollouts:
            entries.extend(dict(item) for item in outcome.recovery_accounting)
    return entries


def _default_provenance(candidate: Candidate) -> Provenance:
    """Provenance for a CalibrationReport derived from its candidate."""
    return Provenance(
        generator=candidate.generator,
        seed=candidate.provenance.seed,
        language=candidate.language,
        tool_versions=_tool_versions(),
        details={"stage": "calibration"},
    )


def assemble_calibration_report(
    run: CalibrationRun,
    *,
    language: str,
    config: BandFilterConfig = DEFAULT_BAND_FILTER,
    difficulty_hint: str = "",
    provenance: Provenance | None = None,
    tier_abilities: dict[str, float] | None = None,
) -> CalibrationReport:
    """Assemble a finalized :class:`CalibrationReport` from a :class:`CalibrationRun`.

    Fits the 2-parameter IRT over the run's per-model solve matrix (via
    :func:`build_calibration_report`), records the full per-call + aggregate
    usage/cost accounting and the run summary under ``details``, then applies the
    band filter to set the terminal ``band_verdict`` + ``reason``. Pure and
    deterministic given the run, so re-assembling an equivalent run reproduces the
    same schema, keep/drop rule, and discrimination direction (VAL-CAL-020).
    """
    details: dict[str, object] = {
        "usage_accounting": build_usage_accounting(run),
        "recovery_accounting": build_recovery_accounting(run),
        "calibration": {
            "band": run.band,
            "difficulty_hint": run.difficulty_hint,
            "validation_calls": run.validation_calls,
            "rollout_calls": run.rollout_calls,
            "total_calls": run.total_calls,
            "cost": run.cost,
            "usage": run.usage.to_dict(),
            "validations": [v.to_dict() for v in run.validations],
        },
    }
    report = build_calibration_report(
        language,
        run.models,
        k=run.k,
        difficulty_hint=difficulty_hint or run.difficulty_hint,
        tier_abilities=tier_abilities,
        provenance=provenance,
        details=details,
    )
    apply_band_filter(report, config=config)
    return report


@dataclass
class CalibrationOutcome:
    """The end-to-end calibration result: the finalized report + the raw run.

    ``report`` is the shippable :class:`CalibrationReport` (IRT + band verdict +
    usage accounting + provenance); ``run`` is the underlying
    :class:`CalibrationRun` (per-model records, validations, raw call accounting)
    kept for auditing and richer CLI/log output.
    """

    report: CalibrationReport
    run: CalibrationRun

    def to_dict(self) -> dict[str, object]:
        return {"report": self.report.to_dict(), "run": self.run.to_dict()}


async def run_calibration(
    candidate: Candidate,
    env_image: EnvImage,
    spec: GeneratedSpec,
    oracle_report: OracleReport,
    panel: list[PanelModel],
    *,
    budget: RolloutBudget = DEFAULT_BUDGET,
    k: int | None = None,
    concurrency: int = DEFAULT_ROLLOUT_CONCURRENCY,
    validate: bool = True,
    config: BandFilterConfig = DEFAULT_BAND_FILTER,
    tier_abilities: dict[str, float] | None = None,
    provenance: Provenance | None = None,
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
    recovery_ledger: RecoveryBudgetLedger | None = None,
) -> CalibrationOutcome:
    """Run the full panel calibration end to end -> a finalized CalibrationReport.

    Drives :func:`run_panel_calibration` (validate-before-bulk; ``k`` independent,
    uncached, concurrency-bounded rollouts per validated model; difficulty-aware
    budget; every rollout scored via the SHARED Docker FAIL->PASS path enforcing
    the FULL ``OracleReport.test_files[]`` suite + P2P), then assembles the report
    (IRT fit + band verdict + per-call/aggregate usage accounting). The panel
    runner resolves the candidate's :class:`LanguageAdapter`, so scoring uses that
    language's test command (cross-language parity, VAL-CAL-023). ``validator`` /
    ``rollout_fn`` are injectable seams for deterministic offline tests; the
    defaults drive the live panel endpoint + the throwaway Docker scorer.
    """
    if oracle_report.verdict != "pass":
        raise CalibrationRunnerError(
            "calibration refused: oracle verdict must be 'pass'; got "
            f"{oracle_report.verdict!r}"
        )
    threshold = (
        oracle_report.final_mutation_evidence.threshold
        if oracle_report.final_mutation_evidence is not None
        else DEFAULT_KILL_THRESHOLD
    )
    problems = [
        *verify_pass_consistency(oracle_report, kill_threshold=threshold),
        *verify_multifault_evidence(oracle_report, candidate=candidate),
    ]
    if problems:
        raise CalibrationRunnerError(
            "calibration refused: final oracle evidence is inconsistent ("
            + "; ".join(dict.fromkeys(problems))
            + ")"
        )
    run = await run_panel_calibration(
        candidate,
        env_image,
        spec,
        oracle_report,
        panel,
        budget=budget,
        k=k,
        concurrency=concurrency,
        validate=validate,
        validate_prompt=validate_prompt,
        validate_max_tokens=validate_max_tokens,
        validate_num_retries=validate_num_retries,
        validate_timeout=validate_timeout,
        max_turns=max_turns,
        max_tokens=max_tokens,
        command_timeout=command_timeout,
        adapter=adapter,
        docker_client=docker_client,
        validator=validator,
        rollout_fn=rollout_fn,
        recovery_ledger=recovery_ledger,
    )
    report = assemble_calibration_report(
        run,
        language=candidate.language,
        config=config,
        difficulty_hint=candidate.difficulty_hint,
        provenance=provenance
        if provenance is not None
        else _default_provenance(candidate),
        tier_abilities=tier_abilities,
    )
    return CalibrationOutcome(report=report, run=run)


__all__ = [
    "CalibrationOutcome",
    "assemble_calibration_report",
    "build_usage_accounting",
    "build_recovery_accounting",
    "run_calibration",
]
