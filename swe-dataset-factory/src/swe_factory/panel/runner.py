"""Hardness panel runner: fixed scaffold rollouts on required models.

Real-PR wave required OpenRouter ids (exact, no silent substitution; VAL-RPANEL-001):
  - ``x-ai/grok-4.5``
  - ``moonshotai/kimi-k2.6``

(Opus is optional / not required for this wave.)

Each physical completion is reserve→complete→settle (or unknown_billing) via
:class:`BudgetLedger`. When remaining budget cannot cover the next worst-case
reservation, the runner **hard-stops** (no further paid calls) and records an
honest ``budget_stop`` — never invents panel scores (VAL-RPANEL-002/004).

Offline unit tests inject a **soft solver** that decides solve/error without
Docker or live LLM. Certified Real-PR panel trials declare Pier / mini-swe-agent
scaffold metadata (VAL-RPANEL-003).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from swe_factory.accounting import AccountingError, BudgetLedger
from swe_factory.config import DEFAULT_PANEL_MODELS
from swe_factory.openrouter import (
    ChatClient,
    ChatResult,
    OpenRouterBillingError,
    OpenRouterClient,
    OpenRouterError,
    TokenUsage,
)
from swe_factory.panel.band import (
    DEFAULT_BAND_FILTER,
    BandDecision,
    BandFilterConfig,
    classify_band,
    compute_discrimination,
    compute_pass_at_k,
    hardness_dict_from_decision,
)
from swe_factory.schema import PanelHardness

# Exact two-model Real-PR pair (no silent substitution).
REQUIRED_PANEL_MODELS: tuple[str, ...] = tuple(DEFAULT_PANEL_MODELS)
REAL_PR_PANEL_MODELS: tuple[str, ...] = REQUIRED_PANEL_MODELS

# Canonical scaffold identity for DeepSWE hardness panel (Pier-loadable packs).
PANEL_SCAFFOLD_NAME = "pier/mini-swe-agent"
PANEL_SCAFFOLD_AGENT = "mini-swe-agent"
PANEL_SCAFFOLD_RUNTIME = "pier"
PANEL_SCAFFOLD_VERSION = "fixed-v3-real-pr"

DEFAULT_PANEL_K = 2
DEFAULT_MAX_TOKENS = 256
DEFAULT_ROLLOUT_RESERVE_USD = Decimal("1.50")

# Fixed scaffold prompt wrapper — problem statement only (no gold / no tests).
SCAFFOLD_SYSTEM = (
    "You are a software engineering agent fixing a multi-file bug. "
    "Respond with ONLY a unified diff patch that fixes the described issue. "
    "Paths must be repository-relative (e.g. --- a/pkg/mod.py). "
    "Do not invent fixtures/ prefixes. Do not include explanations, tool calls, "
    "or analysis outside the patch. If uncertain, still emit your best full patch."
)


class PanelRunnerError(RuntimeError):
    """Raised when panel configuration or keeps fail closed."""


class PanelBudgetStop(PanelRunnerError):
    """Hard stop: remaining budget cannot fund another panel rollout."""

    def __init__(self, message: str, *, remaining_usd: Decimal, need_usd: Decimal) -> None:
        super().__init__(message)
        self.remaining_usd = remaining_usd
        self.need_usd = need_usd


@dataclass(frozen=True, slots=True)
class PanelScaffoldMeta:
    """Pier / mini-swe-agent scaffold identity recorded on every panel result."""

    name: str = PANEL_SCAFFOLD_NAME
    agent: str = PANEL_SCAFFOLD_AGENT
    runtime: str = PANEL_SCAFFOLD_RUNTIME
    version: str = PANEL_SCAFFOLD_VERSION
    pack_path: str | None = None
    pack_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "agent": self.agent,
            "runtime": self.runtime,
            "version": self.version,
            "pack_path": self.pack_path,
            "pack_id": self.pack_id,
            # Legacy alias still used by older ship reports / micro path.
            "scaffold": self.name,
        }


@dataclass(frozen=True, slots=True)
class RolloutOutcome:
    """One fixed-scaffold rollout result."""

    model: str
    index: int
    solved: bool
    physical_call_id: str
    cost_usd: Decimal
    usage: TokenUsage
    error: str | None = None
    text: str = ""
    skipped_budget: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "index": self.index,
            "solved": self.solved,
            "physical_call_id": self.physical_call_id,
            "cost_usd": format(self.cost_usd, "f"),
            "usage": self.usage.to_dict(),
            "error": self.error,
            "text_bytes": len(self.text.encode("utf-8")),
            "skipped_budget": self.skipped_budget,
        }


@dataclass(frozen=True, slots=True)
class ModelRolloutStats:
    model: str
    k: int
    solves: int
    pass_at_k: float
    rollouts: tuple[RolloutOutcome, ...]
    completed_rollouts: int = 0
    incomplete: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "k": self.k,
            "solves": self.solves,
            "pass_at_k": self.pass_at_k,
            "completed_rollouts": self.completed_rollouts,
            "incomplete": self.incomplete,
            "rollouts": [r.to_dict() for r in self.rollouts],
        }


@dataclass
class PanelRunResult:
    """Full hardness panel run over fixed models + band decision."""

    task_id: str
    stage: str
    models: list[ModelRolloutStats]
    decision: BandDecision
    scaffold: str
    total_cost_usd: Decimal
    reserved_models: tuple[str, ...] = REQUIRED_PANEL_MODELS
    scaffold_meta: PanelScaffoldMeta = field(default_factory=PanelScaffoldMeta)
    budget_stop: bool = False
    panel_complete: bool = True
    stop_reason: str | None = None
    planned_rollouts: int = 0
    completed_rollouts: int = 0

    @property
    def is_keep(self) -> bool:
        # Incomplete mid-keep panel without full matrix is never a certified keep
        # (VAL-DPANEL-002) — even if partial pass@k would otherwise look in-band.
        if not self.panel_complete or self.budget_stop:
            return False
        return self.decision.is_keep

    def panel_hardness(self) -> PanelHardness:
        return self.decision.to_panel_hardness()

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "models": [m.to_dict() for m in self.models],
            "decision": self.decision.to_dict(),
            "hardness": hardness_dict_from_decision(self.decision),
            "scaffold": self.scaffold,
            "scaffold_meta": self.scaffold_meta.to_dict(),
            "total_cost_usd": format(self.total_cost_usd, "f"),
            "reserved_models": list(self.reserved_models),
            "is_keep": self.is_keep,
            "budget_stop": self.budget_stop,
            "panel_complete": self.panel_complete,
            "stop_reason": self.stop_reason,
            "planned_rollouts": self.planned_rollouts,
            "completed_rollouts": self.completed_rollouts,
        }


@dataclass
class MultiKeepPanelResult:
    """Batch hardness panel over many keeps with hard budget stop at $0."""

    keep_results: list[PanelRunResult]
    budget_stop: bool
    stop_reason: str | None
    completed_keep_ids: list[str]
    partial_keep_ids: list[str]
    skipped_keep_ids: list[str]
    total_cost_usd: Decimal
    remaining_usd: Decimal
    models: tuple[str, ...]
    scaffold_meta: PanelScaffoldMeta

    def to_dict(self) -> dict[str, object]:
        return {
            "budget_stop": self.budget_stop,
            "stop_reason": self.stop_reason,
            "completed_keep_ids": list(self.completed_keep_ids),
            "partial_keep_ids": list(self.partial_keep_ids),
            "skipped_keep_ids": list(self.skipped_keep_ids),
            "total_cost_usd": format(self.total_cost_usd, "f"),
            "remaining_usd": format(self.remaining_usd, "f"),
            "models": list(self.models),
            "scaffold_meta": self.scaffold_meta.to_dict(),
            "keep_results": [r.to_dict() for r in self.keep_results],
            # Honesty: no fabricated panel scores for unfinished candidates.
            "invented_rewards": False,
        }


# Soft solver: given model + messages returns whether the rollout "solves"
# without needing Docker. Offline unit tests use this exclusively.
SoftSolverFn = Callable[[str, Sequence[dict[str, str]], ChatResult | None], bool]


def build_scaffold_messages(problem_statement: str) -> list[dict[str, str]]:
    """Fixed scaffold messages for panel rollouts (problem statement only)."""
    statement = problem_statement.strip()
    if not statement:
        raise PanelRunnerError("problem_statement must be non-empty")
    return [
        {"role": "system", "content": SCAFFOLD_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Fix the following issue. Return only a unified diff patch.\n\n{statement}"
            ),
        },
    ]


def build_panel_scaffold_meta(
    *,
    pack_path: str | Path | None = None,
    pack_id: str | None = None,
    task_id: str | None = None,
) -> PanelScaffoldMeta:
    """Construct Pier/mini-swe-agent scaffold metadata for a panel run."""
    path_s = str(pack_path).strip() if pack_path is not None else None
    if path_s == "":
        path_s = None
    pid = (pack_id or task_id or "").strip() or None
    return PanelScaffoldMeta(
        name=PANEL_SCAFFOLD_NAME,
        agent=PANEL_SCAFFOLD_AGENT,
        runtime=PANEL_SCAFFOLD_RUNTIME,
        version=PANEL_SCAFFOLD_VERSION,
        pack_path=path_s,
        pack_id=pid,
    )


def _resolve_cost(
    *,
    client: ChatClient,
    result: ChatResult,
    allow_missing_as_zero: bool,
) -> Decimal:
    if result.cost_usd is not None and result.cost_usd >= 0:
        return result.cost_usd
    # Optional generation cost lookup for live OpenRouter client.
    if isinstance(client, OpenRouterClient) and result.request_id:
        try:
            return client.fetch_generation_cost(result.request_id)
        except OpenRouterBillingError:
            pass
    if allow_missing_as_zero:
        return Decimal("0")
    raise OpenRouterBillingError("provider cost unavailable for settlement")


def _estimate_rollout_budget(
    models: Sequence[str],
    k: int,
    reserve_usd: Decimal,
) -> Decimal:
    return reserve_usd * Decimal(len(models) * k)


def canary_affordable(
    ledger: BudgetLedger,
    *,
    models: Sequence[str] = REQUIRED_PANEL_MODELS,
    k: int = 1,
    reserve_usd: Decimal = DEFAULT_ROLLOUT_RESERVE_USD,
) -> bool:
    """Return True if a cheap live canary (k=1 × models) fits remaining budget."""
    need = _estimate_rollout_budget(models, k, reserve_usd)
    return ledger.remaining_usd() >= need and not ledger.has_unknown_billing()


def full_panel_affordable(
    ledger: BudgetLedger,
    *,
    models: Sequence[str] = REQUIRED_PANEL_MODELS,
    k: int = DEFAULT_PANEL_K,
    reserve_usd: Decimal = DEFAULT_ROLLOUT_RESERVE_USD,
) -> bool:
    """Return True if a full panel matrix for one keep fits remaining budget."""
    need = _estimate_rollout_budget(models, k, reserve_usd)
    return ledger.remaining_usd() >= need and not ledger.has_unknown_billing()


def assert_required_panel_models(models: Sequence[str]) -> tuple[str, ...]:
    """Fail closed unless the required Real-PR OpenRouter model ids are present.

    Required pair (VAL-RPANEL-001): ``x-ai/grok-4.5`` + ``moonshotai/kimi-k2.6``.
    Exact string match; silent substitution fails. Extras (e.g. Opus) are allowed
    when callers deliberately opt in, but the paper two must be present in order
    for ``DEFAULT_PANEL_MODELS`` callers.
    """
    models_t = tuple(m.strip() for m in models if m and m.strip())
    if not models_t:
        raise PanelRunnerError("models must be non-empty")
    missing = [m for m in REQUIRED_PANEL_MODELS if m not in models_t]
    if missing:
        raise PanelRunnerError(
            f"panel models must include required set {list(REQUIRED_PANEL_MODELS)}; "
            f"missing {missing}"
        )
    # Reject reordering / silent swap of the required pair: first |models| entries
    # for pure-pair callers should match REQUIRED_PANEL_MODELS exactly when length
    # equals the required set (extras after the pair are fine).
    if len(models_t) == len(REQUIRED_PANEL_MODELS) and models_t != REQUIRED_PANEL_MODELS:
        raise PanelRunnerError(
            f"panel models must be exact pair {list(REQUIRED_PANEL_MODELS)}; "
            f"got {list(models_t)} (no silent substitution)"
        )
    return models_t


def _incomplete_drop_decision(
    *,
    per_model_pass: dict[str, float],
    total_solves: int,
    total_trials: int,
    discrimination: float,
    band_config: BandFilterConfig,
    reason: str,
) -> BandDecision:
    """Build a non-keep decision for budget-stopped / incomplete panels."""
    frontier = 0.0
    if total_trials > 0:
        frontier = total_solves / total_trials
    elif per_model_pass:
        frontier = sum(per_model_pass.values()) / len(per_model_pass)
    return BandDecision(
        verdict="drop",
        rule="budget-stop-incomplete",
        reason=reason,
        frontier_pass_at_k=frontier,
        discrimination=discrimination,
        band_high=band_config.band_high,
        discrimination_floor=band_config.discrimination_floor,
        total_solves=total_solves,
        total_trials=total_trials,
        per_model_pass_at_k=dict(per_model_pass),
    )


def run_panel(
    *,
    task_id: str,
    problem_statement: str,
    ledger: BudgetLedger,
    client: ChatClient | None = None,
    models: Sequence[str] = REQUIRED_PANEL_MODELS,
    k: int = DEFAULT_PANEL_K,
    stage: str = "hardness-panel",
    soft_solver: SoftSolverFn | None = None,
    band_config: BandFilterConfig = DEFAULT_BAND_FILTER,
    reserve_usd: Decimal = DEFAULT_ROLLOUT_RESERVE_USD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    allow_missing_cost_as_zero: bool = False,
    temperature: float = 0.0,
    pack_path: str | Path | None = None,
    pack_id: str | None = None,
    scaffold_meta: PanelScaffoldMeta | None = None,
    stop_on_budget: bool = True,
) -> PanelRunResult:
    """Run fixed scaffold rollouts and apply the hardness band filter.

    Offline unit tests: pass a scripted ``client`` (or stub that raises) and a
    ``soft_solver`` that decides solves from model id / text without Docker.

    Live mode: pass a real :class:`OpenRouterClient` and a soft_solver that
    scores candidate patch text (or a scorer-backed solver). This function always
    reserves before each complete and settles after.

    When ``stop_on_budget`` is True (default), if remaining budget cannot cover
    the next worst-case reservation the runner hard-stops: no further provider
    calls, ``budget_stop=True``, ``panel_complete=False``, and ``is_keep=False``.
    Partial scores are never fabricated into a certified keep (VAL-DPANEL-004/006).
    """
    task_s = task_id.strip()
    if not task_s:
        raise PanelRunnerError("task_id must be non-empty")
    if k <= 0:
        raise PanelRunnerError(f"k must be positive; got {k}")
    models_t = assert_required_panel_models(models)

    meta = scaffold_meta or build_panel_scaffold_meta(
        pack_path=pack_path,
        pack_id=pack_id,
        task_id=task_s,
    )

    messages = build_scaffold_messages(problem_statement)
    if soft_solver is None:
        # Default soft solvers treats non-empty text as not-a-solve so offline
        # without a scorer never accidentally "solves". Live callers should
        # inject a real scorer.
        soft_solver = _default_never_solve

    model_stats: list[ModelRolloutStats] = []
    total_cost = Decimal("0")
    per_model_pass: dict[str, float] = {}
    total_solves = 0
    total_trials = 0
    completed_rollouts = 0
    planned_rollouts = len(models_t) * k
    budget_stop = False
    stop_reason: str | None = None
    halted = False

    for model in models_t:
        if halted:
            # Do not invent model stats for not-started models.
            per_model_pass[model] = 0.0
            model_stats.append(
                ModelRolloutStats(
                    model=model,
                    k=k,
                    solves=0,
                    pass_at_k=0.0,
                    rollouts=(),
                    completed_rollouts=0,
                    incomplete=True,
                )
            )
            continue

        outcomes: list[RolloutOutcome] = []
        solves = 0
        model_completed = 0
        for index in range(k):
            if stop_on_budget:
                remaining = ledger.remaining_usd()
                if remaining < reserve_usd or ledger.has_unknown_billing():
                    budget_stop = True
                    halted = True
                    stop_reason = (
                        f"budget_stop: remaining_usd={format(remaining, 'f')} "
                        f"< reserve_usd={format(reserve_usd, 'f')} "
                        f"(or unknown billing) before {model} rollout {index}; "
                        "refusing further paid panel calls (VAL-RPANEL-002/004)"
                    )
                    break

            try:
                physical = ledger.reserve(
                    stage=stage,
                    task_id=task_s,
                    model=model,
                    reserved_cost_usd=reserve_usd,
                )
            except AccountingError as exc:
                budget_stop = True
                halted = True
                stop_reason = (
                    f"budget_stop: reserve refused for {model} rollout {index}: {exc}; "
                    "no invented panel scores"
                )
                break

            chat: ChatResult | None = None
            err: str | None = None
            cost = Decimal("0")
            usage = TokenUsage()
            text = ""
            try:
                if client is None:
                    raise PanelRunnerError("client is required for live/provider panel rollouts")
                chat = client.complete(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = chat.text
                usage = chat.usage
                try:
                    cost = _resolve_cost(
                        client=client,
                        result=chat,
                        allow_missing_as_zero=allow_missing_cost_as_zero,
                    )
                except OpenRouterBillingError:
                    ledger.mark_unknown_billing(physical, reason_code="missing_provider_cost")
                    raise
                ledger.settle(
                    physical,
                    cost_usd=cost,
                    status="success",
                    usage=usage.to_dict(),
                    request_id=chat.request_id,
                )
                solved = bool(soft_solver(model, messages, chat))
                if solved:
                    solves += 1
                outcomes.append(
                    RolloutOutcome(
                        model=model,
                        index=index,
                        solved=solved,
                        physical_call_id=physical,
                        cost_usd=cost,
                        usage=usage,
                        error=None,
                        text=text,
                    )
                )
                total_cost += cost
                model_completed += 1
                completed_rollouts += 1
            except (OpenRouterError, PanelRunnerError, AccountingError) as exc:
                err = type(exc).__name__
                rec = ledger._records.get(physical)  # noqa: SLF001 — settle if open
                if rec is not None and rec.is_open:
                    try:
                        ledger.settle(
                            physical,
                            cost_usd=Decimal("0"),
                            status="error",
                            usage=usage.to_dict(),
                            request_id="",
                        )
                    except AccountingError:
                        ledger.mark_unknown_billing(physical, reason_code="settle_error")
                outcomes.append(
                    RolloutOutcome(
                        model=model,
                        index=index,
                        solved=False,
                        physical_call_id=physical,
                        cost_usd=cost,
                        usage=usage,
                        error=err,
                        text=text,
                    )
                )
                total_cost += cost
                model_completed += 1
                completed_rollouts += 1

        model_incomplete = model_completed < k
        # Pass@k over planned k only when this model finished all rollouts.
        # Incomplete models contribute only completed trials (no invented results).
        if model_completed >= k:
            pass_k = compute_pass_at_k(solves, k)
            total_solves += solves
            total_trials += k
        else:
            if model_completed == 0:
                pass_k = 0.0
            else:
                pass_k = compute_pass_at_k(solves, max(model_completed, 1))
            total_solves += solves
            total_trials += model_completed
        per_model_pass[model] = pass_k
        model_stats.append(
            ModelRolloutStats(
                model=model,
                k=k,
                solves=solves,
                pass_at_k=pass_k,
                rollouts=tuple(outcomes),
                completed_rollouts=model_completed,
                incomplete=model_incomplete,
            )
        )

    panel_complete = completed_rollouts >= planned_rollouts and not budget_stop

    disc = compute_discrimination(per_model_pass) if per_model_pass else 0.0
    if not panel_complete:
        decision = _incomplete_drop_decision(
            per_model_pass=per_model_pass,
            total_solves=total_solves,
            total_trials=total_trials if total_trials > 0 else planned_rollouts,
            discrimination=disc,
            band_config=band_config,
            reason=stop_reason
            or (
                "incomplete panel matrix — cannot certify keep without full panel "
                f"({completed_rollouts}/{planned_rollouts} rollouts)"
            ),
        )
    else:
        decision = classify_band(
            per_model_pass_at_k=per_model_pass,
            total_solves=total_solves,
            total_trials=total_trials if total_trials > 0 else planned_rollouts,
            discrimination=disc,
            config=band_config,
        )
        # Unknown billing: fail closed — never keep.
        if ledger.has_unknown_billing() and decision.is_keep:
            decision = BandDecision(
                verdict="drop",
                rule="unknown-billing",
                reason=(
                    "drop: ledger has unknown_billing — cannot certify keep without "
                    "exact provider metering (VAL-RPANEL-004 fail closed)"
                ),
                frontier_pass_at_k=decision.frontier_pass_at_k,
                discrimination=decision.discrimination,
                band_high=decision.band_high,
                discrimination_floor=decision.discrimination_floor,
                total_solves=decision.total_solves,
                total_trials=decision.total_trials,
                per_model_pass_at_k=decision.per_model_pass_at_k,
            )

    return PanelRunResult(
        task_id=task_s,
        stage=stage,
        models=model_stats,
        decision=decision,
        scaffold=meta.name,
        total_cost_usd=total_cost,
        reserved_models=models_t,
        scaffold_meta=meta,
        budget_stop=budget_stop,
        panel_complete=panel_complete,
        stop_reason=stop_reason,
        planned_rollouts=planned_rollouts,
        completed_rollouts=completed_rollouts,
    )


def run_panel_until_budget_zero(
    *,
    keeps: Sequence[dict[str, Any]],
    ledger: BudgetLedger,
    client: ChatClient | None = None,
    models: Sequence[str] = REQUIRED_PANEL_MODELS,
    k: int = DEFAULT_PANEL_K,
    stage: str = "hardness-panel",
    soft_solver: SoftSolverFn | None = None,
    band_config: BandFilterConfig = DEFAULT_BAND_FILTER,
    reserve_usd: Decimal = DEFAULT_ROLLOUT_RESERVE_USD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    allow_missing_cost_as_zero: bool = False,
    temperature: float = 0.0,
    require_full_matrix_for_keep: bool = True,
) -> MultiKeepPanelResult:
    """Run full panel on every keep while remaining budget > 0.

    ``keeps`` entries are dicts with at least::

        {"task_id": str, "problem_statement": str,
         "pack_path": optional, "pack_id": optional}

    Stops paid panel when a full matrix cannot fit remaining budget or when a
    mid-keep reserve fails. Completed keeps before the stop retain honest
    evidence; unfinished / partial keeps are **not** certified (no invented
    rewards). VAL-DPANEL-002/004/006.
    """
    models_t = assert_required_panel_models(models)
    meta_global = build_panel_scaffold_meta()
    results: list[PanelRunResult] = []
    completed: list[str] = []
    partial: list[str] = []
    skipped: list[str] = []
    budget_stop = False
    stop_reason: str | None = None
    total_cost = Decimal("0")

    full_need = _estimate_rollout_budget(models_t, k, reserve_usd)
    keep_list = list(keeps)

    for keep_idx, entry in enumerate(keep_list):
        task_id = str(entry.get("task_id") or entry.get("instance_id") or "").strip()
        if not task_id:
            raise PanelRunnerError("keep entry missing task_id")
        problem = str(entry.get("problem_statement") or entry.get("instruction") or "").strip()
        if not problem:
            raise PanelRunnerError(f"keep {task_id!r} missing problem_statement")
        pack_path = entry.get("pack_path")
        pack_id = entry.get("pack_id") or task_id

        remaining = ledger.remaining_usd()
        if remaining <= 0 or remaining < full_need or ledger.has_unknown_billing():
            budget_stop = True
            stop_reason = (
                f"budget_stop: remaining_usd={format(remaining, 'f')} "
                f"< full panel need {format(full_need, 'f')} before keep {task_id!r}; "
                "no further paid panel calls; no invented rewards"
            )
            # Record this and remaining unstarted keeps as skipped.
            for later in keep_list[keep_idx:]:
                later_id = str(later.get("task_id") or later.get("instance_id") or "").strip()
                if later_id and later_id not in skipped:
                    skipped.append(later_id)
            break

        result = run_panel(
            task_id=task_id,
            problem_statement=problem,
            ledger=ledger,
            client=client,
            models=models_t,
            k=k,
            stage=stage,
            soft_solver=soft_solver,
            band_config=band_config,
            reserve_usd=reserve_usd,
            max_tokens=max_tokens,
            allow_missing_cost_as_zero=allow_missing_cost_as_zero,
            temperature=temperature,
            pack_path=pack_path,
            pack_id=str(pack_id) if pack_id else task_id,
            stop_on_budget=True,
        )
        results.append(result)
        total_cost += result.total_cost_usd

        if result.budget_stop or not result.panel_complete:
            budget_stop = True
            stop_reason = result.stop_reason or (
                f"budget_stop: incomplete panel on keep {task_id!r}"
            )
            partial.append(task_id)
            for later in keep_list[keep_idx + 1 :]:
                later_id = str(later.get("task_id") or later.get("instance_id") or "").strip()
                if later_id and later_id not in skipped and later_id != task_id:
                    skipped.append(later_id)
            break

        if require_full_matrix_for_keep and not result.panel_complete:
            partial.append(task_id)
        else:
            completed.append(task_id)

    remaining_after = ledger.remaining_usd()
    if remaining_after <= 0 and not budget_stop:
        budget_stop = True
        stop_reason = stop_reason or (
            f"budget_stop: remaining_usd={format(remaining_after, 'f')} after panel"
        )

    return MultiKeepPanelResult(
        keep_results=results,
        budget_stop=budget_stop,
        stop_reason=stop_reason,
        completed_keep_ids=completed,
        partial_keep_ids=partial,
        skipped_keep_ids=skipped,
        total_cost_usd=total_cost,
        remaining_usd=remaining_after,
        models=models_t,
        scaffold_meta=meta_global,
    )


def _default_never_solve(
    model: str,
    messages: Sequence[dict[str, str]],
    chat: ChatResult | None,
) -> bool:
    del model, messages, chat
    return False


def offline_panel_from_matrix(
    *,
    task_id: str,
    solve_matrix: dict[str, list[bool]],
    ledger: BudgetLedger | None = None,
    ledger_path: Any = None,
    stage: str = "hardness-panel",
    band_config: BandFilterConfig = DEFAULT_BAND_FILTER,
    problem_statement: str = "Offline synthetic hardness matrix.",
    cap_usd: Decimal = Decimal("600"),
    pack_path: str | Path | None = None,
    pack_id: str | None = None,
    stop_on_budget: bool = True,
    reserve_usd: Decimal = Decimal("0.01"),
) -> PanelRunResult:
    """Pure offline panel path from an explicit solve matrix.

    ``solve_matrix`` maps model_id → list of per-rollout bool solves (length = k).
    Uses a scripted client + soft solver. Records reserve/settle with cost 0.
    Ideal for unit tests and band verification without network.
    """
    from swe_factory.openrouter import ScriptedChatClient

    models = list(solve_matrix.keys())
    if not models:
        raise PanelRunnerError("solve_matrix must be non-empty")
    # Ensure required DeepSWE models are present for certified-path demos.
    # Offline band demos may still pass a subset only via assert in run_panel
    # which will raise — callers of offline demos should include the triad.
    k = len(next(iter(solve_matrix.values())))
    if k <= 0:
        raise PanelRunnerError("solve matrix rows must be non-empty")
    for model, row in solve_matrix.items():
        if len(row) != k:
            raise PanelRunnerError(f"model {model!r} has k={len(row)} != expected {k}")

    if ledger is None:
        path = (
            Path(ledger_path)
            if ledger_path is not None
            else Path("/tmp/panel-offline-ledger.jsonl")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger = BudgetLedger(path, cap_usd=cap_usd, worst_case_cost_usd=reserve_usd)

    # One scripted response per (model, rollout)
    responses: list[ChatResult | Exception] = []
    for model in models:
        for i, _solved in enumerate(solve_matrix[model]):
            responses.append(
                ChatResult(
                    model=model,
                    text=f"PATCH {model} #{i}",
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    request_id=f"offline-{model}-{i}",
                    cost_usd=Decimal("0"),
                    finish_reason="stop",
                    raw_usage={},
                )
            )
    client = ScriptedChatClient(responses=responses)

    # Soft solver: sequential call order is models outer, index inner.
    indices: dict[str, int] = {m: 0 for m in models}

    def solver(
        model: str,
        messages: Sequence[dict[str, str]],
        chat: ChatResult | None,
    ) -> bool:
        del messages, chat
        row = solve_matrix[model]
        i = indices[model]
        if i >= len(row):
            return False
        indices[model] = i + 1
        return bool(row[i])

    return run_panel(
        task_id=task_id,
        problem_statement=problem_statement,
        ledger=ledger,
        client=client,
        models=models,
        k=k,
        stage=stage,
        soft_solver=solver,
        band_config=band_config,
        reserve_usd=reserve_usd,
        allow_missing_cost_as_zero=True,
        pack_path=pack_path,
        pack_id=pack_id or task_id,
        stop_on_budget=stop_on_budget,
    )


def offline_tworollout_borderline_matrix(
    models: Sequence[str] | None = None,
) -> dict[str, list[bool]]:
    """Standard offline keep-band matrix for CLI demos (Real-PR pair).

    Default two-model matrix: Grok 0/4, Kimi 2/4 → aggregate 2/8=0.25 with
    discrimination from 0.0 vs 0.5. Extra models (if provided) get zero solvs
    except the last which still gets the 2/4 pinnable row.
    """
    models_t = tuple(models) if models is not None else REQUIRED_PANEL_MODELS
    matrix: dict[str, list[bool]] = {}
    for i, model in enumerate(models_t):
        if i == len(models_t) - 1:
            # Last model gets pinnable partial solves (2 of 4).
            matrix[model] = [True, False, True, False]
        else:
            matrix[model] = [False, False, False, False]
    return matrix


def discover_real_pr_panel_keeps(
    sources: Sequence[str | Path],
) -> list[dict[str, Any]]:
    """Load newly certified / staged Real-PR keeps for live or offline panel.

    Accepts pack directories, a product root containing ``tasks/``, or candidate
    directories that already carry instruction / task.toml metadata. Does **not**
    invent problem statements: missing instruction material is skipped with
    an explicit reason (never fabricated rewards).
    """
    keeps: list[dict[str, Any]] = []
    for raw in sources:
        root = Path(raw)
        candidates: list[Path] = []
        if not root.exists():
            continue
        if (root / "tasks").is_dir():
            candidates.extend(sorted(p for p in (root / "tasks").iterdir() if p.is_dir()))
        elif root.is_dir() and (
            (root / "instruction.md").is_file()
            or (root / "task.toml").is_file()
            or (root / "task.json").is_file()
        ):
            candidates.append(root)
        elif root.is_dir():
            # Pool layout: children are candidate packs / task records.
            for child in sorted(root.iterdir()):
                if child.is_dir() and (
                    (child / "instruction.md").is_file()
                    or (child / "task.toml").is_file()
                    or (child / "problem_statement.md").is_file()
                    or (child / "task.json").is_file()
                ):
                    candidates.append(child)
        for pack in candidates:
            task_id = pack.name
            problem = ""
            for name in (
                "instruction.md",
                "problem_statement.md",
                "PROBLEM.md",
            ):
                p = pack / name
                if p.is_file():
                    problem = p.read_text(encoding="utf-8", errors="replace").strip()
                    break
            if not problem:
                tj = pack / "task.json"
                if tj.is_file():
                    try:
                        blob = json.loads(tj.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        blob = {}
                    if isinstance(blob, dict):
                        problem = str(
                            blob.get("problem_statement") or blob.get("instruction") or ""
                        ).strip()
                        task_id = str(blob.get("instance_id") or blob.get("task_id") or task_id)
            if not problem:
                # Still emit a keep entry only when task.toml proves the pack
                # exists; problem remains required by run_panel — ship code
                # can fill instruction later for true certified packs.
                continue
            keeps.append(
                {
                    "task_id": task_id,
                    "problem_statement": problem,
                    "pack_path": str(pack.resolve()),
                    "pack_id": task_id,
                    "source_track": "real_pr",
                }
            )
    return keeps


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_PANEL_K",
    "DEFAULT_ROLLOUT_RESERVE_USD",
    "PANEL_SCAFFOLD_AGENT",
    "PANEL_SCAFFOLD_NAME",
    "PANEL_SCAFFOLD_RUNTIME",
    "PANEL_SCAFFOLD_VERSION",
    "REAL_PR_PANEL_MODELS",
    "REQUIRED_PANEL_MODELS",
    "SCAFFOLD_SYSTEM",
    "ModelRolloutStats",
    "MultiKeepPanelResult",
    "PanelBudgetStop",
    "PanelRunResult",
    "PanelRunnerError",
    "PanelScaffoldMeta",
    "RolloutOutcome",
    "SoftSolverFn",
    "assert_required_panel_models",
    "build_panel_scaffold_meta",
    "build_scaffold_messages",
    "canary_affordable",
    "discover_real_pr_panel_keeps",
    "full_panel_affordable",
    "offline_panel_from_matrix",
    "offline_tworollout_borderline_matrix",
    "run_panel",
    "run_panel_until_budget_zero",
]
