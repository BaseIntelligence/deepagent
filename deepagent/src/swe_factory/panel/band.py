"""Hardness band filter for frontier panel (\

keep only if ``0 < pass@k_frontier_aggregate ≤ 0.5`` with discrimination floor).

Drops solve-all and solve-none. Deterministic pure math — no provider calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swe_factory.schema import PanelHardness

DEFAULT_BAND_HIGH = 0.5
DEFAULT_DISCRIMINATION_FLOOR = 1.0

RULE_KEEP = "in-band-high-discrimination"
RULE_SOLVE_NONE = "solve-none"
RULE_SOLVE_ALL = "solve-all"
RULE_OUT_OF_BAND = "out-of-band"
RULE_LOW_DISCRIMINATION = "low-discrimination"


class BandFilterError(ValueError):
    """Invalid band filter configuration."""


@dataclass(frozen=True, slots=True)
class BandFilterConfig:
    """Thresholds for the hardness keep band."""

    band_high: float = DEFAULT_BAND_HIGH
    discrimination_floor: float = DEFAULT_DISCRIMINATION_FLOOR

    def __post_init__(self) -> None:
        if not (0.0 < self.band_high <= 1.0):
            raise BandFilterError(f"band_high must be in (0, 1]; got {self.band_high}")
        if self.discrimination_floor < 0.0:
            raise BandFilterError(
                f"discrimination_floor must be >= 0; got {self.discrimination_floor}"
            )


DEFAULT_BAND_FILTER = BandFilterConfig()


@dataclass(frozen=True, slots=True)
class BandDecision:
    """Terminal keep/drop classification over frontier aggregate pass@k."""

    verdict: str
    rule: str
    reason: str
    frontier_pass_at_k: float
    discrimination: float
    band_high: float
    discrimination_floor: float
    total_solves: int
    total_trials: int
    per_model_pass_at_k: dict[str, float]

    @property
    def is_keep(self) -> bool:
        return self.verdict == "keep"

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "rule": self.rule,
            "reason": self.reason,
            "frontier_pass_at_k": self.frontier_pass_at_k,
            "discrimination": self.discrimination,
            "band_high": self.band_high,
            "discrimination_floor": self.discrimination_floor,
            "total_solves": self.total_solves,
            "total_trials": self.total_trials,
            "per_model_pass_at_k": dict(self.per_model_pass_at_k),
        }

    def to_panel_hardness(self) -> PanelHardness:
        """Map decision into schema PanelHardness for certified keep records."""
        pm = self.per_model_pass_at_k
        # Canonical OpenRouter ids used by the mission DeepAgent panel.
        grok = pm.get("x-ai/grok-4.5")
        kimi = pm.get("moonshotai/kimi-k2.6")
        opus = pm.get("anthropic/claude-opus-4.8")
        # Also accept short aliases if runners use them as keys.
        if grok is None:
            for key, val in pm.items():
                if "grok" in key:
                    grok = val
                    break
        if kimi is None:
            for key, val in pm.items():
                if "kimi" in key or "moonshot" in key:
                    kimi = val
                    break
        if opus is None:
            for key, val in pm.items():
                if "opus" in key or "claude" in key:
                    opus = val
                    break
        return PanelHardness(
            grok_4_5=grok,
            kimi_k2_6=kimi,
            opus_4_8=opus,
            pass_at_k=self.frontier_pass_at_k,
            discrimination=self.discrimination,
        )


def compute_pass_at_k(solves: int, k: int) -> float:
    """Baseline pass@k = solves/k clamped to [0, 1]."""
    if k <= 0:
        raise BandFilterError(f"k must be positive; got {k}")
    if solves < 0:
        raise BandFilterError(f"solves must be >= 0; got {solves}")
    return min(1.0, max(0.0, solves / k))


def compute_discrimination(per_model_pass: dict[str, float]) -> float:
    """Simple two-model discrimination: scale of max-min pass@k.

    Floor of 1.0 requires a full swing (0 vs 1) between any two panel models.
    With two frontier models partially separating, use::

        discrimination = (max_p - min_p) / max(min_p, 1e-9)   when min>0
        else max_p / 0.01  when only one solves nontrivialy
        else 0.0 when flat zero

    Mission default floor is 1.0. When both rates are equal and in (0,1],
    discrimination is 0 (no tier separation). When one is 0 and the other
    is p, discrimination = p / 0.25 so that p >= 0.25 yields >= 1.0.
    """
    if not per_model_pass:
        return 0.0
    values = list(per_model_pass.values())
    hi = max(values)
    lo = min(values)
    if hi <= 0.0:
        return 0.0
    if hi == lo:
        # Flat nonzero panel: no discrimination signal.
        return 0.0
    spread = hi - lo
    # Normalize so a 0.0 vs 0.25+ separation meets floor 1.0.
    # (0.25 - 0) / 0.25 = 1.0
    return spread / 0.25


def classify_band(
    *,
    per_model_pass_at_k: dict[str, float],
    total_solves: int,
    total_trials: int,
    discrimination: float | None = None,
    config: BandFilterConfig = DEFAULT_BAND_FILTER,
) -> BandDecision:
    """Classify panel outcomes into keep/drop.

    Priority: solve-none → solve-all → out-of-band → low-discrimination → keep.
    Aggregate frontier pass@k is the mean across required panel models' pass@k
    (or sum(solves)/sum(k) when totals provided consistently).
    """
    if total_trials < 0 or total_solves < 0:
        raise BandFilterError("solves/trials must be non-negative")
    if total_solves > total_trials:
        raise BandFilterError(f"total_solves {total_solves} exceeds total_trials {total_trials}")
    if not per_model_pass_at_k:
        raise BandFilterError("per_model_pass_at_k must be non-empty")

    # Aggregate = mean of per-model pass@k (two frontier models are peers).
    frontier = sum(per_model_pass_at_k.values()) / len(per_model_pass_at_k)
    # Prefer exact trial ratio when available (slightly more precise with uneven k).
    if total_trials > 0:
        frontier = total_solves / total_trials

    disc = (
        discrimination
        if discrimination is not None
        else compute_discrimination(per_model_pass_at_k)
    )

    def decide(verdict: str, rule: str, reason: str) -> BandDecision:
        return BandDecision(
            verdict=verdict,
            rule=rule,
            reason=reason,
            frontier_pass_at_k=frontier,
            discrimination=disc,
            band_high=config.band_high,
            discrimination_floor=config.discrimination_floor,
            total_solves=total_solves,
            total_trials=total_trials,
            per_model_pass_at_k=dict(per_model_pass_at_k),
        )

    if frontier <= 0.0 or total_solves == 0:
        return decide(
            "drop",
            RULE_SOLVE_NONE,
            "solve-none: frontier pass@k is 0.0 (no panel model solved any "
            "rollout); the task is broken or impossible",
        )

    if total_trials > 0 and total_solves >= total_trials:
        return decide(
            "drop",
            RULE_SOLVE_ALL,
            f"solve-all (too easy): every panel rollout solved the task "
            f"({total_solves}/{total_trials})",
        )

    if frontier > config.band_high:
        return decide(
            "drop",
            RULE_OUT_OF_BAND,
            f"out-of-band (too easy): frontier pass@k {frontier:.4f} exceeds the "
            f"upper band edge {config.band_high:.4f}",
        )

    if disc < config.discrimination_floor:
        return decide(
            "drop",
            RULE_LOW_DISCRIMINATION,
            f"low discrimination: discrimination {disc:.4f} is below the keep "
            f"threshold {config.discrimination_floor:.4f}",
        )

    return decide(
        "keep",
        RULE_KEEP,
        f"keep: frontier pass@k {frontier:.4f} is in the low-but-nonzero band "
        f"(0, {config.band_high:.4f}], discrimination {disc:.4f} >= "
        f"{config.discrimination_floor:.4f}, and solves are nonzero",
    )


def apply_band_filter(
    *,
    per_model_pass_at_k: dict[str, float],
    total_solves: int,
    total_trials: int,
    discrimination: float | None = None,
    config: BandFilterConfig = DEFAULT_BAND_FILTER,
) -> BandDecision:
    """Alias of classify_band for symmetry with Agent-SWE filter API."""
    return classify_band(
        per_model_pass_at_k=per_model_pass_at_k,
        total_solves=total_solves,
        total_trials=total_trials,
        discrimination=discrimination,
        config=config,
    )


def decision_from_model_stats(
    model_stats: dict[str, tuple[int, int]],
    *,
    config: BandFilterConfig = DEFAULT_BAND_FILTER,
) -> BandDecision:
    """Build decision from ``{model: (solves, k)}`` maps."""
    per_model: dict[str, float] = {}
    total_solves = 0
    total_trials = 0
    for model, (solves, k) in model_stats.items():
        per_model[model] = compute_pass_at_k(solves, k)
        total_solves += solves
        total_trials += k
    return classify_band(
        per_model_pass_at_k=per_model,
        total_solves=total_solves,
        total_trials=total_trials,
        config=config,
    )


def hardness_dict_from_decision(decision: BandDecision) -> dict[str, Any]:
    """JSON-ready panel hardness fields for task records / reports."""
    h = decision.to_panel_hardness()
    return {
        "grok_4_5": h.grok_4_5,
        "kimi_k2_6": h.kimi_k2_6,
        "opus_4_8": h.opus_4_8,
        "pass_at_k": h.pass_at_k,
        "discrimination": h.discrimination,
        "band_verdict": decision.verdict,
        "band_rule": decision.rule,
        "band_reason": decision.reason,
    }


__all__ = [
    "DEFAULT_BAND_FILTER",
    "DEFAULT_BAND_HIGH",
    "DEFAULT_DISCRIMINATION_FLOOR",
    "RULE_KEEP",
    "RULE_LOW_DISCRIMINATION",
    "RULE_OUT_OF_BAND",
    "RULE_SOLVE_ALL",
    "RULE_SOLVE_NONE",
    "BandDecision",
    "BandFilterConfig",
    "BandFilterError",
    "apply_band_filter",
    "classify_band",
    "compute_discrimination",
    "compute_pass_at_k",
    "decision_from_model_stats",
    "hardness_dict_from_decision",
]
