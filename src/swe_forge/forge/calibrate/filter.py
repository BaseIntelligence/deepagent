"""Stage 4 band filter: the terminal keep/drop verdict over a calibration run.

The IRT fit (:mod:`swe_forge.forge.calibrate.irt`) turns the panel solve matrix
into two numbers -- ``frontier_pass_at_k`` (how often the strongest model solves
the task) and ``irt_discrimination`` (how sharply solve-rate rises from the weak
to the frontier tier). This module applies the *band filter*, the second of the
two mission guarantees ("hard for LLMs"): a task is shippable only if the frontier
finds it hard-but-not-impossible AND the panel separates strong models from weak.

The decision is a deterministic classifier over a :class:`CalibrationReport` (the
teacher proposes; deterministic counting disposes -- nothing here calls a model).
It applies exactly one of five mutually-exclusive rules, in priority order, and
records an attributable reason:

1. **solve-none** -- the frontier solved nothing (``frontier_pass_at_k == 0``):
   the task is broken/impossible. Degenerate all-fail matrices land here with the
   IRT sentinel difficulty (very hard) and zero discrimination, never raising.
2. **solve-all** -- every rollout of every panel model solved the task: trivially
   easy (the degenerate all-pass matrix, IRT sentinel difficulty very easy).
3. **out-of-band** -- the frontier pass-rate is above the upper band edge: too
   easy for the frontier even if some weaker models fail.
4. **low-discrimination** -- the pass-rate is in the low-but-nonzero band but the
   tiers do not separate (``irt_discrimination`` below the keep threshold): no
   useful difficulty signal.
5. **keep** -- in-band frontier pass-rate AND high discrimination AND nonzero
   solves: the only verdict that permits export.

A ``ForgeTask`` may only be assembled from a ``keep`` report (architecture S3
invariant); ``drop`` blocks export. The export boundary enforces this via
:func:`swe_forge.forge.oracle.pipeline.ensure_oracle_exportable` with the band
verdict; this module assigns that verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from swe_forge.forge.models import CalibrationReport

#: Rule labels -- every :class:`BandDecision` maps to exactly one (VAL-CAL-017).
RULE_KEEP = "in-band-high-discrimination"
RULE_SOLVE_NONE = "solve-none"
RULE_SOLVE_ALL = "solve-all"
RULE_OUT_OF_BAND = "out-of-band"
RULE_LOW_DISCRIMINATION = "low-discrimination"

#: Upper band edge: a frontier ``pass_at_k`` strictly above this is "too easy".
#: At-or-below the edge is in-band (the edge is inclusive on the keep side).
DEFAULT_BAND_HIGH = 0.5

#: Keep requires ``irt_discrimination >= DEFAULT_DISCRIMINATION_THRESHOLD``. The
#: fitted slope is ~0 for a flat (non-separating) matrix and >=~1.5 when only the
#: strong tiers solve, so this cleanly separates "tiers separate" from "flat".
DEFAULT_DISCRIMINATION_THRESHOLD = 1.0


class BandFilterError(ValueError):
    """Raised when the band filter is configured with invalid thresholds."""


@dataclass(frozen=True)
class BandFilterConfig:
    """Configurable band-filter thresholds.

    ``band_high`` is the upper edge of the low-but-nonzero target band: a frontier
    ``pass_at_k`` in ``(0, band_high]`` is in-band, and anything above it is too
    easy. ``discrimination_threshold`` is the minimum ``irt_discrimination`` a kept
    task must reach (so weak and strong tiers demonstrably separate).
    """

    band_high: float = DEFAULT_BAND_HIGH
    discrimination_threshold: float = DEFAULT_DISCRIMINATION_THRESHOLD

    def __post_init__(self) -> None:
        if not (0.0 < self.band_high <= 1.0):
            raise BandFilterError(f"band_high must be in (0, 1]; got {self.band_high}")
        if self.discrimination_threshold < 0.0:
            raise BandFilterError(
                "discrimination_threshold must be >= 0; "
                f"got {self.discrimination_threshold}"
            )


#: Default band filter (low-but-nonzero band, high-discrimination keep threshold).
DEFAULT_BAND_FILTER = BandFilterConfig()


@dataclass(frozen=True)
class BandDecision:
    """The terminal keep/drop classification of a :class:`CalibrationReport`.

    ``verdict`` is ``"keep"`` or ``"drop"``; ``rule`` is the single applied rule
    label (one of the ``RULE_*`` constants); ``reason`` is the non-empty
    human-readable justification recorded on the report. The remaining fields
    snapshot the inputs the decision turned on, for provenance/attribution.
    """

    verdict: str
    rule: str
    reason: str
    frontier_pass_at_k: float
    discrimination: float
    band_high: float
    discrimination_threshold: float

    @property
    def is_keep(self) -> bool:
        """``True`` iff the decision keeps the task."""
        return self.verdict == "keep"

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "rule": self.rule,
            "reason": self.reason,
            "frontier_pass_at_k": self.frontier_pass_at_k,
            "discrimination": self.discrimination,
            "band_high": self.band_high,
            "discrimination_threshold": self.discrimination_threshold,
        }


def classify_band(
    report: CalibrationReport,
    *,
    config: BandFilterConfig = DEFAULT_BAND_FILTER,
) -> BandDecision:
    """Classify a calibration report into a keep/drop :class:`BandDecision`.

    Applies the five mutually-exclusive rules in priority order (solve-none ->
    solve-all -> out-of-band -> low-discrimination -> keep) over the report's
    frontier pass-rate, panel solve totals, and fitted discrimination. The report
    is read-only here; use :func:`apply_band_filter` to write the verdict back.
    """
    frontier = report.frontier_pass_at_k()
    discrimination = report.irt_discrimination
    total_trials = sum(record.k for record in report.models)
    total_solves = sum(record.solves for record in report.models)

    def decide(verdict: str, rule: str, reason: str) -> BandDecision:
        return BandDecision(
            verdict=verdict,
            rule=rule,
            reason=reason,
            frontier_pass_at_k=frontier,
            discrimination=discrimination,
            band_high=config.band_high,
            discrimination_threshold=config.discrimination_threshold,
        )

    if frontier <= 0.0:
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

    if discrimination < config.discrimination_threshold:
        return decide(
            "drop",
            RULE_LOW_DISCRIMINATION,
            f"low discrimination: irt_discrimination {discrimination:.4f} is below "
            f"the keep threshold {config.discrimination_threshold:.4f} (weak and "
            f"strong tiers do not separate)",
        )

    return decide(
        "keep",
        RULE_KEEP,
        f"keep: frontier pass@k {frontier:.4f} is in the low-but-nonzero band "
        f"(0, {config.band_high:.4f}], discrimination {discrimination:.4f} >= "
        f"{config.discrimination_threshold:.4f}, and solves are nonzero",
    )


def apply_band_filter(
    report: CalibrationReport,
    *,
    config: BandFilterConfig = DEFAULT_BAND_FILTER,
) -> CalibrationReport:
    """Classify ``report`` and write the terminal band verdict + reason in place.

    Sets ``band_verdict`` (``keep``/``drop``) with the attributable reason via
    :meth:`CalibrationReport.set_band_verdict`, and records the full decision under
    ``details["band_filter"]`` for provenance. Returns the same report so callers
    can chain. Deterministic and idempotent: re-applying yields the same verdict.
    """
    decision = classify_band(report, config=config)
    report.set_band_verdict(decision.verdict, decision.reason)
    report.details["band_filter"] = decision.to_dict()
    return report


__all__ = [
    "DEFAULT_BAND_FILTER",
    "DEFAULT_BAND_HIGH",
    "DEFAULT_DISCRIMINATION_THRESHOLD",
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
]
