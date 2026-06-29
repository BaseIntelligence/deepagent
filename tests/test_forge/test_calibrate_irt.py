"""Unit tests for pass@k + the 2-parameter IRT fit (m5-passk-irt).

Pure-offline coverage (no Docker, no live LLM) of this feature's contract
assertions:

- VAL-CAL-011: pass@k math is a correct nondecreasing function of solves at a
  fixed k; ``solves == 0 -> 0`` and ``solves >= 1 -> > 0``.
- VAL-CAL-010: a gold-solvable but panel-failing task (frontier solves a small
  fraction of k) yields a low frontier pass@k.
- VAL-CAL-012: ``irt_difficulty``/``irt_discrimination`` are numeric and
  matrix-derived (they change when the solve matrix changes materially).
- VAL-CAL-013: high tier-separation -> high discrimination, a flat matrix ->
  low discrimination; kept-style matrices satisfy weak <= mid <= frontier with
  weak pass@k ~ 0 and frontier pass@k > 0.

The live panel + real Docker scoring that *produces* the matrix is exercised by
the runner/solver features and the user-testing validator; here the matrix is a
deterministic fixture.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from swe_forge.forge.calibrate.irt import (
    DEFAULT_TIER_ABILITIES,
    DIFFICULTY_MAX,
    IrtError,
    build_calibration_report,
    compute_pass_at_k,
    fit_irt,
    pass_at_k,
    to_solve_records,
)
from swe_forge.forge.models import (
    BAND_VERDICTS,
    CalibrationReport,
    ModelError,
    ModelSolveRecord,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rec(tier: str, solves: int, k: int, model: str | None = None) -> ModelSolveRecord:
    return ModelSolveRecord(
        model=model or f"m-{tier}",
        tier=tier,
        k=k,
        solves=solves,
        pass_at_k=pass_at_k(solves, k),
    )


def _matrix(weak: int, mid: int, frontier: int, k: int) -> list[ModelSolveRecord]:
    return [
        _rec("weak", weak, k, "openai/gpt-4o-mini"),
        _rec("mid", mid, k, "anthropic/claude-sonnet-4-6"),
        _rec("frontier", frontier, k, "anthropic/claude-opus-4-8"),
    ]


# --------------------------------------------------------------------------- #
# pass@k math (VAL-CAL-011)
# --------------------------------------------------------------------------- #
def test_pass_at_k_zero_solves_is_exactly_zero() -> None:
    assert pass_at_k(0, 5) == 0.0
    assert pass_at_k(0, 1) == 0.0


def test_pass_at_k_any_solve_is_strictly_positive() -> None:
    for solves in range(1, 6):
        assert pass_at_k(solves, 5) > 0.0


def test_pass_at_k_matches_fraction_and_clamps() -> None:
    assert pass_at_k(3, 6) == 0.5
    assert pass_at_k(5, 5) == 1.0
    assert pass_at_k(7, 5) == 1.0  # clamped to [0, 1]


def test_pass_at_k_degenerate_k_is_zero() -> None:
    assert pass_at_k(2, 0) == 0.0
    assert pass_at_k(0, 0) == 0.0
    assert pass_at_k(1, -3) == 0.0


def test_pass_at_k_is_nondecreasing_in_solves_at_fixed_k() -> None:
    k = 8
    values = [pass_at_k(s, k) for s in range(0, k + 1)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[0] == 0.0
    assert values[1] > 0.0


def test_compute_pass_at_k_is_the_same_single_source() -> None:
    # The runner's historical name re-exports the canonical estimator.
    for solves, k in [(0, 5), (1, 5), (3, 6), (7, 5), (2, 0)]:
        assert compute_pass_at_k(solves, k) == pass_at_k(solves, k)


# --------------------------------------------------------------------------- #
# Gold-solvable but panel-failing -> low frontier pass@k (VAL-CAL-010)
# --------------------------------------------------------------------------- #
def test_gold_solvable_but_hard_has_low_frontier_pass_at_k() -> None:
    # Frontier solves 1 of 6 (the gold patch is provably solvable elsewhere),
    # weak/mid solve none: a low-but-nonzero frontier pass@k.
    report = build_calibration_report("python", _matrix(0, 0, 1, 6), k=6)
    assert report.frontier_pass_at_k() == pytest.approx(1 / 6)
    assert 0.0 < report.frontier_pass_at_k() < 0.5
    # weak ~ 0 while frontier > 0
    rates = report.tier_pass_rates()
    assert rates["weak"] == 0.0
    assert rates["frontier"] > 0.0


# --------------------------------------------------------------------------- #
# IRT params numeric + matrix-derived (VAL-CAL-012)
# --------------------------------------------------------------------------- #
def test_irt_params_are_numeric_and_finite() -> None:
    fit = fit_irt(_matrix(0, 1, 3, 5))
    assert isinstance(fit.difficulty, float)
    assert isinstance(fit.discrimination, float)

    assert math.isfinite(fit.difficulty)
    assert math.isfinite(fit.discrimination)


def test_irt_params_change_when_matrix_changes_materially() -> None:
    easy = fit_irt(_matrix(2, 3, 4, 5))  # lots of solves -> low difficulty
    hard = fit_irt(_matrix(0, 0, 1, 5))  # few solves -> high difficulty
    assert hard.difficulty > easy.difficulty
    # the two fits are materially different in BOTH params
    assert hard.difficulty != easy.difficulty
    assert hard.discrimination != easy.discrimination


def test_more_solves_lowers_difficulty() -> None:
    low_solves = fit_irt(_matrix(0, 0, 1, 6)).difficulty
    high_solves = fit_irt(_matrix(0, 2, 5, 6)).difficulty
    assert low_solves > high_solves


# --------------------------------------------------------------------------- #
# Discrimination reflects separation; tier ordering (VAL-CAL-013)
# --------------------------------------------------------------------------- #
def test_high_separation_yields_higher_discrimination_than_flat() -> None:
    separating = fit_irt(_matrix(0, 0, 4, 5))  # only frontier solves
    flat = fit_irt(_matrix(2, 2, 2, 5))  # every tier solves equally
    assert separating.discrimination > flat.discrimination
    assert flat.discrimination == pytest.approx(0.0, abs=1e-6)


def test_perfectly_separating_matrix_has_high_finite_discrimination() -> None:
    fit = fit_irt(_matrix(0, 2, 5, 5))
    assert fit.discrimination > 1.0

    assert math.isfinite(fit.discrimination)


def test_kept_style_matrix_satisfies_tier_ordering() -> None:
    report = build_calibration_report("python", _matrix(0, 1, 3, 6), k=6)
    rates = report.tier_pass_rates()
    assert rates["weak"] <= rates["mid"] <= rates["frontier"]
    assert rates["weak"] == pytest.approx(0.0)
    assert rates["frontier"] > 0.0


def test_no_separation_matrix_has_low_discrimination() -> None:
    fit = fit_irt(_matrix(1, 1, 1, 5))
    assert fit.discrimination == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Degenerate matrices handled without raising (VAL-CAL-015 support)
# --------------------------------------------------------------------------- #
def test_all_fail_matrix_is_very_hard_and_does_not_raise() -> None:
    fit = fit_irt(_matrix(0, 0, 0, 5))
    assert fit.degenerate is True
    assert fit.difficulty == DIFFICULTY_MAX
    assert fit.discrimination == 0.0


def test_all_pass_matrix_is_very_easy_and_does_not_raise() -> None:
    fit = fit_irt(_matrix(5, 5, 5, 5))
    assert fit.degenerate is True
    assert fit.difficulty == -DIFFICULTY_MAX
    assert fit.discrimination == 0.0


def test_solve_none_difficulty_exceeds_solve_some_difficulty() -> None:
    none = fit_irt(_matrix(0, 0, 0, 5)).difficulty
    some = fit_irt(_matrix(0, 0, 1, 5)).difficulty
    assert none > some


# --------------------------------------------------------------------------- #
# fit_irt edge cases
# --------------------------------------------------------------------------- #
def test_fit_irt_empty_matrix_raises() -> None:
    with pytest.raises(IrtError):
        fit_irt([])


def test_fit_irt_all_zero_k_rows_raise() -> None:
    with pytest.raises(IrtError):
        fit_irt([_rec("weak", 0, 0), _rec("frontier", 0, 0)])


def test_fit_irt_unknown_tier_raises() -> None:
    bad = SimpleNamespace(model="x", tier="superhuman", k=5, solves=1)
    with pytest.raises(IrtError):
        fit_irt([bad])


def test_fit_irt_single_tier_no_variation_zero_discrimination() -> None:
    rows = [
        _rec("frontier", 2, 5, "a"),
        _rec("frontier", 3, 5, "b"),
    ]
    fit = fit_irt(rows)
    assert fit.discrimination == 0.0

    assert math.isfinite(fit.difficulty)


def test_fit_irt_respects_custom_tier_abilities() -> None:
    rows = _matrix(0, 0, 4, 5)
    default = fit_irt(rows)
    wider = fit_irt(rows, tier_abilities={"weak": -3.0, "mid": 0.0, "frontier": 3.0})
    # A wider ability spread changes the fitted difficulty scale.
    assert default.difficulty != wider.difficulty


# --------------------------------------------------------------------------- #
# build_calibration_report assembly
# --------------------------------------------------------------------------- #
def test_build_report_populates_fields_and_pending_verdict() -> None:
    report = build_calibration_report(
        "python", _matrix(0, 1, 3, 6), k=6, difficulty_hint="hard"
    )
    assert isinstance(report, CalibrationReport)
    assert report.k == 6
    assert report.difficulty_hint == "hard"
    assert len(report.models) == 3
    assert report.band_verdict == "pending"
    assert report.band_verdict in BAND_VERDICTS
    # IRT fit metadata recorded for provenance.
    assert "irt_fit" in report.details
    assert report.details["tier_abilities"] == DEFAULT_TIER_ABILITIES


def test_build_report_recomputes_canonical_pass_at_k() -> None:
    # Even if an input row carried a bogus pass_at_k, the report uses solves/k.
    rows = _matrix(0, 2, 4, 5)
    report = build_calibration_report("python", rows, k=5)
    by_tier = {r.tier: r for r in report.models}
    assert by_tier["mid"].pass_at_k == pytest.approx(2 / 5)
    assert by_tier["frontier"].pass_at_k == pytest.approx(4 / 5)
    assert by_tier["weak"].pass_at_k == 0.0


def test_report_round_trips_through_dict() -> None:
    report = build_calibration_report("python", _matrix(0, 1, 3, 6), k=6)
    restored = CalibrationReport.from_dict(report.to_dict())
    assert restored.irt_difficulty == pytest.approx(report.irt_difficulty)
    assert restored.irt_discrimination == pytest.approx(report.irt_discrimination)
    assert [m.to_dict() for m in restored.models] == [
        m.to_dict() for m in report.models
    ]


def test_to_solve_records_projects_any_rows() -> None:
    rows = _matrix(0, 1, 3, 6)
    projected = to_solve_records(rows)
    assert all(isinstance(r, ModelSolveRecord) for r in projected)
    assert [r.pass_at_k for r in projected] == [pass_at_k(r.solves, r.k) for r in rows]


# --------------------------------------------------------------------------- #
# ModelSolveRecord pass@k invariant guard
# --------------------------------------------------------------------------- #
def test_model_solve_record_rejects_zero_solves_with_positive_pass_at_k() -> None:
    with pytest.raises(ModelError):
        ModelSolveRecord(model="m", tier="weak", k=5, solves=0, pass_at_k=0.2)


def test_model_solve_record_rejects_positive_solves_with_zero_pass_at_k() -> None:
    with pytest.raises(ModelError):
        ModelSolveRecord(model="m", tier="frontier", k=5, solves=2, pass_at_k=0.0)


def test_model_solve_record_rejects_solves_greater_than_k() -> None:
    with pytest.raises(ModelError):
        ModelSolveRecord(model="m", tier="mid", k=3, solves=4, pass_at_k=1.0)


def test_model_solve_record_rejects_unknown_tier() -> None:
    with pytest.raises(ModelError):
        ModelSolveRecord(model="m", tier="superhuman", k=3, solves=1, pass_at_k=0.33)


# --------------------------------------------------------------------------- #
# CalibrationReport invariants
# --------------------------------------------------------------------------- #
def test_calibration_report_set_band_verdict_records_reason() -> None:
    report = build_calibration_report("python", _matrix(0, 1, 3, 6), k=6)
    report.set_band_verdict("keep", "in-band frontier + high discrimination")
    assert report.is_keep
    assert report.reasons == ["in-band frontier + high discrimination"]


def test_calibration_report_set_band_verdict_requires_reason() -> None:
    report = build_calibration_report("python", _matrix(0, 1, 3, 6), k=6)
    with pytest.raises(ModelError):
        report.set_band_verdict("keep", "")


def test_calibration_report_rejects_bad_band_verdict() -> None:
    with pytest.raises(ModelError):
        CalibrationReport(
            language="python",
            models=[],
            k=0,
            irt_difficulty=0.0,
            irt_discrimination=0.0,
            band_verdict="maybe",
        )


def test_calibration_report_rejects_non_finite_irt() -> None:
    with pytest.raises(ModelError):
        CalibrationReport(
            language="python",
            models=[],
            k=0,
            irt_difficulty=float("inf"),
            irt_discrimination=0.0,
        )
