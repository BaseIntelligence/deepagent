"""Unit tests for the Stage 4 band filter (m5-filter).

Pure-offline coverage (no Docker, no live LLM) of this feature's contract
assertions. The band filter is a deterministic classifier over a
``CalibrationReport`` (which the upstream runner/IRT features produce from the
real panel + Docker scoring); here the report is a deterministic fixture.

- VAL-CAL-014: KEEP only with an in-band frontier pass-rate AND high
  discrimination AND nonzero solves; the recorded reason cites all three.
- VAL-CAL-015: DROP solve-all (too easy) and solve-none (broken/impossible);
  degenerate all-pass / all-fail matrices are handled without raising.
- VAL-CAL-016: DROP an in-band pass-rate with LOW discrimination.
- VAL-CAL-017: ``band_verdict`` + a non-empty, attributable ``reason`` are always
  recorded and map to exactly one applied rule.
- VAL-CAL-018: band edges honored at the boundaries (just-below-edge + disc keep;
  just-above drop too-easy; ==0 drop solve-none).
- VAL-CAL-021: ``keep`` is necessary for export; ``drop`` blocks ForgeTask
  emission (export refuses).
"""

from __future__ import annotations

import pytest

from swe_forge.forge.calibrate.filter import (
    DEFAULT_BAND_FILTER,
    RULE_KEEP,
    RULE_LOW_DISCRIMINATION,
    RULE_OUT_OF_BAND,
    RULE_SOLVE_ALL,
    RULE_SOLVE_NONE,
    BandDecision,
    BandFilterConfig,
    BandFilterError,
    apply_band_filter,
    classify_band,
)
from swe_forge.forge.calibrate.irt import build_calibration_report, pass_at_k
from swe_forge.forge.models import (
    CalibrationReport,
    FinalMutationEvidence,
    ModelError,
    ModelSolveRecord,
    OracleReport,
    OracleTestFile,
)
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.oracle.pipeline import ExportRefusedError, ensure_oracle_exportable


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


def _report(
    weak: int, mid: int, frontier: int, k: int, language: str = "python"
) -> CalibrationReport:
    """Build a real CalibrationReport (IRT fit included) for the given matrix."""
    return build_calibration_report(language, _matrix(weak, mid, frontier, k), k=k)


# --------------------------------------------------------------------------- #
# KEEP (VAL-CAL-014)
# --------------------------------------------------------------------------- #
def test_keep_in_band_high_disc_nonzero_solves() -> None:
    # weak fails, frontier solves a few of k: in-band (0.5) + separating slope.
    report = _report(0, 1, 3, 6)
    assert report.frontier_pass_at_k() <= DEFAULT_BAND_FILTER.band_high
    assert report.irt_discrimination >= DEFAULT_BAND_FILTER.discrimination_threshold

    decision = classify_band(report)
    assert decision.verdict == "keep"
    assert decision.rule == RULE_KEEP
    # The reason cites all three keep conditions.
    reason = decision.reason.lower()
    assert "band" in reason
    assert "discrimination" in reason
    assert "solve" in reason


def test_keep_frontier_only_separating_in_band() -> None:
    # Only frontier solves (steep slope), and its pass@k stays inside the band.
    report = _report(0, 0, 2, 6)
    decision = classify_band(report)
    assert decision.verdict == "keep"
    assert decision.rule == RULE_KEEP


def test_apply_band_filter_sets_keep_verdict_and_records_provenance() -> None:
    report = _report(0, 1, 3, 6)
    apply_band_filter(report)
    assert report.is_keep
    assert report.band_verdict == "keep"
    assert report.reasons and report.reasons[0].strip()
    # Decision recorded under details for attribution/provenance.
    band = report.details["band_filter"]
    assert isinstance(band, dict)
    assert band["verdict"] == "keep"
    assert band["rule"] == RULE_KEEP


# --------------------------------------------------------------------------- #
# DROP solve-all / solve-none + degenerate matrices (VAL-CAL-015)
# --------------------------------------------------------------------------- #
def test_drop_solve_all_degenerate_all_pass() -> None:
    report = _report(6, 6, 6, 6)  # every rollout of every model solves
    # Degenerate all-pass matrix must not raise during the IRT fit.
    assert report.irt_discrimination == 0.0
    decision = classify_band(report)
    assert decision.verdict == "drop"
    assert decision.rule == RULE_SOLVE_ALL
    assert "too easy" in decision.reason.lower()


def test_drop_solve_none_degenerate_all_fail() -> None:
    report = _report(0, 0, 0, 6)  # no model solves any rollout
    assert report.frontier_pass_at_k() == 0.0
    decision = classify_band(report)
    assert decision.verdict == "drop"
    assert decision.rule == RULE_SOLVE_NONE


def test_apply_band_filter_drops_solve_all_and_solve_none() -> None:
    for report in (_report(6, 6, 6, 6), _report(0, 0, 0, 6)):
        apply_band_filter(report)
        assert report.band_verdict == "drop"
        assert not report.is_keep
        assert report.reasons and report.reasons[0].strip()


def test_degenerate_matrices_do_not_raise() -> None:
    # All-pass and all-fail (and a single-tier) must classify without raising.
    for report in (_report(6, 6, 6, 6), _report(0, 0, 0, 6)):
        decision = classify_band(report)
        assert isinstance(decision, BandDecision)
        assert decision.verdict == "drop"


# --------------------------------------------------------------------------- #
# DROP too-easy / out-of-band (VAL-CAL-015 / VAL-CAL-018)
# --------------------------------------------------------------------------- #
def test_drop_out_of_band_too_easy() -> None:
    # frontier solves most of k (0.83) -> above the upper band edge, drop.
    report = _report(1, 3, 5, 6)
    assert report.frontier_pass_at_k() > DEFAULT_BAND_FILTER.band_high
    decision = classify_band(report)
    assert decision.verdict == "drop"
    assert decision.rule == RULE_OUT_OF_BAND
    assert "too easy" in decision.reason.lower()


def test_drop_frontier_solves_all_even_with_high_disc() -> None:
    # frontier 1.0 but weak/mid 0: high discrimination, yet too easy for frontier.
    report = _report(0, 0, 6, 6)
    assert report.frontier_pass_at_k() == 1.0
    decision = classify_band(report)
    assert decision.verdict == "drop"
    # not all-pass (weak/mid fail) -> out-of-band rather than solve-all.
    assert decision.rule == RULE_OUT_OF_BAND


# --------------------------------------------------------------------------- #
# DROP in-band but LOW discrimination (VAL-CAL-016)
# --------------------------------------------------------------------------- #
def test_drop_in_band_low_discrimination_flat() -> None:
    # Every tier solves 2/6: frontier pass@k 0.33 (in band) but a flat slope.
    report = _report(2, 2, 2, 6)
    assert 0.0 < report.frontier_pass_at_k() <= DEFAULT_BAND_FILTER.band_high
    assert report.irt_discrimination < DEFAULT_BAND_FILTER.discrimination_threshold
    decision = classify_band(report)
    assert decision.verdict == "drop"
    assert decision.rule == RULE_LOW_DISCRIMINATION
    assert "discrimination" in decision.reason.lower()


def test_drop_in_band_low_discrimination_weak_separation() -> None:
    # weak 1, mid 2, frontier 2 of 6: in band, only a shallow slope -> drop.
    report = _report(1, 2, 2, 6)
    assert 0.0 < report.frontier_pass_at_k() <= DEFAULT_BAND_FILTER.band_high
    assert report.irt_discrimination < DEFAULT_BAND_FILTER.discrimination_threshold
    decision = classify_band(report)
    assert decision.verdict == "drop"
    assert decision.rule == RULE_LOW_DISCRIMINATION


# --------------------------------------------------------------------------- #
# band_verdict + reason always recorded / attributable (VAL-CAL-017)
# --------------------------------------------------------------------------- #
def test_every_decision_has_nonempty_attributable_reason() -> None:
    matrices = [
        (0, 1, 3, 6),  # keep
        (6, 6, 6, 6),  # solve-all
        (0, 0, 0, 6),  # solve-none
        (2, 2, 2, 6),  # low-disc
        (1, 3, 5, 6),  # out-of-band
    ]
    seen_rules = set()
    valid_rules = {
        RULE_KEEP,
        RULE_SOLVE_ALL,
        RULE_SOLVE_NONE,
        RULE_LOW_DISCRIMINATION,
        RULE_OUT_OF_BAND,
    }
    for w, m, f, k in matrices:
        report = _report(w, m, f, k)
        decision = classify_band(report)
        assert decision.verdict in ("keep", "drop")
        assert decision.reason and decision.reason.strip()
        assert decision.rule in valid_rules
        seen_rules.add(decision.rule)
    # Each of the five distinct rules was exercised by exactly one matrix.
    assert seen_rules == valid_rules


def test_decision_to_dict_is_serializable_and_complete() -> None:
    decision = classify_band(_report(0, 1, 3, 6))
    payload = decision.to_dict()
    for key in (
        "verdict",
        "rule",
        "reason",
        "frontier_pass_at_k",
        "discrimination",
        "band_high",
        "discrimination_threshold",
    ):
        assert key in payload


# --------------------------------------------------------------------------- #
# Boundary / band-edge handling (VAL-CAL-018)
# --------------------------------------------------------------------------- #
def test_boundary_just_below_upper_edge_with_disc_keeps() -> None:
    # band_high configured at 0.55; frontier 5/10 = 0.50 is just inside the band.
    config = BandFilterConfig(band_high=0.55, discrimination_threshold=1.0)
    report = build_calibration_report("python", _matrix(0, 2, 5, 10), k=10)
    assert report.frontier_pass_at_k() == pytest.approx(0.5)
    assert report.irt_discrimination >= config.discrimination_threshold
    decision = classify_band(report, config=config)
    assert decision.verdict == "keep"
    assert decision.rule == RULE_KEEP


def test_boundary_at_upper_edge_inclusive_keeps() -> None:
    # frontier exactly at the upper edge is in-band (<= edge keeps).
    config = BandFilterConfig(band_high=0.5, discrimination_threshold=1.0)
    report = build_calibration_report("python", _matrix(0, 2, 5, 10), k=10)
    assert report.frontier_pass_at_k() == pytest.approx(config.band_high)
    decision = classify_band(report, config=config)
    assert decision.verdict == "keep"


def test_boundary_just_above_upper_edge_drops_too_easy() -> None:
    # frontier 6/10 = 0.60 just above the 0.55 edge -> drop (too easy).
    config = BandFilterConfig(band_high=0.55, discrimination_threshold=1.0)
    report = build_calibration_report("python", _matrix(0, 2, 6, 10), k=10)
    assert report.frontier_pass_at_k() > config.band_high
    decision = classify_band(report, config=config)
    assert decision.verdict == "drop"
    assert decision.rule == RULE_OUT_OF_BAND


def test_boundary_exactly_zero_drops_solve_none() -> None:
    config = BandFilterConfig(band_high=0.55, discrimination_threshold=1.0)
    report = build_calibration_report("python", _matrix(0, 0, 0, 10), k=10)
    assert report.frontier_pass_at_k() == 0.0
    decision = classify_band(report, config=config)
    assert decision.verdict == "drop"
    assert decision.rule == RULE_SOLVE_NONE


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_config_rejects_out_of_range_band_high() -> None:
    with pytest.raises(BandFilterError):
        BandFilterConfig(band_high=0.0)
    with pytest.raises(BandFilterError):
        BandFilterConfig(band_high=1.5)


def test_config_rejects_negative_discrimination_threshold() -> None:
    with pytest.raises(BandFilterError):
        BandFilterConfig(discrimination_threshold=-0.1)


def test_custom_thresholds_change_verdict() -> None:
    # A flat in-band task is dropped under the default threshold but a threshold
    # of 0.0 (accept any discrimination) keeps it -- proving the knob matters.
    report = _report(2, 2, 2, 6)
    assert classify_band(report).verdict == "drop"
    lax = BandFilterConfig(band_high=0.5, discrimination_threshold=0.0)
    assert classify_band(report, config=lax).verdict == "keep"


# --------------------------------------------------------------------------- #
# Export wiring: keep necessary, drop blocks export (VAL-CAL-021)
# --------------------------------------------------------------------------- #
def _passing_oracle() -> OracleReport:
    test_files = [OracleTestFile(path="tests/test_x.py", content="def test_a(): ...")]

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
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=["tests/test_x.py::test_a"],
        pass_to_pass=["tests/test_x.py::test_b"],
        test_files=test_files,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        final_mutation_evidence=FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(test_files),
            mutants_total=10,
            mutants_killed=10,
            threshold=0.8,
            tool="fake-tool",
        ),
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        details={
            "teacher_gates": {
                "differential": {"calls": [call("differential")]},
                "alt_correct": {"calls": [call("alt_correct")]},
            }
        },
    )


def test_keep_is_exportable_with_oracle_pass() -> None:
    report = _report(0, 1, 3, 6)
    apply_band_filter(report)
    assert report.is_keep
    # Oracle pass + calibration keep -> export is permitted (no raise).
    ensure_oracle_exportable(_passing_oracle(), calibration_kept=report.is_keep)


def test_drop_blocks_export_even_with_oracle_pass() -> None:
    report = _report(6, 6, 6, 6)  # solve-all -> drop
    apply_band_filter(report)
    assert not report.is_keep
    with pytest.raises(ExportRefusedError):
        ensure_oracle_exportable(_passing_oracle(), calibration_kept=report.is_keep)


def test_solve_none_drop_blocks_export() -> None:
    report = _report(0, 0, 0, 6)  # solve-none -> drop
    apply_band_filter(report)
    assert not report.is_keep
    with pytest.raises(ExportRefusedError):
        ensure_oracle_exportable(_passing_oracle(), calibration_kept=report.is_keep)


# --------------------------------------------------------------------------- #
# Reproducibility (VAL-CAL-020 support)
# --------------------------------------------------------------------------- #
def test_classify_is_deterministic_for_equivalent_reports() -> None:
    a = classify_band(_report(0, 1, 3, 6))
    b = classify_band(_report(0, 1, 3, 6))
    assert a.to_dict() == b.to_dict()


def test_apply_band_filter_is_idempotent() -> None:
    report = _report(0, 1, 3, 6)
    apply_band_filter(report)
    first = (report.band_verdict, tuple(report.reasons))
    apply_band_filter(report)
    second = (report.band_verdict, tuple(report.reasons))
    assert first == second


def test_set_band_verdict_guard_still_requires_reason() -> None:
    # The data-model guard (reused by the filter) rejects an empty reason.
    report = _report(0, 1, 3, 6)
    with pytest.raises(ModelError):
        report.set_band_verdict("keep", "")
