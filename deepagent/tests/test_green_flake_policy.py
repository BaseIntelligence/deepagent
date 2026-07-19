"""Green-base flake soft-continue policy (VAL-DMED-011).

When green (gold-pass side) fails only on documented non-held-out / P2P-class
flake nodes (e.g. RequestRedirect body param), dual-run may soft-continue iff:
- failures match the allowlist
- F2P cohort still forms >= MIN_F2P
- reason_code stamped
- never ignore held-out F2P failures / potential F2P nodes
"""

from __future__ import annotations

import pytest

from swe_factory.pipeline.hardness_floors import DEFAULT_MIN_F2P_NODES
from swe_factory.producers.harbor_labeling import HarborLabelError, SuiteOutcome
from swe_factory.producers.real_dual_run import (
    DEFAULT_GREEN_FLAKE_ALLOWLIST,
    REASON_GREEN_FLAKE_SOFT_CONTINUE,
    GreenFlakeDecision,
    evaluate_green_base_flakes,
    labels_from_real_suite_outcomes,
    match_green_flake_allowlist,
)

REQUEST_REDIRECT_NODE = "tests.test_exceptions.test_response_body[RequestRedirect]"
F2P_A = "tests.mod.test_behavior_a"
F2P_B = "tests.mod.test_behavior_b"
F2P_C = "tests.mod.test_behavior_c"
F2P_D = "tests.mod.test_behavior_d"
F2P_E = "tests.mod.test_behavior_e"
P2P_OK = "tests.mod.test_stable_p2p"


def test_default_allowlist_covers_request_redirect() -> None:
    assert any("RequestRedirect" in x for x in DEFAULT_GREEN_FLAKE_ALLOWLIST)
    matched = match_green_flake_allowlist([REQUEST_REDIRECT_NODE])
    assert REQUEST_REDIRECT_NODE in matched
    assert match_green_flake_allowlist(["tests.mod.test_real_fail"]) == []


def test_evaluate_soft_continue_when_only_allowlisted_and_f2p_ok() -> None:
    decision = evaluate_green_base_flakes(
        green_failed=(REQUEST_REDIRECT_NODE,),
        green_errors=(),
        green_passed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E, P2P_OK),
        broken_failed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E),
        broken_passed=(P2P_OK, REQUEST_REDIRECT_NODE),
        min_f2p=DEFAULT_MIN_F2P_NODES,
    )
    assert isinstance(decision, GreenFlakeDecision)
    assert decision.soft_continue is True
    assert decision.reason_code == REASON_GREEN_FLAKE_SOFT_CONTINUE
    assert REQUEST_REDIRECT_NODE in decision.ignored_nodes
    assert decision.f2p_count >= DEFAULT_MIN_F2P_NODES


def test_evaluate_refuse_when_non_allowlisted_green_fail() -> None:
    decision = evaluate_green_base_flakes(
        green_failed=(REQUEST_REDIRECT_NODE, "tests.mod.test_real_break"),
        green_errors=(),
        green_passed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E, P2P_OK),
        broken_failed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E, "tests.mod.test_real_break"),
        broken_passed=(P2P_OK,),
        min_f2p=DEFAULT_MIN_F2P_NODES,
    )
    assert decision.soft_continue is False
    assert decision.reason_code != REASON_GREEN_FLAKE_SOFT_CONTINUE


def test_evaluate_refuse_when_f2p_below_min_after_flake() -> None:
    """Addiction: soft-continue never hides empty/thin F2P (abuse on F2P floor)."""
    decision = evaluate_green_base_flakes(
        green_failed=(REQUEST_REDIRECT_NODE,),
        green_errors=(),
        green_passed=(F2P_A, P2P_OK),  # only 1 F2P would form
        broken_failed=(F2P_A,),
        broken_passed=(P2P_OK, REQUEST_REDIRECT_NODE),
        min_f2p=DEFAULT_MIN_F2P_NODES,
    )
    assert decision.soft_continue is False
    assert decision.f2p_count < DEFAULT_MIN_F2P_NODES


def test_evaluate_refuse_allowlist_abuse_on_potential_f2p() -> None:
    """Held-out / potential F2P nodes that fail@base must never soft-ignore on green fail."""
    # Node fails base (would-be F2P) AND fails green — gold did not fix; if we
    # allowlisted it to pretend clean, F2P would still not include it, but this
    # is still abuse of the policy (masking real dual-run gold failures on the
    # PR cohort). Soft-continue must refuse.
    decision = evaluate_green_base_flakes(
        green_failed=(F2P_A,),  # potential F2P fails gold too
        green_errors=(),
        green_passed=(F2P_B, F2P_C, F2P_D, F2P_E, P2P_OK),
        broken_failed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E),
        broken_passed=(P2P_OK,),
        min_f2p=4,
        allowlist=(F2P_A,),  # attacker tries to allowlist held-out F2P node
    )
    assert decision.soft_continue is False
    assert (
        "f2p" in decision.detail.lower()
        or "held" in decision.detail.lower()
        or "abuse" in decision.detail.lower()
    )


def test_labels_from_outcomes_soft_continue_stamps_reason() -> None:
    green = SuiteOutcome.from_summary(
        language="python",
        passed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E, P2P_OK),
        failed=(REQUEST_REDIRECT_NODE,),
    )
    broken = SuiteOutcome.from_summary(
        language="python",
        passed=(P2P_OK, REQUEST_REDIRECT_NODE),
        failed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E),
    )
    labels = labels_from_real_suite_outcomes(
        green,
        broken,
        require_nonempty_f2p=True,
        require_green_clean=True,
        allow_green_flake=True,
        min_f2p_nodes=DEFAULT_MIN_F2P_NODES,
    )
    assert len(labels.f2p_node_ids) >= DEFAULT_MIN_F2P_NODES
    assert REASON_GREEN_FLAKE_SOFT_CONTINUE in (labels.notes.get("green_flake_reason_code"),) or (
        labels.notes.get("green_flake_soft_continue") is True
    )
    assert labels.notes.get("green_flake_reason_code") == REASON_GREEN_FLAKE_SOFT_CONTINUE


def test_labels_from_outcomes_still_refuse_real_green_fail() -> None:
    green = SuiteOutcome.from_summary(
        language="python",
        passed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E, P2P_OK),
        failed=("tests.mod.test_unexpected",),
    )
    broken = SuiteOutcome.from_summary(
        language="python",
        passed=(P2P_OK,),
        failed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E, "tests.mod.test_unexpected"),
    )
    with pytest.raises(HarborLabelError, match="green suite must be clean|green"):
        labels_from_real_suite_outcomes(
            green,
            broken,
            require_nonempty_f2p=True,
            require_green_clean=True,
            allow_green_flake=True,
            min_f2p_nodes=DEFAULT_MIN_F2P_NODES,
        )


def test_labels_disabled_policy_restores_strict_clean() -> None:
    green = SuiteOutcome.from_summary(
        language="python",
        passed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E, P2P_OK),
        failed=(REQUEST_REDIRECT_NODE,),
    )
    broken = SuiteOutcome.from_summary(
        language="python",
        passed=(P2P_OK, REQUEST_REDIRECT_NODE),
        failed=(F2P_A, F2P_B, F2P_C, F2P_D, F2P_E),
    )
    with pytest.raises(HarborLabelError):
        labels_from_real_suite_outcomes(
            green,
            broken,
            require_nonempty_f2p=True,
            require_green_clean=True,
            allow_green_flake=False,  # policy off
            min_f2p_nodes=DEFAULT_MIN_F2P_NODES,
        )
