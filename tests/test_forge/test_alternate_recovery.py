"""Contract tests for the one-shot final alternate recovery controller."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_forge.forge.alternate_recovery import (
    ALTERNATE_RECOVERY_TASK_ID,
    UPDATE_WRAPPER_F2P_COMMAND,
    UPDATE_WRAPPER_F2P_CONTENT,
    UPDATE_WRAPPER_F2P_PATH,
    AlternateRecoveryError,
    RecoveryCertification,
    verify_original_budget,
)
from swe_forge.forge.models import OracleTestFile


def test_update_wrapper_hidden_test_is_upstream_grounded_and_named() -> None:
    """The recovered suite exercises the public wraps behavior, not implementation."""
    assert UPDATE_WRAPPER_F2P_PATH == "test_update_wrapper_wraps_basic.py"
    assert UPDATE_WRAPPER_F2P_COMMAND == (
        "python -m pytest "
        "test_update_wrapper_wraps_basic.py::"
        "test_wraps_basic_regular_function_preserves_metadata_and_wrapped"
    )
    assert "from boltons.funcutils import wraps" in UPDATE_WRAPPER_F2P_CONTENT
    assert "@wraps(source)" in UPDATE_WRAPPER_F2P_CONTENT
    assert "wrapped(3)" in UPDATE_WRAPPER_F2P_CONTENT
    assert "wrapped.__name__ == source.__name__" in UPDATE_WRAPPER_F2P_CONTENT
    assert "wrapped.__doc__ == source.__doc__" in UPDATE_WRAPPER_F2P_CONTENT
    assert "wrapped.__wrapped__ is source" in UPDATE_WRAPPER_F2P_CONTENT


def test_update_wrapper_hidden_test_is_added_without_changing_existing_suite() -> None:
    existing = [
        OracleTestFile(
            path="test_complement_bug.py",
            content="def test_complement():\n    assert True\n",
            origin="provided",
        )
    ]

    recovered = RecoveryCertification.freeze_suite(existing)

    assert [test.path for test in recovered] == [
        "test_complement_bug.py",
        UPDATE_WRAPPER_F2P_PATH,
    ]
    assert recovered[0] is existing[0]
    assert recovered[1].content == UPDATE_WRAPPER_F2P_CONTENT
    assert recovered[1].origin == "provided"


def test_update_wrapper_hidden_test_refuses_conflicting_path() -> None:
    conflicting = [
        OracleTestFile(
            path=UPDATE_WRAPPER_F2P_PATH,
            content="def test_different():\n    assert True\n",
            origin="provided",
        )
    ]

    with pytest.raises(AlternateRecoveryError, match="different content"):
        RecoveryCertification.freeze_suite(conflicting)


def test_original_budget_verification_is_exact_and_caps_incremental_attempt(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "harvest_progress.json"
    progress.write_text(
        json.dumps(
            {
                "budget_usd": 1400.0,
                "spend_usd": 1301.8979,
                "reserved_usd": 0.0,
                "status": "budget_exhausted",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    verified = verify_original_budget(progress)

    assert verified.original_budget_usd == "1400.0"
    assert verified.spent_usd == "1301.8979"
    assert verified.remaining_usd == "98.1021"
    assert verified.incremental_cap_usd == "25"


@pytest.mark.parametrize(
    "payload",
    [
        {"budget_usd": 1400.0, "spend_usd": 1400.01, "reserved_usd": 0.0},
        {"budget_usd": 1300.0, "spend_usd": 1.0, "reserved_usd": 0.0},
        {"budget_usd": 1400.0, "spend_usd": 1.0, "reserved_usd": -0.01},
        {"budget_usd": 1400.0, "spend_usd": "not-money", "reserved_usd": 0.0},
    ],
)
def test_original_budget_verification_rejects_non_authoritative_progress(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    progress = tmp_path / "harvest_progress.json"
    progress.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(AlternateRecoveryError):
        verify_original_budget(progress)


def test_certification_payload_never_marks_stale_task_as_passed() -> None:
    pending = RecoveryCertification.pending(
        run_id="alternate-run",
        previous_generation_id="stale-generation",
        task_id=ALTERNATE_RECOVERY_TASK_ID,
    ).to_dict()
    tombstone = RecoveryCertification.tombstone(
        run_id="alternate-run",
        reason="oracle rejected",
    ).to_dict()

    assert pending["state"] == "pending"
    assert pending["passed"] is False
    assert pending["task_ids"] == [ALTERNATE_RECOVERY_TASK_ID]
    assert tombstone["state"] == "tombstone"
    assert tombstone["passed"] is False
    assert tombstone["task_ids"] == []
