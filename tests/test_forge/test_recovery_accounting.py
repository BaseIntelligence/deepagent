"""Durable, fail-closed accounting for recovery LLM calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from swe_forge.forge.recovery_accounting import (
    RecoveryAccountingError,
    RecoveryBudgetLedger,
    reconcile_recovery_call_evidence,
)
from swe_forge.forge.teacher import TeacherClient, TeacherError, Usage


def _response(
    *,
    request_id: str,
    prompt_tokens: int = 3,
    completion_tokens: int = 2,
    cost: float = 0.0125,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=request_id,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=[]),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        _hidden_params={"response_cost": cost},
    )


async def test_recovery_request_reserves_before_every_attempt_and_fsync_settles(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A failed retry is settled, then the retry is separately pre-reserved."""
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    client = TeacherClient(
        base_url="https://example.invalid",
        api_key="not-a-real-secret",
        model="anthropic/test-model",
        num_retries=1,
        recovery_ledger=ledger,
        recovery_stage="oracle.differential",
        recovery_logical_call_id="logical-differential-1",
    )
    attempts = 0

    async def _completion(**_kwargs: object) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("connection reset")
        return _response(request_id="provider-request-2")

    monkeypatch.setattr("swe_forge.forge.teacher.litellm.acompletion", _completion)

    result = await client.complete_text("propose a variant")

    assert result.usage == Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    assert result.cost == pytest.approx(0.0125)
    accounting = result.recovery_accounting
    assert accounting is not None
    assert accounting["logical_call_id"] == "logical-differential-1"
    calls = accounting["physical_calls"]
    assert isinstance(calls, list) and len(calls) == 2
    assert [call["status"] for call in calls] == ["error", "success"]
    assert [call["retry"] for call in calls] == [0, 1]
    assert calls[1]["request_id"] == "provider-request-2"
    assert calls[0]["error_type"] == "TimeoutError"

    events = ledger.events()
    assert [event["event"] for event in events] == [
        "reserve",
        "settle",
        "reserve",
        "settle",
    ]
    assert [event["physical_call_id"] for event in events[::2]] == [
        call["physical_call_id"] for call in calls
    ]
    assert all("not-a-real-secret" not in str(event) for event in events)
    assert ledger.path.read_bytes().endswith(b"\n")


async def test_recovery_overage_is_durably_settled_then_stops_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Provider-reported overages are auditable before the request fails closed."""
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=0.01,
        worst_case_cost_usd=0.01,
    )
    client = TeacherClient(
        base_url="https://example.invalid",
        api_key="not-a-real-secret",
        model="anthropic/test-model",
        num_retries=1,
        recovery_ledger=ledger,
        recovery_stage="oracle.differential",
        recovery_logical_call_id="logical-overage",
    )
    attempts = 0

    async def _completion(**_kwargs: object) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        return _response(
            request_id="provider-overage",
            cost=0.02,
            finish_reason="prompt contents must never be persisted",
        )

    monkeypatch.setattr("swe_forge.forge.teacher.litellm.acompletion", _completion)

    with pytest.raises(
        RecoveryAccountingError, match="exceeded its pre-reserved worst-case cost"
    ):
        await client.complete_text("propose a variant")

    assert attempts == 1
    assert [event["event"] for event in ledger.events()] == ["reserve", "settle"]
    settled = ledger.settled_calls()
    assert len(settled) == 1
    assert settled[0]["cost"] == "0.02"
    assert settled[0]["finish_reason"] == "other"
    assert "prompt contents" not in ledger.path.read_text(encoding="utf-8")
    assert client.last_recovery_accounting is not None
    with pytest.raises(
        RecoveryAccountingError, match="exceeds its configured budget cap"
    ):
        reconcile_recovery_call_evidence(ledger, [client.last_recovery_accounting])


async def test_recovery_rejects_unsafe_provider_request_ids_without_persisting_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    client = TeacherClient(
        base_url="https://example.invalid",
        api_key="not-a-real-secret",
        model="anthropic/test-model",
        num_retries=0,
        recovery_ledger=ledger,
        recovery_stage="oracle.differential",
    )
    unsafe_id = "provider response contained raw prompt content"

    async def _completion(**_kwargs: object) -> SimpleNamespace:
        return _response(request_id=unsafe_id)

    monkeypatch.setattr("swe_forge.forge.teacher.litellm.acompletion", _completion)

    with pytest.raises(TeacherError, match="safe provider request id"):
        await client.complete_text("propose a variant")

    settled = ledger.settled_calls()
    assert len(settled) == 1
    assert settled[0]["status"] == "error"
    assert settled[0]["request_id"] == ""
    assert settled[0]["error_type"] == "UnsafeRequestId"
    assert unsafe_id not in ledger.path.read_text(encoding="utf-8")


def test_reconciliation_rejects_missing_duplicate_unsettled_and_mismatched_calls(
    tmp_path,
) -> None:
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    first = ledger.reserve(
        logical_call_id="logical-1",
        stage="oracle.alt_correct",
        model="anthropic/test-model",
        retry=0,
    )
    ledger.settle(
        first,
        request_id="provider-1",
        usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        cost=0.01,
        status="success",
        finish_reason="stop",
    )
    unsettled = ledger.reserve(
        logical_call_id="logical-2",
        stage="calibration.rollout",
        model="anthropic/test-model",
        retry=0,
    )

    first_record = ledger.settled_calls()[0]
    valid = {
        "logical_call_id": "logical-1",
        "physical_calls": [first_record],
    }
    with pytest.raises(RecoveryAccountingError, match="unsettled"):
        reconcile_recovery_call_evidence(ledger, [valid])

    ledger.settle(
        unsettled,
        request_id="provider-2",
        usage=Usage(),
        cost=0.0,
        status="error",
        error_type="RuntimeError",
    )
    second_record = ledger.settled_calls()[1]
    duplicate = {
        "logical_call_id": "logical-duplicate",
        "physical_calls": [first_record],
    }
    with pytest.raises(RecoveryAccountingError, match="duplicate"):
        reconcile_recovery_call_evidence(ledger, [valid, duplicate])

    with pytest.raises(RecoveryAccountingError, match="missing"):
        reconcile_recovery_call_evidence(ledger, [valid])

    complete = [
        valid,
        {
            "logical_call_id": "logical-2",
            "physical_calls": [second_record],
        },
    ]
    reconcile_recovery_call_evidence(ledger, complete)

    mismatched = [
        dict(valid, physical_calls=[dict(valid["physical_calls"][0], cost=0.02)]),
        complete[1],
    ]
    with pytest.raises(RecoveryAccountingError, match="mismatched"):
        reconcile_recovery_call_evidence(ledger, mismatched)
