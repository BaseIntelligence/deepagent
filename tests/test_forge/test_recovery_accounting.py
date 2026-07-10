"""Durable, fail-closed accounting for recovery LLM calls."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from swe_forge.forge.recovery_accounting import (
    RecoveryAccountingError,
    RecoveryBudgetLedger,
    reconcile_recovery_call_evidence,
    reconcile_recovery_reports,
)
from swe_forge.forge.teacher import (
    TeacherClient,
    TeacherError,
    UnknownBillingError,
    Usage,
)


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


class _ResponseBearingError(RuntimeError):
    """Provider error carrying only structured, exact metering metadata."""

    def __init__(self, response: SimpleNamespace) -> None:
        super().__init__("provider error details must not be persisted")
        self.response = response


def _oracle_call_with_ledger_linkage(
    ledger: RecoveryBudgetLedger,
    *,
    gate: str,
    call_kind: str = "proposal",
    logical_call_id: str,
    attempts: tuple[tuple[str, Usage, str], ...] = (
        (
            "success",
            Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            "0.01",
        ),
    ),
) -> dict[str, object]:
    """Build one real teacher attestation backed by durable physical attempts."""
    stage = f"oracle.{gate}"
    physical_calls: list[dict[str, object]] = []
    total_usage = Usage()
    total_cost = Decimal()
    for retry, (status, usage, cost) in enumerate(attempts):
        physical = ledger.reserve(
            logical_call_id=logical_call_id,
            stage=stage,
            model="anthropic/test-model",
            retry=retry,
        )
        ledger.settle(
            physical,
            request_id=f"provider-{logical_call_id}-{retry}",
            usage=usage,
            cost=cost,
            status=status,  # type: ignore[arg-type]
            finish_reason="stop" if status == "success" else None,
            error_type="" if status == "success" else "RuntimeError",
        )
        physical_calls.append(
            dict(
                next(
                    call
                    for call in ledger.settled_calls()
                    if call["physical_call_id"] == physical
                )
            )
        )
        total_usage = total_usage + usage
        total_cost += Decimal(cost)
    return {
        "gate": gate,
        "call_kind": call_kind,
        "real_teacher": True,
        "status": attempts[-1][0],
        "model": "anthropic/test-model",
        "usage": total_usage.to_dict(),
        "cost": format(total_cost, "f"),
        "recovery_accounting": {
            "logical_call_id": logical_call_id,
            "physical_calls": physical_calls,
        },
    }


def _linked_oracle_report(
    ledger: RecoveryBudgetLedger,
) -> SimpleNamespace:
    """Return differential, strengthening, and alt-correct call attestations."""
    differential = _oracle_call_with_ledger_linkage(
        ledger,
        gate="differential",
        logical_call_id="differential-proposal",
        attempts=(
            (
                "error",
                Usage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
                "0.01",
            ),
            (
                "success",
                Usage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
                "0.02",
            ),
        ),
    )
    strengthening = _oracle_call_with_ledger_linkage(
        ledger,
        gate="differential",
        call_kind="strengthen",
        logical_call_id="differential-strengthen",
    )
    alt_correct = _oracle_call_with_ledger_linkage(
        ledger,
        gate="alt_correct",
        logical_call_id="alt-correct-proposal",
    )
    return SimpleNamespace(
        details={
            "teacher_gates": {
                "differential": {"calls": [differential, strengthening]},
                "alt_correct": {"calls": [alt_correct]},
            }
        }
    )


async def test_recovery_request_reserves_before_every_attempt_and_fsync_settles(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An exactly metered provider error settles before a separately reserved retry."""
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
            raise _ResponseBearingError(
                _response(
                    request_id="provider-request-1",
                    prompt_tokens=7,
                    completion_tokens=4,
                    cost=0.0175,
                )
            )
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
    assert calls[0]["request_id"] == "provider-request-1"
    assert calls[0]["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 4,
        "total_tokens": 11,
    }
    assert calls[0]["cost"] == "0.0175"
    assert calls[1]["request_id"] == "provider-request-2"
    assert calls[0]["error_type"] == "_ResponseBearingError"

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
    assert "provider error details must not be persisted" not in ledger.path.read_text(
        encoding="utf-8"
    )
    assert ledger.path.read_bytes().endswith(b"\n")
    reloaded = RecoveryBudgetLedger(
        ledger.path,
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    assert reloaded.unsettled_call_ids() == ()
    assert [
        (call["status"], call["usage"], call["cost"])
        for call in reloaded.settled_calls()
    ] == [
        (
            "error",
            {
                "prompt_tokens": 7,
                "completion_tokens": 4,
                "total_tokens": 11,
            },
            "0.0175",
        ),
        (
            "success",
            {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
            "0.0125",
        ),
    ]


async def test_recovery_unknown_billing_keeps_reservation_and_forbids_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A post-send failure without exact metering stays reserved and unpublished."""
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=0.20,
        worst_case_cost_usd=0.20,
    )
    client = TeacherClient(
        base_url="https://example.invalid",
        api_key="not-a-real-secret",
        model="anthropic/test-model",
        num_retries=1,
        recovery_ledger=ledger,
        recovery_stage="oracle.differential",
        recovery_logical_call_id="logical-unknown-billing",
    )
    attempts = 0
    failure_text = (
        "provider endpoint https://example.invalid/v1, "
        "prompt=highly-confidential-provider-response"
    )

    async def _completion(**_kwargs: object) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        raise TimeoutError(failure_text)

    monkeypatch.setattr("swe_forge.forge.teacher.litellm.acompletion", _completion)

    with pytest.raises(UnknownBillingError, match="unknown provider billing"):
        await client.complete_text("prompt carrying not-a-real-secret")

    assert attempts == 1
    assert client.last_recovery_accounting is None
    assert ledger.settled_calls() == []
    assert len(ledger.unsettled_call_ids()) == 1
    assert str(ledger.total_exact_cost) == "0"
    assert ledger.total_active_reservations == Decimal("0.20")
    events = ledger.events()
    assert [event["event"] for event in events] == ["reserve", "unknown_billing"]
    assert events[1]["error_type"] == "TimeoutError"
    assert "usage" not in events[1]
    assert "cost_usd" not in events[1]
    persisted = ledger.path.read_text(encoding="utf-8")
    assert failure_text not in persisted
    assert "not-a-real-secret" not in persisted
    assert "https://example.invalid" not in persisted

    reloaded = RecoveryBudgetLedger(
        ledger.path,
        run_id="recovery-run",
        cap_usd=0.20,
        worst_case_cost_usd=0.20,
    )
    assert reloaded.unsettled_call_ids() == ledger.unsettled_call_ids()
    with pytest.raises(RecoveryAccountingError, match="cap exhausted"):
        reloaded.reserve(
            logical_call_id="blocked-retry",
            stage="oracle.differential",
            model="anthropic/test-model",
            retry=1,
        )
    with pytest.raises(RecoveryAccountingError, match="unknown-billing"):
        reconcile_recovery_call_evidence(reloaded, [])


async def test_recovery_cancellation_keeps_unknown_billing_reservation(
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
        recovery_ledger=ledger,
        recovery_stage="oracle.differential",
    )

    async def _completion(**_kwargs: object) -> SimpleNamespace:
        raise asyncio.CancelledError

    monkeypatch.setattr("swe_forge.forge.teacher.litellm.acompletion", _completion)

    with pytest.raises(asyncio.CancelledError):
        await client.complete_text("propose a variant")

    assert ledger.settled_calls() == []
    assert len(ledger.unsettled_call_ids()) == 1
    assert [event["event"] for event in ledger.events()] == [
        "reserve",
        "unknown_billing",
    ]
    assert ledger.events()[1]["error_type"] == "CancelledError"


def test_recovery_ledger_rejects_replayed_settlement_after_unknown_billing(
    tmp_path,
) -> None:
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    physical_call_id = ledger.reserve(
        logical_call_id="logical-unknown-billing",
        stage="oracle.differential",
        model="anthropic/test-model",
        retry=0,
    )
    ledger.mark_unknown_billing(physical_call_id, error_type="TimeoutError")
    forged_settlement = {
        "schema_version": 1,
        "event": "settle",
        "timestamp": "2026-07-10T00:00:00+00:00",
        "run_id": "recovery-run",
        "logical_call_id": "logical-unknown-billing",
        "physical_call_id": physical_call_id,
        "stage": "oracle.differential",
        "model": "anthropic/test-model",
        "retry": 0,
        "request_id": "forged-request",
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        "cost_usd": "0.01",
        "status": "error",
        "finish_reason": None,
        "error_type": "TimeoutError",
    }
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(forged_settlement) + "\n")

    with pytest.raises(RecoveryAccountingError, match="settles unknown-billing"):
        RecoveryBudgetLedger(
            ledger.path,
            run_id="recovery-run",
            cap_usd=1.0,
            worst_case_cost_usd=0.20,
        )


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


def test_recovery_report_requires_each_real_oracle_call_to_link_ledger_attempts(
    tmp_path,
) -> None:
    """Recovery publication reconciles proposal, strengthening, alt, and retries."""
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    oracle_report = _linked_oracle_report(ledger)

    reconciliation = reconcile_recovery_reports(ledger, oracle_report)

    assert reconciliation == {
        "run_id": "recovery-run",
        "physical_calls": 4,
        "exact_cost_usd": "0.05",
        "status": "reconciled",
    }


def test_historical_oracle_accounting_remains_readable_but_non_authoritative(
    tmp_path,
) -> None:
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    oracle_report = _linked_oracle_report(ledger)
    historical = {
        "logical_call_id": "historical-unlinked",
        "physical_calls": [],
    }
    oracle_report.details["recovery_accounting"] = [historical]

    reconciliation = reconcile_recovery_reports(ledger, oracle_report)

    assert oracle_report.details["recovery_accounting"] == [historical]
    assert reconciliation["physical_calls"] == 4


@pytest.mark.parametrize("linked_value", [None, {}])
def test_recovery_report_rejects_missing_or_empty_oracle_ledger_linkage(
    tmp_path, linked_value: object
) -> None:
    """A real teacher attestation cannot be authorized by calibration-only data."""
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    oracle_report = _linked_oracle_report(ledger)
    calls = oracle_report.details["teacher_gates"]["differential"]["calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    calls[0]["recovery_accounting"] = linked_value
    # Historical/direct entries remain inspectable but cannot replace a call link.
    oracle_report.details["recovery_accounting"] = [
        {
            "logical_call_id": "historical-unlinked",
            "physical_calls": [],
        }
    ]

    with pytest.raises(
        RecoveryAccountingError,
        match="real oracle call.*(no ledger linkage|no logical call id)",
    ):
        reconcile_recovery_reports(ledger, oracle_report)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("stage", "oracle.alt_correct", "mismatched stage"),
        (
            "usage",
            {"prompt_tokens": 99, "completion_tokens": 1, "total_tokens": 100},
            "mismatched usage",
        ),
        ("cost", "0.99", "mismatched cost"),
        ("status", "success", "mismatched status"),
        ("retry", 99, "(mismatched retry|non-contiguous or out-of-order retries)"),
        ("model", "anthropic/other-model", "mismatched model"),
        ("logical_call_id", "wrong-logical-call", "mismatched logical_call_id"),
    ],
)
def test_recovery_report_rejects_mismatched_oracle_ledger_attributes(
    tmp_path, field: str, replacement: object, message: str
) -> None:
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    oracle_report = _linked_oracle_report(ledger)
    calls = oracle_report.details["teacher_gates"]["differential"]["calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    accounting = calls[0]["recovery_accounting"]
    assert isinstance(accounting, dict)
    physical_calls = accounting["physical_calls"]
    assert isinstance(physical_calls, list) and isinstance(physical_calls[0], dict)
    if field == "logical_call_id":
        accounting[field] = replacement
    else:
        physical_calls[0][field] = replacement

    with pytest.raises(RecoveryAccountingError, match=message):
        reconcile_recovery_reports(ledger, oracle_report)


def test_recovery_report_rejects_duplicate_or_unexpected_oracle_call_linkage(
    tmp_path,
) -> None:
    ledger = RecoveryBudgetLedger(
        tmp_path / "recovery-budget.jsonl",
        run_id="recovery-run",
        cap_usd=1.0,
        worst_case_cost_usd=0.20,
    )
    oracle_report = _linked_oracle_report(ledger)
    calls = oracle_report.details["teacher_gates"]["differential"]["calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    duplicate = deepcopy(calls[0])
    duplicate["call_kind"] = "strengthen"
    calls.append(duplicate)

    with pytest.raises(RecoveryAccountingError, match="duplicate physical call"):
        reconcile_recovery_reports(ledger, oracle_report)

    calls.pop()
    accounting = calls[0]["recovery_accounting"]
    assert isinstance(accounting, dict)
    physical = accounting["physical_calls"]
    assert isinstance(physical, list) and isinstance(physical[0], dict)
    physical[0]["physical_call_id"] = "unexpected-physical-call"

    with pytest.raises(RecoveryAccountingError, match="unexpected evidence"):
        reconcile_recovery_reports(ledger, oracle_report)
