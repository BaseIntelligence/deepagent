"""Offline unit tests for reserve/settle ledger (VAL-HARNESS-002)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from swe_factory.accounting import (
    DEFAULT_CAP_USD,
    AccountingError,
    BudgetLedger,
    default_ledger_path,
)


def test_default_cap_is_600() -> None:
    assert Decimal("600") == DEFAULT_CAP_USD


def test_reserve_before_settle_links_stage_and_task(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "ledger.jsonl", cap_usd=600, worst_case_cost_usd=1)
    pid = ledger.reserve(
        stage="hardness-panel",
        task_id="task-abc",
        model="x-ai/grok-4.5",
        reserved_cost_usd="0.50",
    )
    ledger.settle(
        pid,
        cost_usd="0.12",
        status="success",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        request_id="gen-1",
    )
    summary = ledger.summary()
    assert summary.settled_exact_usd == Decimal("0.12")
    assert summary.open_reserved_usd == Decimal("0")
    assert summary.under_cap is True
    assert summary.by_stage["hardness-panel"] == Decimal("0.12")
    assert summary.by_task["task-abc"] == Decimal("0.12")
    assert summary.by_model["x-ai/grok-4.5"] == Decimal("0.12")

    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    reserve_ev = json.loads(lines[0])
    settle_ev = json.loads(lines[1])
    assert reserve_ev["event"] == "reserve"
    assert reserve_ev["stage"] == "hardness-panel"
    assert reserve_ev["task_id"] == "task-abc"
    assert reserve_ev["model"] == "x-ai/grok-4.5"
    assert settle_ev["event"] == "settle"
    assert settle_ev["cost_usd"] == "0.12"
    assert settle_ev["task_id"] == "task-abc"


def test_open_reservation_counts_toward_cap(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "l.jsonl", cap_usd="1.00", worst_case_cost_usd="0.60")
    ledger.reserve(
        stage="teacher",
        task_id="t1",
        model="anthropic/claude-opus-4.8",
        reserved_cost_usd="0.60",
    )
    assert ledger.total_commit_usd() == Decimal("0.60")
    assert ledger.remaining_usd() == Decimal("0.40")
    with pytest.raises(AccountingError, match="exceed cap"):
        ledger.reserve(
            stage="hardness-panel",
            task_id="t2",
            model="x-ai/grok-4.5",
            reserved_cost_usd="0.50",
        )


def test_settle_cannot_exceed_reserved(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "l.jsonl", cap_usd=10, worst_case_cost_usd=1)
    pid = ledger.reserve(
        stage="hardness-panel",
        task_id="t1",
        model="x-ai/grok-4.5",
        reserved_cost_usd="0.25",
    )
    with pytest.raises(AccountingError, match="exceeds reserved"):
        ledger.settle(pid, cost_usd="0.30", status="success")


def test_unknown_billing_fail_closed_blocks_new_reserve(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "l.jsonl", cap_usd=10, worst_case_cost_usd=1)
    pid = ledger.reserve(
        stage="hardness-panel",
        task_id="t1",
        model="anthropic/claude-opus-4.8",
        reserved_cost_usd="1.0",
    )
    ledger.mark_unknown_billing(pid, reason_code="missing_provider_cost")
    assert ledger.has_unknown_billing() is True
    summary = ledger.summary()
    assert summary.has_unknown_billing is True
    assert summary.unknown_billing_count == 1
    # Liability keeps reserved amount under commit.
    assert summary.open_reserved_usd == Decimal("1.0")
    with pytest.raises(AccountingError, match="unknown_billing"):
        ledger.reserve(
            stage="hardness-panel",
            task_id="t2",
            model="x-ai/grok-4.5",
            reserved_cost_usd="0.1",
        )


def test_reload_from_disk_preserves_commit(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = BudgetLedger(path, cap_usd=600, worst_case_cost_usd=2)
    pid = ledger.reserve(
        stage="hardness-panel",
        task_id="task-1",
        model="x-ai/grok-4.5",
        reserved_cost_usd="1.5",
    )
    ledger.settle(pid, cost_usd="0.42", status="success", request_id="r1")
    reloaded = BudgetLedger(path, cap_usd=600, worst_case_cost_usd=2)
    assert reloaded.settled_exact_usd() == Decimal("0.42")
    summary_path = reloaded.write_summary_json()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["under_cap"] is True
    assert payload["settled_exact_usd"] == "0.42"
    assert "hardness-panel" in payload["by_stage"]
    assert "task-1" in payload["by_task"]


def test_default_ledger_path() -> None:
    p = default_ledger_path(Path("/tmp/factory-root"))
    assert p == Path("/tmp/factory-root/datasets/.factory/ledger.jsonl")
