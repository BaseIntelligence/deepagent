"""Exact reserve/settle spend ledger for OpenRouter provider calls.

Durable append-only JSONL under ``datasets/.factory/ledger.jsonl`` (or a test path).
Each physical call is reserved *before* the provider request, then settled with
exact cost (or marked unknown-billing → fail closed). Cap enforcement covers
settled exact spend + unresolved reserved amounts so exact + reserved ≤ cap_usd
(default $600).

Never stores prompts, responses, API keys, or exception text.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Literal

SCHEMA_VERSION: Final = 1
DEFAULT_LEDGER_REL: Final = Path("datasets") / ".factory" / "ledger.jsonl"
DEFAULT_CAP_USD: Final = Decimal("600")
DEFAULT_WORST_CASE_USD: Final = Decimal("2.50")

_EVENT_RESERVE: Final = "reserve"
_EVENT_SETTLE: Final = "settle"
_EVENT_UNKNOWN_BILLING: Final = "unknown_billing"
_SETTLED_STATUSES: Final = frozenset({"success", "error"})


class AccountingError(RuntimeError):
    """Raised when spend evidence is incomplete, inconsistent, or overops cap."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _money(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise AccountingError(f"{field} must be a non-negative decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AccountingError(f"{field} must be a non-negative decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise AccountingError(f"{field} must be a non-negative decimal")
    return parsed


def _money_text(value: Decimal | object, *, field: str) -> str:
    return format(_money(value, field=field), "f")


def _usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    result: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AccountingError(f"usage.{field} must be a non-negative int")
        result[field] = value
    return result


def default_ledger_path(root: Path | None = None) -> Path:
    """Canonical durable ledger path under datasets/.factory/."""
    base = root if root is not None else Path.cwd()
    return base / DEFAULT_LEDGER_REL


@dataclass(frozen=True, slots=True)
class Reservation:
    """One active or settled physical request reconstructed from the ledger."""

    physical_call_id: str
    stage: str
    task_id: str
    model: str
    reserved_cost_usd: Decimal
    reserve_event: dict[str, object]
    settle_event: dict[str, object] | None = None
    unknown_billing_event: dict[str, object] | None = None

    @property
    def is_open(self) -> bool:
        return self.settle_event is None and self.unknown_billing_event is None

    @property
    def is_settled(self) -> bool:
        return self.settle_event is not None

    @property
    def is_unknown_billing(self) -> bool:
        return self.unknown_billing_event is not None


@dataclass(frozen=True, slots=True)
class LedgerSummary:
    """Aggregated spend snapshot (exact + reserved under cap)."""

    path: Path
    cap_usd: Decimal
    settled_exact_usd: Decimal
    open_reserved_usd: Decimal
    total_commit_usd: Decimal  # settled + open reserved
    remaining_usd: Decimal
    open_call_count: int
    settled_call_count: int
    unknown_billing_count: int
    has_unknown_billing: bool
    under_cap: bool
    by_stage: dict[str, Decimal]
    by_task: dict[str, Decimal]
    by_model: dict[str, Decimal]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "cap_usd": _money_text(self.cap_usd, field="cap_usd"),
            "settled_exact_usd": _money_text(self.settled_exact_usd, field="settled"),
            "open_reserved_usd": _money_text(self.open_reserved_usd, field="open"),
            "total_commit_usd": _money_text(self.total_commit_usd, field="total"),
            "remaining_usd": _money_text(self.remaining_usd, field="remaining"),
            "open_call_count": self.open_call_count,
            "settled_call_count": self.settled_call_count,
            "unknown_billing_count": self.unknown_billing_count,
            "has_unknown_billing": self.has_unknown_billing,
            "under_cap": self.under_cap,
            "by_stage": {
                k: _money_text(v, field="stage") for k, v in sorted(self.by_stage.items())
            },
            "by_task": {k: _money_text(v, field="task") for k, v in sorted(self.by_task.items())},
            "by_model": {
                k: _money_text(v, field="model") for k, v in sorted(self.by_model.items())
            },
        }


class BudgetLedger:
    """Append-only fsynced reserve/settle ledger with hard cap enforcement.

    ``cap_usd`` is enforced before every physical request using settled exact
    spend plus all active worst-case reservations. Settlement that exceeds the
    reservation is a hard accounting failure. Unknown billing is fail-closed.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        cap_usd: float | str | Decimal = DEFAULT_CAP_USD,
        worst_case_cost_usd: float | str | Decimal = DEFAULT_WORST_CASE_USD,
        run_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.cap_usd = _money(cap_usd, field="cap_usd")
        self.worst_case_cost_usd = _money(worst_case_cost_usd, field="worst_case_cost_usd")
        if self.cap_usd <= 0:
            raise AccountingError("cap_usd must be greater than zero")
        if self.worst_case_cost_usd <= 0:
            raise AccountingError("worst_case_cost_usd must be greater than zero")
        self.run_id = (run_id or "default").strip() or "default"
        # RLock so concurrent pier trials can reserve/settle race-safely (M20 pool).
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
            self._fsync_file()
            self._fsync_parent()
        self._records = self._load_records()

    def _fsync_file(self) -> None:
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _fsync_parent(self) -> None:
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _append(self, event: dict[str, object]) -> None:
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_parent()

    def _load_records(self) -> dict[str, Reservation]:
        records: dict[str, Reservation] = {}
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return records
        for line_no, raw_line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AccountingError(f"ledger has malformed JSON at line {line_no}") from exc
            if not isinstance(event, dict):
                raise AccountingError(f"ledger event at line {line_no} must be an object")
            if event.get("schema_version") != SCHEMA_VERSION:
                raise AccountingError(f"ledger event at line {line_no} has unsupported schema")
            # Allow multi-run files that share the global path; index by physical id.
            physical = event.get("physical_call_id")
            if not isinstance(physical, str) or not physical:
                raise AccountingError(f"ledger event at line {line_no} has no physical_call_id")
            event_type = event.get("event")
            if event_type == _EVENT_RESERVE:
                if physical in records:
                    raise AccountingError(f"duplicate physical call {physical!r}")
                stage = event.get("stage")
                task_id = event.get("task_id")
                model = event.get("model")
                if (
                    not isinstance(stage, str)
                    or not stage.strip()
                    or not isinstance(task_id, str)
                    or not task_id.strip()
                    or not isinstance(model, str)
                    or not model.strip()
                ):
                    raise AccountingError(
                        f"ledger reserve at line {line_no} missing stage/task_id/model"
                    )
                records[physical] = Reservation(
                    physical_call_id=physical,
                    stage=stage.strip(),
                    task_id=task_id.strip(),
                    model=model.strip(),
                    reserved_cost_usd=_money(
                        event.get("reserved_cost_usd"), field="reserved_cost_usd"
                    ),
                    reserve_event=event,
                )
            elif event_type == _EVENT_SETTLE:
                base = records.get(physical)
                if base is None:
                    raise AccountingError(f"ledger settles unknown physical call {physical!r}")
                if base.settle_event is not None:
                    raise AccountingError(
                        f"ledger settles physical call {physical!r} more than once"
                    )
                if base.unknown_billing_event is not None:
                    raise AccountingError(
                        f"ledger settles unknown-billing physical call {physical!r}"
                    )
                self._validate_settle_keys(event, base)
                cost = _money(event.get("cost_usd"), field="cost_usd")
                if cost > base.reserved_cost_usd:
                    raise AccountingError(
                        f"settled cost {cost} exceeds reserved {base.reserved_cost_usd} "
                        f"for {physical!r}"
                    )
                status = event.get("status")
                if status not in _SETTLED_STATUSES:
                    raise AccountingError(
                        f"settlement for {physical!r} has invalid status {status!r}"
                    )
                raw_usage = event.get("usage")
                _usage_dict(raw_usage if isinstance(raw_usage, dict) else None)
                records[physical] = Reservation(
                    physical_call_id=base.physical_call_id,
                    stage=base.stage,
                    task_id=base.task_id,
                    model=base.model,
                    reserved_cost_usd=base.reserved_cost_usd,
                    reserve_event=base.reserve_event,
                    settle_event=event,
                    unknown_billing_event=None,
                )
            elif event_type == _EVENT_UNKNOWN_BILLING:
                base = records.get(physical)
                if base is None:
                    raise AccountingError(
                        f"ledger marks unknown billing for physical call {physical!r}"
                    )
                if base.settle_event is not None:
                    raise AccountingError(
                        f"ledger marks settled physical call {physical!r} as unknown"
                    )
                if base.unknown_billing_event is not None:
                    raise AccountingError(
                        f"ledger marks physical call {physical!r} unknown more than once"
                    )
                records[physical] = Reservation(
                    physical_call_id=base.physical_call_id,
                    stage=base.stage,
                    task_id=base.task_id,
                    model=base.model,
                    reserved_cost_usd=base.reserved_cost_usd,
                    reserve_event=base.reserve_event,
                    settle_event=None,
                    unknown_billing_event=event,
                )
            else:
                raise AccountingError(
                    f"ledger event at line {line_no} has unknown event type {event_type!r}"
                )
        return records

    @staticmethod
    def _validate_settle_keys(event: dict[str, object], base: Reservation) -> None:
        for field, expected in (
            ("stage", base.stage),
            ("task_id", base.task_id),
            ("model", base.model),
        ):
            if event.get(field) != expected:
                raise AccountingError(
                    f"settlement for {base.physical_call_id!r} mismatches {field}"
                )

    def settled_exact_usd(self) -> Decimal:
        with self._lock:
            total = Decimal("0")
            for rec in self._records.values():
                if rec.settle_event is not None:
                    total += _money(rec.settle_event.get("cost_usd"), field="cost_usd")
            return total

    def open_reserved_usd(self) -> Decimal:
        with self._lock:
            total = Decimal("0")
            for rec in self._records.values():
                if rec.is_open:
                    total += rec.reserved_cost_usd
                elif rec.is_unknown_billing:
                    # Unknown billing keeps full reserved commit until resolved —
                    # fail-closed remains under cap including these liabilities.
                    total += rec.reserved_cost_usd
            return total

    def total_commit_usd(self) -> Decimal:
        with self._lock:
            return self.settled_exact_usd() + self.open_reserved_usd()

    def remaining_usd(self) -> Decimal:
        with self._lock:
            return self.cap_usd - self.total_commit_usd()

    def has_unknown_billing(self) -> bool:
        with self._lock:
            return any(r.is_unknown_billing for r in self._records.values())

    def reserve(
        self,
        *,
        stage: str,
        task_id: str,
        model: str,
        reserved_cost_usd: float | str | Decimal | None = None,
        physical_call_id: str | None = None,
    ) -> str:
        """Reserve worst-case spend before a physical provider call.

        Returns the durable ``physical_call_id``. Raises :class:`AccountingError`
        when the reservation would exceed the hard cap or leave unknown billing.
        Thread-safe for concurrent pier trial pools (M20).
        """
        with self._lock:
            if self.has_unknown_billing():
                raise AccountingError(
                    "ledger has unknown_billing entries; reconcile before new reserves"
                )
            stage_s = stage.strip()
            task_s = task_id.strip()
            model_s = model.strip()
            if not stage_s or not task_s or not model_s:
                raise AccountingError("stage, task_id, and model must be non-empty")
            reserved = _money(
                reserved_cost_usd
                if reserved_cost_usd is not None
                else self.worst_case_cost_usd,
                field="reserved_cost_usd",
            )
            if reserved <= 0:
                raise AccountingError("reserved_cost_usd must be greater than zero")
            commit_after = self.total_commit_usd() + reserved
            if commit_after > self.cap_usd:
                raise AccountingError(
                    f"reserve would exceed cap: commit_after={commit_after} > "
                    f"cap={self.cap_usd} (stage={stage_s!r} task={task_s!r} model={model_s!r})"
                )
            physical = (physical_call_id or uuid.uuid4().hex).strip()
            if not physical:
                raise AccountingError("physical_call_id must be non-empty")
            if physical in self._records:
                raise AccountingError(f"duplicate physical_call_id {physical!r}")
            event: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "event": _EVENT_RESERVE,
                "ts": _timestamp(),
                "run_id": self.run_id,
                "physical_call_id": physical,
                "stage": stage_s,
                "task_id": task_s,
                "model": model_s,
                "reserved_cost_usd": _money_text(reserved, field="reserved_cost_usd"),
            }
            self._append(event)
            self._records[physical] = Reservation(
                physical_call_id=physical,
                stage=stage_s,
                task_id=task_s,
                model=model_s,
                reserved_cost_usd=reserved,
                reserve_event=event,
            )
            return physical

    def settle(
        self,
        physical_call_id: str,
        *,
        cost_usd: float | str | Decimal,
        status: Literal["success", "error"] = "success",
        usage: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        """Settle an open reservation with exact provider cost. Thread-safe."""
        with self._lock:
            physical = physical_call_id.strip()
            base = self._records.get(physical)
            if base is None:
                raise AccountingError(f"settle unknown physical call {physical!r}")
            if not base.is_open:
                raise AccountingError(f"physical call {physical!r} is already closed")
            cost = _money(cost_usd, field="cost_usd")
            if cost > base.reserved_cost_usd:
                raise AccountingError(
                    f"settled cost {cost} exceeds reserved {base.reserved_cost_usd} "
                    f"for {physical!r}"
                )
            if status not in _SETTLED_STATUSES:
                raise AccountingError(f"invalid settle status {status!r}")
            usage_d = _usage_dict(usage)
            event: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "event": _EVENT_SETTLE,
                "ts": _timestamp(),
                "run_id": self.run_id,
                "physical_call_id": physical,
                "stage": base.stage,
                "task_id": base.task_id,
                "model": base.model,
                "status": status,
                "cost_usd": _money_text(cost, field="cost_usd"),
                "usage": usage_d,
                "request_id": request_id or "",
            }
            self._append(event)
            self._records[physical] = Reservation(
                physical_call_id=base.physical_call_id,
                stage=base.stage,
                task_id=base.task_id,
                model=base.model,
                reserved_cost_usd=base.reserved_cost_usd,
                reserve_event=base.reserve_event,
                settle_event=event,
                unknown_billing_event=None,
            )

    def mark_unknown_billing(self, physical_call_id: str, *, reason_code: str) -> None:
        """Mark a reservation as unknown billing (fail closed for keeps)."""
        with self._lock:
            physical = physical_call_id.strip()
            base = self._records.get(physical)
            if base is None:
                raise AccountingError(
                    f"unknown_billing for unknown physical call {physical!r}"
                )
            if not base.is_open:
                raise AccountingError(f"physical call {physical!r} is already closed")
            code = reason_code.strip() or "unknown_billing"
            event: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "event": _EVENT_UNKNOWN_BILLING,
                "ts": _timestamp(),
                "run_id": self.run_id,
                "physical_call_id": physical,
                "stage": base.stage,
                "task_id": base.task_id,
                "model": base.model,
                "reason_code": code,
            }
            self._append(event)
            self._records[physical] = Reservation(
                physical_call_id=base.physical_call_id,
                stage=base.stage,
                task_id=base.task_id,
                model=base.model,
                reserved_cost_usd=base.reserved_cost_usd,
                reserve_event=base.reserve_event,
                settle_event=None,
                unknown_billing_event=event,
            )

    def summary(self) -> LedgerSummary:
        """Compute exact + reserved spend snapshot linked by stage/task/model."""
        with self._lock:
            settled = Decimal("0")
            open_reserved = Decimal("0")
            open_count = 0
            settled_count = 0
            unknown_count = 0
            by_stage: dict[str, Decimal] = {}
            by_task: dict[str, Decimal] = {}
            by_model: dict[str, Decimal] = {}

            def _bump(bucket: dict[str, Decimal], key: str, amount: Decimal) -> None:
                bucket[key] = bucket.get(key, Decimal("0")) + amount

            for rec in self._records.values():
                if rec.settle_event is not None:
                    amount = _money(rec.settle_event.get("cost_usd"), field="cost_usd")
                    settled += amount
                    settled_count += 1
                    _bump(by_stage, rec.stage, amount)
                    _bump(by_task, rec.task_id, amount)
                    _bump(by_model, rec.model, amount)
                elif rec.is_unknown_billing:
                    amount = rec.reserved_cost_usd
                    open_reserved += amount
                    unknown_count += 1
                    _bump(by_stage, rec.stage, amount)
                    _bump(by_task, rec.task_id, amount)
                    _bump(by_model, rec.model, amount)
                else:
                    amount = rec.reserved_cost_usd
                    open_reserved += amount
                    open_count += 1
                    _bump(by_stage, rec.stage, amount)
                    _bump(by_task, rec.task_id, amount)
                    _bump(by_model, rec.model, amount)

            total = settled + open_reserved
            remaining = self.cap_usd - total
            return LedgerSummary(
                path=self.path,
                cap_usd=self.cap_usd,
                settled_exact_usd=settled,
                open_reserved_usd=open_reserved,
                total_commit_usd=total,
                remaining_usd=remaining,
                open_call_count=open_count,
                settled_call_count=settled_count,
                unknown_billing_count=unknown_count,
                has_unknown_billing=unknown_count > 0,
                under_cap=total <= self.cap_usd,
                by_stage=by_stage,
                by_task=by_task,
                by_model=by_model,
            )

    def write_summary_json(self, path: Path | str | None = None) -> Path:
        """Persist ``ledger_summary.json`` next to the ledger by default."""
        summary = self.summary()
        out = Path(path) if path is not None else self.path.parent / "ledger_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return out


__all__ = [
    "DEFAULT_CAP_USD",
    "DEFAULT_LEDGER_REL",
    "DEFAULT_WORST_CASE_USD",
    "AccountingError",
    "BudgetLedger",
    "LedgerSummary",
    "Reservation",
    "SCHEMA_VERSION",
    "default_ledger_path",
]
