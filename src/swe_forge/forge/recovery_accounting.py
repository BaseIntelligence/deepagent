"""Durable, fail-closed LLM cost accounting for recovery recertification.

Recovery work is unusual because a publication must be able to prove every
physical request that contributed to its oracle or calibration evidence.  This
module provides a small append-only JSONL ledger.  A request is reserved and
fsynced *before* it is issued, then durably settled with the provider-reported
usage and cost whether it succeeds or fails.

The ledger intentionally contains operational metadata only.  It never stores
prompts, responses, endpoint URLs, headers, credentials, or exception text.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Literal, Protocol

from swe_forge.forge.models import CalibrationReport
from swe_forge.forge.teacher import Usage

_SCHEMA_VERSION = 1
_EVENT_RESERVE = "reserve"
_EVENT_SETTLE = "settle"
_EVENT_UNKNOWN_BILLING = "unknown_billing"
_SETTLED_STATUSES = frozenset(("success", "error"))
_SAFE_FINISH_REASONS = frozenset(
    ("stop", "length", "tool_calls", "function_call", "content_filter")
)
_SAFE_ERROR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

_ACTIVE_LEDGER: ContextVar[object] = ContextVar(
    "forge_active_recovery_ledger", default=None
)
_ACTIVE_CANDIDATE: ContextVar[str] = ContextVar(
    "forge_active_recovery_candidate", default=""
)
_ACTIVE_STAGE: ContextVar[str] = ContextVar("forge_active_recovery_stage", default="")


class RecoveryAccountingError(RuntimeError):
    """Raised when recovery cost evidence is incomplete or inconsistent."""


@contextmanager
def campaign_call_context(
    ledger: "RecoveryBudgetLedger",
    *,
    candidate_identity: str,
    stage: str,
):
    """Make a campaign ledger the default for nested teacher/panel calls.

    The context is intentionally process-local and scoped. It lets the fresh
    campaign thread its durable ledger through existing stage implementations
    without mutable global provider configuration or bypassing the teacher
    transport boundary.
    """
    candidate = str(candidate_identity).strip()
    stage_name = str(stage).strip()
    if not candidate or not stage_name:
        raise RecoveryAccountingError(
            "campaign call context requires candidate_identity and stage"
        )
    ledger_token = _ACTIVE_LEDGER.set(ledger)
    candidate_token = _ACTIVE_CANDIDATE.set(candidate)
    stage_token = _ACTIVE_STAGE.set(stage_name)
    try:
        yield
    finally:
        _ACTIVE_STAGE.reset(stage_token)
        _ACTIVE_CANDIDATE.reset(candidate_token)
        _ACTIVE_LEDGER.reset(ledger_token)


def active_campaign_context() -> tuple["RecoveryBudgetLedger | None", str, str]:
    """Return the active ledger, candidate identity, and stage for nested calls."""
    active = _ACTIVE_LEDGER.get()
    return (
        active if isinstance(active, RecoveryBudgetLedger) else None,
        _ACTIVE_CANDIDATE.get(),
        _ACTIVE_STAGE.get(),
    )


class _ReportLike(Protocol):
    details: dict[str, object]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _money(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise RecoveryAccountingError(f"{field} must be a non-negative decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RecoveryAccountingError(
            f"{field} must be a non-negative decimal"
        ) from exc
    if not parsed.is_finite() or parsed < 0:
        raise RecoveryAccountingError(f"{field} must be a non-negative decimal")
    return parsed


def _money_text(value: object, *, field: str) -> str:
    """Render a decimal without losing the provider-reported precision."""
    return format(_money(value, field=field), "f")


def _usage_dict(usage: Usage | dict[str, object] | None) -> dict[str, int]:
    if isinstance(usage, Usage):
        payload: dict[str, object] = dict(usage.to_dict())
    elif isinstance(usage, dict):
        payload = usage
    elif usage is None:
        payload = {}
    else:
        raise RecoveryAccountingError("usage must be a Usage object or mapping")
    result: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload.get(field, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RecoveryAccountingError(f"usage.{field} must be a non-negative int")
        result[field] = value
    return result


def _safe_finish_reason(value: object) -> str | None:
    """Persist only known terminal labels, never arbitrary provider content."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in _SAFE_FINISH_REASONS else "other"


def _safe_error_type(value: object) -> str:
    """Persist a bounded exception class name, not provider exception text."""
    return value if isinstance(value, str) and _SAFE_ERROR_TYPE.fullmatch(value) else ""


def sanitize_request_id(value: object) -> str:
    """Return a bounded opaque provider request ID, or no ID if it is unsafe."""
    return value if isinstance(value, str) and _SAFE_REQUEST_ID.fullmatch(value) else ""


@dataclass(frozen=True)
class _Reservation:
    """One active or settled physical request reconstructed from the ledger."""

    physical_call_id: str
    logical_call_id: str
    stage: str
    model: str
    retry: int
    reserved_cost: Decimal
    reserve_event: dict[str, object]
    candidate_identity: str = ""
    settle_event: dict[str, object] | None = None
    unknown_billing_event: dict[str, object] | None = None


class RecoveryBudgetLedger:
    """Append-only, fsynced request reservations for one recovery run.

    ``cap_usd`` is enforced before each physical request using settled exact
    spend plus all active worst-case reservations.  Settlement replaces the
    request's reservation with the exact provider cost, and a provider charge
    exceeding its reserved bound is itself a hard accounting failure.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str,
        cap_usd: float | str | Decimal,
        worst_case_cost_usd: float | str | Decimal,
    ) -> None:
        self.path = Path(path)
        self.run_id = str(run_id).strip()
        self.cap_usd = _money(cap_usd, field="cap_usd")
        self.worst_case_cost_usd = _money(
            worst_case_cost_usd, field="worst_case_cost_usd"
        )
        if not self.run_id:
            raise RecoveryAccountingError("run_id must be non-empty")
        if self.cap_usd <= 0:
            raise RecoveryAccountingError("cap_usd must be greater than zero")
        if self.worst_case_cost_usd <= 0:
            raise RecoveryAccountingError(
                "worst_case_cost_usd must be greater than zero"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
            self._fsync_file()
            self._fsync_parent()
        self._records = self._load_records()
        self._logical_reservations = self._index_logical_reservations(
            self._records.values()
        )

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

    @staticmethod
    def _index_logical_reservations(
        records: Iterable[_Reservation],
    ) -> dict[str, list[_Reservation]]:
        """Return physical reservations grouped by their durable logical owner."""
        logical_reservations: dict[str, list[_Reservation]] = {}
        for reservation in records:
            logical_reservations.setdefault(reservation.logical_call_id, []).append(
                reservation
            )
        return logical_reservations

    @staticmethod
    def _validate_logical_reservation(
        logical_reservations: dict[str, list[_Reservation]],
        *,
        logical_call_id: str,
        stage: str,
        model: str,
        retry: int,
    ) -> None:
        """Require a logical owner to append its sole contiguous retry sequence."""
        previous = logical_reservations.get(logical_call_id, [])
        if not previous:
            if retry != 0:
                raise RecoveryAccountingError(
                    f"logical call {logical_call_id!r} must start at retry 0"
                )
            return

        owner = previous[0]
        if stage != owner.stage:
            raise RecoveryAccountingError(
                f"logical call {logical_call_id!r} is already owned by stage "
                f"{owner.stage!r}"
            )
        if model != owner.model:
            raise RecoveryAccountingError(
                f"logical call {logical_call_id!r} is already owned by model "
                f"{owner.model!r}"
            )

        expected_retry = len(previous)
        if retry < expected_retry:
            raise RecoveryAccountingError(
                f"duplicate logical retry {retry} for logical call {logical_call_id!r}"
            )
        if retry != expected_retry:
            raise RecoveryAccountingError(
                f"logical call {logical_call_id!r} must reserve contiguous retry "
                f"{expected_retry}"
            )

    def _load_records(self) -> dict[str, _Reservation]:
        records: dict[str, _Reservation] = {}
        logical_reservations: dict[str, list[_Reservation]] = {}
        for line_no, raw_line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RecoveryAccountingError(
                    f"ledger has malformed JSON at line {line_no}"
                ) from exc
            if not isinstance(event, dict):
                raise RecoveryAccountingError(
                    f"ledger event at line {line_no} must be an object"
                )
            if event.get("schema_version") != _SCHEMA_VERSION:
                raise RecoveryAccountingError(
                    f"ledger event at line {line_no} has unsupported schema"
                )
            if event.get("run_id") != self.run_id:
                raise RecoveryAccountingError(
                    f"ledger event at line {line_no} belongs to a different run"
                )
            event_type = event.get("event")
            physical = event.get("physical_call_id")
            if not isinstance(physical, str) or not physical:
                raise RecoveryAccountingError(
                    f"ledger event at line {line_no} has no physical_call_id"
                )
            if event_type == _EVENT_RESERVE:
                if physical in records:
                    raise RecoveryAccountingError(
                        f"ledger has duplicate physical call {physical!r}"
                    )
                logical = event.get("logical_call_id")
                stage = event.get("stage")
                model = event.get("model")
                retry = event.get("retry")
                if (
                    not isinstance(logical, str)
                    or not logical
                    or not isinstance(stage, str)
                    or not stage
                    or not isinstance(model, str)
                    or not model
                    or not isinstance(retry, int)
                    or isinstance(retry, bool)
                    or retry < 0
                ):
                    raise RecoveryAccountingError(
                        f"ledger reserve at line {line_no} is malformed"
                    )
                self._validate_logical_reservation(
                    logical_reservations,
                    logical_call_id=logical,
                    stage=stage,
                    model=model,
                    retry=retry,
                )
                reservation = _Reservation(
                    physical_call_id=physical,
                    logical_call_id=logical,
                    stage=stage,
                    model=model,
                    retry=retry,
                    reserved_cost=_money(
                        event.get("reserved_cost_usd"),
                        field="reserved_cost_usd",
                    ),
                    reserve_event=event,
                    candidate_identity=str(event.get("candidate_identity") or ""),
                )
                records[physical] = reservation
                logical_reservations.setdefault(logical, []).append(reservation)
            elif event_type == _EVENT_SETTLE:
                settlement_reservation = records.get(physical)
                if settlement_reservation is None:
                    raise RecoveryAccountingError(
                        f"ledger settles unknown physical call {physical!r}"
                    )
                if settlement_reservation.settle_event is not None:
                    raise RecoveryAccountingError(
                        f"ledger settles physical call {physical!r} more than once"
                    )
                if settlement_reservation.unknown_billing_event is not None:
                    raise RecoveryAccountingError(
                        f"ledger settles unknown-billing physical call {physical!r}"
                    )
                self._validate_settlement(event, settlement_reservation)
                records[physical] = _Reservation(
                    **{
                        **settlement_reservation.__dict__,
                        "settle_event": event,
                    }
                )
            elif event_type == _EVENT_UNKNOWN_BILLING:
                unknown_billing_reservation = records.get(physical)
                if unknown_billing_reservation is None:
                    raise RecoveryAccountingError(
                        f"ledger marks unknown billing for physical call {physical!r}"
                    )
                if unknown_billing_reservation.settle_event is not None:
                    raise RecoveryAccountingError(
                        f"ledger marks settled physical call {physical!r} "
                        "as unknown billing"
                    )
                if unknown_billing_reservation.unknown_billing_event is not None:
                    raise RecoveryAccountingError(
                        f"ledger marks physical call {physical!r} as unknown "
                        "billing more than once"
                    )
                self._validate_unknown_billing(event, unknown_billing_reservation)
                records[physical] = _Reservation(
                    **{
                        **unknown_billing_reservation.__dict__,
                        "unknown_billing_event": event,
                    }
                )
            else:
                raise RecoveryAccountingError(
                    f"ledger event at line {line_no} has unknown event type"
                )
        return records

    def _validate_settlement(
        self, event: dict[str, object], reservation: _Reservation
    ) -> None:
        for field, expected in (
            ("logical_call_id", reservation.logical_call_id),
            ("stage", reservation.stage),
            ("model", reservation.model),
            ("retry", reservation.retry),
        ):
            if event.get(field) != expected:
                raise RecoveryAccountingError(
                    f"settlement for {reservation.physical_call_id!r} mismatches {field}"
                )
        if (
            reservation.candidate_identity
            and event.get("candidate_identity", "") != reservation.candidate_identity
        ):
            raise RecoveryAccountingError(
                f"settlement for {reservation.physical_call_id!r} mismatches "
                "candidate_identity"
            )
        status = event.get("status")
        if status not in _SETTLED_STATUSES:
            raise RecoveryAccountingError(
                f"settlement for {reservation.physical_call_id!r} has invalid status"
            )
        _money(event.get("cost_usd"), field="cost_usd")
        raw_usage = event.get("usage")
        _usage_dict(raw_usage if isinstance(raw_usage, dict) else None)
        if status == "success" and not isinstance(event.get("request_id"), str):
            raise RecoveryAccountingError(
                f"successful settlement for {reservation.physical_call_id!r} "
                "has no request_id"
            )
        if event.get("finish_reason") not in (
            None,
            *_SAFE_FINISH_REASONS,
            "other",
        ):
            raise RecoveryAccountingError(
                f"settlement for {reservation.physical_call_id!r} has invalid "
                "finish_reason"
            )
        if not isinstance(event.get("error_type", ""), str):
            raise RecoveryAccountingError(
                f"settlement for {reservation.physical_call_id!r} has invalid "
                "error_type"
            )

    def _validate_unknown_billing(
        self, event: dict[str, object], reservation: _Reservation
    ) -> None:
        for field, expected in (
            ("logical_call_id", reservation.logical_call_id),
            ("stage", reservation.stage),
            ("model", reservation.model),
            ("retry", reservation.retry),
        ):
            if event.get(field) != expected:
                raise RecoveryAccountingError(
                    f"unknown billing for {reservation.physical_call_id!r} "
                    f"mismatches {field}"
                )
        if (
            reservation.candidate_identity
            and event.get("candidate_identity", "") != reservation.candidate_identity
        ):
            raise RecoveryAccountingError(
                f"unknown billing for {reservation.physical_call_id!r} "
                "mismatches candidate_identity"
            )
        error_type = event.get("error_type")
        if not isinstance(error_type, str) or not _safe_error_type(error_type):
            raise RecoveryAccountingError(
                f"unknown billing for {reservation.physical_call_id!r} "
                "has an invalid error_type"
            )

    @property
    def total_exact_cost(self) -> Decimal:
        return sum(
            (
                _money(record.settle_event.get("cost_usd"), field="cost_usd")
                for record in self._records.values()
                if record.settle_event is not None
            ),
            Decimal(),
        )

    @property
    def total_active_reservations(self) -> Decimal:
        return sum(
            (
                record.reserved_cost
                for record in self._records.values()
                if record.settle_event is None
            ),
            Decimal(),
        )

    def reserve(
        self,
        *,
        logical_call_id: str,
        stage: str,
        model: str,
        retry: int,
        worst_case_cost_usd: float | str | Decimal | None = None,
        candidate_identity: str = "",
    ) -> str:
        """Durably reserve a physical request before calling a provider."""
        logical = str(logical_call_id).strip()
        stage_name = str(stage).strip()
        model_name = str(model).strip()
        if not logical or not stage_name or not model_name:
            raise RecoveryAccountingError(
                "logical_call_id, stage, and model must be non-empty"
            )
        candidate_name = str(candidate_identity).strip()
        if not isinstance(retry, int) or isinstance(retry, bool) or retry < 0:
            raise RecoveryAccountingError("retry must be >= 0")
        self._validate_logical_reservation(
            self._logical_reservations,
            logical_call_id=logical,
            stage=stage_name,
            model=model_name,
            retry=retry,
        )
        reserved = (
            self.worst_case_cost_usd
            if worst_case_cost_usd is None
            else _money(worst_case_cost_usd, field="worst_case_cost_usd")
        )
        projected = self.total_exact_cost + self.total_active_reservations + reserved
        if projected > self.cap_usd:
            raise RecoveryAccountingError(
                "recovery budget cap exhausted before physical LLM request "
                f"(projected={format(projected, 'f')}, cap={format(self.cap_usd, 'f')})"
            )
        physical = uuid.uuid4().hex
        event: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "event": _EVENT_RESERVE,
            "timestamp": _timestamp(),
            "run_id": self.run_id,
            "logical_call_id": logical,
            "physical_call_id": physical,
            "stage": stage_name,
            "model": model_name,
            "retry": retry,
            "reserved_cost_usd": format(reserved, "f"),
        }
        if candidate_name:
            event["candidate_identity"] = candidate_name
        self._append(event)
        self._records[physical] = _Reservation(
            physical_call_id=physical,
            logical_call_id=logical,
            stage=stage_name,
            model=model_name,
            retry=retry,
            reserved_cost=reserved,
            reserve_event=event,
            candidate_identity=candidate_name,
        )
        self._logical_reservations.setdefault(logical, []).append(
            self._records[physical]
        )
        return physical

    def settle(
        self,
        physical_call_id: str,
        *,
        request_id: str = "",
        usage: Usage | dict[str, object] | None = None,
        cost: float | str | Decimal = 0,
        status: Literal["success", "error"],
        finish_reason: str | None = None,
        error_type: str = "",
    ) -> None:
        """Append and fsync the terminal result for a previously reserved call."""
        reservation = self._records.get(physical_call_id)
        if reservation is None:
            raise RecoveryAccountingError(
                f"cannot settle unknown physical call {physical_call_id!r}"
            )
        if reservation.settle_event is not None:
            raise RecoveryAccountingError(
                f"cannot settle physical call {physical_call_id!r} twice"
            )
        if reservation.unknown_billing_event is not None:
            raise RecoveryAccountingError(
                f"cannot settle unknown-billing physical call {physical_call_id!r}"
            )
        event: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "event": _EVENT_SETTLE,
            "timestamp": _timestamp(),
            "run_id": self.run_id,
            "logical_call_id": reservation.logical_call_id,
            "physical_call_id": reservation.physical_call_id,
            "stage": reservation.stage,
            "model": reservation.model,
            "retry": reservation.retry,
            "candidate_identity": reservation.candidate_identity,
            "request_id": sanitize_request_id(request_id),
            "usage": _usage_dict(usage),
            "cost_usd": _money_text(cost, field="cost_usd"),
            "status": status,
            "finish_reason": _safe_finish_reason(finish_reason),
            "error_type": _safe_error_type(error_type),
        }
        self._validate_settlement(event, reservation)
        self._append(event)
        self._records[physical_call_id] = _Reservation(
            **{
                **reservation.__dict__,
                "settle_event": event,
            }
        )
        if _money(event["cost_usd"], field="cost_usd") > reservation.reserved_cost:
            raise RecoveryAccountingError(
                f"settlement for {reservation.physical_call_id!r} exceeded its "
                "pre-reserved worst-case cost"
            )

    def mark_unknown_billing(self, physical_call_id: str, *, error_type: str) -> None:
        """Durably retain an unresolved post-send reservation without fake metering."""
        reservation = self._records.get(physical_call_id)
        if reservation is None:
            raise RecoveryAccountingError(
                f"cannot mark unknown billing for unknown physical call "
                f"{physical_call_id!r}"
            )
        if reservation.settle_event is not None:
            raise RecoveryAccountingError(
                f"cannot mark settled physical call {physical_call_id!r} "
                "as unknown billing"
            )
        if reservation.unknown_billing_event is not None:
            raise RecoveryAccountingError(
                f"physical call {physical_call_id!r} is already unknown billing"
            )
        safe_error_type = _safe_error_type(error_type)
        if not safe_error_type:
            raise RecoveryAccountingError("unknown billing requires a safe error type")
        event: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "event": _EVENT_UNKNOWN_BILLING,
            "timestamp": _timestamp(),
            "run_id": self.run_id,
            "logical_call_id": reservation.logical_call_id,
            "physical_call_id": reservation.physical_call_id,
            "stage": reservation.stage,
            "model": reservation.model,
            "retry": reservation.retry,
            "candidate_identity": reservation.candidate_identity,
            "error_type": safe_error_type,
        }
        self._validate_unknown_billing(event, reservation)
        self._append(event)
        self._records[physical_call_id] = _Reservation(
            **{
                **reservation.__dict__,
                "unknown_billing_event": event,
            }
        )

    def events(self) -> list[dict[str, object]]:
        """Return parsed events for audit or tests, preserving append order."""
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def settled_calls(self) -> list[dict[str, object]]:
        """Return secret-free terminal call records in reservation order."""
        calls: list[dict[str, object]] = []
        for event in self.events():
            if event.get("event") != _EVENT_SETTLE:
                continue
            calls.append(
                {
                    "run_id": event["run_id"],
                    "logical_call_id": event["logical_call_id"],
                    "physical_call_id": event["physical_call_id"],
                    "stage": event["stage"],
                    "model": event["model"],
                    **(
                        {"candidate_identity": event["candidate_identity"]}
                        if isinstance(event.get("candidate_identity"), str)
                        and event["candidate_identity"]
                        else {}
                    ),
                    "retry": event["retry"],
                    "request_id": event["request_id"],
                    "usage": event["usage"],
                    "cost": event["cost_usd"],
                    "status": event["status"],
                    "finish_reason": event["finish_reason"],
                    "error_type": event["error_type"],
                }
            )
        return calls

    def unsettled_call_ids(self) -> tuple[str, ...]:
        return tuple(
            physical
            for physical, record in self._records.items()
            if record.settle_event is None
        )

    def unknown_billing_call_ids(self) -> tuple[str, ...]:
        """Return unresolved calls whose provider billing cannot be proven exact."""
        return tuple(
            physical
            for physical, record in self._records.items()
            if record.unknown_billing_event is not None
        )


def _iter_physical_calls(evidence: Iterable[object]) -> list[dict[str, object]]:
    physical: list[dict[str, object]] = []
    logical_attestations: set[str] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise RecoveryAccountingError(f"recovery evidence {index} is malformed")
        logical = item.get("logical_call_id")
        calls = item.get("physical_calls")
        if not isinstance(logical, str) or not logical or not isinstance(calls, list):
            raise RecoveryAccountingError(
                f"recovery evidence {index} has no logical call or physical calls"
            )
        if logical in logical_attestations:
            raise RecoveryAccountingError(
                f"recovery evidence has duplicate logical attestation {logical!r}"
            )
        logical_attestations.add(logical)
        retries: list[int] = []
        for call in calls:
            if not isinstance(call, dict):
                raise RecoveryAccountingError(
                    f"recovery evidence {index} has a malformed physical call"
                )
            retry = call.get("retry")
            if not isinstance(retry, int) or isinstance(retry, bool) or retry < 0:
                raise RecoveryAccountingError(
                    f"recovery evidence {index} has a malformed retry"
                )
            retries.append(retry)
            physical.append({**call, "logical_call_id": logical})
        if retries != list(range(len(retries))):
            raise RecoveryAccountingError(
                f"recovery evidence {index} has non-contiguous or out-of-order retries"
            )
    return physical


def reconcile_recovery_call_evidence(
    ledger: RecoveryBudgetLedger,
    evidence: Iterable[object],
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Require one-to-one agreement between settled ledger and stage evidence.

    Missing, duplicate, unresolved, or mismatched calls raise before a caller
    can publish a recertified recovery generation.
    """
    unsettled = ledger.unsettled_call_ids()
    unknown_billing = ledger.unknown_billing_call_ids()
    if unknown_billing:
        raise RecoveryAccountingError(
            "recovery ledger has unknown-billing physical calls: "
            + ", ".join(unknown_billing)
        )
    if unsettled:
        raise RecoveryAccountingError(
            "recovery ledger has unsettled physical calls: " + ", ".join(unsettled)
        )
    if ledger.total_exact_cost > ledger.cap_usd:
        raise RecoveryAccountingError(
            "recovery ledger settled cost exceeds its configured budget cap "
            f"(cost={format(ledger.total_exact_cost, 'f')}, "
            f"cap={format(ledger.cap_usd, 'f')})"
        )
    evidence_calls = _iter_physical_calls(evidence)
    evidence_by_id: dict[str, dict[str, object]] = {}
    for call in evidence_calls:
        physical = call.get("physical_call_id")
        if not isinstance(physical, str) or not physical:
            raise RecoveryAccountingError("recovery evidence physical call is missing")
        if physical in evidence_by_id:
            raise RecoveryAccountingError(
                f"recovery evidence has duplicate physical call {physical!r}"
            )
        evidence_by_id[physical] = call

    ledger_calls = {
        str(call["physical_call_id"]): call for call in ledger.settled_calls()
    }
    missing = sorted(set(ledger_calls) - set(evidence_by_id))
    unexpected = sorted(set(evidence_by_id) - set(ledger_calls))
    if require_complete and (missing or unexpected):
        parts = []
        if missing:
            parts.append("missing evidence for " + ", ".join(missing))
        if unexpected:
            parts.append("unexpected evidence for " + ", ".join(unexpected))
        raise RecoveryAccountingError(
            "recovery call evidence is missing: " + "; ".join(parts)
        )

    compare_ids = set(ledger_calls) if require_complete else set(evidence_by_id)
    for physical in compare_ids:
        if physical not in ledger_calls:
            raise RecoveryAccountingError(
                f"recovery evidence cites unknown physical call {physical!r}"
            )
        settled = ledger_calls[physical]
        observed = evidence_by_id[physical]
        for field in (
            "run_id",
            "logical_call_id",
            "stage",
            "model",
            "retry",
            "request_id",
            "status",
            "finish_reason",
            "error_type",
        ):
            if observed.get(field) != settled.get(field):
                raise RecoveryAccountingError(
                    f"recovery call {physical!r} has mismatched {field}"
                )
        raw_observed_usage = observed.get("usage")
        observed_usage = _usage_dict(
            raw_observed_usage if isinstance(raw_observed_usage, dict) else None
        )
        if observed_usage != settled["usage"]:
            raise RecoveryAccountingError(
                f"recovery call {physical!r} has mismatched usage"
            )
        if _money(observed.get("cost"), field="cost") != _money(
            settled["cost"], field="cost"
        ):
            raise RecoveryAccountingError(
                f"recovery call {physical!r} has mismatched cost"
            )
    return {
        "run_id": ledger.run_id,
        "physical_calls": len(ledger_calls),
        "exact_cost_usd": format(ledger.total_exact_cost, "f"),
        "status": "reconciled",
    }


def _oracle_call_context(gate: object, call_kind: object, index: int) -> str:
    """Describe a safe oracle attestation location without serializing content."""
    gate_name = gate if isinstance(gate, str) and gate else "unknown"
    kind_name = call_kind if isinstance(call_kind, str) and call_kind else "unknown"
    return f"real oracle call {gate_name}/{kind_name} at index {index}"


def _require_oracle_recovery_evidence(
    oracle_report: _ReportLike,
) -> list[dict[str, object]]:
    """Validate and collect every authoritative oracle ledger link.

    Recovery reports can retain historical direct accounting for inspection, but
    that legacy material is not an authority to publish.  Only a concrete
    ``real_teacher`` gate attestation can link its logical call to the complete
    set of physical attempts.  The later ledger reconciliation then proves those
    records one-to-one against the durable reserve/settle ledger.
    """
    teacher_gates = oracle_report.details.get("teacher_gates")
    if not isinstance(teacher_gates, dict):
        return []

    evidence: list[dict[str, object]] = []
    for payload_gate, payload in teacher_gates.items():
        if not isinstance(payload, dict):
            continue
        calls = payload.get("calls")
        if not isinstance(calls, list):
            continue
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or call.get("real_teacher") is not True:
                continue
            gate = call.get("gate")
            context = _oracle_call_context(gate, call.get("call_kind"), index)
            if not isinstance(gate, str) or not gate.strip():
                raise RecoveryAccountingError(f"{context} has no gate name")
            if isinstance(payload_gate, str) and payload_gate and gate != payload_gate:
                raise RecoveryAccountingError(
                    f"{context} is recorded under a different teacher gate"
                )
            expected_stage = f"oracle.{gate}"
            accounting = call.get("recovery_accounting")
            if not isinstance(accounting, dict):
                raise RecoveryAccountingError(f"{context} has no ledger linkage")
            logical = accounting.get("logical_call_id")
            physical_calls = accounting.get("physical_calls")
            if not isinstance(logical, str) or not logical.strip():
                raise RecoveryAccountingError(f"{context} has no logical call id")
            if not isinstance(physical_calls, list) or not physical_calls:
                raise RecoveryAccountingError(f"{context} has no physical ledger calls")

            model = call.get("model")
            status = call.get("status")
            raw_usage = call.get("usage")
            if not isinstance(model, str) or not model.strip():
                raise RecoveryAccountingError(f"{context} has no model")
            if status not in _SETTLED_STATUSES:
                raise RecoveryAccountingError(f"{context} has invalid status")
            if not isinstance(raw_usage, dict):
                raise RecoveryAccountingError(f"{context} has malformed usage")
            declared_usage = _usage_dict(raw_usage)
            declared_cost = _money(call.get("cost"), field=f"{context} cost")

            aggregate_usage = Usage()
            aggregate_cost = Decimal()
            retries: list[int] = []
            for physical_index, physical in enumerate(physical_calls):
                if not isinstance(physical, dict):
                    raise RecoveryAccountingError(
                        f"{context} has malformed physical call {physical_index}"
                    )
                child_logical = physical.get("logical_call_id")
                if child_logical is not None and child_logical != logical:
                    raise RecoveryAccountingError(
                        f"{context} has mismatched logical_call_id"
                    )
                if physical.get("stage") != expected_stage:
                    raise RecoveryAccountingError(f"{context} has mismatched stage")
                if physical.get("model") != model:
                    raise RecoveryAccountingError(f"{context} has mismatched model")
                retry = physical.get("retry")
                if not isinstance(retry, int) or isinstance(retry, bool) or retry < 0:
                    raise RecoveryAccountingError(f"{context} has malformed retry")
                retries.append(retry)
                raw_physical_usage = physical.get("usage")
                if not isinstance(raw_physical_usage, dict):
                    raise RecoveryAccountingError(
                        f"{context} has malformed physical usage"
                    )
                aggregate_usage = aggregate_usage + Usage(
                    **_usage_dict(raw_physical_usage)
                )
                aggregate_cost += _money(
                    physical.get("cost"), field=f"{context} physical cost"
                )
            if retries != list(range(len(retries))):
                raise RecoveryAccountingError(
                    f"{context} has non-contiguous or out-of-order retries"
                )
            terminal = physical_calls[-1]
            assert isinstance(terminal, dict)  # checked in the loop above
            if terminal.get("status") != status:
                raise RecoveryAccountingError(f"{context} has mismatched status")
            if aggregate_usage.to_dict() != declared_usage:
                raise RecoveryAccountingError(f"{context} has mismatched usage")
            if aggregate_cost != declared_cost:
                raise RecoveryAccountingError(f"{context} has mismatched cost")
            evidence.append(
                {
                    "logical_call_id": logical,
                    "physical_calls": [dict(item) for item in physical_calls],
                }
            )
    return evidence


def accounting_evidence_from_reports(
    oracle_report: _ReportLike, calibration_report: _ReportLike | None = None
) -> list[dict[str, object]]:
    """Collect only authoritative oracle and calibration recovery evidence."""
    evidence = _require_oracle_recovery_evidence(oracle_report)

    if calibration_report is not None:
        direct_calibration = calibration_report.details.get("recovery_accounting")
        if isinstance(direct_calibration, list):
            evidence.extend(
                item for item in direct_calibration if isinstance(item, dict)
            )
    return evidence


def require_calibration_recovery_evidence(
    calibration_report: CalibrationReport,
) -> None:
    """Require every fresh calibration logical call to cite physical ledger calls.

    A validation probe is one logical call and a solver rollout is one logical
    call that may contain several physical turns. Every row must retain its
    secret-free physical records, and their exact usage/cost sum must match the
    existing calibration accounting row. This prevents a recalibrator from
    returning an empty ledger-linkage list after issuing unmetered requests.
    """
    accounting = calibration_report.details.get("usage_accounting")
    if not isinstance(accounting, dict):
        raise RecoveryAccountingError("fresh calibration has no usage accounting")
    if not calibration_report.models:
        raise RecoveryAccountingError("fresh calibration has no validated models")
    expected_rollout_calls = sum(model.k for model in calibration_report.models)
    per_call_records: list[object] = []
    for section_name in ("validation", "rollout"):
        section = accounting.get(section_name)
        if not isinstance(section, dict):
            raise RecoveryAccountingError(
                f"fresh calibration has no {section_name} accounting"
            )
        calls = section.get("calls")
        rows = section.get("per_call")
        if (
            not isinstance(calls, int)
            or isinstance(calls, bool)
            or calls < 0
            or not isinstance(rows, list)
            or len(rows) != calls
            or (section_name == "validation" and calls < len(calibration_report.models))
            or (section_name == "rollout" and calls != expected_rollout_calls)
        ):
            raise RecoveryAccountingError(
                f"fresh calibration {section_name} call count is inconsistent"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RecoveryAccountingError(
                    f"fresh calibration {section_name} row {index} is malformed"
                )
            raw_records = row.get("recovery_accounting")
            records = (
                [raw_records]
                if isinstance(raw_records, dict)
                else raw_records
                if isinstance(raw_records, list)
                else []
            )
            if not records:
                raise RecoveryAccountingError(
                    f"fresh calibration {section_name} row {index} has no "
                    "physical ledger evidence"
                )
            per_call_records.extend(records)
            physical = _iter_physical_calls(records)
            physical_usage = Usage()
            physical_cost = Decimal()
            for call in physical:
                raw_usage = call.get("usage")
                physical_usage = physical_usage + Usage(
                    **_usage_dict(raw_usage if isinstance(raw_usage, dict) else None)
                )
                physical_cost += _money(call.get("cost"), field="cost")
            row_usage = row.get("usage")
            if (
                _usage_dict(row_usage if isinstance(row_usage, dict) else None)
                != physical_usage.to_dict()
            ):
                raise RecoveryAccountingError(
                    f"fresh calibration {section_name} row {index} has mismatched usage"
                )
            if _money(row.get("cost"), field="cost") != physical_cost:
                raise RecoveryAccountingError(
                    f"fresh calibration {section_name} row {index} has mismatched cost"
                )
    direct = calibration_report.details.get("recovery_accounting")
    if not isinstance(direct, list):
        raise RecoveryAccountingError("fresh calibration has no recovery accounting")
    if direct != per_call_records:
        raise RecoveryAccountingError(
            "fresh calibration recovery accounting does not match its per-call "
            "physical ledger evidence"
        )


def reconcile_recovery_reports(
    ledger: RecoveryBudgetLedger,
    oracle_report: _ReportLike,
    calibration_report: CalibrationReport | None = None,
    *,
    require_complete: bool = True,
    candidate_identity: str | None = None,
) -> dict[str, object]:
    """Reconcile a recovery ledger to canonical oracle/calibration evidence."""
    if calibration_report is not None:
        require_calibration_recovery_evidence(calibration_report)
    evidence = accounting_evidence_from_reports(oracle_report, calibration_report)
    if candidate_identity is not None:
        for call in _iter_physical_calls(evidence):
            observed = call.get("candidate_identity")
            if observed != candidate_identity:
                raise RecoveryAccountingError(
                    "recovery evidence candidate identity does not match campaign"
                )
    return reconcile_recovery_call_evidence(
        ledger,
        evidence,
        require_complete=require_complete,
    )


__all__ = [
    "RecoveryAccountingError",
    "RecoveryBudgetLedger",
    "accounting_evidence_from_reports",
    "reconcile_recovery_call_evidence",
    "reconcile_recovery_reports",
    "require_calibration_recovery_evidence",
    "sanitize_request_id",
]
