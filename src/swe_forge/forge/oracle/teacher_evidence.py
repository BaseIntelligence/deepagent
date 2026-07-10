"""Secret-free, non-vacuity evidence for teacher-proposed oracle gates.

The differential and alt-correct gates may only pass after a real teacher call
produces at least one proposal that was actually executed.  This module keeps the
small auditable record needed to prove that invariant without retaining prompts,
responses, source text, endpoint URLs, or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

from swe_forge.forge.teacher import Usage

TeacherCallStatus = Literal["success", "error", "not_called"]

_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "api_base",
        "base_url",
        "authorization",
        "token",
        "headers",
        "prompt",
        "messages",
        "system",
        "user",
        "response",
        "text",
        "content",
        "raw",
        "raw_response",
        "message",
        "error",
    }
)


@dataclass(frozen=True)
class TeacherGateCallEvidence:
    """One teacher interaction, retaining only safe operational metadata."""

    gate: str
    call_kind: str
    real_teacher: bool
    status: TeacherCallStatus
    response_kind: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    finish_reason: str | None = None
    requested_proposals: int = 0
    received_proposals: int = 0
    parsed_proposals: int = 0
    identical_proposals: int = 0
    invalid_proposals: int = 0
    discarded_proposals: int = 0
    execution_attempted: int = 0
    execution_completed: int = 0
    execution_errors: int = 0
    executable_proposals: int = 0
    error_type: str = ""

    def with_execution(
        self,
        *,
        attempted: int,
        completed: int,
        errors: int = 0,
        executable: int | None = None,
    ) -> TeacherGateCallEvidence:
        """Return this call with deterministic execution disposition counts."""
        return replace(
            self,
            execution_attempted=max(0, attempted),
            execution_completed=max(0, completed),
            execution_errors=max(0, errors),
            executable_proposals=max(
                0, completed if executable is None else executable
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize only secret-free model, usage, and disposition metadata."""
        return {
            "gate": self.gate,
            "call_kind": self.call_kind,
            "real_teacher": self.real_teacher,
            "status": self.status,
            "response_kind": self.response_kind,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "cost": self.cost,
            "finish_reason": self.finish_reason,
            "requested_proposals": self.requested_proposals,
            "received_proposals": self.received_proposals,
            "parsed_proposals": self.parsed_proposals,
            "identical_proposals": self.identical_proposals,
            "invalid_proposals": self.invalid_proposals,
            "discarded_proposals": self.discarded_proposals,
            "execution_attempted": self.execution_attempted,
            "execution_completed": self.execution_completed,
            "execution_errors": self.execution_errors,
            "executable_proposals": self.executable_proposals,
            "error_type": self.error_type,
        }


def evidence_calls(source: object | None) -> list[TeacherGateCallEvidence]:
    """Read typed teacher evidence from a generator without trusting raw output."""
    raw = getattr(source, "teacher_calls", ()) if source is not None else ()
    if not isinstance(raw, (list, tuple)):
        return []
    return [item for item in raw if isinstance(item, TeacherGateCallEvidence)]


def call_records(
    calls: list[TeacherGateCallEvidence] | tuple[TeacherGateCallEvidence, ...],
) -> list[dict[str, object]]:
    """Convert typed records to JSON-safe evidence."""
    return [call.to_dict() for call in calls]


def append_execution(
    calls: list[TeacherGateCallEvidence],
    *,
    call_kind: str,
    attempted: int,
    completed: int,
    errors: int = 0,
    executable: int | None = None,
) -> list[TeacherGateCallEvidence]:
    """Return records with the final matching call augmented by execution counts."""
    for index in range(len(calls) - 1, -1, -1):
        if calls[index].call_kind == call_kind:
            calls[index] = calls[index].with_execution(
                attempted=attempted,
                completed=completed,
                errors=errors,
                executable=executable,
            )
            break
    return calls


def gate_evidence(
    calls: list[TeacherGateCallEvidence] | tuple[TeacherGateCallEvidence, ...],
) -> dict[str, object]:
    """Build the report payload for one teacher-driven oracle gate."""
    return {"calls": call_records(calls)}


def teacher_gate_failure_reason(
    gate: str,
    calls: list[TeacherGateCallEvidence] | tuple[TeacherGateCallEvidence, ...],
) -> str:
    """Return the distinct fail-closed reason for an absent executable proposal."""
    proposal = next(
        (call for call in reversed(calls) if call.call_kind == "proposal"), None
    )
    if proposal is None or not proposal.real_teacher:
        return f"{gate}_no_real_teacher_proposal"
    if proposal.status == "error":
        return f"{gate}_teacher_call_failed:{proposal.error_type or 'unknown'}"
    if proposal.response_kind == "empty":
        return f"{gate}_teacher_empty_output"
    if proposal.response_kind == "unparseable":
        return f"{gate}_teacher_unparseable_output"
    if _proposal_accounting_issues(proposal.to_dict()):
        return f"{gate}_teacher_inconsistent_proposal_accounting"
    if proposal.identical_proposals:
        return f"{gate}_teacher_identical_to_gold"
    if proposal.invalid_proposals:
        return f"{gate}_teacher_invalid_proposals"
    if proposal.discarded_proposals:
        return f"{gate}_teacher_discarded_proposals"
    return f"{gate}_no_executable_teacher_proposals"


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _safe_record(record: dict[str, object]) -> bool:
    """Reject records that carry a known secret/raw-content field."""
    if _FORBIDDEN_KEYS & set(record):
        return False
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return False
    return all(
        _nonnegative_int(usage.get(field))
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    )


_COUNT_FIELDS = (
    "requested_proposals",
    "received_proposals",
    "parsed_proposals",
    "identical_proposals",
    "invalid_proposals",
    "discarded_proposals",
    "execution_attempted",
    "execution_completed",
    "execution_errors",
    "executable_proposals",
)


def _proposal_accounting_issues(record: dict[str, object]) -> list[str]:
    """Return conservation defects for one safe proposal-call record."""
    if not all(_nonnegative_int(record.get(field)) for field in _COUNT_FIELDS):
        return ["proposal or execution counts are malformed"]

    received = _count_value(record, "received_proposals")
    invalid = _count_value(record, "invalid_proposals")
    parsed = _count_value(record, "parsed_proposals")
    identical = _count_value(record, "identical_proposals")
    discarded = _count_value(record, "discarded_proposals")
    executable = _count_value(record, "executable_proposals")
    attempted = _count_value(record, "execution_attempted")
    completed = _count_value(record, "execution_completed")
    errors = _count_value(record, "execution_errors")

    issues: list[str] = []
    if received != invalid + parsed:
        issues.append("received proposals do not equal invalid plus parsed proposals")
    if parsed != identical + discarded + executable:
        issues.append(
            "parsed proposals do not equal identical plus discarded plus executable"
        )
    if attempted != completed + errors:
        issues.append(
            "execution attempted does not equal completed executions plus errors"
        )
    if executable != completed:
        issues.append("executable proposals do not equal completed executions")
    return issues


def _count_value(record: dict[str, object], field: str) -> int:
    """Return a validated non-negative count from a safe evidence record."""
    value = record[field]
    if not _nonnegative_int(value):  # pragma: no cover - guarded by the caller
        raise ValueError(f"{field} is not a non-negative count")
    assert isinstance(value, int)
    return value


def teacher_gate_evidence_issues(
    details: object,
    *,
    gates: Iterable[str] = ("differential", "alt_correct"),
) -> list[str]:
    """Return non-vacuity and hygiene defects for a passing oracle report."""
    if not isinstance(details, dict):
        return ["teacher evidence is missing"]
    requested_gates = tuple(gates)
    gate_payloads = details.get("teacher_gates")
    if not isinstance(gate_payloads, dict):
        return ["teacher evidence is missing"]

    issues: list[str] = []
    for gate in requested_gates:
        payload = gate_payloads.get(gate)
        if not isinstance(payload, dict):
            issues.append(f"{gate}: teacher evidence is missing")
            continue
        calls = payload.get("calls")
        if not isinstance(calls, list):
            issues.append(f"{gate}: teacher calls are missing")
            continue
        if not all(isinstance(call, dict) and _safe_record(call) for call in calls):
            issues.append(f"{gate}: teacher evidence is malformed or unsafe")
            continue

        proposals = [
            call
            for call in calls
            if call.get("call_kind") == "proposal" and call.get("real_teacher") is True
        ]
        if not proposals:
            issues.append(f"{gate}: no real-teacher proposal call was recorded")
            continue

        valid = []
        accounting_problems: list[str] = []
        for call in proposals:
            accounting = _proposal_accounting_issues(call)
            if accounting:
                accounting_problems.extend(accounting)
                continue
            if (
                call.get("status") == "success"
                and call.get("response_kind") == "content"
                and isinstance(call.get("model"), str)
                and bool(str(call["model"]).strip())
                and isinstance(call.get("cost"), (int, float))
                and not isinstance(call.get("cost"), bool)
                and float(call["cost"]) >= 0.0
                and int(call["parsed_proposals"]) > 0
                and int(call["execution_attempted"]) > 0
                and int(call["execution_completed"]) > 0
                and int(call["executable_proposals"]) > 0
            ):
                valid.append(call)
        if not valid:
            if accounting_problems:
                issues.append(
                    f"{gate}: teacher proposal accounting is inconsistent "
                    f"({'; '.join(sorted(set(accounting_problems)))}"
                )
                continue
            issues.append(
                f"{gate}: no successful real-teacher proposal has positive "
                "parse and execution evidence"
            )
    return issues


def aggregate_teacher_gate_usage(details: object) -> tuple[Usage, float]:
    """Sum secret-free teacher call accounting from an OracleReport's evidence."""
    if not isinstance(details, dict):
        return Usage(), 0.0
    gates = details.get("teacher_gates")
    if not isinstance(gates, dict):
        return Usage(), 0.0

    usage = Usage()
    cost = 0.0
    for payload in gates.values():
        if not isinstance(payload, dict):
            continue
        calls = payload.get("calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict) or not _safe_record(call):
                continue
            raw_usage = call["usage"]
            assert isinstance(raw_usage, dict)
            usage = usage + Usage(
                prompt_tokens=int(raw_usage["prompt_tokens"]),
                completion_tokens=int(raw_usage["completion_tokens"]),
                total_tokens=int(raw_usage["total_tokens"]),
            )
            raw_cost = call.get("cost")
            if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
                cost += float(raw_cost)
    return usage, cost


__all__ = [
    "TeacherGateCallEvidence",
    "aggregate_teacher_gate_usage",
    "append_execution",
    "call_records",
    "evidence_calls",
    "gate_evidence",
    "teacher_gate_failure_reason",
    "teacher_gate_evidence_issues",
]
