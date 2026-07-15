"""tasks.jsonl writer for certified / exportable TaskRecord lines.

VAL-EXPORT-002: each line parses and includes required fields for shipped keeps
(instance_id, source_track, repo, base_commit, language, fail_to_pass,
pass_to_pass, environment image digest, panel hardness when present).

The jsonl line retains ``gold_patch`` for internal oracle/harness scoring.
Agent-visible workspaces never receive gold (see :mod:`workspace`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from swe_factory.schema import TaskRecord


class JsonlExportError(ValueError):
    """Raised when tasks.jsonl cannot be written or is incomplete."""


REQUIRED_EXPORT_KEYS: frozenset[str] = frozenset(
    {
        "instance_id",
        "source_track",
        "repo",
        "base_commit",
        "language",
        "fail_to_pass",
        "pass_to_pass",
        "environment",
    }
)


def _as_record(task: TaskRecord | dict[str, Any]) -> TaskRecord:
    if isinstance(task, TaskRecord):
        return task
    return TaskRecord.model_validate(task)


def validate_export_fields(task: TaskRecord, *, require_panel: bool = False) -> None:
    """Fail-closed check that required export keys are present and non-empty.

    When ``require_panel`` is True (certified keep export), panel hardness fields
    must be present: pass_at_k, discrimination, and at least one model rate.
    """
    payload = json.loads(task.model_dump_json())
    missing = sorted(k for k in REQUIRED_EXPORT_KEYS if k not in payload)
    if missing:
        raise JsonlExportError(f"task missing required export fields: {missing}")
    env = payload.get("environment") or {}
    if not str(env.get("image_digest") or "").strip():
        raise JsonlExportError(f"task {task.instance_id!r} missing environment.image_digest")
    if not payload.get("fail_to_pass"):
        raise JsonlExportError(f"task {task.instance_id!r} missing fail_to_pass")
    if not str(payload.get("base_commit") or "").strip():
        raise JsonlExportError(f"task {task.instance_id!r} missing base_commit")
    if not str(payload.get("source_track") or "").strip():
        raise JsonlExportError(f"task {task.instance_id!r} missing source_track")
    if require_panel:
        panel = payload.get("panel")
        if not isinstance(panel, dict) or not panel:
            raise JsonlExportError(
                f"keep export fail-closed: task {task.instance_id!r} lacks panel hardness"
            )
        if panel.get("pass_at_k") is None:
            raise JsonlExportError(
                f"keep export fail-closed: task {task.instance_id!r} panel.pass_at_k missing"
            )
        if panel.get("discrimination") is None:
            raise JsonlExportError(
                f"keep export fail-closed: task {task.instance_id!r} panel.discrimination missing"
            )
        if (
            panel.get("grok_4_5") is None
            and panel.get("kimi_k2_6") is None
            and panel.get("opus_4_8") is None
        ):
            raise JsonlExportError(
                f"keep export fail-closed: task {task.instance_id!r} missing panel model rates"
            )


def record_to_jsonl_line(task: TaskRecord, *, require_panel: bool = False) -> str:
    """Serialize one TaskRecord to a single JSONL line (compact, ordered keys)."""
    validate_export_fields(task, require_panel=require_panel)
    # model_dump_json uses enum values when use_enum_values or serialize mode
    return task.model_dump_json()


def write_tasks_jsonl(
    tasks: Sequence[TaskRecord | dict[str, Any]] | Iterable[TaskRecord | dict[str, Any]],
    path: Path | str,
    *,
    overwrite: bool = True,
    require_panel: bool = False,
) -> Path:
    """Write ``tasks.jsonl`` with one validated TaskRecord per line.

    Returns the path written. Empty input yields a valid empty file.
    When ``require_panel`` is True, every task must carry panel hardness
    (certified keep fail-closed path).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise JsonlExportError(f"refusing to overwrite existing {out}")

    records = [_as_record(t) for t in tasks]
    lines: list[str] = []
    for record in records:
        lines.append(record_to_jsonl_line(record, require_panel=require_panel))

    body = "\n".join(lines) + ("\n" if lines else "")
    out.write_text(body, encoding="utf-8")
    return out


def read_tasks_jsonl(path: Path | str) -> list[TaskRecord]:
    """Parse tasks.jsonl into validated TaskRecord objects."""
    text = Path(path).read_text(encoding="utf-8")
    records: list[TaskRecord] = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(TaskRecord.model_validate_json(stripped))
        except Exception as exc:  # pydantic ValidationError + JSON
            raise JsonlExportError(f"invalid tasks.jsonl line {i}: {exc}") from exc
    return records


__all__ = [
    "REQUIRED_EXPORT_KEYS",
    "JsonlExportError",
    "read_tasks_jsonl",
    "record_to_jsonl_line",
    "validate_export_fields",
    "write_tasks_jsonl",
]
