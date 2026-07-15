"""Agent-visible workspace export (no gold answer key).

VAL-EXPORT-001: agent workspace omits gold solution patch.
VAL-EXPORT-003: export fail-closed when leak scan finds gold/API keys.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_factory.export.jsonl import write_tasks_jsonl
from swe_factory.export.leak_scan import LeakScanResult, scan_export_tree
from swe_factory.schema import TaskRecord

# Paths that must never appear in agent-visible trees.
FORBIDDEN_AGENT_NAMES: frozenset[str] = frozenset(
    {
        "gold.patch",
        "patch.diff",
        "solution.patch",
        "deletion_patch.diff",
        "oracle_hidden",
        ".oracle_candidate.patch",
    }
)


class ExportError(RuntimeError):
    """Raised when export fails schema, layout, or leak checks."""


@dataclass(frozen=True, slots=True)
class ExportBundle:
    """Result of writing an export directory."""

    out_dir: Path
    tasks_jsonl: Path
    workspaces: tuple[Path, ...]
    leak_scan: LeakScanResult
    instance_ids: tuple[str, ...]


def _source_track_value(task: TaskRecord) -> str:
    track = task.source_track
    return track.value if hasattr(track, "value") else str(track)


def agent_meta_for(task: TaskRecord) -> dict[str, Any]:
    """Agent-safe metadata (never includes gold_patch)."""
    return {
        "instance_id": task.instance_id,
        "source_track": _source_track_value(task),
        "repo": task.repo,
        "base_commit": task.base_commit,
        "language": task.language,
        "fail_to_pass": list(task.fail_to_pass),
        "pass_to_pass": list(task.pass_to_pass),
        "environment": {"image_digest": task.environment.image_digest},
        "license": task.license,
        "requirements": task.requirements,
        "gold_present_in_record": False,
        "note": "agent workspace: gold omitted by design",
    }


def _copy_broken_repo(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            "node_modules",
            "gold.patch",
            "patch.diff",
            "solution.patch",
            ".oracle_candidate.patch",
        ),
    )


def export_task_workspace(
    task: TaskRecord,
    *,
    dest: Path | str,
    broken_repo: Path | str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write one agent-visible workspace without gold answer key.

    Layout::

        <dest>/
          problem_statement.md
          task_meta.agent.json   # no gold_patch
          repo/                  # optional broken tree, gold stripped
    """
    workspace = Path(dest)
    if workspace.exists():
        if not overwrite:
            raise ExportError(f"workspace already exists: {workspace}")
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    (workspace / "problem_statement.md").write_text(
        task.problem_statement.rstrip() + "\n",
        encoding="utf-8",
    )
    meta = agent_meta_for(task)
    # Double-guard: never serialize gold into agent meta
    meta.pop("gold_patch", None)
    (workspace / "task_meta.agent.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if broken_repo is not None:
        src = Path(broken_repo)
        if not src.is_dir():
            raise ExportError(f"broken_repo not found: {src}")
        _copy_broken_repo(src, workspace / "repo")

    # Fail-closed if forbidden names slipped in
    for path in workspace.rglob("*"):
        if path.is_file() and path.name in FORBIDDEN_AGENT_NAMES:
            raise ExportError(f"agent workspace contains forbidden path: {path}")

    return workspace


def write_export_bundle(
    tasks: Sequence[TaskRecord],
    out_dir: Path | str,
    *,
    broken_repos: Mapping[str, Path | str] | None = None,
    overwrite: bool = True,
    require_clean_leak_scan: bool = True,
    require_panel: bool = False,
) -> ExportBundle:
    """Write tasks.jsonl + per-task agent workspaces + leak-scan fail closed.

    Parameters
    ----------
    tasks:
        Full TaskRecords (jsonl retains gold for harness; workspaces do not).
    broken_repos:
        Optional map of instance_id → broken source tree to copy under
        ``tasks/<id>/repo``.
    require_panel:
        When True, fail closed if any task lacks panel hardness fields
        (certified keep export).
    """
    base = Path(out_dir)
    if base.exists() and overwrite:
        # Preserve nothing; export is the declaration of truth for this out_dir
        for child in list(base.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    base.mkdir(parents=True, exist_ok=True)

    if not tasks:
        raise ExportError("refusing empty export bundle")

    if require_panel:
        from swe_factory.export.jsonl import validate_export_fields
        from swe_factory.schema import SourceTrack

        for task in tasks:
            try:
                validate_export_fields(task, require_panel=True)
            except Exception as exc:  # JsonlExportError
                raise ExportError(str(exc)) from exc
            track = task.source_track
            value = track.value if hasattr(track, "value") else str(track)
            if value not in {
                SourceTrack.REAL_PR.value,
                SourceTrack.SYNTHETIC_GROUNDED.value,
            }:
                raise ExportError(
                    f"keep export fail-closed: invalid source_track {value!r} "
                    f"on {task.instance_id!r}"
                )

    repos = dict(broken_repos or {})
    workspaces: list[Path] = []
    gold_map: dict[str, str] = {}

    for task in tasks:
        ws = export_task_workspace(
            task,
            dest=base / "tasks" / task.instance_id,
            broken_repo=repos.get(task.instance_id),
            overwrite=True,
        )
        workspaces.append(ws)
        gold_map[task.instance_id] = task.gold_patch

    try:
        tasks_jsonl = write_tasks_jsonl(
            list(tasks),
            base / "tasks.jsonl",
            overwrite=True,
            require_panel=require_panel,
        )
    except Exception as exc:  # pydantic / JsonlExportError
        raise ExportError(f"tasks.jsonl write failed: {exc}") from exc

    # Optional light report for inspectability
    (base / "export_manifest.json").write_text(
        json.dumps(
            {
                "count": len(tasks),
                "instance_ids": [t.instance_id for t in tasks],
                "tasks_jsonl": str(tasks_jsonl.name),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    leak = scan_export_tree(base, gold_patches=gold_map)
    if require_clean_leak_scan and not leak.clean:
        raise ExportError("export leak scan failed: " + "; ".join(leak.findings[:5]))

    return ExportBundle(
        out_dir=base,
        tasks_jsonl=tasks_jsonl,
        workspaces=tuple(workspaces),
        leak_scan=leak,
        instance_ids=tuple(t.instance_id for t in tasks),
    )


__all__ = [
    "FORBIDDEN_AGENT_NAMES",
    "ExportBundle",
    "ExportError",
    "agent_meta_for",
    "export_task_workspace",
    "write_export_bundle",
]
