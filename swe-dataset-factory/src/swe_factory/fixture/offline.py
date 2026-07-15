"""Offline fixture pipeline: schema write + stub gates, zero LLM/provider calls.

VAL-CROSS-001 — Fixture repo produces a gate-passing task offline proving
oracle + export wiring without OpenRouter.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from swe_factory.oracle.gates import GateResult, run_stub_gates
from swe_factory.schema import TaskRecord

FIXTURE_INSTANCE_ID = "fixture__tiny_offline__gate_demo"
_FIXTURE_REL = Path("fixtures") / "tiny_offline"


class OfflineFixtureError(RuntimeError):
    """Raised when the offline fixture tree is missing or invalid."""


@dataclass(frozen=True, slots=True)
class OfflinePipelineResult:
    """Result of the offline gate-demo pipeline."""

    instance_id: str
    task: TaskRecord
    gates: GateResult
    out_dir: Path | None
    provider_calls: int = 0
    tasks_jsonl: Path | None = None
    gate_audit: Path | None = None
    workspace: Path | None = None


def _package_root() -> Path:
    """Repo root containing fixtures/ (editable install layout)."""
    # src/swe_factory/fixture/offline.py -> parents[3] == package root
    return Path(__file__).resolve().parents[3]


def default_fixture_root() -> Path:
    """Locate fixtures/tiny_offline relative to package install or cwd layout."""
    candidates = [
        _package_root() / _FIXTURE_REL,
        Path.cwd() / _FIXTURE_REL,
        Path("/projects/swe-dataset-factory") / _FIXTURE_REL,
    ]
    # Importlib resources fallback if package data were ever packaged
    try:
        pkg = resources.files("swe_factory")
        # not required for M1; fixtures live at repo root
        del pkg
    except (ModuleNotFoundError, TypeError, AttributeError):
        pass
    for path in candidates:
        if path.is_dir() and (path / "task_meta.json").is_file():
            return path
    # Prefer first candidate for error messaging
    return candidates[0]


def _load_meta(fixture_root: Path) -> dict[str, Any]:
    meta_path = fixture_root / "task_meta.json"
    if not meta_path.is_file():
        raise OfflineFixtureError(f"fixture meta missing: {meta_path}")
    raw: object = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OfflineFixtureError(f"fixture meta must be a JSON object: {meta_path}")
    return dict(raw)


def _load_gold(fixture_root: Path) -> str:
    gold_path = fixture_root / "gold.patch"
    if not gold_path.is_file():
        raise OfflineFixtureError(f"fixture gold.patch missing: {gold_path}")
    return gold_path.read_text(encoding="utf-8")


def build_fixture_task(fixture_root: Path | None = None) -> TaskRecord:
    """Build a TaskRecord from the offline fixture tree (schema-validated)."""
    root = fixture_root or default_fixture_root()
    if not root.is_dir():
        raise OfflineFixtureError(f"fixture root not found: {root}")
    meta = _load_meta(root)
    gold = _load_gold(root)
    payload = {
        "instance_id": meta.get("instance_id", FIXTURE_INSTANCE_ID),
        "source_track": meta["source_track"],
        "repo": meta["repo"],
        "base_commit": meta["base_commit"],
        "language": meta["language"],
        "problem_statement": meta["problem_statement"],
        "fail_to_pass": meta["fail_to_pass"],
        "pass_to_pass": meta.get("pass_to_pass", []),
        "gold_patch": gold,
        "environment": meta["environment"],
        "license": meta["license"],
        "requirements": meta.get("requirements"),
        "created_at": datetime.now(UTC),
    }
    return TaskRecord.model_validate(payload)


def export_offline_artifact(
    task: TaskRecord,
    gates: GateResult,
    out_dir: Path,
    *,
    fixture_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write tasks.jsonl, gate_audit.jsonl, and agent-visible workspace (no gold)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / "tasks.jsonl"
    audit_path = out_dir / "gate_audit.jsonl"
    workspace = out_dir / "tasks" / task.instance_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Attach gate proof before colonization
    task_with_proof = task.model_copy(update={"gate_proof": gates.to_gate_proof()})
    line = task_with_proof.model_dump_json()
    tasks_path.write_text(line + "\n", encoding="utf-8")
    audit_path.write_text(
        json.dumps(gates.to_audit_row(task.instance_id), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Agent-visible bits only (VAL-EXPORT-001 style for fixture demo)
    (workspace / "problem_statement.md").write_text(
        task.problem_statement.rstrip() + "\n",
        encoding="utf-8",
    )
    agent_meta = {
        "instance_id": task.instance_id,
        "source_track": task.source_track.value
        if hasattr(task.source_track, "value")
        else str(task.source_track),
        "repo": task.repo,
        "base_commit": task.base_commit,
        "language": task.language,
        "fail_to_pass": task.fail_to_pass,
        "pass_to_pass": task.pass_to_pass,
        "environment": {"image_digest": task.environment.image_digest},
        "license": task.license,
        # Explicitly omit gold_patch
        "gold_present_in_record": False,
        "note": "agent workspace: gold omitted by design",
    }
    (workspace / "task_meta.agent.json").write_text(
        json.dumps(agent_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Copy broken agent tree from fixture if present (still no gold)
    if fixture_root is not None:
        repo_src = fixture_root / "repo"
        if repo_src.is_dir():
            dest = workspace / "repo"
            shutil.copytree(repo_src, dest, dirs_exist_ok=True)

    report = out_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Offline fixture gate-demo report",
                "",
                f"- instance_id: `{task.instance_id}`",
                f"- disposition: `{'accept' if gates.passed else 'reject'}`",
                f"- mode: `{gates.mode}`",
                "- provider_calls: `0`",
                f"- files_touched: `{gates.files_touched}`",
                f"- multi_file: `{gates.multi_file}`",
                f"- reason_codes: `{', '.join(gates.reason_codes)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tasks_path, audit_path, workspace


def run_offline_fixture_pipeline(
    *,
    out_dir: Path | str,
    fixture_root: Path | str | None = None,
    dry_write: bool = False,
) -> OfflinePipelineResult:
    """End-to-end offline path: load fixture → schema → stub gates → export.

    Never contacts OpenRouter or any LLM provider. ``provider_calls`` is always 0.
    When ``dry_write`` is True, skip disk export (unit tests of gates/schema only).
    """
    root = Path(fixture_root) if fixture_root is not None else default_fixture_root()
    if not root.is_dir():
        raise OfflineFixtureError(f"fixture root not found: {root}")

    task = build_fixture_task(root)
    gates = run_stub_gates(task)
    # Always stamp gate proof on the in-memory task for callers
    task = task.model_copy(update={"gate_proof": gates.to_gate_proof()})

    out_path = Path(out_dir)
    tasks_jsonl: Path | None = None
    gate_audit: Path | None = None
    workspace: Path | None = None
    if not dry_write:
        if not gates.passed:
            raise OfflineFixtureError(
                f"stub gates rejected fixture {task.instance_id}: {gates.reason_codes}"
            )
        tasks_jsonl, gate_audit, workspace = export_offline_artifact(
            task, gates, out_path, fixture_root=root
        )

    return OfflinePipelineResult(
        instance_id=task.instance_id,
        task=task,
        gates=gates,
        out_dir=None if dry_write else out_path,
        provider_calls=0,
        tasks_jsonl=tasks_jsonl,
        gate_audit=gate_audit,
        workspace=workspace,
    )


__all__ = [
    "FIXTURE_INSTANCE_ID",
    "OfflineFixtureError",
    "OfflinePipelineResult",
    "build_fixture_task",
    "default_fixture_root",
    "export_offline_artifact",
    "run_offline_fixture_pipeline",
]
