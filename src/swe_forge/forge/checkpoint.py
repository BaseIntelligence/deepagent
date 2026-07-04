"""Incremental Stage-5 checkpointing for the pilot.

Materializes each band-keep :class:`~swe_forge.forge.models.ForgeTask` the moment
it is found instead of waiting for every plan to finish, so a run stopped at any
point (SIGTERM / budget-or-time ceiling / crash) has already shipped every keep
found so far -- the machinery fix for the previously all-or-nothing Stage-5
export that ran only after the whole sweep completed.

It REUSES the Stage-5 export path unchanged -- :func:`assemble_forge_task` (the
fail-fast export gate), :func:`export_forge_task` (the atomic temp-then-rename
workspace writer), and :func:`export_dataset` (jsonl/parquet regeneration) -- and
layers three whole-pipeline guarantees on top:

* **Incremental materialization.** :meth:`PilotCheckpoint.record_keep` ships one
  keep's ``tasks/<id>/`` workspace + jsonl/parquet row + provenance immediately,
  so a stopped run keeps everything shipped so far rather than 0.
* **Crash-safe writes.** The datasets are regenerated to sibling temp files and
  ``os.replace``-d into place, so a mid-write interruption never leaves a corrupt
  jsonl/parquet -- only the previous valid file or the new valid file is ever
  observed. Workspace writes inherit :func:`export_forge_task`'s own
  temp-then-rename atomicity (a failed mid-write leaves no partial ``tasks/<id>/``).
* **Byte-identical final artifact.** The datasets are always regenerated from the
  FULL kept set in PLAN ORDER (keyed by candidate index, not completion order),
  so a completed incremental run produces a dataset byte-identical to a single
  all-at-once export of the same kept set. Checkpointing changes only WHEN a keep
  is written, never WHAT ships.

Only oracle-pass AND band-keep candidates ever reach :meth:`record_keep` (the
orchestrator gates that), and the export layer re-checks the same gate fail-fast,
so a rejection/drop never materializes a workspace or dataset row. A keep that is
refused at write time (e.g. a planted leak) is recorded as ``refused`` and left
out of the datasets, so ``exported < calibration_keep`` surfaces the problem
instead of shipping it.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from swe_forge.forge.adapters import LanguageAdapter
from swe_forge.forge.export import (
    BatchExportResult,
    ExportRequest,
    TaskExportResult,
    assemble_forge_task,
    export_dataset,
    export_forge_task,
)
from swe_forge.forge.models import ExportGateError, ForgeTask
from swe_forge.forge.oracle.pipeline import ExportRefusedError

_KEPT_STATUSES = ("shipped", "skipped")


def _atomic_write_dataset(
    tasks: list[ForgeTask], jsonl_path: Path, parquet_path: Path
) -> None:
    """Regenerate the jsonl + parquet datasets crash-safely (temp-then-rename).

    Reuses :func:`export_dataset` to serialize the FULL kept set into sibling temp
    files, then ``os.replace``-s each into place (atomic on POSIX). A failure
    while serializing removes the temp files and leaves BOTH originals untouched,
    so a mid-write interruption never corrupts an already-shipped dataset.
    """
    suffix = f".ckpt-{uuid.uuid4().hex[:8]}.tmp"
    jsonl_tmp = jsonl_path.with_name(jsonl_path.name + suffix)
    parquet_tmp = parquet_path.with_name(parquet_path.name + suffix)
    try:
        export_dataset(tasks, jsonl_tmp, parquet_tmp)
    except BaseException:
        jsonl_tmp.unlink(missing_ok=True)
        parquet_tmp.unlink(missing_ok=True)
        raise
    os.replace(jsonl_tmp, jsonl_path)
    os.replace(parquet_tmp, parquet_path)


class PilotCheckpoint:
    """Ships each band-keep incrementally, reusing the Stage-5 export path.

    The orchestrator hands one keep at a time to :meth:`record_keep` as its
    candidate finishes processing (keyed by the candidate's plan index). Each keep
    is assembled through the fail-fast gate, its workspace is written atomically,
    and the datasets are regenerated from the full kept set in plan order -- so a
    run stopped at any point has already shipped every keep found so far and the
    completed dataset is byte-identical to a one-shot export.
    """

    def __init__(
        self,
        out_dir: Path | str,
        *,
        overwrite: bool = True,
        adapter: LanguageAdapter | None = None,
        jsonl_name: str = "dataset.jsonl",
        parquet_name: str = "dataset.parquet",
    ) -> None:
        self._out_dir = Path(out_dir)
        self._tasks_dir = self._out_dir / "tasks"
        self._jsonl_path = self._out_dir / jsonl_name
        self._parquet_path = self._out_dir / parquet_name
        self._overwrite = overwrite
        self._adapter = adapter
        self._lock = asyncio.Lock()
        # Kept ForgeTasks and per-candidate export results, keyed by plan index so
        # the datasets + result ledger are always emitted in plan order regardless
        # of the (possibly concurrent) completion order.
        self._kept: dict[int, ForgeTask] = {}
        self._results: dict[int, TaskExportResult] = {}

        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        # A stop before the first keep still leaves valid empty artifacts.
        self._write_dataset()

    @property
    def out_dir(self) -> Path:
        return self._out_dir

    @property
    def tasks_dir(self) -> Path:
        return self._tasks_dir

    @property
    def kept_count(self) -> int:
        return len(self._kept)

    async def record_keep(self, index: int, request: ExportRequest) -> TaskExportResult:
        """Checkpoint one keep now (assemble -> write workspace -> update datasets).

        Serialized by an ``asyncio.Lock`` so concurrent candidates never race a
        dataset rewrite; the blocking file I/O runs in a worker thread so the event
        loop keeps driving the other in-flight candidates.
        """
        async with self._lock:
            return await asyncio.to_thread(self._record_keep_sync, index, request)

    def _record_keep_sync(self, index: int, request: ExportRequest) -> TaskExportResult:
        try:
            task = assemble_forge_task(
                candidate=request.candidate,
                spec=request.spec,
                oracle_report=request.oracle_report,
                calibration_report=request.calibration_report,
                env_image=request.env_image,
                repo_url=request.repo_url,
                base_commit=request.base_commit,
                repo=request.repo,
                task_id=request.task_id,
                adapter=self._adapter,
            )
        except (ExportRefusedError, ExportGateError) as exc:
            # Fail-fast gate: a mis-routed oracle-reject / calibration-drop never
            # materializes anything (should not reach here, but re-checked).
            result = TaskExportResult(
                task_id=request.task_id or request._fallback_id(),
                status="refused",
                reason=str(exc),
            )
            self._results[index] = result
            return result

        result = export_forge_task(
            task,
            self._tasks_dir,
            overwrite=self._overwrite,
            adapter=self._adapter,
            broken_tree=request.broken_tree,
        )
        self._results[index] = result
        if result.status in _KEPT_STATUSES:
            self._kept[index] = task
            self._write_dataset()
        return result

    def _write_dataset(self) -> None:
        tasks = [self._kept[index] for index in sorted(self._kept)]
        _atomic_write_dataset(tasks, self._jsonl_path, self._parquet_path)

    def finalize(self) -> BatchExportResult:
        """Regenerate the plan-ordered datasets and return the batch result.

        Idempotent: rewrites the same bytes the last incremental write produced
        (the full kept set in plan order), so a completed run's dataset is
        byte-identical to a single all-at-once export. Safe to call once on
        completion; the incremental writes already keep the on-disk dataset valid
        if the run is interrupted before finalize.
        """
        self._write_dataset()
        results = [self._results[index] for index in sorted(self._results)]
        return BatchExportResult(
            out_dir=self._out_dir,
            tasks_dir=self._tasks_dir,
            jsonl_path=self._jsonl_path,
            parquet_path=self._parquet_path,
            results=results,
        )


__all__ = ["PilotCheckpoint"]
