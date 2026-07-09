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
* **Crash-safe writes.** Workspaces, JSONL, and Parquet are staged as one
  generation, validated against one ID set, and made visible through one
  recoverable publication pointer. An interruption therefore exposes only the
  prior complete generation or the new complete generation, never a mixed set.
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
from pathlib import Path

from swe_forge.forge.adapters import LanguageAdapter
from swe_forge.forge.export import (
    BatchExportResult,
    ExportError,
    ExportRequest,
    TaskExportResult,
    assemble_forge_task,
    export_dataset,
    export_forge_task,
)
from swe_forge.forge.models import ExportGateError, ForgeTask
from swe_forge.forge.oracle.pipeline import ExportRefusedError
from swe_forge.forge.publication import (
    PublicationEntry,
    PublicationError,
    PublicationOutcome,
    canonical_task_payload,
    load_published_generation,
    publish_generation,
)

_KEPT_STATUSES = ("shipped", "skipped")


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
        self._requests: dict[str, ExportRequest] = {}

        existing = load_published_generation(self._out_dir)
        if existing is None:
            # A stop before the first keep still exposes one valid empty
            # generation.  The same shared publisher is used here and for every
            # subsequent checkpoint update.
            self._publish()
        else:
            for entry in existing.entries:
                self._kept[entry.index] = entry.task
                self._results[entry.index] = TaskExportResult(
                    task_id=entry.task.task_id,
                    status="skipped",
                    path=self._tasks_dir / entry.task.task_id,
                    reason="recovered from committed publication",
                )

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
        except (ExportRefusedError, ExportGateError, ExportError) as exc:
            # Fail-fast gate: a mis-routed oracle-reject / calibration-drop never
            # materializes anything (should not reach here, but re-checked).
            result = TaskExportResult(
                task_id=request.task_id or request._fallback_id(),
                status="refused",
                reason=str(exc),
            )
            self._results[index] = result
            return result

        prior_at_index = self._kept.get(index)
        if prior_at_index is not None:
            if canonical_task_payload(prior_at_index) != canonical_task_payload(task):
                return TaskExportResult(
                    task_id=task.task_id,
                    status="failed",
                    reason=f"conflicting checkpoint payload at plan index {index}",
                )
            return self._results.get(
                index,
                TaskExportResult(
                    task_id=task.task_id,
                    status="skipped",
                    path=self._tasks_dir / task.task_id,
                    reason="checkpoint retry already committed",
                ),
            )

        for existing_index, existing in self._kept.items():
            if existing.task_id != task.task_id:
                continue
            if canonical_task_payload(existing) != canonical_task_payload(task):
                return TaskExportResult(
                    task_id=task.task_id,
                    status="failed",
                    reason=(
                        f"conflicting duplicate task_id {task.task_id!r} "
                        f"against checkpoint plan index {existing_index}"
                    ),
                )
            result = TaskExportResult(
                task_id=task.task_id,
                status="deduplicated",
                path=self._tasks_dir / task.task_id,
                reason=f"identical to checkpoint plan index {existing_index}",
            )
            self._results[index] = result
            return result

        self._requests[task.task_id] = request
        pending = dict(self._kept)
        pending[index] = task
        try:
            outcomes = self._publish(pending)
        except PublicationError as exc:
            self._requests.pop(task.task_id, None)
            raise RuntimeError(f"checkpoint publication failed: {exc}") from exc

        outcome = outcomes[index]
        result = TaskExportResult(
            task_id=outcome.task_id,
            status=outcome.status,
            path=self._tasks_dir / outcome.task_id
            if outcome.status in _KEPT_STATUSES
            else None,
            reason=outcome.reason,
        )
        self._results[index] = result
        self._requests.pop(task.task_id, None)
        return result

    def _publish(
        self, pending: dict[int, ForgeTask] | None = None
    ) -> dict[int, PublicationOutcome]:
        """Commit a complete checkpoint generation through the shared publisher."""
        proposed = pending if pending is not None else self._kept
        entries = [
            PublicationEntry(index=index, task=proposed[index])
            for index in sorted(proposed)
        ]

        def _write_workspace(task: ForgeTask, stage_tasks: Path) -> TaskExportResult:
            request = self._requests.get(task.task_id)
            if request is None:
                raise PublicationError(
                    f"missing checkpoint request for newly staged task {task.task_id!r}"
                )
            return export_forge_task(
                task,
                stage_tasks,
                overwrite=False,
                adapter=self._adapter,
                broken_tree=request.broken_tree,
            )

        generation, outcomes = publish_generation(
            self._out_dir,
            entries,
            workspace_writer=_write_workspace,
            dataset_writer=export_dataset,
            overwrite=self._overwrite,
        )
        self._kept = {entry.index: entry.task for entry in generation.entries}
        return {outcome.index: outcome for outcome in outcomes}

    def finalize(self) -> BatchExportResult:
        """Return the plan-ordered batch result for the committed generation.

        Idempotent: the last record_keep call already committed the full kept set
        in plan order, so finalize never exposes a new partial artifact set.
        """
        results = [self._results[index] for index in sorted(self._results)]
        return BatchExportResult(
            out_dir=self._out_dir,
            tasks_dir=self._tasks_dir,
            jsonl_path=self._jsonl_path,
            parquet_path=self._parquet_path,
            results=results,
        )


__all__ = ["PilotCheckpoint"]
