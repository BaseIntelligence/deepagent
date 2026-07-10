"""Incremental Stage-5 checkpointing for the pilot.

Materializes each band-keep :class:`~swe_forge.forge.models.ForgeTask` the moment
it is found instead of waiting for every plan to finish, so a run stopped at any
point (SIGTERM / budget-or-time ceiling / crash) has already shipped every keep
found so far -- the machinery fix for the previously all-or-nothing Stage-5
export that ran only after the whole sweep completed.

It reuses the Stage-5 private staging path -- :func:`assemble_forge_task` (the
fail-fast export gate), :func:`_write_staged_workspace` (the non-publishing
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
from collections.abc import Iterable
from pathlib import Path

from swe_forge.forge.adapters import LanguageAdapter
from swe_forge.forge.export import (
    BatchExportResult,
    ExportError,
    ExportRequest,
    TaskExportResult,
    _write_staged_workspace,
    assemble_forge_task,
    export_dataset,
)
from swe_forge.forge.models import (
    ExportGateError,
    ForgeTask,
    InstanceGrant,
    RepoSpec,
)
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
        source_specs: Iterable[RepoSpec] = (),
    ) -> None:
        self._out_dir = Path(out_dir)
        self._tasks_dir = self._out_dir / "tasks"
        self._jsonl_path = self._out_dir / jsonl_name
        self._parquet_path = self._out_dir / parquet_name
        self._overwrite = overwrite
        self._adapter = adapter
        self._lock = asyncio.Lock()
        # Admission and persistence are deliberately separate.  A keep is
        # admitted synchronously on the event loop before any blocking copy work
        # begins, then remains in this ledger until the checkpoint writer has
        # reached a terminal result for it.  This lets a graceful shutdown close
        # admission first, then drain every accepted keep without racing the
        # source-tree cleanup in the pilot orchestrator.
        self._accepting = True
        self._accepted: dict[int, ExportRequest] = {}
        self._pending: dict[int, ExportRequest] = {}
        self._writes: dict[int, asyncio.Task[TaskExportResult]] = {}
        # Kept ForgeTasks and per-candidate export results, keyed by plan index so
        # the datasets + result ledger are always emitted in plan order regardless
        # of the (possibly concurrent) completion order.
        self._kept: dict[int, ForgeTask] = {}
        self._results: dict[int, TaskExportResult] = {}
        self._requests: dict[str, ExportRequest] = {}
        # Capacity is admitted in this checkpoint, not in concurrent candidate
        # workers. Keep the first spec for each repo as the canonical
        # SourceRegistry-backed counter and synchronize any same-id aliases so
        # manually constructed plans cannot race independent ``used`` values.
        self._capacity_specs: dict[str, RepoSpec] = {}
        self._capacity_aliases: dict[str, list[RepoSpec]] = {}
        self._capacity_grants: dict[int, InstanceGrant] = {}
        self._released_capacity: set[int] = set()

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
        for spec in source_specs:
            self._register_source(spec)

    @property
    def out_dir(self) -> Path:
        return self._out_dir

    @property
    def tasks_dir(self) -> Path:
        return self._tasks_dir

    @property
    def kept_count(self) -> int:
        return len(self._kept)

    @property
    def committed_indexes(self) -> tuple[int, ...]:
        """Plan indexes already selected by the recovered generation."""
        return tuple(sorted(self._kept))

    @property
    def accepting(self) -> bool:
        """Whether new calibrated keeps may enter this checkpoint."""
        return self._accepting

    @property
    def accepted_indexes(self) -> tuple[int, ...]:
        """All plan indexes admitted during this process, in plan order."""
        return tuple(sorted(self._accepted))

    @property
    def pending_indexes(self) -> tuple[int, ...]:
        """Accepted keeps whose checkpoint I/O has not reached a terminal result."""
        return tuple(sorted(self._pending))

    def capacity_grant(self, index: int) -> InstanceGrant | None:
        """Return the serialized source-cap decision for one qualified keep."""
        return self._capacity_grants.get(index)

    def result_for(self, index: int) -> TaskExportResult | None:
        """Return the terminal checkpoint outcome for one plan index."""
        return self._results.get(index)

    def capacity_snapshot(self) -> list[dict[str, object]]:
        """Return current per-repo capacity accounting in deterministic order."""
        return [
            {
                "repo_id": repo_id,
                "cap": spec.instance_cap,
                "used": spec.used,
                "remaining": spec.remaining,
            }
            for repo_id, spec in sorted(self._capacity_specs.items())
        ]

    def close_admission(self) -> None:
        """Stop accepting new keeps while retaining every prior admission."""
        self._accepting = False

    def was_accepted(self, index: int) -> bool:
        """Return whether a keep was admitted before checkpoint shutdown."""
        return index in self._accepted

    def _published_count(self, repo_id: str) -> int:
        """Count committed tasks for one source in the recovered generation."""
        return sum(
            1 for task in self._kept.values() if task.env_image.repo_id == repo_id
        )

    def _register_source(self, spec: RepoSpec) -> RepoSpec:
        """Restore and share capacity accounting for a source id."""
        canonical = self._capacity_specs.get(spec.repo_id)
        if canonical is None:
            recovered = self._published_count(spec.repo_id)
            if recovered > spec.instance_cap:
                raise RuntimeError(
                    f"checkpoint has {recovered} committed tasks for "
                    f"{spec.repo_id!r}, exceeding instance_cap "
                    f"{spec.instance_cap}"
                )
            if spec.used < recovered:
                spec.used = recovered
            canonical = spec
            self._capacity_specs[spec.repo_id] = canonical
            self._capacity_aliases[spec.repo_id] = [spec]
        else:
            if canonical.instance_cap != spec.instance_cap:
                raise RuntimeError(
                    f"conflicting instance_cap for source {spec.repo_id!r}: "
                    f"{canonical.instance_cap} != {spec.instance_cap}"
                )
            aliases = self._capacity_aliases[spec.repo_id]
            if all(alias is not spec for alias in aliases):
                aliases.append(spec)
            if spec.used > canonical.used:
                canonical.used = spec.used

        if canonical.used > canonical.instance_cap:
            raise RuntimeError(
                f"source {canonical.repo_id!r} has used={canonical.used} above "
                f"instance_cap={canonical.instance_cap}"
            )
        for alias in self._capacity_aliases[canonical.repo_id]:
            alias.used = canonical.used
        return canonical

    def _sync_source_usage(self, repo_id: str) -> None:
        canonical = self._capacity_specs[repo_id]
        for alias in self._capacity_aliases[repo_id]:
            alias.used = canonical.used

    def _release_capacity(self, index: int) -> None:
        """Return a one-shot grant when its Stage-5 export did not ship."""
        if index in self._released_capacity:
            return
        grant = self._capacity_grants.get(index)
        if grant is None or not grant.accepted:
            return
        spec = self._capacity_specs[grant.repo_id]
        spec.release(grant)
        self._sync_source_usage(grant.repo_id)
        self._released_capacity.add(index)

    def admit_keep(
        self,
        index: int,
        request: ExportRequest,
        *,
        source: RepoSpec | None = None,
    ) -> TaskExportResult | None:
        """Record a keep before scheduling copy/publication I/O.

        The source cap is acquired synchronously before a request enters the
        pending publication ledger, so concurrent completions cannot oversubscribe
        it and a cap-rejected keep has no workspace or dataset row. Returns a
        terminal refusal/rejection when admission is closed or capacity is
        exhausted, otherwise ``None``. Repeated calls for an already-admitted
        plan index preserve the original request, which is the only source tree
        the subsequent drain may copy.
        """
        if index in self._accepted:
            return None
        if index in self._kept:
            # A recovered plan index must pass through the normal payload
            # comparison in _record_keep_sync, but it already owns a committed
            # source slot. Queue it without taking a second RepoSpec grant.
            self._accepted[index] = request
            self._pending[index] = request
            return None
        prior = self._results.get(index)
        if prior is not None and prior.status == "cap_rejected":
            return prior
        if not self._accepting:
            return TaskExportResult(
                task_id=request.task_id or request._fallback_id(),
                status="refused",
                reason="checkpoint admission is closed",
            )
        if source is not None:
            spec = self._register_source(source)
            grant = spec.acquire()
            self._capacity_grants[index] = grant
            self._sync_source_usage(spec.repo_id)
            if not grant.accepted:
                result = TaskExportResult(
                    task_id=request.task_id or request._fallback_id(),
                    status="cap_rejected",
                    reason=grant.reason,
                )
                self._results[index] = result
                return result
        self._accepted[index] = request
        self._pending[index] = request
        return None

    async def record_keep(
        self,
        index: int,
        request: ExportRequest,
        *,
        source: RepoSpec | None = None,
    ) -> TaskExportResult:
        """Admit and checkpoint one keep.

        Admission happens before awaiting I/O.  Once admitted, a cancellation of
        the caller cannot make the request disappear: :meth:`drain` will finish it
        in deterministic plan order before the pilot cleans its source trees.
        """
        refused = self.admit_keep(index, request, source=source)
        if refused is not None:
            return refused
        result = await self.drain(indexes=(index,))
        assert isinstance(result, TaskExportResult)
        return result

    async def drain(
        self,
        *,
        indexes: tuple[int, ...] | None = None,
        continue_on_error: bool = False,
    ) -> TaskExportResult | dict[int, TaskExportResult]:
        """Flush accepted keeps in deterministic plan-index order.

        The lock serializes the publication generation.  ``asyncio.shield`` keeps
        the thread-backed write alive if a caller is cancelled; a later drain,
        normally from ``run_pilot``'s ``finally``, waits for the same write before
        source cleanup.  All accepted requests are therefore either committed or
        have a terminal refusal/failure result before their broken tree can vanish.
        A shutdown drain uses ``continue_on_error`` so a failure at one index cannot
        strand later admitted keeps behind it.
        """
        async with self._lock:
            targets = (
                tuple(sorted(self._pending))
                if indexes is None
                else tuple(sorted(index for index in indexes if index in self._pending))
            )
            for pending_index in targets:
                request = self._pending[pending_index]
                write = self._writes.get(pending_index)
                if write is None:
                    write = asyncio.create_task(
                        asyncio.to_thread(
                            self._record_keep_sync, pending_index, request
                        )
                    )
                    self._writes[pending_index] = write
                try:
                    result = await asyncio.shield(write)
                except asyncio.CancelledError:
                    # Do not orphan the worker-thread publication.  The task owns
                    # no source-tree cleanup and remains represented in
                    # ``_pending`` until the shutdown drain observes its result.
                    def _settle(
                        task: asyncio.Future[TaskExportResult], *, plan_index: int
                    ) -> None:
                        if task.cancelled():
                            return
                        try:
                            settled = task.result()
                        except Exception:
                            return
                        self._results[plan_index] = settled
                        if settled.status not in _KEPT_STATUSES:
                            self._release_capacity(plan_index)
                        self._pending.pop(plan_index, None)
                        self._writes.pop(plan_index, None)

                    def _on_done(task: asyncio.Future[TaskExportResult]) -> None:
                        _settle(task, plan_index=pending_index)

                    write.add_done_callback(_on_done)
                    raise
                except Exception as exc:
                    self._requests.pop(request.task_id or request._fallback_id(), None)
                    result = TaskExportResult(
                        task_id=request.task_id or request._fallback_id(),
                        status="failed",
                        reason=f"checkpoint publication failed: {exc}",
                    )
                    self._release_capacity(pending_index)
                    self._results[pending_index] = result
                    self._pending.pop(pending_index, None)
                    self._writes.pop(pending_index, None)
                    if not continue_on_error:
                        raise
                    continue
                self._results[pending_index] = result
                if result.status not in _KEPT_STATUSES:
                    self._release_capacity(pending_index)
                self._pending.pop(pending_index, None)
                self._writes.pop(pending_index, None)

            if indexes is not None:
                index = indexes[0]
                return self._results.get(
                    index,
                    TaskExportResult(
                        task_id=self._accepted[index].task_id
                        or self._accepted[index]._fallback_id(),
                        status="failed",
                        reason="checkpoint write did not reach a terminal result",
                    ),
                )
            return {index: self._results[index] for index in sorted(self._results)}

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
            return _write_staged_workspace(
                task,
                stage_tasks / task.task_id,
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
