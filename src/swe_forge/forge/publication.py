"""Recoverable all-artifact publication for Stage 5 exports.

The public ``tasks/``, JSONL, and Parquet paths are stable symlinks through one
``current`` pointer.  A new generation is built and validated entirely beneath a
private staging directory, renamed into the immutable generations store, then
made visible by replacing that single pointer.  Readers therefore resolve either
the previous complete generation or the next complete generation, never a
mixture of workspaces and datasets.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fcntl

from swe_forge.export.jsonl import import_jsonl
from swe_forge.export.parquet import import_parquet
from swe_forge.forge.models import ForgeTask, OracleReport
from swe_forge.forge.oracle.pipeline import (
    ExportRefusedError,
    ensure_oracle_exportable,
    parse_protected_alt_correct_audit,
)

if TYPE_CHECKING:
    from swe_forge.forge.export import TaskExportResult


_STORE_DIR = ".forge-publications"
_GENERATIONS_DIR = "generations"
_CURRENT_LINK = "current"
_MANIFEST_NAME = "manifest.json"
_SCHEMA_VERSION = 1
_PROTECTED_AUDIT_DIR = ".protected-audit"
_ALT_CORRECT_AUDIT_DIR = "alt-correct"
_TEACHER_RECEIPTS_DIR = "teacher-transport-receipts"


class PublicationError(RuntimeError):
    """Raised when a generation cannot be safely staged or published."""


@dataclass(frozen=True)
class PublicationEntry:
    """One logical keep in a generation, keyed by its deterministic plan index."""

    index: int
    task: ForgeTask


@dataclass(frozen=True)
class PublicationOutcome:
    """The workspace result for one entry attempted during a generation build."""

    index: int
    task_id: str
    status: str
    reason: str = ""
    leak_findings: tuple[str, ...] = ()

    @property
    def kept(self) -> bool:
        return self.status in ("shipped", "skipped")


@dataclass(frozen=True)
class PublishedGeneration:
    """An immutable, validated generation selected by the current pointer."""

    generation_id: str
    root: Path
    tasks_dir: Path
    jsonl_path: Path
    parquet_path: Path
    entries: tuple[PublicationEntry, ...]


WorkspaceWriter = Callable[[ForgeTask, Path], "TaskExportResult"]
DatasetWriter = Callable[[Sequence[ForgeTask], Path, Path], int]
GenerationMetadataWriter = Callable[[Path, Sequence[PublicationEntry]], None]


def _store_root(out_dir: Path) -> Path:
    return out_dir / _STORE_DIR


def _current_path(out_dir: Path) -> Path:
    return _store_root(out_dir) / _CURRENT_LINK


def _generation_dir(out_dir: Path, generation_id: str) -> Path:
    return _store_root(out_dir) / _GENERATIONS_DIR / generation_id


@contextmanager
def _publication_lock(out_dir: Path):
    """Serialize pointer checks and replacement across all local publishers."""
    store = _store_root(out_dir)
    store.mkdir(parents=True, exist_ok=True)
    lock_path = store / ".publication.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _canonical_task_payload(task: ForgeTask) -> str:
    """Serialize a task for duplicate comparison, ignoring generated timestamps."""

    def _without_timestamps(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: _without_timestamps(item)
                for key, item in value.items()
                if key not in {"created_at", "call_id", "receipt_commitment"}
            }
        if isinstance(value, list):
            return [_without_timestamps(item) for item in value]
        return value

    return json.dumps(
        _without_timestamps(task.to_dict()),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_task_payload(task: ForgeTask) -> str:
    """Public duplicate-comparison helper shared by batch and checkpoint callers."""
    return _canonical_task_payload(task)


def _fsync_path(path: Path) -> None:
    """Best-effort fsync for a file or directory before switching ``current``."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Flush staged files and directories so a committed generation is durable."""
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        _fsync_path(path)
    _fsync_path(root)


def _symlink(target: str, path: Path) -> None:
    """Create a relative symlink, rejecting unmanaged legacy output paths."""
    if path.is_symlink():
        if os.readlink(path) != target:
            raise PublicationError(
                f"refusing to replace unmanaged publication link {path}"
            )
        return
    if path.exists():
        raise PublicationError(
            f"refusing to replace existing unmanaged publication artifact {path}"
        )
    os.symlink(target, path)


def _ensure_public_facade(out_dir: Path, extra_artifacts: Sequence[str] = ()) -> None:
    """Create stable legacy paths that all resolve through the one current pointer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    store = _store_root(out_dir)
    (store / _GENERATIONS_DIR).mkdir(parents=True, exist_ok=True)
    _symlink(f"{_STORE_DIR}/{_CURRENT_LINK}/tasks", out_dir / "tasks")
    _symlink(f"{_STORE_DIR}/{_CURRENT_LINK}/dataset.jsonl", out_dir / "dataset.jsonl")
    _symlink(
        f"{_STORE_DIR}/{_CURRENT_LINK}/dataset.parquet",
        out_dir / "dataset.parquet",
    )
    for name in extra_artifacts:
        if (
            not name
            or "/" in name
            or "\\" in name
            or name in {"tasks", "dataset.jsonl", "dataset.parquet"}
        ):
            raise PublicationError(f"invalid public publication artifact {name!r}")
        _symlink(f"{_STORE_DIR}/{_CURRENT_LINK}/{name}", out_dir / name)
    _fsync_path(store / _GENERATIONS_DIR)
    _fsync_path(store)
    _fsync_path(out_dir)


def _read_manifest(root: Path) -> PublishedGeneration:
    manifest_path = root / _MANIFEST_NAME
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(
            f"invalid publication manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA_VERSION:
        raise PublicationError(f"unsupported publication manifest {manifest_path}")
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        raise PublicationError(f"publication manifest {manifest_path} has no entries")

    entries: list[PublicationEntry] = []
    seen_indexes: set[int] = set()
    seen_ids: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict) or not isinstance(raw.get("task"), dict):
            raise PublicationError(
                f"publication manifest {manifest_path} has malformed entry"
            )
        index = raw.get("index")
        if not isinstance(index, int) or index in seen_indexes:
            raise PublicationError(
                f"publication manifest {manifest_path} has duplicate index"
            )
        task = _rehydrate_protected_teacher_receipts(
            root,
            _rehydrate_protected_alt_correct_audit(
                root, ForgeTask.from_dict(raw["task"])
            ),
        )
        try:
            ensure_oracle_exportable(
                task.oracle_report,
                candidate=task.candidate,
                calibration_kept=task.calibration_report.is_keep,
            )
        except ExportRefusedError as exc:
            prefix = (
                "protected alt-correct audit is invalid"
                if "alt_correct:" in str(exc)
                else "protected teacher receipts are invalid"
                if "teacher transport" in str(exc)
                else "protected oracle evidence is invalid"
            )
            raise PublicationError(f"{prefix} for {task.task_id!r}: {exc}") from exc
        if task.task_id in seen_ids:
            raise PublicationError(
                f"publication manifest {manifest_path} has duplicate task id"
            )
        seen_indexes.add(index)
        seen_ids.add(task.task_id)
        entries.append(PublicationEntry(index=index, task=task))

    generation_id = data.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise PublicationError(
            f"publication manifest {manifest_path} has no generation id"
        )
    return PublishedGeneration(
        generation_id=generation_id,
        root=root,
        tasks_dir=root / "tasks",
        jsonl_path=root / "dataset.jsonl",
        parquet_path=root / "dataset.parquet",
        entries=tuple(sorted(entries, key=lambda entry: entry.index)),
    )


def load_published_generation(out_dir: Path | str) -> PublishedGeneration | None:
    """Load only the generation selected by ``current``; staging is never visible."""
    out_path = Path(out_dir)
    current = _current_path(out_path)
    if not current.is_symlink():
        return None
    try:
        root = current.resolve(strict=True)
    except OSError as exc:
        raise PublicationError(
            f"current publication pointer is broken: {current}"
        ) from exc
    generation = _read_manifest(root)
    _validate_staged_generation(
        generation.root,
        [entry.task.task_id for entry in generation.entries],
    )
    return generation


def _validate_staged_generation(stage: Path, expected_ids: Sequence[str]) -> None:
    """Fail closed unless all three artifact surfaces expose the same unique IDs."""
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise PublicationError("generation contains duplicate task ids")

    tasks_dir = stage / "tasks"
    workspace_ids = sorted(
        child.name
        for child in tasks_dir.iterdir()
        if child.is_dir() and (child / "workspace.yaml").is_file()
    )
    if len(workspace_ids) != len(set(workspace_ids)):
        raise PublicationError("generation has duplicate workspace ids")

    try:
        jsonl_ids = [task.id for task in import_jsonl(stage / "dataset.jsonl")]
        parquet_ids = [
            str(row["id"]) for row in import_parquet(stage / "dataset.parquet")
        ]
    except Exception as exc:  # noqa: BLE001 - malformed staged data must never publish
        raise PublicationError(f"generation datasets are unreadable: {exc}") from exc

    if len(jsonl_ids) != len(set(jsonl_ids)):
        raise PublicationError("generation JSONL contains duplicate task ids")
    if len(parquet_ids) != len(set(parquet_ids)):
        raise PublicationError("generation Parquet contains duplicate task ids")
    expected_set = set(expected)
    if set(workspace_ids) != expected_set:
        raise PublicationError("generation workspace ids do not match manifest ids")
    if set(jsonl_ids) != expected_set:
        raise PublicationError("generation JSONL ids do not match manifest ids")
    if set(parquet_ids) != expected_set:
        raise PublicationError("generation Parquet ids do not match manifest ids")


def _write_manifest(
    stage: Path, generation_id: str, entries: Sequence[PublicationEntry]
) -> None:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "generation_id": generation_id,
        "task_ids": [entry.task.task_id for entry in entries],
        "entries": [
            {"index": entry.index, "task": entry.task.to_dict()} for entry in entries
        ],
    }
    (stage / _MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_workspace(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise PublicationError(f"published workspace disappeared before copy: {source}")
    shutil.copytree(source, destination)


def protected_alt_correct_audit_path(root: Path | str, task_id: str) -> Path:
    """Return the non-public location for one task's raw alt-correct audit."""
    return (
        Path(root) / _PROTECTED_AUDIT_DIR / _ALT_CORRECT_AUDIT_DIR / f"{task_id}.json"
    )


def protected_teacher_receipts_path(root: Path | str, task_id: str) -> Path:
    """Return the non-public source-free receipt sidecar for one task."""
    return Path(root) / _PROTECTED_AUDIT_DIR / _TEACHER_RECEIPTS_DIR / f"{task_id}.json"


def _requires_protected_alt_correct_audit(task: ForgeTask) -> bool:
    """Whether this report was emitted by hardened alt-correct validation."""
    return (
        task.oracle_report.alt_correct_accepted
        and "alt_correct" in task.oracle_report.details
    )


def _rehydrate_protected_alt_correct_audit(root: Path, task: ForgeTask) -> ForgeTask:
    """Load and validate a private audit for a hardened public task.

    The public manifest deliberately excludes raw alternative implementations.
    This privileged publication-store loader restores that evidence only in
    memory, then reruns the normal export gate so a missing, malformed, or
    mismatched sidecar cannot flow into a successor generation.
    """
    if not _requires_protected_alt_correct_audit(task):
        # Pre-hardening reports are preserved as read-only recertification
        # inputs. They remain non-exportable because ensure_oracle_exportable()
        # rejects their missing hardened evidence at every write boundary.
        return task

    path = protected_alt_correct_audit_path(root, task.task_id)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PublicationError(
            f"protected alt-correct audit is missing for {task.task_id!r}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise PublicationError(
            f"protected alt-correct audit is not a regular private file for "
            f"{task.task_id!r}"
        )
    if metadata.st_mode & 0o077:
        raise PublicationError(
            f"protected alt-correct audit has unsafe permissions for {task.task_id!r}"
        )
    try:
        payload = parse_protected_alt_correct_audit(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise PublicationError(
            f"protected alt-correct audit is unreadable for {task.task_id!r}"
        ) from exc
    try:
        report = OracleReport.from_protected_dict(
            {
                **task.oracle_report.to_dict(),
                "protected_alt_correct_audit": payload,
            }
        )
    except (ValueError, TypeError) as exc:
        raise PublicationError(
            f"protected alt-correct audit is invalid for {task.task_id!r}: {exc}"
        ) from exc
    return replace(task, oracle_report=report)


def _requires_protected_teacher_receipts(task: ForgeTask) -> bool:
    """Every public teacher-gate record must have a matching private receipt."""
    details = task.oracle_report.details
    return isinstance(details.get("teacher_gates"), dict)


def _rehydrate_protected_teacher_receipts(root: Path, task: ForgeTask) -> ForgeTask:
    """Restore protected receipts, then rerun the export gate before use."""
    if not _requires_protected_teacher_receipts(task):
        return task
    path = protected_teacher_receipts_path(root, task.task_id)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PublicationError(
            f"protected teacher receipts are missing for {task.task_id!r}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise PublicationError(
            f"protected teacher receipts are not a regular private file for "
            f"{task.task_id!r}"
        )
    if metadata.st_mode & 0o077:
        raise PublicationError(
            f"protected teacher receipts have unsafe permissions for {task.task_id!r}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(
            f"protected teacher receipts are unreadable for {task.task_id!r}"
        ) from exc
    if not isinstance(payload, list):
        raise PublicationError(
            f"protected teacher receipts are malformed for {task.task_id!r}"
        )
    try:
        report = OracleReport.from_protected_dict(
            {
                **task.oracle_report.to_protected_dict(),
                "protected_teacher_transport_receipts": payload,
            }
        )
    except (ValueError, TypeError) as exc:
        raise PublicationError(
            f"protected teacher receipts are invalid for {task.task_id!r}: {exc}"
        ) from exc
    return replace(task, oracle_report=report)


def _write_protected_alt_correct_audit(
    stage: Path,
    entry: PublicationEntry,
    previous: PublishedGeneration | None,
) -> None:
    """Persist raw alternative patches only beneath the private generation store."""
    target = protected_alt_correct_audit_path(stage, entry.task.task_id)
    audit = entry.task.oracle_report.protected_alt_correct_audit
    if isinstance(audit, dict):
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(target.parent, 0o700)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(target, 0o600)
        return

    if _requires_protected_alt_correct_audit(entry.task):
        raise PublicationError(
            f"protected alt-correct audit is missing for {entry.task.task_id!r}"
        )


def _write_protected_teacher_receipts(stage: Path, entry: PublicationEntry) -> None:
    """Persist transport authority only in a privileged source-free sidecar."""
    task = entry.task
    if not _requires_protected_teacher_receipts(task):
        return
    receipts = task.oracle_report.protected_teacher_transport_receipts
    if not receipts:
        raise PublicationError(
            f"protected teacher receipts are missing for {task.task_id!r}"
        )
    target = protected_teacher_receipts_path(stage, task.task_id)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(receipts, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(target, 0o600)


def _publish_generation_locked(
    out_dir: Path | str,
    entries: Sequence[PublicationEntry],
    *,
    workspace_writer: WorkspaceWriter,
    dataset_writer: DatasetWriter,
    overwrite: bool,
    replace_existing: bool = False,
    metadata_writer: GenerationMetadataWriter | None = None,
    extra_facade_artifacts: Sequence[str] = (),
) -> tuple[PublishedGeneration, list[PublicationOutcome]]:
    """Build, validate, and atomically select a complete generation.

    ``workspace_writer`` writes only beneath the private stage.  A workspace
    refusal is omitted like the historical batch exporter, while an exception
    during dataset validation or pointer commit leaves ``current`` unchanged.
    ``replace_existing`` is reserved for a recertification that has newly
    verified the same deterministic task id. It stages a complete successor
    generation without deleting the old one before the pointer switch.
    """
    out_path = Path(out_dir)
    unique_entries = list(entries)
    ids = [entry.task.task_id for entry in unique_entries]
    if len(ids) != len(set(ids)):
        raise PublicationError("generation request contains duplicate task ids")
    indexes = [entry.index for entry in unique_entries]
    if len(indexes) != len(set(indexes)):
        raise PublicationError("generation request contains duplicate entry indexes")
    for entry in unique_entries:
        try:
            ensure_oracle_exportable(
                entry.task.oracle_report,
                candidate=entry.task.candidate,
                calibration_kept=entry.task.calibration_report.is_keep,
            )
        except ExportRefusedError as exc:
            raise PublicationError(
                f"task {entry.task.task_id!r} is not exportable: {exc}"
            ) from exc

    _ensure_public_facade(out_path, extra_artifacts=extra_facade_artifacts)
    previous = load_published_generation(out_path)
    prior_by_id = (
        {entry.task.task_id: entry for entry in previous.entries} if previous else {}
    )
    store = _store_root(out_path)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=store))
    outcomes: list[PublicationOutcome] = []
    accepted: list[PublicationEntry] = []

    try:
        stage_tasks = stage / "tasks"
        stage_tasks.mkdir()
        for entry in sorted(unique_entries, key=lambda item: item.index):
            prior = prior_by_id.get(entry.task.task_id)
            if prior is not None:
                if _canonical_task_payload(prior.task) != _canonical_task_payload(
                    entry.task
                ):
                    if not replace_existing:
                        raise PublicationError(
                            f"conflicting duplicate task_id {entry.task.task_id!r} "
                            "against the published generation"
                        )
                else:
                    _copy_workspace(
                        previous.tasks_dir / prior.task.task_id,  # type: ignore[union-attr]
                        stage_tasks / entry.task.task_id,
                    )
                    accepted.append(
                        PublicationEntry(index=entry.index, task=entry.task)
                    )
                    outcomes.append(
                        PublicationOutcome(
                            index=entry.index,
                            task_id=entry.task.task_id,
                            status="skipped",
                            reason="task workspace already published",
                        )
                    )
                    continue

            result = workspace_writer(entry.task, stage_tasks)
            status = str(getattr(result, "status", "failed"))
            reason = str(getattr(result, "reason", ""))
            if status == "failed":
                raise PublicationError(
                    f"workspace export failed for {entry.task.task_id!r}: {reason}"
                )
            if status in ("shipped", "skipped"):
                accepted.append(entry)
            outcomes.append(
                PublicationOutcome(
                    index=entry.index,
                    task_id=entry.task.task_id,
                    status=status,
                    reason=reason,
                    leak_findings=tuple(getattr(result, "leak_findings", ())),
                )
            )

        accepted_ids = [entry.task.task_id for entry in accepted]
        dataset_writer(
            [entry.task for entry in accepted],
            stage / "dataset.jsonl",
            stage / "dataset.parquet",
        )
        for entry in accepted:
            _write_protected_alt_correct_audit(stage, entry, previous)
            _write_protected_teacher_receipts(stage, entry)
        if metadata_writer is not None:
            metadata_writer(stage, accepted)
        generation_id = uuid.uuid4().hex
        _write_manifest(stage, generation_id, accepted)
        _validate_staged_generation(stage, accepted_ids)
        _fsync_tree(stage)

        final_root = _generation_dir(out_path, generation_id)
        os.replace(stage, final_root)
        _fsync_path(final_root.parent)

        pointer_tmp = store / f".current-{uuid.uuid4().hex}"
        os.symlink(f"{_GENERATIONS_DIR}/{generation_id}", pointer_tmp)
        _fsync_path(store)
        os.replace(pointer_tmp, _current_path(out_path))
        _fsync_path(store)
        return (
            PublishedGeneration(
                generation_id=generation_id,
                root=final_root,
                tasks_dir=final_root / "tasks",
                jsonl_path=final_root / "dataset.jsonl",
                parquet_path=final_root / "dataset.parquet",
                entries=tuple(accepted),
            ),
            outcomes,
        )
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def publish_generation(
    out_dir: Path | str,
    entries: Sequence[PublicationEntry],
    *,
    workspace_writer: WorkspaceWriter,
    dataset_writer: DatasetWriter,
    overwrite: bool,
    replace_existing: bool = False,
    metadata_writer: GenerationMetadataWriter | None = None,
    extra_facade_artifacts: Sequence[str] = (),
    expected_current_generation_id: str | None = None,
) -> tuple[PublishedGeneration, list[PublicationOutcome]]:
    """Build and select a generation, optionally compare-and-swapping ``current``.

    The selected predecessor is checked while the publication lock is held and
    remains locked through final pointer replacement.  Therefore an intervening
    publisher cannot be silently overwritten by a terminal recovery writer.
    """
    out_path = Path(out_dir)
    with _publication_lock(out_path):
        _ensure_public_facade(out_path, extra_artifacts=extra_facade_artifacts)
        current = load_published_generation(out_path)
        actual = current.generation_id if current is not None else ""
        if (
            expected_current_generation_id is not None
            and actual != expected_current_generation_id
        ):
            expected = expected_current_generation_id or "<none>"
            selected = actual or "<none>"
            raise PublicationError(
                "expected current generation "
                f"{expected!r} does not match selected generation {selected!r}"
            )
        return _publish_generation_locked(
            out_path,
            entries,
            workspace_writer=workspace_writer,
            dataset_writer=dataset_writer,
            overwrite=overwrite,
            replace_existing=replace_existing,
            metadata_writer=metadata_writer,
            extra_facade_artifacts=extra_facade_artifacts,
        )


__all__ = [
    "PublicationEntry",
    "PublicationError",
    "PublicationOutcome",
    "PublishedGeneration",
    "canonical_task_payload",
    "load_published_generation",
    "publish_generation",
    "protected_alt_correct_audit_path",
    "protected_teacher_receipts_path",
]
