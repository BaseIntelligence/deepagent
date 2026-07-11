"""Durable, machine-global authority for the fixed alternate recovery identity."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal


_SCHEMA_VERSION = 1
_INITIALIZATION_MARKER_PREFIX = ".authority-root-initializing-"


class RecoveryAuthorityError(RuntimeError):
    """Raised when a recovery identity has already been permanently consumed."""


def default_authority_root() -> Path:
    """Return the non-configurable machine-global state location."""
    return Path("/var/lib/swe_forge/recovery-authority")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_regular_at(directory_fd: int, name: str, *, label: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RecoveryAuthorityError(f"{label} must be a regular non-symlink file")


def _authority_root_path(value: Path | str) -> Path:
    """Return a canonical lexical authority root without alias components."""
    raw = os.fspath(value)
    if raw == os.sep:
        return Path(raw)
    if raw.startswith(os.sep):
        components = raw[1:].split(os.sep)
    else:
        components = raw.split(os.sep)
    if not raw or any(component in {"", ".", ".."} for component in components):
        raise RecoveryAuthorityError(
            "recovery authority root path alias is not allowed"
        )
    path = Path(raw)
    return path if path.is_absolute() else Path.cwd() / path


def _initialization_marker_name(root: Path) -> str:
    """Return a collision-resistant marker name for one canonical root."""
    digest = hashlib.sha256(os.fsencode(root)).hexdigest()
    return f"{_INITIALIZATION_MARKER_PREFIX}{digest}"


def _reject_pending_initialization(directory_descriptor: int, marker_name: str) -> None:
    """Reject an authority root whose durable initialization never completed."""
    try:
        os.stat(
            marker_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    raise RecoveryAuthorityError(
        "recovery authority root has incomplete durable initialization"
    )


def _persist_initialization_intent(parent_descriptor: int, marker_name: str) -> None:
    """Durably record intent before creating any authority-root directory."""
    try:
        descriptor = os.open(
            marker_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
    except FileExistsError as exc:
        raise RecoveryAuthorityError(
            "recovery authority root has incomplete durable initialization"
        ) from exc
    os.close(descriptor)
    os.fsync(parent_descriptor)


def _require_no_pending_initialization(root: Path, marker_name: str) -> None:
    """Recheck every root ancestor before granting authority."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in root.parts[1:]:
            _reject_pending_initialization(descriptor, marker_name)
            try:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise RecoveryAuthorityError(
                    "recovery authority root is no longer available"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise RecoveryAuthorityError(
                    "recovery authority root contains a symlink"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise RecoveryAuthorityError(
                    "recovery authority root is not a directory"
                )
            child_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RecoveryAuthorityError(
                        "recovery authority root changed during directory traversal"
                    )
                os.close(descriptor)
                descriptor = child_descriptor
                child_descriptor = -1
            finally:
                if child_descriptor != -1:
                    os.close(child_descriptor)
        _reject_pending_initialization(descriptor, marker_name)
    finally:
        os.close(descriptor)


def _require_directory(path: Path | str) -> tuple[Path, tuple[int, int]]:
    """Create a no-follow authority root and durably persist its ancestry."""
    root = _authority_root_path(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    created: list[tuple[Path, int]] = []
    first_existing_parent: tuple[Path, int] | None = None
    current_path = Path("/")
    marker_name = _initialization_marker_name(root)
    initialization_parent_descriptor: int | None = None

    try:
        for component in root.parts[1:]:
            _reject_pending_initialization(descriptor, marker_name)
            child_path = current_path / component
            created_here = False
            try:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                if initialization_parent_descriptor is None:
                    _persist_initialization_intent(descriptor, marker_name)
                    initialization_parent_descriptor = os.dup(descriptor)
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    created_here = True
                except FileExistsError as exc:
                    raise RecoveryAuthorityError(
                        "recovery authority root changed during directory creation"
                    ) from exc
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)

            if stat.S_ISLNK(metadata.st_mode):
                raise RecoveryAuthorityError(
                    "recovery authority root contains a symlink"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise RecoveryAuthorityError(
                    "recovery authority root must be a directory"
                )

            child_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RecoveryAuthorityError(
                        "recovery authority root changed during directory traversal"
                    )

                if created_here:
                    if first_existing_parent is None:
                        first_existing_parent = (current_path, os.dup(descriptor))
                    created.append((child_path, os.dup(child_descriptor)))
                os.close(descriptor)
                descriptor = child_descriptor
                child_descriptor = -1
                current_path = child_path
            finally:
                if child_descriptor != -1:
                    os.close(child_descriptor)

        if initialization_parent_descriptor is not None:
            for _, created_descriptor in reversed(created):
                os.fsync(created_descriptor)
            if first_existing_parent is not None:
                os.fsync(first_existing_parent[1])
            os.unlink(
                marker_name,
                dir_fd=initialization_parent_descriptor,
            )
            os.fsync(initialization_parent_descriptor)
        _require_no_pending_initialization(root, marker_name)
        metadata = os.fstat(descriptor)
        return root, (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)
        for _, created_descriptor in created:
            os.close(created_descriptor)
        if first_existing_parent is not None:
            os.close(first_existing_parent[1])
        if initialization_parent_descriptor is not None:
            os.close(initialization_parent_descriptor)


def _open_existing_directory(path: Path, expected_identity: tuple[int, int]) -> int:
    """Open a no-follow root only when it remains the verified directory."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            try:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise RecoveryAuthorityError(
                    "recovery authority root is no longer available"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise RecoveryAuthorityError(
                    "recovery authority root contains a symlink"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise RecoveryAuthorityError(
                    "recovery authority root is not a directory"
                )
            try:
                child_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise RecoveryAuthorityError(
                    "recovery authority root changed during directory traversal"
                ) from exc
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise RecoveryAuthorityError(
                        "recovery authority root changed during directory traversal"
                    )
                os.close(descriptor)
                descriptor = child_descriptor
                child_descriptor = -1
            finally:
                if child_descriptor != -1:
                    os.close(child_descriptor)

        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) != expected_identity:
            raise RecoveryAuthorityError("recovery authority root changed after setup")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@dataclass(frozen=True)
class RecoveryAuthorityRecord:
    """One immutable claim plus its durable terminal reconciliation."""

    identity: str
    state: Literal["claimed", "consumed"]
    run_id: str
    claimed_at: str
    expected_current_generation_id: str
    ledger_path: str
    ledger_run_id: str = ""
    certification_run_id: str = ""
    selected_generation_id: str = ""
    terminal_state: Literal["", "keep", "tombstone"] = ""

    @classmethod
    def from_dict(cls, payload: object, *, identity: str) -> "RecoveryAuthorityRecord":
        if not isinstance(payload, dict):
            raise RecoveryAuthorityError("recovery authority record must be an object")
        fields = (
            "identity",
            "state",
            "run_id",
            "claimed_at",
            "expected_current_generation_id",
            "ledger_path",
            "ledger_run_id",
            "certification_run_id",
            "selected_generation_id",
            "terminal_state",
        )
        if payload.get("schema_version") != _SCHEMA_VERSION or any(
            not isinstance(payload.get(field), str) for field in fields
        ):
            raise RecoveryAuthorityError("recovery authority record is malformed")
        if payload["identity"] != identity:
            raise RecoveryAuthorityError("recovery authority identity does not match")
        state = payload["state"]
        terminal = payload["terminal_state"]
        if state not in {"claimed", "consumed"} or terminal not in {
            "",
            "keep",
            "tombstone",
        }:
            raise RecoveryAuthorityError("recovery authority record has invalid state")
        record = cls(
            identity=payload["identity"],
            state=state,
            run_id=payload["run_id"],
            claimed_at=payload["claimed_at"],
            expected_current_generation_id=payload["expected_current_generation_id"],
            ledger_path=payload["ledger_path"],
            ledger_run_id=payload["ledger_run_id"],
            certification_run_id=payload["certification_run_id"],
            selected_generation_id=payload["selected_generation_id"],
            terminal_state=terminal,
        )
        if not record.run_id or not record.claimed_at or not record.ledger_path:
            raise RecoveryAuthorityError("recovery authority record lacks claim fields")
        if record.state == "claimed" and any(
            (
                record.ledger_run_id,
                record.certification_run_id,
                record.selected_generation_id,
                record.terminal_state,
            )
        ):
            raise RecoveryAuthorityError(
                "claimed recovery authority record has terminal fields"
            )
        if record.state == "consumed" and (
            not record.ledger_run_id
            or not record.certification_run_id
            or not record.selected_generation_id
            or not record.terminal_state
        ):
            raise RecoveryAuthorityError(
                "consumed recovery authority record lacks terminal reconciliation"
            )
        return record

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "identity": self.identity,
            "state": self.state,
            "run_id": self.run_id,
            "claimed_at": self.claimed_at,
            "expected_current_generation_id": self.expected_current_generation_id,
            "ledger_path": self.ledger_path,
            "ledger_run_id": self.ledger_run_id,
            "certification_run_id": self.certification_run_id,
            "selected_generation_id": self.selected_generation_id,
            "terminal_state": self.terminal_state,
        }


class RecoveryAttemptAuthority:
    """Serialize and durably consume one fixed recovery identity across roots."""

    def __init__(self, root: Path | str, identity: str) -> None:
        self.identity = identity
        if (
            not identity
            or "/" in identity
            or "\\" in identity
            or identity in {".", ".."}
        ):
            raise RecoveryAuthorityError("recovery identity is not a safe filename")
        self.root, self._root_identity = _require_directory(root)
        self.path = self.root / f"{identity}.json"
        self._lock_path = self.root / f"{identity}.lock"

    @contextmanager
    def _root_directory(self) -> Iterator[int]:
        """Pin the verified root for every authority operation."""
        descriptor = _open_existing_directory(self.root, self._root_identity)
        try:
            _require_regular_at(
                descriptor,
                f"{self.identity}.json",
                label="recovery authority record",
            )
            _require_regular_at(
                descriptor,
                f"{self.identity}.lock",
                label="recovery authority lock",
            )
            yield descriptor
        finally:
            os.close(descriptor)

    @contextmanager
    def _locked(self) -> Iterator[int]:
        with self._root_directory() as root_descriptor:
            try:
                descriptor = os.open(
                    f"{self.identity}.lock",
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=root_descriptor,
                )
            except OSError as exc:
                raise RecoveryAuthorityError(
                    "recovery authority lock could not be opened"
                ) from exc
            try:
                os.fchmod(descriptor, 0o600)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield root_descriptor
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _read_locked(self, root_descriptor: int) -> RecoveryAuthorityRecord | None:
        _require_regular_at(
            root_descriptor,
            f"{self.identity}.json",
            label="recovery authority record",
        )
        try:
            descriptor = os.open(
                f"{self.identity}.json",
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=root_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RecoveryAuthorityError(
                "recovery authority record is unreadable"
            ) from exc
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryAuthorityError(
                "recovery authority record is unreadable"
            ) from exc
        return RecoveryAuthorityRecord.from_dict(payload, identity=self.identity)

    def _write_locked(
        self, root_descriptor: int, record: RecoveryAuthorityRecord
    ) -> None:
        encoded = (
            json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        temporary_name = f".{self.identity}.{uuid.uuid4().hex}"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_descriptor,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary_name,
                f"{self.identity}.json",
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
            )
            os.fsync(root_descriptor)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=root_descriptor)
            except FileNotFoundError:
                pass

    def record(self) -> RecoveryAuthorityRecord | None:
        """Read the current durable record without granting any authority."""
        with self._locked() as root_descriptor:
            return self._read_locked(root_descriptor)

    def claim(
        self,
        *,
        run_id: str,
        expected_current_generation_id: str,
        ledger_path: Path | str,
    ) -> RecoveryAuthorityRecord:
        """Durably consume live authority before certification or live work."""
        if not run_id:
            raise RecoveryAuthorityError("recovery claim requires a run id")
        ledger = Path(ledger_path)
        if not ledger.is_absolute():
            raise RecoveryAuthorityError("recovery ledger path must be absolute")
        with self._locked() as root_descriptor:
            if self._read_locked(root_descriptor) is not None:
                raise RecoveryAuthorityError(
                    f"recovery identity {self.identity!r} is already consumed"
                )
            record = RecoveryAuthorityRecord(
                identity=self.identity,
                state="claimed",
                run_id=run_id,
                claimed_at=_now(),
                expected_current_generation_id=expected_current_generation_id,
                ledger_path=os.fspath(ledger),
            )
            self._write_locked(root_descriptor, record)
            return record

    def consume(
        self,
        claim: RecoveryAuthorityRecord,
        *,
        terminal_state: Literal["keep", "tombstone"],
        certification_run_id: str,
        ledger_run_id: str,
        selected_generation_id: str,
    ) -> RecoveryAuthorityRecord:
        """Attach the selected terminal generation to an existing durable claim."""
        if (
            not certification_run_id
            or not ledger_run_id
            or not selected_generation_id
            or certification_run_id != claim.run_id
            or ledger_run_id != claim.run_id
        ):
            raise RecoveryAuthorityError(
                "terminal recovery evidence does not reconcile to the authority claim"
            )
        with self._locked() as root_descriptor:
            current = self._read_locked(root_descriptor)
            if current is None or current != claim:
                raise RecoveryAuthorityError(
                    "recovery authority claim changed before terminal reconciliation"
                )
            record = RecoveryAuthorityRecord(
                identity=claim.identity,
                state="consumed",
                run_id=claim.run_id,
                claimed_at=claim.claimed_at,
                expected_current_generation_id=claim.expected_current_generation_id,
                ledger_path=claim.ledger_path,
                ledger_run_id=ledger_run_id,
                certification_run_id=certification_run_id,
                selected_generation_id=selected_generation_id,
                terminal_state=terminal_state,
            )
            self._write_locked(root_descriptor, record)
            return record

    def migrate_terminal_tombstone(
        self,
        *,
        run_id: str,
        ledger_run_id: str,
        certification_run_id: str,
        selected_generation_id: str,
        ledger_path: Path | str,
    ) -> RecoveryAuthorityRecord:
        """Persist the legacy terminal tombstone as permanently consumed."""
        ledger = Path(ledger_path)
        if not ledger.is_absolute():
            raise RecoveryAuthorityError("recovery ledger path must be absolute")
        provisional = RecoveryAuthorityRecord(
            identity=self.identity,
            state="claimed",
            run_id=run_id,
            claimed_at=_now(),
            expected_current_generation_id="",
            ledger_path=os.fspath(ledger),
        )
        with self._locked() as root_descriptor:
            current = self._read_locked(root_descriptor)
            if current is not None:
                if (
                    current.state == "consumed"
                    and current.run_id == run_id
                    and current.terminal_state == "tombstone"
                    and current.selected_generation_id == selected_generation_id
                ):
                    return current
                raise RecoveryAuthorityError(
                    f"recovery identity {self.identity!r} is already consumed"
                )
            record = RecoveryAuthorityRecord(
                identity=provisional.identity,
                state="consumed",
                run_id=provisional.run_id,
                claimed_at=provisional.claimed_at,
                expected_current_generation_id="",
                ledger_path=provisional.ledger_path,
                ledger_run_id=ledger_run_id,
                certification_run_id=certification_run_id,
                selected_generation_id=selected_generation_id,
                terminal_state="tombstone",
            )
            if ledger_run_id != run_id or certification_run_id != run_id:
                raise RecoveryAuthorityError(
                    "legacy terminal tombstone run ids do not reconcile"
                )
            self._write_locked(root_descriptor, record)
            return record


__all__ = [
    "RecoveryAttemptAuthority",
    "RecoveryAuthorityError",
    "RecoveryAuthorityRecord",
    "default_authority_root",
]
