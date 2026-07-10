"""Durable, machine-global authority for the fixed alternate recovery identity."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal


_SCHEMA_VERSION = 1


class RecoveryAuthorityError(RuntimeError):
    """Raised when a recovery identity has already been permanently consumed."""


def default_authority_root() -> Path:
    """Return the non-configurable machine-global state location."""
    return Path("/var/lib/swe_forge/recovery-authority")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_regular(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RecoveryAuthorityError(f"{label} must be a regular non-symlink file")


def _require_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        path.mkdir(parents=True, mode=0o700)
        _fsync(path.parent)
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RecoveryAuthorityError(
            "recovery authority root must be a directory, not a symlink"
        )


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
        self.root = Path(root)
        self.identity = identity
        if (
            not identity
            or "/" in identity
            or "\\" in identity
            or identity in {".", ".."}
        ):
            raise RecoveryAuthorityError("recovery identity is not a safe filename")
        _require_directory(self.root)
        self.path = self.root / f"{identity}.json"
        self._lock_path = self.root / f"{identity}.lock"
        _require_regular(self.path, label="recovery authority record")
        _require_regular(self._lock_path, label="recovery authority lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(
            self._lock_path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _read_locked(self) -> RecoveryAuthorityRecord | None:
        _require_regular(self.path, label="recovery authority record")
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RecoveryAuthorityError(
                "recovery authority record is unreadable"
            ) from exc
        return RecoveryAuthorityRecord.from_dict(payload, identity=self.identity)

    def _write_locked(self, record: RecoveryAuthorityRecord) -> None:
        encoded = (
            json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self.identity}.",
            dir=self.root,
        )
        temp = Path(temp_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            _fsync(self.root)
        finally:
            temp.unlink(missing_ok=True)

    def record(self) -> RecoveryAuthorityRecord | None:
        """Read the current durable record without granting any authority."""
        with self._locked():
            return self._read_locked()

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
        with self._locked():
            if self._read_locked() is not None:
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
            self._write_locked(record)
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
        with self._locked():
            current = self._read_locked()
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
            self._write_locked(record)
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
        with self._locked():
            current = self._read_locked()
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
            self._write_locked(record)
            return record


__all__ = [
    "RecoveryAttemptAuthority",
    "RecoveryAuthorityError",
    "RecoveryAuthorityRecord",
    "default_authority_root",
]
