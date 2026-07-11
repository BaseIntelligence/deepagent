"""Host-global signing authority for teacher transport receipts.

The signing key lives outside candidate, report, and publication trees.  Receipt
verification never trusts caller-supplied key material and never creates a key:
only the concrete provider-return path may initialize or use the issuer.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from swe_forge.forge.recovery_authority import (
    _open_existing_directory,
    _require_directory,
)


_KEY_NAME = "issuer-v1.key"
_LOCK_NAME = "issuer-v1.lock"
_KEY_BYTES = 32


class ReceiptAuthorityError(RuntimeError):
    """Raised when the host-held receipt issuer state is unavailable or unsafe."""


def default_authority_root() -> Path:
    """Return the fixed machine-global receipt issuer state directory."""
    return Path("/var/lib/swe_forge/teacher-receipt-authority")


def _require_safe_regular(directory_fd: int, name: str, *, label: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
    ):
        raise ReceiptAuthorityError(f"{label} must be a private regular file")


@contextmanager
def _authority_directory(*, create: bool) -> Iterator[int]:
    """Open the fixed authority root without following aliases or symlinks."""
    root = default_authority_root()
    if create:
        try:
            root, root_identity = _require_directory(root)
        except Exception as exc:
            raise ReceiptAuthorityError(
                "teacher receipt authority root is unavailable"
            ) from exc
    else:
        try:
            metadata = root.lstat()
        except OSError as exc:
            raise ReceiptAuthorityError(
                "teacher receipt authority key is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ReceiptAuthorityError(
                "teacher receipt authority root is not a directory"
            )
        root_identity = (metadata.st_dev, metadata.st_ino)
    try:
        descriptor = _open_existing_directory(root, root_identity)
    except Exception as exc:
        raise ReceiptAuthorityError(
            "teacher receipt authority root is unavailable"
        ) from exc
    try:
        if os.fstat(descriptor).st_mode & 0o077:
            raise ReceiptAuthorityError(
                "teacher receipt authority root has unsafe permissions"
            )
        _require_safe_regular(descriptor, _KEY_NAME, label="teacher receipt key")
        _require_safe_regular(descriptor, _LOCK_NAME, label="teacher receipt lock")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _locked_authority(*, create: bool) -> Iterator[int]:
    with _authority_directory(create=create) as root_descriptor:
        try:
            descriptor = os.open(
                _LOCK_NAME,
                os.O_RDWR | (os.O_CREAT if create else 0) | os.O_NOFOLLOW,
                0o600,
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            raise ReceiptAuthorityError(
                "teacher receipt authority lock is unavailable"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                raise ReceiptAuthorityError(
                    "teacher receipt authority lock has unsafe permissions"
                )
            if create:
                os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield root_descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _read_key(root_descriptor: int) -> bytes | None:
    _require_safe_regular(root_descriptor, _KEY_NAME, label="teacher receipt key")
    try:
        descriptor = os.open(
            _KEY_NAME,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=root_descriptor,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReceiptAuthorityError("teacher receipt key is unreadable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ReceiptAuthorityError("teacher receipt key has unsafe permissions")
        key = os.read(descriptor, _KEY_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(key) != _KEY_BYTES:
        raise ReceiptAuthorityError("teacher receipt key is malformed")
    return key


def _load_or_create_key() -> bytes:
    with _locked_authority(create=True) as root_descriptor:
        key = _read_key(root_descriptor)
        if key is not None:
            return key
        try:
            descriptor = os.open(
                _KEY_NAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            raise ReceiptAuthorityError(
                "teacher receipt key cannot be created"
            ) from exc
        try:
            os.fchmod(descriptor, 0o600)
            key = os.urandom(_KEY_BYTES)
            written = os.write(descriptor, key)
            if written != len(key):
                raise ReceiptAuthorityError("teacher receipt key could not be written")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(root_descriptor)
        return key


def _load_existing_key() -> bytes:
    with _locked_authority(create=False) as root_descriptor:
        key = _read_key(root_descriptor)
        if key is None:
            raise ReceiptAuthorityError("teacher receipt key is unavailable")
        return key


def _key_id(key: bytes) -> str:
    """Return a stable non-secret identifier for the active issuer key."""
    return hashlib.sha256(key).hexdigest()[:32]


def issuer_key_id() -> str:
    """Initialize or load the host-held issuer and return its public key id."""
    return _key_id(_load_or_create_key())


def _sign_claims(claims: bytes) -> tuple[str, str]:
    """Sign canonical claims after a concrete provider transport has returned."""
    key = _load_or_create_key()
    return _key_id(key), hmac.new(key, claims, hashlib.sha256).hexdigest()


def verify_signature(*, key_id: str, claims: bytes, signature: str) -> bool:
    """Reload the host authority and verify a receipt without creating keys."""
    if (
        not isinstance(key_id, str)
        or len(key_id) != 32
        or any(character not in "0123456789abcdef" for character in key_id)
        or not isinstance(signature, str)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        return False
    try:
        key = _load_existing_key()
    except ReceiptAuthorityError:
        return False
    if not hmac.compare_digest(_key_id(key), key_id):
        return False
    expected = hmac.new(key, claims, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


__all__ = [
    "ReceiptAuthorityError",
    "default_authority_root",
    "issuer_key_id",
    "verify_signature",
]
