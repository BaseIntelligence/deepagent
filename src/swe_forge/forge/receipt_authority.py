"""Parent-side facade for the isolated teacher receipt authority.

The authority process owns the provider transport and its ephemeral Ed25519 key.
This module intentionally exports verification and bounded IPC only.  It has no
production signing, issuing, or private-key loading API.
"""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
import stat
import threading
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class ReceiptAuthorityError(RuntimeError):
    """Raised when the isolated authority or its pinned root is unsafe."""


MAX_IPC_BYTES = 1_000_000
ROOT_NAME = "authority-v1.json"
TEST_MARKER_NAME = "test-authority-v1.json"


def default_authority_root() -> Path:
    """Return the fixed machine-global public trust-root directory."""
    return Path("/var/lib/swe_forge/teacher-receipt-authority")


def initialize_test_authority_root(root: Path | str) -> None:
    """Create a non-production root marker for hermetic child-authority tests."""
    from swe_forge.forge import receipt_authority_service

    receipt_authority_service.initialize_test_root(Path(root))


def _open_pinned_root(root: Path) -> int:
    """Open every authority-root ancestor without following symlinks."""
    if not root.is_absolute():
        raise ReceiptAuthorityError("teacher receipt authority root is unsafe")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in root.parts[1:]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ReceiptAuthorityError("teacher receipt authority root is unsafe")
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ReceiptAuthorityError(
            "teacher receipt authority root is unavailable"
        ) from exc
    except Exception:
        os.close(descriptor)
        raise


def _read_pinned_public_key(root: Path) -> tuple[str, bytes]:
    root_fd = _open_pinned_root(root)
    try:
        try:
            names = set(os.listdir(root_fd))
        except OSError as exc:
            raise ReceiptAuthorityError(
                "teacher receipt authority root is unavailable"
            ) from exc
        if not names <= {ROOT_NAME, TEST_MARKER_NAME}:
            raise ReceiptAuthorityError(
                "teacher receipt authority root contains private or unknown material"
            )
        try:
            file_fd = os.open(ROOT_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        except OSError as exc:
            raise ReceiptAuthorityError(
                "teacher receipt public root is unavailable"
            ) from exc
        try:
            file_metadata = os.fstat(file_fd)
            if not stat.S_ISREG(file_metadata.st_mode) or file_metadata.st_mode & 0o077:
                raise ReceiptAuthorityError("teacher receipt public root is unsafe")
            payload = json.loads(os.read(file_fd, 64 * 1024).decode("utf-8"))
            if os.read(file_fd, 1):
                raise ReceiptAuthorityError("teacher receipt public root is malformed")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptAuthorityError(
                "teacher receipt public root is malformed"
            ) from exc
        finally:
            os.close(file_fd)
    finally:
        os.close(root_fd)
    expected = {"version", "algorithm", "environment", "key_id", "public_key"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ReceiptAuthorityError("teacher receipt public root is malformed")
    if (
        payload["version"] != 1
        or payload["algorithm"] != "Ed25519"
        or payload["environment"] not in {"production", "test"}
        or not isinstance(payload["key_id"], str)
        or len(payload["key_id"]) != 64
        or any(char not in "0123456789abcdef" for char in payload["key_id"])
        or not isinstance(payload["public_key"], str)
    ):
        raise ReceiptAuthorityError("teacher receipt public root is malformed")
    try:
        public_key = base64.b64decode(payload["public_key"], validate=True)
    except Exception as exc:
        raise ReceiptAuthorityError("teacher receipt public root is malformed") from exc
    if (
        len(public_key) != 32
        or hashlib.sha256(public_key).hexdigest() != payload["key_id"]
    ):
        raise ReceiptAuthorityError("teacher receipt public root key is malformed")
    return payload["key_id"], public_key


def _authority_entry(connection: Connection, root_text: str) -> None:
    """Import the issuer only after the fresh authority child starts."""
    from swe_forge.forge.receipt_authority_service import authority_process

    authority_process(connection, root_text)


def _parse_message(encoded: bytes) -> dict[str, object]:
    if not isinstance(encoded, bytes) or len(encoded) > MAX_IPC_BYTES:
        raise ReceiptAuthorityError("teacher receipt authority IPC is malformed")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptAuthorityError(
            "teacher receipt authority IPC is malformed"
        ) from exc
    if not isinstance(payload, dict):
        raise ReceiptAuthorityError("teacher receipt authority IPC is malformed")
    return payload


def _canonical_message(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def verify_signature(*, key_id: str, claims: bytes, signature: str) -> bool:
    """Verify against only the Ed25519 key pinned in the durable public root."""
    if (
        not isinstance(key_id, str)
        or len(key_id) != 64
        or any(char not in "0123456789abcdef" for char in key_id)
        or not isinstance(claims, bytes)
        or not isinstance(signature, str)
    ):
        return False
    try:
        pinned_id, public_key = _read_pinned_public_key(default_authority_root())
        if pinned_id != key_id:
            return False
        decoded_signature = base64.b64decode(signature, validate=True)
        if len(decoded_signature) != 64:
            return False
        Ed25519PublicKey.from_public_bytes(public_key).verify(decoded_signature, claims)
    except (ReceiptAuthorityError, InvalidSignature, ValueError):
        return False
    return True


class ReceiptAuthorityClient:
    """One supervised, bounded IPC session with the child authority."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        startup_timeout: float = 5.0,
        request_timeout: float = 120.0,
    ) -> None:
        self._root = Path(default_authority_root() if root is None else root)
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._connection: Connection | None = None
        self._process: BaseProcess | None = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def is_usable(self) -> bool:
        """Whether this supervised session can still accept new completions."""
        return not self._failed

    def _start(self) -> None:
        if self._failed:
            raise ReceiptAuthorityError("teacher receipt authority is unavailable")
        if self._connection is not None and self._process is not None:
            return
        # A fresh interpreter prevents caller-process monkeypatches, globals,
        # and imported fake transports from being inherited by the authority.
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_authority_entry,
            args=(child, str(self._root)),
            name="swe-forge-teacher-authority",
            daemon=True,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        try:
            ready = self._receive(self._startup_timeout)
            if (
                ready.get("type") != "ready"
                or not isinstance(ready.get("key_id"), str)
                or ready.get("environment") not in {"production", "test"}
            ):
                raise ReceiptAuthorityError("teacher receipt authority failed to start")
            pinned_id, _ = _read_pinned_public_key(self._root)
            if ready["key_id"] != pinned_id:
                raise ReceiptAuthorityError("teacher receipt authority root mismatches")
        except Exception:
            self._failed = True
            self.close()
            raise

    def _receive(self, timeout: float) -> dict[str, object]:
        connection = self._connection
        process = self._process
        if connection is None or process is None:
            raise ReceiptAuthorityError("teacher receipt authority is unavailable")
        if timeout <= 0 or not connection.poll(timeout):
            if not process.is_alive():
                raise ReceiptAuthorityError("teacher receipt authority crashed")
            raise ReceiptAuthorityError("teacher receipt authority timed out")
        try:
            encoded = connection.recv_bytes(MAX_IPC_BYTES + 1)
        except (EOFError, OSError) as exc:
            raise ReceiptAuthorityError("teacher receipt authority crashed") from exc
        try:
            return _parse_message(encoded)
        except ReceiptAuthorityError as exc:
            raise ReceiptAuthorityError(
                "teacher receipt authority IPC is malformed"
            ) from exc

    def complete(
        self, request: dict[str, object], *, timeout: float | None = None
    ) -> dict[str, object]:
        """Send one bounded request, failing permanently after a protocol fault."""
        with self._lock:
            try:
                self._start()
                encoded = _canonical_message(request)
                if len(encoded) > MAX_IPC_BYTES:
                    raise ReceiptAuthorityError(
                        "teacher receipt authority request exceeds its bound"
                    )
                assert self._connection is not None
                self._connection.send_bytes(encoded)
                response = self._receive(
                    self._request_timeout if timeout is None else timeout
                )
                if response.get("type") == "error":
                    raise ReceiptAuthorityError(
                        "teacher receipt authority rejected completion"
                    )
                if response.get("type") not in {"result", "provider_error"}:
                    raise ReceiptAuthorityError(
                        "teacher receipt authority response is malformed"
                    )
                return response
            except Exception as exc:
                self._failed = True
                self.close()
                if isinstance(exc, ReceiptAuthorityError):
                    raise
                raise ReceiptAuthorityError("teacher receipt authority failed") from exc

    def close(self) -> None:
        """Close the pipe and reap only the authority process started by this client."""
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


_shared_authorities: dict[Path, tuple[ReceiptAuthorityClient, int]] = {}
_shared_authorities_lock = threading.Lock()


def acquire_shared_authority(*, request_timeout: float) -> ReceiptAuthorityClient:
    """Share one live child per root so its ephemeral signer survives callers."""
    root = default_authority_root().absolute()
    with _shared_authorities_lock:
        existing = _shared_authorities.get(root)
        if existing is not None and existing[0].is_usable:
            client, references = existing
            _shared_authorities[root] = (client, references + 1)
            return client
        client = ReceiptAuthorityClient(root=root, request_timeout=request_timeout)
        _shared_authorities[root] = (client, 1)
        return client


def release_shared_authority(client: ReceiptAuthorityClient) -> None:
    """Release one caller reference and reap the child after the final close."""
    root = client._root.absolute()  # noqa: SLF001 - module-owned lifecycle state
    close = False
    with _shared_authorities_lock:
        existing = _shared_authorities.get(root)
        if existing is None or existing[0] is not client:
            return
        references = existing[1] - 1
        if references <= 0:
            del _shared_authorities[root]
            close = True
        else:
            _shared_authorities[root] = (client, references)
    if close:
        client.close()


__all__ = [
    "ReceiptAuthorityClient",
    "ReceiptAuthorityError",
    "default_authority_root",
    "initialize_test_authority_root",
    "verify_signature",
]
