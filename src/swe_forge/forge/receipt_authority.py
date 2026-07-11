"""Parent-side facade for the isolated teacher receipt authority.

The authority process owns the provider transport and its ephemeral Ed25519 key.
This module intentionally exports verification and bounded IPC only.  It has no
production signing, issuing, or private-key loading API.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import select
import stat
import subprocess
import sys
import threading
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class ReceiptAuthorityError(RuntimeError):
    """Raised when the isolated authority or its pinned root is unsafe."""


MAX_IPC_BYTES = 1_000_000
ROOT_NAME = "authority-v1.json"
TEST_MARKER_NAME = "test-authority-v1.json"
PRODUCTION_MARKER_NAME = "production-authority-v1.json"
TRUST_DOMAINS = frozenset(("production", "test"))


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


def _root_identity(root: Path, environment: str) -> str:
    canonical_root = os.path.normpath(os.path.abspath(os.fspath(root)))
    return hashlib.sha256(
        json.dumps(
            {
                "version": 1,
                "environment": environment,
                "root": canonical_root,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _read_root_environment(root: Path) -> str:
    """Read exactly one immutable trust-domain marker from a root."""
    root_fd = _open_pinned_root(root)
    try:
        names = set(os.listdir(root_fd))
        if not names <= {ROOT_NAME, TEST_MARKER_NAME, PRODUCTION_MARKER_NAME}:
            raise ReceiptAuthorityError(
                "teacher receipt authority root contains private or unknown material"
            )
        has_test = TEST_MARKER_NAME in names
        has_production = PRODUCTION_MARKER_NAME in names
        if has_test == has_production:
            raise ReceiptAuthorityError(
                "teacher receipt authority root has no unique trust domain"
            )
        marker_name, expected = (
            (TEST_MARKER_NAME, {"environment": "test"})
            if has_test
            else (PRODUCTION_MARKER_NAME, {"environment": "production"})
        )
        marker_fd = os.open(marker_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            marker_stat = os.fstat(marker_fd)
            if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_mode & 0o077:
                raise ReceiptAuthorityError(
                    "teacher receipt authority marker is unsafe"
                )
            marker = json.loads(os.read(marker_fd, 4096).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptAuthorityError(
                "teacher receipt authority marker is malformed"
            ) from exc
        finally:
            os.close(marker_fd)
        if marker != expected:
            raise ReceiptAuthorityError("teacher receipt authority marker is malformed")
        return str(expected["environment"])
    finally:
        os.close(root_fd)


def _read_pinned_public_key(root: Path) -> tuple[str, bytes, str]:
    root_fd = _open_pinned_root(root)
    try:
        try:
            names = set(os.listdir(root_fd))
        except OSError as exc:
            raise ReceiptAuthorityError(
                "teacher receipt authority root is unavailable"
            ) from exc
        if not names <= {ROOT_NAME, TEST_MARKER_NAME, PRODUCTION_MARKER_NAME}:
            raise ReceiptAuthorityError(
                "teacher receipt authority root contains private or unknown material"
            )
        environment = _read_root_environment(root)
        allowed_marker = (
            TEST_MARKER_NAME if environment == "test" else PRODUCTION_MARKER_NAME
        )
        if names - {ROOT_NAME, allowed_marker}:
            raise ReceiptAuthorityError(
                "teacher receipt authority root has conflicting trust domains"
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
    expected = {
        "version",
        "algorithm",
        "environment",
        "root_id",
        "key_id",
        "public_key",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ReceiptAuthorityError("teacher receipt public root is malformed")
    if (
        payload["version"] != 1
        or payload["algorithm"] != "Ed25519"
        or payload["environment"] not in {"production", "test"}
        or not isinstance(payload["root_id"], str)
        or len(payload["root_id"]) != 64
        or any(char not in "0123456789abcdef" for char in payload["root_id"])
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
        or payload["environment"] != environment
        or payload["root_id"] != _root_identity(root, environment)
        or _key_id(
            public_key,
            environment=environment,
            root_id=_root_identity(root, environment),
        )
        != payload["key_id"]
    ):
        raise ReceiptAuthorityError("teacher receipt public root key is malformed")
    return payload["key_id"], public_key, environment


def _key_id(public_key: bytes, *, environment: str, root_id: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "algorithm": "Ed25519",
                "environment": environment,
                "root_id": root_id,
                "public_key": base64.b64encode(public_key).decode("ascii"),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


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
        pinned_id, public_key, _ = _read_pinned_public_key(default_authority_root())
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
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._failed = False
        self._closed = False

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def is_usable(self) -> bool:
        """Whether this supervised session can still accept new completions."""
        return not self._failed and not self._closed

    def _start(self) -> None:
        if self._failed or self._closed:
            raise ReceiptAuthorityError("teacher receipt authority is unavailable")
        if self._process is not None:
            return
        environment = "production"
        if self._root.exists():
            environment = _read_root_environment(self._root)
        bootstrap = secrets.token_urlsafe(32)
        child_environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "LITELLM_MODE": "PRODUCTION",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "swe_forge.forge.receipt_authority_service",
                "--root",
                str(self._root),
                "--domain",
                environment,
                "--bootstrap",
                bootstrap,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=child_environment,
            bufsize=0,
        )
        self._process = process
        try:
            ready = self._receive(self._startup_timeout)
            if (
                ready.get("type") != "ready"
                or not isinstance(ready.get("key_id"), str)
                or ready.get("environment") != environment
            ):
                raise ReceiptAuthorityError("teacher receipt authority failed to start")
            pinned_id, _, pinned_environment = _read_pinned_public_key(self._root)
            if ready["key_id"] != pinned_id:
                raise ReceiptAuthorityError("teacher receipt authority root mismatches")
            if pinned_environment != environment:
                raise ReceiptAuthorityError(
                    "teacher receipt authority trust domain mismatches"
                )
        except Exception:
            self._failed = True
            self.close()
            raise

    def _receive(self, timeout: float) -> dict[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise ReceiptAuthorityError("teacher receipt authority is unavailable")
        if timeout <= 0 or not select.select([process.stdout], [], [], timeout)[0]:
            if process.poll() is not None:
                raise ReceiptAuthorityError("teacher receipt authority crashed")
            raise ReceiptAuthorityError("teacher receipt authority timed out")
        try:
            encoded = process.stdout.readline(MAX_IPC_BYTES + 1)
        except OSError as exc:
            raise ReceiptAuthorityError("teacher receipt authority crashed") from exc
        if not encoded or not encoded.endswith(b"\n"):
            raise ReceiptAuthorityError("teacher receipt authority IPC is malformed")
        encoded = encoded[:-1]
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
                process = self._process
                if process is None or process.stdin is None:
                    raise ReceiptAuthorityError(
                        "teacher receipt authority is unavailable"
                    )
                process.stdin.write(encoded + b"\n")
                process.stdin.flush()
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
        self._closed = True
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)


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
