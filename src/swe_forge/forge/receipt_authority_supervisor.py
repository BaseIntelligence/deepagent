"""Trusted, command-scoped production receipt-authority supervisor.

The supervisor is the only process that handles the deployment-provisioned
private-key capability.  It passes that capability to the executable
authority, passes the other end of a Unix socketpair to Forge, and closes the
capability before Forge starts.  Forge therefore receives a socket descriptor,
not a key descriptor, and the authority is reaped when the command ends.

This module is intentionally a small executable wrapper rather than a
long-lived service.  The production root is hard-pinned to the canonical
deployment location and all IPC remains local to the supervised command.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from swe_forge.forge import receipt_authority


AUTHORITY_CLIENT_FD_ENV = "SWE_FORGE_RECEIPT_AUTHORITY_FD"
"""Environment variable containing the inherited Forge-side socket fd number."""

AUTHORITY_MODULE = "swe_forge.forge.receipt_authority_service"
SUPERVISOR_STARTUP_TIMEOUT = 5.0
SUPERVISOR_CHILD_TIMEOUT = 2.0
_SUPERVISOR_CANONICAL_ROOT = Path("/var/lib/swe_forge/teacher-receipt-authority")


class ReceiptAuthoritySupervisorError(RuntimeError):
    """Raised when a supervised authority command cannot be established."""


def _canonical_root() -> Path:
    return _SUPERVISOR_CANONICAL_ROOT


def _validate_production_root(root: Path) -> None:
    """Validate deployment material without creating or changing it."""
    if root.absolute() != _canonical_root():
        raise ReceiptAuthoritySupervisorError(
            "production supervisor accepts only the canonical authority root"
        )
    try:
        key_id, _, environment = receipt_authority._read_pinned_public_key(  # noqa: SLF001
            root
        )
    except receipt_authority.ReceiptAuthorityError as exc:
        raise ReceiptAuthoritySupervisorError(
            "production authority trust material must be provisioned before "
            "supervisor startup"
        ) from exc
    if not key_id or environment != "production":
        raise ReceiptAuthoritySupervisorError(
            "production authority trust material is not provisioned"
        )


def _validate_key_fd(key_fd: int) -> None:
    if key_fd < 3:
        raise ReceiptAuthoritySupervisorError("production key capability fd is invalid")
    try:
        metadata = os.fstat(key_fd)
    except OSError as exc:
        raise ReceiptAuthoritySupervisorError(
            "production key capability fd is unavailable"
        ) from exc
    if (metadata.st_mode & 0o170000) not in {
        0o010000,  # FIFO, the deployment handoff's pipe form
        0o100000,  # regular file, supported for sealed capability handles
    }:
        raise ReceiptAuthoritySupervisorError(
            "production key capability fd has an unsupported type"
        )


def _terminate_kill_reap(
    process: subprocess.Popen[bytes],
    *,
    timeout: float = SUPERVISOR_CHILD_TIMEOUT,
) -> None:
    """Boundedly terminate and reap one supervisor-owned child.

    A child which has already exited still needs ``wait()`` so that its
    process table entry is reaped, but it must not receive unnecessary
    signals.  A live child gets one bounded grace period after SIGTERM.  If
    it does not exit in that period, SIGKILL is followed by a final bounded
    wait to reap it.
    """
    if timeout <= 0:
        raise ValueError("supervisor child timeout must be positive")
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


terminate_kill_reap = _terminate_kill_reap


def _cleanup_supervisor_children(
    forge: subprocess.Popen[bytes] | None,
    authority: subprocess.Popen[bytes] | None,
    *,
    timeout: float,
) -> None:
    """Reap Forge and authority, never allowing Forge cleanup to skip authority."""
    cleanup_error: BaseException | None = None
    try:
        if forge is not None:
            _terminate_kill_reap(forge, timeout=timeout)
    except BaseException as exc:
        cleanup_error = exc
    finally:
        try:
            if authority is not None:
                _terminate_kill_reap(authority, timeout=timeout)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
    if cleanup_error is not None:
        raise cleanup_error


def _wait_for_authority_startup(
    authority: subprocess.Popen[bytes],
    supervisor_socket: socket.socket,
    root: Path,
    timeout: float,
) -> None:
    """Wait until the authority has emitted either ready or startup_error."""
    deadline = time.monotonic() + timeout
    peeked = bytearray()
    while b"\n" not in peeked:
        remaining = deadline - time.monotonic()
        if (
            remaining <= 0
            or not select.select([supervisor_socket], [], [], remaining)[0]
        ):
            if authority.poll() is not None:
                raise ReceiptAuthoritySupervisorError(
                    "production receipt authority exited during startup"
                )
            raise ReceiptAuthoritySupervisorError(
                "production receipt authority startup timed out"
            )
        chunk = supervisor_socket.recv(64 * 1024, socket.MSG_PEEK)
        if not chunk:
            raise ReceiptAuthoritySupervisorError(
                "production receipt authority rejected startup"
            )
        peeked = bytearray(chunk)
        if len(peeked) > receipt_authority.MAX_IPC_BYTES:
            raise ReceiptAuthoritySupervisorError(
                "production receipt authority startup frame is malformed"
            )
    encoded = bytes(peeked[: peeked.index(b"\n")])
    try:
        ready = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptAuthoritySupervisorError(
            "production receipt authority startup frame is malformed"
        ) from exc
    if not isinstance(ready, dict):
        raise ReceiptAuthoritySupervisorError(
            "production receipt authority startup frame is malformed"
        )
    expected_key_id, _, expected_environment = (
        receipt_authority._read_pinned_public_key(  # noqa: SLF001
            root
        )
    )
    if (
        ready.get("type") != "ready"
        or ready.get("environment") != expected_environment
        or ready.get("environment") != "production"
        or ready.get("key_id") != expected_key_id
        or ready.get("root_id") != receipt_authority._root_identity(root, "production")  # noqa: SLF001
    ):
        raise ReceiptAuthoritySupervisorError(
            "production receipt authority ready identity mismatches the canonical root"
        )


def run_supervised(
    command: Sequence[str],
    *,
    key_fd: int,
    root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    startup_timeout: float = SUPERVISOR_STARTUP_TIMEOUT,
    child_timeout: float = SUPERVISOR_CHILD_TIMEOUT,
) -> int:
    """Run ``command`` with a supervisor-owned production authority.

    ``key_fd`` is an inherited deployment capability.  This function never
    reads it, copies it into Python memory, places it in an environment, or
    passes it to the Forge command.  The authority consumes and closes it.
    """
    owned_key_fd = key_fd if key_fd >= 3 else -1
    supervisor_socket: socket.socket | None = None
    authority_socket: socket.socket | None = None
    authority: subprocess.Popen[bytes] | None = None
    forge: subprocess.Popen[bytes] | None = None
    try:
        if not command:
            raise ReceiptAuthoritySupervisorError("supervised Forge command is empty")
        if startup_timeout <= 0:
            raise ReceiptAuthoritySupervisorError(
                "supervisor startup timeout is invalid"
            )
        if child_timeout <= 0:
            raise ReceiptAuthoritySupervisorError("supervisor child timeout is invalid")
        authority_root = _canonical_root() if root is None else Path(root).absolute()
        _validate_production_root(authority_root)
        _validate_key_fd(key_fd)
        supervisor_socket, authority_socket = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
        forge_fd = supervisor_socket.fileno()
        authority_environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "LITELLM_MODE": "PRODUCTION",
        }
        authority = subprocess.Popen(
            [
                sys.executable,
                "-m",
                AUTHORITY_MODULE,
                "--root",
                str(authority_root),
                "--domain",
                "production",
                "--key-fd",
                str(key_fd),
                "--ipc-fd",
                str(authority_socket.fileno()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=authority_environment,
            pass_fds=(key_fd, authority_socket.fileno()),
            close_fds=True,
            bufsize=0,
        )
        # The supervisor must not retain the capability while Forge runs.
        os.close(key_fd)
        key_fd = -1
        owned_key_fd = -1
        authority_socket.close()
        authority_socket = None
        _wait_for_authority_startup(
            authority, supervisor_socket, authority_root, startup_timeout
        )

        forge_environment = dict(os.environ)
        if env is not None:
            forge_environment.update(env)
        # A production key capability is never a Forge input, even if a
        # surrounding launcher or command override advertises an old bootstrap
        # name.
        forge_environment.pop("SWE_FORGE_RECEIPT_AUTHORITY_KEY_FD", None)
        forge_environment.pop("SWE_FORGE_RECEIPT_AUTHORITY_BOOTSTRAP_FD", None)
        forge_environment[AUTHORITY_CLIENT_FD_ENV] = str(forge_fd)
        forge = subprocess.Popen(
            list(command),
            stdin=None,
            stdout=None,
            stderr=None,
            env=forge_environment,
            cwd=None if cwd is None else os.fspath(cwd),
            pass_fds=(forge_fd,),
            close_fds=True,
        )
        supervisor_socket.close()
        supervisor_socket = None
        forge_status = forge.wait()
        if forge_status < 0:
            return 128 + -forge_status
        return int(forge_status)
    except OSError as exc:
        raise ReceiptAuthoritySupervisorError(
            "production receipt authority supervisor failed"
        ) from exc
    finally:
        execution_error = sys.exc_info()[1]
        cleanup_error: BaseException | None = None
        try:
            try:
                _cleanup_supervisor_children(
                    forge,
                    authority,
                    timeout=child_timeout,
                )
            except BaseException as exc:
                cleanup_error = exc
        finally:
            if authority_socket is not None:
                try:
                    authority_socket.close()
                except OSError:
                    pass
            if supervisor_socket is not None:
                try:
                    supervisor_socket.close()
                except OSError:
                    pass
            if owned_key_fd >= 0:
                try:
                    os.close(owned_key_fd)
                except OSError:
                    pass
                owned_key_fd = -1
        if cleanup_error is not None and execution_error is None:
            raise cleanup_error


def supervise(
    command: Sequence[str],
    *,
    key_fd: int,
    root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    startup_timeout: float = SUPERVISOR_STARTUP_TIMEOUT,
    child_timeout: float = SUPERVISOR_CHILD_TIMEOUT,
) -> int:
    """Compatibility alias for callers embedding the trusted wrapper."""
    return run_supervised(
        command,
        key_fd=key_fd,
        root=root,
        env=env,
        cwd=cwd,
        startup_timeout=startup_timeout,
        child_timeout=child_timeout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Forge with a command-scoped production receipt authority"
    )
    parser.add_argument("--key-fd", type=int, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        return run_supervised(
            command,
            key_fd=args.key_fd,
            root=args.root,
            startup_timeout=args.startup_timeout,
        )
    except ReceiptAuthoritySupervisorError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_CLIENT_FD_ENV",
    "SUPERVISOR_CHILD_TIMEOUT",
    "ReceiptAuthoritySupervisorError",
    "terminate_kill_reap",
    "run_supervised",
    "supervise",
]
