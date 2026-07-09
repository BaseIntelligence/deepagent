"""Command execution in Docker containers with streaming and timeout support."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiodocker import Docker

from swe_forge.execution._timeout import (
    new_timeout_marker,
    outer_read_deadline,
    remove_timeout_marker,
    wrap_command_with_timeout,
)


def split_docker_stream(data: bytes) -> tuple[str, str]:
    """Demultiplex Docker's stdout/stderr frames.

    Docker exec with Tty=False returns frames with 8-byte headers:
    - 1 byte: Stream type (1=stdout, 2=stderr)
    - 3 bytes: padding (zeros)
    - 4 bytes: payload size (big-endian uint32)
    """
    if not data:
        return "", ""

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    offset = 0

    while offset + 8 <= len(data):
        stream_type = data[offset]
        if stream_type not in (1, 2):
            break
        payload_size = int.from_bytes(data[offset + 4 : offset + 8], "big")
        offset += 8

        if offset + payload_size > len(data):
            break

        chunk = data[offset : offset + payload_size]
        decoded = chunk.decode("utf-8", errors="replace")
        if stream_type == 1:
            stdout_chunks.append(decoded)
        else:
            stderr_chunks.append(decoded)
        offset += payload_size

    if offset != len(data) or not (stdout_chunks or stderr_chunks):
        # Unit fakes and older daemon paths can provide an unframed payload.
        return data.decode("utf-8", errors="replace"), ""

    return "".join(stdout_chunks), "".join(stderr_chunks)


def demultiplex_docker_stream(data: bytes) -> str:
    """Return combined output for callers retaining the legacy helper contract."""
    stdout, stderr = split_docker_stream(data)
    return stdout + stderr


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from swe_forge.execution.docker_client import DockerClient


class CommandError(Exception):
    """Error during command execution."""

    def __init__(
        self,
        message: str,
        *,
        container_id: str | None = None,
        exit_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ):
        super().__init__(message)
        self.container_id = container_id
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class ExecResult:
    """Result of executing a command in a container."""

    stdout: str
    stderr: str
    exit_code: int
    duration: float

    @property
    def success(self) -> bool:
        return self.exit_code == 0


async def exec_in_container(
    container_id: str,
    cmd: Sequence[str],
    *,
    client: DockerClient | Docker | None = None,
    timeout: float = 120.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
) -> ExecResult:
    """Execute a command in a running container.

    Args:
        container_id: Container ID or name.
        cmd: Command to execute as a sequence of strings.
        client: DockerClient or Docker instance. If None, creates a new connection.
        timeout: Timeout in seconds (default 120s).
        cwd: Working directory for the command.
        env: Environment variables for the command.
        user: User to run the command as.

    Returns:
        ExecResult with stdout, stderr, exit_code, and duration.

    Raises:
        asyncio.TimeoutError: If command execution exceeds timeout. The
            in-container process tree is reaped (process-group SIGKILL) before
            this is raised, so no hung process survives.
        CommandError: If command execution fails.
    """
    start_time = time.monotonic()

    docker, own_connection = _get_docker_instance(client)

    try:
        marker = new_timeout_marker()
        # The session watchdog emits ``marker`` immediately before reaping the
        # tree. That is explicit timeout provenance, unlike target exit codes.
        exec_options = _build_exec_options(
            wrap_command_with_timeout(cmd, timeout, marker=marker), cwd, env, user
        )

        async with docker._query(
            f"containers/{container_id}/exec",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(exec_options),
        ) as response:
            exec_create = await response.json()
        exec_id = exec_create["Id"]

        output_bytes = await _start_and_read_output(
            docker, exec_id, container_id, timeout
        )

        stdout_str, stderr_str = split_docker_stream(output_bytes)
        stdout_str, stdout_timed_out = remove_timeout_marker(stdout_str, marker)
        stderr_str, stderr_timed_out = remove_timeout_marker(stderr_str, marker)

        async with docker._query(
            f"exec/{exec_id}/json",
            method="GET",
        ) as inspect_response:
            exec_info = await inspect_response.json()
        exit_code = exec_info.get("ExitCode", -1)

        if stdout_timed_out or stderr_timed_out:
            raise asyncio.TimeoutError(
                f"command in container {container_id} exceeded its {timeout}s "
                "timeout and was reaped"
            )

        return ExecResult(
            stdout=stdout_str,
            stderr=stderr_str,
            exit_code=exit_code,
            duration=time.monotonic() - start_time,
        )

    finally:
        if own_connection:
            await docker.close()


async def _start_and_read_output(
    docker: Docker,
    exec_id: str,
    container_id: str,
    timeout: float,
) -> bytes:
    """Start the exec and read its full output, bounded by ``timeout`` + grace.

    The watchdog reaps a hung command at the deadline and the stream should
    close on its own. This grace window guards a broken HTTP response: its
    response is closed, its reader cancelled and awaited, and then a timeout is
    raised. The same cleanup happens if the outer caller cancels this coroutine.
    """
    start_response: object | None = None

    async def _read() -> bytes:
        nonlocal start_response
        async with docker._query(
            f"exec/{exec_id}/start",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"Detach": False, "Tty": False}),
        ) as response:
            start_response = response
            return await response.read()

    read_task = asyncio.create_task(_read())
    try:
        done, _pending = await asyncio.wait(
            {read_task}, timeout=outer_read_deadline(timeout)
        )
        if read_task not in done:
            raise asyncio.TimeoutError(
                f"command in container {container_id} exceeded its {timeout}s "
                "timeout; in-container process tree reaped, response closed"
            )
        return read_task.result()
    finally:
        await _close_response_and_wait_for_reader(start_response, read_task)


async def _close_response_and_wait_for_reader(
    response: object | None, read_task: asyncio.Task[bytes]
) -> None:
    """Close an exec-start response, then cancel and await its reader task."""
    if response is not None:
        close = getattr(response, "close", None)
        if close is not None:
            closed = close()
            if inspect.isawaitable(closed):
                await closed

    if not read_task.done():
        read_task.cancel()
    with contextlib.suppress(BaseException):
        await read_task


async def stream_exec(
    container_id: str,
    cmd: Sequence[str],
    *,
    client: DockerClient | Docker | None = None,
    timeout: float = 120.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Stream output from command execution in a container.

    Yields raw output bytes as they arrive, useful for long-running commands.

    Args:
        container_id: Container ID or name.
        cmd: Command to execute as a sequence of strings.
        client: DockerClient or Docker instance. If None, creates a new connection.
        timeout: Timeout in seconds (default 120s).
        cwd: Working directory for the command.
        env: Environment variables for the command.
        user: User to run the command as.

    Yields:
        Raw output bytes from the command.

    Raises:
        asyncio.TimeoutError: If command execution exceeds timeout.
        CommandError: If command execution fails.
    """
    docker, own_connection = _get_docker_instance(client)

    try:
        exec_options = _build_exec_options(cmd, cwd, env, user)

        async with docker._query(
            f"containers/{container_id}/exec",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(exec_options),
        ) as response:
            exec_create = await response.json()
        exec_id = exec_create["Id"]

        async with docker._query(
            f"exec/{exec_id}/start",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"Detach": False, "Tty": False}),
        ) as start_response:
            output = await start_response.read()

        if output:
            yield output

    finally:
        if own_connection:
            await docker.close()


async def exec_with_callback(
    container_id: str,
    cmd: Sequence[str],
    *,
    client: DockerClient | Docker | None = None,
    timeout: float = 120.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    on_output: callable | None = None,
) -> ExecResult:
    """Execute command with callback for streaming output.

    Calls on_output callback for each output chunk as it arrives.

    Args:
        container_id: Container ID or name.
        cmd: Command to execute as a sequence of strings.
        client: DockerClient or Docker instance. If None, creates a new connection.
        timeout: Timeout in seconds (default 120s).
        cwd: Working directory for the command.
        env: Environment variables for the command.
        user: User to run the command as.
        on_output: Callback for output chunks (receives str).

    Returns:
        ExecResult with stdout, stderr, exit_code, and duration.

    Raises:
        asyncio.TimeoutError: If command execution exceeds timeout.
        CommandError: If command execution fails.
    """
    start_time = time.monotonic()

    docker, own_connection = _get_docker_instance(client)

    try:
        exec_options = _build_exec_options(cmd, cwd, env, user)

        async with docker._query(
            f"containers/{container_id}/exec",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(exec_options),
        ) as response:
            exec_create = await response.json()
        exec_id = exec_create["Id"]

        async with docker._query(
            f"exec/{exec_id}/start",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"Detach": False, "Tty": False}),
        ) as start_response:
            output = await start_response.read()

        stdout_str = output.decode("utf-8", errors="replace") if output else ""

        if on_output and stdout_str:
            on_output(stdout_str)

        async with docker._query(
            f"exec/{exec_id}/json",
            method="GET",
        ) as inspect_response:
            exec_info = await inspect_response.json()
        exit_code = exec_info.get("ExitCode", -1)

        duration = time.monotonic() - start_time

        return ExecResult(
            stdout=stdout_str,
            stderr="",
            exit_code=exit_code,
            duration=duration,
        )

    finally:
        if own_connection:
            await docker.close()


def _get_docker_instance(client: DockerClient | Docker | None) -> tuple[Docker, bool]:
    """Get Docker instance from client parameter.

    Returns:
        Tuple of (Docker instance, whether we own the connection).
    """
    if client is None:
        return Docker(), True

    if hasattr(client, "_docker"):
        docker_client = client
        if docker_client._docker is None:
            return Docker(), True
        return docker_client._docker, False

    return client, False


def _build_exec_options(
    cmd: Sequence[str],
    cwd: str | None,
    env: dict[str, str] | None,
    user: str | None,
) -> dict:
    """Build Docker exec create options."""
    options: dict = {
        "Cmd": list(cmd),
        "AttachStdout": True,
        "AttachStderr": True,
        "AttachStdin": False,
        "Tty": False,
    }

    if cwd:
        options["WorkingDir"] = cwd
    if env:
        options["Env"] = [f"{k}={v}" for k, v in env.items()]
    if user:
        options["User"] = user

    return options
