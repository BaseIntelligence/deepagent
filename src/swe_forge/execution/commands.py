"""Command execution in Docker containers with streaming and timeout support."""

from __future__ import annotations

import json
import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiodocker import Docker


def demultiplex_docker_stream(data: bytes) -> str:
    """Demultiplex Docker's multiplexed stream output.

    Docker exec with Tty=False returns frames with 8-byte headers:
    - 1 byte: Stream type (1=stdout, 2=stderr)
    - 3 bytes: padding (zeros)
    - 4 bytes: payload size (big-endian uint32)
    """
    if not data:
        return ""

    result = []
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
        result.append(chunk.decode("utf-8", errors="replace"))
        offset += payload_size

    if not result and data:
        return data.decode("utf-8", errors="replace")

    return "".join(result)


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
            output_bytes = await start_response.read()

        stdout_str = demultiplex_docker_stream(output_bytes)

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
