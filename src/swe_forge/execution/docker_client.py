"""Async Docker client using aiodocker for container management."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Self

from aiodocker import Docker
from aiodocker.exceptions import DockerError as AioDockerError

from swe_forge.execution._timeout import (
    is_timeout_exit,
    outer_read_deadline,
    wrap_command_with_timeout,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class DockerError(Exception):
    """Docker operation error."""

    def __init__(
        self,
        message: str,
        *,
        container_id: str | None = None,
        exit_code: int | None = None,
    ):
        super().__init__(message)
        self.container_id = container_id
        self.exit_code = exit_code


class ContainerStatus(Enum):
    """Status of a container."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    EXITED = "exited"
    REMOVING = "removing"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass
class ExecResult:
    """Result of executing a command in a container."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0


@dataclass
class ContainerConfig:
    """Configuration for creating a new container."""

    name: str
    image: str
    cmd: list[str] | None = None
    env: list[str] = field(default_factory=list)
    working_dir: str | None = None
    volumes: list[str] = field(default_factory=list)
    user: str | None = None
    network_mode: str = "bridge"
    memory_mb: int = 512
    cpu_limit: float = 1.0
    pids_limit: int = 100

    def to_docker_config(self) -> dict[str, Any]:
        """Convert to aiodocker container config format."""
        host_config: dict[str, Any] = {
            "Memory": self.memory_mb * 1024 * 1024,
            "NanoCpus": int(self.cpu_limit * 1_000_000_000),
            "PidsLimit": self.pids_limit,
            "NetworkMode": self.network_mode,
        }
        if self.volumes:
            host_config["Binds"] = self.volumes

        config: dict[str, Any] = {
            "Image": self.image,
            "HostConfig": host_config,
            "Tty": True,
            "AttachStdin": False,
            "AttachStdout": True,
            "AttachStderr": True,
        }
        if self.cmd:
            config["Cmd"] = self.cmd
        if self.env:
            config["Env"] = self.env
        if self.working_dir:
            config["WorkingDir"] = self.working_dir
        if self.user:
            config["User"] = self.user
        return config


class DockerClient:
    """Async Docker client for container operations.

    Usage:
        async with DockerClient() as client:
            await client.ping()
            container_id = await client.create_container(config)
            await client.start_container(container_id)
            result = await client.exec(container_id, ["ls", "-la"])
            await client.stop_container(container_id)
            await client.remove_container(container_id)
    """

    def __init__(self) -> None:
        self._docker: Docker | None = None
        self._own_connection: bool = True

    @classmethod
    def from_docker(cls, docker: Docker) -> Self:
        """Create client from existing Docker instance (doesn't close it)."""
        client = cls()
        client._docker = docker
        client._own_connection = False
        return client

    async def __aenter__(self) -> Self:
        if self._docker is None:
            self._docker = Docker()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._docker and self._own_connection:
            await self._docker.close()
            self._docker = None

    @property
    def _client(self) -> Docker:
        if self._docker is None:
            raise DockerError(
                "Docker client not initialized - use async context manager"
            )
        return self._docker

    async def ping(self) -> bool:
        """Check if Docker daemon is accessible."""
        try:
            await self._client.images.list()
            return True
        except AioDockerError:
            return False
        except OSError:
            return False

    async def create_container(self, config: ContainerConfig) -> str:
        """Create a container with the given configuration.

        Returns:
            Container ID string.
        """
        container = await self._client.containers.create_or_replace(
            name=config.name,
            config=config.to_docker_config(),
        )
        return container.id

    async def start_container(self, container_id: str) -> None:
        """Start a container by ID."""
        container = self._client.containers.container(container_id)
        await container.start()

    async def stop_container(self, container_id: str, timeout: int = 10) -> None:
        """Stop a container by ID.

        Args:
            container_id: Container ID or name.
            timeout: Seconds to wait before sending SIGKILL.
        """
        container = self._client.containers.container(container_id)
        await container.stop(t=timeout)

    async def remove_container(self, container_id: str, force: bool = False) -> None:
        """Remove a container by ID.

        Args:
            container_id: Container ID or name.
            force: Force removal even if running.
        """
        container = self._client.containers.container(container_id)
        await container.delete(v=True, force=force)

    async def list_containers(self, all: bool = False) -> list[dict[str, Any]]:
        """List containers.

        Args:
            all: Include stopped containers.

        Returns:
            List of container info dicts.
        """
        return await self._client.containers.list(all=all)

    async def get_container_status(self, container_id: str) -> ContainerStatus:
        """Get the status of a container."""
        try:
            container = self._client.containers.container(container_id)
            info = await container.show()
            state = info.get("State", {})
            status_str = state.get("Status", "unknown")
            return ContainerStatus(status_str.lower())
        except AioDockerError as e:
            if "No such container" in str(e) or "404" in str(e):
                raise DockerError(
                    f"Container not found: {container_id}",
                    container_id=container_id,
                ) from e
            raise DockerError(
                f"Failed to get container status: {e}",
                container_id=container_id,
            ) from e

    async def exec(
        self,
        container_id: str,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute a command in a running container.

        Args:
            container_id: Container ID or name.
            cmd: Command to execute as list of strings.
            timeout: Optional timeout in seconds.

        Returns:
            ExecResult with exit_code, stdout, and stderr.

        Raises:
            DockerError: If the exec exceeds ``timeout``. When a timeout is set
                the command is wrapped under coreutils ``timeout --signal=KILL``
                so the in-container process tree is reaped (process-group
                SIGKILL); the streaming reader is also cancelled/awaited so no
                hung process or reader survives.
        """
        start_time = time.monotonic()
        container = self._client.containers.container(container_id)

        # Wrap under coreutils ``timeout`` so the daemon reaps the whole
        # in-container process tree at the deadline even if the read below is
        # uncancellable. No wrap when no timeout is requested (behavior
        # unchanged).
        exec_cmd = wrap_command_with_timeout(cmd, timeout) if timeout else cmd

        exec_id = await container.exec(
            cmd=exec_cmd,
            stdout=True,
            stderr=True,
            stdin=False,
            tty=False,
            privileged=False,
        )

        exec_instance = self._client.exec(exec_id)

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        stream = exec_instance.start(detach=False)

        async def read_stream() -> None:
            async for chunk in stream:
                if chunk:
                    chunk_str = (
                        chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                    )
                    if chunk_str:
                        stdout_chunks.append(chunk_str)

        if timeout:
            reader = asyncio.ensure_future(read_stream())
            done, _pending = await asyncio.wait(
                {reader}, timeout=outer_read_deadline(timeout)
            )
            if reader not in done:
                reader.cancel()
                finished, _ = await asyncio.wait({reader}, timeout=5.0)
                for task in finished:
                    with contextlib.suppress(BaseException):
                        task.result()
                raise DockerError(
                    f"Exec timed out after {timeout}s; in-container process "
                    "tree reaped, reader cancelled",
                    container_id=container_id,
                )
            reader.result()
        else:
            await read_stream()

        exec_info = await exec_instance.inspect()
        exit_code = exec_info.get("ExitCode", -1)

        if timeout and is_timeout_exit(
            exit_code, time.monotonic() - start_time, timeout
        ):
            raise DockerError(
                f"Exec timed out after {timeout}s; in-container process tree "
                f"reaped (exit {exit_code})",
                container_id=container_id,
            )

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)

        return ExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    async def get_logs(
        self,
        container_id: str,
        *,
        stdout: bool = True,
        stderr: bool = True,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Get logs from a container.

        Args:
            container_id: Container ID or name.
            stdout: Include stdout.
            stderr: Include stderr.
            follow: Stream logs (returns empty if True, use stream_logs instead).
            tail: Number of lines to return from end.

        Returns:
            Combined logs as string.
        """
        container = self._client.containers.container(container_id)
        kwargs: dict[str, Any] = {
            "stdout": stdout,
            "stderr": stderr,
            "follow": follow,
        }
        if tail is not None:
            kwargs["tail"] = tail

        logs = await container.log(**kwargs)
        return "".join(logs) if isinstance(logs, list) else str(logs)

    async def stream_logs(
        self,
        container_id: str,
        *,
        stdout: bool = True,
        stderr: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Stream logs from a container.

        Args:
            container_id: Container ID or name.
            stdout: Include stdout.
            stderr: Include stderr.

        Yields:
            Log chunks as strings.
        """
        container = self._client.containers.container(container_id)
        stream = container.log(
            stdout=stdout,
            stderr=stderr,
            follow=True,
            stream=True,
        )

        async for chunk in stream:
            if chunk:
                yield chunk.decode() if isinstance(chunk, bytes) else str(chunk)

    async def wait_container(
        self,
        container_id: str,
        *,
        timeout: float | None = None,
    ) -> int:
        """Wait for a container to finish.

        Args:
            container_id: Container ID or name.
            timeout: Optional timeout in seconds.

        Returns:
            Exit code of the container.
        """
        container = self._client.containers.container(container_id)

        async def wait() -> int:
            result = await container.wait(condition="not-running")
            return result.get("StatusCode", -1)

        if timeout:
            try:
                return await asyncio.wait_for(wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise DockerError(
                    f"Wait timed out after {timeout}s",
                    container_id=container_id,
                )
        return await wait()

    async def pull_image(self, image: str) -> None:
        """Pull a Docker image from registry.

        Args:
            image: Image name with optional tag.
        """
        await self._client.images.pull(from_image=image)

    async def image_exists(self, image: str) -> bool:
        """Check if an image exists locally."""
        try:
            await self._client.images.inspect(image)
            return True
        except AioDockerError:
            return False

    async def ensure_image(self, image: str) -> None:
        """Ensure an image exists, pulling if necessary."""
        if not await self.image_exists(image):
            await self.pull_image(image)
