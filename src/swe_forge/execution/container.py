"""Container lifecycle management for Docker execution.

This module provides high-level abstractions for managing container
lifecycle including creation, execution, and cleanup with guaranteed
resource cleanup on exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from swe_forge.execution.docker_client import (
    ContainerConfig,
    ContainerStatus,
    DockerClient,
    DockerError,
    ExecResult,
)

logger = getLogger(__name__)


class ManagedContainerStatus(Enum):
    """Status of a managed container."""

    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return self.value


@dataclass
class VolumeMount:
    """Configuration for mounting a host path into a container.

    Attributes:
        host_path: Path on the host system to mount.
        container_path: Path inside the container where the mount appears.
        read_only: If True, mount is read-only.
    """

    host_path: str
    container_path: str
    read_only: bool = False

    def to_docker_bind(self) -> str:
        """Convert to Docker bind mount format.

        Returns:
            Docker bind mount string like "/host:/container:ro" or "/host:/container:rw"
        """
        mode = "ro" if self.read_only else "rw"
        return f"{self.host_path}:{self.container_path}:{mode}"


@dataclass
class ContainerSpec:
    """Specification for creating and managing a container.

    This is a higher-level configuration than ContainerConfig, providing
    more intuitive interfaces for common operations like volume mounts
    and environment variables.

    Attributes:
        name: Container name.
        image: Docker image to use.
        command: Command to run in the container.
        volumes: List of volume mounts.
        env: Dictionary of environment variables.
        working_dir: Working directory inside the container.
        user: User to run as (e.g., "1000:1000").
        network_mode: Network mode for the container.
        memory_mb: Memory limit in megabytes.
        cpu_limit: CPU limit as fraction (1.0 = 1 CPU).
        pids_limit: Maximum number of processes.
        stop_timeout: Seconds to wait for graceful shutdown before SIGKILL.
    """

    name: str
    image: str
    command: list[str] | None = None
    volumes: list[VolumeMount] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None
    user: str | None = None
    network_mode: str = "bridge"
    memory_mb: int = 512
    cpu_limit: float = 1.0
    pids_limit: int = 100
    stop_timeout: int = 10

    def to_container_config(self) -> ContainerConfig:
        """Convert to ContainerConfig for DockerClient.

        Returns:
            ContainerConfig instance ready for DockerClient.create_container.
        """
        # Convert volume mounts to Docker bind format
        volume_binds = [v.to_docker_bind() for v in self.volumes]

        # Convert env dict to list of "KEY=value" strings
        env_list = [f"{k}={v}" for k, v in self.env.items()]

        return ContainerConfig(
            name=self.name,
            image=self.image,
            cmd=self.command,
            env=env_list,
            working_dir=self.working_dir,
            volumes=volume_binds,
            user=self.user,
            network_mode=self.network_mode,
            memory_mb=self.memory_mb,
            cpu_limit=self.cpu_limit,
            pids_limit=self.pids_limit,
        )


@dataclass
class ManagedContainer:
    """A managed Docker container with lifecycle tracking.

    This class tracks the state of a container and provides
    metadata about its creation and execution.
    """

    id: str
    spec: ContainerSpec
    status: ManagedContainerStatus = ManagedContainerStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_code: int | None = None
    error_message: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if the container is in a terminal state."""
        return self.status in (
            ManagedContainerStatus.COMPLETED,
            ManagedContainerStatus.FAILED,
            ManagedContainerStatus.TIMEOUT,
        )

    @property
    def is_running(self) -> bool:
        """Check if the container is currently running."""
        return self.status == ManagedContainerStatus.RUNNING


class ContainerManager:
    """Async context manager for Docker container lifecycle.

    Provides guaranteed cleanup on both normal exit and exceptions,
    implementing the RAII pattern for container management.

    Usage:
        spec = ContainerSpec(
            name="test-container",
            image="python:3.11-slim",
            volumes=[VolumeMount("/host/path", "/container/path")],
            env={"FOO": "bar"},
        )

        async with ContainerManager(docker_client, spec) as container:
            # Container is running
            result = await container.exec(["ls", "-la"])
            # Automatic cleanup on exit
    """

    def __init__(
        self,
        client: DockerClient,
        spec: ContainerSpec,
        *,
        auto_start: bool = True,
        auto_remove: bool = True,
    ) -> None:
        """Initialize the container manager.

        Args:
            client: DockerClient instance for Docker operations.
            spec: Container specification.
            auto_start: If True, start the container after creation.
            auto_remove: If True, remove container on cleanup.
        """
        self._client = client
        self._spec = spec
        self._auto_start = auto_start
        self._auto_remove = auto_remove
        self._container: ManagedContainer | None = None
        self._started = False

    @classmethod
    def from_existing(
        cls,
        client: DockerClient,
        container_id: str,
        spec: ContainerSpec,
        *,
        auto_remove: bool = True,
    ) -> Self:
        """Create a manager for an existing container.

        This is useful for reconnecting to containers created elsewhere.

        Args:
            client: DockerClient instance for Docker operations.
            container_id: ID of an existing container.
            spec: Container specification (for reference).
            auto_remove: If True, remove container on cleanup.

        Returns:
            ContainerManager instance attached to existing container.
        """
        manager = cls(client, spec, auto_start=False, auto_remove=auto_remove)
        manager._container = ManagedContainer(
            id=container_id,
            spec=spec,
            status=ManagedContainerStatus.PENDING,
        )
        return manager

    async def __aenter__(self) -> Self:
        """Create and optionally start the container.

        Returns:
            Self (ContainerManager) for method chaining.

        Raises:
            DockerError: If container creation or start fails.
        """
        try:
            await self._client.ensure_image(self._spec.image)

            config = self._spec.to_container_config()
            container_id = await self._client.create_container(config)

            self._container = ManagedContainer(
                id=container_id,
                spec=self._spec,
                status=ManagedContainerStatus.PENDING,
            )

            if self._auto_start:
                await self.start()

            return self

        except Exception:
            await self._cleanup(force=True)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Cleanup the container on exit.

        Guarantees container removal even on exceptions.
        Logs warnings if cleanup fails but doesn't mask original exception.
        """
        await self._cleanup(force=exc_type is not None)

    async def _cleanup(self, force: bool = False) -> None:
        if self._container is None:
            return

        container_id = self._container.id

        try:
            if self._started:
                try:
                    await self._client.stop_container(
                        container_id, timeout=self._spec.stop_timeout
                    )
                    self._container.status = ManagedContainerStatus.COMPLETED
                except Exception:
                    force = True

            if self._auto_remove:
                try:
                    await self._client.remove_container(container_id, force=force)
                except Exception:
                    pass

            self._container = None
            self._started = False

        except Exception as e:
            logger.error(f"Container cleanup failed for {container_id}: {e}")

    async def start(self) -> None:
        """Start the container.

        Raises:
            DockerError: If container is not in PENDING state or start fails.
        """
        if self._container is None:
            raise DockerError("Container not created")

        if self._container.status != ManagedContainerStatus.PENDING:
            raise DockerError(
                f"Cannot start container in {self._container.status} state",
                container_id=self._container.id,
            )

        self._container.status = ManagedContainerStatus.CREATING

        try:
            await self._client.start_container(self._container.id)
            self._container.status = ManagedContainerStatus.RUNNING
            self._started = True
        except DockerError as e:
            self._container.status = ManagedContainerStatus.FAILED
            self._container.error_message = str(e)
            raise

    async def stop(self) -> None:
        """Stop the container gracefully.

        Uses the stop_timeout from the spec.
        """
        if self._container is None or not self._started:
            return

        if self._container.status != ManagedContainerStatus.RUNNING:
            return

        try:
            await self._client.stop_container(
                self._container.id, timeout=self._spec.stop_timeout
            )
            self._container.status = ManagedContainerStatus.COMPLETED
            self._started = False
        except DockerError as e:
            self._container.status = ManagedContainerStatus.FAILED
            self._container.error_message = str(e)
            raise

    async def exec(
        self,
        cmd: list[str],
        *,
        timeout: float | None = None,
    ) -> ExecResult:
        """Execute a command in the container.

        Args:
            cmd: Command to execute as list of strings.
            timeout: Optional timeout in seconds.

        Returns:
            ExecResult with exit_code, stdout, and stderr.

        Raises:
            DockerError: If container is not running or exec fails.
        """
        if self._container is None:
            raise DockerError("Container not created")

        if (
            not self._started
            or self._container.status != ManagedContainerStatus.RUNNING
        ):
            raise DockerError(
                f"Cannot exec in container with {self._container.status} state",
                container_id=self._container.id,
            )

        result = await self._client.exec(self._container.id, cmd, timeout=timeout)
        return result

    async def get_logs(
        self,
        *,
        stdout: bool = True,
        stderr: bool = True,
        tail: int | None = None,
    ) -> str:
        """Get logs from the container.

        Args:
            stdout: Include stdout.
            stderr: Include stderr.
            tail: Number of lines to return from end.

        Returns:
            Combined logs as string.
        """
        if self._container is None:
            raise DockerError("Container not created")

        return await self._client.get_logs(
            self._container.id,
            stdout=stdout,
            stderr=stderr,
            tail=tail,
        )

    async def stream_logs(
        self,
        *,
        stdout: bool = True,
        stderr: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Stream logs from the container.

        Args:
            stdout: Include stdout.
            stderr: Include stderr.

        Yields:
            Log chunks as strings.
        """
        if self._container is None:
            raise DockerError("Container not created")

        async for chunk in self._client.stream_logs(
            self._container.id,
            stdout=stdout,
            stderr=stderr,
        ):
            yield chunk

    async def wait(self, *, timeout: float | None = None) -> int:
        """Wait for the container to finish.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            Exit code of the container.

        Raises:
            DockerError: If timeout is reached.
        """
        if self._container is None:
            raise DockerError("Container not created")

        exit_code = await self._client.wait_container(
            self._container.id, timeout=timeout
        )
        self._container.exit_code = exit_code
        self._started = False

        if exit_code == 0:
            self._container.status = ManagedContainerStatus.COMPLETED
        else:
            self._container.status = ManagedContainerStatus.FAILED
            self._container.error_message = f"Exited with code {exit_code}"

        return exit_code

    async def sync_status(self) -> ManagedContainerStatus:
        """Sync local status with actual Docker container status.

        Returns:
            Updated container status.
        """
        if self._container is None:
            raise DockerError("Container not created")

        try:
            status = await self._client.get_container_status(self._container.id)

            status_map = {
                ContainerStatus.CREATED: ManagedContainerStatus.PENDING,
                ContainerStatus.RUNNING: ManagedContainerStatus.RUNNING,
                ContainerStatus.PAUSED: ManagedContainerStatus.RUNNING,
                ContainerStatus.RESTARTING: ManagedContainerStatus.RUNNING,
                ContainerStatus.EXITED: ManagedContainerStatus.COMPLETED,
                ContainerStatus.REMOVING: ManagedContainerStatus.COMPLETED,
                ContainerStatus.DEAD: ManagedContainerStatus.FAILED,
                ContainerStatus.UNKNOWN: ManagedContainerStatus.FAILED,
            }

            self._container.status = status_map.get(
                status, ManagedContainerStatus.FAILED
            )

            return self._container.status

        except DockerError:
            self._container.status = ManagedContainerStatus.FAILED
            return self._container.status

    @property
    def container(self) -> ManagedContainer | None:
        """Get the managed container, or None if not created."""
        return self._container

    @property
    def container_id(self) -> str | None:
        """Get the container ID, or None if not created."""
        return self._container.id if self._container else None
