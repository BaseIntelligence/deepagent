"""High-level Docker sandbox for isolated repository operations.

This module provides DockerSandbox, which combines DockerClient + ContainerManager
for a simplified workflow: create container -> clone repo -> checkout -> run commands.

The sandbox automatically handles container lifecycle and provides a clean interface
for repository isolation tasks.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import getLogger
from typing import TYPE_CHECKING, Any

from swe_forge.execution.container import (
    ContainerManager,
    ContainerSpec,
    ManagedContainerStatus,
    VolumeMount,
)
from swe_forge.execution.docker_client import DockerClient, DockerError
from swe_forge.execution.commands import exec_in_container, ExecResult

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for DockerSandbox.

    Attributes:
        name: Base container name (will be made unique).
        image: Docker image to use.
        workspace_dir: Working directory inside container.
        memory_mb: Memory limit in megabytes.
        cpu_limit: CPU limit as fraction.
        pids_limit: Maximum number of processes.
        clone_timeout: Timeout for git clone in seconds.
        checkout_timeout: Timeout for git checkout in seconds.
        command_timeout: Default timeout for commands in seconds.
        install_timeout: Timeout for dependency installation in seconds.
    """

    name: str = "swe-sandbox"
    image: str = "python:3.11-slim"
    workspace_dir: str = "/repo"
    memory_mb: int = 2048
    cpu_limit: float = 2.0
    pids_limit: int = 200
    clone_timeout: float = 600.0
    checkout_timeout: float = 60.0
    command_timeout: float = 120.0
    install_timeout: float = 300.0


@dataclass
class SandboxState:
    """State tracking for a DockerSandbox instance."""

    container_id: str | None = None
    repo_url: str | None = None
    commit: str | None = None
    workspace_ready: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DockerSandbox:
    """High-level Docker sandbox for isolated repository operations.

    This class provides a simplified interface for:
    - Creating isolated Docker containers
    - Cloning and checking out repositories
    - Installing dependencies
    - Running commands

    Usage:
        async with DockerSandbox(docker_client, image="python:3.11") as sandbox:
            await sandbox.setup_workspace("https://github.com/owner/repo", "abc123")
            await sandbox.install_dependencies(["pytest"])
            result = await sandbox.run_command("pytest tests/")
    """

    def __init__(
        self,
        client: DockerClient,
        config: SandboxConfig | None = None,
        *,
        image: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize DockerSandbox.

        Args:
            client: DockerClient instance for Docker operations.
            config: Optional SandboxConfig. If not provided, uses defaults.
            image: Override image from config (convenience parameter).
            name: Override container name from config (convenience parameter).
        """
        self._client = client
        self._config = config or SandboxConfig()

        # Apply convenience overrides
        if image:
            self._config.image = image
        if name:
            self._config.name = name

        # Generate unique container name to avoid collisions
        unique_suffix = uuid.uuid4().hex[:8]
        self._container_name = f"{self._config.name}-{unique_suffix}"

        self._state = SandboxState()
        self._manager: ContainerManager | None = None

    @classmethod
    def from_spec(
        cls,
        client: DockerClient,
        spec: ContainerSpec,
    ) -> "DockerSandbox":
        """Create DockerSandbox from an existing ContainerSpec.

        Args:
            client: DockerClient instance.
            spec: ContainerSpec to use (must have unique name).

        Returns:
            DockerSandbox instance.
        """
        config = SandboxConfig(
            name=spec.name,
            image=spec.image,
            memory_mb=spec.memory_mb,
            cpu_limit=spec.cpu_limit,
            pids_limit=spec.pids_limit,
        )
        sandbox = cls(client, config)
        sandbox._container_name = spec.name
        return sandbox

    async def __aenter__(self) -> "DockerSandbox":
        """Create and start the sandbox container.

        Returns:
            Self for method chaining.

        Raises:
            DockerError: If container creation fails.
        """
        spec = ContainerSpec(
            name=self._container_name,
            image=self._config.image,
            working_dir=self._config.workspace_dir,
            command=["sleep", "7200"],  # Keep container running
            memory_mb=self._config.memory_mb,
            cpu_limit=self._config.cpu_limit,
            pids_limit=self._config.pids_limit,
        )

        self._manager = ContainerManager(self._client, spec)

        try:
            await self._manager.__aenter__()
            if self._manager.container:
                self._state.container_id = self._manager.container.id
        except Exception:
            self._manager = None
            raise

        logger.info(
            f"DockerSandbox created: container={self._container_name}, "
            f"image={self._config.image}"
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Cleanup the sandbox container.

        Guarantees container removal even on exceptions.
        """
        if self._manager:
            await self._manager.__aexit__(exc_type, exc_val, exc_tb)
            self._manager = None
            self._state.container_id = None

    def _require_running(self) -> str:
        """Assert the container is running and return its ID.

        Returns:
            Container ID string.

        Raises:
            DockerError: If container is not running.
        """
        if not self._manager or not self._state.container_id:
            raise DockerError(
                "Sandbox container not created - use async context manager"
            )

        if (
            self._manager.container
            and self._manager.container.status != ManagedContainerStatus.RUNNING
        ):
            raise DockerError(
                f"Sandbox container not running (status: {self._manager.container.status})",
                container_id=self._state.container_id,
            )

        return self._state.container_id

    def _normalize_repo_url(self, repo_url: str) -> str:
        """Normalize repository URL to HTTPS format.

        Supports:
        - HTTPS: https://github.com/owner/repo
        - SSH: git@github.com:owner/repo
        - Short: owner/repo

        Args:
            repo_url: Repository URL or short name.

        Returns:
            Normalized HTTPS URL.
        """
        repo_url = repo_url.strip()

        # Already HTTPS
        if repo_url.startswith("https://"):
            return repo_url

        # SSH format: git@github.com:owner/repo
        if repo_url.startswith("git@"):
            match = re.match(r"git@([^:]+):(.+)", repo_url)
            if match:
                host, path = match.groups()
                return f"https://{host}/{path}"

        # Short format: owner/repo
        if "/" in repo_url and not repo_url.startswith("/") and ":" not in repo_url:
            return f"https://github.com/{repo_url}"

        return repo_url

    async def setup_workspace(
        self,
        repo_url: str,
        commit: str | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        """Clone repository and checkout to specified commit.

        This method:
        1. Installs git if not present
        2. Clones the repository
        3. Checks out to specified commit (if provided)

        Args:
            repo_url: Repository URL (supports https://, git@, or owner/repo format).
            commit: Optional commit SHA to checkout.
            timeout: Optional override for clone timeout in seconds.

        Raises:
            DockerError: If clone or checkout fails.
        """
        container_id = self._require_running()
        normalized_url = self._normalize_repo_url(repo_url)
        clone_timeout = timeout or self._config.clone_timeout

        self._state.repo_url = normalized_url
        self._state.commit = commit

        # Install git (required for cloning)
        logger.debug(f"Installing git in container {self._container_name}")
        install_result = await self.run_command(
            "apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1",
            timeout=120.0,
        )
        if install_result.exit_code != 0:
            raise DockerError(
                f"Failed to install git: {install_result.stderr}",
                container_id=container_id,
                exit_code=install_result.exit_code,
            )

        # Clone the repository
        logger.info(f"Cloning repository: {normalized_url}")
        workspace_dir = self._config.workspace_dir
        clone_cmd = f"git clone {normalized_url}.git {workspace_dir} 2>&1"
        clone_result = await self.run_command(clone_cmd, timeout=clone_timeout)

        if clone_result.exit_code != 0:
            raise DockerError(
                f"Failed to clone repository: {self._truncate(clone_result.stderr, 500)}",
                container_id=container_id,
                exit_code=clone_result.exit_code,
            )

        # Checkout to specific commit if provided
        if commit:
            logger.info(f"Checking out commit: {commit}")
            checkout_cmd = f"cd {workspace_dir} && git checkout {commit} --force 2>&1"
            checkout_result = await self.run_command(
                checkout_cmd,
                timeout=self._config.checkout_timeout,
            )

            if checkout_result.exit_code != 0:
                raise DockerError(
                    f"Failed to checkout commit {commit}: "
                    f"{self._truncate(checkout_result.stderr, 500)}",
                    container_id=container_id,
                    exit_code=checkout_result.exit_code,
                )

        self._state.workspace_ready = True
        logger.info(f"Workspace ready: {workspace_dir}")

    async def run_command(
        self,
        cmd: str | Sequence[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Execute a command in the sandbox container.

        Args:
            cmd: Command to execute (string or list of arguments).
            cwd: Working directory (defaults to workspace_dir).
            timeout: Timeout in seconds (defaults to command_timeout).
            env: Environment variables for the command.

        Returns:
            ExecResult with exit_code, stdout, stderr, and duration.

        Raises:
            DockerError: If container is not running.
            asyncio.TimeoutError: If command execution exceeds timeout.
        """
        container_id = self._require_running()

        # Convert string command to list if needed
        if isinstance(cmd, str):
            cmd_list = ["bash", "-c", cmd]
        else:
            cmd_list = list(cmd)

        working_dir = cwd or self._config.workspace_dir
        cmd_timeout = timeout or self._config.command_timeout

        result = await exec_in_container(
            container_id,
            cmd_list,
            client=self._client,
            timeout=cmd_timeout,
            cwd=working_dir,
            env=env,
        )

        return result

    async def install_dependencies(
        self,
        packages: Sequence[str] | None = None,
        *,
        requirements_file: str | None = None,
        use_pip: bool = True,
        timeout: float | None = None,
    ) -> ExecResult:
        """Install dependencies in the sandbox.

        Supports pip install or apt-get install.

        Args:
            packages: List of package names to install.
            requirements_file: Path to requirements.txt (alternative to packages).
            use_pip: If True, use pip; if False, use apt-get.
            timeout: Override for install timeout in seconds.

        Returns:
            ExecResult from the install command.

        Raises:
            ValueError: If neither packages nor requirements_file is provided.
            DockerError: If container is not running.
        """
        if not packages and not requirements_file:
            raise ValueError("Must provide either packages or requirements_file")

        container_id = self._require_running()
        install_timeout = timeout or self._config.install_timeout
        workspace_dir = self._config.workspace_dir

        if requirements_file:
            cmd = f"pip install -r {requirements_file}"
        elif use_pip:
            packages_str = " ".join(packages)
            cmd = f"pip install {packages_str}"
        else:
            packages_str = " ".join(packages)
            cmd = f"apt-get update -qq && apt-get install -y -qq {packages_str}"

        logger.info(f"Installing dependencies: {cmd}")
        result = await self.run_command(cmd, timeout=install_timeout)

        if result.exit_code != 0:
            logger.warning(
                f"Dependency installation failed (exit {result.exit_code}): "
                f"{self._truncate(result.stderr, 200)}"
            )

        return result

    async def write_file(
        self,
        path: str,
        content: str,
    ) -> None:
        """Write a file to the sandbox container.

        Args:
            path: Path relative to workspace directory.
            content: File content to write.

        Raises:
            DockerError: If file write fails.
        """
        container_id = self._require_running()

        # Validate path doesn't contain traversal
        if ".." in path or path.startswith("/"):
            raise DockerError(f"Invalid path: {path} (must be relative to workspace)")

        workspace_dir = self._config.workspace_dir
        abs_path = f"{workspace_dir}/{path}"

        # Create parent directory
        parent_dir = "/".join(abs_path.split("/")[:-1])
        mkdir_result = await self.run_command(f"mkdir -p '{parent_dir}'")

        if mkdir_result.exit_code != 0:
            raise DockerError(
                f"Failed to create directory: {parent_dir}",
                container_id=container_id,
            )

        # Write file using cat with heredoc to avoid quoting issues
        # Using base64 encoding to handle special characters safely
        import base64

        encoded = base64.b64encode(content.encode()).decode()
        write_cmd = f"echo '{encoded}' | base64 -d > '{abs_path}'"

        result = await self.run_command(write_cmd)
        if result.exit_code != 0:
            raise DockerError(
                f"Failed to write file {path}: {result.stderr}",
                container_id=container_id,
            )

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox container.

        Args:
            path: Path relative to workspace directory.

        Returns:
            File content as string.

        Raises:
            DockerError: If file read fails.
        """
        container_id = self._require_running()

        # Validate path
        if ".." in path or path.startswith("/"):
            raise DockerError(f"Invalid path: {path} (must be relative to workspace)")

        abs_path = f"{self._config.workspace_dir}/{path}"
        result = await self.run_command(f"cat '{abs_path}'")

        if result.exit_code != 0:
            raise DockerError(
                f"Failed to read file {path}: {result.stderr}",
                container_id=container_id,
                exit_code=result.exit_code,
            )

        return result.stdout

    async def get_current_commit(self) -> str:
        """Get the current git commit SHA in the sandbox.

        Returns:
            Current commit SHA.

        Raises:
            DockerError: If git command fails.
        """
        container_id = self._require_running()
        result = await self.run_command("git rev-parse HEAD")

        if result.exit_code != 0:
            raise DockerError(
                f"Failed to get current commit: {result.stderr}",
                container_id=container_id,
                exit_code=result.exit_code,
            )

        return result.stdout.strip()

    async def get_git_status(self) -> str:
        """Get the git status in the sandbox.

        Returns:
            Git status output (porcelain format).

        Raises:
            DockerError: If git command fails.
        """
        container_id = self._require_running()
        result = await self.run_command("git status --porcelain")

        if result.exit_code != 0:
            raise DockerError(
                f"Failed to get git status: {result.stderr}",
                container_id=container_id,
                exit_code=result.exit_code,
            )

        return result.stdout.strip()

    @property
    def container_id(self) -> str | None:
        """Get the container ID, or None if not created."""
        return self._state.container_id

    @property
    def container_name(self) -> str:
        """Get the container name."""
        return self._container_name

    @property
    def workspace_dir(self) -> str:
        """Get the workspace directory path."""
        return self._config.workspace_dir

    @property
    def workspace_ready(self) -> bool:
        """Check if workspace has been set up."""
        return self._state.workspace_ready

    @property
    def repo_url(self) -> str | None:
        """Get the repository URL, if set."""
        return self._state.repo_url

    @property
    def commit(self) -> str | None:
        """Get the checked out commit, if set."""
        return self._state.commit

    @staticmethod
    def _truncate(s: str, max_len: int) -> str:
        """Truncate string to max length with ellipsis."""
        if len(s) <= max_len:
            return s
        return s[:max_len] + "..."

    @staticmethod
    def image_for_language(language: str) -> str:
        """Get appropriate Docker image for a language.

        Args:
            language: Programming language name.

        Returns:
            Docker image name.
        """
        language_lower = language.lower()

        image_map = {
            "python": "python:3.11-slim",
            "python3": "python:3.11-slim",
            "javascript": "node:20-slim",
            "typescript": "node:20-slim",
            "node": "node:20-slim",
            "nodejs": "node:20-slim",
            "go": "golang:1.21-slim",
            "golang": "golang:1.21-slim",
            "java": "eclipse-temurin:21-jdk-slim",
            "rust": "rust:1.75-slim",
            "ruby": "ruby:3.2-slim",
            "php": "php:8.2-cli-slim",
        }

        return image_map.get(language_lower, "python:3.11-slim")
