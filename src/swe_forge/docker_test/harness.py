"""Docker test harness for building test images and running tests.

Provides:
- DockerTestHarness: Build test images, run tests, apply patches
- TestRunResult: Result of running test commands in a container
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from swe_forge.execution.docker_client import DockerClient, DockerError
from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


DEFAULT_TEST_IMAGE = "ubuntu:24.04"
DEFAULT_TEST_TIMEOUT = 600.0


@dataclass
class TestRunResult:
    """Result of running test commands."""

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    command: str
    duration_seconds: float = 0.0

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"TestRunResult({status}, exit={self.exit_code}, cmd={self.command[:50]}...)"


@dataclass
class BuildResult:
    """Result of building a Docker test image."""

    success: bool
    image_id: str | None = None
    error: str | None = None


class DockerTestHarness:
    """Docker test harness for building and running tests.

    Provides methods to:
    - Build test images from Dockerfile or workspace config
    - Run test commands in containers
    - Apply patches to containers
    - Execute before/after test verification

    Usage:
        async with DockerClient() as client:
            harness = DockerTestHarness(client)
            image_id = await harness.build_test_image(workspace_config)
            result = await harness.run_tests(image_id, tests)
    """

    def __init__(
        self,
        docker_client: DockerClient,
        *,
        default_image: str = DEFAULT_TEST_IMAGE,
        default_timeout: float = DEFAULT_TEST_TIMEOUT,
    ) -> None:
        self.docker_client = docker_client
        self.default_image = default_image
        self.default_timeout = default_timeout

    async def build_test_image(
        self,
        workspace: dict[str, Any] | None = None,
        dockerfile: str | None = None,
        context_dir: str | Path | None = None,
    ) -> str:
        """Build a Docker image for testing.

        Args:
            workspace: Workspace configuration dict (alternative to Dockerfile)
            dockerfile: Dockerfile content (if provided, builds from it)
            context_dir: Build context directory

        Returns:
            Image ID or name

        Raises:
            DockerError: If build fails
        """
        if dockerfile:
            return await self._build_from_dockerfile(dockerfile, context_dir)

        if workspace:
            return await self._build_from_workspace(workspace)

        return await self._ensure_base_image()

    async def _build_from_dockerfile(
        self,
        dockerfile: str,
        context_dir: str | Path | None = None,
    ) -> str:
        """Build image from Dockerfile string."""

        image_tag = f"swe-test-{uuid.uuid4().hex[:8]}"

        if context_dir:
            context_path = Path(context_dir)
        else:
            context_path = Path(tempfile.mkdtemp())

        dockerfile_path = context_path / "Dockerfile"
        dockerfile_path.write_text(dockerfile)

        try:
            build_result = await self._build_image(context_path, image_tag)
            if build_result.success:
                return image_tag
            raise DockerError(f"Build failed: {build_result.error}")
        finally:
            if not context_dir:
                import shutil

                shutil.rmtree(context_path, ignore_errors=True)

    async def _build_from_workspace(self, workspace: dict[str, Any]) -> str:
        """Build image from workspace configuration."""
        _image_tag = f"swe-test-{uuid.uuid4().hex[:8]}"

        base_image = workspace.get("environment", {}).get("image", self.default_image)
        language = workspace.get("language", "python")

        dockerfile_lines = [
            f"FROM {base_image}",
            "WORKDIR /repo",
            "RUN apt-get update && apt-get install -y git curl",
        ]

        install_config = workspace.get("install", {})
        system_packages = install_config.get("system_packages", [])
        if system_packages:
            packages_str = " ".join(system_packages)
            dockerfile_lines.append(f"RUN apt-get install -y {packages_str}")

        if language == "python":
            dockerfile_lines.extend(
                [
                    "RUN apt-get install -y python3 python3-pip python3-venv",
                    "ENV VIRTUAL_ENV=/opt/venv",
                    "RUN python3 -m venv $VIRTUAL_ENV",
                    'ENV PATH="$VIRTUAL_ENV/bin:$PATH"',
                ]
            )

        for cmd in install_config.get("commands", []):
            dockerfile_lines.append(f"RUN {cmd}")

        dockerfile = "\n".join(dockerfile_lines)
        return await self._build_from_dockerfile(dockerfile)

    async def _build_image(self, context_path: Path, tag: str) -> BuildResult:
        """Build Docker image from context path."""
        try:
            import asyncio

            proc = await asyncio.create_subprocess_exec(
                "docker",
                "build",
                "-t",
                tag,
                str(context_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                return BuildResult(success=True, image_id=tag)
            return BuildResult(success=False, error=stderr.decode())
        except Exception as e:
            return BuildResult(success=False, error=str(e))

    async def _ensure_base_image(self) -> str:
        """Ensure base image is available, pulling if needed."""
        await self.docker_client.ensure_image(self.default_image)
        return self.default_image

    async def run_tests(
        self,
        image: str,
        commands: Sequence[str],
        *,
        repo_url: str | None = None,
        commit: str | None = None,
        timeout: float | None = None,
        install_commands: Sequence[str] | None = None,
    ) -> TestRunResult:
        """Run test commands in a Docker container.

        Args:
            image: Docker image to use
            commands: Test commands to run (joined with &&)
            repo_url: Optional repo to clone
            commit: Optional commit to checkout
            timeout: Timeout in seconds
            install_commands: Commands to run before tests

        Returns:
            TestRunResult with pass/fail status
        """
        config = SandboxConfig(
            image=image,
            command_timeout=timeout or self.default_timeout,
            clone_timeout=600.0,
            install_timeout=300.0,
        )

        sandbox = DockerSandbox(self.docker_client, config)

        async with sandbox:
            if repo_url:
                await sandbox.setup_workspace(repo_url, commit)

            if install_commands:
                for cmd in install_commands:
                    result = await sandbox.run_command(
                        cmd, timeout=self.default_timeout
                    )
                    if result.exit_code != 0:
                        logger.warning(f"Install command failed: {cmd}")

            test_command = " && ".join(commands) if len(commands) > 1 else commands[0]

            import time

            start = time.monotonic()
            result = await sandbox.run_command(
                test_command, timeout=timeout or self.default_timeout
            )
            duration = time.monotonic() - start

            return TestRunResult(
                passed=result.exit_code == 0,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                command=test_command,
                duration_seconds=duration,
            )

    async def apply_patch(
        self,
        sandbox: DockerSandbox,
        patch: str,
    ) -> bool:
        """Apply a patch to a running sandbox.

        Args:
            sandbox: Running DockerSandbox instance
            patch: Unified diff patch content

        Returns:
            True if patch applied successfully
        """
        import uuid

        patch_path = f".swe-patch-{uuid.uuid4().hex[:8]}.diff"

        try:
            await sandbox.write_file(patch_path, patch)

            result = await sandbox.run_command(f"git apply {patch_path} 2>&1")

            if result.exit_code != 0:
                logger.warning(
                    f"Patch apply failed: exit={result.exit_code} out={result.stdout[:200]}"
                )
                return False

            await sandbox.run_command(f"rm -f {patch_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply patch: {e}")
            return False

    async def run_test_with_patch(
        self,
        image: str,
        repo_url: str,
        base_commit: str,
        patch: str,
        test_commands: Sequence[str],
        *,
        timeout: float | None = None,
        install_commands: Sequence[str] | None = None,
    ) -> TestRunResult:
        """Clone repo, apply patch, run tests.

        Args:
            image: Docker image to use
            repo_url: Repository URL
            base_commit: Base commit SHA
            patch: Unified diff patch
            test_commands: Commands to run
            timeout: Test timeout
            install_commands: Install commands before patch

        Returns:
            TestRunResult
        """
        config = SandboxConfig(
            image=image,
            command_timeout=timeout or self.default_timeout,
        )

        sandbox = DockerSandbox(self.docker_client, config)

        async with sandbox:
            await sandbox.setup_workspace(repo_url, base_commit)

            if install_commands:
                for cmd in install_commands:
                    await sandbox.run_command(cmd, timeout=self.default_timeout)

            patch_applied = await self.apply_patch(sandbox, patch)
            if not patch_applied:
                return TestRunResult(
                    passed=False,
                    exit_code=-1,
                    stdout="",
                    stderr="Failed to apply patch",
                    command="git apply",
                )

            test_command = (
                " && ".join(test_commands)
                if len(test_commands) > 1
                else test_commands[0]
            )
            result = await sandbox.run_command(
                test_command, timeout=timeout or self.default_timeout
            )

            return TestRunResult(
                passed=result.exit_code == 0,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                command=test_command,
            )
