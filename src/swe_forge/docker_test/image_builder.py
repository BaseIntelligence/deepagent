"""Build Docker images for tasks with pre-installed dependencies.

This module provides functionality to pre-build Docker images containing:
- The repository cloned at base_commit
- All dependencies installed
- Ready for testing (just apply patch + run tests)

Pre-built images dramatically speed up evaluation since dependencies
are already installed.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swe_forge.execution.docker_client import DockerClient

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of building a Docker image."""

    success: bool
    image_name: str | None = None
    error: str | None = None
    duration_seconds: float = 0.0


def generate_dockerfile(
    task: dict[str, Any],
    base_image: str = "ubuntu:24.04",
) -> str:
    """Generate a Dockerfile for a task with pre-installed dependencies.

    Args:
        task: Task dictionary with repo, base_commit, install config
        base_image: Base Docker image to use

    Returns:
        Dockerfile content as string
    """
    repo_info = task.get("repo", {})
    repo_url = repo_info.get("url", "")
    base_commit = repo_info.get("base_commit", "")
    language = task.get("language", "python")

    # If repo_url is "owner/repo" format, convert to full URL
    if repo_url and not repo_url.startswith("http") and "/" in repo_url:
        repo_url = f"https://github.com/{repo_url}.git"

    # Get install commands from various possible locations
    install_config = task.get("install_config", task.get("install", {}))
    install_commands = install_config.get(
        "install_commands", install_config.get("commands", [])
    )

    # Build Dockerfile lines
    lines = [
        f"FROM {base_image}",
        "WORKDIR /repo",
        "",
        "# Install system dependencies",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\",
        "    git \\",
        "    curl \\",
        "    ca-certificates \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
    ]

    # Language-specific setup
    if language == "python":
        lines.extend(
            [
                "# Python setup",
                "RUN apt-get update && apt-get install -y --no-install-recommends \\",
                "    python3 \\",
                "    python3-pip \\",
                "    python3-venv \\",
                "    && rm -rf /var/lib/apt/lists/*",
                "",
                "# Create virtual environment",
                "ENV VIRTUAL_ENV=/opt/venv",
                "RUN python3 -m venv $VIRTUAL_ENV",
                'ENV PATH="$VIRTUAL_ENV/bin:$PATH"',
                "",
                "# Upgrade pip",
                "RUN pip install --upgrade pip",
                "",
            ]
        )
    elif language in ("javascript", "typescript"):
        lines.extend(
            [
                "# Node.js setup",
                "RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \\",
                "    && apt-get install -y --no-install-recommends nodejs \\",
                "    && rm -rf /var/lib/apt/lists/*",
                "",
            ]
        )
    elif language == "rust":
        lines.extend(
            [
                "# Rust setup",
                "RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                'ENV PATH="/root/.cargo/bin:$PATH"',
                "",
            ]
        )
    elif language == "go":
        lines.extend(
            [
                "# Go setup",
                "RUN apt-get update && apt-get install -y --no-install-recommends golang-go \\",
                "    && rm -rf /var/lib/apt/lists/*",
                "",
            ]
        )

    # Clone repository at base commit
    if repo_url:
        lines.extend(
            [
                "# Clone repository",
                f"RUN git clone {repo_url} /repo",
                "WORKDIR /repo",
            ]
        )
        if base_commit:
            lines.append(f"RUN git checkout {base_commit}")
        lines.append("")

    # Run install commands
    if install_commands:
        lines.append("# Install project dependencies")
        for cmd in install_commands:
            if cmd.strip() and not cmd.strip().startswith("#"):
                lines.append(f"RUN cd /repo && {cmd}")
        lines.append("")

    # Set final working directory
    lines.extend(
        [
            "WORKDIR /repo",
            "",
            "# Image is ready for testing",
            "# To test: apply patch, then run test commands",
        ]
    )

    return "\n".join(lines)


async def build_task_image(
    docker_client: "DockerClient",
    task: dict[str, Any],
    image_name: str,
    *,
    base_image: str = "ubuntu:24.04",
    timeout: float = 600.0,
    push: bool = False,
) -> BuildResult:
    """Build a Docker image for a task with repo + deps pre-installed.

    Args:
        docker_client: Docker client instance
        task: Task dictionary with repo, base_commit, install config
        image_name: Image name to tag (e.g., "user/swe-forge-tasks:task-123")
        base_image: Base image to build from
        timeout: Build timeout in seconds
        push: Whether to push image to registry after build

    Returns:
        BuildResult with success status and image name or error
    """
    import time

    start_time = time.monotonic()

    # Generate Dockerfile
    dockerfile = generate_dockerfile(task, base_image=base_image)
    logger.debug(f"Generated Dockerfile:\n{dockerfile}")

    # Create temporary build context
    with tempfile.TemporaryDirectory() as tmpdir:
        context_path = Path(tmpdir)
        dockerfile_path = context_path / "Dockerfile"
        dockerfile_path.write_text(dockerfile)

        try:
            # Run docker build
            logger.info(f"Building image: {image_name}")
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "build",
                "-t",
                image_name,
                "-f",
                str(dockerfile_path),
                str(context_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return BuildResult(
                    success=False,
                    image_name=image_name,
                    error=f"Build timed out after {timeout}s",
                    duration_seconds=time.monotonic() - start_time,
                )

            duration = time.monotonic() - start_time

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown build error"
                logger.error(f"Docker build failed for {image_name}: {error_msg}")
                return BuildResult(
                    success=False,
                    image_name=image_name,
                    error=error_msg,
                    duration_seconds=duration,
                )

            logger.info(f"Successfully built image: {image_name} in {duration:.1f}s")

            # Optionally push to registry
            if push:
                logger.info(f"Pushing image: {image_name}")
                push_proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "push",
                    image_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                push_stdout, push_stderr = await push_proc.communicate()

                if push_proc.returncode != 0:
                    push_error = push_stderr.decode() if push_stderr else "Push failed"
                    logger.warning(f"Failed to push image {image_name}: {push_error}")
                    return BuildResult(
                        success=False,
                        image_name=image_name,
                        error=f"Build succeeded but push failed: {push_error}",
                        duration_seconds=duration,
                    )

                logger.info(f"Successfully pushed image: {image_name}")

            return BuildResult(
                success=True,
                image_name=image_name,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Exception building image {image_name}: {e}")
            return BuildResult(
                success=False,
                image_name=image_name,
                error=str(e),
                duration_seconds=time.monotonic() - start_time,
            )


async def build_images_for_tasks(
    docker_client: "DockerClient",
    tasks: list[dict[str, Any]],
    docker_username: str,
    *,
    base_image: str = "ubuntu:24.04",
    timeout: float = 600.0,
    push: bool = False,
    parallel: int = 2,
) -> list[BuildResult]:
    """Build Docker images for multiple tasks.

    Args:
        docker_client: Docker client instance
        tasks: List of task dictionaries
        docker_username: Docker Hub username for image names
        base_image: Base image to build from
        timeout: Build timeout per image
        push: Whether to push images after building
        parallel: Maximum parallel builds

    Returns:
        List of BuildResult for each task
    """
    results: list[BuildResult] = []
    semaphore = asyncio.Semaphore(parallel)

    async def build_one(task: dict[str, Any]) -> BuildResult:
        task_id = task.get("id", "unknown")
        image_name = f"{docker_username}/swe-forge-tasks:{task_id}"

        async with semaphore:
            return await build_task_image(
                docker_client,
                task,
                image_name,
                base_image=base_image,
                timeout=timeout,
                push=push,
            )

    # Build all images concurrently (respecting semaphore)
    results = await asyncio.gather(*[build_one(t) for t in tasks])

    return list(results)


def task_to_dict(task: Any) -> dict[str, Any]:
    """Convert a SweTask object to dictionary for image building.

    Args:
        task: SweTask instance or dict

    Returns:
        Task as dictionary
    """
    if hasattr(task, "model_dump"):
        return task.model_dump()
    elif hasattr(task, "to_dict"):
        return task.to_dict()
    elif isinstance(task, dict):
        return task
    else:
        raise ValueError(f"Cannot convert {type(task)} to dict")
