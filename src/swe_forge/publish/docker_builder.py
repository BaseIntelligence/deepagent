"""Docker image builder for SWE tasks."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Result of a Docker image build."""

    success: bool
    image_name: str | None = None
    task_id: str | None = None
    error: str | None = None
    push_url: str | None = None


def _generate_dockerfile(workspace: dict) -> str:
    """Generate Dockerfile from workspace.yaml content."""
    repo_info = workspace.get("repo", {})
    repo_url = repo_info.get("url", "")
    base_commit = repo_info.get("base_commit", "")
    language = workspace.get("language", "python")
    install_config = workspace.get("install", {})
    install_commands = install_config.get("commands", [])

    lines = [
        "FROM ubuntu:24.04",
        "RUN apt-get update && apt-get install -y git",
    ]

    if language == "python":
        lines.extend(
            [
                "RUN apt-get update && apt-get install -y python3 python3-pip python3-venv",
                "ENV VIRTUAL_ENV=/opt/venv",
                "RUN python3 -m venv $VIRTUAL_ENV",
                'ENV PATH="$VIRTUAL_ENV/bin:$PATH"',
            ]
        )
        for cmd in install_commands:
            if cmd and not cmd.startswith("#"):
                lines.append(f"RUN {cmd}")
    elif language in ("javascript", "typescript"):
        lines.extend(
            [
                "RUN apt-get update && apt-get install -y nodejs npm",
                "RUN npm install -g pnpm || true",
            ]
        )
        for cmd in install_commands:
            if cmd:
                lines.append(f"RUN {cmd}")
    elif language == "go":
        lines.append("RUN apt-get update && apt-get install -y golang-go")
    elif language == "rust":
        lines.extend(
            [
                "RUN apt-get update && apt-get install -y curl",
                "RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                'ENV PATH="/root/.cargo/bin:$PATH"',
            ]
        )
    else:
        lines.append("RUN apt-get update && apt-get install -y python3 python3-pip")

    if repo_url:
        lines.extend(
            [
                "WORKDIR /repo",
                f"RUN git clone {repo_url} .",
            ]
        )
        if base_commit:
            lines.append(f"RUN git checkout {base_commit}")

    return "\n".join(lines)


async def build_docker_image(
    task_dir: Path,
    docker_user: str,
    push: bool = False,
) -> BuildResult:
    """Build a Docker image for a single task."""
    task_id = task_dir.name
    workspace_path = task_dir / "workspace.yaml"

    if not workspace_path.exists():
        return BuildResult(
            success=False, task_id=task_id, error="workspace.yaml not found"
        )

    try:
        with open(workspace_path) as f:
            workspace = yaml.safe_load(f)

        dockerfile = _generate_dockerfile(workspace)
        image_name = f"{docker_user}/swe-forge:{task_id}"

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile)

            logger.info(f"Building {image_name}...")
            result = subprocess.run(
                ["docker", "build", "-t", image_name, "-f", str(dockerfile_path), "."],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                error_msg = (
                    result.stderr[:500] if result.stderr else "Unknown build error"
                )
                return BuildResult(
                    success=False, task_id=task_id, error=f"Build failed: {error_msg}"
                )

            push_url = None
            if push:
                logger.info(f"Pushing {image_name}...")
                push_result = subprocess.run(
                    ["docker", "push", image_name],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if push_result.returncode != 0:
                    push_error = (
                        push_result.stderr[:500]
                        if push_result.stderr
                        else "Push failed"
                    )
                    return BuildResult(
                        success=False,
                        task_id=task_id,
                        error=f"Push failed: {push_error}",
                    )
                push_url = f"https://hub.docker.com/r/{docker_user}/swe-forge"

            return BuildResult(
                success=True,
                image_name=image_name,
                task_id=task_id,
                push_url=push_url,
            )

    except Exception as e:
        return BuildResult(success=False, task_id=task_id, error=str(e))


async def build_docker_images(
    tasks_dir: Path,
    docker_user: str,
    push: bool = False,
    parallel: int = 4,
    limit: int | None = None,
) -> list[BuildResult]:
    """Build Docker images for all tasks."""
    task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir()])
    if limit:
        task_dirs = task_dirs[:limit]

    sem = asyncio.Semaphore(parallel)

    async def build_with_sem(task_dir: Path) -> BuildResult:
        async with sem:
            return await build_docker_image(task_dir, docker_user, push)

    results = await asyncio.gather(*[build_with_sem(d) for d in task_dirs])
    return list(results)
