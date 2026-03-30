"""Docker image builder for SWE tasks."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _update_workspace_image(workspace_path: Path, image_name: str) -> None:
    """Update workspace.yaml with the built Docker image name."""
    with open(workspace_path) as f:
        workspace = yaml.safe_load(f)
    
    # Update the environment.image field
    if "environment" not in workspace:
        workspace["environment"] = {}
    workspace["environment"]["image"] = image_name
    
    with open(workspace_path, 'w') as f:
        yaml.dump(workspace, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Updated {workspace_path} with image: {image_name}")


@dataclass
class BuildResult:
    """Result of a Docker image build."""

    success: bool
    image_name: str | None = None
    task_id: str | None = None
    error: str | None = None
    push_url: str | None = None


def _generate_run_tests_script(workspace: dict) -> str:
    """Generate run_tests.sh script from workspace.yaml."""
    tests = workspace.get("tests", {})
    fail_to_pass = tests.get("fail_to_pass", [])
    pass_to_pass = tests.get("pass_to_pass", [])
    
    script = '''#!/bin/bash
set -e

echo "=== Running SWE-Forge Tests ==="
echo "Task: ''' + workspace.get("task_id", "unknown") + '''"
echo ""

# Check if patch needs to be applied
if [ -f /workspace/patch.diff ]; then
    echo "Applying patch..."
    cd /repo
    git apply /workspace/patch.diff || echo "Patch already applied or failed"
    echo ""
fi

'''
    
    if fail_to_pass:
        script += 'echo "=== Running fail_to_pass tests ==="\n'
        for test in fail_to_pass:
            script += f'echo "Running: {test}"\n'
            script += f'{test}\n'
        script += '\n'
    
    if pass_to_pass:
        script += 'echo "=== Running pass_to_pass tests ==="\n'
        for test in pass_to_pass:
            script += f'echo "Running: {test}"\n'
            script += f'{test}\n'
    
    script += '''
echo ""
echo "=== All tests completed ==="
'''
    return script


def _generate_dockerfile(workspace: dict, task_dir: Path | None = None) -> str:
    """Generate Dockerfile from workspace.yaml content with tests included."""
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
        lines.extend([
            "RUN apt-get update && apt-get install -y python3 python3-pip python3-venv",
            "ENV VIRTUAL_ENV=/opt/venv",
            "RUN python3 -m venv $VIRTUAL_ENV",
            'ENV PATH="$VIRTUAL_ENV/bin:$PATH"',
        ])
        for cmd in install_commands:
            if cmd and not cmd.startswith("#"):
                lines.append(f"RUN {cmd}")
    elif language in ("javascript", "typescript"):
        lines.extend([
            "RUN apt-get update && apt-get install -y nodejs npm",
            "RUN npm install -g pnpm || true",
        ])
        for cmd in install_commands:
            if cmd:
                lines.append(f"RUN {cmd}")
    elif language == "go":
        lines.append("RUN apt-get update && apt-get install -y golang-go")
    elif language == "rust":
        lines.extend([
            "RUN apt-get update && apt-get install -y curl",
            "RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            'ENV PATH="/root/.cargo/bin:$PATH"',
        ])
    else:
        lines.append("RUN apt-get update && apt-get install -y python3 python3-pip")

    if repo_url:
        lines.extend([
            "WORKDIR /repo",
            f"RUN git clone {repo_url} .",
        ])
        if base_commit:
            lines.append(f"RUN git checkout {base_commit}")

    # Add workspace directory with tests
    lines.extend([
        "",
        "# Create workspace directory with tests",
        "RUN mkdir -p /workspace/tests",
        "WORKDIR /repo",
    ])

    return "\n".join(lines)


async def build_docker_image(
    task_dir: Path,
    docker_user: str,
    push: bool = False,
) -> BuildResult:
    """Build a Docker image for a single task with tests included."""
    task_id = task_dir.name
    workspace_path = task_dir / "workspace.yaml"

    if not workspace_path.exists():
        return BuildResult(
            success=False, task_id=task_id, error="workspace.yaml not found"
        )

    try:
        with open(workspace_path) as f:
            workspace = yaml.safe_load(f)

        image_name = f"{docker_user}/swe-forge:{task_id}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Generate Dockerfile
            dockerfile = _generate_dockerfile(workspace, task_dir)
            dockerfile_path = tmpdir_path / "Dockerfile"
            dockerfile_path.write_text(dockerfile)

            # Create workspace directory structure in build context
            workspace_build = tmpdir_path / "workspace"
            workspace_build.mkdir()
            tests_build = workspace_build / "tests"
            tests_build.mkdir()

            # Copy tests directory if exists
            tests_src = task_dir / "tests"
            if tests_src.exists() and tests_src.is_dir():
                for test_file in tests_src.iterdir():
                    if test_file.is_file():
                        shutil.copy(test_file, tests_build / test_file.name)

            # Copy workspace.yaml
            shutil.copy(workspace_path, workspace_build / "workspace.yaml")

            # Copy patch.diff if exists
            patch_src = task_dir / "patch.diff"
            if patch_src.exists():
                shutil.copy(patch_src, workspace_build / "patch.diff")

            # Generate run_tests.sh
            run_tests_script = _generate_run_tests_script(workspace)
            run_tests_path = workspace_build / "run_tests.sh"
            run_tests_path.write_text(run_tests_script)

            # Create Dockerfile that copies workspace
            dockerfile_with_copy = dockerfile + '''

# Copy workspace files with tests
COPY workspace/ /workspace/
RUN chmod +x /workspace/run_tests.sh
'''
            dockerfile_path.write_text(dockerfile_with_copy)

            logger.info(f"Building {image_name}...")
            result = subprocess.run(
                ["docker", "build", "-t", image_name, "-f", str(dockerfile_path), str(tmpdir_path)],
                capture_output=True,
                text=True,
                timeout=900,
            )

            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "Unknown build error"
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
                    push_error = push_result.stderr[:500] if push_result.stderr else "Push failed"
                    return BuildResult(
                        success=False,
                        task_id=task_id,
                        error=f"Push failed: {push_error}",
                    )
                push_url = f"https://hub.docker.com/r/{docker_user}/swe-forge"

            # Update workspace.yaml with the built image name
            _update_workspace_image(workspace_path, image_name)

            return BuildResult(
                success=True,
                image_name=image_name,
                task_id=task_id,
                push_url=push_url,
            )

    except Exception as e:
        logger.error(f"Error building {task_id}: {e}")
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
