"""Workspace export for SweTask to directory format."""

import re
from pathlib import Path
from typing import Any

import yaml

from swe_forge.swe.models import SweTask


def _extract_test_file_names(test_patch: str) -> list[str]:
    """Extract test file names from test_patch diff."""
    pattern = r"\+\+\+ b/(.*test.*\.py)"
    matches = re.findall(pattern, test_patch)
    return [Path(m).name for m in matches]


def export_task_to_workspace(
    task: SweTask,
    output_folder: Path | str,
    docker_username: str | None = None,
    prebuilt_image: bool = False,
) -> Path:
    """Export single SweTask to workspace directory format.

    Args:
        task: SweTask to export
        output_folder: Directory to export to
        docker_username: Docker Hub username for image names
        prebuilt_image: If True, indicates image is pre-built (build: false)

    Returns:
        Path to the task directory
    """
    output_folder = Path(output_folder)
    task_dir = output_folder / task.id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Docker image
    docker_image = (
        f"{docker_username}/swe-forge-tasks:{task.id}" if docker_username else None
    )

    # Get install commands from config or defaults
    install_commands = task.install_config.get("install_commands", [])
    if not install_commands:
        if task.language == "python":
            install_commands = ["pip install -e .", "pip install pytest"]
        elif task.language in ("javascript", "typescript"):
            install_commands = ["npm install", "npm test"]
        elif task.language == "rust":
            install_commands = ["cargo build", "cargo test"]

    # Get test commands - fallback to test_patch extraction if empty
    fail_to_pass = list(task.fail_to_pass) if task.fail_to_pass else []
    pass_to_pass = list(task.pass_to_pass) if task.pass_to_pass else []

    # Extract test commands from test_patch if available
    if not fail_to_pass and task.test_patch:
        test_files = _extract_test_file_names(task.test_patch)
        if test_files:
            fail_to_pass = [f"pytest {f} -v" for f in test_files]

    # Default test commands as last resort
    if not fail_to_pass:
        if task.language == "python":
            fail_to_pass = ["pytest tests/ -v"]
        elif task.language in ("javascript", "typescript"):
            fail_to_pass = ["npm test"]
        elif task.language == "rust":
            fail_to_pass = ["cargo test"]

    # Build workspace data
    workspace_data: dict[str, Any] = {
        "task_id": task.id,
        "repo": {
            "url": f"https://github.com/{task.repo}.git",
            "base_commit": task.base_commit,
            "merge_commit": task.merge_commit,
        },
        "language": task.language,
        "difficulty_score": task.difficulty_score,
        "prompt": task.prompt,
        "environment": {
            "image": docker_image or "ubuntu:24.04",
            "language_version": (
                task.install_config.get("language_version", "3.12")
                if task.language == "python"
                else "unknown"
            ),
        },
        "install": {
            "commands": install_commands,
        },
        "tests": {
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
        },
    }

    # Add docker section if image specified
    if docker_image:
        workspace_data["docker"] = {
            "image": docker_image,
            "build": not prebuilt_image,  # False if pre-built, True if needs building
            "prebuilt": prebuilt_image,
        }

    if task.meta:
        workspace_data["meta"] = task.meta

    # Write workspace.yaml
    workspace_path = task_dir / "workspace.yaml"
    with open(workspace_path, "w", encoding="utf-8") as f:
        yaml.dump(
            workspace_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    # Write patch
    if task.patch:
        patch_path = task_dir / "patch.diff"
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(task.patch)

    # Write test patch
    if task.test_patch:
        test_patch_path = task_dir / "test_patch.diff"
        with open(test_patch_path, "w", encoding="utf-8") as f:
            f.write(task.test_patch)

        # Extract test files
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        _extract_test_files(task.test_patch, tests_dir)

    return task_dir


def export_tasks_to_workspace(
    tasks: list[SweTask],
    output_folder: Path | str,
    docker_username: str | None = None,
    prebuilt_images: bool = False,
) -> list[Path]:
    """Export list of SweTask to workspace directories.

    Args:
        tasks: List of SweTask to export
        output_folder: Directory to export to
        docker_username: Docker Hub username for image names
        prebuilt_images: If True, indicates images are pre-built

    Returns:
        List of paths to task directories
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    return [
        export_task_to_workspace(task, output_folder, docker_username, prebuilt_images)
        for task in tasks
    ]


def _extract_test_files(test_patch: str, tests_dir: Path) -> None:
    """Extract test files from diff format."""
    pattern = r"#\s*(.+\.py)\n(.+?)(?=#\s*.+\.py\n|$)"
    matches = re.findall(pattern, test_patch, re.DOTALL)

    for file_path, content in matches:
        file_path = file_path.strip()
        if not file_path:
            continue
        test_file = tests_dir / Path(file_path).name
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")


def update_workspace_with_prebuilt_image(
    workspace_path: Path | str,
    image_name: str,
) -> None:
    """Update workspace.yaml to mark Docker image as pre-built.

    Args:
        workspace_path: Path to workspace.yaml file
        image_name: Name of the pre-built Docker image
    """
    workspace_path = Path(workspace_path)
    if not workspace_path.exists():
        return

    with open(workspace_path, "r") as f:
        data = yaml.safe_load(f)

    # Update docker section
    data["docker"] = {
        "image": image_name,
        "build": False,
        "prebuilt": True,
    }

    # Update environment image
    data["environment"]["image"] = image_name

    with open(workspace_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
