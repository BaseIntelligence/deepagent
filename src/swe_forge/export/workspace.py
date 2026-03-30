"""Workspace export for SweTask to directory format."""

import re
from logging import getLogger
from pathlib import Path
from typing import Any

import yaml

from swe_forge.exceptions import DiscoveryError
from swe_forge.export.docker_verify import generate_run_script
from swe_forge.swe.models import SweTask

logger = getLogger(__name__)


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

    # Get install commands from config
    install_commands = task.install_config.get("install_commands", [])
    # If no install commands, skip Dockerfile generation but continue
    has_install_commands = bool(install_commands)

    # Get test commands - fallback to test_patch extraction if empty
    fail_to_pass = list(task.fail_to_pass) if task.fail_to_pass else []
    pass_to_pass = list(task.pass_to_pass) if task.pass_to_pass else []

    # Extract test commands from test_patch if available
    if not fail_to_pass and task.test_patch:
        test_files = _extract_test_file_names(task.test_patch)
        if test_files:
            fail_to_pass = [f"pytest {f} -v" for f in test_files]

    # Warn if no test commands but continue
    if not fail_to_pass and not task.test_patch:
        logger.warning(f"No test commands for task {task.id}")

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
        "prompt": task.dataset_prompt if task.dataset_prompt else task.prompt,
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

    # Generate run_tests.sh script for easy Docker verification
    try:
        generate_run_script(task_dir)
    except Exception as e:
        logger.warning(f"Could not generate run script: {e}")

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
    import shutil
    from logging import getLogger

    logger = getLogger(__name__)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        task_dir = output_folder / task.id
        try:
            path = export_task_to_workspace(
                task, output_folder, docker_username, prebuilt_images
            )
            results.append(path)
        except DiscoveryError as e:
            logger.warning(f"Skipping task {task.id}: {e}")
            # Clean up incomplete directory
            if task_dir.exists():
                shutil.rmtree(task_dir)
        except Exception as e:
            logger.error(f"Error exporting task {task.id}: {e}")
            # Clean up incomplete directory
            if task_dir.exists():
                shutil.rmtree(task_dir)

    return results


def _extract_test_files(test_patch: str, tests_dir: Path) -> None:
    """Extract test files from test_patch.

    Handles multiple formats:
    1. "# Test file: path" format (from TestGenerator) - PRIMARY
    2. Git diff format (diff --git ... +lines)
    3. Truncate long filenames to avoid filesystem errors
    """
    tests_dir.mkdir(parents=True, exist_ok=True)

    def safe_filename(name: str, max_len: int = 200) -> str:
        """Create safe filename, truncating if too long."""
        # Remove invalid characters
        safe = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Truncate to max length
        if len(safe) > max_len:
            safe = safe[:max_len]
        return safe

    # Format 1: "# Test file: path" - Split by markers
    if "# Test file:" in test_patch or "#Test file:" in test_patch:
        parts = re.split(r"#\s*Test file:\s*", test_patch)
        for part in parts:
            if not part.strip():
                continue
            # First line is filename, rest is content
            lines = part.split("\n", 1)
            filename = lines[0].strip()
            content = lines[1] if len(lines) > 1 else ""
            
            if not filename or not content.strip():
                continue
            
            # Use safe filename
            safe_name = safe_filename(filename)
            if not safe_name.endswith(".py"):
                safe_name += ".py"
            
            test_file = tests_dir / safe_name
            try:
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write(content.strip() + "\n")
                logger.debug(f"Extracted test file: {test_file}")
            except OSError as e:
                logger.warning(f"Could not write test file {safe_name}: {e}")
        return

    # Format 2: Git diff format
    if "diff --git" in test_patch:
        diffs = test_patch.split("diff --git")
        for diff in diffs[1:]:
            plus_match = re.search(r"\+\+\+ b/(.+?)(?:\n|$)", diff)
            if not plus_match:
                continue
            file_path = plus_match.group(1).strip()

            content_lines = []
            in_hunk = False

            for line in diff.split("\n"):
                if line.startswith("---") or line.startswith("+++"):
                    continue
                if line.startswith("index") or line.startswith("new file"):
                    continue
                if line.startswith("@@"):
                    in_hunk = True
                    continue
                if not in_hunk:
                    continue
                if line.startswith("+"):
                    content_lines.append(line[1:])

            if content_lines:
                while content_lines and not content_lines[-1].strip():
                    content_lines.pop()
                
                safe_name = safe_filename(Path(file_path).name)
                test_file = tests_dir / safe_name
                try:
                    with open(test_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(content_lines))
                        if content_lines:
                            f.write("\n")
                except OSError as e:
                    logger.warning(f"Could not write test file {safe_name}: {e}")
        return

    # Fallback: Write entire test_patch as test_swe_generated.py
    if test_patch.strip():
        test_file = tests_dir / "test_swe_generated.py"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_patch)


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
