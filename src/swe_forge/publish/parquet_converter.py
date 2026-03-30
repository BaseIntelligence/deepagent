"""Convert tasks to Parquet format for HuggingFace."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_workspace(workspace_path: Path) -> dict:
    """Load a workspace.yaml file."""
    with open(workspace_path) as f:
        return yaml.safe_load(f)


def tasks_to_records(tasks_dir: Path) -> list[dict]:
    """Convert all tasks in directory to list of records."""
    records = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        workspace_path = task_dir / "workspace.yaml"
        if not workspace_path.exists():
            continue

        try:
            ws = load_workspace(workspace_path)
            repo_info = ws.get("repo", {})
            tests_info = ws.get("tests", {})
            install_info = ws.get("install", {})

            record = {
                "task_id": ws.get("task_id", task_dir.name),
                "repo_url": repo_info.get("url", ""),
                "base_commit": repo_info.get("base_commit", ""),
                "merge_commit": repo_info.get("merge_commit", ""),
                "language": ws.get("language", ""),
                "difficulty_score": ws.get("difficulty_score", 0),
                "prompt": ws.get("prompt", ""),
                "docker_image": ws.get("environment", {}).get("image", "ubuntu:24.04"),
                "fail_to_pass": json.dumps(tests_info.get("fail_to_pass", [])),
                "pass_to_pass": json.dumps(tests_info.get("pass_to_pass", [])),
                "install_commands": json.dumps(install_info.get("commands", [])),
            }

            patch_path = task_dir / "patch.diff"
            record["patch"] = patch_path.read_text() if patch_path.exists() else ""

            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to load {workspace_path}: {e}")

    return records


def convert_tasks_to_parquet(tasks_dir: Path, output_path: Path | None = None) -> Path:
    """Convert tasks directory to Parquet file."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for Parquet conversion. Install with: pip install pandas pyarrow"
        )

    records = tasks_to_records(tasks_dir)
    if not records:
        raise ValueError(f"No tasks found in {tasks_dir}")

    df = pd.DataFrame(records)

    if output_path is None:
        output_path = tasks_dir / "swe-forge.parquet"

    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Wrote {len(records)} tasks to {output_path}")

    return output_path
