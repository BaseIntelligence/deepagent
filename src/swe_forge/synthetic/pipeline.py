"""Orchestration for synthetic feature-deletion task creation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from swe_forge.swe.models import SweTask, SweTaskStatus
from swe_forge.synthetic.feature_deletion import build_python_function_deletion
from swe_forge.synthetic.scoring import difficulty_label, estimate_patch_complexity


def _safe_task_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def _default_prompt(repo: str, source_file: Path, symbol: str) -> str:
    return (
        f"Restore the deleted behavior for `{symbol}` in `{source_file.as_posix()}` "
        f"inside `{repo}`. Use the repository tests to infer the expected behavior "
        "and keep the change minimal."
    )


def create_feature_deletion_task(
    *,
    repo_root: Path | str,
    repo: str,
    base_commit: str,
    source_file: Path | str,
    symbol: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str] | None = None,
    install_commands: list[str] | None = None,
    task_id: str | None = None,
    prompt: str | None = None,
    language: str = "python",
) -> SweTask:
    """Create a synthetic ``SweTask`` from a Python feature deletion."""
    deletion = build_python_function_deletion(repo_root, source_file, symbol)
    complexity = estimate_patch_complexity(deletion.oracle_patch)
    generated_task_id = task_id or _safe_task_id(
        f"{repo}-synthetic-feature-deletion-{base_commit[:8]}-{deletion.source_file}-{symbol}"
    )

    metadata = {
        "strategy": "feature_deletion",
        "feature_symbol": symbol,
        "source_file": deletion.source_file.as_posix(),
        "deletion_patch_file": "deletion_patch.diff",
        "oracle_patch_file": "patch.diff",
        "reward_source": "user_supplied_tests",
        "scoring": json.dumps(
            {
                "complexity_score": complexity,
                "difficulty": difficulty_label(complexity),
            },
            sort_keys=True,
        ),
    }

    return SweTask(
        id=generated_task_id,
        repo=repo,
        base_commit=base_commit,
        merge_commit=base_commit,
        language=language,
        source_type="synthetic_feature_deletion",
        deletion_patch=deletion.deletion_patch,
        patch=deletion.oracle_patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass or [],
        install_config={
            "install_commands": install_commands or [],
            "validated": bool(install_commands),
        },
        meta=metadata,
        prompt=prompt or _default_prompt(repo, deletion.source_file, symbol),
        dataset_prompt=prompt or _default_prompt(repo, deletion.source_file, symbol),
        original_pr_body=(
            "Synthetic feature-deletion task generated from a real repository. "
            "The solution is the inverse of the deletion patch."
        ),
        complexity_score=complexity,
        complexity_difficulty=difficulty_label(complexity),
        quality_score=complexity,
        quality_passed=complexity >= 0.05,
        status=SweTaskStatus.READY,
    )
