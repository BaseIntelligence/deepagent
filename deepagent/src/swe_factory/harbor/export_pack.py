"""Emit complete DeepAgent/Harbor pack directory trees.

VAL-HARBOR-001: task.toml, instruction.md, pre_artifacts.sh,
environment/Dockerfile, tests/{Dockerfile,test.sh,grader.py,config.json,test.patch},
solution/{solution.patch,solve.sh}.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_factory.harbor.grader_frame import (
    default_solve_sh,
    default_test_sh,
    render_grader_py,
)
from swe_factory.harbor.pre_artifacts import render_pre_artifacts_sh
from swe_factory.harbor.schema import (
    HarborPackSpec,
    render_task_toml,
    validate_pack_spec,
)

REQUIRED_PACK_RELPATHS: tuple[str, ...] = (
    "task.toml",
    "instruction.md",
    "pre_artifacts.sh",
    "environment/Dockerfile",
    "tests/Dockerfile",
    "tests/test.sh",
    "tests/grader.py",
    "tests/config.json",
    "tests/test.patch",
    "solution/solution.patch",
    "solution/solve.sh",
)


class HarborExportError(RuntimeError):
    """Raised when a Harbor pack cannot be written or verified."""


@dataclass(frozen=True, slots=True)
class HarborPackResult:
    """One emitted pack directory."""

    task_id: str
    pack_dir: Path
    relpaths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HarborExportBundle:
    """Multi-pack export root (e.g. datasets/harbor_v1)."""

    out_dir: Path
    packs: tuple[HarborPackResult, ...]
    pack_manifest: Path


def verify_pack_tree(pack_dir: Path | str) -> list[str]:
    """Return missing required relative paths (empty if complete)."""
    root = Path(pack_dir)
    missing: list[str] = []
    for rel in REQUIRED_PACK_RELPATHS:
        if not (root / rel).is_file():
            missing.append(rel)
    return missing


def _write_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")


def export_harbor_pack(
    spec: HarborPackSpec,
    *,
    dest: Path | str,
    overwrite: bool = True,
    extra_environment_files: dict[str, str] | None = None,
    copy_repo_into_environment: Path | str | None = None,
) -> HarborPackResult:
    """Write one complete Harbor pack tree under ``dest`` (the task directory).

    Parameters
    ----------
    copy_repo_into_environment:
        When set, copy this directory to ``environment/repo/`` for offline
        Dockerfile ``COPY repo/`` builds.
    """
    cleaned = validate_pack_spec(spec)
    pack_dir = Path(dest)
    if pack_dir.exists():
        if not overwrite:
            raise HarborExportError(f"pack already exists: {pack_dir}")
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    base_commit = cleaned.task_toml.metadata.base_commit_hash
    language = cleaned.task_toml.metadata.language

    _write_text(pack_dir / "task.toml", render_task_toml(cleaned.task_toml))
    _write_text(pack_dir / "instruction.md", cleaned.instruction_md)
    _write_exec(
        pack_dir / "pre_artifacts.sh",
        cleaned.pre_artifacts_sh or render_pre_artifacts_sh(base_commit),
    )

    env_dir = pack_dir / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    _write_text(env_dir / "Dockerfile", cleaned.environment_dockerfile)
    if copy_repo_into_environment is not None:
        src = Path(copy_repo_into_environment)
        if not src.is_dir():
            raise HarborExportError(f"repo source not found: {src}")
        dest_repo = env_dir / "repo"
        if dest_repo.exists():
            shutil.rmtree(dest_repo)
        shutil.copytree(
            src,
            dest_repo,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                "*.pyc",
                ".venv",
                "node_modules",
                "gold.patch",
                "solution.patch",
                "test.patch",
            ),
        )
    if extra_environment_files:
        for rel, body in extra_environment_files.items():
            _write_text(env_dir / rel, body)

    tests_dir = pack_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    _write_text(tests_dir / "Dockerfile", cleaned.tests_dockerfile)
    _write_exec(
        tests_dir / "test.sh",
        cleaned.test_sh or default_test_sh(language=language),
    )
    _write_text(tests_dir / "grader.py", cleaned.grader_py or render_grader_py())
    cfg = cleaned.tests_config.model_dump(mode="json")
    (tests_dir / "config.json").write_text(
        json.dumps(cfg, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_text(tests_dir / "test.patch", cleaned.test_patch)

    sol_dir = pack_dir / "solution"
    sol_dir.mkdir(parents=True, exist_ok=True)
    _write_text(sol_dir / "solution.patch", cleaned.solution_patch)
    _write_exec(sol_dir / "solve.sh", cleaned.solve_sh or default_solve_sh())

    missing = verify_pack_tree(pack_dir)
    if missing:
        raise HarborExportError(f"incomplete pack tree under {pack_dir}: missing {missing}")

    return HarborPackResult(
        task_id=cleaned.task_id,
        pack_dir=pack_dir,
        relpaths=REQUIRED_PACK_RELPATHS,
    )


def write_harbor_export(
    specs: Sequence[HarborPackSpec],
    out_dir: Path | str,
    *,
    overwrite: bool = True,
    tasks_subdir: str = "tasks",
    repo_for: dict[str, Path | str] | None = None,
) -> HarborExportBundle:
    """Write one or more Harbor packs under ``out_dir/tasks/<task_id>/``."""
    base = Path(out_dir)
    if not specs:
        raise HarborExportError("refusing empty harbor export")
    if base.exists() and overwrite:
        tasks_root = base / tasks_subdir
        if tasks_root.is_dir():
            shutil.rmtree(tasks_root)
    base.mkdir(parents=True, exist_ok=True)
    tasks_root = base / tasks_subdir
    tasks_root.mkdir(parents=True, exist_ok=True)

    repos = dict(repo_for or {})
    packs: list[HarborPackResult] = []
    for spec in specs:
        collectors = export_harbor_pack(
            spec,
            dest=tasks_root / spec.task_id,
            overwrite=True,
            copy_repo_into_environment=repos.get(spec.task_id),
        )
        packs.append(collectors)

    manifest_path = base / "pack_manifest.json"
    payload: dict[str, Any] = {
        "count": len(packs),
        "task_ids": [p.task_id for p in packs],
        "required_relpaths": list(REQUIRED_PACK_RELPATHS),
        "schema_version_target": "1.1",
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HarborExportBundle(
        out_dir=base,
        packs=tuple(packs),
        pack_manifest=manifest_path,
    )


__all__ = [
    "REQUIRED_PACK_RELPATHS",
    "HarborExportBundle",
    "HarborExportError",
    "HarborPackResult",
    "export_harbor_pack",
    "verify_pack_tree",
    "write_harbor_export",
]
