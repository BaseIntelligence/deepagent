"""Parquet export utilities for SweTask objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from ..swe.models import SweTask, SweTaskStatus

if TYPE_CHECKING:
    from collections.abc import Sequence


def get_parquet_schema() -> pa.Schema:
    """Define PyArrow schema matching SweTask fields.

    Returns:
        PyArrow schema with all SweTask fields.
    """
    return pa.schema(
        [
            ("id", pa.string()),
            ("repo", pa.string()),
            ("base_commit", pa.string()),
            ("merge_commit", pa.string()),
            ("language", pa.string()),
            ("difficulty_score", pa.uint8()),
            ("created_at", pa.timestamp("us", tz="UTC")),
            ("patch", pa.string()),
            ("deletion_patch", pa.string()),
            ("test_patch", pa.string()),
            ("fail_to_pass", pa.list_(pa.string())),
            ("pass_to_pass", pa.list_(pa.string())),
            ("install_config", pa.map_(pa.string(), pa.string())),
            ("meta", pa.map_(pa.string(), pa.string())),
            ("source_type", pa.string()),
            ("prompt", pa.string()),
            ("original_pr_body", pa.string()),
            ("quality_score", pa.float64()),
            ("quality_passed", pa.bool_()),
            ("docker_passed", pa.bool_()),
            ("workspace_path", pa.string()),
            ("status", pa.string()),
        ]
    )


def _swe_task_to_record(task: SweTask) -> dict:
    """Convert SweTask to a Parquet-compatible record dict.

    Args:
        task: SweTask instance to convert.

    Returns:
        Dictionary with all fields converted to Parquet-compatible types.
    """
    return {
        "id": task.id,
        "repo": task.repo,
        "base_commit": task.base_commit,
        "merge_commit": task.merge_commit,
        "language": task.language,
        "difficulty_score": task.difficulty_score,
        "created_at": task.created_at,
        "patch": task.patch,
        "deletion_patch": task.deletion_patch,
        "test_patch": task.test_patch,
        "fail_to_pass": task.fail_to_pass,
        "pass_to_pass": task.pass_to_pass,
        "install_config": [
            (k, json.dumps(v) if not isinstance(v, str) else v)
            for k, v in task.install_config.items()
        ],
        "meta": list(task.meta.items()),
        "source_type": task.source_type,
        "prompt": task.prompt,
        "original_pr_body": task.original_pr_body,
        "quality_score": task.quality_score,
        "quality_passed": task.quality_passed,
        "docker_passed": task.docker_passed,
        "workspace_path": task.workspace_path,
        "status": task.status.value
        if isinstance(task.status, SweTaskStatus)
        else task.status,
    }


def export_parquet(
    tasks: Sequence[SweTask],
    path: str | Path,
    compression: str = "zstd",
) -> int:
    """Write SweTask objects to a Parquet file.

    Uses PyArrow directly (not pandas) for efficient Parquet export.

    Args:
        tasks: Sequence of SweTask objects to export.
        path: Output file path.
        compression: Compression algorithm ('zstd', 'snappy', 'gzip', 'brotli', 'none').
                     Defaults to 'zstd'.

    Returns:
        Number of tasks exported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not tasks:
        schema = get_parquet_schema()
        arrays = [pa.array([], type=field.type) for field in schema]
        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, str(path), compression=compression)
        return 0

    records = [_swe_task_to_record(task) for task in tasks]
    table = pa.Table.from_pylist(records, schema=get_parquet_schema())
    pq.write_table(table, str(path), compression=compression)

    return len(tasks)


def import_parquet(path: str | Path) -> list[dict]:
    """Read Parquet file and return as list of dictionaries.

    Args:
        path: Path to Parquet file.

    Returns:
        List of dictionaries, each representing a task record.
    """
    path = Path(path)
    table = pq.read_table(str(path))
    records = table.to_pylist()

    for record in records:
        for key in ("install_config", "meta"):
            if key in record and isinstance(record[key], list):
                # Arrow maps are returned as list of tuples
                record[key] = dict(record[key])

    return records
