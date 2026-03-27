from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pytest

from swe_forge.export.parquet import (
    export_parquet,
    get_parquet_schema,
    import_parquet,
)
from swe_forge.swe.models import SweTask, SweTaskStatus


class TestGetParquetSchema:
    def test_returns_pyarrow_schema(self):
        schema = get_parquet_schema()
        assert isinstance(schema, pa.Schema)

    def test_has_all_required_fields(self):
        schema = get_parquet_schema()
        field_names = {field.name for field in schema}
        required = {
            "id",
            "repo",
            "base_commit",
            "merge_commit",
            "language",
            "difficulty_score",
            "created_at",
            "patch",
            "test_patch",
            "fail_to_pass",
            "pass_to_pass",
            "install_config",
            "meta",
            "prompt",
            "original_pr_body",
            "quality_score",
            "quality_passed",
            "docker_passed",
            "workspace_path",
            "status",
        }
        assert field_names == required

    def test_field_types(self):
        schema = get_parquet_schema()
        assert schema.field("id").type == pa.string()
        assert schema.field("difficulty_score").type == pa.uint8()
        assert schema.field("created_at").type == pa.timestamp("us", tz="UTC")
        assert schema.field("fail_to_pass").type == pa.list_(pa.string())
        assert schema.field("quality_passed").type == pa.bool_()


class TestExportParquet:
    def test_export_single_task(self, tmp_path: Path):
        task = SweTask(
            id="test-123",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            fail_to_pass=["test_one", "test_two"],
            install_config={"pip": "pytest"},
        )
        output_file = tmp_path / "tasks.parquet"
        count = export_parquet([task], output_file)
        assert count == 1
        assert output_file.exists()

    def test_export_multiple_tasks(self, tmp_path: Path):
        tasks = [SweTask(id=f"task-{i}", repo="owner/repo") for i in range(5)]
        output_file = tmp_path / "tasks.parquet"
        count = export_parquet(tasks, output_file)
        assert count == 5

    def test_export_empty_list(self, tmp_path: Path):
        output_file = tmp_path / "empty.parquet"
        count = export_parquet([], output_file)
        assert count == 0
        assert output_file.exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        output_file = tmp_path / "nested" / "dir" / "tasks.parquet"
        export_parquet([SweTask(id="test", repo="owner/repo")], output_file)
        assert output_file.exists()

    def test_compression_zstd(self, tmp_path: Path):
        task = SweTask(id="test", repo="owner/repo", patch="x" * 10000)
        output_file = tmp_path / "zstd.parquet"
        export_parquet([task], output_file, compression="zstd")
        assert output_file.exists()

    def test_compression_snappy(self, tmp_path: Path):
        task = SweTask(id="test", repo="owner/repo")
        output_file = tmp_path / "snappy.parquet"
        export_parquet([task], output_file, compression="snappy")
        assert output_file.exists()

    def test_compression_none(self, tmp_path: Path):
        task = SweTask(id="test", repo="owner/repo")
        output_file = tmp_path / "none.parquet"
        export_parquet([task], output_file, compression="none")
        assert output_file.exists()


class TestImportParquet:
    def test_import_roundtrip(self, tmp_path: Path):
        task = SweTask(
            id="test-123",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            difficulty_score=5,
            fail_to_pass=["test_one", "test_two"],
            pass_to_pass=["test_pass"],
            install_config={"pip": "pytest", "apt": "libxml2"},
            meta={"source": "github"},
            quality_score=0.95,
            quality_passed=True,
            docker_passed=False,
            workspace_path="/tmp/workspace",
            status=SweTaskStatus.READY,
        )
        output_file = tmp_path / "roundtrip.parquet"
        export_parquet([task], output_file)
        records = import_parquet(output_file)
        assert len(records) == 1
        record = records[0]
        assert record["id"] == "test-123"
        assert record["repo"] == "owner/repo"
        assert record["language"] == "python"
        assert record["difficulty_score"] == 5
        assert record["fail_to_pass"] == ["test_one", "test_two"]
        assert record["pass_to_pass"] == ["test_pass"]
        assert record["install_config"] == {"pip": "pytest", "apt": "libxml2"}
        assert record["meta"] == {"source": "github"}
        assert record["quality_score"] == 0.95
        assert record["quality_passed"] is True
        assert record["docker_passed"] is False
        assert record["workspace_path"] == "/tmp/workspace"
        assert record["status"] == "ready"

    def test_import_preserves_datetime(self, tmp_path: Path):
        created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        task = SweTask(
            id="test",
            repo="owner/repo",
            created_at=created_at,
        )
        output_file = tmp_path / "datetime.parquet"
        export_parquet([task], output_file)
        records = import_parquet(output_file)
        assert records[0]["created_at"] == created_at

    def test_import_empty_file(self, tmp_path: Path):
        output_file = tmp_path / "empty.parquet"
        export_parquet([], output_file)
        records = import_parquet(output_file)
        assert records == []

    def test_import_multiple_tasks(self, tmp_path: Path):
        tasks = [
            SweTask(id=f"task-{i}", repo="owner/repo", difficulty_score=i)
            for i in range(10)
        ]
        output_file = tmp_path / "multi.parquet"
        export_parquet(tasks, output_file)
        records = import_parquet(output_file)
        assert len(records) == 10
        for i, record in enumerate(records):
            assert record["id"] == f"task-{i}"
            assert record["difficulty_score"] == i

    def test_import_handles_null_values(self, tmp_path: Path):
        task = SweTask(
            id="test",
            repo="owner/repo",
            quality_score=None,
            workspace_path=None,
        )
        output_file = tmp_path / "nulls.parquet"
        export_parquet([task], output_file)
        records = import_parquet(output_file)
        assert records[0]["quality_score"] is None
        assert records[0]["workspace_path"] is None

    def test_import_handles_complex_install_config(self, tmp_path: Path):
        """Test that non-string install_config values are JSON-encoded."""
        import json

        task = SweTask(
            id="test-complex",
            repo="owner/repo",
            install_config={
                "commands": ["pip install -e .", "pytest"],
                "python": "3.11",  # string value - not encoded
                "config": {"nested": "value"},
            },
        )
        output_file = tmp_path / "complex.parquet"
        export_parquet([task], output_file)
        records = import_parquet(output_file)

        config = dict(records[0]["install_config"])

        # List values should be JSON-encoded
        commands = json.loads(config["commands"])
        assert commands == ["pip install -e .", "pytest"]

        # String values should remain as-is
        assert config["python"] == "3.11"

        # Dict values should be JSON-encoded
        nested = json.loads(config["config"])
        assert nested == {"nested": "value"}


class TestParquetIntegration:
    def test_large_dataset_roundtrip(self, tmp_path: Path):
        tasks = [
            SweTask(
                id=f"task-{i:04d}",
                repo=f"org/repo-{i % 10}",
                base_commit=f"commit{i}",
                merge_commit=f"merge{i}",
                language="python",
                difficulty_score=i % 256,
                patch=f"diff --git a/file.py b/file.py\n--- auto patch {i}",
                fail_to_pass=[f"test_{j}" for j in range(i % 5)],
                install_config={"pip": f"package=={i}"},
            )
            for i in range(100)
        ]
        output_file = tmp_path / "large.parquet"
        count = export_parquet(tasks, output_file)
        assert count == 100
        records = import_parquet(output_file)
        assert len(records) == 100
        for i, record in enumerate(records):
            assert record["id"] == f"task-{i:04d}"
