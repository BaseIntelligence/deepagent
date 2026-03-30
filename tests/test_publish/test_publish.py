"""Unit tests for publish module."""

import tempfile
from pathlib import Path

import pytest
import yaml


class TestParquetConverter:
    """Tests for parquet_converter module."""

    def test_tasks_to_records_empty_dir(self, tmp_path: Path) -> None:
        """Test converting empty directory."""
        from swe_forge.publish.parquet_converter import tasks_to_records

        records = tasks_to_records(tmp_path)
        assert records == []

    def test_tasks_to_records_with_workspace(self, tmp_path: Path) -> None:
        """Test converting directory with valid workspace."""
        from swe_forge.publish.parquet_converter import tasks_to_records

        task_dir = tmp_path / "owner-repo-123"
        task_dir.mkdir()

        workspace = {
            "task_id": "owner-repo-123",
            "language": "python",
            "repo": {
                "url": "https://github.com/owner/repo.git",
                "base_commit": "abc123",
                "merge_commit": "def456",
            },
            "tests": {
                "fail_to_pass": ["pytest tests/ -v"],
                "pass_to_pass": [],
            },
            "install": {
                "commands": ["pip install -e ."],
            },
        }

        workspace_path = task_dir / "workspace.yaml"
        with open(workspace_path, "w") as f:
            yaml.dump(workspace, f)

        patch_path = task_dir / "patch.diff"
        patch_path.write_text("diff --git a/file.py b/file.py\n")

        records = tasks_to_records(tmp_path)

        assert len(records) == 1
        assert records[0]["task_id"] == "owner-repo-123"
        assert records[0]["language"] == "python"
        assert records[0]["repo_url"] == "https://github.com/owner/repo.git"

    def test_convert_tasks_to_parquet_no_tasks(self, tmp_path: Path) -> None:
        """Test error when no tasks found."""
        from swe_forge.publish.parquet_converter import convert_tasks_to_parquet

        with pytest.raises(ValueError, match="No tasks found"):
            convert_tasks_to_parquet(tmp_path)


class TestDockerBuilder:
    """Tests for docker_builder module."""

    def test_build_result_dataclass(self) -> None:
        """Test BuildResult dataclass."""
        from swe_forge.publish.docker_builder import BuildResult

        result = BuildResult(
            success=True, image_name="test/image:tag", task_id="test-123"
        )
        assert result.success is True
        assert result.image_name == "test/image:tag"
        assert result.task_id == "test-123"
        assert result.error is None
        assert result.push_url is None

    def test_build_result_failure(self) -> None:
        """Test BuildResult for failure case."""
        from swe_forge.publish.docker_builder import BuildResult

        result = BuildResult(success=False, task_id="test-456", error="Build failed")
        assert result.success is False
        assert result.error == "Build failed"

    def test_generate_dockerfile_python(self) -> None:
        """Test Dockerfile generation for Python."""
        from swe_forge.publish.docker_builder import _generate_dockerfile

        workspace = {
            "language": "python",
            "repo": {
                "url": "https://github.com/test/repo.git",
                "base_commit": "abc123",
            },
            "install": {
                "commands": ["pip install -e ."],
            },
        }

        dockerfile = _generate_dockerfile(workspace)

        assert "FROM ubuntu:24.04" in dockerfile
        assert "python3" in dockerfile
        assert "git clone https://github.com/test/repo.git" in dockerfile
        assert "git checkout abc123" in dockerfile

    def test_generate_dockerfile_javascript(self) -> None:
        """Test Dockerfile generation for JavaScript."""
        from swe_forge.publish.docker_builder import _generate_dockerfile

        workspace = {
            "language": "javascript",
            "repo": {
                "url": "https://github.com/test/repo.git",
                "base_commit": "def456",
            },
            "install": {
                "commands": ["npm install"],
            },
        }

        dockerfile = _generate_dockerfile(workspace)

        assert "FROM ubuntu:24.04" in dockerfile
        assert "nodejs" in dockerfile


class TestHfUploader:
    """Tests for hf_uploader module."""

    def test_upload_dataset_missing_token(self, tmp_path: Path) -> None:
        """Test error when token missing."""
        from swe_forge.publish.hf_uploader import upload_dataset

        parquet_path = tmp_path / "test.parquet"
        parquet_path.write_text("test")

        import os

        old_token = os.environ.pop("HF_TOKEN", None)

        try:
            with pytest.raises(ValueError, match="HF_TOKEN not provided"):
                upload_dataset(parquet_path, "test/repo")
        finally:
            if old_token:
                os.environ["HF_TOKEN"] = old_token
