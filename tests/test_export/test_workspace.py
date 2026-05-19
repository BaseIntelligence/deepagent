"""Tests for workspace export dedup functionality."""

from pathlib import Path

from swe_forge.swe.models import SweTask
from swe_forge.export.workspace import export_task_to_workspace


class TestWorkspaceExportDedup:
    """Tests for workspace export skip existing."""

    def test_export_skips_existing_directory(self, tmp_path: Path):
        task = SweTask(
            id="owner-repo-123",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test task",
            patch="diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n",
        )

        existing_dir = tmp_path / task.id
        existing_dir.mkdir(parents=True)
        (existing_dir / "existing_file.txt").write_text("existing content")

        result = export_task_to_workspace(task, tmp_path)

        assert result is None
        assert (existing_dir / "existing_file.txt").exists()
        assert (existing_dir / "existing_file.txt").read_text() == "existing content"

    def test_export_overwrites_with_flag(self, tmp_path: Path):
        task = SweTask(
            id="owner-repo-456",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test task",
            patch="diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n",
        )

        existing_dir = tmp_path / task.id
        existing_dir.mkdir(parents=True)
        (existing_dir / "old_file.txt").write_text("old content")

        result = export_task_to_workspace(task, tmp_path, overwrite=True)

        assert result is not None
        assert (result / "workspace.yaml").exists()
        assert not (result / "old_file.txt").exists()

    def test_export_new_directory(self, tmp_path: Path):
        task = SweTask(
            id="owner-repo-789",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test task",
            patch="diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n",
        )

        result = export_task_to_workspace(task, tmp_path)

        assert result is not None
        assert result == tmp_path / task.id
        assert (result / "workspace.yaml").exists()
        assert (result / "patch.diff").exists()

    def test_export_writes_synthetic_deletion_patch(self, tmp_path: Path):
        task = SweTask(
            id="owner-repo-synth",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="abc123",
            language="python",
            source_type="synthetic_feature_deletion",
            prompt="Restore deleted feature",
            patch="diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n",
            deletion_patch="diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n",
            fail_to_pass=["pytest tests/test_pkg.py -v"],
            meta={"strategy": "feature_deletion"},
        )

        result = export_task_to_workspace(task, tmp_path)

        assert result is not None
        assert (result / "deletion_patch.diff").exists()
        workspace = (result / "workspace.yaml").read_text()
        assert "synthetic_feature_deletion" in workspace
        assert "deletion_patch.diff" in workspace
        assert (
            "Applying synthetic deletion patch" in (result / "evaluate.sh").read_text()
        )
