"""Tests for the export CLI command."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from swe_forge.cli.export import app as export_app
from swe_forge.cli.export import validate_format
from swe_forge.swe.models import SweTask

runner = CliRunner()


class TestValidateFormat:
    """Tests for format validation."""

    def test_valid_jsonl_format(self):
        assert validate_format("jsonl") is True

    def test_valid_parquet_format(self):
        assert validate_format("parquet") is True

    def test_valid_hf_format(self):
        assert validate_format("hf") is True

    def test_invalid_format(self):
        assert validate_format("csv") is False

    def test_invalid_format_uppercase(self):
        assert validate_format("JSONL") is False

    def test_empty_format(self):
        assert validate_format("") is False

    def test_none_format(self):
        assert validate_format(None) is False


class TestExportCommandParsing:
    """Tests for export command argument parsing."""

    @pytest.fixture
    def sample_jsonl(self, tmp_path: Path) -> Path:
        """Create a sample JSONL file with test tasks."""
        jsonl_path = tmp_path / "tasks.jsonl"
        tasks = [
            SweTask(
                id="test-1",
                repo="owner/repo",
                base_commit="abc123",
                merge_commit="def456",
                language="python",
                prompt="Test prompt 1",
            ),
        ]
        from swe_forge.export.jsonl import export_jsonl

        export_jsonl(tasks, jsonl_path)
        return jsonl_path

    def test_export_help(self):
        """Test that --help works and shows expected options."""
        result = runner.invoke(export_app, ["--help"])
        assert result.exit_code == 0
        assert "input" in result.output.lower() or "--input" in result.output
        assert "format" in result.output.lower() or "--format" in result.output
        assert "output" in result.output.lower() or "--output" in result.output
        assert "repo" in result.output.lower() or "--repo" in result.output

    def test_export_missing_input(self):
        """Test that missing input file exits with error."""
        result = runner.invoke(
            export_app,
            [
                "--input",
                "/nonexistent/file.jsonl",
                "--format",
                "parquet",
                "--output",
                "out.parquet",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_export_invalid_format(self, sample_jsonl: Path):
        """Test that invalid format exits with error."""
        result = runner.invoke(
            export_app,
            ["--input", str(sample_jsonl), "--format", "csv", "--output", "out.csv"],
        )
        assert result.exit_code == 1
        assert "Invalid format" in result.output

    def test_export_parquet_missing_output(self, sample_jsonl: Path):
        """Test that --output is required for parquet format."""
        result = runner.invoke(
            export_app, ["--input", str(sample_jsonl), "--format", "parquet"]
        )
        assert result.exit_code == 1
        assert "--output is required" in result.output

    def test_export_jsonl_missing_output(self, sample_jsonl: Path):
        """Test that --output is required for jsonl format."""
        result = runner.invoke(
            export_app, ["--input", str(sample_jsonl), "--format", "jsonl"]
        )
        assert result.exit_code == 1
        assert "--output is required" in result.output

    def test_export_hf_missing_repo(self, sample_jsonl: Path):
        """Test that --repo is required for hf format."""
        result = runner.invoke(
            export_app, ["--input", str(sample_jsonl), "--format", "hf"]
        )
        assert result.exit_code == 1
        assert "--repo is required" in result.output


class TestExportCommandIntegration:
    """Integration tests for export command execution."""

    @pytest.fixture
    def sample_jsonl(self, tmp_path: Path) -> Path:
        """Create a sample JSONL file with test tasks."""
        jsonl_path = tmp_path / "tasks.jsonl"
        tasks = [
            SweTask(
                id="test-1",
                repo="owner/repo",
                base_commit="abc123",
                merge_commit="def456",
                language="python",
                prompt="Test prompt 1",
            ),
            SweTask(
                id="test-2",
                repo="owner/repo",
                base_commit="ghi789",
                merge_commit="jkl012",
                language="python",
                prompt="Test prompt 2",
            ),
        ]
        from swe_forge.export.jsonl import export_jsonl

        export_jsonl(tasks, jsonl_path)
        return jsonl_path

    def test_export_to_parquet(self, sample_jsonl: Path, tmp_path: Path):
        """Test exporting to parquet format."""
        output_path = tmp_path / "output.parquet"
        result = runner.invoke(
            export_app,
            [
                "--input",
                str(sample_jsonl),
                "--format",
                "parquet",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        assert "Exported 2 tasks" in result.output

    def test_export_to_jsonl(self, sample_jsonl: Path, tmp_path: Path):
        """Test exporting to jsonl format."""
        output_path = tmp_path / "output.jsonl"
        result = runner.invoke(
            export_app,
            [
                "--input",
                str(sample_jsonl),
                "--format",
                "jsonl",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        assert "Exported 2 tasks" in result.output

    def test_export_to_hf(self, sample_jsonl: Path):
        """Test exporting to HuggingFace format."""
        with patch("swe_forge.cli.export.upload_to_hf") as mock_upload:
            mock_upload.return_value = True
            result = runner.invoke(
                export_app,
                [
                    "--input",
                    str(sample_jsonl),
                    "--format",
                    "hf",
                    "--repo",
                    "org/dataset",
                ],
            )
            assert result.exit_code == 0
            mock_upload.assert_called_once()
            assert "Uploaded 2 tasks" in result.output

    def test_export_empty_file(self, tmp_path: Path):
        """Test exporting empty input file."""
        empty_jsonl = tmp_path / "empty.jsonl"
        empty_jsonl.write_text("")
        output_path = tmp_path / "output.parquet"

        result = runner.invoke(
            export_app,
            [
                "--input",
                str(empty_jsonl),
                "--format",
                "parquet",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert "No tasks to export" in result.output

    def test_verbose_mode(self, sample_jsonl: Path, tmp_path: Path):
        """Test that verbose mode works."""
        output_path = tmp_path / "output.parquet"
        result = runner.invoke(
            export_app,
            [
                "--input",
                str(sample_jsonl),
                "--format",
                "parquet",
                "--output",
                str(output_path),
                "--verbose",
            ],
        )
        assert result.exit_code == 0

    def test_hf_upload_error(self, sample_jsonl: Path):
        """Test that HF upload errors are handled."""
        with patch("swe_forge.cli.export.upload_to_hf") as mock_upload:
            from swe_forge.export.hf_upload import HfUploadError

            mock_upload.side_effect = HfUploadError("Upload failed")
            result = runner.invoke(
                export_app,
                [
                    "--input",
                    str(sample_jsonl),
                    "--format",
                    "hf",
                    "--repo",
                    "org/dataset",
                ],
            )
            assert result.exit_code == 1
            assert "Error" in result.output


class TestExportAppGroup:
    """Tests for export app being a Typer app."""

    def test_app_is_typer_app(self):
        """Verify that export_app is a Typer app."""
        import typer

        assert isinstance(export_app, typer.Typer)

    def test_app_has_export_command(self):
        """Verify export command is registered."""
        assert len(export_app.registered_commands) >= 1
        assert export_app.info.name == "export"
