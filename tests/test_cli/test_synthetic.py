from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from swe_forge.cli.synthetic import app

runner = CliRunner()


def test_synthetic_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "audit" not in result.output
    assert "sanitize" not in result.output


def test_synthetic_generate_generates_workspace(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "pkg.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    output = tmp_path / "tasks"

    with patch("swe_forge.cli.synthetic._current_commit", return_value="abc123"):
        result = runner.invoke(
            app,
            [
                "--repo-path",
                str(repo),
                "--repo",
                "owner/repo",
                "--source-file",
                "pkg.py",
                "--symbol",
                "add",
                "--fail-to-pass",
                "pytest tests/test_pkg.py -v",
                "--output-folder",
                str(output),
            ],
        )

    assert result.exit_code == 0
    task_dirs = list(output.iterdir())
    assert len(task_dirs) == 1
    assert (task_dirs[0] / "workspace.yaml").exists()
    assert (task_dirs[0] / "deletion_patch.diff").exists()
