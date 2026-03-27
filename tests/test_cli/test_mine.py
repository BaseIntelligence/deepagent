"""Tests for the mine CLI command."""

from unittest.mock import patch

from typer.testing import CliRunner

from swe_forge.cli.mine import app as mine_app
from swe_forge.cli.mine import validate_repo_format

runner = CliRunner()


class TestValidateRepoFormat:
    """Tests for repository format validation."""

    def test_valid_owner_repo(self):
        assert validate_repo_format("owner/repo") is True

    def test_valid_org_project(self):
        assert validate_repo_format("org/project-name") is True

    def test_invalid_missing_slash(self):
        assert validate_repo_format("ownerrepo") is False

    def test_invalid_too_many_slashes(self):
        assert validate_repo_format("owner/repo/extra") is False

    def test_invalid_empty_owner(self):
        assert validate_repo_format("/repo") is False

    def test_invalid_empty_repo(self):
        assert validate_repo_format("owner/") is False

    def test_invalid_empty_string(self):
        assert validate_repo_format("") is False


class TestMineCommandParsing:
    """Tests for mine command argument parsing."""

    def test_mine_help(self):
        """Test that --help works and shows expected options."""
        result = runner.invoke(mine_app, ["mine", "--help"])
        assert result.exit_code == 0
        assert "repo" in result.output.lower() or "--repo" in result.output
        assert "limit" in result.output.lower() or "--limit" in result.output

    def test_mine_default_options(self):
        """Test default values are applied correctly."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine"])
            assert result.exit_code == 0

    def test_mine_with_repo_option(self):
        """Test --repo option is parsed correctly."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--repo", "owner/repo"])
            assert result.exit_code == 0

    def test_mine_with_limit(self):
        """Test --limit option is parsed correctly."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--limit", "5"])
            assert result.exit_code == 0

    def test_mine_with_output(self):
        """Test --output option is parsed correctly."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--output", "/tmp/test.jsonl"])
            assert result.exit_code == 0

    def test_mine_with_difficulty(self):
        """Test --difficulty option is parsed correctly."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass

            @dataclass
            class MockResult:
                tasks: list = list
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--difficulty", "easy"])
            assert result.exit_code == 0

    def test_mine_with_model(self):
        """Test --model option is parsed correctly."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--model", "gpt-4"])
            assert result.exit_code == 0

    def test_mine_once_flag(self):
        """Test --once flag is recognized."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--once"])
            assert result.exit_code == 0

    def test_mine_continuous_flag(self):
        """Test --continuous flag is recognized."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--continuous"])
            assert result.exit_code == 0


class TestMineCommandValidation:
    """Tests for mine command input validation."""

    def test_invalid_repo_format(self):
        """Test that invalid repo format exits with error."""
        result = runner.invoke(mine_app, ["mine", "--repo", "invalidrepo"])
        assert result.exit_code == 1
        assert "owner/repo" in result.output

    def test_invalid_difficulty(self):
        """Test that invalid difficulty exits with error."""
        result = runner.invoke(mine_app, ["mine", "--difficulty", "impossible"])
        assert result.exit_code == 1

    def test_limit_too_low(self, caplog):
        """Test that limit below minimum shows error."""
        result = runner.invoke(mine_app, ["mine", "--limit", "0"])
        assert result.exit_code != 0


class TestMineCommandIntegration:
    """Integration tests for mine command execution."""

    def test_mine_creates_output_file(self, tmp_path):
        """Test that mine command creates output file."""
        from swe_forge.swe.models import SweTask

        output_file = tmp_path / "output.jsonl"

        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            task = SweTask(
                id="test-1",
                repo="owner/repo",
                base_commit="abc123",
                merge_commit="def456",
                language="python",
                prompt="Test prompt",
            )
            mock_run.return_value = MockResult(tasks=[task], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--output", str(output_file)])
            assert result.exit_code == 0
            assert output_file.exists()

    def test_verbose_mode(self):
        """Test that verbose mode works."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--verbose"])
            assert result.exit_code == 0

    def test_language_option(self):
        """Test --language option is parsed."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--language", "typescript"])
            assert result.exit_code == 0

    def test_min_stars_option(self):
        """Test --min-stars option is parsed."""
        with patch("swe_forge.cli.mine._run_pipeline") as mock_run:
            from dataclasses import dataclass, field

            @dataclass
            class MockResult:
                tasks: list = field(default_factory=list)
                benchmark_metrics: object = None

            mock_run.return_value = MockResult(tasks=[], benchmark_metrics=None)
            result = runner.invoke(mine_app, ["mine", "--min-stars", "50"])
            assert result.exit_code == 0


class TestMineAppGroup:
    """Tests for mine app being a Typer app."""

    def test_app_is_typer_app(self):
        """Verify that mine_app is a Typer app."""
        import typer

        assert isinstance(mine_app, typer.Typer)

    def test_app_has_mine_command(self):
        """Verify mine command is registered."""
        assert len(mine_app.registered_commands) >= 1
        assert mine_app.info.name == "mine"
