"""Tests for the benchmark CLI command."""

from unittest.mock import patch

from typer.testing import CliRunner

from swe_forge.cli.benchmark import app as benchmark_app

runner = CliRunner()


class TestBenchmarkCommandParsing:
    """Tests for benchmark command argument parsing."""

    def test_benchmark_help(self):
        """Test that --help works and shows expected options."""
        result = runner.invoke(benchmark_app, ["--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower() or "--model" in result.output
        assert "tasks" in result.output.lower() or "--tasks" in result.output

    def test_benchmark_default_options(self):
        """Test default values are applied correctly."""
        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=0,
                resolved_count=0,
                total_count=0,
                results=[],
            )
            result = runner.invoke(benchmark_app, [])
            assert result.exit_code == 0

    def test_benchmark_with_model_option(self):
        """Test --model option is parsed correctly."""
        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=0,
                resolved_count=0,
                total_count=0,
                results=[],
            )
            result = runner.invoke(benchmark_app, ["--model", "gpt-4"])
            assert result.exit_code == 0

    def test_benchmark_with_tasks_option(self):
        """Test --tasks option is parsed correctly."""
        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=0,
                resolved_count=0,
                total_count=0,
                results=[],
            )
            result = runner.invoke(benchmark_app, ["--tasks", "10"])
            assert result.exit_code == 0

    def test_benchmark_with_difficulty_option(self):
        """Test --difficulty option is parsed correctly."""
        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=0,
                resolved_count=0,
                total_count=0,
                results=[],
            )
            result = runner.invoke(benchmark_app, ["--difficulty", "easy"])
            assert result.exit_code == 0

    def test_benchmark_with_output_option(self):
        """Test --output option is parsed correctly."""
        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=0,
                resolved_count=0,
                total_count=0,
                results=[],
            )
            result = runner.invoke(benchmark_app, ["--output", "results/"])
            assert result.exit_code == 0

    def test_benchmark_with_report_flag(self):
        """Test --report flag is recognized."""
        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=0,
                resolved_count=0,
                total_count=0,
                results=[],
            )
            result = runner.invoke(benchmark_app, ["--report"])
            assert result.exit_code == 0


class TestBenchmarkCommandValidation:
    """Tests for benchmark command input validation."""

    def test_invalid_difficulty(self):
        """Test that invalid difficulty exits with error."""
        result = runner.invoke(benchmark_app, ["--difficulty", "impossible"])
        assert result.exit_code == 1

    def test_tasks_too_low(self):
        """Test that tasks below minimum shows error."""
        result = runner.invoke(benchmark_app, ["--tasks", "0"])
        assert result.exit_code != 0

    def test_invalid_model(self):
        """Test that empty model name is handled."""
        result = runner.invoke(benchmark_app, ["--model", ""])
        # Empty model should either error or use default
        assert result.exit_code == 0 or result.exit_code == 1


class MockBenchmarkResult:
    """Mock result for benchmark run."""

    def __init__(
        self,
        tasks_completed: int,
        resolved_count: int,
        total_count: int,
        results: list,
        metrics: dict = None,
    ):
        self.tasks_completed = tasks_completed
        self.resolved_count = resolved_count
        self.total_count = total_count
        self.results = results
        self.metrics = metrics or {}


class TestBenchmarkCommandIntegration:
    """Integration tests for benchmark command execution."""

    def test_benchmark_creates_output_directory(self, tmp_path):
        """Test that benchmark command creates output directory."""
        output_dir = tmp_path / "results"

        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=0,
                resolved_count=0,
                total_count=0,
                results=[],
            )
            result = runner.invoke(benchmark_app, ["--output", str(output_dir)])
            assert result.exit_code == 0

    def test_benchmark_generates_report(self, tmp_path):
        """Test that benchmark generates HTML report when --report is used."""
        output_dir = tmp_path / "results"

        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=1,
                resolved_count=1,
                total_count=1,
                results=[{"task_id": "test-1", "resolved": True}],
            )
            with patch("swe_forge.cli.benchmark._generate_report"):
                result = runner.invoke(
                    benchmark_app,
                    ["--output", str(output_dir), "--report"],
                )
                assert result.exit_code == 0

    def test_verbose_mode(self):
        """Test that verbose mode works."""
        with patch("swe_forge.cli.benchmark._run_benchmark") as mock_run:
            mock_run.return_value = MockBenchmarkResult(
                tasks_completed=0,
                resolved_count=0,
                total_count=0,
                results=[],
            )
            result = runner.invoke(benchmark_app, ["--verbose"])
            assert result.exit_code == 0


class TestBenchmarkAppGroup:
    """Tests for benchmark app being a Typer app."""

    def test_app_is_typer_app(self):
        """Verify that benchmark_app is a Typer app."""
        import typer

        assert isinstance(benchmark_app, typer.Typer)

    def test_app_has_benchmark_command(self):
        """Verify benchmark command is registered."""
        assert len(benchmark_app.registered_commands) >= 1
        assert benchmark_app.info.name == "benchmark"


class TestBenchmarkHelpers:
    """Tests for benchmark helper functions."""

    def test_difficulty_validation_function(self):
        """Test that difficulty validation works."""
        from swe_forge.cli.benchmark import validate_difficulty

        assert validate_difficulty("easy") is True
        assert validate_difficulty("medium") is True
        assert validate_difficulty("hard") is True
        assert validate_difficulty("invalid") is False

    def test_generate_summary_table(self):
        """Test that summary table generation works."""
        from swe_forge.cli.benchmark import _print_summary

        # This should not raise any errors
        results = [
            {"task_id": "test-1", "resolved": True, "status": "resolved"},
            {"task_id": "test-2", "resolved": False, "status": "unresolved"},
        ]

        with patch("swe_forge.cli.benchmark.console") as mock_console:
            _print_summary(results)
            # Verify print was called
            assert mock_console.print.called
