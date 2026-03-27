"""Tests for harness report generation."""

import json
import tempfile
from pathlib import Path


from swe_forge.swe.harness import HarnessResult, HarnessStatus
from swe_forge.swe.report import HarnessReport


class TestHarnessReport:
    """Tests for HarnessReport class."""

    def test_write_report_creates_file(self):
        """Test that write_report creates the output file."""
        report = HarnessReport()
        results = [
            HarnessResult(
                task_id="test-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "reports" / "report.json"
            report.write_report(results, str(output_path))

            assert output_path.exists()

    def test_write_report_json_structure(self):
        """Test JSON report has correct structure."""
        report = HarnessReport()
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=10.5,
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=20.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            report.write_report(results, str(output_path))

            with open(output_path) as f:
                data = json.load(f)

            assert "generated_at" in data
            assert data["total_tasks"] == 2
            assert data["resolved"] == 1
            assert len(data["results"]) == 2

    def test_write_report_includes_test_results(self):
        """Test JSON report includes test results."""
        report = HarnessReport()
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                fail_to_pass_results=[
                    ("pytest test_foo.py", True),
                    ("pytest test_bar.py", True),
                ],
                pass_to_pass_results=[
                    ("pytest test_baz.py", True),
                ],
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            report.write_report(results, str(output_path))

            with open(output_path) as f:
                data = json.load(f)

            result = data["results"][0]
            assert len(result["fail_to_pass_results"]) == 2
            assert result["fail_to_pass_results"][0]["command"] == "pytest test_foo.py"
            assert result["fail_to_pass_results"][0]["passed"] is True
            assert len(result["pass_to_pass_results"]) == 1

    def test_print_summary_empty_results(self):
        """Test summary with no results."""
        report = HarnessReport()
        summary = report.print_summary([])

        assert summary == "Harness Results: 0 tasks"

    def test_print_summary_basic(self):
        """Test basic summary output."""
        report = HarnessReport()
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=10.0,
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=20.0,
            ),
            HarnessResult(
                task_id="task-3",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=30.0,
            ),
        ]

        summary = report.print_summary(results)

        assert "3 tasks" in summary
        assert "2 resolved" in summary
        assert "66.7%" in summary
        assert "RESOLVED: 2" in summary
        assert "UNRESOLVED: 1" in summary

    def test_print_summary_includes_errors(self):
        """Test summary counts error statuses."""
        report = HarnessReport()
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=10.0,
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.AGENT_ERROR,
                resolved=False,
                duration_seconds=5.0,
                error_message="Agent timeout",
            ),
            HarnessResult(
                task_id="task-3",
                status=HarnessStatus.SETUP_ERROR,
                resolved=False,
                duration_seconds=2.0,
                error_message="Setup failed",
            ),
        ]

        summary = report.print_summary(results)

        assert "ERROR: 2" in summary

    def test_print_summary_average_duration(self):
        """Test summary includes average duration."""
        report = HarnessReport()
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=30.0,
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=60.0,
            ),
        ]

        summary = report.print_summary(results)

        assert "Avg duration: 45.0s" in summary

    def test_write_report_includes_error_message(self):
        """Test JSON report includes error message when present."""
        report = HarnessReport()
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.AGENT_ERROR,
                resolved=False,
                error_message="Agent timed out after 600s",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            report.write_report(results, str(output_path))

            with open(output_path) as f:
                data = json.load(f)

            assert data["results"][0]["error_message"] == "Agent timed out after 600s"
