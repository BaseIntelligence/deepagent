"""Tests for dataset validation - FRESH container testing."""

import json

from swe_forge.swe.dataset_validator import (
    ValidationResult,
    DatasetValidationReport,
    print_validation_report,
)
from swe_forge.swe.models import SweTask


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = ValidationResult(
            task_id="test-001",
            passed=True,
            phase="complete",
            install_commands_worked=["pip install -e ."],
            test_results={"base:pytest": False, "patch:pytest": True},
        )

        assert result.passed is True
        assert result.phase == "complete"
        assert len(result.install_commands_worked) == 1

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ValidationResult(
            task_id="test-002",
            passed=False,
            phase="install",
            error_message="Install failed",
            install_commands_failed=["pip install -e ."],
        )

        data = result.to_dict()

        assert data["task_id"] == "test-002"
        assert data["passed"] is False
        assert data["error_message"] == "Install failed"

    def test_failed_install_tracking(self):
        """Test tracking of failed install commands."""
        result = ValidationResult(task_id="test-003", passed=False, phase="install")

        result.install_commands_worked.append("apt-get update")
        result.install_commands_failed.append("pip install broken-package")

        assert len(result.install_commands_worked) == 1
        assert len(result.install_commands_failed) == 1


class TestDatasetValidationReport:
    """Test DatasetValidationReport."""

    def test_empty_report(self):
        """Test empty report."""
        report = DatasetValidationReport()

        assert report.total_tasks == 0
        assert report.pass_rate == 0.0

    def test_report_with_results(self):
        """Test report with validation results."""
        report = DatasetValidationReport(
            total_tasks=3,
            passed_tasks=2,
            failed_tasks=1,
            validation_results=[
                ValidationResult(task_id="t1", passed=True, phase="complete"),
                ValidationResult(task_id="t2", passed=True, phase="complete"),
                ValidationResult(
                    task_id="t3",
                    passed=False,
                    phase="test_base",
                    error_message="Test failed",
                ),
            ],
        )

        assert report.pass_rate == 2 / 3
        assert report.failed_tasks == 1

    def test_report_to_json(self):
        """Test report JSON serialization."""
        report = DatasetValidationReport(
            total_tasks=2,
            passed_tasks=1,
            failed_tasks=1,
        )

        json_str = report.to_json()
        data = json.loads(json_str)

        assert data["total_tasks"] == 2
        assert data["pass_rate"] == 0.5


class TestPrintValidationReport:
    """Test print_validation_report function."""

    def test_print_empty_report(self, capsys):
        """Test printing empty report."""
        report = DatasetValidationReport()
        print_validation_report(report)

        captured = capsys.readouterr()
        assert "Total tasks: 0" in captured.out
        assert "Pass rate:" in captured.out

    def test_print_report_with_results(self, capsys):
        """Test printing report with results."""
        report = DatasetValidationReport(
            total_tasks=5,
            passed_tasks=3,
            failed_tasks=2,
            validation_results=[
                ValidationResult(task_id="t1", passed=True, phase="complete"),
                ValidationResult(
                    task_id="t2",
                    passed=False,
                    phase="test_base",
                    error_message="Test failed on base",
                ),
            ],
        )

        print_validation_report(report)

        captured = capsys.readouterr()
        assert "Passed: 3" in captured.out
        assert "Failed: 2" in captured.out
        assert "0.6" in captured.out or "60" in captured.out


class TestSweTaskInstallConfig:
    """Test SweTask with install_config field."""

    def test_task_with_install_config(self):
        """Test task creation with install_config."""
        task = SweTask(
            id="test-001",
            repo="test/test",
            base_commit="abc123",
            merge_commit="def456",
            install_config={
                "python_version": "3.11",
                "package_manager": "poetry",
                "install_commands": ["poetry install"],
                "validated": True,
            },
        )

        assert task.install_config["python_version"] == "3.11"
        assert task.install_config["package_manager"] == "poetry"

    def test_task_is_install_ready(self):
        """Test is_install_ready method."""
        # Not ready
        task1 = SweTask(
            id="test-001",
            install_config={},
        )
        assert task1.is_install_ready() is False

        # Ready
        task2 = SweTask(
            id="test-002",
            install_config={
                "install_commands": ["pip install -e ."],
                "validated": True,
            },
        )
        assert task2.is_install_ready() is True
