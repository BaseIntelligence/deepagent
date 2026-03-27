"""Tests for swe_forge.swe.sanity_check module."""

import pytest
from dataclasses import dataclass

from swe_forge.swe.sanity_check import (
    SanityCheckResult,
    run_sanity_check,
)
from swe_forge.swe.models import SweTask


@dataclass
class MockExecResult:
    stdout: str
    stderr: str
    exit_code: int


class MockSandbox:
    def __init__(self, results: list[MockExecResult] | None = None):
        self.results = results or []
        self.call_count = 0
        self.commands: list[tuple[str, float | None]] = []
        self._container_id = "test-container-123"
        self._workspace_dir = "/repo"

    async def run_command(self, cmd: str, *, timeout: float | None = None):
        self.commands.append((cmd, timeout))
        if self.call_count < len(self.results):
            result = self.results[self.call_count]
            self.call_count += 1
            return result
        return MockExecResult(stdout="", stderr="", exit_code=0)

    @property
    def container_id(self):
        return self._container_id

    @property
    def workspace_dir(self):
        return self._workspace_dir


def create_task(
    *,
    task_id: str = "test-task-1",
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
) -> SweTask:
    return SweTask(
        id=task_id,
        repo="owner/repo",
        base_commit="abc123def456",
        merge_commit="def456",
        language="python",
        fail_to_pass=fail_to_pass or [],
        pass_to_pass=pass_to_pass or [],
        prompt="Fix the bug",
    )


class TestSanityCheckResult:
    def test_result_creation(self):
        result = SanityCheckResult(
            passed=True,
            fail_to_pass_ok=["pytest test_a.py"],
            pass_to_pass_ok=["pytest test_b.py"],
        )
        assert result.passed is True
        assert len(result.fail_to_pass_ok) == 1
        assert len(result.pass_to_pass_ok) == 1

    def test_result_defaults(self):
        result = SanityCheckResult(passed=False)
        assert result.fail_to_pass_ok == []
        assert result.fail_to_pass_bad == []
        assert result.pass_to_pass_ok == []
        assert result.pass_to_pass_bad == []

    def test_summary_passed(self):
        result = SanityCheckResult(
            passed=True,
            fail_to_pass_ok=["test1", "test2"],
            pass_to_pass_ok=["test3"],
        )
        assert "PASSED" in result.summary
        assert "f2p: 2 ok" in result.summary
        assert "p2p: 1 ok" in result.summary

    def test_summary_failed(self):
        result = SanityCheckResult(
            passed=False,
            fail_to_pass_bad=[("test1", "error")],
            pass_to_pass_bad=[("test2", "error")],
        )
        assert "FAILED" in result.summary
        assert "f2p: 0 ok, 1 bad" in result.summary


class TestRunSanityCheck:
    @pytest.mark.asyncio
    async def test_all_tests_behave_correctly(self):
        """Pass case: f2p fails, p2p passes on base commit."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=["pytest test_existing.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="FAILED", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)

        result = await run_sanity_check(sandbox, task)

        assert result.passed is True
        assert result.fail_to_pass_ok == ["pytest test_bug.py"]
        assert result.fail_to_pass_bad == []
        assert result.pass_to_pass_ok == ["pytest test_existing.py"]
        assert result.pass_to_pass_bad == []

    @pytest.mark.asyncio
    async def test_fail_to_pass_unexpectedly_passes(self):
        """Fail case: f2p test passes on base commit (sanity fail)."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)

        result = await run_sanity_check(sandbox, task)

        assert result.passed is False
        assert result.fail_to_pass_ok == []
        assert len(result.fail_to_pass_bad) == 1
        assert result.fail_to_pass_bad[0][0] == "pytest test_bug.py"
        assert "already passes" in result.fail_to_pass_bad[0][1]

    @pytest.mark.asyncio
    async def test_pass_to_pass_unexpectedly_fails(self):
        """Fail case: p2p test fails on base commit (sanity fail)."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=["pytest test_existing.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="FAILED", exit_code=1),
            MockExecResult(stdout="", stderr="ERROR", exit_code=1),
        ]
        sandbox = MockSandbox(results=results)

        result = await run_sanity_check(sandbox, task)

        assert result.passed is False
        assert result.pass_to_pass_ok == []
        assert len(result.pass_to_pass_bad) == 1
        assert result.pass_to_pass_bad[0][0] == "pytest test_existing.py"
        assert "fails on base" in result.pass_to_pass_bad[0][1]

    @pytest.mark.asyncio
    async def test_multiple_f2p_tests_mixed_results(self):
        """Multiple f2p tests with mixed outcomes."""
        task = create_task(
            fail_to_pass=["pytest a.py", "pytest b.py", "pytest c.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="", stderr="", exit_code=1),
        ]
        sandbox = MockSandbox(results=results)

        result = await run_sanity_check(sandbox, task)

        assert result.passed is False
        assert result.fail_to_pass_ok == ["pytest a.py", "pytest c.py"]
        assert len(result.fail_to_pass_bad) == 1
        assert result.fail_to_pass_bad[0][0] == "pytest b.py"

    @pytest.mark.asyncio
    async def test_multiple_p2p_tests_mixed_results(self):
        """Multiple p2p tests with mixed outcomes."""
        task = create_task(
            fail_to_pass=[],
            pass_to_pass=["pytest a.py", "pytest b.py", "pytest c.py"],
        )

        results = [
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="", stderr="FAIL", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)

        result = await run_sanity_check(sandbox, task)

        assert result.passed is False
        assert result.pass_to_pass_ok == ["pytest a.py", "pytest c.py"]
        assert len(result.pass_to_pass_bad) == 1
        assert result.pass_to_pass_bad[0][0] == "pytest b.py"

    @pytest.mark.asyncio
    async def test_empty_test_lists(self):
        """Empty test lists should pass (vacuously true)."""
        task = create_task(
            fail_to_pass=[],
            pass_to_pass=[],
        )

        sandbox = MockSandbox(results=[])

        result = await run_sanity_check(sandbox, task)

        assert result.passed is True
        assert result.fail_to_pass_ok == []
        assert result.fail_to_pass_bad == []
        assert result.pass_to_pass_ok == []
        assert result.pass_to_pass_bad == []

    @pytest.mark.asyncio
    async def test_exception_on_f2p_treated_as_failure(self):
        """Exception during f2p test is treated as test failure (expected)."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=[],
        )

        class ExceptionSandbox(MockSandbox):
            async def run_command(self, cmd: str, *, timeout: float | None = None):
                raise RuntimeError("Command crashed")

        sandbox = ExceptionSandbox(results=[])

        result = await run_sanity_check(sandbox, task)

        assert result.passed is True
        assert result.fail_to_pass_ok == ["pytest test_bug.py"]

    @pytest.mark.asyncio
    async def test_exception_on_p2p_is_error(self):
        """Exception during p2p test is a sanity check failure."""
        task = create_task(
            fail_to_pass=[],
            pass_to_pass=["pytest test_existing.py"],
        )

        class ExceptionSandbox(MockSandbox):
            async def run_command(self, cmd: str, *, timeout: float | None = None):
                raise RuntimeError("Command crashed")

        sandbox = ExceptionSandbox(results=[])

        result = await run_sanity_check(sandbox, task)

        assert result.passed is False
        assert len(result.pass_to_pass_bad) == 1

    @pytest.mark.asyncio
    async def test_timeout_passed_to_sandbox(self):
        """Custom timeout should be passed to sandbox."""
        task = create_task(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        results = [MockExecResult(stdout="", stderr="", exit_code=1)]
        sandbox = MockSandbox(results=results)

        await run_sanity_check(sandbox, task, timeout=60.0)

        assert sandbox.commands[0][1] == 60.0
