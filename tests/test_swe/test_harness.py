"""Tests for swe_forge.swe.harness module."""

import pytest
from dataclasses import dataclass

from swe_forge.swe.harness import (
    HarnessConfig,
    HarnessResult,
    HarnessRunner,
    HarnessStatus,
    HarnessTestResult,
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
        self.written_files: dict[str, str] = {}
        self._container_id = "test-container-123"
        self._workspace_dir = "/repo"
        self._git_status = ""

    async def run_command(self, cmd: str, *, timeout: float | None = None):
        self.commands.append((cmd, timeout))
        if self.call_count < len(self.results):
            result = self.results[self.call_count]
            self.call_count += 1
            return result
        return MockExecResult(stdout="", stderr="", exit_code=0)

    async def write_file(self, path: str, content: str):
        self.written_files[path] = content

    async def get_git_status(self):
        return self._git_status

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
    prompt: str = "Fix the bug",
    base_commit: str = "abc123def456",
) -> SweTask:
    return SweTask(
        id=task_id,
        repo="owner/repo",
        base_commit=base_commit,
        merge_commit="def456",
        language="python",
        fail_to_pass=fail_to_pass or [],
        pass_to_pass=pass_to_pass or [],
        prompt=prompt,
    )


class TestHarnessStatus:
    def test_status_values(self):
        assert HarnessStatus.RESOLVED.value == "resolved"
        assert HarnessStatus.UNRESOLVED.value == "unresolved"
        assert HarnessStatus.AGENT_ERROR.value == "agent_error"
        assert HarnessStatus.TEST_ERROR.value == "test_error"
        assert HarnessStatus.SETUP_ERROR.value == "setup_error"
        assert HarnessStatus.SANITY_FAIL.value == "sanity_fail"

    def test_status_is_string_enum(self):
        assert isinstance(HarnessStatus.RESOLVED, str)
        assert HarnessStatus.RESOLVED == "resolved"


class TestHarnessTestResult:
    def test_test_result_creation(self):
        result = HarnessTestResult(
            command="pytest test.py",
            exit_code=0,
            stdout="OK",
            stderr="",
            passed=True,
            duration_ms=1500.0,
        )
        assert result.command == "pytest test.py"
        assert result.exit_code == 0
        assert result.passed is True

    def test_test_result_failed(self):
        result = HarnessTestResult(
            command="pytest test.py",
            exit_code=1,
            stdout="",
            stderr="FAILED",
            passed=False,
            duration_ms=500.0,
        )
        assert result.passed is False


class TestHarnessResult:
    def test_result_creation(self):
        result = HarnessResult(
            task_id="test-1",
            status=HarnessStatus.RESOLVED,
            resolved=True,
            fail_to_pass_results=[("pytest test.py", True)],
            pass_to_pass_results=[("pytest other.py", True)],
        )
        assert result.task_id == "test-1"
        assert result.status == HarnessStatus.RESOLVED
        assert result.resolved is True
        assert result.patch_applied is False

    def test_result_defaults(self):
        result = HarnessResult(
            task_id="test-1",
            status=HarnessStatus.UNRESOLVED,
            resolved=False,
        )
        assert result.fail_to_pass_results == []
        assert result.pass_to_pass_results == []
        assert result.error_message is None
        assert result.agent_output is None

    def test_passed_counts(self):
        result = HarnessResult(
            task_id="test-1",
            status=HarnessStatus.UNRESOLVED,
            resolved=False,
            fail_to_pass_results=[
                ("pytest a.py", True),
                ("pytest b.py", False),
                ("pytest c.py", True),
            ],
            pass_to_pass_results=[
                ("pytest d.py", True),
                ("pytest e.py", True),
            ],
        )
        assert result.fail_to_pass_passed_count == 2
        assert result.pass_to_pass_passed_count == 2

    def test_summary(self):
        result = HarnessResult(
            task_id="test-123",
            status=HarnessStatus.RESOLVED,
            resolved=True,
            fail_to_pass_results=[("pytest test.py", True)],
            pass_to_pass_results=[("pytest other.py", True)],
        )
        assert "test-123" in result.summary
        assert "resolved" in result.summary


class TestHarnessConfig:
    def test_config_defaults(self):
        config = HarnessConfig()
        assert config.agent_timeout_seconds == 600.0
        assert config.test_timeout_seconds == 120.0
        assert config.setup_timeout_seconds == 180.0
        assert config.keep_containers is False
        assert config.capture_patch is True
        assert config.agent_script is None

    def test_config_custom_values(self):
        config = HarnessConfig(
            agent_timeout_seconds=300.0,
            test_timeout_seconds=60.0,
            keep_containers=True,
        )
        assert config.agent_timeout_seconds == 300.0
        assert config.test_timeout_seconds == 60.0
        assert config.keep_containers is True


class TestHarnessRunner:
    def test_runner_initialization(self):
        runner = HarnessRunner()
        assert runner._config is not None

    def test_runner_with_config(self):
        config = HarnessConfig(agent_timeout_seconds=300.0)
        runner = HarnessRunner(config=config)
        assert runner._config.agent_timeout_seconds == 300.0


class TestHarnessRunnerSanityCheck:
    @pytest.mark.asyncio
    async def test_sanity_fail_f2p_already_passes(self):
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.status == HarnessStatus.SANITY_FAIL
        assert "already passes on base" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_sanity_fail_p2p_fails_on_base(self):
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=["pytest test_existing.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="FAILED", exit_code=1),
            MockExecResult(stdout="", stderr="ERROR", exit_code=1),
        ]
        sandbox = MockSandbox(results=results)

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.status == HarnessStatus.SANITY_FAIL
        assert "fails on base" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_sanity_check_passes(self):
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=["pytest test_existing.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="FAILED", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="", stderr="", exit_code=0),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.status == HarnessStatus.RESOLVED
        assert result.resolved is True


class TestHarnessRunnerTests:
    @pytest.mark.asyncio
    async def test_all_tests_pass_resolved(self):
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=["pytest test_existing.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="", stderr="", exit_code=0),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.status == HarnessStatus.RESOLVED
        assert result.resolved is True
        assert result.fail_to_pass_passed_count == 1
        assert result.pass_to_pass_passed_count == 1

    @pytest.mark.asyncio
    async def test_f2p_still_fails_unresolved(self):
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="FAILED", exit_code=1),
            MockExecResult(stdout="", stderr="STILL FAILED", exit_code=1),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.status == HarnessStatus.UNRESOLVED
        assert result.resolved is False
        assert result.fail_to_pass_passed_count == 0

    @pytest.mark.asyncio
    async def test_p2p_fails_unresolved(self):
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=["pytest test_regression.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="", stderr="REGRESSION", exit_code=1),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.status == HarnessStatus.UNRESOLVED
        assert result.pass_to_pass_passed_count == 0

    @pytest.mark.asyncio
    async def test_multiple_tests_mixed_results(self):
        task = create_task(
            fail_to_pass=["pytest a.py", "pytest b.py"],
            pass_to_pass=["pytest c.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.status == HarnessStatus.UNRESOLVED
        assert result.fail_to_pass_passed_count == 1
        assert result.pass_to_pass_passed_count == 1


class TestHarnessRunnerAgent:
    @pytest.mark.asyncio
    async def test_agent_script_execution(self):
        task = create_task(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="Agent output", stderr="", exit_code=0),
            MockExecResult(stdout="diff --git a/file.py", stderr="", exit_code=0),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = "M file.py"

        config = HarnessConfig(agent_script="python fix_bug.py")
        runner = HarnessRunner(config=config)

        result = await runner.run_harness(task, sandbox, agent_script="python fix.py")

        assert result.agent_output is not None
        assert result.patch_applied is True

    @pytest.mark.asyncio
    async def test_agent_timeout_detection(self):
        task = create_task(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="", stderr="timed out after 60s", exit_code=137),
        ]
        sandbox = MockSandbox(results=results)

        config = HarnessConfig(agent_script="python slow_agent.py")
        runner = HarnessRunner(config=config)

        result = await runner.run_harness(task, sandbox)

        assert result.status == HarnessStatus.AGENT_ERROR
        assert (
            "timed out" in (result.error_message or "").lower()
            or result.status == HarnessStatus.AGENT_ERROR
        )

    @pytest.mark.asyncio
    async def test_agent_non_zero_exit_continues(self):
        task = create_task(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="Agent done", stderr="warning", exit_code=1),
            MockExecResult(stdout="", stderr="", exit_code=0),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = "M file.py"

        config = HarnessConfig(agent_script="python agent.py")
        runner = HarnessRunner(config=config)

        result = await runner.run_harness(task, sandbox)

        # Agent exited non-zero but test passed -> RESOLVED
        assert result.status == HarnessStatus.RESOLVED


class TestHarnessRunnerEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_test_lists(self):
        task = create_task(
            fail_to_pass=[],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        # No tests to run means resolved (vacuously true)
        assert result.status == HarnessStatus.RESOLVED
        assert result.resolved is True

    @pytest.mark.asyncio
    async def test_exception_during_test_command(self):
        task = create_task(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        class ExceptionSandbox(MockSandbox):
            def __init__(self):
                super().__init__()
                self._call_count = 0

            async def run_command(self, cmd: str, *, timeout: float | None = None):
                self._call_count += 1
                if self._call_count == 3:  # First test after sanity
                    raise RuntimeError("Test crashed")
                if self._call_count <= 2:
                    return MockExecResult(stdout="", stderr="", exit_code=1)
                return MockExecResult(stdout="", stderr="", exit_code=0)

        sandbox = ExceptionSandbox()
        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        # Should still return a result, tests marked as failed
        assert len(result.fail_to_pass_results) == 1
        assert result.fail_to_pass_results[0] == ("pytest test.py", False)

    @pytest.mark.asyncio
    async def test_capture_patch_disabled(self):
        task = create_task(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = "M file.py"

        config = HarnessConfig(capture_patch=False, agent_script="python agent.py")
        runner = HarnessRunner(config=config)

        result = await runner.run_harness(task, sandbox)

        assert result.patch_applied is False

    @pytest.mark.asyncio
    async def test_container_id_captured(self):
        task = create_task(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._container_id = "container-xyz-789"
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.container_id == "container-xyz-789"


class TestHarnessRunnerSummary:
    @pytest.mark.asyncio
    async def test_summary_includes_task_id(self):
        task = create_task(
            task_id="unique-task-42",
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert "unique-task-42" in result.summary

    @pytest.mark.asyncio
    async def test_duration_recorded(self):
        task = create_task(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=1),
            MockExecResult(stdout="OK", stderr="", exit_code=0),
        ]
        sandbox = MockSandbox(results=results)
        sandbox._git_status = ""

        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)

        assert result.duration_seconds >= 0
