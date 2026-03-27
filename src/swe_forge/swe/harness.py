"""SWE-bench evaluation harness.

Runs an external agent on mined SWE tasks inside Docker containers,
then verifies results by executing fail_to_pass / pass_to_pass test commands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from swe_forge.swe.models import SweTask

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Status Enum
# ---------------------------------------------------------------------------


class HarnessStatus(str, Enum):
    """Status of harness evaluation for a SWE task."""

    RESOLVED = "resolved"
    """All tests pass - bug is fixed."""

    UNRESOLVED = "unresolved"
    """Some tests still fail - bug not fully fixed."""

    AGENT_ERROR = "agent_error"
    """Agent execution failed (timeout, crash, etc.)."""

    TEST_ERROR = "test_error"
    """Test execution error (not test failure)."""

    SETUP_ERROR = "setup_error"
    """Container setup failed (clone, install, etc.)."""

    SANITY_FAIL = "sanity_fail"
    """Sanity check failed - tests don't behave as expected on base commit."""


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------


@dataclass
class HarnessTestResult:
    """Result of a single test command execution.

    Attributes:
        command: The test command that was run.
        exit_code: Exit code of the test command.
        stdout: Standard output from the test.
        stderr: Standard error from the test.
        passed: Whether the test passed (exit_code == 0).
        duration_ms: Duration of test execution in milliseconds.
    """

    command: str
    exit_code: int
    stdout: str
    stderr: str
    passed: bool
    duration_ms: float


@dataclass
class HarnessResult:
    """Result of harness evaluation for a single SWE task.

    Attributes:
        task_id: Unique identifier for the task.
        status: Final evaluation status.
        resolved: True if all tests pass (RESOLVED status).
        patch_applied: True if agent changes were captured.
        fail_to_pass_results: Results of fail_to_pass tests after agent.
            Each tuple is (test_command, passed).
        pass_to_pass_results: Results of pass_to_pass tests after agent.
            Each tuple is (test_command, passed).
        duration_seconds: Total time for harness execution.
        error_message: Error message if status is error type.
        agent_output: Captured stdout/stderr from agent execution.
        container_id: Docker container ID used for isolation.
    """

    task_id: str
    status: HarnessStatus
    resolved: bool
    patch_applied: bool = False
    fail_to_pass_results: list[tuple[str, bool]] = field(default_factory=list)
    pass_to_pass_results: list[tuple[str, bool]] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: str | None = None
    agent_output: str | None = None
    container_id: str | None = None

    @property
    def fail_to_pass_passed_count(self) -> int:
        """Count of fail_to_pass tests that passed."""
        return sum(1 for _, passed in self.fail_to_pass_results if passed)

    @property
    def pass_to_pass_passed_count(self) -> int:
        """Count of pass_to_pass tests that passed."""
        return sum(1 for _, passed in self.pass_to_pass_results if passed)

    @property
    def summary(self) -> str:
        """Human-readable summary of results."""
        f2p_total = len(self.fail_to_pass_results)
        f2p_passed = self.fail_to_pass_passed_count
        p2p_total = len(self.pass_to_pass_results)
        p2p_passed = self.pass_to_pass_passed_count

        return (
            f"Task {self.task_id}: {self.status.value} "
            f"(f2p: {f2p_passed}/{f2p_total}, p2p: {p2p_passed}/{p2p_total})"
        )


# ---------------------------------------------------------------------------
# Sandbox Protocol
# ---------------------------------------------------------------------------


class SandboxProtocol(Protocol):
    """Protocol for sandbox implementations used by HarnessRunner."""

    async def run_command(
        self, cmd: str, *, timeout: float | None = None
    ) -> "ExecResultProtocol":
        """Execute a command in the sandbox."""
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox."""
        ...

    async def get_git_status(self) -> str:
        """Get git status in porcelain format."""
        ...

    @property
    def container_id(self) -> str | None:
        """Get container ID."""
        ...

    @property
    def workspace_dir(self) -> str:
        """Get workspace directory."""
        ...


class ExecResultProtocol(Protocol):
    """Protocol for command execution results."""

    @property
    def exit_code(self) -> int: ...

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...


# ---------------------------------------------------------------------------
# Harness Runner
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """Configuration for HarnessRunner.

    Attributes:
        agent_timeout_seconds: Timeout for agent execution.
        test_timeout_seconds: Timeout for test command execution.
        setup_timeout_seconds: Timeout for setup operations.
        keep_containers: If True, don't remove containers after completion.
        capture_patch: If True, capture git diff as agent_patch.
        agent_script: Optional script to run as the agent.
    """

    agent_timeout_seconds: float = 600.0
    test_timeout_seconds: float = 120.0
    setup_timeout_seconds: float = 180.0
    keep_containers: bool = False
    capture_patch: bool = True
    agent_script: str | None = None


class HarnessRunner:
    """Runs evaluation harness for SWE tasks.

    The harness performs the following steps:
    1. Set up container with task repository at base commit
    2. Run sanity check (fail_to_pass must FAIL, pass_to_pass must PASS)
    3. Run agent to make changes (if provided)
    4. Capture agent's changes (git diff)
    5. Run test verification
    6. Generate result report

    Example:
        runner = HarnessRunner()
        result = await runner.run_harness(task, sandbox)
        if result.resolved:
            print(f"Task resolved: {result.task_id}")
    """

    def __init__(
        self,
        config: HarnessConfig | None = None,
    ):
        """Initialize HarnessRunner.

        Args:
            config: Optional HarnessConfig. Uses defaults if not provided.
        """
        self._config = config or HarnessConfig()

    async def run_harness(
        self,
        task: "SweTask",
        sandbox: SandboxProtocol,
        agent_script: str | None = None,
    ) -> HarnessResult:
        """Run the evaluation harness for a single SWE task.

        Args:
            task: The SWE task to evaluate.
            sandbox: Sandbox with repository at base commit.
            agent_script: Optional agent script override.

        Returns:
            HarnessResult with evaluation outcome.
        """
        start_time = time.monotonic()
        agent_script = agent_script or self._config.agent_script

        # Initialize result
        result = HarnessResult(
            task_id=task.id,
            status=HarnessStatus.SETUP_ERROR,
            resolved=False,
            container_id=sandbox.container_id,
        )

        try:
            # Step 1: Run sanity check on base commit
            sanity_passed = await self._run_sanity_check(task, sandbox, result)
            if not sanity_passed:
                result.duration_seconds = time.monotonic() - start_time
                return result

            # Step 2: Run agent if provided
            agent_output = ""
            if agent_script:
                agent_output = await self._run_agent(
                    task, sandbox, agent_script, result
                )
                result.agent_output = self._truncate(agent_output, 10000)

                if result.status == HarnessStatus.AGENT_ERROR:
                    result.duration_seconds = time.monotonic() - start_time
                    return result

            # Step 3: Capture patch if agent made changes
            if self._config.capture_patch:
                patch = await self._capture_patch(sandbox)
                result.patch_applied = bool(patch)

            # Step 4: Run test verification
            await self._run_tests(task, sandbox, result)

            # Step 5: Determine final status
            self._determine_status(result)

        except Exception as e:
            logger.exception(f"Harness execution failed for task {task.id}")
            result.status = HarnessStatus.SETUP_ERROR
            result.error_message = str(e)

        result.duration_seconds = time.monotonic() - start_time
        return result

    async def _run_sanity_check(
        self,
        task: "SweTask",
        sandbox: SandboxProtocol,
        result: HarnessResult,
    ) -> bool:
        """Run sanity check: fail_to_pass must FAIL, pass_to_pass must PASS.

        Args:
            task: The SWE task.
            sandbox: Sandbox at base commit.
            result: Result to update with status.

        Returns:
            True if sanity check passed, False otherwise.
        """
        logger.info(f"Running sanity check for task {task.id}")

        # Check fail_to_pass tests - must FAIL on base
        for cmd in task.fail_to_pass:
            try:
                exec_result = await sandbox.run_command(
                    cmd, timeout=self._config.test_timeout_seconds
                )
                if exec_result.exit_code == 0:
                    # Test passes on base - sanity fail
                    result.status = HarnessStatus.SANITY_FAIL
                    result.error_message = (
                        f"fail_to_pass command already passes on base commit: {cmd}"
                    )
                    logger.warning(result.error_message)
                    return False
            except Exception as e:
                # Command error is treated as failure (expected)
                logger.debug(f"fail_to_pass command error (expected): {e}")

        # Check pass_to_pass tests - must PASS on base
        for cmd in task.pass_to_pass:
            try:
                exec_result = await sandbox.run_command(
                    cmd, timeout=self._config.test_timeout_seconds
                )
                if exec_result.exit_code != 0:
                    # Test fails on base - sanity fail
                    result.status = HarnessStatus.SANITY_FAIL
                    result.error_message = (
                        f"pass_to_pass command fails on base commit: {cmd}"
                    )
                    logger.warning(result.error_message)
                    return False
            except Exception as e:
                # Command error is not expected for pass_to_pass
                result.status = HarnessStatus.SANITY_FAIL
                result.error_message = (
                    f"pass_to_pass command error on base commit: {cmd} - {e}"
                )
                logger.warning(result.error_message)
                return False

        logger.info(f"Sanity check passed for task {task.id}")
        return True

    async def _run_agent(
        self,
        task: "SweTask",
        sandbox: SandboxProtocol,
        agent_script: str,
        result: HarnessResult,
    ) -> str:
        """Run the agent script in the sandbox.

        Args:
            task: The SWE task.
            sandbox: Sandbox at base commit.
            agent_script: Script to run as agent.
            result: Result to update with status.

        Returns:
            Agent output (stdout + stderr).
        """
        logger.info(f"Running agent for task {task.id}")

        # Prepare the agent command
        # If it's a file path, execute it; otherwise run as inline script
        if agent_script.startswith("/") or agent_script.startswith("./"):
            # It's a path to a script file
            agent_cmd = f"cd {sandbox.workspace_dir} && {agent_script}"
        else:
            # Inline script or command
            agent_cmd = f"cd {sandbox.workspace_dir} && {agent_script}"

        # If task has a prompt, we might pass it as an argument
        # Some agents accept --instruction or similar
        if task.prompt:
            prompt_escaped = task.prompt.replace("'", "'\\''")
            agent_cmd = f"{agent_cmd} --instruction '{prompt_escaped}'"

        try:
            exec_result = await sandbox.run_command(
                agent_cmd, timeout=self._config.agent_timeout_seconds
            )

            output = f"{exec_result.stdout}\n{exec_result.stderr}".strip()

            # Check if agent timed out
            if exec_result.exit_code != 0:
                if "timed out" in exec_result.stderr.lower():
                    result.status = HarnessStatus.AGENT_ERROR
                    result.error_message = (
                        f"Agent timed out after {self._config.agent_timeout_seconds}s"
                    )
                else:
                    # Agent exited with non-zero - continue anyway
                    logger.warning(
                        f"Agent exited with code {exec_result.exit_code}, "
                        "continuing to test"
                    )

            return output

        except TimeoutError:
            result.status = HarnessStatus.AGENT_ERROR
            result.error_message = (
                f"Agent timed out after {self._config.agent_timeout_seconds}s"
            )
            logger.error(result.error_message)
            return ""

        except Exception as e:
            result.status = HarnessStatus.AGENT_ERROR
            result.error_message = f"Agent execution error: {e}"
            logger.exception("Agent execution failed")
            return ""

    async def _capture_patch(self, sandbox: SandboxProtocol) -> str:
        """Capture git diff from agent changes.

        Args:
            sandbox: Sandbox with agent changes.

        Returns:
            Git diff string, or empty string if no changes.
        """
        try:
            status = await sandbox.get_git_status()
            if not status.strip():
                return ""

            diff_result = await sandbox.run_command("git diff", timeout=30.0)
            return diff_result.stdout

        except Exception as e:
            logger.warning(f"Failed to capture patch: {e}")
            return ""

    async def _run_tests(
        self,
        task: "SweTask",
        sandbox: SandboxProtocol,
        result: HarnessResult,
    ) -> None:
        """Run test verification after agent changes.

        Args:
            task: The SWE task.
            sandbox: Sandbox with agent changes.
            result: Result to populate with test outcomes.
        """
        logger.info(f"Running test verification for task {task.id}")

        # Run fail_to_pass tests
        for cmd in task.fail_to_pass:
            try:
                start = time.monotonic()
                exec_result = await sandbox.run_command(
                    cmd, timeout=self._config.test_timeout_seconds
                )
                _duration_ms = (time.monotonic() - start) * 1000

                passed = exec_result.exit_code == 0
                result.fail_to_pass_results.append((cmd, passed))

                logger.debug(
                    f"fail_to_pass test '{cmd}': {'PASS' if passed else 'FAIL'}"
                )

            except Exception as e:
                logger.warning(f"fail_to_pass test error: {cmd} - {e}")
                result.fail_to_pass_results.append((cmd, False))

        # Run pass_to_pass tests
        for cmd in task.pass_to_pass:
            try:
                start = time.monotonic()
                exec_result = await sandbox.run_command(
                    cmd, timeout=self._config.test_timeout_seconds
                )
                _duration_ms = (time.monotonic() - start) * 1000

                passed = exec_result.exit_code == 0
                result.pass_to_pass_results.append((cmd, passed))

                logger.debug(
                    f"pass_to_pass test '{cmd}': {'PASS' if passed else 'FAIL'}"
                )

            except Exception as e:
                logger.warning(f"pass_to_pass test error: {cmd} - {e}")
                result.pass_to_pass_results.append((cmd, False))

    def _determine_status(self, result: HarnessResult) -> None:
        """Determine final status based on test results.

        Args:
            result: Result to update with final status.
        """
        # All fail_to_pass must pass
        all_f2p_pass = all(passed for _, passed in result.fail_to_pass_results)

        # All pass_to_pass must pass
        all_p2p_pass = all(passed for _, passed in result.pass_to_pass_results)

        if all_f2p_pass and all_p2p_pass:
            result.status = HarnessStatus.RESOLVED
            result.resolved = True
            logger.info(f"Task {result.task_id}: RESOLVED")
        else:
            result.status = HarnessStatus.UNRESOLVED
            result.resolved = False
            f2p_passed = result.fail_to_pass_passed_count
            p2p_passed = result.pass_to_pass_passed_count
            logger.info(
                f"Task {result.task_id}: UNRESOLVED "
                f"(f2p: {f2p_passed}/{len(result.fail_to_pass_results)}, "
                f"p2p: {p2p_passed}/{len(result.pass_to_pass_results)})"
            )

    @staticmethod
    def _truncate(s: str, max_len: int) -> str:
        """Truncate string to max length with ellipsis.

        Args:
            s: String to truncate.
            max_len: Maximum length.

        Returns:
            Truncated string.
        """
        if len(s) <= max_len:
            return s

        # Find a safe boundary for truncation (UTF-8)
        end = max_len
        while end > 0 and not s[end - 1].isprintable():
            end -= 1

        return s[:end] + "... [truncated]"
