"""Sanity check logic for preflight test validation.

Verifies that tests behave correctly on the base commit:
- fail_to_pass tests MUST FAIL on base commit
- pass_to_pass tests MUST PASS on base commit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from swe_forge.swe.models import SweTask

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------


@dataclass
class SanityCheckResult:
    """Result of sanity check validation.

    Attributes:
        passed: True if all sanity checks passed.
        fail_to_pass_ok: Tests that failed as expected (good).
        fail_to_pass_bad: Tests that unexpectedly passed - (test_command, error_message).
        pass_to_pass_ok: Tests that passed as expected (good).
        pass_to_pass_bad: Tests that unexpectedly failed - (test_command, error_message).
    """

    passed: bool
    fail_to_pass_ok: list[str] = field(default_factory=list)
    fail_to_pass_bad: list[tuple[str, str]] = field(default_factory=list)
    pass_to_pass_ok: list[str] = field(default_factory=list)
    pass_to_pass_bad: list[tuple[str, str]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Human-readable summary of sanity check results."""
        f2p_ok = len(self.fail_to_pass_ok)
        f2p_bad = len(self.fail_to_pass_bad)
        p2p_ok = len(self.pass_to_pass_ok)
        p2p_bad = len(self.pass_to_pass_bad)

        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Sanity check {status} "
            f"(f2p: {f2p_ok} ok, {f2p_bad} bad; p2p: {p2p_ok} ok, {p2p_bad} bad)"
        )


# ---------------------------------------------------------------------------
# Sandbox Protocol
# ---------------------------------------------------------------------------


class SandboxProtocol(Protocol):
    """Protocol for sandbox implementations used by sanity check."""

    async def run_command(
        self, cmd: str, *, timeout: float | None = None
    ) -> "ExecResultProtocol":
        """Execute a command in the sandbox."""
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
# Sanity Check Function
# ---------------------------------------------------------------------------


async def run_sanity_check(
    sandbox: SandboxProtocol,
    task: "SweTask",
    timeout: float = 120.0,
) -> SanityCheckResult:
    """Run sanity check to verify tests behave correctly on base commit.

    This verifies that:
    - fail_to_pass tests FAIL on base commit (they should fail before the fix)
    - pass_to_pass tests PASS on base commit (they should always pass)

    Args:
        sandbox: Sandbox with repository at base commit.
        task: The SWE task with test commands.
        timeout: Timeout for each test command in seconds.

    Returns:
        SanityCheckResult with pass/fail status and details.
    """
    logger.info(f"Running sanity check for task {task.id}")

    result = SanityCheckResult(passed=True)

    # Check fail_to_pass tests - must FAIL on base commit
    for cmd in task.fail_to_pass:
        try:
            exec_result = await sandbox.run_command(cmd, timeout=timeout)
            if exec_result.exit_code == 0:
                # Test passes on base - this is BAD (sanity fail)
                error_msg = "fail_to_pass command already passes on base commit"
                result.fail_to_pass_bad.append((cmd, error_msg))
                result.passed = False
                logger.warning(f"{error_msg}: {cmd}")
            else:
                # Test fails on base - this is GOOD (expected)
                result.fail_to_pass_ok.append(cmd)
                logger.debug(f"fail_to_pass command fails as expected: {cmd}")

        except Exception as e:
            # Command error is treated as failure (expected for fail_to_pass)
            # The test command itself failed to execute, which we treat as "test failed"
            result.fail_to_pass_ok.append(cmd)
            logger.debug(f"fail_to_pass command error (treated as fail): {cmd} - {e}")

    # Check pass_to_pass tests - must PASS on base commit
    for cmd in task.pass_to_pass:
        try:
            exec_result = await sandbox.run_command(cmd, timeout=timeout)
            if exec_result.exit_code != 0:
                # Test fails on base - this is BAD (sanity fail)
                error_msg = "pass_to_pass command fails on base commit"
                stderr = exec_result.stderr.strip() if exec_result.stderr else ""
                detail = f"{error_msg}: {stderr}" if stderr else error_msg
                result.pass_to_pass_bad.append((cmd, detail))
                result.passed = False
                logger.warning(f"{error_msg}: {cmd}")
            else:
                # Test passes on base - this is GOOD (expected)
                result.pass_to_pass_ok.append(cmd)
                logger.debug(f"pass_to_pass command passes as expected: {cmd}")

        except Exception as e:
            # Command error is NOT expected for pass_to_pass
            error_msg = f"pass_to_pass command error: {e}"
            result.pass_to_pass_bad.append((cmd, error_msg))
            result.passed = False
            logger.warning(f"pass_to_pass command error: {cmd} - {e}")

    logger.info(result.summary)
    return result
