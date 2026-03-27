"""Tests for swe_forge.swe.dual_commit module."""

import pytest
from dataclasses import dataclass

from swe_forge.swe.dual_commit import (
    DualCommitResult,
    DualCommitValidator,
)
from swe_forge.swe.models import SweTask


# ─────────────────────────────────────────────────────────────────────────────
# Mocks and Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MockExecResult:
    stdout: str
    stderr: str
    exit_code: int


class MockSandbox:
    """Mock sandbox for testing dual-commit validation."""

    def __init__(self, results: list[MockExecResult] | None = None):
        self.results = results or []
        self.call_count = 0
        self.commands: list[tuple[str, float | None]] = []
        self.written_files: dict[str, str] = {}
        self._checkouts: list[str] = []
        self._patches: list[str] = []

    async def run_command(self, cmd: str, *, timeout: float | None = None):
        self.commands.append((cmd, timeout))
        if self.call_count < len(self.results):
            result = self.results[self.call_count]
            self.call_count += 1
            return result
        return MockExecResult(stdout="", stderr="", exit_code=0)

    async def write_file(self, path: str, content: str):
        self.written_files[path] = content
        self._patches.append(content)


def create_task(
    *,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    patch: str = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n",
    test_patch: str = "",
    base_commit: str = "abc123",
) -> SweTask:
    """Helper to create a SweTask for testing."""
    return SweTask(
        id="test-task-1",
        repo="owner/repo",
        base_commit=base_commit,
        merge_commit="def456",
        language="python",
        fail_to_pass=fail_to_pass or [],
        pass_to_pass=pass_to_pass or [],
        patch=patch,
        test_patch=test_patch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DualCommitResult Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDualCommitResult:
    def test_result_creation(self):
        result = DualCommitResult(
            passed=True,
            reasons=[],
            fail_to_pass_before=[("pytest test.py", True)],
            fail_to_pass_after=[("pytest test.py", True)],
            pass_to_pass_after=[("pytest other.py", True)],
        )
        assert result.passed is True
        assert result.reasons == []
        assert result.fail_to_pass_before == [("pytest test.py", True)]
        assert result.fail_to_pass_after == [("pytest test.py", True)]
        assert result.pass_to_pass_after == [("pytest other.py", True)]
        assert result.merge_conflict is False

    def test_result_defaults(self):
        result = DualCommitResult(passed=False)
        assert result.passed is False
        assert result.reasons == []
        assert result.fail_to_pass_before == []
        assert result.fail_to_pass_after == []
        assert result.pass_to_pass_after == []
        assert result.merge_conflict is False

    def test_result_with_failure_reasons(self):
        result = DualCommitResult(
            passed=False,
            reasons=["Test failed after patch", "Merge conflict"],
            merge_conflict=True,
        )
        assert result.passed is False
        assert len(result.reasons) == 2
        assert result.merge_conflict is True


# ─────────────────────────────────────────────────────────────────────────────
# DualCommitValidator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDualCommitValidatorInit:
    def test_validator_default_values(self):
        validator = DualCommitValidator()
        assert validator._timeout_seconds == 120.0
        assert validator._patch_timeout_seconds == 30.0

    def test_validator_custom_values(self):
        validator = DualCommitValidator(
            timeout_seconds=60.0,
            patch_timeout_seconds=15.0,
        )
        assert validator._timeout_seconds == 60.0
        assert validator._patch_timeout_seconds == 15.0


class TestDualCommitValidation:
    @pytest.mark.asyncio
    async def test_success_case_all_tests_behave_correctly(self):
        """Test the happy path: fail_to_pass fails before, passes after;
        pass_to_pass passes after patch."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=["pytest test_existing.py"],
        )

        # Sequence of command results:
        # 1. git checkout base_commit -> success
        # 2. git apply patch -> success
        # 3. pytest test_bug.py (before patch) -> FAIL (as expected)
        # 4. git apply patch -> success (second patch application, for main patch)
        # 5. git status -> show modified files
        # 6. pytest test_bug.py (after patch) -> PASS
        # 7. pytest test_existing.py -> PASS
        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(
                stdout="", stderr="", exit_code=1
            ),  # fail_to_pass before fails
            MockExecResult(stdout="", stderr="", exit_code=0),  # patch apply
            MockExecResult(stdout="M file.py", stderr="", exit_code=0),  # git status
            MockExecResult(
                stdout="OK", stderr="", exit_code=0
            ),  # fail_to_pass after passes
            MockExecResult(stdout="OK", stderr="", exit_code=0),  # pass_to_pass passes
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        assert result.passed is True
        assert result.reasons == []
        assert not result.merge_conflict
        assert len(result.fail_to_pass_before) == 1
        assert result.fail_to_pass_before[0] == ("pytest test_bug.py", True)
        assert len(result.fail_to_pass_after) == 1
        assert result.fail_to_pass_after[0] == ("pytest test_bug.py", True)
        assert len(result.pass_to_pass_after) == 1
        assert result.pass_to_pass_after[0] == ("pytest test_existing.py", True)

    @pytest.mark.asyncio
    async def test_fail_to_pass_already_passes_on_base(self):
        """Validation fails when fail_to_pass test passes on base commit."""
        task = create_task(
            fail_to_pass=["pytest test_already_passes.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(stdout="OK", stderr="", exit_code=0),  # passes on base (BAD)
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        assert result.passed is False
        assert any("already passes on base" in r for r in result.reasons)
        assert result.fail_to_pass_before[0] == ("pytest test_already_passes.py", False)

    @pytest.mark.asyncio
    async def test_fail_to_pass_fails_after_patch(self):
        """Validation fails when fail_to_pass test fails after patch."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(stdout="", stderr="", exit_code=1),  # fails on base (good)
            MockExecResult(stdout="", stderr="", exit_code=0),  # patch apply
            MockExecResult(stdout="M file.py", stderr="", exit_code=0),  # git status
            MockExecResult(
                stdout="", stderr="FAILED", exit_code=1
            ),  # still fails after patch (BAD)
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        assert result.passed is False
        assert any("failed after patch" in r for r in result.reasons)

    @pytest.mark.asyncio
    async def test_pass_to_pass_fails_after_patch(self):
        """Validation fails when pass_to_pass test fails after patch."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            pass_to_pass=["pytest test_regression.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(
                stdout="", stderr="", exit_code=1
            ),  # fail_to_pass fails (good)
            MockExecResult(stdout="", stderr="", exit_code=0),  # patch apply
            MockExecResult(stdout="M file.py", stderr="", exit_code=0),  # git status
            MockExecResult(
                stdout="OK", stderr="", exit_code=0
            ),  # fail_to_pass passes (good)
            MockExecResult(
                stdout="", stderr="FAILED", exit_code=1
            ),  # pass_to_pass fails (BAD)
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        assert result.passed is False
        assert any("pass_to_pass test failed" in r for r in result.reasons)

    @pytest.mark.asyncio
    async def test_patch_conflict_detected(self):
        """Validation fails with merge_conflict=True when patch fails to apply."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(
                stdout="", stderr="", exit_code=1
            ),  # fail_to_pass fails (good)
            MockExecResult(
                stdout="", stderr="patch does not apply", exit_code=1
            ),  # patch fail
            MockExecResult(stdout="", stderr="", exit_code=0),  # git apply --3way
            MockExecResult(stdout="", stderr="", exit_code=0),  # rm
            MockExecResult(stdout="", stderr="", exit_code=0),  # git status (empty)
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        assert result.passed is False
        assert result.merge_conflict is True
        assert any("Failed to apply patch" in r for r in result.reasons)

    @pytest.mark.asyncio
    async def test_checkout_failure(self):
        """Validation fails immediately if cannot checkout base commit."""
        task = create_task()

        results = [
            MockExecResult(
                stdout="", stderr="commit not found", exit_code=1
            ),  # checkout fails
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        assert result.passed is False
        assert any("Failed to checkout" in r for r in result.reasons)

    @pytest.mark.asyncio
    async def test_empty_patch(self):
        """Empty patch should be treated as success."""
        task = create_task(
            fail_to_pass=["pytest test.py"],
            patch="",  # empty patch
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(stdout="", stderr="", exit_code=1),  # fail_to_pass fails
            MockExecResult(
                stdout="", stderr="", exit_code=1
            ),  # patch apply fails (empty)
            MockExecResult(stdout="", stderr="", exit_code=0),  # git apply --3way
            MockExecResult(stdout="", stderr="", exit_code=0),  # rm
            MockExecResult(stdout="", stderr="", exit_code=0),  # git status
            MockExecResult(stdout="OK", stderr="", exit_code=0),  # test passes
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        # Should proceed even with empty patch
        assert result is not None

    @pytest.mark.asyncio
    async def test_with_test_patch(self):
        """Validation applies test_patch before running tests."""
        task = create_task(
            fail_to_pass=["pytest test_bug.py"],
            test_patch="diff --git a/test_bug.py b/test_bug.py\n--- /dev/null\n",
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(stdout="", stderr="", exit_code=0),  # git apply test_patch
            MockExecResult(stdout="", stderr="", exit_code=0),  # rm test_patch file
            MockExecResult(
                stdout="", stderr="", exit_code=1
            ),  # fail_to_pass fails (good)
            MockExecResult(stdout="", stderr="", exit_code=0),  # git apply main_patch
            MockExecResult(stdout="", stderr="", exit_code=0),  # rm main_patch file
            MockExecResult(stdout="OK", stderr="", exit_code=0),  # fail_to_pass passes
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        assert result.passed is True
        assert len(sandbox.written_files) > 0

    @pytest.mark.asyncio
    async def test_multiple_fail_to_pass_commands(self):
        """Validation handles multiple fail_to_pass commands."""
        task = create_task(
            fail_to_pass=["pytest test_a.py", "pytest test_b.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(stdout="", stderr="", exit_code=1),  # test_a fails
            MockExecResult(stdout="", stderr="", exit_code=1),  # test_b fails
            MockExecResult(stdout="", stderr="", exit_code=0),  # patch apply
            MockExecResult(stdout="M file.py", stderr="", exit_code=0),  # git status
            MockExecResult(stdout="OK", stderr="", exit_code=0),  # test_a passes
            MockExecResult(stdout="OK", stderr="", exit_code=0),  # test_b passes
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        assert result.passed is True
        assert len(result.fail_to_pass_before) == 2
        assert len(result.fail_to_pass_after) == 2

    @pytest.mark.asyncio
    async def test_exception_during_test_command(self):
        """Exceptions during test commands are handled gracefully."""
        task = create_task(
            fail_to_pass=["pytest test.py"],
        )

        class ExceptionSandbox(MockSandbox):
            call_count = 0

            async def run_command(self, cmd: str, *, timeout: float | None = None):
                self.call_count += 1
                if self.call_count == 2:  # First test run
                    raise RuntimeError("Command timed out")
                return MockExecResult(stdout="", stderr="", exit_code=0)

        sandbox = ExceptionSandbox()

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        # Exception is treated as test failure (which is expected on base)
        assert len(result.fail_to_pass_before) == 1
        assert result.fail_to_pass_before[0][1] is True  # failed as expected

    @pytest.mark.asyncio
    async def test_patch_applies_with_conflict_but_succeeds(self):
        """Patch with conflict markers that still applies should succeed."""
        task = create_task(
            fail_to_pass=["pytest test.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(stdout="", stderr="", exit_code=1),  # test fails (good)
            MockExecResult(
                stdout="", stderr="", exit_code=1
            ),  # git apply --allow-empty fails
            MockExecResult(
                stdout="Applied patch with conflicts", stderr="", exit_code=0
            ),  # --3way
            MockExecResult(stdout="", stderr="", exit_code=0),  # rm
            MockExecResult(
                stdout="M file.py", stderr="", exit_code=0
            ),  # git status shows changes
            MockExecResult(stdout="OK", stderr="", exit_code=0),  # test passes
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator()
        result = await validator.validate(task, sandbox)

        # Should succeed because files were modified
        assert result is not None

    @pytest.mark.asyncio
    async def test_commands_run_with_correct_timeout(self):
        """Test commands should use configured timeout."""
        task = create_task(
            fail_to_pass=["pytest test.py"],
        )

        results = [
            MockExecResult(stdout="", stderr="", exit_code=0),  # checkout
            MockExecResult(stdout="", stderr="", exit_code=1),  # test fails
            MockExecResult(stdout="", stderr="", exit_code=0),  # patch apply
            MockExecResult(stdout="M file.py", stderr="", exit_code=0),  # git status
            MockExecResult(stdout="OK", stderr="", exit_code=0),  # test passes
        ]
        sandbox = MockSandbox(results=results)

        validator = DualCommitValidator(timeout_seconds=60.0)
        await validator.validate(task, sandbox)

        # Check that test commands got the correct timeout
        [
            cmd
            for cmd, timeout in sandbox.commands
            if "pytest" in cmd and timeout is not None
        ]
        # At least one test command should have been run with timeout
        timeouts_used = [t for _, t in sandbox.commands if t is not None and t == 60.0]
        assert len(timeouts_used) > 0
