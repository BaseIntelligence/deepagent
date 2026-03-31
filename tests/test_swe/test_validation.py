"""Tests for Docker image validation logic."""

from unittest.mock import AsyncMock, MagicMock, patch
import subprocess

import pytest

from swe_forge.publish.docker_builder import verify_docker_image, VerifyResult


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for Docker operations."""
    with patch("swe_forge.publish.docker_builder.subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def workspace_with_tests() -> dict:
    """Create a sample workspace dict with test configuration."""
    return {
        "task_id": "test-owner-repo-123",
        "tests": {
            "fail_to_pass": ["pytest tests/test_feature.py -v"],
            "pass_to_pass": ["pytest tests/test_other.py -v"],
        },
        "repo": {
            "url": "https://github.com/test/repo.git",
            "base_commit": "abc123",
        },
        "language": "python",
    }


class TestVerifyResultDataclass:
    """Tests for VerifyResult dataclass behavior."""

    def test_default_values(self):
        """Test VerifyResult default values."""
        result = VerifyResult(success=True)
        assert result.success is True
        assert result.before_patch_fail is False
        assert result.after_patch_pass is False
        assert result.pass_to_pass_ok is True
        assert result.error is None
        assert result.details is None

    def test_custom_values(self):
        """Test VerifyResult with custom values."""
        result = VerifyResult(
            success=True,
            before_patch_fail=True,
            after_patch_pass=True,
            pass_to_pass_ok=True,
            details={"before": [], "after": []},
        )
        assert result.success is True
        assert result.before_patch_fail is True
        assert result.after_patch_pass is True
        assert result.pass_to_pass_ok is True
        assert result.details == {"before": [], "after": []}

    def test_failure_result(self):
        """Test VerifyResult failure case."""
        result = VerifyResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_equality(self):
        """Test VerifyResult equality."""
        result1 = VerifyResult(success=True, before_patch_fail=True)
        result2 = VerifyResult(success=True, before_patch_fail=True)
        assert result1 == result2


class TestVerifyDockerImage:
    """Tests for verify_docker_image function."""

    @pytest.mark.asyncio
    async def test_verify_passes_when_tests_fail_before_and_pass_after(
        self, workspace_with_tests
    ):
        """Test that verification passes when tests fail before patch and pass after."""
        mock_results = []
        call_count = {"count": 0}

        def mock_run_side_effect(*args, **kwargs):
            call_count["count"] += 1
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""

            # Handle different Docker commands
            if "rm" in args[0] and "-f" in args[0]:
                # Docker rm command
                return mock_result
            elif "run" in args[0] and "-d" in args[0]:
                # Docker run command
                return mock_result
            elif "exec" in args[0] and "git apply" in str(args[0]):
                # Patch application
                return mock_result
            elif "exec" in args[0] and "pytest" in str(args[0]):
                # Test execution
                test_phase = call_count["count"]
                mock_result.returncode = 1 if test_phase < 50 else 0
                return mock_result
            elif "exec" in args[0]:
                # Other exec commands (bash setup)
                return mock_result
            return mock_result

        with patch(
            "swe_forge.publish.docker_builder.subprocess.run",
            side_effect=mock_run_side_effect,
        ):
            with patch("swe_forge.publish.docker_builder.time.sleep"):
                with patch(
                    "swe_forge.publish.docker_builder._run_test_in_container"
                ) as mock_run_test:
                    # Before patch: tests fail
                    # After patch: tests pass
                    mock_run_test.side_effect = [
                        # Before patch - tests fail (exit_code != 0)
                        {
                            "command": "pytest tests/test_feature.py -v",
                            "exit_code": 1,
                            "success": False,
                            "output": "FAILED",
                            "error": "AssertionError",
                        },
                        # After patch - tests pass (exit_code == 0)
                        {
                            "command": "pytest tests/test_feature.py -v",
                            "exit_code": 0,
                            "success": True,
                            "output": "PASSED",
                            "error": "",
                        },
                        # pass_to_pass test - passes
                        {
                            "command": "pytest tests/test_other.py -v",
                            "exit_code": 0,
                            "success": True,
                            "output": "PASSED",
                            "error": "",
                        },
                    ]
                    result = await verify_docker_image(
                        "test-image:latest", workspace_with_tests
                    )

        assert result.success is True
        assert result.before_patch_fail is True
        assert result.after_patch_pass is True
        assert result.pass_to_pass_ok is True

    @pytest.mark.asyncio
    async def test_verify_fails_when_tests_pass_before_patch(
        self, workspace_with_tests
    ):
        """Test that verification fails when tests pass before patch (bug doesn't exist)."""
        with patch("swe_forge.publish.docker_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            with patch("swe_forge.publish.docker_builder.time.sleep"):
                with patch(
                    "swe_forge.publish.docker_builder._run_test_in_container"
                ) as mock_run_test:
                    # Before patch - tests PASS (shouldn't happen for valid task)
                    mock_run_test.return_value = {
                        "command": "pytest tests/test_feature.py -v",
                        "exit_code": 0,
                        "success": True,
                        "output": "PASSED",
                        "error": "",
                    }

                    result = await verify_docker_image(
                        "test-image:latest", workspace_with_tests
                    )

        assert result.success is False
        assert result.before_patch_fail is False
        assert "PASS before patch" in result.error

    @pytest.mark.asyncio
    async def test_verify_fails_when_tests_fail_after_patch(self, workspace_with_tests):
        """Test that verification fails when tests still fail after patch (patch doesn't fix bug)."""
        mock_run_test_calls = []

        def mock_run_test_side_effect(container_name, test_cmd, timeout=120):
            call_count = len(mock_run_test_calls)
            mock_run_test_calls.append(call_count)

            # First call: before patch - test fails
            if call_count == 0:
                return {
                    "command": test_cmd,
                    "exit_code": 1,
                    "success": False,
                    "output": "FAILED",
                    "error": "AssertionError",
                }
            # Second call: after patch - test still fails
            return {
                "command": test_cmd,
                "exit_code": 1,
                "success": False,
                "output": "FAILED",
                "error": "AssertionError still",
            }

        with patch("swe_forge.publish.docker_builder.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            with patch("swe_forge.publish.docker_builder.time.sleep"):
                with patch(
                    "swe_forge.publish.docker_builder._run_test_in_container",
                    side_effect=mock_run_test_side_effect,
                ):
                    result = await verify_docker_image(
                        "test-image:latest", workspace_with_tests
                    )

        assert result.success is False
        assert result.before_patch_fail is True
        assert result.after_patch_pass is False
        assert "FAIL after patch" in result.error

    @pytest.mark.asyncio
    async def test_verify_fails_with_no_fail_to_pass_tests(self):
        """Test that verification fails when no fail_to_pass tests are defined."""
        workspace = {
            "task_id": "test-owner-repo-123",
            "tests": {
                "fail_to_pass": [],  # Empty
                "pass_to_pass": ["pytest tests/test_other.py -v"],
            },
        }

        result = await verify_docker_image("test-image:latest", workspace)

        assert result.success is False
        assert "No fail_to_pass tests" in result.error

    @pytest.mark.asyncio
    async def test_verify_handles_patch_failure(self, workspace_with_tests):
        """Test that verification handles patch application failures."""
        mock_run_test_calls = []

        def mock_run_test_side_effect(container_name, test_cmd, timeout=120):
            mock_run_test_calls.append(1)
            return {
                "command": test_cmd,
                "exit_code": 1,
                "success": False,
                "output": "FAILED",
                "error": "AssertionError",
            }

        def mock_subprocess_run(*args, **kwargs):
            mock_result = MagicMock()
            # Docker exec with git apply fails
            if "exec" in args[0] and "git apply" in str(args[0]):
                mock_result.returncode = 1
                mock_result.stderr = "error: patch failed"
            else:
                mock_result.returncode = 0
            mock_result.stdout = ""
            return mock_result

        with patch(
            "swe_forge.publish.docker_builder.subprocess.run",
            side_effect=mock_subprocess_run,
        ):
            with patch("swe_forge.publish.docker_builder.time.sleep"):
                with patch(
                    "swe_forge.publish.docker_builder._run_test_in_container",
                    side_effect=mock_run_test_side_effect,
                ):
                    result = await verify_docker_image(
                        "test-image:latest", workspace_with_tests
                    )

        assert result.success is False
        assert "Failed to apply patch" in result.error

    @pytest.mark.asyncio
    async def test_verify_handles_timeout(self, workspace_with_tests):
        """Test that verification handles timeout gracefully."""
        call_count = {"count": 0}

        def mock_run_side_effect(*args, **kwargs):
            call_count["count"] += 1
            # First call is docker rm cleanup (before starting) - should succeed
            if call_count["count"] == 1:
                return MagicMock(returncode=0, stdout="", stderr="")
            # Second call is docker run - raise timeout
            if call_count["count"] == 2:
                raise subprocess.TimeoutExpired(cmd="docker", timeout=300)
            # Finally block calls docker rm again - should succeed for cleanup
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch(
            "swe_forge.publish.docker_builder.subprocess.run",
            side_effect=mock_run_side_effect,
        ):
            with patch("swe_forge.publish.docker_builder.time.sleep"):
                result = await verify_docker_image(
                    "test-image:latest", workspace_with_tests, timeout=300
                )

        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_verify_handles_regression(self, workspace_with_tests):
        """Test detection of pass_to_pass regression (tests that should stay passing fail)."""
        with patch("swe_forge.publish.docker_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            with patch("swe_forge.publish.docker_builder.time.sleep"):
                with patch(
                    "swe_forge.publish.docker_builder._run_test_in_container"
                ) as mock_run_test:
                    mock_run_test.side_effect = [
                        # Before patch - fail_to_pass test fails
                        {
                            "command": "pytest tests/test_feature.py -v",
                            "exit_code": 1,
                            "success": False,
                            "output": "FAILED",
                            "error": "",
                        },
                        # After patch - fail_to_pass test passes
                        {
                            "command": "pytest tests/test_feature.py -v",
                            "exit_code": 0,
                            "success": True,
                            "output": "PASSED",
                            "error": "",
                        },
                        # pass_to_pass test FAILS (regression!)
                        {
                            "command": "pytest tests/test_other.py -v",
                            "exit_code": 1,
                            "success": False,
                            "output": "FAILED",
                            "error": "Regression",
                        },
                    ]

                    result = await verify_docker_image(
                        "test-image:latest", workspace_with_tests
                    )

        assert result.success is True  # Main verification passes
        assert result.before_patch_fail is True
        assert result.after_patch_pass is True
        assert result.pass_to_pass_ok is False  # Regression detected


class TestVerifyDockerImageFlakyTests:
    """Tests for flaky test handling."""

    @pytest.mark.asyncio
    async def test_verify_with_partial_before_failure(self, workspace_with_tests):
        """Test handling when only some tests fail before patch (partial failure)."""
        workspace_with_tests["tests"]["fail_to_pass"] = [
            "pytest tests/test_a.py -v",
            "pytest tests/test_b.py -v",
        ]

        call_index = {"count": 0}

        def mock_run_test_side_effect(container_name, test_cmd, timeout=120):
            call_index["count"] += 1
            count = call_index["count"]

            # Calls 1-2: before patch (test_a fails, test_b passes - partial)
            # Calls 3-4: after patch (both pass)
            if count == 1:
                return {
                    "command": test_cmd,
                    "exit_code": 1,
                    "success": False,
                    "output": "FAILED",
                    "error": "",
                }
            elif count == 2:
                # This is called but we check if ALL passed before patch
                return {
                    "command": test_cmd,
                    "exit_code": 0,  # One test passes before patch
                    "success": True,
                    "output": "PASSED",
                    "error": "",
                }
            elif count == 3:
                return {
                    "command": test_cmd,
                    "exit_code": 0,
                    "success": True,
                    "output": "PASSED",
                    "error": "",
                }
            elif count == 4:
                return {
                    "command": test_cmd,
                    "exit_code": 0,
                    "success": True,
                    "output": "PASSED",
                    "error": "",
                }
            return {
                "command": test_cmd,
                "exit_code": 0,
                "success": True,
                "output": "PASSED",
                "error": "",
            }

        with patch("swe_forge.publish.docker_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            with patch("swe_forge.publish.docker_builder.time.sleep"):
                with patch(
                    "swe_forge.publish.docker_builder._run_test_in_container",
                    side_effect=mock_run_test_side_effect,
                ):
                    result = await verify_docker_image(
                        "test-image:latest", workspace_with_tests
                    )

        # Should succeed because not all tests passed before patch (at least one failed)
        assert result.success is True
        assert result.before_patch_fail is True
        assert result.after_patch_pass is True

    @pytest.mark.asyncio
    async def test_verify_with_multiple_fail_to_pass_tests(self, workspace_with_tests):
        """Test handling with multiple fail_to_pass tests."""
        workspace_with_tests["tests"]["fail_to_pass"] = [
            "pytest tests/test_a.py -v",
            "pytest tests/test_b.py -v",
        ]

        call_index = {"count": 0}

        def mock_run_test_side_effect(container_name, test_cmd, timeout=120):
            call_index["count"] += 1
            count = call_index["count"]

            # Before patch: both fail
            if count in (1, 2):
                return {
                    "command": test_cmd,
                    "exit_code": 1,
                    "success": False,
                    "output": "FAILED",
                    "error": "",
                }
            # After patch: both pass
            return {
                "command": test_cmd,
                "exit_code": 0,
                "success": True,
                "output": "PASSED",
                "error": "",
            }

        with patch("swe_forge.publish.docker_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            with patch("swe_forge.publish.docker_builder.time.sleep"):
                with patch(
                    "swe_forge.publish.docker_builder._run_test_in_container",
                    side_effect=mock_run_test_side_effect,
                ):
                    result = await verify_docker_image(
                        "test-image:latest", workspace_with_tests
                    )

        assert result.success is True
        assert result.before_patch_fail is True
        assert result.after_patch_pass is True
