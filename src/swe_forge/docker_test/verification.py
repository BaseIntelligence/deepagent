"""Verification logic for before/after test execution.

This module provides the core verification that:
1. Tests FAIL on base commit (proves bug exists)
2. Tests PASS after patch applied (proves fix works)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from swe_forge.docker_test.harness import TestRunResult

if TYPE_CHECKING:
    from swe_forge.docker_test.harness import DockerTestHarness

logger = logging.getLogger(__name__)


@dataclass
class TestFile:
    path: str
    content: str


@dataclass
class VerificationResult:
    """Result of patch verification."""

    passed: bool
    before_passed: bool
    after_passed: bool
    before_output: str
    after_output: str
    error: str | None = None

    def __repr__(self) -> str:
        if self.passed:
            return "VerificationResult(PASSED: tests failed before, passed after)"
        return f"VerificationResult(FAILED: before={self.before_passed}, after={self.after_passed})"


def load_workspace_config(workspace_dir: Path | str) -> dict[str, Any]:
    """Load workspace configuration from workspace.yaml.

    Args:
        workspace_dir: Path to workspace directory or workspace.yaml file

    Returns:
        Workspace configuration dictionary
    """
    import yaml

    workspace_path = Path(workspace_dir)
    if workspace_path.is_file() and workspace_path.name == "workspace.yaml":
        yaml_path = workspace_path
    else:
        yaml_path = workspace_path / "workspace.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"workspace.yaml not found at {yaml_path}")

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_docker_image_for_workspace(workspace_config: dict[str, Any]) -> str | None:
    """Get appropriate Docker image for workspace.

    If docker.prebuilt is True or docker.build is False,
    returns the pre-built image name.

    Args:
        workspace_config: Workspace configuration dictionary

    Returns:
        Docker image name or None if should build from scratch
    """
    docker_config = workspace_config.get("docker", {})

    # Check if pre-built image is available
    if docker_config.get("prebuilt") or docker_config.get("build") is False:
        return docker_config.get("image")

    # Fall back to environment image
    env_config = workspace_config.get("environment", {})
    return env_config.get("image")


async def verify_patch_fixes_issue(
    harness: "DockerTestHarness",
    repo_url: str,
    base_commit: str,
    patch: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str] | None = None,
    install_commands: Sequence[str] | None = None,
    test_files: Sequence[TestFile] | None = None,
    *,
    test_timeout: float = 600.0,
    image: str | None = None,
    python_version: str = "3.11",
    workspace_config: dict[str, Any] | None = None,
) -> VerificationResult:
    """Verify that a patch fixes the issue by running tests before and after.

    The verification logic:
    1. Clone repo at base commit
    2. Write test files (if provided)
    3. Run fail_to_pass tests - they MUST FAIL (proves bug exists)
    4. Apply patch
    5. Run fail_to_pass tests - they MUST PASS (proves fix works)
    6. Run pass_to_pass tests - they MUST PASS (no regression)

    This function supports pre-built Docker images. If workspace_config is
    provided and contains docker.prebuilt=True, the pre-built image is used
    directly without rebuilding, making verification much faster.

    Args:
        harness: DockerTestHarness instance
        repo_url: Repository URL
        base_commit: Base commit SHA
        patch: Unified diff patch to apply
        fail_to_pass: Tests that should fail before, pass after
        pass_to_pass: Tests that should pass both before and after
        install_commands: Commands to install dependencies
        test_files: Test files to write before running tests
        test_timeout: Timeout for test execution
        image: Docker image to use (default: python:${python_version}-slim)
        python_version: Python version for default image
        workspace_config: Optional workspace config for pre-built image detection

    Returns:
        VerificationResult indicating success or failure
    """
    from swe_forge.execution.sandbox import SandboxConfig, DockerSandbox

    # Determine image from workspace config if provided
    if workspace_config and image is None:
        prebuilt_image = get_docker_image_for_workspace(workspace_config)
        if prebuilt_image:
            image = prebuilt_image
            logger.info(f"Using pre-built image: {image}")

    if image is None:
        image = "ubuntu:24.04"
    pass_to_pass = pass_to_pass or []

    docker_client = harness.docker_client
    config = SandboxConfig(
        image=image,
        command_timeout=test_timeout,
        clone_timeout=600.0,
        install_timeout=300.0,
    )

    sandbox = DockerSandbox(docker_client, config)

    try:
        async with sandbox:
            # Check if image is pre-built (contains repo already)
            docker_config = (
                workspace_config.get("docker", {}) if workspace_config else {}
            )
            is_prebuilt = (
                docker_config.get("prebuilt") or docker_config.get("build") is False
            )

            if not is_prebuilt:
                # Standard flow: clone repo and install
                logger.info(f"Cloning {repo_url} at {base_commit[:8]}")
                await sandbox.setup_workspace(repo_url, base_commit)

                # Install Python and git first (ubuntu:24.04 has no Python pre-installed)
                logger.info("Installing Python and git...")
                await sandbox.run_command(
                    "apt-get update && apt-get install -y python3 python3-pip git",
                    timeout=120.0,
                )

                # Always install pytest for running tests
                await sandbox.run_command(
                    "pip3 install pytest --break-system-packages 2>&1", timeout=60.0
                )

                # Install package if it's a Python project with pyproject.toml/setup.py
                result = await sandbox.run_command(
                    "ls pyproject.toml setup.py 2>/dev/null | head -1"
                )
                has_python_project = result.exit_code == 0 and result.stdout.strip()

                if has_python_project:
                    logger.info("Installing Python package...")
                    await sandbox.run_command(
                        "cd /repo && pip3 install -e . --break-system-packages 2>&1 || pip3 install . --break-system-packages 2>&1",
                        timeout=300.0,
                    )

                if install_commands:
                    logger.info("Running install commands")
                    for cmd in install_commands:
                        if cmd.strip() and not cmd.strip().startswith("#"):
                            result = await sandbox.run_command(cmd, timeout=300.0)
                            if result.exit_code != 0:
                                logger.warning(
                                    f"Install command may have failed: {cmd}"
                                )
            else:
                # Pre-built image: repo and deps already installed
                logger.info(f"Using pre-built image {image}, skipping clone/install")

            if test_files:
                logger.info(f"Writing {len(test_files)} test files")
                for tf in test_files:
                    await sandbox.write_file(tf.path, tf.content)
                    logger.debug(f"Wrote test file: {tf.path}")

            before_result = await _run_fail_to_pass_tests(
                sandbox, fail_to_pass, test_timeout
            )
            before_passed = before_result.passed
            before_output = (
                f"STDOUT:\n{before_result.stdout}\n\nSTDERR:\n{before_result.stderr}"
            )

            if before_passed:
                logger.warning(
                    "Tests PASSED before patch - bug may not exist or tests are wrong"
                )
                return VerificationResult(
                    passed=False,
                    before_passed=True,
                    after_passed=False,
                    before_output=before_output,
                    after_output="",
                    error="Tests passed on base commit - expected them to FAIL",
                )

            logger.info("Tests FAILED before patch (good) - applying patch")
            patch_applied = await harness.apply_patch(sandbox, patch)

            if not patch_applied:
                logger.error("Failed to apply patch")
                return VerificationResult(
                    passed=False,
                    before_passed=False,
                    after_passed=False,
                    before_output=before_output,
                    after_output="",
                    error="Failed to apply patch",
                )

            after_result = await _run_fail_to_pass_tests(
                sandbox, fail_to_pass, test_timeout
            )
            after_passed = after_result.passed
            after_output = (
                f"STDOUT:\n{after_result.stdout}\n\nSTDERR:\n{after_result.stderr}"
            )

            if not after_passed:
                logger.error("Tests FAILED after patch - fix does not work")
                return VerificationResult(
                    passed=False,
                    before_passed=False,
                    after_passed=False,
                    before_output=before_output,
                    after_output=after_output,
                    error="Tests failed after patch - expected them to PASS",
                )

            if pass_to_pass:
                logger.info("Running pass_to_pass tests")
                p2p_result = await _run_pass_to_pass_tests(
                    sandbox, pass_to_pass, test_timeout
                )

                if not p2p_result.passed:
                    logger.warning(
                        "pass_to_pass tests FAILED - but continuing since fail_to_pass passed"
                    )

            logger.info("VERIFICATION PASSED")
            return VerificationResult(
                passed=True,
                before_passed=False,
                after_passed=True,
                before_output=before_output,
                after_output=after_output,
            )

    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        return VerificationResult(
            passed=False,
            before_passed=False,
            after_passed=False,
            before_output="",
            after_output="",
            error=str(e),
        )


async def _run_fail_to_pass_tests(
    sandbox,
    tests: Sequence[str],
    timeout: float,
) -> TestRunResult:
    """Run fail_to_pass tests."""
    if not tests:
        return TestRunResult(
            passed=True,
            exit_code=0,
            stdout="No tests provided",
            stderr="",
            command="",
        )

    import time

    test_cmd = " && ".join(tests) if len(tests) > 1 else tests[0]

    start = time.monotonic()
    result = await sandbox.run_command(test_cmd, timeout=timeout)
    duration = time.monotonic() - start

    return TestRunResult(
        passed=result.exit_code == 0,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        command=test_cmd,
        duration_seconds=duration,
    )


async def _run_pass_to_pass_tests(
    sandbox,
    tests: Sequence[str],
    timeout: float,
) -> TestRunResult:
    """Run pass_to_pass tests."""
    return await _run_fail_to_pass_tests(sandbox, tests, timeout)


async def run_before_after_verification(
    harness: "DockerTestHarness",
    repo_url: str,
    base_commit: str,
    patch: str,
    test_commands: Sequence[str],
    install_commands: Sequence[str] | None = None,
    *,
    test_timeout: float = 600.0,
    workspace_config: dict[str, Any] | None = None,
) -> tuple[bool, bool]:
    """Run simple before/after verification.

    Supports pre-built Docker images for faster verification.

    Returns:
        Tuple of (before_failed, after_passed)
    """
    from swe_forge.execution.sandbox import SandboxConfig, DockerSandbox

    # Determine image from workspace config
    image = harness.default_image
    if workspace_config:
        prebuilt_image = get_docker_image_for_workspace(workspace_config)
        if prebuilt_image:
            image = prebuilt_image

    config = SandboxConfig(
        image=image,
        command_timeout=test_timeout,
    )

    sandbox = DockerSandbox(harness.docker_client, config)

    async with sandbox:
        docker_config = workspace_config.get("docker", {}) if workspace_config else {}
        is_prebuilt = (
            docker_config.get("prebuilt") or docker_config.get("build") is False
        )

        if not is_prebuilt:
            await sandbox.setup_workspace(repo_url, base_commit)

            if install_commands:
                for cmd in install_commands:
                    await sandbox.run_command(cmd, timeout=300.0)

        before_result = await sandbox.run_command(
            " && ".join(test_commands),
            timeout=test_timeout,
        )
        before_failed = before_result.exit_code != 0

        await harness.apply_patch(sandbox, patch)

        after_result = await sandbox.run_command(
            " && ".join(test_commands),
            timeout=test_timeout,
        )
        after_passed = after_result.exit_code == 0

        return before_failed, after_passed


async def verify_workspace(
    workspace_dir: Path | str,
    harness: "DockerTestHarness",
    *,
    test_timeout: float = 600.0,
) -> VerificationResult:
    """Verify a workspace using its configuration.

    This function loads workspace.yaml and runs verification
    using pre-built images if available.

    Args:
        workspace_dir: Path to workspace directory
        harness: DockerTestHarness instance
        test_timeout: Timeout for test execution

    Returns:
        VerificationResult
    """
    workspace_config = load_workspace_config(workspace_dir)
    workspace_path = (
        Path(workspace_dir)
        if Path(workspace_dir).is_dir()
        else Path(workspace_dir).parent
    )

    # Get repo info
    repo_info = workspace_config.get("repo", {})
    repo_url = repo_info.get("url", "")
    base_commit = repo_info.get("base_commit", "")

    # Get tests
    tests_config = workspace_config.get("tests", {})
    fail_to_pass = tests_config.get("fail_to_pass", [])
    pass_to_pass = tests_config.get("pass_to_pass", [])

    # Get install commands
    install_config = workspace_config.get("install", {})
    install_commands = install_config.get("commands", [])

    # Load patch
    patch_path = workspace_path / "patch.diff"
    patch = ""
    if patch_path.exists():
        patch = patch_path.read_text()

    if not patch:
        return VerificationResult(
            passed=False,
            before_passed=False,
            after_passed=False,
            before_output="",
            after_output="",
            error="No patch.diff found in workspace",
        )

    return await verify_patch_fixes_issue(
        harness,
        repo_url=repo_url,
        base_commit=base_commit,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        install_commands=install_commands,
        test_timeout=test_timeout,
        workspace_config=workspace_config,
    )
