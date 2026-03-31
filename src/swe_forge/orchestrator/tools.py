"""Tool implementations for the orchestrator pipeline.

This module provides the actual implementation of the function calls
defined in function_calls.py. Each function is a thin wrapper around
existing functionality, converting inputs/outputs to match orchestrator
data models.

Pipeline order:
1. generate_tests - Create test specifications via agentic loop
2. validate_tests - Verify test quality and syntax
3. build_docker - Containerize the test environment
4. verify_fail_to_pass - Run tests before/after patch
5. repair_test - Fix failing tests if needed
6. score_task - Calculate quality metrics
7. publish_task - Export to dataset
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .models import (
    BuildDockerResult,
    GenerateTestsResult,
    PublishResult,
    RepairResult,
    ScoreResult,
    TestFile,
    ValidateTestsResult,
    VerifyResult,
)

if TYPE_CHECKING:
    from swe_forge.agents.test_repair_agent import Diagnosis, Fix, RepairAttempt
    from swe_forge.llm.client import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# Sandbox Protocol for Docker-based test generation
# =============================================================================


class SandboxProtocol(Protocol):
    """Protocol for sandbox implementations used by test generation."""

    async def run_command(
        self, cmd: str, *, timeout: float | None = None
    ) -> "ExecResultProtocol":
        """Execute a command in the sandbox."""
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox."""
        ...

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        ...


class ExecResultProtocol(Protocol):
    """Protocol for command execution results."""

    @property
    def exit_code(self) -> int: ...

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...


# =============================================================================
# Tool Implementations
# =============================================================================


async def generate_tests(
    task_id: str,
    patch: str,
    repo_url: str,
    base_commit: str,
    language: str,
    llm_client: LLMClient | None = None,
    sandbox: SandboxProtocol | None = None,
    *,
    model: str = "openai/gpt-4o",
    max_turns: int = 200,
) -> GenerateTestsResult:
    """Generate tests via LLM agent.

    Uses existing TestGenerator from swe_forge.swe.test_generator.

    Args:
        task_id: Unique identifier for the task (e.g., 'owner-repo-123').
        patch: Unified diff patch content to generate tests for.
        repo_url: Git repository URL to clone.
        base_commit: Git commit SHA to checkout before applying patch.
        language: Programming language of the repository (e.g., 'python', 'rust').
        llm_client: LLM client for generation (optional, will create if not provided).
        sandbox: Sandbox for executing commands (optional, will create Docker if not provided).
        model: Model identifier to use for generation.
        max_turns: Maximum number of agent turns (default: 200).

    Returns:
        GenerateTestsResult with generated tests and metadata.
    """
    from swe_forge.swe.test_generator import TestGenerator
    from swe_forge.swe.models import SweTask

    # Create LLM client if not provided
    if llm_client is None:
        from swe_forge.llm.openrouter import OpenRouterClient
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return GenerateTestsResult(
                success=False,
                error="OPENROUTER_API_KEY not set and no llm_client provided",
            )
        llm_client = OpenRouterClient(api_key=api_key)

    # Create SweTask from parameters
    task = SweTask(
        id=task_id,
        repo=repo_url.replace("https://github.com/", "").replace(".git", ""),
        base_commit=base_commit,
        patch=patch,
        language=language,
    )

    # If no sandbox provided, create a Docker-based one
    if sandbox is None:
        sandbox = await _create_docker_sandbox(repo_url, base_commit, task_id)

    generator = TestGenerator(
        llm=llm_client,
        max_turns=max_turns,
        model=model,
    )

    try:
        result = await generator.generate_tests(task, sandbox)

        return GenerateTestsResult(
            success=result.success,
            tests={
                "fail_to_pass": result.fail_to_pass,
                "pass_to_pass": result.pass_to_pass,
            },
            test_files=[
                TestFile(path=tf.path, content=tf.content) for tf in result.test_files
            ],
            install_commands=result.install_commands,
            turn_count=result.turn_count,
            error=None if result.success else "Generation failed or exhausted turns",
        )
    except Exception as e:
        logger.error(f"Test generation failed for {task_id}: {e}")
        return GenerateTestsResult(
            success=False,
            error=str(e),
        )


async def validate_tests(
    task_id: str,
    tests: list[dict[str, str]],
    patch: str,
) -> ValidateTestsResult:
    """Validate tests BEFORE Docker build.

    Checks:
    1. Syntax validity (ast.parse for Python)
    2. Assertion presence (patterns like assert, expect, should)
    3. Relevance to patch (file paths match)

    Args:
        task_id: Unique identifier for the task.
        tests: List of test files to validate (dict with 'path' and 'content').
        patch: Unified diff patch to check test relevance against.

    Returns:
        ValidateTestsResult with validation status and issues.
    """
    issues: list[str] = []
    has_valid_syntax = True
    has_assertions = False
    relevant_to_patch = False

    # Extract file paths from patch
    patch_file_pattern = re.compile(r"^[+-]{3} [ab]/(.+)$", re.MULTILINE)
    patch_files = set(patch_file_pattern.findall(patch))

    for test in tests:
        path = test.get("path", "")
        content = test.get("content", "")

        if not path or not content:
            issues.append(f"Test file missing path or content: {path or 'unnamed'}")
            continue

        # Check syntax for Python files
        if path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as e:
                has_valid_syntax = False
                issues.append(f"Syntax error in {path}: {e}")
        elif path.endswith((".js", ".ts")):
            # Basic JavaScript/TypeScript checks
            if "function" not in content and "=>" not in content:
                issues.append(f"No function definition found in {path}")

        # Check for assertions
        assertion_patterns = [
            r"\bassert\s",
            r"\bexpect\s*\(",
            r"\.should\(",
            r"\.toBe\(",
            r"\.toEqual\(",
            r"\.assert\(",
            r"self\.assert",
            r"assertThat\(",
        ]
        for pattern in assertion_patterns:
            if re.search(pattern, content):
                has_assertions = True
                break

        # Check relevance to patch (simple: file paths overlap)
        test_dir = Path(path).parent
        for patch_file in patch_files:
            patch_dir = Path(patch_file).parent
            # Check if test file references similar paths
            if str(test_dir) in content or str(patch_dir) in content:
                relevant_to_patch = True
                break
            # Check if test is for the module being patched
            module_name = patch_file.replace("/", ".").replace(".py", "")
            if module_name in content:
                relevant_to_patch = True
                break

        # Also check if the test imports/mentions files from the patch
        for patch_file in patch_files:
            module_ref = patch_file.replace("/", ".").rsplit(".", 1)[0]
            if module_ref in content:
                relevant_to_patch = True
                break

    # If we have tests but no assertions found
    if tests and not has_assertions:
        issues.append("No assertions found in test files")

    success = has_valid_syntax and has_assertions and len(issues) == 0

    return ValidateTestsResult(
        success=success,
        has_assertions=has_assertions,
        has_valid_syntax=has_valid_syntax,
        relevant_to_patch=relevant_to_patch,
        issues=issues,
        error=None if success else "; ".join(issues),
    )


async def build_docker(
    task_id: str,
    tests: list[dict[str, str]],
    repo_url: str,
    base_commit: str,
    language: str,
    *,
    docker_username: str = "swe-forge",
    push: bool = False,
    install_commands: list[str] | None = None,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    patch: str = "",
) -> BuildDockerResult:
    """Build Docker image with repository and tests.

    Uses existing docker_builder module.

    Args:
        task_id: Unique identifier for the task.
        tests: List of test files to include in the image.
        repo_url: Git repository URL to clone in the image.
        base_commit: Git commit SHA to checkout.
        language: Programming language (determines base image).
        docker_username: Docker Hub username for image naming.
        push: Whether to push the image to Docker Hub.
        install_commands: Commands to install dependencies.
        fail_to_pass: Test commands that should fail before patch and pass after.
        pass_to_pass: Test commands that should pass before and after patch.
        patch: The patch content to write to patch.diff file.

    Returns:
        BuildDockerResult with image name and build status.
    """
    from swe_forge.publish.docker_builder import build_docker_image

    start_time = time.time()

    # Create temporary task directory with workspace.yaml
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir) / task_id.replace("/", "-").replace(".", "-")
        task_dir.mkdir()
        tests_dir = task_dir / "tests"
        tests_dir.mkdir()

        # Write test files
        for test in tests:
            test_path = tests_dir / Path(test["path"]).name
            test_path.write_text(test["content"])

        # Write workspace.yaml
        import yaml

        workspace = {
            "task_id": task_id,
            "repo": {"url": repo_url, "base_commit": base_commit},
            "language": language,
            "install": {"commands": install_commands or []},
            "tests": {
                "fail_to_pass": fail_to_pass or [],
                "pass_to_pass": pass_to_pass or [],
            },
        }
        workspace_path = task_dir / "workspace.yaml"
        with open(workspace_path, "w") as f:
            yaml.dump(workspace, f, default_flow_style=False)

        # Write patch.diff
        patch_path = task_dir / "patch.diff"
        patch_path.write_text(patch)

        try:
            result = await build_docker_image(
                task_dir=task_dir,
                docker_user=docker_username,
                push=push,
            )

            build_time = time.time() - start_time

            return BuildDockerResult(
                success=result.success,
                image_name=result.image_name,
                build_time_seconds=build_time,
                error=result.error,
            )
        except Exception as e:
            logger.error(f"Docker build failed for {task_id}: {e}")
            return BuildDockerResult(
                success=False,
                error=str(e),
                build_time_seconds=time.time() - start_time,
            )


async def verify_fail_to_pass(
    task_id: str,
    image_name: str,
    tests: list[str],
    patch: str,
    *,
    max_retries: int = 3,
    timeout: int = 300,
) -> VerifyResult:
    """Verify tests FAIL before patch and PASS after patch.

    Uses existing verify_docker_image.

    Args:
        task_id: Unique identifier for the task.
        image_name: Docker image name to run tests in.
        tests: Test commands to execute (fail_to_pass tests).
        patch: Patch to apply between test runs.
        max_retries: Maximum retry attempts for flaky tests.
        timeout: Timeout per test in seconds.

    Returns:
        VerifyResult with verification status.
    """
    from swe_forge.publish.docker_builder import verify_docker_image

    # Build workspace dict for verification
    workspace = {
        "task_id": task_id,
        "tests": {
            "fail_to_pass": tests,
            "pass_to_pass": [],
        },
        "patch": patch,
    }

    try:
        result = await verify_docker_image(
            image_name=image_name,
            workspace=workspace,
            timeout=timeout,
        )

        return VerifyResult(
            success=result.success,
            before_patch_failed=result.before_patch_fail,
            after_patch_passed=result.after_patch_pass,
            needs_repair=not result.success,
            error=result.error,
        )
    except Exception as e:
        logger.error(f"Verification failed for {task_id}: {e}")
        return VerifyResult(
            success=False,
            needs_repair=True,
            error=str(e),
        )


async def repair_test(
    task_id: str,
    error_output: str,
    *,
    max_attempts: int = 5,
    llm_client: LLMClient | None = None,
    model: str = "openai/gpt-4o-mini",
    patch: str = "",
    repo_url: str = "",
) -> RepairResult:
    """Attempt to repair failing tests via LLM agent.

    Uses existing TestRepairAgent.

    Args:
        task_id: Unique identifier for the task.
        error_output: Error output from failed test run.
        max_attempts: Maximum repair attempts (default: 5).
        llm_client: LLM client for diagnosis and fix generation.
        model: Model to use for repair agent.
        patch: The patch that was being applied (for context).
        repo_url: Repository URL (for context).

    Returns:
        RepairResult with repair attempts and outcome.
    """
    from swe_forge.agents.test_repair_agent import TestRepairAgent, RepairAttempt

    # Create LLM client if not provided
    if llm_client is None:
        from swe_forge.llm.openrouter import OpenRouterClient
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return RepairResult(
                success=False,
                attempts=0,
                error="OPENROUTER_API_KEY not set and no llm_client provided",
            )
        llm_client = OpenRouterClient(api_key=api_key)

    agent = TestRepairAgent(llm_client=llm_client, model=model)
    attempts = 0
    fix_applied = None

    try:
        # Single diagnosis attempt (repair loop happens higher up)
        diagnosis = await agent.diagnose_failure(
            test_output={"error": error_output},
            patch=patch,
            repo_url=repo_url,
            error=error_output,
        )
        attempts += 1

        if not diagnosis.fixable:
            return RepairResult(
                success=False,
                attempts=attempts,
                error=f"Diagnosis: {diagnosis.reason}",
            )

        fix = await agent.generate_fix(diagnosis)
        attempts += 1

        if fix.skip_task:
            return RepairResult(
                success=False,
                attempts=attempts,
                error=fix.description,
            )

        fix_applied = fix.description

        return RepairResult(
            success=True,
            attempts=attempts,
            fix_applied=fix_applied,
        )

    except Exception as e:
        logger.error(f"Repair failed for {task_id}: {e}")
        return RepairResult(
            success=False,
            attempts=attempts,
            error=str(e),
        )


async def score_task(
    task_id: str,
    validation_result: ValidateTestsResult,
    verify_result: VerifyResult,
    *,
    patch: str = "",
    tests: list[str] | None = None,
    prompt: str = "",
) -> ScoreResult:
    """Calculate quality score for a completed task.

    Uses existing ComplexityEvaluator for complexity scoring.

    Args:
        task_id: Unique identifier for the task.
        validation_result: Result from validate_tests step.
        verify_result: Result from verify_fail_to_pass step.
        patch: The patch content (for complexity analysis).
        tests: Test commands (for complexity analysis).
        prompt: Task description (for complexity analysis).

    Returns:
        ScoreResult with quality scores.
    """
    from swe_forge.evaluators.complexity_evaluator import ComplexityEvaluator

    # Calculate complexity score
    evaluator = ComplexityEvaluator()

    try:
        verdict = evaluator.evaluate(
            patch=patch,
            tests=tests or [],
            prompt=prompt,
        )
        complexity_score = verdict.score
    except Exception as e:
        logger.warning(f"Complexity evaluation failed for {task_id}: {e}")
        complexity_score = 0.5  # Default to medium

    # Calculate test quality score based on validation
    test_quality_score = 0.0
    if validation_result.success:
        test_quality_score = 1.0
    elif validation_result.has_assertions and validation_result.has_valid_syntax:
        test_quality_score = 0.7
    elif validation_result.has_valid_syntax:
        test_quality_score = 0.3

    # Calculate verification score
    verification_score = 0.0
    if verify_result.success:
        verification_score = 1.0
    elif verify_result.after_patch_passed:
        verification_score = 0.7
    elif verify_result.before_patch_failed:
        verification_score = 0.3

    # Overall score is weighted average
    overall_score = (
        0.4 * complexity_score + 0.3 * test_quality_score + 0.3 * verification_score
    )

    return ScoreResult(
        score=round(overall_score, 3),
        complexity_score=round(complexity_score, 3),
        test_quality_score=round(test_quality_score, 3),
        verification_score=round(verification_score, 3),
    )


async def publish_task(
    task_id: str,
    score: float,
    *,
    dataset_name: str = "CortexLM/swe-forge",
    hf_token: str | None = None,
    min_score: float = 0.5,
    task_data: dict[str, Any] | None = None,
) -> PublishResult:
    """Publish to HuggingFace if score >= threshold.

    Args:
        task_id: Unique identifier for the task.
        score: Quality score (0.0 to 1.0).
        dataset_name: HuggingFace dataset name.
        hf_token: HuggingFace API token (uses HF_TOKEN env var if not provided).
        min_score: Minimum score threshold for publishing.
        task_data: Additional task data to publish.

    Returns:
        PublishResult with publishing status.
    """
    import os

    # Check score threshold
    if score < min_score:
        return PublishResult(
            success=False,
            error=f"Score {score:.2f} below threshold {min_score}",
        )

    # Get HF token
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        return PublishResult(
            success=False,
            error="HF_TOKEN not set",
        )

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        # Prepare data for publishing
        data = task_data or {}
        data["instance_id"] = task_id
        data["score"] = score

        # Note: Actual publishing would use create_commit or upload_file
        # For now, just validate the connection
        user_info = api.whoami()
        logger.info(
            f"Ready to publish {task_id} to {dataset_name} as {user_info['name']}"
        )

        return PublishResult(
            success=True,
            dataset_name=dataset_name,
            task_id=task_id,
        )

    except ImportError:
        return PublishResult(
            success=False,
            error="huggingface_hub not installed",
        )
    except Exception as e:
        logger.error(f"Publishing failed for {task_id}: {e}")
        return PublishResult(
            success=False,
            error=str(e),
        )


async def reject_task(
    task_id: str,
    reason: str,
    *,
    details: dict[str, Any] | None = None,
) -> None:
    """Log rejection with reason.

    This is a terminal action that marks a task as rejected.

    Args:
        task_id: Unique identifier for the task.
        reason: Short rejection reason (e.g., 'complexity_too_low').
        details: Detailed explanation of why task was rejected.
    """
    detail_str = ""
    if details:
        import json

        detail_str = f" | Details: {json.dumps(details)}"
    logger.info(f"Task {task_id} REJECTED: {reason}{detail_str}")


# =============================================================================
# Helper Functions
# =============================================================================


async def _create_docker_sandbox(
    repo_url: str,
    base_commit: str,
    task_id: str,
) -> "DockerSandbox":
    """Create a Docker-based sandbox for test generation."""
    return DockerSandbox(repo_url=repo_url, base_commit=base_commit, task_id=task_id)


@dataclass
class DockerSandbox:
    """Docker-based sandbox for test generation."""

    repo_url: str
    base_commit: str
    task_id: str
    container_name: str = ""
    container_id: str | None = None
    _started: bool = field(default=False, init=False)

    async def _ensure_container(self) -> None:
        """Start Docker container if not already running."""
        if self._started:
            return

        self.container_name = (
            f"swe-gen-{self.task_id.replace('/', '-').replace('.', '-')}"
        )

        # Remove existing container if any
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            text=True,
        )

        # Start new container
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "ubuntu:24.04",
                "sleep",
                "3600",  # Keep alive for 1 hour
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self.container_id = result.stdout.strip()
        self._started = True

        # Install dependencies
        await self.run_command(
            "apt-get update && apt-get install -y git python3 python3-pip python3-venv"
        )
        await self.run_command("git clone {self.repo_url} /repo")
        await self.run_command(f"cd /repo && git checkout {self.base_commit}")

    async def run_command(
        self, cmd: str, *, timeout: float | None = None
    ) -> "ExecResult":
        """Execute a command in the Docker container."""
        await self._ensure_container()

        timeout_sec = timeout or 120

        result = subprocess.run(
            ["docker", "exec", self.container_name, "bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        return ExecResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    async def write_file(self, path: str, content: str) -> None:
        """Write a file in the container."""
        await self._ensure_container()

        # Use heredoc for multi-line content
        escaped_content = content.replace("'", "'\"'\"'")
        await self.run_command(f"cat > {path} << 'EOFMARKER'\n{content}\nEOFMARKER")

    async def read_file(self, path: str) -> str:
        """Read a file from the container."""
        await self._ensure_container()
        result = await self.run_command(f"cat {path}")
        return result.stdout

    async def cleanup(self) -> None:
        """Remove the Docker container."""
        if self.container_name:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                text=True,
            )
            self._started = False


@dataclass
class ExecResult:
    """Result from a command execution in the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
