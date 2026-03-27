"""Dataset validator - tests datasets in FRESH Docker containers.

CRITICAL: Tests run in isolated, fresh containers to ensure
the dataset will work in production.

NO HARDCODING: All commands come from the agentic configuration.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any

from swe_forge.swe.agentic_config import detect_repository_config
from swe_forge.swe.models import SweTask

if TYPE_CHECKING:
    from swe_forge.llm.client import LLMClient

logger = getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a dataset task."""

    task_id: str
    passed: bool
    phase: str  # "setup", "install", "test_base", "test_patch"
    error_message: str | None = None
    install_commands_worked: list[str] = field(default_factory=list)
    install_commands_failed: list[str] = field(default_factory=list)
    test_results: dict[str, bool] = field(default_factory=dict)
    container_logs: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "phase": self.phase,
            "error_message": self.error_message,
            "install_commands_worked": self.install_commands_worked,
            "install_commands_failed": self.install_commands_failed,
            "test_results": self.test_results,
        }


@dataclass
class DatasetValidationReport:
    """Full validation report for a dataset."""

    total_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    validation_results: list[ValidationResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.passed_tasks / self.total_tasks

    def to_json(self) -> str:
        return json.dumps(
            {
                "total_tasks": self.total_tasks,
                "passed_tasks": self.passed_tasks,
                "failed_tasks": self.failed_tasks,
                "pass_rate": self.pass_rate,
                "results": [r.to_dict() for r in self.validation_results],
            },
            indent=2,
        )


async def validate_task_in_fresh_container(
    task: SweTask,
    llm_client: "LLMClient",
    docker_client: Any,
    timeout: int = 600,
) -> ValidationResult:
    """Validate a single task in a FRESH Docker container.

    This ensures the task dataset will work in production:
    1. Clone repo at base commit in fresh container
    2. Run agentic config detection for install commands
    3. Run fail_to_pass tests - must FAIL on base
    4. Apply patch
    5. Run fail_to_pass tests - must PASS after patch
    6. Run pass_to_pass tests - must PASS both times

    NO HARDCODING: All commands come from LLM agent detection.
    """
    from swe_forge.execution.sandbox import DockerSandbox

    result = ValidationResult(task_id=task.id, passed=False, phase="setup")

    # Get repository info
    owner, repo = task.repo.split("/")
    repo_url = f"https://github.com/{task.repo}.git"

    # Create fresh sandbox - NO HARDCODED IMAGE
    # Use the task's detected config or let agent detect
    if task.install_config:
        docker_image = task.install_config.get("docker_image", "python:latest")
    else:
        docker_image = "ubuntu:24.04"  # Default, will be detected

    sandbox = DockerSandbox(
        docker_client=docker_client,
        image=docker_image,
        workdir="/workspace",
    )

    try:
        async with sandbox:
            # PHASE 1: Setup - Clone and detect config
            result.phase = "setup"

            # Clone repository
            clone_result = await sandbox.run_command(
                f"git clone {repo_url} /workspace/repo", timeout=120
            )
            if clone_result.exit_code != 0:
                result.error_message = f"Clone failed: {clone_result.stderr}"
                return result

            # Checkout base commit
            checkout_result = await sandbox.run_command(
                f"cd /workspace/repo && git checkout {task.base_commit}", timeout=60
            )
            if checkout_result.exit_code != 0:
                result.error_message = f"Checkout failed: {checkout_result.stderr}"
                return result

            # PHASE 2: Agentic config detection - NO HARDCODING
            result.phase = "install"

            # If task has install_config, use it. Otherwise detect.
            if task.install_config and task.install_config.get("commands"):
                install_commands = task.install_config["commands"]
            else:
                # AGENTIC DETECTION - LLM figures it out
                logger.info(f"Detecting configuration for {task.repo} via agent...")
                config = await detect_repository_config(
                    llm_client=llm_client,
                    sandbox=sandbox,
                    repo_url=repo_url,
                    commit_sha=task.base_commit,
                    max_turns=30,
                )

                if not config.is_valid():
                    result.error_message = (
                        f"Config detection failed: {config.validation_errors}"
                    )
                    return result

                install_commands = config.install_commands
                task.test_command = config.test_command

            # Run install commands - track which work
            for cmd in install_commands:
                install_result = await sandbox.run_command(
                    f"cd /workspace/repo && {cmd}", timeout=300
                )
                if install_result.exit_code == 0:
                    result.install_commands_worked.append(cmd)
                else:
                    result.install_commands_failed.append(cmd)
                    logger.warning(
                        f"Install command failed: {cmd} -> {install_result.stderr[:200]}"
                    )

            # PHASE 3: Test on base commit (fail_to_pass MUST FAIL)
            result.phase = "test_base"

            for test_cmd in task.fail_to_pass:
                test_result = await sandbox.run_command(
                    f"cd /workspace/repo && {test_cmd}", timeout=timeout
                )

                # On base commit, fail_to_pass tests SHOULD FAIL
                base_passed = test_result.exit_code == 0
                result.test_results[f"base:{test_cmd}"] = base_passed

                if base_passed:
                    # This is an ERROR - test should fail on base
                    result.error_message = (
                        f"FAIL_TO_PASS test '{test_cmd}' PASSED on base commit - "
                        f"this means it's not a valid fail_to_pass test!"
                    )
                    return result

            # PHASE 4: Apply patch
            result.phase = "patch"

            # Write patch to file
            patch_result = await sandbox.run_command(
                f"cd /workspace/repo && git apply --allow-empty <<< '{task.patch}'",
                timeout=60,
            )
            if patch_result.exit_code != 0:
                # Try with 3-way merge
                patch_result = await sandbox.run_command(
                    f"cd /workspace/repo && git apply --3way --allow-empty <<< '{task.patch}'",
                    timeout=60,
                )
                if patch_result.exit_code != 0:
                    result.error_message = f"Patch apply failed: {patch_result.stderr}"
                    return result

            # PHASE 5: Test after patch (fail_to_pass MUST PASS)
            result.phase = "test_patch"

            all_tests_passed = True

            for test_cmd in task.fail_to_pass:
                test_result = await sandbox.run_command(
                    f"cd /workspace/repo && {test_cmd}", timeout=timeout
                )

                # After patch, fail_to_pass tests SHOULD PASS
                passed = test_result.exit_code == 0
                result.test_results[f"patch:{test_cmd}"] = passed

                if not passed:
                    all_tests_passed = False

            # PHASE 6: pass_to_pass tests
            for test_cmd in task.pass_to_pass:
                test_result = await sandbox.run_command(
                    f"cd /workspace/repo && {test_cmd}", timeout=timeout
                )

                passed = test_result.exit_code == 0
                result.test_results[f"passtopass:{test_cmd}"] = passed

                if not passed:
                    all_tests_passed = False

            # Final result
            result.passed = all_tests_passed
            result.phase = "complete"

            if all_tests_passed:
                logger.info(f"✅ Task {task.id} validated successfully!")
            else:
                logger.warning(f"⚠️ Task {task.id} failed validation")

    except Exception as e:
        result.error_message = f"Exception: {str(e)}"
        logger.error(f"Validation exception for {task.id}: {e}")

    return result


async def validate_dataset(
    tasks: list[SweTask],
    llm_client: "LLMClient",
    docker_client: Any,
    max_concurrent: int = 4,
    output_path: str | None = None,
) -> DatasetValidationReport:
    """Validate an entire dataset in FRESH containers.

    Args:
        tasks: List of tasks to validate
        llm_client: LLM client for agentic detection
        docker_client: Docker client
        max_concurrent: Maximum concurrent validations
        output_path: Optional path to save validation report

    Returns:
        DatasetValidationReport with results
    """
    report = DatasetValidationReport(total_tasks=len(tasks))

    # Run validations with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)

    async def validate_with_semaphore(task: SweTask) -> ValidationResult:
        async with semaphore:
            return await validate_task_in_fresh_container(
                task=task,
                llm_client=llm_client,
                docker_client=docker_client,
            )

    # Run all validations
    results = await asyncio.gather(
        *[validate_with_semaphore(task) for task in tasks],
        return_exceptions=True,
    )

    for r in results:
        if isinstance(r, Exception):
            result = ValidationResult(
                task_id="unknown",
                passed=False,
                phase="error",
                error_message=str(r),
            )
        else:
            result = r

        report.validation_results.append(result)
        if result.passed:
            report.passed_tasks += 1
        else:
            report.failed_tasks += 1

    # Save report if path provided
    if output_path:
        Path(output_path).write_text(report.to_json())
        logger.info(f"Validation report saved to {output_path}")

    return report


def print_validation_report(report: DatasetValidationReport) -> None:
    """Print validation report summary."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"Total tasks: {report.total_tasks}")
    print(f"Passed: {report.passed_tasks}")
    print(f"Failed: {report.failed_tasks}")
    print(f"Pass rate: {report.pass_rate:.1%}")
    print("=" * 70)

    if report.failed_tasks > 0:
        print("\nFailed tasks:")
        for result in report.validation_results:
            if not result.passed:
                print(f"  - {result.task_id}: {result.phase} - {result.error_message}")

    print()
