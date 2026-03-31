"""DatasetOrchestrator - Orchestrates processing of a single task through the pipeline.

This module provides the per-task orchestrator that runs the sequential pipeline:
Generate -> Validate -> Build -> Verify -> Repair -> Score -> Publish

The MasterOrchestrator creates multiple DatasetOrchestrator instances
for parallel processing of multiple tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .models import (
    BuildDockerResult,
    GenerateTestsResult,
    OrchestratorTask,
    TaskState,
    ValidateTestsResult,
)
from .tools import (
    build_docker,
    generate_tests,
    publish_task,
    reject_task,
    repair_test,
    score_task,
    validate_tests,
    verify_fail_to_pass,
)

if TYPE_CHECKING:
    from swe_forge.llm.client import LLMClient

logger = logging.getLogger(__name__)


class DatasetOrchestrator:
    """Orchestrates processing of a single dataset task through the pipeline.

    Pipeline order (CRITICAL):
    1. Generate Tests (LLM agent)
    2. Validate Tests (syntax, assertions, relevance)
    3. Build Docker (if validation passes)
    4. Verify FAIL→PASS
    5. Repair (if verification fails, up to max_repair_attempts)
    6. Score (complexity + quality)
    7. Publish (if score >= threshold)

    Attributes:
        orchestrator_id: Unique identifier for this orchestrator instance.
        llm_client: LLM client for test generation and repair.
        hf_token: HuggingFace API token for publishing.
        min_score_threshold: Minimum score required to publish.
        max_repair_attempts: Maximum repair attempts before rejecting.
        model: Model to use for LLM operations.
    """

    def __init__(
        self,
        orchestrator_id: int = 0,
        llm_client: LLMClient | None = None,
        hf_token: str | None = None,
        min_score_threshold: float = 0.5,
        max_repair_attempts: int = 5,
        model: str = "openai/gpt-4o",
        docker_username: str = "swe-forge",
        push_images: bool = False,
        skip_generation: bool = False,
        use_existing_image: bool = False,
    ) -> None:
        """Initialize the DatasetOrchestrator.

        Args:
            orchestrator_id: Unique identifier for this orchestrator.
            llm_client: LLM client for test generation and repair.
            hf_token: HuggingFace API token for publishing.
            min_score_threshold: Minimum score required to publish (default: 0.5).
            max_repair_attempts: Maximum repair attempts before rejecting (default: 5).
            model: Model to use for LLM operations (default: gpt-4o).
            docker_username: Docker Hub username for image naming.
            push_images: Whether to push Docker images to registry.
            skip_generation: Whether to skip test generation if tests already exist.
        """
        self.orchestrator_id = orchestrator_id
        self.llm_client = llm_client
        self.hf_token = hf_token
        self.min_score_threshold = min_score_threshold
        self.max_repair_attempts = max_repair_attempts
        self.model = model
        self.docker_username = docker_username
        self.push_images = push_images
        self.skip_generation = skip_generation
        self.use_existing_image = use_existing_image

    async def run_pipeline(self, task: OrchestratorTask) -> OrchestratorTask:
        """Run the full pipeline for a single task.

        This method executes the sequential pipeline:
        Generate -> Validate -> Build -> Verify -> Repair -> Score -> Publish

        Args:
            task: The task to process. Will be mutated with state transitions
                  and results attached.

        Returns:
            The task with updated state and results. Check task.state for
            final status (COMPLETED, REJECTED, or FAILED).
        """
        logger.info(
            f"[{self.orchestrator_id}] Starting pipeline for task {task.task_id}"
        )

        try:
            return await self._run_pipeline_inner(task)
        except Exception as e:
            logger.exception(
                f"[{self.orchestrator_id}] Pipeline failed for {task.task_id}: {e}"
            )
            task.transition_to(TaskState.FAILED)
            task.metadata["failure_reason"] = str(e)
            return task

    async def _run_pipeline_inner(self, task: OrchestratorTask) -> OrchestratorTask:
        """Inner pipeline implementation with exception handling at top level."""
        # 1. GENERATE_TESTS (skip if tests already exist)
        if self.skip_generation and task.tests.get("fail_to_pass"):
            logger.info(
                f"[{self.orchestrator_id}] Skipping test generation, using existing tests"
            )
            task.transition_to(TaskState.GENERATING_TESTS)
            generate_result = GenerateTestsResult(
                success=True,
                tests=task.tests,
                test_files=task.test_files,
                install_commands=task.install_commands,
            )
        else:
            task.transition_to(TaskState.GENERATING_TESTS)
            generate_result = await generate_tests(
                task_id=task.task_id,
                patch=task.patch,
                repo_url=task.repo_url,
                base_commit=task.base_commit,
                language=task.language,
                llm_client=self.llm_client,
                model=self.model,
            )
        task.generate_result = generate_result
        task.tests = generate_result.tests

        if not generate_result.success:
            return self._reject(task, "Test generation failed", generate_result.error)

        # 2. VALIDATING_TESTS (skip if no test files to validate)
        if generate_result.test_files:
            task.transition_to(TaskState.VALIDATING_TESTS)
            test_files_for_validation = [
                {"path": tf.path, "content": tf.content}
                for tf in generate_result.test_files
            ]
            validate_result = await validate_tests(
                task_id=task.task_id,
                tests=test_files_for_validation,
                patch=task.patch,
            )
            task.validate_result = validate_result

            if not validate_result.success:
                return self._reject(
                    task,
                    "Test validation failed",
                    "; ".join(validate_result.issues)
                    if validate_result.issues
                    else validate_result.error,
                )
        else:
            # No test files to validate, skip validation
            logger.info(
                f"[{self.orchestrator_id}] Skipping test validation (no test files)"
            )
            task.transition_to(TaskState.VALIDATING_TESTS)
            task.validate_result = ValidateTestsResult(
                success=True,
                has_valid_syntax=True,
                has_assertions=True,
                relevant_to_patch=True,
            )

        # 3. BUILDING_DOCKER (skip if using existing image)
        if self.use_existing_image and task.docker_image:
            logger.info(
                f"[{self.orchestrator_id}] Using existing Docker image: {task.docker_image}"
            )
            task.transition_to(TaskState.BUILDING_DOCKER)
            build_result = BuildDockerResult(
                success=True,
                image_name=task.docker_image,
                build_time_seconds=0.0,
            )
        else:
            task.transition_to(TaskState.BUILDING_DOCKER)
            test_files_for_build = [
                {"path": tf.path, "content": tf.content}
                for tf in generate_result.test_files
            ]
            build_result = await build_docker(
                task_id=task.task_id,
                tests=test_files_for_build,
                repo_url=task.repo_url,
                base_commit=task.base_commit,
                language=task.language,
                docker_username=self.docker_username,
                push=self.push_images,
                install_commands=generate_result.install_commands,
                fail_to_pass=generate_result.tests.get("fail_to_pass", []),
                pass_to_pass=generate_result.tests.get("pass_to_pass", []),
                patch=task.patch,
            )
        task.build_result = build_result

        if not build_result.success:
            return self._reject(task, "Docker build failed", build_result.error)

        task.metadata["docker_image"] = build_result.image_name

        # 4. VERIFYING
        task.transition_to(TaskState.VERIFYING)
        fail_to_pass_tests = task.tests.get("fail_to_pass", [])
        verify_result = await verify_fail_to_pass(
            task_id=task.task_id,
            image_name=build_result.image_name or "",
            tests=fail_to_pass_tests,
            patch=task.patch,
        )
        task.verify_result = verify_result

        if not verify_result.success:
            # 5. REPAIRING (if needed)
            if verify_result.needs_repair:
                task = await self._attempt_repair(task, build_result.image_name or "")
                if task.state == TaskState.REJECTED:
                    return task
            else:
                return self._reject(task, "Verification failed", verify_result.error)

        # 6. SCORING
        task.transition_to(TaskState.SCORING)
        score_result = await score_task(
            task_id=task.task_id,
            validation_result=task.validate_result,
            verify_result=task.verify_result,
            patch=task.patch,
            tests=fail_to_pass_tests,
            prompt=task.metadata.get("prompt", ""),
        )
        task.score_result = score_result

        # 7. PUBLISHING or COMPLETED
        if score_result.score >= self.min_score_threshold:
            task.transition_to(TaskState.PUBLISHING)
            publish_result = await publish_task(
                task_id=task.task_id,
                score=score_result.score,
                hf_token=self.hf_token,
                min_score=self.min_score_threshold,
                task_data={
                    "repo_url": task.repo_url,
                    "base_commit": task.base_commit,
                    "language": task.language,
                    "patch": task.patch,
                    "tests": task.tests,
                    "docker_image": build_result.image_name,
                },
            )
            task.publish_result = publish_result

            if publish_result.success:
                task.transition_to(TaskState.COMPLETED)
                logger.info(
                    f"[{self.orchestrator_id}] Task {task.task_id} COMPLETED "
                    f"with score {score_result.score:.3f}"
                )
            else:
                task.transition_to(TaskState.REJECTED)
                task.metadata["rejection_reason"] = (
                    f"Publish failed: {publish_result.error}"
                )
                logger.warning(
                    f"[{self.orchestrator_id}] Task {task.task_id} REJECTED: "
                    f"Publish failed - {publish_result.error}"
                )
        else:
            task.transition_to(TaskState.REJECTED)
            task.metadata["rejection_reason"] = (
                f"Score {score_result.score:.3f} below threshold {self.min_score_threshold}"
            )
            logger.warning(
                f"[{self.orchestrator_id}] Task {task.task_id} REJECTED: "
                f"Score {score_result.score:.3f} < {self.min_score_threshold}"
            )

        return task

    async def _attempt_repair(
        self, task: OrchestratorTask, image_name: str
    ) -> OrchestratorTask:
        """Attempt to repair a failing task.

        Args:
            task: The task that needs repair.
            image_name: Docker image name for re-verification.

        Returns:
            The task with updated state and repair results.
        """
        task.transition_to(TaskState.REPAIRING)
        repair_attempts = 0
        error_output = task.verify_result.error if task.verify_result else ""

        while repair_attempts < self.max_repair_attempts:
            repair_attempts += 1
            logger.info(
                f"[{self.orchestrator_id}] Repair attempt {repair_attempts}/{self.max_repair_attempts} "
                f"for task {task.task_id}"
            )

            repair_result = await repair_test(
                task_id=task.task_id,
                error_output=error_output or "",
                max_attempts=1,  # Single attempt per loop iteration
                llm_client=self.llm_client,
                model=self.model,
                patch=task.patch,
                repo_url=task.repo_url,
            )
            task.repair_result = repair_result

            if not repair_result.success:
                if repair_attempts >= self.max_repair_attempts:
                    return self._reject(
                        task,
                        f"Repair failed after {repair_attempts} attempts",
                        repair_result.error,
                    )
                continue

            # Re-verify after repair
            fail_to_pass_tests = task.tests.get("fail_to_pass", [])
            verify_result = await verify_fail_to_pass(
                task_id=task.task_id,
                image_name=image_name,
                tests=fail_to_pass_tests,
                patch=task.patch,
            )
            task.verify_result = verify_result

            if verify_result.success:
                logger.info(
                    f"[{self.orchestrator_id}] Repair succeeded for {task.task_id} "
                    f"after {repair_attempts} attempts"
                )
                return task

            error_output = verify_result.error

        # All repair attempts exhausted
        return self._reject(
            task,
            f"Repair failed after {repair_attempts} attempts",
            f"Max repair attempts ({self.max_repair_attempts}) exhausted",
        )

    def _reject(
        self, task: OrchestratorTask, reason: str, details: str | None = None
    ) -> OrchestratorTask:
        """Reject a task with reason and details.

        Args:
            task: The task to reject.
            reason: Short rejection reason (e.g., 'Test generation failed').
            details: Detailed error message or None.

        Returns:
            The rejected task with REJECTED state.
        """
        task.transition_to(TaskState.REJECTED)
        task.metadata["rejection_reason"] = reason
        if details:
            task.metadata["rejection_details"] = details

        # Log rejection via the reject_task tool
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                reject_task(
                    task_id=task.task_id,
                    reason=reason,
                    details={"details": details} if details else None,
                )
            )
        except RuntimeError:
            # No running loop, run synchronously
            asyncio.run(
                reject_task(
                    task_id=task.task_id,
                    reason=reason,
                    details={"details": details} if details else None,
                )
            )

        logger.warning(
            f"[{self.orchestrator_id}] Task {task.task_id} REJECTED: {reason}"
            f"{' - ' + details if details else ''}"
        )

        return task

    def __repr__(self) -> str:
        """Return string representation of the orchestrator."""
        return (
            f"DatasetOrchestrator("
            f"id={self.orchestrator_id}, "
            f"min_score={self.min_score_threshold}, "
            f"max_repair={self.max_repair_attempts}, "
            f"use_existing_image={self.use_existing_image}"
            f")"
        )
