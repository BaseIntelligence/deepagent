"""Complete A-Z mining pipeline with Docker verification.

This module implements the full mining workflow:
1. Fetch PR from GitHub
2. Detect language (rule-based)
3. Discover commands (agentic from CI/CD)
4. LLM generates workspace.yml prompt
5. LLM generates tests from patch
6. Build Docker test image
7. Run tests BEFORE patch (must fail)
8. Apply patch
9. Run tests AFTER patch (must pass)
10. Export validated task only if all checks pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from swe_forge.detection import detect_language_from_files
from swe_forge.discovery import AgenticCommandDiscovery
from swe_forge.execution.docker_client import DockerClient
from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig
from swe_forge.export.jsonl import export_jsonl
from swe_forge.swe.github_api import GitHubClient, PullRequest, PRFile
from swe_forge.swe.models import SweTask, SweTaskStatus
from swe_forge.swe.test_generator import GeneratedTests, TestGenerator

if TYPE_CHECKING:
    from swe_forge.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ValidatedTask:
    """A task that has passed all verification checks."""

    task: SweTask
    before_tests_passed: bool
    after_tests_passed: bool
    before_test_output: str
    after_test_output: str
    verification_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.id,
            "repo": self.task.repo,
            "before_tests_passed": self.before_tests_passed,
            "after_tests_passed": self.after_tests_passed,
            "verification_timestamp": self.verification_timestamp.isoformat(),
        }


@dataclass
class PipelineResult:
    """Result of running the complete mining pipeline."""

    success: bool
    validated_task: ValidatedTask | None = None
    error: str | None = None
    stage: str = ""
    pr_data: dict[str, Any] = field(default_factory=dict)
    language_detected: str = ""
    commands_discovered: dict[str, Any] = field(default_factory=dict)
    tests_generated: dict[str, Any] = field(default_factory=dict)


class CompleteMiningPipeline:
    """Complete A-Z mining pipeline with Docker verification.

    This pipeline ensures that:
    - Tests FAIL before applying the patch (proves bug exists)
    - Tests PASS after applying the patch (proves fix works)
    - Only then is the task exported

    Usage:
        async with GitHubClient(token) as gh:
            pipeline = CompleteMiningPipeline(gh_client=gh, llm_client=llm)
            result = await pipeline.mine_pr('python/cpython', 12345)
            if result:
                print(f"Validated: {result.task.id}")
    """

    def __init__(
        self,
        gh_client: GitHubClient,
        llm_client: LLMClient | None = None,
        docker_client: DockerClient | None = None,
        *,
        model: str = "openai/gpt-5.4",
        max_test_turns: int = 200,
        test_timeout: float = 600.0,
        install_timeout: float = 300.0,
        max_context_tokens: int = 150000,
    ) -> None:
        self.gh_client = gh_client
        self.llm_client = llm_client
        self.docker_client = docker_client
        self.model = model
        self.max_test_turns = max_test_turns
        self.test_timeout = test_timeout
        self.install_timeout = install_timeout
        self.max_context_tokens = max_context_tokens
        self._own_docker = docker_client is None

    async def mine_pr(
        self,
        repo: str,
        pr_number: int,
    ) -> ValidatedTask | None:
        """Mine a single PR through the complete A-Z pipeline.

        Args:
            repo: Repository in owner/repo format
            pr_number: Pull request number

        Returns:
            ValidatedTask if all checks pass, None otherwise
        """
        result = await self._run_pipeline(repo, pr_number)
        return result.validated_task

    async def verify_task(self, task: SweTask) -> bool:
        """Verify an existing SweTask against Docker tests.

        Args:
            task: The SweTask to verify

        Returns:
            True if task passes verification, False otherwise
        """
        from swe_forge.docker_test import DockerTestHarness, verify_patch_fixes_issue

        docker_client = self.docker_client or DockerClient()

        try:
            async with docker_client:
                harness = DockerTestHarness(docker_client)

                verification = await verify_patch_fixes_issue(
                    harness=harness,
                    repo_url=f"https://github.com/{task.repo}",
                    base_commit=task.base_commit,
                    patch=task.patch,
                    fail_to_pass=task.fail_to_pass,
                    pass_to_pass=task.pass_to_pass,
                    install_commands=task.install_config.get("install_commands", []),
                    test_timeout=self.test_timeout,
                )

                return verification.passed
        finally:
            if self._own_docker and docker_client:
                pass  # DockerClient context handles cleanup

    async def _run_pipeline(self, repo: str, pr_number: int) -> PipelineResult:
        """Execute the full pipeline for a single PR."""

        # Stage 1: Fetch PR
        pr, files, diff = await self._fetch_pr(repo, pr_number)
        if not pr:
            return PipelineResult(
                success=False,
                error="Failed to fetch PR",
                stage="fetch",
            )

        # Stage 2: Detect language (rule-based)
        filenames = [f.filename for f in files]
        language = detect_language_from_files(filenames)

        # Stage 3: Discover commands (agentic from CI/CD)
        parts = repo.split("/")
        owner, repo_name = parts[0], parts[1]
        ci_cd_files = await self.gh_client.get_ci_cd_files(
            owner, repo_name, ref=pr.base_sha
        )

        file_contents = dict(ci_cd_files)
        for f in files[:20]:
            if f.patch:
                file_contents[f.filename] = f.patch

        discovery = AgenticCommandDiscovery(llm_client=self.llm_client)
        discovered = await discovery.discover_install_commands(language, file_contents)
        test_discovered = await discovery.discover_test_commands(
            language, file_contents
        )

        discovered.install_commands.extend(test_discovered.test_commands)

        # Stage 4: Create initial task
        task = SweTask(
            id=f"{repo.replace('/', '-')}-{pr_number}",
            repo=repo,
            base_commit=pr.base_sha,
            merge_commit=pr.head_sha,
            language=language.value,
            prompt=pr.title,
            original_pr_body=pr.body or "",
            patch=diff,
            status=SweTaskStatus.CANDIDATE,
            install_config={
                "install_commands": discovered.install_commands,
                "validated": bool(discovered.install_commands),
            },
        )

        # Stage 5: Generate tests via LLM
        if self.llm_client and discovered.install_commands:
            generated = await self._generate_tests(task, pr, files)
            if generated and generated.success:
                task.fail_to_pass = generated.fail_to_pass
                task.pass_to_pass = generated.pass_to_pass
                task.test_patch = (
                    "\n".join(
                        f"# {tf.path}\n{tf.content}" for tf in generated.test_files
                    )
                    if generated.test_files
                    else ""
                )
                if generated.install_commands:
                    task.install_config["install_commands"] = generated.install_commands
                    task.install_config["validated"] = True

        # Stage 6-10: Docker verification
        if not task.fail_to_pass:
            logger.warning(f"No fail_to_pass tests generated for {task.id}")
            return PipelineResult(
                success=False,
                error="No tests generated",
                stage="test_generation",
                pr_data={"number": pr_number, "repo": repo},
                language_detected=language.value,
            )

        verification_result = await self._run_docker_verification(task)

        if verification_result:
            task.status = SweTaskStatus.VALIDATED
            task.docker_passed = True
            return PipelineResult(
                success=True,
                validated_task=verification_result,
                stage="complete",
                pr_data={"number": pr_number, "repo": repo},
                language_detected=language.value,
            )

        return PipelineResult(
            success=False,
            error="Docker verification failed",
            stage="docker_verification",
            pr_data={"number": pr_number, "repo": repo},
            language_detected=language.value,
        )

    async def _fetch_pr(
        self,
        repo: str,
        pr_number: int,
    ) -> tuple[PullRequest | None, list[PRFile], str]:
        """Fetch PR data from GitHub."""
        parts = repo.split("/")
        if len(parts) != 2:
            logger.error(f"Invalid repo format: {repo}")
            return None, [], ""

        owner, repo_name = parts

        try:
            pr = await self.gh_client.get_pr(owner, repo_name, pr_number)
            files = await self.gh_client.get_pr_files(owner, repo_name, pr_number)
            diff = await self.gh_client.get_pr_diff(owner, repo_name, pr_number)
            return pr, files, diff
        except Exception as e:
            logger.error(f"Failed to fetch PR {repo}#{pr_number}: {e}")
            return None, [], ""

    async def _generate_tests(
        self,
        task: SweTask,
        pr: PullRequest,
        files: list[PRFile],
    ) -> GeneratedTests | None:
        """Generate test commands using TestGenerator."""
        if not self.llm_client:
            return None

        docker_client = self.docker_client or DockerClient()

        try:
            async with docker_client:
                config = SandboxConfig(
                    image="ubuntu:24.04",
                    command_timeout=self.test_timeout,
                    install_timeout=self.install_timeout,
                )
                sandbox = DockerSandbox(docker_client, config)

                async with sandbox:
                    await sandbox.setup_workspace(
                        f"https://github.com/{task.repo}",
                        task.base_commit,
                    )

                    generator = TestGenerator(
                        llm=self.llm_client,
                        model=self.model,
                        max_turns=self.max_test_turns,
                        max_context_tokens=self.max_context_tokens,
                    )

                    return await generator.generate_tests(task, sandbox)
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return None
        finally:
            if self._own_docker:
                pass

    async def _run_docker_verification(self, task: SweTask) -> ValidatedTask | None:
        """Run Docker verification: tests must fail before, pass after patch."""
        from swe_forge.docker_test import DockerTestHarness, verify_patch_fixes_issue
        from swe_forge.docker_test.verification import TestFile

        docker_client = self.docker_client or DockerClient()

        try:
            async with docker_client:
                harness = DockerTestHarness(docker_client)

                test_files = []
                if task.test_patch:
                    lines = task.test_patch.split("\n")
                    current_file = None
                    current_content = []
                    for line in lines:
                        if line.startswith("# Test file: ") or line.startswith("# "):
                            if current_file and current_content:
                                test_files.append(
                                    TestFile(
                                        path=current_file,
                                        content="\n".join(current_content),
                                    )
                                )
                            current_file = (
                                line.replace("# Test file: ", "")
                                .replace("# ", "")
                                .strip()
                            )
                            current_content = []
                        else:
                            current_content.append(line)
                    if current_file and current_content:
                        test_files.append(
                            TestFile(
                                path=current_file, content="\n".join(current_content)
                            )
                        )

                verification = await verify_patch_fixes_issue(
                    harness=harness,
                    repo_url=f"https://github.com/{task.repo}",
                    base_commit=task.base_commit,
                    patch=task.patch,
                    fail_to_pass=task.fail_to_pass,
                    pass_to_pass=task.pass_to_pass,
                    install_commands=task.install_config.get("install_commands", []),
                    test_files=test_files,
                    test_timeout=self.test_timeout,
                )

                if verification.passed:
                    return ValidatedTask(
                        task=task,
                        before_tests_passed=verification.before_passed,
                        after_tests_passed=verification.after_passed,
                        before_test_output=verification.before_output,
                        after_test_output=verification.after_output,
                    )

                return None
        except Exception as e:
            logger.error(f"Docker verification failed: {e}")
            return None


async def run_complete_mining(
    gh_client: GitHubClient,
    llm_client: LLMClient | None,
    repo: str,
    pr_number: int,
    output_path: str | Path | None = None,
) -> ValidatedTask | None:
    """Convenience function to run complete mining for a single PR.

    Args:
        gh_client: GitHub API client
        llm_client: LLM client for test generation
        repo: Repository in owner/repo format
        pr_number: Pull request number
        output_path: Optional path to export validated task

    Returns:
        ValidatedTask if successful, None otherwise
    """
    pipeline = CompleteMiningPipeline(gh_client=gh_client, llm_client=llm_client)
    result = await pipeline.mine_pr(repo, pr_number)

    if result and output_path:
        export_jsonl([result.task], output_path, append=True)

    return result
