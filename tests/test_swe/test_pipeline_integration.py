"""Integration tests for pipeline+generator connection.

These tests verify that the SwePipeline correctly integrates with TestGenerator
during the _deep_stage and _run_test_generation methods.
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
from typing import TYPE_CHECKING

import pytest

from swe_forge.swe.enricher import EnrichedPullRequest
from swe_forge.swe.github_api import GitHubClient
from swe_forge.swe.models import SweTask, SweTaskStatus
from swe_forge.swe.pipeline import (
    BenchmarkMetrics,
    SwePipeline,
    SwePipelineConfig,
    SwePipelineEvent,
    SwePipelineEventType,
)
from swe_forge.swe.test_generator import GeneratedTests, TestFile


if TYPE_CHECKING:
    pass


@dataclass
class MockExecResult:
    stdout: str
    stderr: str
    exit_code: int


class MockSandbox:
    """Mock sandbox that supports async context manager protocol."""

    def __init__(self):
        self.files: dict[str, str] = {}
        self.commands: list[tuple[str, float | None]] = []

    async def run_command(self, cmd: str, *, timeout: float | None = None):
        self.commands.append((cmd, timeout))
        return MockExecResult(stdout="output", stderr="", exit_code=0)

    async def write_file(self, path: str, content: str):
        self.files[path] = content

    async def read_file(self, path: str):
        return self.files.get(path, "")

    async def setup_workspace(self, repo_url: str, commit: str):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockTestGenerator:
    """Mock test generator that returns predictable GeneratedTests."""

    def __init__(self, result: GeneratedTests):
        self.result = result
        self.generate_tests_called = False
        self.last_task: SweTask | None = None
        self.last_sandbox: MockSandbox | None = None

    async def generate_tests(self, task: SweTask, sandbox) -> GeneratedTests:
        self.generate_tests_called = True
        self.last_task = task
        self.last_sandbox = sandbox
        return self.result


class FailingMockGenerator:
    """Mock generator that raises an exception."""

    def __init__(self, error_message: str = "LLM connection failed"):
        self.error_message = error_message
        self.generate_tests_called = False

    async def generate_tests(self, task: SweTask, sandbox):
        self.generate_tests_called = True
        raise RuntimeError(self.error_message)


@pytest.fixture
def mock_gh_client() -> GitHubClient:
    client = MagicMock(spec=GitHubClient)
    client._session = MagicMock()
    client._session.closed = False
    return client


@pytest.fixture
def mock_gh_archive_client():
    client = MagicMock()
    client._session = MagicMock()
    client._session.closed = False
    client.close = AsyncMock()
    client._ensure_session = AsyncMock()
    return client


@pytest.fixture
def sample_enriched_pr() -> EnrichedPullRequest:
    return EnrichedPullRequest(
        id="test-pr-123",
        repo="owner/repo",
        number=42,
        title="Fix bug in module",
        body="This PR fixes a critical bug.",
        base_commit="abc123def456",
        merge_commit="def456abc123",
        language="python",
        files_changed=3,
        additions=50,
        deletions=20,
        changed_files=["src/main.py", "src/utils.py", "tests/test_main.py"],
        stars=100,
    )


def _bind_mock_sandbox(pipeline: SwePipeline, mock_sandbox: MockSandbox):
    """Bind mock sandbox to pipeline's _create_sandbox.

    The source code calls _create_sandbox without await, so we bind a
    sync function that returns the mock directly (not a coroutine).
    """

    def create_sandbox_sync(self, repo_url: str, base_commit: str):
        return mock_sandbox

    pipeline._create_sandbox = create_sandbox_sync.__get__(pipeline, type(pipeline))


class TestGeneratorCalledWhenConfigured:
    @pytest.mark.asyncio
    async def test_generator_called_with_task(self, mock_gh_client, sample_enriched_pr):
        generated_result = GeneratedTests(
            fail_to_pass=["pytest tests/test_main.py::test_fix"],
            pass_to_pass=["pytest tests/test_utils.py"],
            test_files=[
                TestFile(
                    path="tests/test_generated.py",
                    content="def test_generated(): assert True",
                )
            ],
            install_commands=["pip install -e ."],
            turn_count=3,
            success=True,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123def456",
            merge_commit="def456abc123",
            language="python",
            prompt="Fix bug in module",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert mock_generator.generate_tests_called is True
        assert mock_generator.last_task.id == task.id
        assert mock_generator.last_sandbox is mock_sandbox

    @pytest.mark.asyncio
    async def test_generator_not_called_when_not_configured(
        self, mock_gh_client, sample_enriched_pr
    ):
        config = SwePipelineConfig(test_generator=None)
        pipeline = SwePipeline(mock_gh_client, config=config)

        SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123def456",
            language="python",
            prompt="Test",
            status=SweTaskStatus.CANDIDATE,
        )

        with patch.object(
            pipeline, "_extract_patch", return_value=("diff --git a/file.py", "")
        ):
            metrics = BenchmarkMetrics()
            result = await pipeline._deep_stage(
                sample_enriched_pr,
                asyncio.Semaphore(1),
                metrics,
                None,
            )

        assert result is not None
        assert result.quality_score is not None
        assert result.quality_score >= 0.7


class TestGeneratorResultsMappedToTask:
    @pytest.mark.asyncio
    async def test_generator_results_mapped(self, mock_gh_client, sample_enriched_pr):
        generated_result = GeneratedTests(
            fail_to_pass=["pytest tests/test_main.py::test_bug_fix"],
            pass_to_pass=["pytest tests/test_other.py"],
            test_files=[
                TestFile(
                    path="tests/test_new.py",
                    content="# New test file\ndef test_new_feature(): pass",
                ),
                TestFile(
                    path="tests/test_extra.py",
                    content="# Extra test\ndef test_extra(): pass",
                ),
            ],
            install_commands=["pip install -e .", "pip install pytest"],
            turn_count=5,
            success=True,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123def456",
            merge_commit="def456abc123",
            language="python",
            prompt="Test prompt",
            test_patch="# Original test patch",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert task.fail_to_pass == ["pytest tests/test_main.py::test_bug_fix"]
        assert task.pass_to_pass == ["pytest tests/test_other.py"]

        assert "tests/test_new.py" in task.test_patch
        assert "tests/test_extra.py" in task.test_patch
        assert "def test_new_feature" in task.test_patch

        assert "install_commands" in task.install_config
        assert "pip install -e ." in task.install_config["install_commands"]
        assert task.install_config.get("validated") is True

        assert task.quality_score == 1.0
        assert task.quality_passed is True
        assert task.status == SweTaskStatus.READY

    @pytest.mark.asyncio
    async def test_quality_score_and_difficulty_from_turn_count(
        self, mock_gh_client, sample_enriched_pr
    ):
        generated_result = GeneratedTests(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
            test_files=[],
            install_commands=["pip install -e ."],
            turn_count=10,
            success=True,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert task.difficulty_score == 3
        assert metrics.difficulty_hard == 1

    @pytest.mark.asyncio
    async def test_low_turn_count_easy_difficulty(
        self, mock_gh_client, sample_enriched_pr
    ):
        generated_result = GeneratedTests(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
            test_files=[],
            install_commands=["pip install -e ."],
            turn_count=2,
            success=True,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert task.difficulty_score == 1
        assert metrics.difficulty_easy == 1


class TestGeneratorFailureRejected:
    @pytest.mark.asyncio
    async def test_generator_failure_marks_rejected(
        self, mock_gh_client, sample_enriched_pr
    ):
        generated_result = GeneratedTests(
            fail_to_pass=[],
            pass_to_pass=[],
            test_files=[],
            install_commands=[],
            turn_count=0,
            success=False,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123def456",
            merge_commit="def456abc123",
            language="python",
            prompt="Test prompt",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert task.status == SweTaskStatus.REJECTED
        assert task.quality_score == 0.0
        assert task.quality_passed is False

        assert metrics.quality_failed == 1
        assert metrics.quality_passed == 0

    @pytest.mark.asyncio
    async def test_generator_exception_marks_rejected(
        self, mock_gh_client, sample_enriched_pr
    ):
        failing_generator = FailingMockGenerator("LLM connection failed")
        config = SwePipelineConfig(test_generator=failing_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert task.status == SweTaskStatus.REJECTED
        assert task.quality_score == 0.0
        assert task.quality_passed is False
        assert metrics.quality_failed == 1


class TestEventEmission:
    @pytest.mark.asyncio
    async def test_quality_scored_event_emitted_on_success(
        self, mock_gh_client, sample_enriched_pr
    ):
        generated_result = GeneratedTests(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
            test_files=[],
            install_commands=["pip install -e ."],
            turn_count=4,
            success=True,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="test-task-123",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        event_queue: asyncio.Queue[SwePipelineEvent] = asyncio.Queue()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(
            sample_enriched_pr, task, metrics, event_queue
        )

        event = event_queue.get_nowait()
        assert event.event_type == SwePipelineEventType.QUALITY_SCORED
        assert event.data["task_id"] == "test-task-123"
        assert event.data["score"] == 1.0
        assert event.data["passed"] is True
        assert event.data["turn_count"] == 4


class TestInstallConfigMapping:
    @pytest.mark.asyncio
    async def test_install_commands_mapped_to_task(
        self, mock_gh_client, sample_enriched_pr
    ):
        generated_result = GeneratedTests(
            fail_to_pass=["pytest test.py"],
            pass_to_pass=[],
            test_files=[],
            install_commands=[
                "apt-get update",
                "apt-get install -y libffi-dev",
                "pip install -e .",
                "pip install pytest-cov",
            ],
            turn_count=5,
            success=True,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert len(task.install_config["install_commands"]) == 4
        assert "apt-get update" in task.install_config["install_commands"]
        assert "pip install -e ." in task.install_config["install_commands"]
        assert task.install_config["validated"] is True


class TestTestFilesMerged:
    @pytest.mark.asyncio
    async def test_test_files_appended_to_test_patch(
        self, mock_gh_client, sample_enriched_pr
    ):
        generated_result = GeneratedTests(
            fail_to_pass=["pytest tests/"],
            pass_to_pass=[],
            test_files=[
                TestFile(
                    path="tests/test_feature.py",
                    content="def test_feature():\n    assert 1 + 1 == 2\n",
                ),
            ],
            install_commands=["pip install pytest"],
            turn_count=3,
            success=True,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test",
            test_patch="# Original test patch\ndef test_original(): pass",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert "# Original test patch" in task.test_patch
        assert "tests/test_feature.py" in task.test_patch
        assert "def test_feature" in task.test_patch

    @pytest.mark.asyncio
    async def test_test_files_creates_test_patch_if_empty(
        self, mock_gh_client, sample_enriched_pr
    ):
        generated_result = GeneratedTests(
            fail_to_pass=["pytest tests/"],
            pass_to_pass=[],
            test_files=[
                TestFile(
                    path="tests/test_new.py",
                    content="# Brand new test\ndef test_new(): assert True",
                ),
            ],
            install_commands=["pip install pytest"],
            turn_count=2,
            success=True,
        )

        mock_generator = MockTestGenerator(generated_result)
        config = SwePipelineConfig(test_generator=mock_generator)
        pipeline = SwePipeline(mock_gh_client, config=config)

        task = SweTask(
            id="owner-repo-42",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            prompt="Test",
            test_patch="",
            status=SweTaskStatus.CANDIDATE,
        )

        metrics = BenchmarkMetrics()
        mock_sandbox = MockSandbox()
        _bind_mock_sandbox(pipeline, mock_sandbox)

        await pipeline._run_test_generation(sample_enriched_pr, task, metrics, None)

        assert "tests/test_new.py" in task.test_patch
        assert "def test_new" in task.test_patch
