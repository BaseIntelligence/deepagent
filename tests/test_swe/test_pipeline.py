"""Tests for the SWE mining pipeline orchestrator."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.swe.gharchive import GhArchiveEvent
from swe_forge.swe.github_api import GitHubClient
from swe_forge.swe.models import SweTaskStatus
from swe_forge.swe.pipeline import (
    BenchmarkMetrics,
    DifficultyTargets,
    SwePipeline,
    SwePipelineConfig,
    SwePipelineEvent,
    SwePipelineEventType,
    SwePipelineRunResult,
    run_pipeline_once,
)


@pytest.fixture
def mock_gh_client() -> GitHubClient:
    """Create a mock GitHub client."""
    client = MagicMock(spec=GitHubClient)
    client._session = MagicMock()
    client._session.closed = False
    return client


@pytest.fixture
def mock_gh_archive_client():
    """Create a mock GH Archive client."""
    client = MagicMock()
    client._session = MagicMock()
    client._session.closed = False
    client.close = AsyncMock()
    client._ensure_session = AsyncMock()
    return client


@pytest.fixture
def sample_event() -> GhArchiveEvent:
    """Create a sample GH Archive event."""
    return GhArchiveEvent(
        id="evt-12345",
        event_type="PullRequestEvent",
        repository="owner/repo",
        actor="developer",
        action="merged",
        pull_number=42,
        base_sha="abc123",
        merge_sha="def456",
        title="Fix bug in module",
        body="This PR fixes a critical bug.",
        language_hint="python",
        stars=100,
        has_org=True,
        created_at=datetime.now(timezone.utc),
        merged_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_events(sample_event) -> list[GhArchiveEvent]:
    """Create sample GH Archive events."""
    return [
        sample_event,
        GhArchiveEvent(
            id="evt-67890",
            event_type="PullRequestEvent",
            repository="org/project",
            actor="contributor",
            action="merged",
            pull_number=99,
            base_sha="111aaa",
            merge_sha="222bbb",
            title="Add feature X",
            body="Implements feature X.",
            language_hint="python",
            stars=200,
            has_org=True,
            created_at=datetime.now(timezone.utc),
            merged_at=datetime.now(timezone.utc),
        ),
    ]


class TestSwePipelineConfig:
    """Tests for SwePipelineConfig dataclass."""

    def test_default_values(self):
        config = SwePipelineConfig()
        assert config.min_stars == 100
        assert config.max_candidates == 50
        assert config.max_tasks == 1
        assert config.once is True
        assert config.languages == []
        assert config.concurrency_enrich == 20
        assert config.concurrency_preclassify == 25
        assert config.concurrency_deep == 8
        assert config.difficulty_filter is None
        assert config.difficulty_targets is None
        assert config.target_valid_tasks == 10
        assert config.max_hours_back == 168
        assert config.batch_size_hours == 6
        assert config.min_complexity == 0.25
        assert config.verify_docker is True

    def test_custom_values(self):
        config = SwePipelineConfig(
            min_stars=50,
            languages=["python", "typescript"],
            max_candidates=100,
            max_tasks=10,
            once=False,
            concurrency_enrich=30,
            concurrency_preclassify=40,
            concurrency_deep=15,
            difficulty_filter="medium",
        )
        assert config.min_stars == 50
        assert config.languages == ["python", "typescript"]
        assert config.max_candidates == 100
        assert config.max_tasks == 10
        assert config.once is False
        assert config.concurrency_enrich == 30
        assert config.concurrency_preclassify == 40
        assert config.concurrency_deep == 15
        assert config.difficulty_filter == "medium"


class TestDifficultyTargets:
    """Tests for DifficultyTargets dataclass."""

    def test_all_met_empty(self):
        targets = DifficultyTargets()
        assert targets.all_met({}) is True

    def test_all_met_satisfied(self):
        targets = DifficultyTargets(targets={"easy": 2, "medium": 1})
        counts = {"easy": 2, "medium": 1}
        assert targets.all_met(counts) is True

    def test_all_met_not_satisfied(self):
        targets = DifficultyTargets(targets={"easy": 2, "medium": 1})
        counts = {"easy": 1, "medium": 1}
        assert targets.all_met(counts) is False

    def test_all_met_partial_counts(self):
        targets = DifficultyTargets(targets={"easy": 2, "medium": 1})
        counts = {"easy": 2}
        assert targets.all_met(counts) is False


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics dataclass."""

    def test_default_values(self):
        metrics = BenchmarkMetrics()
        assert metrics.total_raw_events == 0
        assert metrics.enriched_count == 0
        assert metrics.accepted_count == 0
        assert metrics.languages == {}

    def test_custom_values(self):
        metrics = BenchmarkMetrics(
            total_raw_events=100,
            enriched_count=80,
            accepted_count=5,
            languages={"python": 3, "typescript": 2},
        )
        assert metrics.total_raw_events == 100
        assert metrics.enriched_count == 80
        assert metrics.accepted_count == 5
        assert metrics.languages["python"] == 3


class TestSwePipelineRunResult:
    """Tests for SwePipelineRunResult dataclass."""

    def test_default_values(self):
        result = SwePipelineRunResult()
        assert result.tasks == []
        assert result.filtered == 0
        assert result.extracted == 0
        assert result.scored == 0
        assert result.benchmark_metrics is None

    def test_with_tasks(self):
        MagicMock(spec=SweTaskStatus)
        result = SwePipelineRunResult(
            tasks=[],
            filtered=10,
            extracted=5,
            scored=3,
        )
        assert result.filtered == 10
        assert result.extracted == 5
        assert result.scored == 3


class TestSwePipelineEvent:
    """Tests for SwePipelineEvent dataclass."""

    def test_event_creation(self):
        event = SwePipelineEvent(
            SwePipelineEventType.COLLECTION_STARTED,
            {"requested": 50},
        )
        assert event.event_type == SwePipelineEventType.COLLECTION_STARTED
        assert event.data["requested"] == 50

    def test_event_default_data(self):
        event = SwePipelineEvent(SwePipelineEventType.PIPELINE_COMPLETED)
        assert event.data == {}


class TestSwePipeline:
    """Tests for SwePipeline class."""

    def test_init(self, mock_gh_client):
        pipeline = SwePipeline(mock_gh_client)
        assert pipeline.gh_client == mock_gh_client
        assert pipeline.config is not None
        assert pipeline._active is False

    def test_init_with_config(self, mock_gh_client):
        config = SwePipelineConfig(min_stars=100, max_tasks=5)
        pipeline = SwePipeline(mock_gh_client, config=config)
        assert pipeline.config.min_stars == 100
        assert pipeline.config.max_tasks == 5

    def test_build_filter_config(self, mock_gh_client):
        config = SwePipelineConfig(min_stars=50, languages=["python", "go"])
        pipeline = SwePipeline(mock_gh_client, config=config)
        filter_config = pipeline._build_filter_config()
        assert filter_config.min_stars == 50

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_gh_client, mock_gh_archive_client):
        pipeline = SwePipeline(mock_gh_client, gh_archive_client=mock_gh_archive_client)
        async with pipeline as p:
            assert p._active is True
        assert pipeline._active is False
        mock_gh_archive_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_event_to_queue(self, mock_gh_client):
        pipeline = SwePipeline(mock_gh_client)
        queue = asyncio.Queue(maxsize=10)
        event = SwePipelineEvent(SwePipelineEventType.COLLECTION_STARTED, {"test": 1})
        await pipeline._emit_event(queue, event)
        assert queue.qsize() == 1
        received = await queue.get()
        assert received.event_type == SwePipelineEventType.COLLECTION_STARTED

    @pytest.mark.asyncio
    async def test_emit_event_none_queue(self, mock_gh_client):
        pipeline = SwePipeline(mock_gh_client)
        event = SwePipelineEvent(SwePipelineEventType.COLLECTION_STARTED)
        await pipeline._emit_event(None, event)

    def test_should_skip_event_bot(self, mock_gh_client, sample_event):
        pipeline = SwePipeline(mock_gh_client)
        sample_event.actor = "test[bot]"
        assert pipeline._should_skip_event(sample_event) is True
        sample_event.actor = "dependabot"
        assert pipeline._should_skip_event(sample_event) is True

    def test_should_skip_event_no_org(self, mock_gh_client, sample_event):
        pipeline = SwePipeline(mock_gh_client)
        sample_event.has_org = False
        assert pipeline._should_skip_event(sample_event) is True

    def test_should_skip_event_valid(self, mock_gh_client, sample_event):
        pipeline = SwePipeline(mock_gh_client)
        sample_event.has_org = True
        sample_event.actor = "developer"
        assert pipeline._should_skip_event(sample_event) is False


class TestSwePipelineRun:
    """Tests for pipeline run methods."""

    @pytest.mark.asyncio
    async def test_run_with_mocked_fetch(
        self, mock_gh_client, mock_gh_archive_client, sample_events
    ):
        """Test pipeline run with mocked event fetching."""
        config = SwePipelineConfig(max_tasks=1, max_candidates=2)
        pipeline = SwePipeline(
            mock_gh_client,
            gh_archive_client=mock_gh_archive_client,
            config=config,
        )

        async def mock_enrich(event, client):
            from swe_forge.swe.enricher import EnrichedPullRequest

            return EnrichedPullRequest(
                id=event.id,
                repo=event.repository,
                number=event.pull_number,
                title=event.title,
                body=event.body or "",
                base_commit=event.base_sha,
                merge_commit=event.merge_sha,
                language=event.language_hint or "python",
                files_changed=2,
                additions=10,
                deletions=5,
                changed_files=["test.py"],
                stars=event.stars,
            )

        with patch.object(pipeline, "_fetch_events", return_value=sample_events):
            with patch("swe_forge.swe.pipeline.enrich_pr", mock_enrich):
                events = []
                async with pipeline:
                    async for event in pipeline.run_with_progress():
                        events.append(event)

                assert len(events) > 0
                assert events[0].event_type == SwePipelineEventType.COLLECTION_STARTED
                assert events[-1].event_type == SwePipelineEventType.PIPELINE_COMPLETED


class TestPipelineHelpers:
    """Tests for pipeline helper functions."""

    def test_difficulty_matches_filter_no_filter(self, mock_gh_client):
        config = SwePipelineConfig()
        pipeline = SwePipeline(mock_gh_client, config=config)
        assert pipeline._difficulty_matches_filter("easy", config) is True
        assert pipeline._difficulty_matches_filter("medium", config) is True
        assert pipeline._difficulty_matches_filter("hard", config) is True

    def test_difficulty_matches_filter_with_filter(self, mock_gh_client):
        config = SwePipelineConfig(difficulty_filter="medium")
        pipeline = SwePipeline(mock_gh_client, config=config)
        assert pipeline._difficulty_matches_filter("medium", config) is True
        assert pipeline._difficulty_matches_filter("easy", config) is False

    def test_difficulty_matches_filter_with_targets(self, mock_gh_client):
        config = SwePipelineConfig(
            difficulty_targets=DifficultyTargets(targets={"easy": 2, "hard": 1})
        )
        pipeline = SwePipeline(mock_gh_client, config=config)
        assert pipeline._difficulty_matches_filter("easy", config) is True
        assert pipeline._difficulty_matches_filter("hard", config) is True
        assert pipeline._difficulty_matches_filter("medium", config) is False


class TestRunPipelineOnce:
    """Tests for run_pipeline_once convenience function."""

    @pytest.mark.asyncio
    async def test_run_pipeline_once_mocked(self, mock_gh_client):
        """Test that run_pipeline_once creates pipeline and runs it."""
        config = SwePipelineConfig(max_tasks=1)

        with patch("swe_forge.swe.pipeline.SwePipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.run = AsyncMock(
                return_value=SwePipelineRunResult(
                    tasks=[],
                    filtered=5,
                    extracted=0,
                )
            )
            MockPipeline.return_value = mock_instance

            result = await run_pipeline_once(mock_gh_client, config)
            assert result.filtered == 5


class TestPipelineEventType:
    """Tests for pipeline event types."""

    def test_event_types_exist(self):
        assert SwePipelineEventType.COLLECTION_STARTED.value == "collection_started"
        assert SwePipelineEventType.BATCH_FETCHED.value == "batch_fetched"
        assert SwePipelineEventType.PIPELINE_PROGRESS.value == "pipeline_progress"
        assert SwePipelineEventType.CANDIDATE_FILTERED.value == "candidate_filtered"
        assert SwePipelineEventType.TASK_EXTRACTED.value == "task_extracted"
        assert SwePipelineEventType.QUALITY_SCORED.value == "quality_scored"
        assert SwePipelineEventType.PIPELINE_COMPLETED.value == "pipeline_completed"


class TestDifficultyClassifierIntegration:
    """Tests for difficulty classifier integration with pipeline."""

    @pytest.mark.asyncio
    async def test_difficulty_classifier_integration(self, mock_gh_client):
        """Verify classifier called when llm_client available."""
        from unittest.mock import AsyncMock

        from swe_forge.swe.difficulty import (
            ClassifyResponse,
            DifficultyClassifier,
            TriageResponse,
        )
        from swe_forge.swe.enricher import EnrichedPullRequest

        mock_classifier = MagicMock(spec=DifficultyClassifier)
        mock_classifier.classify_triage = AsyncMock(
            return_value=TriageResponse(difficulty="hard", reasoning="Complex changes")
        )
        mock_classifier.classify_full = AsyncMock(
            return_value=ClassifyResponse(
                difficulty="hard",
                score=0.8,
                quality_good=True,
                reasoning="High complexity",
            )
        )

        config = SwePipelineConfig(difficulty_classifier=mock_classifier)
        pipeline = SwePipeline(mock_gh_client, config=config)

        enriched = EnrichedPullRequest(
            id="test-1",
            repo="owner/repo",
            number=1,
            title="Fix critical bug",
            body="This PR fixes a critical bug",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            files_changed=10,
            additions=100,
            deletions=50,
            changed_files=["src/main.py", "src/utils.py"],
            stars=100,
        )

        metrics = BenchmarkMetrics()
        semaphore = asyncio.Semaphore(10)

        result = await pipeline._preclassify_stage(enriched, semaphore, metrics)

        assert result == "hard"
        mock_classifier.classify_triage.assert_called_once()
        assert metrics.preclassify_hard == 1

    @pytest.mark.asyncio
    async def test_difficulty_classifier_fallback(self, mock_gh_client):
        """Verify NO HARDCODED HEURISTICS when llm_client=None."""
        from swe_forge.swe.enricher import EnrichedPullRequest

        config = SwePipelineConfig()
        pipeline = SwePipeline(mock_gh_client, config=config)

        enriched = EnrichedPullRequest(
            id="test-1",
            repo="owner/repo",
            number=1,
            title="Fix bug",
            body="Bug fix",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            files_changed=1,
            additions=10,
            deletions=5,
            changed_files=["src/main.py"],
            stars=100,
        )

        metrics = BenchmarkMetrics()
        semaphore = asyncio.Semaphore(10)

        result = await pipeline._preclassify_stage(enriched, semaphore, metrics)

        # NO HARDCODED HEURISTICS - returns None when no classifier
        assert result is None
        # No classification happened
        assert metrics.preclassify_easy == 0

    @pytest.mark.asyncio
    async def test_difficulty_classifier_score_mapping(self, mock_gh_client):
        """Verify score conversion (0.5 -> 5, 0.3 -> 3, etc.)."""
        from unittest.mock import AsyncMock

        from swe_forge.swe.difficulty import (
            ClassifyResponse,
            DifficultyClassifier,
        )
        from swe_forge.swe.enricher import EnrichedPullRequest

        test_cases = [
            (0.5, 5),
            (0.3, 3),
            (0.1, 1),
            (1.0, 10),
            (0.95, 9),
            (0.05, 1),
        ]

        for score, expected_difficulty_score in test_cases:
            mock_classifier = MagicMock(spec=DifficultyClassifier)
            mock_classifier.classify_full = AsyncMock(
                return_value=ClassifyResponse(
                    difficulty="medium",
                    score=score,
                    quality_good=True,
                    reasoning="Test",
                )
            )

            config = SwePipelineConfig(difficulty_classifier=mock_classifier)
            pipeline = SwePipeline(mock_gh_client, config=config)

            enriched = EnrichedPullRequest(
                id="test-1",
                repo="owner/repo",
                number=1,
                title="Fix bug",
                body="Bug fix",
                base_commit="abc123",
                merge_commit="def456",
                language="python",
                files_changed=3,
                additions=30,
                deletions=10,
                changed_files=["src/main.py"],
                stars=100,
            )

            metrics = BenchmarkMetrics()
            semaphore = asyncio.Semaphore(10)
            event_queue = asyncio.Queue(maxsize=10)

            with patch.object(pipeline, "_extract_patch", return_value=("patch", "")):
                task = await pipeline._deep_stage(
                    enriched, semaphore, metrics, event_queue
                )

            assert task is not None
            assert task.difficulty_score == expected_difficulty_score, (
                f"Score {score} should map to {expected_difficulty_score}, got {task.difficulty_score}"
            )

    def test_get_classifier_with_preconfigured(self, mock_gh_client):
        """Verify _get_classifier returns pre-configured classifier."""
        from swe_forge.swe.difficulty import DifficultyClassifier

        mock_classifier = MagicMock(spec=DifficultyClassifier)
        config = SwePipelineConfig(difficulty_classifier=mock_classifier)
        pipeline = SwePipeline(mock_gh_client, config=config)

        result = pipeline._get_classifier()

        assert result is mock_classifier

    def test_get_classifier_without_llm_client(self, mock_gh_client):
        """Verify _get_classifier returns None when no llm_client configured."""
        config = SwePipelineConfig()
        pipeline = SwePipeline(mock_gh_client, config=config)

        result = pipeline._get_classifier()

        assert result is None
