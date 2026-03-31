"""Tests for dedup integration with pipeline."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.swe.dedup import DedupManager
from swe_forge.swe.gharchive import GhArchiveEvent
from swe_forge.swe.github_api import GitHubClient
from swe_forge.swe.pipeline import (
    BenchmarkMetrics,
    SwePipeline,
    SwePipelineConfig,
    SwePipelineEventType,
)
from swe_forge.swe.pr_cache import PRCache


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
def sample_event() -> GhArchiveEvent:
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


class TestPipelineDedup:
    """Tests for pipeline dedup integration."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.mark.asyncio
    async def test_pipeline_skips_duplicates(
        self, mock_gh_client, mock_gh_archive_client, sample_event, temp_cache_dir
    ):
        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()
        await pr_cache.mark_processed("owner/repo/42")

        dedup_manager = DedupManager(pr_cache=pr_cache)

        config = SwePipelineConfig(
            max_tasks=1,
            max_candidates=1,
            dedup_manager=dedup_manager,
        )

        pipeline = SwePipeline(
            mock_gh_client,
            gh_archive_client=mock_gh_archive_client,
            config=config,
        )

        enriched_call_count = 0

        async def mock_enrich(event, client):
            nonlocal enriched_call_count
            enriched_call_count += 1
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

        with patch.object(pipeline, "_fetch_events", return_value=[sample_event]):
            with patch("swe_forge.swe.pipeline.enrich_pr", mock_enrich):
                events = []
                async with pipeline:
                    async for event in pipeline.run_with_progress():
                        events.append(event)

                assert enriched_call_count == 0
                final_event = events[-1]
                assert final_event.event_type == SwePipelineEventType.PIPELINE_COMPLETED
                metrics = final_event.data.get("metrics")
                assert metrics is not None
                assert metrics.duplicates_skipped == 1

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_pipeline_marks_processed(
        self, mock_gh_client, mock_gh_archive_client, sample_event, temp_cache_dir
    ):
        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        assert not await pr_cache.is_processed("owner/repo/42")

        dedup_manager = DedupManager(pr_cache=pr_cache)

        config = SwePipelineConfig(
            max_tasks=1,
            max_candidates=1,
            dedup_manager=dedup_manager,
        )

        pipeline = SwePipeline(
            mock_gh_client,
            gh_archive_client=mock_gh_archive_client,
            config=config,
        )

        from swe_forge.swe.enricher import EnrichedPullRequest

        async def mock_enrich(event, client):
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

        async def mock_extract_patch(enriched):
            return "patch content", ""

        with patch.object(pipeline, "_fetch_events", return_value=[sample_event]):
            with patch("swe_forge.swe.pipeline.enrich_pr", mock_enrich):
                with patch.object(pipeline, "_extract_patch", mock_extract_patch):
                    events = []
                    async with pipeline:
                        async for event in pipeline.run_with_progress():
                            events.append(event)

                    assert await pr_cache.is_processed("owner/repo/42")

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_pipeline_without_dedup_works_normally(
        self, mock_gh_client, mock_gh_archive_client, sample_event
    ):
        config = SwePipelineConfig(max_tasks=1, max_candidates=1)

        pipeline = SwePipeline(
            mock_gh_client,
            gh_archive_client=mock_gh_archive_client,
            config=config,
        )

        from swe_forge.swe.enricher import EnrichedPullRequest

        async def mock_enrich(event, client):
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

        with patch.object(pipeline, "_fetch_events", return_value=[sample_event]):
            with patch("swe_forge.swe.pipeline.enrich_pr", mock_enrich):
                events = []
                async with pipeline:
                    async for event in pipeline.run_with_progress():
                        events.append(event)

                assert len(events) > 0
                final_event = events[-1]
                assert final_event.event_type == SwePipelineEventType.PIPELINE_COMPLETED
                metrics = final_event.data.get("metrics")
                assert metrics is not None
                assert metrics.duplicates_skipped == 0
