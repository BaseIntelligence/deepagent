"""Integration tests for dedup functionality."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.swe.dedup import DedupManager
from swe_forge.swe.gharchive import GhArchiveEvent
from swe_forge.swe.github_api import GitHubClient
from swe_forge.swe.models import SweTask, SweTaskStatus
from swe_forge.swe.pipeline import (
    SwePipeline,
    SwePipelineConfig,
    SwePipelineEventType,
)
from swe_forge.swe.pr_cache import PRCache


class TestDedupIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_gh_client(self) -> GitHubClient:
        client = MagicMock(spec=GitHubClient)
        client._session = MagicMock()
        client._session.closed = False
        return client

    @pytest.fixture
    def mock_gh_archive_client(self):
        client = MagicMock()
        client._session = MagicMock()
        client._session.closed = False
        client.close = AsyncMock()
        client._ensure_session = AsyncMock()
        return client

    def create_event(self, repo: str, pr_number: int) -> GhArchiveEvent:
        return GhArchiveEvent(
            id=f"evt-{repo}-{pr_number}",
            event_type="PullRequestEvent",
            repository=repo,
            actor="developer",
            action="merged",
            pull_number=pr_number,
            base_sha="abc123",
            merge_sha="def456",
            title=f"Fix bug in {repo}#{pr_number}",
            body="This PR fixes a bug.",
            language_hint="python",
            stars=100,
            has_org=True,
            created_at=datetime.now(timezone.utc),
            merged_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_parallel_dedup(
        self, mock_gh_client, mock_gh_archive_client, temp_cache_dir
    ):
        """Thread-safety with parallel dedup enabled."""
        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        dedup_manager = DedupManager(pr_cache=pr_cache)

        async def mark_task(task_id: str):
            await dedup_manager.mark_processed(task_id)

        tasks_to_mark = [f"owner-repo-{i}" for i in range(20)]

        await asyncio.gather(*[mark_task(t) for t in tasks_to_mark])

        count = await pr_cache.count()
        assert count == 20

        for task_id in tasks_to_mark:
            assert await dedup_manager.is_processed(task_id)

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_full_workflow_duplicate_detection(
        self, mock_gh_client, mock_gh_archive_client, temp_cache_dir
    ):
        """Full workflow: process task, mark as processed, verify dedup works."""
        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        dedup_manager = DedupManager(pr_cache=pr_cache)

        task_id = "python-cpython-12345"

        assert not await dedup_manager.is_processed(task_id)

        await dedup_manager.mark_processed(task_id)

        assert await dedup_manager.is_processed(task_id)

        await pr_cache.close()

        pr_cache2 = PRCache(temp_cache_dir)
        await pr_cache2.open()

        dedup_manager2 = DedupManager(pr_cache=pr_cache2)

        assert await dedup_manager2.is_processed(task_id)

        await pr_cache2.close()

    @pytest.mark.asyncio
    async def test_pipeline_with_dedup_processes_unique_only(
        self, mock_gh_client, mock_gh_archive_client, temp_cache_dir
    ):
        """Pipeline with dedup processes unique tasks only."""
        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        event1 = self.create_event("owner/repo", 1)
        event2 = self.create_event("owner/repo", 2)
        event3 = self.create_event("owner/repo", 1)

        await pr_cache.mark_processed("owner/repo/1")

        dedup_manager = DedupManager(pr_cache=pr_cache)

        config = SwePipelineConfig(
            max_tasks=3,
            max_candidates=3,
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

        events = [event1, event2, event3]

        with patch.object(pipeline, "_fetch_events", return_value=events):
            with patch("swe_forge.swe.pipeline.enrich_pr", mock_enrich):
                with patch.object(pipeline, "_extract_patch", mock_extract_patch):
                    results = []
                    async with pipeline:
                        async for event in pipeline.run_with_progress():
                            if event.event_type == SwePipelineEventType.TASK_EXTRACTED:
                                results.append(event.data.get("task"))

        assert len(results) <= 2

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_concurrent_mark_processed_no_duplicates(
        self, temp_cache_dir, mock_gh_client, mock_gh_archive_client
    ):
        """Concurrent mark_processed creates single entry."""
        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        async def mark_same_pr():
            await pr_cache.mark_processed("owner/repo/123")

        await asyncio.gather(*[mark_same_pr() for _ in range(50)])

        assert await pr_cache.count() == 1
        assert await pr_cache.is_processed("owner/repo/123")

        await pr_cache.close()
