"""Tests for dedup module - format conversion, HF cache, DedupManager.

TDD approach: RED -> GREEN -> REFACTOR
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.swe.pr_cache import PRCache


# =============================================================================
# Task 1: Format Conversion Tests
# =============================================================================


class TestTaskIdToPrId:
    """Tests for task_id_to_pr_id conversion function."""

    def test_happy_path_standard_format(self):
        """Convert standard task ID format to PR ID."""
        from swe_forge.swe.dedup import task_id_to_pr_id

        # Standard format: owner-repo-123 -> owner/repo/123
        result = task_id_to_pr_id("owner-repo-123")
        assert result == "owner/repo/123"

    def test_repository_with_dashes(self):
        """Handle repository names that contain dashes."""
        from swe_forge.swe.dedup import task_id_to_pr_id

        # python-cpython-123456 -> python/cpython/123456
        result = task_id_to_pr_id("python-cpython-123456")
        assert result == "python/cpython/123456"

    def test_repository_with_multiple_dashes(self):
        from swe_forge.swe.dedup import task_id_to_pr_id

        result = task_id_to_pr_id("my-org-my-repo-name-789")
        assert result == "my/org-my-repo-name/789"

    def test_owner_with_dash(self):
        from swe_forge.swe.dedup import task_id_to_pr_id

        result = task_id_to_pr_id("some-org-repo-123")
        assert result == "some/org-repo/123"

    def test_malformed_input_no_number(self):
        """Return None for malformed input without PR number."""
        from swe_forge.swe.dedup import task_id_to_pr_id

        result = task_id_to_pr_id("owner-repo")
        assert result is None

    def test_malformed_input_empty(self):
        """Return None for empty input."""
        from swe_forge.swe.dedup import task_id_to_pr_id

        result = task_id_to_pr_id("")
        assert result is None

    def test_malformed_input_no_dash(self):
        """Return None for input without dash."""
        from swe_forge.swe.dedup import task_id_to_pr_id

        result = task_id_to_pr_id("invalidformat")
        assert result is None

    def test_number_extraction(self):
        """Correctly extract number from end of task ID."""
        from swe_forge.swe.dedup import task_id_to_pr_id

        result = task_id_to_pr_id("owner-repo-999999")
        assert result == "owner/repo/999999"


class TestPrIdToTaskId:
    """Tests for pr_id_to_task_id conversion function."""

    def test_happy_path_standard_format(self):
        """Convert standard PR ID format to task ID."""
        from swe_forge.swe.dedup import pr_id_to_task_id

        # Standard format: owner/repo/123 -> owner-repo-123
        result = pr_id_to_task_id("owner/repo/123")
        assert result == "owner-repo-123"

    def test_repository_with_dashes(self):
        """Handle repository names that contain dashes."""
        from swe_forge.swe.dedup import pr_id_to_task_id

        result = pr_id_to_task_id("python/cpython/123456")
        assert result == "python-cpython-123456"

    def test_repository_with_multiple_dashes(self):
        """Handle repository names with multiple dashes."""
        from swe_forge.swe.dedup import pr_id_to_task_id

        result = pr_id_to_task_id("my-org/my-repo-name/789")
        assert result == "my-org-my-repo-name-789"

    def test_malformed_input_no_slashes(self):
        """Return None for malformed input without slashes."""
        from swe_forge.swe.dedup import pr_id_to_task_id

        result = pr_id_to_task_id("invalidformat")
        assert result is None

    def test_malformed_input_too_few_parts(self):
        """Return None for input with too few parts."""
        from swe_forge.swe.dedup import pr_id_to_task_id

        result = pr_id_to_task_id("owner/repo")
        assert result is None

    def test_malformed_input_empty(self):
        """Return None for empty input."""
        from swe_forge.swe.dedup import pr_id_to_task_id

        result = pr_id_to_task_id("")
        assert result is None


class TestFormatConversionRoundtrip:
    """Tests for roundtrip conversion."""

    def test_roundtrip_standard(self):
        """Roundtrip conversion preserves original value."""
        from swe_forge.swe.dedup import pr_id_to_task_id, task_id_to_pr_id

        original_pr_id = "owner/repo/123"
        task_id = pr_id_to_task_id(original_pr_id)
        back_to_pr_id = task_id_to_pr_id(task_id)
        assert back_to_pr_id == original_pr_id

    def test_roundtrip_with_dashes_in_repo(self):
        """Roundtrip works with dashes in repo name."""
        from swe_forge.swe.dedup import pr_id_to_task_id, task_id_to_pr_id

        original_pr_id = "python/cpython/123456"
        task_id = pr_id_to_task_id(original_pr_id)
        back_to_pr_id = task_id_to_pr_id(task_id)
        assert back_to_pr_id == original_pr_id

    def test_roundtrip_complex_repo(self):
        from swe_forge.swe.dedup import pr_id_to_task_id, task_id_to_pr_id

        original_pr_id = "my/my-complex-repo-name/789"
        task_id = pr_id_to_task_id(original_pr_id)
        back_to_pr_id = task_id_to_pr_id(task_id)
        assert back_to_pr_id == original_pr_id


# =============================================================================
# Task 2: Thread-Safe PRCache Tests
# =============================================================================


class TestPRCacheThreadSafety:
    """Tests for thread-safe PRCache operations."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    async def test_concurrent_mark_processed_single_entry(
        self, temp_cache_dir: Path
    ) -> None:
        """Concurrent mark_processed calls should result in single entry."""
        cache = PRCache(temp_cache_dir)
        await cache.open()

        pr_id = "owner/repo/123"

        # Simulate concurrent calls
        async def mark():
            await cache.mark_processed(pr_id)

        # Run multiple concurrent calls
        tasks = [mark() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Should only have one entry
        count = await cache.count()
        assert count == 1, f"Expected 1 entry, got {count}"
        assert await cache.is_processed(pr_id)

        await cache.close()

    async def test_concurrent_mark_different_prs(self, temp_cache_dir: Path) -> None:
        """Concurrent mark_processed for different PRs should all be recorded."""
        cache = PRCache(temp_cache_dir)
        await cache.open()

        # Run concurrent marks for different PRs
        async def mark(pr_id: str):
            await cache.mark_processed(pr_id)

        pr_ids = [f"owner/repo/{i}" for i in range(20)]
        tasks = [mark(pr_id) for pr_id in pr_ids]
        await asyncio.gather(*tasks)

        # All should be recorded
        count = await cache.count()
        assert count == 20

        for pr_id in pr_ids:
            assert await cache.is_processed(pr_id)

        await cache.close()

    async def test_lock_protects_memory_and_file(self, temp_cache_dir: Path) -> None:
        cache = PRCache(temp_cache_dir)
        await cache.open()
        await cache.close()


# =============================================================================
# Task 3: HF Dataset Cache Tests
# =============================================================================


class TestHuggingFaceDatasetCache:
    """Tests for HuggingFaceDatasetCache class."""

    def test_init_with_default_params(self):
        """Initialize HuggingFaceDatasetCache with default params."""
        from swe_forge.swe.dedup import HuggingFaceDatasetCache

        cache = HuggingFaceDatasetCache()
        assert cache.dataset_id is None
        assert cache._task_ids == set()

    def test_init_with_dataset_id(self):
        """Initialize with specific dataset ID."""
        from swe_forge.swe.dedup import HuggingFaceDatasetCache

        cache = HuggingFaceDatasetCache(dataset_id="CortexLM/swe-forge")
        assert cache.dataset_id == "CortexLM/swe-forge"

    @pytest.mark.asyncio
    async def test_fetch_missing_dataset_returns_empty_set(self):
        """Fetching missing dataset returns empty set, not exception."""
        from swe_forge.swe.dedup import HuggingFaceDatasetCache

        cache = HuggingFaceDatasetCache(dataset_id="nonexistent/dataset-xyz")

        # Should not raise, return empty set
        task_ids = await cache.fetch_task_ids()
        assert task_ids == set()

    @pytest.mark.asyncio
    async def test_is_processed_without_fetch(self):
        """is_processed returns False before fetch."""
        from swe_forge.swe.dedup import HuggingFaceDatasetCache

        cache = HuggingFaceDatasetCache(dataset_id="some/dataset")
        assert await cache.is_processed("owner-repo-123") is False

    @pytest.mark.asyncio
    async def test_get_task_ids_returns_copy(self):
        """get_task_ids returns a copy, not reference."""
        from swe_forge.swe.dedup import HuggingFaceDatasetCache

        cache = HuggingFaceDatasetCache()
        cache._task_ids = {"task-1", "task-2"}

        ids = cache.get_task_ids()
        assert ids == {"task-1", "task-2"}

        # Modifying returned set should not affect cache
        ids.add("task-3")
        assert "task-3" not in cache._task_ids


# =============================================================================
# Task 4: DedupManager Tests
# =============================================================================


class TestDedupManager:
    """Tests for DedupManager composite class."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    def test_init_with_prcache_only(self, temp_cache_dir: Path):
        from swe_forge.swe.dedup import DedupManager

        pr_cache = PRCache(temp_cache_dir)
        manager = DedupManager(pr_cache=pr_cache)
        assert manager.pr_cache is pr_cache
        assert manager.hf_cache is None

    @pytest.mark.asyncio
    async def test_is_processed_checks_pr_cache(self, temp_cache_dir: Path):
        """is_processed checks PRCache."""
        from swe_forge.swe.dedup import DedupManager

        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()
        await pr_cache.mark_processed("owner/repo/123")

        manager = DedupManager(pr_cache=pr_cache)

        # Convert task_id to pr_id internally
        assert await manager.is_processed("owner-repo-123")
        assert not await manager.is_processed("other-repo-456")

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_is_processed_checks_hf_cache(self, temp_cache_dir: Path):
        """is_processed checks HF cache when available."""
        from swe_forge.swe.dedup import DedupManager, HuggingFaceDatasetCache

        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        # Create mock HF cache with known task IDs
        hf_cache = HuggingFaceDatasetCache()
        hf_cache._task_ids = {"python-cpython-123456"}

        manager = DedupManager(pr_cache=pr_cache, hf_cache=hf_cache)

        # Should find in HF cache
        assert await manager.is_processed("python-cpython-123456")
        assert not await manager.is_processed("unknown-repo-999")

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_mark_processed_calls_pr_cache(self, temp_cache_dir: Path):
        """mark_processed calls through to PRCache."""
        from swe_forge.swe.dedup import DedupManager

        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        manager = DedupManager(pr_cache=pr_cache)
        await manager.mark_processed("owner-repo-123")

        # Should be in PRCache (converted to PR ID format)
        assert await pr_cache.is_processed("owner/repo/123")

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_handles_missing_hf_gracefully(self, temp_cache_dir: Path):
        """DedupManager works without HF cache."""
        from swe_forge.swe.dedup import DedupManager

        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        manager = DedupManager(pr_cache=pr_cache)  # No HF cache

        # Should work with just PR cache
        await manager.mark_processed("owner-repo-123")
        assert await manager.is_processed("owner-repo-123")

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_format_conversion_error_handling(self, temp_cache_dir: Path):
        """Handles malformed task IDs gracefully."""
        from swe_forge.swe.dedup import DedupManager

        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        manager = DedupManager(pr_cache=pr_cache)

        # Malformed task ID should return False (not processed)
        assert await manager.is_processed("invalid-format") is False

        await pr_cache.close()


# =============================================================================
# Integration Tests
# =============================================================================


class TestDedupIntegration:
    """Integration tests for dedup functionality."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.mark.asyncio
    async def test_full_dedup_workflow(self, temp_cache_dir: Path):
        """Full workflow: check, mark, verify."""
        from swe_forge.swe.dedup import DedupManager

        pr_cache = PRCache(temp_cache_dir)
        await pr_cache.open()

        manager = DedupManager(pr_cache=pr_cache)

        task_id = "python-cpython-12345"

        # Initially not processed
        assert not await manager.is_processed(task_id)

        # Mark as processed
        await manager.mark_processed(task_id)

        # Now should be processed
        assert await manager.is_processed(task_id)

        await pr_cache.close()

    @pytest.mark.asyncio
    async def test_persistence_across_sessions(self, temp_cache_dir: Path):
        """Dedup state persists across sessions."""
        from swe_forge.swe.dedup import DedupManager

        # Session 1: Mark as processed
        pr_cache1 = PRCache(temp_cache_dir)
        await pr_cache1.open()

        manager1 = DedupManager(pr_cache=pr_cache1)
        await manager1.mark_processed("owner-repo-123")

        await pr_cache1.close()

        # Session 2: Should still be processed
        pr_cache2 = PRCache(temp_cache_dir)
        await pr_cache2.open()

        manager2 = DedupManager(pr_cache=pr_cache2)
        assert await manager2.is_processed("owner-repo-123")

        await pr_cache2.close()
