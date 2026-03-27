"""Tests for PR cache implementation."""

import json
from pathlib import Path

import pytest

from swe_forge.swe.pr_cache import OptionalPRCache, PRCache


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
async def pr_cache(temp_cache_dir: Path) -> PRCache:
    """Create and open a PR cache without SQLite."""
    cache = PRCache(temp_cache_dir)
    await cache.open()
    yield cache
    await cache.close()


@pytest.fixture
async def pr_cache_with_sqlite(temp_cache_dir: Path) -> PRCache:
    """Create and open a PR cache with SQLite backend."""
    cache = PRCache(temp_cache_dir, use_sqlite=True)
    await cache.open()
    yield cache
    await cache.close()


class TestPRCacheBasic:
    """Basic PRCache functionality tests."""

    async def test_open_creates_directory(self, tmp_path: Path) -> None:
        """Opening cache creates the directory if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        cache = PRCache(cache_dir)
        await cache.open()
        assert cache_dir.exists()
        await cache.close()

    async def test_is_processed_returns_false_for_new_pr(
        self, pr_cache: PRCache
    ) -> None:
        """is_processed returns False for a PR not in cache."""
        result = await pr_cache.is_processed("owner/repo/123")
        assert result is False

    async def test_mark_processed_and_is_processed(self, pr_cache: PRCache) -> None:
        """mark_processed adds PR to cache and is_processed returns True."""
        pr_id = "owner/repo/123"
        await pr_cache.mark_processed(pr_id)

        assert await pr_cache.is_processed(pr_id)

    async def test_mark_processed_with_metadata(self, pr_cache: PRCache) -> None:
        """mark_processed stores metadata correctly."""
        pr_id = "owner/repo/456"
        metadata = {"status": "success", "score": 0.85}
        await pr_cache.mark_processed(pr_id, metadata=metadata)

        assert await pr_cache.is_processed(pr_id)

    async def test_get_all_processed(self, pr_cache: PRCache) -> None:
        """get_all_processed returns set of all processed PR IDs."""
        pr_ids = ["owner/repo/1", "owner/repo/2", "other/repo/3"]
        for pr_id in pr_ids:
            await pr_cache.mark_processed(pr_id)

        processed = await pr_cache.get_all_processed()

        assert processed == set(pr_ids)

    async def test_count(self, pr_cache: PRCache) -> None:
        """count returns correct number of processed PRs."""
        assert await pr_cache.count() == 0

        await pr_cache.mark_processed("owner/repo/1")
        assert await pr_cache.count() == 1

        await pr_cache.mark_processed("owner/repo/2")
        assert await pr_cache.count() == 2


class TestPRCacheJSONL:
    """Tests for JSONL file handling."""

    async def test_jsonl_file_created(self, temp_cache_dir: Path) -> None:
        """JSONL file is created in cache directory."""
        cache = PRCache(temp_cache_dir)
        await cache.open()
        await cache.mark_processed("owner/repo/123")
        await cache.close()

        jsonl_path = temp_cache_dir / "cache.jsonl"
        assert jsonl_path.exists()

    async def test_jsonl_format(self, temp_cache_dir: Path) -> None:
        """JSONL file contains valid JSON lines."""
        cache = PRCache(temp_cache_dir)
        await cache.open()
        await cache.mark_processed("owner/repo/123", status="success")
        await cache.close()

        jsonl_path = temp_cache_dir / "cache.jsonl"
        content = jsonl_path.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["id"] == "owner/repo/123"
        assert entry["status"] == "success"
        assert "processed_at" in entry

    async def test_append_mode(self, temp_cache_dir: Path) -> None:
        """Multiple mark_processed calls append to JSONL."""
        cache = PRCache(temp_cache_dir)
        await cache.open()
        await cache.mark_processed("owner/repo/1")
        await cache.mark_processed("owner/repo/2")
        await cache.close()

        jsonl_path = temp_cache_dir / "cache.jsonl"
        content = jsonl_path.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        assert len(lines) == 2

    async def test_resume_from_existing_cache(self, temp_cache_dir: Path) -> None:
        """Cache loads existing entries from JSONL on open."""
        jsonl_path = temp_cache_dir / "cache.jsonl"
        existing_entries = [
            {"id": "owner/repo/100", "processed_at": "2023-01-01T00:00:00Z"},
            {"id": "owner/repo/101", "processed_at": "2023-01-01T00:01:00Z"},
        ]
        with open(jsonl_path, "w") as f:
            for entry in existing_entries:
                f.write(json.dumps(entry) + "\n")

        cache = PRCache(temp_cache_dir)
        await cache.open()

        assert await cache.is_processed("owner/repo/100")
        assert await cache.is_processed("owner/repo/101")
        assert not await cache.is_processed("owner/repo/999")

        await cache.close()

    async def test_handle_corrupted_jsonl(self, temp_cache_dir: Path) -> None:
        """Cache handles corrupted JSONL lines gracefully."""
        jsonl_path = temp_cache_dir / "cache.jsonl"
        content = """
{"id": "owner/repo/1", "processed_at": "2023-01-01T00:00:00Z"}
this is not valid json
{"id": "owner/repo/2", "processed_at": "2023-01-01T00:01:00Z"}
{"broken": missing quotes}
{"id": "owner/repo/3", "processed_at": "2023-01-01T00:02:00Z"}
"""
        jsonl_path.write_text(content)

        cache = PRCache(temp_cache_dir)
        await cache.open()

        processed = await cache.get_all_processed()
        assert "owner/repo/1" in processed
        assert "owner/repo/2" in processed
        assert "owner/repo/3" in processed
        assert len(processed) == 3

        await cache.close()


class TestPRCacheSQLite:
    """Tests for SQLite backend."""

    async def test_sqlite_file_created(self, temp_cache_dir: Path) -> None:
        """SQLite database is created when use_sqlite=True."""
        cache = PRCache(temp_cache_dir, use_sqlite=True)
        await cache.open()
        await cache.mark_processed("owner/repo/123")
        await cache.close()

        sqlite_path = temp_cache_dir / "cache.db"
        assert sqlite_path.exists()

    async def test_get_metadata_from_sqlite(
        self, pr_cache_with_sqlite: PRCache
    ) -> None:
        """get_metadata returns data from SQLite."""
        pr_id = "owner/repo/123"
        await pr_cache_with_sqlite.mark_processed(pr_id, metadata={"score": 0.9})

        metadata = await pr_cache_with_sqlite.get_metadata(pr_id)
        assert metadata is not None
        assert metadata["metadata"]["score"] == 0.9
        assert "processed_at" in metadata

    async def test_get_metadata_none_for_unknown(
        self, pr_cache_with_sqlite: PRCache
    ) -> None:
        """get_metadata returns None for unknown PR."""
        metadata = await pr_cache_with_sqlite.get_metadata("unknown/pr/999")
        assert metadata is None

    async def test_upsert_updates_existing(self, pr_cache_with_sqlite: PRCache) -> None:
        """mark_processed updates existing entry in SQLite."""
        pr_id = "owner/repo/123"
        await pr_cache_with_sqlite.mark_processed(pr_id, status="pending")
        await pr_cache_with_sqlite.mark_processed(pr_id, status="success")

        metadata = await pr_cache_with_sqlite.get_metadata(pr_id)
        assert metadata["status"] == "success"


class TestPRCacheContextManager:
    """Tests for async context manager."""

    async def test_context_manager(self, temp_cache_dir: Path) -> None:
        """PRCache works as async context manager."""
        async with PRCache(temp_cache_dir) as cache:
            assert cache.is_initialized()
            await cache.mark_processed("owner/repo/1")
            assert await cache.is_processed("owner/repo/1")

        assert not cache.is_initialized()


class TestPRCacheClear:
    """Tests for cache clearing."""

    async def test_clear_memory(self, pr_cache: PRCache) -> None:
        """clear removes entries from memory."""
        await pr_cache.mark_processed("owner/repo/1")
        await pr_cache.mark_processed("owner/repo/2")

        await pr_cache.clear()

        assert await pr_cache.count() == 0
        assert not await pr_cache.is_processed("owner/repo/1")

    async def test_clear_all_files(self, pr_cache: PRCache) -> None:
        """clear_all_files deletes cache files."""
        await pr_cache.mark_processed("owner/repo/1")
        jsonl_path = pr_cache.jsonl_path

        assert jsonl_path.exists()

        await pr_cache.clear_all_files()

        assert not jsonl_path.exists()
        assert await pr_cache.count() == 0


class TestOptionalPRCache:
    """Tests for OptionalPRCache wrapper."""

    async def test_none_returns_false_for_is_processed(self) -> None:
        """OptionalPRCache.none() returns False for is_processed."""
        cache = OptionalPRCache.none()
        assert await cache.is_processed("any/pr/123") is False

    async def test_none_no_op_for_mark_processed(self) -> None:
        """OptionalPRCache.none() no-ops for mark_processed."""
        cache = OptionalPRCache.none()
        await cache.mark_processed("any/pr/123")

    async def test_none_empty_set_for_get_all_processed(self) -> None:
        """OptionalPRCache.none() returns empty set."""
        cache = OptionalPRCache.none()
        result = await cache.get_all_processed()
        assert result == set()

    async def test_some_works_like_pr_cache(self, temp_cache_dir: Path) -> None:
        """OptionalPRCache.some() works like regular PRCache."""
        inner_cache = PRCache(temp_cache_dir)
        await inner_cache.open()

        cache = OptionalPRCache.some(inner_cache)

        assert not await cache.is_processed("owner/repo/1")
        await cache.mark_processed("owner/repo/1")
        assert await cache.is_processed("owner/repo/1")

        processed = await cache.get_all_processed()
        assert "owner/repo/1" in processed

        await inner_cache.close()

    def test_repr(self) -> None:
        """OptionalPRCache has helpful repr."""
        none_cache = OptionalPRCache.none()
        assert "none" in repr(none_cache).lower()


class TestPRCacheErrorHandling:
    """Tests for error handling."""

    async def test_is_processed_without_open(self, temp_cache_dir: Path) -> None:
        """is_processed returns False if cache not initialized."""
        cache = PRCache(temp_cache_dir)
        result = await cache.is_processed("owner/repo/1")
        assert result is False

    async def test_mark_processed_without_open_raises(
        self, temp_cache_dir: Path
    ) -> None:
        """mark_processed raises RuntimeError if not initialized."""
        cache = PRCache(temp_cache_dir)
        with pytest.raises(RuntimeError, match="not initialized"):
            await cache.mark_processed("owner/repo/1")


class TestPRCacheConcurrency:
    """Tests for concurrent access scenarios."""

    async def test_multiple_opens_same_directory(self, temp_cache_dir: Path) -> None:
        """Multiple cache instances can share the same directory."""
        cache1 = PRCache(temp_cache_dir)
        await cache1.open()
        await cache1.mark_processed("owner/repo/1")
        await cache1.close()

        cache2 = PRCache(temp_cache_dir)
        await cache2.open()
        assert await cache2.is_processed("owner/repo/1")
        await cache2.close()

    async def test_resume_preserves_all_entries(self, temp_cache_dir: Path) -> None:
        """Resume preserves all entries after multiple sessions."""
        cache = PRCache(temp_cache_dir)

        for i in range(10):
            await cache.open()
            await cache.mark_processed(f"owner/repo/{i}")
            await cache.close()

        cache = PRCache(temp_cache_dir)
        await cache.open()

        for i in range(10):
            assert await cache.is_processed(f"owner/repo/{i}")

        await cache.close()
