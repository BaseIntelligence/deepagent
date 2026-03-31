"""PR cache using JSONL append-only storage with optional SQLite backend.

Supports:
- JSONL file for simple append-only storage
- Optional SQLite backend for richer queries
- Resume from interrupted runs (reads existing cache on init)
- Deduplication on read
- Graceful handling of corrupted files
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiosqlite

logger = logging.getLogger(__name__)

DEFAULT_JSONL_FILENAME = "cache.jsonl"
DEFAULT_SQLITE_FILENAME = "cache.db"


@dataclass
class PRMetadata:
    """Metadata for a processed PR."""

    pr_id: str  # Format: "owner/repo/number" or "repo/number"
    processed_at: str
    status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)


class PRCache:
    """Async PR cache with JSONL and optional SQLite backend.

    Uses JSONL for simple append-only storage and optionally SQLite
    for structured queries. On initialization, loads existing PR IDs
    into memory for fast deduplication checks.

    Example:
        cache = PRCache("./cache")
        await cache.open()

        if not await cache.is_processed("owner/repo/123"):
            # process PR...
            await cache.mark_processed("owner/repo/123", {"status": "success"})

        await cache.close()
    """

    def __init__(
        self,
        cache_dir: str | Path,
        use_sqlite: bool = False,
        jsonl_filename: str = DEFAULT_JSONL_FILENAME,
        sqlite_filename: str = DEFAULT_SQLITE_FILENAME,
    ):
        self.cache_dir = Path(cache_dir)
        self.use_sqlite = use_sqlite
        self.jsonl_filename = jsonl_filename
        self.sqlite_filename = sqlite_filename

        self._processed_ids: set[str] = set()
        self._jsonl_path = self.cache_dir / self.jsonl_filename
        self._sqlite_path = self.cache_dir / self.sqlite_filename
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def jsonl_path(self) -> Path:
        """Path to JSONL cache file."""
        return self._jsonl_path

    @property
    def sqlite_path(self) -> Path:
        """Path to SQLite database file."""
        return self._sqlite_path

    async def open(self) -> None:
        """Open the cache and load existing entries.

        Creates cache directory if needed. Loads existing PR IDs into
        memory for fast deduplication. Initializes SQLite if enabled.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing entries from JSONL
        await self._load_existing_entries()

        # Initialize SQLite if enabled
        if self.use_sqlite:
            await self._init_sqlite()

        self._initialized = True
        logger.info(
            f"PR cache opened: {len(self._processed_ids)} entries loaded, "
            f"sqlite={self.use_sqlite}"
        )

    async def _load_existing_entries(self) -> None:
        """Load existing PR IDs from JSONL file into memory.

        Handles corrupted files gracefully by skipping bad lines.
        """
        if not self._jsonl_path.exists():
            return

        try:
            async with aiofiles.open(self._jsonl_path, mode="r") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pr_id = data.get("id")
                        if pr_id:
                            self._processed_ids.add(pr_id)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping corrupted cache line: {e}")
                        continue
        except OSError as e:
            logger.warning(f"Failed to read cache file: {e}")

    async def _init_sqlite(self) -> None:
        """Initialize SQLite database with schema."""
        self._db = await aiosqlite.connect(self._sqlite_path)
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_prs (
                id TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL,
                status TEXT,
                metadata TEXT
            )
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        """Close the cache, flushing any pending data."""
        if self._db:
            await self._db.close()
            self._db = None
        self._initialized = False

    async def __aenter__(self) -> "PRCache":
        """Async context manager entry."""
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def is_initialized(self) -> bool:
        """Check if cache is initialized."""
        return self._initialized

    async def is_processed(self, pr_id: str) -> bool:
        """Check if a PR has already been processed.

        Uses in-memory set for fast lookup.

        Args:
            pr_id: PR identifier (format: "owner/repo/number" or "repo/number")

        Returns:
            True if PR was already processed, False otherwise
        """
        if not self._initialized:
            logger.warning("Cache not initialized, returning False")
            return False
        return pr_id in self._processed_ids

    async def mark_processed(
        self,
        pr_id: str,
        metadata: Optional[dict[str, Any]] = None,
        status: str = "success",
    ) -> None:
        if not self._initialized:
            raise RuntimeError("Cache not initialized. Call open() first.")

        processed_at = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": pr_id,
            "processed_at": processed_at,
            "status": status,
            **(metadata or {}),
        }

        async with self._lock:
            await self._append_to_jsonl(entry)
            self._processed_ids.add(pr_id)

            if self._db:
                await self._upsert_sqlite(pr_id, processed_at, status, metadata)

    async def _append_to_jsonl(self, entry: dict[str, Any]) -> None:
        """Append entry to JSONL file."""
        line = json.dumps(entry, separators=(",", ":"))
        async with aiofiles.open(self._jsonl_path, mode="a") as f:
            await f.write(line + "\n")

    async def _upsert_sqlite(
        self,
        pr_id: str,
        processed_at: str,
        status: str,
        metadata: Optional[dict[str, Any]],
    ) -> None:
        """Upsert entry into SQLite database."""
        if not self._db:
            return

        metadata_json = json.dumps(metadata) if metadata else None
        await self._db.execute(
            """
            INSERT INTO processed_prs (id, processed_at, status, metadata)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                processed_at = excluded.processed_at,
                status = excluded.status,
                metadata = excluded.metadata
            """,
            (pr_id, processed_at, status, metadata_json),
        )
        await self._db.commit()

    async def get_all_processed(self) -> set[str]:
        """Get set of all processed PR IDs.

        Returns:
            Set of PR identifiers that have been processed
        """
        return self._processed_ids.copy()

    async def get_metadata(self, pr_id: str) -> Optional[dict[str, Any]]:
        """Get metadata for a specific PR from SQLite.

        Args:
            pr_id: PR identifier

        Returns:
            Metadata dict if found, None otherwise

        Note:
            Only works if SQLite backend is enabled. Falls back to
            reading from JSONL if SQLite is not available.
        """
        if self._db:
            async with self._db.execute(
                "SELECT processed_at, status, metadata FROM processed_prs WHERE id = ?",
                (pr_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "processed_at": row[0],
                        "status": row[1],
                        "metadata": json.loads(row[2]) if row[2] else {},
                    }
            return None

        # Fallback: read from JSONL (slower)
        return await self._read_metadata_from_jsonl(pr_id)

    async def _read_metadata_from_jsonl(self, pr_id: str) -> Optional[dict[str, Any]]:
        """Read metadata for a PR from JSONL file."""
        if not self._jsonl_path.exists():
            return None

        try:
            async with aiofiles.open(self._jsonl_path, mode="r") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("id") == pr_id:
                            return data
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return None

    async def count(self) -> int:
        """Get count of processed PRs.

        Returns:
            Number of processed PRs
        """
        return len(self._processed_ids)

    async def clear(self) -> None:
        """Clear the cache by deleting all entries.

        Warning: This does not delete the files, only clears the in-memory
        state. Use clear_all_files() to delete files too.
        """
        self._processed_ids.clear()

        if self._db:
            await self._db.execute("DELETE FROM processed_prs")
            await self._db.commit()

    async def clear_all_files(self) -> None:
        """Delete cache files and clear in-memory state.

        This completely resets the cache.
        """
        await self.clear()

        if self._jsonl_path.exists():
            self._jsonl_path.unlink()

        if self.use_sqlite and self._sqlite_path.exists():
            self._sqlite_path.unlink()


class OptionalPRCache:
    """Wrapper for optional PRCache that works with or without caching.

    Similar to Rust's OptionalCache - allows pipeline to work
    regardless of whether caching is enabled.
    """

    def __init__(self, cache: Optional[PRCache] = None):
        """Initialize with optional cache.

        Args:
            cache: Optional PRCache instance
        """
        self._cache = cache

    @classmethod
    def none(cls) -> "OptionalPRCache":
        """Create an OptionalPRCache with no caching."""
        return cls(None)

    @classmethod
    def some(cls, cache: PRCache) -> "OptionalPRCache":
        """Create an OptionalPRCache with caching enabled."""
        return cls(cache)

    async def is_processed(self, pr_id: str) -> bool:
        """Check if PR is processed (returns False if no cache)."""
        if self._cache:
            return await self._cache.is_processed(pr_id)
        return False

    async def mark_processed(
        self,
        pr_id: str,
        metadata: Optional[dict[str, Any]] = None,
        status: str = "success",
    ) -> None:
        """Mark PR as processed (no-op if no cache)."""
        if self._cache:
            await self._cache.mark_processed(pr_id, metadata, status)

    async def get_all_processed(self) -> set[str]:
        """Get all processed PR IDs (returns empty set if no cache)."""
        if self._cache:
            return await self._cache.get_all_processed()
        return set()

    def __repr__(self) -> str:
        if self._cache:
            return f"OptionalPRCache(active, {len(self._cache._processed_ids)} entries)"
        return "OptionalPRCache(none)"
