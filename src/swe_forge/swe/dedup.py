"""Deduplication module for SWE mining pipeline.

Provides:
- Format conversion between task IDs and PR IDs
- HuggingFace dataset cache for remote dedup
- DedupManager composite for unified dedup interface
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swe_forge.swe.pr_cache import PRCache

logger = logging.getLogger(__name__)


def task_id_to_pr_id(task_id: str) -> str | None:
    """Convert task ID format to PR ID format.

    Task ID format: "owner-repo-123" (dashes, PR number at end)
    PR ID format: "owner/repo/123" (slashes, PR number at end)

    The conversion handles repos with dashes by parsing from the end:
    - The last segment (digits) is the PR number
    - The first segment is the owner
    - Everything in between is the repo name

    Args:
        task_id: Task ID in format "owner-repo-number"

    Returns:
        PR ID in format "owner/repo/number", or None if malformed
    """
    if not task_id:
        return None

    if "-" not in task_id:
        return None

    parts = task_id.split("-")

    # Need at least 3 parts: owner, repo, number
    if len(parts) < 3:
        return None

    # Last part must be a number (PR number)
    number = parts[-1]
    if not number.isdigit():
        return None

    # First part is owner, rest (except last) is repo
    owner = parts[0]
    repo = "-".join(parts[1:-1])

    if not owner or not repo:
        return None

    return f"{owner}/{repo}/{number}"


def pr_id_to_task_id(pr_id: str) -> str | None:
    """Convert PR ID format to task ID format.

    PR ID format: "owner/repo/123" (slashes)
    Task ID format: "owner-repo-123" (dashes)

    Args:
        pr_id: PR ID in format "owner/repo/number"

    Returns:
        Task ID in format "owner-repo-number", or None if malformed
    """
    if not pr_id:
        return None

    parts = pr_id.split("/")

    # Need exactly 3 parts: owner, repo, number
    if len(parts) != 3:
        return None

    owner, repo, number = parts

    if not owner or not repo or not number:
        return None

    return f"{owner}-{repo}-{number}"


@dataclass
class HuggingFaceDatasetCache:
    """Cache for task IDs from HuggingFace datasets.

    Fetches task IDs from a HF dataset at startup for dedup checking.
    Uses streaming mode to handle large datasets.

    Attributes:
        dataset_id: HuggingFace dataset ID (e.g., "CortexLM/swe-forge")
        _task_ids: Set of task IDs from the dataset
    """

    dataset_id: str | None = None
    _task_ids: set[str] = field(default_factory=set, repr=False)

    async def fetch_task_ids(self) -> set[str]:
        """Fetch task IDs from the HuggingFace dataset.

        Uses streaming mode and handles missing datasets gracefully.

        Returns:
            Set of task IDs from the dataset (empty if fetch fails)
        """
        if not self.dataset_id:
            return set()

        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning("datasets library not installed, skipping HF fetch")
            return set()

        try:
            # Use streaming for large datasets
            dataset = load_dataset(
                self.dataset_id,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

            task_ids: set[str] = set()
            for row in dataset:
                # Look for task_id, instance_id, or id field
                task_id = row.get("task_id") or row.get("instance_id") or row.get("id")
                if task_id:
                    task_ids.add(str(task_id))

            self._task_ids = task_ids
            logger.info(
                f"Loaded {len(task_ids)} task IDs from HF dataset {self.dataset_id}"
            )
            return task_ids

        except Exception as e:
            logger.warning(f"Failed to fetch HF dataset {self.dataset_id}: {e}")
            return set()

    async def is_processed(self, task_id: str) -> bool:
        """Check if a task ID exists in the HF dataset.

        Args:
            task_id: Task ID to check

        Returns:
            True if task was processed (exists in dataset), False otherwise
        """
        return task_id in self._task_ids

    def get_task_ids(self) -> set[str]:
        """Get a copy of the cached task IDs.

        Returns:
            Copy of the task IDs set
        """
        return self._task_ids.copy()


@dataclass
class DedupManager:
    """Composite deduplication manager.

    Wraps PRCache (local) and HuggingFaceDatasetCache (remote) to provide
    a unified dedup interface. Handles format conversion internally.
    """

    pr_cache: PRCache | None = None
    hf_cache: HuggingFaceDatasetCache | None = None

    async def is_processed(self, task_id: str) -> bool:
        """Check if a task has been processed.

        Checks both local cache and HF cache (if available).
        Handles format conversion internally.

        Args:
            task_id: Task ID in format "owner-repo-number"

        Returns:
            True if task was processed, False otherwise
        """
        if self.pr_cache:
            pr_id = task_id_to_pr_id(task_id)
            if pr_id:
                if await self.pr_cache.is_processed(pr_id):
                    return True

        if self.hf_cache:
            if await self.hf_cache.is_processed(task_id):
                return True

        return False

    async def mark_processed(self, task_id: str) -> None:
        """Mark a task as processed.

        Only marks in local cache (HF cache is read-only).

        Args:
            task_id: Task ID in format "owner-repo-number"
        """
        if self.pr_cache:
            pr_id = task_id_to_pr_id(task_id)
            if pr_id:
                await self.pr_cache.mark_processed(pr_id)
