"""GH Archive client for fetching GitHub event data.

This module provides an async client for fetching and parsing GH Archive
hourly event files, focusing on merged PullRequestEvents for SWE task mining.
"""

import gzip
import logging
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

GH_ARCHIVE_BASE_URL = "https://data.gharchive.org"


class GhArchiveEvent(BaseModel):
    """Parsed GH Archive event representing a merged PR."""

    id: str
    event_type: str
    repository: str
    actor: str
    action: str
    pull_number: int
    base_sha: str = ""
    merge_sha: str = ""
    title: str = ""
    body: str = ""
    language_hint: str | None = None
    stars: int = 0
    has_org: bool = False
    created_at: datetime
    merged_at: datetime | None = None
    base_ref: str = ""
    head_ref: str = ""
    user: str = ""


class GhArchiveError(Exception):
    """Base error for GH Archive operations."""

    pass


class GhArchiveNotFoundError(GhArchiveError):
    """Raised when a GH Archive hour file is not found (404)."""

    pass


class GhArchiveClient:
    """Async client for fetching GH Archive event data.

    This client fetches hourly event files from GH Archive, decompresses them,
    and parses events focusing on merged PullRequestEvents.

    Attributes:
        token: Optional GitHub token for authentication.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
    """

    def __init__(
        self,
        token: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialize the GH Archive client.

        Args:
            token: Optional GitHub token for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for transient errors.
        """
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "GhArchiveClient":
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure an aiohttp session exists."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_hour_url(self, date: datetime, hour: int) -> str:
        """Build the GH Archive URL for a specific hour.

        Args:
            date: The date for the hour.
            hour: The hour (0-23).

        Returns:
            The full URL to the gzipped JSON file.
        """
        hour_key = f"{date.strftime('%Y-%m-%d')}-{hour}"
        return f"{GH_ARCHIVE_BASE_URL}/{hour_key}.json.gz"

    async def fetch_hour(self, date: datetime, hour: int) -> bytes:
        """Fetch a single hour file from GH Archive.

        Args:
            date: The date to fetch.
            hour: The hour (0-23).

        Returns:
            Decompressed JSON content as bytes.

        Raises:
            GhArchiveNotFoundError: If the hour file doesn't exist (404).
            GhArchiveError: For other HTTP or network errors.
        """
        url = self._build_hour_url(date, hour)
        session = await self._ensure_session()

        headers = {"User-Agent": "swe_forge/1.0"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (aiohttp.ClientError, aiohttp.ServerTimeoutError)
            ),
            reraise=True,
        )

        async def _fetch_with_retry() -> bytes:
            async for attempt in retryer:
                with attempt:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 404:
                            raise GhArchiveNotFoundError(
                                f"GH Archive file not found: {url}"
                            )
                        if response.status != 200:
                            raise GhArchiveError(
                                f"GH Archive returned HTTP {response.status} for {url}"
                            )
                        compressed = await response.read()
                        return gzip.decompress(compressed)

        try:
            return await _fetch_with_retry()
        except GhArchiveNotFoundError:
            # Propagate 404s without retrying
            raise
        except aiohttp.ClientError as e:
            raise GhArchiveError(f"Failed to fetch GH Archive hour file: {e}") from e

    async def fetch_range(
        self,
        start_date: datetime,
        end_date: datetime,
        skip_missing: bool = True,
    ) -> list[bytes]:
        """Fetch multiple hour files from GH Archive.

        Fetches all hours between start_date and end_date (inclusive).

        Args:
            start_date: Start datetime (inclusive).
            end_date: End datetime (inclusive).
            skip_missing: If True, skip missing hours with a warning.
                         If False, raise on missing hours.

        Returns:
            List of decompressed JSON content for each hour.
        """
        results: list[bytes] = []
        current = start_date

        while current <= end_date:
            for hour in range(24):
                hour_dt = current.replace(hour=hour, minute=0, second=0, microsecond=0)
                if hour_dt > end_date:
                    break

                try:
                    content = await self.fetch_hour(current, hour)
                    results.append(content)
                    logger.info(
                        f"Fetched GH Archive hour: {current.strftime('%Y-%m-%d')}-{hour}"
                    )
                except GhArchiveNotFoundError:
                    if skip_missing:
                        logger.warning(
                            f"GH Archive hour file not found, skipping: "
                            f"{current.strftime('%Y-%m-%d')}-{hour}"
                        )
                    else:
                        raise
                except GhArchiveError as e:
                    logger.error(f"Error fetching GH Archive hour: {e}")
                    if not skip_missing:
                        raise

            current = current + timedelta(days=1)

        return results

    def parse_events(self, data: bytes) -> list[dict[str, Any]]:
        """Parse JSON events from decompressed GH Archive data.

        Each line in the file is a separate JSON object.

        Args:
            data: Decompressed JSON content.

        Returns:
            List of parsed event dictionaries.
        """
        import json

        events: list[dict[str, Any]] = []
        for line in data.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                logger.debug(f"Skipping malformed GH Archive line: {e}")
                continue

        return events

    def filter_merged_prs(self, events: list[dict[str, Any]]) -> list[GhArchiveEvent]:
        """Filter events for merged PullRequestEvents only.

        Args:
            events: List of parsed event dictionaries.

        Returns:
            List of GhArchiveEvent objects for merged PRs.
        """
        merged_prs: list[GhArchiveEvent] = []

        for event in events:
            if event.get("type") != "PullRequestEvent":
                continue

            payload = event.get("payload", {})
            if payload.get("action") != "closed":
                continue

            pr = payload.get("pull_request", {})
            if not pr.get("merged", False):
                continue

            parsed = self._parse_pr_event(event)
            if parsed:
                merged_prs.append(parsed)

        return merged_prs

    def _parse_pr_event(self, event: dict[str, Any]) -> GhArchiveEvent | None:
        """Parse a PullRequestEvent into a GhArchiveEvent.

        Args:
            event: Raw event dictionary.

        Returns:
            Parsed GhArchiveEvent or None if parsing fails.
        """
        try:
            payload = event.get("payload", {})
            pr = payload.get("pull_request", {})
            repo = event.get("repo", {})
            actor = event.get("actor", {})

            # Parse created_at
            created_at_str = event.get("created_at", "")
            created_at = self._parse_datetime(created_at_str)

            # Parse merged_at
            merged_at_str = pr.get("merged_at")
            merged_at = self._parse_datetime(merged_at_str) if merged_at_str else None

            # Get repository info from payload if available
            payload_repo = payload.get("repository", {})
            stars = payload_repo.get("watchers_count", 0) or 0

            # Get language from head repo
            head_repo = pr.get("head", {}).get("repo", {})
            language_hint = head_repo.get("language")
            if language_hint:
                language_hint = language_hint.lower()

            return GhArchiveEvent(
                id=f"evt-{event.get('id', '')}",
                event_type="PullRequestEvent",
                repository=repo.get("name", "unknown/repo"),
                actor=actor.get("login", "unknown"),
                action="merged",
                pull_number=pr.get("number", 0),
                base_sha=pr.get("base", {}).get("sha", ""),
                merge_sha=pr.get("merge_commit_sha", ""),
                title=pr.get("title", "Untitled change"),
                body=pr.get("body", "") or "",
                language_hint=language_hint,
                stars=int(stars),
                has_org="org" in event,
                created_at=created_at,
                merged_at=merged_at,
                base_ref=pr.get("base", {}).get("ref", ""),
                head_ref=pr.get("head", {}).get("ref", ""),
                user=pr.get("user", {}).get("login", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to parse PR event: {e}")
            return None

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse an ISO 8601 datetime string.

        Args:
            dt_str: ISO 8601 datetime string.

        Returns:
            Parsed datetime object.
        """
        # Handle various ISO 8601 formats
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return datetime.fromisoformat(dt_str)

    async def fetch_merged_prs(
        self,
        start_date: datetime,
        end_date: datetime,
        skip_missing: bool = True,
    ) -> list[GhArchiveEvent]:
        """Fetch and filter merged PRs from GH Archive.

        Convenience method that combines fetch_range, parse_events, and filter_merged_prs.

        Args:
            start_date: Start datetime (inclusive).
            end_date: End datetime (inclusive).
            skip_missing: If True, skip missing hours with a warning.

        Returns:
            List of GhArchiveEvent objects for merged PRs.
        """
        raw_data_list = await self.fetch_range(start_date, end_date, skip_missing)

        all_events: list[dict[str, Any]] = []
        for raw_data in raw_data_list:
            events = self.parse_events(raw_data)
            all_events.extend(events)

        return self.filter_merged_prs(all_events)
