"""GitHub API client for fetching PR data.

Provides async methods to interact with GitHub's REST API for:
- Fetching PR details
- Fetching PR file changes
- Fetching PR diffs

Includes rate limit handling with automatic wait and retry logic.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger()


class GitHubApiError(Exception):
    """Base exception for GitHub API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(GitHubApiError):
    """Raised when rate limit is exceeded."""

    def __init__(self, reset_time: int) -> None:
        super().__init__(f"Rate limit exceeded, resets at {reset_time}", 403)
        self.reset_time = reset_time


class NotFoundError(GitHubApiError):
    """Raised when a resource is not found (404)."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, 404)


class ForbiddenError(GitHubApiError):
    """Raised for forbidden access (403) not due to rate limits."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 403)


class ServerError(GitHubApiError):
    """Raised for 5xx server errors."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message, status_code)


class PRState(str, Enum):
    """PR state enumeration."""

    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


@dataclass
class PullRequest:
    """Pull request data from GitHub API."""

    number: int
    title: str
    body: str | None
    state: str
    merged: bool
    merged_at: datetime | None
    user_login: str
    base_sha: str
    base_ref: str
    head_sha: str
    head_ref: str
    additions: int
    deletions: int
    changed_files: int

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PullRequest":
        """Create PullRequest from GitHub API response."""
        merged_at = data.get("merged_at")
        return cls(
            number=data["number"],
            title=data.get("title", ""),
            body=data.get("body"),
            state=data.get("state", "open"),
            merged=data.get("merged", False),
            merged_at=datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
            if merged_at
            else None,
            user_login=data.get("user", {}).get("login", "unknown"),
            base_sha=data.get("base", {}).get("sha", ""),
            base_ref=data.get("base", {}).get("ref", ""),
            head_sha=data.get("head", {}).get("sha", ""),
            head_ref=data.get("head", {}).get("ref", ""),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            changed_files=data.get("changed_files", 0),
        )


@dataclass
class PRFile:
    """Changed file in a pull request."""

    filename: str
    status: str  # added, modified, removed, renamed
    additions: int
    deletions: int
    changes: int
    raw_url: str | None = None
    blob_url: str | None = None
    patch: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "PRFile":
        """Create PRFile from GitHub API response."""
        return cls(
            filename=data["filename"],
            status=data.get("status", "modified"),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            changes=data.get("changes", 0),
            raw_url=data.get("raw_url"),
            blob_url=data.get("blob_url"),
            patch=data.get("patch"),
        )


@dataclass
class RateLimitInfo:
    """GitHub API rate limit information."""

    limit: int
    remaining: int
    reset_time: int
    used: int

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "RateLimitInfo":
        """Create RateLimitInfo from response headers."""
        return cls(
            limit=int(headers.get("x-ratelimit-limit", "5000")),
            remaining=int(headers.get("x-ratelimit-remaining", "5000")),
            reset_time=int(headers.get("x-ratelimit-reset", "0")),
            used=int(headers.get("x-ratelimit-used", "0")),
        )


class GitHubClient:
    """Async GitHub API client with rate limit handling.

    Usage:
        async with GitHubClient(token) as client:
            pr = await client.get_pr("owner", "repo", 123)
            files = await client.get_pr_files("owner", "repo", 123)
            diff = await client.get_pr_diff("owner", "repo", 123)
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, timeout: float = 30.0) -> None:
        """Initialize the GitHub client.

        Args:
            token: GitHub personal access token.
            timeout: Request timeout in seconds.
        """
        self.token = token
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._rate_limit_info: RateLimitInfo | None = None

    def _headers(
        self, accept: str = "application/vnd.github.v3+json"
    ) -> dict[str, str]:
        """Build request headers."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": accept,
            "User-Agent": "swe-forge/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def __aenter__(self) -> "GitHubClient":
        """Create aiohttp session on context entry."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self._headers(),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close aiohttp session on context exit."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get the current session or raise an error."""
        if not self._session:
            raise RuntimeError("GitHubClient must be used as async context manager")
        return self._session

    async def _update_rate_limit(self, headers: dict[str, str]) -> None:
        """Update rate limit info from response headers."""
        try:
            self._rate_limit_info = RateLimitInfo.from_headers(headers)
        except (ValueError, KeyError):
            pass

    async def _handle_rate_limit(self, headers: dict[str, str]) -> None:
        """Handle rate limit by waiting until reset."""
        remaining = int(headers.get("x-ratelimit-remaining", "1"))
        if remaining == 0:
            reset_time = int(headers.get("x-ratelimit-reset", "0"))
            if reset_time:
                wait_seconds = max(0, reset_time - time.time() + 1)
                logger.warning(
                    "rate_limit_exceeded",
                    reset_time=reset_time,
                    wait_seconds=wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                raise RateLimitError(reset_time)

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, dict[str, str], str]:
        """Make an HTTP request and handle common errors.

        Returns:
            Tuple of (status_code, response_headers, response_text).

        Raises:
            NotFoundError: For 404 responses.
            ForbiddenError: For 403 responses (non-rate-limit).
            ServerError: For 5xx responses.
            GitHubApiError: For other error responses.
        """
        session = self._get_session()
        request_headers = headers or self._headers()

        async with session.request(
            method, url, headers=request_headers, **kwargs
        ) as response:
            response_headers = dict(response.headers)
            await self._update_rate_limit(response_headers)

            text = await response.text()

            if response.status == 200:
                return response.status, response_headers, text

            if response.status == 404:
                raise NotFoundError(f"Resource not found: {url}")

            if response.status == 403:
                await self._handle_rate_limit(response_headers)
                raise ForbiddenError(f"Forbidden: {url}")

            if response.status >= 500:
                raise ServerError(response.status, f"Server error: {text}")

            raise GitHubApiError(f"HTTP {response.status}: {text}", response.status)

    @retry(
        retry=retry_if_exception_type(
            (aiohttp.ClientError, ServerError, RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_pr(self, owner: str, repo: str, number: int) -> PullRequest:
        """Fetch a pull request by number.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Pull request number.

        Returns:
            PullRequest object with PR details.

        Raises:
            NotFoundError: If PR doesn't exist.
            GitHubApiError: For other API errors.
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{number}"
        _, _, text = await self._request("GET", url)

        import json

        data = json.loads(text)
        return PullRequest.from_api_response(data)

    @retry(
        retry=retry_if_exception_type(
            (aiohttp.ClientError, ServerError, RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_pr_files(
        self, owner: str, repo: str, number: int, per_page: int = 100
    ) -> list[PRFile]:
        """Fetch files changed in a pull request.

        GitHub API paginates this endpoint at 100 items per page.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Pull request number.
            per_page: Items per page (max 100).

        Returns:
            List of PRFile objects.

        Raises:
            NotFoundError: If PR doesn't exist.
            GitHubApiError: For other API errors.
        """
        import json

        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{number}/files"
        all_files: list[PRFile] = []
        page = 1

        while True:
            _, _, text = await self._request(
                "GET",
                url,
                params={"per_page": str(min(per_page, 100)), "page": str(page)},
            )

            items = json.loads(text)
            if not items:
                break

            for item in items:
                all_files.append(PRFile.from_api_response(item))

            if len(items) < per_page:
                break

            page += 1

        return all_files

    @retry(
        retry=retry_if_exception_type(
            (aiohttp.ClientError, ServerError, RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_pr_diff(self, owner: str, repo: str, number: int) -> str:
        """Fetch the unified diff for a pull request.

        Uses Accept: application/vnd.github.v3.diff header.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Pull request number.

        Returns:
            Unified diff as a string.

        Raises:
            NotFoundError: If PR doesn't exist.
            GitHubApiError: For other API errors.
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/pulls/{number}"
        headers = self._headers(accept="application/vnd.github.v3.diff")

        _, _, text = await self._request("GET", url, headers=headers)
        return text

    async def get_rate_limit(self) -> RateLimitInfo:
        """Fetch current rate limit status.

        Returns:
            RateLimitInfo with current limits.
        """
        import json

        url = f"{self.BASE_URL}/rate_limit"
        _, _, text = await self._request("GET", url)

        data = json.loads(text)
        resources = data.get("resources", {})
        core = resources.get("core", {})

        return RateLimitInfo(
            limit=core.get("limit", 5000),
            remaining=core.get("remaining", 5000),
            reset_time=core.get("reset", 0),
            used=core.get("used", 0),
        )

    @property
    def rate_limit_info(self) -> RateLimitInfo | None:
        """Get cached rate limit info from last request."""
        return self._rate_limit_info


async def create_client(token: str | None = None) -> GitHubClient:
    """Create a GitHub client, loading token from environment if not provided.

    Args:
        token: GitHub token, or None to load from GITHUB_TOKEN env var.

    Returns:
        Initialized GitHubClient (caller must close it).

    Raises:
        ValueError: If no token provided and GITHUB_TOKEN not set.
    """
    import os

    if not token:
        token = os.environ.get("GITHUB_TOKEN")

    if not token:
        raise ValueError(
            "GitHub token required. Set GITHUB_TOKEN env var or pass token."
        )

    return GitHubClient(token)
