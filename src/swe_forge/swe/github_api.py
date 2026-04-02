"""GitHub API client for fetching PR data.

Provides async methods to interact with GitHub's REST API for:
- Fetching PR details
- Fetching PR file changes
- Fetching PR diffs

Includes rate limit handling with automatic wait and retry logic.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
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


class DiffTooLargeError(GitHubApiError):
    """Raised when diff is too large (406 - diff exceeded limits)."""

    def __init__(self, message: str = "Diff too large") -> None:
        super().__init__(message, 406)


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
    stars: int = 0

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
            stars=data.get("base", {}).get("repo", {}).get("stargazers_count", 0),
        )


@dataclass
class RepoInfo:
    """Repository info from GitHub API."""

    id: int
    name: str
    owner: str
    full_name: str
    description: str
    stars: int
    language: str | None
    default_branch: str
    created_at: str
    updated_at: str

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "RepoInfo":
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            owner=data.get("owner", {}).get("login", ""),
            full_name=data.get("full_name", ""),
            description=data.get("description", "") or "",
            stars=data.get("stargazers_count", 0),
            language=data.get("language"),
            default_branch=data.get("default_branch", "main"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
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
        self._rate_limit_remaining: int = 5000  # Track remaining API calls

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

            if response.status == 406:
                raise DiffTooLargeError(f"Diff too large: {text}")

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

    async def get_pr_diff_via_sparse_fetch(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        base_sha: str,
    ) -> str:
        """Fetch PR diff using sparse git fetch (downloads only needed commits).

        Uses git's partial clone feature to fetch only the PR head and base commits,
        not the entire repository history. This is much faster for large repos.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            base_sha: Base commit SHA to diff against.

        Returns:
            Unified diff as a string.
        """
        repo_url = f"https://github.com/{owner}/{repo}.git"
        tmpdir = tempfile.mkdtemp(prefix=f"swe-sparse-{owner}-{repo}-{pr_number}-")

        try:
            logger.info(
                "sparse_fetch_started",
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                tmpdir=tmpdir,
            )

            # Set git config for faster operations
            env = {
                **os.environ,
                "GIT_HTTP_LOW_SPEED_LIMIT": "1000",
                "GIT_HTTP_LOW_SPEED_TIME": "30",
            }

            # Init empty repo
            subprocess.run(
                ["git", "init"],
                cwd=tmpdir,
                check=True,
                timeout=10,
                capture_output=True,
            )

            # Add remote
            subprocess.run(
                ["git", "remote", "add", "origin", repo_url],
                cwd=tmpdir,
                check=True,
                timeout=10,
                capture_output=True,
            )

            # Fetch PR head (shallow, no blobs initially)
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--depth=1",
                    "--filter=blob:none",
                    "origin",
                    f"refs/pull/{pr_number}/head:pr-head",
                ],
                cwd=tmpdir,
                check=True,
                timeout=30,
                capture_output=True,
                env=env,
            )

            # Fetch base commit (shallow, no blobs initially)
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--depth=1",
                    "--filter=blob:none",
                    "origin",
                    f"{base_sha}:base",
                ],
                cwd=tmpdir,
                check=True,
                timeout=30,
                capture_output=True,
                env=env,
            )

            # Get diff (blobs are fetched lazily only for changed files)
            result = subprocess.run(
                ["git", "diff", "base", "pr-head"],
                cwd=tmpdir,
                capture_output=True,
                check=True,
                timeout=60,
            )

            diff = result.stdout.decode("utf-8", errors="replace")

            logger.info(
                "sparse_fetch_success",
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                diff_size=len(diff),
            )

            return diff

        except subprocess.TimeoutExpired as e:
            logger.warning(
                "sparse_fetch_timeout",
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                error=str(e),
            )
            raise GitHubApiError(f"Sparse fetch timeout: {e}") from e
        except subprocess.CalledProcessError as e:
            logger.warning(
                "sparse_fetch_failed",
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                stderr=e.stderr.decode("utf-8", errors="replace") if e.stderr else "",
            )
            raise GitHubApiError(
                f"Sparse fetch failed: {e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)}"
            ) from e
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    async def get_pr_diff_via_api_compare(
        self,
        owner: str,
        repo: str,
        base_sha: str,
        head_sha: str,
    ) -> str | None:
        """Get diff using GitHub's compare API (fallback).

        Uses GitHub's REST API compare endpoint. Consumes rate limit.

        Args:
            owner: Repository owner.
            repo: Repository name.
            base_sha: Base commit SHA.
            head_sha: Head commit SHA.

        Returns:
            Unified diff string, or None if rate limited or error.
        """
        # Check rate limit
        if self._rate_limit_remaining < 100:
            logger.warning("rate_limit_low", remaining=self._rate_limit_remaining)
            return None

        url = f"{self.BASE_URL}/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"
        headers = self._headers()

        try:
            import json

            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 403:
                    logger.warning("api_rate_limited", owner=owner, repo=repo)
                    return None

                if resp.status != 200:
                    logger.warning(
                        "api_compare_error",
                        owner=owner,
                        repo=repo,
                        status=resp.status,
                    )
                    return None

                text = await resp.text()
                data = json.loads(text)

                # Build unified diff from files array
                diff_lines = []
                for file in data.get("files", []):
                    filename = file.get("filename", "")
                    diff_lines.append(f"diff --git a/{filename} b/{filename}")

                    if "patch" in file:
                        diff_lines.append(file["patch"])

                diff = "\n".join(diff_lines)

                logger.info(
                    "api_compare_success",
                    owner=owner,
                    repo=repo,
                    diff_size=len(diff),
                )

                # Update rate limit counter
                self._rate_limit_remaining -= 1

                return diff

        except Exception as e:
            logger.warning("api_compare_failed", owner=owner, repo=repo, error=str(e))
            return None

    async def get_pr_diff_via_git(
        self,
        owner: str,
        repo: str,
        number: int,
    ) -> str:
        """Fetch PR diff using git clone (fallback for large diffs).

        Used when GitHub API returns 406 "diff too large".
        Performs a shallow clone and fetches the PR as a branch.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Pull request number.

        Returns:
            Unified diff as a string.
        """
        repo_url = f"https://github.com/{owner}/{repo}.git"
        tmpdir = tempfile.mkdtemp(prefix=f"swe-forge-{owner}-{repo}-{number}-")

        try:
            logger.info(
                "git_clone_fallback_started",
                owner=owner,
                repo=repo,
                pr_number=number,
                tmpdir=tmpdir,
            )

            subprocess.run(
                ["git", "clone", "--depth=1", "--filter=blob:none", repo_url, tmpdir],
                capture_output=True,
                check=True,
                timeout=180,
            )

            subprocess.run(
                ["git", "fetch", "origin", f"pull/{number}/head:pr-{number}"],
                capture_output=True,
                check=True,
                timeout=180,
                cwd=tmpdir,
            )

            result = subprocess.run(
                ["git", "diff", "HEAD", f"pr-{number}"],
                capture_output=True,
                check=True,
                timeout=180,
                cwd=tmpdir,
            )

            diff = result.stdout.decode("utf-8", errors="replace")
            logger.info(
                "git_clone_fallback_success",
                owner=owner,
                repo=repo,
                pr_number=number,
                diff_size=len(diff),
            )
            return diff

        except subprocess.TimeoutExpired as e:
            logger.error(
                "git_clone_fallback_timeout",
                owner=owner,
                repo=repo,
                pr_number=number,
                error=str(e),
            )
            raise GitHubApiError(f"Git clone timeout: {e}") from e
        except subprocess.CalledProcessError as e:
            logger.error(
                "git_clone_fallback_failed",
                owner=owner,
                repo=repo,
                pr_number=number,
                stderr=e.stderr.decode("utf-8", errors="replace") if e.stderr else "",
            )
            raise GitHubApiError(
                f"Git clone failed: {e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)}"
            ) from e
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    @retry(
        retry=retry_if_exception_type(
            (aiohttp.ClientError, ServerError, RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
    ) -> str | None:
        """Fetch file content from a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: File path in the repository.
            ref: Git ref (branch, tag, or commit). Defaults to default branch.

        Returns:
            File content as string, or None if file doesn't exist.
        """
        import base64
        import json

        url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}"
        if ref:
            url = f"{url}?ref={ref}"

        try:
            _, _, text = await self._request("GET", url)
            data = json.loads(text)

            if isinstance(data, dict) and data.get("type") == "file":
                content = data.get("content", "")
                if content:
                    return base64.b64decode(content).decode("utf-8", errors="replace")
            return None
        except NotFoundError:
            return None

    async def get_ci_cd_files(
        self,
        owner: str,
        repo: str,
        ref: str | None = None,
    ) -> dict[str, str]:
        """Fetch CI/CD configuration files from a repository.

        Fetches common CI/CD files:
        - .github/workflows/*.yml (GitHub Actions)
        - .gitlab-ci.yml
        - Dockerfile
        - Makefile
        - pyproject.toml
        - setup.py
        - package.json
        - Cargo.toml

        Args:
            owner: Repository owner.
            repo: Repository name.
            ref: Git ref (branch, tag, or commit).

        Returns:
            Dict mapping filename to content for files that exist.
        """
        files: dict[str, str] = {}

        # CI/CD config paths to check
        ci_paths = [
            ".github/workflows/ci.yml",
            ".github/workflows/test.yml",
            ".github/workflows/main.yml",
            ".github/workflows/build.yml",
            ".github/workflows/python-package.yml",
            ".gitlab-ci.yml",
            "Dockerfile",
            "Makefile",
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
        ]

        # Fetch files in parallel
        import asyncio

        async def fetch_file(path: str) -> tuple[str, str | None]:
            content = await self.get_file_content(owner, repo, path, ref)
            return path, content

        results = await asyncio.gather(*[fetch_file(p) for p in ci_paths])

        for path, content in results:
            if content is not None:
                files[path] = content

        return files

    async def get_ci_cd_files_optimized(
        self, owner: str, repo: str, ref: str | None = None
    ) -> dict[str, str]:
        """Get CI/CD files using UNgh (no rate limit).

        Falls back to REST API if UNgh fails.

        Args:
            owner: Repository owner.
            repo: Repository name.
            ref: Git ref (branch, tag, or commit). Defaults to default branch.

        Returns:
            Dict mapping filename to content for files that exist.
        """
        from .ungh_client import UnghClient

        branch = ref or "main"

        try:
            async with UnghClient() as ungh:
                files = await ungh.get_ci_cd_files(owner, repo, branch)
                if files:
                    return files
        except Exception as e:
            logger.warning(f"UNgh failed, falling back to REST API: {e}")

        # Fallback to existing REST API method
        return await self.get_ci_cd_files(owner, repo, ref)

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

    async def get_repo(self, owner: str, repo: str) -> RepoInfo:
        """Fetch repository information.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            RepoInfo object with repository details.

        Raises:
            NotFoundError: If repo doesn't exist.
            GitHubApiError: For other API errors.
        """
        import json

        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        _, _, text = await self._request("GET", url)
        data = json.loads(text)
        return RepoInfo.from_api_response(data)

    async def get_repo_stars(self, owner: str, repo: str) -> int:
        """Fetch repository star count.

        Convenience method that calls get_repo and extracts star count.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            Number of stargazers (0 if error).
        """
        try:
            repo_info = await self.get_repo(owner, repo)
            return repo_info.stars
        except Exception as e:
            logger.warning(f"Failed to get stars for {owner}/{repo}: {e}")
            return 0


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
