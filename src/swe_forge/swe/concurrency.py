"""Concurrency control module with semaphores for pipeline stages."""

import asyncio
from dataclasses import dataclass, field


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency limits across pipeline stages.

    Values match Rust implementation:
    - GH_ARCHIVE_FETCH: Hardcoded at 8 concurrent GH Archive fetches
    - ENRICHMENT: Default concurrency_enrich (20)
    - PRECLASSIFY: Default concurrency_preclassify (25)
    - DEEP_PROCESSING: Default concurrency_deep (8)
    - DEEP_BACKLOG_MULTIPLIER: Multiplier for backlog semaphores (5)
    - GITHUB_SEARCH_AUTH: Rate limit for authenticated GitHub searches (5)
    - GITHUB_SEARCH_UNAUTH: Rate limit for unauthenticated GitHub searches (2)
    - DOCKER_CONTAINERS: Maximum concurrent Docker containers (8)
    """

    GH_ARCHIVE_FETCH: int = 8
    ENRICHMENT: int = 20
    PRECLASSIFY: int = 25
    DEEP_PROCESSING: int = 8
    DEEP_BACKLOG_MULTIPLIER: int = 5
    GITHUB_SEARCH_AUTH: int = 5
    GITHUB_SEARCH_UNAUTH: int = 2
    DOCKER_CONTAINERS: int = 8


@dataclass
class PipelineSemaphores:
    """Pipeline semaphores for controlling concurrency across stages.

    Each semaphore controls access to a specific pipeline stage,
    preventing resource exhaustion and rate limit violations.
    """

    config: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Semaphores initialized post-creation since asyncio.Semaphore needs event loop
    _enrich_sem: asyncio.Semaphore | None = field(default=None, init=False)
    _preclassify_sem: asyncio.Semaphore | None = field(default=None, init=False)
    _deep_sem: asyncio.Semaphore | None = field(default=None, init=False)
    _backlog_sem: asyncio.Semaphore | None = field(default=None, init=False)
    _gh_archive_sem: asyncio.Semaphore | None = field(default=None, init=False)
    _github_search_auth_sem: asyncio.Semaphore | None = field(default=None, init=False)
    _github_search_unauth_sem: asyncio.Semaphore | None = field(
        default=None, init=False
    )
    _docker_sem: asyncio.Semaphore | None = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    def _ensure_initialized(self) -> None:
        """Initialize semaphores on first use (requires event loop)."""
        if not self._initialized:
            self._enrich_sem = asyncio.Semaphore(self.config.ENRICHMENT)
            self._preclassify_sem = asyncio.Semaphore(self.config.PRECLASSIFY)
            self._deep_sem = asyncio.Semaphore(self.config.DEEP_PROCESSING)
            self._backlog_sem = asyncio.Semaphore(
                self.config.DEEP_PROCESSING * self.config.DEEP_BACKLOG_MULTIPLIER
            )
            self._gh_archive_sem = asyncio.Semaphore(self.config.GH_ARCHIVE_FETCH)
            self._github_search_auth_sem = asyncio.Semaphore(
                self.config.GITHUB_SEARCH_AUTH
            )
            self._github_search_unauth_sem = asyncio.Semaphore(
                self.config.GITHUB_SEARCH_UNAUTH
            )
            self._docker_sem = asyncio.Semaphore(self.config.DOCKER_CONTAINERS)
            self._initialized = True

    @property
    def enrich_sem(self) -> asyncio.Semaphore:
        """Semaphore for enrichment stage."""
        self._ensure_initialized()
        assert self._enrich_sem is not None
        return self._enrich_sem

    @property
    def preclassify_sem(self) -> asyncio.Semaphore:
        """Semaphore for preclassify stage."""
        self._ensure_initialized()
        assert self._preclassify_sem is not None
        return self._preclassify_sem

    @property
    def deep_sem(self) -> asyncio.Semaphore:
        """Semaphore for deep processing stage."""
        self._ensure_initialized()
        assert self._deep_sem is not None
        return self._deep_sem

    @property
    def backlog_sem(self) -> asyncio.Semaphore:
        """Semaphore for backlog processing (multiplied)."""
        self._ensure_initialized()
        assert self._backlog_sem is not None
        return self._backlog_sem

    @property
    def gh_archive_sem(self) -> asyncio.Semaphore:
        """Semaphore for GH Archive fetches."""
        self._ensure_initialized()
        assert self._gh_archive_sem is not None
        return self._gh_archive_sem

    @property
    def github_search_auth_sem(self) -> asyncio.Semaphore:
        """Semaphore for authenticated GitHub searches."""
        self._ensure_initialized()
        assert self._github_search_auth_sem is not None
        return self._github_search_auth_sem

    @property
    def github_search_unauth_sem(self) -> asyncio.Semaphore:
        """Semaphore for unauthenticated GitHub searches."""
        self._ensure_initialized()
        assert self._github_search_unauth_sem is not None
        return self._github_search_unauth_sem

    @property
    def docker_sem(self) -> asyncio.Semaphore:
        """Semaphore for Docker container creation."""
        self._ensure_initialized()
        assert self._docker_sem is not None
        return self._docker_sem


# GitHub API rate limiting constants
GITHUB_RATE_LIMIT_DELAY: float = 2.1  # seconds between pages

_docker_sem: asyncio.Semaphore | None = None
_docker_containers_limit: int = 8


def set_docker_containers_limit(limit: int) -> None:
    """Set the maximum concurrent Docker containers.

    Must be called before any container creation.
    Resets the semaphore to use the new limit.

    Args:
        limit: Maximum number of concurrent Docker containers.
    """
    global _docker_sem, _docker_containers_limit
    _docker_containers_limit = limit
    _docker_sem = None  # Reset to force re-initialization with new limit


def get_docker_semaphore() -> asyncio.Semaphore:
    """Get the global Docker semaphore for container creation.

    Lazy initialization - creates on first call.

    Returns:
        asyncio.Semaphore for limiting concurrent Docker containers.
    """
    global _docker_sem
    if _docker_sem is None:
        _docker_sem = asyncio.Semaphore(_docker_containers_limit)
    return _docker_sem


async def rate_limit_delay() -> None:
    """Apply rate limiting delay for GitHub API pagination.

    Waits 2.1 seconds between API page requests to avoid rate limits.
    This matches the delay used in the Rust implementation.
    """
    await asyncio.sleep(GITHUB_RATE_LIMIT_DELAY)


__all__ = [
    "ConcurrencyConfig",
    "PipelineSemaphores",
    "rate_limit_delay",
    "GITHUB_RATE_LIMIT_DELAY",
    "get_docker_semaphore",
    "set_docker_containers_limit",
]
