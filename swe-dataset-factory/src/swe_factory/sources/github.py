"""GitHub REST client for real_pr mining (stdlib-friendly httpx).

Discovery paths (product live mine):
  - discovery_path=list_pulls  → client.list_pulls (per-repo closed/merged PRs)
  - discovery_path=search      → client.search_issues / search_merged_pull_requests
                                (GET /search/issues?q=is:pr+is:merged+...)

Token resolution order (never logged):
  explicit → GITHUB_TOKEN → GH_TOKEN → ``gh auth token`` subprocess.

Respects Retry-After / X-RateLimit-* with injectable sleep; fail closed on
401/403/exhausted rate limits with reason codes. Routes only to api.github.com
(or injectable base_url) — never Oxylabs Realtime for REST/Search shapes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

GITHUB_API_DEFAULT = "https://api.github.com"
USER_AGENT = "swe-dataset-factory-pr-miner"
API_VERSION = "2022-11-28"

# Candidate ledger discovery_path labels (VAL-LMINE-003 / VAL-LMINE-006).
DISCOVERY_PATH_SEARCH = "search"
DISCOVERY_PATH_LIST_PULLS = "list_pulls"
DISCOVERY_PATHS = frozenset({DISCOVERY_PATH_SEARCH, DISCOVERY_PATH_LIST_PULLS})

# Bound secondary-rate sleep so unit tests / storms do not wait forever.
_MAX_BACKOFF_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 2


class GitHubError(RuntimeError):
    """Raised when a GitHub REST call fails or returns unexpected payload."""

    def __init__(self, message: str, *, reason_code: str = "github_error") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class GitHubAuthError(GitHubError):
    """401/403 auth failures — fail closed, do not invent product N."""

    def __init__(self, message: str, *, reason_code: str = "auth_failed") -> None:
        super().__init__(message, reason_code=reason_code)


class GitHubRateLimitError(GitHubError):
    """Rate limit / secondary limit exhausted after backoff (VAL-LMINE-004/010)."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "rate_limited",
        rate_limit: RateLimitInfo | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code)
        self.rate_limit = rate_limit


@dataclass(frozen=True, slots=True)
class RateLimitInfo:
    """Parsed Retry-After / X-RateLimit-* response headers."""

    retry_after: float | None = None
    limit: int | None = None
    remaining: int | None = None
    reset: int | None = None


@dataclass(frozen=True, slots=True)
class MockHttpResponse:
    """Error or envelope response for offline DictGitHubTransport tests."""

    status_code: int
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)


def parse_rate_limit_headers(headers: Mapping[str, str] | None) -> RateLimitInfo:
    """Parse Retry-After and X-RateLimit-* (case-insensitive keys)."""
    if not headers:
        return RateLimitInfo()
    lower = {str(k).lower(): str(v) for k, v in headers.items()}

    def _float(key: str) -> float | None:
        raw = lower.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _int(key: str) -> int | None:
        raw = lower.get(key)
        if raw is None or raw == "":
            return None
        try:
            return int(float(raw))
        except ValueError:
            return None

    return RateLimitInfo(
        retry_after=_float("retry-after"),
        limit=_int("x-ratelimit-limit"),
        remaining=_int("x-ratelimit-remaining"),
        reset=_int("x-ratelimit-reset"),
    )


def _gh_auth_token() -> str | None:
    """Non-interactive ``gh auth token``; never log the value."""
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    token = (proc.stdout or "").strip()
    return token or None


def resolve_github_token(explicit: str | None = None) -> str | None:
    """Order: explicit → GITHUB_TOKEN → GH_TOKEN → gh auth token. Never log."""
    if explicit is not None:
        cleaned = explicit.strip()
        return cleaned or None
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        raw = os.environ.get(key)
        if raw and raw.strip():
            return raw.strip()
    return _gh_auth_token()


def build_merged_pr_search_query(
    *,
    language: str | None = None,
    repo: str | None = None,
    extra_qualifiers: Sequence[str] | None = None,
) -> str:
    """Build Search Issues/PRs query for merged PRs (multi-lang ready)."""
    parts: list[str] = ["is:pr", "is:merged"]
    if language:
        parts.append(f"language:{language.strip()}")
    if repo:
        cleaned = repo.strip().removesuffix(".git")
        if cleaned.startswith("https://github.com/"):
            cleaned = cleaned[len("https://github.com/") :]
        if cleaned.startswith("github.com/"):
            cleaned = cleaned[len("github.com/") :]
        parts.append(f"repo:{cleaned}")
    if extra_qualifiers:
        for q in extra_qualifiers:
            q = str(q).strip()
            if q:
                parts.append(q)
    return " ".join(parts)


class GitHubTransport(Protocol):
    """Minimal HTTP transport for GET JSON endpoints.

    Implementations may raise GitHubError / Auth / RateLimit from get_json.
    Optional ``headers`` on MockHttpResponse-style returns are honored by
    GitHubClient when the transport yields a MockHttpResponse object.
    """

    def get_json(self, path: str, *, params: Mapping[str, str] | None = None) -> Any: ...


@dataclass
class HttpxGitHubTransport:
    """Live GitHub REST transport over httpx (timeouts, auth header only).

    Never logs Authorization. Base URL defaults to api.github.com (not Oxylabs).
    """

    token: str | None = None
    base_url: str = GITHUB_API_DEFAULT
    timeout: float = 30.0
    user_agent: str = USER_AGENT
    _client: httpx.Client | None = field(default=None, repr=False)

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            headers: dict[str, str] = {
                "Accept": "application/vnd.github+json",
                "User-Agent": self.user_agent,
                "X-GitHub-Api-Version": API_VERSION,
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.Client(
                base_url=self.base_url.rstrip("/"),
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def get_json(self, path: str, *, params: Mapping[str, str] | None = None) -> Any:
        client = self._ensure_client()
        url = path if path.startswith("/") else f"/{path}"
        try:
            response = client.get(url, params=dict(params or {}))
        except httpx.HTTPError as exc:
            # Never interpolate headers/tokens into the message.
            raise GitHubError(
                f"GitHub request failed: {type(exc).__name__}",
                reason_code="transport_error",
            ) from exc
        rate = parse_rate_limit_headers(response.headers)
        status = response.status_code
        if status in (401, 403, 429) or status >= 400:
            # Prefer structured raise so GitHubClient can backoff/rethrow.
            body: Any
            try:
                body = response.json()
            except ValueError:
                body = None
            return MockHttpResponse(status_code=status, body=body, headers=dict(response.headers))
        if rate.remaining == 0 and rate.retry_after:
            # Remaining zero with body may still be 200 only rarely; just return JSON.
            pass
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubError(
                f"GitHub returned non-JSON for {url}",
                reason_code="bad_payload",
            ) from exc


@dataclass
class DictGitHubTransport:
    """Offline map of exact path(+query) → JSON / MockHttpResponse for tests.

    Optional ``dynamic`` callback receives (path, params) and may return
    MockHttpResponse envelopes so rate-limit / auth paths are unit-testable.
    """

    routes: MutableMapping[str, Any]
    default: Any | None = None
    calls: list[str] = field(default_factory=list)
    dynamic: Callable[[str, Mapping[str, str] | None], Any] | None = None

    def get_json(self, path: str, *, params: Mapping[str, str] | None = None) -> Any:
        key = path if path.startswith("/") else f"/{path}"
        if params:
            qs = urlencode(sorted((str(k), str(v)) for k, v in params.items()))
            full = f"{key}?{qs}"
        else:
            full = key
        self.calls.append(full)
        if self.dynamic is not None:
            return self.dynamic(key, params)
        if full in self.routes:
            return self.routes[full]
        if key in self.routes:
            return self.routes[key]
        if self.default is not None:
            return self.default
        raise GitHubError(f"mock GitHub has no route for {full}", reason_code="mock_miss")


def _is_rate_limited(status: int, body: Any, rate: RateLimitInfo) -> bool:
    if status == 429:
        return True
    if status != 403:
        return rate.remaining == 0 and (rate.retry_after is not None or rate.reset is not None)
    message = ""
    if isinstance(body, dict):
        message = str(body.get("message") or "").lower()
    rate_phrases = (
        "rate limit",
        "secondary rate",
        "abuse detection",
        "api rate limit exceeded",
        "you have exceeded a secondary rate limit",
    )
    if any(p in message for p in rate_phrases):
        return True
    if rate.remaining == 0:
        return True
    return rate.retry_after is not None


def _backoff_seconds(rate: RateLimitInfo) -> float:
    if rate.retry_after is not None and rate.retry_after >= 0:
        return min(float(rate.retry_after), _MAX_BACKOFF_SECONDS)
    # Conservative default when headers omit Retry-After
    return min(5.0, _MAX_BACKOFF_SECONDS)


@dataclass
class GitHubClient:
    """Thin wrapper over a transport for PR mining + Search endpoints.

    discovery_path labels for candidate ledgers:
      - ``search`` when results come from search_issues / search_merged_pull_requests
      - ``list_pulls`` when results come from list_pulls
    """

    transport: GitHubTransport
    per_page: int = 50
    sleep_fn: Callable[[float], None] = field(default=time.sleep, repr=False)
    max_retries: int = _DEFAULT_MAX_RETRIES

    @classmethod
    def from_env(
        cls,
        *,
        token: str | None = None,
        base_url: str = GITHUB_API_DEFAULT,
        timeout: float = 30.0,
        sleep_fn: Callable[[float], None] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> GitHubClient:
        resolved = resolve_github_token(token)
        return cls(
            transport=HttpxGitHubTransport(
                token=resolved,
                base_url=base_url,
                timeout=timeout,
            ),
            sleep_fn=sleep_fn if sleep_fn is not None else time.sleep,
            max_retries=max_retries,
        )

    def _request(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        allow_retry: bool = True,
    ) -> Any:
        """GET JSON with auth fail-closed + rate-limit Retry-After backoff."""
        attempts = 0
        while True:
            attempts += 1
            raw = self.transport.get_json(path, params=params)
            if not isinstance(raw, MockHttpResponse):
                return raw
            status = int(raw.status_code)
            rate = parse_rate_limit_headers(raw.headers)
            url_hint = path if path.startswith("/") else f"/{path}"

            if _is_rate_limited(status, raw.body, rate):
                if allow_retry and attempts <= self.max_retries:
                    delay = _backoff_seconds(rate)
                    logger.info(
                        "GitHub rate limit on %s (status=%s remaining=%s); backoff %.1fs",
                        url_hint,
                        status,
                        rate.remaining,
                        delay,
                    )
                    self.sleep_fn(delay)
                    continue
                reason = "rate_limited"
                body_msg = ""
                if isinstance(raw.body, dict):
                    body_msg = str(raw.body.get("message") or "")
                if "secondary" in body_msg.lower():
                    reason = "secondary_rate_limit"
                elif path.lstrip("/").startswith("search"):
                    reason = "search_rate_limited"
                raise GitHubRateLimitError(
                    f"GitHub rate limited (HTTP {status}) for {url_hint}",
                    reason_code=reason,
                    rate_limit=rate,
                )

            if status == 401:
                raise GitHubAuthError(
                    f"GitHub HTTP 401 for {url_hint}",
                    reason_code="unauthorized",
                )
            if status == 403:
                raise GitHubAuthError(
                    f"GitHub HTTP 403 for {url_hint}",
                    reason_code="forbidden",
                )
            if status >= 400:
                raise GitHubError(
                    f"GitHub HTTP {status} for {url_hint}",
                    reason_code=f"http_{status}",
                )
            # Unexpected 2xx mock envelope
            return raw.body

    def get_pull(self, repo: str, number: int) -> dict[str, Any]:
        owner, name = _split_repo(repo)
        payload = self._request(f"/repos/{owner}/{name}/pulls/{int(number)}")
        if not isinstance(payload, dict):
            raise GitHubError(
                f"expected pull object for {repo}#{number}",
                reason_code="bad_payload",
            )
        return payload

    def list_pulls(
        self,
        repo: str,
        *,
        state: str = "closed",
        sort: str = "updated",
        direction: str = "desc",
        page: int = 1,
        per_page: int | None = None,
    ) -> list[dict[str, Any]]:
        """List repository PRs. Label discovery_path=list_pulls on candidates."""
        owner, name = _split_repo(repo)
        pp = per_page if per_page is not None else self.per_page
        payload = self._request(
            f"/repos/{owner}/{name}/pulls",
            params={
                "state": state,
                "sort": sort,
                "direction": direction,
                "page": str(page),
                "per_page": str(pp),
            },
        )
        if not isinstance(payload, list):
            raise GitHubError(
                f"expected pull list for {repo}",
                reason_code="bad_payload",
            )
        return [item for item in payload if isinstance(item, dict)]

    def search_issues(
        self,
        query: str,
        *,
        page: int = 1,
        per_page: int | None = None,
        sort: str = "updated",
        order: str = "desc",
    ) -> dict[str, Any]:
        """GET /search/issues — primary Search Issues/PRs surface.

        For live mine, callers should use queries containing is:pr is:merged
        and label candidates discovery_path=search.
        """
        pp = per_page if per_page is not None else min(100, self.per_page)
        payload = self._request(
            "/search/issues",
            params={
                "q": query,
                "page": str(page),
                "per_page": str(pp),
                "sort": sort,
                "order": order,
            },
        )
        if not isinstance(payload, dict):
            raise GitHubError(
                "expected search issues object",
                reason_code="bad_payload",
            )
        return payload

    def search_merged_pull_requests(
        self,
        *,
        language: str | None = None,
        repo: str | None = None,
        extra_qualifiers: Sequence[str] | None = None,
        page: int = 1,
        per_page: int | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search merged PRs (multi-lang). Returns items[]; discovery_path=search."""
        q = (
            query
            if query is not None
            else build_merged_pr_search_query(
                language=language,
                repo=repo,
                extra_qualifiers=extra_qualifiers,
            )
        )
        payload = self.search_issues(q, page=page, per_page=per_page)
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def get_pull_files(
        self,
        repo: str,
        number: int,
        *,
        page: int = 1,
        per_page: int | None = None,
    ) -> list[dict[str, Any]]:
        owner, name = _split_repo(repo)
        pp = per_page if per_page is not None else min(100, self.per_page)
        payload = self._request(
            f"/repos/{owner}/{name}/pulls/{int(number)}/files",
            params={"page": str(page), "per_page": str(pp)},
        )
        if not isinstance(payload, list):
            raise GitHubError(
                f"expected files list for {repo}#{number}",
                reason_code="bad_payload",
            )
        return [item for item in payload if isinstance(item, dict)]

    def list_all_pull_files(
        self,
        repo: str,
        number: int,
        *,
        max_pages: int = 5,
    ) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            batch = self.get_pull_files(repo, number, page=page, per_page=100)
            files.extend(batch)
            if len(batch) < 100:
                break
        return files


def _split_repo(repo: str) -> tuple[str, str]:
    cleaned = repo.strip().removesuffix(".git")
    if cleaned.startswith("https://github.com/"):
        cleaned = cleaned[len("https://github.com/") :]
    if cleaned.startswith("github.com/"):
        cleaned = cleaned[len("github.com/") :]
    parts = [p for p in cleaned.split("/") if p]
    if len(parts) < 2:
        raise GitHubError(
            f"repo must be owner/name; got {repo!r}",
            reason_code="bad_repo",
        )
    return parts[0], parts[1]


JsonLoader = Callable[[], Any]


def load_offline_routes(entries: Sequence[tuple[str, Any]]) -> DictGitHubTransport:
    """Build a mock transport from (path, payload) pairs."""
    routes: dict[str, Any] = {}
    for path, payload in entries:
        key = path if path.startswith("/") else f"/{path}"
        routes[key] = payload
    return DictGitHubTransport(routes=routes)


__all__ = [
    "API_VERSION",
    "DISCOVERY_PATHS",
    "DISCOVERY_PATH_LIST_PULLS",
    "DISCOVERY_PATH_SEARCH",
    "GITHUB_API_DEFAULT",
    "DictGitHubTransport",
    "GitHubAuthError",
    "GitHubClient",
    "GitHubError",
    "GitHubRateLimitError",
    "GitHubTransport",
    "HttpxGitHubTransport",
    "MockHttpResponse",
    "RateLimitInfo",
    "build_merged_pr_search_query",
    "load_offline_routes",
    "parse_rate_limit_headers",
    "resolve_github_token",
]
