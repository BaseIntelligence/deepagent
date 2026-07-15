"""Oxylabs Web Scraper API client (GitHub HTTP via source=universal only).

Force-path for all github.com page/raw HTTP fetches in the DeepSWE real-PR
pipeline. Never uses amazon/marketplace templates. Credentials load only
from environment; secrets are never logged or included in repr.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

OXYLABS_REALTIME_URL = "https://realtime.oxylabs.io/v1/queries"
ALLOWED_SOURCE = "universal"
# Explicit agent-denylisted templates for this mission (never send live).
FORBIDDEN_SOURCES: frozenset[str] = frozenset(
    {
        "amazon_product",
        "amazon_search",
        "amazon_pricing",
        "amazon_sellers",
        "amazon_bestsellers",
        "amazon",
        "google_search",
        "google_shopping_product",
        "google_shopping_search",
        "ebay_product",
        "ebay_search",
        "walmart_product",
        "walmart_search",
        "target_product",
        "target_search",
        "etsy_product",
        "etsy_search",
        # Generic marketplace spoof labels that must never ship.
        "marketplace",
        "marketplace_product",
    }
)


class OxylabsError(RuntimeError):
    """Base error for Oxylabs client failures."""


class OxylabsAuthError(OxylabsError):
    """Missing or rejected Oxylabs credentials."""


class OxylabsSourceError(OxylabsError):
    """Requested source template is forbidden for this mission."""


class OxylabsTransportError(OxylabsError):
    """Network or provider HTTP failure (no secret material in message)."""


@dataclass(frozen=True, slots=True)
class OxylabsCredentials:
    """Resolved username/password. Repr is always masked."""

    username: str
    password: str

    def __repr__(self) -> str:
        user_hint = "***" if self.username else ""
        return f"OxylabsCredentials(username={user_hint!r}, password='***')"

    def __str__(self) -> str:
        return repr(self)

    def as_basic_auth(self) -> tuple[str, str]:
        return self.username, self.password


@dataclass(frozen=True, slots=True)
class OxylabsFetchResult:
    """Normalized scrape result (no credentials)."""

    url: str
    status_code: int
    content: str
    job_id: str | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300 and bool(self.content)


class OxylabsTransport(Protocol):
    """Injectable POST transport for unit tests."""

    def post_json(
        self,
        url: str,
        *,
        auth: tuple[str, str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]: ...


def resolve_oxylabs_credentials(
    *,
    username: str | None = None,
    password: str | None = None,
    env: Mapping[str, str] | None = None,
) -> OxylabsCredentials:
    """Load credentials from explicit args or OXYLABS_* env. Fail closed if missing.

    Never logs username/password values.
    """
    environ = env if env is not None else os.environ
    user = (username if username is not None else environ.get("OXYLABS_USERNAME", "")).strip()
    pwd = (password if password is not None else environ.get("OXYLABS_PASSWORD", "")).strip()
    if not user or not pwd:
        raise OxylabsAuthError(
            "OXYLABS_USERNAME and OXYLABS_PASSWORD are required for GitHub HTTP "
            "via Oxylabs (set them in the gitignored .env). Missing credentials; "
            "refusing network call."
        )
    return OxylabsCredentials(username=user, password=pwd)


def has_oxylabs_credentials(env: Mapping[str, str] | None = None) -> bool:
    """True when both OXYLABS_USERNAME and OXYLABS_PASSWORD are non-empty."""
    environ = env if env is not None else os.environ
    user = environ.get("OXYLABS_USERNAME", "").strip()
    pwd = environ.get("OXYLABS_PASSWORD", "").strip()
    return bool(user and pwd)


def assert_allowed_source(source: str) -> str:
    """Return source only if it is the allowed universal scraper; else raise.

    Mission rule: GitHub HTTP only via source=universal. Amazon / marketplace
    (and other non-universal templates) must fail closed before any network I/O.
    """
    cleaned = (source or "").strip().lower()
    if not cleaned:
        raise OxylabsSourceError("Oxylabs source must be non-empty; only 'universal' is allowed")
    if cleaned in FORBIDDEN_SOURCES:
        raise OxylabsSourceError(
            f"Oxylabs source {source!r} is forbidden for this factory "
            f"(amazon/marketplace templates are not permitted). "
            f"Only source={ALLOWED_SOURCE!r} is allowed."
        )
    if cleaned != ALLOWED_SOURCE:
        raise OxylabsSourceError(
            f"Oxylabs source {source!r} refused; only source={ALLOWED_SOURCE!r} "
            "is permitted for GitHub HTTP fetches."
        )
    return ALLOWED_SOURCE


def build_universal_payload(
    url: str,
    *,
    source: str = ALLOWED_SOURCE,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the realtime API request body for a URL scrape.

    Always forces ``source=universal`` after validating the requested source.
    """
    allowed = assert_allowed_source(source)
    target = (url or "").strip()
    if not target:
        raise OxylabsError("url must be non-empty for universal scrape")
    parsed = urlparse(target)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise OxylabsError(f"url must be absolute http(s); got {url!r}")
    payload: dict[str, Any] = {
        "source": allowed,
        "url": target,
    }
    if extra:
        for key, value in extra.items():
            if key in ("source", "url"):
                continue
            payload[key] = value
    # Re-assert after merge so callers cannot override source via extra tricks
    # by putting source only in extra without passing source= (already handled);
    # and enforce postcondition.
    if payload.get("source") != ALLOWED_SOURCE:
        raise OxylabsSourceError("payload source drifted from universal")
    return payload


def _content_from_results(results: list[Any]) -> tuple[int, str]:
    """Extract HTTP status + body text from Oxylabs results array."""
    if not results:
        return 0, ""
    first = results[0]
    if not isinstance(first, dict):
        return 0, ""
    status = first.get("status_code")
    try:
        status_i = int(status) if status is not None else 0
    except (TypeError, ValueError):
        status_i = 0
    content = first.get("content")
    if content is None:
        content_s = ""
    elif isinstance(content, str):
        content_s = content
    elif isinstance(content, (dict, list)):
        # parsed JSON blob — stringify for miner consumers
        import json

        content_s = json.dumps(content)
    else:
        content_s = str(content)
    return status_i, content_s


@dataclass
class HttpxOxylabsTransport:
    """Live realtime transport over httpx Basic Auth."""

    timeout: float = 120.0
    endpoint: str = OXYLABS_REALTIME_URL
    _client: httpx.Client | None = field(default=None, repr=False)
    _owns_client: bool = field(default=True, repr=False)

    def __init__(
        self,
        *,
        timeout: float = 120.0,
        endpoint: str = OXYLABS_REALTIME_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.timeout = timeout
        self.endpoint = endpoint
        self._client = http_client
        self._owns_client = http_client is None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
            self._owns_client = True
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def post_json(
        self,
        url: str,
        *,
        auth: tuple[str, str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            response = client.post(
                url,
                json=dict(payload),
                auth=auth,
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            # Never embed auth headers into the exception message.
            raise OxylabsTransportError(f"Oxylabs transport error: {type(exc).__name__}") from exc
        if response.status_code in (401, 403):
            raise OxylabsAuthError(f"Oxylabs auth rejected (status={response.status_code})")
        if response.status_code >= 400:
            raise OxylabsTransportError(f"Oxylabs HTTP {response.status_code} from realtime API")
        try:
            data = response.json()
        except ValueError as exc:
            raise OxylabsTransportError("Oxylabs returned non-JSON body") from exc
        if not isinstance(data, dict):
            raise OxylabsTransportError("Oxylabs JSON payload must be an object")
        return data


@dataclass
class DictOxylabsTransport:
    """Offline mock that records posts and returns canned responses by url key."""

    responses: dict[str, dict[str, Any]] | None = None
    default: dict[str, Any] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def post_json(
        self,
        url: str,
        *,
        auth: tuple[str, str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Record call with *redacted* auth so tests can assert no secret leak
        # by inspecting str(calls) and that password never appears.
        self.calls.append(
            {
                "url": url,
                "auth_user": auth[0],
                "auth_password_set": bool(auth[1]),
                "payload": dict(payload),
            }
        )
        target = str(payload.get("url") or "")
        if self.responses and target in self.responses:
            return self.responses[target]
        if self.default is not None:
            return self.default
        raise OxylabsTransportError(f"mock Oxylabs has no response for url={target!r}")


@dataclass
class OxylabsClient:
    """Web Scraper API client constrained to source=universal."""

    credentials: OxylabsCredentials
    transport: OxylabsTransport
    endpoint: str = OXYLABS_REALTIME_URL
    _owns_transport: bool = field(default=False, repr=False)

    def __repr__(self) -> str:
        return f"OxylabsClient(credentials={self.credentials!r}, endpoint={self.endpoint!r})"

    @classmethod
    def from_env(
        cls,
        *,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
        env: Mapping[str, str] | None = None,
    ) -> OxylabsClient:
        """Construct a live client from env. Raises OxylabsAuthError if unset."""
        creds = resolve_oxylabs_credentials(username=username, password=password, env=env)
        transport = HttpxOxylabsTransport(timeout=timeout, http_client=http_client)
        return cls(credentials=creds, transport=transport, _owns_transport=True)

    def close(self) -> None:
        closer = getattr(self.transport, "close", None)
        if self._owns_transport and callable(closer):
            closer()

    def __enter__(self) -> OxylabsClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def scrape_url(
        self,
        url: str,
        *,
        source: str = ALLOWED_SOURCE,
        extra: Mapping[str, Any] | None = None,
    ) -> OxylabsFetchResult:
        """Fetch ``url`` via Oxylabs universal realtime scrape.

        Fails closed for non-universal ``source`` before any network I/O.
        """
        payload = build_universal_payload(url, source=source, extra=extra)
        logger.info(
            "oxylabs scrape source=%s url=%s",
            payload["source"],
            payload["url"],
        )
        # Guard: never log credentials at INFO or above.
        data = self.transport.post_json(
            self.endpoint,
            auth=self.credentials.as_basic_auth(),
            payload=payload,
        )
        results_raw = data.get("results")
        results: list[Any] = results_raw if isinstance(results_raw, list) else []
        status_code, content = _content_from_results(results)
        job_id: str | None = None
        if isinstance(data.get("job"), dict) and isinstance(data["job"].get("id"), str):
            job_id = data["job"]["id"]
        elif isinstance(data.get("id"), str):
            job_id = data["id"]
        if status_code == 0 and not content:
            # Provider returned job wrapper without content — still surface structure.
            raise OxylabsTransportError(
                f"Oxylabs returned empty results for url={payload['url']!r}"
            )
        return OxylabsFetchResult(
            url=str(payload["url"]),
            status_code=status_code,
            content=content,
            job_id=job_id,
            raw=data,
        )


def scrape_github_http(
    url: str,
    *,
    client: OxylabsClient | None = None,
    env: Mapping[str, str] | None = None,
) -> OxylabsFetchResult:
    """Convenience: one-shot universal fetch with env credentials.

    Prefer injecting a long-lived :class:`OxylabsClient` when bulk mining.
    """
    owns = client is None
    active = client or OxylabsClient.from_env(env=env)
    try:
        return active.scrape_url(url, source=ALLOWED_SOURCE)
    finally:
        if owns:
            active.close()


__all__ = [
    "ALLOWED_SOURCE",
    "FORBIDDEN_SOURCES",
    "OXYLABS_REALTIME_URL",
    "DictOxylabsTransport",
    "HttpxOxylabsTransport",
    "OxylabsAuthError",
    "OxylabsClient",
    "OxylabsCredentials",
    "OxylabsError",
    "OxylabsFetchResult",
    "OxylabsSourceError",
    "OxylabsTransport",
    "OxylabsTransportError",
    "assert_allowed_source",
    "build_universal_payload",
    "has_oxylabs_credentials",
    "resolve_oxylabs_credentials",
    "scrape_github_http",
]
