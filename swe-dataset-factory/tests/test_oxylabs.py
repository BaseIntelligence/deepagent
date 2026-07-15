"""Unit tests for Oxylabs universal-only Web Scraper client (offline mocks)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from swe_factory.sources.oxylabs import (
    ALLOWED_SOURCE,
    FORBIDDEN_SOURCES,
    OXYLABS_REALTIME_URL,
    DictOxylabsTransport,
    OxylabsAuthError,
    OxylabsClient,
    OxylabsCredentials,
    OxylabsError,
    OxylabsFetchResult,
    OxylabsSourceError,
    OxylabsTransportError,
    assert_allowed_source,
    build_universal_payload,
    has_oxylabs_credentials,
    resolve_oxylabs_credentials,
    scrape_github_http,
)

SAMPLE_GITHUB = "https://github.com/psf/requests"


def _mock_results(
    *,
    content: str = "<html>ok</html>",
    status: int = 200,
    job_id: str = "job-1",
) -> dict[str, Any]:
    return {
        "results": [
            {
                "content": content,
                "status_code": status,
                "url": SAMPLE_GITHUB,
            }
        ],
        "job": {"id": job_id, "status": "done"},
    }


def test_allowed_source_constant_is_universal() -> None:
    assert ALLOWED_SOURCE == "universal"
    assert "amazon_product" in FORBIDDEN_SOURCES
    assert "marketplace" in FORBIDDEN_SOURCES


def test_assert_allowed_source_accepts_universal() -> None:
    assert assert_allowed_source("universal") == "universal"
    assert assert_allowed_source("Universal") == "universal"


@pytest.mark.parametrize(
    "bad",
    [
        "amazon_product",
        "amazon_search",
        "amazon",
        "marketplace",
        "marketplace_product",
        "google_search",
        "ebay_product",
        "walmart_product",
        "google",  # non-universal / not allowed
        "github",
        "",
    ],
)
def test_rejects_non_universal_and_amazon_marketplace_sources(bad: str) -> None:
    with pytest.raises(OxylabsSourceError):
        assert_allowed_source(bad)


def test_build_universal_payload_correct() -> None:
    payload = build_universal_payload(SAMPLE_GITHUB)
    assert payload == {"source": "universal", "url": SAMPLE_GITHUB}
    # extra must not override source
    payload2 = build_universal_payload(
        SAMPLE_GITHUB,
        source="universal",
        extra={"render": "html", "source": "amazon_product"},
    )
    assert payload2["source"] == "universal"
    assert payload2["url"] == SAMPLE_GITHUB
    assert payload2.get("render") == "html"


def test_build_payload_rejects_amazon_source() -> None:
    with pytest.raises(OxylabsSourceError):
        build_universal_payload(SAMPLE_GITHUB, source="amazon_product")


def test_build_payload_rejects_bad_url() -> None:
    with pytest.raises(OxylabsError):
        build_universal_payload("")
    with pytest.raises(OxylabsError):
        build_universal_payload("not-a-url")


def test_resolve_credentials_from_env_mapping() -> None:
    creds = resolve_oxylabs_credentials(
        env={"OXYLABS_USERNAME": "user-a", "OXYLABS_PASSWORD": "secret-pass-xyz"}
    )
    assert creds.username == "user-a"
    assert creds.password == "secret-pass-xyz"
    # Secrets must never appear in repr/str
    r = repr(creds)
    assert "secret-pass-xyz" not in r
    assert "user-a" not in r
    assert "***" in r


def test_missing_credentials_fail_closed() -> None:
    with pytest.raises(OxylabsAuthError) as ei:
        resolve_oxylabs_credentials(env={})
    msg = str(ei.value)
    assert "OXYLABS_USERNAME" in msg or "OXYLABS_PASSWORD" in msg
    assert "refusing" in msg.lower() or "required" in msg.lower()

    with pytest.raises(OxylabsAuthError):
        resolve_oxylabs_credentials(env={"OXYLABS_USERNAME": "only-user"})
    with pytest.raises(OxylabsAuthError):
        resolve_oxylabs_credentials(env={"OXYLABS_PASSWORD": "only-pass"})


def test_has_oxylabs_credentials() -> None:
    assert has_oxylabs_credentials({"OXYLABS_USERNAME": "u", "OXYLABS_PASSWORD": "p"}) is True
    assert has_oxylabs_credentials({}) is False
    assert has_oxylabs_credentials({"OXYLABS_USERNAME": "u", "OXYLABS_PASSWORD": ""}) is False


def test_from_env_missing_creds_no_network() -> None:
    """Fail closed before constructing a network path that could spam the API."""
    calls: list[str] = []

    class BoomTransport:
        def post_json(self, url: str, *, auth: tuple[str, str], payload: dict[str, Any]) -> Any:
            calls.append(url)
            raise AssertionError("must not reach transport without credentials")

    with pytest.raises(OxylabsAuthError):
        OxylabsClient.from_env(env={})
    assert calls == []


def test_scrape_url_universal_payload_under_mock() -> None:
    transport = DictOxylabsTransport(default=_mock_results(content="hello-github"))
    client = OxylabsClient(
        credentials=OxylabsCredentials("mock-user", "mock-secret-PASSWORD"),
        transport=transport,
    )
    result = client.scrape_url(SAMPLE_GITHUB)
    assert isinstance(result, OxylabsFetchResult)
    assert result.ok is True
    assert result.content == "hello-github"
    assert result.status_code == 200
    assert result.job_id == "job-1"
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == OXYLABS_REALTIME_URL
    assert call["payload"]["source"] == "universal"
    assert call["payload"]["url"] == SAMPLE_GITHUB
    # password recorded only as presence flag — not the secret string
    blob = json.dumps(transport.calls)
    assert "mock-secret-PASSWORD" not in blob
    assert call["auth_password_set"] is True


def test_scrape_rejects_amazon_before_transport() -> None:
    transport = DictOxylabsTransport(default=_mock_results())
    client = OxylabsClient(
        credentials=OxylabsCredentials("u", "p"),
        transport=transport,
    )
    with pytest.raises(OxylabsSourceError):
        client.scrape_url(SAMPLE_GITHUB, source="amazon_product")
    assert transport.calls == []


def test_httpx_mock_transport_builds_auth_and_payload() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization", "")
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_mock_results(content="via-httpx"))

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OxylabsClient.from_env(
        username="oxy-user",
        password="oxy-pass-SECRET",
        http_client=http,
        env={},  # force explicit only
    )
    result = client.scrape_url(SAMPLE_GITHUB)
    assert result.content == "via-httpx"
    assert captured["body"]["source"] == "universal"
    assert captured["body"]["url"] == SAMPLE_GITHUB
    assert captured["url"].startswith("https://realtime.oxylabs.io/")
    # Basic auth present but we never print password in client repr
    assert captured["auth"].startswith("Basic ")
    assert "oxy-pass-SECRET" not in repr(client)
    assert "oxy-pass-SECRET" not in str(client.credentials)
    client.close()


def test_auth_401_raises_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OxylabsClient.from_env(
        username="u",
        password="p",
        http_client=http,
        env={},
    )
    with pytest.raises(OxylabsAuthError):
        client.scrape_url(SAMPLE_GITHUB)
    client.close()


def test_scrape_github_http_helper() -> None:
    transport = DictOxylabsTransport(default=_mock_results(content="helper-body"))
    client = OxylabsClient(
        credentials=OxylabsCredentials("u", "p"),
        transport=transport,
    )
    out = scrape_github_http(SAMPLE_GITHUB, client=client)
    assert out.content == "helper-body"


def test_empty_results_raise_transport_error() -> None:
    transport = DictOxylabsTransport(default={"results": []})
    client = OxylabsClient(
        credentials=OxylabsCredentials("u", "p"),
        transport=transport,
    )
    with pytest.raises(OxylabsTransportError):
        client.scrape_url(SAMPLE_GITHUB)


def test_no_secrets_in_info_logs(caplog: pytest.LogCaptureFixture) -> None:
    secret = "super-secret-oxy-password-ZZZ"
    transport = DictOxylabsTransport(default=_mock_results())
    client = OxylabsClient(
        credentials=OxylabsCredentials("oxy-user-logtest", secret),
        transport=transport,
    )
    with caplog.at_level(logging.INFO, logger="swe_factory.sources.oxylabs"):
        client.scrape_url(SAMPLE_GITHUB)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in joined
    assert "oxy-user-logtest" not in joined
    assert "source=universal" in joined or "universal" in joined


def test_source_code_has_no_hardcoded_oxylabs_password() -> None:
    """Repo scan: client module must not embed real secret assignments."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "swe_factory" / "sources" / "oxylabs.py"
    text = src.read_text(encoding="utf-8")
    assert "OXYLABS_PASSWORD=" not in text
    assert "password = '" not in text.lower().replace("password: str", "")
    # Credential fields pull from env keys only
    assert "OXYLABS_USERNAME" in text
    assert "OXYLABS_PASSWORD" in text


@pytest.mark.integration
def test_live_universal_probe_when_creds_present() -> None:
    """Optional live smoke: skip unless OXYLABS_* are set in the process env."""
    import os

    if not has_oxylabs_credentials(os.environ):
        pytest.skip("OXYLABS_USERNAME/PASSWORD not set; live probe skipped")
    with OxylabsClient.from_env() as client:
        result = client.scrape_url("https://github.com/psf/requests")
    # Do not assert exact github HTML; just non-empty success-ish content.
    assert result.status_code in (200, 0) or result.content
    assert len(result.content) > 0
    # Password must not appear in content (paranoia)
    assert os.environ.get("OXYLABS_PASSWORD", "___never___") not in result.content
