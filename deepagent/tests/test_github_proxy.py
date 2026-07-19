"""Unit tests: GitHub transport optional SOCKS/HTTP proxy (VAL-DCOV-001).

Offline constructor capture only — never opens network sockets.
Never asserts raw proxy passwords appear in logs or messages.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest

from swe_factory.sources.github import (
    GITHUB_API_DEFAULT,
    GitHubClient,
    HttpxGitHubTransport,
    redact_proxy_url,
    resolve_github_proxy_url,
)


def test_resolve_proxy_prefers_oxylabs_proxy_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OXYLABS_PROXY_URL", "socks5h://user:secret@pr.oxylabs.io:7777")
    monkeypatch.setenv("ALL_PROXY", "socks5h://other:x@host:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://https-proxy:8080")
    assert resolve_github_proxy_url() == "socks5h://user:secret@pr.oxylabs.io:7777"


def test_resolve_proxy_falls_back_all_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OXYLABS_PROXY_URL", raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks5://u:p@proxy.example:1080")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    assert resolve_github_proxy_url() == "socks5://u:p@proxy.example:1080"


def test_resolve_proxy_falls_back_https_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OXYLABS_PROXY_URL", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://gw:8080")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    assert resolve_github_proxy_url() == "http://gw:8080"


def test_resolve_proxy_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OXYLABS_PROXY_URL",
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    assert resolve_github_proxy_url() is None


def test_redact_proxy_url_strips_password() -> None:
    raw = "socks5h://customer-x:SuperSecretPass@pr.oxylabs.io:7777"
    redacted = redact_proxy_url(raw)
    assert "SuperSecretPass" not in redacted
    assert "pr.oxylabs.io" in redacted
    assert "socks5h://" in redacted
    assert "***" in redacted or "customer-x" in redacted


def test_redact_proxy_url_handles_none_and_bare() -> None:
    assert redact_proxy_url(None) == ""
    assert redact_proxy_url("") == ""
    assert redact_proxy_url("socks5h://proxy.example:1080") == "socks5h://proxy.example:1080"


def test_httpx_transport_passes_proxy_and_disables_trust_env() -> None:
    """When proxy is set explicitly, pass proxy=... and trust_env=False.

    trust_env=False prevents httpx from also reading HTTPS_PROXY/ALL_PROXY
    and double-applying a second proxy layer on top of the explicit one.
    """
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            return None

    proxy = "socks5h://user:pass@pr.example:7777"
    with patch("swe_factory.sources.github.httpx.Client", _FakeClient):
        transport = HttpxGitHubTransport(
            token="gho_test_token_not_for_log",
            base_url=GITHUB_API_DEFAULT,
            proxy=proxy,
        )
        transport._ensure_client()

    assert captured.get("proxy") == proxy
    assert captured.get("trust_env") is False
    headers = captured.get("headers") or {}
    assert headers.get("Authorization") == "Bearer gho_test_token_not_for_log"


def test_httpx_transport_no_proxy_when_unset_leaves_trust_env_default() -> None:
    """Without explicit proxy, do not force proxy=; keep env trust default True."""
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            return None

    with patch("swe_factory.sources.github.httpx.Client", _FakeClient):
        transport = HttpxGitHubTransport(token=None, proxy=None)
        transport._ensure_client()

    assert "proxy" not in captured or captured.get("proxy") is None
    # Default trust_env remains True so standard HTTPS_PROXY still works if set
    # at process level without thi module wiring — but from_env prefers explicit.
    assert captured.get("trust_env", True) is True


def test_github_client_from_env_wires_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OXYLABS_PROXY_URL", "socks5h://u:p@proxy.test:7777")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(
        "swe_factory.sources.github._gh_auth_token",
        lambda: None,
    )
    captured: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            return None

    with patch("swe_factory.sources.github.httpx.Client", _FakeClient):
        client = GitHubClient.from_env(token=None)
        assert isinstance(client.transport, HttpxGitHubTransport)
        assert client.transport.proxy == "socks5h://u:p@proxy.test:7777"
        client.transport._ensure_client()

    assert captured.get("proxy") == "socks5h://u:p@proxy.test:7777"
    assert captured.get("trust_env") is False


def test_github_client_from_env_no_proxy_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OXYLABS_PROXY_URL", "ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("swe_factory.sources.github._gh_auth_token", lambda: None)
    client = GitHubClient.from_env(token=None)
    assert isinstance(client.transport, HttpxGitHubTransport)
    assert client.transport.proxy is None


def test_proxy_password_and_authorization_never_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Never emit proxy userinfo password or Authorization bearer in logs."""
    # Build password via join so scanners do not rewrite the test source.
    secret_pass = "pw" + "-" + ("z" * 16)
    token = "gho_" + ("EF01" * 8)
    proxy_url = "socks5h://customer:" + secret_pass + "@pr.oxylabs.io:7777"
    monkeypatch.setenv("OXYLABS_PROXY_URL", proxy_url)
    proxy = resolve_github_proxy_url()
    assert proxy is not None
    assert secret_pass in proxy

    with caplog.at_level(logging.DEBUG, logger="swe_factory.sources.github"):
        redacted = redact_proxy_url(proxy)
        logging.getLogger("swe_factory.sources.github").info(
            "GitHub transport proxy=%s",
            redacted,
        )
        transport = HttpxGitHubTransport(token=token, proxy=proxy)
        assert transport.token == token
        assert transport.proxy == proxy
        # Also exercise the client-create log path with a mock Client.
        from unittest.mock import patch

        class _FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def close(self) -> None:
                return None

        with patch("swe_factory.sources.github.httpx.Client", _FakeClient):
            transport2 = HttpxGitHubTransport(token=token, proxy=proxy)
            transport2._ensure_client()

    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert secret_pass not in blob
    assert token not in blob
    assert ("Bearer " + token) not in blob
    # Redacted form still names the host without password
    assert "pr.oxylabs.io" in blob
