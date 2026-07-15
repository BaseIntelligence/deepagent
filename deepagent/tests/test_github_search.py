"""Unit tests for GitHub Search + list_pulls client (VAL-LMINE-001/004/006/009/010).

Offline DictGitHubTransport / MockHttpResponse only (no network). Covers:
- Search Issues/PRs parse for is:pr is:merged multi-lang queries
- Token resolution order GITHUB_TOKEN | GH_TOKEN | gh auth token
- Retry-After / X-RateLimit-* backoff hooks
- Fail closed on 401/403 with reason codes
- list_pulls remains functional
- No secrets in errors/logs; direct api.github.com (never Oxylabs)
- discovery_path labels search|list_pulls documented for guarantees
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from swe_factory.sources.github import (
    DISCOVERY_PATH_LIST_PULLS,
    DISCOVERY_PATH_SEARCH,
    DISCOVERY_PATHS,
    GITHUB_API_DEFAULT,
    DictGitHubTransport,
    GitHubAuthError,
    GitHubClient,
    GitHubError,
    GitHubRateLimitError,
    HttpxGitHubTransport,
    MockHttpResponse,
    RateLimitInfo,
    build_merged_pr_search_query,
    parse_rate_limit_headers,
    resolve_github_token,
)


def test_discovery_path_labels_are_search_and_list_pulls() -> None:
    """Documented discovery_path labels for candidate guarantees (VAL-LMINE-003/006)."""
    assert DISCOVERY_PATH_SEARCH == "search"
    assert DISCOVERY_PATH_LIST_PULLS == "list_pulls"
    assert DISCOVERY_PATH_SEARCH in DISCOVERY_PATHS
    assert DISCOVERY_PATH_LIST_PULLS in DISCOVERY_PATHS
    assert len(DISCOVERY_PATHS) == 2


def test_github_api_default_is_direct_not_oxylabs() -> None:
    """Live mine REST/Search must hit api.github.com, never Oxylabs (VAL-LMINE-009)."""
    assert GITHUB_API_DEFAULT == "https://api.github.com"
    assert "oxylabs" not in GITHUB_API_DEFAULT.lower()
    transport = HttpxGitHubTransport(token=None, base_url=GITHUB_API_DEFAULT)
    assert transport.base_url.rstrip("/") == "https://api.github.com"
    assert "oxylabs" not in transport.base_url.lower()


def test_resolve_token_prefers_github_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "gho_from_github")
    monkeypatch.setenv("GH_TOKEN", "gho_from_gh")
    assert resolve_github_token() == "gho_from_github"


def test_resolve_token_falls_back_to_gh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "gho_from_gh_only")
    assert resolve_github_token() == "gho_from_gh_only"


def test_resolve_token_falls_back_to_gh_auth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    def _fake_gh_auth() -> str | None:
        return "gho_from_cli"

    monkeypatch.setattr(
        "swe_factory.sources.github._gh_auth_token",
        _fake_gh_auth,
    )
    assert resolve_github_token() == "gho_from_cli"


def test_resolve_token_explicit_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "gho_env")
    assert resolve_github_token("gho_explicit") == "gho_explicit"


def test_build_merged_pr_search_query_multi_lang() -> None:
    q_py = build_merged_pr_search_query(language="python")
    assert "is:pr" in q_py
    assert "is:merged" in q_py
    assert "language:python" in q_py

    q_rs = build_merged_pr_search_query(language="rust", extra_qualifiers=["stars:>100"])
    assert "language:rust" in q_rs
    assert "stars:>100" in q_rs
    assert "is:merged" in q_rs


def test_search_issues_parses_items() -> None:
    payload = {
        "total_count": 2,
        "incomplete_results": False,
        "items": [
            {
                "number": 101,
                "title": "Hard multi-file fix",
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/lib/pulls/101",
                    "merged_at": "2024-01-02T00:00:00Z",
                },
                "repository_url": "https://api.github.com/repos/acme/lib",
                "html_url": "https://github.com/acme/lib/pull/101",
                "state": "closed",
            },
            {
                "number": 102,
                "title": "Another",
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/lib/pulls/102",
                    "merged_at": "2024-01-03T00:00:00Z",
                },
                "repository_url": "https://api.github.com/repos/acme/lib",
                "html_url": "https://github.com/acme/lib/pull/102",
                "state": "closed",
            },
        ],
    }
    transport = DictGitHubTransport(
        routes={
            "/search/issues": payload,
        }
    )
    client = GitHubClient(transport=transport)
    result = client.search_issues("is:pr is:merged language:python")
    assert result["total_count"] == 2
    items = client.search_merged_pull_requests(language="python")
    assert len(items) == 2
    assert items[0]["number"] == 101
    # Transport records search path for discovery_path=search evidence
    assert any(c.startswith("/search/issues") for c in transport.calls)


def test_search_merged_pull_requests_query_params() -> None:
    transport = DictGitHubTransport(
        routes={
            "/search/issues": {
                "total_count": 0,
                "incomplete_results": False,
                "items": [],
            }
        }
    )
    client = GitHubClient(transport=transport, per_page=30)
    client.search_merged_pull_requests(language="go", page=2, per_page=10)
    assert transport.calls, "expected search call"
    call = transport.calls[-1]
    assert call.startswith("/search/issues?")
    parsed = urlparse(call)
    qs = parse_qs(parsed.query)
    assert "q" in qs
    q = qs["q"][0]
    assert "is:pr" in q
    assert "is:merged" in q
    assert "language:go" in q
    assert qs["page"] == ["2"]
    assert qs["per_page"] == ["10"]


def test_list_pulls_path_still_works() -> None:
    pulls = [
        {
            "number": 7,
            "title": "merged",
            "merged_at": "2024-06-01T12:00:00Z",
            "state": "closed",
            "base": {"sha": "a" * 40},
            "head": {"sha": "b" * 40},
        }
    ]
    transport = DictGitHubTransport(
        routes={
            "/repos/owner/demo/pulls": pulls,
        }
    )
    client = GitHubClient(transport=transport)
    found = client.list_pulls("owner/demo", state="closed", page=1, per_page=30)
    assert len(found) == 1
    assert found[0]["number"] == 7
    assert DISCOVERY_PATH_LIST_PULLS == "list_pulls"
    assert any("/repos/owner/demo/pulls" in c for c in transport.calls)


def test_parse_rate_limit_headers_retry_after_and_x_headers() -> None:
    info = parse_rate_limit_headers(
        {
            "Retry-After": "12",
            "X-RateLimit-Limit": "5000",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1710000000",
        }
    )
    assert isinstance(info, RateLimitInfo)
    assert info.retry_after == 12.0
    assert info.limit == 5000
    assert info.remaining == 0
    assert info.reset == 1710000000


def test_fail_closed_on_401() -> None:
    transport = DictGitHubTransport(
        routes={
            "/repos/owner/x/pulls/1": MockHttpResponse(
                status_code=401,
                body={"message": "Bad credentials"},
                headers={},
            )
        }
    )
    client = GitHubClient(transport=transport)
    with pytest.raises(GitHubAuthError) as ei:
        client.get_pull("owner/x", 1)
    err = ei.value
    assert err.reason_code in {"auth_failed", "unauthorized", "http_401"}
    assert "401" in str(err)
    assert "gho_" not in str(err).lower()
    assert "Bearer" not in str(err)


def test_fail_closed_on_403() -> None:
    transport = DictGitHubTransport(
        routes={
            "/search/issues": MockHttpResponse(
                status_code=403,
                body={"message": "API rate limit exceeded"},
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "1710000000",
                    "Retry-After": "5",
                },
            )
        }
    )
    client = GitHubClient(transport=transport)
    # Prefer rate-limit surface when headers indicate exhaustion
    with pytest.raises((GitHubRateLimitError, GitHubAuthError)) as ei:
        client.search_issues("is:pr is:merged")
    err = ei.value
    assert getattr(err, "reason_code", None)
    msg = str(err)
    assert "token" not in msg.lower() or "rate" in msg.lower() or "403" in msg
    assert "gho_" not in msg
    assert "Bearer" not in msg


def test_rate_limit_retry_after_backoff_hook() -> None:
    """Secondary/rate-limit uses sleep hook with Retry-After then may retry (VAL-LMINE-010)."""
    sleeps: list[float] = []

    class Flaky:
        def __init__(self) -> None:
            self.n = 0

        def next_payload(self) -> Any:
            self.n += 1
            if self.n == 1:
                return MockHttpResponse(
                    status_code=429,
                    body={"message": "You have exceeded a secondary rate limit"},
                    headers={
                        "Retry-After": "3",
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Limit": "30",
                        "X-RateLimit-Reset": "1710000100",
                    },
                )
            return {
                "total_count": 1,
                "incomplete_results": False,
                "items": [{"number": 9, "title": "ok"}],
            }

    flaky = Flaky()
    transport = DictGitHubTransport(
        routes={},
        default=None,
        dynamic=lambda path, params: flaky.next_payload(),
    )
    client = GitHubClient(
        transport=transport,
        sleep_fn=lambda s: sleeps.append(float(s)),
        max_retries=2,
    )
    result = client.search_issues("is:pr is:merged")
    assert result["total_count"] == 1
    assert sleeps, "expected Retry-After backoff sleep"
    assert sleeps[0] == 3.0


def test_rate_limit_exhausts_retries_raises() -> None:
    transport = DictGitHubTransport(
        routes={
            "/search/issues": MockHttpResponse(
                status_code=429,
                body={"message": "secondary rate limit"},
                headers={"Retry-After": "10", "X-RateLimit-Remaining": "0"},
            )
        }
    )
    sleeps: list[float] = []
    client = GitHubClient(
        transport=transport,
        sleep_fn=lambda s: sleeps.append(float(s)),
        max_retries=1,
    )
    with pytest.raises(GitHubRateLimitError) as ei:
        client.search_issues("is:pr is:merged language:python")
    err = ei.value
    assert err.reason_code in {
        "rate_limited",
        "secondary_rate_limit",
        "http_429",
        "search_rate_limited",
    }
    assert err.rate_limit is not None
    assert err.rate_limit.retry_after == 10.0
    assert sleeps  # attempted backoff before fail closed


def test_token_never_appears_in_client_error_or_logs(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "gho_SUPER_SECRET_TOKEN_VALUE_NEVER_LOG"
    monkeypatch.setenv("GITHUB_TOKEN", secret)
    transport = DictGitHubTransport(
        routes={
            "/repos/owner/x/pulls/1": MockHttpResponse(
                status_code=401,
                body={"message": "Bad credentials"},
                headers={},
            )
        }
    )
    client = GitHubClient(transport=transport)
    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(GitHubError) as ei,
    ):
        client.get_pull("owner/x", 1)
    blob = str(ei.value) + "\n" + caplog.text
    assert secret not in blob
    assert "gho_SUPER" not in blob
    assert "Bearer" not in blob


def test_search_path_not_oxylabs_route() -> None:
    """Search route keys are GitHub REST paths, not realtime.oxylabs shapes."""
    transport = DictGitHubTransport(
        routes={
            "/search/issues": {
                "total_count": 0,
                "incomplete_results": False,
                "items": [],
            }
        }
    )
    client = GitHubClient(transport=transport)
    client.search_issues("is:pr is:merged")
    for call in transport.calls:
        assert "oxylabs" not in call.lower()
        assert call.startswith("/search/issues")
