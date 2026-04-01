"""Tests for GitHub API client."""

import json
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

from swe_forge.swe.github_api import (
    DiffTooLargeError,
    ForbiddenError,
    GitHubApiError,
    GitHubClient,
    NotFoundError,
    PRFile,
    PRState,
    PullRequest,
    RateLimitError,
    RateLimitInfo,
    ServerError,
    create_client,
)


def make_pr_response(number: int = 123, **kwargs) -> dict:
    defaults = {
        "number": number,
        "title": "Test PR",
        "body": "PR description",
        "state": "closed",
        "merged": True,
        "merged_at": "2023-01-01T12:00:00Z",
        "user": {"login": "testuser"},
        "base": {"sha": "abc123", "ref": "main"},
        "head": {"sha": "def456", "ref": "feature"},
        "additions": 10,
        "deletions": 5,
        "changed_files": 3,
    }
    defaults.update(kwargs)
    return defaults


def make_file_response(filename: str = "test.py", **kwargs) -> dict:
    defaults = {
        "filename": filename,
        "status": "modified",
        "additions": 5,
        "deletions": 2,
        "changes": 7,
        "raw_url": f"https://raw.githubusercontent.com/owner/repo/main/{filename}",
        "blob_url": f"https://github.com/owner/repo/blob/main/{filename}",
        "patch": "@@ -1,3 +1,5 @@\n+new line\n context\n+another new line",
    }
    defaults.update(kwargs)
    return defaults


class MockResponse:
    def __init__(
        self,
        status: int = 200,
        text_data: str = "",
        headers: dict | None = None,
    ) -> None:
        self.status = status
        self._text_data = text_data
        self.headers = headers or {
            "x-ratelimit-limit": "5000",
            "x-ratelimit-remaining": "4999",
            "x-ratelimit-reset": "1700000000",
            "x-ratelimit-used": "1",
        }

    async def __aenter__(self) -> "MockResponse":
        return self

    async def __aexit__(self, *args) -> None:
        pass

    async def text(self) -> str:
        return self._text_data


def create_mock_session(responses: list[MockResponse]) -> MagicMock:
    session = MagicMock(spec=aiohttp.ClientSession)
    session.headers = {}
    response_iter = iter(responses)

    def request_side_effect(*args, **kwargs):
        return next(response_iter)

    session.request = MagicMock(side_effect=request_side_effect)
    session.close = MagicMock()
    return session


class TestPullRequest:
    def test_from_api_response(self) -> None:
        data = make_pr_response()
        pr = PullRequest.from_api_response(data)

        assert pr.number == 123
        assert pr.title == "Test PR"
        assert pr.body == "PR description"
        assert pr.state == "closed"
        assert pr.merged is True
        assert pr.user_login == "testuser"
        assert pr.base_sha == "abc123"
        assert pr.head_sha == "def456"
        assert pr.additions == 10
        assert pr.deletions == 5
        assert pr.changed_files == 3

    def test_from_api_response_minimal(self) -> None:
        data = {"number": 1}
        pr = PullRequest.from_api_response(data)

        assert pr.number == 1
        assert pr.title == ""
        assert pr.body is None
        assert pr.state == "open"
        assert pr.merged is False

    def test_merged_at_parsing(self) -> None:
        data = make_pr_response(merged_at="2024-06-15T08:30:00Z")
        pr = PullRequest.from_api_response(data)

        assert pr.merged_at is not None
        assert pr.merged_at.year == 2024
        assert pr.merged_at.month == 6
        assert pr.merged_at.day == 15


class TestPRFile:
    def test_from_api_response(self) -> None:
        data = make_file_response()
        f = PRFile.from_api_response(data)

        assert f.filename == "test.py"
        assert f.status == "modified"
        assert f.additions == 5
        assert f.deletions == 2
        assert f.changes == 7

    def test_status_types(self) -> None:
        for status in ["added", "modified", "removed", "renamed"]:
            data = make_file_response(status=status)
            f = PRFile.from_api_response(data)
            assert f.status == status


class TestRateLimitInfo:
    def test_from_headers(self) -> None:
        headers = {
            "x-ratelimit-limit": "5000",
            "x-ratelimit-remaining": "4500",
            "x-ratelimit-reset": "1700000000",
            "x-ratelimit-used": "500",
        }
        info = RateLimitInfo.from_headers(headers)

        assert info.limit == 5000
        assert info.remaining == 4500
        assert info.reset_time == 1700000000
        assert info.used == 500

    def test_from_headers_defaults(self) -> None:
        info = RateLimitInfo.from_headers({})

        assert info.limit == 5000
        assert info.remaining == 5000


class TestGitHubClient:
    def test_init(self) -> None:
        client = GitHubClient(token="secret", timeout=60.0)

        assert client.token == "secret"
        assert client.timeout == 60.0
        assert client._session is None

    def test_headers(self) -> None:
        client = GitHubClient(token="test-token")
        headers = client._headers()

        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "swe-forge/1.0"

    def test_headers_diff(self) -> None:
        client = GitHubClient(token="test-token")
        headers = client._headers(accept="application/vnd.github.v3.diff")

        assert headers["Accept"] == "application/vnd.github.v3.diff"

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with GitHubClient(token="test") as client:
            assert client._session is not None
            assert isinstance(client._session, aiohttp.ClientSession)

    @pytest.mark.asyncio
    async def test_context_manager_closes(self) -> None:
        client = GitHubClient(token="test")
        async with client:
            pass
        assert client._session is None

    @pytest.mark.asyncio
    async def test_get_session_raises_without_context(self) -> None:
        client = GitHubClient(token="test")
        with pytest.raises(RuntimeError, match="context manager"):
            client._get_session()

    @pytest.mark.asyncio
    async def test_get_pr_success(self) -> None:
        pr_data = make_pr_response()
        mock_response = MockResponse(status=200, text_data=json.dumps(pr_data))
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        pr = await client.get_pr("owner", "repo", 123)

        assert pr.number == 123
        assert pr.title == "Test PR"

    @pytest.mark.asyncio
    async def test_get_pr_not_found(self) -> None:
        mock_response = MockResponse(status=404, text_data='{"message": "Not Found"}')
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        with pytest.raises(NotFoundError):
            await client.get_pr("owner", "repo", 999)

    @pytest.mark.asyncio
    async def test_get_pr_forbidden(self) -> None:
        mock_response = MockResponse(
            status=403,
            text_data='{"message": "Forbidden"}',
            headers={"x-ratelimit-remaining": "1"},
        )
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        with pytest.raises(ForbiddenError):
            await client.get_pr("owner", "repo", 123)

    @pytest.mark.asyncio
    async def test_get_pr_server_error(self) -> None:
        mock_response = MockResponse(status=500, text_data="Internal Server Error")
        # tenacity retries up to 3 times
        mock_session = create_mock_session(
            [mock_response, mock_response, mock_response]
        )

        client = GitHubClient(token="test-token")
        client._session = mock_session

        with pytest.raises(ServerError) as exc_info:
            await client.get_pr("owner", "repo", 123)

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_pr_files_success(self) -> None:
        files_data = [make_file_response("file1.py"), make_file_response("file2.py")]
        mock_response = MockResponse(status=200, text_data=json.dumps(files_data))
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        files = await client.get_pr_files("owner", "repo", 123)

        assert len(files) == 2
        assert files[0].filename == "file1.py"
        assert files[1].filename == "file2.py"

    @pytest.mark.asyncio
    async def test_get_pr_files_paging(self) -> None:
        page1 = [make_file_response(f"file{i}.py") for i in range(100)]
        page2 = [make_file_response("file100.py")]

        mock_session = create_mock_session(
            [
                MockResponse(status=200, text_data=json.dumps(page1)),
                MockResponse(status=200, text_data=json.dumps(page2)),
            ]
        )

        client = GitHubClient(token="test-token")
        client._session = mock_session

        files = await client.get_pr_files("owner", "repo", 123)

        assert len(files) == 101

    @pytest.mark.asyncio
    async def test_get_pr_files_empty(self) -> None:
        mock_response = MockResponse(status=200, text_data="[]")
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        files = await client.get_pr_files("owner", "repo", 123)

        assert files == []

    @pytest.mark.asyncio
    async def test_get_pr_diff_success(self) -> None:
        diff_text = "diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n"
        mock_response = MockResponse(status=200, text_data=diff_text)
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        diff = await client.get_pr_diff("owner", "repo", 123)

        assert diff == diff_text

    @pytest.mark.asyncio
    async def test_get_pr_diff_not_found(self) -> None:
        mock_response = MockResponse(status=404, text_data="Not Found")
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        with pytest.raises(NotFoundError):
            await client.get_pr_diff("owner", "repo", 999)

    @pytest.mark.asyncio
    async def test_get_pr_diff_too_large(self) -> None:
        mock_response = MockResponse(
            status=406, text_data='{"message": "diff exceeded 300 files"}'
        )
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        with pytest.raises(DiffTooLargeError):
            await client.get_pr_diff("owner", "repo", 123)

    @pytest.mark.asyncio
    async def test_get_rate_limit(self) -> None:
        rate_data = {
            "resources": {
                "core": {
                    "limit": 5000,
                    "remaining": 4999,
                    "reset": 1700000000,
                    "used": 1,
                }
            }
        }
        mock_response = MockResponse(status=200, text_data=json.dumps(rate_data))
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        info = await client.get_rate_limit()

        assert info.limit == 5000
        assert info.remaining == 4999

    @pytest.mark.asyncio
    async def test_rate_limit_info_updates(self) -> None:
        pr_data = make_pr_response()
        mock_response = MockResponse(
            status=200,
            text_data=json.dumps(pr_data),
            headers={
                "x-ratelimit-limit": "5000",
                "x-ratelimit-remaining": "4500",
                "x-ratelimit-reset": "1700000000",
                "x-ratelimit-used": "500",
            },
        )
        mock_session = create_mock_session([mock_response])

        client = GitHubClient(token="test-token")
        client._session = mock_session

        await client.get_pr("owner", "repo", 123)

        assert client.rate_limit_info is not None
        assert client.rate_limit_info.remaining == 4500


class TestRateLimitHandling:
    @pytest.mark.asyncio
    async def test_rate_limit_sleep_and_raise(self) -> None:
        now_ts = 1700000000

        with patch("swe_forge.swe.github_api.time.time", return_value=now_ts - 5):
            with patch("swe_forge.swe.github_api.asyncio.sleep"):
                rate_limited_response = MockResponse(
                    status=403,
                    text_data="",
                    headers={
                        "x-ratelimit-remaining": "0",
                        "x-ratelimit-reset": str(now_ts),
                    },
                )
                mock_session = create_mock_session(
                    [
                        rate_limited_response,
                        rate_limited_response,
                        rate_limited_response,
                        rate_limited_response,
                    ]
                )

                client = GitHubClient(token="test-token")
                client._session = mock_session

                with pytest.raises(RateLimitError) as exc_info:
                    await client.get_pr("owner", "repo", 123)

                assert exc_info.value.reset_time == now_ts


class TestClientFactory:
    @pytest.mark.asyncio
    async def test_create_client_with_token(self) -> None:
        client = await create_client("my-token")

        assert client.token == "my-token"

    @pytest.mark.asyncio
    async def test_create_client_from_env(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "env-token"}):
            client = await create_client()

            assert client.token == "env-token"

    @pytest.mark.asyncio
    async def test_create_client_no_token_raises(self) -> None:
        import os

        original = os.environ.get("GITHUB_TOKEN")
        if "GITHUB_TOKEN" in os.environ:
            del os.environ["GITHUB_TOKEN"]

        try:
            with pytest.raises(ValueError, match="GitHub token"):
                await create_client()
        finally:
            if original:
                os.environ["GITHUB_TOKEN"] = original


class TestExceptions:
    def test_github_api_error(self) -> None:
        err = GitHubApiError("test error", 400)

        assert str(err) == "test error"
        assert err.status_code == 400

    def test_rate_limit_error(self) -> None:
        err = RateLimitError(1700000000)

        assert "Rate limit exceeded" in str(err)
        assert err.reset_time == 1700000000

    def test_not_found_error(self) -> None:
        err = NotFoundError("PR not found")

        assert "not found" in str(err).lower()
        assert err.status_code == 404

    def test_forbidden_error(self) -> None:
        err = ForbiddenError("Access denied")

        assert err.status_code == 403

    def test_server_error(self) -> None:
        err = ServerError(502, "Bad Gateway")

        assert err.status_code == 502

    def test_diff_too_large_error(self) -> None:
        err = DiffTooLargeError("Diff exceeded 300 files")

        assert "diff" in str(err).lower()
        assert err.status_code == 406

    def test_diff_too_large_error_default_message(self) -> None:
        err = DiffTooLargeError()

        assert "too large" in str(err).lower()
        assert err.status_code == 406


class TestPRState:
    def test_values(self) -> None:
        assert PRState.OPEN == "open"
        assert PRState.CLOSED == "closed"
        assert PRState.ALL == "all"
