"""Tests for GH Archive client."""

import gzip
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.swe.gharchive import (
    GH_ARCHIVE_BASE_URL,
    GhArchiveClient,
    GhArchiveError,
    GhArchiveEvent,
    GhArchiveNotFoundError,
)


def make_pr_event(
    repo: str = "owner/repo",
    number: int = 123,
    merged: bool = True,
    action: str = "closed",
    actor: str = "testuser",
    merged_at: str = "2023-01-01T12:00:00Z",
) -> dict:
    """Create a test PullRequestEvent."""
    return {
        "id": "1234567890",
        "type": "PullRequestEvent",
        "actor": {"login": actor},
        "repo": {"name": repo},
        "payload": {
            "action": action,
            "pull_request": {
                "number": number,
                "merged": merged,
                "merged_at": merged_at,
                "title": f"PR #{number}: Test PR",
                "body": "Test PR body",
                "user": {"login": actor},
                "base": {"ref": "main", "sha": "abc123"},
                "head": {"ref": "feature-branch", "sha": "def456"},
                "merge_commit_sha": "merged_sha_123",
            },
            "repository": {"watchers_count": 100},
        },
        "created_at": merged_at,
        "org": {"login": "testorg"},
    }


def make_issues_event() -> dict:
    """Create a test IssuesEvent."""
    return {
        "id": "9876543210",
        "type": "IssuesEvent",
        "actor": {"login": "testuser"},
        "repo": {"name": "owner/repo"},
        "payload": {"action": "opened", "issue": {"number": 456}},
        "created_at": "2023-01-01T12:00:00Z",
    }


class TestGhArchiveEvent:
    def test_create_event(self):
        event = GhArchiveEvent(
            id="evt-123",
            event_type="PullRequestEvent",
            repository="owner/repo",
            actor="testuser",
            action="merged",
            pull_number=123,
            created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        assert event.id == "evt-123"
        assert event.event_type == "PullRequestEvent"
        assert event.repository == "owner/repo"
        assert event.action == "merged"

    def test_defaults(self):
        event = GhArchiveEvent(
            id="evt-123",
            event_type="PullRequestEvent",
            repository="owner/repo",
            actor="testuser",
            action="merged",
            pull_number=123,
            created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        assert event.base_sha == ""
        assert event.merge_sha == ""
        assert event.title == ""
        assert event.body == ""
        assert event.stars == 0


class TestGhArchiveClientInit:
    def test_default_init(self):
        client = GhArchiveClient()
        assert client.token is None
        assert client.timeout == 60
        assert client.max_retries == 3

    def test_custom_init(self):
        client = GhArchiveClient(token="test_token", timeout=30, max_retries=5)
        assert client.token == "test_token"
        assert client.timeout == 30
        assert client.max_retries == 5


class TestGhArchiveClientBuildUrl:
    def test_build_hour_url(self):
        client = GhArchiveClient()
        date = datetime(2023, 1, 15, tzinfo=timezone.utc)
        url = client._build_hour_url(date, 10)
        assert url == f"{GH_ARCHIVE_BASE_URL}/2023-01-15-10.json.gz"

    def test_build_hour_url_midnight(self):
        client = GhArchiveClient()
        date = datetime(2023, 1, 15, tzinfo=timezone.utc)
        url = client._build_hour_url(date, 0)
        assert url == f"{GH_ARCHIVE_BASE_URL}/2023-01-15-0.json.gz"

    def test_build_hour_url_23rd_hour(self):
        client = GhArchiveClient()
        date = datetime(2023, 1, 15, tzinfo=timezone.utc)
        url = client._build_hour_url(date, 23)
        assert url == f"{GH_ARCHIVE_BASE_URL}/2023-01-15-23.json.gz"


class TestGhArchiveClientParseEvents:
    def test_parse_events_single_line(self):
        client = GhArchiveClient()
        event = make_pr_event()
        data = json.dumps(event).encode("utf-8")
        events = client.parse_events(data)
        assert len(events) == 1
        assert events[0]["type"] == "PullRequestEvent"

    def test_parse_events_multiple_lines(self):
        client = GhArchiveClient()
        event1 = make_pr_event(number=1)
        event2 = make_pr_event(number=2)
        data = f"{json.dumps(event1)}\n{json.dumps(event2)}".encode("utf-8")
        events = client.parse_events(data)
        assert len(events) == 2

    def test_parse_events_handles_empty_lines(self):
        client = GhArchiveClient()
        event = make_pr_event()
        data = f"\n{json.dumps(event)}\n\n".encode("utf-8")
        events = client.parse_events(data)
        assert len(events) == 1

    def test_parse_events_handles_malformed_json(self):
        client = GhArchiveClient()
        event = make_pr_event()
        data = f"{json.dumps(event)}\n{{invalid json}}\n{json.dumps(event)}".encode(
            "utf-8"
        )
        events = client.parse_events(data)
        assert len(events) == 2


class TestGhArchiveClientFilterMergedPRs:
    def test_filters_merged_prs(self):
        client = GhArchiveClient()
        events = [make_pr_event(merged=True), make_pr_event(merged=False)]
        merged = client.filter_merged_prs(events)
        assert len(merged) == 1
        assert merged[0].pull_number == 123

    def test_filters_by_event_type(self):
        client = GhArchiveClient()
        events = [make_pr_event(), make_issues_event()]
        merged = client.filter_merged_prs(events)
        assert len(merged) == 1

    def test_filters_by_action_closed(self):
        client = GhArchiveClient()
        events = [
            make_pr_event(action="closed", merged=True),
            make_pr_event(action="opened", merged=False),
        ]
        merged = client.filter_merged_prs(events)
        assert len(merged) == 1
        assert merged[0].action == "merged"

    def test_returns_empty_for_no_merged_prs(self):
        client = GhArchiveClient()
        events = [
            make_pr_event(merged=False),
            make_pr_event(action="opened"),
            make_issues_event(),
        ]
        merged = client.filter_merged_prs(events)
        assert len(merged) == 0


class TestGhArchiveClientParsePREvent:
    def test_parse_basic_pr_event(self):
        client = GhArchiveClient()
        event = make_pr_event()
        parsed = client._parse_pr_event(event)
        assert parsed is not None
        assert parsed.event_type == "PullRequestEvent"
        assert parsed.repository == "owner/repo"
        assert parsed.pull_number == 123
        assert parsed.action == "merged"
        assert parsed.actor == "testuser"
        assert parsed.user == "testuser"

    def test_parse_pr_event_with_all_fields(self):
        client = GhArchiveClient()
        event = make_pr_event()
        parsed = client._parse_pr_event(event)
        assert parsed is not None
        assert parsed.title == "PR #123: Test PR"
        assert parsed.body == "Test PR body"
        assert parsed.base_ref == "main"
        assert parsed.head_ref == "feature-branch"
        assert parsed.merge_sha == "merged_sha_123"
        assert parsed.stars == 100
        assert parsed.has_org is True

    def test_parse_pr_event_handles_missing_fields(self):
        client = GhArchiveClient()
        event = {
            "id": "123",
            "type": "PullRequestEvent",
            "repo": {"name": "owner/repo"},
            "actor": {"login": "testuser"},
            "payload": {
                "action": "closed",
                "pull_request": {
                    "number": 123,
                    "merged": True,
                    "merged_at": "2023-01-01T12:00:00Z",
                },
            },
            "created_at": "2023-01-01T12:00:00Z",
        }
        parsed = client._parse_pr_event(event)
        assert parsed is not None
        assert parsed.repository == "owner/repo"
        assert parsed.title == "Untitled change"
        assert parsed.body == ""


class TestGhArchiveClientFetchHour:
    @pytest.mark.asyncio
    async def test_fetch_hour_success(self):
        client = GhArchiveClient()
        event = make_pr_event()
        compressed = gzip.compress(json.dumps(event).encode("utf-8"))

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=compressed)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch.object(client, "_ensure_session", return_value=mock_session):
            result = await client.fetch_hour(datetime(2023, 1, 1), 0)
            assert result == json.dumps(event).encode("utf-8")

    @pytest.mark.asyncio
    async def test_fetch_hour_404_raises_not_found(self):
        client = GhArchiveClient()

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch.object(client, "_ensure_session", return_value=mock_session):
            with pytest.raises(GhArchiveNotFoundError):
                await client.fetch_hour(datetime(2023, 1, 1), 0)

    @pytest.mark.asyncio
    async def test_fetch_hour_non_200_raises_error(self):
        client = GhArchiveClient()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch.object(client, "_ensure_session", return_value=mock_session):
            with pytest.raises(GhArchiveError):
                await client.fetch_hour(datetime(2023, 1, 1), 0)


class TestGhArchiveClientFetchRange:
    @pytest.mark.asyncio
    async def test_fetch_range_single_hour(self):
        client = GhArchiveClient()

        async def mock_fetch_hour(date, hour):
            event = make_pr_event()
            return json.dumps(event).encode("utf-8")

        with patch.object(client, "fetch_hour", side_effect=mock_fetch_hour):
            start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            results = await client.fetch_range(start, end)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_fetch_range_skips_missing(self):
        client = GhArchiveClient()
        event = make_pr_event()
        compressed = gzip.compress(json.dumps(event).encode("utf-8"))

        call_count = 0

        async def mock_fetch_hour(date, hour):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise GhArchiveNotFoundError("Not found")
            return compressed

        with patch.object(client, "fetch_hour", side_effect=mock_fetch_hour):
            start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end = datetime(2023, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
            results = await client.fetch_range(start, end, skip_missing=True)
            # First hour missing, second hour OK
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_fetch_range_raises_on_missing_if_skip_false(self):
        client = GhArchiveClient()

        async def mock_fetch_hour(date, hour):
            raise GhArchiveNotFoundError("Not found")

        with patch.object(client, "fetch_hour", side_effect=mock_fetch_hour):
            start = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            with pytest.raises(GhArchiveNotFoundError):
                await client.fetch_range(start, end, skip_missing=False)


class TestGhArchiveClientParseDatetime:
    def test_parse_datetime_with_z(self):
        client = GhArchiveClient()
        result = client._parse_datetime("2023-01-01T12:00:00Z")
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12

    def test_parse_datetime_with_timezone(self):
        client = GhArchiveClient()
        result = client._parse_datetime("2023-01-01T12:00:00+00:00")
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12


class TestGhArchiveClientContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with GhArchiveClient() as client:
            assert client._session is not None
        assert client._session is None or client._session.closed

    @pytest.mark.asyncio
    async def test_close_can_be_called_multiple_times(self):
        client = GhArchiveClient()
        await client.close()
        await client.close()


class TestGhArchiveClientFetchMergedPRs:
    @pytest.mark.asyncio
    async def test_fetch_merged_prs(self):
        client = GhArchiveClient()
        merged_pr = make_pr_event(merged=True)
        closed_pr = make_pr_event(merged=False, number=456)
        data = f"{json.dumps(merged_pr)}\n{json.dumps(closed_pr)}".encode("utf-8")

        async def mock_fetch_range(start, end, skip_missing=True):
            return [data]

        with patch.object(client, "fetch_range", side_effect=mock_fetch_range):
            start = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end = datetime(2023, 1, 1, tzinfo=timezone.utc)
            results = await client.fetch_merged_prs(start, end)
            assert len(results) == 1
            assert results[0].action == "merged"


class TestGhArchiveClientSession:
    @pytest.mark.asyncio
    async def test_ensure_session_creates_new(self):
        client = GhArchiveClient()
        assert client._session is None
        session = await client._ensure_session()
        assert session is not None
        assert client._session is session
        await client.close()

    @pytest.mark.asyncio
    async def test_ensure_session_reuses_existing(self):
        client = GhArchiveClient()
        session1 = await client._ensure_session()
        session2 = await client._ensure_session()
        assert session1 is session2
        await client.close()


class TestGhArchiveNotFoundError:
    def test_is_gh_archive_error(self):
        assert issubclass(GhArchiveNotFoundError, GhArchiveError)

    def test_message(self):
        error = GhArchiveNotFoundError("test message")
        assert str(error) == "test message"


class TestGhArchiveError:
    def test_is_exception(self):
        assert issubclass(GhArchiveError, Exception)

    def test_message(self):
        error = GhArchiveError("test error")
        assert str(error) == "test error"
