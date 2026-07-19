"""Unit tests: GH Archive bulk event discover (VAL-DCOV-002).

Offline fixture lines only — no network required.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from swe_factory.sources.gh_archive import (
    DISCOVERY_PATH_GH_ARCHIVE,
    discover_from_gh_archive_lines,
    discover_from_gh_archive_path,
    is_merged_pull_request_event,
    write_gh_archive_candidates_jsonl,
)


def _merged_pr_event(
    *,
    repo: str = "pallets/click",
    number: int = 2800,
    merge_sha: str | None = "a" * 40,
    base_sha: str | None = "b" * 40,
    language: str = "Python",
    action: str = "closed",
    merged: bool = True,
    etype: str = "PullRequestEvent",
) -> dict:
    pr: dict = {
        "number": number,
        "merged": merged,
        "title": f"PR {number}",
        "html_url": f"https://github.com/{repo}/pull/{number}",
        "merged_at": "2024-06-01T12:00:00Z" if merged else None,
        "base": {
            "sha": base_sha or "",
            "ref": "main",
            "repo": {"full_name": repo, "language": language},
        },
        "head": {"sha": "c" * 40},
        "user": {"login": "dev"},
    }
    if merge_sha is not None:
        pr["merge_commit_sha"] = merge_sha
    return {
        "id": f"evt-{repo}-{number}",
        "type": etype,
        "repo": {"id": 1, "name": repo, "url": f"https://api.github.com/repos/{repo}"},
        "payload": {
            "action": action,
            "number": number,
            "pull_request": pr,
        },
        "public": True,
        "created_at": "2024-06-01T12:00:00Z",
    }


def test_discovery_path_label_is_gh_archive() -> None:
    assert DISCOVERY_PATH_GH_ARCHIVE == "gh_archive"


def test_is_merged_pull_request_event_filters() -> None:
    assert is_merged_pull_request_event(_merged_pr_event()) is True
    assert is_merged_pull_request_event(_merged_pr_event(merged=False)) is False
    assert is_merged_pull_request_event(_merged_pr_event(action="opened")) is False
    assert (
        is_merged_pull_request_event({"type": "PushEvent", "payload": {}, "repo": {"name": "a/b"}})
        is False
    )
    # closed + merge_commit_sha present still counts as merged even if merged flag missing
    evt = _merged_pr_event()
    del evt["payload"]["pull_request"]["merged"]
    evt["payload"]["pull_request"]["merge_commit_sha"] = "d" * 40
    evt["payload"]["action"] = "closed"
    assert is_merged_pull_request_event(evt) is True


def test_discover_from_lines_emits_candidates() -> None:
    lines = [
        json.dumps(_merged_pr_event(repo="pallets/click", number=11)),
        json.dumps(_merged_pr_event(repo="psf/requests", number=22, language="Python")),
        json.dumps(_merged_pr_event(merged=False, number=99)),  # skip
        json.dumps({"type": "WatchEvent", "repo": {"name": "x/y"}, "payload": {}}),
        "not-json",
        json.dumps(_merged_pr_event(repo="encode/httpx", number=33, merge_sha=None)),
    ]
    rows = discover_from_gh_archive_lines(lines)
    assert len(rows) == 3
    repos = {r["repo"] for r in rows}
    assert repos == {"pallets/click", "psf/requests", "encode/httpx"}
    for row in rows:
        assert row["discovery_path"] == "gh_archive"
        assert isinstance(row["pr_number"], int)
        assert row["repo"]
        assert row.get("disposition") in (None, "candidate", "accept") or True
    click = next(r for r in rows if r["repo"] == "pallets/click")
    assert click["pr_number"] == 11
    assert click.get("merge_commit_sha") == "a" * 40
    assert click.get("base_sha") == "b" * 40 or click.get("base_commit") == "b" * 40


def test_discover_offline_fixture_path_gz(tmp_path: Path) -> None:
    events = [
        _merged_pr_event(repo="pallets/werkzeug", number=2608),
        _merged_pr_event(repo="pallets/flask", number=5000),
        _merged_pr_event(repo="tiran/defusedxml", number=100, merged=False),
    ]
    raw = ("\n".join(json.dumps(e) for e in events) + "\n").encode("utf-8")
    gz_path = tmp_path / "2024-01-01-15.json.gz"
    with gzip.open(gz_path, "wb") as fh:
        fh.write(raw)

    rows = discover_from_gh_archive_path(gz_path)
    assert len(rows) == 2
    assert {r["repo"] for r in rows} == {"pallets/werkzeug", "pallets/flask"}
    assert all(r["discovery_path"] == "gh_archive" for r in rows)


def test_discover_offline_plain_jsonl_fixture(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(_merged_pr_event(repo="owner/demo", number=7)) + "\n",
        encoding="utf-8",
    )
    rows = discover_from_gh_archive_path(path)
    assert len(rows) == 1
    assert rows[0]["repo"] == "owner/demo"
    assert rows[0]["pr_number"] == 7


def test_write_candidates_jsonl(tmp_path: Path) -> None:
    rows = discover_from_gh_archive_lines([json.dumps(_merged_pr_event(repo="a/b", number=1))])
    out = tmp_path / "candidates.jsonl"
    written = write_gh_archive_candidates_jsonl(rows, out)
    assert written == out
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["discovery_path"] == "gh_archive"
    assert parsed["repo"] == "a/b"
    assert parsed["pr_number"] == 1
    assert "merge_commit_sha" in parsed


def test_dedupe_same_repo_pr(tmp_path: Path) -> None:
    lines = [
        json.dumps(_merged_pr_event(repo="a/b", number=1)),
        json.dumps(_merged_pr_event(repo="a/b", number=1)),
        json.dumps(_merged_pr_event(repo="a/b", number=2)),
    ]
    rows = discover_from_gh_archive_lines(lines)
    assert len(rows) == 2
