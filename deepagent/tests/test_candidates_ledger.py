"""Unit tests for durable real_pr candidates.jsonl ledger (VAL-LMINE-003/007, VAL-LX-003).

Offline DictGitHubTransport only unless marked integration. Covers:
- Stable candidates.jsonl schema (repo, pr_number, base_sha 40, language,
  file/hunk stats, license, discovery_path)
- Live keep/reject rows label discovery_path as search|list_pulls only
- offline_fixture path remains engineering-only and never claims product N
  or mislabels synthetic rows as search|list_pulls live paths
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from swe_factory.producers.pr_miner import PrFileChange
from swe_factory.sources.github import (
    DISCOVERY_PATH_LIST_PULLS,
    DISCOVERY_PATH_SEARCH,
    DISCOVERY_PATHS,
    DictGitHubTransport,
    GitHubClient,
)
from swe_factory.sources.real_pr_pool import (
    CANDIDATE_LEDGER_REQUIRED_FIELDS,
    DISCOVERY_PATH_OFFLINE_FIXTURE,
    RealPrPoolError,
    build_candidate_ledger_row,
    mine_live_merged_pr_pool,
    mine_offline_real_pr_pool,
    mine_real_pr_pool,
    write_candidates_jsonl,
)

_FULL_SHA = "b" * 40
_FULL_SHA2 = "c" * 40


def _source(path: str, patch: str = "@@ -1 +1 @@\n-a\n+b\n") -> PrFileChange:
    return PrFileChange(path=path, status="modified", patch=patch)


def _hard_files() -> list[dict[str, Any]]:
    """M27 product-eligible: ≥4 source files, ≥14 source hunks + tests.

    Pre-M27 mocks used 2 files / 11 hunks (old ≥10 floor) and now fail
    PRODUCT_MULTI_FILE_FLOOR=4 and PRODUCT_SOURCE_HUNK_FLOOR=14.
    """
    # 4 + 4 + 3 + 3 = 14 source hunks across four product files
    hunks_a = "".join(f"@@ -{i},1 +{i},1 @@\n-old_a{i}\n+new_a{i}\n" for i in range(1, 5))
    hunks_b = "".join(f"@@ -{i},1 +{i},1 @@\n-old_b{i}\n+new_b{i}\n" for i in range(1, 5))
    hunks_c = "".join(f"@@ -{i},1 +{i},1 @@\n-old_c{i}\n+new_c{i}\n" for i in range(1, 4))
    hunks_d = "".join(f"@@ -{i},1 +{i},1 @@\n-old_d{i}\n+new_d{i}\n" for i in range(1, 4))
    return [
        {"filename": "pkg/a.py", "status": "modified", "patch": hunks_a},
        {"filename": "pkg/b.py", "status": "modified", "patch": hunks_b},
        {"filename": "pkg/c.py", "status": "modified", "patch": hunks_c},
        {"filename": "pkg/d.py", "status": "modified", "patch": hunks_d},
        {
            "filename": "tests/test_a.py",
            "status": "added",
            "patch": "@@ -0,0 +1,2 @@\n+def test_a():\n+    assert True\n",
        },
    ]


def _mock_list_pulls_routes() -> dict[str, Any]:
    files = _hard_files()
    return {
        "/repos/owner/demo/pulls": [
            {
                "number": 42,
                "title": "Multi-file hard PR",
                "body": "body",
                "merged_at": "2026-01-02T00:00:00Z",
                "merge_commit_sha": "m" * 40,
                "html_url": "https://github.com/owner/demo/pull/42",
                "base": {"sha": _FULL_SHA, "ref": "main"},
            },
            {
                "number": 7,
                "title": "WIP not merged",
                "body": "",
                "merged_at": None,
                "base": {"sha": _FULL_SHA2},
            },
        ],
        "/repos/owner/demo/pulls/42": {
            "number": 42,
            "title": "Multi-file hard PR",
            "body": "body",
            "merged_at": "2026-01-02T00:00:00Z",
            "merge_commit_sha": "m" * 40,
            "html_url": "https://github.com/owner/demo/pull/42",
            "base": {"sha": _FULL_SHA, "ref": "main"},
        },
        "/repos/owner/demo/pulls/42/files": files,
        "/repos/owner/demo/pulls/7": {
            "number": 7,
            "title": "WIP",
            "body": "",
            "merged_at": None,
            "base": {"sha": _FULL_SHA2},
        },
        "/repos/owner/demo/pulls/7/files": [],
    }


def _mock_search_routes() -> dict[str, Any]:
    files = _hard_files()
    return {
        "/search/issues": {
            "total_count": 1,
            "incomplete_results": False,
            "items": [
                {
                    "number": 99,
                    "title": "Search-found PR",
                    "body": "from search",
                    "html_url": "https://github.com/acme/lib/pull/99",
                    "pull_request": {
                        "url": "https://api.github.com/repos/acme/lib/pulls/99",
                        "merged_at": "2026-03-01T00:00:00Z",
                    },
                    "repository_url": "https://api.github.com/repos/acme/lib",
                }
            ],
        },
        "/repos/acme/lib/pulls/99": {
            "number": 99,
            "title": "Search-found PR",
            "body": "from search",
            "merged_at": "2026-03-01T00:00:00Z",
            "merge_commit_sha": "d" * 40,
            "html_url": "https://github.com/acme/lib/pull/99",
            "base": {"sha": "e" * 40, "ref": "main"},
        },
        "/repos/acme/lib/pulls/99/files": files,
    }


def test_build_candidate_ledger_row_schema_stable() -> None:
    row = build_candidate_ledger_row(
        repo="owner/demo",
        pr_number=42,
        base_sha=_FULL_SHA,
        language="python",
        license="MIT",
        discovery_path=DISCOVERY_PATH_LIST_PULLS,
        source_hunk_count=12,
        source_file_count=2,
        test_file_count=1,
        disposition="accept",
    )
    for field in CANDIDATE_LEDGER_REQUIRED_FIELDS:
        assert field in row, f"missing required field {field}"
    assert row["repo"] == "owner/demo"
    assert row["pr_number"] == 42
    assert row["base_sha"] == _FULL_SHA
    assert len(row["base_sha"]) == 40
    assert row["language"] == "python"
    assert row["license"] == "MIT"
    assert row["discovery_path"] == DISCOVERY_PATH_LIST_PULLS
    assert row["source_hunk_count"] == 12
    assert row["source_file_count"] == 2
    assert row["test_file_count"] == 1
    assert row["disposition"] == "accept"


def test_live_list_pulls_writes_candidates_jsonl_with_discovery_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mine via list_pulls emits durable rows with discovery_path=list_pulls."""
    from swe_factory.sources.allowlist import SeedRepo

    seed = SeedRepo(
        seed_id="test_owner_demo",
        language="python",
        repo="owner/demo",
        base_commit=_FULL_SHA,
        license="MIT",
        mine_priority=1,
    )
    monkeypatch.setattr(
        "swe_factory.sources.real_pr_pool.real_pr_seed_pool",
        lambda **_kwargs: [seed],
    )

    client = GitHubClient(transport=DictGitHubTransport(routes=_mock_list_pulls_routes()))
    report = mine_live_merged_pr_pool(
        client,
        work_root=tmp_path / "live_pool",
        seed_ids=["test_owner_demo"],
        target_candidates=1,
        max_scan_per_repo=5,
        max_keeps_per_repo=1,
        max_seeds=1,
        discovery_paths=[DISCOVERY_PATH_LIST_PULLS],
        product_mode=True,
        require_token=False,
    )
    assert report.mode == "live_github_rest"
    assert report.offline is False
    assert report.network_required is True
    assert report.product_n_evidence is True
    ledger_path = tmp_path / "live_pool" / "candidates.jsonl"
    assert ledger_path.is_file()
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    assert rows, "candidates.jsonl must not be empty for live discovery"
    live_rows = [r for r in rows if r.get("disposition") == "accept"]
    assert live_rows, "expected at least one accepted live candidate"
    for row in live_rows:
        assert row["discovery_path"] == DISCOVERY_PATH_LIST_PULLS
        assert row["discovery_path"] in DISCOVERY_PATHS
        assert len(str(row["base_sha"])) == 40
        assert row["repo"]
        assert row["pr_number"]
        assert "source_hunk_count" in row
        assert "license" in row
        assert "language" in row
        assert "source_file_count" in row or "gold_files" in row


def test_live_search_path_labels_discovery_path_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Search discovery path stamps discovery_path=search on ledger rows."""
    from swe_factory.sources.allowlist import SeedRepo

    seed = SeedRepo(
        seed_id="test_acme_lib",
        language="python",
        repo="acme/lib",
        base_commit="e" * 40,
        license="Apache-2.0",
        mine_priority=1,
    )
    monkeypatch.setattr(
        "swe_factory.sources.real_pr_pool.real_pr_seed_pool",
        lambda **_kwargs: [seed],
    )
    client = GitHubClient(transport=DictGitHubTransport(routes=_mock_search_routes()))
    report = mine_live_merged_pr_pool(
        client,
        work_root=tmp_path / "search_pool",
        seed_ids=["test_acme_lib"],
        target_candidates=1,
        max_scan_per_repo=5,
        max_keeps_per_repo=1,
        max_seeds=1,
        discovery_paths=[DISCOVERY_PATH_SEARCH],
        product_mode=True,
        require_token=False,
    )
    assert report.keep_count >= 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "search_pool" / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    accepted = [r for r in rows if r.get("disposition") == "accept"]
    assert accepted
    for row in accepted:
        assert row["discovery_path"] == DISCOVERY_PATH_SEARCH
        assert row["discovery_path"] in DISCOVERY_PATHS
        assert len(str(row["base_sha"])) == 40
        assert row["source_hunk_count"] >= 10


def test_offline_fixture_candidates_never_claim_live_discovery_path(tmp_path: Path) -> None:
    """offline_fixture rows must not use discovery_path=search|list_pulls (VAL-LMINE-007)."""
    report = mine_offline_real_pr_pool(
        work_root=tmp_path / "offline_pool",
        target_candidates=3,
        synthetic_repo_diversity=3,
    )
    assert report.mode == "offline_fixture"
    assert report.offline is True
    assert report.product_n_evidence is False
    assert report.engineering_only is True
    payload = report.to_dict()
    assert payload["product_n_evidence"] is False
    assert payload["engineering_only"] is True
    assert payload["mode"] == "offline_fixture"
    # product N must never be justified from offline mode
    assert payload.get("honesty", {}).get("product_n_from_offline_fixture") is False

    ledger_path = tmp_path / "offline_pool" / "candidates.jsonl"
    assert ledger_path.is_file()
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    assert rows
    for row in rows:
        path = row.get("discovery_path")
        assert path not in DISCOVERY_PATHS, (
            f"offline_fixture must not claim live discovery_path; got {path!r}"
        )
        assert path == DISCOVERY_PATH_OFFLINE_FIXTURE
        # Explicit engineering label preferred
        assert row.get("product_n_evidence") is False
        assert row.get("engineering_only") is True


def test_write_candidates_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [
        build_candidate_ledger_row(
            repo="a/b",
            pr_number=1,
            base_sha=_FULL_SHA,
            language="go",
            license="BSD-3-Clause",
            discovery_path=DISCOVERY_PATH_SEARCH,
            source_hunk_count=15,
            source_file_count=3,
            test_file_count=2,
            disposition="accept",
        ),
        build_candidate_ledger_row(
            repo="a/b",
            pr_number=2,
            base_sha=_FULL_SHA2,
            language="go",
            license="MIT",
            discovery_path=DISCOVERY_PATH_LIST_PULLS,
            source_hunk_count=3,
            source_file_count=1,
            test_file_count=0,
            disposition="reject",
            reason_code="source_hunks_below_floor",
        ),
    ]
    path = write_candidates_jsonl(rows, tmp_path / "candidates.jsonl")
    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(loaded) == 2
    assert loaded[0]["discovery_path"] == DISCOVERY_PATH_SEARCH
    assert loaded[1]["discovery_path"] == DISCOVERY_PATH_LIST_PULLS
    assert loaded[1]["disposition"] == "reject"


def test_mine_real_pr_pool_live_requires_token_when_flagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Live path fail-closed when require_token and no resolvable token."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    # Patch the name bound in real_pr_pool (imported at module load).
    monkeypatch.setattr(
        "swe_factory.sources.real_pr_pool.resolve_github_token",
        lambda _explicit=None: None,
    )
    monkeypatch.setattr(
        "swe_factory.sources.github.resolve_github_token",
        lambda _explicit=None: None,
    )
    with pytest.raises(RealPrPoolError, match="token|GITHUB_TOKEN|auth"):
        mine_real_pr_pool(
            work_root=tmp_path / "no_token",
            offline=False,
            target_candidates=1,
            require_token=True,
        )


def test_cli_offline_fixture_report_not_product_n(tmp_path: Path) -> None:
    """CLI offline mode report must not claim product N evidence."""
    from typer.testing import CliRunner

    from swe_factory.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "real-pr-pool",
            "--offline-fixture",
            "--target",
            "3",
            "--json",
            "--out",
            str(tmp_path / "cli_offline"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "offline_fixture"
    assert payload.get("product_n_evidence") is False
    assert payload.get("engineering_only") is True
    ledger = tmp_path / "cli_offline" / "candidates.jsonl"
    assert ledger.is_file()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert row.get("discovery_path") not in DISCOVERY_PATHS


def test_cli_live_without_token_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from swe_factory.cli import app

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(
        "swe_factory.sources.github.resolve_github_token",
        lambda _explicit=None: None,
    )
    # settings may still load empty github_token
    monkeypatch.setattr(
        "swe_factory.cli.load_settings",
        lambda: type(
            "S",
            (),
            {
                "github_token": None,
            },
        )(),
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "real-pr-pool",
            "--live",
            "--target",
            "1",
            "--max-scan",
            "2",
            "--json",
            "--out",
            str(tmp_path / "cli_live_no_token"),
        ],
    )
    assert result.exit_code != 0
    assert "token" in (result.output + result.stdout + str(result.exception)).lower()
