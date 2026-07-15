"""Unit tests for Real-PR merged-PR-only pool (VAL-RPR-002..005).

Offline only (no network). Covers:
- Offline real_pr fixture still produces valid TaskRecord
- Reject single-file / no-test PRs
- Reject harbor motors / hybrid_curated as candidates
- Emit candidate pool suitable to later select ≥5 repos
- Gold source-only + held-out test_patch
- Honesty: hybrid never mis-labeled real_pr
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from swe_factory.oracle.gates import count_files_in_patch
from swe_factory.producers.pr_miner import (
    PrFileChange,
    PrMineError,
    PrMiner,
    extract_gold_from_files,
    is_test_path,
    multi_file_source_filter,
    produce_offline_fixture,
)
from swe_factory.schema import SourceTrack
from swe_factory.sources.allowlist import HARBOR_MOTOR_SEEDS
from swe_factory.sources.discover import DiscoverError, sanitize_api_candidate
from swe_factory.sources.github import DictGitHubTransport, GitHubClient
from swe_factory.sources.real_pr_pool import (
    RealPrPoolError,
    assert_not_motor_or_hybrid,
    is_motor_or_hybrid_identity,
    mine_offline_real_pr_pool,
    real_pr_seed_pool,
    reject_hybrid_motor_discover_attempt,
    sanitize_real_pr_api_candidate,
    summary_seed_pool_for_select5,
)


def _source(path: str, patch: str = "@@ -1 +1 @@\n-a\n+b\n") -> PrFileChange:
    return PrFileChange(path=path, status="modified", patch=patch)


def _test(path: str, patch: str = "@@ -0,0 +1 @@\n+assert True\n") -> PrFileChange:
    return PrFileChange(path=path, status="modified", patch=patch)


_FULL_SHA = "b" * 40


# --- VAL-RPR-002: merged-PR only / multi-file / reject motors ---


def test_offline_fixture_still_valid_taskrecord(tmp_path: Path) -> None:
    candidate = produce_offline_fixture(work_root=tmp_path / "fx", run_stub_oracle=True)
    assert candidate.task.source_track == SourceTrack.REAL_PR
    assert candidate.source_track == "real_pr"
    assert candidate.task.base_commit
    assert len(candidate.gold_files) >= 2
    assert candidate.test_patch.strip()
    assert candidate.repository_url
    assert all(not is_test_path(p) for p in candidate.gold_files)


def test_rejects_single_file_and_no_test_prs() -> None:
    single = [_source("only.py"), _test("tests/t.py")]
    assert multi_file_source_filter(single) is False
    no_tests = [_source("a.py"), _source("b.py")]
    assert multi_file_source_filter(no_tests, require_tests=True) is False
    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client)
    with pytest.raises(PrMineError, match="multi-file"):
        miner.select_from_files(
            repo="owner/demo",
            number=1,
            title="single",
            body="",
            base_commit=_FULL_SHA,
            merge_commit_sha=None,
            html_url="",
            files_payload=[
                {
                    "filename": "only.py",
                    "status": "modified",
                    "patch": "@@ -1 +1 @@\n-a\n+b\n",
                },
                {
                    "filename": "tests/test_only.py",
                    "status": "modified",
                    "patch": "@@ -0,0 +1 @@\n+x\n",
                },
            ],
        )


def test_pr_miner_rejects_motor_repo_identity() -> None:
    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client)
    with pytest.raises(PrMineError, match="motor_or_hybrid"):
        miner.select_from_files(
            repo="fixtures/harbor_motors/python_orders",
            number=1,
            title="motor",
            body="",
            base_commit=_FULL_SHA,
            merge_commit_sha=None,
            html_url="",
            files_payload=[
                {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b\n"},
                {"filename": "b.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b\n"},
                {
                    "filename": "tests/t.py",
                    "status": "modified",
                    "patch": "@@ -0,0 +1 @@\n+x\n",
                },
            ],
        )


def test_is_motor_or_hybrid_identity_blocks_harbor_seeds() -> None:
    for motor in HARBOR_MOTOR_SEEDS:
        banned, reason = is_motor_or_hybrid_identity(motor.repo, seed_id=motor.seed_id)
        assert banned is True
        assert "motor" in reason.lower() or "hybrid" in reason.lower() or "harbor" in reason.lower()
    with pytest.raises(RealPrPoolError, match="motor_or_hybrid"):
        assert_not_motor_or_hybrid(
            "fixtures/harbor_motors/python_orders",
            seed_id="harbor_python_orders",
        )
    banned_track, _ = is_motor_or_hybrid_identity("mahmoud/boltons", source_track="hybrid_curated")
    assert banned_track is True


def test_discover_real_pr_sanitizer_rejects_motors() -> None:
    files = [_source("a.py"), _source("b.py"), _test("tests/t.py")]
    with pytest.raises(DiscoverError, match="motor_or_hybrid|repository_url|real_pr"):
        sanitize_api_candidate(
            repo="fixtures/harbor_motors/python_orders",
            repository_url="file://fixtures/harbor_motors/python_orders",
            base_commit=_FULL_SHA,
            files=files,
            license="MIT",
            kind="real_pr",
        )
    # public multi-file still accepted
    ok = sanitize_real_pr_api_candidate(
        repo="owner/demo",
        base_commit=_FULL_SHA,
        files=files,
        license="MIT",
        language="python",
        pr_number=99,
        title="multi-file",
    )
    assert ok.kind == "real_pr"
    assert len(ok.gold_files) >= 2
    assert ok.test_patch.strip()
    assert ok.meta.get("hybrid_motors_allowed") is False


def test_reject_hybrid_motor_discover_attempt_returns_audit_row() -> None:
    row = reject_hybrid_motor_discover_attempt(
        repo="fixtures/harbor_motors/go_kvstore",
        source_track="hybrid_curated",
    )
    assert row is not None
    assert row.reason_code == "motor_or_hybrid_rejected"
    assert "hybrid" in row.detail.lower() or "motor" in row.detail.lower()
    assert reject_hybrid_motor_discover_attempt(repo="mahmoud/boltons") is None


# --- VAL-RPR-003: gold multi-file source; tests held out ---


def test_gold_source_only_test_patch_held_out() -> None:
    files = [
        _source("pkg/a.py", "@@ -1 +1 @@\n-old_a\n+new_a\n"),
        _source("pkg/b.py", "@@ -1 +1 @@\n-old_b\n+new_b\n"),
        _test("tests/test_a.py", "@@ -0,0 +1,2 @@\n+def test_a():\n+    assert True\n"),
    ]
    gold = extract_gold_from_files(files)
    gold_paths = count_files_in_patch(gold)
    assert len(gold_paths) >= 2
    assert all(not is_test_path(p) for p in gold_paths)
    from swe_factory.sources.discover import extract_test_patch_from_files

    test_patch = extract_test_patch_from_files(files)
    assert test_patch.strip()
    assert any(is_test_path(p) for p in count_files_in_patch(test_patch))


# --- VAL-RPR-004 / inventory for ≥5 ---


def test_offline_real_pr_pool_emits_select5_inventory(tmp_path: Path) -> None:
    report = mine_offline_real_pr_pool(
        work_root=tmp_path / "pool",
        target_candidates=5,
        synthetic_repo_diversity=6,
    )
    assert report.offline is True
    assert report.network_required is False
    assert report.provider_calls == 0
    assert report.keep_count >= 5
    assert report.repo_diversity >= 5
    assert report.source_track == "real_pr"
    assert report.hybrid_motors_allowed is False
    # every keep is real_pr
    for item in report.kept:
        track = getattr(item, "source_track", None)
        if hasattr(item, "task"):
            track = item.task.source_track
            if hasattr(track, "value"):
                track = track.value
        assert str(track) == "real_pr"
        if hasattr(item, "test_patch"):
            assert (item.test_patch or "").strip()
        if hasattr(item, "gold_files"):
            assert len(item.gold_files) >= 2
    # motor rejects documented (honesty)
    assert report.motor_rejects
    assert all(r.reason_code == "motor_or_hybrid_rejected" for r in report.motor_rejects)
    payload = report.to_dict()
    assert payload["honesty"]["motors_excluded"] is True
    assert payload["honesty"]["hybrid_as_real_pr_false_claim"] is False
    assert (tmp_path / "pool" / "real_pr_pool_report.json").is_file()
    assert (tmp_path / "pool" / "candidates.jsonl").is_file()
    assert (tmp_path / "pool" / "tasks.jsonl").is_file()
    # tasks.jsonl all real_pr
    for line in (tmp_path / "pool" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["source_track"] == "real_pr"


def test_real_pr_seed_pool_excludes_motors_and_meets_select5() -> None:
    seeds = real_pr_seed_pool()
    assert len(seeds) >= 5
    motor_ids = {s.seed_id for s in HARBOR_MOTOR_SEEDS}
    assert not motor_ids.intersection({s.seed_id for s in seeds})
    for seed in seeds:
        assert seed.repository_url.startswith("https://")
        banned, _ = is_motor_or_hybrid_identity(seed.repo, seed_id=seed.seed_id)
        assert banned is False
    summary = summary_seed_pool_for_select5()
    assert summary["meets_select5_inventory"] is True
    assert summary["motors_excluded"] is True
    assert "harbor_python_orders" in summary["motor_seed_ids_blocked"]


def test_ensure_source_track_rejects_hybrid_label() -> None:
    from swe_factory.producers.pr_miner import ensure_source_track

    with pytest.raises(PrMineError, match="hybrid|motor|real_pr"):
        ensure_source_track("hybrid_curated")


def test_mocked_merged_pr_fetch_rejects_unmerged_and_accepts_valid() -> None:
    """VAL-RPR-002 live shape with DictGitHubTransport (no network)."""
    routes: dict[str, Any] = {
        "/repos/owner/demo/pulls/42": {
            "number": 42,
            "title": "Multi-file fix",
            "body": "Fixes add and reverse.",
            "merged_at": "2026-01-02T00:00:00Z",
            "merge_commit_sha": "m" * 40,
            "html_url": "https://github.com/owner/demo/pull/42",
            "base": {"sha": "b" * 40, "ref": "main"},
        },
        "/repos/owner/demo/pulls/42/files": [
            {
                "filename": "pkg/a.py",
                "status": "modified",
                "patch": "@@ -1 +1 @@\n-olda\n+newa\n",
            },
            {
                "filename": "pkg/b.py",
                "status": "modified",
                "patch": "@@ -1 +1 @@\n-oldb\n+newb\n",
            },
            {
                "filename": "tests/test_a.py",
                "status": "added",
                "patch": "@@ -0,0 +1,2 @@\n+def test_a():\n+    assert True\n",
            },
        ],
        "/repos/owner/demo/pulls/7": {
            "number": 7,
            "title": "WIP",
            "body": "",
            "merged_at": None,
            "base": {"sha": "c" * 40},
        },
        "/repos/owner/demo/pulls/7/files": [],
    }
    client = GitHubClient(transport=DictGitHubTransport(routes=routes))
    miner = PrMiner(client=client)
    pr = miner.fetch_merged_pr("owner/demo", 42, language="python")
    assert pr.base_commit == "b" * 40
    candidate = miner.produce(pr, instance_suffix="mock", run_stub_oracle=True)
    assert candidate.source_track == "real_pr"
    assert all(not is_test_path(p) for p in candidate.gold_files)
    assert candidate.test_patch.strip()
    with pytest.raises(PrMineError, match="not merged"):
        miner.fetch_merged_pr("owner/demo", 7)
