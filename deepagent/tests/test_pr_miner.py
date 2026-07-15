"""Unit tests for real_pr miner (VAL-PROD-001 / VAL-PROD-004).

Offline mocked GitHub API only (no network). Covers:
- source_track=real_pr labeling
- multi-file PR filter (reject single-file / test-only)
- invalid/missing source_track rejection
- gold extraction + F2P/P2P derivation
- stub + fake certified oracle path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from swe_factory.oracle import codes as C
from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite
from swe_factory.oracle.gates import count_files_in_patch
from swe_factory.producers.pr_miner import (
    PrFileChange,
    PrMineError,
    PrMiner,
    build_problem_statement,
    derive_test_commands,
    ensure_source_track,
    extract_gold_from_files,
    is_source_path,
    is_test_path,
    multi_file_source_filter,
    offline_fixture_pr,
    produce_offline_fixture,
)
from swe_factory.schema import VALID_SOURCE_TRACKS, SourceTrack
from swe_factory.sources.github import DictGitHubTransport, GitHubClient, GitHubError


def _source(path: str, patch: str = "@@ -1 +1 @@\n-a\n+b\n") -> PrFileChange:
    return PrFileChange(path=path, status="modified", patch=patch)


def _test(path: str, patch: str = "@@ -0,0 +1 @@\n+assert True\n") -> PrFileChange:
    return PrFileChange(path=path, status="modified", patch=patch)


def test_is_test_and_source_path_heuristics() -> None:
    assert is_test_path("tests/test_math.py")
    assert is_test_path("src/foo_test.go")
    assert is_test_path("lib/util.test.js")
    assert is_test_path("pkg/spec/bar_test.go") or is_test_path("pkg/foo_test.go")
    assert is_source_path("demo_pkg/math_ops.py")
    assert is_source_path("lib/stringify.js")
    assert not is_source_path("README.md")
    assert not is_source_path("node_modules/x/index.js")


def test_multi_file_pr_filter_accepts_source_plus_tests() -> None:
    files = [
        _source("a.py"),
        _source("b.py"),
        _test("tests/test_a.py"),
    ]
    assert multi_file_source_filter(files) is True


def test_multi_file_pr_filter_rejects_single_source() -> None:
    files = [_source("only.py"), _test("tests/test_only.py")]
    assert multi_file_source_filter(files) is False


def test_multi_file_pr_filter_rejects_test_only() -> None:
    files = [_test("tests/test_a.py"), _test("tests/test_b.py")]
    assert multi_file_source_filter(files) is False


def test_multi_file_pr_filter_rejects_missing_tests_when_required() -> None:
    files = [_source("a.py"), _source("b.py")]
    assert multi_file_source_filter(files, require_tests=True) is False
    assert multi_file_source_filter(files, require_tests=False) is True


def test_extract_gold_is_multi_file_source_only() -> None:
    files = [
        _source("pkg/a.py", "@@ -1 +1 @@\n-old_a\n+new_a\n"),
        _source("pkg/b.py", "@@ -1 +1 @@\n-old_b\n+new_b\n"),
        _test("tests/test_a.py", "@@ -0,0 +1 @@\n+def test_a():\n+    pass\n"),
    ]
    gold = extract_gold_from_files(files)
    touched = count_files_in_patch(gold)
    assert len(touched) >= 2
    assert "pkg/a.py" in touched
    assert "pkg/b.py" in touched
    assert not any(is_test_path(p) for p in touched)


def test_wrap_file_diff_emits_new_file_headers_for_added_status() -> None:
    from swe_factory.producers.pr_miner import wrap_file_diff

    body = wrap_file_diff(
        "pkg/new.py",
        "@@ -0,0 +1,2 @@\n+def f():\n+    return 1\n",
        status="added",
    )
    assert "new file mode 100644" in body
    assert "--- /dev/null\n" in body
    assert "+++ b/pkg/new.py\n" in body
    assert "--- a/pkg/new.py" not in body


def test_wrap_file_diff_infers_create_from_hunk_even_without_status() -> None:
    from swe_factory.producers.pr_miner import wrap_file_diff

    body = wrap_file_diff(
        "tests/test_new.py",
        "@@ -0,0 +1,1 @@\n+def test_new():\n+    assert True\n",
    )
    assert "new file mode" in body
    assert "--- /dev/null" in body


def test_repair_pseudo_create_file_headers_rewrites_legacy_materials() -> None:
    from swe_factory.producers.pr_miner import repair_pseudo_create_file_headers

    legacy = (
        "diff --git a/itemadapter/_json_schema.py b/itemadapter/_json_schema.py\n"
        "--- a/itemadapter/_json_schema.py\n"
        "+++ b/itemadapter/_json_schema.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+def schema():\n"
        "+    return {}\n"
        "+\n"
    )
    fixed = repair_pseudo_create_file_headers(legacy)
    assert "new file mode 100644" in fixed
    assert "--- /dev/null\n" in fixed
    assert "+++ b/itemadapter/_json_schema.py\n" in fixed
    assert "--- a/itemadapter/_json_schema.py" not in fixed
    # modify hunks stay unmodified
    modify = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    assert repair_pseudo_create_file_headers(modify) == modify


def test_derive_test_commands_python_prefer_pr_tests() -> None:
    f2p, p2p = derive_test_commands(
        language="python",
        test_files=["tests/test_math.py", "tests/test_text.py"],
    )
    assert f2p
    assert "tests/test_math.py" in f2p[0]
    assert "tests/test_text.py" in f2p[0]
    assert p2p  # broader baseline present


def test_ensure_source_track_rejects_invalid_and_missing() -> None:
    assert ensure_source_track("real_pr") is SourceTrack.REAL_PR
    assert ensure_source_track(SourceTrack.REAL_PR) is SourceTrack.REAL_PR
    # valid alternate track parses, but pr_miner.produce still requires real_pr
    assert ensure_source_track("synthetic_grounded") is SourceTrack.SYNTHETIC_GROUNDED
    with pytest.raises(PrMineError, match="source_track"):
        ensure_source_track(None)
    with pytest.raises(PrMineError, match="source_track"):
        ensure_source_track("")
    with pytest.raises(PrMineError, match="invalid source_track"):
        ensure_source_track("mixed_unknown")


def test_offline_fixture_pr_labeled_real_pr(tmp_path: Path) -> None:
    candidate = produce_offline_fixture(work_root=tmp_path / "work", run_stub_oracle=True)
    task = candidate.task
    assert task.source_track == SourceTrack.REAL_PR
    assert candidate.source_track == "real_pr"
    assert task.repo
    assert task.base_commit
    assert task.gold_patch.strip()
    assert task.problem_statement.strip()
    assert len(candidate.gold_files) >= 2
    assert candidate.provenance["gold_provenance"]
    assert candidate.provenance["source_track"] == "real_pr"
    assert candidate.gates is not None and candidate.gates.passed
    assert C.G4_MULTI_FILE_OK in candidate.gates.reason_codes
    # tasks.jsonl-shape dump works
    row = json.loads(task.model_dump_json())
    assert row["source_track"] == "real_pr"
    assert row["base_commit"]
    assert row["gold_patch"]


def test_produce_rejects_non_real_pr_track(tmp_path: Path) -> None:
    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client, work_root=tmp_path)
    pr = offline_fixture_pr()
    with pytest.raises(PrMineError, match="real_pr"):
        miner.produce(pr, source_track="synthetic_grounded", run_stub_oracle=False)


def test_select_rejects_single_file_pr() -> None:
    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client)
    with pytest.raises(PrMineError, match="multi-file"):
        miner.select_from_files(
            repo="owner/repo",
            number=9,
            title="tiny",
            body="",
            base_commit="abc123",
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


def test_mocked_github_fetch_merged_pr() -> None:
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
                "additions": 1,
                "deletions": 1,
            },
            {
                "filename": "pkg/b.py",
                "status": "modified",
                "patch": "@@ -1 +1 @@\n-oldb\n+newb\n",
                "additions": 1,
                "deletions": 1,
            },
            {
                "filename": "tests/test_a.py",
                "status": "added",
                "patch": "@@ -0,0 +1,3 @@\n+def test_a():\n+    assert True\n",
                "additions": 3,
                "deletions": 0,
            },
        ],
    }
    client = GitHubClient(transport=DictGitHubTransport(routes=routes))
    miner = PrMiner(client=client)
    pr = miner.fetch_merged_pr("owner/demo", 42, language="python")
    assert pr.base_commit == "b" * 40
    assert len(pr.source_files) >= 2
    assert pr.test_files
    candidate = miner.produce(pr, instance_suffix="mock", run_stub_oracle=True)
    assert candidate.task.source_track == SourceTrack.REAL_PR
    assert candidate.task.base_commit == "b" * 40
    assert candidate.task.repo == "owner/demo"
    assert "pkg/a.py" in candidate.gold_files or "pkg/a.py" in str(candidate.gold_files)


def test_mocked_github_rejects_unmerged() -> None:
    routes = {
        "/repos/owner/demo/pulls/7": {
            "number": 7,
            "title": "WIP",
            "body": "",
            "merged_at": None,
            "base": {"sha": "c" * 40},
        },
        "/repos/owner/demo/pulls/7/files": [],
    }
    miner = PrMiner(client=GitHubClient(transport=DictGitHubTransport(routes=routes)))
    with pytest.raises(PrMineError, match="not merged"):
        miner.fetch_merged_pr("owner/demo", 7)


def test_mocked_transport_missing_route_raises() -> None:
    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    with pytest.raises(GitHubError):
        client.get_pull("owner/x", 1)


def test_produce_and_certify_with_fake_oracle(tmp_path: Path) -> None:
    # Use tiny_offline broken workspace for stronger realism
    fixture_repo = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_offline" / "repo"
    assert fixture_repo.is_dir()
    runner = FakeOracleRunner(
        broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        gold_runs=[
            ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
            ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
        ],
        null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
    )
    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client, work_root=tmp_path / "work")
    pr = offline_fixture_pr()
    audit = tmp_path / "gate_audit.jsonl"
    candidate = miner.produce_and_certify(
        pr,
        runner=runner,
        workspace=fixture_repo,
        instance_suffix="certfake",
        fail_to_pass=["python -m pytest tests/test_math.py tests/test_text.py -q"],
        pass_to_pass=["python -m pytest tests/test_ok.py -q"],
        audit_out=audit,
    )
    assert candidate.gates is not None and candidate.gates.passed
    assert C.ORACLE_PASS in candidate.gates.reason_codes
    assert candidate.task.source_track == SourceTrack.REAL_PR
    row = json.loads(audit.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["source_track"] == "real_pr"
    assert row["disposition"] == "accept"


def test_problem_statement_includes_title_and_files() -> None:
    pr = offline_fixture_pr()
    prompt = build_problem_statement(pr=pr)
    assert prompt.strip()
    assert "demo_pkg/math_ops.py" in prompt or "math_ops" in prompt
    assert str(pr.number) in prompt


def test_valid_source_tracks_known() -> None:
    assert "real_pr" in VALID_SOURCE_TRACKS
    assert "synthetic_grounded" in VALID_SOURCE_TRACKS
