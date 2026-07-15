"""Offline unit tests for materials-from-PR bridge (VAL-LMAT-001/004, VAL-LMINE-008).

DictGitHubTransport / MergedPR only — no network. Covers:
- inventory.json + {task_id}/solution.patch + test.patch under non-fixture root
- inventory fields repo/pr/base(sha)/language
- refuse fixtures/real_pr_ship as live materials root
- empty solution fails closed
- batch materialize from MergedPR list
- product-compatible inventory loadable by ship_real_pr.load_real_pr_materials
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from swe_factory.pipeline.ship_real_pr import load_real_pr_materials
from swe_factory.producers.materialize_from_pr import (
    DEFAULT_LIVE_MATERIALS_ROOT,
    FIXTURE_MATERIALS_MARKER,
    INVENTORY_REQUIRED_FIELDS,
    MaterializeError,
    assert_non_fixture_materials_root,
    inventory_stats,
    is_fixture_materials_root,
    materialize_accepted_candidates,
    materialize_from_pr_number,
    materialize_merged_pr,
    materials_task_id,
    read_inventory,
)
from swe_factory.producers.pr_miner import MergedPR, PrFileChange, PrMiner
from swe_factory.sources.github import DictGitHubTransport, GitHubClient

_FULL_SHA = "a" * 40
_MERGE_SHA = "b" * 40


def _source(path: str, patch: str | None = None) -> PrFileChange:
    body = patch or ("@@ -1,3 +1,4 @@\n def a():\n-    return 1\n+    return 2\n+\n")
    return PrFileChange(path=path, status="modified", patch=body)


def _test_file(path: str, patch: str | None = None) -> PrFileChange:
    body = patch or "@@ -0,0 +1,3 @@\n+def test_a():\n+    assert True\n+\n"
    return PrFileChange(path=path, status="added", patch=body)


def _merged_pr(
    *,
    repo: str = "owner/demo",
    number: int = 42,
    files: list[PrFileChange] | None = None,
    language: str = "python",
    license: str = "MIT",
    base: str = _FULL_SHA,
) -> MergedPR:
    changes = files or [
        _source("pkg/a.py"),
        _source("pkg/b.py"),
        _test_file("tests/test_a.py"),
    ]
    return MergedPR(
        repo=repo,
        number=number,
        title=f"Hard fix #{number}",
        body="multi-file product source + tests",
        base_commit=base,
        merge_commit_sha=_MERGE_SHA,
        language=language,
        html_url=f"https://github.com/{repo}/pull/{number}",
        files=tuple(changes),
        license=license,
        merged_at="2026-01-02T00:00:00Z",
        source_hunk_count=2,
    )


def _hard_files_payload() -> list[dict[str, Any]]:
    hunks_a = "".join(f"@@ -{i},1 +{i},1 @@\n-old{i}\n+new{i}\n" for i in range(1, 7))
    hunks_b = "".join(f"@@ -{i},1 +{i},1 @@\n-ob{i}\n+nb{i}\n" for i in range(1, 6))
    return [
        {"filename": "pkg/a.py", "status": "modified", "patch": hunks_a},
        {"filename": "pkg/b.py", "status": "modified", "patch": hunks_b},
        {
            "filename": "tests/test_a.py",
            "status": "added",
            "patch": "@@ -0,0 +1,2 @@\n+def test_a():\n+    assert True\n",
        },
    ]


def _mock_pr_routes(repo: str = "owner/demo", number: int = 42) -> dict[str, Any]:
    files = _hard_files_payload()
    return {
        f"/repos/{repo}/pulls/{number}": {
            "number": number,
            "title": "Hard multi-file PR",
            "body": "body",
            "merged_at": "2026-01-02T00:00:00Z",
            "merge_commit_sha": _MERGE_SHA,
            "html_url": f"https://github.com/{repo}/pull/{number}",
            "base": {"sha": _FULL_SHA, "ref": "main"},
        },
        f"/repos/{repo}/pulls/{number}/files": files,
    }


def test_default_live_materials_root_outside_fixture_shortlist() -> None:
    assert FIXTURE_MATERIALS_MARKER not in str(DEFAULT_LIVE_MATERIALS_ROOT)
    assert "real_pr_ship" not in str(DEFAULT_LIVE_MATERIALS_ROOT)
    assert is_fixture_materials_root(FIXTURE_MATERIALS_MARKER)
    assert is_fixture_materials_root("fixtures/real_pr_ship")
    assert is_fixture_materials_root(Path("fixtures/real_pr_ship"))
    assert not is_fixture_materials_root(DEFAULT_LIVE_MATERIALS_ROOT)
    assert not is_fixture_materials_root("datasets/live_materials")


def test_assert_non_fixture_materials_root_refuses_real_pr_ship() -> None:
    with pytest.raises(MaterializeError, match="fixture shortlist"):
        assert_non_fixture_materials_root("fixtures/real_pr_ship")
    with pytest.raises(MaterializeError, match="fixture"):
        assert_non_fixture_materials_root(Path("/tmp") / "fixtures" / "real_pr_ship")
    ok = assert_non_fixture_materials_root("datasets/live_materials")
    assert ok == Path("datasets/live_materials")


def test_materials_task_id_style() -> None:
    assert materials_task_id("jaraco/zipp", 154) == "realpr-zipp-154"
    assert materials_task_id("pallets/click", 3581) == "realpr-click-3581"
    assert materials_task_id("owner/demo", 42) == "realpr-demo-42"


def test_materialize_merged_pr_writes_inventory_and_patches(tmp_path: Path) -> None:
    """VAL-LMAT-001: durable tree with solution.patch, test.patch, inventory fields."""
    root = tmp_path / "live_materials"
    pr = _merged_pr()
    task = materialize_merged_pr(pr, root, discovery_path="list_pulls")

    assert task.task_id == "realpr-demo-42"
    task_dir = root / task.task_id
    sol = task_dir / "solution.patch"
    test = task_dir / "test.patch"
    meta = task_dir / "meta.json"
    inv = root / "inventory.json"

    assert sol.is_file() and sol.read_text(encoding="utf-8").strip()
    assert test.is_file() and test.read_text(encoding="utf-8").strip()
    assert meta.is_file()
    assert inv.is_file()

    # solution is source-only; test holds verifier patch
    sol_body = sol.read_text(encoding="utf-8")
    test_body = test.read_text(encoding="utf-8")
    assert "pkg/a.py" in sol_body
    assert "pkg/b.py" in sol_body
    assert "tests/test_a.py" not in sol_body or "test_a" not in sol_body.split("pkg/a")[0]
    assert "tests/test_a.py" in test_body

    rows = read_inventory(root)
    assert len(rows) == 1
    row = rows[0]
    for field in INVENTORY_REQUIRED_FIELDS:
        assert field in row, f"missing inventory field {field}"
    # expectedBehavior: Inventory fields repo/pr/sha/lang
    assert row["repo"] == "owner/demo"
    assert row["pr"] == 42
    assert row["base"] == _FULL_SHA
    assert len(row["base"]) == 40
    assert row["language"] == "python"
    assert row["task_id"] == "realpr-demo-42"
    assert row["discovery_path"] == "list_pulls"
    assert "src" in row and len(row["src"]) >= 2
    assert "tests" in row and len(row["tests"]) >= 1


def test_materialize_refuses_fixture_root_by_default(tmp_path: Path) -> None:
    """VAL-LMINE-008 / VAL-LMAT-001: live bridge must not write fixture shortlist."""
    fixture_root = tmp_path / "fixtures" / "real_pr_ship"
    fixture_root.mkdir(parents=True)
    pr = _merged_pr()
    with pytest.raises(MaterializeError, match="fixture"):
        materialize_merged_pr(pr, fixture_root, allow_fixture_root=False)


def test_materialize_rejects_empty_solution(tmp_path: Path) -> None:
    """VAL-LMAT-001: empty solution.patch for a supposed keep fails."""
    root = tmp_path / "live_materials"
    # Test-only files → gold extraction fails / empty
    pr = _merged_pr(
        files=[
            _test_file("tests/test_a.py"),
            _test_file("tests/test_b.py"),
        ]
    )
    with pytest.raises(MaterializeError, match="usable source|empty solution|no usable"):
        materialize_merged_pr(pr, root)


def test_materialize_rejects_non_full_sha(tmp_path: Path) -> None:
    root = tmp_path / "live_materials"
    pr = _merged_pr(base="abc1234")
    with pytest.raises(MaterializeError, match="40-char"):
        materialize_merged_pr(pr, root)


def test_materialize_from_pr_number_offline_mock_transport(tmp_path: Path) -> None:
    """VAL-LMAT-004: offline DictGitHubTransport materialize without network."""
    root = tmp_path / "live_materials"
    transport = DictGitHubTransport(routes=_mock_pr_routes("owner/demo", 42))
    client = GitHubClient(transport=transport)
    task = materialize_from_pr_number(
        client,
        "owner/demo",
        42,
        root,
        discovery_path="search",
        product_mode=False,
    )
    assert task.repo == "owner/demo"
    assert task.pr_number == 42
    assert task.base_sha == _FULL_SHA
    assert (root / task.task_id / "solution.patch").is_file()
    assert (root / "inventory.json").is_file()
    rows = read_inventory(root)
    assert rows[0]["language"] == "python"
    assert rows[0]["discovery_path"] == "search"
    # transport was hit (offline routes), not fixture inventory
    assert any("/pulls/42" in c for c in transport.calls)


def test_materialize_accepted_batch_from_merged_prs(tmp_path: Path) -> None:
    root = tmp_path / "live_materials"
    prs = [
        _merged_pr(repo="owner/alpha", number=1),
        _merged_pr(repo="owner/beta", number=2),
    ]
    report = materialize_accepted_candidates(None, prs, root)
    assert len(report.tasks) == 2
    assert report.product_materials is True
    assert report.engineering_fixture is False
    assert FIXTURE_MATERIALS_MARKER not in report.materials_root
    rows = read_inventory(root)
    assert {r["task_id"] for r in rows} == {"realpr-alpha-1", "realpr-beta-2"}
    stats = inventory_stats(root)
    assert stats["count"] == 2
    assert stats["is_fixture_root"] is False


def test_upsert_inventory_replaces_same_task_id(tmp_path: Path) -> None:
    root = tmp_path / "live_materials"
    pr = _merged_pr(number=10)
    materialize_merged_pr(pr, root)
    # re-materialize with different title path should replace, not duplicate
    pr2 = _merged_pr(
        number=10,
        files=[
            _source("pkg/x.py"),
            _source("pkg/y.py"),
            _test_file("tests/test_x.py"),
        ],
    )
    materialize_merged_pr(pr2, root)
    rows = read_inventory(root)
    assert len(rows) == 1
    assert rows[0]["task_id"] == "realpr-demo-10"
    sol = (root / "realpr-demo-10" / "solution.patch").read_text(encoding="utf-8")
    assert "pkg/x.py" in sol


def test_ship_loader_accepts_live_materials_tree(tmp_path: Path) -> None:
    """Materials bridge output can feed product ship loaders (not fixture-only)."""
    root = tmp_path / "live_materials"
    # Need ≥2 product sources + ≥1 test path in inventory/patches.
    pr = _merged_pr(
        repo="owner/shipable",
        number=99,
        files=[
            _source("lib/one.py"),
            _source("lib/two.py"),
            _test_file("tests/test_one.py"),
        ],
    )
    materialize_merged_pr(pr, root)
    materials = load_real_pr_materials(root)
    assert len(materials) == 1
    mat = materials[0]
    assert mat.task_id == "realpr-shipable-99"
    assert mat.pr_number == 99
    assert mat.base_commit == _FULL_SHA
    assert mat.language == "python"
    assert mat.solution_patch.strip()
    assert mat.test_patch.strip()
    assert FIXTURE_MATERIALS_MARKER not in mat.materials_dir


def test_batch_mapping_candidates_require_client_or_merged_pr(tmp_path: Path) -> None:
    root = tmp_path / "live_materials"
    with pytest.raises(MaterializeError, match="no materials materialized|GitHubClient"):
        materialize_accepted_candidates(
            None,
            [{"repo": "owner/demo", "pr_number": 1, "discovery_path": "list_pulls"}],
            root,
        )


def test_batch_mapping_with_mock_client(tmp_path: Path) -> None:
    root = tmp_path / "live_materials"
    transport = DictGitHubTransport(routes=_mock_pr_routes("owner/demo", 42))
    client = GitHubClient(transport=transport)
    report = materialize_accepted_candidates(
        client,
        [
            {
                "repo": "owner/demo",
                "pr_number": 42,
                "language": "python",
                "license": "MIT",
                "discovery_path": "list_pulls",
            }
        ],
        root,
        product_mode=False,
    )
    assert len(report.tasks) == 1
    assert report.tasks[0].pr_number == 42
    inv = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
    assert inv[0]["repo"] == "owner/demo"
    assert inv[0]["base"] == _FULL_SHA
    assert inv[0]["language"] == "python"


def test_materialize_from_pr_miner_produce_bridge(tmp_path: Path) -> None:
    """Accepted PR via PrMiner offline transport → materials tree."""
    root = tmp_path / "live_materials"
    transport = DictGitHubTransport(routes=_mock_pr_routes("owner/demo", 42))
    client = GitHubClient(transport=transport)
    miner = PrMiner(client=client, product_mode=False)
    pr = miner.fetch_merged_pr("owner/demo", 42, require_full_base_sha=True, enforce_license=True)
    task = materialize_merged_pr(pr, root, discovery_path="list_pulls")
    assert task.base_sha == _FULL_SHA
    assert (root / task.task_id / "test.patch").stat().st_size > 0


def test_rebuild_inventory_from_task_dirs_recovers_truncated_inventory(
    tmp_path: Path,
) -> None:
    """TDD: truncated inventory.json must be rebuilt to cover all loadable tasks.

    m14 under-yield: ~24 task dirs but inventory listed ~4; ship trusted incomplete inv.
    """
    from swe_factory.producers.materialize_from_pr import (
        inventory_completeness,
        rebuild_inventory_from_task_dirs,
        write_inventory,
    )

    root = tmp_path / "live_materials"
    root.mkdir()
    # Materialize three tasks, then truncate inventory to one row only (simulating bug).
    for n, repo in ((1, "owner/alpha"), (2, "owner/beta"), (3, "owner/gamma")):
        materialize_merged_pr(_merged_pr(repo=repo, number=n), root)

    full_before = read_inventory(root)
    assert len(full_before) == 3

    # Truncate inventory anten — dirs remain on disk with patches/meta
    write_inventory(root, [full_before[0]])
    assert len(read_inventory(root)) == 1

    completeness = inventory_completeness(root)
    assert completeness["complete"] is False
    assert len(completeness["missing_from_inventory"]) == 2
    assert completeness["loadable_task_dirs"] == 3

    rebuilt = rebuild_inventory_from_task_dirs(root, merge_existing=True, write=True)
    assert len(rebuilt) == 3
    assert {r["task_id"] for r in rebuilt} == {
        "realpr-alpha-1",
        "realpr-beta-2",
        "realpr-gamma-3",
    }
    # Required inventory fields present on recovered rows
    for row in rebuilt:
        for field in INVENTORY_REQUIRED_FIELDS:
            assert field in row
        assert len(str(row["base"])) == 40

    completeness2 = inventory_completeness(root)
    assert completeness2["complete"] is True
    assert completeness2["missing_from_inventory"] == []

    # Ship loader must now see all 3 (not the truncated 1)
    materials = load_real_pr_materials(root, rebuild_inventory=True)
    assert len(materials) == 3


def test_load_real_pr_materials_auto_rebuilds_incomplete_inventory(
    tmp_path: Path,
) -> None:
    """load_real_pr_materials repairs truncated inventory without caller rewrite."""
    from swe_factory.producers.materialize_from_pr import write_inventory

    root = tmp_path / "live_materials"
    for n, repo in ((10, "owner/a"), (11, "owner/b")):
        materialize_merged_pr(_merged_pr(repo=repo, number=n), root)
    only = [read_inventory(root)[0]]
    write_inventory(root, only)
    # Broken state
    assert len(read_inventory(root)) == 1
    mats = load_real_pr_materials(root, rebuild_inventory=True)
    assert len(mats) == 2
    # Inventory file rewritten for completeness too
    assert len(read_inventory(root)) == 2
