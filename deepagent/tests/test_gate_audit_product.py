"""Product dual-truth gate_audit (VAL-LSHIP-007)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.pipeline.gate_audit_product import (
    LABEL_METHOD_LIVE,
    ProductGateAuditError,
    audit_keep_dual_truth,
    rebuild_product_dual_truth_from_tasks,
    require_gate_audit_pass,
    write_product_gate_audit,
)


def test_audit_keep_accepts_live_dual_truth() -> None:
    row = audit_keep_dual_truth(
        task_id="realpr-demo-1",
        materials_root="datasets/live_materials",
        live_mine=True,
        label_method=LABEL_METHOD_LIVE,
        # VAL-DMED-001: product floor requires F2P ≥ MIN_F2P_NODES (default 5)
        f2p_node_ids=[
            "tests.test_x::test_a",
            "tests.test_x::test_b",
            "tests.test_x::test_c",
            "tests.test_x::test_d",
            "tests.test_x::test_e",
        ],
        p2p_node_ids=["tests.test_x::test_p2p"],
        backend_class="HarborDockerVerifier",
        agent_image="harbor-sdf-agent-rpr00:oracle",
        tests_image="harbor-sdf-tests-rpr00:oracle",
        solution_reward=1,
        null_reward=0,
        source_track="real_pr",
        source_hunk_count=16,
        discovery_path="list_pulls",
    )
    assert row.accepted
    assert not row.reasons


def test_audit_keep_rejects_synthetic_and_empty_images() -> None:
    row = audit_keep_dual_truth(
        task_id="bad",
        materials_root="fixtures/real_pr_ship",
        live_mine=True,
        label_method="synthetic_patch_seed",
        f2p_node_ids=["tests.test_ok::test_always_ok"],
        p2p_node_ids=["tests.test_ok::test_always_ok"],
        backend_class="ScriptedDockerVerifier",
        agent_image="",
        tests_image="",
        solution_reward=1,
        null_reward=1,
        source_track="real_pr",
        source_hunk_count=3,
        discovery_path="offline_fixture",
    )
    assert not row.accepted
    joined = " ".join(row.reasons)
    assert "fixture" in joined or "label" in joined or "synthetic" in joined
    assert "empty_docker_images" in row.reasons or "backend_not" in joined


def test_write_and_require_gate_audit(tmp_path: Path) -> None:
    good = audit_keep_dual_truth(
        task_id="k1",
        materials_root="datasets/live_materials",
        live_mine=True,
        label_method=LABEL_METHOD_LIVE,
        f2p_node_ids=["n1", "n2", "n3", "n4", "n5"],
        p2p_node_ids=["p1"],
        backend_class="HarborDockerVerifier",
        agent_image="a:tag",
        tests_image="t:tag",
        solution_reward=1,
        null_reward=0,
        source_hunk_count=16,
        discovery_path="search",
    )
    path = tmp_path / "gate_audit.jsonl"
    result = write_product_gate_audit(
        [good],
        path,
        materials_root="datasets/live_materials",
        live_mine=True,
        seed5_archived=True,
        min_accepted=1,
        require_all_accepted=True,
    )
    assert result.ok
    assert path.is_file()
    # summary is sibling rename: gate_audit.jsonl → gate_audit_summary.json
    summary_path = tmp_path / "gate_audit_summary.json"
    assert summary_path.is_file(), list(tmp_path.iterdir())
    require_gate_audit_pass(result)

    bad = audit_keep_dual_truth(
        task_id="k2",
        materials_root="datasets/live_materials",
        live_mine=True,
        label_method="synthetic_patch_seed",
        f2p_node_ids=[],
        p2p_node_ids=[],
        backend_class="Fake",
        agent_image="",
        tests_image="",
        solution_reward=0,
        null_reward=1,
    )
    fail = write_product_gate_audit(
        [bad],
        tmp_path / "gate_fail.jsonl",
        materials_root="datasets/live_materials",
        live_mine=True,
        seed5_archived=True,
        min_accepted=1,
        require_all_accepted=True,
    )
    assert not fail.ok
    with pytest.raises(ProductGateAuditError):
        require_gate_audit_pass(fail)
    with pytest.raises(ProductGateAuditError):
        require_gate_audit_pass(None)
    summary = json.loads((tmp_path / "gate_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True


def _seed_product_keep(
    product: Path,
    *,
    task_id: str,
    language: str,
    agent_image: str,
    tests_image: str,
    f2p: list[str],
    p2p: list[str] | None = None,
    source_hunk_count: int = 16,
    repo: str = "example/repo",
    base: str = "a" * 40,
    discovery_path: str = "list_pulls",
    materials_root: Path | None = None,
) -> None:
    task = product / "tasks" / task_id
    (task / "tests").mkdir(parents=True, exist_ok=True)
    (task / "solution").mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text(
        "\n".join(
            [
                f'name = "swe-factory/{task_id}"',
                f'task_id = "{task_id}"',
                f'language = "{language}"',
                f'repository_url = "https://github.com/{repo}.git"',
                f'base_commit_hash = "{base}"',
                'source_track = "real_pr"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (task / "tests" / "config.json").write_text(
        json.dumps(
            {
                "base_commit": base,
                "f2p_node_ids": f2p,
                "p2p_node_ids": p2p or ["p2p-a"],
                "label_method": LABEL_METHOD_LIVE,
                "source_track": "real_pr",
                "suite_command": "npm test",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    docker = product / "evidence" / "docker"
    docker.mkdir(parents=True, exist_ok=True)
    (docker / f"{task_id}.json").write_text(
        json.dumps(
            {
                "backend": "docker",
                "certified": True,
                "disposition": "accept",
                "base_commit_hash": base,
                "isolation_status": "clean",
                "null_reward": 0,
                "oracle": {
                    "agent_image": agent_image,
                    "tests_image": tests_image,
                    "solution_reward": 1,
                    "null_reward": 0,
                    "passed": True,
                    "mode": "docker",
                },
                "pack_meta": {
                    "language": language,
                    "repository_url": f"https://github.com/{repo}.git",
                    "base_commit_hash": base,
                },
                "solution_reward": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if materials_root is not None:
        mat = materials_root / task_id
        mat.mkdir(parents=True, exist_ok=True)
        (mat / "meta.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "repo": repo,
                    "base": base,
                    "language": language,
                    "source_hunk_count": source_hunk_count,
                    "discovery_path": discovery_path,
                    "materials_root": str(materials_root),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def test_rebuild_product_dual_truth_from_tasks_full_rewrite(tmp_path: Path) -> None:
    """Additive lag case: tasks/* N=20-class product with stale accepted_count=17 audit."""
    product = tmp_path / "deepagent_v1"
    mats = tmp_path / "live_materials"
    # 17 baseline python + 3 additive (qsx2, bitflags) — scale down to 5 for unit speed
    base_ids = [f"realpr-base-{i}" for i in range(1, 4)]
    additive = ["realpr-qs-487", "realpr-qs-488", "realpr-bitflags-483"]
    for i, tid in enumerate(base_ids):
        _seed_product_keep(
            product,
            task_id=tid,
            language="python",
            agent_image=f"harbor-sdf-agent-rpr{i:02d}:oracle",
            tests_image=f"harbor-sdf-tests-rpr{i:02d}:oracle",
            f2p=[
                f"tests.t::test_{i}_a",
                f"tests.t::test_{i}_b",
                f"tests.t::test_{i}_c",
                f"tests.t::test_{i}_d",
                f"tests.t::test_{i}_e",
            ],
            materials_root=mats,
            source_hunk_count=16 + i,
            repo=f"py/pkg{i}",
        )
    _seed_product_keep(
        product,
        task_id="realpr-qs-487",
        language="javascript",
        agent_image="harbor-sdf-agent-ml300:oracle",
        tests_image="harbor-sdf-tests-ml300:oracle",
        # Synthetic gate_audit fixtures must clear M27 floors (real qs-487 is thin).
        f2p=[
            "should be deeply equivalent",
            "should be strictly equal",
            "should throw",
            "should encode nested",
            "should decode nested",
        ],
        materials_root=mats,
        source_hunk_count=16,
        repo="ljharb/qs",
        base="04f422fe91985103d2fdca0280ee362ecf5e43f2",
    )
    _seed_product_keep(
        product,
        task_id="realpr-qs-488",
        language="javascript",
        agent_image="harbor-sdf-agent-ml301:oracle",
        tests_image="harbor-sdf-tests-ml301:oracle",
        f2p=[
            "should be strictly equal",
            "should throw",
            "should coerce undefined",
            "should encode nested",
            "should decode nested",
        ],
        materials_root=mats,
        source_hunk_count=16,
        repo="ljharb/qs",
        base="5f0449fff1d9fb236d297cd0d3650b42d2d93b8a",
    )
    _seed_product_keep(
        product,
        task_id="realpr-bitflags-483",
        language="rust",
        agent_image="harbor-sdf-agent-mlrust483b:oracle",
        tests_image="harbor-sdf-tests-mlrust483b:oracle",
        f2p=[
            "pass",
            "tests/compile-pass/bitflags_flag_name.rs",
            "tests/compile-fail/flag_value.rs",
            "tests/compile-pass/extra_a.rs",
            "tests/compile-pass/extra_b.rs",
        ],
        p2p=[],
        materials_root=mats,
        source_hunk_count=16,
        repo="bitflags/bitflags",
        base="4ed9ffa949970239cd2d87c775e9fdcf9c438fb5",
    )

    # Stale gate_audit (lagging accepted_count; missing additive)
    stale_rows = []
    for tid in base_ids:
        stale_rows.append(
            {
                "task_id": tid,
                "accepted": True,
                "reasons": [],
                "fields": {
                    "agent_image": "stale",
                    "tests_image": "stale",
                    "backend_class": "HarborDockerVerifier",
                    "solution_reward": 1,
                    "null_reward": 0,
                    "label_method": LABEL_METHOD_LIVE,
                    "source_hunk_count": 16,
                },
                "stage": "gate_audit_dual_truth",
            }
        )
    (product / "gate_audit_summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "accepted_count": len(base_ids),
                "accepted_ids": list(base_ids),
                "intended_count": len(base_ids),
                "rows": stale_rows,
                "timestamp_utc": "2026-07-15T10:36:12.535265+00:00",
                "gate": "product_dual_truth",
                "seed5_archived": True,
                "live_mine": True,
                "materials_root": str(mats),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # Stale oracle missing qs
    (product / "oracle_evidence.json").write_text(
        json.dumps(
            {
                "backend": "docker",
                "certified_count": 3,
                "records": [
                    {
                        "task_id": tid,
                        "certified": True,
                        "solution_reward": 1,
                        "null_reward": 0,
                        "backend_class": "HarborDockerVerifier",
                        "agent_image": f"harbor-sdf-agent-{tid}:oracle",
                        "tests_image": f"harbor-sdf-tests-{tid}:oracle",
                    }
                    for tid in base_ids + ["realpr-bitflags-483"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (product / "ship_summary.json").write_text(
        json.dumps(
            {
                "certified_count": 6,
                "gate_audit_pass": True,
                "ok": True,
                "live_mine": True,
                "seed5_archived": True,
                "materials_root": str(mats),
                "materials_is_fixture": False,
                "languages": {"python": 3},
                "mlang_additive": {
                    "added": additive,
                    "records": [
                        {
                            "task_id": "realpr-qs-487",
                            "language": "javascript",
                            "sol": 1,
                            "null": 0,
                            "source_hunk_count": 16,
                            "docker_ok": True,
                            "label_method": LABEL_METHOD_LIVE,
                            "f2p": [
                                "should be deeply equivalent",
                                "should be strictly equal",
                                "should throw",
                                "should encode nested",
                                "should decode nested",
                            ],
                            "agent_image": "harbor-sdf-agent-ml300:oracle",
                            "tests_image": "harbor-sdf-tests-ml300:oracle",
                        }
                    ],
                },
                "gate_audit": {
                    "accepted_count": len(base_ids),
                    "accepted_ids": list(base_ids),
                    "ok": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (product / "pack_manifest.json").write_text(
        json.dumps(
            {
                "count": 6,
                "pack_count": 6,
                "task_ids": list(base_ids),
                "packs": [],
                "languages": {"python": 3},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (product / "report.md").write_text(
        "# DeepAgent v1 ship report\n\n- Certified packs N: **3** (stale)\n",
        encoding="utf-8",
    )

    expected_n = len(base_ids) + len(additive)
    out = rebuild_product_dual_truth_from_tasks(
        product,
        materials_roots=[mats],
        live_mine=True,
        seed5_archived=True,
        min_accepted=expected_n,
        require_all_accepted=True,
    )
    assert out["ok"] is True
    assert out["accepted_count"] == expected_n
    assert out["task_count"] == expected_n
    for tid in additive:
        assert tid in out["accepted_ids"]

    summary = json.loads((product / "gate_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_count"] == expected_n
    assert summary["intended_count"] == expected_n
    assert summary["ok"] is True
    for tid in additive:
        assert tid in summary["accepted_ids"]

    oe = json.loads((product / "oracle_evidence.json").read_text(encoding="utf-8"))
    oe_ids = {r["task_id"] for r in oe["records"] if r.get("certified")}
    for tid in additive + base_ids:
        assert tid in oe_ids
    for rec in oe["records"]:
        if rec["task_id"] in additive + base_ids:
            assert rec.get("solution_reward") == 1 or rec.get("sol") == 1
            assert rec.get("null_reward") == 0 or rec.get("null") == 0
            assert rec.get("backend_class") == "HarborDockerVerifier"
            assert rec.get("agent_image")
            assert rec.get("tests_image")

    ship = json.loads((product / "ship_summary.json").read_text(encoding="utf-8"))
    assert ship["gate_audit_pass"] is True
    assert ship["gate_audit"]["accepted_count"] == expected_n
    assert ship["certified_count"] == expected_n
    assert set(ship["languages"]) >= {"python", "javascript", "rust"}

    pack = json.loads((product / "pack_manifest.json").read_text(encoding="utf-8"))
    assert pack["pack_count"] == expected_n
    assert set(pack["task_ids"]) == set(base_ids + additive)
    assert (product / "evidence" / "docker" / "realpr-qs-487.oracle_evidence.json").is_file()
    assert (product / "gate_audit.jsonl").is_file()
    # Packs not wiped
    assert (product / "tasks" / "realpr-qs-487").is_dir()


def test_gate_audit_product_cli_smoke(tmp_path: Path) -> None:
    product = tmp_path / "deepagent_v1"
    mats = tmp_path / "live_materials"
    _seed_product_keep(
        product,
        task_id="realpr-demo-1",
        language="python",
        agent_image="harbor-sdf-agent-rpr00:oracle",
        tests_image="harbor-sdf-tests-rpr00:oracle",
        f2p=[
            "tests.t::test_a",
            "tests.t::test_b",
            "tests.t::test_c",
            "tests.t::test_d",
            "tests.t::test_e",
        ],
        materials_root=mats,
        source_hunk_count=16,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "gate-audit-product",
            "--product",
            str(product),
            "--materials",
            str(mats),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["accepted_count"] == 1
    assert "realpr-demo-1" in payload["accepted_ids"]


def test_audit_keep_refuses_thin_f2p_below_floor() -> None:
    """VAL-DHARD-002: f2p=1 refuse on product gate_audit by default."""
    row = audit_keep_dual_truth(
        task_id="thin-f2p",
        materials_root="datasets/live_materials",
        live_mine=True,
        label_method=LABEL_METHOD_LIVE,
        f2p_node_ids=["tests.t::only_one"],
        p2p_node_ids=["tests.t::p2p"],
        backend_class="HarborDockerVerifier",
        agent_image="a:tag",
        tests_image="t:tag",
        solution_reward=1,
        null_reward=0,
        source_track="real_pr",
        source_hunk_count=16,
        discovery_path="list_pulls",
    )
    assert not row.accepted
    joined = " ".join(row.reasons)
    assert "f2p_nodes_below_floor" in joined
    assert "thin_f2p_easy_class" in joined


def test_audit_keep_refuses_hunks_below_m27_floor() -> None:
    """VAL-DMED-001: source_hunk_count=13 refuse (floor 14)."""
    row = audit_keep_dual_truth(
        task_id="thin-hunks",
        materials_root="datasets/live_materials",
        live_mine=True,
        label_method=LABEL_METHOD_LIVE,
        f2p_node_ids=["a", "b", "c", "d", "e"],
        p2p_node_ids=["p"],
        backend_class="HarborDockerVerifier",
        agent_image="a:tag",
        tests_image="t:tag",
        solution_reward=1,
        null_reward=0,
        source_track="real_pr",
        source_hunk_count=13,
        discovery_path="list_pulls",
    )
    assert not row.accepted
    assert any("source_hunks_below_floor" in r for r in row.reasons)


def test_audit_keep_engineering_opt_out_skips_f2p_floor() -> None:
    """Offline / engineering opt-out may skip MIN_F2P (never product default)."""
    row = audit_keep_dual_truth(
        task_id="eng-opt-out",
        materials_root="fixtures/real_pr_ship",
        live_mine=False,
        label_method=LABEL_METHOD_LIVE,
        f2p_node_ids=["tests.t::only_one"],
        p2p_node_ids=["tests.t::p2p"],
        backend_class="HarborDockerVerifier",
        agent_image="a:tag",
        tests_image="t:tag",
        solution_reward=1,
        null_reward=0,
        source_hunk_count=12,
        engineering_opt_out=True,
    )
    assert row.accepted
    assert "engineering_opt_out" in row.reasons
