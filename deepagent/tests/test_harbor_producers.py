"""Harbor multi-lang producers (VAL-HARBOR-007 + DeepAgent materials).

Covers offline motors for Python, Go, TypeScript:
- multi-file solution.patch hard floor
- held-out test.patch not in agent package
- non-empty f2p/p2p node ids
- long-horizon behavioral instruction.md
- avoid sole NotImplemented hard-set faults
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree
from swe_factory.harbor.harbor_docker import (
    list_agent_context_paths,
    scan_agent_context_forbidden,
    stage_agent_context,
)
from swe_factory.oracle.gates import count_files_in_patch
from swe_factory.producers.harbor_labeling import (
    HarborLabelError,
    SuiteOutcome,
    assert_broken_matches_labels,
    compute_dual_run_labels,
    label_cohorts_from_outcomes,
    pytest_nodeid_to_harbor,
)
from swe_factory.producers.harbor_motors import (
    HARD_MULTI_FILE_FLOOR,
    MOTOR_SEEDS,
    HarborMotorError,
    _apply_fault_plan,
    _copy_tree,
    build_held_out_test_patch,
    build_long_horizon_instruction,
    get_motor_seed,
    list_motor_seeds,
    produce_all_offline_motors,
    produce_harbor_materials,
    produce_harbor_pack,
)
from swe_factory.producers.harbor_variants import SHIP_MOTOR_SEEDS
from swe_factory.sources.allowlist import (
    HARBOR_MOTOR_SEEDS,
    harbor_motor_seeds,
    local_offline_seeds,
)

runner = CliRunner()


def test_motor_seeds_cover_python_go_ts() -> None:
    langs = {s.language for s in MOTOR_SEEDS}
    assert langs >= {"python", "go", "typescript"}
    assert len(MOTOR_SEEDS) >= 3
    assert all(s.hard_track for s in MOTOR_SEEDS)
    assert all(len(s.green_modules) >= 2 for s in MOTOR_SEEDS)
    assert all(not s.fault.uses_not_implemented for s in MOTOR_SEEDS)
    # Allowlist projection posts the same offline motors
    offline_ids = {s.seed_id for s in local_offline_seeds()}
    for seed in HARBOR_MOTOR_SEEDS:
        assert seed.seed_id in offline_ids
        assert seed.modular
    assert len(harbor_motor_seeds()) == 3


@pytest.mark.parametrize("seed_id", [s.seed_id for s in MOTOR_SEEDS])
def test_produce_materials_multi_file_and_held_out(tmp_path: Path, seed_id: str) -> None:
    """VAL-HARBOR-007: hard packs touch ≥2 source files; held-out tests absent from agent."""
    seed = get_motor_seed(seed_id)
    materials = produce_harbor_materials(
        seed,
        work_root=tmp_path / "work",
        instance_suffix="ut",
    )
    assert materials.provider_calls == 0
    assert materials.multi_file_ok is True
    assert len(materials.solution_files) >= HARD_MULTI_FILE_FLOOR
    files = count_files_in_patch(materials.solution_patch)
    product = [p for p in files if not p.startswith("tests/")]
    assert len(product) >= HARD_MULTI_FILE_FLOOR
    assert materials.test_patch.strip()
    assert materials.f2p_node_ids
    assert all(materials.f2p_node_ids)
    # Held-out content not baked into broken agent tree
    held = materials.broken_workspace / seed.held_out.relative_path
    assert not held.exists()
    # Instruction is long-horizon behavioral
    assert len(materials.instruction_md) >= 400
    assert "Fail-to-pass" in materials.instruction_md or "fail-to-pass" in materials.instruction_md
    assert "multi-file" in materials.instruction_md.lower()
    # No NotImplemented as sole hard set approach
    assert materials.notes.get("uses_not_implemented") is False
    assert "NotImplementedError" not in materials.solution_patch or len(product) >= 2


@pytest.mark.parametrize("seed_id", [s.seed_id for s in MOTOR_SEEDS])
def test_produce_pack_tree_and_agent_isolation(tmp_path: Path, seed_id: str) -> None:
    seed = get_motor_seed(seed_id)
    result = produce_harbor_pack(
        seed,
        out_dir=tmp_path / "out",
        work_root=tmp_path / "work",
        instance_suffix="pack",
    )
    assert result.pack_dir is not None
    assert result.missing == ()
    missing = verify_pack_tree(result.pack_dir)
    assert missing == []
    for rel in REQUIRED_PACK_RELPATHS:
        assert (result.pack_dir / rel).is_file()

    sol = (result.pack_dir / "solution" / "solution.patch").read_text(encoding="utf-8")
    product = [p for p in count_files_in_patch(sol) if not p.startswith("tests/")]
    assert len(product) >= HARD_MULTI_FILE_FLOOR

    cfg = json.loads((result.pack_dir / "tests" / "config.json").read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"]
    assert (result.pack_dir / "tests" / "test.patch").read_text(encoding="utf-8").strip()

    # Held-out patch not in agent environment/repo
    env_repo = result.pack_dir / "environment" / "repo"
    assert env_repo.is_dir()
    assert not (env_repo / seed.held_out.relative_path).exists()
    assert not (result.pack_dir / "environment" / "solution").exists()
    assert not (env_repo / "tests" / "test.patch").exists()

    # Stage agent context like docker path and scan isolation
    ctx = stage_agent_context(result.pack_dir, tmp_path / f"agent_{seed_id}")
    hits = scan_agent_context_forbidden(ctx.context_dir)
    assert hits == []
    paths = list_agent_context_paths(ctx.context_dir)
    assert not any(p.endswith("test.patch") for p in paths)
    assert not any(p.startswith("solution") or "/solution/" in p for p in paths)


def test_produce_all_offline_motors(tmp_path: Path) -> None:
    results = produce_all_offline_motors(
        out_dir=tmp_path / "all",
        work_root=tmp_path / "work",
        instance_suffix="all",
    )
    assert len(results) == 3
    langs = {r.materials.language for r in results}
    assert langs == {"python", "go", "typescript"}
    manifest = json.loads((tmp_path / "all" / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 3
    assert manifest["provider_calls"] == 0
    assert set(manifest["languages"]) == {"python", "go", "typescript"}
    for task_id, files in manifest["multi_file"].items():
        assert len(files) >= HARD_MULTI_FILE_FLOOR, task_id


def test_held_out_test_patch_has_new_file_header() -> None:
    seed = get_motor_seed("harbor_python_orders")
    patch = build_held_out_test_patch(seed.held_out)
    assert "diff --git" in patch
    assert "new file mode" in patch
    assert seed.held_out.relative_path in patch
    assert "test_multi_module_checkout_contract" in patch


def test_long_horizon_instruction_mentions_modules() -> None:
    seed = list_motor_seeds(language="go")[0]
    text = build_long_horizon_instruction(
        seed=seed,
        fault_files=list(seed.green_modules),
        f2p=seed.base_f2p_node_ids + seed.held_out.f2p_node_ids,
        p2p=seed.base_p2p_node_ids,
    )
    assert "store.go" in text
    assert "router.go" in text
    assert "Behavioural requirements" in text or "Behavioral" in text or "Behavioural" in text


def test_fault_span_missing_raises(tmp_path: Path) -> None:
    seed = get_motor_seed("harbor_python_orders")
    # Build a sibling tree with truncated modules so fault old-spans are absent.
    broken_root = tmp_path / "bad_seed"
    dest = broken_root / "repo"
    dest.mkdir(parents=True)
    (dest / "orderlib").mkdir()
    (dest / "orderlib" / "pricing.py").write_text(
        "def x():\n    return 1\n",
        encoding="utf-8",
    )
    (dest / "orderlib" / "inventory.py").write_text(
        "class Inventory: pass\n",
        encoding="utf-8",
    )
    (dest / "orderlib" / "checkout.py").write_text(
        "class CheckoutService: pass\n",
        encoding="utf-8",
    )
    from swe_factory.producers.harbor_motors import _apply_fault_plan

    with pytest.raises(HarborMotorError, match="fault old-span|fault target"):
        _apply_fault_plan(dest, seed.fault)


def test_cli_harbor_produce_offline(tmp_path: Path) -> None:
    out = tmp_path / "cli_out"
    result = runner.invoke(
        app,
        [
            "harbor-produce",
            "--offline",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["count"] == 3
    assert payload["hard_multi_file_ok"] is True
    assert payload["provider_calls"] == 0
    assert set(payload["languages"]) == {"python", "go", "typescript"}
    for pack in payload["packs"]:
        assert len(pack["solution_files"]) >= 2
        assert pack["f2p_node_ids"]
        assert Path(pack["pack_dir"]).is_dir()


def test_cli_harbor_produce_language_filter(tmp_path: Path) -> None:
    out = tmp_path / "py_only"
    result = runner.invoke(
        app,
        [
            "harbor-produce",
            "--offline",
            "--language",
            "python",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["languages"] == ["python"]


def test_cli_list_seeds() -> None:
    result = runner.invoke(app, ["harbor-produce", "--list-seeds", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    ids = {row["seed_id"] for row in payload["seeds"]}
    assert "harbor_python_orders" in ids
    assert "harbor_go_kvstore" in ids
    assert "harbor_ts_registry" in ids


def test_label_cohorts_from_outcomes_pure() -> None:
    f2p, p2p = label_cohorts_from_outcomes(
        green_passed={"a", "b", "c", "d"},
        green_failed=set(),
        broken_passed={"b", "d"},
        broken_failed={"a", "c"},
    )
    assert f2p == ("a", "c")
    assert p2p == ("b", "d")


def test_assert_broken_matches_labels_rejects_mislabel() -> None:
    broken = SuiteOutcome(
        language="python",
        passed=("p2p_ok", "mis_f2p_still_green"),
        failed=("true_f2p", "mis_p2p_red"),
    )
    with pytest.raises(HarborLabelError):
        assert_broken_matches_labels(
            broken=broken,
            f2p_node_ids=["true_f2p", "mis_f2p_still_green"],
            p2p_node_ids=["p2p_ok", "mis_p2p_red"],
        )
    assert_broken_matches_labels(
        broken=broken,
        f2p_node_ids=["true_f2p", "mis_p2p_red"],
        p2p_node_ids=["p2p_ok", "mis_f2p_still_green"],
    )


def test_pytest_nodeid_to_harbor() -> None:
    assert (
        pytest_nodeid_to_harbor("tests/test_inventory.py::test_reserve_atomic")
        == "tests.test_inventory.test_reserve_atomic"
    )


@pytest.mark.parametrize("seed_id", [s.seed_id for s in MOTOR_SEEDS])
def test_dual_run_labels_broken_fails_exactly_f2p(tmp_path: Path, seed_id: str) -> None:
    """Broken suite fails every F2P node and passes every P2P node (VAL-HARBOR-006)."""
    seed = get_motor_seed(seed_id)
    green = tmp_path / "green"
    broken = tmp_path / "broken"
    _copy_tree(seed.green_repo(), green)
    _copy_tree(seed.green_repo(), broken)
    _apply_fault_plan(broken, seed.fault)

    labels = compute_dual_run_labels(
        language=seed.language,
        green_repo=green,
        broken_repo=broken,
        held_out_relative_path=seed.held_out.relative_path,
        held_out_content=seed.held_out.content,
    )
    assert labels.f2p_node_ids, "F2P set must be non-empty"
    assert set(labels.f2p_node_ids).isdisjoint(labels.p2p_node_ids)
    assert_broken_matches_labels(
        broken=labels.broken,
        f2p_node_ids=labels.f2p_node_ids,
        p2p_node_ids=labels.p2p_node_ids,
    )
    # Green must keep every labeled node.
    assert set(labels.f2p_node_ids) <= labels.green.passed_set
    assert set(labels.p2p_node_ids) <= labels.green.passed_set
    # Regression proof vs historical mislabels for base motors:
    if seed_id == "harbor_python_orders":
        assert "tests.test_inventory.test_reserve_atomic" in labels.p2p_node_ids
        assert "tests.test_inventory.test_reserve_rejects_partial" in labels.f2p_node_ids
    if seed_id == "harbor_go_kvstore":
        assert "TestStoreSetGet" in labels.p2p_node_ids
        assert "TestRouterRemoveMissing" in labels.f2p_node_ids
    if seed_id == "harbor_ts_registry":
        assert "catalog add and get" in labels.p2p_node_ids
        assert "catalog findByTag" in labels.f2p_node_ids


@pytest.mark.parametrize("seed_id", [s.seed_id for s in MOTOR_SEEDS])
def test_produce_materials_uses_dual_run_labels(tmp_path: Path, seed_id: str) -> None:
    seed = get_motor_seed(seed_id)
    materials = produce_harbor_materials(
        seed,
        work_root=tmp_path / "work",
        instance_suffix="labels",
    )
    assert materials.dual_run is not None
    assert materials.notes.get("label_method") == "dual_run_broken_vs_green"
    assert materials.f2p_node_ids == materials.dual_run.f2p_node_ids
    assert materials.p2p_node_ids == materials.dual_run.p2p_node_ids
    assert_broken_matches_labels(
        broken=materials.dual_run.broken,
        f2p_node_ids=materials.f2p_node_ids,
        p2p_node_ids=materials.p2p_node_ids,
    )


@pytest.mark.parametrize(
    "seed_id",
    [s.seed_id for s in SHIP_MOTOR_SEEDS],
    ids=[s.seed_id for s in SHIP_MOTOR_SEEDS],
)
def test_ship_variants_dual_run_labels(tmp_path: Path, seed_id: str) -> None:
    """Each ship fault-variant gets its own dual-run F2P/P2P (not a shared hand guess)."""
    from swe_factory.producers.harbor_variants import get_ship_motor_seed

    seed = get_ship_motor_seed(seed_id)
    materials = produce_harbor_materials(
        seed,
        work_root=tmp_path / "work",
        instance_suffix="varlabel",
    )
    assert materials.f2p_node_ids
    assert materials.dual_run is not None
    assert_broken_matches_labels(
        broken=materials.dual_run.broken,
        f2p_node_ids=materials.f2p_node_ids,
        p2p_node_ids=materials.p2p_node_ids,
    )
    # Labels may legitimately differ from advisory base_* defaults under other faults.
    cfg_style_f2p = set(materials.f2p_node_ids)
    # Broken fails F2P and none of P2P (already asserted); green passes all labels.
    assert cfg_style_f2p <= set(materials.dual_run.green.passed)


def test_python_motor_fixture_green_tests_pass() -> None:
    """Offline motor green tree itself is multi-module and importable."""
    import os
    import subprocess
    import sys

    seed = get_motor_seed("harbor_python_orders")
    repo = seed.green_repo()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.integration
def test_python_harbor_motor_oracle_docker_smoke(tmp_path: Path) -> None:
    """Docker smoke: harbor-oracle on python multi-file motor pack (solution=1 null=0)."""
    from swe_factory.harbor.harbor_oracle import HarborDockerVerifier, run_harbor_oracle

    result = produce_harbor_pack(
        "harbor_python_orders",
        out_dir=tmp_path / "out",
        work_root=tmp_path / "work",
        instance_suffix="dock",
    )
    assert result.pack_dir is not None
    assert len(result.materials.solution_files) >= 2
    try:
        oracle = run_harbor_oracle(
            result.pack_dir,
            backend=HarborDockerVerifier(run_id="hmotord"),
            mode="docker",
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"docker harbor oracle unavailable: {exc}")
    assert oracle.solution.reward == 1
    assert oracle.null.reward == 0
    assert oracle.agent_isolated is True
    assert oracle.passed is True
