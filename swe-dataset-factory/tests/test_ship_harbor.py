"""Ship Harbor pack wave (VAL-HARBOR-008/009/010, VAL-CROSS-007)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree
from swe_factory.pipeline.ship_harbor import (
    DEFAULT_MAX_PACKS,
    DEFAULT_MIN_PACKS,
    run_harbor_load_smoke,
    run_ship_harbor,
)
from swe_factory.producers.harbor_variants import SHIP_MOTOR_SEEDS, list_ship_motor_seeds

runner = CliRunner()


def test_ship_motor_seed_inventory() -> None:
    assert len(SHIP_MOTOR_SEEDS) >= 12
    langs = {s.language for s in SHIP_MOTOR_SEEDS}
    assert langs >= {"python", "go", "typescript"}
    # each is multi-module hard
    assert all(s.hard_track for s in SHIP_MOTOR_SEEDS)
    assert all(len(s.green_modules) >= 2 for s in SHIP_MOTOR_SEEDS)
    assert all(not s.fault.uses_not_implemented for s in SHIP_MOTOR_SEEDS)
    # unique seed ids
    ids = [s.seed_id for s in SHIP_MOTOR_SEEDS]
    assert len(ids) == len(set(ids))
    # M8 scaled seed catalog (>=4 historical + expansion for ≥30 target).
    assert len(list_ship_motor_seeds(language="python")) >= 4
    assert len(list_ship_motor_seeds(language="go")) >= 4
    assert len(list_ship_motor_seeds(language="ts")) >= 4


def test_run_ship_harbor_offline_band(tmp_path: Path) -> None:
    """Fake-oracle ship produces 10–15 certified DeepSWE packs with Python/Go/TS."""
    out = tmp_path / "harbor_v1"
    result = run_ship_harbor(
        out_dir=out,
        work_root=tmp_path / "work",
        target_packs=12,
        min_packs=10,
        max_packs=15,
        oracle_mode="fake",
    )
    assert result.ok, result.reason
    assert DEFAULT_MIN_PACKS <= result.certified_count <= DEFAULT_MAX_PACKS
    assert result.languages.get("python", 0) >= 1
    assert result.languages.get("go", 0) >= 1
    assert result.languages.get("typescript", 0) >= 1
    # Artifacts
    assert result.report_path is not None and result.report_path.is_file()
    assert result.pack_manifest_path is not None and result.pack_manifest_path.is_file()
    assert result.ledger_summary_path is not None and result.ledger_summary_path.is_file()
    assert result.oracle_evidence_path is not None and result.oracle_evidence_path.is_file()
    assert result.under_cap is True
    assert result.provider_calls == 0
    assert result.harbor_load_smoke.get("ok") is True

    report = result.report_path.read_text(encoding="utf-8")
    assert "Language mix" in report or "language" in report.lower()
    assert "javascript=0" in report or "javascript" in report.lower()
    assert "rust" in report.lower()

    manifest = json.loads(result.pack_manifest_path.read_text(encoding="utf-8"))
    assert manifest["count"] == result.certified_count
    assert 10 <= manifest["count"] <= 15
    assert set(manifest["task_ids"]) == {r.task_id for r in result.records if r.certified}

    # Every certified pack: complete tree + multi-file + oracle evidence
    for rec in result.records:
        if not rec.certified:
            continue
        missing = verify_pack_tree(rec.pack_dir)
        assert missing == [], (rec.task_id, missing)
        for rel in REQUIRED_PACK_RELPATHS:
            assert (rec.pack_dir / rel).is_file()
        assert rec.solution_reward == 1
        assert rec.null_reward == 0
        assert rec.agent_isolated is True
        assert rec.multi_file_ok is True
        assert len(rec.solution_files) >= 2
        cfg = json.loads((rec.pack_dir / "tests" / "config.json").read_text(encoding="utf-8"))
        assert cfg.get("f2p_node_ids")
        assert (rec.pack_dir / "tests" / "test.patch").read_text(encoding="utf-8").strip()
        # Isolation of held-out
        assert not (rec.pack_dir / "environment" / "solution").exists()
        env_repo = rec.pack_dir / "environment" / "repo"
        if env_repo.is_dir():
            assert not list(env_repo.rglob("test.patch"))

    # Ledger under cap
    ledger = json.loads(result.ledger_summary_path.read_text(encoding="utf-8"))
    assert ledger.get("under_cap") is True
    total = float(ledger.get("total_commit_usd") or ledger.get("settled_exact_usd") or 0)
    assert total <= 600.0


def test_harbor_load_smoke_on_shipped_pack(tmp_path: Path) -> None:
    out = tmp_path / "harbor_v1"
    result = run_ship_harbor(
        out_dir=out,
        work_root=tmp_path / "work",
        target_packs=10,
        min_packs=10,
        max_packs=12,
        oracle_mode="fake",
        seed_limit=12,
    )
    assert result.ok
    smoke = run_harbor_load_smoke(out / "tasks")
    assert smoke["ok"] is True
    assert smoke["task_config_ok"] is True
    assert smoke["paths_ok"] is True
    assert smoke["listed"]


def test_ship_harbor_cli(tmp_path: Path) -> None:
    out = tmp_path / "harbor_v1_cli"
    res = runner.invoke(
        app,
        [
            "ship-harbor",
            "--out",
            str(out),
            "--work",
            str(tmp_path / "work_cli"),
            "--target",
            "10",
            "--min-packs",
            "10",
            "--max-packs",
            "12",
            "--oracle",
            "fake",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert 10 <= payload["certified_count"] <= 12
    assert (out / "report.md").is_file()
    assert (out / "pack_manifest.json").is_file()
    assert (out / "ledger_summary.json").is_file()
    assert (out / "tasks").is_dir()


@pytest.mark.parametrize("seed_id", [s.seed_id for s in SHIP_MOTOR_SEEDS[:3]])
def test_variant_materials_multi_file(tmp_path: Path, seed_id: str) -> None:
    from swe_factory.producers.harbor_motors import produce_harbor_materials
    from swe_factory.producers.harbor_variants import get_ship_motor_seed

    seed = get_ship_motor_seed(seed_id)
    materials = produce_harbor_materials(
        seed,
        work_root=tmp_path / "w",
        instance_suffix="ut",
    )
    assert materials.multi_file_ok
    assert len(materials.solution_files) >= 2
    assert materials.provider_calls == 0
