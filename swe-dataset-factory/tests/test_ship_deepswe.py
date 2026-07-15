"""Ship DeepSWE product surface (VAL-SHIP-001..005/008/010, VAL-XDEEP-001..006)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harbor.deepswe_cert import (
    FakeBackendRejected,
    is_real_base_sha,
    is_real_repository_url,
)
from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree
from swe_factory.harbor.harbor_oracle import FakeHarborVerifier, VerifierRunResult
from swe_factory.pipeline.ship_deepswe import (
    DEFAULT_MIN_PACKS,
    DEFAULT_TARGET_PACKS,
    pick_hybrid_identity,
    refuse_fake_ship_dest,
    run_ship_deepswe,
)

runner = CliRunner()


@dataclass
class ScriptedDockerVerifier:
    """Docker-named injectable backend for offline ship unit tests."""

    solution_reward: int | float = 1
    null_reward: int | float = 0
    cleaned: bool = False
    call_log: list[str] = field(default_factory=list)

    def run_solution(self, pack_dir: Path) -> VerifierRunResult:
        del pack_dir
        self.call_log.append("solution")
        reward = self.solution_reward
        return VerifierRunResult(
            phase="solution",
            reward=reward,
            reward_json={
                "reward": reward,
                "f2p_total": 2,
                "f2p_passed": 2 if reward == 1 else 0,
                "p2p_total": 1,
                "p2p_passed": 1 if reward == 1 else 0,
            },
            logs="scripted docker solution",
            ok=True,
        )

    def run_null(self, pack_dir: Path) -> VerifierRunResult:
        del pack_dir
        self.call_log.append("null")
        reward = self.null_reward
        return VerifierRunResult(
            phase="null",
            reward=reward,
            reward_json={
                "reward": reward,
                "f2p_total": 2,
                "f2p_passed": 0,
                "p2p_total": 1,
                "p2p_passed": 1 if reward == 0 else 0,
            },
            logs="scripted docker null",
            ok=True,
        )

    def cleanup(self) -> None:
        self.cleaned = True


def test_refuse_fake_ship_dest(tmp_path: Path) -> None:
    dest = tmp_path / "datasets" / "deepswe_v1"
    dest.mkdir(parents=True)
    with pytest.raises(FakeBackendRejected):
        refuse_fake_ship_dest("fake", out_dir=dest)
    with pytest.raises(FakeBackendRejected):
        refuse_fake_ship_dest("stub", out_dir=str(dest).replace("deepswe_v1", "deepswe-v1"))
    # docker ok
    refuse_fake_ship_dest("docker", out_dir=dest)


def test_pick_hybrid_identity_real_pins() -> None:
    used: set[str] = set()
    ids: list[Any] = []
    for lang in ("python", "go", "typescript", "python", "go"):
        ident = pick_hybrid_identity(lang, pack_index=len(ids), used=used)
        ids.append(ident)
        assert is_real_repository_url(ident.repository_url)
        assert is_real_base_sha(ident.base_commit)
        assert ident.license
        assert ident.source_track == "hybrid_curated"
        assert not ident.repository_url.lower().startswith("file://")
    assert len(ids) == 5


def test_run_ship_deepswe_offline_m10_band(tmp_path: Path) -> None:
    """Offline historical hybrid path: ≥113 certified packs (regression only)."""
    out = tmp_path / "deepswe_v1"
    backend = ScriptedDockerVerifier()
    result = run_ship_deepswe(
        out_dir=out,
        work_root=tmp_path / "work",
        target_packs=113,
        min_packs=113,
        max_packs=130,
        oracle_mode="docker",
        panel_mode="offline",
        pier_mode="scripted",
        docker_backend=backend,
        pier_jobs_root=Path("/tmp/harbor-deepswe-jobs-ut-ship"),
    )

    assert result.ok, result.reason
    assert result.certified_count >= DEFAULT_MIN_PACKS
    assert result.certified_count >= DEFAULT_TARGET_PACKS
    assert result.certified_count >= 113
    assert result.mode == "docker"
    assert result.under_cap is True
    assert result.harbor_load_smoke.get("ok") is True
    # Offline panel must not claim live spend; budget_stop false with remaining > 0.
    assert result.budget_stop is False

    # Artifacts (VAL-SHIP-002)
    assert result.report_path is not None and result.report_path.is_file()
    assert result.pack_manifest_path is not None and result.pack_manifest_path.is_file()
    assert result.ledger_summary_path is not None and result.ledger_summary_path.is_file()
    assert result.provenance_path is not None and result.provenance_path.is_file()
    assert result.oracle_evidence_path is not None and result.oracle_evidence_path.is_file()
    assert result.pier_evidence_path is not None and result.pier_evidence_path.is_file()
    assert result.e2e_drip_path is not None and result.e2e_drip_path.is_file()

    report = result.report_path.read_text(encoding="utf-8")
    assert "deepswe_v1" in report.lower() or "DeepSWE" in report
    assert "harbor_v1" in report.lower()  # labeled as fixture
    assert "language" in report.lower()
    assert "javascript" in report.lower()
    assert "rust" in report.lower()
    # VAL-SHIP-009 honesty + VAL-SHIP-007 parity comparison sealed in report.
    assert "under-supply" in report.lower()
    assert "113" in report
    assert "DeepSWE parity" in report or "parity comparison" in report.lower()
    assert result.provenance_path.is_file()
    assert any("javascript=0" in u or "rust=0" in u for u in result.under_supply_reasons)

    provenance = result.provenance_path.read_text(encoding="utf-8")
    # One row per certified pack
    cert_ids = [r.task_id for r in result.records if r.certified]
    for tid in cert_ids:
        assert tid in provenance
    assert provenance.count("| `") >= result.certified_count

    manifest = json.loads(result.pack_manifest_path.read_text(encoding="utf-8"))
    assert manifest["count"] == result.certified_count
    assert manifest["refuse_fake"] is True
    assert manifest["product_surface"] == "datasets/deepswe_v1"
    assert "datasets/harbor_v1" in manifest["historical_fixtures_only"]
    assert set(manifest["task_ids"]) == set(cert_ids)

    oracle_idx = json.loads(result.oracle_evidence_path.read_text(encoding="utf-8"))
    assert oracle_idx["backend"] == "docker"
    assert oracle_idx["refuse_fake"] is True

    languages_seen: set[str] = set()
    secrets_blob = ""
    for rec in result.records:
        if not rec.certified:
            continue
        languages_seen.add(rec.language)
        missing = verify_pack_tree(rec.pack_dir)
        assert missing == [], (rec.task_id, missing)
        for rel in REQUIRED_PACK_RELPATHS:
            assert (rec.pack_dir / rel).is_file()
        assert rec.solution_reward == 1
        assert rec.null_reward == 0
        assert rec.docker_oracle_certified is True
        assert rec.pier_certified is True
        assert rec.pier_oracle_reward == 1
        assert rec.pier_null_reward == 0
        assert rec.agent_isolated is True
        assert rec.multi_file_ok is True
        assert len(rec.solution_files) >= 2
        assert rec.hybrid is not None
        assert is_real_repository_url(rec.hybrid.repository_url)
        assert is_real_base_sha(rec.hybrid.base_commit)
        # Pack metadata matches hybrid bind
        toml = (rec.pack_dir / "task.toml").read_text(encoding="utf-8")
        assert rec.hybrid.repository_url in toml
        assert rec.hybrid.base_commit in toml
        assert "file://" not in toml
        assert "oracle_mode=fake" not in toml
        secrets_blob += toml
        secrets_blob += (rec.pack_dir / "instruction.md").read_text(encoding="utf-8")
        # Isolation
        assert not (rec.pack_dir / "environment" / "solution").exists()
        # E2E drip stages
        stages = [row["stage"] for row in rec.drip]
        for must in (
            "produce",
            "hybrid_bind",
            "real_pack",
            "docker_oracle",
            "pier_cert",
            "panel",
            "promote",
        ):
            assert must in stages, (rec.task_id, stages)

    # M10 language ambition: ≥3 languages among Py/TS/Go when funnel allows
    assert len(languages_seen & {"python", "go", "typescript"}) >= 3

    # No secrets in ship outputs (VAL-XDEEP-006)
    for token in ("sk-", "OPENROUTER_API_KEY=", "OXYLABS_PASSWORD="):
        assert token not in secrets_blob
        assert token not in report
        assert token not in provenance

    ledger = json.loads(result.ledger_summary_path.read_text(encoding="utf-8"))
    assert ledger.get("under_cap") is True
    total = float(ledger.get("total_commit_usd") or ledger.get("settled_exact_usd") or 0)
    assert total <= 600.0

    drip_text = result.e2e_drip_path.read_text(encoding="utf-8")
    assert "docker_oracle" in drip_text
    assert "pier_cert" in drip_text


def test_ship_deepswe_rejects_fake_oracle(tmp_path: Path) -> None:
    out = tmp_path / "deepswe_v1"
    with pytest.raises(FakeBackendRejected):
        run_ship_deepswe(
            out_dir=out,
            work_root=tmp_path / "work",
            target_packs=1,
            min_packs=1,
            max_packs=3,
            oracle_mode="fake",
            panel_mode="skip",
            pier_mode="scripted",
            docker_backend=FakeHarborVerifier(),
            pier_jobs_root=Path("/tmp/harbor-deepswe-jobs-ut-fake"),
        )


def test_ship_deepswe_cli_refuses_fake(tmp_path: Path) -> None:
    out = tmp_path / "deepswe_v1"
    res = runner.invoke(
        app,
        [
            "ship-deepswe",
            "--out",
            str(out),
            "--oracle",
            "fake",
            "--target",
            "1",
            "--min-packs",
            "1",
            "--json",
        ],
    )
    assert res.exit_code == 2, res.output
    assert "refuse" in res.output.lower() or "docker only" in res.output.lower()


def test_ship_deepswe_cli_help() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "ship-deepswe" in res.output
    # Real-PR product path discoverable from top-level help (VAL-RX-006).
    assert "real_pr" in res.output.lower() or "Real-PR" in res.output or "≥5" in res.output
    res2 = runner.invoke(app, ["ship-deepswe", "--help"])
    assert res2.exit_code == 0
    assert "deepswe_v1" in res2.output
    assert "docker" in res2.output.lower()
    assert "real_pr" in res2.output
    # Real-PR defaults ≥5; historical hybrid path remains documented in --source help.
    assert "5" in res2.output
    assert "hybrid" in res2.output.lower()


def test_ship_deepswe_cli_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI path ensures docker-only ship with injected offline docker backend.

    Uses a compact band so CLI wiring is covered without re-running the full M10
    113-pack offline matrix (that scale gate lives in
    ``test_run_ship_deepswe_offline_m10_band``).
    """
    import swe_factory.pipeline.ship_deepswe as ship_mod

    out = tmp_path / "deepswe_v1"
    original = ship_mod.run_ship_deepswe

    def wrapped(**kwargs: Any):
        kwargs["docker_backend"] = ScriptedDockerVerifier()
        kwargs.setdefault("pier_jobs_root", Path("/tmp/harbor-deepswe-jobs-ut-cli"))
        kwargs.setdefault("panel_mode", "offline")
        kwargs.setdefault("pier_mode", "scripted")
        return original(**kwargs)

    monkeypatch.setattr(ship_mod, "run_ship_deepswe", wrapped)

    res = runner.invoke(
        app,
        [
            "ship-deepswe",
            "--source",
            "hybrid_curated",
            "--out",
            str(out),
            "--work",
            str(tmp_path / "work_cli"),
            "--target",
            "5",
            "--min-packs",
            "5",
            "--max-packs",
            "12",
            "--oracle",
            "docker",
            "--panel",
            "offline",
            "--pier",
            "scripted",
            "--pier-jobs",
            "/tmp/harbor-deepswe-jobs-ut-cli",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["certified_count"] >= 5
    assert (out / "PROVENANCE.md").is_file()
    assert (out / "report.md").is_file()
    assert (out / "pack_manifest.json").is_file()
    assert (out / "tasks").is_dir()
    assert len(list((out / "tasks").iterdir())) >= 5


def test_ship_rejects_historical_fixture_out(tmp_path: Path) -> None:
    from swe_factory.pipeline.ship_deepswe import ShipDeepSWEError

    with pytest.raises(ShipDeepSWEError):
        run_ship_deepswe(
            out_dir=tmp_path / "harbor_v1",
            work_root=tmp_path / "work",
            target_packs=1,
            min_packs=1,
            max_packs=2,
            oracle_mode="docker",
            panel_mode="skip",
            pier_mode="scripted",
            docker_backend=ScriptedDockerVerifier(),
            pier_jobs_root=Path("/tmp/harbor-deepswe-jobs-ut-hist"),
        )


def test_no_secret_patterns_in_module_strings() -> None:
    src = Path("src/swe_factory/pipeline/ship_deepswe.py").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9]{10,}", src)
    assert "OXYLABS_PASSWORD=" not in src


def test_list_ship_motor_seeds_m10_scale() -> None:
    from swe_factory.producers.harbor_variants import (
        M10_SHIP_VARIANTS_PER_LANG,
        list_ship_motor_seeds,
    )

    seeds = list_ship_motor_seeds()
    assert len(seeds) >= 113
    assert len(seeds) >= M10_SHIP_VARIANTS_PER_LANG * 3
    langs = {s.language for s in seeds}
    assert {"python", "go", "typescript"} <= langs
    # Unique motor seed ids and synthetic base commits (hybrid ship rewrites SHA).
    assert len({s.seed_id for s in seeds}) == len(seeds)
    assert len({s.base_commit for s in seeds}) == len(seeds)


def test_run_ship_deepswe_offline_m9_band_compat(tmp_path: Path) -> None:
    """Earlier M9 band still works (compat gate at 70)."""
    out = tmp_path / "deepswe_v1"
    backend = ScriptedDockerVerifier()
    result = run_ship_deepswe(
        out_dir=out,
        work_root=tmp_path / "work",
        target_packs=70,
        min_packs=70,
        max_packs=90,
        oracle_mode="docker",
        panel_mode="offline",
        pier_mode="scripted",
        docker_backend=backend,
        pier_jobs_root=Path("/tmp/harbor-deepswe-jobs-ut-ship-m9"),
    )
    assert result.ok, result.reason
    assert result.certified_count >= 70
    assert result.mode == "docker"


def test_run_ship_deepswe_offline_m8_band_compat(tmp_path: Path) -> None:
    """Earlier M8 band still works (compat gate at 30)."""
    out = tmp_path / "deepswe_v1"
    backend = ScriptedDockerVerifier()
    result = run_ship_deepswe(
        out_dir=out,
        work_root=tmp_path / "work",
        target_packs=30,
        min_packs=30,
        max_packs=40,
        oracle_mode="docker",
        panel_mode="offline",
        pier_mode="scripted",
        docker_backend=backend,
        pier_jobs_root=Path("/tmp/harbor-deepswe-jobs-ut-ship-m8"),
    )
    assert result.ok, result.reason
    assert result.certified_count >= 30
    assert result.mode == "docker"


def test_run_ship_deepswe_offline_m7_band_compat(tmp_path: Path) -> None:
    """Smaller band still works (compat for unit gate without full 113)."""
    out = tmp_path / "deepswe_v1"
    backend = ScriptedDockerVerifier()
    result = run_ship_deepswe(
        out_dir=out,
        work_root=tmp_path / "work",
        target_packs=5,
        min_packs=5,
        max_packs=12,
        oracle_mode="docker",
        panel_mode="offline",
        pier_mode="scripted",
        docker_backend=backend,
        pier_jobs_root=Path("/tmp/harbor-deepswe-jobs-ut-ship-m7"),
    )
    assert result.ok, result.reason
    assert result.certified_count >= 5
    assert result.mode == "docker"
