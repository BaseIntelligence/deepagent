"""Offline fixture end-to-end path (VAL-CROSS-001).

Proves package wiring offline: fixture mono -> TaskRecord schema write ->
stub oracle gates pass -> exported gate-demo artifact, with zero OpenRouter
or LLM provider calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.fixture.offline import (
    FIXTURE_INSTANCE_ID,
    OfflineFixtureError,
    default_fixture_root,
    run_offline_fixture_pipeline,
)
from swe_factory.oracle.gates import GateResult, run_stub_gates
from swe_factory.schema import SourceTrack, TaskRecord

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_fixture_root_exists() -> None:
    root = default_fixture_root()
    assert root.is_dir()
    assert (root / "task_meta.json").is_file()
    assert (root / "gold.patch").is_file()
    assert (root / "repo").is_dir()
    # Tiny multi-file monomer: at least two source modules + tests
    assert (root / "repo" / "demo_pkg" / "math_ops.py").is_file()
    assert (root / "repo" / "demo_pkg" / "text_ops.py").is_file()


def test_offline_pipeline_writes_schema_valid_artifact(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixture_out"
    result = run_offline_fixture_pipeline(out_dir=out_dir)

    assert result.instance_id == FIXTURE_INSTANCE_ID
    assert result.task.source_track == SourceTrack.SYNTHETIC_GROUNDED
    assert result.gates.passed is True
    assert result.gates.accepted is True
    assert result.provider_calls == 0

    tasks_path = out_dir / "tasks.jsonl"
    assert tasks_path.is_file()
    lines = [ln for ln in tasks_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = TaskRecord.model_validate_json(lines[0])
    assert record.instance_id == FIXTURE_INSTANCE_ID
    assert record.gold_patch.strip()
    assert len(record.fail_to_pass) >= 1
    assert record.environment.image_digest
    assert record.gate_proof is not None
    assert record.gate_proof.get("mode") == "stub_offline"
    assert record.gate_proof.get("passed") is True

    audit_path = out_dir / "gate_audit.jsonl"
    assert audit_path.is_file()
    audit_line = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert audit_line["instance_id"] == FIXTURE_INSTANCE_ID
    assert audit_line["disposition"] == "accept"
    assert "reason_codes" in audit_line

    # Gold must not appear in the agent-visible workspace layout
    workspace = out_dir / "tasks" / FIXTURE_INSTANCE_ID
    assert workspace.is_dir()
    assert (workspace / "problem_statement.md").is_file()
    assert not (workspace / "gold.patch").exists()
    assert not (workspace / "patch.diff").exists()
    gold_snippet = record.gold_patch.strip().splitlines()[0]
    problem_text = (workspace / "problem_statement.md").read_text(encoding="utf-8")
    assert gold_snippet not in problem_text


def test_stub_gates_pass_multi_file_fixture_task() -> None:
    # Build via pipeline internals on the stock fixture
    out = run_offline_fixture_pipeline(out_dir=Path("/tmp/unused"), dry_write=True)
    gate: GateResult = run_stub_gates(out.task)
    assert gate.passed
    assert "G1" in gate.reasons or any(c.startswith("G1") for c in gate.reason_codes)
    codes = set(gate.reason_codes)
    assert any(c.startswith("G") for c in codes)
    # Multi-file hard floor for demo
    assert gate.multi_file is True
    assert gate.files_touched >= 2


def test_stub_gates_reject_empty_f2p() -> None:
    # Empty F2P fails schema for TaskRecord; exercise structural gate helper directly.
    from swe_factory.oracle.gates import evaluate_stub_gate_fields

    result = evaluate_stub_gate_fields(
        gold_patch="diff --git a/a.py b/a.py\n",
        fail_to_pass=[],
        problem_statement="something is wrong",
        image_digest="sha256:fixture",
    )
    assert result.passed is False
    assert "G1_EMPTY_F2P" in result.reason_codes or any("F2P" in c for c in result.reason_codes)


def test_stub_gates_reject_single_file_gold() -> None:
    from swe_factory.oracle.gates import evaluate_stub_gate_fields

    single = "diff --git a/only.py b/only.py\n--- a/only.py\n+++ b/only.py\n@@ -1 +1 @@\n-x\n+y\n"
    result = evaluate_stub_gate_fields(
        gold_patch=single,
        fail_to_pass=["pytest tests/test_math.py"],
        problem_statement="fix only one file",
        image_digest="sha256:fixture",
    )
    assert result.passed is False
    assert "G4_MULTI_FILE" in result.reason_codes or any("MULTI" in c for c in result.reason_codes)


def test_cli_offline_fixture_command(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_fixture"
    # Ensure no OpenRouter key is needed: clear any leaked env in process
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)

    result = runner.invoke(
        app,
        ["offline-fixture", "--out", str(out_dir)],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert "offline" in result.output.lower() or "fixture" in result.output.lower()
    assert "provider_calls=0" in result.output or "provider calls: 0" in result.output.lower()
    assert (out_dir / "tasks.jsonl").is_file()
    record = TaskRecord.model_validate_json(
        (out_dir / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert record.instance_id == FIXTURE_INSTANCE_ID
    assert record.gate_proof is not None
    assert record.gate_proof.get("passed") is True


def test_cli_help_lists_offline_fixture() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "offline-fixture" in result.output


def test_offline_pipeline_never_hits_openrouter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guard: any accidental HTTP to OpenRouter must fail the test path."""

    def _block_http(*_a: object, **_k: object) -> None:
        raise AssertionError("unexpected network/provider call during offline fixture")

    monkeypatch.setattr("httpx.Client.request", _block_http, raising=False)
    monkeypatch.setattr("httpx.AsyncClient.request", _block_http, raising=False)

    out = run_offline_fixture_pipeline(out_dir=tmp_path / "blocked_net")
    assert out.provider_calls == 0
    assert out.gates.passed


def test_missing_fixture_raises(tmp_path: Path) -> None:
    with pytest.raises(OfflineFixtureError):
        run_offline_fixture_pipeline(fixture_root=tmp_path / "missing", out_dir=tmp_path / "o")


def test_no_sdf_containers_left_after_offline() -> None:
    """Offline path must not spawn Docker containers; residual sdf-* list empty."""
    import subprocess

    before = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=sdf-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    # If docker missing, skip residual assertion (offline doesn't need it)
    if before.returncode != 0:
        pytest.skip("docker unavailable")

    names_before = {n for n in before.stdout.splitlines() if n.strip()}
    run_offline_fixture_pipeline(out_dir=REPO_ROOT / "datasets" / ".tmp_fixture_test")
    after = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=sdf-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    names_after = {n for n in after.stdout.splitlines() if n.strip()}
    created = names_after - names_before
    assert not created, f"offline fixture left sdf-* containers: {created}"
