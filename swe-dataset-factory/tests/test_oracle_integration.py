"""Docker integration tests for certified oracle gates on tiny_offline fixture.

Marked ``integration``. Validates:
- VAL-ORACLE-001 G1 F2P fail on broken
- VAL-ORACLE-002 G2 gold dual-run F2P+P2P pass
- VAL-ORACLE-003 G3 null patch scores zero / does not resolve
- VAL-ORACLE-004 G4 multi-file (structural, stock fixture)
- VAL-ORACLE-005 flake path reused from unit; dual-run must agree
- VAL-ORACLE-006 gate_audit.jsonl with stable codes
- VAL-ENV-003 no leftover sdf-* containers; off-limits untouched
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers
from swe_factory.envbuild.fixture import default_offline_fixture_root
from swe_factory.oracle import codes as C
from swe_factory.oracle.docker_run import OracleDockerRunner
from swe_factory.oracle.gates import append_gate_audit, run_certified_gates

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]


def _docker_ok() -> bool:
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=30,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.fixture(scope="module")
def docker_ready() -> DockerCLI:
    if not _docker_ok():
        pytest.skip("Docker daemon not available")
    return DockerCLI()


def _snapshot_off_limits() -> dict[str, str]:
    completed = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    out: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        name, _, status = line.partition("\t")
        if any(token in name for token in ("mission-test-pg", "challenge-prism", "acproxy")):
            out[name] = status
    return out


def _load_gold_and_workspace() -> tuple[str, Path, list[str], list[str]]:
    root = default_offline_fixture_root()
    gold = (root / "gold.patch").read_text(encoding="utf-8")
    meta = json.loads((root / "task_meta.json").read_text(encoding="utf-8"))
    f2p = list(meta["fail_to_pass"])
    p2p = list(meta.get("pass_to_pass") or [])
    return gold, root / "repo", f2p, p2p


def test_certified_gates_tiny_offline_docker(docker_ready: DockerCLI, tmp_path: Path) -> None:
    docker = docker_ready
    before_ol = _snapshot_off_limits()
    before_sdf = [n for n in docker.list_containers(all_containers=True) if n.startswith("sdf-")]

    gold, workspace, f2p, p2p = _load_gold_and_workspace()
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "problem_statement.md").write_text(
        "Restore multi-module add/reverse behaviour.\n", encoding="utf-8"
    )

    runner = OracleDockerRunner(
        docker=docker,
        base_image="python:3.12-slim",
        install_commands=["pip install -q pytest"],
        run_id="orit01",
        memory_mb=1024,
        cpus=1.0,
        command_timeout=120.0,
        keep_eval_image=False,
    )

    try:
        result = run_certified_gates(
            gold_patch=gold,
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            problem_statement="Restore multi-module behaviour for add and reverse_words.",
            image_digest="sha256:oracle_integration_fixture",
            workspace=workspace,
            runner=runner,
            agent_workspace=agent,
            require_multi_file=True,
            dual_runs=2,
            check_null_patch=True,
            check_leak=True,
        )
        # Hygiene sweep
        leftover = remove_leftover_sdf_containers(docker)

        assert result.passed is True, (
            f"certified gates failed: {result.reason_codes}\n{result.reasons}\n{result.details}"
        )
        assert C.G1_F2P_FAIL_OK in result.reason_codes
        assert C.G2_GOLD_DUAL_PASS in result.reason_codes
        assert C.G3_NULL_NOT_RESOLVE in result.reason_codes
        assert C.G4_MULTI_FILE_OK in result.reason_codes
        assert C.G5_LEAK_CLEAN in result.reason_codes
        assert C.ORACLE_PASS in result.reason_codes
        assert result.multi_file is True
        assert result.files_touched >= 2

        # Dual gold runs observed
        gold_runs = result.details.get("gold_runs") or []
        assert len(gold_runs) == 2
        for run in gold_runs:
            assert run.get("resolve") is True

        broken = result.details.get("broken") or {}
        assert broken.get("all_f2p_failed") is True

        null = result.details.get("null") or {}
        assert null.get("resolve") is False

        audit_path = tmp_path / "gate_audit.jsonl"
        append_gate_audit(audit_path, result, "fixture__tiny_offline__oracle")
        row = json.loads(audit_path.read_text(encoding="utf-8").strip())
        assert row["disposition"] == "accept"
        assert "G1_F2P_FAIL_OK" in row["reason_codes"]
        assert "G2_GOLD_DUAL_PASS" in row["reason_codes"]

        after_sdf = [n for n in docker.list_containers(all_containers=True) if n.startswith("sdf-")]
        new = set(after_sdf) - set(before_sdf) - set(leftover)
        assert not new, f"leftover sdf containers after oracle: {new}"

        after_ol = _snapshot_off_limits()
        for name in before_ol:
            assert name in after_ol
    finally:
        remove_leftover_sdf_containers(docker)


def test_null_patch_does_not_resolve_alone(docker_ready: DockerCLI) -> None:
    """Explicit G3 check using runner without full gate stack."""
    docker = docker_ready
    _, workspace, f2p, p2p = _load_gold_and_workspace()
    runner = OracleDockerRunner(
        docker=docker,
        base_image="python:3.12-slim",
        install_commands=["pip install -q pytest"],
        run_id="orit02",
        memory_mb=1024,
        cpus=1.0,
        command_timeout=120.0,
    )
    try:
        null = runner.run_with_patch(
            workspace=workspace,
            patch="",
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            phase="null",
        )
        assert null.resolve() is False
        assert null.all_f2p_passed() is False
    finally:
        runner.cleanup()
        remove_leftover_sdf_containers(docker)
