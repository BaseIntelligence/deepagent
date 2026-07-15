"""Docker integration: Harbor separate-verifier oracle on offline fixture.

Marked ``integration``. Validates:
- VAL-HARBOR-003 solution reward=1
- VAL-HARBOR-004 null reward=0
- VAL-HARBOR-005 agent context isolation
- VAL-HARBOR-006 config f2p + non-empty test.patch
- sdf-/harbor-sdf- container hygiene; off-limits untouched
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers
from swe_factory.harbor.harbor_docker import (
    scan_agent_context_forbidden,
    stage_agent_context,
)
from swe_factory.harbor.harbor_oracle import (
    HarborDockerVerifier,
    run_harbor_oracle,
)
from swe_factory.harbor.offline_fixture import run_offline_harbor_fixture

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


def test_harbor_oracle_solution_pass_null_fail_docker(
    docker_ready: DockerCLI, tmp_path: Path
) -> None:
    del docker_ready  # used for availability only; runner owns builds
    before_ol = _snapshot_off_limits()

    pack_result = run_offline_harbor_fixture(out_dir=tmp_path / "harbor_pack")
    pack = pack_result.pack_dir
    # Structural isolation before docker
    staged = stage_agent_context(pack, tmp_path / "agent_only")
    assert scan_agent_context_forbidden(staged.context_dir) == []
    assert not (staged.context_dir / "solution").exists()
    assert not any(p.name == "test.patch" for p in staged.context_dir.rglob("*") if p.is_file())

    verifier = HarborDockerVerifier(
        run_id="horit01",
        memory_mb=1024,
        cpus=1.0,
        build_timeout=600.0,
        run_timeout=240.0,
        remove_images_on_cleanup=True,
    )
    try:
        oracle = run_harbor_oracle(
            pack,
            backend=verifier,
            task_id=pack_result.task_id,
            mode="docker",
            cleanup=True,
        )
        leftover = remove_leftover_sdf_containers(DockerCLI())

        assert oracle.passed is True, (
            f"harbor oracle failed: {oracle.reasons}\n"
            f"solution={oracle.solution.reward} null={oracle.null.reward}\n"
            f"sol_logs={oracle.solution.logs[-1500:]}\n"
            f"null_logs={oracle.null.logs[-1500:]}"
        )
        assert oracle.solution.reward == 1
        assert oracle.null.reward == 0
        assert oracle.agent_isolated is True
        assert oracle.config_ok is True
        assert oracle.test_patch_ok is True

        # leftover sdf- from this run should be cleaned
        after = DockerCLI().list_containers(all_containers=True)
        leftover_run = [n for n in after if n.startswith("sdf-") and "horit01" in n]
        assert (
            leftover_run == []
        ), f"leftover mission containers: {leftover_run} (+sweep {leftover})"

        after_ol = _snapshot_off_limits()
        assert after_ol == before_ol, "off-limits containers changed"
    finally:
        with pytest.MonkeyPatch.context() as _:
            pass
        # ensure cleanup even if asserts fail
        verifier.cleanup()


def test_harbor_agent_dockerfile_build_isolated(docker_ready: DockerCLI, tmp_path: Path) -> None:
    """Agent image builds only from environment/ context."""
    from swe_factory.harbor.harbor_docker import build_agent_and_tests_images, remove_images

    del docker_ready
    pack = run_offline_harbor_fixture(out_dir=tmp_path / "pack").pack_dir
    agent_tag = "harbor-sdf-agent-unitit:iso"
    tests_tag = "harbor-sdf-tests-unitit:iso"
    try:
        pair = build_agent_and_tests_images(
            pack,
            work_dir=tmp_path / "contexts",
            agent_tag=agent_tag,
            tests_tag=tests_tag,
            stage_only=False,
        )
        hits = scan_agent_context_forbidden(pair.agent_context)
        assert hits == []
        # solution files must not be in agent context listing
        listing = "\n".join(p.name for p in pair.agent_context.rglob("*") if p.is_file())
        assert "solution.patch" not in listing
        assert "test.patch" not in listing
        # verifier context must include them
        assert (pair.tests_context / "test.patch").is_file()
        cfg = json.loads((pair.tests_context / "config.json").read_text(encoding="utf-8"))
        assert cfg["f2p_node_ids"]
    finally:
        remove_images([agent_tag, tests_tag])
