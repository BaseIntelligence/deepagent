"""Docker integration for harness scoring on tiny_offline (VAL-HARNESS-001)."""

from __future__ import annotations

import json
import subprocess

import pytest

from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers
from swe_factory.fixture.offline import build_fixture_task, default_fixture_root
from swe_factory.harness.score import score_gold_and_null
from swe_factory.oracle.docker_run import OracleDockerRunner

pytestmark = pytest.mark.integration


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


def test_harness_gold_true_null_false_docker(docker_ready: DockerCLI) -> None:
    del docker_ready
    task = build_fixture_task()
    workspace = default_fixture_root() / "repo"
    runner = OracleDockerRunner(
        docker=DockerCLI(),
        base_image="python:3.12-slim",
        install_commands=["pip install -q pytest"],
        command_timeout=180.0,
    )
    try:
        pair = score_gold_and_null(task=task, workspace=workspace, runner=runner)
        assert pair.gold.resolve is True, pair.gold.to_dict()
        assert pair.null.resolve is False, pair.null.to_dict()
        assert pair.passed is True
        assert pair.gold.score == 1.0
        assert pair.null.score == 0.0
    finally:
        remove_leftover_sdf_containers()


def test_harness_cli_docker_fixture() -> None:
    if not _docker_ok():
        pytest.skip("Docker daemon not available")
    from typer.testing import CliRunner

    from swe_factory.cli import app

    cli = CliRunner()
    result = cli.invoke(
        app,
        ["score", "--from-fixture", "--backend", "docker", "--json"],
    )
    assert result.exit_code == 0, result.output
    # Find JSON object in output
    text = result.output.strip()
    # May print multi-line JSON
    start = text.find("{")
    assert start >= 0, result.output
    payload = json.loads(text[start:])
    assert payload["gold"]["resolve"] is True
    assert payload["null"]["resolve"] is False
    assert payload["passed"] is True

    # No leftover sdf containers
    leftover = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=sdf-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    names = [n for n in leftover.stdout.splitlines() if n.strip()]
    assert not names, f"leftover sdf containers: {names}"
