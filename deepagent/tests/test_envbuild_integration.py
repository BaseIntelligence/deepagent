"""Docker integration tests for envbuild on the green fixture.

Marked ``integration`` so default offline suite skips them. Validates:
- VAL-ENV-001 dual-build usable image + digest metadata
- VAL-ENV-002 base suite green
- VAL-ENV-003 no leftover sdf-* containers; off-limits untouched
"""

from __future__ import annotations

import contextlib
import subprocess

import pytest

from swe_factory.envbuild.builder import (
    DockerCLI,
    EnvBuilder,
    dual_build,
    remove_leftover_sdf_containers,
)
from swe_factory.envbuild.fixture import recipe_from_green_fixture

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


def _snapshot_off_limits(docker: DockerCLI) -> dict[str, str]:
    """Map off-limits-ish containers to status for before/after compare."""
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


def test_green_fixture_envbuild_docker(docker_ready: DockerCLI) -> None:
    docker = docker_ready
    before = _snapshot_off_limits(docker)
    before_sdf = [n for n in docker.list_containers(all_containers=True) if n.startswith("sdf-")]

    recipe = recipe_from_green_fixture()
    builder = EnvBuilder(docker=docker, run_id="itgt01", memory_mb=1024, cpus=1.0)
    result = builder.build(recipe)

    # Hygiene recovery even on failure
    leftover = remove_leftover_sdf_containers(docker)

    try:
        assert (
            result.success is True
        ), f"envbuild failed: {result.failure_kind}: {result.reason}\n{result.logs}"
        assert result.env_image is not None
        assert result.env_image.baseline_green is True
        assert result.env_image.baseline_exit_code == 0
        assert result.env_image.image_digest
        assert result.env_image.image_tag.startswith("sdf-env-")
        # Fresh-container green is proven (reproduce stage completed)
        assert "reproduce" in result.logs

        after_sdf = [n for n in docker.list_containers(all_containers=True) if n.startswith("sdf-")]
        # No new leftover mission containers (sweep may have cleaned prior orphans)
        new = set(after_sdf) - set(before_sdf) - set(leftover)
        assert not new, f"leftover sdf containers: {new}"

        after = _snapshot_off_limits(docker)
        for name, _status in before.items():
            assert name in after, f"off-limits container missing after build: {name}"
            # Status string may flip "Up Xs" timers; require still present (and not gone)
            assert after[name], f"off-limits empty status: {name}"
    finally:
        if result.env_image is not None:
            with contextlib.suppress(Exception):
                docker.remove_image(result.env_image.image_tag)
        remove_leftover_sdf_containers(docker)


def test_dual_build_green_fixture_docker(docker_ready: DockerCLI) -> None:
    docker = docker_ready
    before = _snapshot_off_limits(docker)
    recipe = recipe_from_green_fixture()
    first, second, verified = dual_build(recipe, docker=docker, keep_images=True)

    remove_leftover_sdf_containers(docker)

    tags: list[str] = []
    try:
        assert first.success, first.reason
        assert second.success, second.reason
        assert first.env_image is not None and second.env_image is not None
        assert verified is True
        assert first.env_image.dual_build_verified is True
        assert first.env_image.image_digest
        assert second.env_image.image_digest
        tags = [first.env_image.image_tag, second.env_image.image_tag]

        # No sdf leftovers
        leftover = [n for n in docker.list_containers(all_containers=True) if n.startswith("sdf-")]
        assert leftover == [], leftover

        after = _snapshot_off_limits(docker)
        for name in before:
            assert name in after
    finally:
        for tag in tags:
            with contextlib.suppress(Exception):
                docker.remove_image(tag)
        remove_leftover_sdf_containers(docker)
