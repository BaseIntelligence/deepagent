"""Unit tests for envbuild (offline / mocked docker seam).

Covers VAL-ENV-001 (digest metadata + dual_build), VAL-ENV-002 (base green),
VAL-ENV-003 (sdf-* teardown, off-limits untouched) without requiring a daemon.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from swe_factory.envbuild.builder import (
    BASELINE_FAILED,
    CHECKOUT_FAILED,
    INSTALL_FAILED,
    REPRODUCE_FAILED,
    DockerCLI,
    EnvBuilder,
    EnvBuildError,
    ExecOutcome,
    dual_build,
    image_tag_for,
    remove_leftover_sdf_containers,
    scoped_container_name,
)
from swe_factory.envbuild.fixture import (
    default_green_fixture_root,
    recipe_from_green_fixture,
    recipe_from_offline_broken_fixture,
)
from swe_factory.envbuild.models import EnvImage, EnvRecipe
from swe_factory.schema import EnvironmentMeta, SourceTrack, TaskRecord


class FakeDocker:
    """In-memory docker seam for deterministic unit tests."""

    def __init__(
        self,
        *,
        install_exit: int = 0,
        baseline_exit: int = 0,
        reproduce_exit: int = 0,
        fail_pull: bool = False,
        digest_sequence: list[str] | None = None,
    ) -> None:
        self.install_exit = install_exit
        self.baseline_exit = baseline_exit
        self.reproduce_exit = reproduce_exit
        self.fail_pull = fail_pull
        self.containers: dict[str, dict[str, Any]] = {}
        self.images: dict[str, str] = {"python:3.12-slim": "sha256:baseimg"}
        self.removed_containers: list[str] = []
        self.removed_images: list[str] = []
        self.run_detached_names: list[str] = []
        self.ephemeral_names: list[str] = []
        self.copy_calls: list[tuple[str, str]] = []
        self.exec_scripts: list[str] = []
        self._digest_seq = list(digest_sequence or [])
        self._digest_i = 0
        self._commit_n = 0

    def version(self) -> str:
        return "fake-29"

    def image_id(self, ref: str) -> str | None:
        return self.images.get(ref)

    def image_digest(self, ref: str) -> str | None:
        if self._digest_seq:
            if self._digest_i < len(self._digest_seq):
                d = self._digest_seq[self._digest_i]
                self._digest_i += 1
                return d
            return self._digest_seq[-1]
        return self.images.get(ref)

    def ensure_image(self, ref: str, *, timeout: float) -> None:
        if self.fail_pull and ref not in self.images:
            raise EnvBuildError(f"failed to pull base image {ref!r}")
        self.images.setdefault(ref, f"sha256:pulled-{ref}")

    def run_detached(
        self,
        *,
        name: str,
        image: str,
        workdir: str,
        memory_mb: int,
        cpus: float,
        pids_limit: int,
    ) -> str:
        if not name.startswith("sdf-"):
            raise EnvBuildError(f"bad name {name}")
        self.run_detached_names.append(name)
        self.containers[name] = {"image": image, "workdir": workdir}
        return f"id-{name}"

    def copy_into(self, container: str, src_dir: Path | str, dest_dir: str) -> None:
        self.copy_calls.append((container, str(src_dir)))
        if container not in self.containers:
            raise EnvBuildError(f"missing container {container}")

    def exec(self, container: str, script: str, *, workdir: str, timeout: float) -> ExecOutcome:
        self.exec_scripts.append(script)
        # Prep is best-effort and never gates success.
        if script.strip().startswith("set +e") or "apt-get" in script:
            return ExecOutcome(0, stdout="prep ok\n")
        # Install uses an explicit set -e multi-command script.
        if "set -e" in script or "pip install" in script or "npm install" in script:
            return ExecOutcome(
                self.install_exit,
                stdout="install ok\n" if self.install_exit == 0 else "install fail\n",
            )
        # Baseline is the remaining suite command.
        if self.baseline_exit == 0:
            return ExecOutcome(0, stdout="=== 3 passed in 0.01s ===\n")
        return ExecOutcome(self.baseline_exit, stdout="=== 1 failed ===\n")

    def commit(self, container: str, tag: str, *, workdir: str) -> str:
        self._commit_n += 1
        cid = f"sha256:commit{self._commit_n:04d}"
        self.images[tag] = cid
        return cid

    def run_ephemeral(
        self,
        *,
        name: str,
        image: str,
        script: str,
        workdir: str,
        timeout: float,
        memory_mb: int,
        cpus: float,
        pids_limit: int,
    ) -> ExecOutcome:
        if not name.startswith("sdf-"):
            raise EnvBuildError(f"bad name {name}")
        self.ephemeral_names.append(name)
        # ephemerals auto-remove; do not leave in containers
        if self.reproduce_exit == 0:
            return ExecOutcome(0, stdout="=== 3 passed in 0.02s ===\n")
        return ExecOutcome(self.reproduce_exit, stdout="reproduce fail\n")

    def remove_container(self, ref: str) -> None:
        if not (
            ref.startswith("sdf-") or ref.startswith("deepswe-") or ref.startswith("harbor-sdf-")
        ):
            raise EnvBuildError(f"refuse non-owned {ref}")
        self.removed_containers.append(ref)
        self.containers.pop(ref, None)

    def remove_image(self, ref: str) -> None:
        self.removed_images.append(ref)
        self.images.pop(ref, None)

    def list_containers(self, *, all_containers: bool = True) -> list[str]:
        return list(self.containers.keys())

    def list_images(self) -> list[str]:
        return list(self.images.keys())


def test_green_fixture_root_exists() -> None:
    root = default_green_fixture_root()
    assert root.is_dir()
    assert (root / "repo" / "demo_pkg" / "math_ops.py").is_file()
    assert (root / "task_meta.json").is_file()


def test_recipe_from_green_fixture() -> None:
    recipe = recipe_from_green_fixture()
    assert recipe.repo_id
    assert recipe.base_commit
    assert recipe.local_path is not None
    assert Path(recipe.local_path).is_dir()
    assert "pytest" in " ".join(recipe.install_commands)


def test_scoped_container_names_are_sdf_prefixed() -> None:
    name = scoped_container_name("build", "abc12345")
    assert name.startswith("sdf-")
    assert "build" in name


def test_image_tag_for_stable() -> None:
    t1 = image_tag_for("fixtures/tiny_green", "green00000000000000000000000000000001")
    t2 = image_tag_for("fixtures/tiny_green", "green00000000000000000000000000000001")
    assert t1 == t2
    assert t1.startswith("sdf-env-")


def test_happy_path_stores_digest_and_green_baseline() -> None:
    docker = FakeDocker(digest_sequence=["sha256:aaa111"])
    recipe = recipe_from_green_fixture()
    builder = EnvBuilder(docker=docker, run_id="unit01")
    result = builder.build(recipe)

    assert result.success is True
    assert result.stage == "complete"
    assert result.env_image is not None
    assert result.env_image.baseline_green is True
    assert result.env_image.baseline_exit_code == 0
    assert result.env_image.image_digest == "sha256:aaa111"
    assert result.env_image.image_tag.startswith("sdf-env-")
    assert result.env_image.base_commit == recipe.base_commit
    # digest metadata is TaskRecord-compatible
    meta = EnvironmentMeta(image_digest=result.env_image.image_digest)
    assert meta.image_digest.startswith("sha256:")


def test_containers_named_sdf_and_removed() -> None:
    docker = FakeDocker()
    recipe = recipe_from_green_fixture()
    builder = EnvBuilder(docker=docker, run_id="unit02")
    result = builder.build(recipe)
    assert result.success
    assert docker.run_detached_names
    assert all(n.startswith("sdf-") for n in docker.run_detached_names)
    assert all(n.startswith("sdf-") for n in docker.ephemeral_names)
    # Build container must be removed; no leftovers
    assert docker.containers == {}
    assert any(n.startswith("sdf-build") for n in docker.removed_containers)


def test_install_failure_distinct_from_baseline() -> None:
    docker = FakeDocker(install_exit=1)
    result = EnvBuilder(docker=docker).build(recipe_from_green_fixture())
    assert result.success is False
    assert result.failure_kind == INSTALL_FAILED
    assert result.stage == "install"
    assert result.env_image is None
    assert docker.containers == {}


def test_baseline_failure_rejects_image() -> None:
    docker = FakeDocker(baseline_exit=1)
    result = EnvBuilder(docker=docker).build(recipe_from_green_fixture())
    assert result.success is False
    assert result.failure_kind == BASELINE_FAILED
    assert result.stage == "baseline"
    assert result.env_image is None
    assert docker.containers == {}


def test_broken_offline_fixture_fails_baseline() -> None:
    """Structural check: offline fixture recipe points at broken tree + baseline cmd."""
    recipe = recipe_from_offline_broken_fixture()
    assert recipe.local_path is not None
    math_ops = Path(recipe.local_path) / "demo_pkg" / "math_ops.py"
    assert "return a - b" in math_ops.read_text(encoding="utf-8")
    docker = FakeDocker(baseline_exit=1)
    result = EnvBuilder(docker=docker).build(recipe)
    assert result.success is False
    assert result.failure_kind == BASELINE_FAILED


def test_reproduce_failure_discards_image() -> None:
    docker = FakeDocker(reproduce_exit=1)
    result = EnvBuilder(docker=docker).build(recipe_from_green_fixture())
    assert result.success is False
    assert result.failure_kind == REPRODUCE_FAILED
    assert result.env_image is None
    # committed tag should be discarded
    assert docker.removed_images


def test_missing_local_path_checkout_failed() -> None:
    recipe = EnvRecipe(
        repo_id="missing",
        base_commit="abc",
        local_path="/tmp/does-not-exist-envbuild-xyz",
    )
    result = EnvBuilder(docker=FakeDocker()).build(recipe)
    assert result.success is False
    assert result.failure_kind == CHECKOUT_FAILED


def test_dual_build_marks_verified_and_stores_digests() -> None:
    docker = FakeDocker(digest_sequence=["sha256:same", "sha256:same", "sha256:same"])
    recipe = recipe_from_green_fixture()
    first, second, ok = dual_build(recipe, docker=docker)
    assert ok is True
    assert first.success and second.success
    assert first.env_image is not None and second.env_image is not None
    assert first.env_image.dual_build_verified is True
    assert second.env_image.dual_build_verified is True
    assert len(first.env_image.dual_build_digests) == 2
    # usable image references recorded
    assert first.env_image.image_digest
    assert second.env_image.image_digest


def test_dual_build_usable_when_commit_ids_diverge() -> None:
    """Docker commit often yields distinct image IDs; recipe still dual-build usable."""
    docker = FakeDocker(digest_sequence=["sha256:one", "sha256:two", "sha256:one", "sha256:two"])
    recipe = recipe_from_green_fixture()
    first, second, ok = dual_build(recipe, docker=docker)
    assert ok is True  # recipe-usable path
    assert first.env_image is not None
    assert first.env_image.dual_build_verified is True


def test_docker_cli_refuses_off_limits_names() -> None:
    cli = DockerCLI(binary="docker")
    with pytest.raises(EnvBuildError, match="off-limits|must start"):
        cli.remove_container("mission-test-pg")
    with pytest.raises(EnvBuildError, match="off-limits|must start"):
        cli.remove_container("challenge-prism-worker.1.abc")
    with pytest.raises(EnvBuildError, match="off-limits|must start"):
        cli.remove_container("acproxy")
    with pytest.raises(EnvBuildError, match="must start"):
        cli.remove_container("random-other")


def test_remove_leftover_only_sdf() -> None:
    class ListingFake(FakeDocker):
        def list_containers(self, *, all_containers: bool = True) -> list[str]:
            return [
                "sdf-orphan-leftover",
                "mission-test-pg",
                "challenge-prism.1.x",
                "acproxy",
                "other",
            ]

        def remove_container(self, ref: str) -> None:
            super().remove_container(ref)

    fake = ListingFake()
    fake.containers["sdf-orphan-leftover"] = {}
    removed = remove_leftover_sdf_containers(fake)
    assert removed == ["sdf-orphan-leftover"]
    assert "mission-test-pg" not in removed


def test_env_image_round_trip_and_task_environment() -> None:
    img = EnvImage(
        repo_id="fixtures/tiny_green",
        base_commit="green01",
        language="python",
        image_tag="sdf-env-fixtures_tiny_green:green01",
        image_digest="sha256:deadbeef",
        base_image="python:3.12-slim",
        workspace_dir="/workspace/repo",
        install_commands=["pip install -q pytest"],
        baseline_test_command="python -m pytest -q",
        baseline_green=True,
        baseline_exit_code=0,
    )
    restored = EnvImage.from_dict(img.to_dict())
    assert restored.image_digest == "sha256:deadbeef"
    # Can embed into TaskRecord environment
    record = TaskRecord(
        instance_id="envbuild_meta_check",
        source_track=SourceTrack.SYNTHETIC_GROUNDED,
        repo=img.repo_id,
        base_commit=img.base_commit,
        language="python",
        problem_statement="check",
        fail_to_pass=["python -m pytest -q"],
        pass_to_pass=[],
        gold_patch="diff --git a/x b/x\n",
        environment=EnvironmentMeta(image_digest=img.image_digest),
        license="MIT",
    )
    assert record.environment.image_digest == "sha256:deadbeef"
    payload = json.loads(record.model_dump_json())
    assert payload["environment"]["image_digest"] == "sha256:deadbeef"
