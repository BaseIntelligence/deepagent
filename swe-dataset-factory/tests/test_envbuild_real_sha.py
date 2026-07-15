"""Unit tests for DeepSWE real-SHA envbuild (VAL-ENVR-001..008).

Offline / mocked docker only — no daemon, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.test_envbuild import FakeDocker

from swe_factory.envbuild.agent_recipe import (
    ALLOW_INTERNET_FALSE,
    HOOKS_OFF_LINE,
    RUNTIME_OFFLINE_ENV,
    RUNTIME_OFFLINE_LABEL,
    default_agent_contract,
    render_agent_dockerfile,
    render_task_toml_env_snippet,
)
from swe_factory.envbuild.builder import (
    DISK_FAILED,
    SHA_FAILED,
    DockerCLI,
    EnvBuilder,
    EnvBuildError,
    dual_build,
    prune_owned_env_images,
    remove_leftover_sdf_containers,
)
from swe_factory.envbuild.fixture import (
    recipe_for_language,
    recipe_from_clone,
    recipe_from_go_fixture,
    recipe_from_green_fixture,
    recipe_from_ts_fixture,
)
from swe_factory.envbuild.hygiene import (
    CONCURRENCY_HINT,
    DEFAULT_MIN_FREE_DISK_BYTES,
    MAX_CONCURRENT_ENVBUILD_JOBS,
    HygieneError,
    assert_safe_container_name,
    check_disk_for_envbuild,
    is_off_limits_name,
    is_owned_container_name,
    is_owned_image_ref,
    prune_owned_images,
    remove_leftover_owned_containers,
    require_disk_for_envbuild,
)
from swe_factory.envbuild.models import EnvRecipe
from swe_factory.envbuild.sha import (
    BaseCommitError,
    is_full_sha,
    isolation_scan,
    looks_synthetic_sha,
    require_full_sha,
    scrub_git_history,
    write_base_commit_marker,
)

# ---------------------------------------------------------------------------
# VAL-ENVR-001: real base SHA pin / refuse fake
# ---------------------------------------------------------------------------


def test_require_full_sha_accepts_real() -> None:
    sha = "a1b2c3d4e5f6789012345678abcdef0123456789"
    assert is_full_sha(sha)
    assert require_full_sha(sha) == sha.lower()
    assert not looks_synthetic_sha(sha)


def test_require_full_sha_rejects_short_and_synthetic() -> None:
    with pytest.raises(BaseCommitError):
        require_full_sha("abc123")
    with pytest.raises(BaseCommitError):
        require_full_sha("a100000000000000000000000000000000000001")
    with pytest.raises(BaseCommitError):
        require_full_sha("0000000000000000000000000000000000000001")
    with pytest.raises(BaseCommitError):
        require_full_sha("local")
    assert looks_synthetic_sha("b100000000000000000000000000000000000001")


def test_builder_requires_real_sha_on_clone_recipe() -> None:
    docker = FakeDocker()
    recipe = recipe_from_clone(
        repo_id="owner/repo",
        base_commit="a100000000000000000000000000000000000001",
        require_real_sha=True,
    )
    result = EnvBuilder(docker=docker, enforce_disk_gate=False).build(recipe)
    assert result.success is False
    assert result.failure_kind in {SHA_FAILED, "checkout_failed", "sha_failed"}
    assert "synthetic" in result.reason or "40-char" in result.reason or "SHA" in result.reason


def test_builder_green_fixture_still_pins_base_commit() -> None:
    docker = FakeDocker(digest_sequence=["sha256:pinok"])
    recipe = recipe_from_green_fixture()
    result = EnvBuilder(docker=docker, enforce_disk_gate=False).build(recipe)
    assert result.success
    assert result.env_image is not None
    assert result.env_image.base_commit == recipe.base_commit
    # Marker path / head recorded in logs or image
    assert result.env_image.resolved_head or result.logs.get("checkout_head") is not None


# ---------------------------------------------------------------------------
# VAL-ENVR-002: runtime offline allow_internet=false
# ---------------------------------------------------------------------------


def test_agent_dockerfile_documents_allow_internet_false() -> None:
    df = render_agent_dockerfile(
        base_commit="a1b2c3d4e5f6789012345678abcdef0123456789",
        language="python",
    )
    assert ALLOW_INTERNET_FALSE in df
    assert RUNTIME_OFFLINE_LABEL in df
    assert RUNTIME_OFFLINE_ENV in df
    assert "HARBOR_ALLOW_INTERNET=false" in df
    # Build-time install only comment
    assert "Build-time dependency install" in df or "pip install" in df
    # No runtime network-install hooks instructed
    assert "allow_internet=true" not in df


def test_task_toml_snippet_allow_internet_false() -> None:
    assert render_task_toml_env_snippet() == "allow_internet = false"
    assert "true" in render_task_toml_env_snippet(allow_internet=True)


def test_env_image_records_offline_runtime_contract() -> None:
    docker = FakeDocker(digest_sequence=["sha256:off"])
    recipe = recipe_from_green_fixture()
    assert recipe.allow_internet is False
    result = EnvBuilder(docker=docker, enforce_disk_gate=False).build(recipe)
    assert result.success and result.env_image is not None
    assert result.env_image.allow_internet is False
    assert result.logs.get("runtime_contract") == ALLOW_INTERNET_FALSE
    assert ALLOW_INTERNET_FALSE in (
        result.env_image.dockerfile_excerpt or result.logs.get("dockerfile_excerpt", "")
    )
    assert result.env_image.provenance.get("runtime_network_policy") == ALLOW_INTERNET_FALSE


def test_agent_contract_default() -> None:
    c = default_agent_contract()
    assert c.allow_internet is False
    assert c.hooks_path == "/dev/null"
    assert c.as_dict()["runtime_network_policy"] == ALLOW_INTERNET_FALSE


# ---------------------------------------------------------------------------
# VAL-ENVR-003: dual-build both base-line green
# ---------------------------------------------------------------------------


def test_dual_build_both_baseline_green() -> None:
    docker = FakeDocker(digest_sequence=["sha256:d1", "sha256:d2", "sha256:d1", "sha256:d2"])
    recipe = recipe_from_green_fixture()
    first, second, ok = dual_build(
        recipe,
        docker=docker,
        builder_factory=lambda **kw: EnvBuilder(enforce_disk_gate=False, **kw),
    )
    assert ok is True
    assert first.success and second.success
    assert first.env_image is not None and second.env_image is not None
    assert first.env_image.baseline_green is True
    assert second.env_image.baseline_green is True
    assert first.env_image.image_tag
    assert second.env_image.image_tag


# ---------------------------------------------------------------------------
# VAL-ENVR-004: porcelain + hooks off
# ---------------------------------------------------------------------------


def test_dockerfile_disables_hooks() -> None:
    df = render_agent_dockerfile(base_commit="a" * 40, language="python")
    assert HOOKS_OFF_LINE in df or "core.hooksPath /dev/null" in df
    assert "git status --porcelain" in df


def test_build_sets_hooks_path_and_porcelain_fields() -> None:
    docker = FakeDocker(digest_sequence=["sha256:hooks"])
    # Capture prep/scrub scripts
    recipe = recipe_from_green_fixture()
    result = EnvBuilder(docker=docker, enforce_disk_gate=False).build(recipe)
    assert result.success and result.env_image is not None
    assert result.env_image.hooks_path == "/dev/null"
    # Scrub script executed
    scrubbed = any("hooksPath" in s or "reflog" in s for s in docker.exec_scripts)
    assert scrubbed or result.env_image.history_scrubbed


# ---------------------------------------------------------------------------
# VAL-ENVR-005: no future-history / solution leakage
# ---------------------------------------------------------------------------


def test_dockerfile_scrubs_history_and_forbids_solution() -> None:
    df = render_agent_dockerfile(
        base_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        language="python",
        repo_url="https://github.com/example/repo.git",
        copy_context=False,
    )
    assert "git remote remove origin" in df or "reflog expire" in df
    assert "solution" in df.lower()
    assert "gc --prune" in df


def test_isolation_scan_detects_solution_tree(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x=1\n", encoding="utf-8")
    clean = isolation_scan(tmp_path)
    assert clean["clean"] is True
    sol = tmp_path / "solution"
    sol.mkdir()
    (sol / "solve.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    dirty = isolation_scan(tmp_path)
    assert dirty["clean"] is False
    assert any("solution" in h for h in dirty["hits"])  # type: ignore[operator]


def test_write_marker_and_scrub(tmp_path: Path) -> None:
    # minimal git repo
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "f.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    sha = "abcdef0123456789abcdef0123456789abcdef01"
    marker = write_base_commit_marker(tmp_path, sha)
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip() == sha
    actions = scrub_git_history(tmp_path)
    assert actions
    assert any("hooksPath" in a or "hooks" in a for a in actions)


# ---------------------------------------------------------------------------
# VAL-ENVR-006: owned prefixes + off-limits untouched
# ---------------------------------------------------------------------------


def test_owned_and_off_limits_name_rules() -> None:
    assert is_owned_container_name("sdf-build-abc")
    assert is_owned_container_name("deepswe-build-1")
    assert is_owned_container_name("harbor-sdf-agent-x")
    assert not is_owned_container_name("mission-test-pg")
    assert not is_owned_container_name("challenge-prism.1")
    assert not is_owned_container_name("acproxy")
    assert not is_owned_container_name("random")
    assert is_off_limits_name("mission-test-pg")
    assert is_off_limits_name("challenge-prism-worker.1")
    assert is_off_limits_name("acproxy")
    assert is_owned_image_ref("sdf-env-repo:abc")
    assert is_owned_image_ref("deepswe-env-foo:deadbeef")
    assert is_owned_image_ref("harbor-sdf-agent:local")
    assert not is_owned_image_ref("python:3.12-slim")
    assert not is_owned_image_ref("mission-test-pg:latest")


def test_assert_safe_container_name_refuses_off_limits() -> None:
    with pytest.raises(HygieneError):
        assert_safe_container_name("mission-test-pg")
    with pytest.raises(HygieneError):
        assert_safe_container_name("other")
    assert_safe_container_name("sdf-ok-1")
    assert_safe_container_name("deepswe-ok-1")


def test_leftover_sweep_only_owned() -> None:
    class ListingFake(FakeDocker):
        def list_containers(self, *, all_containers: bool = True) -> list[str]:
            return [
                "sdf-orphan",
                "deepswe-orphan",
                "harbor-sdf-old",
                "mission-test-pg",
                "challenge-prism.1.x",
                "acproxy",
                "other",
            ]

        def remove_container(self, ref: str) -> None:
            super().remove_container(ref)

    fake = ListingFake()
    for n in ("sdf-orphan", "deepswe-orphan", "harbor-sdf-old"):
        fake.containers[n] = {}
    removed = remove_leftover_owned_containers(fake)
    assert set(removed) == {"sdf-orphan", "deepswe-orphan", "harbor-sdf-old"}
    assert "mission-test-pg" not in removed

    # Compat export — use a listing that reports sdf-x
    class ListingFake2(FakeDocker):
        def list_containers(self, *, all_containers: bool = True) -> list[str]:
            return ["sdf-x", "mission-test-pg"]

        def remove_container(self, ref: str) -> None:
            super().remove_container(ref)

    fake2 = ListingFake2()
    fake2.containers["sdf-x"] = {}
    assert remove_leftover_sdf_containers(fake2) == ["sdf-x"]


def test_docker_cli_refuses_deepswe_ok_and_offlimits() -> None:
    cli = DockerCLI(binary="docker")
    with pytest.raises(EnvBuildError, match="off-limits|must start"):
        cli.remove_container("mission-test-pg")
    with pytest.raises(EnvBuildError, match="must start"):
        cli.remove_container("unrelated")


# ---------------------------------------------------------------------------
# VAL-ENVR-007: multi-lang recipes (python + go + typescript)
# ---------------------------------------------------------------------------


def test_multi_lang_fixture_recipes() -> None:
    py = recipe_for_language("python")
    go = recipe_from_go_fixture()
    ts = recipe_from_ts_fixture()
    assert py.language == "python"
    assert go.language == "go"
    assert ts.language == "typescript"
    assert "python" in go.base_image or "golang" in go.base_image
    assert "node" in ts.base_image
    assert go.local_path and Path(go.local_path).is_dir()
    assert ts.local_path and Path(ts.local_path).is_dir()
    assert go.allow_internet is False
    assert ts.allow_internet is False
    assert "go test" in go.baseline_test_command
    assert "npm" in ts.baseline_test_command or "test" in ts.baseline_test_command


def test_multi_lang_build_mocked_go_and_ts() -> None:
    # FakeDocker treats install+baseline as green regardless of language cmd.
    for recipe in (recipe_from_go_fixture(), recipe_from_ts_fixture()):
        docker = FakeDocker(digest_sequence=[f"sha256:{recipe.language}01"])
        # register language base images so ensure_image is happy without pull fail
        docker.images[recipe.base_image] = f"sha256:base-{recipe.language}"
        result = EnvBuilder(docker=docker, enforce_disk_gate=False).build(recipe)
        assert result.success, (recipe.language, result.failure_kind, result.reason)
        assert result.env_image is not None
        assert result.env_image.baseline_green is True
        assert result.env_image.language == recipe.language
        assert result.env_image.image_tag.startswith("sdf-env-")


def test_dockerfile_language_bases() -> None:
    go_df = render_agent_dockerfile(base_commit="a" * 40, language="go")
    ts_df = render_agent_dockerfile(base_commit="a" * 40, language="typescript")
    assert "golang" in go_df
    assert "node:" in ts_df
    assert ALLOW_INTERNET_FALSE in go_df
    assert ALLOW_INTERNET_FALSE in ts_df


# ---------------------------------------------------------------------------
# VAL-ENVR-008: prune + disk fail-closed + concurrency ceiling
# ---------------------------------------------------------------------------


def test_concurrency_ceiling_documented() -> None:
    assert 16 <= MAX_CONCURRENT_ENVBUILD_JOBS <= 24
    assert "16" in CONCURRENCY_HINT
    c = default_agent_contract()
    assert 16 <= c.concurrency_ceiling <= 24
    df = render_agent_dockerfile(base_commit="a" * 40)
    assert "16" in df or "24" in df


def test_disk_gate_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "swe_factory.envbuild.hygiene.free_disk_bytes",
        lambda path="/": 100,
    )
    result = check_disk_for_envbuild(min_free_bytes=DEFAULT_MIN_FREE_DISK_BYTES)
    assert result.ok is False
    with pytest.raises(HygieneError, match="insufficient free disk"):
        require_disk_for_envbuild(min_free_bytes=DEFAULT_MIN_FREE_DISK_BYTES)

    docker = FakeDocker()
    recipe = recipe_from_green_fixture()
    builder = EnvBuilder(
        docker=docker,
        enforce_disk_gate=True,
        min_free_disk_bytes=10**15,  # absurd threshold → fail
    )
    out = builder.build(recipe)
    assert out.success is False
    assert out.failure_kind == DISK_FAILED


def test_prune_owned_images_only() -> None:
    class ImgFake(FakeDocker):
        def __init__(self) -> None:
            super().__init__()
            self.images = {
                "sdf-env-foo:abc": "sha256:1",
                "deepswe-env-bar:def": "sha256:2",
                "harbor-sdf-agent:local": "sha256:3",
                "python:3.12-slim": "sha256:base",
                "mission-test-pg:latest": "sha256:nope",
            }

        def list_images(self) -> list[str]:
            return list(self.images.keys())

    fake = ImgFake()
    pruned = prune_owned_images(fake)
    assert "sdf-env-foo:abc" in pruned
    assert "deepswe-env-bar:def" in pruned
    assert "harbor-sdf-agent:local" in pruned
    assert "python:3.12-slim" not in pruned
    assert "mission-test-pg:latest" not in pruned
    # prune_owned_env_images export
    fake2 = ImgFake()
    p2 = prune_owned_env_images(fake2, image_refs=["sdf-env-foo:abc", "python:3.12-slim"])
    assert p2 == ["sdf-env-foo:abc"]


def test_dual_build_disk_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "swe_factory.envbuild.hygiene.free_disk_bytes",
        lambda path="/": 1,
    )
    recipe = recipe_from_green_fixture()
    first, second, ok = dual_build(recipe, docker=FakeDocker())
    assert ok is False
    assert first.failure_kind == DISK_FAILED
    assert second.failure_kind == DISK_FAILED


def test_recipe_from_clone_defaults_real_sha() -> None:
    real = "abcdef0123456789abcdef0123456789abcdef01"
    r = recipe_from_clone(repo_id="org/lib", base_commit=real, language="python")
    assert r.require_real_sha is True
    assert r.allow_internet is False
    assert r.image_namespace.startswith("deepswe")
    assert r.clone_url and "github.com/org/lib" in r.clone_url


def test_env_recipe_fields_to_dict() -> None:
    recipe = EnvRecipe(
        repo_id="x",
        base_commit="a" * 40,
        require_real_sha=True,
        allow_internet=False,
        history_scrub=True,
    )
    d = recipe.to_dict()
    assert d["require_real_sha"] is True
    assert d["allow_internet"] is False
