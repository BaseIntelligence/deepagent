"""Unit tests for Stage 1 env-first build (offline; Docker mocked).

Covers the m2-envbuild contract (VAL-ENV-012..014, 018..022) at the logic level
with a fake docker CLI, plus the adapter baseline capability and the
green-baseline downstream precondition. The real-Docker end-to-end build for
Python/JS/Go is exercised by the ``integration``-marked test at the bottom
(deselected from the milestone gate) and by manual CLI verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_forge.forge.adapters import build_default_registry
from swe_forge.forge.cli import app as forge_app
from swe_forge.forge.envbuild import (
    EnvBuildError,
    EnvBuilder,
    EnvBuildResult,
    ExecOutcome,
)
from swe_forge.forge.envbuild.builder import (
    BASELINE_FAILED,
    DETECT_FAILED,
    IMAGE_PULL_FAILED,
    INSTALL_FAILED,
    REPRODUCE_FAILED,
)
from swe_forge.forge.models import (
    BaselineNotGreenError,
    EnvImage,
    RepoSpec,
    require_green_baseline,
)

runner = CliRunner()

_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write(root: Path, rel: str, content: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Adapter baseline capability (offline).
# --------------------------------------------------------------------------- #
class TestAdapterBaseline:
    def test_python_plain_manifest_ensures_pytest(self, tmp_path: Path) -> None:
        repo = tmp_path / "py"
        _write(repo, "pyproject.toml", "[project]\nname = 'demo'\n")
        adapter = build_default_registry().get("python")
        cmds = adapter.baseline_install_commands(repo)
        assert cmds == ["pip install -e .", "pip install pytest"]
        # The bare runtime install is distinct and does NOT add pytest.
        assert adapter.install_commands(repo) == ["pip install -e ."]
        # Baseline runner is the repo suite; standard runner is unchanged.
        assert adapter.baseline_test_command(repo) == "python -m pytest"
        assert adapter.test_command() == "python -m pytest"

    def test_python_optional_dependencies_extra(self, tmp_path: Path) -> None:
        repo = tmp_path / "py"
        _write(
            repo,
            "pyproject.toml",
            "[project]\nname = 'demo'\n"
            "[project.optional-dependencies]\n"
            'test = ["pytest", "coverage"]\n',
        )
        cmds = build_default_registry().get("python").baseline_install_commands(repo)
        assert "pip install -e '.[test]'" in cmds

    def test_python_pep735_dependency_group(self, tmp_path: Path) -> None:
        repo = tmp_path / "py"
        _write(
            repo,
            "pyproject.toml",
            "[project]\nname = 'demo'\n"
            "[dependency-groups]\n"
            'dev = ["pytest>=8", "pytest-xdist>=3"]\n',
        )
        cmds = build_default_registry().get("python").baseline_install_commands(repo)
        assert cmds[0] == "pip install -e ."
        assert any("pytest>=8" in c and "pytest-xdist>=3" in c for c in cmds)

    def test_python_requirements_dev_file(self, tmp_path: Path) -> None:
        repo = tmp_path / "py"
        _write(repo, "requirements-dev.txt", "pytest\n")
        _write(repo, "app.py", "x = 1\n")
        cmds = build_default_registry().get("python").baseline_install_commands(repo)
        assert "pip install -r requirements-dev.txt" in cmds

    def test_javascript_baseline_runs_npm_test(self, tmp_path: Path) -> None:
        repo = tmp_path / "js"
        _write(repo, "package.json", '{"name":"demo","scripts":{"test":"ava"}}\n')
        adapter = build_default_registry().get("javascript")
        assert adapter.baseline_test_command(repo) == "npm test"
        # Standard runner (for synthesized F2P tests) stays node --test.
        assert adapter.test_command() == "node --test"
        assert adapter.baseline_install_commands(repo) == ["npm install"]

    def test_go_baseline_uses_defaults(self, tmp_path: Path) -> None:
        repo = tmp_path / "go"
        _write(repo, "go.mod", "module demo\n\ngo 1.21\n")
        adapter = build_default_registry().get("go")
        assert adapter.baseline_test_command(repo) == "go test ./..."
        assert adapter.baseline_install_commands(repo) == ["go mod download"]


# --------------------------------------------------------------------------- #
# Adapter P2P-exclusion application (offline).
# --------------------------------------------------------------------------- #
class TestAdapterP2PExclusions:
    def test_python_appends_k_not_filter(self) -> None:
        adapter = build_default_registry().get("python")
        cmd = adapter.apply_p2p_exclusions(
            "python -m pytest", ["test_flaky", "test_other"]
        )
        assert cmd == "python -m pytest -k 'not (test_flaky or test_other)'"

    def test_python_no_exclusions_is_noop(self) -> None:
        adapter = build_default_registry().get("python")
        assert (
            adapter.apply_p2p_exclusions("python -m pytest", []) == "python -m pytest"
        )

    def test_javascript_appends_mocha_invert(self) -> None:
        adapter = build_default_registry().get("javascript")
        cmd = adapter.apply_p2p_exclusions(
            "npm test", ["should export the version number"]
        )
        assert cmd.startswith("npm test -- --grep ")
        assert "--invert" in cmd
        # The excluded title appears as a regex (whitespace escaped by re.escape).
        assert "version" in cmd and "export" in cmd

    def test_javascript_no_exclusions_is_noop(self) -> None:
        adapter = build_default_registry().get("javascript")
        assert adapter.apply_p2p_exclusions("npm test", []) == "npm test"

    def test_go_appends_skip_filter(self) -> None:
        adapter = build_default_registry().get("go")
        cmd = adapter.apply_p2p_exclusions(
            "go test ./...", ["TestMapClaims_ZeroIsExpired", "TestOther"]
        )
        # Go 1.20+ -skip de-selects the named tests, anchored so a prefix never
        # accidentally skips a sibling.
        assert cmd == (
            "go test ./... -skip '^(TestMapClaims_ZeroIsExpired|TestOther)$'"
        )

    def test_go_no_exclusions_is_noop(self) -> None:
        adapter = build_default_registry().get("go")
        assert adapter.apply_p2p_exclusions("go test ./...", []) == "go test ./..."


# --------------------------------------------------------------------------- #
# Adapter positive test selection (pr_mirror F2P via the repo's own runner).
# --------------------------------------------------------------------------- #
class TestAdapterSelectTests:
    def test_python_appends_k_select_filter(self) -> None:
        adapter = build_default_registry().get("python")
        cmd = adapter.select_tests("python -m pytest", ["test_between", "test_other"])
        assert cmd == "python -m pytest -k '(test_between or test_other)'"
        assert "not (" not in cmd

    def test_javascript_select_uses_mocha_grep_without_invert(self) -> None:
        adapter = build_default_registry().get("javascript")
        cmd = adapter.select_tests("npm test", ["should validate ISO 8601 dates"])
        assert cmd.startswith("npm test -- --grep ")
        assert "--invert" not in cmd
        assert "ISO" in cmd and "8601" in cmd

    def test_go_select_uses_run_anchored(self) -> None:
        adapter = build_default_registry().get("go")
        cmd = adapter.select_tests("go test ./...", ["TestVersion7Monotonicity"])
        assert cmd == "go test ./... -run '^(TestVersion7Monotonicity)$'"

    def test_select_no_names_is_noop(self) -> None:
        registry = build_default_registry()
        for language, base in (
            ("python", "python -m pytest"),
            ("javascript", "npm test"),
            ("go", "go test ./..."),
        ):
            adapter = registry.get(language)
            assert adapter.select_tests(base, []) == base


# --------------------------------------------------------------------------- #
# Per-repo override threading (offline; Docker mocked).
# --------------------------------------------------------------------------- #
class TestBuilderOverrides:
    def _spec(self, **overrides: object) -> RepoSpec:
        base: dict[str, object] = {
            "repo_id": "python-validators/validators#411",
            "url": "https://github.com/python-validators/validators.git",
            "commit": "25fcef05c293def421fd3f6714403a54fce993cf",
            "commit_date": "2025-03-28T20:52:14Z",
            "language": "python",
            "license": "MIT",
            "instance_cap": 2,
        }
        base.update(overrides)
        return RepoSpec(**base)  # type: ignore[arg-type]

    def _builder(self, fake: FakeDocker) -> EnvBuilder:
        return EnvBuilder(
            docker=fake, registry=build_default_registry(), run_id="testrun"
        )

    def test_install_and_baseline_overrides_threaded(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[_OK, _OK, ExecOutcome(0, "10 passed")])
        spec = self._spec(
            baseline_install=[
                "pip install -e '.[crypto-eth-addresses]'",
                "pip install pytest",
            ],
            baseline_test="python -m pytest -q",
        )
        result = self._builder(fake).build(spec, workdir=py_repo)

        assert result.success is True
        env = result.env_image
        assert env is not None
        assert env.install_commands == [
            "pip install -e '.[crypto-eth-addresses]'",
            "pip install pytest",
        ]
        assert env.baseline_test_command == "python -m pytest -q"
        # The override install commands were actually executed in the container,
        # and the baseline run used the override command (exec order: prep,
        # install, baseline).
        assert "crypto-eth-addresses" in fake.exec_scripts[1]
        assert fake.exec_scripts[2] == "python -m pytest -q"

    def test_p2p_exclusions_applied_to_recorded_baseline(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[_OK, _OK, ExecOutcome(0, "passed")])
        spec = self._spec(p2p_exclusions=["test_version_selftest"])
        result = self._builder(fake).build(spec, workdir=py_repo)

        assert result.success is True
        env = result.env_image
        assert env is not None
        # The excluded self-test is de-selected from the recorded P2P command,
        # and that exact command is what the container ran.
        assert "not (test_version_selftest)" in env.baseline_test_command
        assert fake.exec_scripts[2] == env.baseline_test_command

    def test_no_overrides_falls_back_to_adapter_defaults(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[_OK, _OK, ExecOutcome(0, "passed")])
        spec = self._spec()  # no per-repo overrides set
        result = self._builder(fake).build(spec, workdir=py_repo)

        env = result.env_image
        assert env is not None
        assert env.install_commands == ["pip install -e .", "pip install pytest"]
        assert env.baseline_test_command == "python -m pytest"


# --------------------------------------------------------------------------- #
# Fake docker CLI for offline builder logic tests.
# --------------------------------------------------------------------------- #
class FakeDocker:
    """In-memory stand-in for :class:`DockerCLI` recording every interaction."""

    def __init__(
        self,
        *,
        exec_results: list[ExecOutcome] | None = None,
        reproduce: ExecOutcome | None = None,
        image_id_before: str | None = None,
        commit_id: str = "sha256:newimage",
        pull_ok: bool = True,
    ) -> None:
        self._exec_queue = list(exec_results or [])
        self._reproduce = (
            reproduce if reproduce is not None else ExecOutcome(0, "49 passed")
        )
        self._image_id_before = image_id_before
        self._commit_id = commit_id
        self._pull_ok = pull_ok
        self.run_names: list[str] = []
        self.ephemeral_names: list[str] = []
        self.exec_scripts: list[str] = []
        self.committed: list[str] = []
        self.removed_containers: list[str] = []
        self.removed_images: list[str] = []

    def version(self) -> str:
        return "29.2.1"

    def ensure_image(self, ref: str, *, timeout: float) -> None:
        if not self._pull_ok:
            raise EnvBuildError(f"failed to pull base image {ref!r}")

    def image_id(self, ref: str) -> str | None:
        return self._image_id_before

    def run_detached(self, *, name: str, image: str, workdir: str, **_: object) -> str:
        self.run_names.append(name)
        return f"cid-{name}"

    def copy_into(self, container: str, src_dir: object, dest_dir: str) -> None:
        pass

    def exec(
        self, container: str, script: str, *, workdir: str, timeout: float
    ) -> ExecOutcome:
        self.exec_scripts.append(script)
        return self._exec_queue.pop(0)

    def commit(self, container: str, tag: str, *, workdir: str) -> str:
        self.committed.append(tag)
        return self._commit_id

    def run_ephemeral(self, *, name: str, image: str, **_: object) -> ExecOutcome:
        self.ephemeral_names.append(name)
        return self._reproduce

    def remove_container(self, ref: str) -> None:
        self.removed_containers.append(ref)

    def remove_image(self, ref: str) -> None:
        self.removed_images.append(ref)


@pytest.fixture
def py_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "demo_py"
    _write(repo, "pyproject.toml", "[project]\nname = 'demo'\n")
    _write(repo, "demo.py", "def add(a, b):\n    return a + b\n")
    return repo


def _build(fake: FakeDocker, repo: Path, repo_id: str = "demo/py") -> EnvBuildResult:
    builder = EnvBuilder(
        docker=fake, registry=build_default_registry(), run_id="testrun"
    )
    return builder.build_from_path(repo, repo_id=repo_id, commit=_COMMIT)


_OK = ExecOutcome(0, "ok")


class TestBuilderHappyPath:
    def test_green_baseline_produces_env_image(self, py_repo: Path) -> None:
        fake = FakeDocker(
            exec_results=[_OK, _OK, ExecOutcome(0, "49 passed in 0.03s")],
            reproduce=ExecOutcome(0, "49 passed in 0.02s"),
        )
        result = _build(fake, py_repo)

        assert result.success is True
        assert result.failure_kind == ""
        assert result.install_exit_code == 0
        assert result.baseline_exit_code == 0
        env = result.env_image
        assert env is not None
        assert env.language == "python"
        assert env.base_image == "python:3.12-slim"
        assert env.image_tag == "swe-forge-env-demo_py:0123456789ab"
        assert env.commit == _COMMIT
        assert env.baseline_test_command == "python -m pytest"
        assert env.install_commands == ["pip install -e .", "pip install pytest"]
        assert env.baseline_green is True
        assert env.baseline_exit_code == 0
        assert "49 passed" in env.baseline_summary
        assert env.provenance["docker_version"] == "29.2.1"

    def test_image_committed_and_baseline_reproduced_fresh(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[_OK, _OK, _OK])
        result = _build(fake, py_repo)
        # Exactly one image committed (net +1) and a FRESH ephemeral re-run done.
        assert fake.committed == ["swe-forge-env-demo_py:0123456789ab"]
        assert len(fake.ephemeral_names) == 1
        assert result.success is True

    def test_unique_scoped_container_names_and_teardown(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[_OK, _OK, _OK])
        _build(fake, py_repo)
        assert len(fake.run_names) == 1
        build_name = fake.run_names[0]
        baseline_name = fake.ephemeral_names[0]
        assert build_name.startswith("swe-forge-env-build-testrun-")
        assert baseline_name.startswith("swe-forge-env-baseline-testrun-")
        assert build_name != baseline_name
        # The build container is torn down by id even on success.
        assert fake.removed_containers == [f"cid-{build_name}"]

    def test_rebuild_removes_superseded_image_no_dangling(self, py_repo: Path) -> None:
        fake = FakeDocker(
            exec_results=[_OK, _OK, _OK],
            image_id_before="sha256:oldimage",
            commit_id="sha256:newimage",
        )
        _build(fake, py_repo)
        # The previously-tagged image id is removed so no dangling image remains.
        assert "sha256:oldimage" in fake.removed_images


class TestBuilderRejections:
    def test_install_failure_is_distinct_and_no_image(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[_OK, ExecOutcome(1, "could not install")])
        result = _build(fake, py_repo)

        assert result.success is False
        assert result.failure_kind == INSTALL_FAILED
        assert result.stage == "install"
        assert "install/build failed" in result.reason
        assert result.install_exit_code == 1
        assert result.baseline_exit_code is None
        assert result.env_image is None
        # No commit, no ephemeral baseline run, build container still torn down.
        assert fake.committed == []
        assert fake.ephemeral_names == []
        assert len(fake.removed_containers) == 1

    def test_baseline_failure_distinct_from_install(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[_OK, _OK, ExecOutcome(1, "1 failed")])
        result = _build(fake, py_repo)

        assert result.success is False
        assert result.failure_kind == BASELINE_FAILED
        assert result.stage == "baseline"
        assert "baseline tests failed" in result.reason
        assert result.install_exit_code == 0
        assert result.baseline_exit_code == 1
        assert result.env_image is None
        assert fake.committed == []
        assert len(fake.removed_containers) == 1

    def test_red_baseline_emits_no_green_artifact(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[_OK, _OK, ExecOutcome(2, "FAILED")])
        result = _build(fake, py_repo)
        payload = result.to_dict()
        assert payload["env_image"] is None
        # Never a baseline_green:true artifact for a red repo.
        assert json.dumps(payload).find('"baseline_green": true') == -1

    def test_non_reproducible_baseline_discards_image(self, py_repo: Path) -> None:
        fake = FakeDocker(
            exec_results=[_OK, _OK, ExecOutcome(0, "passed")],
            reproduce=ExecOutcome(1, "failed on fresh container"),
        )
        result = _build(fake, py_repo)

        assert result.success is False
        assert result.failure_kind == REPRODUCE_FAILED
        # Committed then removed -> no shipped image.
        assert fake.committed == ["swe-forge-env-demo_py:0123456789ab"]
        assert "swe-forge-env-demo_py:0123456789ab" in fake.removed_images
        assert result.env_image is None

    def test_detect_failure_on_unknown_repo(self, tmp_path: Path) -> None:
        empty = tmp_path / "mystery"
        empty.mkdir()
        fake = FakeDocker(exec_results=[])
        result = _build(fake, empty, repo_id="x/mystery")
        assert result.success is False
        assert result.failure_kind == DETECT_FAILED
        # Detection fails before any container is created.
        assert fake.run_names == []

    def test_image_pull_failure(self, py_repo: Path) -> None:
        fake = FakeDocker(exec_results=[], pull_ok=False)
        result = _build(fake, py_repo)
        assert result.success is False
        assert result.failure_kind == IMAGE_PULL_FAILED
        assert fake.run_names == []

    def test_only_own_resources_touched(self, py_repo: Path) -> None:
        """Hygiene: every container/image the builder removes is its own."""
        fake = FakeDocker(exec_results=[_OK, _OK, ExecOutcome(1, "fail")])
        _build(fake, py_repo)
        for ref in fake.removed_containers + fake.removed_images:
            assert ref.startswith("cid-swe-forge-env-") or ref.startswith(
                "swe-forge-env-"
            )


# --------------------------------------------------------------------------- #
# Green-baseline downstream precondition (VAL-ENV-022).
# --------------------------------------------------------------------------- #
class TestRequireGreenBaseline:
    def _green(self) -> EnvImage:
        return EnvImage(
            repo_id="demo/py",
            language="python",
            image_tag="swe-forge-env-demo_py:0123456789ab",
            base_image="python:3.12-slim",
            commit=_COMMIT,
            workspace_dir="/workspace/repo",
            install_commands=["pip install -e ."],
            baseline_test_command="python -m pytest",
            baseline_green=True,
            baseline_exit_code=0,
        )

    def test_missing_env_image_blocks(self) -> None:
        with pytest.raises(BaselineNotGreenError):
            require_green_baseline(None)

    def test_non_green_blocks(self) -> None:
        env = self._green()
        env.baseline_green = False
        with pytest.raises(BaselineNotGreenError):
            require_green_baseline(env)

    def test_green_passes(self) -> None:
        env = self._green()
        assert require_green_baseline(env) is env

    def test_env_image_roundtrip(self) -> None:
        env = self._green()
        restored = EnvImage.from_dict(env.to_dict())
        assert restored.to_dict() == env.to_dict()


# --------------------------------------------------------------------------- #
# CLI surface (builder stubbed; offline).
# --------------------------------------------------------------------------- #
class _StubBuilder:
    def __init__(self, result: EnvBuildResult) -> None:
        self._result = result

    def build(self, spec: RepoSpec, **_: object) -> EnvBuildResult:
        return self._result

    def build_from_path(self, path: object, **_: object) -> EnvBuildResult:
        return self._result


def _green_result() -> EnvBuildResult:
    env = EnvImage(
        repo_id="demo/py",
        language="python",
        image_tag="swe-forge-env-demo_py:0123456789ab",
        base_image="python:3.12-slim",
        commit=_COMMIT,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e .", "pip install pytest"],
        baseline_test_command="python -m pytest",
        baseline_green=True,
        baseline_exit_code=0,
        baseline_summary="49 passed",
    )
    return EnvBuildResult(
        repo_id="demo/py",
        language="python",
        success=True,
        stage="complete",
        image_tag=env.image_tag,
        env_image=env,
        install_exit_code=0,
        baseline_exit_code=0,
    )


def _red_result() -> EnvBuildResult:
    return EnvBuildResult(
        repo_id="pytest-dev/iniconfig",
        language="python",
        success=False,
        stage="baseline",
        failure_kind=BASELINE_FAILED,
        reason="baseline tests failed (exit 1); baseline suite is not green",
        install_exit_code=0,
        baseline_exit_code=1,
    )


class TestBuildEnvCli:
    def test_path_green_build_emits_env_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "demo_py"
        repo.mkdir()
        emit = tmp_path / "env.json"
        monkeypatch.setattr(
            "swe_forge.forge.cli.EnvBuilder", lambda: _StubBuilder(_green_result())
        )
        res = runner.invoke(
            forge_app,
            ["build-env", "--path", str(repo), "--emit", str(emit), "--json"],
        )
        assert res.exit_code == 0, res.output
        payload = json.loads(res.output)
        assert payload["success"] is True
        assert payload["env_image"]["baseline_green"] is True
        # --emit wrote the EnvImage JSON.
        assert json.loads(emit.read_text())["image_tag"] == payload["image_tag"]

    def test_advance_blocks_non_green_repo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "swe_forge.forge.cli.EnvBuilder", lambda: _StubBuilder(_red_result())
        )
        res = runner.invoke(
            forge_app,
            ["build-env", "--repo", "pytest-dev/iniconfig", "--advance", "--json"],
        )
        assert res.exit_code == 1, res.output
        payload = json.loads(res.output)
        assert payload["advance"]["advanced"] is False
        assert "no green baseline" in payload["advance"]["reason"]
        # No downstream artifact: env_image stays null.
        assert payload["env_image"] is None

    def test_requires_repo_or_path(self) -> None:
        res = runner.invoke(forge_app, ["build-env", "--json"])
        assert res.exit_code == 1

    def test_repo_and_path_mutually_exclusive(self, tmp_path: Path) -> None:
        res = runner.invoke(
            forge_app, ["build-env", "--repo", "x", "--path", str(tmp_path)]
        )
        assert res.exit_code == 1

    def test_unknown_repo_id(self) -> None:
        res = runner.invoke(forge_app, ["build-env", "--repo", "not/a-real-repo"])
        assert res.exit_code == 1


# --------------------------------------------------------------------------- #
# Real-Docker end-to-end (deselected from the gate).
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_go_seed_real_green_baseline() -> None:
    """Build the Go seed for real and confirm a reproducible green baseline."""
    from swe_forge.forge.envbuild.builder import DockerCLI
    from swe_forge.forge.sources import build_source_registry

    spec = build_source_registry().get("golang-jwt/jwt")
    docker = DockerCLI()
    builder = EnvBuilder(docker=docker)
    result = builder.build(spec)
    try:
        assert result.success is True, result.reason
        env = result.env_image
        assert env is not None
        assert env.base_image == "golang:1.22"
        assert env.baseline_test_command == "go test ./..."
        assert env.baseline_green is True
        # Reproduce once more in a fresh throwaway container.
        repro = docker.run_ephemeral(
            name="swe-forge-env-itest-go",
            image=env.image_tag,
            script=env.baseline_test_command,
            workdir=env.workspace_dir,
            timeout=600.0,
            memory_mb=4096,
            cpus=4.0,
            pids_limit=2048,
        )
        assert repro.exit_code == 0
    finally:
        if result.image_tag:
            docker.remove_image(result.image_tag)
