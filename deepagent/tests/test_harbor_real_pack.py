"""Harbor real-pack export gates (VAL-HPACK-001..009) — offline unit tests.

Fixtures motors may dispatch file:// / short SHA; DeepAgent product packs under
datasets/deepagent_v1 require real HTTPS remotes, full 40-char SHAs, multi-file
gold, isolation, held-out tests, and pre_artifacts nonempty capture after sol.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import tomllib
from pathlib import Path

import pytest

from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree
from swe_factory.harbor.pre_artifacts import render_pre_artifacts_sh
from swe_factory.harbor.real_pack import (
    RealPackError,
    RealPackValidationResult,
    export_real_harbor_pack,
    product_source_files,
    run_pre_artifacts_capture,
    scan_instruction_gold_leak,
    validate_real_harbor_pack,
)
from swe_factory.harbor.schema import (
    MODEL_PATCH_ARTIFACT,
    HarborMetadata,
    HarborPackSpec,
    HarborTaskIdentity,
    HarborTaskToml,
    HarborVerifier,
    TestsConfig,
)

# Offline-looking but form-valid real identity fields.
_REAL_URL = "https://github.com/pallets/click.git"
_REAL_SHA = "a1b2c3d4e5f6789012345678901234567890abcd"
_MULTI_SOL = textwrap.dedent(
    """\
    diff --git a/src/pkg/core.py b/src/pkg/core.py
    --- a/src/pkg/core.py
    +++ b/src/pkg/core.py
    @@ -1,3 +1,4 @@
     def run():
    -    return 0
    +    return 1
    +
    diff --git a/src/pkg/util.py b/src/pkg/util.py
    --- a/src/pkg/util.py
    +++ b/src/pkg/util.py
    @@ -1,2 +1,3 @@
     def helper():
    -    return "x"
    +    return "y"
    """
)
_TEST_PATCH = textwrap.dedent(
    """\
    diff --git a/tests/test_held_out.py b/tests/test_held_out.py
    new file mode 100644
    --- /dev/null
    +++ b/tests/test_held_out.py
    @@ -0,0 +1,3 @@
    +def test_held():
    +    from src.pkg.core import run
    +    assert run() == 1
    """
)
_INSTRUCTION = textwrap.dedent(
    """\
    # Restore multi-module package behavior

    The package under `src/pkg` no longer returns expected values after a
    partial refactor. Fix the implementation so existing tests pass and the
    new behavioral contract for `run()` / `helper()` holds.

    IMPORTANT: commit your work on a branch. Do not leave uncommitted changes.
    """
)


def _real_spec(**overrides: object) -> HarborPackSpec:
    base: dict[str, object] = {
        "task_id": "real-click-hpack-demo",
        "instruction_md": _INSTRUCTION,
        "task_toml": HarborTaskToml(
            schema_version="1.1",
            artifacts=[MODEL_PATCH_ARTIFACT],
            task=HarborTaskIdentity(name="swe-factory/real-click-hpack-demo"),
            metadata=HarborMetadata(
                language="python",
                repository_url=_REAL_URL,
                base_commit_hash=_REAL_SHA,
                task_id="real-click-hpack-demo",
                source_track="real_pr",
                license="BSD-3-Clause",
            ),
            verifier=HarborVerifier(environment_mode="separate", timeout_sec=1800.0),
        ),
        "tests_config": TestsConfig(
            base_commit=_REAL_SHA,
            f2p_node_ids=["tests.test_held_out.test_held"],
            p2p_node_ids=["tests.test_ok.test_always_ok"],
        ),
        "solution_patch": _MULTI_SOL,
        "test_patch": _TEST_PATCH,
        "environment_dockerfile": (
            "FROM python:3.12-slim\n"
            f'LABEL harbor.allow_internet="false"\n'
            f'LABEL swe_factory.base_commit="{_REAL_SHA}"\n'
            "ENV HARBOR_ALLOW_INTERNET=false\n"
            f"ENV BASE_COMMIT={_REAL_SHA}\n"
            "WORKDIR /app\n"
            f"ARG BASE_SHA={_REAL_SHA}\n"
            # Product real_pr: clone@SHA authority (not motor COPY repo/).
            f'RUN git clone --filter=blob:none "{_REAL_URL}" . \\\n'
            ' && git fetch --depth 1 origin "$BASE_SHA" || git fetch origin "$BASE_SHA" \\\n'
            ' && git checkout --force "$BASE_SHA" \\\n'
            ' && test "$(git rev-parse HEAD)" = "$BASE_SHA" \\\n'
            ' && git rev-parse --verify "${BASE_SHA}^{commit}" \\\n'
            " && git rev-parse HEAD > /app/.harbor_base_commit\n"
        ),
        "tests_dockerfile": (
            "FROM deepagent-agent:local\n"
            "COPY test.sh /tests/test.sh\n"
            "COPY grader.py /tests/grader.py\n"
            "COPY config.json /tests/config.json\n"
            "COPY test.patch /tests/test.patch\n"
        ),
    }
    base.update(overrides)
    return HarborPackSpec.model_validate(base)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@local",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@local",
        },
    )


def _init_repo_with_sol(tmp_path: Path) -> tuple[Path, str]:
    """Create a repo at real-looking SHA not required; return (repo, base_sha)."""
    repo = tmp_path / "app"
    repo.mkdir(parents=True)
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "core.py").write_text("def run():\n    return 0\n", encoding="utf-8")
    (repo / "src" / "pkg" / "util.py").write_text(
        'def helper():\n    return "x"\n', encoding="utf-8"
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@local")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-B", "main")
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / ".harbor_base_commit").write_text(sha + "\n", encoding="utf-8")
    # Solution commit
    (repo / "src" / "pkg" / "core.py").write_text("def run():\n    return 1\n\n", encoding="utf-8")
    (repo / "src" / "pkg" / "util.py").write_text(
        'def helper():\n    return "y"\n', encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=oracle",
        "-c",
        "user.email=oracle@local",
        "commit",
        "-q",
        "--no-verify",
        "-m",
        "Apply reference solution",
    )
    return repo, sha


# --- pure helpers -----------------------------------------------------------------


def test_product_source_files_excludes_tests() -> None:
    """VAL-HPACK-007: ≥2 product/source files, not pure test-only gold."""
    files = product_source_files(_MULTI_SOL)
    assert len(files) >= 2
    assert all(not f.startswith("tests/") for f in files)
    pure_test = textwrap.dedent(
        """\
        diff --git a/tests/a.py b/tests/a.py
        --- a/tests/a.py
        +++ b/tests/a.py
        @@ -1 +1 @@
        -x
        +y
        diff --git a/tests/b.py b/tests/b.py
        --- a/tests/b.py
        +++ b/tests/b.py
        @@ -1 +1 @@
        -a
        +b
        """
    )
    assert product_source_files(pure_test) == []


def test_instruction_gold_leak_detection() -> None:
    """VAL-HPACK-009: instruction must not embed solution.diff gold bodies."""
    clean = scan_instruction_gold_leak(_INSTRUCTION, _MULTI_SOL)
    assert clean == []
    leaky = _INSTRUCTION + "\n\n```diff\n" + _MULTI_SOL + "\n```\n"
    hits = scan_instruction_gold_leak(leaky, _MULTI_SOL)
    assert hits, "embedding full solution.patch in instruction must be a leak"


def test_export_real_pack_tree_complete(tmp_path: Path) -> None:
    """VAL-HPACK-001: full DeepAgent relpath set present after real export."""
    result = export_real_harbor_pack(_real_spec(), dest=tmp_path / "real-click-hpack-demo")
    assert result.validation.ok
    missing = verify_pack_tree(result.pack_dir)
    assert missing == []
    for rel in REQUIRED_PACK_RELPATHS:
        assert (result.pack_dir / rel).is_file(), rel


def test_export_real_pack_metadata_real_url_and_sha(tmp_path: Path) -> None:
    """VAL-HPACK-002: real HTTPS repository_url + full 40-char base SHA + separate."""
    pack = export_real_harbor_pack(_real_spec(), dest=tmp_path / "real-click-hpack-demo")
    data = tomllib.loads((pack.pack_dir / "task.toml").read_text(encoding="utf-8"))
    assert data["schema_version"].startswith("1.")
    assert data["metadata"]["repository_url"] == _REAL_URL
    assert data["metadata"]["repository_url"].startswith("https://")
    assert not data["metadata"]["repository_url"].startswith("file://")
    assert len(data["metadata"]["base_commit_hash"]) == 40
    assert all(c in "0123456789abcdef" for c in data["metadata"]["base_commit_hash"].lower())
    assert data["verifier"]["environment_mode"] == "separate"
    assert pack.validation.real_url_ok
    assert pack.validation.real_sha_ok


def test_export_real_pack_rejects_file_url(tmp_path: Path) -> None:
    """VAL-HPACK-002: file:// motor remotes fail real packing gate."""
    with pytest.raises(RealPackError, match="repository_url|real"):
        export_real_harbor_pack(
            _real_spec(
                task_toml=HarborTaskToml(
                    schema_version="1.1",
                    artifacts=[MODEL_PATCH_ARTIFACT],
                    task=HarborTaskIdentity(name="swe-factory/motor"),
                    metadata=HarborMetadata(
                        language="python",
                        repository_url="file://fixtures/tiny_offline",
                        base_commit_hash=_REAL_SHA,
                        task_id="motor-bad-url",
                    ),
                    verifier=HarborVerifier(environment_mode="separate"),
                ),
                tests_config=TestsConfig(
                    base_commit=_REAL_SHA,
                    f2p_node_ids=["tests.t.test_a"],
                ),
                task_id="motor-bad-url",
            ),
            dest=tmp_path / "motor-bad-url",
        )


def test_export_real_pack_rejects_synthetic_sha(tmp_path: Path) -> None:
    """VAL-HPACK-002/003: synthetic a1000… / short SHA refused for real packs."""
    bad_sha = "a100000000000000000000000000000000000001"
    with pytest.raises(RealPackError, match="base_commit|SHA|sha"):
        export_real_harbor_pack(
            _real_spec(
                task_toml=HarborTaskToml(
                    schema_version="1.1",
                    artifacts=[MODEL_PATCH_ARTIFACT],
                    task=HarborTaskIdentity(name="swe-factory/syn"),
                    metadata=HarborMetadata(
                        language="python",
                        repository_url=_REAL_URL,
                        base_commit_hash=bad_sha,
                        task_id="syn-sha",
                    ),
                    verifier=HarborVerifier(environment_mode="separate"),
                ),
                tests_config=TestsConfig(
                    base_commit=bad_sha,
                    f2p_node_ids=["tests.t.test_a"],
                ),
                task_id="syn-sha",
            ),
            dest=tmp_path / "syn-sha",
        )

    with pytest.raises(RealPackError, match="base_commit|SHA|sha"):
        export_real_harbor_pack(
            _real_spec(
                task_toml=HarborTaskToml(
                    schema_version="1.1",
                    artifacts=[MODEL_PATCH_ARTIFACT],
                    task=HarborTaskIdentity(name="swe-factory/short"),
                    metadata=HarborMetadata(
                        language="python",
                        repository_url=_REAL_URL,
                        base_commit_hash="deadbeef",
                        task_id="short-sha",
                    ),
                    verifier=HarborVerifier(environment_mode="separate"),
                ),
                tests_config=TestsConfig(
                    base_commit="deadbeef",
                    f2p_node_ids=["tests.t.test_a"],
                ),
                task_id="short-sha",
            ),
            dest=tmp_path / "short-sha",
        )


def test_export_real_pack_rejects_shared_verifier(tmp_path: Path) -> None:
    """VAL-HPACK-005: shared verifier mode is refused for real DeepAgent packs."""
    with pytest.raises((RealPackError, ValueError), match="separate"):
        export_real_harbor_pack(
            _real_spec(
                task_toml=HarborTaskToml(
                    schema_version="1.1",
                    artifacts=[MODEL_PATCH_ARTIFACT],
                    task=HarborTaskIdentity(name="swe-factory/shared"),
                    metadata=HarborMetadata(
                        language="python",
                        repository_url=_REAL_URL,
                        base_commit_hash=_REAL_SHA,
                        task_id="shared-mode",
                    ),
                    verifier=HarborVerifier(environment_mode="shared"),
                ),
                task_id="shared-mode",
            ),
            dest=tmp_path / "shared-mode",
        )


def test_validate_held_out_and_node_ids(tmp_path: Path) -> None:
    """VAL-HPACK-006: nonempty test.patch + f2p_node_ids present settings."""
    pack = export_real_harbor_pack(_real_spec(), dest=tmp_path / "real-click-hpack-demo")
    v = validate_real_harbor_pack(pack.pack_dir)
    assert v.ok
    assert v.test_patch_ok
    assert v.config_ok
    assert v.f2p_count >= 1
    cfg = json.loads((pack.pack_dir / "tests" / "config.json").read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"]
    assert (pack.pack_dir / "tests" / "test.patch").read_text(encoding="utf-8").strip()


def test_export_real_pack_rejects_single_file_solution(tmp_path: Path) -> None:
    """VAL-HPACK-007: single product file fails multi-file floor."""
    single = textwrap.dedent(
        """\
        diff --git a/src/only.py b/src/only.py
        --- a/src/only.py
        +++ b/src/only.py
        @@ -1 +1 @@
        -x
        +y
        """
    )
    with pytest.raises(RealPackError, match="multi-file|product|files"):
        export_real_harbor_pack(
            _real_spec(solution_patch=single),
            dest=tmp_path / "single-file",
        )


def test_agent_isolation_solution_and_held_out(tmp_path: Path) -> None:
    """VAL-HPACK-008: agent environment context never ships solution/ or test.patch."""
    pack = export_real_harbor_pack(
        _real_spec(),
        dest=tmp_path / "real-click-hpack-demo",
        copy_repo_into_environment=None,
    )
    v = validate_real_harbor_pack(pack.pack_dir)
    assert v.isolation_clean
    env = pack.pack_dir / "environment"
    assert not (env / "solution").exists()
    assert not (env / "solution.patch").exists()
    assert not (env / "test.patch").exists()
    assert not (env / "tests").exists()
    # solution still exists at pack level (oracle path)
    assert (pack.pack_dir / "solution" / "solution.patch").is_file()
    # held-out stays under tests/
    assert (pack.pack_dir / "tests" / "test.patch").is_file()


def test_instruction_nonempty_without_gold_on_export(tmp_path: Path) -> None:
    """VAL-HPACK-009: instruction present and clean of gold body."""
    pack = export_real_harbor_pack(_real_spec(), dest=tmp_path / "real-click-hpack-demo")
    instr = (pack.pack_dir / "instruction.md").read_text(encoding="utf-8")
    assert instr.strip()
    assert "diff --git" not in instr
    assert pack.validation.instruction_ok


def test_instruction_gold_leak_refuses_export(tmp_path: Path) -> None:
    leaky = _INSTRUCTION + "\n\n" + _MULTI_SOL
    with pytest.raises(RealPackError, match="instruction|gold|leak"):
        export_real_harbor_pack(
            _real_spec(instruction_md=leaky),
            dest=tmp_path / "leaky",
        )


def test_pre_artifacts_nonempty_after_solution(tmp_path: Path) -> None:
    """VAL-HPACK-004: pre_artifacts.sh after sol commit → model.patch size > 0."""
    repo, real_sha = _init_repo_with_sol(tmp_path)
    # Export pack then prove capture using the emitted pre_artifacts template path
    pack = export_real_harbor_pack(
        _real_spec(
            task_toml=HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name="swe-factory/preart"),
                metadata=HarborMetadata(
                    language="python",
                    repository_url=_REAL_URL,
                    # Keep real-shaped metadata; actual object is local real_sha
                    base_commit_hash=_REAL_SHA,
                    task_id="preart-demo",
                    source_track="real_pr",
                ),
                verifier=HarborVerifier(environment_mode="separate"),
            ),
            tests_config=TestsConfig(
                base_commit=_REAL_SHA,
                f2p_node_ids=["tests.t.t"],
            ),
            task_id="preart-demo",
        ),
        dest=tmp_path / "preart-demo",
    )
    pre = (pack.pack_dir / "pre_artifacts.sh").read_text(encoding="utf-8")
    assert "model.patch" in pre
    assert "rev-list --max-parents=0" in pre
    # Simulate script against local sol-committed worktree
    capture = run_pre_artifacts_capture(
        repo,
        pre_artifacts_sh=render_pre_artifacts_sh(real_sha),
        logs_dir=tmp_path / "logs",
    )
    assert capture.ok
    assert capture.byte_size > 0
    assert "core.py" in capture.patch_text or "util.py" in capture.patch_text


def test_base_commit_resolves_as_git_object(tmp_path: Path) -> None:
    """VAL-HPACK-003: local workspace can rev-parse the real base SHA as a commit."""
    repo, real_sha = _init_repo_with_sol(tmp_path)
    # Identity check: full 40-char hex present after initial commit
    assert len(real_sha) == 40
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{real_sha}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    # Validation helper surfaces the same proof when repo given
    v = validate_real_harbor_pack(
        export_real_harbor_pack(
            _real_spec(
                task_toml=HarborTaskToml(
                    schema_version="1.1",
                    artifacts=[MODEL_PATCH_ARTIFACT],
                    task=HarborTaskIdentity(name="swe-factory/resolve"),
                    metadata=HarborMetadata(
                        language="python",
                        repository_url=_REAL_URL,
                        base_commit_hash=real_sha,
                        task_id="resolve-demo",
                        source_track="real_pr",
                    ),
                    verifier=HarborVerifier(environment_mode="separate"),
                ),
                tests_config=TestsConfig(
                    base_commit=real_sha,
                    f2p_node_ids=["tests.t.t"],
                ),
                # resolve uses a real local sha; Dockerfile must match that pin.
                environment_dockerfile=(
                    "FROM python:3.12-slim\n"
                    'LABEL harbor.allow_internet="false"\n'
                    f'LABEL swe_factory.base_commit="{real_sha}"\n'
                    "ENV HARBOR_ALLOW_INTERNET=false\n"
                    f"ENV BASE_COMMIT={real_sha}\n"
                    "WORKDIR /app\n"
                    f"ARG BASE_SHA={real_sha}\n"
                    f'RUN git clone --filter=blob:none "{_REAL_URL}" . \\\n'
                    ' && (git fetch --depth 1 origin "$BASE_SHA"'
                    ' || git fetch origin "$BASE_SHA") \\\n'
                    ' && git checkout --force "$BASE_SHA" \\\n'
                    ' && test "$(git rev-parse HEAD)" = "$BASE_SHA" \\\n'
                    ' && git rev-parse --verify "${BASE_SHA}^{commit}" \\\n'
                    " && git rev-parse HEAD > /app/.harbor_base_commit\n"
                ),
                task_id="resolve-demo",
            ),
            dest=tmp_path / "resolve-demo",
        ).pack_dir,
        workspace_repo=repo,
    )
    assert v.base_object_ok
    assert v.ok


def test_validate_result_to_dict_fields(tmp_path: Path) -> None:
    pack = export_real_harbor_pack(_real_spec(), dest=tmp_path / "real-click-hpack-demo")
    d = pack.validation.to_dict()
    for key in (
        "ok",
        "tree_complete",
        "real_url_ok",
        "real_sha_ok",
        "separate_verifier",
        "test_patch_ok",
        "config_ok",
        "multi_file_ok",
        "isolation_clean",
        "instruction_ok",
        "solution_files",
        "reason_codes",
    ):
        assert key in d
    assert isinstance(pack.validation, RealPackValidationResult)


def test_export_real_pack_refuses_copy_repo_hybrid_bind(tmp_path: Path) -> None:
    """Product real_pr export refuses COPY-bind hybrid trees (VAL-RCLN-002).

    Offline non-product motors may still COPY via ``export_harbor_pack``;
    certified real packs must materialize agent trees via clone@SHA only.
    """
    repo_src = tmp_path / "src_repo"
    repo_src.mkdir()
    (repo_src / "pkg.py").write_text("x=1\n", encoding="utf-8")
    (repo_src / "orderlib").mkdir()
    (repo_src / "orderlib" / "pricing.py").write_text("x=1\n", encoding="utf-8")
    with pytest.raises(RealPackError, match="copy_repo_into_environment|hybrid|clone@SHA"):
        export_real_harbor_pack(
            _real_spec(),
            dest=tmp_path / "real-click-hpack-demo",
            copy_repo_into_environment=repo_src,
        )

    # Isolation still holds when export succeeds without hybrid bind: no solution/
    pack = export_real_harbor_pack(
        _real_spec(),
        dest=tmp_path / "real-click-hpack-demo",
    )
    env = pack.pack_dir / "environment"
    assert env.is_dir()
    assert not (env / "solution").exists()
    assert not (env / "solution.patch").exists()
    assert not (env / "repo").exists()  # no motor bind tree stamped on product packs
    df = (env / "Dockerfile").read_text(encoding="utf-8")
    assert "git clone" in df
    assert "COPY repo/" not in df
    assert pack.validation.isolation_clean
