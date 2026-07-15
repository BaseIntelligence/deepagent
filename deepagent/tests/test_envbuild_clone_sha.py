"""VAL-RCLN-001..004: real clone@SHA agent Dockerfile + refuse motor hybrid.

Unit tests only (no docker daemon, no network). Covers:

- Product real_pr recipe materializes git clone of repository_url at full SHA
- Product path never emits motor-only ``COPY repo/`` while claiming a real URL
  (e.g. boltons metadata paired with orderlib motor body)
- Runtime offline ``allow_internet=false`` markers
- Export / assert gates reject motor hybrid for real_pr
- Fake / short SHA rejected for product clone recipe
"""

from __future__ import annotations

import textwrap

import pytest

from swe_factory.envbuild.agent_recipe import (
    ALLOW_INTERNET_FALSE,
    RealPrDockerfileError,
    assert_real_pr_agent_dockerfile,
    dockerfile_has_clone_at_sha,
    looks_motor_fixture_copy,
    render_agent_dockerfile,
    render_real_pr_agent_dockerfile,
)
from swe_factory.envbuild.fixture import recipe_from_clone
from swe_factory.harbor.grader_frame import (
    default_environment_dockerfile,
    offline_environment_dockerfile,
)
from swe_factory.harbor.real_pack import (
    RealPackError,
    assert_real_pack_spec,
    export_real_harbor_pack,
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

# Mixed real boltons-like identity (public remote + full SHA).
_BOLTONS_URL = "https://github.com/mahmoud/boltons.git"
_REAL_SHA = "a1b2c3d4e5f6789012345678abcdef0123456789"

_MULTI_SOL = textwrap.dedent(
    """\
    diff --git a/src/pkg/core.py b/src/pkg/core.py
    --- a/src/pkg/core.py
    +++ b/src/pkg/core.py
    @@ -1,2 +1,3 @@
     def run():
    -    return 0
    +    return 1
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
    @@ -0,0 +1,2 @@
    +def test_held():
    +    assert True
    """
)


# ---------------------------------------------------------------------------
# VAL-RCLN-001: clone template + pin
# ---------------------------------------------------------------------------


def test_render_real_pr_agent_dockerfile_clones_at_sha() -> None:
    df = render_real_pr_agent_dockerfile(
        repository_url=_BOLTONS_URL,
        base_commit=_REAL_SHA,
        language="python",
    )
    assert "git clone" in df
    assert _BOLTONS_URL in df
    assert _REAL_SHA in df
    assert "BASE_SHA" in df
    assert "git checkout --force" in df
    assert "rev-parse --verify" in df or "rev-parse HEAD" in df
    assert dockerfile_has_clone_at_sha(df)
    # Product: no motor COPY body
    assert "COPY repo/" not in df
    assert "orderlib" not in df
    assert "allow_internet=false" in df or "HARBOR_ALLOW_INTERNET=false" in df


def test_render_agent_dockerfile_public_url_never_copies_motor() -> None:
    """Even when copy_context=True, public HTTPS forces clone@SHA (mutual exclusion)."""
    df = render_agent_dockerfile(
        base_commit=_REAL_SHA,
        language="python",
        repo_url=_BOLTONS_URL,
        copy_context=True,  # tempt hybrid
        source_track="real_pr",
    )
    assert "git clone" in df
    assert "COPY repo/" not in df
    assert "HARBOR_ALLOW_INTERNET=false" in df


def test_builder_recipe_from_clone_is_product_real() -> None:
    recipe = recipe_from_clone(
        repo_id="mahmoud/boltons",
        base_commit=_REAL_SHA,
        language="python",
        clone_url=_BOLTONS_URL,
    )
    assert recipe.require_real_sha is True
    assert recipe.allow_internet is False
    assert recipe.clone_url == _BOLTONS_URL


def test_default_environment_dockerfile_uses_clone_for_real_url() -> None:
    df = default_environment_dockerfile(
        repo_url=_BOLTONS_URL,
        base_commit=_REAL_SHA,
    )
    assert "git clone" in df
    assert "COPY repo/" not in df
    assert ALLOW_INTERNET_FALSE in df or "HARBOR_ALLOW_INTERNET=false" in df
    assert dockerfile_has_clone_at_sha(df)


# ---------------------------------------------------------------------------
# VAL-RCLN-002: refuse motor COPY hybrid for product real_pr
# ---------------------------------------------------------------------------


def test_looks_motor_fixture_copy_for_offline_motor() -> None:
    offline = offline_environment_dockerfile()
    assert "COPY repo/" in offline
    assert looks_motor_fixture_copy(offline)
    assert not dockerfile_has_clone_at_sha(offline)


def test_assert_real_pr_refuses_motor_only_copy_with_boltons_url() -> None:
    """Classic hybrid smell: boltons repository_url + orderlib COPY repo body."""
    hybrid_df = textwrap.dedent(
        """\
        FROM python:3.12-slim
        WORKDIR /app
        # Hybrid: claim boltons, copy orderlib motor fixture
        COPY repo/ /app/
        # orderlib/pricing.py lives in COPY'd motor tree
        RUN git init && git commit -m base || true
        """
    )
    with pytest.raises(RealPrDockerfileError, match="motor|COPY|clone"):
        assert_real_pr_agent_dockerfile(
            hybrid_df,
            repository_url=_BOLTONS_URL,
            base_commit=_REAL_SHA,
            source_track="real_pr",
        )


def test_assert_real_pr_refuses_offline_motor_dockerfile() -> None:
    motor_df = offline_environment_dockerfile()
    with pytest.raises(RealPrDockerfileError):
        assert_real_pr_agent_dockerfile(
            motor_df,
            repository_url=_BOLTONS_URL,
            base_commit=_REAL_SHA,
            source_track="real_pr",
        )


def test_export_real_harbor_pack_refuses_motor_dockerfile(tmp_path) -> None:  # noqa: ANN001
    motor_df = offline_environment_dockerfile()
    spec = HarborPackSpec.model_validate(
        {
            "task_id": "hybrid-boltons-orderlib",
            "instruction_md": (
                "Fix the multi-module package so tests pass. Do not leave uncommitted changes."
            ),
            "task_toml": HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name="swe-factory/hybrid-boltons-orderlib"),
                metadata=HarborMetadata(
                    language="python",
                    repository_url=_BOLTONS_URL,
                    base_commit_hash=_REAL_SHA,
                    task_id="hybrid-boltons-orderlib",
                    source_track="real_pr",
                    license="BSD",
                ),
                verifier=HarborVerifier(environment_mode="separate", timeout_sec=1800.0),
            ),
            "tests_config": TestsConfig(
                base_commit=_REAL_SHA,
                f2p_node_ids=["tests.test_held.test_one"],
                p2p_node_ids=[],
            ),
            "solution_patch": _MULTI_SOL,
            "test_patch": _TEST_PATCH,
            "environment_dockerfile": motor_df,
            "tests_dockerfile": (
                "FROM deepagent-agent:local\n"
                "COPY test.sh /tests/test.sh\n"
                "COPY grader.py /tests/grader.py\n"
                "COPY config.json /tests/config.json\n"
                "COPY test.patch /tests/test.patch\n"
            ),
        }
    )
    with pytest.raises(RealPackError, match="motor|COPY|clone|real_pr"):
        export_real_harbor_pack(spec, dest=tmp_path / "hybrid-boltons-orderlib")


def test_export_real_harbor_pack_refuses_copy_repo_bind(tmp_path) -> None:  # noqa: ANN001
    good_df = render_real_pr_agent_dockerfile(
        repository_url=_BOLTONS_URL,
        base_commit=_REAL_SHA,
    )
    # Make minimal green tree that would have been hybrid-bound
    motor_tree = tmp_path / "orderlib_motor"
    (motor_tree / "orderlib").mkdir(parents=True)
    (motor_tree / "orderlib" / "pricing.py").write_text("x=1\n", encoding="utf-8")
    spec = HarborPackSpec.model_validate(
        {
            "task_id": "real-boltons-no-bind",
            "instruction_md": (
                "Restore package behavior after partial refactor. Do not leave uncommitted changes."
            ),
            "task_toml": HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name="swe-factory/real-boltons-no-bind"),
                metadata=HarborMetadata(
                    language="python",
                    repository_url=_BOLTONS_URL,
                    base_commit_hash=_REAL_SHA,
                    task_id="real-boltons-no-bind",
                    source_track="real_pr",
                    license="BSD",
                ),
                verifier=HarborVerifier(environment_mode="separate", timeout_sec=1800.0),
            ),
            "tests_config": TestsConfig(
                base_commit=_REAL_SHA,
                f2p_node_ids=["tests.test_held.test_one"],
                p2p_node_ids=[],
            ),
            "solution_patch": _MULTI_SOL,
            "test_patch": _TEST_PATCH,
            "environment_dockerfile": good_df,
            "tests_dockerfile": (
                "FROM deepagent-agent:local\n"
                "COPY test.sh /tests/test.sh\n"
                "COPY grader.py /tests/grader.py\n"
                "COPY config.json /tests/config.json\n"
                "COPY test.patch /tests/test.patch\n"
            ),
        }
    )
    with pytest.raises(RealPackError, match="copy_repo_into_environment|hybrid|clone@SHA"):
        export_real_harbor_pack(
            spec,
            dest=tmp_path / "real-boltons-no-bind",
            copy_repo_into_environment=motor_tree,
        )


def test_assert_real_pack_spec_accepts_clone_dockerfile() -> None:
    good_df = render_real_pr_agent_dockerfile(
        repository_url=_BOLTONS_URL,
        base_commit=_REAL_SHA,
    )
    spec = HarborPackSpec.model_validate(
        {
            "task_id": "real-boltons-ok",
            "instruction_md": (
                "Restore package behavior after partial refactor. Do not leave uncommitted changes."
            ),
            "task_toml": HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name="swe-factory/real-boltons-ok"),
                metadata=HarborMetadata(
                    language="python",
                    repository_url=_BOLTONS_URL,
                    base_commit_hash=_REAL_SHA,
                    task_id="real-boltons-ok",
                    source_track="real_pr",
                    license="BSD",
                ),
                verifier=HarborVerifier(environment_mode="separate", timeout_sec=1800.0),
            ),
            "tests_config": TestsConfig(
                base_commit=_REAL_SHA,
                f2p_node_ids=["tests.test_held.test_one"],
                p2p_node_ids=[],
            ),
            "solution_patch": _MULTI_SOL,
            "test_patch": _TEST_PATCH,
            "environment_dockerfile": good_df,
            "tests_dockerfile": (
                "FROM deepagent-agent:local\n"
                "COPY test.sh /tests/test.sh\n"
                "COPY grader.py /tests/grader.py\n"
                "COPY config.json /tests/config.json\n"
                "COPY test.patch /tests/test.patch\n"
            ),
        }
    )
    cleaned = assert_real_pack_spec(spec)
    assert "git clone" in cleaned.environment_dockerfile
    assert "COPY repo/" not in cleaned.environment_dockerfile


# ---------------------------------------------------------------------------
# VAL-RCLN-003: full SHA required; fake rejected
# ---------------------------------------------------------------------------


def test_render_real_pr_rejects_short_or_synthetic_sha() -> None:
    with pytest.raises(RealPrDockerfileError, match="40-char|base_commit"):
        render_real_pr_agent_dockerfile(
            repository_url=_BOLTONS_URL,
            base_commit="abc123",
        )
    with pytest.raises(RealPrDockerfileError, match="HTTPS|repository_url"):
        render_real_pr_agent_dockerfile(
            repository_url="file:///tmp/motor.git",
            base_commit=_REAL_SHA,
        )


def test_assert_real_pr_rejects_fake_sha_on_product_path() -> None:
    df = render_real_pr_agent_dockerfile(
        repository_url=_BOLTONS_URL,
        base_commit=_REAL_SHA,
    )
    with pytest.raises(RealPrDockerfileError, match="40-char"):
        assert_real_pr_agent_dockerfile(
            df,
            repository_url=_BOLTONS_URL,
            base_commit="notasha",
            source_track="real_pr",
        )


# ---------------------------------------------------------------------------
# VAL-RCLN-004: offline runtime allow_internet=false
# ---------------------------------------------------------------------------


def test_real_pr_dockerfile_documents_offline_runtime() -> None:
    df = render_real_pr_agent_dockerfile(
        repository_url=_BOLTONS_URL,
        base_commit=_REAL_SHA,
    )
    assert "allow_internet=false" in df
    assert "HARBOR_ALLOW_INTERNET=false" in df
    assert "Build-time dependency install" in df or "pip install" in df


def test_offline_motor_allowed_under_fixture_allowlist_flag() -> None:
    """Motor Dockerfiles remain valid for offline fixture tests (non-product)."""
    motor_df = offline_environment_dockerfile()
    # Does not raise when allow_fixture_only
    assert_real_pr_agent_dockerfile(
        motor_df,
        repository_url="file:///tmp/motor",
        base_commit="a100000000000000000000000000000000000001",
        source_track="synthetic_grounded",
        allow_fixture_only=True,
    )


def test_lang_motor_dockerfiles_still_copy_for_offline() -> None:
    # ensure offline motors still produce COPY layout (fixture path)
    df = offline_environment_dockerfile()
    assert "COPY repo/" in df
    assert looks_motor_fixture_copy(df)
    # Go/TS offline recipes in harbor_motors retain COPY repo/ as well.
    go_offline = "FROM golang:1.22-bookworm\nWORKDIR /app\nCOPY repo/ /app/\nRUN git init\n"
    assert looks_motor_fixture_copy(go_offline)
