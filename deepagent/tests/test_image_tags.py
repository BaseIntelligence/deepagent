"""Unit tests for pier/Harbor pack-scoped agent mintag resolution (M15).

Product packs used to ship ``tests/Dockerfile`` with ``FROM deepagent-agent:local``.
Sharing that global tag across multi-pack pier prep short-circuits later packs
onto the first pack's environment Dockerfile. These tests cover:

- pack-scoped digests / tags (two packs → distinct tags)
- ensure() paints FROM to pack-scoped tag
- short-circuit checks pack-scoped tag only (not the global)
- pier_cert mintag prepare fail-closed behaviour
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swe_factory.harbor.harbor_docker import (
    DEEPAGENT_AGENT_LOCAL_TAG,
    HarborDockerError,
    ensure_deepagent_agent_local,
    environment_content_digest,
    pack_scoped_agent_image_tag,
    paint_tests_dockerfile_from,
    resolve_pier_agent_image_tag,
    rewrite_tests_dockerfile_from,
    stage_agent_context,
)
from swe_factory.harbor.pier_cert import (
    ScriptedPierRunner,
    certify_pier_pack,
)


def _minimal_pack(
    root: Path,
    *,
    env_docker: str = "FROM python:3.12-slim\nWORKDIR /app\n",
    tests_from: str = "deepagent-agent:local",
) -> Path:
    """Create a minimal pack with environment/ + tests/ Dockerfiles."""
    env = root / "environment"
    env.mkdir(parents=True, exist_ok=True)
    (env / "Dockerfile").write_text(env_docker, encoding="utf-8")
    tests = root / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "Dockerfile").write_text(
        f"FROM {tests_from}\nCOPY test.sh /tests/test.sh\n",
        encoding="utf-8",
    )
    (tests / "config.json").write_text(
        '{"base_commit": "' + ("a" * 40) + '", "f2p_node_ids": ["t"]}',
        encoding="utf-8",
    )
    (tests / "test.patch").write_text(
        "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
        encoding="utf-8",
    )
    (tests / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (tests / "grader.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "solution").mkdir(exist_ok=True)
    (root / "solution" / "solution.patch").write_text(
        "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
        encoding="utf-8",
    )
    return root


def test_default_mintag_is_deepagent_agent_local() -> None:
    assert DEEPAGENT_AGENT_LOCAL_TAG == "deepagent-agent:local"
    # Without pack_dir, resolve still returns the legacy global default.
    assert resolve_pier_agent_image_tag() == "deepagent-agent:local"


def test_resolve_pier_agent_image_tag_prefers_explicit() -> None:
    assert (
        resolve_pier_agent_image_tag(agent_image="harbor-sdf-agent-item:oracle")
        == "harbor-sdf-agent-item:oracle"
    )
    assert (
        resolve_pier_agent_image_tag(agent_image="  deepagent-agent:local  ")
        == "deepagent-agent:local"
    )
    assert resolve_pier_agent_image_tag(agent_image="") == DEEPAGENT_AGENT_LOCAL_TAG
    assert resolve_pier_agent_image_tag(agent_image=None) == DEEPAGENT_AGENT_LOCAL_TAG


def test_resolve_pier_agent_image_tag_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWE_FACTORY_AGENT_IMAGE", "env-agent:tag")
    assert resolve_pier_agent_image_tag() == "env-agent:tag"
    # explicit still wins over env
    assert resolve_pier_agent_image_tag(agent_image="explicit:1") == "explicit:1"
    monkeypatch.delenv("SWE_FACTORY_AGENT_IMAGE", raising=False)
    assert resolve_pier_agent_image_tag() == DEEPAGENT_AGENT_LOCAL_TAG


def test_resolve_with_pack_dir_returns_pack_scoped(tmp_path: Path) -> None:
    pack = _minimal_pack(tmp_path / "p1")
    tag = resolve_pier_agent_image_tag(pack_dir=pack)
    assert tag.startswith("deepagent-agent-")
    assert tag.endswith(":local")
    assert tag != DEEPAGENT_AGENT_LOCAL_TAG


def test_rewrite_tests_dockerfile_from_replaces_deepagent_agent_local() -> None:
    src = "FROM deepagent-agent:local\nCOPY test.sh /tests/test.sh\n"
    out = rewrite_tests_dockerfile_from(src, agent_image="harbor-sdf-agent-demo:oracle")
    assert out.startswith("FROM harbor-sdf-agent-demo:oracle\n")
    assert "deepagent-agent:local" not in out
    assert "COPY test.sh" in out


def test_rewrite_preserves_deepagent_agent_local_when_requested() -> None:
    src = "FROM deepagent-agent:local\nCOPY grader.py /tests/grader.py\n"
    out = rewrite_tests_dockerfile_from(src, agent_image=DEEPAGENT_AGENT_LOCAL_TAG)
    assert out.splitlines()[0] == "FROM deepagent-agent:local"


def test_two_packs_different_dockerfiles_produce_distinct_tags(tmp_path: Path) -> None:
    """Core multi-pack isolation: different env Dockerfiles → different mintags."""
    pack_a = _minimal_pack(
        tmp_path / "pack-a",
        env_docker="FROM python:3.12-slim\nWORKDIR /app\n# pack A\nRUN echo a\n",
    )
    pack_b = _minimal_pack(
        tmp_path / "pack-b",
        env_docker="FROM python:3.12-slim\nWORKDIR /app\n# pack B\nRUN echo b\n",
    )
    dig_a = environment_content_digest(pack_a)
    dig_b = environment_content_digest(pack_b)
    assert dig_a != dig_b
    tag_a = pack_scoped_agent_image_tag(pack_a)
    tag_b = pack_scoped_agent_image_tag(pack_b)
    assert tag_a != tag_b
    assert tag_a == f"deepagent-agent-{dig_a}:local"
    assert tag_b == f"deepagent-agent-{dig_b}:local"

    work_a = tmp_path / "work-a"
    work_b = tmp_path / "work-b"
    out_a = ensure_deepagent_agent_local(pack_a, work_dir=work_a, stage_only=True)
    out_b = ensure_deepagent_agent_local(pack_b, work_dir=work_b, stage_only=True)
    assert out_a == tag_a
    assert out_b == tag_b
    assert out_a != out_b
    # Each pack's tests/Dockerfile painted to its own pack-scoped FROM
    from_a = (pack_a / "tests" / "Dockerfile").read_text(encoding="utf-8").splitlines()[0]
    from_b = (pack_b / "tests" / "Dockerfile").read_text(encoding="utf-8").splitlines()[0]
    assert from_a == f"FROM {tag_a}"
    assert from_b == f"FROM {tag_b}"
    assert "deepagent-agent:local" not in from_a
    assert "deepagent-agent:local" not in from_b


def test_identical_env_dockerfiles_share_digest(tmp_path: Path) -> None:
    body = "FROM python:3.12-slim\nWORKDIR /app\nRUN echo same\n"
    pack_a = _minimal_pack(tmp_path / "same-a", env_docker=body)
    pack_b = _minimal_pack(tmp_path / "same-b", env_docker=body)
    assert environment_content_digest(pack_a) == environment_content_digest(pack_b)
    assert pack_scoped_agent_image_tag(pack_a) == pack_scoped_agent_image_tag(pack_b)


def test_ensure_deepagent_agent_local_stage_only(tmp_path: Path) -> None:
    """Stage-only path builds environment context and returns pack-scoped mintag."""
    pack = _minimal_pack(tmp_path / "pack")
    work = tmp_path / "work"
    tag = ensure_deepagent_agent_local(
        pack,
        work_dir=work,
        stage_only=True,
        binary="docker-not-invoked",
    )
    expected = pack_scoped_agent_image_tag(pack)
    assert tag == expected
    assert tag != DEEPAGENT_AGENT_LOCAL_TAG
    agent_ctx = work / "agent_context"
    assert (agent_ctx / "Dockerfile").is_file()
    assert "python:3.12-slim" in (agent_ctx / "Dockerfile").read_text(encoding="utf-8")
    # agent context isolation: no solution/tests
    stubs = stage_agent_context(pack, work / "agent_context_again")
    assert stubs.forbidden_hits == ()
    # FROM painted
    painted = (pack / "tests" / "Dockerfile").read_text(encoding="utf-8")
    assert painted.startswith(f"FROM {tag}\n")


def test_ensure_does_not_short_circuit_on_global_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existence of global deepagent-agent:local must not skip pack-scoped build."""
    pack = _minimal_pack(
        tmp_path / "pack",
        env_docker="FROM python:3.12-slim\nWORKDIR /app\n# unique-body\n",
    )
    scoped = pack_scoped_agent_image_tag(pack)
    build_calls: list[str] = []

    def fake_exists(tag: str, *, binary: str = "docker") -> bool:
        # Pretend legacy global is present, pack-scoped is not.
        return tag == DEEPAGENT_AGENT_LOCAL_TAG

    def fake_build(*, context_dir: Any, tag: str, **_: Any) -> None:
        build_calls.append(tag)

    monkeypatch.setattr(
        "swe_factory.harbor.harbor_docker.docker_image_exists",
        fake_exists,
    )
    monkeypatch.setattr(
        "swe_factory.harbor.harbor_docker.docker_build",
        fake_build,
    )
    out = ensure_deepagent_agent_local(
        pack,
        work_dir=tmp_path / "work",
        stage_only=False,
        force_rebuild=False,
        binary="docker-mock",
    )
    assert out == scoped
    assert build_calls == [scoped]


def test_ensure_short_circuits_when_pack_scoped_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = _minimal_pack(tmp_path / "pack")
    scoped = pack_scoped_agent_image_tag(pack)
    build_calls: list[str] = []

    def fake_exists(tag: str, *, binary: str = "docker") -> bool:
        return tag == scoped

    def fake_build(*, context_dir: Any, tag: str, **_: Any) -> None:
        build_calls.append(tag)

    monkeypatch.setattr(
        "swe_factory.harbor.harbor_docker.docker_image_exists",
        fake_exists,
    )
    monkeypatch.setattr(
        "swe_factory.harbor.harbor_docker.docker_build",
        fake_build,
    )
    out = ensure_deepagent_agent_local(
        pack,
        work_dir=tmp_path / "work",
        stage_only=False,
        force_rebuild=False,
    )
    assert out == scoped
    assert build_calls == []


def test_paint_tests_dockerfile_from_in_place(tmp_path: Path) -> None:
    pack = _minimal_pack(tmp_path / "pack")
    tag = "deepagent-agent-deadbeefcafe:local"
    out = paint_tests_dockerfile_from(pack, agent_image=tag, in_place=True)
    assert out.startswith(f"FROM {tag}\n")
    assert (pack / "tests" / "Dockerfile").read_text(encoding="utf-8").startswith(f"FROM {tag}\n")


def test_ensure_deepagent_agent_local_missing_env_fails(tmp_path: Path) -> None:
    pack = tmp_path / "empty"
    pack.mkdir()
    with pytest.raises(HarborDockerError, match="environment"):
        ensure_deepagent_agent_local(pack, work_dir=tmp_path / "w", stage_only=True)


def test_product_itemadapter_and_click_get_distinct_tags() -> None:
    """Live product packs with unique env Dockerfiles mint distinct tags."""
    root = Path("datasets/deepagent_v1/tasks")
    item_p = root / "realpr-itemadapter-101"
    click_p = root / "realpr-click-3645"
    if not item_p.is_dir() or not click_p.is_dir():
        pytest.skip("product packs not present")
    dig_i = environment_content_digest(item_p)
    dig_c = environment_content_digest(click_p)
    assert dig_i != dig_c
    tag_i = pack_scoped_agent_image_tag(item_p)
    tag_c = pack_scoped_agent_image_tag(click_p)
    assert tag_i != tag_c
    assert tag_i.startswith("deepagent-agent-") and tag_i.endswith(":local")
    assert tag_c.startswith("deepagent-agent-") and tag_c.endswith(":local")


def test_invoke_pier_mini_swe_panel_public_api() -> None:
    """M15: invoke_pier_mini_swe_panel must be importable from swe_factory.panel."""
    from swe_factory.panel import invoke_pier_mini_swe_panel as from_panel
    from swe_factory.panel.pier_scaffold import (
        invoke_pier_mini_swe_panel as from_scaffold,
    )

    assert from_panel is from_scaffold
    # dry-run must not invent rewards
    inv = from_panel(
        pack_path="/nonexistent/pack",
        pack_id="demo",
        model="x-ai/grok-4.5",
        dry_run=True,
    )
    assert inv.ok is True
    assert inv.invented_reward is False
    assert inv.reward is None
    assert inv.mode == "dry-run"


def _export_pier_cert_pack(tmp_path: Path) -> Path:
    """Export a schema-valid pier-certifiable pack for fail-closed tests."""
    from swe_factory.harbor.export_pack import export_harbor_pack
    from swe_factory.harbor.schema import (
        MODEL_PATCH_ARTIFACT,
        HarborMetadata,
        HarborPackSpec,
        HarborTaskIdentity,
        HarborTaskToml,
        HarborVerifier,
        TestsConfig,
    )

    _REAL_URL = "https://github.com/pallets/click.git"
    _REAL_SHA = "a1b2c3d4e5f6789012345678901234567890abcd"
    task_id = "deepagent-pier-mintag-demo"
    spec = HarborPackSpec.model_validate(
        {
            "task_id": task_id,
            "instruction_md": "Fix multi-file bug via Pier-loadable pack.\n",
            "task_toml": HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name=f"swe-factory/{task_id}"),
                metadata=HarborMetadata(
                    language="python",
                    repository_url=_REAL_URL,
                    base_commit_hash=_REAL_SHA,
                    task_id=task_id,
                    source_track="real_pr",
                    license="BSD-3-Clause",
                ),
                verifier=HarborVerifier(environment_mode="separate", timeout_sec=120.0),
            ),
            "tests_config": TestsConfig(
                base_commit=_REAL_SHA,
                f2p_node_ids=["tests.test_math.test_add"],
                p2p_node_ids=["tests.test_ok.test_always_ok"],
            ),
            "solution_patch": (
                "diff --git a/src/a.py b/src/a.py\n"
                "--- a/src/a.py\n"
                "+++ b/src/a.py\n"
                "@@ -1 +1 @@\n"
                "-broken\n"
                "+fixed\n"
            ),
            "test_patch": (
                "diff --git a/tests/new.py b/tests/new.py\n"
                "--- /dev/null\n"
                "+++ b/tests/new.py\n"
                "@@ -0,0 +1,2 @@\n"
                "+def test_add():\n"
                "+    assert True\n"
            ),
            "environment_dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n",
            "tests_dockerfile": "FROM deepagent-agent:local\nCOPY test.sh /tests/test.sh\n",
        }
    )
    pack = export_harbor_pack(spec, dest=tmp_path / "tasks" / task_id)
    return pack.pack_dir


def test_pier_cert_fail_closed_on_mintag_prepare_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mintag prepare failure must reject cert (not a soft note only)."""
    from swe_factory.harbor import pier_cert as pier_mod
    from swe_factory.harbor.harbor_docker import HarborDockerError

    pack = _export_pier_cert_pack(tmp_path)
    jobs = Path("/tmp/harbor-deepagent-jobs-mintag-fail-test")
    jobs.mkdir(parents=True, exist_ok=True)

    def boom(*_a: Any, **_k: Any) -> str:
        raise HarborDockerError("simulated mintag build failure")

    monkeypatch.setattr(
        "swe_factory.harbor.harbor_docker.ensure_deepagent_agent_local",
        boom,
    )
    # isinstance(active_runner, SubprocessPierRunner) gates mintag prep.
    runner = pier_mod.SubprocessPierRunner(pier_bin="/tmp/does-not-exist-pier-xyz")
    # ensure raises before any real pier binary is needed
    result = certify_pier_pack(
        pack,
        runner=runner,
        jobs_root=jobs,
        run_oracle=True,
        run_null=True,
        run_load_smoke=False,
        force_build=False,
        timeout_sec=5.0,
    )
    assert result.certified is False
    assert result.disposition == "reject"
    assert "PIER_AGENT_MINTAG_FAIL" in result.reason_codes
    assert result.oracle_run is None
    assert result.null_run is None
    assert any("fail-closed" in r for r in result.reasons)
    # Ensure scripted path still works without mintag block
    scripted = ScriptedPierRunner(oracle_reward=1, null_reward=0)
    ok = certify_pier_pack(
        pack,
        runner=scripted,
        jobs_root=jobs,
        run_oracle=True,
        run_null=True,
        run_load_smoke=False,
    )
    assert ok.certified is True
    assert "PIER_AGENT_MINTAG_FAIL" not in ok.reason_codes
