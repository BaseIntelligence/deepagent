"""Separate-verifier Harbor oracle (VAL-HARBOR-003..006) — offline fakes + structural."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harbor.export_pack import export_harbor_pack
from swe_factory.harbor.harbor_docker import (
    HarborDockerError,
    assert_certified_test_patch,
    assert_certified_tests_config,
    build_agent_and_tests_images,
    list_agent_context_paths,
    scan_agent_context_forbidden,
    stage_agent_context,
    stage_tests_context,
    summarize_agent_context,
)
from swe_factory.harbor.harbor_oracle import (
    FakeHarborVerifier,
    HarborOracleError,
    run_harbor_oracle,
    run_offline_harbor_oracle_fixture,
)
from swe_factory.harbor.offline_fixture import (
    HARBOR_FIXTURE_TASK_ID,
    build_offline_harbor_spec,
    run_offline_harbor_fixture,
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

runner = CliRunner()


def _minimal_spec(**overrides: object) -> HarborPackSpec:
    base: dict[str, object] = {
        "task_id": "demo-harbor-oracle",
        "instruction_md": "Fix multi-file bug.\n",
        "task_toml": HarborTaskToml(
            schema_version="1.1",
            artifacts=[MODEL_PATCH_ARTIFACT],
            task=HarborTaskIdentity(name="swe-factory/demo-harbor-oracle"),
            metadata=HarborMetadata(
                language="python",
                repository_url="https://example.com/demo.git",
                base_commit_hash="abc123def456",
                task_id="demo-harbor-oracle",
            ),
            verifier=HarborVerifier(environment_mode="separate", timeout_sec=120.0),
        ),
        "tests_config": TestsConfig(
            base_commit="abc123def456",
            f2p_node_ids=["tests.test_math.test_add"],
            p2p_node_ids=["tests.test_ok.test_always_ok"],
        ),
        "solution_patch": (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        ),
        "test_patch": (
            "diff --git a/tests/new.py b/tests/new.py\n"
            "--- /dev/null\n"
            "+++ b/tests/new.py\n"
            "@@ -0,0 +1 @@\n"
            "+def test_x():\n"
            "+    assert True\n"
        ),
        "environment_dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n",
        "tests_dockerfile": "FROM demo-agent:local\nCOPY test.sh /tests/test.sh\n",
    }
    base.update(overrides)
    return HarborPackSpec.model_validate(base)


def test_agent_context_excludes_solution_and_test_patch(tmp_path: Path) -> None:
    """VAL-HARBOR-005: agent build context omits solution/ and tests/test.patch."""
    pack = export_harbor_pack(
        _minimal_spec(),
        dest=tmp_path / "pack",
        extra_environment_files={"repo/main.py": "print('hi')\n"},
    )
    # Plant a leak candidate inside environment copy to ensure ignore patterns work
    # (export itself does not put solution into environment/).
    ctx = stage_agent_context(pack.pack_dir, tmp_path / "agent_ctx")
    # Inject a red-team file via copying tests into context — should flag.
    assert (pack.pack_dir / "solution" / "solution.patch").is_file()
    assert (pack.pack_dir / "tests" / "test.patch").is_file()
    assert not (ctx.context_dir / "solution").exists()
    paths = list_agent_context_paths(ctx.context_dir)
    assert not any(p.endswith("test.patch") for p in paths)
    assert not any("solution" in p for p in paths)
    hits = scan_agent_context_forbidden(ctx.context_dir)
    assert hits == []
    summary = summarize_agent_context(ctx.context_dir)
    assert summary["isolated"] is True
    assert summary["has_test_patch"] is False
    assert summary["has_solution_dir"] is False


def test_scan_detects_leaked_test_patch(tmp_path: Path) -> None:
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (agent / "test.patch").write_text("leak\n", encoding="utf-8")
    hits = scan_agent_context_forbidden(agent)
    assert any("test.patch" in h for h in hits)


def test_stage_tests_context_has_held_out(tmp_path: Path) -> None:
    pack = export_harbor_pack(_minimal_spec(), dest=tmp_path / "pack")
    tests = stage_tests_context(pack.pack_dir, tmp_path / "tests_ctx")
    assert (tests.context_dir / "test.patch").read_text(encoding="utf-8").strip()
    assert (tests.context_dir / "config.json").is_file()
    assert (tests.context_dir / "grader.py").is_file()


def test_certified_config_and_test_patch_enforced(tmp_path: Path) -> None:
    """VAL-HARBOR-006: non-empty f2p_node_ids + non-empty test.patch."""
    pack = export_harbor_pack(_minimal_spec(), dest=tmp_path / "pack")
    cfg = assert_certified_tests_config(pack.pack_dir / "tests" / "config.json")
    assert cfg["f2p_node_ids"]
    body = assert_certified_test_patch(pack.pack_dir / "tests" / "test.patch")
    assert body.strip()

    empty_patch = tmp_path / "empty.patch"
    empty_patch.write_text("\n", encoding="utf-8")
    with pytest.raises(HarborDockerError, match="non-empty"):
        assert_certified_test_patch(empty_patch)

    bad_cfg = tmp_path / "bad.json"
    bad_cfg.write_text(
        json.dumps({"base_commit": "x", "f2p_node_ids": []}),
        encoding="utf-8",
    )
    with pytest.raises(HarborDockerError, match="f2p_node_ids"):
        assert_certified_tests_config(bad_cfg)


def test_build_contexts_stage_only_no_docker(tmp_path: Path) -> None:
    pack = export_harbor_pack(_minimal_spec(), dest=tmp_path / "pack")
    pair = build_agent_and_tests_images(
        pack.pack_dir,
        work_dir=tmp_path / "work",
        agent_tag="harbor-sdf-agent-unit:test",
        tests_tag="harbor-sdf-tests-unit:test",
        stage_only=True,
    )
    assert pair.agent_context.is_dir()
    assert pair.tests_context.is_dir()
    assert scan_agent_context_forbidden(pair.agent_context) == []
    # Dockerfile rewritten to agent tag
    df = (pair.tests_context / "Dockerfile").read_text(encoding="utf-8")
    assert "harbor-sdf-agent-unit:test" in df


def test_fake_oracle_solution_pass_null_fail(tmp_path: Path) -> None:
    """VAL-HARBOR-003/004 offline: solution reward=1, null reward=0."""
    result = run_offline_harbor_fixture(out_dir=tmp_path / "harbor_fixture")
    fake = FakeHarborVerifier(solution_reward=1, null_reward=0)
    oracle = run_harbor_oracle(
        result.pack_dir,
        backend=fake,
        task_id=result.task_id,
    )
    assert oracle.passed is True, oracle.reasons
    assert oracle.solution.reward == 1
    assert oracle.null.reward == 0
    assert oracle.agent_isolated is True
    assert oracle.config_ok is True
    assert oracle.test_patch_ok is True
    assert fake.cleaned is True
    assert oracle.mode == "fake"


def test_fake_oracle_rejects_when_null_resolves(tmp_path: Path) -> None:
    pack = run_offline_harbor_fixture(out_dir=tmp_path / "h").pack_dir
    fake = FakeHarborVerifier(solution_reward=1, null_reward=1)
    oracle = run_harbor_oracle(pack, backend=fake)
    assert oracle.passed is False
    assert any("null reward" in r for r in oracle.reasons)


def test_fake_oracle_rejects_when_solution_fails(tmp_path: Path) -> None:
    pack = run_offline_harbor_fixture(out_dir=tmp_path / "h").pack_dir
    fake = FakeHarborVerifier(solution_reward=0, null_reward=0)
    oracle = run_harbor_oracle(pack, backend=fake)
    assert oracle.passed is False
    assert any("solution reward" in r for r in oracle.reasons)


def test_offline_fixture_helper_writes_evidence(tmp_path: Path) -> None:
    """VAL-CROSS-006 path: pack + local oracle without live LLM."""
    pack, oracle = run_offline_harbor_oracle_fixture(out_dir=tmp_path / "bundle")
    assert pack.provider_calls == 0
    assert pack.task_id == HARBOR_FIXTURE_TASK_ID
    assert oracle.passed is True
    evidence = tmp_path / "bundle" / "oracle_evidence.json"
    assert evidence.is_file()
    data = json.loads(evidence.read_text(encoding="utf-8"))
    assert data["solution_reward"] == 1
    assert data["null_reward"] == 0
    assert data["agent_isolated"] is True


def test_offline_spec_f2p_nodes_junit_shape() -> None:
    """config.json node ids align with in-container junit classname.name form."""
    spec = build_offline_harbor_spec()
    for nid in spec.tests_config.f2p_node_ids:
        assert nid.startswith("tests.")
        assert "test_" in nid


def test_cli_harbor_oracle_fake(tmp_path: Path) -> None:
    """CLI harbor-oracle --from-fixture --backend fake."""
    out = tmp_path / "cli_out"
    result = runner.invoke(
        app,
        [
            "harbor-oracle",
            "--from-fixture",
            "--backend",
            "fake",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["passed"] is True
    assert payload["solution_reward"] == 1
    assert payload["null_reward"] == 0
    assert payload["agent_isolated"] is True
    assert payload["provider_calls"] == 0


def test_cli_harbor_oracle_help() -> None:
    result = runner.invoke(app, ["harbor-oracle", "--help"])
    assert result.exit_code == 0
    assert "from-fixture" in result.output.replace("_", "-")
    assert "fake" in result.output


def test_cli_help_lists_harbor_oracle() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "harbor-oracle" in result.output


def test_missing_pack_raises() -> None:
    with pytest.raises(HarborOracleError, match="not found"):
        run_harbor_oracle("/tmp/does-not-exist-harbor-pack-xyz")
