"""Harbor/DeepSWE pack schema + offline export (VAL-HARBOR-001/002, VAL-CROSS-006)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harbor.export_pack import (
    REQUIRED_PACK_RELPATHS,
    HarborExportError,
    export_harbor_pack,
    verify_pack_tree,
    write_harbor_export,
)
from swe_factory.harbor.offline_fixture import (
    HARBOR_FIXTURE_TASK_ID,
    build_offline_harbor_spec,
    run_offline_harbor_fixture,
)
from swe_factory.harbor.pre_artifacts import render_pre_artifacts_sh
from swe_factory.harbor.schema import (
    MODEL_PATCH_ARTIFACT,
    HarborMetadata,
    HarborPackSpec,
    HarborTaskIdentity,
    HarborTaskToml,
    HarborVerifier,
    TestsConfig,
    render_task_toml,
    validate_pack_spec,
)

runner = CliRunner()


def _minimal_spec(**overrides: object) -> HarborPackSpec:
    base = {
        "task_id": "demo-harbor-task",
        "instruction_md": "Fix the multi-file bug.\n\nIMPORTANT: commit on a branch.\n",
        "task_toml": HarborTaskToml(
            schema_version="1.1",
            artifacts=[MODEL_PATCH_ARTIFACT],
            task=HarborTaskIdentity(name="swe-factory/demo-harbor-task"),
            metadata=HarborMetadata(
                language="python",
                repository_url="https://example.com/demo.git",
                base_commit_hash="abc123def456",
                task_id="demo-harbor-task",
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


def test_task_toml_requires_schema_and_separate_mode() -> None:
    """VAL-HARBOR-002: schema_version + separate verifier mode + required metadata."""
    task = HarborTaskToml(
        schema_version="1.1",
        artifacts=[MODEL_PATCH_ARTIFACT],
        task=HarborTaskIdentity(name="org/name"),
        metadata=HarborMetadata(
            language="python",
            repository_url="https://github.com/example/repo",
            base_commit_hash="deadbeef",
        ),
        verifier=HarborVerifier(environment_mode="separate"),
    )
    task.ensure_required()
    text = render_task_toml(task)
    parsed = tomllib.loads(text)
    assert parsed["schema_version"] == "1.1"
    assert MODEL_PATCH_ARTIFACT in parsed["artifacts"]
    assert parsed["metadata"]["language"] == "python"
    assert parsed["metadata"]["repository_url"].startswith("https://")
    assert parsed["metadata"]["base_commit_hash"] == "deadbeef"
    assert parsed["verifier"]["environment_mode"] == "separate"
    assert parsed["verifier"]["timeout_sec"] > 0
    assert parsed["agent"]["timeout_sec"] > 0


def test_task_toml_rejects_shared_mode_for_deepswe() -> None:
    task = HarborTaskToml(
        schema_version="1.1",
        artifacts=[MODEL_PATCH_ARTIFACT],
        task=HarborTaskIdentity(name="org/name"),
        metadata=HarborMetadata(
            language="go",
            repository_url="https://github.com/example/go",
            base_commit_hash="abc",
        ),
        verifier=HarborVerifier(environment_mode="shared"),
    )
    with pytest.raises(ValueError, match="separate"):
        task.ensure_required()


def test_task_toml_requires_model_patch_artifact() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HarborTaskToml(
            schema_version="1.1",
            artifacts=["/tmp/other"],
            task=HarborTaskIdentity(name="org/name"),
            metadata=HarborMetadata(
                language="python",
                repository_url="https://example.com/r",
                base_commit_hash="x",
            ),
        )


def test_tests_config_requires_f2p() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TestsConfig(base_commit="abc", f2p_node_ids=[])


def test_validate_pack_spec_commit_match() -> None:
    spec = _minimal_spec()
    ok = validate_pack_spec(spec)
    assert ok.task_id == "demo-harbor-task"

    bad = _minimal_spec(
        tests_config=TestsConfig(
            base_commit="other",
            f2p_node_ids=["t1"],
        )
    )
    with pytest.raises(ValueError, match="base_commit"):
        validate_pack_spec(bad)


def test_export_pack_tree_complete(tmp_path: Path) -> None:
    """VAL-HARBOR-001: every required path exists."""
    pack = export_harbor_pack(_minimal_spec(), dest=tmp_path / "demo-harbor-task")
    assert pack.pack_dir.is_dir()
    missing = verify_pack_tree(pack.pack_dir)
    assert missing == []
    for rel in REQUIRED_PACK_RELPATHS:
        assert (pack.pack_dir / rel).is_file(), rel

    # task.toml parseable
    text = (pack.pack_dir / "task.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    assert data["verifier"]["environment_mode"] == "separate"
    assert data["metadata"]["base_commit_hash"] == "abc123def456"

    cfg = json.loads((pack.pack_dir / "tests" / "config.json").read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"]
    assert cfg["base_commit"] == "abc123def456"
    assert (pack.pack_dir / "tests" / "test.patch").read_text(encoding="utf-8").strip()
    assert (pack.pack_dir / "solution" / "solution.patch").read_text(encoding="utf-8").strip()
    pre = (pack.pack_dir / "pre_artifacts.sh").read_text(encoding="utf-8")
    # Preferred configured SHA may still appear as preference; capture remains
    # robust via dynamic root fallback when placeholder is not in image history.
    assert "abc123def456" in pre
    assert "model.patch" in pre
    assert "rev-list --max-parents=0" in pre


def test_pre_artifacts_prefers_configured_and_falls_back_to_root() -> None:
    body = render_pre_artifacts_sh("cabba9e")
    assert "cabba9e" in body
    assert "/logs/artifacts/model.patch" in body
    assert "rev-list --max-parents=0" in body
    assert "resolve_base_ref" in body
    # Empty configured SHA still renders (dynamic root handles it).
    empty = render_pre_artifacts_sh("  ")
    assert "model.patch" in empty
    assert "rev-list --max-parents=0" in empty


def test_write_harbor_export_manifest(tmp_path: Path) -> None:
    bundle = write_harbor_export([_minimal_spec()], tmp_path / "harbor_out")
    assert bundle.pack_manifest.is_file()
    man = json.loads(bundle.pack_manifest.read_text(encoding="utf-8"))
    assert man["count"] == 1
    assert man["task_ids"] == ["demo-harbor-task"]
    assert (tmp_path / "harbor_out" / "tasks" / "demo-harbor-task" / "task.toml").is_file()


def test_write_harbor_export_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(HarborExportError, match="empty"):
        write_harbor_export([], tmp_path)


def test_offline_fixture_pack_complete(tmp_path: Path) -> None:
    """VAL-CROSS-006: offline fixture yields complete Harbor pack tree."""
    result = run_offline_harbor_fixture(out_dir=tmp_path / "harbor_fixture")
    assert result.provider_calls == 0
    assert result.task_id == HARBOR_FIXTURE_TASK_ID
    assert result.missing == ()
    missing = verify_pack_tree(result.pack_dir)
    assert missing == []

    data = tomllib.loads((result.pack_dir / "task.toml").read_text(encoding="utf-8"))
    assert data["schema_version"].startswith("1.")
    assert data["verifier"]["environment_mode"] == "separate"
    assert data["metadata"]["language"]
    assert data["metadata"]["repository_url"]
    assert data["metadata"]["base_commit_hash"]
    assert MODEL_PATCH_ARTIFACT in data["artifacts"]

    cfg = json.loads((result.pack_dir / "tests" / "config.json").read_text(encoding="utf-8"))
    assert len(cfg["f2p_node_ids"]) >= 1
    assert (result.pack_dir / "environment" / "repo").is_dir()
    # Agent context does not embed solution at environment/repo level
    assert not (result.pack_dir / "environment" / "repo" / "solution.patch").exists()
    assert not (result.pack_dir / "environment" / "repo" / "gold.patch").exists()


def test_offline_spec_validates() -> None:
    spec = build_offline_harbor_spec()
    assert spec.task_id == HARBOR_FIXTURE_TASK_ID
    assert "math" in " ".join(spec.tests_config.f2p_node_ids) or spec.tests_config.f2p_node_ids


def test_cli_export_harbor_from_fixture(tmp_path: Path) -> None:
    """CLI export-harbor works offline."""
    out = tmp_path / "cli_harbor"
    result = runner.invoke(
        app,
        ["export-harbor", "--from-fixture", "--out", str(out), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["complete"] is True
    assert payload["provider_calls"] == 0
    assert Path(payload["pack_dir"]).is_dir()
    assert verify_pack_tree(payload["pack_dir"]) == []


def test_cli_export_harbor_help() -> None:
    result = runner.invoke(app, ["export-harbor", "--help"])
    assert result.exit_code == 0
    assert "from-fixture" in result.output.replace("_", "-")


def test_cli_help_lists_export_harbor() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "export-harbor" in result.output


def test_v1_export_still_works(tmp_path: Path) -> None:
    """Keep V1 export intact alongside Harbor export."""
    out = tmp_path / "v1_export"
    result = runner.invoke(
        app,
        ["export", "--from-fixture", "--out", str(out), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert (out / "tasks.jsonl").is_file()
