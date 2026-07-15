"""CLI help and documented command surface."""

from __future__ import annotations

from typer.testing import CliRunner

from swe_factory import __version__
from swe_factory.cli import app
from swe_factory.schema import TaskRecordStub

runner = CliRunner()


def test_help_lists_factory_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "build" in text
    assert "export" in text
    assert "score" in text
    assert "config" in text
    # High-level product wording for first-visit navigability
    assert "swe" in text or "factory" in text
    # M14 live-mine discoverability (VAL-LX-006)
    assert "real-pr-pool" in text
    assert "ship-deepagent" in text
    assert "live-mine" in text
    # M15 DeepAgent-grade eval discoverability (VAL-DEVAL-007)
    assert "eval-deepagent" in text


def test_build_export_score_help() -> None:
    for cmd in (
        "build",
        "export",
        "score",
        "config",
        "version",
        "envbuild",
        "offline-fixture",
        "synth",
        "pr-mine",
        "discover",
    ):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, (cmd, result.output)


def test_help_lists_pr_mine() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pr-mine" in result.output.lower() or "pr_mine" in result.output.lower()


def test_help_lists_discover() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "discover" in result.output.lower()


def test_help_lists_envbuild() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "envbuild" in result.output.lower()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_build_export_score_dry_run() -> None:
    for args in (
        ["build", "--dry-run"],
        ["export", "--dry-run"],
        ["score", "--dry-run", "--task-id", "demo"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.output)
        assert "dry-run" in result.output.lower()


def test_schema_stub_importable() -> None:
    record = TaskRecordStub(
        instance_id="demo-1",
        source_track="synthetic_grounded",
        repo="example/demo",
        base_commit="abc123",
        language="python",
        problem_statement="Fix the bug",
        fail_to_pass=["pytest tests/test_bug.py"],
        pass_to_pass=["pytest tests/test_ok.py"],
        gold_patch="diff --git a/x.py b/x.py\n",
        environment={"image_digest": "sha256:demo"},
        license="MIT",
    )
    assert record.instance_id == "demo-1"
