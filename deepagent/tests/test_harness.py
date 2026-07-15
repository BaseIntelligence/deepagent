"""Harness scoring: gold resolves, null does not (VAL-HARNESS-001).

Unit tests use FakeOracleRunner (offline). Docker fixture path is marked
integration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.harness.score import (
    HarnessScoreResult,
    score_candidate,
    score_gold_and_null,
)
from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite
from swe_factory.schema import EnvironmentMeta, SourceTrack, TaskRecord

runner = CliRunner()


def _task() -> TaskRecord:
    return TaskRecord.model_validate(
        {
            "instance_id": "harness__sample__1",
            "source_track": SourceTrack.SYNTHETIC_GROUNDED,
            "repo": "fixtures/sample",
            "base_commit": "abc123",
            "language": "python",
            "problem_statement": "Fix multi-file bugs",
            "fail_to_pass": ["pytest tests/test_math.py"],
            "pass_to_pass": ["pytest tests/test_ok.py"],
            "gold_patch": (
                "diff --git a/a.py b/a.py\n+++ b/a.py\ndiff --git a/b.py b/b.py\n+++ b/b.py\n"
            ),
            "environment": EnvironmentMeta(image_digest="sha256:harness"),
            "license": "MIT",
        }
    )


def test_score_gold_resolves_true_null_false_offline(tmp_path: Path) -> None:
    """VAL-HARNESS-001 offline: gold resolve=true, null resolve=false."""
    task = _task()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "dummy.py").write_text("x=1\n", encoding="utf-8")

    fake = FakeOracleRunner(
        broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        gold_runs=[ScriptedSuite(f2p_exits=[0], p2p_exits=[0])],
        null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
    )
    gold = score_candidate(
        task=task,
        workspace=workspace,
        patch=task.gold_patch,
        runner=fake,
        label="gold",
    )
    null = score_candidate(
        task=task,
        workspace=workspace,
        patch="",
        runner=fake,
        label="null",
    )
    assert isinstance(gold, HarnessScoreResult)
    assert gold.resolve is True
    assert gold.score == 1.0
    assert null.resolve is False
    assert null.score == 0.0


def test_score_gold_and_null_helper(tmp_path: Path) -> None:
    task = _task()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    fake = FakeOracleRunner(
        broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        gold_runs=[ScriptedSuite(f2p_exits=[0], p2p_exits=[0])],
        null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
    )
    pair = score_gold_and_null(task=task, workspace=workspace, runner=fake)
    assert pair.gold.resolve is True
    assert pair.null.resolve is False
    assert pair.passed is True  # gold true and null false
    assert fake.cleaned is True


def test_score_null_that_accidentally_resolves_fails_pair(tmp_path: Path) -> None:
    task = _task()
    workspace = tmp_path / "repo"
    workspace.mkdir()
    # Broken F2P already green → null would resolve (invalid task for harness)
    fake = FakeOracleRunner(
        broken=ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
        gold_runs=[ScriptedSuite(f2p_exits=[0], p2p_exits=[0])],
        null=ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
    )
    pair = score_gold_and_null(task=task, workspace=workspace, runner=fake)
    assert pair.gold.resolve is True
    assert pair.null.resolve is True
    assert pair.passed is False


def test_cli_score_fixture_dry_path_offline(tmp_path: Path) -> None:
    """score --from-fixture --backend fake proves CLI wiring offline."""
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    result = runner.invoke(
        app,
        [
            "score",
            "--from-fixture",
            "--backend",
            "fake",
            "--json",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output
    # Pretty-printed JSON from --json
    text = result.output.strip()
    start = text.find("{")
    assert start >= 0, result.output
    payload = json.loads(text[start:])
    assert payload["gold"]["resolve"] is True
    assert payload["null"]["resolve"] is False
    assert payload["passed"] is True
