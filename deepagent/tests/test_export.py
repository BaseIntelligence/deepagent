"""Export workspace, tasks.jsonl writer, and leak scanner (VAL-EXPORT-001/002/003, VAL-HARNESS-003).

Offline tests first. Agent workspaces must omit gold; leak scan must stay clean
on fixture export and catch planted gold/API keys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.export.jsonl import write_tasks_jsonl
from swe_factory.export.leak_scan import LeakScanResult, scan_export_tree
from swe_factory.export.workspace import (
    ExportBundle,
    export_task_workspace,
    write_export_bundle,
)
from swe_factory.fixture.offline import (
    FIXTURE_INSTANCE_ID,
    build_fixture_task,
    default_fixture_root,
)
from swe_factory.schema import EnvironmentMeta, SourceTrack, TaskRecord

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample_task(**overrides: object) -> TaskRecord:
    base = {
        "instance_id": "export__sample__1",
        "source_track": SourceTrack.SYNTHETIC_GROUNDED,
        "repo": "fixtures/sample",
        "base_commit": "abc123def456",
        "language": "python",
        "problem_statement": "Restore correct multi-file behaviour.",
        "fail_to_pass": ["python -m pytest tests/test_math.py -q"],
        "pass_to_pass": ["python -m pytest tests/test_ok.py -q"],
        "gold_patch": (
            "diff --git a/demo_pkg/math_ops.py b/demo_pkg/math_ops.py\n"
            "--- a/demo_pkg/math_ops.py\n"
            "+++ b/demo_pkg/math_ops.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def add(a: int, b: int) -> int:\n"
            "-    return a - b\n"
            "+    return a + b\n"
            "\n"
            "diff --git a/demo_pkg/text_ops.py b/demo_pkg/text_ops.py\n"
            "--- a/demo_pkg/text_ops.py\n"
            "+++ b/demo_pkg/text_ops.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def reverse_words(text: str) -> str:\n"
            "-    return text\n"
            '+    return " ".join(reversed(text.split()))\n'
        ),
        "environment": EnvironmentMeta(image_digest="sha256:export_fixture_v1"),
        "license": "MIT",
        "gate_proof": {"mode": "stub_offline", "passed": True},
        "panel": {
            "grok_4_5": 0.25,
            "opus_4_8": 0.0,
            "pass_at_k": 0.125,
            "discrimination": 1.0,
        },
    }
    base.update(overrides)
    return TaskRecord.model_validate(base)


def test_export_workspace_omits_gold(tmp_path: Path) -> None:
    """VAL-EXPORT-001: agent workspace must not contain gold answer key."""
    task = _sample_task()
    fixture_repo = default_fixture_root() / "repo"
    workspace = export_task_workspace(
        task,
        dest=tmp_path / "tasks" / task.instance_id,
        broken_repo=fixture_repo,
    )

    assert workspace.is_dir()
    assert (workspace / "problem_statement.md").is_file()
    assert (workspace / "task_meta.agent.json").is_file()
    assert (workspace / "repo").is_dir()

    # Forbidden gold answer-key locations
    assert not (workspace / "gold.patch").exists()
    assert not (workspace / "patch.diff").exists()
    assert not (workspace / "solution.patch").exists()
    assert not list(workspace.rglob("gold.patch"))
    assert not list(workspace.rglob("patch.diff"))

    agent_meta = json.loads((workspace / "task_meta.agent.json").read_text(encoding="utf-8"))
    assert "gold_patch" not in agent_meta
    assert agent_meta.get("instance_id") == task.instance_id
    assert agent_meta.get("environment", {}).get("image_digest") == task.environment.image_digest

    # Gold body must not appear in agent-visible prompt/meta (answer key)
    gold_line = "+    return a + b"
    problem = (workspace / "problem_statement.md").read_text(encoding="utf-8")
    assert gold_line not in problem
    assert task.gold_patch not in problem
    assert gold_line not in (workspace / "task_meta.agent.json").read_text(encoding="utf-8")


def test_write_tasks_jsonl_schema_complete(tmp_path: Path) -> None:
    """VAL-EXPORT-002: tasks.jsonl lines parse with required fields for keeps."""
    task = _sample_task()
    path = write_tasks_jsonl([task], tmp_path / "tasks.jsonl")
    assert path.is_file()

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = TaskRecord.model_validate_json(lines[0])
    raw = json.loads(lines[0])

    for key in (
        "instance_id",
        "source_track",
        "repo",
        "base_commit",
        "language",
        "fail_to_pass",
        "pass_to_pass",
        "environment",
        "panel",
    ):
        assert key in raw, f"missing required export field {key}"

    assert record.environment.image_digest.startswith("sha256:")
    assert record.panel is not None
    assert record.panel.pass_at_k is not None
    assert 0.0 < float(record.panel.pass_at_k) <= 0.5
    assert record.fail_to_pass
    # Internal jsonl retains gold_patch for harness; agent trees never get it.
    assert record.gold_patch.strip()


def test_write_tasks_jsonl_multiple_records(tmp_path: Path) -> None:
    tasks = [
        _sample_task(instance_id="export__a"),
        _sample_task(instance_id="export__b", source_track=SourceTrack.REAL_PR),
    ]
    path = write_tasks_jsonl(tasks, tmp_path / "tasks.jsonl")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    ids = {TaskRecord.model_validate_json(ln).instance_id for ln in lines}
    assert ids == {"export__a", "export__b"}


def test_leak_scanner_clean_on_fixture_export(tmp_path: Path) -> None:
    """VAL-EXPORT-003: leak scanner clean on fixture export; VAL-HARNESS-003 no secrets."""
    task = build_fixture_task()
    bundle = write_export_bundle(
        tasks=[task],
        out_dir=tmp_path / "export",
        broken_repos={task.instance_id: default_fixture_root() / "repo"},
    )
    assert isinstance(bundle, ExportBundle)
    result = scan_export_tree(bundle.out_dir)
    assert isinstance(result, LeakScanResult)
    assert result.clean is True
    assert list(result.findings) == []
    assert result.files_scanned > 0


def test_leak_scanner_catches_planted_gold_and_key(tmp_path: Path) -> None:
    task = _sample_task()
    workspace = export_task_workspace(task, dest=tmp_path / "bad_ws")
    # Plant gold answer key + a fake API key string
    (workspace / "gold.patch").write_text(task.gold_patch, encoding="utf-8")
    (workspace / "notes.txt").write_text(
        "OPENROUTER_API_KEY=*********************************************\n",
        encoding="utf-8",
    )

    result = scan_export_tree(tmp_path)
    assert result.clean is False
    joined = " ".join(result.findings).lower()
    assert "gold" in joined or "patch" in joined or "diff" in joined
    assert "api" in joined or "key" in joined or "secret" in joined


def test_leak_scanner_catches_gho_and_bearer_tokens() -> None:
    """VAL-LX-005: gho_/Bearer raw must be detected (never in ship tree)."""
    from swe_factory.export.leak_scan import scan_text_for_secrets

    hits_gho = scan_text_for_secrets(
        "token=gho_SUPER_SECRET_TOKEN_VALUE_NEVER_LOG_ABCDEF",
        rel="evidence/bad.json",
    )
    assert hits_gho, "expected gho_ token finding"
    hits_bearer = scan_text_for_secrets(
        "Authorization: Bearer gho_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
        rel="evidence/headers.txt",
    )
    assert hits_bearer, "expected Bearer finding"
    hits_or = scan_text_for_secrets(
        "OPENROUTER_API_KEY=sk-or-v1-" + ("a" * 24),
        rel="report.md",
    )
    assert hits_or, "expected OPENROUTER raw key finding"
    # Documented env var names without values are fine; raw assignment is not.
    clean = scan_text_for_secrets(
        "Export GITHUB_TOKEN via gh auth token; never commit secrets.",
        rel="README.md",
    )
    assert clean == []


def test_write_export_bundle_layout(tmp_path: Path) -> None:
    task = build_fixture_task()
    out_dir = tmp_path / "v1"
    bundle = write_export_bundle(
        tasks=[task],
        out_dir=out_dir,
        broken_repos={task.instance_id: default_fixture_root() / "repo"},
    )
    assert bundle.tasks_jsonl.is_file()
    assert (bundle.tasks_jsonl.parent / "tasks").is_dir()
    assert (out_dir / "tasks" / task.instance_id / "problem_statement.md").is_file()
    assert not (out_dir / "tasks" / task.instance_id / "gold.patch").exists()

    # Leak scan drive-by proof
    scan = scan_export_tree(out_dir)
    assert scan.clean


def test_cli_export_fixture_path(tmp_path: Path) -> None:
    """CLI export --from-fixture writes agent workspaces + tasks.jsonl offline."""
    out_dir = tmp_path / "cli_export"
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)

    result = runner.invoke(
        app,
        ["export", "--from-fixture", "--out", str(out_dir)],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "tasks.jsonl").is_file()
    record = TaskRecord.model_validate_json(
        (out_dir / "tasks.jsonl").read_text(encoding="utf-8").strip().splitlines()[0]
    )
    assert record.instance_id == FIXTURE_INSTANCE_ID
    workspace = out_dir / "tasks" / FIXTURE_INSTANCE_ID
    assert workspace.is_dir()
    assert not (workspace / "gold.patch").exists()
    # Scan report or at least clean exit mentions leak/export success
    assert "export" in result.output.lower() or "ok" in result.output.lower()


def test_cli_export_dry_run_still_works() -> None:
    result = runner.invoke(app, ["export", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.output.lower()


def test_leak_scan_rejects_unified_diff_in_agent_meta(tmp_path: Path) -> None:
    ws = tmp_path / "tasks" / "leaky"
    ws.mkdir(parents=True)
    (ws / "problem_statement.md").write_text("fix me\n", encoding="utf-8")
    (ws / "hint.md").write_text(
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x\n+y\n",
        encoding="utf-8",
    )
    result = scan_export_tree(tmp_path)
    assert result.clean is False
