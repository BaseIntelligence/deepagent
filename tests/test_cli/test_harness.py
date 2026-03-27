"""Tests for swe_forge.cli.harness module."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


from swe_forge.cli.harness import app, _load_tasks, _output_results, _print_summary
from swe_forge.swe.models import SweTask

runner = CliRunner()


def create_task_jsonl(tasks: list[dict], path: Path) -> None:
    """Create a temporary JSONL file with tasks."""
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


class TestLoadTasks:
    def test_load_tasks_from_valid_jsonl(self, tmp_path: Path):
        jsonl_path = tmp_path / "tasks.jsonl"
        create_task_jsonl(
            [
                {
                    "id": "task-1",
                    "repo": "owner/repo",
                    "base_commit": "abc123",
                    "fail_to_pass": ["pytest test.py"],
                    "pass_to_pass": [],
                },
                {
                    "id": "task-2",
                    "repo": "owner/repo",
                    "base_commit": "def456",
                    "fail_to_pass": ["pytest test2.py"],
                    "pass_to_pass": [],
                },
            ],
            jsonl_path,
        )

        tasks = _load_tasks(jsonl_path)

        assert len(tasks) == 2
        assert tasks[0].id == "task-1"
        assert tasks[1].id == "task-2"

    def test_load_tasks_empty_file(self, tmp_path: Path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        tasks = _load_tasks(jsonl_path)

        assert tasks == []

    def test_load_tasks_skips_empty_lines(self, tmp_path: Path):
        jsonl_path = tmp_path / "tasks.jsonl"
        jsonl_path.write_text(
            '{"id": "task-1", "repo": "owner/repo", "base_commit": "abc123"}\n'
            "\n"
            '{"id": "task-2", "repo": "owner/repo", "base_commit": "def456"}\n'
        )

        tasks = _load_tasks(jsonl_path)

        assert len(tasks) == 2


class TestOutputResults:
    def test_output_results_writes_json(self, tmp_path: Path):
        output_path = tmp_path / "results.json"
        results = [
            {"task_id": "task-1", "status": "resolved"},
            {"task_id": "task-2", "status": "unresolved"},
        ]

        _output_results(results, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["task_id"] == "task-1"


class TestPrintSummary:
    def test_print_summary_counts_statuses(self, capsys):
        results = [
            {"status": "resolved"},
            {"status": "resolved"},
            {"status": "unresolved"},
            {"status": "agent_error"},
        ]

        _print_summary(results)

        captured = capsys.readouterr()
        assert "resolved" in captured.out
        assert "unresolved" in captured.out
        assert "agent_error" in captured.out

    def test_print_summary_handles_empty_results(self, capsys):
        _print_summary([])

        captured = capsys.readouterr()
        assert "Resolution rate:" in captured.out


class TestHarnessCommand:
    def test_harness_command_with_valid_input(self, tmp_path: Path):
        jsonl_path = tmp_path / "tasks.jsonl"
        output_path = tmp_path / "results.json"
        create_task_jsonl(
            [
                {
                    "id": "task-1",
                    "repo": "owner/repo",
                    "base_commit": "abc123",
                    "fail_to_pass": ["pytest test.py"],
                    "pass_to_pass": [],
                },
            ],
            jsonl_path,
        )

        result = runner.invoke(
            app,
            [
                "--input",
                str(jsonl_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()

    def test_harness_command_with_missing_input(self):
        result = runner.invoke(
            app,
            ["--input", "/nonexistent/file.jsonl"],
        )

        assert result.exit_code != 0

    def test_harness_command_with_agent_script(self, tmp_path: Path):
        jsonl_path = tmp_path / "tasks.jsonl"
        output_path = tmp_path / "results.json"
        agent_script = tmp_path / "agent.sh"
        agent_script.write_text("#!/bin/bash\necho 'agent ran'")

        create_task_jsonl(
            [
                {
                    "id": "task-1",
                    "repo": "owner/repo",
                    "base_commit": "abc123",
                    "fail_to_pass": [],
                    "pass_to_pass": [],
                },
            ],
            jsonl_path,
        )

        result = runner.invoke(
            app,
            [
                "--input",
                str(jsonl_path),
                "--agent-script",
                str(agent_script),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0

    def test_harness_command_custom_timeout(self, tmp_path: Path):
        jsonl_path = tmp_path / "tasks.jsonl"
        output_path = tmp_path / "results.json"
        create_task_jsonl(
            [{"id": "task-1", "repo": "owner/repo", "base_commit": "abc123"}],
            jsonl_path,
        )

        result = runner.invoke(
            app,
            [
                "--input",
                str(jsonl_path),
                "--timeout",
                "300",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0

    def test_harness_command_parallel_option(self, tmp_path: Path):
        jsonl_path = tmp_path / "tasks.jsonl"
        output_path = tmp_path / "results.json"
        create_task_jsonl(
            [
                {"id": "task-1", "repo": "owner/repo", "base_commit": "abc123"},
                {"id": "task-2", "repo": "owner/repo", "base_commit": "def456"},
            ],
            jsonl_path,
        )

        result = runner.invoke(
            app,
            [
                "--input",
                str(jsonl_path),
                "--parallel",
                "2",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        with open(output_path) as f:
            results = json.load(f)
        assert len(results) == 2

    def test_harness_command_empty_tasks_file(self, tmp_path: Path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        result = runner.invoke(
            app,
            ["--input", str(jsonl_path)],
        )

        assert result.exit_code == 0
        assert "No tasks to process" in result.output


class TestProcessSingleTask:
    @pytest.mark.asyncio
    async def test_process_single_task_returns_result_dict(self):
        from swe_forge.cli.harness import (
            HarnessConfig,
            HarnessRunner,
            _process_single_task,
        )

        task = SweTask(
            id="test-task",
            repo="owner/repo",
            base_commit="abc123",
            fail_to_pass=[],
            pass_to_pass=[],
        )
        config = HarnessConfig()
        runner = HarnessRunner(config=config)

        result = await _process_single_task(runner, task, config)

        assert "task_id" in result
        assert result["task_id"] == "test-task"
        assert "status" in result
        assert "resolved" in result


class TestCLIIntegration:
    def test_harness_cli_registers_help(self):
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Run SWE evaluation harness" in result.output

    def test_harness_cli_shows_options_in_help(self):
        result = runner.invoke(app, ["harness", "--help"])
        output = strip_ansi(result.output)

        assert "--input" in output
        assert "--agent-script" in output
        assert "--timeout" in output
        assert "--parallel" in output
        assert "--output" in output


class TestEdgeCases:
    def test_harness_handles_malformed_json(self, tmp_path: Path):
        jsonl_path = tmp_path / "bad.jsonl"
        jsonl_path.write_text('{"id": "task-1"\n')

        result = runner.invoke(app, ["--input", str(jsonl_path)])

        assert result.exit_code != 0

    def test_harness_handles_missing_required_fields(self, tmp_path: Path):
        jsonl_path = tmp_path / "incomplete.jsonl"
        jsonl_path.write_text('{"repo": "owner/repo"}\n')

        result = runner.invoke(app, ["--input", str(jsonl_path)])

        assert result.exit_code != 0
