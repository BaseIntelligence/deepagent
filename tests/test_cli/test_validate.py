"""Tests for the validate CLI command."""

import json

import pytest
from typer.testing import CliRunner

from swe_forge.cli.validate import app, validate_jsonl_file


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def valid_task_data():
    return {
        "id": "task-1",
        "repo": "owner/repo",
        "base_commit": "abc123def456",
        "merge_commit": "def456abc123",
        "language": "python",
        "difficulty_score": 5,
        "patch": "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old\n+new\n",
    }


@pytest.fixture
def invalid_task_data():
    return {
        "id": "task-2",
        "repo": "",
        "base_commit": "",
        "merge_commit": "",
    }


@pytest.fixture
def valid_jsonl_file(tmp_path, valid_task_data):
    file_path = tmp_path / "valid_tasks.jsonl"
    with file_path.open("w") as f:
        f.write(json.dumps(valid_task_data) + "\n")
    return file_path


@pytest.fixture
def invalid_jsonl_file(tmp_path, invalid_task_data):
    file_path = tmp_path / "invalid_tasks.jsonl"
    with file_path.open("w") as f:
        f.write(json.dumps(invalid_task_data) + "\n")
    return file_path


@pytest.fixture
def mixed_jsonl_file(tmp_path, valid_task_data, invalid_task_data):
    file_path = tmp_path / "mixed_tasks.jsonl"
    with file_path.open("w") as f:
        f.write(json.dumps(valid_task_data) + "\n")
        f.write(json.dumps(invalid_task_data) + "\n")
    return file_path


def test_validate_command_valid_file(runner, valid_jsonl_file):
    result = runner.invoke(app, ["--input", str(valid_jsonl_file)])
    assert result.exit_code == 0
    assert "Valid" in result.output
    assert "1" in result.output


def test_validate_command_invalid_file(runner, invalid_jsonl_file):
    result = runner.invoke(app, ["--input", str(invalid_jsonl_file)])
    assert result.exit_code == 1
    assert "Invalid" in result.output


def test_validate_command_mixed_file(runner, mixed_jsonl_file):
    result = runner.invoke(app, ["--input", str(mixed_jsonl_file)])
    assert result.exit_code == 1
    assert "│ Valid" in result.output
    assert "│ Invalid" in result.output


def test_validate_command_with_output(runner, valid_jsonl_file, tmp_path):
    output_path = tmp_path / "output.jsonl"
    result = runner.invoke(
        app, ["--input", str(valid_jsonl_file), "--output", str(output_path)]
    )
    assert result.exit_code == 0
    assert output_path.exists()
    content = output_path.read_text()
    assert "task-1" in content


def test_validate_command_file_not_found(runner):
    result = runner.invoke(app, ["--input", "/nonexistent/file.jsonl"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_validate_command_verbose(runner, invalid_jsonl_file):
    result = runner.invoke(app, ["--input", str(invalid_jsonl_file), "--verbose"])
    assert result.exit_code == 1
    assert "Detailed Errors" in result.output


def test_validate_jsonl_file_valid(valid_jsonl_file):
    valid_tasks, results = validate_jsonl_file(valid_jsonl_file)
    assert len(valid_tasks) == 1
    assert len(results) == 1
    assert results[0].valid is True


def test_validate_jsonl_file_invalid(invalid_jsonl_file):
    valid_tasks, results = validate_jsonl_file(invalid_jsonl_file)
    assert len(valid_tasks) == 0
    assert len(results) == 1
    assert results[0].valid is False
    assert len(results[0].errors) > 0


def test_validate_jsonl_file_mixed(mixed_jsonl_file):
    valid_tasks, results = validate_jsonl_file(mixed_jsonl_file)
    assert len(valid_tasks) == 1
    assert len(results) == 2
    assert sum(1 for r in results if r.valid) == 1
    assert sum(1 for r in results if not r.valid) == 1


def test_validate_jsonl_file_with_fix(tmp_path, invalid_task_data):
    fixable_task = {
        "id": "fixable-task",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "merge_commit": "def456",
    }
    file_path = tmp_path / "fixable.jsonl"
    with file_path.open("w") as f:
        f.write(json.dumps(fixable_task) + "\n")

    valid_tasks, results = validate_jsonl_file(file_path, fix=True)
    assert len(valid_tasks) == 1
    assert results[0].valid is True


def test_validate_empty_jsonl_file(tmp_path):
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("")
    valid_tasks, results = validate_jsonl_file(file_path)
    assert len(valid_tasks) == 0
    assert len(results) == 0


def test_validate_malformed_json(tmp_path):
    file_path = tmp_path / "malformed.jsonl"
    file_path.write_text("{invalid json\n")
    valid_tasks, results = validate_jsonl_file(file_path)
    assert len(valid_tasks) == 0
    assert len(results) == 1
    assert results[0].valid is False
    assert "Invalid JSON" in results[0].errors[0]


def test_validate_patch_format_valid():
    from swe_forge.cli.validate import validate_patch_format

    valid_patch = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,3 @@\n-old\n+new\n"
    is_valid, errors = validate_patch_format(valid_patch)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_patch_format_invalid():
    from swe_forge.cli.validate import validate_patch_format

    invalid_patch = "not a valid patch"
    is_valid, errors = validate_patch_format(invalid_patch)
    assert is_valid is False
    assert len(errors) > 0


def test_validate_patch_empty():
    from swe_forge.cli.validate import validate_patch_format

    is_valid, errors = validate_patch_format("")
    assert is_valid is True
    assert len(errors) == 0


def test_validate_required_fields():
    from swe_forge.cli.validate import validate_required_fields

    valid_data = {
        "id": "test",
        "repo": "owner/repo",
        "base_commit": "abc",
        "merge_commit": "def",
    }
    is_valid, errors = validate_required_fields(valid_data)
    assert is_valid is True

    invalid_data = {"id": "test"}
    is_valid, errors = validate_required_fields(invalid_data)
    assert is_valid is False
    assert len(errors) > 0
