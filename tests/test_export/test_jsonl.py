
import pytest

from swe_forge.export.jsonl import export_jsonl, import_jsonl, stream_jsonl
from swe_forge.swe.models import SweTask, SweTaskStatus


@pytest.fixture
def sample_tasks():
    return [
        SweTask(
            id="task-1",
            repo="owner/repo",
            base_commit="abc123",
            merge_commit="def456",
            language="python",
            difficulty_score=5,
            prompt="Fix the bug",
        ),
        SweTask(
            id="task-2",
            repo="owner/repo2",
            base_commit="111111",
            merge_commit="222222",
            language="javascript",
            difficulty_score=3,
            status=SweTaskStatus.READY,
        ),
    ]


def test_roundtrip(sample_tasks, tmp_path):
    file_path = tmp_path / "tasks.jsonl"
    export_jsonl(sample_tasks, file_path)
    imported = import_jsonl(file_path)
    assert len(imported) == 2
    assert imported[0].id == "task-1"
    assert imported[0].repo == "owner/repo"
    assert imported[1].id == "task-2"
    assert imported[1].status == SweTaskStatus.READY


def test_append_mode(sample_tasks, tmp_path):
    file_path = tmp_path / "tasks.jsonl"
    export_jsonl([sample_tasks[0]], file_path)
    export_jsonl([sample_tasks[1]], file_path, append=True)
    imported = import_jsonl(file_path)
    assert len(imported) == 2


def test_stream_jsonl(sample_tasks, tmp_path):
    file_path = tmp_path / "tasks.jsonl"
    export_jsonl(sample_tasks, file_path)
    tasks_gen = stream_jsonl(file_path)
    first = next(tasks_gen)
    assert first.id == "task-1"
    second = next(tasks_gen)
    assert second.id == "task-2"
    with pytest.raises(StopIteration):
        next(tasks_gen)


def test_empty_file(tmp_path):
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("")
    imported = import_jsonl(file_path)
    assert imported == []


def test_empty_file_with_blank_lines(tmp_path):
    file_path = tmp_path / "blank.jsonl"
    file_path.write_text("\n\n\n")
    imported = import_jsonl(file_path)
    assert imported == []


def test_invalid_json(tmp_path):
    file_path = tmp_path / "invalid.jsonl"
    file_path.write_text('{"id": "task-1"}\nnot valid json\n')
    with pytest.raises(ValueError, match="Invalid JSON"):
        import_jsonl(file_path)


def test_invalid_swe_task_data(tmp_path):
    file_path = tmp_path / "invalid_task.jsonl"
    file_path.write_text('{"id": "task-1", "difficulty_score": 999}\n')
    with pytest.raises(ValueError, match="Invalid SweTask data"):
        import_jsonl(file_path)


def test_export_creates_parent_dirs(tmp_path):
    nested_path = tmp_path / "nested" / "deep" / "tasks.jsonl"
    task = SweTask(
        id="task-1", repo="owner/repo", base_commit="abc", merge_commit="def"
    )
    export_jsonl([task], nested_path)
    assert nested_path.exists()


def test_overwrite_mode(sample_tasks, tmp_path):
    file_path = tmp_path / "tasks.jsonl"
    export_jsonl([sample_tasks[0]], file_path)
    export_jsonl([sample_tasks[1]], file_path, append=False)
    imported = import_jsonl(file_path)
    assert len(imported) == 1
    assert imported[0].id == "task-2"
