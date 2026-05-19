from pathlib import Path

from swe_forge.synthetic.feature_deletion import build_python_function_deletion
from swe_forge.synthetic.pipeline import create_feature_deletion_task


def test_build_python_function_deletion_generates_inverse_patch(tmp_path: Path):
    source = tmp_path / "pkg.py"
    source.write_text(
        "def add(a, b):\n    total = a + b\n    return total\n",
        encoding="utf-8",
    )

    result = build_python_function_deletion(tmp_path, source, "add")

    assert "NotImplementedError" in result.deletion_patch
    assert "-    total = a + b" in result.deletion_patch
    assert "+    total = a + b" in result.oracle_patch
    assert result.source_file == Path("pkg.py")


def test_create_feature_deletion_task_sets_synthetic_fields(tmp_path: Path):
    source = tmp_path / "pkg.py"
    source.write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    task = create_feature_deletion_task(
        repo_root=tmp_path,
        repo="owner/repo",
        base_commit="abc123",
        source_file="pkg.py",
        symbol="add",
        fail_to_pass=["pytest tests/test_pkg.py -v"],
    )

    assert task.source_type == "synthetic_feature_deletion"
    assert task.deletion_patch
    assert task.patch
    assert task.meta["strategy"] == "feature_deletion"
    assert task.fail_to_pass == ["pytest tests/test_pkg.py -v"]
