from pathlib import Path

from swe_forge.synthetic.sanitizer import sanitize_tree


def test_sanitize_tree_removes_common_artifacts(tmp_path: Path):
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    pyc = cache_dir / "mod.cpython-312.pyc"
    pyc.write_bytes(b"cache")
    log = tmp_path / "run.log"
    log.write_text("log", encoding="utf-8")

    result = sanitize_tree(tmp_path)

    assert cache_dir in result.removed_paths
    assert log in result.removed_paths
    assert not cache_dir.exists()
    assert not log.exists()


def test_sanitize_tree_dry_run_does_not_delete(tmp_path: Path):
    dist = tmp_path / "dist"
    dist.mkdir()

    result = sanitize_tree(tmp_path, dry_run=True)

    assert dist in result.removed_paths
    assert dist.exists()
