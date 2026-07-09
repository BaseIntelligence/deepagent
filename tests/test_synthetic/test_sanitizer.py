from pathlib import Path

import pytest

from swe_forge.artifacts import (
    GENERATED_ARTIFACT_GITIGNORE_PATTERNS,
    generated_artifact_gitignore_patterns,
    is_generated_artifact,
)
from swe_forge.synthetic.sanitizer import is_leaky_artifact, sanitize_tree


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


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/__pycache__/module.cpython-312.pyc", True),
        ("pkg.egg-info/PKG-INFO", True),
        ("node_modules/pkg/index.js", True),
        (".nyc_output/out.json", True),
        ("coverage/lcov.info", True),
        ("coverage.out", True),
        ("package.test", True),
        ("build/classes/App.class", True),
        ("dist/app.jar", True),
        ("target/classes/App.class", True),
        (".gradle/caches/modules.bin", True),
        (".cache/tool/state", True),
        ("logs/build.txt", True),
        ("bin/solver.exe", True),
        ("src/build_tools.py", False),
        ("src/targeting.go", False),
        ("src/catalog.ts", False),
        ("src/App.java", False),
        ("docs/build.md", False),
        ("gradlew", False),
    ],
)
def test_generated_artifact_policy_is_shared_with_sanitization(
    path: str, expected: bool
) -> None:
    artifact_path = Path(path)

    assert is_generated_artifact(artifact_path) is expected
    assert is_leaky_artifact(artifact_path) is expected


def test_generated_artifact_policy_exports_the_exact_calibration_ignore_rules() -> None:
    patterns = generated_artifact_gitignore_patterns()

    assert patterns == GENERATED_ARTIFACT_GITIGNORE_PATTERNS
    for expected in (
        "__pycache__/",
        "*.pyc",
        "node_modules/",
        ".nyc_output/",
        "coverage/",
        "build/",
        "dist/",
        "target/",
        "*.class",
        "*.jar",
        "*.test",
        "*.log",
    ):
        assert expected in patterns
