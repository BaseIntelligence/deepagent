"""Canonical generated-artifact policy for agent-visible source trees.

The calibration solver's orphan baseline and synthetic sanitization must make
the same keep-or-remove decision.  This module is intentionally outside
``forge`` so the existing synthetic pipeline can depend on it without gaining a
forge dependency.
"""

from __future__ import annotations

from pathlib import Path

# Every rule supplies both the filesystem classification and the exact
# gitignore spelling. This is the single canonical policy definition, so
# sanitizer traversal and calibration's orphan baseline cannot diverge.
_GENERATED_ARTIFACT_RULES: tuple[tuple[str, str, str], ...] = (
    ("directory", "__pycache__", "__pycache__/"),
    ("file_suffix", ".pyc", "*.pyc"),
    ("file_suffix", ".pyo", "*.pyo"),
    ("directory_suffix", ".egg-info", "*.egg-info/"),
    ("directory", ".pytest_cache", ".pytest_cache/"),
    ("directory", ".mypy_cache", ".mypy_cache/"),
    ("directory", ".ruff_cache", ".ruff_cache/"),
    ("directory", ".tox", ".tox/"),
    ("directory", ".nox", ".nox/"),
    ("directory", ".cache", ".cache/"),
    ("directory", "node_modules", "node_modules/"),
    ("directory", ".nyc_output", ".nyc_output/"),
    ("file_name", ".coverage", ".coverage"),
    ("directory", "coverage", "coverage/"),
    ("file_name", "coverage.out", "coverage.out"),
    ("file_name", "lcov.info", "lcov.info"),
    ("directory", "htmlcov", "htmlcov/"),
    ("directory", "build", "build/"),
    ("directory", "dist", "dist/"),
    ("directory", "target", "target/"),
    ("directory", ".gradle", ".gradle/"),
    ("directory", "logs", "logs/"),
    ("file_suffix", ".class", "*.class"),
    ("file_suffix", ".jar", "*.jar"),
    ("file_suffix", ".log", "*.log"),
    ("file_suffix", ".test", "*.test"),
    ("file_suffix", ".exe", "*.exe"),
    ("file_suffix", ".dll", "*.dll"),
    ("file_suffix", ".so", "*.so"),
    ("file_suffix", ".dylib", "*.dylib"),
)

GENERATED_ARTIFACT_DIRECTORY_NAMES = frozenset(
    value for kind, value, _pattern in _GENERATED_ARTIFACT_RULES if kind == "directory"
)
GENERATED_ARTIFACT_DIRECTORY_SUFFIXES = tuple(
    value
    for kind, value, _pattern in _GENERATED_ARTIFACT_RULES
    if kind == "directory_suffix"
)
GENERATED_ARTIFACT_FILE_NAMES = frozenset(
    value for kind, value, _pattern in _GENERATED_ARTIFACT_RULES if kind == "file_name"
)
GENERATED_ARTIFACT_FILE_SUFFIXES = frozenset(
    value
    for kind, value, _pattern in _GENERATED_ARTIFACT_RULES
    if kind == "file_suffix"
)
GENERATED_ARTIFACT_GITIGNORE_PATTERNS = tuple(
    pattern for _kind, _value, pattern in _GENERATED_ARTIFACT_RULES
)


def is_generated_artifact(path: Path | str) -> bool:
    """Return whether ``path`` is a generated artifact under this policy."""
    candidate = Path(path)
    for component in candidate.parts:
        if component in GENERATED_ARTIFACT_DIRECTORY_NAMES:
            return True
        if component.endswith(GENERATED_ARTIFACT_DIRECTORY_SUFFIXES):
            return True

    return (
        candidate.name in GENERATED_ARTIFACT_FILE_NAMES
        or candidate.suffix in GENERATED_ARTIFACT_FILE_SUFFIXES
    )


def generated_artifact_gitignore_patterns() -> tuple[str, ...]:
    """Return the exact ignore rules used to build solver orphan baselines."""
    return GENERATED_ARTIFACT_GITIGNORE_PATTERNS
