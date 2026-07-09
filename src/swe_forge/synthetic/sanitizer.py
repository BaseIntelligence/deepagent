"""Sanitizers for synthetic task workspaces."""

from __future__ import annotations

import shutil
from pathlib import Path

from swe_forge.artifacts import (
    GENERATED_ARTIFACT_DIRECTORY_NAMES,
    GENERATED_ARTIFACT_FILE_SUFFIXES,
    is_generated_artifact,
)
from swe_forge.synthetic.models import SanitizerResult

# Compatibility aliases. The policy itself lives in ``swe_forge.artifacts`` so
# calibration and sanitization cannot silently diverge.
ARTIFACT_NAMES = GENERATED_ARTIFACT_DIRECTORY_NAMES
ARTIFACT_SUFFIXES = GENERATED_ARTIFACT_FILE_SUFFIXES


def is_leaky_artifact(path: Path) -> bool:
    """Return whether a path is a cache/build artifact that can leak answers."""
    return is_generated_artifact(path)


def sanitize_tree(root: Path | str, *, dry_run: bool = False) -> SanitizerResult:
    """Remove common build/cache artifacts under ``root``.

    This is intentionally conservative and never follows paths outside ``root``.
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    removed: list[Path] = []
    skipped: list[Path] = []

    for path in sorted(root_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        resolved = path.resolve()
        if root_path not in (resolved, *resolved.parents):
            skipped.append(path)
            continue
        if not is_leaky_artifact(path):
            continue
        removed.append(path)
        if dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    return SanitizerResult(removed_paths=removed, skipped_paths=skipped)
