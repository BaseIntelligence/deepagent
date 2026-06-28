"""Filesystem-marker detection helpers shared by the concrete language adapters.

Detection is rule-based and, for single-language repositories, mutually
exclusive: an adapter matches a repo when it finds either a root-level ecosystem
manifest (e.g. ``pyproject.toml`` / ``package.json`` / ``go.mod``) or at least
one source file carrying a language-specific extension somewhere in the tree.
Heavy or vendored directories (``.git``, ``node_modules``, ``vendor`` ...) are
pruned so the walk stays cheap and never matches on dependency code.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from swe_forge.forge.adapters.base import PathLike

#: Directories never worth scanning for first-party source: VCS metadata,
#: virtualenvs, dependency caches, and build outputs.
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "bower_components",
        ".pnpm-store",
        "vendor",
        "dist",
        "build",
        "out",
        "target",
        ".next",
        ".nuxt",
        ".gradle",
        ".idea",
        ".vscode",
    }
)

#: Bound the walk so a pathological tree cannot make detection unbounded.
_MAX_SCAN_FILES = 20000


def has_root_marker(repo_path: PathLike, names: Iterable[str]) -> bool:
    """Return ``True`` iff any of ``names`` is a file at the repo root."""
    root = Path(repo_path)
    return any((root / name).is_file() for name in names)


def has_source_file(repo_path: PathLike, extensions: Iterable[str]) -> bool:
    """Return ``True`` iff a file with one of ``extensions`` exists in the tree.

    The walk prunes ignored and hidden directories so dependency/build trees do
    not trigger a false positive, and is bounded by ``_MAX_SCAN_FILES``.
    """
    suffixes = tuple(extensions)
    if not suffixes:
        return False
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [
            d for d in dirnames if d not in _IGNORED_DIRS and not d.startswith(".")
        ]
        for filename in filenames:
            scanned += 1
            if scanned > _MAX_SCAN_FILES:
                return False
            if filename.endswith(suffixes):
                return True
    return False
