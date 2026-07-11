"""The source-only policy shared by immutable hidden-test manifests.

Approved trees bind only source-test and wrapper bytes. CPython cache
directories and bytecode files are runtime artifacts, so they never affect an
approved tree's fingerprint, file count, snapshot, or rehydrated test suite.
This module deliberately classifies paths only; callers retain their existing
no-follow traversal and fail-closed checks for every approved source path.
"""

from __future__ import annotations

from pathlib import PurePath

_RUNTIME_BYTECODE_DIRECTORY = "__pycache__"
_RUNTIME_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})


def is_runtime_bytecode_directory(path: str | PurePath) -> bool:
    """Return whether ``path`` names a CPython runtime bytecode directory."""

    return PurePath(path).name == _RUNTIME_BYTECODE_DIRECTORY


def is_runtime_bytecode_file(path: str | PurePath) -> bool:
    """Return whether ``path`` names a CPython runtime bytecode file."""

    return PurePath(path).suffix in _RUNTIME_BYTECODE_SUFFIXES


def is_approved_source_path(path: str | PurePath) -> bool:
    """Return whether a relative path belongs in a source-only manifest."""

    candidate = PurePath(path)
    return (
        _RUNTIME_BYTECODE_DIRECTORY not in candidate.parts
        and not is_runtime_bytecode_file(candidate)
    )
