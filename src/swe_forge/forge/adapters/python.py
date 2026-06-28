"""Python language adapter.

Implements the language-specific behavior the env-first build and downstream
stages need for Python repositories: filesystem detection, the pinned base
image, dependency-install commands, the pytest test command (selection-aware),
test-file classification, and the mutation-tool metadata hook. AST parsing and
mutation (:meth:`parse_symbols`, :meth:`mutate_ast`) and the in-Docker mutation
run (:meth:`mutation_tool_run`) are implemented by later milestones and still
raise :class:`NotImplementedError`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from swe_forge.forge.adapters._fsdetect import has_root_marker, has_source_file
from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutantStats,
    MutationOp,
    Patch,
    PathLike,
    Symbol,
)

_TODO = "PythonAdapter.{method}() is not implemented yet (later milestone)."

# Root manifests that unambiguously mark a Python project.
_PY_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "tox.ini",
)
_PY_EXTENSIONS = (".py",)
_REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "dev-requirements.txt",
)
_BUILD_MANIFESTS = ("pyproject.toml", "setup.py", "setup.cfg")
_TEST_FILE_RE = re.compile(r"(?:test_.*|.*_test)\.py$")


class PythonAdapter(LanguageAdapter):
    """Adapter for Python repositories (``pyproject.toml`` / ``*.py``)."""

    name = "python"
    mutation_tool = "mutmut"
    mutation_tools = ("mutmut", "cosmic-ray")

    def detect(self, repo_path: PathLike) -> bool:
        return has_root_marker(repo_path, _PY_MARKERS) or has_source_file(
            repo_path, _PY_EXTENSIONS
        )

    def base_image(self) -> str:
        return "python:3.12-slim"

    def install_commands(self, repo_path: PathLike) -> list[str]:
        root = Path(repo_path)
        commands: list[str] = []
        for req in _REQUIREMENT_FILES:
            if (root / req).is_file():
                commands.append(f"pip install -r {req}")
        if any((root / manifest).is_file() for manifest in _BUILD_MANIFESTS):
            commands.append("pip install -e .")
        if not commands:
            commands.append("pip install pytest")
        return commands

    def test_command(self, selection: Sequence[str] | None = None) -> str:
        base = "python -m pytest"
        if selection:
            return f"{base} {' '.join(selection)}"
        return base

    def is_test_file(self, path: PathLike) -> bool:
        return bool(_TEST_FILE_RE.fullmatch(Path(path).name))

    def parse_symbols(self, file: PathLike) -> list[Symbol]:
        raise NotImplementedError(_TODO.format(method="parse_symbols"))

    def mutate_ast(self, file: PathLike, symbol: Symbol, op: MutationOp) -> Patch:
        raise NotImplementedError(_TODO.format(method="mutate_ast"))

    def mutation_tool_run(
        self,
        image: str,
        repo_path: PathLike,
        *,
        paths: Sequence[str] | None = None,
        test_command: str | None = None,
    ) -> MutantStats:
        raise NotImplementedError(_TODO.format(method="mutation_tool_run"))
