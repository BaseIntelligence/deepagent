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

import ast
import re
from collections.abc import Sequence
from pathlib import Path

from swe_forge.forge.adapters._fsdetect import has_root_marker, has_source_file
from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutantStats,
    MutationOp,
    ParseError,
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

_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


def _signature(node: _FuncDef) -> str:
    """Render ``def name(args)`` for a function/method, async-aware, no body."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    return f"{prefix} {node.name}({args})"


class _SymbolCollector(ast.NodeVisitor):
    """Collect every function/method declaration with its location.

    A function is classified ``"method"`` when its nearest enclosing scope is a
    class body and ``"function"`` otherwise; nested functions are included so
    every mutable definition in the file is locatable. Line spans are 1-based
    and inclusive, taken from the ``def`` line (decorators excluded) through the
    function's last line.
    """

    def __init__(self, file: str) -> None:
        self.file = file
        self.symbols: list[Symbol] = []
        self._class_depth = 0

    def _add(self, node: _FuncDef) -> None:
        self.symbols.append(
            Symbol(
                name=node.name,
                kind="method" if self._class_depth > 0 else "function",
                file=self.file,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=_signature(node),
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: _FuncDef) -> None:
        self._add(node)
        # A function body is not a class scope: its nested defs are functions.
        saved = self._class_depth
        self._class_depth = 0
        for child in node.body:
            self.visit(child)
        self._class_depth = saved

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_depth += 1
        for child in node.body:
            self.visit(child)
        self._class_depth -= 1


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
        path = Path(file)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            location = f"{path}:{exc.lineno or 0}"
            raise ParseError(
                f"failed to parse Python source {location}: {exc.msg}"
            ) from exc
        collector = _SymbolCollector(str(path))
        collector.visit(tree)
        return collector.symbols

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
