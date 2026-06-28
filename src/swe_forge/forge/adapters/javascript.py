"""JavaScript / TypeScript language adapter.

A single adapter covers both plain-JS and TS repositories (both resolve to
``"javascript"``). Implements detection, the pinned base image, dependency
install (lockfile-aware), the ``node --test`` command (selection-aware),
test-file classification, and the mutation-tool metadata hook. AST parsing and
mutation (:meth:`parse_symbols`, :meth:`mutate_ast`) and the in-Docker mutation
run (:meth:`mutation_tool_run`) are implemented by later milestones and still
raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from swe_forge.forge.adapters._fsdetect import has_root_marker, has_source_file
from swe_forge.forge.adapters._treesitter import parse_js_symbols
from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutantStats,
    MutationOp,
    Patch,
    PathLike,
    Symbol,
)

_TODO = "JavaScriptAdapter.{method}() is not implemented yet (later milestone)."

# Root manifests/config that mark a JS or TS project (TS resolves here too).
_JS_MARKERS = (
    "package.json",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
    "bun.lock",
    "tsconfig.json",
    "jsconfig.json",
)
_JS_EXTENSIONS = (
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
)
_TEST_INFIXES = (".test", ".spec")


class JavaScriptAdapter(LanguageAdapter):
    """Adapter for JavaScript and TypeScript repositories (``package.json``)."""

    name = "javascript"
    mutation_tool = "stryker"
    mutation_tools = ("stryker",)

    def detect(self, repo_path: PathLike) -> bool:
        return has_root_marker(repo_path, _JS_MARKERS) or has_source_file(
            repo_path, _JS_EXTENSIONS
        )

    def base_image(self) -> str:
        return "node:22-slim"

    def install_commands(self, repo_path: PathLike) -> list[str]:
        root = Path(repo_path)
        if (root / "package-lock.json").is_file() or (
            root / "npm-shrinkwrap.json"
        ).is_file():
            return ["npm ci"]
        if (root / "yarn.lock").is_file():
            return ["yarn install --frozen-lockfile"]
        if (root / "pnpm-lock.yaml").is_file():
            return ["pnpm install --frozen-lockfile"]
        if (root / "bun.lockb").is_file() or (root / "bun.lock").is_file():
            return ["bun install"]
        return ["npm install"]

    def test_command(self, selection: Sequence[str] | None = None) -> str:
        base = "node --test"
        if selection:
            return f"{base} {' '.join(selection)}"
        return base

    def is_test_file(self, path: PathLike) -> bool:
        p = Path(path)
        if "__tests__" in p.parts:
            return True
        name = p.name
        for ext in _JS_EXTENSIONS:
            if name.endswith(ext):
                stem = name[: -len(ext)]
                return stem.endswith(_TEST_INFIXES)
        return False

    def parse_symbols(self, file: PathLike) -> list[Symbol]:
        return parse_js_symbols(file)

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
