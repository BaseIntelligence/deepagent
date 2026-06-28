"""Go language adapter.

Implements detection, the pinned base image, dependency download, the
``go test`` command (selection-aware via package paths and/or ``-run`` names),
test-file classification, and the mutation-tool metadata hook. AST parsing and
mutation (:meth:`parse_symbols`, :meth:`mutate_ast`) and the in-Docker mutation
run (:meth:`mutation_tool_run`) are implemented by later milestones and still
raise :class:`NotImplementedError`.
"""

from __future__ import annotations

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

_TODO = "GoAdapter.{method}() is not implemented yet (later milestone)."

_GO_MARKERS = ("go.mod", "go.sum", "go.work")
_GO_EXTENSIONS = (".go",)


def _is_go_package(token: str) -> bool:
    """Return ``True`` iff ``token`` names a package/path rather than a test name.

    Go test selections mix package import paths (``./pkg/foo``, ``.``) with test
    function names (``TestAdd``). Anything containing a path separator or a
    leading ``.`` is treated as a package; everything else is a ``-run`` name.
    """
    return token == "." or token.startswith((".", "/")) or "/" in token


class GoAdapter(LanguageAdapter):
    """Adapter for Go repositories (``go.mod`` / ``*.go``)."""

    name = "go"
    mutation_tool = "go-mutesting"
    mutation_tools = ("go-mutesting",)

    def detect(self, repo_path: PathLike) -> bool:
        return has_root_marker(repo_path, _GO_MARKERS) or has_source_file(
            repo_path, _GO_EXTENSIONS
        )

    def base_image(self) -> str:
        return "golang:1.22"

    def install_commands(self, repo_path: PathLike) -> list[str]:
        return ["go mod download"]

    def test_command(self, selection: Sequence[str] | None = None) -> str:
        if not selection:
            return "go test ./..."
        packages = [token for token in selection if _is_go_package(token)]
        names = [token for token in selection if not _is_go_package(token)]
        scope = " ".join(packages) if packages else "./..."
        if names:
            pattern = "|".join(names)
            return f"go test -run '^({pattern})$' {scope}"
        return f"go test {scope}"

    def is_test_file(self, path: PathLike) -> bool:
        return Path(path).name.endswith("_test.go")

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
