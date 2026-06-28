"""Go language adapter (stub).

Foundational stub registered in the default :class:`AdapterRegistry`. The
behavior (detection, install/test commands, ``go/ast`` symbol parsing and
mutation, go-mutesting runs) is implemented in the env-first build milestone;
every method here raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Sequence

from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutantStats,
    MutationOp,
    Patch,
    PathLike,
    Symbol,
)

_TODO = "GoAdapter.{method}() is not implemented yet (env-first build milestone)."


class GoAdapter(LanguageAdapter):
    """Adapter for Go repositories. Behavior pending implementation."""

    name = "go"

    def detect(self, repo_path: PathLike) -> bool:
        raise NotImplementedError(_TODO.format(method="detect"))

    def base_image(self) -> str:
        raise NotImplementedError(_TODO.format(method="base_image"))

    def install_commands(self, repo_path: PathLike) -> list[str]:
        raise NotImplementedError(_TODO.format(method="install_commands"))

    def test_command(self, selection: Sequence[str] | None = None) -> str:
        raise NotImplementedError(_TODO.format(method="test_command"))

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

    def is_test_file(self, path: PathLike) -> bool:
        raise NotImplementedError(_TODO.format(method="is_test_file"))
