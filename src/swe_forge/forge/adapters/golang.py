"""Go language adapter.

Implements detection, the pinned base image, dependency download, the
``go test`` command (selection-aware via package paths and/or ``-run`` names),
test-file classification, AST parsing/mutation, and the in-Docker mutation run
(:meth:`mutation_tool_run`, via go-mutesting) used by the mutation-adequacy gate.
"""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from pathlib import Path

from swe_forge.forge.adapters._fsdetect import has_root_marker, has_source_file
from swe_forge.forge.adapters._goast import parse_go_symbols
from swe_forge.forge.adapters._mutate import mutate_source
from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutantStats,
    MutationExecutor,
    MutationOp,
    Patch,
    PathLike,
    Symbol,
)

_GO_MARKERS = ("go.mod", "go.sum", "go.work")
_GO_EXTENSIONS = (".go",)


def _is_go_package(token: str) -> bool:
    """Return ``True`` iff ``token`` names a package/path rather than a test name.

    Go test selections mix package import paths (``./pkg/foo``, ``.``) with test
    function names (``TestAdd``). Anything containing a path separator or a
    leading ``.`` is treated as a package; everything else is a ``-run`` name.
    """
    return token == "." or token.startswith((".", "/")) or "/" in token


def _is_go_test_file(token: str) -> bool:
    """Return ``True`` iff ``token`` is a Go source FILE path (e.g. ``a/b_test.go``)."""
    return token.endswith(_GO_EXTENSIONS)


def _go_package_dir(token: str) -> str:
    """Map a Go source file path to the package directory ``go test`` can run.

    ``go test`` takes package import paths (directories or ``./...``), never a
    bare ``*.go`` file, so a single ``foo/bar_test.go`` selection must run the
    ``./foo`` package (which also compiles the package's other sources).
    """
    directory = posixpath.dirname(token).strip("/")
    return f"./{directory}" if directory else "."


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
        packages: list[str] = []
        names: list[str] = []
        for token in selection:
            if _is_go_test_file(token):
                packages.append(_go_package_dir(token))
            elif _is_go_package(token):
                packages.append(token)
            else:
                names.append(token)
        deduped: list[str] = []
        for pkg in packages:
            if pkg not in deduped:
                deduped.append(pkg)
        scope = " ".join(deduped) if deduped else "./..."
        if names:
            pattern = "|".join(names)
            return f"go test -run '^({pattern})$' {scope}"
        return f"go test {scope}"

    def is_test_file(self, path: PathLike) -> bool:
        return Path(path).name.endswith("_test.go")

    def parse_symbols(self, file: PathLike) -> list[Symbol]:
        return parse_go_symbols(file)

    def mutate_ast(self, file: PathLike, symbol: Symbol, op: MutationOp) -> Patch:
        return mutate_source(self.name, file, symbol, op)

    async def mutation_tool_run(
        self,
        executor: MutationExecutor,
        *,
        target_files: Sequence[str],
        timeout: float = 1200.0,
    ) -> MutantStats:
        """Run go-mutesting against the target package(s) inside ``executor``.

        go-mutesting is installed on demand (``go install``) and scoped to the
        target files' packages to bound runtime; ``PASS`` mutants are killed and
        ``FAIL`` mutants survive. Counts are aggregated across packages.
        """
        from swe_forge.forge.adapters._mutation_tools import (
            GO_MUTESTING_SETUP,
            MutationToolError,
            gomutesting_command,
            parse_gomutesting,
        )

        sources = [str(f) for f in target_files if not self.is_test_file(f)]
        if not sources:
            raise MutationToolError("no non-test target files to mutate (go)")

        for cmd in GO_MUTESTING_SETUP:
            res = await executor.run_command(cmd, timeout=timeout)
            if res.exit_code != 0:
                raise MutationToolError(
                    f"go-mutesting install failed (exit {res.exit_code}): "
                    f"{(res.stderr or res.stdout)[:400]}"
                )

        packages = sorted({posixpath.dirname(src) or "." for src in sources})
        total = 0
        killed = 0
        survivors: list[str] = []
        for pkg in packages:
            res = await executor.run_command(gomutesting_command(pkg), timeout=timeout)
            counts = parse_gomutesting(res.stdout + "\n" + res.stderr)
            total += counts.total
            killed += counts.killed
            survivors.extend(counts.survivors)

        return MutantStats(
            total=total,
            killed=killed,
            survived=total - killed,
            tool="go-mutesting",
            survivors=tuple(survivors),
        )
