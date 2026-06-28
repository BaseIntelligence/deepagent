"""JavaScript / TypeScript language adapter.

A single adapter covers both plain-JS and TS repositories (both resolve to
``"javascript"``). Implements detection, the pinned base image, dependency
install (lockfile-aware), the ``node --test`` command (selection-aware),
test-file classification, AST parsing/mutation, and the in-Docker mutation run
(:meth:`mutation_tool_run`, via Stryker) used by the mutation-adequacy gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from swe_forge.forge.adapters._fsdetect import has_root_marker, has_source_file
from swe_forge.forge.adapters._mutate import mutate_source
from swe_forge.forge.adapters._treesitter import parse_js_symbols
from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutantStats,
    MutationExecutor,
    MutationOp,
    Patch,
    PathLike,
    Symbol,
)


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

    def baseline_test_command(self, repo_path: PathLike) -> str:
        """Run the repo's configured suite via its ``test`` npm script.

        A JS/TS repo's real suite is whatever ``npm test`` invokes (e.g.
        ``ava && tsd``), which is generally NOT the ``node --test`` standard
        runner. ``baseline_install_commands`` (the default lockfile-aware
        ``npm ci``/``npm install``) brings in the devDependencies that script
        needs, and ``node --test`` remains available for synthesized F2P tests.
        """
        return "npm test"

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
        return mutate_source(self.name, file, symbol, op)

    async def mutation_tool_run(
        self,
        executor: MutationExecutor,
        *,
        target_files: Sequence[str],
        timeout: float = 1200.0,
    ) -> MutantStats:
        """Run Stryker against the gold target file(s) inside ``executor``.

        Uses Stryker's command test runner (a non-zero ``npm test`` exit = a kill)
        so any JS/TS suite works without a framework plugin, fetched on demand via
        ``npx``. Mutant statuses are read from Stryker's JSON report.
        """
        from swe_forge.forge.adapters._mutation_tools import (
            STRYKER_CONFIG,
            STRYKER_REPORT,
            STRYKER_SETUP,
            MutationToolError,
            parse_stryker_json,
            stryker_config,
            stryker_run_command,
        )

        sources = [str(f) for f in target_files if not self.is_test_file(f)]
        if not sources:
            raise MutationToolError("no non-test target files to mutate (javascript)")

        for cmd in STRYKER_SETUP:
            res = await executor.run_command(cmd, timeout=timeout)
            if res.exit_code != 0:
                raise MutationToolError(
                    "Stryker requires the 'ps' utility (procps), which the "
                    "node slim base image lacks and which could not be installed "
                    f"(exit {res.exit_code}): {(res.stderr or res.stdout)[:300]}"
                )

        await executor.write_file(
            STRYKER_CONFIG, stryker_config(sources, test_command="npm test")
        )
        run = await executor.run_command(stryker_run_command(), timeout=timeout)
        try:
            report = await executor.read_file(STRYKER_REPORT)
        except Exception as exc:
            raise MutationToolError(
                f"Stryker produced no JSON report (exit {run.exit_code}): "
                f"{(run.stderr or run.stdout)[:400]}"
            ) from exc
        counts = parse_stryker_json(report)
        return MutantStats(
            total=counts.total,
            killed=counts.killed,
            survived=counts.survived,
            tool="stryker",
            survivors=counts.survivors,
        )
