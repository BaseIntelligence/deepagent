"""Language-adapter interface, supporting primitives, and the adapter registry.

Everything language-specific in the forge pipeline lives behind
:class:`LanguageAdapter` so the synthesis/oracle/calibration stages stay
language-agnostic and never branch on language. Concrete adapters (Python,
JS/TS, Go) are selected at runtime by :class:`AdapterRegistry` via their
``detect()`` predicate.

This module defines the *interface only*: the ABC, the small value types its
methods exchange (:class:`Symbol`, :class:`Patch`, :class:`MutantStats`,
:class:`MutationOp`), and the registry that holds adapters and selects one for a
repository. The concrete per-language behavior (detection, install/test
commands, AST parsing, mutation) is implemented by the build milestone; the
shipped adapter classes are stubs that raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

PathLike: TypeAlias = str | Path


class MutationOp(str, Enum):
    """A deterministic AST mutation operator applied to a target symbol."""

    OPERATOR_SWAP = "operator_swap"
    OFF_BY_ONE = "off_by_one"
    BRANCH_REMOVAL = "branch_removal"
    COMPARISON_FLIP = "comparison_flip"
    BOOLEAN_FLIP = "boolean_flip"
    CONSTANT_TWEAK = "constant_tweak"


@dataclass(frozen=True)
class Symbol:
    """A located source declaration the pipeline can target for mutation.

    Carries enough to both identify and locate/mutate the declaration: its
    ``name`` and ``kind`` plus the file and 1-based inclusive line span. An
    optional ``signature`` records the public interface (used later to pin the
    expected API surface so correct solutions are not failed on naming).
    """

    name: str
    kind: str
    file: str
    start_line: int
    end_line: int
    signature: str | None = None


@dataclass(frozen=True)
class Patch:
    """A unified-diff edit produced by :meth:`LanguageAdapter.mutate_ast`.

    ``diff`` is a git-applyable unified diff; ``files`` lists the repo-relative
    paths it touches.
    """

    diff: str
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class MutantStats:
    """Result of running a language mutation tool against the gold code.

    ``survivors`` carries short human-readable descriptions of the mutants the
    suite failed to kill (used to guide the mutation-adequacy gate's auto-test
    synthesis); it is optional so existing callers that only need the counts are
    unaffected.
    """

    total: int
    killed: int
    survived: int = 0
    tool: str = ""
    survivors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def kill_ratio(self) -> float:
        """Fraction of generated mutants the suite killed (0.0 when none ran)."""
        return self.killed / self.total if self.total else 0.0


class AdapterError(RuntimeError):
    """Base error for the language-adapter layer."""


class ParseError(AdapterError):
    """Raised when an adapter cannot parse malformed source.

    Distinguishes a genuine syntax error in the target file from a programming
    error in the adapter, so callers can surface a clean "parse error" instead
    of crashing. The message names the offending file and the underlying reason.
    """


class DuplicateAdapterError(AdapterError):
    """Raised when registering a second adapter under an existing name."""


class NoAdapterFoundError(AdapterError):
    """Raised when no registered adapter matches a repository or name."""


@runtime_checkable
class MutationExecResult(Protocol):
    """The exit code + captured output of a command run in a sandbox."""

    @property
    def exit_code(self) -> int: ...
    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...


@runtime_checkable
class MutationExecutor(Protocol):
    """Minimal sandbox surface ``mutation_tool_run`` drives (DockerSandbox-compat).

    A throwaway container whose image is the candidate's green ``EnvImage`` (the
    gold repo already checked out at :attr:`workspace_dir`). The adapter installs
    its mutation tool, writes a config, runs the tool, and reads any report file
    through this surface, all language-agnostically from the gate's perspective.
    """

    @property
    def workspace_dir(self) -> str: ...

    async def run_command(
        self,
        cmd: str,
        *,
        cwd: str | None = ...,
        timeout: float | None = ...,
        env: dict[str, str] | None = ...,
    ) -> MutationExecResult: ...

    async def write_file(self, path: str, content: str) -> None: ...

    async def read_file(self, path: str) -> str: ...


class LanguageAdapter(ABC):
    """Interface every per-language adapter implements.

    Subclasses must set :attr:`name` to a non-empty canonical identifier
    (``"python"`` | ``"javascript"`` | ``"go"``) and implement every abstract
    method. Stages call the adapter exclusively; they never inspect the language
    directly.
    """

    #: Canonical adapter name, e.g. ``"python"`` | ``"javascript"`` | ``"go"``.
    #: Concrete adapters must override this with a non-empty value.
    name: str = ""

    #: Primary mutation tool this language uses (``mutmut`` | ``stryker`` |
    #: ``go-mutesting``). Metadata only; the actual run is :meth:`mutation_tool_run`.
    mutation_tool: str = ""

    #: All mutation tools usable for this language, primary first (e.g. Python
    #: accepts both ``mutmut`` and ``cosmic-ray``).
    mutation_tools: tuple[str, ...] = ()

    @abstractmethod
    def detect(self, repo_path: PathLike) -> bool:
        """Return ``True`` iff this adapter's language is the repo's language."""

    @abstractmethod
    def base_image(self) -> str:
        """Return the pinned Docker base image for this language."""

    @abstractmethod
    def install_commands(self, repo_path: PathLike) -> list[str]:
        """Return the ecosystem-appropriate dependency-install commands."""

    @abstractmethod
    def test_command(self, selection: Sequence[str] | None = None) -> str:
        """Return the test runner command, narrowed to ``selection`` when given.

        This is the *adapter standard runner* (pytest / ``node --test`` /
        ``go test``) used to execute the synthesized hidden F2P tests added in
        later stages. It is distinct from the *baseline* capability below, which
        runs the repository's own configured suite.
        """

    def baseline_install_commands(self, repo_path: PathLike) -> list[str]:
        """Return the install commands for a green *baseline* of the repo's suite.

        Unlike :meth:`install_commands` (the bare runtime install), this includes
        the repository's TEST/DEV dependencies so its own test suite can run.
        The default is :meth:`install_commands`; adapters whose ecosystems split
        test deps out of the runtime install (e.g. Python) override this.
        """
        return self.install_commands(repo_path)

    def baseline_test_command(self, repo_path: PathLike) -> str:
        """Return the command that runs the repository's OWN configured suite.

        This is the command proven green at build time and recorded in
        ``EnvImage.baseline_test_command`` (used for baseline + P2P regression).
        It is distinct from the selection-aware :meth:`test_command` standard
        runner. The default delegates to :meth:`test_command` (no selection);
        adapters whose configured suite differs from the standard runner (e.g.
        JS/TS running ``npm test``) override this.
        """
        return self.test_command()

    def apply_p2p_exclusions(self, command: str, exclusions: Sequence[str]) -> str:
        """Return ``command`` narrowed to SKIP the named fix-independent tests.

        Some repos carry self-tests that are green/red independently of the
        manufactured fault (e.g. a version-constant assertion that lags
        ``package.json`` at a non-release commit). Such a test is NOT part of the
        P2P regression contract, so the env build (and the P2P set recorded on the
        ``EnvImage``) must exclude it. The exclusion is applied language-agnostic
        from the stage's perspective (the stage only calls this method); the
        per-language test-runner syntax lives here. The default is a no-op (return
        ``command`` unchanged); adapters whose runner supports name-based skipping
        override it.
        """
        return command

    def select_tests(self, command: str, names: Sequence[str]) -> str:
        """Return ``command`` narrowed to RUN ONLY the named tests.

        The positive counterpart of :meth:`apply_p2p_exclusions`: given the
        repo's own baseline command (its real runner) and the names of the
        fault-isolating F2P tests, return a command that runs exactly those tests
        via that runner. This lets a ``pr_mirror`` candidate confirm its isolated
        F2P (fails-on-broken / passes-on-gold) using the repo's configured runner
        rather than the standard :meth:`test_command` runner, which matters when
        the two differ (e.g. JS/TS ``npm test`` driving Mocha vs ``node --test``).
        The stage only calls this method; the per-language runner syntax lives in
        the concrete adapter. The default returns ``command`` unchanged (run the
        whole suite, which still FAILS on the broken tree because a named test
        fails); adapters whose runner supports name-based selection override it.
        """
        return command

    def parse_test_failures(self, output: str) -> list[str]:
        """Return the de-selectable names of the tests that FAILED in ``output``.

        Parses this language's test-runner output (pytest / Mocha / ``go test``)
        and returns the identifiers of the tests that failed, in first-seen order
        with duplicates removed. The names are exactly the form :meth:`select_tests`
        / :meth:`apply_p2p_exclusions` consume, so a STRUCTURAL mutation's
        fault-independent collateral failures (e.g. a removed function's own
        doctests) can be derived per candidate and excluded from the establish
        P2P set -- never the synthesized F2P, never a gate loosening.

        The default returns ``[]`` (a conservative no-op for runners whose output
        is not safely parseable); adapters whose output can be parsed override it.
        """
        return []

    @abstractmethod
    def parse_symbols(self, file: PathLike) -> list[Symbol]:
        """Parse ``file`` and return its declared, locatable symbols."""

    @abstractmethod
    def mutate_ast(self, file: PathLike, symbol: Symbol, op: MutationOp) -> Patch:
        """Apply mutation ``op`` to ``symbol`` in ``file`` and return the patch."""

    def remove_function_body(self, file: PathLike, symbol: Symbol) -> Patch:
        """Remove ``symbol``'s body (keeping its signature) and return the patch.

        Replaces the function/method body with a language-appropriate stub that
        keeps the file parseable; the inverse gold patch re-inserts the original
        body exactly. Returns an empty :class:`Patch` when ``symbol`` has no
        removable/meaningful body. Language-agnostic by delegating the excision
        to the shared tree-sitter driver keyed on :attr:`name`.
        """
        from swe_forge.forge.adapters._removal import remove_body

        return remove_body(self.name, file, symbol)

    @abstractmethod
    async def mutation_tool_run(
        self,
        executor: MutationExecutor,
        *,
        target_files: Sequence[str],
        timeout: float = 1200.0,
        target_regions: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    ) -> MutantStats:
        """Run this language's mutation tool against the gold code in ``executor``.

        Mutates the non-test ``target_files`` (the candidate's gold source) inside
        the throwaway container, runs the established hidden suite against every
        mutant, and returns the :class:`MutantStats` (``total``/``killed`` plus
        survivor descriptions) for the mutation-adequacy gate. The gate writes the
        hidden test files into the workspace before calling, so the tool's test
        command collects them alongside the repo's own suite.

        ``target_regions`` optionally maps a target file to changed-symbol line
        ranges (1-based, inclusive) so an over-large mutation run can be SCOPED to
        the actually-mutated region -- the difficulty amplifiers
        (``bug_combination`` / ``multi_file``) on a large modular module would not
        otherwise finish within ``timeout``. Adapters whose tool is already
        file-scoped (Go, JS) accept and ignore it; the Python adapter (cosmic-ray,
        which mutates a whole module) honors it.
        """

    @abstractmethod
    def is_test_file(self, path: PathLike) -> bool:
        """Return ``True`` iff ``path`` is a test file in this language."""


class AdapterRegistry:
    """Holds language adapters and selects one for a repository by ``detect()``.

    Registration order is preserved; :meth:`detect` returns the first adapter
    whose predicate matches so selection is deterministic. Names are unique.
    """

    def __init__(self) -> None:
        self._adapters: list[LanguageAdapter] = []

    def register(self, adapter: LanguageAdapter) -> LanguageAdapter:
        """Register ``adapter``; reject empty or duplicate names. Returns it."""
        if not adapter.name:
            raise ValueError(
                "adapter must declare a non-empty name before registration"
            )
        if any(existing.name == adapter.name for existing in self._adapters):
            raise DuplicateAdapterError(
                f"an adapter named {adapter.name!r} is already registered"
            )
        self._adapters.append(adapter)
        return adapter

    def __len__(self) -> int:
        return len(self._adapters)

    def __iter__(self) -> Iterator[LanguageAdapter]:
        return iter(tuple(self._adapters))

    def adapters(self) -> tuple[LanguageAdapter, ...]:
        """Return the registered adapters in registration order."""
        return tuple(self._adapters)

    def names(self) -> tuple[str, ...]:
        """Return the registered adapter names in registration order."""
        return tuple(adapter.name for adapter in self._adapters)

    def get(self, name: str) -> LanguageAdapter:
        """Return the adapter registered under ``name`` or raise."""
        for adapter in self._adapters:
            if adapter.name == name:
                return adapter
        raise NoAdapterFoundError(f"no adapter registered under the name {name!r}")

    def detect_all(self, repo_path: PathLike) -> list[LanguageAdapter]:
        """Return every registered adapter whose ``detect()`` matches the repo."""
        return [adapter for adapter in self._adapters if adapter.detect(repo_path)]

    def detect(self, repo_path: PathLike) -> LanguageAdapter:
        """Return the first adapter matching ``repo_path`` or raise."""
        for adapter in self._adapters:
            if adapter.detect(repo_path):
                return adapter
        raise NoAdapterFoundError(
            f"no registered adapter matched the repository at {repo_path!r}"
        )

    def detect_optional(self, repo_path: PathLike) -> LanguageAdapter | None:
        """Return the first matching adapter, or ``None`` if none match."""
        for adapter in self._adapters:
            if adapter.detect(repo_path):
                return adapter
        return None
