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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeAlias

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
    """Result of running a language mutation tool against the gold code."""

    total: int
    killed: int
    survived: int = 0
    tool: str = ""

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
        """Return the test runner command, narrowed to ``selection`` when given."""

    @abstractmethod
    def parse_symbols(self, file: PathLike) -> list[Symbol]:
        """Parse ``file`` and return its declared, locatable symbols."""

    @abstractmethod
    def mutate_ast(self, file: PathLike, symbol: Symbol, op: MutationOp) -> Patch:
        """Apply mutation ``op`` to ``symbol`` in ``file`` and return the patch."""

    @abstractmethod
    def mutation_tool_run(
        self,
        image: str,
        repo_path: PathLike,
        *,
        paths: Sequence[str] | None = None,
        test_command: str | None = None,
    ) -> MutantStats:
        """Run the language mutation tool in ``image`` and return mutant stats."""

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
