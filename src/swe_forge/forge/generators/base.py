"""Bug-generator interface and registry (Stage 2 synthesis menu).

Every generator manufactures a :class:`~swe_forge.forge.models.Candidate` from
known-good code: a forward ``mutation_patch`` paired with its inverse gold
``oracle_patch``. The interface keeps the synthesis stage language-agnostic by
routing all language-specific work through a
:class:`~swe_forge.forge.adapters.base.LanguageAdapter`.

This module defines the ABC, the :class:`GenerationRequest` it consumes, the
:class:`GenerationError` raised on a failed generation, and the
:class:`GeneratorRegistry`. Concrete generators live in sibling modules and are
wired into the default registry in this package's ``__init__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from swe_forge.forge.adapters.base import LanguageAdapter
from swe_forge.forge.models import Candidate, EnvImage


class GenerationError(RuntimeError):
    """Raised when a generator cannot produce a valid, round-tripping Candidate.

    Signals the caller to abort and emit NO Candidate artifact (a partial or
    non-inverting result must never ship).
    """


@dataclass
class GenerationRequest:
    """Inputs for one generation attempt.

    ``repo_root`` is the host checkout the generator reads and patches. ``seed``
    makes deterministic generators reproducible. ``file``/``symbol``/``op`` are
    optional target hints (repo-relative file, symbol name, generator-specific
    operator) that pin the mutation; when omitted the generator selects a target
    deterministically from ``seed``. ``env_image``, when supplied, gates
    generation on a proven green baseline (the hard Stage-1 precondition).
    """

    repo_root: Path
    seed: int = 0
    file: str | None = None
    symbol: str | None = None
    op: str | None = None
    env_image: EnvImage | None = None
    params: dict[str, object] = field(default_factory=dict)


class BugGenerator(ABC):
    """Interface every Stage-2 bug generator implements."""

    #: Canonical generator name; must be one of
    #: :data:`~swe_forge.forge.models.GENERATOR_NAMES`.
    name: str = ""

    @abstractmethod
    def generate(
        self, request: GenerationRequest, adapter: LanguageAdapter
    ) -> Candidate:
        """Produce one verified :class:`Candidate` or raise :class:`GenerationError`."""


class GeneratorRegistry:
    """Holds bug generators and looks them up by name (registration order kept)."""

    def __init__(self) -> None:
        self._generators: list[BugGenerator] = []

    def register(self, generator: BugGenerator) -> BugGenerator:
        """Register ``generator``; reject empty or duplicate names. Returns it."""
        if not generator.name:
            raise ValueError(
                "generator must declare a non-empty name before registration"
            )
        if any(existing.name == generator.name for existing in self._generators):
            raise ValueError(
                f"a generator named {generator.name!r} is already registered"
            )
        self._generators.append(generator)
        return generator

    def __len__(self) -> int:
        return len(self._generators)

    def __iter__(self) -> Iterator[BugGenerator]:
        return iter(tuple(self._generators))

    def names(self) -> tuple[str, ...]:
        """Return the registered generator names in registration order."""
        return tuple(generator.name for generator in self._generators)

    def get(self, name: str) -> BugGenerator:
        """Return the generator registered under ``name`` or raise ``KeyError``."""
        for generator in self._generators:
            if generator.name == name:
                return generator
        raise KeyError(
            f"no generator registered under {name!r}; known: {', '.join(self.names())}"
        )
