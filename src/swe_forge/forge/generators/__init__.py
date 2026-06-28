"""Stage-2 bug-generator menu: interface, registry, and concrete generators.

Exposes the :class:`BugGenerator` interface and the :class:`GeneratorRegistry`,
plus :func:`build_default_generator_registry`, which returns a registry holding
the generators implemented so far. The full six-generator menu is filled in
across the m3 milestone; this package grows one generator at a time.
"""

from __future__ import annotations

from swe_forge.forge.generators.ast_mutation import AstMutationGenerator
from swe_forge.forge.generators.base import (
    BugGenerator,
    GenerationError,
    GenerationRequest,
    GeneratorRegistry,
)


def build_default_generator_registry() -> GeneratorRegistry:
    """Return a registry holding the available bug generators."""
    registry = GeneratorRegistry()
    registry.register(AstMutationGenerator())
    return registry


__all__ = [
    "AstMutationGenerator",
    "BugGenerator",
    "GenerationError",
    "GenerationRequest",
    "GeneratorRegistry",
    "build_default_generator_registry",
]
