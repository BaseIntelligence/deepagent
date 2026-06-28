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
from swe_forge.forge.generators.bug_combination import BugCombinationGenerator
from swe_forge.forge.generators.function_removal import FunctionRemovalGenerator
from swe_forge.forge.generators.lm_authored import LmAuthoredGenerator
from swe_forge.forge.generators.menu import (
    CellResult,
    MenuCellSpec,
    MenuReport,
    RoundTripResult,
    build_coverage_matrix,
    build_menu_cell_specs,
    evaluate_cell,
    evaluate_coverage,
    run_menu_selfcheck,
    schema_completeness,
    verify_candidate_roundtrip,
)
from swe_forge.forge.generators.multi_file import MultiFileGenerator
from swe_forge.forge.generators.pr_mirror import PrMirrorGenerator


def build_default_generator_registry() -> GeneratorRegistry:
    """Return a registry holding the available bug generators."""
    registry = GeneratorRegistry()
    registry.register(AstMutationGenerator())
    registry.register(LmAuthoredGenerator())
    registry.register(PrMirrorGenerator())
    registry.register(FunctionRemovalGenerator())
    registry.register(MultiFileGenerator())
    registry.register(BugCombinationGenerator())
    return registry


__all__ = [
    "AstMutationGenerator",
    "BugCombinationGenerator",
    "BugGenerator",
    "CellResult",
    "FunctionRemovalGenerator",
    "GenerationError",
    "GenerationRequest",
    "GeneratorRegistry",
    "LmAuthoredGenerator",
    "MenuCellSpec",
    "MenuReport",
    "MultiFileGenerator",
    "PrMirrorGenerator",
    "RoundTripResult",
    "build_coverage_matrix",
    "build_default_generator_registry",
    "build_menu_cell_specs",
    "evaluate_cell",
    "evaluate_coverage",
    "run_menu_selfcheck",
    "schema_completeness",
    "verify_candidate_roundtrip",
]
