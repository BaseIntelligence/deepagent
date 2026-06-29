"""Offline coverage of the per-candidate P2P-exclusion derivation (m6-pilot-difficulty).

Exercises the structural-mutation collateral derivation deterministically -- no
Docker -- via the pure reducer plus a fake recipe driving the recipe-level core:

- a structural mutation's fault-independent collateral failures (a removed
  function's own doctests in boltons) are derived from the broken-tree baseline
  output and reduced to the P2P exclusion set;
- the synthesized/provided F2P is NEVER excluded (``protected`` is filtered out);
- a green broken tree yields no exclusions; the derivation reduces failures via
  the language adapter's ``parse_test_failures`` so the stage stays
  language-agnostic.
"""

from __future__ import annotations

import asyncio

from swe_forge.forge.adapters import build_default_registry
from swe_forge.forge.oracle.establish import TestRun, TreeState
from swe_forge.forge.oracle.p2p_derive import (
    STRUCTURAL_GENERATORS,
    compute_collateral_exclusions,
    derive_from_recipe,
)


# --------------------------------------------------------------------------- #
# Pure reducer: collateral = broken failures minus the protected F2P
# --------------------------------------------------------------------------- #
def test_compute_collateral_excludes_all_failures_when_unprotected() -> None:
    result = compute_collateral_exclusions(["test_a", "test_b", "slugify"])
    assert result.exclusions == ("test_a", "test_b", "slugify")
    assert result.broken_failures == ("test_a", "test_b", "slugify")
    assert result.p2p_green_on_broken is False
    assert result.has_exclusions is True


def test_compute_collateral_never_excludes_the_protected_f2p() -> None:
    # The F2P (provided/synthesized) must never be excluded from P2P.
    result = compute_collateral_exclusions(
        ["test_collateral", "test_the_f2p"],
        protected=["test_the_f2p"],
    )
    assert "test_the_f2p" not in result.exclusions
    assert result.exclusions == ("test_collateral",)
    assert "test_the_f2p" in result.protected
    # The full failure set is still recorded (audit), F2P included.
    assert set(result.broken_failures) == {"test_collateral", "test_the_f2p"}


def test_compute_collateral_dedupes_in_first_seen_order() -> None:
    result = compute_collateral_exclusions(["b", "a", "b", " a ", "", "c"])
    assert result.exclusions == ("b", "a", "c")


def test_compute_collateral_green_broken_has_no_exclusions() -> None:
    result = compute_collateral_exclusions([])
    assert result.exclusions == ()
    assert result.p2p_green_on_broken is True
    assert result.has_exclusions is False


# --------------------------------------------------------------------------- #
# Recipe-driven core: run broken P2P once, parse via the adapter, reduce
# --------------------------------------------------------------------------- #
class _FakeRecipe:
    """A minimal RecipeProtocol stand-in that scripts the broken-tree P2P run."""

    def __init__(
        self, p2p_run: TestRun, *, p2p_command: str = "python -m pytest"
    ) -> None:
        self.p2p_command = p2p_command
        self.language = "python"
        self._p2p_run = p2p_run
        self.states: list[TreeState] = []

    async def set_state(self, state: TreeState) -> None:
        self.states.append(state)

    async def run_p2p(self) -> TestRun:
        return self._p2p_run

    async def run_test(self, test: object) -> TestRun:  # pragma: no cover - unused
        raise AssertionError("derive must not run individual tests")

    async def write_test(self, test: object) -> None:  # pragma: no cover - unused
        raise AssertionError

    async def remove_test(self, test: object) -> None:  # pragma: no cover - unused
        raise AssertionError


def _derive(recipe: _FakeRecipe, **kw: object):
    adapter = build_default_registry().get("python")
    return asyncio.run(derive_from_recipe(recipe, adapter, **kw))  # type: ignore[arg-type]


def test_derive_from_recipe_extracts_collateral_from_broken_output() -> None:
    # boltons canonical case: removing a function deletes its doctest, which the
    # baseline collects -> a fault-independent collateral failure.
    output = (
        "=== short test summary info ===\n"
        "FAILED boltons/strutils.py::boltons.strutils.slugify\n"
        "FAILED tests/test_strutils.py::test_slugify_basic\n"
    )
    recipe = _FakeRecipe(
        TestRun("python -m pytest", exit_code=1, passed=False, stdout=output)
    )
    result = _derive(recipe)
    assert recipe.states[0] == TreeState.BROKEN
    assert result.exclusions == ("slugify", "test_slugify_basic")
    assert result.p2p_green_on_broken is False


def test_derive_from_recipe_protects_the_f2p_name() -> None:
    output = (
        "FAILED tests/test_x.py::test_collateral\nFAILED tests/test_x.py::test_f2p\n"
    )
    recipe = _FakeRecipe(
        TestRun("python -m pytest", exit_code=1, passed=False, stdout=output)
    )
    result = _derive(recipe, protected_names=["test_f2p"])
    assert "test_f2p" not in result.exclusions
    assert result.exclusions == ("test_collateral",)


def test_derive_from_recipe_green_broken_yields_no_exclusions() -> None:
    recipe = _FakeRecipe(
        TestRun("python -m pytest", exit_code=0, passed=True, stdout="all green")
    )
    result = _derive(recipe)
    assert result.exclusions == ()
    assert result.p2p_green_on_broken is True


def test_structural_generators_set_excludes_pr_mirror_and_lm_authored() -> None:
    assert "pr_mirror" not in STRUCTURAL_GENERATORS
    assert "lm_authored" not in STRUCTURAL_GENERATORS
    assert {"function_removal", "bug_combination", "multi_file", "ast_mutation"} == set(
        STRUCTURAL_GENERATORS
    )
