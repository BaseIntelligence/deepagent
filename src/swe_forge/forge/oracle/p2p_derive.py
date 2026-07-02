"""Per-candidate P2P-exclusion derivation for STRUCTURAL mutations.

A structural mutation (``ast_mutation`` / ``function_removal`` / ``bug_combination``
/ ``multi_file``) on a MODULAR repo breaks some of the target's OWN existing tests
as collateral damage -- the canonical case being a removed function's doctests in
boltons -- so the repo's full baseline suite goes RED on the broken tree and the
establish gate rejects with ``p2p_not_green_on_broken`` even though the
manufactured fault is otherwise isolable.

``pr_mirror`` solves this per-REPO: a curated :attr:`RepoSpec.p2p_exclusions` list
(the flipping self-tests) is baked into the EnvImage baseline so establish's P2P is
green on broken BY CONSTRUCTION. Structural mutations need the same thing computed
per-CANDIDATE, because WHICH existing tests break depends on which symbol the
generator picked. This module derives, for one candidate, the set of existing tests
the BROKEN tree fails and produces a P2P exclusion list of exactly those
fault-independent collateral tests.

Invariants (never loosen a gate):

* The derivation NEVER excludes the synthesized/provided F2P. The structural F2P is
  a NEW test file the establish/differential P2P run never collects (hidden tests
  are written, run, then removed before any P2P run), so it cannot appear among the
  existing-test failures; any ``protected`` F2P name is additionally filtered out
  defensively.
* The exclusion is applied to BOTH the gold and broken P2P via the baseline command
  (mirroring ``pr_mirror``); gold stays green because excluding tests from an
  already-green suite keeps it green, and the establish gate still re-checks
  ``p2p_gold`` is green as a safety net.
* The fault stays detectable by the F2P and the downstream mutation/differential/
  alt-correct gates still harden the suite, so removing a collateral failure (a test
  the structural mutation inherently breaks, which therefore cannot be a P2P
  regression test) is sound, not a relaxation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import Candidate, EnvImage, require_green_baseline
from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    RecipeProtocol,
    TreeState,
)

if TYPE_CHECKING:
    from swe_forge.execution.docker_client import DockerClient

#: Generators whose forward mutation is STRUCTURAL (breaks the target's own
#: existing tests as collateral damage) and therefore benefits from per-candidate
#: P2P-exclusion derivation. ``pr_mirror`` is excluded (it reintroduces an
#: isolated real-PR fault with curated per-repo exclusions); ``lm_authored`` is a
#: subtle single-function change whose flipped test is the intended F2P.
STRUCTURAL_GENERATORS: frozenset[str] = frozenset(
    {"ast_mutation", "function_removal", "bug_combination", "multi_file"}
)


@dataclass(frozen=True)
class P2PDerivation:
    """The per-candidate P2P-exclusion derivation result.

    ``exclusions`` are the collateral existing-test names to keep OUT of the
    establish P2P set; ``file_exclusions`` are whole test MODULES the structural
    fault makes uncollectable (import error, no per-test name) to ignore
    wholesale; ``broken_failures`` is every existing test that failed on the
    broken tree (the superset before protecting the F2P); ``protected`` are the
    F2P names that were filtered out and never excluded.
    """

    exclusions: tuple[str, ...] = ()
    file_exclusions: tuple[str, ...] = ()
    broken_failures: tuple[str, ...] = ()
    protected: tuple[str, ...] = ()
    p2p_green_on_broken: bool = True
    details: dict[str, object] = field(default_factory=dict)

    @property
    def has_exclusions(self) -> bool:
        return bool(self.exclusions or self.file_exclusions)

    def to_dict(self) -> dict[str, object]:
        return {
            "exclusions": list(self.exclusions),
            "file_exclusions": list(self.file_exclusions),
            "broken_failures": list(self.broken_failures),
            "protected": list(self.protected),
            "p2p_green_on_broken": self.p2p_green_on_broken,
            "details": dict(self.details),
        }


def compute_collateral_exclusions(
    broken_failures: Sequence[str],
    *,
    protected: Sequence[str] = (),
    collection_error_files: Sequence[str] = (),
) -> P2PDerivation:
    """Reduce broken-tree failures to the collateral P2P exclusion set (pure).

    De-duplicates ``broken_failures`` in first-seen order and removes any name in
    ``protected`` (the synthesized/provided F2P) so the F2P is never excluded.
    ``collection_error_files`` are whole test modules the structural fault makes
    uncollectable (an IMPORT-TIME collateral failure with no per-test name to
    skip); they are de-duplicated into ``file_exclusions`` so the module is
    ignored wholesale. Returns the resulting :class:`P2PDerivation`. With neither
    per-test failures nor collection-error files the broken P2P is already green
    and both exclusion lists are empty.
    """
    protected_set = {p.strip() for p in protected if p.strip()}
    failures: list[str] = []
    exclusions: list[str] = []
    seen: set[str] = set()
    for raw in broken_failures:
        name = raw.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        failures.append(name)
        if name not in protected_set:
            exclusions.append(name)
    file_exclusions: list[str] = []
    seen_files: set[str] = set()
    for raw in collection_error_files:
        path = raw.strip()
        if not path or path in seen_files:
            continue
        seen_files.add(path)
        file_exclusions.append(path)
    return P2PDerivation(
        exclusions=tuple(exclusions),
        file_exclusions=tuple(file_exclusions),
        broken_failures=tuple(failures),
        protected=tuple(sorted(protected_set)),
        p2p_green_on_broken=not (failures or file_exclusions),
    )


async def derive_from_recipe(
    recipe: RecipeProtocol,
    adapter: LanguageAdapter,
    *,
    protected_names: Sequence[str] = (),
) -> P2PDerivation:
    """Derive the collateral exclusions by running the BROKEN-tree P2P via ``recipe``.

    Sets the tree to broken, runs the repo's full baseline (P2P) suite once, and
    -- when it is red -- parses the failing test names via
    :meth:`LanguageAdapter.parse_test_failures` and reduces them to the collateral
    exclusion set (protecting any F2P name). This is the offline-testable core
    (drive it with a fake recipe + adapter); the Docker wrapper is
    :func:`derive_structural_p2p_exclusions`.
    """
    await recipe.set_state(TreeState.BROKEN)
    run = await recipe.run_p2p()
    if run.passed:
        return P2PDerivation(
            exclusions=(),
            broken_failures=(),
            protected=tuple(sorted({p.strip() for p in protected_names if p.strip()})),
            p2p_green_on_broken=True,
            details={"p2p_command": recipe.p2p_command, "broken_p2p_exit": 0},
        )
    output = "\n".join(part for part in (run.stdout, run.stderr) if part)
    failures = adapter.parse_test_failures(output)
    collection_files = adapter.parse_collection_error_files(output)
    derivation = compute_collateral_exclusions(
        failures,
        protected=protected_names,
        collection_error_files=collection_files,
    )
    return P2PDerivation(
        exclusions=derivation.exclusions,
        file_exclusions=derivation.file_exclusions,
        broken_failures=derivation.broken_failures,
        protected=derivation.protected,
        p2p_green_on_broken=derivation.p2p_green_on_broken,
        details={
            "p2p_command": recipe.p2p_command,
            "broken_p2p_exit": run.exit_code,
            "parsed_failures": list(failures),
            "collection_error_files": list(collection_files),
        },
    )


async def derive_structural_p2p_exclusions(
    candidate: Candidate,
    env_image: EnvImage,
    adapter: LanguageAdapter | None = None,
    *,
    baseline_command: str = "",
    protected_names: Sequence[str] = (),
    command_timeout: float = 600.0,
    docker_client: "DockerClient | None" = None,
) -> P2PDerivation:
    """Derive a structural candidate's P2P exclusions in a throwaway Docker sandbox.

    A green baseline is a hard precondition. Opens a ``--rm`` DockerSandbox on the
    candidate's EnvImage, applies the forward mutation (broken tree), runs the
    repo's baseline (P2P) suite, and reduces its failures to the collateral
    exclusion set (never the F2P). Returns the :class:`P2PDerivation`; the caller
    bakes ``exclusions`` into a derived EnvImage baseline command (via
    :func:`apply_p2p_exclusions`) so establish's P2P is green on broken.
    """
    require_green_baseline(env_image)
    if adapter is None:
        adapter = build_default_registry().get(candidate.language)

    from swe_forge.execution.docker_client import DockerClient
    from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

    p2p_command = baseline_command or env_image.baseline_test_command
    client = docker_client or DockerClient()
    config = SandboxConfig(
        name="swe-forge-oracle-p2p-derive",
        image=env_image.image_tag,
        workspace_dir=env_image.workspace_dir,
        command_timeout=command_timeout,
    )
    sandbox = DockerSandbox(client, config)
    async with sandbox:
        recipe = DockerOracleRecipe(
            sandbox,
            language=candidate.language,
            workspace_dir=env_image.workspace_dir,
            mutation_patch=candidate.mutation_patch,
            oracle_patch=candidate.oracle_patch,
            p2p_command=p2p_command,
            command_timeout=command_timeout,
        )
        return await derive_from_recipe(
            recipe, adapter, protected_names=protected_names
        )


__all__ = [
    "STRUCTURAL_GENERATORS",
    "P2PDerivation",
    "compute_collateral_exclusions",
    "derive_from_recipe",
    "derive_structural_p2p_exclusions",
]
