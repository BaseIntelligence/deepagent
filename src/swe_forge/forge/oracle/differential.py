"""Differential-vs-gold gate: prove gold is uniquely accepted (PatchDiff-style).

The fourth oracle-hardening gate (architecture S6, Stage 3.4). Establish +
flakiness + mutation guarantee a *deterministic, mutation-adequate* FAIL->PASS
contract, but a suite can still admit a whole *plausible-but-wrong*
implementation that happens to satisfy every assertion (the "pass-but-wrong" gap
at the patch level rather than the single-mutant level). This gate closes that
gap PatchDiff-style: it generates plausible-but-wrong **variants** of the gold
code and runs the established F2P+P2P suite against each, requiring that *only*
the gold tree passes while every behaviorally-divergent variant FAILS >=1 test.

Flow (``the teacher proposes, deterministic execution disposes``):

1. Confirm the **gold** tree passes its own established F2P+P2P suite (the
   by-construction guarantee; a defensive re-check at this gate).
2. Score every plausible-wrong **variant** through the SAME Docker F2P/P2P
   primitives. A variant that FAILS >=1 test is *killed* (good - the suite
   distinguishes it from gold). A variant that passes the full suite is a
   **survivor** (the suite cannot tell it apart from gold).
3. For each survivor, enter a **bounded** test-strengthening loop: the teacher
   proposes an extra hidden test; each proposal is *confirmed* by execution -
   kept only if it PASSES on gold AND makes a surviving variant FAIL (i.e. it
   separates that variant from gold). New test ids are appended and survivors
   re-measured until none remain.
4. If the bounded loop leaves an indistinguishable survivor, the candidate is
   **rejected** with ``differential_pass == False`` and a reason citing the
   surviving variant (the oracle accepts a wrong behavior).

The gate scores through :class:`~swe_forge.forge.oracle.establish.DockerOracleRecipe`
so "F2P pass AND P2P green" means exactly the same thing everywhere, and it honors
the Python re-test ``.pyc`` determinism invariant baked into the recipe.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Protocol

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    ModelError,
    OracleReport,
    OracleTestFile,
    Provenance,
    require_green_baseline,
)
from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    HiddenTest,
    HiddenTestFile,
    TreeState,
)
from swe_forge.forge.oracle.teacher_evidence import (
    append_execution,
    evidence_calls,
    gate_evidence,
    teacher_gate_failure_reason,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from swe_forge.execution.docker_client import DockerClient

#: How many test-strengthening rounds the gate attempts before rejecting.
DEFAULT_MAX_STRENGTHEN_ROUNDS = 3
#: Default number of plausible-wrong variants the teacher generator proposes.
DEFAULT_NUM_VARIANTS = 3

# Attributable reject reason prefixes (stable keys the contract/CLI gate on). Both
# begin with ``differential`` so a reject is always traceable to this gate.
REASON_DIFFERENTIAL_SURVIVOR = "differential_indistinguishable_variant"
REASON_DIFFERENTIAL_GOLD_NOT_GREEN = "differential_gold_not_green"
REASON_DIFFERENTIAL_NO_EXECUTABLE = "differential_no_executable_teacher_proposals"


class DifferentialError(RuntimeError):
    """Raised for an unrecoverable failure while driving the differential gate."""


@dataclass(frozen=True)
class VariantFile:
    """One source file overwrite that materializes a plausible-wrong variant."""

    path: str
    content: str


@dataclass(frozen=True)
class Variant:
    """A plausible-but-wrong implementation of the gold code.

    ``files`` overwrite the gold source on top of a pristine (gold) checkout to
    produce the variant tree; ``description`` is a short human-readable note of
    the injected mistake (used to guide test-strengthening). A variant is
    expected to be *behaviorally divergent* from gold yet realistic (the kind of
    subtly-wrong patch a strong model might submit).
    """

    variant_id: str
    files: tuple[VariantFile, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.variant_id).strip():
            raise ModelError("Variant.variant_id must be non-empty")
        if not self.files or any(not str(f.path).strip() for f in self.files):
            raise ModelError(
                "Variant.files must be a non-empty list of file overwrites"
            )

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)


@dataclass(frozen=True)
class VariantScore:
    """The outcome of running the F2P+P2P suite against one tree (gold/variant)."""

    f2p_passed: bool
    p2p_passed: bool
    failing_test_ids: tuple[str, ...] = ()

    @property
    def passes_suite(self) -> bool:
        """``True`` iff every F2P test passed AND the P2P/regression suite is green.

        For a variant this means *survivor* (bad); for gold it means *accepted*
        (the required by-construction guarantee).
        """
        return self.f2p_passed and self.p2p_passed

    def summary(self) -> dict[str, object]:
        return {
            "f2p_passed": self.f2p_passed,
            "p2p_passed": self.p2p_passed,
            "passes_suite": self.passes_suite,
            "failing_test_ids": list(self.failing_test_ids),
        }


class DifferentialRunner(Protocol):
    """The scoring surface the gate drives (Docker-backed in production).

    ``exclude`` is the set of base-suite test ids to skip for a run: the gate uses
    it to drop an inherited SYNTHESIZED discriminator (a later-gate survivor-killer
    in ``test_files`` but NOT in ``fail_to_pass``) that a mis-modeling teacher left
    red-on-gold, so an invalid discriminator can no longer make gold fail and
    reject the candidate. ``discardable_tests`` are exactly those discardable
    inherited discriminators (the established F2P and the P2P suite are never
    discardable - if they are red on gold that is a genuine ``gold_not_green``).
    """

    @property
    def language(self) -> str: ...

    @property
    def discardable_tests(self) -> tuple[HiddenTest, ...]: ...

    async def score_gold(
        self, extra_tests: Sequence[HiddenTest], *, exclude: Collection[str] = ()
    ) -> VariantScore: ...

    async def score_variant(
        self,
        variant: Variant,
        extra_tests: Sequence[HiddenTest],
        *,
        exclude: Collection[str] = (),
    ) -> VariantScore: ...

    async def read_sources(self) -> dict[str, str]: ...


@dataclass
class DifferentialSynthesisContext:
    """Inputs handed to a :class:`VariantStrengthSynthesizer` for one round.

    ``gold_sources`` maps each gold target file to its content, ``survivors`` are
    the plausible-wrong variants still passing the suite, and
    ``existing_test_paths`` are the paths already added this run (so a synthesizer
    never collides).
    """

    candidate: Candidate
    adapter: LanguageAdapter
    gold_sources: dict[str, str]
    survivors: tuple[Variant, ...]
    round_index: int
    existing_test_paths: tuple[str, ...] = ()

    @property
    def language(self) -> str:
        return self.candidate.language


class VariantStrengthSynthesizer(Protocol):
    """Proposes extra hidden tests that separate a surviving variant from gold."""

    async def __call__(self, ctx: DifferentialSynthesisContext) -> list[HiddenTest]: ...


class NullVariantSynthesizer:
    """A synthesizer that proposes nothing (offline/deterministic default)."""

    async def __call__(self, ctx: DifferentialSynthesisContext) -> list[HiddenTest]:
        return []


@dataclass
class VariantGenerationContext:
    """Inputs handed to a :class:`VariantGenerator`."""

    candidate: Candidate
    adapter: LanguageAdapter
    gold_sources: dict[str, str]
    num_variants: int = DEFAULT_NUM_VARIANTS

    @property
    def language(self) -> str:
        return self.candidate.language


class VariantGenerator(Protocol):
    """Proposes plausible-but-wrong variants of the gold code (PatchDiff-style)."""

    async def __call__(self, ctx: VariantGenerationContext) -> list[Variant]: ...


class NullVariantGenerator:
    """A generator that proposes nothing (offline/deterministic default)."""

    async def __call__(self, ctx: VariantGenerationContext) -> list[Variant]:
        return []


@dataclass
class DifferentialOutcome:
    """The result of the differential gate (folded into an :class:`OracleReport`)."""

    verdict: str
    reasons: list[str]
    differential_pass: bool
    variants_total: int
    variants_killed: int
    added_tests: list[HiddenTest] = field(default_factory=list)
    discarded_test_paths: list[str] = field(default_factory=list)
    rounds: int = 0
    survivors: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        return self.verdict == "pass"


async def assess_differential(
    runner: DifferentialRunner,
    variants: Sequence[Variant],
    *,
    synthesizer: VariantStrengthSynthesizer | None = None,
    context_template: DifferentialSynthesisContext | None = None,
    max_rounds: int = DEFAULT_MAX_STRENGTHEN_ROUNDS,
) -> DifferentialOutcome:
    """Run the differential gate and auto-strengthen until variants are separated.

    Before judging distinguishability the gate VETS the established gold suite: a
    mis-modeling teacher may have left an inherited SYNTHESIZED discriminator (a
    later-gate survivor-killer, in ``test_files`` but NOT in ``fail_to_pass``) that
    is RED on gold. Such a test is an invalid discriminator (a TEACHER error), not
    evidence the candidate is bad, so it is DISCARDED (and pruned from the shipped
    suite) rather than counted as a ``differential_gold_not_green`` reject of the
    candidate. Only when gold still fails after every discardable discriminator is
    dropped - i.e. an established F2P or the P2P/regression suite is itself red on
    gold - does the gate reject ``differential_gold_not_green`` (case B, a genuine
    gold-not-green).

    Then it scores every variant. Survivors (a variant that passes the full suite)
    drive up to ``max_rounds`` strengthening rounds; in each round the
    ``synthesizer`` proposes extra tests and every proposal is confirmed by
    execution - kept only if it keeps gold green AND makes a surviving variant
    fail (the bounded re-synthesis of VALID gold-green discriminators). Rejects
    (``differential_pass == False``) when a survivor cannot be separated from gold
    within the bounded loop (``differential_indistinguishable_variant``) - the
    gate's purpose stays intact.
    """
    discardable = tuple(getattr(runner, "discardable_tests", ()) or ())
    discardable_by_id = {t.test_id: t for t in discardable}
    exclude: set[str] = set()

    gold_base = await runner.score_gold([])
    if not gold_base.passes_suite and discardable:
        # A mis-modeled inherited discriminator may be red on gold. Drop the
        # discardable discriminators one at a time (bisecting toward the genuine
        # cause) and re-vet gold; keep only the drops that are actually needed to
        # make gold green. If gold is green with NONE excluded the failure is a
        # protected test (established F2P / P2P) and stays a real reject.
        failing = set(gold_base.failing_test_ids)
        for test in discardable:
            if test.test_id in failing or not failing:
                exclude.add(test.test_id)
                trial = await runner.score_gold([], exclude=exclude)
                gold_base = trial
                if trial.passes_suite:
                    break
                failing = set(trial.failing_test_ids)
        # Minimize the exclusion set: keep only discriminators whose removal is
        # required for gold to pass (so a still-red discriminator is not blamed).
        if gold_base.passes_suite and exclude:
            minimal: set[str] = set()
            for test_id in list(exclude):
                probe = await runner.score_gold([], exclude=exclude - {test_id})
                if not probe.passes_suite:
                    minimal.add(test_id)
            exclude = minimal
            gold_base = await runner.score_gold([], exclude=exclude)

    discarded = sorted(
        {
            f.path
            for tid in exclude
            for f in discardable_by_id.get(tid, HiddenTest(tid)).files
        }
    )
    round_records: list[dict[str, object]] = []
    details: dict[str, object] = {
        "stage": "differential",
        "variants_total": len(variants),
        "gold_base": gold_base.summary(),
        "discarded_discriminators": discarded,
        "rounds": round_records,
    }

    if not gold_base.passes_suite:
        return DifferentialOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_DIFFERENTIAL_GOLD_NOT_GREEN}: the gold tree did not pass "
                f"its own established F2P+P2P suite (failing: "
                f"{list(gold_base.failing_test_ids)}) even after discarding "
                f"{len(discarded)} mis-modeled synthesized discriminator(s); the "
                "differential gate cannot establish uniqueness"
            ],
            differential_pass=False,
            variants_total=len(variants),
            variants_killed=0,
            discarded_test_paths=discarded,
            details=details,
        )

    accepted: list[HiddenTest] = []
    variant_scores: dict[str, VariantScore] = {}
    survivors: list[Variant] = []
    for variant in variants:
        score = await runner.score_variant(variant, [], exclude=exclude)
        variant_scores[variant.variant_id] = score
        if score.passes_suite:
            survivors.append(variant)
    details["initial"] = {
        "killed": [vid for vid, s in variant_scores.items() if not s.passes_suite],
        "survivors": [v.variant_id for v in survivors],
        "per_variant": {vid: s.summary() for vid, s in variant_scores.items()},
    }

    if not variants:
        return DifferentialOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_DIFFERENTIAL_NO_EXECUTABLE}: no plausible-but-wrong "
                "teacher proposal was executable through the differential gate"
            ],
            differential_pass=False,
            variants_total=0,
            variants_killed=0,
            discarded_test_paths=discarded,
            details=details,
        )

    if survivors and synthesizer is not None:
        if context_template is None:
            raise DifferentialError(
                "a synthesis context template is required when a synthesizer is set"
            )
        gold_sources: dict[str, str] = {}
        with contextlib.suppress(Exception):
            gold_sources = await runner.read_sources()

        for round_index in range(1, max_rounds + 1):
            ctx = dataclasses.replace(
                context_template,
                gold_sources=gold_sources,
                survivors=tuple(survivors),
                round_index=round_index,
                existing_test_paths=tuple(f.path for t in accepted for f in t.files),
            )
            proposals = await synthesizer(ctx)
            accepted_this_round = 0
            for proposal in proposals:
                gold_trial = await runner.score_gold(
                    [*accepted, proposal], exclude=exclude
                )
                if not gold_trial.passes_suite:
                    continue
                newly_killed: list[Variant] = []
                for variant in survivors:
                    trial = await runner.score_variant(
                        variant, [*accepted, proposal], exclude=exclude
                    )
                    if not trial.passes_suite:
                        newly_killed.append(variant)
                        variant_scores[variant.variant_id] = trial
                if newly_killed:
                    accepted.append(proposal)
                    killed_ids = {v.variant_id for v in newly_killed}
                    survivors = [v for v in survivors if v.variant_id not in killed_ids]
                    accepted_this_round += 1
                    if not survivors:
                        break
            round_records.append(
                {
                    "round": round_index,
                    "proposed": len(proposals),
                    "accepted": accepted_this_round,
                    "survivors_after": [v.variant_id for v in survivors],
                }
            )
            if not survivors or accepted_this_round == 0:
                break

    variants_total = len(variants)
    variants_killed = variants_total - len(survivors)
    details["final"] = {
        "survivors": [v.variant_id for v in survivors],
        "variants_killed": variants_killed,
    }
    details["added_test_paths"] = [f.path for t in accepted for f in t.files]

    if not survivors:
        return DifferentialOutcome(
            verdict="pass",
            reasons=[],
            differential_pass=True,
            variants_total=variants_total,
            variants_killed=variants_killed,
            added_tests=accepted,
            discarded_test_paths=discarded,
            rounds=len(round_records),
            survivors=[],
            details=details,
        )

    survivor_ids = [v.variant_id for v in survivors]
    reason = (
        f"{REASON_DIFFERENTIAL_SURVIVOR}: {len(survivors)} plausible-but-wrong "
        f"variant(s) {survivor_ids} still pass the full F2P+P2P suite after "
        f"{len(round_records)} strengthening round(s); the oracle cannot "
        "distinguish them from gold (a wrong behavior would be accepted)"
    )
    return DifferentialOutcome(
        verdict="reject",
        reasons=[reason],
        differential_pass=False,
        variants_total=variants_total,
        variants_killed=variants_killed,
        added_tests=accepted,
        discarded_test_paths=discarded,
        rounds=len(round_records),
        survivors=survivor_ids,
        details=details,
    )


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def _tool_versions(extra: dict[str, str] | None = None) -> dict[str, str]:
    versions: dict[str, str] = {}
    with contextlib.suppress(metadata.PackageNotFoundError):
        versions["litellm"] = metadata.version("litellm")
    if extra:
        versions.update(extra)
    return versions


def build_differential_report(
    candidate: Candidate,
    prior_report: OracleReport,
    outcome: DifferentialOutcome,
    *,
    env_image: EnvImage | None = None,
    extra_details: dict[str, object] | None = None,
) -> OracleReport:
    """Fold a :class:`DifferentialOutcome` into the running :class:`OracleReport`.

    Carries the establish + flakiness + mutation fields forward, sets
    ``differential_pass``, appends any survivor-killing tests to ``test_files``,
    PRUNES any inherited synthesized discriminator the gate discarded as
    red-on-gold (so the shipped suite stays gold=100%), and sets the terminal
    verdict (``pass`` when every variant is separated from gold; ``reject`` with
    an attributable differential reason - a surviving indistinguishable variant,
    or a genuine gold-not-green after discards - otherwise).
    """
    details: dict[str, object] = dict(prior_report.details)
    details["differential"] = outcome.details
    # Differential may append survivor-killers or prune a red-on-gold inherited
    # discriminator. Either change invalidates the earlier mutation evidence;
    # the pipeline must remeasure the final suite without synthesis.
    if prior_report.final_mutation_evidence is not None:
        details["mutation_evidence_invalidated"] = {
            "stage": "differential",
            "reason": "hidden_suite_changed",
        }
    if env_image is not None:
        details.setdefault("env_image", env_image.image_tag)
    if extra_details:
        teacher_gates = extra_details.get("teacher_gates")
        if isinstance(teacher_gates, dict):
            inherited = details.get("teacher_gates")
            merged = dict(inherited) if isinstance(inherited, dict) else {}
            merged.update(teacher_gates)
            details["teacher_gates"] = merged
        details.update(
            {
                key: value
                for key, value in extra_details.items()
                if key != "teacher_gates"
            }
        )

    discarded = set(outcome.discarded_test_paths)
    test_files = [tf for tf in prior_report.test_files if tf.path not in discarded]
    seen = {tf.path for tf in test_files}
    for test in outcome.added_tests:
        for file in test.files:
            if file.path in seen:
                continue
            seen.add(file.path)
            test_files.append(
                OracleTestFile(
                    path=file.path, content=file.content, origin="synthesized"
                )
            )

    base_prov = prior_report.provenance
    provenance = Provenance(
        generator=candidate.generator,
        seed=candidate.provenance.seed,
        language=candidate.language,
        tool_versions=dict(base_prov.tool_versions) if base_prov else _tool_versions(),
        details={
            "stage": "oracle.differential",
            "variants_total": outcome.variants_total,
            "variants_killed": outcome.variants_killed,
            "differential_pass": outcome.differential_pass,
            "strengthen_rounds": outcome.rounds,
            "added_tests": [f.path for t in outcome.added_tests for f in t.files],
            "discarded_discriminators": list(outcome.discarded_test_paths),
            "survivors": list(outcome.survivors),
        },
    )

    return OracleReport(
        language=prior_report.language,
        generator=prior_report.generator,
        verdict=outcome.verdict,
        reasons=list(outcome.reasons),
        fail_to_pass=list(prior_report.fail_to_pass),
        pass_to_pass=list(prior_report.pass_to_pass),
        test_files=test_files,
        flakiness_runs=prior_report.flakiness_runs,
        mutants_total=prior_report.mutants_total,
        mutants_killed=prior_report.mutants_killed,
        final_mutation_evidence=None,
        differential_pass=outcome.differential_pass,
        provenance=provenance,
        details=details,
    )


# --------------------------------------------------------------------------- #
# Docker-backed runner + top-level gate
# --------------------------------------------------------------------------- #
def reconstruct_suite_tests(
    adapter: LanguageAdapter,
    fail_to_pass: Sequence[str],
    test_files: Sequence[OracleTestFile],
) -> list[HiddenTest]:
    """Rebuild the full hidden suite (F2P + later-gate tests) as runnable tests.

    Each established F2P entry keeps its exact selection command; any other hidden
    test file (e.g. a mutation-gate survivor-killing test, recorded in
    ``test_files`` but not in ``fail_to_pass``) is given a selection command via
    the adapter so the differential gate runs the entire hidden suite against
    gold and each variant.
    """
    tests: list[HiddenTest] = []
    for tf in test_files:
        if not tf.content:
            continue
        test_id: str | None = None
        for cmd in fail_to_pass:
            if tf.path and tf.path in cmd:
                test_id = cmd
                break
        if test_id is None:
            test_id = adapter.test_command((tf.path,))
        tests.append(
            HiddenTest(
                test_id=test_id,
                files=(HiddenTestFile(path=tf.path, content=tf.content),),
                origin=tf.origin,
            )
        )
    return tests


class DockerDifferentialRunner:
    """A :class:`DifferentialRunner` that scores via throwaway Docker sandboxes.

    Each scoring opens a fresh ``--rm`` :class:`~swe_forge.execution.sandbox.DockerSandbox`
    on the candidate's green ``EnvImage`` (the repo already checked out, pristine =
    gold), optionally overwrites the gold source with a variant's files, writes the
    established + extra hidden tests, and runs the F2P + P2P suite through the shared
    :class:`~swe_forge.forge.oracle.establish.DockerOracleRecipe`.
    """

    def __init__(
        self,
        candidate: Candidate,
        env_image: EnvImage,
        adapter: LanguageAdapter,
        *,
        base_tests: Sequence[HiddenTest] = (),
        fail_to_pass: Sequence[str] = (),
        p2p_command: str = "",
        command_timeout: float = 600.0,
        docker_client: "DockerClient | None" = None,
    ) -> None:
        self._candidate = candidate
        self._env_image = env_image
        self._adapter = adapter
        self._base_tests = list(base_tests)
        # A base test is DISCARDABLE iff it is not an established F2P (its id is not
        # in ``fail_to_pass``): those are the later-gate SYNTHESIZED discriminators
        # (mutation-gate survivor-killers) that were never gold-vetted. The
        # established F2P (gold-confirmed by establish) and the P2P suite are
        # PROTECTED - if they are red on gold that is a genuine gold-not-green.
        ftp = set(fail_to_pass)
        self._discardable = tuple(t for t in self._base_tests if t.test_id not in ftp)
        self._p2p_command = p2p_command or env_image.baseline_test_command
        self._timeout = command_timeout
        self._docker_client = docker_client

    @property
    def language(self) -> str:
        return self._candidate.language

    @property
    def discardable_tests(self) -> tuple[HiddenTest, ...]:
        return self._discardable

    @property
    def _target_files(self) -> list[str]:
        return [
            f for f in self._candidate.target.files if not self._adapter.is_test_file(f)
        ]

    @contextlib.asynccontextmanager
    async def _recipe(self) -> "AsyncIterator[DockerOracleRecipe]":
        from swe_forge.execution.docker_client import DockerClient
        from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

        client = self._docker_client or DockerClient()
        config = SandboxConfig(
            name="swe-forge-oracle-differential",
            image=self._env_image.image_tag,
            workspace_dir=self._env_image.workspace_dir,
            command_timeout=self._timeout,
        )
        sandbox = DockerSandbox(client, config)
        async with sandbox:
            yield DockerOracleRecipe(
                sandbox,
                language=self._candidate.language,
                workspace_dir=self._env_image.workspace_dir,
                mutation_patch=self._candidate.mutation_patch,
                oracle_patch=self._candidate.oracle_patch,
                p2p_command=self._p2p_command,
                command_timeout=self._timeout,
            )

    async def _score(
        self,
        variant_files: Sequence[VariantFile],
        extra_tests: Sequence[HiddenTest],
        *,
        exclude: Collection[str] = (),
    ) -> VariantScore:
        excluded = set(exclude)
        async with self._recipe() as recipe:
            # The EnvImage checkout is pristine = gold; overwrite target files to
            # materialize a variant (no-op when scoring gold).
            await recipe.set_state(TreeState.GOLD)
            for file in variant_files:
                await recipe.sandbox.write_file(file.path, file.content)

            # P2P/regression runs with NO hidden test present (a fail-on-variant
            # hidden test must not make the repo's own suite look red).
            p2p = await recipe.run_p2p()

            failing: list[str] = []
            f2p_passed = True
            for test in [*self._base_tests, *extra_tests]:
                if test.test_id in excluded:
                    continue
                await recipe.write_test(test)
                run = await recipe.run_test(test)
                await recipe.remove_test(test)
                if not run.passed:
                    failing.append(test.test_id)
                    f2p_passed = False

            if not p2p.passed:
                failing.append(recipe.p2p_command)
            return VariantScore(
                f2p_passed=f2p_passed,
                p2p_passed=p2p.passed,
                failing_test_ids=tuple(failing),
            )

    async def score_gold(
        self, extra_tests: Sequence[HiddenTest], *, exclude: Collection[str] = ()
    ) -> VariantScore:
        return await self._score((), extra_tests, exclude=exclude)

    async def score_variant(
        self,
        variant: Variant,
        extra_tests: Sequence[HiddenTest],
        *,
        exclude: Collection[str] = (),
    ) -> VariantScore:
        return await self._score(variant.files, extra_tests, exclude=exclude)

    async def read_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        async with self._recipe() as recipe:
            await recipe.set_state(TreeState.GOLD)
            for path in self._target_files:
                with contextlib.suppress(Exception):
                    sources[path] = await recipe.sandbox.read_file(path)
        return sources


async def run_differential_gate(
    candidate: Candidate,
    env_image: EnvImage,
    prior_report: OracleReport,
    *,
    variant_generator: VariantGenerator | None = None,
    synthesizer: VariantStrengthSynthesizer | None = None,
    num_variants: int = DEFAULT_NUM_VARIANTS,
    max_rounds: int = DEFAULT_MAX_STRENGTHEN_ROUNDS,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 600.0,
) -> OracleReport:
    """Run the differential-vs-gold gate in throwaway Docker sandboxes.

    A green baseline is a hard precondition and the prior gate (mutation) must
    have passed. Builds a :class:`DockerDifferentialRunner`, asks the
    ``variant_generator`` for plausible-wrong variants, scores them with bounded
    test-strengthening, and returns the extended :class:`OracleReport` (with
    ``differential_pass`` set).
    """
    require_green_baseline(env_image)
    if prior_report.verdict != "pass":
        raise DifferentialError(
            "differential gate requires a passing prior (mutation) report; got "
            f"verdict {prior_report.verdict!r}"
        )

    if adapter is None:
        adapter = build_default_registry().get(candidate.language)

    base_tests = reconstruct_suite_tests(
        adapter, prior_report.fail_to_pass, prior_report.test_files
    )
    p2p_command = (
        prior_report.pass_to_pass[0]
        if prior_report.pass_to_pass
        else env_image.baseline_test_command
    )
    runner = DockerDifferentialRunner(
        candidate,
        env_image,
        adapter,
        base_tests=base_tests,
        fail_to_pass=prior_report.fail_to_pass,
        p2p_command=p2p_command,
        command_timeout=command_timeout,
        docker_client=docker_client,
    )

    variants: list[Variant] = []
    teacher_calls = []
    if variant_generator is not None:
        gold_sources = await runner.read_sources()
        gen_ctx = VariantGenerationContext(
            candidate=candidate,
            adapter=adapter,
            gold_sources=gold_sources,
            num_variants=num_variants,
        )
        variants = await variant_generator(gen_ctx)
        teacher_calls.extend(evidence_calls(variant_generator))

    template = DifferentialSynthesisContext(
        candidate=candidate,
        adapter=adapter,
        gold_sources={},
        survivors=(),
        round_index=0,
    )
    outcome = await assess_differential(
        runner,
        variants,
        synthesizer=synthesizer,
        context_template=template,
        max_rounds=max_rounds,
    )
    teacher_calls.extend(evidence_calls(synthesizer))
    append_execution(
        teacher_calls,
        call_kind="proposal",
        attempted=outcome.variants_total,
        completed=outcome.variants_total,
        executable=outcome.variants_total,
    )
    if not variants:
        outcome.reasons = [teacher_gate_failure_reason("differential", teacher_calls)]
    return build_differential_report(
        candidate,
        prior_report,
        outcome,
        env_image=env_image,
        extra_details={"teacher_gates": {"differential": gate_evidence(teacher_calls)}},
    )
