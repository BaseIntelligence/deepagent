"""Alt-correct false-negative gate: a correct alternative must be ACCEPTED.

The fifth oracle-hardening gate (architecture S6, Stage 3.5). Establish +
flakiness + mutation + differential prove the hidden suite is a *deterministic,
mutation-adequate, gold-unique* FAIL->PASS contract - but a suite hardened that
hard can swing the other way and become **over-fit**: it can encode an
implementation detail of the *gold* code (a private symbol name, a specific data
structure, an incidental output) so tightly that a *genuinely-correct but
differently-written* solution is wrongly FAILED. That is a false-negative the
solver would suffer through no fault of its own. This gate closes that gap.

Flow (``the teacher proposes, deterministic execution disposes``):

1. The teacher writes 1-2 **genuinely-correct alternative** implementations of
   the gold code: different internal style and different private symbol names,
   but honoring the published ``interface_block`` signatures (so they import and
   run against the same public surface). These are PROPOSALS.
2. Confirm the **gold** tree still passes its own established F2P+P2P suite (the
   by-construction guarantee; a defensive re-check at this gate).
3. Score every alternative through the SAME Docker F2P/P2P primitives. A correct
   alternative MUST pass the full suite (``alt_correct_accepted == True``). The
   Interface pinning is what makes this possible: a correct solution is never
   failed merely for naming the symbols differently.
4. If an alternative FAILS, the suite is **over-fit** (a false-negative). The
   conservative, correctness-first default is to **reject** the candidate with an
   attributable ``alt_correct`` reason (never ship an over-fit suite silently).
   Optionally (``relax=True``) the gate may instead drop the offending over-fit
   hidden test(s) - but only when doing so leaves an F2P transition intact and
   makes every alternative pass - recording the relax action; if it cannot relax
   safely it rejects.

The gate scores through :class:`~swe_forge.forge.oracle.establish.DockerOracleRecipe`
so "F2P pass AND P2P green" means exactly the same thing everywhere, and it honors
the Python re-test ``.pyc`` determinism invariant baked into the recipe.
"""

from __future__ import annotations

import contextlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Protocol

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    GeneratedSpec,
    ModelError,
    OracleReport,
    Provenance,
    require_green_baseline,
)
from swe_forge.forge.oracle.differential import reconstruct_suite_tests
from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    HiddenTest,
    TreeState,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from swe_forge.execution.docker_client import DockerClient

#: Default number of genuinely-correct alternatives the teacher proposes.
DEFAULT_NUM_ALTERNATIVES = 2

# Attributable reject reason prefixes (stable keys the contract/CLI gate on). Both
# begin with ``alt_correct`` so a reject is always traceable to this gate.
REASON_ALT_CORRECT_OVERFIT = "alt_correct_overfit"
REASON_ALT_CORRECT_GOLD_NOT_GREEN = "alt_correct_gold_not_green"


class AltCorrectError(RuntimeError):
    """Raised for an unrecoverable failure while driving the alt-correct gate."""


@dataclass(frozen=True)
class AltImplFile:
    """One source file overwrite that materializes an alternative implementation."""

    path: str
    content: str


@dataclass(frozen=True)
class AltImpl:
    """A genuinely-correct alternative implementation of the gold code.

    ``files`` overwrite the gold source on top of a pristine (gold) checkout to
    produce the alternative tree; ``description`` is a short human-readable note.
    An alternative is *behaviorally equivalent* to gold (correct) yet written in a
    different internal style with different private symbol names, honoring the
    published Interface signatures so it exposes the same public surface.
    """

    impl_id: str
    files: tuple[AltImplFile, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.impl_id).strip():
            raise ModelError("AltImpl.impl_id must be non-empty")
        if not self.files or any(not str(f.path).strip() for f in self.files):
            raise ModelError(
                "AltImpl.files must be a non-empty list of file overwrites"
            )

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.files)


@dataclass(frozen=True)
class AltScore:
    """The outcome of running the F2P+P2P suite against one tree (gold/alt)."""

    f2p_passed: bool
    p2p_passed: bool
    failing_test_ids: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        """``True`` iff every F2P test passed AND the P2P/regression suite is green.

        For an alternative this means *accepted* (good - the correct alternative
        is not falsely rejected); for gold it means *accepted* (the required
        by-construction guarantee).
        """
        return self.f2p_passed and self.p2p_passed

    def summary(self) -> dict[str, object]:
        return {
            "f2p_passed": self.f2p_passed,
            "p2p_passed": self.p2p_passed,
            "accepted": self.accepted,
            "failing_test_ids": list(self.failing_test_ids),
        }


class AltCorrectRunner(Protocol):
    """The scoring surface the gate drives (Docker-backed in production).

    ``exclude`` is the set of hidden-test ids to skip for a run (used only by the
    optional relax path, to re-score with the over-fit test(s) removed).
    """

    @property
    def language(self) -> str: ...

    async def score_gold(self, exclude: Collection[str] = ()) -> AltScore: ...

    async def score_alt(
        self, alt: AltImpl, exclude: Collection[str] = ()
    ) -> AltScore: ...

    async def read_sources(self) -> dict[str, str]: ...


@dataclass
class AltCorrectGenerationContext:
    """Inputs handed to an :class:`AltCorrectGenerator`.

    ``gold_sources`` maps each gold target file to its content, and
    ``interface_block`` is the published Interface (expected public
    names/signatures) the alternatives must honor so a correct solution is not
    failed for a naming difference.
    """

    candidate: Candidate
    adapter: LanguageAdapter
    gold_sources: dict[str, str]
    interface_block: str = ""
    num_alternatives: int = DEFAULT_NUM_ALTERNATIVES

    @property
    def language(self) -> str:
        return self.candidate.language


class AltCorrectGenerator(Protocol):
    """Proposes genuinely-correct alternative implementations of the gold code."""

    async def __call__(self, ctx: AltCorrectGenerationContext) -> list[AltImpl]: ...


class NullAltCorrectGenerator:
    """A generator that proposes nothing (offline/deterministic default)."""

    async def __call__(self, ctx: AltCorrectGenerationContext) -> list[AltImpl]:
        return []


@dataclass
class AltCorrectOutcome:
    """The result of the alt-correct gate (folded into an :class:`OracleReport`)."""

    verdict: str
    reasons: list[str]
    alt_correct_accepted: bool
    alternatives_total: int
    alternatives_accepted: int
    rejected: list[str] = field(default_factory=list)
    relaxed: bool = False
    relaxed_test_ids: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        return self.verdict == "pass"


def _overfit_reason(rejected: Sequence[tuple[AltImpl, AltScore]]) -> str:
    """Build the attributable over-fit reject reason citing the failing alts."""
    parts = []
    for alt, score in rejected:
        parts.append(f"{alt.impl_id} (failed: {list(score.failing_test_ids)})")
    return (
        f"{REASON_ALT_CORRECT_OVERFIT}: {len(rejected)} genuinely-correct "
        f"alternative(s) {parts} were FAILED by the F2P+P2P suite; the suite is "
        "over-fit to the gold implementation (a false-negative) and must be "
        "relaxed or the candidate rejected"
    )


async def assess_alt_correct(
    runner: AltCorrectRunner,
    alternatives: Sequence[AltImpl],
    *,
    fail_to_pass: Sequence[str] = (),
    relax: bool = False,
) -> AltCorrectOutcome:
    """Run the alt-correct gate: every correct alternative must be accepted.

    Confirms gold passes its own suite, then scores every alternative. When every
    alternative is accepted the gate passes (``alt_correct_accepted == True``).
    When an alternative is FAILED the suite is over-fit: the default is to reject
    with an attributable reason; with ``relax=True`` the gate instead drops the
    offending over-fit hidden test(s) when that leaves an F2P transition intact
    and makes every alternative pass (a recorded relax action), else rejects.
    """
    gold_base = await runner.score_gold()
    details: dict[str, object] = {
        "stage": "alt_correct",
        "alternatives_total": len(alternatives),
        "gold_base": gold_base.summary(),
        "relax_enabled": relax,
    }

    if not gold_base.accepted:
        return AltCorrectOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_ALT_CORRECT_GOLD_NOT_GREEN}: the gold tree did not pass "
                f"its own established F2P+P2P suite (failing: "
                f"{list(gold_base.failing_test_ids)}); the alt-correct gate cannot "
                "assess false-negatives"
            ],
            alt_correct_accepted=False,
            alternatives_total=len(alternatives),
            alternatives_accepted=0,
            details=details,
        )

    per_alt: dict[str, AltScore] = {}
    rejected: list[tuple[AltImpl, AltScore]] = []
    for alt in alternatives:
        score = await runner.score_alt(alt)
        per_alt[alt.impl_id] = score
        if not score.accepted:
            rejected.append((alt, score))
    accepted_count = len(alternatives) - len(rejected)
    details["initial"] = {
        "accepted": [aid for aid, s in per_alt.items() if s.accepted],
        "rejected": [a.impl_id for a, _ in rejected],
        "per_alt": {aid: s.summary() for aid, s in per_alt.items()},
    }

    if not rejected:
        return AltCorrectOutcome(
            verdict="pass",
            reasons=[],
            alt_correct_accepted=True,
            alternatives_total=len(alternatives),
            alternatives_accepted=accepted_count,
            rejected=[],
            details=details,
        )

    if not relax:
        return AltCorrectOutcome(
            verdict="reject",
            reasons=[_overfit_reason(rejected)],
            alt_correct_accepted=False,
            alternatives_total=len(alternatives),
            alternatives_accepted=accepted_count,
            rejected=[a.impl_id for a, _ in rejected],
            details=details,
        )

    return await _attempt_relax(
        runner, alternatives, rejected, accepted_count, fail_to_pass, details
    )


async def _attempt_relax(
    runner: AltCorrectRunner,
    alternatives: Sequence[AltImpl],
    rejected: Sequence[tuple[AltImpl, AltScore]],
    accepted_count: int,
    fail_to_pass: Sequence[str],
    details: dict[str, object],
) -> AltCorrectOutcome:
    """Try to drop the over-fit hidden test(s) so correct alternatives pass.

    Only HIDDEN-test (F2P/added) failures are relaxable: an alternative that fails
    the P2P/regression suite broke the repo's own tests, so it is not a
    false-negative and the baseline cannot be relaxed. A relax never removes the
    last F2P transition. The relax succeeds only if, after excluding the over-fit
    tests, gold still passes and every alternative is accepted.
    """
    total = len(alternatives)
    rejected_ids = [a.impl_id for a, _ in rejected]

    p2p_failures = [a.impl_id for a, s in rejected if not s.p2p_passed]
    if p2p_failures:
        details["relax"] = {
            "attempted": False,
            "reason": "p2p_failure_not_relaxable",
            "p2p_failing_alts": p2p_failures,
        }
        return AltCorrectOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_ALT_CORRECT_OVERFIT}: alternative(s) {p2p_failures} fail "
                "the P2P/regression suite (they break the repo's own tests, not a "
                "hidden-test over-fit); the baseline cannot be relaxed -> reject"
            ],
            alt_correct_accepted=False,
            alternatives_total=total,
            alternatives_accepted=accepted_count,
            rejected=rejected_ids,
            details=details,
        )

    overfit_ids = sorted({tid for _, s in rejected for tid in s.failing_test_ids})
    f2p_set = set(fail_to_pass)
    remaining_f2p = [tid for tid in fail_to_pass if tid not in overfit_ids]
    if f2p_set and not remaining_f2p:
        details["relax"] = {
            "attempted": True,
            "succeeded": False,
            "overfit_test_ids": overfit_ids,
            "reason": "would_remove_last_f2p",
        }
        return AltCorrectOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_ALT_CORRECT_OVERFIT}: relaxing the over-fit test(s) "
                f"{overfit_ids} would remove the last F2P transition; the oracle "
                "would lose its discriminating test -> reject"
            ],
            alt_correct_accepted=False,
            alternatives_total=total,
            alternatives_accepted=accepted_count,
            rejected=rejected_ids,
            details=details,
        )

    gold_relaxed = await runner.score_gold(exclude=overfit_ids)
    still_failing: list[str] = []
    for alt, _ in rejected:
        relaxed_score = await runner.score_alt(alt, exclude=overfit_ids)
        if not relaxed_score.accepted:
            still_failing.append(alt.impl_id)

    if gold_relaxed.accepted and not still_failing:
        details["relax"] = {
            "attempted": True,
            "succeeded": True,
            "relaxed_test_ids": overfit_ids,
            "remaining_fail_to_pass": remaining_f2p,
        }
        return AltCorrectOutcome(
            verdict="pass",
            reasons=[],
            alt_correct_accepted=True,
            alternatives_total=total,
            alternatives_accepted=total,
            rejected=[],
            relaxed=True,
            relaxed_test_ids=overfit_ids,
            details=details,
        )

    details["relax"] = {
        "attempted": True,
        "succeeded": False,
        "overfit_test_ids": overfit_ids,
        "gold_after_relax": gold_relaxed.summary(),
        "still_failing": still_failing,
        "reason": "relax_insufficient",
    }
    return AltCorrectOutcome(
        verdict="reject",
        reasons=[
            f"{REASON_ALT_CORRECT_OVERFIT}: alternative(s) {still_failing or rejected_ids} "
            f"still fail after relaxing the over-fit test(s) {overfit_ids}; the "
            "suite cannot be safely relaxed to accept a correct alternative -> reject"
        ],
        alt_correct_accepted=False,
        alternatives_total=total,
        alternatives_accepted=accepted_count,
        rejected=rejected_ids,
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


def _relaxed_paths(
    base_tests: Sequence[HiddenTest], relaxed_test_ids: Collection[str]
) -> set[str]:
    """Map relaxed hidden-test ids back to the file paths they wrote."""
    relaxed = set(relaxed_test_ids)
    paths: set[str] = set()
    for test in base_tests:
        if test.test_id in relaxed:
            paths.update(f.path for f in test.files)
    return paths


def build_alt_correct_report(
    candidate: Candidate,
    prior_report: OracleReport,
    outcome: AltCorrectOutcome,
    *,
    env_image: EnvImage | None = None,
    base_tests: Sequence[HiddenTest] = (),
    extra_details: dict[str, object] | None = None,
) -> OracleReport:
    """Fold an :class:`AltCorrectOutcome` into the running :class:`OracleReport`.

    Carries the establish + flakiness + mutation + differential fields forward,
    sets ``alt_correct_accepted``, and sets the terminal verdict (``pass`` when
    every correct alternative is accepted; ``reject`` with an attributable
    over-fit reason otherwise). When a relax action was recorded the dropped
    over-fit test(s) are removed from ``fail_to_pass``/``test_files``.
    """
    details: dict[str, object] = dict(prior_report.details)
    details["alt_correct"] = outcome.details
    # A relaxation removes hidden tests. Do not let counts from a prior suite
    # survive it; final pipeline remeasurement must bind the retained suite.
    if prior_report.final_mutation_evidence is not None:
        details["mutation_evidence_invalidated"] = {
            "stage": "alt_correct",
            "reason": "hidden_suite_changed",
        }
    if env_image is not None:
        details.setdefault("env_image", env_image.image_tag)
    if extra_details:
        details.update(extra_details)

    fail_to_pass = list(prior_report.fail_to_pass)
    test_files = list(prior_report.test_files)
    if outcome.relaxed and outcome.relaxed_test_ids:
        relaxed_ids = set(outcome.relaxed_test_ids)
        fail_to_pass = [tid for tid in fail_to_pass if tid not in relaxed_ids]
        drop_paths = _relaxed_paths(base_tests, relaxed_ids)
        if drop_paths:
            test_files = [tf for tf in test_files if tf.path not in drop_paths]

    base_prov = prior_report.provenance
    provenance = Provenance(
        generator=candidate.generator,
        seed=candidate.provenance.seed,
        language=candidate.language,
        tool_versions=dict(base_prov.tool_versions) if base_prov else _tool_versions(),
        details={
            "stage": "oracle.alt_correct",
            "alternatives_total": outcome.alternatives_total,
            "alternatives_accepted": outcome.alternatives_accepted,
            "alt_correct_accepted": outcome.alt_correct_accepted,
            "rejected": list(outcome.rejected),
            "relaxed": outcome.relaxed,
            "relaxed_test_ids": list(outcome.relaxed_test_ids),
        },
    )

    return OracleReport(
        language=prior_report.language,
        generator=prior_report.generator,
        verdict=outcome.verdict,
        reasons=list(outcome.reasons),
        fail_to_pass=fail_to_pass,
        pass_to_pass=list(prior_report.pass_to_pass),
        test_files=test_files,
        flakiness_runs=prior_report.flakiness_runs,
        mutants_total=prior_report.mutants_total,
        mutants_killed=prior_report.mutants_killed,
        final_mutation_evidence=None,
        differential_pass=prior_report.differential_pass,
        alt_correct_accepted=outcome.alt_correct_accepted,
        leak_audit=prior_report.leak_audit,
        provenance=provenance,
        details=details,
    )


# --------------------------------------------------------------------------- #
# Docker-backed runner + top-level gate
# --------------------------------------------------------------------------- #
class DockerAltCorrectRunner:
    """An :class:`AltCorrectRunner` that scores via throwaway Docker sandboxes.

    Each scoring opens a fresh ``--rm`` :class:`~swe_forge.execution.sandbox.DockerSandbox`
    on the candidate's green ``EnvImage`` (the repo already checked out, pristine =
    gold), optionally overwrites the gold source with an alternative's files, writes
    the established + later-gate hidden tests, and runs the F2P + P2P suite through
    the shared :class:`~swe_forge.forge.oracle.establish.DockerOracleRecipe`.
    """

    def __init__(
        self,
        candidate: Candidate,
        env_image: EnvImage,
        adapter: LanguageAdapter,
        *,
        base_tests: Sequence[HiddenTest] = (),
        p2p_command: str = "",
        command_timeout: float = 600.0,
        docker_client: "DockerClient | None" = None,
    ) -> None:
        self._candidate = candidate
        self._env_image = env_image
        self._adapter = adapter
        self._base_tests = list(base_tests)
        self._p2p_command = p2p_command or env_image.baseline_test_command
        self._timeout = command_timeout
        self._docker_client = docker_client

    @property
    def language(self) -> str:
        return self._candidate.language

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
            name="swe-forge-oracle-alt-correct",
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
        alt_files: Sequence[AltImplFile],
        exclude: Collection[str],
    ) -> AltScore:
        skip = set(exclude)
        async with self._recipe() as recipe:
            # The EnvImage checkout is pristine = gold; overwrite target files to
            # materialize an alternative (no-op when scoring gold).
            await recipe.set_state(TreeState.GOLD)
            for file in alt_files:
                await recipe.sandbox.write_file(file.path, file.content)

            # P2P/regression runs with NO hidden test present (a hidden test must
            # not make the repo's own suite look red).
            p2p = await recipe.run_p2p()

            failing: list[str] = []
            f2p_passed = True
            for test in self._base_tests:
                if test.test_id in skip:
                    continue
                await recipe.write_test(test)
                run = await recipe.run_test(test)
                await recipe.remove_test(test)
                if not run.passed:
                    failing.append(test.test_id)
                    f2p_passed = False

            if not p2p.passed:
                failing.append(recipe.p2p_command)
            return AltScore(
                f2p_passed=f2p_passed,
                p2p_passed=p2p.passed,
                failing_test_ids=tuple(failing),
            )

    async def score_gold(self, exclude: Collection[str] = ()) -> AltScore:
        return await self._score((), exclude)

    async def score_alt(self, alt: AltImpl, exclude: Collection[str] = ()) -> AltScore:
        return await self._score(alt.files, exclude)

    async def read_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        async with self._recipe() as recipe:
            await recipe.set_state(TreeState.GOLD)
            for path in self._target_files:
                with contextlib.suppress(Exception):
                    sources[path] = await recipe.sandbox.read_file(path)
        return sources


async def run_alt_correct_gate(
    candidate: Candidate,
    env_image: EnvImage,
    prior_report: OracleReport,
    *,
    spec: GeneratedSpec | None = None,
    alt_generator: AltCorrectGenerator | None = None,
    num_alternatives: int = DEFAULT_NUM_ALTERNATIVES,
    relax: bool = False,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 600.0,
) -> OracleReport:
    """Run the alt-correct false-negative gate in throwaway Docker sandboxes.

    A green baseline is a hard precondition and the prior gate (differential) must
    have passed. Builds a :class:`DockerAltCorrectRunner`, asks the
    ``alt_generator`` for genuinely-correct alternatives honoring the spec's
    Interface, scores each against the established F2P+P2P suite, and returns the
    extended :class:`OracleReport` (with ``alt_correct_accepted`` set).
    """
    require_green_baseline(env_image)
    if prior_report.verdict != "pass":
        raise AltCorrectError(
            "alt-correct gate requires a passing prior (differential) report; got "
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
    runner = DockerAltCorrectRunner(
        candidate,
        env_image,
        adapter,
        base_tests=base_tests,
        p2p_command=p2p_command,
        command_timeout=command_timeout,
        docker_client=docker_client,
    )

    alternatives: list[AltImpl] = []
    if alt_generator is not None:
        gold_sources = await runner.read_sources()
        gen_ctx = AltCorrectGenerationContext(
            candidate=candidate,
            adapter=adapter,
            gold_sources=gold_sources,
            interface_block=spec.interface_block if spec is not None else "",
            num_alternatives=num_alternatives,
        )
        alternatives = await alt_generator(gen_ctx)

    outcome = await assess_alt_correct(
        runner,
        alternatives,
        fail_to_pass=prior_report.fail_to_pass,
        relax=relax,
    )
    return build_alt_correct_report(
        candidate, prior_report, outcome, env_image=env_image, base_tests=base_tests
    )
