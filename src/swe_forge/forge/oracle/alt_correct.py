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
import hashlib
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
from swe_forge.forge.oracle.teacher_evidence import (
    append_execution,
    evidence_calls,
    gate_evidence,
    teacher_gate_failure_reason,
    teacher_gate_evidence_issues,
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
REASON_ALT_CORRECT_NO_EXECUTABLE = "alt_correct_no_executable_teacher_proposals"
REASON_ALT_CORRECT_INVALID_TEACHER_PROPOSAL = "invalid_teacher_proposal"
REASON_ALT_CORRECT_PUBLIC_SUITE_UNAVAILABLE = "alt_correct_public_suite_unavailable"


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
    """The outcome of public, hidden-F2P, and filtered-P2P checks on one tree."""

    f2p_passed: bool
    p2p_passed: bool
    public_suite_passed: bool = False
    public_suite_exit_code: int | None = None
    p2p_exit_code: int | None = None
    hidden_test_exits: tuple[tuple[str, int], ...] = ()
    failing_test_ids: tuple[str, ...] = ()

    @property
    def public_valid(self) -> bool:
        """Whether a tree passed the original unfiltered upstream/public suite."""
        return self.public_suite_passed

    @property
    def accepted(self) -> bool:
        """True iff public validation, every F2P, and filtered P2P are green.

        For an alternative this means accepted (good, the correct alternative is
        not falsely rejected); for gold it means accepted (the required
        by-construction guarantee).
        """
        return self.public_valid and self.f2p_passed and self.p2p_passed

    def summary(self) -> dict[str, object]:
        return {
            "public_suite_passed": self.public_suite_passed,
            "public_suite_exit_code": self.public_suite_exit_code,
            "f2p_passed": self.f2p_passed,
            "p2p_passed": self.p2p_passed,
            "p2p_exit_code": self.p2p_exit_code,
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
    # Raw materialized proposals, their digests, and per-node exit evidence are
    # never agent-facing.  This is intentionally excluded from normal report
    # serialization and written only by the protected publication audit store.
    protected_audit: dict[str, object] = field(default_factory=dict, repr=False)

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


def _proposal_digest(alt: AltImpl) -> str:
    """Return a stable digest of an alternative's materialized source patches."""
    digest = hashlib.sha256()
    for file in sorted(alt.files, key=lambda item: item.path):
        digest.update(file.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _protected_alt_record(alt: AltImpl, score: AltScore) -> dict[str, object]:
    """Build audit-only source and execution evidence for one materialized alt."""
    return {
        "proposal_sha256": _proposal_digest(alt),
        "patches": [{"path": file.path, "content": file.content} for file in alt.files],
        "public": {
            "passed": score.public_suite_passed,
            "exit_code": score.public_suite_exit_code,
        },
        "filtered_p2p": {
            "passed": score.p2p_passed,
            "exit_code": score.p2p_exit_code,
        },
        "hidden": [
            {"test_id": test_id, "exit_code": exit_code}
            for test_id, exit_code in score.hidden_test_exits
        ],
    }


def _safe_public_audit(
    *,
    public_command: str,
    gold: AltScore,
    scores: dict[str, AltScore],
) -> dict[str, object]:
    """Return a source-free summary fit for the normal OracleReport details."""
    return {
        "public_suite_sha256": hashlib.sha256(
            public_command.encode("utf-8")
        ).hexdigest(),
        "gold_public_suite_passed": gold.public_suite_passed,
        "public_valid_alternatives": sum(
            score.public_valid for score in scores.values()
        ),
        "invalid_teacher_proposals": sorted(
            alt_id for alt_id, score in scores.items() if not score.public_valid
        ),
    }


async def assess_alt_correct(
    runner: AltCorrectRunner,
    alternatives: Sequence[AltImpl],
    *,
    fail_to_pass: Sequence[str] = (),
    relax: bool = False,
    original_public_command: str = "",
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
    protected_audit: dict[str, object] = {
        "version": 1,
        "original_public_suite_sha256": hashlib.sha256(
            original_public_command.encode("utf-8")
        ).hexdigest(),
        "gold": {
            "public": {
                "passed": gold_base.public_suite_passed,
                "exit_code": gold_base.public_suite_exit_code,
            },
            "filtered_p2p": {
                "passed": gold_base.p2p_passed,
                "exit_code": gold_base.p2p_exit_code,
            },
            "hidden": [
                {"test_id": test_id, "exit_code": exit_code}
                for test_id, exit_code in gold_base.hidden_test_exits
            ],
        },
        "alternatives": {},
    }
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
            protected_audit=protected_audit,
        )

    if not alternatives:
        return AltCorrectOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_ALT_CORRECT_NO_EXECUTABLE}: no genuinely-correct "
                "teacher alternative was executable through the alt-correct gate"
            ],
            alt_correct_accepted=False,
            alternatives_total=0,
            alternatives_accepted=0,
            details=details,
            protected_audit=protected_audit,
        )

    per_alt: dict[str, AltScore] = {}
    rejected: list[tuple[AltImpl, AltScore]] = []
    public_invalid: list[str] = []
    public_valid_alternatives: list[AltImpl] = []
    for alt in alternatives:
        score = await runner.score_alt(alt)
        per_alt[alt.impl_id] = score
        audit_alternatives = protected_audit["alternatives"]
        assert isinstance(audit_alternatives, dict)
        audit_alternatives[alt.impl_id] = _protected_alt_record(alt, score)
        if not score.public_valid:
            public_invalid.append(alt.impl_id)
            continue
        public_valid_alternatives.append(alt)
        if not score.accepted:
            rejected.append((alt, score))
    accepted_count = len(public_valid_alternatives) - len(rejected)
    details["initial"] = {
        "accepted": [
            aid
            for aid, score in per_alt.items()
            if score.public_valid and score.accepted
        ],
        "rejected": [alt.impl_id for alt, _ in rejected],
        "public_invalid": sorted(public_invalid),
    }
    details.update(
        _safe_public_audit(
            public_command=original_public_command,
            gold=gold_base,
            scores=per_alt,
        )
    )

    if not public_valid_alternatives:
        return AltCorrectOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_ALT_CORRECT_INVALID_TEACHER_PROPOSAL}: "
                f"alternative(s) {sorted(public_invalid)} failed the original "
                "unfiltered upstream/public suite"
            ],
            alt_correct_accepted=False,
            alternatives_total=len(alternatives),
            alternatives_accepted=0,
            rejected=sorted(public_invalid),
            details=details,
            protected_audit=protected_audit,
        )

    if not rejected:
        return AltCorrectOutcome(
            verdict="pass",
            reasons=[],
            alt_correct_accepted=True,
            alternatives_total=len(alternatives),
            alternatives_accepted=accepted_count,
            rejected=[],
            details=details,
            protected_audit=protected_audit,
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
            protected_audit=protected_audit,
        )

    return await _attempt_relax(
        runner,
        public_valid_alternatives,
        rejected,
        accepted_count,
        fail_to_pass,
        details,
        protected_audit,
    )


async def _attempt_relax(
    runner: AltCorrectRunner,
    alternatives: Sequence[AltImpl],
    rejected: Sequence[tuple[AltImpl, AltScore]],
    accepted_count: int,
    fail_to_pass: Sequence[str],
    details: dict[str, object],
    protected_audit: dict[str, object],
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
            protected_audit=protected_audit,
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
            protected_audit=protected_audit,
        )

    gold_relaxed = await runner.score_gold(exclude=overfit_ids)
    still_failing: list[str] = []
    for alt, _ in rejected:
        relaxed_score = await runner.score_alt(alt, exclude=overfit_ids)
        audit_alternatives = protected_audit["alternatives"]
        assert isinstance(audit_alternatives, dict)
        audit_alternatives[alt.impl_id]["relaxed"] = _protected_alt_record(
            alt, relaxed_score
        )
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
            protected_audit=protected_audit,
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
        protected_audit=protected_audit,
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
        protected_alt_correct_audit=(
            dict(outcome.protected_audit) if outcome.protected_audit else None
        ),
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
        original_public_command: str = "",
        command_timeout: float = 600.0,
        docker_client: "DockerClient | None" = None,
    ) -> None:
        self._candidate = candidate
        self._env_image = env_image
        self._adapter = adapter
        self._base_tests = list(base_tests)
        self._p2p_command = p2p_command or env_image.baseline_test_command
        self._original_public_command = original_public_command
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

            # The original upstream/public suite is a hard precondition.  It
            # deliberately runs before filtered P2P or any hidden test, so a
            # public-red teacher proposal can never be misclassified as hidden
            # overfit or motivate hidden-suite relaxation.
            await recipe.purge_pycache()
            public = await recipe.sandbox.run_command(
                self._original_public_command,
                timeout=self._timeout,
                env={"PYTHONDONTWRITEBYTECODE": "1"}
                if self.language == "python"
                else None,
            )
            if public.exit_code != 0:
                return AltScore(
                    f2p_passed=False,
                    p2p_passed=False,
                    public_suite_passed=False,
                    public_suite_exit_code=public.exit_code,
                )

            # Filtered P2P/regression runs with NO hidden test present (a hidden
            # test must not make the repo's own suite look red), but only after
            # the original public command has proven this tree valid.
            p2p = await recipe.run_p2p()

            failing: list[str] = []
            f2p_passed = True
            hidden_exits: list[tuple[str, int]] = []
            for test in self._base_tests:
                if test.test_id in skip:
                    continue
                await recipe.write_test(test)
                run = await recipe.run_test(test)
                await recipe.remove_test(test)
                hidden_exits.append((test.test_id, run.exit_code))
                if not run.passed:
                    failing.append(test.test_id)
                    f2p_passed = False

            if not p2p.passed:
                failing.append(recipe.p2p_command)
            return AltScore(
                f2p_passed=f2p_passed,
                p2p_passed=p2p.passed,
                public_suite_passed=True,
                public_suite_exit_code=public.exit_code,
                p2p_exit_code=p2p.exit_code,
                hidden_test_exits=tuple(hidden_exits),
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
    original_public_command = env_image.original_public_test_command.strip()
    if not original_public_command:
        outcome = AltCorrectOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_ALT_CORRECT_PUBLIC_SUITE_UNAVAILABLE}: EnvImage does not "
                "retain the original unfiltered upstream/public test command"
            ],
            alt_correct_accepted=False,
            alternatives_total=0,
            alternatives_accepted=0,
            details={"stage": "alt_correct"},
        )
        return build_alt_correct_report(
            candidate,
            prior_report,
            outcome,
            env_image=env_image,
            extra_details={"teacher_gates": {"alt_correct": gate_evidence([])}},
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
        original_public_command=original_public_command,
        command_timeout=command_timeout,
        docker_client=docker_client,
    )

    alternatives: list[AltImpl] = []
    teacher_calls = []
    authoritative_teacher_generator = False
    if alt_generator is not None:
        # A public gate must not let a programmatic test seam stand in for a
        # paid, uncached concrete-teacher call. The generic assessor remains
        # reusable; only the exact production generator can issue
        # authoritative transport-attested evidence.
        from swe_forge.forge.oracle.alt_correct_synth import (
            TeacherAltCorrectGenerator,
        )

        authoritative_teacher_generator = (
            type(alt_generator) is TeacherAltCorrectGenerator
        )
        gold_sources = await runner.read_sources()
        gen_ctx = AltCorrectGenerationContext(
            candidate=candidate,
            adapter=adapter,
            gold_sources=gold_sources,
            interface_block=spec.interface_block if spec is not None else "",
            num_alternatives=num_alternatives,
        )
        alternatives = await alt_generator(gen_ctx)
        teacher_calls.extend(evidence_calls(alt_generator))

    outcome = await assess_alt_correct(
        runner,
        alternatives,
        fail_to_pass=prior_report.fail_to_pass,
        relax=relax,
        original_public_command=original_public_command,
    )
    append_execution(
        teacher_calls,
        call_kind="proposal",
        attempted=outcome.alternatives_total,
        completed=outcome.alternatives_total,
        executable=outcome.alternatives_total,
    )
    evidence = gate_evidence(teacher_calls)
    evidence_issues = teacher_gate_evidence_issues(
        {"teacher_gates": {"alt_correct": evidence}},
        gates=("alt_correct",),
    )
    if not authoritative_teacher_generator or evidence_issues:
        outcome.verdict = "reject"
        outcome.alt_correct_accepted = False
        if not authoritative_teacher_generator:
            reason = "alt_correct_no_real_teacher_proposal"
        elif not alternatives:
            reason = teacher_gate_failure_reason("alt_correct", teacher_calls)
        else:
            reason = "alt_correct_teacher_evidence_invalid: " + "; ".join(
                evidence_issues
            )
        outcome.reasons = [reason]
    return build_alt_correct_report(
        candidate,
        prior_report,
        outcome,
        env_image=env_image,
        base_tests=base_tests,
        extra_details={"teacher_gates": {"alt_correct": evidence}},
    )
