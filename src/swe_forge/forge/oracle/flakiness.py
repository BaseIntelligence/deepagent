"""Flakiness gate: determinism over repeated, fresh-container validation.

The second oracle-hardening gate (architecture S6, Stage 3.2). Once the establish
gate has fixed the hidden F2P tests and the P2P/regression suite, a shipped task
must be *reproducible*: the same validation has to yield the same verdict every
time. This gate re-runs the established F2P+P2P validation ``>=3`` times, each in a
**fresh throwaway** :class:`~swe_forge.execution.sandbox.DockerSandbox`, and looks
for any test whose pass/fail verdict varies across the runs.

Outcomes:

* a **deterministic** suite yields identical per-test verdicts across every run and
  passes; ``OracleReport.flakiness_runs`` records the number of distinct container
  runs (always ``>=3`` whenever the gate executed),
* a **non-deterministic F2P** test (time/random/ordering dependent) is **dropped**
  from ``fail_to_pass``/``test_files``; if dropping removes the last F2P (the
  oracle would have no sound discriminating test left) the candidate is
  **rejected** with a flakiness reason,
* a **non-deterministic P2P/regression** suite cannot simply be dropped (it would
  leave the oracle unsound), so the candidate is **rejected** with a flakiness
  reason.

The gate reuses the establish gate's :class:`~swe_forge.forge.oracle.establish.DockerOracleRecipe`
(so "F2P fails on broken / passes on gold, P2P green on both" means exactly the
same thing everywhere) and its Python re-test determinism invariant
(``PYTHONDONTWRITEBYTECODE=1`` + ``__pycache__``/``.pyc`` purge before each run) so a
*deterministic* suite is never mislabelled flaky by a stale ``.pyc``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Callable

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    OracleReport,
    OracleTestFile,
    Provenance,
    require_green_baseline,
)
from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    HiddenTest,
    HiddenTestFile,
    RecipeProtocol,
    TestRun,
    TreeState,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from swe_forge.execution.docker_client import DockerClient

#: How many fresh-container validation runs the gate performs by default. The gate
#: never runs fewer than :data:`MIN_FLAKINESS_RUNS` so the ``>=3`` invariant holds.
DEFAULT_FLAKINESS_RUNS = 3
MIN_FLAKINESS_RUNS = 3

# Attributable reject reason prefixes (stable keys the contract/CLI gate on). Both
# begin with ``flakiness`` so a reject is always traceable to this gate.
REASON_FLAKY_P2P = "flakiness_nondeterministic_p2p"
REASON_FLAKY_LAST_F2P = "flakiness_dropped_last_f2p"


class FlakinessError(RuntimeError):
    """Raised for an unrecoverable failure while driving the flakiness gate."""


#: A factory that yields a *fresh* recipe (a new throwaway container) each call.
RecipeFactory = Callable[[], AbstractAsyncContextManager[RecipeProtocol]]


@dataclass
class SingleRun:
    """The verdicts observed in one fresh-container validation pass.

    Records the P2P/regression run on both trees and every F2P test's run on the
    broken tree (expected to FAIL) and the gold tree (expected to PASS). Verdicts
    are compared across runs to detect non-determinism; exit codes are retained as
    evidence.
    """

    index: int
    p2p_gold: TestRun
    p2p_broken: TestRun
    f2p_broken: dict[str, TestRun] = field(default_factory=dict)
    f2p_gold: dict[str, TestRun] = field(default_factory=dict)


@dataclass
class FlakinessOutcome:
    """The result of the flakiness gate (folded into an :class:`OracleReport`)."""

    verdict: str
    reasons: list[str]
    flakiness_runs: int
    fail_to_pass: list[str] = field(default_factory=list)
    dropped_test_ids: list[str] = field(default_factory=list)
    surviving_tests: list[HiddenTest] = field(default_factory=list)
    dropped_tests: list[HiddenTest] = field(default_factory=list)
    flaky_p2p: bool = False
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        return self.verdict == "pass"


def reconstruct_hidden_tests(
    fail_to_pass: Sequence[str], test_files: Sequence[OracleTestFile]
) -> list[HiddenTest]:
    """Rebuild :class:`HiddenTest` objects from a prior gate's report fields.

    Each ``fail_to_pass`` entry is a selection-aware test command that embeds the
    test file's path (built via the adapter's ``test_command``), so a test file is
    associated with the command(s) whose text contains its path. Files that do not
    match any command (e.g. an existing repo test already on disk) are simply not
    carried, which is correct: the gate only needs to re-materialize the
    synthesized/provided hidden test bodies.
    """
    tests: list[HiddenTest] = []
    for test_id in fail_to_pass:
        files = tuple(
            HiddenTestFile(path=tf.path, content=tf.content)
            for tf in test_files
            if tf.path and tf.path in test_id
        )
        origin = "synthesized"
        for tf in test_files:
            if tf.path and tf.path in test_id:
                origin = tf.origin
                break
        tests.append(HiddenTest(test_id=test_id, files=files, origin=origin))
    return tests


async def _validate_once(
    recipe: RecipeProtocol, f2p_tests: Sequence[HiddenTest], *, index: int
) -> SingleRun:
    """Run the full established F2P+P2P validation once on ``recipe``.

    The P2P/regression suite runs with NO hidden test present (so a fail-on-broken
    hidden test never makes the full suite look red); each F2P test is written
    immediately before, and removed immediately after, its own run so it never
    perturbs the suite or another test's verdict.
    """
    await recipe.set_state(TreeState.GOLD)
    p2p_gold = await recipe.run_p2p()
    await recipe.set_state(TreeState.BROKEN)
    p2p_broken = await recipe.run_p2p()

    # F2P on the broken tree (expected to FAIL) - we are already on broken.
    f2p_broken: dict[str, TestRun] = {}
    for test in f2p_tests:
        await recipe.write_test(test)
        f2p_broken[test.test_id] = await recipe.run_test(test)
        await recipe.remove_test(test)

    # F2P on the gold tree (expected to PASS).
    await recipe.set_state(TreeState.GOLD)
    f2p_gold: dict[str, TestRun] = {}
    for test in f2p_tests:
        await recipe.write_test(test)
        f2p_gold[test.test_id] = await recipe.run_test(test)
        await recipe.remove_test(test)

    return SingleRun(
        index=index,
        p2p_gold=p2p_gold,
        p2p_broken=p2p_broken,
        f2p_broken=f2p_broken,
        f2p_gold=f2p_gold,
    )


def _is_consistent(values: Sequence[bool]) -> bool:
    """``True`` iff every observed verdict is identical (a deterministic test)."""
    return len(set(values)) <= 1


def _summarize(
    runs: Sequence[SingleRun], f2p_tests: Sequence[HiddenTest]
) -> FlakinessOutcome:
    """Fold the per-run verdicts into a verdict + dropped/surviving partition."""
    reasons: list[str] = []
    flakiness_runs = len(runs)

    # -- P2P determinism: a flaky regression suite cannot be dropped --------- #
    p2p_gold_passed = [r.p2p_gold.passed for r in runs]
    p2p_broken_passed = [r.p2p_broken.passed for r in runs]
    flaky_p2p = not _is_consistent(p2p_gold_passed) or not _is_consistent(
        p2p_broken_passed
    )
    p2p_details: dict[str, object] = {
        "flaky": flaky_p2p,
        "gold_exit_codes": [r.p2p_gold.exit_code for r in runs],
        "broken_exit_codes": [r.p2p_broken.exit_code for r in runs],
        "gold_passed": p2p_gold_passed,
        "broken_passed": p2p_broken_passed,
    }

    # -- F2P determinism: a flaky F2P test is dropped ------------------------ #
    surviving: list[HiddenTest] = []
    dropped: list[HiddenTest] = []
    per_test: dict[str, object] = {}
    for test in f2p_tests:
        broken_passed = [r.f2p_broken[test.test_id].passed for r in runs]
        gold_passed = [r.f2p_gold[test.test_id].passed for r in runs]
        is_flaky = not _is_consistent(broken_passed) or not _is_consistent(gold_passed)
        per_test[test.test_id] = {
            "flaky": is_flaky,
            "broken_exit_codes": [r.f2p_broken[test.test_id].exit_code for r in runs],
            "gold_exit_codes": [r.f2p_gold[test.test_id].exit_code for r in runs],
            "broken_passed": broken_passed,
            "gold_passed": gold_passed,
        }
        (dropped if is_flaky else surviving).append(test)

    surviving_ids = [t.test_id for t in surviving]
    dropped_ids = [t.test_id for t in dropped]

    if flaky_p2p:
        reasons.append(
            f"{REASON_FLAKY_P2P}: the P2P/regression suite returned different "
            f"verdicts across {flakiness_runs} fresh-container runs; the oracle is "
            "non-deterministic and the regression suite cannot be dropped"
        )
    if not surviving_ids:
        reasons.append(
            f"{REASON_FLAKY_LAST_F2P}: dropping the flaky F2P test(s) {dropped_ids} "
            "removed the last F2P; no sound discriminating test remains"
        )

    verdict = "reject" if reasons else "pass"
    details: dict[str, object] = {
        "stage": "flakiness",
        "runs": flakiness_runs,
        "p2p": p2p_details,
        "per_test": per_test,
        "dropped": dropped_ids,
    }
    return FlakinessOutcome(
        verdict=verdict,
        reasons=reasons,
        flakiness_runs=flakiness_runs,
        fail_to_pass=surviving_ids,
        dropped_test_ids=dropped_ids,
        surviving_tests=surviving,
        dropped_tests=dropped,
        flaky_p2p=flaky_p2p,
        details=details,
    )


async def assess_flakiness(
    recipe_factory: RecipeFactory,
    *,
    f2p_tests: Sequence[HiddenTest],
    p2p_command: str,
    runs: int = DEFAULT_FLAKINESS_RUNS,
) -> FlakinessOutcome:
    """Re-run the established validation ``runs`` times in fresh recipes.

    ``recipe_factory`` yields a brand-new recipe (a fresh throwaway container) on
    every call so the runs are genuinely independent. The run count is clamped up
    to :data:`MIN_FLAKINESS_RUNS` so the ``flakiness_runs >= 3`` invariant always
    holds when the gate executes.
    """
    if not f2p_tests:
        raise FlakinessError(
            "flakiness gate requires at least one established F2P test; "
            "establish must pass before flakiness runs"
        )
    runs = max(runs, MIN_FLAKINESS_RUNS)

    records: list[SingleRun] = []
    for index in range(runs):
        async with recipe_factory() as recipe:
            records.append(await _validate_once(recipe, f2p_tests, index=index))

    outcome = _summarize(records, f2p_tests)
    outcome.details["p2p_command"] = p2p_command
    return outcome


def _tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    with contextlib.suppress(metadata.PackageNotFoundError):
        versions["litellm"] = metadata.version("litellm")
    return versions


def build_flakiness_report(
    candidate: Candidate,
    establish_report: OracleReport,
    outcome: FlakinessOutcome,
    *,
    env_image: EnvImage | None = None,
    extra_details: dict[str, object] | None = None,
) -> OracleReport:
    """Fold a :class:`FlakinessOutcome` into the running :class:`OracleReport`.

    Carries the establish gate's fields forward, sets ``flakiness_runs``, drops any
    flaky F2P from ``fail_to_pass``/``test_files``, and sets the terminal verdict
    (``pass`` when the surviving suite is deterministic; ``reject`` with an
    attributable flakiness reason when the regression suite is non-deterministic or
    dropping removed the last F2P).
    """
    details: dict[str, object] = dict(establish_report.details)
    details["flakiness"] = outcome.details
    if env_image is not None:
        details.setdefault("env_image", env_image.image_tag)
    if extra_details:
        details.update(extra_details)

    # Drop the test files belonging only to dropped (flaky) tests; keep any file a
    # surviving test still needs.
    dropped_paths = {f.path for t in outcome.dropped_tests for f in t.files}
    surviving_paths = {f.path for t in outcome.surviving_tests for f in t.files}
    remove_paths = dropped_paths - surviving_paths
    test_files = [
        tf for tf in establish_report.test_files if tf.path not in remove_paths
    ]

    base_prov = establish_report.provenance
    provenance = Provenance(
        generator=candidate.generator,
        seed=candidate.provenance.seed,
        language=candidate.language,
        tool_versions=dict(base_prov.tool_versions) if base_prov else _tool_versions(),
        details={
            "stage": "oracle.flakiness",
            "flakiness_runs": outcome.flakiness_runs,
            "fail_to_pass": list(outcome.fail_to_pass),
            "dropped": list(outcome.dropped_test_ids),
            "flaky_p2p": outcome.flaky_p2p,
        },
    )

    return OracleReport(
        language=establish_report.language,
        generator=establish_report.generator,
        verdict=outcome.verdict,
        reasons=list(outcome.reasons),
        fail_to_pass=list(outcome.fail_to_pass),
        pass_to_pass=list(establish_report.pass_to_pass),
        test_files=test_files,
        flakiness_runs=outcome.flakiness_runs,
        provenance=provenance,
        details=details,
    )


@asynccontextmanager
async def _docker_recipe(
    candidate: Candidate,
    env_image: EnvImage,
    *,
    command_timeout: float,
    docker_client: "DockerClient | None",
) -> "AsyncIterator[DockerOracleRecipe]":
    """Yield a recipe on a FRESH throwaway sandbox built from the EnvImage.

    Each call creates a new uniquely-named ``--rm`` :class:`DockerSandbox`, so the
    flakiness runs are genuinely independent containers; the sandbox tears down on
    exit even if a run raises.
    """
    from swe_forge.execution.docker_client import DockerClient
    from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

    client = docker_client or DockerClient()
    config = SandboxConfig(
        name="swe-forge-oracle-flakiness",
        image=env_image.image_tag,
        workspace_dir=env_image.workspace_dir,
        command_timeout=command_timeout,
    )
    sandbox = DockerSandbox(client, config)
    async with sandbox:
        yield DockerOracleRecipe(
            sandbox,
            language=candidate.language,
            workspace_dir=env_image.workspace_dir,
            mutation_patch=candidate.mutation_patch,
            oracle_patch=candidate.oracle_patch,
            p2p_command=env_image.baseline_test_command,
            command_timeout=command_timeout,
        )


async def run_flakiness_gate(
    candidate: Candidate,
    env_image: EnvImage,
    establish_report: OracleReport,
    *,
    runs: int = DEFAULT_FLAKINESS_RUNS,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 600.0,
) -> OracleReport:
    """Run the flakiness gate in fresh throwaway Docker sandboxes on ``env_image``.

    A green baseline is a hard precondition (``require_green_baseline``) and the
    establish gate must have passed (it fixes the F2P/P2P set this gate re-runs).
    Each of the ``>=3`` validation passes uses a brand-new container; the report is
    extended in place with the determinism verdict.
    """
    require_green_baseline(env_image)
    if establish_report.verdict != "pass":
        raise FlakinessError(
            "flakiness gate requires a passing establish report; got verdict "
            f"{establish_report.verdict!r}"
        )

    if adapter is None:
        adapter = build_default_registry().get(candidate.language)

    f2p_tests = reconstruct_hidden_tests(
        establish_report.fail_to_pass, establish_report.test_files
    )
    p2p_command = (
        establish_report.pass_to_pass[0]
        if establish_report.pass_to_pass
        else env_image.baseline_test_command
    )

    def factory() -> AbstractAsyncContextManager[RecipeProtocol]:
        return _docker_recipe(
            candidate,
            env_image,
            command_timeout=command_timeout,
            docker_client=docker_client,
        )

    outcome = await assess_flakiness(
        factory, f2p_tests=f2p_tests, p2p_command=p2p_command, runs=runs
    )
    return build_flakiness_report(
        candidate, establish_report, outcome, env_image=env_image
    )
