"""Establish gate: the FAIL->PASS contract + the reusable Docker test recipe.

This is the first oracle-hardening gate (architecture S6, Stage 3.1). On the
candidate's :class:`~swe_forge.forge.models.EnvImage`, inside a throwaway
:class:`~swe_forge.execution.sandbox.DockerSandbox`, it mechanically confirms the
two by-construction guarantees of a synthetic task:

* the **broken** tree (forward ``mutation_patch`` applied) FAILS the hidden F2P
  tests,
* the **gold** tree (inverse ``oracle_patch`` applied) makes every F2P test PASS,
* and the repo's own suite stays green as **P2P** on both trees (no collateral
  damage; the fault is detectable only by the hidden discriminating test).

Where the manufactured fault is not already covered by a discriminating test, the
agentic test generator (rewired onto :mod:`swe_forge.forge.teacher` - see
:mod:`swe_forge.forge.oracle.test_synth`) synthesizes hidden tests; *the teacher
proposes, deterministic execution disposes*: every proposed test is confirmed to
fail-on-broken and pass-on-gold here before it is recorded. The gate REJECTs (with
an attributable reason) when no F2P transition can be established, when the gold
patch fails to fix an intended F2P, or when an intended P2P is not green on broken.

The module also owns the **reusable Docker F2P/P2P execution recipe**
(:class:`DockerOracleRecipe`) that the later oracle gates and the calibration
runner score through, so a "solve" means the same thing everywhere. The recipe
bakes in the Python re-test determinism invariant from ``AGENTS.md``: it sets
``PYTHONDONTWRITEBYTECODE=1`` and purges ``__pycache__``/``*.pyc`` before every
re-run, so a same-second mutation revert + re-test never loads a stale ``.pyc``
and a green re-test never spuriously looks broken.
"""

from __future__ import annotations

import contextlib
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from swe_forge.forge.adapters import (
    LanguageAdapter,
    build_default_registry,
)
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    OracleReport,
    OracleTestFile,
    Provenance,
    require_green_baseline,
)

if TYPE_CHECKING:
    from swe_forge.execution.docker_client import DockerClient

# Where the recipe stages the candidate's patches inside the workspace. Kept out
# of the repo's own test-discovery paths so a P2P run never collects them.
_PATCH_DIR = ".swe_forge_oracle"
_MUTATION_PATCH_REL = f"{_PATCH_DIR}/mutation.patch"
_ORACLE_PATCH_REL = f"{_PATCH_DIR}/oracle.patch"

# Attributable reject reason prefixes (stable keys the contract/CLI gate on).
REASON_GOLD_P2P_NOT_GREEN = "gold_p2p_not_green"
REASON_P2P_NOT_GREEN_ON_BROKEN = "p2p_not_green_on_broken"
REASON_GOLD_NOT_FIX = "gold_not_fix"
REASON_NO_F2P = "no_f2p_established"


class EstablishError(RuntimeError):
    """Raised for an unrecoverable failure while driving the establish recipe."""


class TreeState(str, Enum):
    """Which source tree the workspace currently holds."""

    #: Known-good / gold source (the ``EnvImage`` checkout is pristine = gold).
    GOLD = "gold"
    #: Broken source = pristine + the candidate's forward ``mutation_patch``.
    BROKEN = "broken"


@dataclass(frozen=True)
class TestRun:
    """The outcome of one test/suite execution in the sandbox."""

    __test__ = False  # not a pytest test class

    command: str
    exit_code: int
    passed: bool
    stdout: str = ""
    stderr: str = ""

    def summary(self) -> dict[str, object]:
        """A compact, log-safe view (exit code + pass flag, no full output)."""
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class HiddenTestFile:
    """One file written into the workspace to materialize a hidden test."""

    path: str
    content: str


@dataclass(frozen=True)
class HiddenTest:
    """A candidate hidden F2P test: a run command plus the file(s) it needs.

    ``test_id`` is the exact command recorded in ``fail_to_pass`` (built via the
    adapter's selection-aware ``test_command``); ``files`` are written into the
    workspace before the command runs; ``origin`` is ``"synthesized"`` (authored
    by the agentic generator) or ``"provided"`` (a caller-declared/intended test
    whose failure to round-trip is a hard reject, not a silent drop).
    """

    test_id: str
    files: tuple[HiddenTestFile, ...] = ()
    origin: str = "synthesized"

    @property
    def intended(self) -> bool:
        """``True`` for a caller-declared test that MUST establish a transition."""
        return self.origin == "provided"


# --------------------------------------------------------------------------- #
# Sandbox / recipe protocols (so the gate is unit-testable without real Docker)
# --------------------------------------------------------------------------- #
@runtime_checkable
class _ExecResult(Protocol):
    @property
    def exit_code(self) -> int: ...
    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...


class SandboxProtocol(Protocol):
    """Minimal async sandbox surface the recipe needs (DockerSandbox-compatible)."""

    async def run_command(
        self,
        cmd: str,
        *,
        cwd: str | None = ...,
        timeout: float | None = ...,
        env: dict[str, str] | None = ...,
    ) -> _ExecResult: ...

    async def write_file(self, path: str, content: str) -> None: ...

    async def read_file(self, path: str) -> str: ...


class RecipeProtocol(Protocol):
    """The execution recipe surface the establish orchestration drives."""

    language: str
    p2p_command: str

    async def set_state(self, state: TreeState) -> None: ...
    async def run_p2p(self) -> TestRun: ...
    async def write_test(self, test: HiddenTest) -> None: ...
    async def remove_test(self, test: HiddenTest) -> None: ...
    async def run_test(self, test: HiddenTest) -> TestRun: ...


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


class DockerOracleRecipe:
    """Reusable Docker F2P/P2P execution recipe on an :class:`EnvImage`.

    Wraps a running :class:`SandboxProtocol` whose image is the candidate's
    EnvImage (the repo is already checked out, pristine = gold, at
    ``workspace_dir`` with deps installed). Owns the tree-state transitions
    (apply the forward ``mutation_patch`` for broken, the inverse ``oracle_patch``
    for gold), the selection-aware test execution, and the Python re-test
    determinism invariant (``PYTHONDONTWRITEBYTECODE=1`` + ``__pycache__``/``.pyc``
    purge before every run). The same recipe is consumed by the later oracle
    gates and the calibration scorer so "all F2P pass AND P2P green" means the
    same thing everywhere.
    """

    def __init__(
        self,
        sandbox: SandboxProtocol,
        *,
        language: str,
        workspace_dir: str,
        mutation_patch: str,
        oracle_patch: str,
        p2p_command: str,
        command_timeout: float = 600.0,
    ) -> None:
        self._sandbox = sandbox
        self.language = language
        self._workspace_dir = workspace_dir
        self._mutation_patch = mutation_patch
        self._oracle_patch = oracle_patch
        self.p2p_command = p2p_command
        self._timeout = command_timeout
        self._state = TreeState.GOLD
        self._prepared = False

    @property
    def sandbox(self) -> SandboxProtocol:
        """The underlying sandbox (so a synthesizer can explore the tree)."""
        return self._sandbox

    @property
    def state(self) -> TreeState:
        return self._state

    async def prepare(self) -> None:
        """Stage the candidate's patches in the workspace (idempotent)."""
        if self._prepared:
            return
        await self._sandbox.write_file(
            _MUTATION_PATCH_REL, _ensure_trailing_newline(self._mutation_patch)
        )
        await self._sandbox.write_file(
            _ORACLE_PATCH_REL, _ensure_trailing_newline(self._oracle_patch)
        )
        self._prepared = True

    async def set_state(self, state: TreeState) -> None:
        """Transition the working tree to ``state`` (no-op when already there)."""
        await self.prepare()
        if state == self._state:
            return
        # gold -> broken applies the forward mutation; broken -> gold applies the
        # inverse oracle patch (the contract's "broken (mutation)" / "gold (oracle)").
        rel = _MUTATION_PATCH_REL if state == TreeState.BROKEN else _ORACLE_PATCH_REL
        await self._apply_patch(rel)
        self._state = state

    async def _apply_patch(self, rel: str) -> None:
        primary = f"git apply --whitespace=nowarn {shlex.quote(rel)}"
        result = await self._raw(primary)
        if result.exit_code == 0:
            return
        fallback = f"git apply --3way --whitespace=nowarn {shlex.quote(rel)}"
        result2 = await self._raw(fallback)
        if result2.exit_code != 0:
            raise EstablishError(
                f"failed to apply patch {rel!r}: "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )

    async def purge_pycache(self) -> None:
        """Clear CPython's second-resolution bytecode cache (Python only).

        Defeats the stale-``.pyc`` hazard from ``AGENTS.md``: a same-second
        mutation revert + re-test could otherwise load a cached ``.pyc`` and a
        green re-test would spuriously look broken.
        """
        if self.language != "python":
            return
        await self._raw(
            "find . -name '__pycache__' -type d -prune -exec rm -rf {} + "
            "2>/dev/null; find . -name '*.pyc' -delete 2>/dev/null; true"
        )

    def _test_env(self) -> dict[str, str] | None:
        if self.language == "python":
            return {"PYTHONDONTWRITEBYTECODE": "1"}
        return None

    async def run_p2p(self) -> TestRun:
        """Run the repo's baseline (P2P/regression) suite on the current tree."""
        await self.purge_pycache()
        return await self._run_command(self.p2p_command)

    async def run_test(self, test: HiddenTest) -> TestRun:
        """Run a hidden test's selection command on the current tree."""
        await self.purge_pycache()
        return await self._run_command(test.test_id)

    async def write_test(self, test: HiddenTest) -> None:
        for file in test.files:
            await self._sandbox.write_file(file.path, file.content)

    async def remove_test(self, test: HiddenTest) -> None:
        for file in test.files:
            await self._raw(f"rm -f {shlex.quote(file.path)}")

    async def _run_command(self, command: str) -> TestRun:
        result = await self._sandbox.run_command(
            command, timeout=self._timeout, env=self._test_env()
        )
        return TestRun(
            command=command,
            exit_code=result.exit_code,
            passed=result.exit_code == 0,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    async def _raw(self, command: str) -> _ExecResult:
        return await self._sandbox.run_command(command, timeout=self._timeout)


# --------------------------------------------------------------------------- #
# Hidden-test synthesizer (teacher proposes; this gate disposes)
# --------------------------------------------------------------------------- #
@dataclass
class SynthesisContext:
    """Inputs handed to a :class:`HiddenTestSynthesizer`.

    The synthesizer explores the *broken* tree via ``recipe.sandbox`` and the
    ``adapter`` (for the selection-aware test command), and proposes hidden
    test(s); the establish gate confirms each before recording it.
    """

    candidate: Candidate
    env_image: EnvImage
    adapter: LanguageAdapter
    recipe: DockerOracleRecipe
    max_attempts: int = 1

    @property
    def language(self) -> str:
        return self.candidate.language

    @property
    def target_files(self) -> tuple[str, ...]:
        return tuple(self.candidate.target.files)


class HiddenTestSynthesizer(Protocol):
    """Proposes discriminating hidden tests for a manufactured fault."""

    async def __call__(self, ctx: SynthesisContext) -> list[HiddenTest]: ...


class NullSynthesizer:
    """A synthesizer that proposes nothing (offline/deterministic default).

    Used when synthesis is disabled or unavailable: the establish gate then
    relies solely on caller-provided tests and rejects with ``no_f2p_established``
    if none discriminate.
    """

    async def __call__(self, ctx: SynthesisContext) -> list[HiddenTest]:
        return []


# --------------------------------------------------------------------------- #
# Establish orchestration
# --------------------------------------------------------------------------- #
@dataclass
class _Confirmation:
    test: HiddenTest
    fails_on_broken: bool
    passes_on_gold: bool

    @property
    def confirmed(self) -> bool:
        return self.fails_on_broken and self.passes_on_gold


@dataclass
class EstablishOutcome:
    """The result of the establish gate (folded into an :class:`OracleReport`)."""

    verdict: str
    reasons: list[str]
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    test_files: list[OracleTestFile] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        return self.verdict == "pass"


async def _confirm_batch(
    recipe: RecipeProtocol, proposals: Sequence[HiddenTest]
) -> list[_Confirmation]:
    """Confirm each proposal's fail-on-broken / pass-on-gold via the recipe.

    Uses exactly two state transitions regardless of the number of proposals:
    run every proposal on the broken tree, then every proposal on the gold tree.
    Test files are written immediately before and removed immediately after each
    run so a proposal never perturbs another's verdict.
    """
    if not proposals:
        return []

    fails_on_broken: list[bool] = []
    await recipe.set_state(TreeState.BROKEN)
    for test in proposals:
        await recipe.write_test(test)
        run = await recipe.run_test(test)
        await recipe.remove_test(test)
        fails_on_broken.append(not run.passed)

    passes_on_gold: list[bool] = []
    await recipe.set_state(TreeState.GOLD)
    for test in proposals:
        await recipe.write_test(test)
        run = await recipe.run_test(test)
        await recipe.remove_test(test)
        passes_on_gold.append(run.passed)

    return [
        _Confirmation(
            test=test,
            fails_on_broken=fails_on_broken[idx],
            passes_on_gold=passes_on_gold[idx],
        )
        for idx, test in enumerate(proposals)
    ]


def _record_confirmations(
    confirmations: Sequence[_Confirmation],
    *,
    confirmed: list[HiddenTest],
    seen_ids: set[str],
    reasons: list[str],
) -> None:
    """Fold a batch of confirmations into the running confirmed list + reasons."""
    for conf in confirmations:
        if conf.confirmed:
            if conf.test.test_id not in seen_ids:
                seen_ids.add(conf.test.test_id)
                confirmed.append(conf.test)
        elif conf.test.intended and conf.fails_on_broken and not conf.passes_on_gold:
            reasons.append(
                f"{REASON_GOLD_NOT_FIX}: gold patch does not make the intended "
                f"F2P test {conf.test.test_id!r} pass"
            )


async def establish_oracle(
    recipe: RecipeProtocol,
    *,
    provided_tests: Sequence[HiddenTest] = (),
    synthesizer: HiddenTestSynthesizer | None = None,
    synthesis_context: SynthesisContext | None = None,
) -> EstablishOutcome:
    """Run the establish gate against ``recipe`` and return its outcome.

    Steps (gate order is honored - an early P2P failure rejects before any F2P
    work): confirm the P2P/regression suite is green on the gold tree and on the
    broken tree; confirm/synthesize hidden F2P tests that fail-on-broken and
    pass-on-gold; reject (with an attributable reason) when no F2P transition can
    be established, when the gold patch fails to fix an intended F2P, or when an
    intended P2P is not green on broken.
    """
    reasons: list[str] = []
    details: dict[str, object] = {}
    pass_to_pass = [recipe.p2p_command] if recipe.p2p_command else []

    # -- P2P: full regression must be green on gold AND on broken ---------- #
    await recipe.set_state(TreeState.GOLD)
    p2p_gold = await recipe.run_p2p()
    details["p2p_gold"] = p2p_gold.summary()
    if not p2p_gold.passed:
        reasons.append(
            f"{REASON_GOLD_P2P_NOT_GREEN}: the repo suite failed on the gold tree "
            f"(exit {p2p_gold.exit_code}); the gold/baseline is not green"
        )

    await recipe.set_state(TreeState.BROKEN)
    p2p_broken = await recipe.run_p2p()
    details["p2p_broken"] = p2p_broken.summary()
    if not p2p_broken.passed:
        reasons.append(
            f"{REASON_P2P_NOT_GREEN_ON_BROKEN}: an intended pass_to_pass test "
            f"failed on the broken tree (exit {p2p_broken.exit_code}); the fault "
            "has collateral damage beyond the hidden F2P"
        )

    if reasons:
        return EstablishOutcome(
            verdict="reject",
            reasons=reasons,
            fail_to_pass=[],
            pass_to_pass=pass_to_pass,
            test_files=[],
            details=details,
        )

    # -- F2P: confirm provided tests, synthesize when none discriminate ---- #
    confirmed: list[HiddenTest] = []
    seen_ids: set[str] = set()

    provided_conf = await _confirm_batch(recipe, list(provided_tests))
    _record_confirmations(
        provided_conf, confirmed=confirmed, seen_ids=seen_ids, reasons=reasons
    )
    details["provided_confirmations"] = [
        {
            "test_id": c.test.test_id,
            "fails_on_broken": c.fails_on_broken,
            "passes_on_gold": c.passes_on_gold,
            "confirmed": c.confirmed,
        }
        for c in provided_conf
    ]

    synthesized_count = 0
    if not confirmed and synthesizer is not None and synthesis_context is not None:
        await recipe.set_state(TreeState.BROKEN)
        proposed = await synthesizer(synthesis_context)
        synthesized_count = len(proposed)
        synth_conf = await _confirm_batch(recipe, proposed)
        _record_confirmations(
            synth_conf, confirmed=confirmed, seen_ids=seen_ids, reasons=reasons
        )
        details["synthesized_confirmations"] = [
            {
                "test_id": c.test.test_id,
                "fails_on_broken": c.fails_on_broken,
                "passes_on_gold": c.passes_on_gold,
                "confirmed": c.confirmed,
            }
            for c in synth_conf
        ]
    details["synthesized_proposed"] = synthesized_count

    if not confirmed:
        reasons.append(
            f"{REASON_NO_F2P}: no hidden test FAILS on the broken tree and PASSES "
            "on the gold tree (after best-effort synthesis); no F2P transition"
        )

    fail_to_pass = [test.test_id for test in confirmed]
    test_files: list[OracleTestFile] = []
    test_file_seen: set[str] = set()
    for test in confirmed:
        for file in test.files:
            if file.path in test_file_seen:
                continue
            test_file_seen.add(file.path)
            test_files.append(
                OracleTestFile(path=file.path, content=file.content, origin=test.origin)
            )

    verdict = "reject" if reasons else "pass"
    return EstablishOutcome(
        verdict=verdict,
        reasons=reasons,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        test_files=test_files,
        details=details,
    )


# --------------------------------------------------------------------------- #
# Report assembly + top-level gate runner (real Docker)
# --------------------------------------------------------------------------- #
def _tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    with contextlib.suppress(metadata.PackageNotFoundError):
        versions["litellm"] = metadata.version("litellm")
    return versions


def build_establish_report(
    candidate: Candidate,
    outcome: EstablishOutcome,
    *,
    env_image: EnvImage | None = None,
    extra_details: dict[str, object] | None = None,
) -> OracleReport:
    """Fold an :class:`EstablishOutcome` into an :class:`OracleReport`.

    Only the establish-stage fields are populated; the later gates
    (flakiness/mutation/differential/alt-correct/leak) extend the report.
    """
    details: dict[str, object] = {"stage": "establish", **outcome.details}
    if env_image is not None:
        details["env_image"] = env_image.image_tag
        details["p2p_command"] = env_image.baseline_test_command
    if extra_details:
        details.update(extra_details)

    provenance = Provenance(
        generator=candidate.generator,
        seed=candidate.provenance.seed,
        language=candidate.language,
        tool_versions=_tool_versions(),
        details={
            "stage": "oracle.establish",
            "fail_to_pass": list(outcome.fail_to_pass),
            "pass_to_pass": list(outcome.pass_to_pass),
            "test_files": [tf.path for tf in outcome.test_files],
        },
    )
    return OracleReport(
        language=candidate.language,
        generator=candidate.generator,
        verdict=outcome.verdict,
        reasons=list(outcome.reasons),
        fail_to_pass=list(outcome.fail_to_pass),
        pass_to_pass=list(outcome.pass_to_pass),
        test_files=list(outcome.test_files),
        provenance=provenance,
        details=details,
    )


async def run_establish_gate(
    candidate: Candidate,
    env_image: EnvImage,
    *,
    provided_tests: Sequence[HiddenTest] = (),
    synthesizer: HiddenTestSynthesizer | None = None,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 600.0,
) -> OracleReport:
    """Run the establish gate in a throwaway Docker sandbox on ``env_image``.

    A green baseline is a hard precondition (``require_green_baseline``). The
    sandbox uses the EnvImage directly (repo already checked out = gold), drives
    the :class:`DockerOracleRecipe`, and tears the container down even on failure
    (the container is uniquely named and ``--rm`` via :class:`DockerSandbox`).
    """
    from swe_forge.execution.docker_client import DockerClient
    from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

    require_green_baseline(env_image)

    if adapter is None:
        adapter = build_default_registry().get(candidate.language)

    client = docker_client or DockerClient()
    config = SandboxConfig(
        name="swe-forge-oracle-establish",
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
            p2p_command=env_image.baseline_test_command,
            command_timeout=command_timeout,
        )
        ctx = SynthesisContext(
            candidate=candidate,
            env_image=env_image,
            adapter=adapter,
            recipe=recipe,
        )
        outcome = await establish_oracle(
            recipe,
            provided_tests=provided_tests,
            synthesizer=synthesizer,
            synthesis_context=ctx,
        )

    return build_establish_report(candidate, outcome, env_image=env_image)
