"""Pilot orchestration: the `swe-forge forge build --pilot` end-to-end run.

This is the capstone that drives Stage 0 -> Stage 5 in one invocation and proves
the two mission headlines on a real, shipped set:

* **(A)** every shipped task's gold patch scores ``{"score": 1}`` in Docker, and
* **(B)** the frontier panel solve-rate over the shipped set is low (below the
  stated threshold) but > 0.

The orchestrator is deliberately thin and language-agnostic: it threads a fixed
candidate through the SAME stage functions the per-stage CLI subcommands call
(env build, generate, spec, oracle pipeline, calibration, export, gold-eval,
report), so the ``--pilot`` path and the per-stage path agree by construction
(VAL-CROSS-020). The heavy per-candidate work lives behind the
:class:`CandidateProcessor` seam: :class:`LiveCandidateProcessor` wires the real
Docker + live-endpoint stages, while a fake processor lets the funnel/gate/
reconciliation invariants be unit-tested offline and deterministically.

Whole-pipeline invariants enforced here:

* the per-stage **funnel is monotone**: ``sourced >= env_built >= synthesized >=
  oracle_pass >= calibration_keep == exported`` (VAL-CROSS-013);
* **rejections never propagate**: an oracle-reject candidate is never calibrated
  and never exported; a calibration-drop candidate is never exported; only an
  oracle-pass AND band-keep candidate becomes an :class:`ExportRequest`
  (VAL-CROSS-012) -- and the export layer re-checks the same gate, so a
  mis-routed task is still refused;
* **usage/cost is surfaced**: teacher (spec + oracle synthesis) and panel
  (calibration) per-call usage is aggregated into the run summary
  (VAL-CROSS-022);
* **Docker hygiene with guaranteed teardown**: every container is throwaway and
  torn down by the stage that owns it (even on an induced failure); the pilot
  only ever names mission-scoped resources, so off-limits containers are never
  touched (VAL-CROSS-018/019).

Secrets never reach any pilot surface: the orchestrator only handles model ids,
counts, usage, and verdicts -- never the API key (VAL-CROSS-023).
"""

from __future__ import annotations

import asyncio
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Protocol

from swe_forge.forge.adapters import (
    AdapterRegistry,
    LanguageAdapter,
    NoAdapterFoundError,
    build_default_registry,
)
from swe_forge.forge.calibrate.filter import DEFAULT_BAND_FILTER, BandFilterConfig
from swe_forge.forge.calibrate.pipeline import run_calibration
from swe_forge.forge.checkpoint import PilotCheckpoint
from swe_forge.forge.export import (
    BatchExportResult,
    ExportRequest,
)
from swe_forge.forge.gold_eval import GoldEvalError, GoldEvalReport, run_gold_eval
from swe_forge.forge.generators import (
    GenerationError,
    GenerationRequest,
    build_default_generator_registry,
)
from swe_forge.forge.envbuild import EnvBuilder
from swe_forge.forge.models import (
    BaselineNotGreenError,
    CalibrationReport,
    Candidate,
    EnvImage,
    GeneratedSpec,
    OracleReport,
    RepoSpec,
)
from swe_forge.forge.oracle import (
    DEFAULT_FLAKINESS_RUNS,
    DEFAULT_KILL_THRESHOLD,
    STRUCTURAL_GENERATORS,
    DockerOracleRecipe,
    HiddenTest,
    P2PDerivation,
    SynthesisContext,
    TreeState,
    derive_structural_p2p_exclusions,
    run_oracle_pipeline,
)
from swe_forge.forge.oracle.establish import EstablishError
from swe_forge.forge.oracle.alt_correct_synth import TeacherAltCorrectGenerator
from swe_forge.forge.oracle.differential_synth import (
    DifferentialKillSynthesizer,
    TeacherVariantGenerator,
)
from swe_forge.forge.oracle.mutation_synth import MutationKillSynthesizer
from swe_forge.forge.oracle.test_synth import AgenticTestSynthesizer
from swe_forge.forge.oracle.teacher_evidence import aggregate_teacher_gate_usage
from swe_forge.forge.panel import (
    PanelModel,
    build_panel_from_env,
    resolve_panel_endpoint,
)
from swe_forge.forge.report import (
    DEFAULT_FRONTIER_THRESHOLD,
    BenchmarkReport,
    build_benchmark_report,
    write_report,
)
from swe_forge.forge.sources import SourceRegistry, build_source_registry
from swe_forge.forge.spec import (
    F2PTrace,
    FailingTest,
    SpecError,
    TemplateSpecAuthor,
    generate_spec,
)
from swe_forge.forge.teacher import (
    MissingCredentialsError,
    ModelRoutingError,
    Usage,
)


class PilotError(RuntimeError):
    """Raised for an unrecoverable failure while orchestrating the pilot."""


class StructuralF2PProtectionError(PilotError):
    """Raised when a structural F2P cannot be protected without ambiguity."""


@dataclass(frozen=True)
class StructuralF2PProtection:
    """A pre-P2P, broken-tree-verified structural F2P proposal.

    Structural P2P derivation runs before establish. Its file- and name-level
    collateral exclusions therefore need the identity of every F2P the establish
    gate will later confirm. This immutable bundle carries the exact proposal
    objects, their safely discovered test paths, and every parser-derived test
    name. An empty or ambiguous identity is rejected before derivation, never
    treated as permission to weaken the P2P suite.
    """

    tests: tuple[HiddenTest, ...]
    protected_names: tuple[str, ...]
    protected_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tests:
            raise StructuralF2PProtectionError(
                "structural F2P protection requires at least one proposed test"
            )
        if not self.protected_files:
            raise StructuralF2PProtectionError(
                "structural F2P protection is missing every test-file path"
            )
        if any(test.origin != "provided" for test in self.tests):
            raise StructuralF2PProtectionError(
                "structural F2P protection must retain every proposal as intended"
            )
        for path in self.protected_files:
            candidate = PurePosixPath(path)
            if (
                not path
                or candidate.is_absolute()
                or ".." in candidate.parts
                or path != candidate.as_posix()
            ):
                raise StructuralF2PProtectionError(
                    f"structural F2P protection has an unsafe test path {path!r}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "test_ids": [test.test_id for test in self.tests],
            "protected_names": list(self.protected_names),
            "protected_files": list(self.protected_files),
            "preflight": "failed_on_broken",
        }


@dataclass(frozen=True)
class _StructuralF2PObservation:
    """One preflight run used to derive fail-closed F2P protection metadata."""

    test: HiddenTest
    failed_on_broken: bool
    stdout: str
    stderr: str


def _dedupe_nonempty(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize non-empty strings in first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _matches_protected_file(path: str, protected_files: Sequence[str]) -> bool:
    """Match the same path/basename policy used by P2P collection protection."""
    candidate = PurePosixPath(path.strip())
    if not candidate.name:
        return False
    for protected in protected_files:
        known = PurePosixPath(protected)
        if candidate.as_posix() == known.as_posix() or candidate.name == known.name:
            return True
    return False


def _build_structural_f2p_protection(
    observations: Sequence[_StructuralF2PObservation], adapter: LanguageAdapter
) -> StructuralF2PProtection:
    """Validate preflight observations into the metadata P2P derivation consumes.

    Every proposed structural F2P must fail on the broken tree, identify at least
    one parser-recognized test name or collection file, and use only its own
    declared test paths for any collection error. Unknown output or a collection
    path outside the proposal is ambiguous metadata, so it rejects before the
    P2P command can be narrowed.
    """
    if not observations:
        raise StructuralF2PProtectionError(
            "structural F2P synthesis produced no test proposals"
        )

    tests: list[HiddenTest] = []
    names: list[str] = []
    files: list[str] = []
    seen_test_ids: set[str] = set()
    for observation in observations:
        test = observation.test
        if not test.test_id.strip():
            raise StructuralF2PProtectionError(
                "structural F2P proposal has an empty test command"
            )
        if test.test_id in seen_test_ids:
            raise StructuralF2PProtectionError(
                f"structural F2P proposal is duplicated: {test.test_id!r}"
            )
        seen_test_ids.add(test.test_id)
        if not test.files:
            raise StructuralF2PProtectionError(
                f"structural F2P proposal {test.test_id!r} has no test file"
            )
        proposal_files = _dedupe_nonempty([file.path for file in test.files])
        if len(proposal_files) != len(test.files):
            raise StructuralF2PProtectionError(
                f"structural F2P proposal {test.test_id!r} has ambiguous test paths"
            )
        for path in proposal_files:
            parsed = PurePosixPath(path)
            if (
                parsed.is_absolute()
                or ".." in parsed.parts
                or path != parsed.as_posix()
            ):
                raise StructuralF2PProtectionError(
                    f"structural F2P proposal has an unsafe test path {path!r}"
                )

        if not observation.failed_on_broken:
            raise StructuralF2PProtectionError(
                f"structural F2P proposal {test.test_id!r} passed on the broken tree"
            )
        output = "\n".join(
            part for part in (observation.stdout, observation.stderr) if part
        )
        parsed_names = _dedupe_nonempty(adapter.parse_test_failures(output))
        collection_files = _dedupe_nonempty(
            adapter.parse_collection_error_files(output)
        )
        if not parsed_names and not collection_files:
            raise StructuralF2PProtectionError(
                f"structural F2P proposal {test.test_id!r} failed with "
                "unparseable protection metadata"
            )
        for path in collection_files:
            if not _matches_protected_file(path, proposal_files):
                raise StructuralF2PProtectionError(
                    f"structural F2P collection error {path!r} is not owned by "
                    f"proposal {test.test_id!r}"
                )

        tests.append(
            HiddenTest(test_id=test.test_id, files=test.files, origin="provided")
        )
        names.extend(parsed_names)
        files.extend(proposal_files)

    return StructuralF2PProtection(
        tests=tuple(tests),
        protected_names=_dedupe_nonempty(names),
        protected_files=_dedupe_nonempty(files),
    )


# --------------------------------------------------------------------------- #
# Stage-count funnel
# --------------------------------------------------------------------------- #
@dataclass
class StageCounts:
    """The monotone per-stage funnel of the pilot run.

    Each gate only reduces or holds the population, and the terminal stage equals
    the exported workspace count: ``sourced >= env_built >= synthesized >=
    oracle_pass >= calibration_keep >= cap_admitted == exported``. A refused
    Stage-5 export intentionally breaks the final equality so the run fails
    closed instead of hiding an artifact/publication defect.
    """

    sourced: int = 0
    env_built: int = 0
    synthesized: int = 0
    oracle_pass: int = 0
    calibration_keep: int = 0
    exported: int = 0
    cap_admitted: int = 0
    export_refused: int = 0

    @property
    def monotone(self) -> bool:
        """``True`` iff eligibility, cap admission, and publication reconcile."""
        return (
            self.sourced
            >= self.env_built
            >= self.synthesized
            >= self.oracle_pass
            >= self.calibration_keep
            >= self.cap_admitted
            and self.cap_admitted == self.exported
            and self.export_refused == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sourced": self.sourced,
            "env_built": self.env_built,
            "synthesized": self.synthesized,
            "oracle_pass": self.oracle_pass,
            "calibration_keep": self.calibration_keep,
            "cap_admitted": self.cap_admitted,
            "exported": self.exported,
            "export_refused": self.export_refused,
            "monotone": self.monotone,
        }


# --------------------------------------------------------------------------- #
# Aggregate usage/cost (teacher + panel)
# --------------------------------------------------------------------------- #
@dataclass
class PilotUsage:
    """Aggregate LLM usage/cost for the run, split by teacher vs panel.

    ``teacher`` pools the spec-author + oracle-synthesis calls; ``panel`` pools
    the calibration validation + rollout calls. Both are surfaced so VAL-CROSS-022
    can confirm non-zero token usage and cost attributable to each.
    """

    teacher: Usage = field(default_factory=Usage)
    teacher_cost: float = 0.0
    panel: Usage = field(default_factory=Usage)
    panel_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.teacher.total_tokens + self.panel.total_tokens

    @property
    def total_cost(self) -> float:
        return self.teacher_cost + self.panel_cost

    def add(self, artifacts: CandidateArtifacts) -> None:
        self.teacher = self.teacher + artifacts.teacher_usage
        self.teacher_cost += artifacts.teacher_cost
        self.panel = self.panel + artifacts.panel_usage
        self.panel_cost += artifacts.panel_cost

    def to_dict(self) -> dict[str, object]:
        return {
            "teacher": {"usage": self.teacher.to_dict(), "cost": self.teacher_cost},
            "panel": {"usage": self.panel.to_dict(), "cost": self.panel_cost},
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }


# --------------------------------------------------------------------------- #
# Plans, per-candidate artifacts, dispositions
# --------------------------------------------------------------------------- #
@dataclass
class CandidatePlan:
    """One candidate to attempt: a (repo, generator, seed[, target]) tuple."""

    repo: RepoSpec
    generator: str
    seed: int
    file: str | None = None
    symbol: str | None = None
    op: str | None = None
    params: dict[str, object] = field(default_factory=dict)

    @property
    def language(self) -> str:
        return self.repo.language

    @property
    def label(self) -> str:
        return f"{self.repo.repo_id}::{self.generator}#{self.seed}"

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo.repo_id,
            "language": self.language,
            "generator": self.generator,
            "seed": self.seed,
            "file": self.file,
            "symbol": self.symbol,
            "op": self.op,
            "params": dict(self.params),
        }


@dataclass
class CandidateArtifacts:
    """Everything Stage 1-4 produced for one candidate plan.

    A ``None`` for an early field marks where the candidate dropped out of the
    funnel (env build red, generation failed, oracle reject, calibration drop).
    ``cleanup_paths`` are extra host dirs (beyond the orchestrator-owned workdir)
    a processor may ask to be removed after export.
    """

    plan: CandidatePlan
    env_image: EnvImage | None = None
    candidate: Candidate | None = None
    spec: GeneratedSpec | None = None
    oracle_report: OracleReport | None = None
    calibration_report: CalibrationReport | None = None
    broken_tree: Path | None = None
    repo_url: str = ""
    base_commit: str = ""
    teacher_usage: Usage = field(default_factory=Usage)
    teacher_cost: float = 0.0
    panel_usage: Usage = field(default_factory=Usage)
    panel_cost: float = 0.0
    structural_f2p_protection: StructuralF2PProtection | None = None
    p2p_derivation: P2PDerivation | None = None
    failure_reason: str = ""
    cleanup_paths: list[Path] = field(default_factory=list)


@dataclass
class CandidateDisposition:
    """Where one candidate exited the funnel (one row of the run ledger)."""

    plan: CandidatePlan
    stage: str  # env_failed | synth_failed | oracle_reject | calib_drop | cap_rejected | kept
    oracle_verdict: str = ""
    band_verdict: str = ""
    task_id: str = ""
    reason: str = ""
    # The serialized RepoSpec.acquire decision for a calibrated keep. This is
    # present for cap-admitted and cap-rejected dispositions only, letting the
    # ledger prove capacity without turning a rejection into an artifact.
    cap_grant: dict[str, object] | None = None
    # Observability only (never read by any gate/band/funnel logic): the per-tier
    # pass@k + discrimination + applied band rule of the candidate's calibration
    # run, present iff calibration ran (kept / calib_drop). Lets a harvest ledger
    # tell a solve-all/too-easy drop from a solve-none/too-hard or a
    # low-discrimination drop without re-parsing the free-text reason.
    calibration: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            **self.plan.to_dict(),
            "stage": self.stage,
            "oracle_verdict": self.oracle_verdict,
            "band_verdict": self.band_verdict,
            "task_id": self.task_id,
            "reason": self.reason,
            "cap_grant": self.cap_grant,
            "calibration": self.calibration,
        }


# --------------------------------------------------------------------------- #
# The candidate processor seam (Stage 1-4)
# --------------------------------------------------------------------------- #
class CandidateProcessor(Protocol):
    """Runs Stage 1-4 for one candidate plan and returns its artifacts.

    The orchestrator owns ``workdir`` (a fresh host dir under the run-scoped temp
    root) and tears it down after the run -- even if processing raises -- so the
    processor never needs to manage host cleanup. The live implementation drives
    real Docker + the live endpoint; offline tests inject a deterministic fake so
    the orchestrator's funnel/gate/reconciliation logic is exercised without
    Docker or the network.
    """

    async def process(
        self, plan: CandidatePlan, workdir: Path
    ) -> CandidateArtifacts: ...


class GoldEvalFn(Protocol):
    """The Stage-5 gold-eval seam (real Docker by default; injectable for tests)."""

    def __call__(self, tasks_dir: Path | str, *, runs: int = ...) -> GoldEvalReport: ...


def candidate_trace(candidate: Candidate) -> F2PTrace:
    """A minimal F2P trace for the spec, keyed to the candidate's target symbol.

    The spec's problem statement/requirements are backtranslated from the failing
    behavior of the target (never the diff); the real Docker F2P establishment
    happens in Stage 3. One named failing test is enough to ground the spec's
    requirements (Stage 2 contract).
    """
    files = [f for f in candidate.target.files if f]
    symbols = [s for s in candidate.target.symbols if s]
    target = symbols[0] if symbols else (files[0] if files else candidate.generator)
    file = files[0] if files else ""
    name = f"{file}::{target}" if file else str(target)
    return F2PTrace(
        tests=(
            FailingTest(
                name=name,
                file=file,
                message=(
                    f"{target} returns an incorrect result for the manufactured fault"
                ),
            ),
        )
    )


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _materialize_broken_tree(checkout: Path, candidate: Candidate, dest: Path) -> Path:
    """Copy the gold checkout, apply the forward mutation, return the broken tree.

    The export layer re-inits this tree to a single orphan commit before shipping
    it (so gold is unrecoverable from ``.git`` history); here we only need the
    mutated working tree the agent will see.
    """
    shutil.copytree(checkout, dest)
    patch_file = dest / ".forge_mutation.patch"
    patch_file.write_text(
        _ensure_trailing_newline(candidate.mutation_patch), encoding="utf-8"
    )
    apply = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_file.name)],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        fallback = subprocess.run(
            ["git", "apply", "--3way", "--whitespace=nowarn", str(patch_file.name)],
            cwd=dest,
            capture_output=True,
            text=True,
        )
        if fallback.returncode != 0:
            raise PilotError(
                f"could not apply mutation to broken tree: "
                f"{(apply.stderr or fallback.stderr).strip()[:300]}"
            )
    patch_file.unlink(missing_ok=True)
    return dest


class LiveCandidateProcessor:
    """The real Stage 1-4 processor: Docker env build + live-endpoint gates.

    Order: env build (green baseline, host checkout retained) -> generate the
    candidate -> author the spec (teacher, deterministic-template fallback) ->
    run the FULL oracle pipeline NON-OFFLINE with real teacher generators (so the
    differential/alt-correct gates never pass vacuously) -> calibrate via the
    panel when oracle passes. Each stage owns its own throwaway-container hygiene;
    the host checkout/broken-tree dirs live under the orchestrator-owned
    ``workdir`` (torn down after export, even on failure).
    """

    def __init__(
        self,
        *,
        panel: list[PanelModel],
        band_config: BandFilterConfig = DEFAULT_BAND_FILTER,
        kill_threshold: float = DEFAULT_KILL_THRESHOLD,
        flakiness_runs: int = DEFAULT_FLAKINESS_RUNS,
        k: int | None = None,
        concurrency: int = 4,
        validate_models: bool = True,
        command_timeout: float = 600.0,
        mutation_timeout: float = 1200.0,
        registry: AdapterRegistry | None = None,
    ) -> None:
        self._panel = panel
        self._band_config = band_config
        self._kill_threshold = kill_threshold
        self._flakiness_runs = flakiness_runs
        self._k = k
        self._concurrency = concurrency
        self._validate_models = validate_models
        self._command_timeout = command_timeout
        self._mutation_timeout = mutation_timeout
        self._registry = registry or build_default_registry()
        self._generators = build_default_generator_registry()
        # Per-repo env-image cache: a repo's green-baseline image is built once
        # per run and reused across its candidates (same image tag). The per-repo
        # lock serializes concurrent same-repo builds so two candidates never race
        # to commit the same tag (which could remove an image another is using).
        self._env_cache: dict[str, EnvImage] = {}
        self._env_failures: dict[str, str] = {}
        self._env_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._env_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._env_locks[key] = lock
            return lock

    async def _acquire_env_image(
        self, plan: CandidatePlan, checkout: Path
    ) -> tuple[EnvImage | None, str]:
        """Build (once per repo per run) or reuse the cached green-baseline image.

        The blocking Docker build runs in a worker thread so concurrent candidates
        for OTHER repos overlap; the per-repo lock ensures a single build per repo.
        Returns ``(env_image, "")`` on success or ``(None, reason)`` on failure
        (the failure is cached so a red repo is not rebuilt for every candidate).
        """
        key = plan.repo.repo_id
        lock = await self._lock_for(key)
        async with lock:
            cached = self._env_cache.get(key)
            if cached is not None:
                return cached, ""
            failed = self._env_failures.get(key)
            if failed is not None:
                return None, failed
            build = await asyncio.to_thread(
                EnvBuilder().build, plan.repo, workdir=checkout
            )
            if not build.success or build.env_image is None:
                reason = f"env build {build.failure_kind}: {build.reason}"
                self._env_failures[key] = reason
                return None, reason
            self._env_cache[key] = build.env_image
            return build.env_image, ""

    async def process(self, plan: CandidatePlan, workdir: Path) -> CandidateArtifacts:
        art = CandidateArtifacts(plan=plan)
        try:
            return await self._process(plan, workdir, art)
        except (
            BaselineNotGreenError,
            GenerationError,
            SpecError,
            MissingCredentialsError,
            ModelRoutingError,
            PilotError,
        ) as exc:
            art.failure_reason = f"{type(exc).__name__}: {exc}"
            return art
        except Exception as exc:  # noqa: BLE001
            # An UNEXPECTED per-candidate failure (a mutation-tool misconfig, a
            # transient Docker/endpoint error, a parser edge case, ...) must NOT
            # abort the whole sweep -- it is recorded as this candidate's failure
            # (dropping it out of the funnel at the stage it reached) so the
            # remaining candidates still run. This never masks a vacuous oracle
            # pass: the candidate is DROPPED, not passed (its oracle_report stays
            # unset -> counted as an oracle_reject with the reason). ``asyncio``
            # cancellation is a ``BaseException`` and still propagates.
            art.failure_reason = f"{type(exc).__name__}: {exc}"
            return art

    async def _process(
        self, plan: CandidatePlan, work: Path, art: CandidateArtifacts
    ) -> CandidateArtifacts:
        try:
            adapter = self._registry.get(plan.language)
        except NoAdapterFoundError as exc:
            art.failure_reason = str(exc)
            return art

        # -- Stage 1: env-first build (host checkout retained) ------------- #
        checkout = work / "checkout"
        checkout.mkdir()
        clone = await asyncio.to_thread(_checkout_repo, plan.repo, checkout)
        if clone is not None:
            art.failure_reason = f"checkout failed: {clone}"
            return art

        env_image, reason = await self._acquire_env_image(plan, checkout)
        if env_image is None:
            art.failure_reason = reason
            return art
        art.env_image = env_image
        art.repo_url = plan.repo.url
        art.base_commit = env_image.commit

        # -- Stage 2a: generate the candidate ------------------------------ #
        request = GenerationRequest(
            repo_root=checkout,
            seed=plan.seed,
            file=plan.file,
            symbol=plan.symbol,
            op=plan.op,
            env_image=art.env_image,
            params=dict(plan.params),
        )
        candidate = self._generators.get(plan.generator).generate(request, adapter)
        art.candidate = candidate

        # -- Stage 2b: author the spec (teacher; template fallback) -------- #
        trace = candidate_trace(candidate)
        spec = _author_spec(candidate, trace, checkout, adapter)
        art.spec = spec
        _accumulate_spec_usage(art, spec)

        # -- Stage 3: oracle pipeline (NON-OFFLINE; real teacher) ---------- #
        baseline_command = plan.repo.baseline_test or adapter.baseline_test_command(
            checkout
        )
        provided_tests = _pr_mirror_provided_tests(
            candidate, adapter, plan.repo, baseline_command
        )
        establish_synthesizer: AgenticTestSynthesizer | None = AgenticTestSynthesizer()

        # Per-candidate P2P exclusions for STRUCTURAL faults: a structural
        # mutation breaks the target's OWN existing tests as collateral damage,
        # so the full baseline goes red on the broken tree. Derive exactly those
        # fault-independent collateral failures and bake them into a derived
        # EnvImage baseline (the per-candidate analogue of the curated per-repo
        # RepoSpec.p2p_exclusions pr_mirror gets) so establish's P2P is green on
        # broken -- never the synthesized F2P, never a gate loosening. The derived
        # baseline threads through establish -> pass_to_pass -> calibration/export.
        if candidate.generator in STRUCTURAL_GENERATORS:
            # Propose structural F2P tests BEFORE observing and narrowing the
            # broken P2P. Their parser-derived names and declared file paths are
            # the protection boundary for name/file collateral derivation. A
            # missing, unparseable, or ambiguously owned proposal raises and the
            # outer process records a dropped candidate, never a weakened suite.
            protection = await self._propose_structural_f2p(
                candidate, art.env_image, adapter
            )
            art.structural_f2p_protection = protection
            provided_tests = list(protection.tests)
            # The exact pre-P2P proposal set must be confirmed by establish. Do
            # not generate a different fallback set after derivation because it
            # would no longer be protected metadata for the baseline we derived.
            establish_synthesizer = None
            art.env_image = await self._apply_structural_p2p_exclusions(
                candidate, art.env_image, adapter, art, protection
            )

        oracle = await run_oracle_pipeline(
            candidate,
            art.env_image,
            provided_tests=provided_tests,
            establish_synthesizer=establish_synthesizer,
            mutation_synthesizer=MutationKillSynthesizer(),
            variant_generator=TeacherVariantGenerator(),
            differential_synthesizer=DifferentialKillSynthesizer(),
            alt_generator=TeacherAltCorrectGenerator(),
            spec=spec,
            flakiness_runs=self._flakiness_runs,
            kill_threshold=self._kill_threshold,
            adapter=adapter,
            command_timeout=self._command_timeout,
            mutation_timeout=self._mutation_timeout,
        )
        art.oracle_report = oracle
        _accumulate_oracle_usage(art, oracle)
        if not oracle.is_pass:
            return art

        # -- Stage 4: calibration (panel; only on oracle-pass) ------------- #
        outcome = await run_calibration(
            candidate,
            art.env_image,
            spec,
            oracle,
            self._panel,
            k=self._k,
            concurrency=self._concurrency,
            validate=self._validate_models,
            config=self._band_config,
            command_timeout=self._command_timeout,
            adapter=adapter,
        )
        art.calibration_report = outcome.report
        _accumulate_panel_usage(art, outcome.report)

        if outcome.report.is_keep:
            broken = await asyncio.to_thread(
                _materialize_broken_tree, checkout, candidate, work / "broken"
            )
            art.broken_tree = broken
        return art

    async def _propose_structural_f2p(
        self,
        candidate: Candidate,
        env_image: EnvImage,
        adapter: LanguageAdapter,
    ) -> StructuralF2PProtection:
        """Propose and identify structural F2P tests before P2P derivation.

        The proposal runs only on the broken tree in a disposable sandbox. It is
        not accepted as an oracle test yet, because establish must independently
        verify the required broken-fails/gold-passes transition after P2P
        derivation. It does, however, provide the only safe source of protected
        F2P paths/names for derivation. Any unknown or ambiguous identity fails
        closed before the derived P2P command could exclude a module.
        """
        from swe_forge.execution.docker_client import DockerClient
        from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

        sandbox = DockerSandbox(
            DockerClient(),
            SandboxConfig(
                name="swe-forge-oracle-f2p-propose",
                image=env_image.image_tag,
                workspace_dir=env_image.workspace_dir,
                command_timeout=self._command_timeout,
            ),
        )
        async with sandbox:
            recipe = DockerOracleRecipe(
                sandbox,
                language=candidate.language,
                workspace_dir=env_image.workspace_dir,
                mutation_patch=candidate.mutation_patch,
                oracle_patch=candidate.oracle_patch,
                p2p_command=env_image.baseline_test_command,
                command_timeout=self._command_timeout,
            )
            await recipe.set_state(TreeState.BROKEN)
            proposals = await AgenticTestSynthesizer()(
                SynthesisContext(
                    candidate=candidate,
                    env_image=env_image,
                    adapter=adapter,
                    recipe=recipe,
                )
            )
            observations: list[_StructuralF2PObservation] = []
            for test in proposals:
                await recipe.write_test(test)
                try:
                    run = await recipe.run_test(test)
                finally:
                    await recipe.remove_test(test)
                observations.append(
                    _StructuralF2PObservation(
                        test=test,
                        failed_on_broken=not run.passed,
                        stdout=run.stdout,
                        stderr=run.stderr,
                    )
                )
        return _build_structural_f2p_protection(observations, adapter)

    async def _apply_structural_p2p_exclusions(
        self,
        candidate: Candidate,
        env_image: EnvImage,
        adapter: LanguageAdapter,
        art: CandidateArtifacts,
        protection: StructuralF2PProtection | None = None,
    ) -> EnvImage:
        """Derive a structural candidate's collateral P2P exclusions; bake them in.

        Runs the broken-tree baseline once in a throwaway sandbox to find the
        existing tests the structural fault breaks for fault-independent reasons,
        then returns a derived :class:`EnvImage` whose baseline command excludes
        exactly those (via ``adapter.apply_p2p_exclusions``) so the establish P2P
        is green on broken. Best-effort: a Docker/parse hiccup leaves the baseline
        untouched (establish then rejects with ``p2p_not_green_on_broken`` if the
        collateral is real), never weakening a gate. The exclusions are recorded
        on the artifact and the derived image provenance for audit.
        """
        if protection is None:
            raise StructuralF2PProtectionError(
                "structural P2P derivation requires validated F2P protection metadata"
            )
        try:
            derivation = await derive_structural_p2p_exclusions(
                candidate,
                env_image,
                adapter,
                baseline_command=env_image.baseline_test_command,
                protected_names=protection.protected_names,
                protected_files=protection.protected_files,
                command_timeout=self._command_timeout,
            )
        except (EstablishError, PilotError):
            return env_image
        derivation.details["structural_f2p_protection"] = protection.to_dict()
        art.p2p_derivation = derivation
        if derivation.has_protected_conflict:
            # An un-importable module IS (or contains) the F2P's own module: a
            # fault that breaks the F2P's own import is a real defect, not
            # collateral. Leave the baseline UNTOUCHED so establish rejects on the
            # still-red broken P2P (never a vacuous pass by excluding the F2P).
            return env_image
        if not derivation.has_exclusions:
            return env_image
        return _with_p2p_exclusions(env_image, adapter, derivation)


def _with_p2p_exclusions(
    env_image: EnvImage, adapter: LanguageAdapter, derivation: P2PDerivation
) -> EnvImage:
    """Return a derived :class:`EnvImage` whose baseline excludes the collateral.

    Narrows the proven-green baseline command with ``adapter.apply_p2p_exclusions``
    (per-test names) and ``adapter.apply_p2p_file_exclusions`` (whole test modules
    the fault makes uncollectable -- IMPORT-TIME collateral with no per-test name)
    so the per-candidate collateral is skipped from the P2P/regression set. The
    image tag is unchanged (same Docker image); only the recorded command and
    provenance change. Excluding tests from an already-green suite keeps gold
    green (and the establish gate re-checks ``p2p_gold`` defensively), so
    ``baseline_green`` is preserved.
    """
    new_command = adapter.apply_p2p_exclusions(
        env_image.baseline_test_command, derivation.exclusions
    )
    new_command = adapter.apply_p2p_file_exclusions(
        new_command, derivation.file_exclusions
    )
    provenance = dict(env_image.provenance)
    provenance["per_candidate_p2p_exclusions"] = derivation.to_dict()
    return replace(
        env_image,
        baseline_test_command=new_command,
        provenance=provenance,
    )


def _pr_mirror_provided_tests(
    candidate: Candidate,
    adapter: LanguageAdapter,
    repo: RepoSpec,
    baseline_command: str,
) -> list[HiddenTest]:
    """Build the isolated F2P tests a ``pr_mirror`` candidate reintroduces.

    A merged bug-fix PR ships the test that pins the fixed behavior; the current
    checkout already contains it, so reverting only the source (the mutation)
    makes that test FAIL on the broken tree and PASS on gold -- an isolated F2P.

    When the repo curates the flipping-test names (``RepoSpec.p2p_exclusions``,
    the very tests the baked baseline excludes to keep P2P green on broken), the
    F2P runs exactly those tests via the repo's OWN runner -- ``select_tests``
    narrows ``baseline_command`` positively, the mirror of the baked exclusion.
    A repo may further pin a SINGLE discriminating assertion via
    ``RepoSpec.pr_f2p_names`` (preferred when set): in semantic-correctness
    domains (URL/email validators) a single precise F2P assertion -- rather than a
    whole flipping test -- keeps the differential synthesizer's discriminators
    gold-green (fewer ``differential_gold_not_green`` rejects). This is what makes
    a non-standard-runner repo (e.g. JS/TS ``npm test`` driving Mocha, which the
    ``node --test`` standard runner cannot execute) confirm its F2P at all.
    Otherwise it falls back to running each recorded PR test file via the standard
    runner. ``files=()`` so nothing is written/removed and the test the mutation
    leaves in place is exactly what runs.
    """
    if candidate.generator != "pr_mirror":
        return []
    f2p_names = [n.strip() for n in repo.pr_f2p_names if n.strip()] or [
        n.strip() for n in repo.p2p_exclusions if n.strip()
    ]
    if f2p_names and baseline_command.strip():
        return [
            HiddenTest(
                test_id=adapter.select_tests(baseline_command, f2p_names),
                files=(),
                origin="provided",
            )
        ]
    details = candidate.provenance.details
    raw = details.get("test_files") if isinstance(details, dict) else None
    if not isinstance(raw, list):
        return []
    tests: list[HiddenTest] = []
    for path in raw:
        if not isinstance(path, str) or not path.strip():
            continue
        tests.append(
            HiddenTest(
                test_id=adapter.test_command((path,)),
                files=(),
                origin="provided",
            )
        )
    return tests


def _checkout_repo(repo: RepoSpec, dest: Path) -> str | None:
    """Clone+checkout the pinned commit into ``dest``; ``None`` on success."""
    for command in repo.checkout_commands():
        completed = subprocess.run(
            command,
            shell=True,
            cwd=dest,
            capture_output=True,
            text=True,
            timeout=600.0,
        )
        if completed.returncode != 0:
            return (completed.stderr or completed.stdout).strip()[:300]
    return None


def _author_spec(
    candidate: Candidate,
    trace: F2PTrace,
    checkout: Path,
    adapter: object,
) -> GeneratedSpec:
    """Author the spec via the teacher; fall back to the deterministic template.

    The teacher (the default) backtranslates a test-conditioned statement; the
    template author is a deterministic, offline fallback so a transient endpoint
    hiccup never drops an otherwise-valid candidate at the spec step. (The
    shipped task's ORACLE gates still run non-offline with a real teacher.)
    """
    try:
        return generate_spec(candidate, trace, checkout, adapter)  # type: ignore[arg-type]
    except (SpecError, MissingCredentialsError, ModelRoutingError):
        return generate_spec(
            candidate,
            trace,
            checkout,
            adapter,  # type: ignore[arg-type]
            author=TemplateSpecAuthor(),
        )


def _accumulate_spec_usage(art: CandidateArtifacts, spec: GeneratedSpec) -> None:
    teacher = spec.provenance.details.get("teacher")
    if isinstance(teacher, dict):
        usage = teacher.get("usage")
        if isinstance(usage, dict):
            art.teacher_usage = art.teacher_usage + Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
            )
        cost = teacher.get("cost")
        if isinstance(cost, (int, float)):
            art.teacher_cost += float(cost)


def _accumulate_oracle_usage(art: CandidateArtifacts, report: OracleReport) -> None:
    """Add every recorded oracle teacher call, including rejected candidates."""
    usage, cost = aggregate_teacher_gate_usage(report.details)
    art.teacher_usage = art.teacher_usage + usage
    art.teacher_cost += cost


def _accumulate_panel_usage(art: CandidateArtifacts, report: CalibrationReport) -> None:
    accounting = report.details.get("usage_accounting")
    if not isinstance(accounting, dict):
        return
    aggregate = accounting.get("aggregate")
    if isinstance(aggregate, dict):
        usage = aggregate.get("usage")
        if isinstance(usage, dict):
            art.panel_usage = art.panel_usage + Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
            )
        cost = aggregate.get("cost")
        if isinstance(cost, (int, float)):
            art.panel_cost += float(cost)


# --------------------------------------------------------------------------- #
# Pilot config + plan building
# --------------------------------------------------------------------------- #
#: Generators exercised per language for the pilot. Every generator family is
#: deterministic+verifiable on Python; the cross-language families (ast_mutation,
#: function_removal, and the difficulty amplifiers bug_combination/multi_file)
#: round-trip on JS/Go too. The amplifiers (``bug_combination`` merges >=2
#: independently-behavior-changing faults; ``multi_file`` coordinates edits across
#: >=2 files) raise candidate DIFFICULTY so the mix spans easy->hard and the band
#: filter can SELECT the in-band ones -- they run only on the diversified MODULAR
#: structural-source repos (where a structural fault isolates), guarded by
#: ``RepoSpec.structural_source`` in :func:`build_pilot_plans`. >=2 generators
#: keeps VAL-CROSS-005.
DEFAULT_GENERATORS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "python": (
        "ast_mutation",
        "function_removal",
        "bug_combination",
        "multi_file",
        "lm_authored",
    ),
    "javascript": ("ast_mutation", "function_removal", "bug_combination", "multi_file"),
    "go": ("ast_mutation", "function_removal", "bug_combination", "multi_file"),
}

#: Difficulty amplifier ("m6-band-supply"). The m6-pilot-build measurement showed
#: ``bug_combination`` difficulty against the elite sealed panel is BIMODAL on the
#: fault-COUNT knob (faults=2 -> solve-all pass@k=1.0; faults=3 -> solve-none 0.0
#: AND more import-collateral rejects), so the keep band is a knife-edge on that
#: axis. This ladder instead centers difficulty on the ORTHOGONAL "how hard is the
#: fault to LOCATE" axis: it fixes faults=2 (the count that clears the oracle
#: cleanly) and varies the TARGET-SYMBOL SIZE -- a subtle operator/off-by-one
#: fault buried in a LARGER function is a genuine needle for the frontier yet
#: still flips exactly one hidden test (isolable), while the same fault in a tiny
#: helper is obvious. Each rung is a ``{faults, min_symbol_lines, prefer}`` param
#: set handed to the ``bug_combination`` generator; sweeping the rungs x seeds
#: spreads oracle-passing candidates ACROSS the difficulty spectrum so the band
#: filter can SELECT the in-band ones (never by loosening the band). ``prefer``:
#: ``largest`` tries the biggest symbols first (needle), ``smallest`` the easy end.
DEFAULT_AMPLIFIER_LADDER: tuple[dict[str, object], ...] = (
    {"faults": 2, "min_symbol_lines": 0, "prefer": "smallest"},
    {"faults": 2, "min_symbol_lines": 10, "prefer": "largest"},
    {"faults": 2, "min_symbol_lines": 20, "prefer": "largest"},
    {"faults": 2, "min_symbol_lines": 30, "prefer": "largest"},
)

#: Structural generators the amplifier ladder is applied to. Both multi-fault
#: amplifiers (``bug_combination`` and ``multi_file``) combine >=2 distinct-file
#: faults through the same targeting, so the large-symbol rungs give each a
#: needle that reliably reaches calibration on the modular repos and lets the
#: band filter select in-band keeps from >1 generator (VAL-CROSS-005, per
#: m6-pilot-build).
AMPLIFIER_GENERATORS: frozenset[str] = frozenset({"bug_combination", "multi_file"})


@dataclass
class PilotConfig:
    """All knobs for one pilot run (Stage 0 -> Stage 5)."""

    plans: list[CandidatePlan]
    out_dir: Path
    band_config: BandFilterConfig = DEFAULT_BAND_FILTER
    frontier_threshold: float = DEFAULT_FRONTIER_THRESHOLD
    kill_threshold: float = DEFAULT_KILL_THRESHOLD
    flakiness_runs: int = DEFAULT_FLAKINESS_RUNS
    k: int | None = None
    concurrency: int = 4
    candidate_concurrency: int = 1
    validate_models: bool = True
    command_timeout: float = 600.0
    mutation_timeout: float = 1200.0
    gold_eval_runs: int = 2
    run_gold_eval: bool = True
    write_report: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Require two fresh Docker proofs before Headline A can be claimed."""
        if self.gold_eval_runs < 2:
            raise PilotError(
                "gold_eval_runs must be >= 2 for strict Headline A proof; "
                f"got {self.gold_eval_runs}"
            )


def build_pilot_plans(
    *,
    registry: SourceRegistry | None = None,
    generators_by_language: dict[str, tuple[str, ...]] | None = None,
    seeds_per_cell: int = 2,
    languages: Sequence[str] | None = None,
    max_plans: int | None = None,
    include_pr_mirror: bool = True,
    include_structural: bool = True,
    amplifier_ladder: Sequence[dict[str, object]] | None = None,
) -> list[CandidatePlan]:
    """Build the candidate plan list (Stage 0 sourcing) for a pilot run.

    Two plan families are emitted from the curated source registry:

    * a **``pr_mirror``** plan for every allowlist entry
      (:attr:`RepoSpec.has_pr_mirror`), pinned to that entry's own
      ``base_commit`` and carrying ``params={"repo", "pr_number"}`` so the
      generator reconstructs the reverted (isolated-F2P) fault; and
    * **structural-generator** plans (``ast_mutation``/``function_removal``/...)
      for every diversified MODULAR repo (:attr:`RepoSpec.structural_source`),
      enumerated over ``generator x seed`` cells.

    The difficulty **amplifier** generators (``bug_combination``, ``multi_file``)
    are additionally
    swept over the :data:`DEFAULT_AMPLIFIER_LADDER` (``amplifier_ladder``
    override): each ``(generator, seed)`` cell is expanded to one plan per ladder
    rung, and the rung's ``{faults, min_symbol_lines, prefer}`` are threaded as
    generation ``params`` so per-fault difficulty is CENTERED on the band (varying
    the fault-LOCATE difficulty via target-symbol size) rather than bimodal on the
    fault-COUNT knob. Non-amplifier structural generators keep the plain
    one-plan-per-cell behavior.

    The per-repo cap bounds the SHIPPED set (acquired at keep time), not the
    candidate count, so the funnel is fed many candidates per repo and the band
    filter selects the in-band ones. The GitHub token the ``pr_mirror`` generator
    needs is read from the environment at run time and never appears in a plan.
    """
    src = registry or build_source_registry()
    gens = generators_by_language or DEFAULT_GENERATORS_BY_LANGUAGE
    ladder = list(
        amplifier_ladder if amplifier_ladder is not None else DEFAULT_AMPLIFIER_LADDER
    )
    plans: list[CandidatePlan] = []

    def _capped() -> bool:
        return max_plans is not None and len(plans) >= max_plans

    for repo in src.specs():
        if languages is not None and repo.language not in languages:
            continue
        if include_pr_mirror and repo.has_pr_mirror:
            plans.append(
                CandidatePlan(
                    repo=repo,
                    generator=repo.preferred_generator,
                    seed=0,
                    params=repo.pr_params(),
                )
            )
            if _capped():
                return plans
        if include_structural and repo.structural_source:
            for generator in gens.get(repo.language, ("ast_mutation",)):
                rungs: Sequence[dict[str, object]] = (
                    ladder if generator in AMPLIFIER_GENERATORS and ladder else ({},)
                )
                for seed in range(seeds_per_cell):
                    for rung in rungs:
                        plans.append(
                            CandidatePlan(
                                repo=repo,
                                generator=generator,
                                seed=seed,
                                params=dict(rung),
                            )
                        )
                        if _capped():
                            return plans
    return plans


def default_pilot_config(
    out_dir: Path | str,
    *,
    seeds_per_cell: int = 4,
    languages: Sequence[str] | None = None,
    max_plans: int | None = None,
    k: int | None = None,
    **overrides: object,
) -> PilotConfig:
    """The ``--pilot`` preset: many candidates across all three languages.

    Budgeted to feed the funnel enough borderline candidates that the band filter
    can select ~10-30 in-band keeps (the frontier is bimodal per task, so a large
    candidate count + an adequate hard-band ``k`` is how the shipped set is built
    -- never by hand-tuning one task or loosening the band).
    """
    plans = build_pilot_plans(
        seeds_per_cell=seeds_per_cell, languages=languages, max_plans=max_plans
    )
    config = PilotConfig(plans=plans, out_dir=Path(out_dir), k=k)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.validate()
    return config


# --------------------------------------------------------------------------- #
# Pilot outcome
# --------------------------------------------------------------------------- #
@dataclass
class PilotOutcome:
    """The end-to-end result of a pilot run."""

    out_dir: Path
    counts: StageCounts
    usage: PilotUsage
    dispositions: list[CandidateDisposition]
    export_result: BatchExportResult | None = None
    gold: GoldEvalReport | None = None
    report: BenchmarkReport | None = None
    capacity: list[dict[str, object]] = field(default_factory=list)

    @property
    def shipped_count(self) -> int:
        return self.counts.exported

    @property
    def generators_used(self) -> list[str]:
        return sorted(
            {d.plan.generator for d in self.dispositions if d.stage == "kept"}
        )

    @property
    def languages_shipped(self) -> list[str]:
        return sorted({d.plan.language for d in self.dispositions if d.stage == "kept"})

    @property
    def headline_a_pass(self) -> bool:
        return self.report is not None and self.report.headline_a_pass

    @property
    def headline_b_pass(self) -> bool:
        return self.report is not None and self.report.headline_b_pass

    @property
    def in_band(self) -> bool:
        """True iff the shipped count lands in the documented ~10-30 band."""
        return 10 <= self.shipped_count <= 30

    @property
    def ok(self) -> bool:
        """Exit-0 condition: funnel monotone and (if shipped) both headlines hold."""
        if not self.counts.monotone:
            return False
        if self.shipped_count == 0:
            return False
        return (
            self.report is not None
            and self.report.passed
            and len(self.generators_used) >= 2
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "out_dir": str(self.out_dir),
            "counts": self.counts.to_dict(),
            "usage": self.usage.to_dict(),
            "shipped_count": self.shipped_count,
            "generators_used": self.generators_used,
            "languages_shipped": self.languages_shipped,
            "in_band": self.in_band,
            "headline_a_pass": self.headline_a_pass,
            "headline_b_pass": self.headline_b_pass,
            "report": self.report.to_dict() if self.report is not None else None,
            "capacity": [dict(snapshot) for snapshot in self.capacity],
            "dispositions": [d.to_dict() for d in self.dispositions],
            "ok": self.ok,
        }


# --------------------------------------------------------------------------- #
# Orchestration (Stage 0 -> Stage 5)
# --------------------------------------------------------------------------- #
async def _process_plans(
    plans: Sequence[CandidatePlan],
    processor: CandidateProcessor,
    run_root: Path,
    concurrency: int,
    on_complete: Callable[[int, CandidatePlan, CandidateArtifacts], Awaitable[None]]
    | None = None,
    can_start: Callable[[], bool] | None = None,
) -> list[CandidateArtifacts]:
    """Run Stage 1-4 for every plan, returning artifacts in plan order.

    Each candidate gets its own run-scoped ``workdir`` (created up front, torn
    down with ``run_root`` by the caller). Candidates run concurrently under a
    semaphore of size ``concurrency`` (``1`` = the sequential default, which
    preserves the historical single-candidate-at-a-time behavior); the heavy
    per-candidate work overlaps so a large sweep is wall-clock tractable while
    every stage still owns its own throwaway-container hygiene. Results are
    placed by index so the funnel counting + export downstream stay deterministic
    regardless of completion order.

    ``on_complete`` (when given) is awaited for each candidate the instant its
    artifacts are ready -- WHILE the sweep is still running and still holding this
    candidate's semaphore slot -- so a keep is checkpointed/shipped as it is found
    rather than after every plan finishes. If a processor (or the callback) raises,
    the remaining tasks are cancelled and the error propagates (the caller's
    ``finally`` still cleans up), but every keep already checkpointed stays shipped.
    """
    width = max(1, int(concurrency))
    if width == 1:
        # Do not pre-schedule the next candidate in sequential mode.  Releasing a
        # semaphore while a failing task unwinds lets the next waiter start before
        # ``gather`` observes the exception and cancels it, which can checkpoint an
        # unexpected extra keep after a budget/time interruption.
        sequential_results: list[CandidateArtifacts] = []
        for index, plan in enumerate(plans):
            if can_start is not None and not can_start():
                break
            workdir = run_root / f"cand-{index:04d}"
            workdir.mkdir(parents=True, exist_ok=True)
            art = await processor.process(plan, workdir)
            if on_complete is not None:
                await on_complete(index, plan, art)
            sequential_results.append(art)
        return sequential_results

    semaphore = asyncio.Semaphore(width)
    results: list[CandidateArtifacts | None] = [None] * len(plans)

    async def _one(index: int, plan: CandidatePlan) -> None:
        if can_start is not None and not can_start():
            return
        workdir = run_root / f"cand-{index:04d}"
        workdir.mkdir(parents=True, exist_ok=True)
        async with semaphore:
            if can_start is not None and not can_start():
                return
            art = await processor.process(plan, workdir)
            results[index] = art
            if on_complete is not None:
                await on_complete(index, plan, art)

    tasks = [asyncio.create_task(_one(index, plan)) for index, plan in enumerate(plans)]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [art for art in results if art is not None]


@dataclass
class _PilotShutdown:
    """Coordinates signal admission closure with the checkpoint writer drain."""

    checkpoint: PilotCheckpoint
    requested: bool = False

    def request(self) -> bool:
        """Close keep admission once, returning whether this is the first signal."""
        if self.requested:
            return False
        self.requested = True
        self.checkpoint.close_admission()
        return True


async def _drain_checkpoint_before_cleanup(checkpoint: PilotCheckpoint) -> bool:
    """Drain all accepted checkpoint I/O, deferring cancellation until it settles.

    ``asyncio.shield`` alone protects the inner task but still raises
    ``CancelledError`` in the outer task. Re-await the same drain task until it
    completes so a follow-up cancellation can never let source cleanup race a
    background ``copytree``. Returns whether cancellation was deferred.
    """
    drain = asyncio.create_task(checkpoint.drain(continue_on_error=True))
    deferred_cancellation = False
    while True:
        try:
            await asyncio.shield(drain)
            return deferred_cancellation
        except asyncio.CancelledError:
            deferred_cancellation = True


def _keep_export_request(art: CandidateArtifacts) -> ExportRequest | None:
    """The :class:`ExportRequest` for an oracle-pass AND band-keep candidate.

    Returns ``None`` for a candidate that failed env build / generation / oracle /
    calibration, so a rejection/drop is never checkpointed (never materializes a
    workspace or dataset row). This mirrors exactly the funnel gate the post-sweep
    counting loop applies, so the checkpointed set == the kept set.
    """
    oracle = art.oracle_report
    calibration = art.calibration_report
    if (
        art.env_image is None
        or art.candidate is None
        or oracle is None
        or not oracle.is_pass
        or calibration is None
        or not calibration.is_keep
    ):
        return None
    return ExportRequest(
        candidate=art.candidate,
        spec=_require_spec(art),
        oracle_report=oracle,
        calibration_report=calibration,
        env_image=art.env_image,
        repo_url=art.repo_url,
        base_commit=art.base_commit,
        broken_tree=art.broken_tree,
    )


def _install_shutdown_handlers(shutdown: _PilotShutdown) -> Callable[[], None]:
    """Close keep admission then cancel the running pilot on SIGTERM/SIGINT.

    The first signal closes checkpoint admission *before* cancelling the sweep.
    This creates a stable boundary: every keep admitted before the signal is
    drained by ``run_pilot``'s cleanup path, and a keep completing after it cannot
    enter a new copy/publication. Repeated signals are ignored while the drain is
    in progress so they cannot interrupt source-tree protection. Best-effort: a
    platform without ``add_signal_handler`` (or a non-main thread) simply runs
    without the hook.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - run_pilot is always awaited
        return lambda: None
    task = asyncio.current_task()
    if task is None:  # pragma: no cover - defensive
        return lambda: None

    installed: list[signal.Signals] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                sig,
                lambda: task.cancel() if shutdown.request() else None,
            )
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):  # pragma: no cover
            continue

    def _remove() -> None:
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):  # pragma: no cover
                continue

    return _remove


async def run_pilot(
    config: PilotConfig,
    *,
    processor: CandidateProcessor | None = None,
    gold_eval_fn: GoldEvalFn | None = None,
    handle_signals: bool = False,
) -> PilotOutcome:
    """Drive Stage 0 -> Stage 5 for ``config`` and return the :class:`PilotOutcome`.

    For each sourced candidate plan the processor runs Stage 1-4; the moment a
    candidate is calibrated oracle-pass AND band-keep it is CHECKPOINTED -- its
    workspace + jsonl/parquet row + provenance are materialized immediately via
    the reused Stage-5 export path (see :class:`PilotCheckpoint`) -- so a run
    stopped at any point (SIGTERM / budget-or-time ceiling / crash) has already
    shipped every keep found so far, not 0. Rejections/drops never propagate (only
    an oracle-pass AND band-keep candidate is ever materialized, and the export
    layer re-checks the gate). After the sweep the orchestrator counts the monotone
    funnel, finalizes the plan-ordered datasets (byte-identical to a one-shot
    export of the same kept set), runs gold-eval in Docker (Headline A), and builds
    the benchmark report (Headline B + provenance audit + count reconciliation).

    With ``handle_signals=True`` a clean shutdown hook cancels the in-flight sweep
    on SIGTERM/SIGINT so throwaway containers are torn down (by id, via each
    stage's ``async with`` sandbox) and the already-written checkpoint is left
    valid; off-limits containers are never named, so they are never touched. Host
    temp dirs are always cleaned up.
    """
    config.validate()
    if processor is None:
        processor = _live_processor(config)

    usage = PilotUsage()
    dispositions: list[CandidateDisposition] = []
    run_root = Path(tempfile.mkdtemp(prefix="forge-pilot-run-"))
    cleanup: list[Path] = [run_root]

    # Stage 5 is now INCREMENTAL: each band-keep is materialized (workspace +
    # jsonl/parquet row + provenance) the instant it is found, reusing the export
    # path unchanged, so an interruption never loses the keeps found so far.
    checkpoint = PilotCheckpoint(
        config.out_dir,
        overwrite=True,
        source_specs=(plan.repo for plan in config.plans),
    )
    recovered = checkpoint.kept_count
    # A continuation starts from the committed generation selected by the
    # publication pointer. These tasks were already oracle-pass, calibration-
    # keep, and cap-admitted, so their baseline funnel count must match the
    # restored source usage before new completions are processed.
    counts = StageCounts(
        sourced=recovered,
        env_built=recovered,
        synthesized=recovered,
        oracle_pass=recovered,
        calibration_keep=recovered,
        cap_admitted=recovered,
        exported=recovered,
    )
    recovered_indexes = frozenset(checkpoint.committed_indexes)
    shutdown = _PilotShutdown(checkpoint)
    completed: dict[int, CandidateArtifacts] = {}
    deferred_cancellation = False

    async def _on_complete(
        index: int, plan: CandidatePlan, art: CandidateArtifacts
    ) -> None:
        # Record finished candidates before awaiting checkpoint I/O.  A signal can
        # cancel that await, but the checkpoint admission/pending ledger still
        # retains a keep that entered before the shutdown boundary.
        completed[index] = art
        cleanup.extend(art.cleanup_paths)
        # A process restart may replay the full plan list. A task already in the
        # committed generation is part of the recovered funnel baseline, so do
        # not acquire capacity or publish it a second time.
        if index in recovered_indexes:
            return
        request = _keep_export_request(art)
        if request is not None:
            # Admission has no await boundary, so SIGTERM either closes the gate
            # before this keep enters it or the keep is durably represented in the
            # pending ledger before cancellation can reach checkpoint I/O.
            if checkpoint.admit_keep(index, request, source=plan.repo) is None:
                await checkpoint.drain(indexes=(index,))

    remove_signals = (
        _install_shutdown_handlers(shutdown) if handle_signals else (lambda: None)
    )
    try:
        interrupted = False
        try:
            artifacts = await _process_plans(
                config.plans,
                processor,
                run_root,
                getattr(config, "candidate_concurrency", 1) or 1,
                _on_complete,
                lambda: checkpoint.accepting,
            )
        except asyncio.CancelledError:
            # Only the installed signal hook converts cancellation into a
            # graceful checkpoint drain. Direct caller cancellation retains the
            # normal CancelledError contract, while ``finally`` still protects
            # any already-admitted copy from source cleanup.
            if not shutdown.requested:
                raise
            interrupted = True
        else:
            for index, art in enumerate(artifacts):
                completed.setdefault(index, art)

        # Admission is now closed for both graceful interruption and ordinary
        # completion. Shield the deterministic writer drain from a follow-up
        # cancellation before deleting the run root or any cleanup path used as a
        # source by copytree/export_forge_task.
        checkpoint.close_admission()
        deferred_cancellation = await _drain_checkpoint_before_cleanup(checkpoint)

        for index in sorted(completed):
            if index in recovered_indexes:
                continue
            plan = config.plans[index]
            art = completed[index]
            counts.sourced += 1
            usage.add(art)

            if art.env_image is None:
                dispositions.append(
                    CandidateDisposition(plan, "env_failed", reason=art.failure_reason)
                )
                continue
            counts.env_built += 1

            if art.candidate is None:
                dispositions.append(
                    CandidateDisposition(
                        plan, "synth_failed", reason=art.failure_reason
                    )
                )
                continue
            counts.synthesized += 1

            oracle = art.oracle_report
            if oracle is None or not oracle.is_pass:
                dispositions.append(
                    CandidateDisposition(
                        plan,
                        "oracle_reject",
                        oracle_verdict=oracle.verdict if oracle else "missing",
                        reason="; ".join(oracle.reasons)
                        if oracle
                        else art.failure_reason,
                    )
                )
                continue
            counts.oracle_pass += 1

            calibration = art.calibration_report
            if calibration is None or not calibration.is_keep:
                dispositions.append(
                    CandidateDisposition(
                        plan,
                        "calib_drop",
                        oracle_verdict=oracle.verdict,
                        band_verdict=calibration.band_verdict
                        if calibration
                        else "missing",
                        reason=calibration.reasons[0]
                        if calibration and calibration.reasons
                        else "",
                        calibration=_calibration_summary(calibration),
                    )
                )
                continue

            counts.calibration_keep += 1
            cap_grant = checkpoint.capacity_grant(index)
            if cap_grant is not None and not cap_grant.accepted:
                dispositions.append(
                    CandidateDisposition(
                        plan,
                        "cap_rejected",
                        oracle_verdict=oracle.verdict,
                        band_verdict=calibration.band_verdict,
                        reason=cap_grant.reason,
                        cap_grant=cap_grant.to_dict(),
                        calibration=_calibration_summary(calibration),
                    )
                )
                continue

            # Oracle pass AND band keep is retained only if it crossed the
            # checkpoint admission boundary. A SIGTERM may let an in-flight
            # candidate finish calculation, but it cannot add a new keep after
            # admission has closed.
            if not checkpoint.was_accepted(index):
                dispositions.append(
                    CandidateDisposition(
                        plan,
                        "shutdown_rejected",
                        oracle_verdict=oracle.verdict,
                        band_verdict=calibration.band_verdict,
                        reason="checkpoint admission closed before keep acceptance",
                        calibration=_calibration_summary(calibration),
                    )
                )
                continue

            checkpoint_result = checkpoint.result_for(index)
            if checkpoint_result is not None and checkpoint_result.status not in (
                "shipped",
                "skipped",
            ):
                counts.export_refused += 1
                dispositions.append(
                    CandidateDisposition(
                        plan,
                        "export_refused",
                        oracle_verdict=oracle.verdict,
                        band_verdict=calibration.band_verdict,
                        reason=checkpoint_result.reason,
                        cap_grant=cap_grant.to_dict()
                        if cap_grant is not None
                        else None,
                        calibration=_calibration_summary(calibration),
                    )
                )
                continue

            # An accepted SourceRegistry grant was serialized before any
            # checkpoint I/O. Only these cap-admitted keeps may publish.
            counts.cap_admitted += 1
            dispositions.append(
                CandidateDisposition(
                    plan,
                    "kept",
                    oracle_verdict=oracle.verdict,
                    band_verdict=calibration.band_verdict,
                    cap_grant=cap_grant.to_dict() if cap_grant is not None else None,
                    calibration=_calibration_summary(calibration),
                )
            )

        # Finalize: rewrite the plan-ordered datasets (idempotent + byte-identical
        # to a one-shot export of the same kept set) and collect the per-candidate
        # ship/refuse ledger. A refused keep (e.g. a planted leak) is left out of
        # the datasets, so ``exported < calibration_keep`` surfaces the problem.
        export_result = checkpoint.finalize()
        counts.exported = len(export_result.kept)
        _attach_task_ids(dispositions, export_result)

        gold: GoldEvalReport | None = None
        if not interrupted and config.run_gold_eval and export_result.shipped:
            evaluate = gold_eval_fn or run_gold_eval
            try:
                gold = evaluate(config.out_dir, runs=config.gold_eval_runs)
            except GoldEvalError:
                gold = None

        capacity = checkpoint.capacity_snapshot()
        report = (
            _build_report(
                config,
                gold,
                funnel=counts.to_dict(),
                source_capacity=capacity,
            )
            if not interrupted
            else None
        )
        if not interrupted and config.write_report and report is not None:
            write_report(report, config.out_dir)

        return PilotOutcome(
            out_dir=Path(config.out_dir),
            counts=counts,
            usage=usage,
            dispositions=dispositions,
            export_result=export_result,
            gold=gold,
            report=report,
            capacity=capacity,
        )
    finally:
        # This final drain also covers direct task cancellation.  It is essential
        # that all accepted worker-thread copies settle before deleting run_root
        # or processor-provided cleanup paths that can be their broken_tree source.
        checkpoint.close_admission()
        try:
            deferred_cancellation = (
                await _drain_checkpoint_before_cleanup(checkpoint)
            ) or deferred_cancellation
        finally:
            remove_signals()
            for path in cleanup:
                shutil.rmtree(path, ignore_errors=True)
            if deferred_cancellation and not shutdown.requested:
                raise asyncio.CancelledError


def _live_processor(config: PilotConfig) -> LiveCandidateProcessor:
    """Build the real Stage 1-4 processor from the env-configured panel."""
    base_url, api_key = resolve_panel_endpoint()
    if not base_url or not api_key:
        raise PilotError(
            "no panel endpoint/credentials configured; export TEACHER_LLM_* "
            "(and optionally PANEL_LLM_*) before running the pilot"
        )
    panel = build_panel_from_env()
    return LiveCandidateProcessor(
        panel=panel,
        band_config=config.band_config,
        kill_threshold=config.kill_threshold,
        flakiness_runs=config.flakiness_runs,
        k=config.k,
        concurrency=config.concurrency,
        validate_models=config.validate_models,
        command_timeout=config.command_timeout,
        mutation_timeout=config.mutation_timeout,
    )


def _require_spec(art: CandidateArtifacts) -> GeneratedSpec:
    if art.spec is None:  # pragma: no cover - a kept candidate always has a spec
        raise PilotError(
            f"kept candidate {art.plan.label!r} is missing its GeneratedSpec"
        )
    return art.spec


def _calibration_summary(
    report: CalibrationReport | None,
) -> dict[str, object] | None:
    """Observability snapshot of a calibration run (``None`` if none ran).

    Surfaces the per-tier pass@k, the frontier pass@k, the fitted discrimination,
    and the applied band rule so a run ledger can attribute WHY a candidate was
    kept or calib-dropped (solve-all/too-easy vs solve-none/too-hard vs
    low-discrimination) without re-parsing the free-text reason. Read-only: it
    never influences the keep/drop decision.
    """
    if report is None:
        return None
    band = report.details.get("band_filter")
    rule = band.get("rule") if isinstance(band, dict) else None
    return {
        "frontier_pass_at_k": report.frontier_pass_at_k(),
        "tier_pass_rates": report.tier_pass_rates(),
        "discrimination": report.irt_discrimination,
        "band_rule": rule,
    }


def _attach_task_ids(
    dispositions: list[CandidateDisposition], result: BatchExportResult
) -> None:
    """Best-effort: map each kept disposition to its exported task id (by order)."""
    shipped_ids = [
        r.task_id for r in result.results if r.status in ("shipped", "skipped")
    ]
    kept = [d for d in dispositions if d.stage == "kept"]
    for disposition, task_id in zip(kept, shipped_ids):
        disposition.task_id = task_id


def _build_report(
    config: PilotConfig,
    gold: GoldEvalReport | None,
    *,
    funnel: dict[str, object] | None = None,
    source_capacity: list[dict[str, object]] | None = None,
) -> BenchmarkReport | None:
    try:
        return build_benchmark_report(
            config.out_dir,
            gold=gold,
            frontier_threshold=config.frontier_threshold,
            band_config=config.band_config,
            kill_threshold=config.kill_threshold,
            funnel=funnel,
            source_capacity=source_capacity,
        )
    except Exception:  # noqa: BLE001 - a report build failure is surfaced as None
        return None


__all__ = [
    "AMPLIFIER_GENERATORS",
    "DEFAULT_AMPLIFIER_LADDER",
    "DEFAULT_GENERATORS_BY_LANGUAGE",
    "CandidateArtifacts",
    "CandidateDisposition",
    "CandidatePlan",
    "CandidateProcessor",
    "LiveCandidateProcessor",
    "PilotConfig",
    "PilotError",
    "PilotOutcome",
    "PilotUsage",
    "StageCounts",
    "build_pilot_plans",
    "candidate_trace",
    "default_pilot_config",
    "run_pilot",
]
