"""Mutation-adequacy gate: prove the hidden suite pins down the gold behavior.

The third oracle-hardening gate (architecture S6, Stage 3.3). Establish + flakiness
guarantee a *deterministic* FAIL->PASS contract, but a suite can be deterministic
and still under-determined: it can pass the gold yet fail to distinguish gold from
*subtly wrong* code (the "pass-but-wrong" gap). This gate closes that gap by
mutating the **gold** code with the language's mutation tool (cosmic-ray | Stryker |
go-mutesting, via the :class:`~swe_forge.forge.adapters.base.LanguageAdapter`) and
requiring the established F2P+P2P suite to *kill* at least a configured fraction of
the generated mutants.

Flow (``the teacher proposes, deterministic execution disposes``):

1. Run the mutation tool against the gold target file(s) in a throwaway
   :class:`~swe_forge.execution.sandbox.DockerSandbox` with the established hidden
   tests written in; record ``mutants_total``/``mutants_killed`` consistent with the
   tool's output.
2. If the kill ratio is already ``>= threshold`` the gate passes with no synthesis.
3. Otherwise enter a **bounded** auto-synthesis loop: the teacher proposes extra
   tests targeting the surviving mutants; each proposal is *confirmed* by re-running
   the mutation tool and kept only if it actually reduces the survivor count (i.e.
   it killed a previously-surviving mutant). New test ids are appended and the ratio
   is re-measured until the threshold is met.
4. If the bounded loop cannot reach the threshold, the candidate is **rejected**
   with a reason citing the surviving mutants (the oracle is provably
   under-determined).

The gate scores through the SAME Docker primitives as the rest of the oracle, and
honors the Python ``.pyc`` re-test determinism invariant (handled inside the
adapter's ``mutation_tool_run``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Protocol

from swe_forge.forge.adapters import (
    LanguageAdapter,
    MutantStats,
    MutationExecutor,
    build_default_registry,
)
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    FinalMutationEvidence,
    OracleReport,
    OracleTestFile,
    Provenance,
    require_green_baseline,
)
from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from swe_forge.execution.docker_client import DockerClient

#: Default minimum fraction of generated mutants the suite must kill.
DEFAULT_KILL_THRESHOLD = 0.8
#: How many auto-synthesis rounds the gate attempts before rejecting.
DEFAULT_MAX_SYNTHESIS_ROUNDS = 3

#: Difficulty-amplifier generators whose mutation run is scoped to the changed
#: region. A ``bug_combination`` / ``multi_file`` candidate mutates >=2 symbols on
#: large MODULAR modules; mutating each WHOLE module is hundreds of mutants and
#: does not finish within ``mutation_timeout``, so the runner restricts cosmic-ray
#: to the changed-symbol line ranges (the same file/region scoping the Go/JS
#: tooling already do).
AMPLIFIER_GENERATORS: frozenset[str] = frozenset({"bug_combination", "multi_file"})

#: Above this many lines, a faulted symbol is a LARGE function whose WHOLE-span
#: cosmic-ray run does not finish within ``mutation_timeout``. The m6-pilot-build
#: sweep measured ~28min mutation timeouts on boltons ``bug_combination`` HARD
#: rungs (``min_symbol_lines=20/30``, ``prefer=largest``): the amplifier buries a
#: subtle single fault inside a LARGE function, so scoping to the WHOLE enclosing
#: symbol is still hundreds of mutants and blocks a candidate-concurrency slot ~28
#: minutes before the exec-timeout reaper drops it. For such a symbol the run is
#: narrowed to only the actually-CHANGED lines (the ``mutation_patch`` hunks,
#: padded by :data:`REGION_PAD_LINES` and clamped to the symbol) -- bounding the
#: run to the fault region so it finishes fast instead of timing out. A symbol
#: at/under this size keeps the whole-span behavior, so candidates whose mutation
#: run already finished in budget compute the SAME kill ratio / verdict as before.
MAX_SCOPED_SYMBOL_LINES = 40
#: Lines of context kept on each side of a changed-line hunk when a LARGE symbol
#: span is narrowed, so cosmic-ray still enumerates the mutable nodes flanking the
#: fault (a meaningful adequacy sample) without mutating the whole large function.
REGION_PAD_LINES = 8

# ``@@ -old_start,old_count +new_start,new_count @@`` unified-diff hunk header.
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
# ``+++ b/path`` (or ``+++ path``) target-file marker, tab-trimmed.
_PLUS_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+?)(?:\t.*)?$")

# Attributable reject reason prefixes (stable keys the contract/CLI gate on).
REASON_MUTATION_INADEQUATE = "mutation_adequacy_below_threshold"
REASON_NO_MUTANTS = "mutation_no_mutants_generated"


class MutationError(RuntimeError):
    """Raised for an unrecoverable failure while driving the mutation gate."""


def patch_old_side_regions(patch_text: str) -> dict[str, list[tuple[int, int]]]:
    """Return per-file GOLD-side (old) changed line ranges from a unified diff.

    The forward ``mutation_patch`` transforms gold -> broken, so each hunk's OLD
    side (``@@ -old_start,old_count``) is the GOLD line range the fault touches --
    exactly the lines cosmic-ray must mutate. Pure-insertion hunks (``old_count ==
    0``) contribute no range. Ranges are 1-based and inclusive, keyed by the
    repo-relative path from the ``+++ b/<path>`` marker.
    """
    regions: dict[str, list[tuple[int, int]]] = {}
    current: str | None = None
    for line in patch_text.splitlines():
        file_match = _PLUS_FILE_RE.match(line)
        if file_match:
            path = file_match.group(1).strip()
            current = None if path == "/dev/null" else path
            continue
        hunk = _HUNK_RE.match(line)
        if hunk and current is not None:
            start = int(hunk.group(1))
            count = int(hunk.group(2)) if hunk.group(2) is not None else 1
            if count > 0:
                regions.setdefault(current, []).append((start, start + count - 1))
    return regions


def _provenance_symbol_regions(
    candidate: Candidate,
) -> dict[str, list[tuple[int, int]]]:
    """Per-file changed-symbol line spans recorded in the candidate provenance.

    The difficulty-amplifier generators record each constituent fault/edit's
    enclosing-symbol span (``start_line``/``end_line``) under ``faults``/``edits``.
    A SMALL faulted symbol is mutated whole (the span keeps enough mutable nodes
    for a meaningful adequacy measurement); a LARGE one (over
    :data:`MAX_SCOPED_SYMBOL_LINES`) is narrowed by :func:`candidate_mutation_regions`
    to only the changed lines so the run does not time out on the whole function.
    """
    regions: dict[str, list[tuple[int, int]]] = {}
    details = candidate.provenance.details if candidate.provenance else {}
    if not isinstance(details, dict):
        return regions
    records = details.get("faults")
    if not isinstance(records, list):
        records = details.get("edits")
    if not isinstance(records, list):
        return regions
    for record in records:
        if not isinstance(record, dict):
            continue
        path = record.get("file")
        start = record.get("start_line")
        end = record.get("end_line")
        if (
            isinstance(path, str)
            and path
            and isinstance(start, int)
            and isinstance(end, int)
            and end >= start
        ):
            regions.setdefault(path, []).append((start, end))
    return regions


def _narrow_to_changed_lines(
    hunks: Sequence[tuple[int, int]],
    symbol_spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Bound a LARGE faulted symbol's mutation run to its actually-CHANGED lines.

    Pads each ``mutation_patch`` hunk range by :data:`REGION_PAD_LINES` on each
    side (keeping the mutable nodes flanking the fault) and clamps it to the
    enclosing symbol span so the scoped region never spills past the changed
    symbol. This replaces the whole-large-function span with a small,
    fault-centered region so cosmic-ray finishes fast instead of timing out; the
    padding still leaves a meaningful adequacy sample.
    """
    narrowed: list[tuple[int, int]] = []
    for lo, hi in hunks:
        padded_lo = max(1, lo - REGION_PAD_LINES)
        padded_hi = hi + REGION_PAD_LINES
        enclosing = [(s, e) for s, e in symbol_spans if s <= lo and hi <= e]
        if enclosing:
            sym_lo = min(s for s, _ in enclosing)
            sym_hi = max(e for _, e in enclosing)
            padded_lo = max(padded_lo, sym_lo)
            padded_hi = min(padded_hi, sym_hi)
        narrowed.append((padded_lo, padded_hi))
    return narrowed


def candidate_mutation_regions(
    candidate: Candidate,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Per-file GOLD line ranges to scope an amplifier candidate's mutation run.

    Per target file, the scoped region depends on the size of the faulted symbol:

    * a SMALL symbol (span at/under :data:`MAX_SCOPED_SYMBOL_LINES`) keeps the
      union of the provenance-recorded symbol span with the ``mutation_patch``
      hunk ranges -- the generous whole-symbol scoping, so a candidate whose
      mutation run already finished in budget computes the SAME kill ratio /
      verdict as before;
    * a LARGE symbol (the ``bug_combination`` / ``multi_file`` HARD rungs whose
      whole-span run times out ~28min) is narrowed to only the changed lines via
      :func:`_narrow_to_changed_lines`, bounding the run to the fault region.

    Returns ``{}`` when no ranges can be derived (the runner then mutates whole
    files, the safe default). A large symbol with no derivable hunks also keeps
    its whole span (never worse than the pre-narrowing behavior).
    """
    symbol_regions = _provenance_symbol_regions(candidate)
    hunk_regions = patch_old_side_regions(candidate.mutation_patch)
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for path in set(symbol_regions) | set(hunk_regions):
        symbols = symbol_regions.get(path, [])
        hunks = hunk_regions.get(path, [])
        largest_symbol = max((hi - lo + 1 for lo, hi in symbols), default=0)
        if largest_symbol > MAX_SCOPED_SYMBOL_LINES and hunks:
            ranges = _narrow_to_changed_lines(hunks, symbols)
        else:
            ranges = [*symbols, *hunks]
        if ranges:
            result[path] = tuple(ranges)
    return result


@dataclass(frozen=True)
class MutationMeasurement:
    """One mutation-tool run against the gold code with the current suite."""

    total: int
    killed: int
    tool: str = ""
    survivors: tuple[str, ...] = ()

    @property
    def survived(self) -> int:
        return self.total - self.killed

    @property
    def kill_ratio(self) -> float:
        return self.killed / self.total if self.total else 0.0

    @classmethod
    def from_stats(cls, stats: MutantStats) -> MutationMeasurement:
        return cls(
            total=stats.total,
            killed=stats.killed,
            tool=stats.tool,
            survivors=tuple(stats.survivors),
        )

    def summary(self) -> dict[str, object]:
        return {
            "total": self.total,
            "killed": self.killed,
            "survived": self.survived,
            "kill_ratio": round(self.kill_ratio, 4),
            "tool": self.tool,
        }


class MutationRunner(Protocol):
    """The mutation-measurement surface the gate drives (Docker-backed in prod)."""

    @property
    def language(self) -> str: ...

    async def measure(
        self, extra_tests: Sequence[HiddenTest]
    ) -> MutationMeasurement: ...

    async def read_sources(self) -> dict[str, str]: ...


@dataclass
class MutationSynthesisContext:
    """Inputs handed to a :class:`MutationTestSynthesizer` for one round.

    ``sources`` maps each gold target file to its content, ``survivors`` are the
    tool's descriptions of the mutants still alive, and ``existing_test_paths``
    are the paths already added this run (so a synthesizer never collides).
    """

    candidate: Candidate
    adapter: LanguageAdapter
    sources: dict[str, str]
    survivors: tuple[str, ...]
    round_index: int
    existing_test_paths: tuple[str, ...] = ()

    @property
    def language(self) -> str:
        return self.candidate.language


class MutationTestSynthesizer(Protocol):
    """Proposes extra hidden tests intended to kill surviving mutants."""

    async def __call__(self, ctx: MutationSynthesisContext) -> list[HiddenTest]: ...


class NullMutationSynthesizer:
    """A synthesizer that proposes nothing (offline/deterministic default)."""

    async def __call__(self, ctx: MutationSynthesisContext) -> list[HiddenTest]:
        return []


@dataclass
class MutationOutcome:
    """The result of the mutation-adequacy gate (folded into an OracleReport)."""

    verdict: str
    reasons: list[str]
    mutants_total: int
    mutants_killed: int
    threshold: float
    tool: str = ""
    added_tests: list[HiddenTest] = field(default_factory=list)
    rounds: int = 0
    survivors: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def kill_ratio(self) -> float:
        return self.mutants_killed / self.mutants_total if self.mutants_total else 0.0

    @property
    def is_pass(self) -> bool:
        return self.verdict == "pass"


async def assess_mutation(
    runner: MutationRunner,
    *,
    synthesizer: MutationTestSynthesizer | None = None,
    context_template: MutationSynthesisContext | None = None,
    threshold: float = DEFAULT_KILL_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_SYNTHESIS_ROUNDS,
) -> MutationOutcome:
    """Measure mutation adequacy and auto-synthesize tests until threshold met.

    Starts from the established suite (``runner.measure([])``). If the kill ratio
    is below ``threshold``, runs up to ``max_rounds`` synthesis rounds; in each
    round the ``synthesizer`` proposes extra tests and every proposal is confirmed
    by re-measuring - kept only if it reduces the survivor count (it killed a
    previously-surviving mutant). Rejects when no mutants are generated or the
    bounded loop cannot reach the threshold.
    """
    if not 0.0 < threshold <= 1.0:
        raise MutationError(f"threshold must be in (0, 1]; got {threshold}")

    base = await runner.measure([])
    round_records: list[dict[str, object]] = []
    details: dict[str, object] = {
        "stage": "mutation",
        "threshold": threshold,
        "tool": base.tool,
        "initial": base.summary(),
        "rounds": round_records,
    }

    if base.total == 0:
        return MutationOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_NO_MUTANTS}: the mutation tool generated 0 mutants for the "
                "gold target; adequacy cannot be established"
            ],
            mutants_total=0,
            mutants_killed=0,
            threshold=threshold,
            tool=base.tool,
            details=details,
        )

    current = base
    accepted: list[HiddenTest] = []

    if current.kill_ratio < threshold and synthesizer is not None:
        if context_template is None:
            raise MutationError(
                "a synthesis context template is required when a synthesizer is set"
            )
        sources: dict[str, str] = {}
        with contextlib.suppress(Exception):
            sources = await runner.read_sources()

        for round_index in range(1, max_rounds + 1):
            ctx = dataclasses.replace(
                context_template,
                sources=sources,
                survivors=current.survivors,
                round_index=round_index,
                existing_test_paths=tuple(f.path for t in accepted for f in t.files),
            )
            proposals = await synthesizer(ctx)
            accepted_this_round = 0
            for proposal in proposals:
                trial = await runner.measure([*accepted, proposal])
                if trial.survived < current.survived:
                    accepted.append(proposal)
                    current = trial
                    accepted_this_round += 1
                    if current.kill_ratio >= threshold:
                        break
            round_records.append(
                {
                    "round": round_index,
                    "proposed": len(proposals),
                    "accepted": accepted_this_round,
                    "survived_after": current.survived,
                    "kill_ratio_after": round(current.kill_ratio, 4),
                }
            )
            if current.kill_ratio >= threshold or accepted_this_round == 0:
                break

    details["final"] = current.summary()
    details["added_test_paths"] = [f.path for t in accepted for f in t.files]

    if current.kill_ratio >= threshold:
        return MutationOutcome(
            verdict="pass",
            reasons=[],
            mutants_total=current.total,
            mutants_killed=current.killed,
            threshold=threshold,
            tool=current.tool,
            added_tests=accepted,
            rounds=len(round_records),
            survivors=list(current.survivors),
            details=details,
        )

    reason = (
        f"{REASON_MUTATION_INADEQUATE}: kill ratio "
        f"{current.kill_ratio:.2f} < threshold {threshold:.2f} after "
        f"{len(round_records)} synthesis round(s); {current.survived} surviving "
        f"mutant(s) remain (oracle under-determined)"
    )
    return MutationOutcome(
        verdict="reject",
        reasons=[reason],
        mutants_total=current.total,
        mutants_killed=current.killed,
        threshold=threshold,
        tool=current.tool,
        added_tests=accepted,
        rounds=len(round_records),
        survivors=list(current.survivors),
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


def build_mutation_report(
    candidate: Candidate,
    prior_report: OracleReport,
    outcome: MutationOutcome,
    *,
    env_image: EnvImage | None = None,
    extra_details: dict[str, object] | None = None,
) -> OracleReport:
    """Fold a :class:`MutationOutcome` into the running :class:`OracleReport`.

    Carries the establish + flakiness fields forward, sets
    ``mutants_total``/``mutants_killed``, appends any synthesized
    survivor-killing tests to ``test_files``, and sets the terminal verdict
    (``pass`` when the kill ratio reached threshold; ``reject`` with an
    attributable mutation-adequacy reason citing surviving mutants otherwise).
    """
    details: dict[str, object] = dict(prior_report.details)
    details["mutation"] = outcome.details
    if env_image is not None:
        details.setdefault("env_image", env_image.image_tag)
    if extra_details:
        details.update(extra_details)

    test_files = list(prior_report.test_files)
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
        tool_versions=_tool_versions({outcome.tool: ""} if outcome.tool else None)
        if base_prov is None
        else {
            **base_prov.tool_versions,
            **({outcome.tool: ""} if outcome.tool else {}),
        },
        details={
            "stage": "oracle.mutation",
            "mutants_total": outcome.mutants_total,
            "mutants_killed": outcome.mutants_killed,
            "kill_ratio": round(outcome.kill_ratio, 4),
            "threshold": outcome.threshold,
            "synthesis_rounds": outcome.rounds,
            "added_tests": [f.path for t in outcome.added_tests for f in t.files],
            "mutation_tool": outcome.tool,
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
        mutants_total=outcome.mutants_total,
        mutants_killed=outcome.mutants_killed,
        provenance=provenance,
        details=details,
    )


def final_suite_fingerprint(test_files: Sequence[OracleTestFile]) -> str:
    """Return a canonical SHA-256 fingerprint for exported hidden test bodies.

    Export writes only non-empty hidden test files and does not preserve their
    in-memory ordering. The fingerprint therefore sorts by path and hashes each
    exact body, making it stable under ordering changes but sensitive to every
    exported test path/content change. Duplicate paths are ambiguous and reject
    before an evidence claim can be trusted.
    """
    files: list[dict[str, str]] = []
    seen: set[str] = set()
    for test_file in test_files:
        if not test_file.content:
            continue
        if test_file.path in seen:
            raise MutationError(
                "cannot fingerprint final hidden suite with duplicate test path "
                f"{test_file.path!r}"
            )
        seen.add(test_file.path)
        # ``write_workspace`` appends exactly one newline when a non-empty test
        # body lacks one. Fingerprint the same canonical bytes that export writes.
        content = (
            test_file.content
            if test_file.content.endswith("\n")
            else test_file.content + "\n"
        )
        files.append(
            {
                "path": test_file.path,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    canonical = json.dumps(
        sorted(files, key=lambda item: item["path"]),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_final_mutation_report(
    candidate: Candidate,
    prior_report: OracleReport,
    outcome: MutationOutcome,
    *,
    threshold: float,
    env_image: EnvImage | None = None,
) -> OracleReport:
    """Replace mutation evidence after suite-mutating oracle gates complete.

    The initial mutation gate may synthesize tests, and the differential or
    alt-correct gates can later add or prune them. This final measurement is
    deliberately no-synthesis: it records whether the *already final* suite is
    adequate without changing the suite it certifies.
    """
    if outcome.added_tests:
        raise MutationError(
            "final mutation remeasurement must not synthesize or append test files"
        )
    if outcome.threshold != threshold:
        raise MutationError(
            "final mutation remeasurement threshold does not match configured "
            f"threshold ({outcome.threshold} != {threshold})"
        )

    evidence = FinalMutationEvidence(
        suite_fingerprint=final_suite_fingerprint(prior_report.test_files),
        mutants_total=outcome.mutants_total,
        mutants_killed=outcome.mutants_killed,
        threshold=threshold,
        tool=outcome.tool,
    )
    details = dict(prior_report.details)
    details["mutation_final"] = {
        "no_synthesis": True,
        "suite_fingerprint": evidence.suite_fingerprint,
        "replacement_counts": {
            "total": outcome.mutants_total,
            "killed": outcome.mutants_killed,
            "kill_ratio": round(outcome.kill_ratio, 4),
            "tool": outcome.tool,
        },
        "threshold": threshold,
        "measurement": outcome.details,
    }
    details.pop("mutation_evidence_invalidated", None)
    if env_image is not None:
        details.setdefault("env_image", env_image.image_tag)

    base_prov = prior_report.provenance
    provenance = Provenance(
        generator=candidate.generator,
        seed=candidate.provenance.seed,
        language=candidate.language,
        tool_versions={
            **(base_prov.tool_versions if base_prov else _tool_versions()),
            **({outcome.tool: ""} if outcome.tool else {}),
        },
        details={
            "stage": "oracle.mutation.final",
            "final_mutation_suite_fingerprint": evidence.suite_fingerprint,
            "mutants_total": outcome.mutants_total,
            "mutants_killed": outcome.mutants_killed,
            "threshold": threshold,
            "mutation_tool": outcome.tool,
            "no_synthesis": True,
        },
    )
    return OracleReport(
        language=prior_report.language,
        generator=prior_report.generator,
        verdict=outcome.verdict,
        reasons=list(outcome.reasons),
        fail_to_pass=list(prior_report.fail_to_pass),
        pass_to_pass=list(prior_report.pass_to_pass),
        test_files=list(prior_report.test_files),
        flakiness_runs=prior_report.flakiness_runs,
        mutants_total=outcome.mutants_total,
        mutants_killed=outcome.mutants_killed,
        final_mutation_evidence=evidence,
        differential_pass=prior_report.differential_pass,
        alt_correct_accepted=prior_report.alt_correct_accepted,
        leak_audit=prior_report.leak_audit,
        provenance=provenance,
        details=details,
        protected_alt_correct_audit=prior_report.protected_alt_correct_audit,
    )


# --------------------------------------------------------------------------- #
# Docker-backed runner + top-level gate
# --------------------------------------------------------------------------- #
def reconstruct_base_tests(
    test_files: Sequence[OracleTestFile],
) -> list[HiddenTest]:
    """Rebuild :class:`HiddenTest` objects (file bodies only) from report fields.

    The mutation tool collects every test file present in the workspace, so the
    base suite only needs the established hidden test *bodies* written back in;
    the ``test_id`` is informational here (the tool runs the whole suite).
    """
    tests: list[HiddenTest] = []
    for tf in test_files:
        if not tf.content:
            continue
        tests.append(
            HiddenTest(
                test_id=tf.path,
                files=(HiddenTestFile(path=tf.path, content=tf.content),),
                origin=tf.origin,
            )
        )
    return tests


class DockerMutationRunner:
    """A :class:`MutationRunner` that scores via throwaway Docker sandboxes.

    Each :meth:`measure` opens a fresh ``--rm`` :class:`DockerSandbox` on the
    candidate's green ``EnvImage`` (the repo already checked out = gold), writes
    the established hidden tests plus the round's extra tests, and runs the
    adapter's mutation tool against the gold target file(s).
    """

    def __init__(
        self,
        candidate: Candidate,
        env_image: EnvImage,
        adapter: LanguageAdapter,
        *,
        base_tests: Sequence[HiddenTest] = (),
        command_timeout: float = 1200.0,
        docker_client: "DockerClient | None" = None,
    ) -> None:
        self._candidate = candidate
        self._env_image = env_image
        self._adapter = adapter
        self._base_tests = list(base_tests)
        self._timeout = command_timeout
        self._docker_client = docker_client
        # Difficulty-amplifier candidates mutate >=2 symbols on large modular
        # modules; scope cosmic-ray to the changed regions so the run finishes
        # within the timeout (other generators mutate whole files, unchanged).
        self._regions: dict[str, tuple[tuple[int, int], ...]] | None = (
            candidate_mutation_regions(candidate)
            if candidate.generator in AMPLIFIER_GENERATORS
            else None
        )

    @property
    def language(self) -> str:
        return self._candidate.language

    @property
    def _target_files(self) -> list[str]:
        return [
            f for f in self._candidate.target.files if not self._adapter.is_test_file(f)
        ]

    @contextlib.asynccontextmanager
    async def _sandbox(self) -> "AsyncIterator[MutationExecutor]":
        from swe_forge.execution.docker_client import DockerClient
        from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

        client = self._docker_client or DockerClient()
        config = SandboxConfig(
            name="swe-forge-oracle-mutation",
            image=self._env_image.image_tag,
            workspace_dir=self._env_image.workspace_dir,
            command_timeout=self._timeout,
        )
        sandbox = DockerSandbox(client, config)
        async with sandbox:
            yield sandbox

    async def measure(self, extra_tests: Sequence[HiddenTest]) -> MutationMeasurement:
        async with self._sandbox() as sandbox:
            for test in [*self._base_tests, *extra_tests]:
                for file in test.files:
                    await sandbox.write_file(file.path, file.content)
            stats = await self._adapter.mutation_tool_run(
                sandbox,
                target_files=self._target_files,
                timeout=self._timeout,
                target_regions=self._regions,
            )
        return MutationMeasurement.from_stats(stats)

    async def read_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        async with self._sandbox() as sandbox:
            for path in self._target_files:
                with contextlib.suppress(Exception):
                    sources[path] = await sandbox.read_file(path)
        return sources


async def run_mutation_gate(
    candidate: Candidate,
    env_image: EnvImage,
    prior_report: OracleReport,
    *,
    synthesizer: MutationTestSynthesizer | None = None,
    threshold: float = DEFAULT_KILL_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_SYNTHESIS_ROUNDS,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 1200.0,
) -> OracleReport:
    """Run the mutation-adequacy gate in throwaway Docker sandboxes on ``env_image``.

    A green baseline is a hard precondition and the prior gate (flakiness) must
    have passed. Builds a :class:`DockerMutationRunner`, measures adequacy with
    bounded auto-synthesis, and returns the extended :class:`OracleReport`.
    """
    require_green_baseline(env_image)
    if prior_report.verdict != "pass":
        raise MutationError(
            "mutation gate requires a passing prior (flakiness) report; got verdict "
            f"{prior_report.verdict!r}"
        )

    if adapter is None:
        adapter = build_default_registry().get(candidate.language)

    base_tests = reconstruct_base_tests(prior_report.test_files)
    runner = DockerMutationRunner(
        candidate,
        env_image,
        adapter,
        base_tests=base_tests,
        command_timeout=command_timeout,
        docker_client=docker_client,
    )
    template = MutationSynthesisContext(
        candidate=candidate,
        adapter=adapter,
        sources={},
        survivors=(),
        round_index=0,
    )
    outcome = await assess_mutation(
        runner,
        synthesizer=synthesizer,
        context_template=template,
        threshold=threshold,
        max_rounds=max_rounds,
    )
    return build_mutation_report(candidate, prior_report, outcome, env_image=env_image)


async def run_final_mutation_gate(
    candidate: Candidate,
    env_image: EnvImage,
    prior_report: OracleReport,
    *,
    threshold: float = DEFAULT_KILL_THRESHOLD,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 1200.0,
) -> OracleReport:
    """Re-measure adequacy against the final suite without synthesizing tests."""
    require_green_baseline(env_image)
    if prior_report.verdict != "pass":
        raise MutationError(
            "final mutation remeasurement requires a passing prior report; got "
            f"verdict {prior_report.verdict!r}"
        )

    if adapter is None:
        adapter = build_default_registry().get(candidate.language)
    runner = DockerMutationRunner(
        candidate,
        env_image,
        adapter,
        base_tests=reconstruct_base_tests(prior_report.test_files),
        command_timeout=command_timeout,
        docker_client=docker_client,
    )
    outcome = await assess_mutation(
        runner,
        synthesizer=None,
        threshold=threshold,
        max_rounds=0,
    )
    return build_final_mutation_report(
        candidate,
        prior_report,
        outcome,
        threshold=threshold,
        env_image=env_image,
    )
