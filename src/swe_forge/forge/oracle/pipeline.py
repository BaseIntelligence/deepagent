"""Oracle pipeline: orchestrate the hardening gates into one OracleReport.

The final Stage 3 step (architecture S6, Stage 3 "pipeline"). The individual
gates - establish, flakiness, mutation, differential, alt-correct, multifault,
 leak - each
extend a running :class:`~swe_forge.forge.models.OracleReport`; this module runs
them **in that fixed order** and decides the terminal verdict:

* ``verdict == "pass"`` iff EVERY gate passed, leaving ``reasons == []`` and all
  gate fields mutually consistent (``fail_to_pass`` non-empty, ``flakiness_runs
  >= 3``, mutant kill ratio ``>= threshold``, ``differential_pass``,
  ``alt_correct_accepted``, and a clean/sanitized ``leak_audit``);
* a single gate failure forces ``verdict == "reject"`` with a non-empty,
  attributable ``reasons`` list - and the pipeline STOPS at the first failure, so
  the later gates are never credited (their fields stay at the dataclass defaults
  ``differential_pass == False`` / ``alt_correct_accepted == False`` /
  ``leak_audit == ""`` rather than spuriously ``True``).

A rejected candidate must never be exported. The export layer (a later stage)
calls :func:`ensure_oracle_exportable`, which encodes the architecture invariant
"a ForgeTask may only be created if ``OracleReport.verdict == pass`` AND
``CalibrationReport.band_verdict == keep``": an oracle pass is *necessary* but,
once calibration exists, not *sufficient*.

Every gate runs in a throwaway :class:`~swe_forge.execution.sandbox.DockerSandbox`
on the candidate's :class:`~swe_forge.forge.models.EnvImage`; the gate runners own
the container hygiene (``--rm``, unique names, teardown by id even on failure).
The pipeline is language-agnostic: it threads the same :class:`LanguageAdapter`
through every gate and never branches on language itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    GeneratedSpec,
    OracleReport,
    require_green_baseline,
)
from swe_forge.forge.oracle.alt_correct import (
    DEFAULT_NUM_ALTERNATIVES,
    AltCorrectGenerator,
    run_alt_correct_gate,
)
from swe_forge.forge.oracle.differential import (
    DEFAULT_MAX_STRENGTHEN_ROUNDS,
    DEFAULT_NUM_VARIANTS,
    VariantGenerator,
    VariantStrengthSynthesizer,
    run_differential_gate,
)
from swe_forge.forge.oracle.establish import (
    HiddenTest,
    HiddenTestSynthesizer,
    run_establish_gate,
)
from swe_forge.forge.oracle.flakiness import (
    DEFAULT_FLAKINESS_RUNS,
    MIN_FLAKINESS_RUNS,
    run_flakiness_gate,
)
from swe_forge.forge.oracle.leak import run_leak_gate
from swe_forge.forge.oracle.mutation import (
    DEFAULT_KILL_THRESHOLD,
    DEFAULT_MAX_SYNTHESIS_ROUNDS,
    MutationTestSynthesizer,
    final_suite_fingerprint,
    run_final_mutation_gate,
    run_mutation_gate,
)
from swe_forge.forge.oracle.multifault import (
    run_multifault_completeness_gate,
    verify_multifault_evidence,
)
from swe_forge.forge.oracle.teacher_evidence import teacher_gate_evidence_issues

if TYPE_CHECKING:
    from swe_forge.execution.docker_client import DockerClient

#: The mandated gate order. Every shipped task is hardened in exactly this
#: sequence; the pipeline stops at the first non-pass so later gates are never
#: credited on an earlier failure.
GATE_ORDER: tuple[str, ...] = (
    "establish",
    "flakiness",
    "mutation",
    "differential",
    "alt_correct",
    "multifault",
    "leak",
)

#: Reason prefix used when every gate reported pass but the folded report's gate
#: fields are not mutually consistent (a defensive guard - should never fire in a
#: correct gate, but the pipeline refuses to emit an inconsistent ``pass``).
REASON_PIPELINE_INCONSISTENT = "pipeline_inconsistent_pass"

#: One bound gate: consumes the prior report (``None`` for the first gate) and
#: returns the extended report. The orchestration threads these uniformly.
GateStep = Callable[["OracleReport | None"], Awaitable[OracleReport]]


class OraclePipelineError(RuntimeError):
    """Raised for an unrecoverable failure while orchestrating the gates."""


class ExportRefusedError(RuntimeError):
    """Raised by :func:`ensure_oracle_exportable` for a non-shippable candidate."""


def _alt_correct_public_validity_issues(report: OracleReport) -> list[str]:
    """Validate the non-agent-facing evidence for alt-correct public validity."""
    # Artifacts emitted before public-validity hardening lack the entire
    # alt-correct record. They can remain immutable inputs to fresh
    # recertification, which executes the hardened gate and records the audit.
    # Any report produced by the hardened gate has this record and therefore
    # cannot bypass the checks below.
    if "alt_correct" not in report.details:
        return []

    audit = report.protected_alt_correct_audit
    if not isinstance(audit, dict):
        return ["alt_correct: protected public-validity audit is missing"]
    gold = audit.get("gold")
    if not isinstance(gold, dict):
        return ["alt_correct: protected gold public result is missing"]
    gold_public = gold.get("public")
    if not isinstance(gold_public, dict) or gold_public.get("passed") is not True:
        return ["alt_correct: gold did not pass the original public suite"]
    alternatives = audit.get("alternatives")
    if not isinstance(alternatives, dict) or not alternatives:
        return ["alt_correct: protected alternative audit is missing"]

    public_green = 0
    for alt_id, record in alternatives.items():
        if not isinstance(alt_id, str) or not isinstance(record, dict):
            return ["alt_correct: protected alternative audit is malformed"]
        proposal_digest = record.get("proposal_sha256")
        patches = record.get("patches")
        public = record.get("public")
        if (
            not isinstance(proposal_digest, str)
            or len(proposal_digest) != 64
            or not isinstance(patches, list)
            or not patches
            or not isinstance(public, dict)
        ):
            return ["alt_correct: protected alternative audit is incomplete"]
        if public.get("passed") is True:
            public_green += 1
            hidden = record.get("hidden")
            if not isinstance(hidden, list) or not hidden:
                return [
                    "alt_correct: public-green alternative has no hidden per-node "
                    "execution evidence"
                ]
    if public_green < 1:
        return [
            "alt_correct: no executable real-teacher alternative passed the "
            "original public suite"
        ]

    public_details = report.details.get("alt_correct")
    if not isinstance(public_details, dict):
        return ["alt_correct: public validity summary is missing"]
    if not isinstance(public_details.get("public_suite_sha256"), str):
        return ["alt_correct: public suite digest is missing"]
    if public_details.get("gold_public_suite_passed") is not True:
        return ["alt_correct: public gold summary is not green"]
    if public_details.get("public_valid_alternatives") != public_green:
        return ["alt_correct: public-valid alternative count does not match audit"]
    return []


def verify_pass_consistency(
    report: OracleReport, *, kill_threshold: float = DEFAULT_KILL_THRESHOLD
) -> list[str]:
    """Return the ways a ``pass`` report's gate fields are mutually inconsistent.

    Encodes VAL-ORACLE-016: a genuine ``pass`` must have an empty ``reasons`` list
    and every gate's evidence field set consistently. Returns ``[]`` for a
    consistent pass (or for any non-pass verdict, which this function does not
    police). A non-empty result means a gate reported pass without leaving sound
    evidence and the pipeline must not ship the candidate.
    """
    if report.verdict != "pass":
        return []

    problems: list[str] = []
    if report.reasons:
        problems.append(f"a pass verdict carries reasons {report.reasons!r}")
    if not report.fail_to_pass:
        problems.append("establish: fail_to_pass is empty")
    if report.flakiness_runs < MIN_FLAKINESS_RUNS:
        problems.append(
            f"flakiness: flakiness_runs {report.flakiness_runs} < {MIN_FLAKINESS_RUNS}"
        )
    if report.mutants_total <= 0:
        problems.append("mutation: mutants_total is 0")
    elif report.mutants_killed / report.mutants_total < kill_threshold:
        ratio = report.mutants_killed / report.mutants_total
        problems.append(
            f"mutation: kill ratio {ratio:.2f} < threshold {kill_threshold:.2f}"
        )
    evidence = report.final_mutation_evidence
    if evidence is None:
        problems.append("mutation: final mutation evidence is missing")
    else:
        try:
            expected_fingerprint = final_suite_fingerprint(report.test_files)
        except Exception as exc:  # noqa: BLE001 - fail closed on malformed suites
            problems.append(
                "mutation: final mutation evidence cannot fingerprint final "
                f"hidden suite ({type(exc).__name__}: {exc})"
            )
        else:
            if evidence.suite_fingerprint != expected_fingerprint:
                problems.append(
                    "mutation: final mutation evidence suite fingerprint does not "
                    "match final hidden tests"
                )
        if (evidence.mutants_total, evidence.mutants_killed) != (
            report.mutants_total,
            report.mutants_killed,
        ):
            problems.append(
                "mutation: final mutation evidence replacement counts do not "
                "match OracleReport mutants_total/mutants_killed"
            )
        if evidence.threshold != kill_threshold:
            problems.append(
                "mutation: final mutation evidence threshold "
                f"{evidence.threshold:.2f} != configured threshold {kill_threshold:.2f}"
            )
        elif evidence.kill_ratio < evidence.threshold:
            problems.append(
                "mutation: final mutation evidence kill ratio "
                f"{evidence.kill_ratio:.2f} < threshold {evidence.threshold:.2f}"
            )
    problems.extend(verify_multifault_evidence(report))
    if not report.differential_pass:
        problems.append("differential: differential_pass is false")
    if not report.alt_correct_accepted:
        problems.append("alt_correct: alt_correct_accepted is false")
    problems.extend(_alt_correct_public_validity_issues(report))
    problems.extend(teacher_gate_evidence_issues(report.details))
    if not (
        report.leak_audit.startswith("clean")
        or report.leak_audit.startswith("sanitized")
    ):
        problems.append(f"leak: leak_audit {report.leak_audit!r} is not clean")
    return problems


def _reject_for_inconsistency(
    report: OracleReport, problems: Sequence[str]
) -> OracleReport:
    """Re-cast a (spuriously) pass report as a reject citing each inconsistency."""
    data = report.to_dict()
    data["verdict"] = "reject"
    data["reasons"] = [f"{REASON_PIPELINE_INCONSISTENT}: {p}" for p in problems]
    return OracleReport.from_dict(data)


async def orchestrate_gates(
    gates: Sequence[tuple[str, GateStep]],
    *,
    kill_threshold: float = DEFAULT_KILL_THRESHOLD,
) -> OracleReport:
    """Run ``gates`` in order, stopping at the first that does not pass.

    Threads each gate's report into the next; the first non-pass report is the
    pipeline's result (so later gates never run and their fields are never
    credited). When every gate passes, the folded report is checked for field
    consistency (:func:`verify_pass_consistency`) and demoted to ``reject`` if a
    gate passed without sound evidence. The result records a ``pipeline`` summary
    in ``details`` (configured order, gates actually run, earliest failed gate).
    """
    if not gates:
        raise OraclePipelineError("the oracle pipeline requires at least one gate")

    report: OracleReport | None = None
    gates_run: list[str] = []
    failed_gate: str | None = None

    for name, step in gates:
        report = await step(report)
        gates_run.append(name)
        if report.verdict != "pass":
            failed_gate = name
            break

    if report is None:  # pragma: no cover - guarded by the empty-gates check above
        raise OraclePipelineError("no gate produced a report")

    if failed_gate is None:
        problems = verify_pass_consistency(report, kill_threshold=kill_threshold)
        if problems:
            report = _reject_for_inconsistency(report, problems)
            failed_gate = "consistency"

    report.details["pipeline"] = {
        "gate_order": [name for name, _ in gates],
        "gates_run": gates_run,
        "failed_gate": failed_gate,
        "verdict": report.verdict,
    }
    return report


def build_default_gates(
    candidate: Candidate,
    env_image: EnvImage,
    *,
    provided_tests: Sequence[HiddenTest] = (),
    establish_synthesizer: HiddenTestSynthesizer | None = None,
    mutation_synthesizer: MutationTestSynthesizer | None = None,
    variant_generator: VariantGenerator | None = None,
    differential_synthesizer: VariantStrengthSynthesizer | None = None,
    alt_generator: AltCorrectGenerator | None = None,
    spec: GeneratedSpec | None = None,
    flakiness_runs: int = DEFAULT_FLAKINESS_RUNS,
    kill_threshold: float = DEFAULT_KILL_THRESHOLD,
    max_mutation_rounds: int = DEFAULT_MAX_SYNTHESIS_ROUNDS,
    num_variants: int = DEFAULT_NUM_VARIANTS,
    max_differential_rounds: int = DEFAULT_MAX_STRENGTHEN_ROUNDS,
    num_alternatives: int = DEFAULT_NUM_ALTERNATIVES,
    relax_alt_correct: bool = False,
    sanitize: bool = True,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 600.0,
    mutation_timeout: float = 1200.0,
) -> list[tuple[str, GateStep]]:
    """Build the ordered, Docker-backed gate steps for one candidate.

    Each step closes over the candidate/env image and its gate-specific knobs and
    exposes the uniform :data:`GateStep` signature. The synthesizers/generators
    are injected (the LLM-backed defaults are wired by the CLI) so the pipeline
    module stays free of LLM coupling and trivially unit-testable with fakes.
    """

    async def establish(_prior: OracleReport | None) -> OracleReport:
        return await run_establish_gate(
            candidate,
            env_image,
            provided_tests=provided_tests,
            synthesizer=establish_synthesizer,
            adapter=adapter,
            docker_client=docker_client,
            command_timeout=command_timeout,
        )

    async def flakiness(prior: OracleReport | None) -> OracleReport:
        assert prior is not None
        return await run_flakiness_gate(
            candidate,
            env_image,
            prior,
            runs=flakiness_runs,
            adapter=adapter,
            docker_client=docker_client,
            command_timeout=command_timeout,
        )

    async def mutation(prior: OracleReport | None) -> OracleReport:
        assert prior is not None
        return await run_mutation_gate(
            candidate,
            env_image,
            prior,
            synthesizer=mutation_synthesizer,
            threshold=kill_threshold,
            max_rounds=max_mutation_rounds,
            adapter=adapter,
            docker_client=docker_client,
            command_timeout=mutation_timeout,
        )

    async def differential(prior: OracleReport | None) -> OracleReport:
        assert prior is not None
        return await run_differential_gate(
            candidate,
            env_image,
            prior,
            variant_generator=variant_generator,
            synthesizer=differential_synthesizer,
            num_variants=num_variants,
            max_rounds=max_differential_rounds,
            adapter=adapter,
            docker_client=docker_client,
            command_timeout=command_timeout,
        )

    async def alt_correct(prior: OracleReport | None) -> OracleReport:
        assert prior is not None
        return await run_alt_correct_gate(
            candidate,
            env_image,
            prior,
            spec=spec,
            alt_generator=alt_generator,
            num_alternatives=num_alternatives,
            relax=relax_alt_correct,
            adapter=adapter,
            docker_client=docker_client,
            command_timeout=command_timeout,
        )

    async def multifault(prior: OracleReport | None) -> OracleReport:
        assert prior is not None
        # Differential can strengthen or prune tests, and alt-correct can relax
        # them. Re-measure the finalized suite before proving every multi-fault
        # constituent is independently required, never synthesizing new tests
        # after this point.
        final_mutation = await run_final_mutation_gate(
            candidate,
            env_image,
            prior,
            threshold=kill_threshold,
            adapter=adapter,
            docker_client=docker_client,
            command_timeout=mutation_timeout,
        )
        if final_mutation.verdict != "pass":
            return final_mutation
        return await run_multifault_completeness_gate(
            candidate,
            env_image,
            final_mutation,
            adapter=adapter,
            docker_client=docker_client,
            command_timeout=command_timeout,
        )

    async def leak(prior: OracleReport | None) -> OracleReport:
        assert prior is not None
        return await run_leak_gate(
            candidate,
            env_image,
            prior,
            adapter=adapter,
            sanitize=sanitize,
            docker_client=docker_client,
            command_timeout=command_timeout,
        )

    return [
        ("establish", establish),
        ("flakiness", flakiness),
        ("mutation", mutation),
        ("differential", differential),
        ("alt_correct", alt_correct),
        ("multifault", multifault),
        ("leak", leak),
    ]


async def run_oracle_pipeline(
    candidate: Candidate,
    env_image: EnvImage,
    *,
    provided_tests: Sequence[HiddenTest] = (),
    establish_synthesizer: HiddenTestSynthesizer | None = None,
    mutation_synthesizer: MutationTestSynthesizer | None = None,
    variant_generator: VariantGenerator | None = None,
    differential_synthesizer: VariantStrengthSynthesizer | None = None,
    alt_generator: AltCorrectGenerator | None = None,
    spec: GeneratedSpec | None = None,
    flakiness_runs: int = DEFAULT_FLAKINESS_RUNS,
    kill_threshold: float = DEFAULT_KILL_THRESHOLD,
    max_mutation_rounds: int = DEFAULT_MAX_SYNTHESIS_ROUNDS,
    num_variants: int = DEFAULT_NUM_VARIANTS,
    max_differential_rounds: int = DEFAULT_MAX_STRENGTHEN_ROUNDS,
    num_alternatives: int = DEFAULT_NUM_ALTERNATIVES,
    relax_alt_correct: bool = False,
    sanitize: bool = True,
    adapter: LanguageAdapter | None = None,
    docker_client: "DockerClient | None" = None,
    command_timeout: float = 600.0,
    mutation_timeout: float = 1200.0,
    gates: Sequence[tuple[str, GateStep]] | None = None,
) -> OracleReport:
    """Run the full oracle pipeline on a candidate and return its OracleReport.

    A green baseline is a hard precondition (:func:`require_green_baseline`). The
    gates run in :data:`GATE_ORDER` on the candidate's EnvImage in throwaway
    Docker sandboxes; the verdict is ``pass`` only when every gate passes with
    consistent fields, else ``reject`` with attributable reasons citing the
    earliest failed gate. ``gates`` may be supplied to inject a custom (e.g.
    test) gate sequence; otherwise the default Docker-backed gates are built.
    """
    require_green_baseline(env_image)

    if adapter is None:
        adapter = build_default_registry().get(candidate.language)

    if gates is None:
        gates = build_default_gates(
            candidate,
            env_image,
            provided_tests=provided_tests,
            establish_synthesizer=establish_synthesizer,
            mutation_synthesizer=mutation_synthesizer,
            variant_generator=variant_generator,
            differential_synthesizer=differential_synthesizer,
            alt_generator=alt_generator,
            spec=spec,
            flakiness_runs=flakiness_runs,
            kill_threshold=kill_threshold,
            max_mutation_rounds=max_mutation_rounds,
            num_variants=num_variants,
            max_differential_rounds=max_differential_rounds,
            num_alternatives=num_alternatives,
            relax_alt_correct=relax_alt_correct,
            sanitize=sanitize,
            adapter=adapter,
            docker_client=docker_client,
            command_timeout=command_timeout,
            mutation_timeout=mutation_timeout,
        )

    return await orchestrate_gates(gates, kill_threshold=kill_threshold)


def is_oracle_exportable(
    report: OracleReport, *, candidate: Candidate | None = None
) -> bool:
    """``True`` iff the oracle verdict permits export (a necessary condition).

    An oracle pass is *necessary* for export but not *sufficient*: calibration
    must also keep the candidate. Use :func:`ensure_oracle_exportable` to enforce
    both at the export boundary.
    """
    threshold = (
        report.final_mutation_evidence.threshold
        if report.final_mutation_evidence is not None
        else DEFAULT_KILL_THRESHOLD
    )
    return (
        report.verdict == "pass"
        and not verify_pass_consistency(report, kill_threshold=threshold)
        and not verify_multifault_evidence(report, candidate=candidate)
    )


def ensure_oracle_exportable(
    report: OracleReport,
    *,
    candidate: Candidate | None = None,
    calibration_kept: bool | None = None,
    kill_threshold: float | None = None,
) -> None:
    """Raise :class:`ExportRefusedError` unless the candidate may be exported.

    Encodes the architecture export invariant: a rejected candidate is NEVER
    exported (oracle pass is necessary), and once calibration has run an oracle
    pass is only exportable together with a calibration ``keep``
    (``calibration_kept`` is ``True``). ``calibration_kept=None`` checks only the
    necessary oracle condition (calibration not yet available).
    """
    if report.verdict != "pass":
        raise ExportRefusedError(
            f"export refused: oracle verdict is {report.verdict!r} "
            f"(reasons={list(report.reasons)}); a rejected candidate is never exported"
        )
    final_threshold = kill_threshold
    if final_threshold is None:
        final_threshold = (
            report.final_mutation_evidence.threshold
            if report.final_mutation_evidence is not None
            else DEFAULT_KILL_THRESHOLD
        )
    consistency_problems = verify_pass_consistency(
        report, kill_threshold=final_threshold
    )
    consistency_problems.extend(verify_multifault_evidence(report, candidate=candidate))
    if consistency_problems:
        raise ExportRefusedError(
            "export refused: final mutation evidence or oracle gate consistency "
            "is invalid (" + "; ".join(consistency_problems) + ")"
        )
    if calibration_kept is False:
        raise ExportRefusedError(
            "export refused: oracle passed but calibration band_verdict is 'drop'; "
            "an oracle pass is necessary but calibration keep is also required"
        )
