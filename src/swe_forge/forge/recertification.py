"""Fail-closed recertification of the single certified multi-fault recovery.

This module deliberately accepts one immutable recovery publication only. It
first re-runs the normal, live teacher differential and alternative-correct
gates, then re-runs final mutation, constituent-completeness, and leak gates.
Calibration reuse is permitted only when the final hidden-suite fingerprint is
byte-for-byte unchanged. Otherwise a caller must perform fresh panel
calibration rather than accidentally publishing stale difficulty evidence.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from swe_forge.forge.export import BatchExportResult, ExportRequest, export_batch
from swe_forge.forge.models import CalibrationReport, ForgeTask, OracleReport
from swe_forge.forge.oracle.alt_correct import (
    AltCorrectGenerator,
    run_alt_correct_gate,
)
from swe_forge.forge.oracle.alt_correct_synth import TeacherAltCorrectGenerator
from swe_forge.forge.oracle.differential import (
    VariantGenerator,
    VariantStrengthSynthesizer,
    run_differential_gate,
)
from swe_forge.forge.oracle.differential_synth import (
    DifferentialKillSynthesizer,
    TeacherVariantGenerator,
)
from swe_forge.forge.oracle.leak import run_leak_gate
from swe_forge.forge.oracle.mutation import (
    final_suite_fingerprint,
    run_final_mutation_gate,
)
from swe_forge.forge.oracle.multifault import (
    run_multifault_completeness_gate,
    verify_multifault_evidence,
)
from swe_forge.forge.oracle.pipeline import (
    ensure_oracle_exportable,
    verify_pass_consistency,
)
from swe_forge.forge.publication import (
    PublishedGeneration,
    load_published_generation,
)

if TYPE_CHECKING:
    from swe_forge.execution.docker_client import DockerClient
    from swe_forge.forge.adapters import LanguageAdapter

# The recovery feature creates this exact replacement.  It is intentionally not
# inferred from a stale pilot directory, which prevents accidental restoration of
# the five invalidated legacy tasks.
CERTIFIED_RECOVERY_TASK_ID = "mahmoud-boltons__bug_combination__242d17b94d94"
CERTIFIED_RECOVERY_SOURCE = Path("results/multifault_recovery_certified")


class RecertificationError(RuntimeError):
    """Raised when the certified recovery cannot be safely republished."""


@dataclass(frozen=True)
class RecertificationResult:
    """The recertified report plus its one transactional export result."""

    source_generation_id: str
    task_id: str
    oracle_report: OracleReport
    export: BatchExportResult


Recalibrator = Callable[[ForgeTask, OracleReport], Awaitable[CalibrationReport]]


def _final_fingerprint(report: OracleReport) -> str:
    """Return trusted final-suite evidence or reject before any export."""
    evidence = report.final_mutation_evidence
    if evidence is None:
        raise RecertificationError("final mutation evidence is missing")
    actual = final_suite_fingerprint(report.test_files)
    if evidence.suite_fingerprint != actual:
        raise RecertificationError(
            "final hidden-suite fingerprint does not match mutation evidence"
        )
    return actual


def _final_evidence(report: OracleReport):
    """Return final mutation evidence after making its absence a hard failure."""
    evidence = report.final_mutation_evidence
    if evidence is None:
        raise RecertificationError("final mutation evidence is missing")
    return evidence


def _validate_recovery_identity(task: ForgeTask) -> None:
    """Require the immutable recovery identity and retained calibration."""
    if task.task_id != CERTIFIED_RECOVERY_TASK_ID:
        raise RecertificationError(
            "recovery publication does not contain the approved genuine task "
            f"{CERTIFIED_RECOVERY_TASK_ID!r}"
        )
    if task.generator != "bug_combination":
        raise RecertificationError("certified recovery must remain a bug_combination")
    if task.oracle_report.verdict != "pass":
        raise RecertificationError("certified recovery requires an oracle pass")
    if not task.calibration_report.is_keep:
        raise RecertificationError("certified recovery requires a calibration keep")
    _final_fingerprint(task.oracle_report)


def validate_certified_recovery_source(task: ForgeTask) -> None:
    """Validate a genuine old recovery before fresh teacher-gate evidence exists.

    The recovery was produced before the non-vacuous teacher-proof schema. Its
    old report may therefore lack that newly-required evidence, but its identity,
    calibration, final mutation evidence, and constituent proof must still be
    sound before it can become input to a new live recertification.
    """
    _validate_recovery_identity(task)
    if not task.oracle_report.differential_pass:
        raise RecertificationError(
            "certified recovery requires a prior differential oracle pass"
        )
    if not task.oracle_report.alt_correct_accepted:
        raise RecertificationError(
            "certified recovery requires a prior alternative-correct oracle pass"
        )
    problems = [
        *verify_multifault_evidence(task.oracle_report, candidate=task.candidate),
    ]
    if problems:
        raise RecertificationError(
            "certified recovery oracle evidence is inconsistent: "
            + "; ".join(dict.fromkeys(problems))
        )


def validate_certified_recovery_task(task: ForgeTask) -> None:
    """Require the genuine recovery task and every current export invariant."""
    validate_certified_recovery_source(task)
    threshold = _final_evidence(task.oracle_report).threshold
    problems = verify_pass_consistency(task.oracle_report, kill_threshold=threshold)
    if problems:
        raise RecertificationError(
            "certified recovery oracle evidence is inconsistent: "
            + "; ".join(dict.fromkeys(problems))
        )
    try:
        ensure_oracle_exportable(
            task.oracle_report,
            candidate=task.candidate,
            calibration_kept=task.calibration_report.is_keep,
            kill_threshold=threshold,
        )
    except Exception as exc:  # ExportRefusedError is intentionally hidden here.
        raise RecertificationError(
            f"certified recovery is not exportable: {exc}"
        ) from exc


def load_certified_recovery(
    source_dir: Path | str = CERTIFIED_RECOVERY_SOURCE,
) -> tuple[PublishedGeneration, ForgeTask]:
    """Load only the recovery publication's current immutable generation."""
    generation = load_published_generation(source_dir)
    if generation is None:
        raise RecertificationError(
            f"no published certified recovery generation under {source_dir}"
        )
    if len(generation.entries) != 1:
        raise RecertificationError(
            "certified recovery must contain exactly one approved task, found "
            f"{len(generation.entries)}"
        )
    task = generation.entries[0].task
    validate_certified_recovery_source(task)
    return generation, task


def require_unchanged_suite_for_calibration(
    original: OracleReport, recertified: OracleReport
) -> None:
    """Reject calibration reuse whenever a recertification changed hidden tests."""
    original_fingerprint = _final_fingerprint(original)
    recertified_fingerprint = _final_fingerprint(recertified)
    if original_fingerprint != recertified_fingerprint:
        raise RecertificationError(
            "final hidden suite changed during recertification; recalibration is "
            "required and prior calibration evidence cannot be reused"
        )


async def recertify_final_oracle(
    task: ForgeTask,
    *,
    command_timeout: float = 600.0,
    mutation_timeout: float = 1200.0,
    variant_generator: VariantGenerator | None = None,
    differential_synthesizer: VariantStrengthSynthesizer | None = None,
    alt_generator: AltCorrectGenerator | None = None,
    adapter: LanguageAdapter | None = None,
    docker_client: DockerClient | None = None,
) -> OracleReport:
    """Re-run every suite-dependent final gate against the recovered task.

    The original recovery predates non-vacuous teacher evidence. The ordinary
    live differential and alternative-correct gates must therefore execute again
    before any final-suite evidence is retained. The function then remeasures
    mutation adequacy, proves every constituent is necessary, and repeats the
    agent-tree leak audit. Every retained evidence field thus belongs to the
    exact recertified export suite.
    """
    validate_certified_recovery_source(task)
    threshold = _final_evidence(task.oracle_report).threshold
    differential = await run_differential_gate(
        task.candidate,
        task.env_image,
        task.oracle_report,
        variant_generator=variant_generator or TeacherVariantGenerator(),
        synthesizer=differential_synthesizer or DifferentialKillSynthesizer(),
        adapter=adapter,
        docker_client=docker_client,
        command_timeout=command_timeout,
    )
    if not differential.is_pass:
        return differential
    alt_correct = await run_alt_correct_gate(
        task.candidate,
        task.env_image,
        differential,
        spec=task.spec,
        alt_generator=alt_generator or TeacherAltCorrectGenerator(),
        adapter=adapter,
        docker_client=docker_client,
        command_timeout=command_timeout,
    )
    if not alt_correct.is_pass:
        return alt_correct
    final_mutation = await run_final_mutation_gate(
        task.candidate,
        task.env_image,
        alt_correct,
        threshold=threshold,
        adapter=adapter,
        docker_client=docker_client,
        command_timeout=mutation_timeout,
    )
    if not final_mutation.is_pass:
        return final_mutation
    multifault = await run_multifault_completeness_gate(
        task.candidate,
        task.env_image,
        final_mutation,
        adapter=adapter,
        docker_client=docker_client,
        command_timeout=command_timeout,
    )
    if not multifault.is_pass:
        return multifault
    recertified = await run_leak_gate(
        task.candidate,
        task.env_image,
        multifault,
        adapter=adapter,
        docker_client=docker_client,
        command_timeout=command_timeout,
    )
    if recertified.is_pass:
        problems = [
            *verify_pass_consistency(recertified, kill_threshold=threshold),
            *verify_multifault_evidence(recertified, candidate=task.candidate),
        ]
        if problems:
            raise RecertificationError(
                "recertified oracle evidence is inconsistent: "
                + "; ".join(dict.fromkeys(problems))
            )
    return recertified


def build_recertification_request(
    generation: PublishedGeneration,
    oracle_report: OracleReport,
    calibration_report: CalibrationReport | None = None,
) -> ExportRequest:
    """Construct the only permissible transactional export request.

    The calibration report is reused only after the caller has shown the
    recertified suite is identical. A freshly recalibrated keep may be supplied
    if the normal teacher gates changed the hidden-suite fingerprint. The
    deterministic task id is retained, so JSONL, Parquet, and workspace
    reconciliation all target the same singleton.
    """
    if len(generation.entries) != 1:
        raise RecertificationError("cannot re-export a non-singleton recovery set")
    source = generation.entries[0].task
    validate_certified_recovery_source(source)
    if oracle_report.verdict != "pass":
        raise RecertificationError(
            "recertified oracle did not pass: " + "; ".join(oracle_report.reasons)
        )
    calibration = calibration_report or source.calibration_report
    if calibration_report is None:
        require_unchanged_suite_for_calibration(source.oracle_report, oracle_report)
    if not calibration.is_keep:
        raise RecertificationError(
            "recertified recovery requires a fresh calibration keep"
        )
    try:
        ensure_oracle_exportable(
            oracle_report,
            candidate=source.candidate,
            calibration_kept=calibration.is_keep,
            kill_threshold=_final_evidence(oracle_report).threshold,
        )
    except Exception as exc:
        raise RecertificationError(
            f"recertified task is not exportable: {exc}"
        ) from exc
    return ExportRequest(
        candidate=source.candidate,
        spec=source.spec,
        oracle_report=oracle_report,
        calibration_report=calibration,
        env_image=source.env_image,
        repo_url=source.repo_url,
        base_commit=source.base_commit,
        repo=source.repo,
        task_id=source.task_id,
    )


async def recertify_recovery_export(
    out_dir: Path | str,
    *,
    source_dir: Path | str = CERTIFIED_RECOVERY_SOURCE,
    overwrite: bool = True,
    command_timeout: float = 600.0,
    mutation_timeout: float = 1200.0,
    recalibrator: Recalibrator | None = None,
    variant_generator: VariantGenerator | None = None,
    differential_synthesizer: VariantStrengthSynthesizer | None = None,
    alt_generator: AltCorrectGenerator | None = None,
    adapter: LanguageAdapter | None = None,
    docker_client: DockerClient | None = None,
) -> RecertificationResult:
    """Re-certify the recovery and publish exactly its genuine singleton."""
    generation, source_task = load_certified_recovery(source_dir)
    oracle_report = await recertify_final_oracle(
        source_task,
        command_timeout=command_timeout,
        mutation_timeout=mutation_timeout,
        variant_generator=variant_generator,
        differential_synthesizer=differential_synthesizer,
        alt_generator=alt_generator,
        adapter=adapter,
        docker_client=docker_client,
    )
    if not oracle_report.is_pass:
        raise RecertificationError(
            "recertified oracle did not pass: " + "; ".join(oracle_report.reasons)
        )
    try:
        require_unchanged_suite_for_calibration(
            source_task.oracle_report, oracle_report
        )
    except RecertificationError:
        if recalibrator is None:
            raise RecertificationError(
                "final hidden suite changed during recertification and no "
                "recalibrator was supplied"
            ) from None
        calibration = await recalibrator(source_task, oracle_report)
    else:
        calibration = source_task.calibration_report

    request = build_recertification_request(
        generation, oracle_report, calibration_report=calibration
    )
    result = export_batch(
        [request],
        out_dir,
        overwrite=overwrite,
        replace_existing=True,
    )
    if len(result.kept) != 1 or len(result.refused) != 0:
        raise RecertificationError(
            "recertification export did not publish exactly one genuine task"
        )
    return RecertificationResult(
        source_generation_id=generation.generation_id,
        task_id=source_task.task_id,
        oracle_report=oracle_report,
        export=result,
    )


__all__ = [
    "CERTIFIED_RECOVERY_SOURCE",
    "CERTIFIED_RECOVERY_TASK_ID",
    "RecertificationError",
    "RecertificationResult",
    "build_recertification_request",
    "load_certified_recovery",
    "recertify_final_oracle",
    "recertify_recovery_export",
    "require_unchanged_suite_for_calibration",
    "validate_certified_recovery_source",
    "validate_certified_recovery_task",
]
