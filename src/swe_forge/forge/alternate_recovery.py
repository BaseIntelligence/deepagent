"""One-shot, fail-closed final recovery for the retained boltons candidate.

The controller intentionally recognizes exactly one immutable input workspace.
It is not a general retry facility: after the single fresh calibration either a
fully reconciled keep is published or the canonical output is tombstoned.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, Sequence

import yaml  # type: ignore[import-untyped]

from swe_forge.forge.calibrate.filter import BandFilterConfig
from swe_forge.forge.calibrate.pipeline import run_calibration
from swe_forge.forge.export import ExportRequest, export_batch
from swe_forge.forge.gold_eval import run_gold_eval
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    CalibrationReport,
    EnvImage,
    GeneratedSpec,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.alt_correct import run_alt_correct_gate
from swe_forge.forge.oracle.alt_correct_synth import TeacherAltCorrectGenerator
from swe_forge.forge.oracle.differential import (
    NullVariantSynthesizer,
    run_differential_gate,
)
from swe_forge.forge.oracle.differential_synth import TeacherVariantGenerator
from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    HiddenTest,
    HiddenTestFile,
    TreeState,
    run_establish_gate,
)
from swe_forge.forge.oracle.flakiness import run_flakiness_gate
from swe_forge.forge.oracle.leak import run_leak_gate
from swe_forge.forge.oracle.multifault import (
    run_multifault_completeness_gate,
    verify_multifault_evidence,
)
from swe_forge.forge.oracle.mutation import run_final_mutation_gate
from swe_forge.forge.oracle.pipeline import verify_pass_consistency
from swe_forge.forge.panel import build_panel_from_env
from swe_forge.forge.publication import load_published_generation
from swe_forge.forge.recovery_accounting import (
    RecoveryBudgetLedger,
    reconcile_recovery_reports,
)
from swe_forge.forge.report import GoldSummary, build_benchmark_report, write_report
from swe_forge.forge.teacher import TeacherClient

ALTERNATE_RECOVERY_TASK_ID = "mahmoud-boltons__bug_combination__7bb4e61cc98c"
ORIGINAL_MISSION_BUDGET_USD = Decimal("1400")
INCREMENTAL_RECOVERY_CAP_USD = Decimal("25")

UPDATE_WRAPPER_F2P_PATH = "test_update_wrapper_wraps_basic.py"
UPDATE_WRAPPER_F2P_NODE = (
    "test_update_wrapper_wraps_basic.py::"
    "test_wraps_basic_regular_function_preserves_metadata_and_wrapped"
)
UPDATE_WRAPPER_F2P_COMMAND = f"python -m pytest {UPDATE_WRAPPER_F2P_NODE}"
UPDATE_WRAPPER_F2P_CONTENT = """\
from boltons.funcutils import wraps


def test_wraps_basic_regular_function_preserves_metadata_and_wrapped():
    def source(value):
        '''Return a value through the original callable.'''
        return value * 2

    @wraps(source)
    def wrapped(*args, **kwargs):
        return source(*args, **kwargs)

    assert wrapped(3) == 6
    assert wrapped.__name__ == source.__name__
    assert wrapped.__doc__ == source.__doc__
    assert wrapped.__wrapped__ is source
"""

_CANONICAL_AUDIT_ARTIFACTS = (
    "certification.json",
    "gold_eval.json",
    "report.json",
    "report.md",
)


class AlternateRecoveryError(RuntimeError):
    """Raised when the sole allowed alternate-recovery path cannot continue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise AlternateRecoveryError(f"{field} must be a non-negative decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AlternateRecoveryError(f"{field} must be a non-negative decimal") from exc
    if not result.is_finite() or result < 0:
        raise AlternateRecoveryError(f"{field} must be a non-negative decimal")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True)
class VerifiedOriginalBudget:
    """A durable verification of the original mission's remaining capacity."""

    original_budget_usd: str
    spent_usd: str
    remaining_usd: str
    incremental_cap_usd: str
    source_sha256: str
    source_path: str


def verify_original_budget(progress_path: Path | str) -> VerifiedOriginalBudget:
    """Verify original-$1400 accounting before authorizing any new reservation.

    The alternate path accepts the terminal harvest ledger only when it reports
    the original ceiling exactly, has no active reservation or in-flight batch,
    and its accounted spend is non-negative and no greater than that ceiling.
    """

    path = Path(progress_path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AlternateRecoveryError(
            f"cannot durably verify original mission budget from {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise AlternateRecoveryError("original budget record must be an object")
    budget = _decimal(payload.get("budget_usd"), field="budget_usd")
    spent = _decimal(payload.get("spend_usd"), field="spend_usd")
    reserved = _decimal(payload.get("reserved_usd"), field="reserved_usd")
    if budget != ORIGINAL_MISSION_BUDGET_USD:
        raise AlternateRecoveryError(
            "budget record does not prove the original $1400 mission budget"
        )
    if spent > budget:
        raise AlternateRecoveryError("original mission spend exceeds its budget")
    if reserved != 0:
        raise AlternateRecoveryError(
            "original mission budget record has an unresolved reservation"
        )
    if payload.get("in_flight_batch") is not None:
        raise AlternateRecoveryError(
            "original mission budget record has an in-flight batch"
        )
    remaining = budget - spent
    cap = min(INCREMENTAL_RECOVERY_CAP_USD, remaining)
    if cap <= 0:
        raise AlternateRecoveryError("no verified original mission budget remains")
    return VerifiedOriginalBudget(
        original_budget_usd=_decimal_text(budget),
        spent_usd=_decimal_text(spent),
        remaining_usd=_decimal_text(remaining),
        incremental_cap_usd=_decimal_text(cap),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_path=str(path),
    )


@dataclass(frozen=True)
class RecoveryCertification:
    """The explicit current-state certificate validators must inspect."""

    run_id: str
    state: Literal["pending", "keep", "tombstone"]
    passed: bool
    task_ids: tuple[str, ...]
    created_at: str
    previous_generation_id: str = ""
    reason: str = ""

    @classmethod
    def pending(
        cls, *, run_id: str, previous_generation_id: str, task_id: str
    ) -> "RecoveryCertification":
        return cls(
            run_id=run_id,
            state="pending",
            passed=False,
            task_ids=(task_id,),
            created_at=_utc_now(),
            previous_generation_id=previous_generation_id,
            reason="final alternate recovery is not certified",
        )

    @classmethod
    def tombstone(cls, *, run_id: str, reason: str) -> "RecoveryCertification":
        return cls(
            run_id=run_id,
            state="tombstone",
            passed=False,
            task_ids=(),
            created_at=_utc_now(),
            reason=reason,
        )

    @classmethod
    def keep(cls, *, run_id: str, task_id: str) -> "RecoveryCertification":
        return cls(
            run_id=run_id,
            state="keep",
            passed=True,
            task_ids=(task_id,),
            created_at=_utc_now(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "state": self.state,
            "passed": self.passed,
            "task_ids": list(self.task_ids),
            "created_at": self.created_at,
            "previous_generation_id": self.previous_generation_id,
            "reason": self.reason,
        }

    @staticmethod
    def freeze_suite(
        tests: Sequence[OracleTestFile],
    ) -> list[OracleTestFile]:
        """Append the fixed upstream-grounded update-wrapper test exactly once."""

        frozen = list(tests)
        existing = next(
            (test for test in frozen if test.path == UPDATE_WRAPPER_F2P_PATH),
            None,
        )
        if existing is None:
            frozen.append(
                OracleTestFile(
                    path=UPDATE_WRAPPER_F2P_PATH,
                    content=UPDATE_WRAPPER_F2P_CONTENT,
                    origin="provided",
                )
            )
        elif existing.content != UPDATE_WRAPPER_F2P_CONTENT:
            raise AlternateRecoveryError(
                "alternate recovery update-wrapper test path has different content"
            )
        return frozen


def write_recovery_certification(
    out_dir: Path | str, certification: RecoveryCertification
) -> Path:
    """Atomically write the active certification state before any live call."""

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "certification.json"
    temp = root / f".certification-{uuid.uuid4().hex}"
    encoded = json.dumps(certification.to_dict(), indent=2, sort_keys=True) + "\n"
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temp.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class RehydratedAlternate:
    """Immutable candidate input rebuilt from the retained audit workspace."""

    task_id: str
    source_workspace: Path
    source_sha256: dict[str, str]
    candidate: Candidate
    spec: GeneratedSpec
    env_image: EnvImage
    repo_url: str
    repo: str
    base_commit: str
    broken_tree: Path
    tests: tuple[OracleTestFile, ...]


def _read_workspace_yaml(path: Path) -> dict[str, object]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AlternateRecoveryError(
            f"invalid retained workspace yaml: {path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise AlternateRecoveryError("retained workspace yaml must be a mapping")
    return loaded


def _workspace_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AlternateRecoveryError(
            f"retained alternate artifact is missing: {path}"
        ) from exc


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlternateRecoveryError(f"retained alternate workspace lacks {name}")
    return value


def rehydrate_alternate(
    source_workspace: Path | str,
) -> RehydratedAlternate:
    """Rebuild the precise retained candidate without regenerating any patch."""

    root = Path(source_workspace)
    workspace = _read_workspace_yaml(root / "workspace.yaml")
    task_id = _required_text(workspace.get("task_id"), name="task_id")
    if task_id != ALTERNATE_RECOVERY_TASK_ID:
        raise AlternateRecoveryError(
            "retained workspace is not the approved final alternate candidate"
        )
    patch_path = root / "patch.diff"
    mutation_path = root / "deletion_patch.diff"
    provenance_path = root / "provenance.json"
    repo = root / "repo"
    tests_dir = root / "tests"
    if not repo.is_dir() or not tests_dir.is_dir():
        raise AlternateRecoveryError("retained alternate workspace is incomplete")
    try:
        source_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlternateRecoveryError(
            "retained alternate provenance is unreadable"
        ) from exc
    if not isinstance(source_provenance, dict):
        raise AlternateRecoveryError("retained alternate provenance must be an object")
    details = source_provenance.get("details")
    if not isinstance(details, dict):
        raise AlternateRecoveryError("retained alternate provenance lacks details")
    if details.get("mutants_total") != 59 or details.get("mutants_killed") != 59:
        raise AlternateRecoveryError(
            "retained alternate mutation evidence is not the required 59/59 input"
        )
    synthetic = workspace.get("synthetic")
    environment = workspace.get("environment")
    repo_data = workspace.get("repo")
    if not isinstance(synthetic, dict) or not isinstance(environment, dict):
        raise AlternateRecoveryError("retained alternate workspace lacks metadata")
    if not isinstance(repo_data, dict):
        raise AlternateRecoveryError("retained alternate workspace lacks repository")
    oracle_patch = patch_path.read_text(encoding="utf-8")
    mutation_patch = mutation_path.read_text(encoding="utf-8")
    if not oracle_patch.endswith("\n") or not mutation_patch.endswith("\n"):
        raise AlternateRecoveryError("retained alternate patches must end in newlines")
    frozen_tests: list[OracleTestFile] = []
    for test_path in sorted(path for path in tests_dir.rglob("*") if path.is_file()):
        frozen_tests.append(
            OracleTestFile(
                path=test_path.relative_to(tests_dir).as_posix(),
                content=test_path.read_text(encoding="utf-8"),
                origin="provided",
            )
        )
    frozen_tests = RecoveryCertification.freeze_suite(frozen_tests)
    target = CandidateTarget(
        files=("boltons/funcutils.py", "boltons/setutils.py"),
        symbols=("update_wrapper", "complement"),
    )
    candidate = Candidate(
        language="python",
        generator="bug_combination",
        target=target,
        mutation_patch=mutation_patch,
        oracle_patch=oracle_patch,
        difficulty_hint="high",
        provenance=Provenance(
            generator="bug_combination",
            seed=int(source_provenance.get("seed", 30)),
            language="python",
            created_at=str(source_provenance.get("created_at", "")),
            tool_versions=dict(source_provenance.get("tool_versions", {})),
            details={
                "constituents": [
                    {
                        "index": 0,
                        "file": "boltons/funcutils.py",
                        "mutation_patch": _single_file_patch(
                            mutation_patch, "boltons/funcutils.py"
                        ),
                        "inverse_patch": _single_file_patch(
                            oracle_patch, "boltons/funcutils.py"
                        ),
                    },
                    {
                        "index": 1,
                        "file": "boltons/setutils.py",
                        "mutation_patch": _single_file_patch(
                            mutation_patch, "boltons/setutils.py"
                        ),
                        "inverse_patch": _single_file_patch(
                            oracle_patch, "boltons/setutils.py"
                        ),
                    },
                ],
                "recovery": {
                    "source_task_id": task_id,
                    "immutable_source_sha256": {
                        "patch.diff": _workspace_sha256(patch_path),
                        "deletion_patch.diff": _workspace_sha256(mutation_path),
                        "provenance.json": _workspace_sha256(provenance_path),
                    },
                },
            },
        ),
    )
    requirements = workspace.get("requirements", [])
    if not isinstance(requirements, list):
        raise AlternateRecoveryError(
            "retained alternate workspace requirements invalid"
        )
    spec = GeneratedSpec(
        problem_statement=_required_text(workspace.get("prompt"), name="prompt"),
        requirements=[
            str(item) for item in requirements if isinstance(item, str) and item.strip()
        ],
        interface_block=_required_text(workspace.get("interface"), name="interface"),
        provenance=candidate.provenance,
    )
    install = workspace.get("install")
    install_commands = (
        [str(item) for item in install.get("commands", []) if isinstance(item, str)]
        if isinstance(install, dict)
        else []
    )
    p2p = (
        "python -m pytest -k 'not (test_wraps_basic or test_wraps_injected or "
        "test_wraps_update_dict or test_wraps_expected or test_wraps_py3 or "
        "test_remove_kwonly_arg or test_wraps_inner_kwarg_only or test_wraps_async "
        "or test_wraps_hide_wrapped or test_complement_set)'"
    )
    env_image = EnvImage(
        repo_id="mahmoud-boltons",
        language="python",
        image_tag=_required_text(environment.get("image"), name="environment.image"),
        base_image=str(environment.get("base_image", "python:3.12-slim")),
        commit=_required_text(repo_data.get("base_commit"), name="repo.base_commit"),
        workspace_dir=_required_text(environment.get("repo_path"), name="repo_path"),
        install_commands=install_commands,
        baseline_test_command=p2p,
        original_public_test_command="python -m pytest",
        baseline_green=True,
        baseline_exit_code=0,
        baseline_summary="rehydrated immutable alternate, public suite reverified",
        provenance={"alternate_recovery": task_id},
    )
    return RehydratedAlternate(
        task_id=task_id,
        source_workspace=root,
        source_sha256={
            "patch.diff": _workspace_sha256(patch_path),
            "deletion_patch.diff": _workspace_sha256(mutation_path),
            "provenance.json": _workspace_sha256(provenance_path),
        },
        candidate=candidate,
        spec=spec,
        env_image=env_image,
        repo_url=_required_text(repo_data.get("url"), name="repo.url"),
        repo="mahmoud/boltons",
        base_commit=env_image.commit,
        broken_tree=repo,
        tests=tuple(frozen_tests),
    )


def _single_file_patch(patch: str, path: str) -> str:
    """Extract an executable one-file patch from a two-file git diff."""

    chunks = [
        "diff --git a/" + chunk for chunk in patch.split("diff --git a/") if chunk
    ]
    selected = next(
        (
            chunk
            for chunk in chunks
            if chunk.startswith(f"diff --git a/{path} b/{path}\n")
        ),
        "",
    )
    if not selected:
        raise AlternateRecoveryError(
            f"retained alternate patch does not contain constituent {path}"
        )
    return selected if selected.endswith("\n") else selected + "\n"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


async def verify_unfiltered_public_gold(
    alternate: RehydratedAlternate, *, command_timeout: float = 600.0
) -> None:
    """Prove the upstream unfiltered suite on immutable gold before teacher calls."""

    from swe_forge.execution.docker_client import DockerClient
    from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

    sandbox = DockerSandbox(
        DockerClient(),
        SandboxConfig(
            name="swe-forge-alternate-public",
            image=alternate.env_image.image_tag,
            workspace_dir=alternate.env_image.workspace_dir,
            command_timeout=command_timeout,
        ),
    )
    async with sandbox:
        recipe = DockerOracleRecipe(
            sandbox,
            language="python",
            workspace_dir=alternate.env_image.workspace_dir,
            mutation_patch=alternate.candidate.mutation_patch,
            oracle_patch=alternate.candidate.oracle_patch,
            p2p_command=alternate.env_image.original_public_test_command,
            command_timeout=command_timeout,
        )
        await recipe.set_state(TreeState.GOLD)
        result = await recipe.run_p2p()
    if not result.passed:
        raise AlternateRecoveryError(
            "unfiltered upstream/public suite is not gold-green "
            f"(exit {result.exit_code})"
        )


def _hidden_tests(tests: Sequence[OracleTestFile]) -> tuple[HiddenTest, ...]:
    result: list[HiddenTest] = []
    for test in tests:
        command = (
            UPDATE_WRAPPER_F2P_COMMAND
            if test.path == UPDATE_WRAPPER_F2P_PATH
            else f"python -m pytest {test.path}"
        )
        result.append(
            HiddenTest(
                test_id=command,
                files=(HiddenTestFile(path=test.path, content=test.content),),
                origin="provided",
            )
        )
    return tuple(result)


async def run_normal_oracle_gates(
    alternate: RehydratedAlternate,
    *,
    ledger: RecoveryBudgetLedger,
    command_timeout: float = 600.0,
    mutation_timeout: float = 1200.0,
) -> OracleReport:
    """Run the frozen normal gates with real teacher calls and zero retries."""

    established = await run_establish_gate(
        alternate.candidate,
        alternate.env_image,
        provided_tests=_hidden_tests(alternate.tests),
        synthesizer=None,
        command_timeout=command_timeout,
    )
    if not established.is_pass:
        return established
    flakiness = await run_flakiness_gate(
        alternate.candidate,
        alternate.env_image,
        established,
        command_timeout=command_timeout,
    )
    if not flakiness.is_pass:
        return flakiness
    initial_multifault = await run_multifault_completeness_gate(
        alternate.candidate,
        alternate.env_image,
        flakiness,
        command_timeout=command_timeout,
    )
    if not initial_multifault.is_pass:
        return initial_multifault
    differential_client = TeacherClient.from_settings(
        recovery_ledger=ledger,
        recovery_stage="oracle.differential",
        num_retries=0,
    )
    differential = await run_differential_gate(
        alternate.candidate,
        alternate.env_image,
        initial_multifault,
        variant_generator=TeacherVariantGenerator(client=differential_client),
        synthesizer=NullVariantSynthesizer(),
        command_timeout=command_timeout,
    )
    if not differential.is_pass:
        return differential
    alt_client = TeacherClient.from_settings(
        recovery_ledger=ledger,
        recovery_stage="oracle.alt_correct",
        num_retries=0,
    )
    alt_correct = await run_alt_correct_gate(
        alternate.candidate,
        alternate.env_image,
        differential,
        spec=alternate.spec,
        alt_generator=TeacherAltCorrectGenerator(client=alt_client),
        command_timeout=command_timeout,
    )
    if not alt_correct.is_pass:
        return alt_correct
    final_mutation = await run_final_mutation_gate(
        alternate.candidate,
        alternate.env_image,
        alt_correct,
        threshold=0.8,
        command_timeout=mutation_timeout,
    )
    if not final_mutation.is_pass:
        return final_mutation
    multifault = await run_multifault_completeness_gate(
        alternate.candidate,
        alternate.env_image,
        final_mutation,
        command_timeout=command_timeout,
    )
    if not multifault.is_pass:
        return multifault
    problems = [
        *verify_pass_consistency(multifault, kill_threshold=0.8),
        *verify_multifault_evidence(multifault, candidate=alternate.candidate),
    ]
    if problems:
        raise AlternateRecoveryError(
            "final oracle evidence is inconsistent: " + "; ".join(problems)
        )
    return await run_leak_gate(
        alternate.candidate,
        alternate.env_image,
        multifault,
        command_timeout=command_timeout,
    )


def _tombstone_stage_writer(certification: RecoveryCertification, reason: str):
    def write(stage: Path, _entries: Sequence[object]) -> None:
        _write_json(stage / "certification.json", certification.to_dict())
        _write_json(
            stage / "invalidation.json",
            {
                "schema_version": 1,
                "state": "tombstone",
                "passed": False,
                "task_ids": [],
                "reason": reason,
            },
        )
        _write_json(
            stage / "report.json",
            {
                "shipped_count": 0,
                "passed": False,
                "invalidation_reason": reason,
            },
        )
        (stage / "report.md").write_text(
            "# SWE-Forge Benchmark Report\n\n- Overall: **INVALIDATED**\n"
            f"- Reason: {reason}\n",
            encoding="utf-8",
        )
        _write_json(
            stage / "gold_eval.json",
            {
                "shipped_count": 0,
                "gold_count": 0,
                "passed": False,
                "invalidation_reason": reason,
            },
        )

    return write


def _promote_regular_audit_artifacts_to_prior_generation(output: Path) -> None:
    """Preserve stale root evidence in its immutable audit generation.

    Older publications predate report/certification facades.  Before switching
    to the recovery's generation-backed audit surfaces, move only those
    root-level audit files beneath the old selected generation. The stale
    generation remains available for review but the pending certificate makes
    it non-shippable.
    """

    previous = load_published_generation(output)
    for name in _CANONICAL_AUDIT_ARTIFACTS:
        source = output / name
        if source.is_symlink() or not source.is_file():
            continue
        if previous is not None:
            destination = previous.root / name
            if not destination.exists():
                shutil.copy2(source, destination)
        source.unlink()


def _keep_stage_writer(
    *,
    certification: RecoveryCertification,
    ledger_path: Path,
    alternate: RehydratedAlternate,
    oracle_report: OracleReport,
    calibration_report: CalibrationReport,
):
    def write(stage: Path, entries: Sequence[object]) -> None:
        gold = run_gold_eval(stage, runs=2, name_prefix="swe-forge-alternate-gold")
        if not gold.passed:
            raise AlternateRecoveryError("strict two-run gold proof failed")
        gold_payload = gold.to_dict()
        gold_payload["tasks_dir"] = "tasks"
        _write_json(stage / "gold_eval.json", gold_payload)
        report = build_benchmark_report(
            stage,
            gold=GoldSummary.from_gold_eval(gold),
            frontier_threshold=0.51,
            band_config=BandFilterConfig(band_high=0.5, discrimination_threshold=1.0),
        )
        if not report.passed:
            raise AlternateRecoveryError("benchmark report/audit reconciliation failed")
        write_report(report, stage)
        if len(entries) != 1:
            raise AlternateRecoveryError(
                "alternate keep did not stage exactly one task"
            )
        _write_json(stage / "certification.json", certification.to_dict())
        _write_json(
            stage / "recovery-evidence.json",
            {
                "schema_version": 1,
                "task_id": alternate.task_id,
                "source_workspace": str(alternate.source_workspace),
                "source_sha256": dict(alternate.source_sha256),
                "suite_fingerprint": oracle_report.final_mutation_evidence.suite_fingerprint
                if oracle_report.final_mutation_evidence
                else "",
                "mutants_total": oracle_report.mutants_total,
                "mutants_killed": oracle_report.mutants_killed,
                "calibration": calibration_report.to_dict(),
            },
        )
        (stage / "recovery-ledger.jsonl").write_bytes(ledger_path.read_bytes())

    return write


@dataclass(frozen=True)
class AlternateRecoveryResult:
    """The terminal state of the sole authorized alternate recovery attempt."""

    run_id: str
    status: Literal["kept", "tombstoned"]
    reason: str
    task_id: str = ""


async def run_final_alternate_recovery(
    *,
    out_dir: Path | str = Path("results/pilot_final"),
    source_workspace: Path | str = Path(
        "results/pilot_keeps/tasks/" + ALTERNATE_RECOVERY_TASK_ID
    ),
    budget_progress: Path | str = Path("results/pilot_keeps/harvest_progress.json"),
    work_root: Path | str = Path("results/final_alternate_recovery"),
) -> AlternateRecoveryResult:
    """Execute exactly one no-retry recovery, keeping or tombstoning atomically."""

    output = Path(out_dir)
    work = Path(work_root)
    run_id = f"alternate-{uuid.uuid4().hex}"
    previous = load_published_generation(output)
    pending = RecoveryCertification.pending(
        run_id=run_id,
        previous_generation_id=previous.generation_id if previous else "",
        task_id=ALTERNATE_RECOVERY_TASK_ID,
    )
    # This root-level marker is deliberately set before any Docker/LLM request:
    # a stale selected generation cannot be accepted while recertification runs.
    write_recovery_certification(output, pending)
    reason = ""
    try:
        verified_budget = verify_original_budget(budget_progress)
        work.mkdir(parents=True, exist_ok=False)
        _write_json(work / "budget-verification.json", verified_budget.__dict__)
        ledger_path = work / "recovery-ledger.jsonl"
        ledger = RecoveryBudgetLedger(
            ledger_path,
            run_id=run_id,
            cap_usd=verified_budget.incremental_cap_usd,
            worst_case_cost_usd="3.00",
        )
        alternate = rehydrate_alternate(source_workspace)
        _write_json(
            work / "preflight.json",
            {
                "run_id": run_id,
                "task_id": alternate.task_id,
                "max_retries": 0,
                "k": 6,
                "band_high": 0.5,
                "discrimination_threshold": 1.0,
                "source_sha256": alternate.source_sha256,
                "budget": verified_budget.__dict__,
            },
        )
        await verify_unfiltered_public_gold(alternate)
        oracle = await run_normal_oracle_gates(alternate, ledger=ledger)
        _write_json(work / "oracle-report.json", oracle.to_dict())
        if not oracle.is_pass:
            raise AlternateRecoveryError(
                "normal oracle gates rejected: " + "; ".join(oracle.reasons)
            )
        calibration = (
            await run_calibration(
                alternate.candidate,
                alternate.env_image,
                alternate.spec,
                oracle,
                build_panel_from_env(),
                k=6,
                concurrency=4,
                validate=True,
                config=BandFilterConfig(band_high=0.5, discrimination_threshold=1.0),
                validate_num_retries=0,
                rollout_num_retries=0,
                recovery_ledger=ledger,
            )
        ).report
        if not calibration.is_keep:
            raise AlternateRecoveryError(
                "fresh calibration dropped: " + "; ".join(calibration.reasons)
            )
        reconcile_recovery_reports(ledger, oracle, calibration)
        keep = RecoveryCertification.keep(run_id=run_id, task_id=alternate.task_id)
        _promote_regular_audit_artifacts_to_prior_generation(output)
        export = export_batch(
            [
                ExportRequest(
                    candidate=alternate.candidate,
                    spec=alternate.spec,
                    oracle_report=oracle,
                    calibration_report=calibration,
                    env_image=alternate.env_image,
                    repo_url=alternate.repo_url,
                    base_commit=alternate.base_commit,
                    repo=alternate.repo,
                    task_id=alternate.task_id,
                    broken_tree=alternate.broken_tree,
                )
            ],
            output,
            overwrite=True,
            replace_existing=True,
            generation_metadata_writer=_keep_stage_writer(
                certification=keep,
                ledger_path=ledger_path,
                alternate=alternate,
                oracle_report=oracle,
                calibration_report=calibration,
            ),
            extra_facade_artifacts=_CANONICAL_AUDIT_ARTIFACTS,
        )
        if len(export.kept) != 1 or len(export.refused) != 0:
            raise AlternateRecoveryError("alternate keep export did not reconcile")
        _write_json(
            work / "result.json",
            {"run_id": run_id, "status": "kept", "task_id": alternate.task_id},
        )
        return AlternateRecoveryResult(
            run_id=run_id, status="kept", reason="", task_id=alternate.task_id
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        tombstone = RecoveryCertification.tombstone(run_id=run_id, reason=reason)
        try:
            _promote_regular_audit_artifacts_to_prior_generation(output)
            export_batch(
                [],
                output,
                overwrite=True,
                generation_metadata_writer=_tombstone_stage_writer(tombstone, reason),
                extra_facade_artifacts=_CANONICAL_AUDIT_ARTIFACTS,
            )
            work.mkdir(parents=True, exist_ok=True)
            _write_json(
                work / "result.json",
                {"run_id": run_id, "status": "tombstoned", "reason": reason},
            )
        except Exception as tombstone_error:
            raise AlternateRecoveryError(
                f"alternate recovery failed and tombstone publication failed: {tombstone_error}"
            ) from tombstone_error
        return AlternateRecoveryResult(
            run_id=run_id, status="tombstoned", reason=reason
        )


__all__ = [
    "ALTERNATE_RECOVERY_TASK_ID",
    "INCREMENTAL_RECOVERY_CAP_USD",
    "ORIGINAL_MISSION_BUDGET_USD",
    "UPDATE_WRAPPER_F2P_COMMAND",
    "UPDATE_WRAPPER_F2P_CONTENT",
    "UPDATE_WRAPPER_F2P_NODE",
    "UPDATE_WRAPPER_F2P_PATH",
    "AlternateRecoveryError",
    "RecoveryCertification",
    "RehydratedAlternate",
    "AlternateRecoveryResult",
    "VerifiedOriginalBudget",
    "rehydrate_alternate",
    "run_final_alternate_recovery",
    "run_normal_oracle_gates",
    "verify_original_budget",
    "verify_unfiltered_public_gold",
    "write_recovery_certification",
]
