"""Fail-closed final-suite completeness gate for multi-fault candidates.

``bug_combination`` and ``multi_file`` manufacture several constituent faults in
one candidate.  Their combined FAIL->PASS proof is insufficient: a hidden suite
can accidentally accept a repair that restores only some constituents.  This
gate proves every constituent is independently required by the *final* hidden
suite by repairing every other constituent and leaving that one fault broken.

The proof is intentionally performed after all suite-mutating gates have
completed.  Every leave-one-broken tree must keep the final P2P command green
and fail at least one final hidden F2P test.  Missing metadata, a failed inverse
patch application, a P2P regression, or an accepted partial repair is a reject.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shlex
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from swe_forge.forge.adapters import LanguageAdapter, build_default_registry
from swe_forge.forge.models import (
    Candidate,
    EnvImage,
    OracleReport,
    OracleTestFile,
    require_green_baseline,
)
from swe_forge.forge.oracle.differential import reconstruct_suite_tests
from swe_forge.forge.oracle.establish import DockerOracleRecipe, HiddenTest, TreeState
from swe_forge.forge.oracle.mutation import final_suite_fingerprint

if TYPE_CHECKING:
    from swe_forge.execution.docker_client import DockerClient

MULTIFAULT_GENERATORS: frozenset[str] = frozenset({"bug_combination", "multi_file"})

REASON_METADATA = "multifault_metadata_invalid"
REASON_PARTIAL_APPLY = "multifault_partial_repair_apply_failed"
REASON_PARTIAL_P2P = "multifault_partial_repair_p2p_not_green"
REASON_PARTIAL_ACCEPTED = "multifault_partial_repair_accepted"


class MultiFaultError(RuntimeError):
    """Raised when constituent metadata cannot support a sound proof."""


@dataclass(frozen=True)
class ConstituentInversePatch:
    """One indexed executable inverse patch recorded by a structural generator."""

    index: int
    file: str
    mutation_patch: str
    inverse_patch: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise MultiFaultError("constituent index must be >= 0")
        if not self.file.strip():
            raise MultiFaultError("constituent file must be non-empty")
        if not self.mutation_patch.strip():
            raise MultiFaultError(
                f"constituent {self.index} mutation_patch must be non-empty"
            )
        if not self.inverse_patch.strip():
            raise MultiFaultError(
                f"constituent {self.index} inverse_patch must be non-empty"
            )

    @property
    def inverse_patch_sha256(self) -> str:
        return hashlib.sha256(self.inverse_patch.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PartialRepairScore:
    """Execution result for one leave-one-constituent-broken tree."""

    other_inverse_patches_applied: bool
    p2p_passed: bool
    failed_f2p_test_ids: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class ConstituentVerdict:
    """Persisted final-suite verdict for one constituent fault."""

    index: int
    file: str
    inverse_patch_sha256: str
    repaired_indices: tuple[int, ...]
    other_inverse_patches_applied: bool
    p2p_passed: bool
    failed_f2p_test_ids: tuple[str, ...]
    verdict: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.index < 0:
            raise MultiFaultError("constituent verdict index must be >= 0")
        if not self.file.strip():
            raise MultiFaultError("constituent verdict file must be non-empty")
        if len(self.inverse_patch_sha256) != 64:
            raise MultiFaultError(
                "constituent inverse_patch_sha256 must be a SHA-256 digest"
            )
        if self.verdict not in {"pass", "reject"}:
            raise MultiFaultError(
                f"constituent verdict must be 'pass' or 'reject'; got {self.verdict!r}"
            )
        if self.verdict == "reject" and not self.reason:
            raise MultiFaultError("rejected constituent verdict requires a reason")
        if self.verdict == "pass" and self.reason:
            raise MultiFaultError("passing constituent verdict must not carry a reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "file": self.file,
            "inverse_patch_sha256": self.inverse_patch_sha256,
            "repaired_indices": list(self.repaired_indices),
            "other_inverse_patches_applied": self.other_inverse_patches_applied,
            "p2p_passed": self.p2p_passed,
            "failed_f2p_test_ids": list(self.failed_f2p_test_ids),
            "verdict": self.verdict,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ConstituentVerdict:
        repaired = data.get("repaired_indices", [])
        failed = data.get("failed_f2p_test_ids", [])
        return cls(
            index=int(str(data["index"])),
            file=str(data["file"]),
            inverse_patch_sha256=str(data["inverse_patch_sha256"]),
            repaired_indices=tuple(int(str(item)) for item in repaired)
            if isinstance(repaired, list)
            else (),
            other_inverse_patches_applied=bool(
                data.get("other_inverse_patches_applied", False)
            ),
            p2p_passed=bool(data.get("p2p_passed", False)),
            failed_f2p_test_ids=tuple(str(item) for item in failed)
            if isinstance(failed, list)
            else (),
            verdict=str(data["verdict"]),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class MultiFaultCompletenessEvidence:
    """Final-suite proof that every constituent fault is independently required."""

    suite_fingerprint: str
    p2p_command: str
    constituents: tuple[ConstituentVerdict, ...]

    def __post_init__(self) -> None:
        if len(self.suite_fingerprint) != 64:
            raise MultiFaultError(
                "multifault suite_fingerprint must be a SHA-256 digest"
            )
        if not self.p2p_command.strip():
            raise MultiFaultError("multifault p2p_command must be non-empty")
        if not self.constituents:
            raise MultiFaultError("multifault evidence requires constituent verdicts")
        indexes = [record.index for record in self.constituents]
        if indexes != list(range(len(indexes))):
            raise MultiFaultError(
                "multifault constituent verdict indexes must be contiguous from zero"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "suite_fingerprint": self.suite_fingerprint,
            "p2p_command": self.p2p_command,
            "constituents": [record.to_dict() for record in self.constituents],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MultiFaultCompletenessEvidence:
        records = data.get("constituents", [])
        if not isinstance(records, list):
            raise MultiFaultError("multifault evidence constituents must be a list")
        return cls(
            suite_fingerprint=str(data["suite_fingerprint"]),
            p2p_command=str(data["p2p_command"]),
            constituents=tuple(
                ConstituentVerdict.from_dict(record)
                for record in records
                if isinstance(record, dict)
            ),
        )


@dataclass
class MultiFaultOutcome:
    """Outcome of the final-suite completeness gate."""

    verdict: str
    reasons: list[str]
    evidence: MultiFaultCompletenessEvidence | None
    details: dict[str, object] = field(default_factory=dict)


class PartialRepairRunner(Protocol):
    """Scores one partial repair against the final P2P and hidden F2P suite."""

    async def score(
        self,
        leave_broken: ConstituentInversePatch,
        repairs: Sequence[ConstituentInversePatch],
        tests: Sequence[HiddenTest],
        *,
        p2p_command: str,
    ) -> PartialRepairScore: ...


def _raw_constituents(candidate: Candidate) -> object:
    details = candidate.provenance.details if candidate.provenance else {}
    return details.get("constituents") if isinstance(details, dict) else None


def normalize_constituent_inverse_patches(
    candidate: Candidate,
) -> tuple[ConstituentInversePatch, ...]:
    """Validate and normalize structural-generator constituent metadata.

    Both multi-fault generators emit the canonical ``constituents`` list.  Older
    ``faults``/``edits`` metadata is deliberately not accepted because it lacks
    the uniform, indexed executable inverse-patch contract required by this
    final-suite gate.
    """

    if candidate.generator not in MULTIFAULT_GENERATORS:
        return ()
    raw = _raw_constituents(candidate)
    if not isinstance(raw, list) or len(raw) < 2:
        raise MultiFaultError(
            "constituent metadata must be a list with at least two records"
        )

    records: list[ConstituentInversePatch] = []
    for position, value in enumerate(raw):
        if not isinstance(value, dict):
            raise MultiFaultError(
                f"constituent metadata record {position} must be an object"
            )
        index = value.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise MultiFaultError(
                f"constituent metadata record {position} has no integer index"
            )
        file = value.get("file")
        mutation_patch = value.get("mutation_patch")
        inverse_patch = value.get("inverse_patch")
        if not isinstance(file, str):
            raise MultiFaultError(f"constituent metadata record {index} has no file")
        if not isinstance(mutation_patch, str):
            raise MultiFaultError(
                f"constituent metadata record {index} has no mutation_patch"
            )
        if not isinstance(inverse_patch, str):
            raise MultiFaultError(
                f"constituent metadata record {index} has no inverse_patch"
            )
        records.append(
            ConstituentInversePatch(
                index=index,
                file=file,
                mutation_patch=mutation_patch,
                inverse_patch=inverse_patch,
            )
        )

    ordered = tuple(sorted(records, key=lambda record: record.index))
    indexes = [record.index for record in ordered]
    if indexes != list(range(len(ordered))):
        raise MultiFaultError(
            "constituent metadata indexes must be contiguous and unique from zero"
        )
    files = [record.file for record in ordered]
    if len(set(files)) != len(files):
        raise MultiFaultError("constituent metadata must contain distinct files")
    if set(files) != set(candidate.target.files):
        raise MultiFaultError(
            "constituent metadata files must exactly cover Candidate.target.files"
        )
    digests = [record.inverse_patch_sha256 for record in ordered]
    if len(set(digests)) != len(digests):
        raise MultiFaultError(
            "constituent metadata must contain distinct inverse patches"
        )
    return ordered


def constituent_metadata_fingerprint(
    constituents: Sequence[ConstituentInversePatch],
) -> str:
    """Canonical digest binding an evidence record to indexed inverse patches."""

    canonical = [
        {
            "index": record.index,
            "file": record.file,
            "inverse_patch_sha256": record.inverse_patch_sha256,
        }
        for record in constituents
    ]
    encoded = json.dumps(
        canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verdict(
    leave_broken: ConstituentInversePatch,
    repairs: Sequence[ConstituentInversePatch],
    score: PartialRepairScore,
) -> ConstituentVerdict:
    if not score.other_inverse_patches_applied:
        reason = (
            f"{REASON_PARTIAL_APPLY}: constituent {leave_broken.index}: "
            f"{score.error or 'one or more other constituent inverse patches failed'}"
        )
    elif not score.p2p_passed:
        reason = (
            f"{REASON_PARTIAL_P2P}: constituent {leave_broken.index}: "
            "final P2P is not green when this fault remains broken"
        )
    elif not score.failed_f2p_test_ids:
        reason = (
            f"{REASON_PARTIAL_ACCEPTED}: constituent {leave_broken.index}: "
            "final hidden F2P all passed after repairing every other constituent"
        )
    else:
        reason = ""
    return ConstituentVerdict(
        index=leave_broken.index,
        file=leave_broken.file,
        inverse_patch_sha256=leave_broken.inverse_patch_sha256,
        repaired_indices=tuple(record.index for record in repairs),
        other_inverse_patches_applied=score.other_inverse_patches_applied,
        p2p_passed=score.p2p_passed,
        failed_f2p_test_ids=tuple(score.failed_f2p_test_ids),
        verdict="pass" if not reason else "reject",
        reason=reason,
    )


async def assess_multifault_completeness(
    candidate: Candidate,
    test_files: Sequence[OracleTestFile],
    *,
    p2p_command: str,
    runner: PartialRepairRunner,
    adapter: LanguageAdapter | None = None,
    fail_to_pass: Sequence[str] = (),
) -> MultiFaultOutcome:
    """Run the deterministic leave-one-fault-broken proof for one candidate."""

    if candidate.generator not in MULTIFAULT_GENERATORS:
        return MultiFaultOutcome(verdict="pass", reasons=[], evidence=None)

    try:
        constituents = normalize_constituent_inverse_patches(candidate)
    except MultiFaultError as exc:
        return MultiFaultOutcome(
            verdict="reject",
            reasons=[f"{REASON_METADATA}: {exc}"],
            evidence=None,
        )
    if adapter is None:
        adapter = build_default_registry().get(candidate.language)
    tests = reconstruct_suite_tests(adapter, fail_to_pass, test_files)
    if not tests:
        return MultiFaultOutcome(
            verdict="reject",
            reasons=[
                f"{REASON_METADATA}: final hidden suite contains no executable F2P tests"
            ],
            evidence=None,
        )
    if not p2p_command.strip():
        return MultiFaultOutcome(
            verdict="reject",
            reasons=[f"{REASON_METADATA}: final P2P command is missing"],
            evidence=None,
        )

    verdicts: list[ConstituentVerdict] = []
    for leave_broken in constituents:
        repairs = tuple(
            record for record in constituents if record.index != leave_broken.index
        )
        score = await runner.score(
            leave_broken, repairs, tests, p2p_command=p2p_command
        )
        verdicts.append(_verdict(leave_broken, repairs, score))

    evidence = MultiFaultCompletenessEvidence(
        suite_fingerprint=final_suite_fingerprint(test_files),
        p2p_command=p2p_command,
        constituents=tuple(verdicts),
    )
    reasons = [record.reason for record in verdicts if record.reason]
    return MultiFaultOutcome(
        verdict="pass" if not reasons else "reject",
        reasons=reasons,
        evidence=evidence,
        details={
            "stage": "oracle.multifault",
            "constituent_metadata_fingerprint": constituent_metadata_fingerprint(
                constituents
            ),
            "suite_fingerprint": evidence.suite_fingerprint,
            "p2p_command": p2p_command,
            "constituents": [record.to_dict() for record in evidence.constituents],
        },
    )


class DockerPartialRepairRunner:
    """Docker-backed runner using the shared FAIL->PASS recipe in fresh sandboxes."""

    def __init__(
        self,
        candidate: Candidate,
        env_image: EnvImage,
        adapter: LanguageAdapter,
        *,
        command_timeout: float = 600.0,
        docker_client: DockerClient | None = None,
    ) -> None:
        self._candidate = candidate
        self._env_image = env_image
        self._adapter = adapter
        self._timeout = command_timeout
        self._docker_client = docker_client

    @contextlib.asynccontextmanager
    async def _recipe(self, p2p_command: str) -> AsyncIterator[DockerOracleRecipe]:
        from swe_forge.execution.docker_client import DockerClient
        from swe_forge.execution.sandbox import DockerSandbox, SandboxConfig

        client = self._docker_client or DockerClient()
        sandbox = DockerSandbox(
            client,
            SandboxConfig(
                name="swe-forge-oracle-multifault",
                image=self._env_image.image_tag,
                workspace_dir=self._env_image.workspace_dir,
                command_timeout=self._timeout,
            ),
        )
        async with sandbox:
            yield DockerOracleRecipe(
                sandbox,
                language=self._candidate.language,
                workspace_dir=self._env_image.workspace_dir,
                mutation_patch=self._candidate.mutation_patch,
                oracle_patch=self._candidate.oracle_patch,
                p2p_command=p2p_command,
                command_timeout=self._timeout,
            )

    async def _apply_inverse(
        self, recipe: DockerOracleRecipe, constituent: ConstituentInversePatch
    ) -> str | None:
        path = f".swe_forge_oracle/constituent-{constituent.index}.patch"
        content = constituent.inverse_patch
        await recipe.sandbox.write_file(
            path, content if content.endswith("\n") else content + "\n"
        )
        primary = await recipe.sandbox.run_command(
            f"git apply --whitespace=nowarn {shlex.quote(path)}", timeout=self._timeout
        )
        if primary.exit_code == 0:
            return None
        fallback = await recipe.sandbox.run_command(
            f"git apply --3way --whitespace=nowarn {shlex.quote(path)}",
            timeout=self._timeout,
        )
        if fallback.exit_code == 0:
            return None
        output = fallback.stderr or fallback.stdout or primary.stderr or primary.stdout
        return output.strip()[:500] or "git apply rejected the inverse patch"

    async def _verify_partial_tree(
        self,
        recipe: DockerOracleRecipe,
        leave_broken: ConstituentInversePatch,
        repairs: Sequence[ConstituentInversePatch],
    ) -> str | None:
        """Prove repairs restored every other target file to the gold checkout."""

        for repair in repairs:
            result = await recipe.sandbox.run_command(
                f"git diff --quiet -- {shlex.quote(repair.file)}",
                timeout=self._timeout,
            )
            if result.exit_code != 0:
                output = (result.stderr or result.stdout).strip()[:500]
                return (
                    f"constituent {repair.index} inverse patch did not restore "
                    f"{repair.file!r} to the gold tree"
                    + (f": {output}" if output else "")
                )
        remaining = await recipe.sandbox.run_command(
            f"git diff --quiet -- {shlex.quote(leave_broken.file)}",
            timeout=self._timeout,
        )
        if remaining.exit_code == 0:
            return (
                f"constituent {leave_broken.index} is no longer broken after "
                "the other inverse patches were applied"
            )
        if remaining.exit_code != 1:
            output = (remaining.stderr or remaining.stdout).strip()[:500]
            return (
                f"could not verify that constituent {leave_broken.index} remains "
                f"broken (git diff exit {remaining.exit_code})"
                + (f": {output}" if output else "")
            )
        return None

    async def score(
        self,
        leave_broken: ConstituentInversePatch,
        repairs: Sequence[ConstituentInversePatch],
        tests: Sequence[HiddenTest],
        *,
        p2p_command: str,
    ) -> PartialRepairScore:
        async with self._recipe(p2p_command) as recipe:
            await recipe.set_state(TreeState.BROKEN)
            for repair in repairs:
                error = await self._apply_inverse(recipe, repair)
                if error:
                    return PartialRepairScore(
                        other_inverse_patches_applied=False,
                        p2p_passed=False,
                        error=(
                            f"repairing constituent {repair.index} while leaving "
                            f"{leave_broken.index} broken failed: {error}"
                        ),
                    )
            verification_error = await self._verify_partial_tree(
                recipe, leave_broken, repairs
            )
            if verification_error:
                return PartialRepairScore(
                    other_inverse_patches_applied=False,
                    p2p_passed=False,
                    error=verification_error,
                )

            p2p = await recipe.run_p2p()
            failed: list[str] = []
            for test in tests:
                await recipe.write_test(test)
                run = await recipe.run_test(test)
                await recipe.remove_test(test)
                if not run.passed:
                    failed.append(test.test_id)
            return PartialRepairScore(
                other_inverse_patches_applied=True,
                p2p_passed=p2p.passed,
                failed_f2p_test_ids=tuple(failed),
            )


def build_multifault_report(
    prior_report: OracleReport,
    outcome: MultiFaultOutcome,
) -> OracleReport:
    """Fold completeness evidence into the running oracle report."""

    details = dict(prior_report.details)
    details["multifault"] = outcome.details
    return OracleReport(
        language=prior_report.language,
        generator=prior_report.generator,
        verdict=outcome.verdict,
        reasons=list(outcome.reasons),
        fail_to_pass=list(prior_report.fail_to_pass),
        pass_to_pass=list(prior_report.pass_to_pass),
        test_files=list(prior_report.test_files),
        flakiness_runs=prior_report.flakiness_runs,
        mutants_total=prior_report.mutants_total,
        mutants_killed=prior_report.mutants_killed,
        final_mutation_evidence=prior_report.final_mutation_evidence,
        differential_pass=prior_report.differential_pass,
        alt_correct_accepted=prior_report.alt_correct_accepted,
        leak_audit=prior_report.leak_audit,
        multifault_evidence=outcome.evidence,
        provenance=prior_report.provenance,
        details=details,
    )


async def run_multifault_completeness_gate(
    candidate: Candidate,
    env_image: EnvImage,
    prior_report: OracleReport,
    *,
    adapter: LanguageAdapter | None = None,
    docker_client: DockerClient | None = None,
    command_timeout: float = 600.0,
) -> OracleReport:
    """Run the completeness proof after final mutation remeasurement."""

    require_green_baseline(env_image)
    if prior_report.verdict != "pass":
        raise MultiFaultError(
            "multifault gate requires a passing prior report; got "
            f"{prior_report.verdict!r}"
        )
    if candidate.generator not in MULTIFAULT_GENERATORS:
        return prior_report
    if adapter is None:
        adapter = build_default_registry().get(candidate.language)
    p2p_command = (
        prior_report.pass_to_pass[0]
        if prior_report.pass_to_pass
        else env_image.baseline_test_command
    )
    outcome = await assess_multifault_completeness(
        candidate,
        prior_report.test_files,
        p2p_command=p2p_command,
        runner=DockerPartialRepairRunner(
            candidate,
            env_image,
            adapter,
            command_timeout=command_timeout,
            docker_client=docker_client,
        ),
        adapter=adapter,
        fail_to_pass=prior_report.fail_to_pass,
    )
    return build_multifault_report(prior_report, outcome)


def verify_multifault_evidence(
    report: OracleReport,
    *,
    candidate: Candidate | None = None,
) -> list[str]:
    """Return any missing, stale, or unsound multi-fault final-suite evidence."""

    generator = candidate.generator if candidate is not None else report.generator
    if generator not in MULTIFAULT_GENERATORS:
        return []
    problems: list[str] = []
    if candidate is not None and candidate.generator != report.generator:
        problems.append(
            "multifault: candidate generator does not match OracleReport generator"
        )
    evidence = report.multifault_evidence
    if evidence is None:
        return ["multifault: final-suite constituent evidence is missing"]
    try:
        expected_suite = final_suite_fingerprint(report.test_files)
    except Exception as exc:  # noqa: BLE001 - malformed suites must fail closed
        problems.append(
            "multifault: cannot fingerprint final hidden suite "
            f"({type(exc).__name__}: {exc})"
        )
    else:
        if evidence.suite_fingerprint != expected_suite:
            problems.append(
                "multifault: final-suite fingerprint does not match final hidden tests"
            )
    expected_p2p = report.pass_to_pass[0] if report.pass_to_pass else ""
    if not expected_p2p:
        problems.append("multifault: final P2P command is missing from OracleReport")
    elif evidence.p2p_command != expected_p2p:
        problems.append(
            "multifault: final P2P command does not match OracleReport pass_to_pass"
        )

    records = evidence.constituents
    indexes = [record.index for record in records]
    if indexes != list(range(len(indexes))):
        problems.append("multifault: constituent verdict indexes are incomplete")
    for record in records:
        if (
            record.verdict != "pass"
            or not record.other_inverse_patches_applied
            or not record.p2p_passed
            or not record.failed_f2p_test_ids
        ):
            problems.append(
                f"multifault: constituent {record.index} does not prove "
                "P2P-green/F2P-failing leave-one-broken behavior"
            )

    if candidate is not None:
        try:
            constituents = normalize_constituent_inverse_patches(candidate)
        except MultiFaultError as exc:
            problems.append(f"multifault: candidate metadata invalid ({exc})")
        else:
            if len(records) != len(constituents):
                problems.append(
                    "multifault: evidence does not cover every candidate constituent"
                )
            for expected, actual in zip(constituents, records, strict=False):
                if (
                    expected.index != actual.index
                    or expected.file != actual.file
                    or expected.inverse_patch_sha256 != actual.inverse_patch_sha256
                    or set(actual.repaired_indices)
                    != {
                        item.index
                        for item in constituents
                        if item.index != expected.index
                    }
                ):
                    problems.append(
                        f"multifault: evidence record {expected.index} does not "
                        "match the candidate's indexed constituent inverse patch"
                    )
    return problems


__all__ = [
    "MULTIFAULT_GENERATORS",
    "REASON_METADATA",
    "REASON_PARTIAL_ACCEPTED",
    "REASON_PARTIAL_APPLY",
    "REASON_PARTIAL_P2P",
    "ConstituentInversePatch",
    "ConstituentVerdict",
    "DockerPartialRepairRunner",
    "MultiFaultCompletenessEvidence",
    "MultiFaultError",
    "MultiFaultOutcome",
    "PartialRepairRunner",
    "PartialRepairScore",
    "assess_multifault_completeness",
    "build_multifault_report",
    "constituent_metadata_fingerprint",
    "normalize_constituent_inverse_patches",
    "run_multifault_completeness_gate",
    "verify_multifault_evidence",
]
