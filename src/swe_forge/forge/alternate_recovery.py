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
import stat
import tempfile
import unicodedata
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
from swe_forge.forge.recovery_authority import (
    RecoveryAttemptAuthority,
    RecoveryAuthorityError,
    RecoveryAuthorityRecord,
    default_authority_root,
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

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_RECOVERY_OUTPUT = _PROJECT_ROOT / "results" / "pilot_final"
_LEGACY_RECOVERY_WORK_ROOT = _PROJECT_ROOT / "results" / "final_alternate_recovery"
_LEGACY_TOMBSTONE_GENERATION_ID = "eb215983bd91465cbc07dce80cf5e015"
_APPROVED_MANIFEST_RESOURCE = Path(__file__).with_name(
    "approved_alternate_recovery_inputs.json"
)
# This pins the semantic manifest digest outside the retained workspace.  Editing
# the retained bytes cannot amend the approval, and editing the manifest resource
# without the reviewed source change below fails closed.
APPROVED_INPUT_MANIFEST_DIGEST = (
    "a3130d75dfd2b4fe37d9ecc785ed837cf3e4ffd0945fbc227ca49ca28a4a2e5a"
)


class AlternateRecoveryError(RuntimeError):
    """Raised when the sole allowed alternate-recovery path cannot continue."""


@dataclass(frozen=True)
class ApprovedTree:
    """Digest commitment to a canonical, complete retained tree."""

    path: str
    file_count: int
    tree_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "file_count": self.file_count,
            "tree_sha256": self.tree_sha256,
        }


@dataclass(frozen=True)
class ApprovedInputManifest:
    """Separately reviewed commitment to every recovery-consumed byte."""

    manifest_id: str
    task_id: str
    workspace_relative: str
    budget_relative: str
    budget_sha256: str
    workspace_files: dict[str, str]
    hidden_tests: ApprovedTree
    repository: ApprovedTree
    schema_version: int = 1

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "task_id": self.task_id,
            "workspace_relative": self.workspace_relative,
            "budget_relative": self.budget_relative,
            "budget_sha256": self.budget_sha256,
            "workspace_files": dict(sorted(self.workspace_files.items())),
            "hidden_tests": self.hidden_tests.to_dict(),
            "repository": self.repository.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ApprovedInputManifest":
        if not isinstance(payload, dict):
            raise AlternateRecoveryError("approved manifest must be an object")

        def text(name: str) -> str:
            value = payload.get(name)
            if not isinstance(value, str) or not value:
                raise AlternateRecoveryError(f"approved manifest lacks {name}")
            return value

        def tree(name: str) -> ApprovedTree:
            value = payload.get(name)
            if not isinstance(value, dict):
                raise AlternateRecoveryError(f"approved manifest lacks {name} tree")
            path = value.get("path")
            count = value.get("file_count")
            digest = value.get("tree_sha256")
            if (
                not isinstance(path, str)
                or not isinstance(count, int)
                or count < 0
                or not isinstance(digest, str)
                or not _is_sha256(digest)
            ):
                raise AlternateRecoveryError(
                    f"approved manifest has invalid {name} tree"
                )
            return ApprovedTree(path=path, file_count=count, tree_sha256=digest)

        workspace_files = payload.get("workspace_files")
        if not isinstance(workspace_files, dict) or not workspace_files:
            raise AlternateRecoveryError("approved manifest lacks workspace files")
        files: dict[str, str] = {}
        for path, digest in workspace_files.items():
            if not isinstance(path, str) or not _is_sha256(digest):
                raise AlternateRecoveryError(
                    "approved manifest has invalid file digest"
                )
            _validate_relative_path(path, label="approved manifest path")
            files[path] = digest
        schema = payload.get("schema_version")
        if schema != 1:
            raise AlternateRecoveryError("approved manifest schema is unsupported")
        budget_digest = text("budget_sha256")
        if not _is_sha256(budget_digest):
            raise AlternateRecoveryError("approved manifest has invalid budget digest")
        manifest = cls(
            manifest_id=text("manifest_id"),
            task_id=text("task_id"),
            workspace_relative=text("workspace_relative"),
            budget_relative=text("budget_relative"),
            budget_sha256=budget_digest,
            workspace_files=files,
            hidden_tests=tree("hidden_tests"),
            repository=tree("repository"),
            schema_version=schema,
        )
        _validate_relative_path(manifest.workspace_relative, label="workspace path")
        _validate_relative_path(manifest.budget_relative, label="budget path")
        for committed in (manifest.hidden_tests.path, manifest.repository.path):
            _validate_relative_path(committed, label="approved tree path")
        if set(manifest.workspace_files) != {
            "workspace.yaml",
            "patch.diff",
            "deletion_patch.diff",
            "provenance.json",
        }:
            raise AlternateRecoveryError(
                "approved manifest must enumerate the four retained root artifacts"
            )
        if manifest.hidden_tests.path != "tests" or manifest.repository.path != "repo":
            raise AlternateRecoveryError(
                "approved manifest must enumerate hidden tests and repository roots"
            )
        return manifest


@dataclass(frozen=True)
class VerifiedRecoveryInputs:
    """A private snapshot that is safe to consume after manifest verification."""

    manifest_id: str
    manifest_digest: str
    snapshot_root: Path
    budget_bytes: bytes
    workspace_digests: dict[str, str]

    def audit_evidence(self) -> dict[str, object]:
        """Return the only recovery-input evidence safe for publication."""

        return {
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "verified_input_digests": dict(sorted(self.workspace_digests.items())),
        }

    def cleanup(self) -> None:
        shutil.rmtree(self.snapshot_root, ignore_errors=True)


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_relative_path(path: str, *, label: str) -> None:
    candidate = Path(path)
    if (
        not path
        or candidate.is_absolute()
        or "\\" in path
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.as_posix() != path
    ):
        raise AlternateRecoveryError(f"{label} is not a safe canonical relative path")


def _strict_absolute_path(value: Path | str, *, label: str) -> Path:
    raw = Path(value)
    if any(part in (".", "..") for part in raw.parts):
        raise AlternateRecoveryError(f"{label} path alias is not allowed")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    if not absolute.is_absolute():
        raise AlternateRecoveryError(f"{label} path must be absolute")
    return absolute


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    """Open a directory only after rejecting every symlinked path component."""

    if not path.is_absolute():
        raise AlternateRecoveryError(f"{label} path must be absolute")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            if component in ("", ".", ".."):
                raise AlternateRecoveryError(f"{label} has a traversal component")
            try:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise AlternateRecoveryError(f"{label} directory is missing") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise AlternateRecoveryError(f"{label} contains a symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise AlternateRecoveryError(f"{label} is not a directory")
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_at(
    directory_fd: int, name: str, *, label: str
) -> tuple[bytes, tuple[int, int]]:
    """Read one regular file from a pinned directory descriptor, never following."""

    _validate_relative_path(name, label=label)
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise AlternateRecoveryError(
            f"approved manifest input is missing: {label}"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        raise AlternateRecoveryError(f"approved manifest input is a symlink: {label}")
    if not stat.S_ISREG(before.st_mode):
        raise AlternateRecoveryError(
            f"approved manifest input is not a regular file: {label}"
        )
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as exc:
        raise AlternateRecoveryError(
            f"approved manifest input changed: {label}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise AlternateRecoveryError(
                f"approved manifest input target swapped: {label}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), (opened.st_dev, opened.st_ino)
    finally:
        os.close(descriptor)


def _read_tree_at(
    directory_fd: int,
    *,
    label: str,
    ignored_top_level: frozenset[str] = frozenset(),
) -> dict[str, bytes]:
    """Canonical, symlink-safe walk over every consumed file in a retained tree."""

    entries: dict[str, bytes] = {}

    def walk(current_fd: int, prefix: str, *, is_root: bool = False) -> None:
        try:
            names = sorted(os.listdir(current_fd))
        except OSError as exc:
            raise AlternateRecoveryError(
                f"approved manifest tree unreadable: {label}"
            ) from exc
        canonical_names: set[str] = set()
        for name in names:
            if is_root and name in ignored_top_level:
                continue
            normalized = unicodedata.normalize("NFC", name)
            alias_key = normalized.casefold()
            if name != normalized or alias_key in canonical_names:
                raise AlternateRecoveryError(
                    f"approved manifest path alias detected in {label}: {name!r}"
                )
            canonical_names.add(alias_key)
            _validate_relative_path(name, label=f"{label} entry")
            relative = f"{prefix}/{name}" if prefix else name
            try:
                metadata = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            except OSError as exc:
                raise AlternateRecoveryError(
                    f"approved manifest tree entry disappeared: {relative}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise AlternateRecoveryError(
                    f"approved manifest tree input is a symlink: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current_fd,
                )
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                content, _identity = _read_regular_at(
                    current_fd, name, label=f"{label}/{relative}"
                )
                entries[relative] = content
            else:
                raise AlternateRecoveryError(
                    f"approved manifest tree input is not a regular file: {relative}"
                )

    walk(directory_fd, "", is_root=True)
    return entries


def _canonical_tree_digest(entries: dict[str, bytes]) -> str:
    return hashlib.sha256(
        b"".join(
            path.encode("utf-8")
            + b"\0"
            + hashlib.sha256(content).hexdigest().encode("ascii")
            + b"\n"
            for path, content in sorted(entries.items())
        )
    ).hexdigest()


def _verified_tree_digests(
    entries: dict[str, bytes], tree: ApprovedTree, *, label: str
) -> str:
    if len(entries) != tree.file_count:
        raise AlternateRecoveryError(
            f"approved manifest {label} tree has missing or extra inputs"
        )
    digest = _canonical_tree_digest(entries)
    if digest != tree.tree_sha256:
        raise AlternateRecoveryError(f"approved manifest {label} tree digest mismatch")
    return digest


def _load_approved_input_manifest() -> ApprovedInputManifest:
    try:
        raw = _APPROVED_MANIFEST_RESOURCE.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise AlternateRecoveryError("approved input manifest is unavailable") from exc
    manifest = ApprovedInputManifest.from_dict(payload)
    if manifest.digest != APPROVED_INPUT_MANIFEST_DIGEST:
        raise AlternateRecoveryError("approved input manifest digest is not pinned")
    if manifest.task_id != ALTERNATE_RECOVERY_TASK_ID:
        raise AlternateRecoveryError("approved input manifest task identity is invalid")
    return manifest


APPROVED_INPUT_MANIFEST = _load_approved_input_manifest()


def _read_approved_manifest_material(
    workspace: Path,
    budget: Path,
    repository_root: Path,
    manifest: ApprovedInputManifest,
) -> tuple[dict[str, bytes], bytes]:
    """Read every approved byte through no-follow descriptors, or fail closed."""

    expected_workspace = repository_root / manifest.workspace_relative
    expected_budget = repository_root / manifest.budget_relative
    if workspace != expected_workspace or budget != expected_budget:
        raise AlternateRecoveryError(
            "alternate recovery accepts only the exact approved input paths"
        )
    workspace_fd = _open_directory_nofollow(workspace, label="retained workspace")
    budget_parent_fd = _open_directory_nofollow(
        budget.parent, label="retained budget parent"
    )
    try:
        root_names = set(os.listdir(workspace_fd))
        allowed_root_names = {
            *manifest.workspace_files,
            manifest.hidden_tests.path,
            manifest.repository.path,
            "evaluate.sh",  # Retained but never recovery-consumed.
        }
        if root_names - allowed_root_names:
            raise AlternateRecoveryError(
                "approved manifest workspace has extra recovery inputs"
            )
        if allowed_root_names - root_names - {"evaluate.sh"}:
            raise AlternateRecoveryError(
                "approved manifest workspace has missing recovery inputs"
            )
        files: dict[str, bytes] = {}
        for name, approved_digest in sorted(manifest.workspace_files.items()):
            content, _identity = _read_regular_at(
                workspace_fd, name, label=f"workspace/{name}"
            )
            actual_digest = hashlib.sha256(content).hexdigest()
            if actual_digest != approved_digest:
                raise AlternateRecoveryError(
                    f"approved manifest digest mismatch for workspace/{name}"
                )
            files[name] = content
        for tree, label, ignored in (
            (manifest.hidden_tests, "hidden tests", frozenset()),
            (manifest.repository, "repository", frozenset({".git"})),
        ):
            try:
                metadata = os.stat(
                    tree.path, dir_fd=workspace_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise AlternateRecoveryError(
                    f"approved manifest {label} root is missing"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise AlternateRecoveryError(
                    f"approved manifest {label} root is a symlink"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise AlternateRecoveryError(
                    f"approved manifest {label} root is not a directory"
                )
            tree_fd = os.open(
                tree.path,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=workspace_fd,
            )
            try:
                entries = _read_tree_at(tree_fd, label=label, ignored_top_level=ignored)
            finally:
                os.close(tree_fd)
            _verified_tree_digests(entries, tree, label=label)
            for relative, content in entries.items():
                files[f"{tree.path}/{relative}"] = content
        budget_bytes, _identity = _read_regular_at(
            budget_parent_fd, budget.name, label="budget progress"
        )
        if hashlib.sha256(budget_bytes).hexdigest() != manifest.budget_sha256:
            raise AlternateRecoveryError("approved manifest budget digest mismatch")
        return files, budget_bytes
    finally:
        os.close(budget_parent_fd)
        os.close(workspace_fd)


def _snapshot_approved_material(
    workspace_files: dict[str, bytes], *, manifest: ApprovedInputManifest
) -> Path:
    snapshot = Path(tempfile.mkdtemp(prefix="swe-forge-approved-recovery-"))
    try:
        for relative, content in workspace_files.items():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return snapshot
    except Exception:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def _verify_approved_recovery_inputs(
    source_workspace: Path | str,
    budget_progress: Path | str,
    *,
    repository_root: Path | str = _PROJECT_ROOT,
    manifest: ApprovedInputManifest,
) -> VerifiedRecoveryInputs:
    """Pin, fully enumerate, and snapshot recovery inputs before any side effect."""

    root = _strict_absolute_path(repository_root, label="repository root")
    workspace = _strict_absolute_path(source_workspace, label="source workspace")
    budget = _strict_absolute_path(budget_progress, label="budget progress")
    first_files, first_budget = _read_approved_manifest_material(
        workspace, budget, root, manifest
    )
    # Re-read the exact descriptor-contained path set before making a snapshot.
    # A swap between passes is rejected, and the later pipeline uses only snapshot
    # bytes, never the mutable retained workspace.
    second_files, second_budget = _read_approved_manifest_material(
        workspace, budget, root, manifest
    )
    if first_files != second_files or first_budget != second_budget:
        raise AlternateRecoveryError(
            "approved manifest inputs changed during verification"
        )
    snapshot = _snapshot_approved_material(first_files, manifest=manifest)
    digests = dict(manifest.workspace_files)
    digests[manifest.hidden_tests.path] = manifest.hidden_tests.tree_sha256
    digests[manifest.repository.path] = manifest.repository.tree_sha256
    digests["budget"] = manifest.budget_sha256
    return VerifiedRecoveryInputs(
        manifest_id=manifest.manifest_id,
        manifest_digest=manifest.digest,
        snapshot_root=snapshot,
        budget_bytes=first_budget,
        workspace_digests=digests,
    )


def verify_approved_recovery_inputs(
    source_workspace: Path | str,
    budget_progress: Path | str,
    *,
    repository_root: Path | str = _PROJECT_ROOT,
) -> VerifiedRecoveryInputs:
    """Verify only the pinned, code-reviewed alternate-recovery manifest."""

    return _verify_approved_recovery_inputs(
        source_workspace,
        budget_progress,
        repository_root=repository_root,
        manifest=APPROVED_INPUT_MANIFEST,
    )


def _verify_original_budget_bytes(raw: bytes) -> VerifiedOriginalBudget:
    """Verify original-$1400 accounting from already-approved immutable bytes.

    The alternate path accepts the terminal harvest ledger only when it reports
    the original ceiling exactly, has no active reservation or in-flight batch,
    and its accounted spend is non-negative and no greater than that ceiling.
    """

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AlternateRecoveryError(
            "cannot durably verify original mission budget"
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
    )


def verify_original_budget(progress_path: Path | str) -> VerifiedOriginalBudget:
    """Verify an accounting file for standalone inspection.

    Recovery execution itself uses :func:`verify_approved_recovery_inputs` and
    passes only its verified in-memory snapshot to ``_verify_original_budget_bytes``.
    """

    try:
        raw = Path(progress_path).read_bytes()
    except OSError as exc:
        raise AlternateRecoveryError(
            "cannot durably verify original mission budget"
        ) from exc
    return _verify_original_budget_bytes(raw)


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
    def tombstone(
        cls, *, run_id: str, reason: str, previous_generation_id: str = ""
    ) -> "RecoveryCertification":
        return cls(
            run_id=run_id,
            state="tombstone",
            passed=False,
            task_ids=(),
            created_at=_utc_now(),
            previous_generation_id=previous_generation_id,
            reason=reason,
        )

    @classmethod
    def keep(
        cls, *, run_id: str, task_id: str, previous_generation_id: str = ""
    ) -> "RecoveryCertification":
        return cls(
            run_id=run_id,
            state="keep",
            passed=True,
            task_ids=(task_id,),
            created_at=_utc_now(),
            previous_generation_id=previous_generation_id,
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


def _require_canonical_recovery_output(out_dir: Path | str) -> Path:
    """Refuse output aliases so the global authority has one publication target."""
    output = _strict_absolute_path(out_dir, label="recovery output")
    if output != _CANONICAL_RECOVERY_OUTPUT:
        raise AlternateRecoveryError(
            "alternate recovery accepts only the canonical pilot_final output path"
        )
    return output


def _terminal_ledger_run_id(path: Path, *, empty_run_id: str = "") -> str:
    """Read only the run identity from a terminal ledger without trusting content."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AlternateRecoveryError("terminal recovery ledger is unavailable") from exc
    if not lines:
        if empty_run_id:
            return empty_run_id
        raise AlternateRecoveryError("terminal recovery ledger is empty")
    run_ids: set[str] = set()
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AlternateRecoveryError(
                "terminal recovery ledger is malformed"
            ) from exc
        run_id = event.get("run_id") if isinstance(event, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise AlternateRecoveryError("terminal recovery ledger lacks a run id")
        run_ids.add(run_id)
    if len(run_ids) != 1:
        raise AlternateRecoveryError("terminal recovery ledger has mixed run ids")
    return next(iter(run_ids))


def _migrate_existing_terminal_tombstone(
    authority: RecoveryAttemptAuthority, output: Path
) -> bool:
    """Record the one pre-authorized terminal tombstone before any live activity."""
    selected = load_published_generation(output)
    if selected is None or selected.generation_id != _LEGACY_TOMBSTONE_GENERATION_ID:
        return False
    certification_path = selected.root / "certification.json"
    try:
        certification = json.loads(certification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlternateRecoveryError(
            "legacy terminal certification is unavailable"
        ) from exc
    if (
        not isinstance(certification, dict)
        or certification.get("state") != "tombstone"
        or certification.get("passed") is not False
        or certification.get("task_ids") != []
        or not isinstance(certification.get("run_id"), str)
        or not certification["run_id"]
    ):
        raise AlternateRecoveryError("legacy terminal certification is invalid")
    ledger = selected.root / "recovery-ledger.jsonl"
    if not ledger.is_file():
        ledger = _LEGACY_RECOVERY_WORK_ROOT / "recovery-ledger.jsonl"
    ledger_run_id = _terminal_ledger_run_id(ledger)
    try:
        authority.migrate_terminal_tombstone(
            run_id=certification["run_id"],
            ledger_run_id=ledger_run_id,
            certification_run_id=certification["run_id"],
            selected_generation_id=selected.generation_id,
            ledger_path=ledger.resolve(strict=True),
        )
    except RecoveryAuthorityError as exc:
        raise AlternateRecoveryError(str(exc)) from exc
    return True


def _consume_terminal_authority(
    authority: RecoveryAttemptAuthority,
    claim: RecoveryAuthorityRecord,
    *,
    terminal_state: Literal["keep", "tombstone"],
    output: Path,
) -> None:
    """Require selected certification and ledger evidence before consuming a claim."""
    selected = load_published_generation(output)
    if selected is None:
        raise AlternateRecoveryError("terminal recovery publication is not selected")
    certification_path = selected.root / "certification.json"
    ledger_path = selected.root / "recovery-ledger.jsonl"
    try:
        certification = json.loads(certification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlternateRecoveryError(
            "terminal recovery certification is unavailable"
        ) from exc
    if (
        not isinstance(certification, dict)
        or certification.get("run_id") != claim.run_id
        or certification.get("state") != terminal_state
        or certification.get("previous_generation_id")
        != claim.expected_current_generation_id
    ):
        raise AlternateRecoveryError(
            "terminal certification does not reconcile to the authority claim"
        )
    try:
        authority.consume(
            claim,
            terminal_state=terminal_state,
            certification_run_id=certification["run_id"],
            ledger_run_id=_terminal_ledger_run_id(
                ledger_path, empty_run_id=claim.run_id
            ),
            selected_generation_id=selected.generation_id,
        )
    except RecoveryAuthorityError as exc:
        raise AlternateRecoveryError(str(exc)) from exc


def _reconcile_selected_terminal(
    authority: RecoveryAttemptAuthority,
    claim: RecoveryAuthorityRecord,
    *,
    output: Path,
) -> AlternateRecoveryResult | None:
    """Consume a claim when a matching terminal publication already selected."""
    selected = load_published_generation(output)
    if selected is None:
        return None
    try:
        certification = json.loads(
            (selected.root / "certification.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(certification, dict):
        return None
    terminal_state = certification.get("state")
    if (
        terminal_state not in {"keep", "tombstone"}
        or certification.get("run_id") != claim.run_id
        or certification.get("previous_generation_id")
        != claim.expected_current_generation_id
    ):
        return None
    _consume_terminal_authority(
        authority,
        claim,
        terminal_state=terminal_state,
        output=output,
    )
    return AlternateRecoveryResult(
        run_id=claim.run_id,
        status="kept" if terminal_state == "keep" else "tombstoned",
        reason=(
            ""
            if terminal_state == "keep"
            else str(certification.get("reason", "terminal reconciliation"))
        ),
        task_id=ALTERNATE_RECOVERY_TASK_ID if terminal_state == "keep" else "",
    )


def _reconcile_claim_to_tombstone(
    authority: RecoveryAttemptAuthority,
    claim: RecoveryAuthorityRecord,
    *,
    output: Path,
    cap_usd: str,
    reason: str,
) -> AlternateRecoveryResult:
    """Terminally reconcile a crashed claim without Docker or teacher activity."""
    ledger_path = Path(claim.ledger_path)
    if not ledger_path.exists():
        RecoveryBudgetLedger(
            ledger_path,
            run_id=claim.run_id,
            cap_usd=cap_usd,
            worst_case_cost_usd="3.00",
        )
    tombstone = RecoveryCertification.tombstone(
        run_id=claim.run_id,
        reason=reason,
        previous_generation_id=claim.expected_current_generation_id,
    )
    _promote_regular_audit_artifacts_to_prior_generation(output)
    export_batch(
        [],
        output,
        overwrite=True,
        generation_metadata_writer=_tombstone_stage_writer(
            tombstone, reason, ledger_path=ledger_path
        ),
        extra_facade_artifacts=_CANONICAL_AUDIT_ARTIFACTS,
        expected_current_generation_id=claim.expected_current_generation_id,
    )
    _consume_terminal_authority(
        authority,
        claim,
        terminal_state="tombstone",
        output=output,
    )
    _write_json(
        ledger_path.parent / "result.json",
        {"run_id": claim.run_id, "status": "tombstoned", "reason": reason},
    )
    return AlternateRecoveryResult(
        run_id=claim.run_id, status="tombstoned", reason=reason
    )


@dataclass(frozen=True)
class RehydratedAlternate:
    """Immutable candidate input rebuilt from the retained audit workspace."""

    task_id: str
    manifest_id: str
    manifest_digest: str
    verified_input_digests: dict[str, str]
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


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlternateRecoveryError(f"retained alternate workspace lacks {name}")
    return value


def rehydrate_alternate(
    approved_inputs: VerifiedRecoveryInputs,
) -> RehydratedAlternate:
    """Rebuild a candidate exclusively from a verified private snapshot."""

    if not isinstance(approved_inputs, VerifiedRecoveryInputs):
        raise AlternateRecoveryError(
            "alternate recovery requires approved immutable recovery inputs"
        )
    root = approved_inputs.snapshot_root
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
                    "approved_manifest_id": approved_inputs.manifest_id,
                    "approved_manifest_digest": approved_inputs.manifest_digest,
                    "verified_input_digests": approved_inputs.workspace_digests,
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
        manifest_id=approved_inputs.manifest_id,
        manifest_digest=approved_inputs.manifest_digest,
        verified_input_digests=approved_inputs.workspace_digests,
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


def _tombstone_stage_writer(
    certification: RecoveryCertification, reason: str, *, ledger_path: Path
):
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
        (stage / "recovery-ledger.jsonl").write_bytes(ledger_path.read_bytes())

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
                "approved_manifest_id": alternate.manifest_id,
                "approved_manifest_digest": alternate.manifest_digest,
                "verified_input_digests": dict(alternate.verified_input_digests),
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
    out_dir: Path | str = _CANONICAL_RECOVERY_OUTPUT,
    source_workspace: Path | str = Path(
        "results/pilot_keeps/tasks/" + ALTERNATE_RECOVERY_TASK_ID
    ),
    budget_progress: Path | str = Path("results/pilot_keeps/harvest_progress.json"),
    work_root: Path | str = _LEGACY_RECOVERY_WORK_ROOT,
    repository_root: Path | str = _PROJECT_ROOT,
) -> AlternateRecoveryResult:
    """Execute exactly one no-retry recovery, keeping or tombstoning atomically."""

    # This preflight must finish before certificate, ledger, Docker, LLM, or
    # publication activity.  A manifest failure is deliberately propagated,
    # rather than tombstoned, because even a tombstone is a publication effect.
    approved_inputs = verify_approved_recovery_inputs(
        source_workspace,
        budget_progress,
        repository_root=repository_root,
    )
    verified_budget = _verify_original_budget_bytes(approved_inputs.budget_bytes)
    try:
        output = _require_canonical_recovery_output(out_dir)
        authority = RecoveryAttemptAuthority(
            default_authority_root(), ALTERNATE_RECOVERY_TASK_ID
        )
        if _migrate_existing_terminal_tombstone(authority, output):
            raise AlternateRecoveryError(
                "final alternate recovery is already consumed by the terminal tombstone"
            )
        existing_claim = authority.record()
        if existing_claim is not None:
            if existing_claim.state == "claimed":
                selected_terminal = _reconcile_selected_terminal(
                    authority,
                    existing_claim,
                    output=output,
                )
                if selected_terminal is not None:
                    return selected_terminal
                return _reconcile_claim_to_tombstone(
                    authority,
                    existing_claim,
                    output=output,
                    cap_usd=verified_budget.incremental_cap_usd,
                    reason=(
                        "AlternateRecoveryError: durable authority claim was "
                        "interrupted before terminal reconciliation"
                    ),
                )
            raise AlternateRecoveryError(
                "final alternate recovery is already consumed by a durable authority"
            )

        work = _strict_absolute_path(work_root, label="recovery work root")
        work.mkdir(parents=True, exist_ok=False)
        run_id = f"alternate-{uuid.uuid4().hex}"
        ledger_path = (work / "recovery-ledger.jsonl").resolve()
        previous = load_published_generation(output)
        try:
            claim = authority.claim(
                run_id=run_id,
                expected_current_generation_id=(
                    previous.generation_id if previous else ""
                ),
                ledger_path=ledger_path,
            )
        except RecoveryAuthorityError as exc:
            raise AlternateRecoveryError(str(exc)) from exc

        # The claim was fsynced before this certification, ledger, Docker, or
        # live teacher work. A crash at any later point leaves authority consumed.
        ledger = RecoveryBudgetLedger(
            ledger_path,
            run_id=run_id,
            cap_usd=verified_budget.incremental_cap_usd,
            worst_case_cost_usd="3.00",
        )
        pending = RecoveryCertification.pending(
            run_id=run_id,
            previous_generation_id=claim.expected_current_generation_id,
            task_id=ALTERNATE_RECOVERY_TASK_ID,
        )
        write_recovery_certification(output, pending)
        alternate = rehydrate_alternate(approved_inputs)
        _write_json(work / "budget-verification.json", verified_budget.__dict__)
        _write_json(
            work / "preflight.json",
            {
                "run_id": run_id,
                "task_id": alternate.task_id,
                "max_retries": 0,
                "k": 6,
                "band_high": 0.5,
                "discrimination_threshold": 1.0,
                **approved_inputs.audit_evidence(),
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
        keep = RecoveryCertification.keep(
            run_id=run_id,
            task_id=alternate.task_id,
            previous_generation_id=claim.expected_current_generation_id,
        )
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
            expected_current_generation_id=claim.expected_current_generation_id,
        )
        if len(export.kept) != 1 or len(export.refused) != 0:
            raise AlternateRecoveryError("alternate keep export did not reconcile")
        _consume_terminal_authority(
            authority,
            claim,
            terminal_state="keep",
            output=output,
        )
        _write_json(
            work / "result.json",
            {"run_id": run_id, "status": "kept", "task_id": alternate.task_id},
        )
        return AlternateRecoveryResult(
            run_id=run_id, status="kept", reason="", task_id=alternate.task_id
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        # Do not publish another terminal generation for validation, alias, or
        # already-consumed failures. Only a durable claim may be reconciled.
        claimed_record = locals().get("claim")
        if not isinstance(claimed_record, RecoveryAuthorityRecord):
            raise
        try:
            return _reconcile_claim_to_tombstone(
                authority,
                claimed_record,
                output=output,
                cap_usd=verified_budget.incremental_cap_usd,
                reason=reason,
            )
        except Exception as tombstone_error:
            raise AlternateRecoveryError(
                f"alternate recovery failed and tombstone publication failed: {tombstone_error}"
            ) from tombstone_error
    finally:
        approved_inputs.cleanup()


__all__ = [
    "ALTERNATE_RECOVERY_TASK_ID",
    "APPROVED_INPUT_MANIFEST",
    "APPROVED_INPUT_MANIFEST_DIGEST",
    "INCREMENTAL_RECOVERY_CAP_USD",
    "ORIGINAL_MISSION_BUDGET_USD",
    "UPDATE_WRAPPER_F2P_COMMAND",
    "UPDATE_WRAPPER_F2P_CONTENT",
    "UPDATE_WRAPPER_F2P_NODE",
    "UPDATE_WRAPPER_F2P_PATH",
    "ApprovedInputManifest",
    "ApprovedTree",
    "AlternateRecoveryError",
    "RecoveryCertification",
    "RehydratedAlternate",
    "AlternateRecoveryResult",
    "VerifiedOriginalBudget",
    "VerifiedRecoveryInputs",
    "rehydrate_alternate",
    "run_final_alternate_recovery",
    "run_normal_oracle_gates",
    "verify_original_budget",
    "verify_approved_recovery_inputs",
    "verify_unfiltered_public_gold",
    "write_recovery_certification",
]
