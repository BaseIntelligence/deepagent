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

import hashlib
import json
import re
from pathlib import PurePosixPath
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
    reconstruct_suite_tests,
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
_ALT_CORRECT_AUDIT_VERSION = 2
_ALT_CORRECT_PUBLIC_SUMMARY_KEYS = frozenset(
    {
        "public_suite_sha256",
        "gold_public_suite_passed",
        "public_valid_alternatives",
        "invalid_teacher_proposals",
        "pre_relax_suite_fingerprint",
        "final_suite_fingerprint",
    }
)
_ALT_CORRECT_AUDIT_KEYS = frozenset(
    {
        "version",
        "original_public_suite_sha256",
        "pre_relax_suite",
        "final_suite",
        "gold",
        "alternatives",
    }
)
_ALT_CORRECT_SUITE_RECEIPT_KEYS = frozenset(
    {
        "identities",
        "identity_count",
        "identity_sha256",
        "suite_fingerprint",
        "files",
    }
)
_ALT_CORRECT_SUITE_FILE_KEYS = frozenset({"path", "content_sha256"})
_ALT_CORRECT_GOLD_RECORD_KEYS = frozenset({"public", "filtered_p2p", "hidden"})
_ALT_CORRECT_ALTERNATIVE_RECORD_KEYS = frozenset(
    {"proposal_sha256", "patches", "public", "filtered_p2p", "hidden"}
)
_ALT_CORRECT_RED_ALTERNATIVE_RECORD_KEYS = frozenset(
    {"proposal_sha256", "patches", "public"}
)
_ALT_CORRECT_EXECUTION_KEYS = frozenset({"public", "filtered_p2p", "hidden"})
_ALT_CORRECT_STATUS_KEYS = frozenset({"passed", "exit_code"})
_ALT_CORRECT_HIDDEN_RESULT_KEYS = frozenset({"test_id", "exit_code"})
_ALT_CORRECT_PATCH_KEYS = frozenset({"path", "content"})
_AUDIT_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")

#: One bound gate: consumes the prior report (``None`` for the first gate) and
#: returns the extended report. The orchestration threads these uniformly.
GateStep = Callable[["OracleReport | None"], Awaitable[OracleReport]]


class OraclePipelineError(RuntimeError):
    """Raised for an unrecoverable failure while orchestrating the gates."""


class ExportRefusedError(RuntimeError):
    """Raised by :func:`ensure_oracle_exportable` for a non-shippable candidate."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_exit_code(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255


def parse_protected_alt_correct_audit(value: str) -> dict[str, object]:
    """Parse a private audit while rejecting JSON duplicate object members."""

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, item in pairs:
            if key in parsed:
                raise ValueError(
                    f"protected alt-correct audit contains duplicate object key {key!r}"
                )
            parsed[key] = item
        return parsed

    payload = json.loads(value, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("protected alt-correct audit must be an object")
    return payload


def _has_exact_keys(value: object, keys: frozenset[str]) -> bool:
    return isinstance(value, dict) and set(value) == keys


def _is_audit_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(_AUDIT_IDENTIFIER_RE.fullmatch(value))


def _is_audit_test_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and "\x00" not in value
        and "\n" not in value
        and "\r" not in value
    )


def _is_audit_patch_path(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.parts != (".",)


def _audit_status(value: object, label: str) -> tuple[bool, int] | str:
    if not _has_exact_keys(value, _ALT_CORRECT_STATUS_KEYS):
        return f"alt_correct: {label} is malformed"
    assert isinstance(value, dict)
    passed = value["passed"]
    exit_code = value["exit_code"]
    if not isinstance(passed, bool) or not _is_exit_code(exit_code):
        return f"alt_correct: {label} is malformed"
    assert isinstance(exit_code, int)
    if passed is not (exit_code == 0):
        return f"alt_correct: {label} is malformed"
    return passed, exit_code


def _audit_hidden_results(value: object, label: str) -> tuple[set[str], bool] | str:
    if not isinstance(value, list) or not value:
        return f"alt_correct: {label} is incomplete"
    identities: set[str] = set()
    all_green = True
    for result in value:
        if not _has_exact_keys(result, _ALT_CORRECT_HIDDEN_RESULT_KEYS):
            return f"alt_correct: {label} is malformed"
        assert isinstance(result, dict)
        test_id = result["test_id"]
        exit_code = result["exit_code"]
        if not _is_audit_test_id(test_id) or not _is_exit_code(exit_code):
            return f"alt_correct: {label} is malformed"
        assert isinstance(test_id, str) and isinstance(exit_code, int)
        if test_id in identities:
            return "alt_correct: duplicate hidden test identity"
        identities.add(test_id)
        all_green = all_green and exit_code == 0
    return identities, all_green


def _audit_patches(value: object) -> str | None:
    if not isinstance(value, list) or not value:
        return "alt_correct: protected alternative audit is incomplete"
    paths: set[str] = set()
    for patch in value:
        if not _has_exact_keys(patch, _ALT_CORRECT_PATCH_KEYS):
            return "alt_correct: alternative patch is malformed"
        assert isinstance(patch, dict)
        path = patch["path"]
        content = patch["content"]
        if not _is_audit_patch_path(path) or not isinstance(content, str):
            return "alt_correct: alternative patch is malformed"
        assert isinstance(path, str)
        if path in paths:
            return "alt_correct: duplicate patch path"
        paths.add(path)
    return None


def _expected_hidden_test_ids(report: OracleReport) -> set[str] | None:
    """Rebuild the exact hidden-test identities recorded by the alt gate."""
    try:
        adapter = build_default_registry().get(report.language)
        hidden_tests = reconstruct_suite_tests(
            adapter, report.fail_to_pass, report.test_files
        )
    except Exception:  # noqa: BLE001 - malformed report evidence is non-exportable
        return None
    expected = {test.test_id for test in hidden_tests}
    if len(expected) != len(hidden_tests):
        return None
    for test_id in expected:
        if not _is_audit_test_id(test_id):
            return None
    return expected or None


def _hidden_identity_digest(identities: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{identity}\n" for identity in sorted(identities)).encode("utf-8")
    ).hexdigest()


def _suite_file_manifest(
    report: OracleReport,
) -> tuple[tuple[str, str], ...] | None:
    files: list[tuple[str, str]] = []
    paths: set[str] = set()
    for test_file in report.test_files:
        if not test_file.content:
            continue
        if not _is_audit_patch_path(test_file.path) or test_file.path in paths:
            return None
        paths.add(test_file.path)
        content = (
            test_file.content
            if test_file.content.endswith("\n")
            else test_file.content + "\n"
        )
        files.append(
            (test_file.path, hashlib.sha256(content.encode("utf-8")).hexdigest())
        )
    return tuple(sorted(files))


def _suite_fingerprint_from_manifest(files: Sequence[tuple[str, str]]) -> str:
    canonical = json.dumps(
        [
            {"path": path, "content_sha256": content_sha256}
            for path, content_sha256 in sorted(files)
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _audit_suite_receipt(
    value: object, label: str
) -> tuple[tuple[str, ...], str, tuple[tuple[str, str], ...]] | str:
    """Validate a receipt binding hidden-test identities to a concrete suite."""
    if not _has_exact_keys(value, _ALT_CORRECT_SUITE_RECEIPT_KEYS):
        return f"alt_correct: {label} suite receipt has an unsafe schema"
    assert isinstance(value, dict)
    identities_value = value["identities"]
    if not isinstance(identities_value, list) or not identities_value:
        return f"alt_correct: {label} suite identities are incomplete"
    if any(not _is_audit_test_id(identity) for identity in identities_value):
        return f"alt_correct: {label} suite identities are malformed"
    identities = tuple(identities_value)
    if len(set(identities)) != len(identities):
        return f"alt_correct: {label} suite identities are duplicated"
    if identities != tuple(sorted(identities)):
        return f"alt_correct: {label} suite identities are not canonical"
    identity_count = value["identity_count"]
    if (
        not isinstance(identity_count, int)
        or isinstance(identity_count, bool)
        or identity_count != len(identities)
    ):
        return f"alt_correct: {label} suite identity count is inconsistent"
    identity_sha256 = value["identity_sha256"]
    if not _is_sha256(identity_sha256) or identity_sha256 != _hidden_identity_digest(
        identities
    ):
        return f"alt_correct: {label} suite identity digest is inconsistent"
    suite_fingerprint = value["suite_fingerprint"]
    if not _is_sha256(suite_fingerprint):
        return f"alt_correct: {label} suite fingerprint is malformed"
    files_value = value["files"]
    if not isinstance(files_value, list) or not files_value:
        return f"alt_correct: {label} suite file manifest is incomplete"
    files: list[tuple[str, str]] = []
    paths: set[str] = set()
    for item in files_value:
        if not _has_exact_keys(item, _ALT_CORRECT_SUITE_FILE_KEYS):
            return f"alt_correct: {label} suite file manifest is malformed"
        assert isinstance(item, dict)
        path = item["path"]
        content_sha256 = item["content_sha256"]
        if (
            not _is_audit_patch_path(path)
            or not _is_sha256(content_sha256)
            or path in paths
        ):
            return f"alt_correct: {label} suite file manifest is malformed"
        assert isinstance(path, str) and isinstance(content_sha256, str)
        paths.add(path)
        files.append((path, content_sha256))
    manifest = tuple(sorted(files))
    if suite_fingerprint != _suite_fingerprint_from_manifest(manifest):
        return f"alt_correct: {label} suite fingerprint does not match file manifest"
    return identities, suite_fingerprint, manifest


def _audit_execution_record(
    value: object, label: str
) -> tuple[bool, bool, set[str], bool] | str:
    if not _has_exact_keys(value, _ALT_CORRECT_EXECUTION_KEYS):
        return f"alt_correct: {label} has an unsafe schema"
    assert isinstance(value, dict)
    public = _audit_status(value["public"], f"{label} public result")
    if isinstance(public, str):
        return public
    filtered = _audit_status(value["filtered_p2p"], f"{label} filtered P2P result")
    if isinstance(filtered, str):
        return filtered
    hidden = _audit_hidden_results(value["hidden"], f"{label} hidden result")
    if isinstance(hidden, str):
        return hidden
    public_passed, _ = public
    filtered_passed, _ = filtered
    hidden_ids, hidden_green = hidden
    return public_passed, filtered_passed, hidden_ids, hidden_green


def _patches_digest(patches: Sequence[dict[str, object]]) -> str:
    """Recompute the canonical digest recorded for a materialized proposal."""
    digest = hashlib.sha256()
    for patch in sorted(patches, key=lambda item: str(item["path"])):
        digest.update(str(patch["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(patch["content"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _alt_correct_public_validity_issues(report: OracleReport) -> list[str]:
    """Validate the non-agent-facing evidence for alt-correct public validity."""
    # Historical artifacts remain readable as immutable recertification inputs,
    # but this validator is an export boundary: a passing report that claims
    # alternative correctness can never use the old missing-evidence shape to
    # authorize public publication.
    if not report.alt_correct_accepted:
        return []

    public_details = report.details.get("alt_correct")
    if not isinstance(public_details, dict):
        return ["alt_correct: source-free public validity summary is missing"]
    if set(public_details) != _ALT_CORRECT_PUBLIC_SUMMARY_KEYS:
        return ["alt_correct: public validity summary has an unsafe schema"]

    audit = report.protected_alt_correct_audit
    if not isinstance(audit, dict):
        return ["alt_correct: protected public-validity audit is missing"]
    if set(audit) != _ALT_CORRECT_AUDIT_KEYS:
        return ["alt_correct: protected public-validity audit has an unsafe schema"]
    version = audit["version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != _ALT_CORRECT_AUDIT_VERSION
    ):
        return ["alt_correct: protected public-validity audit version is unsupported"]
    audit_digest = audit["original_public_suite_sha256"]
    if not _is_sha256(audit_digest):
        return ["alt_correct: protected public suite digest is malformed"]
    summary_digest = public_details.get("public_suite_sha256")
    if not _is_sha256(summary_digest):
        return ["alt_correct: public suite digest is missing or malformed"]
    if summary_digest != audit_digest:
        return ["alt_correct: public suite digest does not match protected audit"]
    pre_relax_suite = _audit_suite_receipt(audit["pre_relax_suite"], "pre-relax")
    if isinstance(pre_relax_suite, str):
        return [pre_relax_suite]
    final_suite = _audit_suite_receipt(audit["final_suite"], "final")
    if isinstance(final_suite, str):
        return [final_suite]
    (
        pre_relax_hidden_ids,
        pre_relax_fingerprint,
        pre_relax_files,
    ) = pre_relax_suite
    final_hidden_ids, final_fingerprint, final_files = final_suite
    if public_details["pre_relax_suite_fingerprint"] != pre_relax_fingerprint:
        return [
            "alt_correct: public pre-relax suite fingerprint does not match "
            "protected audit"
        ]
    if public_details["final_suite_fingerprint"] != final_fingerprint:
        return [
            "alt_correct: public final suite fingerprint does not match protected audit"
        ]
    expected_final_hidden_ids = _expected_hidden_test_ids(report)
    expected_final_files = _suite_file_manifest(report)
    if expected_final_hidden_ids is None or final_hidden_ids != tuple(
        sorted(expected_final_hidden_ids)
    ):
        return ["alt_correct: final suite identities do not match final hidden suite"]
    if expected_final_files is None or final_files != expected_final_files:
        return [
            "alt_correct: final suite file manifest does not match final hidden suite"
        ]
    try:
        expected_final_fingerprint = final_suite_fingerprint(report.test_files)
    except Exception:  # noqa: BLE001 - malformed suite evidence is non-exportable
        return ["alt_correct: final suite fingerprint cannot be reconciled"]
    if final_fingerprint != expected_final_fingerprint:
        return [
            "alt_correct: final suite fingerprint does not match final hidden suite"
        ]
    gold = audit["gold"]
    if not isinstance(gold, dict) or set(gold) not in (
        _ALT_CORRECT_GOLD_RECORD_KEYS,
        _ALT_CORRECT_GOLD_RECORD_KEYS | {"relaxed"},
    ):
        return ["alt_correct: protected gold record has an unsafe schema"]
    assert isinstance(gold, dict)
    gold_evidence = _audit_execution_record(
        {key: gold[key] for key in _ALT_CORRECT_EXECUTION_KEYS}, "gold"
    )
    if isinstance(gold_evidence, str):
        return [gold_evidence]
    gold_passed, gold_p2p_passed, gold_hidden_ids, gold_hidden_green = gold_evidence
    if gold_passed is not True:
        return ["alt_correct: gold did not pass the original public suite"]
    if not gold_p2p_passed:
        return ["alt_correct: gold filtered P2P result is not green"]
    if not gold_hidden_green:
        return ["alt_correct: gold hidden result is not green"]
    if gold_hidden_ids != set(pre_relax_hidden_ids):
        return ["alt_correct: gold hidden test identities do not match pre-relax suite"]
    effective_gold_hidden_ids = gold_hidden_ids
    relaxed_audit = "relaxed" in gold
    if not relaxed_audit and (
        pre_relax_hidden_ids != final_hidden_ids
        or pre_relax_fingerprint != final_fingerprint
        or pre_relax_files != final_files
    ):
        return [
            "alt_correct: non-relaxed audit pre-relax suite does not match final suite"
        ]
    if relaxed_audit:
        gold_relaxed = _audit_execution_record(gold["relaxed"], "relaxed gold")
        if isinstance(gold_relaxed, str):
            return [gold_relaxed]
        (
            gold_relaxed_public_passed,
            gold_relaxed_p2p_passed,
            effective_gold_hidden_ids,
            gold_relaxed_hidden_green,
        ) = gold_relaxed
        if not gold_relaxed_public_passed:
            return ["alt_correct: relaxed gold public result is not green"]
        if not gold_relaxed_p2p_passed:
            return ["alt_correct: relaxed gold filtered P2P result is not green"]
        if not gold_relaxed_hidden_green:
            return ["alt_correct: relaxed gold hidden result is not green"]
        if effective_gold_hidden_ids != set(final_hidden_ids):
            return [
                "alt_correct: relaxed gold hidden test identities do not match "
                "final suite"
            ]

    alternatives = audit["alternatives"]
    if not isinstance(alternatives, dict) or not alternatives:
        return ["alt_correct: protected alternative audit is missing"]

    public_green = 0
    invalid_alternatives: list[str] = []
    seen_alt_ids: set[str] = set()
    for alt_id, record in alternatives.items():
        if not _is_audit_identifier(alt_id):
            return ["alt_correct: alternative identity is malformed"]
        assert isinstance(alt_id, str)
        if alt_id in seen_alt_ids:
            return ["alt_correct: duplicate alternative identity"]
        seen_alt_ids.add(alt_id)
        if not isinstance(record, dict):
            return ["alt_correct: protected alternative audit is malformed"]
        public = _audit_status(record.get("public"), "alternative public result")
        if isinstance(public, str):
            return [public]
        public_passed, _ = public
        required_record_keys = (
            _ALT_CORRECT_ALTERNATIVE_RECORD_KEYS
            if public_passed
            else _ALT_CORRECT_RED_ALTERNATIVE_RECORD_KEYS
        )
        has_relaxed_record = "relaxed" in record
        if set(record) not in (
            required_record_keys,
            required_record_keys | {"relaxed"},
        ):
            return ["alt_correct: alternative record has an unsafe schema"]
        if not _is_sha256(record["proposal_sha256"]):
            return ["alt_correct: protected alternative audit is incomplete"]
        patches = record["patches"]
        patch_issue = _audit_patches(patches)
        if patch_issue:
            return [patch_issue]
        assert isinstance(patches, list)
        typed_patches = [patch for patch in patches if isinstance(patch, dict)]
        proposal_digest = record["proposal_sha256"]
        if proposal_digest != _patches_digest(typed_patches):
            return [
                "alt_correct: protected alternative proposal digest does not "
                "match materialized patches"
            ]
        if public_passed:
            initial = _audit_execution_record(
                {key: record[key] for key in _ALT_CORRECT_EXECUTION_KEYS},
                "alternative",
            )
            if isinstance(initial, str):
                return [initial]
            _, initial_p2p_passed, initial_hidden_ids, initial_hidden_green = initial
            if not initial_p2p_passed:
                return [
                    "alt_correct: public-green alternative has a non-green filtered P2P result"
                ]
            if not initial_hidden_green and not relaxed_audit:
                return [
                    "alt_correct: public-green alternative has a non-green hidden result"
                ]
            if initial_hidden_ids != set(pre_relax_hidden_ids):
                return [
                    "alt_correct: alternative hidden test identities do not match "
                    "pre-relax suite"
                ]
            public_green += 1
        else:
            invalid_alternatives.append(alt_id)

        if not relaxed_audit and has_relaxed_record:
            return [
                "alt_correct: alternative relaxed record exists without effective "
                "gold evidence"
            ]
        if relaxed_audit and public_passed and not has_relaxed_record:
            return [
                "alt_correct: relaxed audit is missing effective alternative evidence"
            ]
        if not has_relaxed_record:
            continue
        if not public_passed:
            return ["alt_correct: public-red alternative has a relaxed record"]
        relaxed = _audit_execution_record(record["relaxed"], "relaxed alternative")
        if isinstance(relaxed, str):
            return [relaxed]
        (
            relaxed_public_passed,
            relaxed_p2p_passed,
            relaxed_hidden_ids,
            relaxed_hidden_green,
        ) = relaxed
        if not relaxed_public_passed:
            return ["alt_correct: relaxed alternative public result is not green"]
        if not relaxed_p2p_passed:
            return [
                "alt_correct: relaxed public-green alternative has a non-green "
                "filtered P2P result"
            ]
        if not relaxed_hidden_green:
            return [
                "alt_correct: relaxed public-green alternative has a non-green "
                "hidden result"
            ]
        if relaxed_hidden_ids != set(final_hidden_ids):
            return [
                "alt_correct: relaxed alternative hidden test identities do not "
                "match final suite"
            ]
    if public_green < 1:
        return [
            "alt_correct: no executable real-teacher alternative passed the "
            "original public suite"
        ]

    if public_details.get("gold_public_suite_passed") is not True:
        return ["alt_correct: public gold summary is not green"]
    summary_count = public_details.get("public_valid_alternatives")
    if (
        not isinstance(summary_count, int)
        or isinstance(summary_count, bool)
        or summary_count < 0
        or summary_count > len(alternatives)
        or summary_count != public_green
    ):
        return ["alt_correct: public-valid alternative count does not match audit"]
    invalid_summary = public_details.get("invalid_teacher_proposals")
    if (
        not isinstance(invalid_summary, list)
        or any(not isinstance(item, str) for item in invalid_summary)
        or sorted(invalid_summary) != sorted(invalid_alternatives)
        or len(set(invalid_summary)) != len(invalid_summary)
    ):
        return ["alt_correct: invalid alternative identities do not match audit"]
    return []


def verify_pass_consistency(
    report: OracleReport,
    *,
    candidate: Candidate | None = None,
    kill_threshold: float = DEFAULT_KILL_THRESHOLD,
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
    # Preserve public-evidence hygiene checks for legacy readers, then enforce
    # receipt authority whenever the boundary has the candidate binding.
    problems.extend(teacher_gate_evidence_issues(report.details))
    if candidate is not None:
        problems.extend(
            teacher_gate_evidence_issues(
                report.details,
                candidate=candidate,
                protected_receipts=report.protected_teacher_transport_receipts,
            )
        )
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
    candidate: Candidate | None = None,
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
        problems = verify_pass_consistency(
            report, candidate=candidate, kill_threshold=kill_threshold
        )
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

    return await orchestrate_gates(
        gates, candidate=candidate, kill_threshold=kill_threshold
    )


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
        and not verify_pass_consistency(
            report, candidate=candidate, kill_threshold=threshold
        )
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
        report, candidate=candidate, kill_threshold=final_threshold
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
