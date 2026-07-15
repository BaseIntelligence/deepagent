"""Documented funnel skip / reject reason codes for M9 scale ≥70.

Every reject or skip along the mine → envbuild → label → oracle funnel
must use a stable, documented code. Reports aggregate these tallies so
under-supply and yield loss are auditable (never silent).

Codes are additive: prefer new constants over renames. Legacy discover/
git_mine string codes remain accepted and appear in SKIP_REASON_DOCS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Final

# ---------------------------------------------------------------------------
# Core discover / mine skip codes (legacy + M9)
# ---------------------------------------------------------------------------
SKIP_LICENSE_COPYLEFT: Final = "license_copyleft"
SKIP_LICENSE_MISSING: Final = "license_missing"
SKIP_LICENSE_UNKNOWN: Final = "license_unknown"
SKIP_LICENSE_REJECTED: Final = "license_rejected"
SKIP_CLONE_FAILED: Final = "clone_failed"
SKIP_MERGE_SCAN_FAILED: Final = "merge_scan_failed"
SKIP_BASE_COMMIT_NOT_FULL_SHA: Final = "base_commit_not_full_sha"
SKIP_MULTI_FILE_FLOOR: Final = "multi_file_floor_rejected"
SKIP_TESTS_MISSING: Final = "tests_missing"
SKIP_DISCOVER_REJECTED: Final = "discover_rejected"
SKIP_DEDUPED_PATCH: Final = "deduped_patch"
SKIP_TARGET_SATURATED: Final = "target_saturated"

# M9 funnel hardening codes
SKIP_MONOREPO: Final = "monorepo_skip"
SKIP_REPO_TOO_LARGE: Final = "repo_too_large"
SKIP_FLAKE_GATE: Final = "flake_gate"
SKIP_CACHE_MISS_FATAL: Final = "clone_cache_miss_fatal"
SKIP_PARALLEL_CAP_WAIT_TIMEOUT: Final = "parallel_cap_wait_timeout"
SKIP_OFF_LIMITS_DOCKER: Final = "off_limits_docker_refused"
SKIP_DISK_GATE: Final = "disk_gate"
SKIP_ENVBUILD_FAILED: Final = "envbuild_failed"
SKIP_ORACLE_FLAKE: Final = "oracle_flake_reject"

# M14 product hard-filter codes (live-mine keep floors)
SKIP_SOURCE_HUNKS_BELOW_FLOOR: Final = "source_hunks_below_floor"
SKIP_DOCS_CHORE_ONLY: Final = "docs_chore_only"
SKIP_NOT_MERGED: Final = "not_merged"
SKIP_SUITE_REPORTER_UNAVAILABLE: Final = "suite_reporter_unavailable"
SKIP_MOTOR_OR_HYBRID: Final = "motor_or_hybrid_rejected"
SKIP_CHORE_ONLY: Final = "chore_only_rejected"

# Alias map for normalize_reason_code (legacy / synonym → canonical)
_REASON_ALIASES: Final[Mapping[str, str]] = {
    "G2_FLAKE": SKIP_ORACLE_FLAKE,
    "FLAKE_REJECT": SKIP_ORACLE_FLAKE,
    "license_copyleft": SKIP_LICENSE_COPYLEFT,
    "copyleft": SKIP_LICENSE_COPYLEFT,
    "license_missing": SKIP_LICENSE_MISSING,
    "license_unknown": SKIP_LICENSE_UNKNOWN,
    "license_rejected": SKIP_LICENSE_REJECTED,
    "clone_failed": SKIP_CLONE_FAILED,
    "merge_scan_failed": SKIP_MERGE_SCAN_FAILED,
    "base_commit_not_full_sha": SKIP_BASE_COMMIT_NOT_FULL_SHA,
    "multi_file_floor_rejected": SKIP_MULTI_FILE_FLOOR,
    "tests_missing": SKIP_TESTS_MISSING,
    "discover_rejected": SKIP_DISCOVER_REJECTED,
    "monorepo": SKIP_MONOREPO,
    "monorepo_skip": SKIP_MONOREPO,
    "too_large": SKIP_REPO_TOO_LARGE,
    "repo_too_large": SKIP_REPO_TOO_LARGE,
    "flake": SKIP_FLAKE_GATE,
    "flake_gate": SKIP_FLAKE_GATE,
    "disk_failed": SKIP_DISK_GATE,
    "disk_gate": SKIP_DISK_GATE,
    "source_hunks_below_floor": SKIP_SOURCE_HUNKS_BELOW_FLOOR,
    "source_hunk_floor": SKIP_SOURCE_HUNKS_BELOW_FLOOR,
    "docs_chore_only": SKIP_DOCS_CHORE_ONLY,
    "docs_only": SKIP_DOCS_CHORE_ONLY,
    "chore_only": SKIP_CHORE_ONLY,
    "chore_only_rejected": SKIP_CHORE_ONLY,
    "not_merged": SKIP_NOT_MERGED,
    "unmerged": SKIP_NOT_MERGED,
    "suite_reporter_unavailable": SKIP_SUITE_REPORTER_UNAVAILABLE,
    "motor_or_hybrid_rejected": SKIP_MOTOR_OR_HYBRID,
    "motor_or_hybrid": SKIP_MOTOR_OR_HYBRID,
}


# Human documentation for each stable skip code (used in reports + CLI).
SKIP_REASON_DOCS: Final[dict[str, str]] = {
    SKIP_LICENSE_COPYLEFT: (
        "Copyleft / non-permissive license refused before clone or keep (fail-closed Val-MINE-003)."
    ),
    SKIP_LICENSE_MISSING: "No license declared; refuse DeepSWE candidate.",
    SKIP_LICENSE_UNKNOWN: "License string not on permissive allowlist; refuse keep.",
    SKIP_LICENSE_REJECTED: "License gate rejected (copyleft / missing / unknown).",
    SKIP_CLONE_FAILED: "git clone / fetch of public HTTPS remote failed.",
    SKIP_MERGE_SCAN_FAILED: "git log merge/first-parent scan failed on clone.",
    SKIP_BASE_COMMIT_NOT_FULL_SHA: ("base_commit is not a full 40-char immutable hex SHA."),
    SKIP_MULTI_FILE_FLOOR: ("Gold/source path count below multi-file floor (≥2 product files)."),
    SKIP_TESTS_MISSING: "Candidate lacks held-out / changed test paths when required.",
    SKIP_DISCOVER_REJECTED: "Generic discover gate reject (see reject.detail).",
    SKIP_DEDUPED_PATCH: ("Patch / (repo, base, gold signature) already kept earlier in funnel."),
    SKIP_TARGET_SATURATED: "Scale target already met; remaining seeds not mined.",
    SKIP_MONOREPO: (
        "Monorepo skip: multi-package workspace markers (pnpm-workspace, lerna, "
        "go.work, Cargo workspace, multi-setup.py trees, etc.) exceed modular "
        "constraints for reliable single-pack envbuild."
    ),
    SKIP_REPO_TOO_LARGE: ("Working tree or tracked file count exceeds scale envbuild size budget."),
    SKIP_FLAKE_GATE: (
        "Pre-cert flake screen: dual-run suite signatures disagree "
        "(G2_FLAKE / FLAKE_REJECT semantics)."
    ),
    SKIP_CACHE_MISS_FATAL: "Required clone cache entry missing when offline-only.",
    SKIP_PARALLEL_CAP_WAIT_TIMEOUT: (
        "Parallel envbuild semaphore wait exceeded budget (cap enforced)."
    ),
    SKIP_OFF_LIMITS_DOCKER: (
        "Docker op refused: name is off-limits (mission-test-pg / "
        "challenge-prism* / acproxy) or non-owned prefix."
    ),
    SKIP_DISK_GATE: "Free disk below envbuild fail-closed threshold.",
    SKIP_ENVBUILD_FAILED: "Envbuild agent image bake failed (non-docker-damage).",
    SKIP_ORACLE_FLAKE: (
        "Docker/oracle dual-run flake: gold or null suite signatures disagree; "
        "never certify (VAL-ORCD-007)."
    ),
    SKIP_SOURCE_HUNKS_BELOW_FLOOR: (
        "Product keep requires ≥10 product-source unified-diff hunks "
        "(soft MULTI_FILE_FLOOR=2 is engineering-only; VAL-LHARD-001)."
    ),
    SKIP_DOCS_CHORE_ONLY: (
        "Pure docs/README/chore/config-only candidate; no product source keep (VAL-LHARD-003)."
    ),
    SKIP_NOT_MERGED: "PR is not merged (merged_at missing); live mine refuses open/unmerged.",
    SKIP_SUITE_REPORTER_UNAVAILABLE: (
        "No real suite reporter for candidate language; product dual-run refuse (VAL-LHARD-002)."
    ),
    SKIP_MOTOR_OR_HYBRID: (
        "Motor/hybrid/synthetic fixture identity refused on live product mine (VAL-LMINE-005)."
    ),
    SKIP_CHORE_ONLY: "Chore/config-only maintenance PR refused for product keep.",
}


@dataclass(frozen=True, slots=True)
class SkipReason:
    """One documented funnel skip event."""

    code: str
    detail: str = ""
    stage: str = "discover"  # discover | mine | envbuild | label | oracle | ship
    repo: str = ""
    candidate_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["documentation"] = describe_skip_reason(self.code)
        return row


def normalize_reason_code(code: str | None) -> str:
    """Map synonyms / legacy codes onto the stable documented set when known."""
    if not code:
        return SKIP_DISCOVER_REJECTED
    cleaned = str(code).strip()
    if cleaned in SKIP_REASON_DOCS:
        return cleaned
    alias = _REASON_ALIASES.get(cleaned) or _REASON_ALIASES.get(cleaned.lower())
    if alias:
        return alias
    return cleaned


def describe_skip_reason(code: str | None) -> str:
    """Return human documentation for a skip code (unknown → generic note)."""
    norm = normalize_reason_code(code)
    if norm in SKIP_REASON_DOCS:
        return SKIP_REASON_DOCS[norm]
    return f"Undocumented / stage-specific reject code {norm!r}; treat as audit detail."


def document_all_skip_reasons() -> list[dict[str, str]]:
    """Return sorted [{code, documentation}] for CLI/report emission."""
    return [{"code": code, "documentation": doc} for code, doc in sorted(SKIP_REASON_DOCS.items())]


def tally_skip_reasons(events: list[SkipReason] | list[Mapping[str, Any]]) -> dict[str, int]:
    """Aggregate skip events by normalized code."""
    tallies: dict[str, int] = {}
    for event in events:
        if isinstance(event, SkipReason):
            code = normalize_reason_code(event.code)
        else:
            code = normalize_reason_code(str(event.get("code") or event.get("reason_code") or ""))
        tallies[code] = tallies.get(code, 0) + 1
    return tallies


__all__ = [
    "SKIP_BASE_COMMIT_NOT_FULL_SHA",
    "SKIP_CACHE_MISS_FATAL",
    "SKIP_CHORE_ONLY",
    "SKIP_CLONE_FAILED",
    "SKIP_DEDUPED_PATCH",
    "SKIP_DISCOVER_REJECTED",
    "SKIP_DISK_GATE",
    "SKIP_DOCS_CHORE_ONLY",
    "SKIP_ENVBUILD_FAILED",
    "SKIP_FLAKE_GATE",
    "SKIP_LICENSE_COPYLEFT",
    "SKIP_LICENSE_MISSING",
    "SKIP_LICENSE_REJECTED",
    "SKIP_LICENSE_UNKNOWN",
    "SKIP_MERGE_SCAN_FAILED",
    "SKIP_MONOREPO",
    "SKIP_MOTOR_OR_HYBRID",
    "SKIP_MULTI_FILE_FLOOR",
    "SKIP_NOT_MERGED",
    "SKIP_OFF_LIMITS_DOCKER",
    "SKIP_ORACLE_FLAKE",
    "SKIP_PARALLEL_CAP_WAIT_TIMEOUT",
    "SKIP_REASON_DOCS",
    "SKIP_REPO_TOO_LARGE",
    "SKIP_SOURCE_HUNKS_BELOW_FLOOR",
    "SKIP_SUITE_REPORTER_UNAVAILABLE",
    "SKIP_TARGET_SATURATED",
    "SKIP_TESTS_MISSING",
    "SkipReason",
    "describe_skip_reason",
    "document_all_skip_reasons",
    "normalize_reason_code",
    "tally_skip_reasons",
]
