"""M9 funnel hardening orchestration: skip docs + monorepo + flake + cache + parallel.

Provides a small API the mine / ship workers use for scale ≥70:

1. Documented skip reasons (always tallied)
2. Monorepo / size skip before expensive envbuild
3. Clone cache for reused remotes
4. Flake pre-gate helpers (delegate to label/oracle detectors)
5. Bounded parallel envbuild advisories

Does not claim ship N≥70 by itself — that is the following pipeline feature.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_factory.oracle.codes import FLAKE_REJECT, G2_FLAKE
from swe_factory.sources.clone_cache import (
    CloneCache,
    default_clone_cache_root,
    ensure_cached_clone,
)
from swe_factory.sources.monorepo import (
    MonorepoDecision,
    evaluate_monorepo_gate,
    paths_look_monorepo,
    scan_monorepo_signals,
)
from swe_factory.sources.skip_reasons import (
    SKIP_DEDUPED_PATCH,
    SKIP_FLAKE_GATE,
    SKIP_MONOREPO,
    SKIP_ORACLE_FLAKE,
    SKIP_REASON_DOCS,
    SkipReason,
    describe_skip_reason,
    document_all_skip_reasons,
    normalize_reason_code,
    tally_skip_reasons,
)

# Local copies of concurrency constants to avoid circular import with envbuild.parallel
# (sources/__init__ → funnel → envbuild.parallel → sources.skip_reasons → sources/__init__).
DEFAULT_ENVBUILD_WORKERS = 16
HARD_MAX_ENVBUILD_WORKERS = 24
MAX_CONCURRENT_ENVBUILD_JOBS = 16
CONCURRENCY_HINT = (
    "Recommended concurrent envbuild/Pier jobs: ≤16 (hard ceiling 24). "
    f"Code default MAX_CONCURRENT_ENVBUILD_JOBS={MAX_CONCURRENT_ENVBUILD_JOBS}."
)


def clamp_envbuild_workers(
    requested: int | None, *, default: int = DEFAULT_ENVBUILD_WORKERS
) -> int:
    """Bound worker count to [1, HARD_MAX] (mirrors envbuild.parallel.clamp)."""
    if requested is None:
        n = default
    else:
        try:
            n = int(requested)
        except (TypeError, ValueError):
            n = default
    if n < 1:
        n = 1
    if n > HARD_MAX_ENVBUILD_WORKERS:
        n = HARD_MAX_ENVBUILD_WORKERS
    return n


@dataclass
class FunnelConfig:
    """Tunables for M9 scale funnel hardening."""

    # Parallel envbuild
    max_envbuild_workers: int = DEFAULT_ENVBUILD_WORKERS
    # Monorepo
    monorepo_enabled: bool = True
    max_package_markers: int = 8
    max_tracked_files: int = 25_000
    max_tree_bytes: int = 800 * 1024 * 1024
    # Clone cache
    clone_cache_enabled: bool = True
    clone_cache_root: str = str(default_clone_cache_root())
    clone_cache_depth: int = 80
    # Flake gate is always on for cert paths
    flake_gate_enabled: bool = True
    # Dedup gold signatures
    dedupe_enabled: bool = True

    def clamped_workers(self) -> int:
        return clamp_envbuild_workers(self.max_envbuild_workers)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["clamped_envbuild_workers"] = self.clamped_workers()
        d["hard_max_envbuild_workers"] = HARD_MAX_ENVBUILD_WORKERS
        d["hygiene_ceiling"] = MAX_CONCURRENT_ENVBUILD_JOBS
        d["concurrency_hint"] = CONCURRENCY_HINT
        return d


@dataclass
class FunnelCounters:
    """Running tallies for a funnel batch."""

    considered: int = 0
    kept: int = 0
    skipped: int = 0
    flake_rejected: int = 0
    monorepo_skipped: int = 0
    size_skipped: int = 0
    deduped: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    skip_by_code: dict[str, int] = field(default_factory=dict)

    def record_skip(self, code: str) -> None:
        norm = normalize_reason_code(code)
        self.skipped += 1
        self.skip_by_code[norm] = self.skip_by_code.get(norm, 0) + 1
        if norm == SKIP_MONOREPO:
            self.monorepo_skipped += 1
        elif norm in {SKIP_FLAKE_GATE, SKIP_ORACLE_FLAKE, G2_FLAKE, FLAKE_REJECT}:
            self.flake_rejected += 1
        elif norm == SKIP_DEDUPED_PATCH:
            self.deduped += 1
        elif norm == "repo_too_large":
            self.size_skipped += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunnelReport:
    """Serializable funnel optimization report for ship / CLI."""

    config: FunnelConfig = field(default_factory=FunnelConfig)
    counters: FunnelCounters = field(default_factory=FunnelCounters)
    skips: list[SkipReason] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    produced_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def add_skip(self, reason: SkipReason) -> None:
        self.skips.append(reason)
        self.counters.record_skip(reason.code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "milestone": "m9-scale-70",
            "feature": "m9-funnel-hardening",
            "produced_at": self.produced_at,
            "config": self.config.to_dict(),
            "counters": self.counters.to_dict(),
            "skip_reasons_tallied": tally_skip_reasons(self.skips),
            "documented_skip_catalog": document_all_skip_reasons(),
            "skips": [s.to_dict() for s in self.skips],
            "notes": list(self.notes),
            "parallelism_bounded": self.config.clamped_workers() <= HARD_MAX_ENVBUILD_WORKERS,
            "off_limits_docker_policy": (
                "never touch mission-test-pg / challenge-prism* / acproxy; "
                "only sdf-/deepswe-/harbor-sdf- prefixes"
            ),
        }

    def write_json(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return out


def gold_signature(
    *,
    repo: str,
    base_commit: str,
    gold_patch: str,
    source_files: Sequence[str] | None = None,
) -> str:
    """Stable signature for dedupe across discover keeps."""
    parts = [
        (repo or "").strip().lower(),
        (base_commit or "").strip().lower(),
        hashlib.sha256((gold_patch or "").encode()).hexdigest(),
    ]
    if source_files:
        parts.append("|".join(sorted(source_files)))
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()


def apply_monorepo_skip(
    *,
    root: Path | str | None = None,
    paths: Sequence[str] | None = None,
    config: FunnelConfig | None = None,
    repo: str = "",
    candidate_id: str = "",
    stage: str = "mine",
) -> tuple[bool, SkipReason | None, MonorepoDecision]:
    """Run monorepo / size gate; return (should_skip, skip_reason, decision)."""
    cfg = config or FunnelConfig()
    if not cfg.monorepo_enabled:
        return False, None, MonorepoDecision(skip=False)

    if root is not None:
        decision = evaluate_monorepo_gate(
            root,
            max_package_markers=cfg.max_package_markers,
            max_tracked_files=cfg.max_tracked_files,
            max_tree_bytes=cfg.max_tree_bytes,
        )
    elif paths is not None:
        decision = paths_look_monorepo(paths)
    else:
        decision = MonorepoDecision(skip=False)

    if not decision.skip:
        return False, None, decision
    reason = decision.to_skip_reason(stage=stage, repo=repo, candidate_id=candidate_id)
    return True, reason, decision


def apply_flake_gate(
    *,
    is_flake: bool,
    reason_codes: Sequence[str] | None = None,
    details: Mapping[str, Any] | None = None,
    repo: str = "",
    candidate_id: str = "",
    stage: str = "label",
    config: FunnelConfig | None = None,
) -> tuple[bool, SkipReason | None]:
    """Map flake detector output onto a documented funnel skip."""
    cfg = config or FunnelConfig()
    if not cfg.flake_gate_enabled or not is_flake:
        return False, None
    codes = [normalize_reason_code(c) for c in (reason_codes or (SKIP_FLAKE_GATE,))]
    primary = codes[0] if codes else SKIP_FLAKE_GATE
    # Prefer oracle flake code when G2/FLAKE present
    if any(c in {SKIP_ORACLE_FLAKE, G2_FLAKE, FLAKE_REJECT} for c in (reason_codes or ())):
        primary = SKIP_ORACLE_FLAKE
    reason = SkipReason(
        code=primary,
        detail=describe_skip_reason(primary)
        + (f"; codes={list(reason_codes or ())}" if reason_codes else ""),
        stage=stage,
        repo=repo,
        candidate_id=candidate_id,
        meta=dict(details or {}),
    )
    return True, reason


def default_funnel_config_for_scale(target: int = 70) -> FunnelConfig:
    """Config used when mining/envbuild aiming for scale ≥70."""
    # Keep workers at hygiene ceiling; callers may raise to hard max deliberately.
    workers = DEFAULT_ENVBUILD_WORKERS
    if target >= 70:
        workers = DEFAULT_ENVBUILD_WORKERS
    return FunnelConfig(
        max_envbuild_workers=workers,
        monorepo_enabled=True,
        clone_cache_enabled=True,
        flake_gate_enabled=True,
        dedupe_enabled=True,
    )


def make_scale_funnel_report(
    config: FunnelConfig | None = None,
    *,
    notes: Sequence[str] | None = None,
) -> FunnelReport:
    """Fresh report skeleton for a scale ≥70 run."""
    cfg = config or default_funnel_config_for_scale(70)
    report = FunnelReport(config=cfg)
    report.notes.extend(
        [
            "M9 funnel hardening active: flake gates, monorepo skip, clone cache, "
            "parallel envbuild with caps",
            f"parallel workers clamped to {cfg.clamped_workers()} "
            f"(hard max {HARD_MAX_ENVBUILD_WORKERS})",
            "documented skip reasons catalog available via funnel-report",
            "off-limits docker: mission-test-pg / challenge-prism* / acproxy never touched",
        ]
    )
    if notes:
        report.notes.extend(notes)
    return report


def clone_cache_from_config(config: FunnelConfig) -> CloneCache:
    """Build a CloneCache from FunnelConfig."""
    return CloneCache(
        root=config.clone_cache_root if config.clone_cache_enabled else default_clone_cache_root(),
        depth=config.clone_cache_depth,
    )


# Re-export key symbols for one-import convenience
__all__ = [
    "CONCURRENCY_HINT",
    "DEFAULT_ENVBUILD_WORKERS",
    "FUNNEL_SKIP_REASON_DOCS",
    "FunnelConfig",
    "FunnelCounters",
    "FunnelReport",
    "HARD_MAX_ENVBUILD_WORKERS",
    "MAX_CONCURRENT_ENVBUILD_JOBS",
    "SKIP_REASON_DOCS",
    "apply_flake_gate",
    "apply_monorepo_skip",
    "clamp_envbuild_workers",
    "clone_cache_from_config",
    "default_funnel_config_for_scale",
    "describe_skip_reason",
    "document_all_skip_reasons",
    "ensure_cached_clone",
    "evaluate_monorepo_gate",
    "gold_signature",
    "make_scale_funnel_report",
    "normalize_reason_code",
    "paths_look_monorepo",
    "scan_monorepo_signals",
    "tally_skip_reasons",
]

# Alias for external docs
FUNNEL_SKIP_REASON_DOCS = SKIP_REASON_DOCS
