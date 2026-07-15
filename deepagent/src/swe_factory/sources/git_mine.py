"""Git-clone-only multi-lang allowlist miner (M8 default live path).

Until OXYLABS_* credentials are present, DeepAgent discover uses **local git
clone/fetch** of public HTTPS remotes and merge-commit history as the sole
authority for multi-file source+test candidates. Oxylabs is never called for
diffs or history; it remains optional for GitHub HTTP page/raw only.

Yield goals:
- Expand REMOTE_SEEDS multi-lang inventory (Py/TS/Go/JS/Rust)
- Emit enough real_pr / hybrid-eligible DiscoverCandidate rows to feed m8-ship ≥30
- Record language histogram + honest under-supply when a language yields zero
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from swe_factory.oracle.gates import MULTI_FILE_FLOOR
from swe_factory.sources.allowlist import (
    SCALE_LANGUAGES,
    SeedRepo,
    normalize_language,
    remote_mine_seeds,
    scale_inventory_report,
    under_supply_reasons,
)
from swe_factory.sources.clone import CloneError, is_immutable_sha
from swe_factory.sources.clone_cache import CloneCache
from swe_factory.sources.discover import (
    DiscoverCandidate,
    DiscoverError,
    DiscoverReject,
    DiscoverReport,
    discover_merge_range_from_git,
    is_full_sha,
    write_candidate_artifacts,
)
from swe_factory.sources.funnel import (
    FunnelConfig,
    FunnelReport,
    apply_monorepo_skip,
    default_funnel_config_for_scale,
    gold_signature,
    make_scale_funnel_report,
)
from swe_factory.sources.license_gate import LicenseGateError, assert_permissive_license
from swe_factory.sources.skip_reasons import (
    SKIP_DEDUPED_PATCH,
    SKIP_TARGET_SATURATED,
    SkipReason,
    describe_skip_reason,
    document_all_skip_reasons,
    normalize_reason_code,
)

logger = logging.getLogger(__name__)

MineMode = Literal["git_clone_only"]


class GitMineError(RuntimeError):
    """Unrecoverable allowlist git mine failure."""


@dataclass(frozen=True, slots=True)
class MergeRange:
    """One merge commit with its first-parent base (history authority = git)."""

    merge_sha: str
    base_sha: str
    subject: str = ""


@dataclass
class SeedMineStats:
    seed_id: str
    language: str
    repo: str
    license: str
    clone_ok: bool = False
    merges_scanned: int = 0
    kept: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GitMineReport:
    """Multi-lang git-clone mine funnel with honest under-supply stats."""

    mode: MineMode
    kept: list[DiscoverCandidate] = field(default_factory=list)
    rejected: list[DiscoverReject] = field(default_factory=list)
    seed_stats: list[SeedMineStats] = field(default_factory=list)
    language_kept: dict[str, int] = field(default_factory=dict)
    language_inventory: dict[str, int] = field(default_factory=dict)
    under_supply: list[str] = field(default_factory=list)
    provider_calls: int = 0
    network_required: bool = True  # git HTTPS clone may touch network
    history_authority: str = "git"
    http_metadata_source: str = "none"
    target_candidates: int = 30
    work_root: str = ""
    inventory: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    oxylabs_status: str = "not_used"
    # M9 funnel hardening attachment
    funnel: dict[str, Any] = field(default_factory=dict)
    clone_cache_stats: dict[str, Any] = field(default_factory=dict)
    documented_skip_reasons: list[dict[str, str]] = field(default_factory=list)

    @property
    def keep_count(self) -> int:
        return len(self.kept)

    @property
    def reject_count(self) -> int:
        return len(self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "mode": self.mode,
            "keep_count": self.keep_count,
            "reject_count": self.reject_count,
            "provider_calls": self.provider_calls,
            "network_required": self.network_required,
            "history_authority": self.history_authority,
            "http_metadata_source": self.http_metadata_source,
            "target_candidates": self.target_candidates,
            "language_kept": dict(self.language_kept),
            "language_inventory": dict(self.language_inventory),
            "under_supply": list(self.under_supply),
            "oxylabs_status": self.oxylabs_status,
            "work_root": self.work_root,
            "inventory": dict(self.inventory),
            "notes": list(self.notes),
            "seed_stats": [s.to_dict() for s in self.seed_stats],
            "kept": [c.to_dict() for c in self.kept],
            "rejected": [r.to_dict() for r in self.rejected],
            "reject_reasons": _tally_rejects(self.rejected),
            "funnel": dict(self.funnel),
            "clone_cache_stats": dict(self.clone_cache_stats),
            "documented_skip_reasons": list(
                self.documented_skip_reasons or document_all_skip_reasons()
            ),
            "produced_at": datetime.now(UTC).isoformat(),
        }

    def as_discover_report(self) -> DiscoverReport:
        return DiscoverReport(
            kept=tuple(self.kept),
            rejected=tuple(self.rejected),
            provider_calls=self.provider_calls,
            network_required=self.network_required,
            offline=False,
            history_authority=self.history_authority,
        )


def _tally_rejects(rows: Sequence[DiscoverReject]) -> dict[str, int]:
    tallies: dict[str, int] = {}
    for row in rows:
        tallies[row.reason_code] = tallies.get(row.reason_code, 0) + 1
    return tallies


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def clone_seed_repo(
    seed: SeedRepo,
    *,
    dest_root: Path,
    depth: int = 80,
    reuse: bool = True,
    clone_cache: CloneCache | None = None,
) -> Path:
    """Shallow-ish clone of a public allowlisted HTTPS remote for history mine.

    History authority remains local git. No Oxylabs / GitHub REST required.

    When *clone_cache* is provided (M9 default), resolve the durable cache entry
    first and materialize / reuse a work dir under *dest_root* so repeated scale
    mines avoid re-cloning remotes.
    """
    assert_permissive_license(seed.license, repo=seed.repo)
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    clone_dir = dest_root / f"{seed.seed_id}_mine"

    if clone_cache is not None:
        entry = clone_cache.ensure_seed(seed, refresh=reuse)
        # Reuse dest when already present and reuse=True
        if reuse and clone_dir.is_dir() and (clone_dir / ".git").exists():
            _run_git(["fetch", "--depth", str(max(1, depth)), "origin"], cwd=clone_dir)
            return clone_dir
        # Materialize from cache (path may be the cache itself for mine walk when safe)
        cache_path = Path(entry.path)
        if cache_path.is_dir() and (cache_path / ".git").exists():
            # Prefer work-dir copy so mine mutations don't trash the cache
            if clone_dir.exists():
                shutil.rmtree(clone_dir)
            # Fast path: for mine (read-only-ish git log) we may hard-use cache path
            # when dest would force a full copy — use cache path directly for log.
            return cache_path
        raise CloneError(f"clone cache entry unusable for {seed.repo}: {entry.path}")

    if reuse and clone_dir.is_dir() and (clone_dir / ".git").exists():
        # Refresh a small window of history when possible
        _run_git(["fetch", "--depth", str(max(1, depth)), "origin"], cwd=clone_dir)
        return clone_dir

    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    url = seed.repository_url
    if not url.startswith("https://") and not url.startswith("http://"):
        url = f"https://github.com/{seed.repo}.git"
    if not url.endswith(".git") and "github.com" in url:
        url = url.rstrip("/") + ".git"

    clone = _run_git(
        [
            "clone",
            "--filter=blob:none",
            f"--depth={max(20, depth)}",
            "--no-single-branch",
            url,
            str(clone_dir),
        ],
        timeout=300,
    )
    if clone.returncode != 0:
        # Retry without depth (some remotes reject shallow)
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        clone = _run_git(
            ["clone", "--filter=blob:none", url, str(clone_dir)],
            timeout=600,
        )
        if clone.returncode != 0:
            raise CloneError(
                f"git clone failed for {seed.repo}: "
                f"{(clone.stderr or clone.stdout or '').strip()[:400]}"
            )
    # Ensure default branch tip is checked out for rev-list
    _run_git(["checkout", "--force", "HEAD"], cwd=clone_dir)
    return clone_dir


def list_merge_ranges(
    repo_path: Path,
    *,
    max_merges: int = 40,
) -> list[MergeRange]:
    """Enumerate recent first-parent merge commits: base = parent1, head = merge."""
    if not Path(repo_path).is_dir():
        raise GitMineError(f"repo path missing: {repo_path}")
    # Prefer --first-parent merge walk for PR-like ranges
    proc = _run_git(
        [
            "log",
            "--merges",
            "--first-parent",
            f"-n{max(1, max_merges)}",
            "--pretty=format:%H%x09%s",
        ],
        cwd=repo_path,
        timeout=120,
    )
    if proc.returncode != 0:
        # Fall back to all merges
        proc = _run_git(
            [
                "log",
                "--merges",
                f"-n{max(1, max_merges)}",
                "--pretty=format:%H%x09%s",
            ],
            cwd=repo_path,
            timeout=120,
        )
    if proc.returncode != 0:
        raise GitMineError(
            f"git log --merges failed: {(proc.stderr or proc.stdout or '').strip()[:300]}"
        )

    ranges: list[MergeRange] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        merge_sha = parts[0].strip().lower()
        subject = parts[1].strip() if len(parts) > 1 else ""
        if not is_full_sha(merge_sha) and not is_immutable_sha(merge_sha):
            continue
        parents = _run_git(["rev-parse", f"{merge_sha}^1"], cwd=repo_path)
        if parents.returncode != 0:
            continue
        base_sha = parents.stdout.strip().lower()
        if not is_full_sha(base_sha):
            # expand short
            full = _run_git(["rev-parse", base_sha], cwd=repo_path)
            base_sha = full.stdout.strip().lower()
        if not is_full_sha(base_sha):
            continue
        # normalize merge to full
        merge_full = _run_git(["rev-parse", merge_sha], cwd=repo_path)
        merge_resolved = merge_full.stdout.strip().lower()
        if not is_full_sha(merge_resolved):
            continue
        ranges.append(
            MergeRange(merge_sha=merge_resolved, base_sha=base_sha, subject=subject[:200])
        )
    return ranges


def list_non_merge_ranges(
    repo_path: Path,
    *,
    max_commits: int = 30,
    min_parents: int = 1,
) -> list[MergeRange]:
    """Fallback: consecutive first-parent commits when merge yield is thin.

    Treats commit^1..commit as a mini PR range. Still git-history authority.
    """
    del min_parents  # reserved
    proc = _run_git(
        [
            "log",
            "--first-parent",
            f"-n{max(2, max_commits + 1)}",
            "--pretty=format:%H%x09%s",
        ],
        cwd=repo_path,
        timeout=120,
    )
    if proc.returncode != 0:
        return []
    shas: list[tuple[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split("\t", 1)
        sha = parts[0].strip().lower()
        subject = parts[1].strip() if len(parts) > 1 else ""
        if is_full_sha(sha) or is_immutable_sha(sha):
            full = _run_git(["rev-parse", sha], cwd=repo_path).stdout.strip().lower()
            if is_full_sha(full):
                shas.append((full, subject[:200]))
    ranges: list[MergeRange] = []
    for i in range(len(shas) - 1):
        head_sha, subject = shas[i]
        base_sha = shas[i + 1][0]
        if head_sha == base_sha:
            continue
        ranges.append(MergeRange(merge_sha=head_sha, base_sha=base_sha, subject=subject))
        if len(ranges) >= max_commits:
            break
    return ranges


def mine_seed_merges(
    seed: SeedRepo,
    *,
    dest_root: Path,
    max_merges: int = 40,
    max_keeps: int = 8,
    min_source_files: int = MULTI_FILE_FLOOR,
    require_tests: bool = True,
    allow_non_merge_fallback: bool = True,
    reuse_clone: bool = True,
    clone_cache: CloneCache | None = None,
    funnel_config: FunnelConfig | None = None,
    funnel_report: FunnelReport | None = None,
    seen_signatures: set[str] | None = None,
) -> tuple[list[DiscoverCandidate], list[DiscoverReject], SeedMineStats]:
    """Mine one allowlisted remote via git only; return keeps + rejects + stats."""
    stats = SeedMineStats(
        seed_id=seed.seed_id,
        language=seed.language,
        repo=seed.repo,
        license=seed.license,
    )
    keeps: list[DiscoverCandidate] = []
    rejects: list[DiscoverReject] = []

    try:
        assert_permissive_license(seed.license, repo=seed.repo)
    except LicenseGateError as exc:
        stats.error = str(exc)
        rejects.append(
            DiscoverReject(
                repo=seed.repo,
                reason_code=exc.reason_code,
                detail=str(exc),
                repository_url=seed.repository_url,
            )
        )
        stats.rejected += 1
        stats.reject_reasons[exc.reason_code] = stats.reject_reasons.get(exc.reason_code, 0) + 1
        return keeps, rejects, stats

    try:
        repo_path = clone_seed_repo(
            seed,
            dest_root=dest_root,
            reuse=reuse_clone,
            clone_cache=clone_cache,
        )
        stats.clone_ok = True
    except (CloneError, GitMineError, OSError) as exc:
        stats.error = str(exc)
        rejects.append(
            DiscoverReject(
                repo=seed.repo,
                reason_code="clone_failed",
                detail=str(exc)[:500],
                repository_url=seed.repository_url,
            )
        )
        stats.rejected += 1
        stats.reject_reasons["clone_failed"] = 1
        if funnel_report is not None:
            funnel_report.add_skip(
                SkipReason(
                    code="clone_failed",
                    detail=str(exc)[:400],
                    stage="mine",
                    repo=seed.repo,
                )
            )
        return keeps, rejects, stats

    # M9 monorepo / size gate (documented skip, no silent drop)
    cfg = funnel_config or FunnelConfig()
    should_skip, mono_reason, mono_decision = apply_monorepo_skip(
        root=repo_path,
        config=cfg,
        repo=seed.repo,
        stage="mine",
    )
    if should_skip and mono_reason is not None:
        stats.error = mono_reason.detail
        rejects.append(
            DiscoverReject(
                repo=seed.repo,
                reason_code=normalize_reason_code(mono_reason.code),
                detail=mono_reason.detail[:500],
                repository_url=seed.repository_url,
                meta={
                    "monorepo": mono_decision.to_dict(),
                    "documentation": describe_skip_reason(mono_reason.code),
                },
            )
        )
        stats.rejected += 1
        stats.reject_reasons[mono_reason.code] = stats.reject_reasons.get(mono_reason.code, 0) + 1
        if funnel_report is not None:
            funnel_report.add_skip(mono_reason)
        return keeps, rejects, stats

    try:
        merges = list_merge_ranges(repo_path, max_merges=max_merges)
        if not merges and allow_non_merge_fallback:
            merges = list_non_merge_ranges(repo_path, max_commits=max_merges)
    except GitMineError as exc:
        stats.error = str(exc)
        rejects.append(
            DiscoverReject(
                repo=seed.repo,
                reason_code="merge_scan_failed",
                detail=str(exc)[:500],
                repository_url=seed.repository_url,
            )
        )
        stats.rejected += 1
        stats.reject_reasons["merge_scan_failed"] = 1
        return keeps, rejects, stats

    stats.merges_scanned = len(merges)

    for merge in merges:
        if len(keeps) >= max_keeps:
            break
        try:
            candidate = discover_merge_range_from_git(
                repo_path,
                base=merge.base_sha,
                head=merge.merge_sha,
                repo=seed.repo,
                repository_url=seed.repository_url,
                license=seed.license,
                language=seed.language,
                title=merge.subject or f"merge {merge.merge_sha[:12]}",
                merge_commit_sha=merge.merge_sha,
                min_source_files=min_source_files,
                require_tests=require_tests,
                kind="real_pr",
            )
        except (DiscoverError, LicenseGateError, CloneError) as exc:
            reason = getattr(exc, "reason_code", None) or _reason_from_message(str(exc))
            reason = normalize_reason_code(reason)
            rejects.append(
                DiscoverReject(
                    repo=seed.repo,
                    reason_code=reason,
                    detail=str(exc)[:400],
                    repository_url=seed.repository_url,
                    base_commit=merge.base_sha,
                    meta={
                        "merge_sha": merge.merge_sha,
                        "seed_id": seed.seed_id,
                        "language": seed.language,
                        "documentation": describe_skip_reason(reason),
                    },
                )
            )
            stats.rejected += 1
            stats.reject_reasons[reason] = stats.reject_reasons.get(reason, 0) + 1
            if funnel_report is not None:
                funnel_report.add_skip(
                    SkipReason(
                        code=reason,
                        detail=str(exc)[:400],
                        stage="mine",
                        repo=seed.repo,
                        meta={"merge_sha": merge.merge_sha},
                    )
                )
            continue

        # M9 path-list monorepo heuristic on gold+test files (extra guard)
        path_skip, path_reason, _path_dec = apply_monorepo_skip(
            paths=[*candidate.source_files, *candidate.test_files, *candidate.gold_files],
            config=cfg,
            repo=seed.repo,
            candidate_id=candidate.candidate_id,
            stage="mine",
        )
        if path_skip and path_reason is not None:
            rejects.append(
                DiscoverReject(
                    repo=seed.repo,
                    reason_code=path_reason.code,
                    detail=path_reason.detail[:500],
                    repository_url=seed.repository_url,
                    base_commit=candidate.base_commit,
                    meta={
                        "merge_sha": merge.merge_sha,
                        "seed_id": seed.seed_id,
                        "documentation": describe_skip_reason(path_reason.code),
                    },
                )
            )
            stats.rejected += 1
            stats.reject_reasons[path_reason.code] = (
                stats.reject_reasons.get(path_reason.code, 0) + 1
            )
            if funnel_report is not None:
                funnel_report.add_skip(path_reason)
            continue

        # M9 gold signature dedupe
        sig = gold_signature(
            repo=candidate.repo or seed.repo,
            base_commit=candidate.base_commit,
            gold_patch=candidate.gold_patch,
            source_files=candidate.gold_files or candidate.source_files,
        )
        if seen_signatures is not None and cfg.dedupe_enabled and sig in seen_signatures:
            rejects.append(
                DiscoverReject(
                    repo=seed.repo,
                    reason_code=SKIP_DEDUPED_PATCH,
                    detail=f"duplicate gold signature {sig[:12]}… already kept",
                    repository_url=seed.repository_url,
                    base_commit=candidate.base_commit,
                    meta={
                        "gold_signature": sig,
                        "documentation": describe_skip_reason(SKIP_DEDUPED_PATCH),
                    },
                )
            )
            stats.rejected += 1
            stats.reject_reasons[SKIP_DEDUPED_PATCH] = (
                stats.reject_reasons.get(SKIP_DEDUPED_PATCH, 0) + 1
            )
            if funnel_report is not None:
                funnel_report.add_skip(
                    SkipReason(
                        code=SKIP_DEDUPED_PATCH,
                        detail=f"duplicate gold signature {sig[:12]}",
                        stage="mine",
                        repo=seed.repo,
                        candidate_id=candidate.candidate_id,
                        meta={"gold_signature": sig},
                    )
                )
            continue

        # Stamp seed metadata for complete funnel tables
        meta = dict(candidate.meta)
        meta.update(
            {
                "seed_id": seed.seed_id,
                "allowlist_license": seed.license,
                "mine_mode": "git_clone_only",
                "merge_subject": merge.subject,
                "gold_signature": sig,
                "funnel_hardening": "m9",
            }
        )
        # Recreate frozen dataclass with extended meta (DiscoverCandidate is frozen)
        candidate = DiscoverCandidate(
            candidate_id=candidate.candidate_id,
            kind=candidate.kind,
            repository_url=candidate.repository_url,
            repo=candidate.repo,
            base_commit=candidate.base_commit,
            language=candidate.language or seed.language,
            license=candidate.license,
            title=candidate.title,
            problem_statement=candidate.problem_statement,
            source_files=candidate.source_files,
            test_files=candidate.test_files,
            gold_patch=candidate.gold_patch,
            test_patch=candidate.test_patch,
            gold_files=candidate.gold_files,
            history_authority="git",
            http_metadata_source="none",
            merge_commit_sha=candidate.merge_commit_sha or merge.merge_sha,
            pr_number=candidate.pr_number,
            html_url=candidate.html_url,
            reject_reason=None,
            meta=meta,
        )
        keeps.append(candidate)
        stats.kept += 1
        if seen_signatures is not None:
            seen_signatures.add(sig)
        if funnel_report is not None:
            funnel_report.counters.kept += 1
            funnel_report.counters.considered += 1

    return keeps, rejects, stats


def _reason_from_message(message: str) -> str:
    lower = message.lower()
    if "copyleft" in lower or "license" in lower:
        return "license_rejected"
    if "40-char" in lower or ("full" in lower and "sha" in lower):
        return "base_commit_not_full_sha"
    if "multi-file" in lower or "floor" in lower:
        return "multi_file_floor_rejected"
    if "require_tests" in lower or "no test" in lower:
        return "tests_missing"
    if "clone" in lower:
        return "clone_failed"
    return "discover_rejected"


def mine_allowlist_git_only(
    *,
    work_root: Path,
    target_candidates: int = 30,
    max_merges_per_seed: int = 40,
    max_keeps_per_seed: int = 6,
    languages: Sequence[str] | None = None,
    seed_ids: Sequence[str] | None = None,
    write_artifacts: bool = True,
    reuse_clones: bool = True,
    require_tests: bool = True,
    min_source_files: int = MULTI_FILE_FLOOR,
    funnel_config: FunnelConfig | None = None,
    use_clone_cache: bool = True,
    clone_cache_root: Path | str | None = None,
) -> GitMineReport:
    """Run git-clone-only allowlist discovery across multi-lang remotes.

    Never calls Oxylabs. Records language classification + under-supply.

    M9 funnel hardening (defaults on for scale paths):
    - documented skip reasons on every reject
    - monorepo / size skip
    - clone cache reuse
    - gold signature dedupe
    """
    from swe_factory.envbuild.hygiene import MAX_CONCURRENT_ENVBUILD_JOBS

    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    inventory = scale_inventory_report()

    cfg = funnel_config or default_funnel_config_for_scale(
        target_candidates if target_candidates >= 70 else 30
    )
    funnel_report = make_scale_funnel_report(
        cfg,
        notes=[
            f"mine target_candidates={target_candidates}",
            f"clone_cache={'on' if use_clone_cache and cfg.clone_cache_enabled else 'off'}",
        ],
    )
    seen_signatures: set[str] = set()

    cache: CloneCache | None = None
    if use_clone_cache and cfg.clone_cache_enabled:
        if clone_cache_root is not None:
            cache_root = Path(clone_cache_root)
        elif target_candidates >= 70:
            cache_root = work_root / "clone_cache"
        else:
            cache_root = Path(cfg.clone_cache_root)
        cache = CloneCache(root=cache_root, depth=cfg.clone_cache_depth)

    # Select seeds
    if seed_ids:
        wanted = {s.strip() for s in seed_ids if s and s.strip()}
        seeds = [s for s in remote_mine_seeds() if s.seed_id in wanted]
    elif languages:
        seeds = []
        for lang in languages:
            seeds.extend(remote_mine_seeds(language=lang))
        # de-dupe while preserving priority order
        seen: set[str] = set()
        deduped: list[SeedRepo] = []
        for s in seeds:
            if s.seed_id in seen:
                continue
            seen.add(s.seed_id)
            deduped.append(s)
        seeds = deduped
    else:
        seeds = remote_mine_seeds()

    inv_hist_raw = inventory.get("remote_language_histogram") or {}
    inv_hist = cast(dict[str, int], dict(inv_hist_raw)) if isinstance(inv_hist_raw, dict) else {}
    report = GitMineReport(
        mode="git_clone_only",
        target_candidates=target_candidates,
        work_root=str(work_root),
        inventory=inventory,
        language_inventory={str(k): int(v) for k, v in inv_hist.items()},
        language_kept={lang: 0 for lang in SCALE_LANGUAGES},
        notes=[
            "git history authority for all patches (VAL-MINE-007)",
            "Oxylabs not used in this path (http_metadata_source=none)",
            "copyleft licenses fail closed before clone",
            "M9 funnel: monorepo skip + gold dedupe + documented rejects + clone cache",
            f"parallel envbuild workers capped at {cfg.clamped_workers()} "
            f"(hygiene ceiling {MAX_CONCURRENT_ENVBUILD_JOBS}, hard max 24)",
        ],
        oxylabs_status="not_used",
        documented_skip_reasons=document_all_skip_reasons(),
    )

    clone_root = work_root / "clones"
    clone_root.mkdir(parents=True, exist_ok=True)
    cand_root = work_root / "candidates"
    cand_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        if report.keep_count >= target_candidates:
            funnel_report.add_skip(
                SkipReason(
                    code=SKIP_TARGET_SATURATED,
                    detail=(
                        f"target {target_candidates} already met; seed {seed.seed_id} not mined"
                    ),
                    stage="mine",
                    repo=seed.repo,
                )
            )
            break
        remaining = max(1, target_candidates - report.keep_count)
        per_seed_cap = min(max_keeps_per_seed, remaining)
        funnel_report.counters.considered += 1
        keeps, rejects, stats = mine_seed_merges(
            seed,
            dest_root=clone_root,
            max_merges=max_merges_per_seed,
            max_keeps=per_seed_cap,
            min_source_files=min_source_files,
            require_tests=require_tests,
            reuse_clone=reuse_clones,
            clone_cache=cache,
            funnel_config=cfg,
            funnel_report=funnel_report,
            seen_signatures=seen_signatures,
        )
        report.seed_stats.append(stats)
        for cand in keeps:
            if report.keep_count >= target_candidates:
                break
            report.kept.append(cand)
            lang = normalize_language(cand.language or seed.language)
            report.language_kept[lang] = report.language_kept.get(lang, 0) + 1
            if write_artifacts:
                write_candidate_artifacts(cand, cand_root)
        report.rejected.extend(rejects)

    # Honest under-supply for languages with inventory but zero keeps (or zero inventory)
    for lang in SCALE_LANGUAGES:
        kept_n = int(report.language_kept.get(lang, 0))
        inv_n = int(report.language_inventory.get(lang, 0))
        if kept_n == 0:
            if inv_n == 0:
                report.under_supply.append(
                    f"{lang}: inventory empty — no permissive modular remotes seeded "
                    "for this language; best-effort under-supply (not silent omission)."
                )
            else:
                report.under_supply.append(
                    f"{lang}: inventory={inv_n} but mine kept=0 after multi-file+tests+"
                    "license funnel (honest yield shortfall, not silent omission)."
                )

    # Invert also for keep-histogram vs inventory zeros already established in allowlist
    report.under_supply.extend(
        [r for r in under_supply_reasons(report.language_inventory) if r not in report.under_supply]
    )

    if report.keep_count < target_candidates:
        report.notes.append(
            f"under target: kept={report.keep_count} < target={target_candidates}; "
            "expand max_merges / max_keeps or allowlist depth (do not invent packs)."
        )

    if cache is not None:
        report.clone_cache_stats = dict(cache.stats_dict())
        funnel_report.counters.cache_hits = int(cache.stats.hits)
        funnel_report.counters.cache_misses = int(cache.stats.misses)
    funnel_report.counters.kept = report.keep_count
    report.funnel = funnel_report.to_dict()

    # Persist report for ship consumption
    report_path = work_root / "git_mine_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    funnel_report.write_json(work_root / "funnel_report.json")
    # candidates.jsonl for funnel inspection
    lines = [json.dumps(c.to_dict(), sort_keys=True) for c in report.kept]
    (work_root / "candidates.jsonl").write_text(
        ("\n".join(lines) + ("\n" if lines else "")),
        encoding="utf-8",
    )
    # language stats side-car
    lang_stats = {
        "language_inventory": report.language_inventory,
        "language_kept": report.language_kept,
        "under_supply": report.under_supply,
        "keep_count": report.keep_count,
        "target_candidates": report.target_candidates,
        "mode": report.mode,
        "oxylabs_status": report.oxylabs_status,
        "funnel_skip_tallies": funnel_report.counters.skip_by_code,
        "parallel_envbuild_workers": cfg.clamped_workers(),
    }
    (work_root / "language_stats.json").write_text(
        json.dumps(lang_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (work_root / "skip_reasons.json").write_text(
        json.dumps(
            {
                "catalog": document_all_skip_reasons(),
                "tallied": funnel_report.counters.skip_by_code,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def probe_oxylabs_live(
    *,
    url: str = "https://github.com/psf/requests",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """VAL-OXY-005 live probe: succeed only when OXYLABS_* set and fetch works.

    If credentials are missing, return ``status=blocked`` with evidence (do not
    fake pass). Never logs secrets.
    """
    import os

    from swe_factory.sources.oxylabs import (
        OxylabsAuthError,
        OxylabsClient,
        OxylabsError,
        has_oxylabs_credentials,
    )

    environ = env if env is not None else dict(os.environ)
    evidence: dict[str, Any] = {
        "assertion": "VAL-OXY-005",
        "url": url,
        "source": "universal",
        "credentials_present": has_oxylabs_credentials(environ),
        "status": "pending",
        "ok": False,
        "content_bytes": 0,
        "http_status": None,
        "reason": "",
        "produced_at": datetime.now(UTC).isoformat(),
    }
    if not evidence["credentials_present"]:
        evidence["status"] = "blocked"
        evidence["reason"] = (
            "OXYLABS_USERNAME/OXYLABS_PASSWORD unset; live GitHub HTTP probe "
            "blocked pending operator credentials. Do not invent a pass. "
            "Git-clone-only mining remains the default live path."
        )
        return evidence
    try:
        with OxylabsClient.from_env(env=environ) as client:
            result = client.scrape_url(url, source="universal")
        evidence["http_status"] = result.status_code
        evidence["content_bytes"] = len(result.content or "")
        evidence["job_id"] = result.job_id
        if result.ok or (result.content and len(result.content) > 0):
            evidence["status"] = "passed"
            evidence["ok"] = True
            evidence["reason"] = "live universal github fetch returned non-empty body"
        else:
            evidence["status"] = "failed"
            evidence["reason"] = f"empty content or non-OK status_code={result.status_code}"
    except OxylabsAuthError as exc:
        evidence["status"] = "blocked"
        evidence["reason"] = f"auth error (no secrets): {type(exc).__name__}"
    except OxylabsError as exc:
        evidence["status"] = "failed"
        evidence["reason"] = f"{type(exc).__name__}: transport/scrape failed (no secrets)"
    return evidence


__all__ = [
    "GitMineError",
    "GitMineReport",
    "MergeRange",
    "SeedMineStats",
    "clone_seed_repo",
    "list_merge_ranges",
    "list_non_merge_ranges",
    "mine_allowlist_git_only",
    "mine_seed_merges",
    "probe_oxylabs_live",
]
