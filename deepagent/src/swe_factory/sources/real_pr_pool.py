"""Real-PR candidate pool for M11+ (merged GitHub PRs only).

Product discovery for ``source_track=real_pr`` must:

1. Accept **merged GitHub PRs only** (not synthetic motors / hybrid_curated).
2. Require multi-file product source (≥2 non-test files) + ≥1 test path.
3. Pin ``base_commit`` to the PR base SHA (full 40-char when live).
4. Emit gold = source hunks only; test hunks → held-out ``test_patch``.
5. Fail closed on copyleft; never stage harbor motors as real_pr candidates.

Offline path builds a synthesized multi-repo pool from the offline PR fixture
(no network). Live path uses GitHub public REST (token optional, never logged).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from swe_factory.oracle.gates import MULTI_FILE_FLOOR
from swe_factory.producers.hard_filter import measure_source_hunk_count
from swe_factory.producers.pr_miner import (
    MergedPR,
    PrMineError,
    PrMiner,
    RealPrCandidate,
    offline_fixture_pr,
    produce_offline_fixture,
)
from swe_factory.schema import SourceTrack
from swe_factory.sources.allowlist import (
    HARBOR_MOTOR_SEEDS,
    SeedRepo,
    remote_mine_seeds,
)
from swe_factory.sources.discover import (
    DiscoverCandidate,
    DiscoverError,
    DiscoverReject,
    DiscoverReport,
    is_full_sha,
    is_real_repository_url,
    repository_url_for,
    sanitize_api_candidate,
    write_candidate_artifacts,
)
from swe_factory.sources.github import (
    DISCOVERY_PATH_LIST_PULLS,
    DISCOVERY_PATH_SEARCH,
    DISCOVERY_PATHS,
    GitHubClient,
    GitHubError,
    resolve_github_token,
)
from swe_factory.sources.license_gate import LicenseGateError, assert_permissive_license

# Engineering-only discovery label for offline_fixture ledger rows (never product N).
DISCOVERY_PATH_OFFLINE_FIXTURE = "offline_fixture"

# Stable candidates.jsonl schema fields (VAL-LMINE-003 / VAL-LX-003).
CANDIDATE_LEDGER_REQUIRED_FIELDS: tuple[str, ...] = (
    "repo",
    "pr_number",
    "base_sha",
    "language",
    "license",
    "discovery_path",
    "source_hunk_count",
    "source_file_count",
    "test_file_count",
    "disposition",
)

# ---------------------------------------------------------------------------
# Motor / hybrid markers (VAL-RPR-002 / VAL-RPR-005)
# ---------------------------------------------------------------------------

_MOTOR_REPO_MARKERS: tuple[str, ...] = (
    "fixtures/harbor_motors",
    "harbor_motors/",
    "fixtures/tiny_green",
    "fixtures/tiny_offline",
    "orderlib",
    "kvstore",
    "ts_registry",
    "python_orders",
    "go_kvstore",
)

_MOTOR_SEED_IDS: frozenset[str] = frozenset(s.seed_id for s in HARBOR_MOTOR_SEEDS)

_HYBRID_TRACK_ALIASES: frozenset[str] = frozenset(
    {
        "hybrid_curated",
        "hybrid",
        "motor",
        "harbor_motor",
        "synthetic_motor",
    }
)

# Prefer small / modular independent repos for the first wave (select ≥5 later).
# Explicit seed_ids kept small-to-medium and multi-lang.
DEFAULT_REAL_PR_SEED_IDS: tuple[str, ...] = (
    # Python-first dual-run-survivable surface (N base before multi-lang).
    "python_boltons",
    "python_cachetools",
    "python_zipp",
    "python_itsdangerous",
    "python_markupsafe",
    "python_click",
    "python_httpx",
    "python_packaging",
    "python_httpcore",
    "python_jinja",
    "python_flask",
    "python_werkzeug",
    "python_attrs",
    "python_urllib3",
    "python_idna",
    "python_more_itertools",
    "python_platformdirs",
    "python_blinker",
    "python_requests",
    "python_rich",
    "python_jsonschema",
    "python_tldextract",
    "python_itemadapter",
    "python_charset_normalizer",
    # Multi-lang diversity seeds (best-effort AFTER python N base).
    "go_cast",
    "go_uuid",
    "go_xid",
    "go_semver",
    "go_cleanhttp",
    "go_chi",
    "go_mapstructure",
    "go_multierror",
    "js_qs",
    "js_debug",
    "js_slash",
    "js_is_plain_obj",
    "js_ansi_styles",
    "js_validator",
    "js_chalk",
    "js_uuid",
    "ts_emittery",
    "ts_tslib",
    "ts_zod",
    "ts_type_fest",
    "rust_log",
    "rust_bitflags",
    "rust_byteorder",
    "rust_thiserror",
    "rust_anyhow",
)

PoolMode = Literal["offline_fixture", "live_github_rest"]


class RealPrPoolError(RuntimeError):
    """Raised when real_pr pool construction fails fatally."""


@dataclass(frozen=True, slots=True)
class MotorReject:
    """Audit row for a motor / hybrid rejection on the real_pr path."""

    identity: str
    reason_code: str
    detail: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RealPrPoolReport:
    """Candidate pool for later selection of ≥5 small real packs."""

    mode: PoolMode
    kept: list[RealPrCandidate | DiscoverCandidate] = field(default_factory=list)
    rejected: list[DiscoverReject | MotorReject] = field(default_factory=list)
    motor_rejects: list[MotorReject] = field(default_factory=list)
    repos_considered: list[str] = field(default_factory=list)
    source_track: str = SourceTrack.REAL_PR.value
    provider_calls: int = 0
    network_required: bool = False
    offline: bool = False
    min_source_files: int = MULTI_FILE_FLOOR
    target_candidates: int = 5
    notes: list[str] = field(default_factory=list)
    work_root: str = ""
    merged_only: bool = True
    hybrid_motors_allowed: bool = False
    # Durable candidates.jsonl rows (accept + honest reject); VAL-LMINE-003/LX-003
    ledger_rows: list[dict[str, Any]] = field(default_factory=list)
    # Explicit product-N honesty: offline_fixture never counts toward product N
    product_n_evidence: bool = False
    engineering_only: bool = False

    @property
    def keep_count(self) -> int:
        return len(self.kept)

    @property
    def reject_count(self) -> int:
        return len(self.rejected) + len(self.motor_rejects)

    @property
    def repo_diversity(self) -> int:
        repos: set[str] = set()
        for item in self.kept:
            repo = getattr(item, "repo", None) or (
                item.task.repo if hasattr(item, "task") else None
            )
            if repo:
                repos.add(str(repo))
        return len(repos)

    def to_dict(self) -> dict[str, Any]:
        kept_rows: list[dict[str, Any]] = []
        for item in self.kept:
            if isinstance(item, RealPrCandidate):
                task = item.task
                track = (
                    task.source_track.value
                    if hasattr(task.source_track, "value")
                    else str(task.source_track)
                )
                keep_row: dict[str, Any] = {
                    "kind": "task_record",
                    "instance_id": task.instance_id,
                    "source_track": track,
                    "repo": task.repo,
                    "repository_url": item.repository_url,
                    "base_commit": task.base_commit,
                    "base_sha": task.base_commit,
                    "pr_number": item.pr.number,
                    "gold_files": list(item.gold_files),
                    "test_files": list(item.pr.test_files),
                    "test_patch_nonempty": bool((item.test_patch or "").strip()),
                    "license": task.license,
                    "language": task.language,
                    "source_hunk_count": int(
                        item.provenance.get("source_hunk_count") or item.pr.source_hunk_count or 0
                    ),
                    "source_file_count": len(item.pr.source_files),
                    "test_file_count": len(item.pr.test_files),
                    "discovery_path": item.provenance.get("discovery_path"),
                    "product_n_evidence": self.product_n_evidence,
                    "engineering_only": self.engineering_only,
                }
                kept_rows.append(keep_row)
            else:
                row = item.to_dict()
                row.setdefault("base_sha", row.get("base_commit"))
                row.setdefault("product_n_evidence", self.product_n_evidence)
                row.setdefault("engineering_only", self.engineering_only)
                kept_rows.append(row)
        return {
            "ok": True,
            "mode": self.mode,
            "source_track": self.source_track,
            "keep_count": self.keep_count,
            "reject_count": self.reject_count,
            "repo_diversity": self.repo_diversity,
            "repos_considered": list(self.repos_considered),
            "provider_calls": self.provider_calls,
            "network_required": self.network_required,
            "offline": self.offline,
            "merged_only": self.merged_only,
            "hybrid_motors_allowed": self.hybrid_motors_allowed,
            "min_source_files": self.min_source_files,
            "target_candidates": self.target_candidates,
            "notes": list(self.notes),
            "work_root": self.work_root,
            "product_n_evidence": self.product_n_evidence,
            "engineering_only": self.engineering_only,
            "kept": kept_rows,
            "rejected": [r.to_dict() for r in self.rejected],
            "motor_rejects": [r.to_dict() for r in self.motor_rejects],
            "ledger_row_count": len(self.ledger_rows),
            "produced_at": datetime.now(UTC).isoformat(),
            "honesty": {
                "hybrid_as_real_pr_false_claim": False,
                "product_track": SourceTrack.REAL_PR.value,
                "motors_excluded": True,
                "product_n_from_offline_fixture": False,
                "product_n_evidence": self.product_n_evidence,
                "engineering_only": self.engineering_only,
            },
        }

    def as_discover_report(self) -> DiscoverReport:
        discover_keeps: list[DiscoverCandidate] = []
        discover_rejects: list[DiscoverReject] = []
        for item in self.kept:
            if isinstance(item, DiscoverCandidate):
                discover_keeps.append(item)
            elif isinstance(item, RealPrCandidate):
                discover_keeps.append(_candidate_from_real_pr(item))
        for row in self.rejected:
            if isinstance(row, DiscoverReject):
                discover_rejects.append(row)
        for motor in self.motor_rejects:
            discover_rejects.append(
                DiscoverReject(
                    repo=motor.identity,
                    reason_code=motor.reason_code,
                    detail=motor.detail,
                    meta=dict(motor.meta),
                )
            )
        return DiscoverReport(
            kept=tuple(discover_keeps),
            rejected=tuple(discover_rejects),
            provider_calls=self.provider_calls,
            network_required=self.network_required,
            offline=self.offline,
            history_authority="api_metadata_only" if not self.offline else "offline_fixture",
        )


def build_candidate_ledger_row(
    *,
    repo: str,
    pr_number: int | None,
    base_sha: str | None,
    language: str | None,
    license: str | None,
    discovery_path: str | None,
    source_hunk_count: int = 0,
    source_file_count: int = 0,
    test_file_count: int = 0,
    disposition: str = "accept",
    reason_code: str | None = None,
    reason_codes: Sequence[str] | None = None,
    detail: str | None = None,
    product_n_evidence: bool | None = None,
    engineering_only: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a durable candidates.jsonl row with the stable M14 schema.

    Live product rows must set ``discovery_path`` to ``search`` or ``list_pulls``.
    Offline engineering rows use ``offline_fixture`` and never claim product N.
    """
    path = (discovery_path or "").strip() or None
    if product_n_evidence is None:
        product_n_evidence = path in DISCOVERY_PATHS
    if engineering_only is None:
        engineering_only = path == DISCOVERY_PATH_OFFLINE_FIXTURE or path not in DISCOVERY_PATHS
    row: dict[str, Any] = {
        "repo": repo,
        "pr_number": pr_number,
        "base_sha": base_sha or "",
        "base_commit": base_sha or "",
        "language": language or "",
        "license": license or "",
        "discovery_path": path,
        "source_hunk_count": int(source_hunk_count or 0),
        "source_file_count": int(source_file_count or 0),
        "test_file_count": int(test_file_count or 0),
        "disposition": disposition,
        "reason_code": reason_code,
        "reason_codes": list(reason_codes or ([reason_code] if reason_code else [])),
        "detail": detail or "",
        "product_n_evidence": bool(product_n_evidence),
        "engineering_only": bool(engineering_only),
    }
    if extra:
        row.update(dict(extra))
    return row


def write_candidates_jsonl(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> Path:
    """Write durable candidates.jsonl (one JSON object per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), sort_keys=True) for row in rows]
    out.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
    return out


def candidate_ledger_row_from_real_pr(
    item: RealPrCandidate,
    *,
    discovery_path: str | None,
    disposition: str = "accept",
    product_n_evidence: bool = True,
    engineering_only: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a RealPrCandidate keep into the stable candidates.jsonl schema."""
    pr = item.pr
    task = item.task
    hunk_count = int(
        item.provenance.get("source_hunk_count")
        or pr.source_hunk_count
        or measure_source_hunk_count(pr.files)
        or 0
    )
    path = discovery_path or item.provenance.get("discovery_path")
    row = build_candidate_ledger_row(
        repo=task.repo,
        pr_number=pr.number,
        base_sha=task.base_commit,
        language=task.language,
        license=task.license or pr.license,
        discovery_path=str(path) if path else None,
        source_hunk_count=hunk_count,
        source_file_count=len(pr.source_files),
        test_file_count=len(pr.test_files),
        disposition=disposition,
        product_n_evidence=product_n_evidence,
        engineering_only=engineering_only,
        extra={
            "instance_id": task.instance_id,
            "repository_url": item.repository_url,
            "gold_files": list(item.gold_files),
            "test_files": list(pr.test_files),
            "source_track": item.source_track,
            "kind": "task_record",
            "test_patch_nonempty": bool((item.test_patch or "").strip()),
            "merge_commit_sha": pr.merge_commit_sha,
            **(dict(extra) if extra else {}),
        },
    )
    return row


def is_motor_or_hybrid_identity(
    identity: str | None,
    *,
    source_track: str | None = None,
    seed_id: str | None = None,
    kind: str | None = None,
) -> tuple[bool, str]:
    """Return (is_banned, reason) for product real_pr mine paths.

    Motors / hybrid_curated / fixture harbor trees are never real_pr candidates.
    """
    track = (source_track or "").strip().lower()
    if track in _HYBRID_TRACK_ALIASES:
        return True, f"source_track={track!r} is hybrid/motor; product path requires real_pr"
    kind_n = (kind or "").strip().lower()
    if kind_n in {"curated", "hybrid", "motor", "hybrid_curated"}:
        return True, f"kind={kind_n!r} is not allowed on real_pr-only product path"
    if seed_id and seed_id in _MOTOR_SEED_IDS:
        return True, f"seed_id={seed_id!r} is a harbor motor seed; excluded from real_pr pool"
    if seed_id and seed_id.startswith("harbor_"):
        return True, f"seed_id={seed_id!r} looks like a harbor motor"
    raw = (identity or "").strip().lower()
    if not raw:
        return False, ""
    if any(marker in raw for marker in _MOTOR_REPO_MARKERS):
        return True, f"identity {identity!r} matches motor/fixture marker"
    if raw.startswith("file://"):
        return True, f"file:// identity {identity!r} is not a public PR source"
    if "fixtures/" in raw or raw.startswith("fixture"):
        # offline fixture pool uses fixtures/tiny_real_pr intentionally via special path
        if "tiny_real_pr" in raw:
            return False, ""
        return True, f"fixture path {identity!r} cannot be a product real_pr candidate"
    return False, ""


def assert_not_motor_or_hybrid(
    identity: str | None,
    *,
    source_track: str | None = None,
    seed_id: str | None = None,
    kind: str | None = None,
) -> None:
    """Raise RealPrPoolError when identity/track is a banned motor or hybrid."""
    banned, reason = is_motor_or_hybrid_identity(
        identity,
        source_track=source_track,
        seed_id=seed_id,
        kind=kind,
    )
    if banned:
        raise RealPrPoolError(f"motor_or_hybrid_rejected: {reason}")


def real_pr_seed_pool(
    *,
    seed_ids: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    max_seeds: int | None = None,
) -> list[SeedRepo]:
    """Return allowlisted public remotes suitable for real_pr merged-PR mining.

    Explicitly excludes :data:`HARBOR_MOTOR_SEEDS` and fixture-only seeds.
    """
    wanted = {s.strip() for s in (seed_ids or DEFAULT_REAL_PR_SEED_IDS) if s and s.strip()}
    seeds: list[SeedRepo] = []
    for seed in remote_mine_seeds():
        if seed.seed_id not in wanted and seed_ids is None:
            # default path: only names in DEFAULT_REAL_PR_SEED_IDS
            if seed.seed_id not in DEFAULT_REAL_PR_SEED_IDS:
                continue
        elif seed_ids is not None and seed.seed_id not in wanted:
            continue
        banned, _ = is_motor_or_hybrid_identity(seed.repo, seed_id=seed.seed_id)
        if banned:
            continue
        if languages is not None:
            langs = {str(x).strip().lower() for x in languages if x}
            if seed.language not in langs:
                continue
        # Prefer seeds that already prove real HTTPS + full pin form
        if not is_real_repository_url(seed.repository_url):
            continue
        if not is_full_sha(seed.base_commit):
            # base_commit on allowlist is inventory HEAD, not PR base; still
            # require hex pin form so motors (a1000…) stay out.
            continue
        seeds.append(seed)
        if max_seeds is not None and len(seeds) >= max_seeds:
            break
    # Stable priority order (mine_priority then id)
    seeds.sort(key=lambda s: (s.mine_priority, s.seed_id))
    return seeds


def candidate_from_merged_pr(
    pr: MergedPR,
    *,
    work_root: Path | None = None,
    run_stub_oracle: bool = True,
) -> RealPrCandidate:
    """Produce a real_pr TaskRecord from an already-selected MergedPR."""
    assert_not_motor_or_hybrid(pr.repo, source_track=SourceTrack.REAL_PR.value)
    assert_permissive_license(pr.license or "MIT", repo=pr.repo)
    from swe_factory.sources.github import DictGitHubTransport

    # Offline construction: miner only needs client for live fetch; produce is pure.
    client = GitHubClient(transport=DictGitHubTransport(routes={}))
    miner = PrMiner(client=client, work_root=work_root, license=pr.license or "MIT")
    return miner.produce(
        pr,
        instance_suffix=f"pool{uuid.uuid4().hex[:6]}",
        run_stub_oracle=run_stub_oracle,
        source_track=SourceTrack.REAL_PR,
    )


def _candidate_from_real_pr(item: RealPrCandidate) -> DiscoverCandidate:
    """Project RealPrCandidate → DiscoverCandidate for funnel consumers."""
    task = item.task
    return DiscoverCandidate(
        candidate_id=task.instance_id,
        kind="real_pr",
        repository_url=item.repository_url or repository_url_for(task.repo),
        repo=task.repo,
        base_commit=task.base_commit,
        language=task.language,
        license=task.license,
        title=item.pr.title,
        problem_statement=task.problem_statement,
        source_files=tuple(item.pr.source_files),
        test_files=tuple(item.pr.test_files),
        gold_patch=task.gold_patch,
        test_patch=item.test_patch or "",
        gold_files=tuple(item.gold_files),
        history_authority="api_metadata_only",
        http_metadata_source="github_api",
        merge_commit_sha=item.pr.merge_commit_sha,
        pr_number=item.pr.number,
        html_url=item.pr.html_url,
        meta={
            "source_track": SourceTrack.REAL_PR.value,
            "producer": "real_pr_pool",
            "gold_provenance": item.provenance.get("gold_provenance"),
            "merged_only": True,
            "hybrid_motors_allowed": False,
        },
    )


def mine_offline_real_pr_pool(
    *,
    work_root: Path | None = None,
    target_candidates: int = 5,
    synthetic_repo_diversity: int = 5,
) -> RealPrPoolReport:
    """Offline multi-candidate pool from the real_pr fixture (no network).

    Emits ≥``target_candidates`` labeled real_pr TaskRecords with distinct
    synthetic owner/repo ids so later ship can *select* ≥5 distinct identities.
    Never includes hybrid motors.

    **Engineering-only:** mode=offline_fixture never counts as product N evidence
    (VAL-LMINE-007 / VAL-LX-003). Ledger rows use discovery_path=offline_fixture
    and never search|list_pulls.
    """
    if target_candidates < 1:
        raise RealPrPoolError("target_candidates must be ≥1")
    diversity = max(target_candidates, synthetic_repo_diversity)
    root = Path(work_root) if work_root is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)

    report = RealPrPoolReport(
        mode="offline_fixture",
        target_candidates=target_candidates,
        network_required=False,
        offline=True,
        provider_calls=0,
        work_root=str(root) if root else "",
        product_n_evidence=False,
        engineering_only=True,
        notes=[
            "offline real_pr fixture pool (VAL-MINE-004 / VAL-RPR-002 wiring)",
            "source_track=real_pr only; hybrid motors excluded",
            "synthetic owner diversity for selection floor only; not live commits",
            "engineering_only: offline_fixture is NOT product N / live mine evidence",
        ],
    )

    # Refuse motors if accidentally mixed in.
    for motor in HARBOR_MOTOR_SEEDS:
        banned, reason = is_motor_or_hybrid_identity(motor.repo, seed_id=motor.seed_id)
        if banned:
            report.motor_rejects.append(
                MotorReject(
                    identity=motor.seed_id,
                    reason_code="motor_or_hybrid_rejected",
                    detail=reason,
                    meta={"repo": motor.repo, "track": "hybrid_curated"},
                )
            )

    base_pr = offline_fixture_pr()
    # Produce diverse synthetic repos: fixtures/tiny_real_pr__N under distinct names
    # that still pass multi-file/tests filter when re-seeded as MergedPR.
    for i in range(diversity):
        if report.keep_count >= target_candidates and report.repo_diversity >= min(
            5, target_candidates
        ):
            break
        # Use firmed-up title indices that still look PR-numbered
        bacteria = [
            "acme",
            "bolt",
            "cache",
            "delta",
            "echo",
            "flux",
            "gamma",
            "helix",
            "ion",
            "jade",
        ]
        owner = bacteria[i % len(bacteria)]
        name = f"lib{i + 1:02d}"
        repo = f"{owner}/{name}"
        report.repos_considered.append(repo)
        pr = MergedPR(
            repo=repo,
            number=1000 + i,
            title=base_pr.title,
            body=base_pr.body,
            # Distinct 40-char hex pins (offline labels, not live git objects).
            base_commit=f"{i + 1:040x}",
            merge_commit_sha=f"{0xFFFFFF - i:040x}"[-40:],
            language=base_pr.language,
            html_url=f"https://github.com/{repo}/pull/{1000 + i}",
            files=base_pr.files,
            license="MIT",
        )
        try:
            assert_permissive_license(pr.license, repo=repo)
            novelty_root = (root / "cases") if root is not None else None
            candidate = candidate_from_merged_pr(
                pr,
                work_root=novelty_root,
                run_stub_oracle=True,
            )
        except (PrMineError, RealPrPoolError, LicenseGateError, DiscoverError) as exc:
            report.rejected.append(
                DiscoverReject(
                    repo=repo,
                    reason_code="offline_pool_reject",
                    detail=str(exc)[:400],
                )
            )
            continue
        # Strict: real_pr track only
        if candidate.source_track != SourceTrack.REAL_PR.value:
            report.rejected.append(
                DiscoverReject(
                    repo=repo,
                    reason_code="source_track_not_real_pr",
                    detail=f"got {candidate.source_track!r}",
                )
            )
            continue
        if not candidate.test_patch.strip():
            report.rejected.append(
                DiscoverReject(
                    repo=repo,
                    reason_code="test_patch_empty",
                    detail="held-out test.patch required for real_pr product path",
                )
            )
            continue
        if len(candidate.gold_files) < MULTI_FILE_FLOOR:
            report.rejected.append(
                DiscoverReject(
                    repo=repo,
                    reason_code="multi_file_floor_rejected",
                    detail=f"gold_files={list(candidate.gold_files)}",
                )
            )
            continue
        # Stamp offline discovery_path so ledger never claims live search|list_pulls
        candidate.provenance["discovery_path"] = DISCOVERY_PATH_OFFLINE_FIXTURE
        candidate.provenance["product_n_evidence"] = False
        candidate.provenance["engineering_only"] = True
        report.kept.append(candidate)
        report.ledger_rows.append(
            candidate_ledger_row_from_real_pr(
                candidate,
                discovery_path=DISCOVERY_PATH_OFFLINE_FIXTURE,
                product_n_evidence=False,
                engineering_only=True,
            )
        )
        if root is not None:
            disc = _candidate_from_real_pr(candidate)
            write_candidate_artifacts(disc, root / "candidates")

    if report.keep_count < target_candidates:
        report.notes.append(f"under target offline: kept={report.keep_count} < {target_candidates}")
    if report.repo_diversity < min(5, target_candidates):
        report.notes.append(
            f"repo_diversity={report.repo_diversity} below select-5 floor "
            f"(offline synthetic diversity may need re-run)"
        )

    # Always include one pure offline fixture candidate row as the "known-good" poison pill absence
    try:
        fixture_cand = produce_offline_fixture(
            work_root=(root / "fixture_case") if root is not None else None,
            run_stub_oracle=True,
        )
        # Only append if under target or to prove fixture still green
        if all(
            not (
                isinstance(k, RealPrCandidate)
                and k.task.instance_id == fixture_cand.task.instance_id
            )
            for k in report.kept
        ):
            # Don't count fixture id toward diversity of public repos; still a valid TaskRecord
            report.notes.append(
                f"offline fixture TaskRecord ok: {fixture_cand.task.instance_id} "
                f"(source_track={fixture_cand.source_track})"
            )
            fixture_cand.provenance["discovery_path"] = DISCOVERY_PATH_OFFLINE_FIXTURE
            fixture_cand.provenance["product_n_evidence"] = False
            fixture_cand.provenance["engineering_only"] = True
            report.kept.append(fixture_cand)
            report.ledger_rows.append(
                candidate_ledger_row_from_real_pr(
                    fixture_cand,
                    discovery_path=DISCOVERY_PATH_OFFLINE_FIXTURE,
                    product_n_evidence=False,
                    engineering_only=True,
                )
            )
    except PrMineError as exc:
        report.notes.append(f"offline fixture produce failed: {exc}")

    # Append honest reject ledger rows (engineering_only)
    for row in report.rejected:
        if isinstance(row, DiscoverReject):
            report.ledger_rows.append(
                build_candidate_ledger_row(
                    repo=row.repo,
                    pr_number=row.pr_number,
                    base_sha=row.base_commit,
                    language=None,
                    license=None,
                    discovery_path=DISCOVERY_PATH_OFFLINE_FIXTURE,
                    disposition="reject",
                    reason_code=row.reason_code,
                    detail=row.detail,
                    product_n_evidence=False,
                    engineering_only=True,
                )
            )

    if root is not None:
        _write_pool_report(report, root)
    return report


def mine_live_merged_pr_pool(
    client: GitHubClient,
    *,
    work_root: Path | None = None,
    seed_ids: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    target_candidates: int = 5,
    max_scan_per_repo: int = 80,
    max_keeps_per_repo: int = 8,
    max_seeds: int | None = 36,
    discovery_paths: Sequence[str] | None = None,
    product_mode: bool = True,
    require_token: bool = False,
    token: str | None = None,
) -> RealPrPoolReport:
    """Live merged-PR pool via GitHub REST list_pulls and/or Search.

    Never accepts harbor motors. License gate + multi-file filter applied per PR.
    Durable ``candidates.jsonl`` rows label ``discovery_path=search|list_pulls``
    (VAL-LMINE-003 / VAL-LX-003). Live path is product-N evidence.

    When ``require_token`` is True (CLI --live), fail closed unless a GitHub
    token is resolvable (GITHUB_TOKEN|GH_TOKEN|gh auth token).
    """
    if require_token and not resolve_github_token(token):
        raise RealPrPoolError(
            "live real-pr-pool requires network+token "
            "(set GITHUB_TOKEN / GH_TOKEN or run `gh auth token`); "
            "refusing unauth mass mine as product N evidence"
        )

    seeds = real_pr_seed_pool(seed_ids=seed_ids, languages=languages, max_seeds=max_seeds)
    if not seeds:
        raise RealPrPoolError("no real_pr seeds available after motor/hybrid exclusion")

    paths = (
        list(discovery_paths)
        if discovery_paths is not None
        else [
            DISCOVERY_PATH_LIST_PULLS,
            DISCOVERY_PATH_SEARCH,
        ]
    )
    # Normalize + validate discovery paths (product live only)
    clean_paths: list[str] = []
    for p in paths:
        p_norm = str(p).strip()
        if p_norm not in DISCOVERY_PATHS:
            raise RealPrPoolError(
                f"invalid live discovery_path={p_norm!r}; must be one of {sorted(DISCOVERY_PATHS)}"
            )
        if p_norm not in clean_paths:
            clean_paths.append(p_norm)
    if not clean_paths:
        clean_paths = [DISCOVERY_PATH_LIST_PULLS, DISCOVERY_PATH_SEARCH]

    root = Path(work_root) if work_root is not None else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)

    report = RealPrPoolReport(
        mode="live_github_rest",
        target_candidates=target_candidates,
        network_required=True,
        offline=False,
        work_root=str(root) if root else "",
        product_n_evidence=True,
        engineering_only=False,
        notes=[
            "live merged PRs only (VAL-RPR-002 / VAL-LMINE-003)",
            "base_commit = PR base.sha (full 40-char)",
            "gold=source hunks; test_patch=test hunks",
            "hybrid motors excluded from seed inventory",
            f"discovery_paths={clean_paths}",
            "candidates.jsonl durable ledger with discovery_path labels",
            "GITHUB_TOKEN used when present (never logged)",
        ],
    )
    miner = PrMiner(client=client, work_root=root, product_mode=product_mode)
    provider_calls = 0
    # Dedupe keeps by (repo, pr_number) across discovery paths
    seen_prs: set[tuple[str, int]] = set()

    def _record_reject(
        *,
        repo: str,
        reason_code: str,
        detail: str,
        repository_url: str | None = None,
        base_commit: str | None = None,
        pr_number: int | None = None,
        discovery_path: str,
        language: str | None = None,
        license: str | None = None,
        source_hunk_count: int = 0,
        source_file_count: int = 0,
        test_file_count: int = 0,
    ) -> None:
        report.rejected.append(
            DiscoverReject(
                repo=repo,
                reason_code=reason_code,
                detail=detail[:400],
                repository_url=repository_url or repository_url_for(repo),
                base_commit=base_commit,
                pr_number=pr_number,
                meta={"discovery_path": discovery_path},
            )
        )
        report.ledger_rows.append(
            build_candidate_ledger_row(
                repo=repo,
                pr_number=pr_number,
                base_sha=base_commit,
                language=language,
                license=license,
                discovery_path=discovery_path,
                source_hunk_count=source_hunk_count,
                source_file_count=source_file_count,
                test_file_count=test_file_count,
                disposition="reject",
                reason_code=reason_code,
                detail=detail[:400],
                product_n_evidence=True,
                engineering_only=False,
            )
        )

    def _try_produce(
        pr: MergedPR,
        *,
        seed: SeedRepo,
        discovery_path: str,
        kept_for_repo: int,
    ) -> int:
        nonlocal provider_calls
        if report.keep_count >= target_candidates or kept_for_repo >= max_keeps_per_repo:
            return kept_for_repo
        key = (pr.repo.lower(), int(pr.number))
        if key in seen_prs:
            return kept_for_repo
        try:
            assert_not_motor_or_hybrid(pr.repo)
            if not is_full_sha(pr.base_commit):
                raise PrMineError(
                    f"live PR {pr.repo}#{pr.number} base_commit not full SHA ({pr.base_commit!r})"
                )
            pr_fixed = MergedPR(
                repo=pr.repo,
                number=pr.number,
                title=pr.title,
                body=pr.body,
                base_commit=pr.base_commit,
                merge_commit_sha=pr.merge_commit_sha,
                language=seed.language or pr.language,
                html_url=pr.html_url,
                files=pr.files,
                license=seed.license or pr.license,
                merged_at=pr.merged_at,
                source_hunk_count=pr.source_hunk_count,
            )
            candidate = miner.produce(
                pr_fixed,
                instance_suffix=f"live{uuid.uuid4().hex[:6]}",
                run_stub_oracle=True,
                source_track=SourceTrack.REAL_PR,
            )
        except (PrMineError, RealPrPoolError, LicenseGateError) as query_exc:
            _record_reject(
                repo=seed.repo,
                reason_code="pr_produce_rejected",
                detail=str(query_exc),
                repository_url=seed.repository_url,
                base_commit=getattr(pr, "base_commit", None),
                pr_number=getattr(pr, "number", None),
                discovery_path=discovery_path,
                language=seed.language,
                license=seed.license,
                source_hunk_count=int(getattr(pr, "source_hunk_count", 0) or 0),
                source_file_count=len(getattr(pr, "source_files", ()) or ()),
                test_file_count=len(getattr(pr, "test_files", ()) or ()),
            )
            return kept_for_repo
        if candidate.source_track != SourceTrack.REAL_PR.value:
            _record_reject(
                repo=seed.repo,
                reason_code="source_track_not_real_pr",
                detail=f"got {candidate.source_track!r}",
                pr_number=pr.number,
                discovery_path=discovery_path,
            )
            return kept_for_repo
        candidate.provenance["discovery_path"] = discovery_path
        candidate.provenance["product_n_evidence"] = True
        candidate.provenance["engineering_only"] = False
        report.kept.append(candidate)
        report.ledger_rows.append(
            candidate_ledger_row_from_real_pr(
                candidate,
                discovery_path=discovery_path,
                product_n_evidence=True,
                engineering_only=False,
            )
        )
        seen_prs.add(key)
        kept_for_repo += 1
        if root is not None:
            write_candidate_artifacts(_candidate_from_real_pr(candidate), root / "candidates")
        return kept_for_repo

    for seed in seeds:
        if report.keep_count >= target_candidates:
            break
        banned, reason = is_motor_or_hybrid_identity(seed.repo, seed_id=seed.seed_id)
        if banned:
            report.motor_rejects.append(
                MotorReject(
                    identity=seed.seed_id,
                    reason_code="motor_or_hybrid_rejected",
                    detail=reason,
                    meta={"repo": seed.repo},
                )
            )
            continue
        try:
            assert_permissive_license(seed.license, repo=seed.repo)
        except LicenseGateError as exc:
            _record_reject(
                repo=seed.repo,
                reason_code=getattr(exc, "reason_code", None) or "license_rejected",
                detail=str(exc)[:400],
                repository_url=seed.repository_url,
                discovery_path=clean_paths[0],
                language=seed.language,
                license=seed.license,
            )
            continue

        report.repos_considered.append(seed.repo)
        kept_for_repo = 0

        # --- list_pulls discovery path ---
        if DISCOVERY_PATH_LIST_PULLS in clean_paths and report.keep_count < target_candidates:
            try:
                prs = miner.list_candidate_prs(
                    seed.repo,
                    max_scan=max_scan_per_repo,
                    max_keep=max_keeps_per_repo,
                    language=seed.language,
                )
                provider_calls += max(1, max_scan_per_repo // 30) + len(prs) * 2
            except (PrMineError, GitHubError) as exc:
                _record_reject(
                    repo=seed.repo,
                    reason_code=getattr(exc, "reason_code", None) or "github_api_failed",
                    detail=str(exc)[:400],
                    repository_url=seed.repository_url,
                    discovery_path=DISCOVERY_PATH_LIST_PULLS,
                    language=seed.language,
                    license=seed.license,
                )
                prs = []

            for pr in prs:
                if report.keep_count >= target_candidates or kept_for_repo >= max_keeps_per_repo:
                    break
                kept_for_repo = _try_produce(
                    pr,
                    seed=seed,
                    discovery_path=DISCOVERY_PATH_LIST_PULLS,
                    kept_for_repo=kept_for_repo,
                )

        # --- search discovery path ---
        if DISCOVERY_PATH_SEARCH in clean_paths and report.keep_count < target_candidates:
            if kept_for_repo >= max_keeps_per_repo:
                continue
            try:
                items = client.search_merged_pull_requests(
                    language=seed.language,
                    repo=seed.repo,
                    per_page=min(30, max_scan_per_repo),
                )
                provider_calls += 1 + len(items) * 2
            except (GitHubError, PrMineError) as exc:
                _record_reject(
                    repo=seed.repo,
                    reason_code=getattr(exc, "reason_code", None) or "search_failed",
                    detail=str(exc)[:400],
                    repository_url=seed.repository_url,
                    discovery_path=DISCOVERY_PATH_SEARCH,
                    language=seed.language,
                    license=seed.license,
                )
                items = []

            for item in items:
                if report.keep_count >= target_candidates or kept_for_repo >= max_keeps_per_repo:
                    break
                number = int(item.get("number") or 0)
                if number <= 0:
                    continue
                # Search Issues returns PR-like items; fetch full PR for base SHA + files
                try:
                    pr = miner.fetch_merged_pr(
                        seed.repo,
                        number,
                        language=seed.language,
                        license=seed.license,
                        product_hard_filter=product_mode,
                    )
                    provider_calls += 2
                except (PrMineError, GitHubError) as fetch_exc:
                    _record_reject(
                        repo=seed.repo,
                        reason_code=getattr(fetch_exc, "reason_code", None) or "pr_fetch_rejected",
                        detail=str(fetch_exc)[:400],
                        repository_url=seed.repository_url,
                        pr_number=number,
                        discovery_path=DISCOVERY_PATH_SEARCH,
                        language=seed.language,
                        license=seed.license,
                    )
                    continue
                kept_for_repo = _try_produce(
                    pr,
                    seed=seed,
                    discovery_path=DISCOVERY_PATH_SEARCH,
                    kept_for_repo=kept_for_repo,
                )

    report.provider_calls = provider_calls
    if report.keep_count < target_candidates:
        report.notes.append(
            f"under target live: kept={report.keep_count} < {target_candidates}; "
            "expand max_scan / seeds or set GITHUB_TOKEN for rate headroom "
            "(do not invent PRs or promote motors)."
        )
    if root is not None:
        _write_pool_report(report, root)
    return report


def mine_real_pr_pool(
    *,
    work_root: Path | None = None,
    offline: bool = True,
    client: GitHubClient | None = None,
    target_candidates: int = 5,
    seed_ids: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    max_scan_per_repo: int = 80,
    max_keeps_per_repo: int = 8,
    max_seeds: int | None = 36,
    discovery_paths: Sequence[str] | None = None,
    product_mode: bool = True,
    require_token: bool = False,
    token: str | None = None,
) -> RealPrPoolReport:
    """Convenience entry: offline fixture pool or live REST/Search pool."""
    if offline:
        return mine_offline_real_pr_pool(
            work_root=work_root,
            target_candidates=target_candidates,
        )
    if require_token and not resolve_github_token(token):
        raise RealPrPoolError(
            "live real-pr-pool requires network+token "
            "(set GITHUB_TOKEN / GH_TOKEN or run `gh auth token`); "
            "refusing unauth mass mine as product N evidence"
        )
    if client is None:
        client = GitHubClient.from_env(token=token)
    return mine_live_merged_pr_pool(
        client,
        work_root=work_root,
        seed_ids=seed_ids,
        languages=languages,
        target_candidates=target_candidates,
        max_scan_per_repo=max_scan_per_repo,
        max_keeps_per_repo=max_keeps_per_repo,
        max_seeds=max_seeds,
        discovery_paths=discovery_paths,
        product_mode=product_mode,
        require_token=require_token,
        token=token,
    )


def reject_hybrid_motor_discover_attempt(
    *,
    repo: str,
    repository_url: str | None = None,
    base_commit: str | None = None,
    source_track: str | None = None,
    kind: str | None = None,
    seed_id: str | None = None,
) -> DiscoverReject | None:
    """Return a DiscoverReject when hybrid/motor markers appear; else None."""
    for identity in (repo, repository_url or "", seed_id or ""):
        banned, reason = is_motor_or_hybrid_identity(
            identity,
            source_track=source_track,
            seed_id=seed_id,
            kind=kind,
        )
        if banned:
            return DiscoverReject(
                repo=repo,
                reason_code="motor_or_hybrid_rejected",
                detail=reason,
                repository_url=repository_url or repository_url_for(repo),
                base_commit=base_commit,
                meta={
                    "honesty": "hybrid never claimed as real_pr",
                    "product_track_required": SourceTrack.REAL_PR.value,
                },
            )
    return None


def sanitize_real_pr_api_candidate(
    *,
    repo: str,
    base_commit: str,
    files: Sequence[Any],
    license: str,
    repository_url: str | None = None,
    language: str | None = None,
    title: str = "",
    body: str = "",
    pr_number: int | None = None,
    html_url: str = "",
    merge_commit_sha: str | None = None,
    min_source_files: int = MULTI_FILE_FLOOR,
    require_tests: bool = True,
) -> DiscoverCandidate:
    """Discover sanitizer for real_pr product path (merged PR APIfixture payloads).

    Always forces kind=real_pr; refuses motors/hybrid tracks.
    """
    reject = reject_hybrid_motor_discover_attempt(
        repo=repo,
        repository_url=repository_url,
        base_commit=base_commit,
        source_track=SourceTrack.REAL_PR.value,
        kind="real_pr",
    )
    if reject is not None:
        raise DiscoverError(reject.detail)
    assert_not_motor_or_hybrid(repo)
    if repository_url:
        assert_not_motor_or_hybrid(repository_url)

    candidate = sanitize_api_candidate(
        repo=repo,
        base_commit=base_commit,
        files=files,
        license=license,
        repository_url=repository_url,
        language=language,
        title=title,
        body=body,
        pr_number=pr_number,
        html_url=html_url,
        merge_commit_sha=merge_commit_sha,
        min_source_files=min_source_files,
        require_tests=require_tests,
        kind="real_pr",
        history_authority="api_metadata_only",
        http_metadata_source="github_api",
    )
    # Stamp real_pr honesty meta
    meta = dict(candidate.meta)
    meta.update(
        {
            "source_track": SourceTrack.REAL_PR.value,
            "merged_only": True,
            "hybrid_motors_allowed": False,
            "gold_source_only": True,
        }
    )
    return DiscoverCandidate(
        candidate_id=candidate.candidate_id,
        kind="real_pr",
        repository_url=candidate.repository_url,
        repo=candidate.repo,
        base_commit=candidate.base_commit,
        language=candidate.language,
        license=candidate.license,
        title=candidate.title,
        problem_statement=candidate.problem_statement,
        source_files=candidate.source_files,
        test_files=candidate.test_files,
        gold_patch=candidate.gold_patch,
        test_patch=candidate.test_patch,
        gold_files=candidate.gold_files,
        history_authority=candidate.history_authority,
        http_metadata_source=candidate.http_metadata_source,
        merge_commit_sha=candidate.merge_commit_sha,
        pr_number=candidate.pr_number,
        html_url=candidate.html_url,
        reject_reason=None,
        meta=meta,
    )


def _write_pool_report(report: RealPrPoolReport, root: Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    (root / "real_pr_pool_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Prefer durable ledger_rows (accept + reject with discovery_path schema).
    # Fall back to kept rows when ledger was not populated (legacy callers).
    if report.ledger_rows:
        write_candidates_jsonl(report.ledger_rows, root / "candidates.jsonl")
    else:
        lines: list[str] = []
        for row in payload.get("kept") or []:
            # Backfill base_sha alias for schema stability
            if "base_sha" not in row and row.get("base_commit"):
                row = dict(row)
                row["base_sha"] = row.get("base_commit")
            lines.append(json.dumps(row, sort_keys=True))
        (root / "candidates.jsonl").write_text(
            ("\n".join(lines) + ("\n" if lines else "")),
            encoding="utf-8",
        )
    # Tasks.jsonl for RealPrCandidate keeps
    task_lines: list[str] = []
    for item in report.kept:
        if isinstance(item, RealPrCandidate):
            task_lines.append(item.task.model_dump_json())
    if task_lines:
        (root / "tasks.jsonl").write_text("\n".join(task_lines) + "\n", encoding="utf-8")


def summary_seed_pool_for_select5() -> dict[str, Any]:
    """Inventory of seeds suitable to later select ≥5 small repos for cert."""
    seeds = real_pr_seed_pool()
    by_lang: dict[str, int] = {}
    for seed in seeds:
        by_lang[seed.language] = by_lang.get(seed.language, 0) + 1
    return {
        "seed_count": len(seeds),
        "meets_select5_inventory": len(seeds) >= 5,
        "languages": by_lang,
        "seed_ids": [s.seed_id for s in seeds],
        "repos": [s.repo for s in seeds],
        "motors_excluded": True,
        "motor_seed_ids_blocked": sorted(_MOTOR_SEED_IDS),
        "merged_prs_only": True,
        "default_target_candidates": 5,
        "notes": (
            "Inventory only — livemine still requires multi-file merged PRs per repo. "
            "Offline pool uses fixture diversity when REST is rate-limited."
        ),
    }


__all__ = [
    "CANDIDATE_LEDGER_REQUIRED_FIELDS",
    "DEFAULT_REAL_PR_SEED_IDS",
    "DISCOVERY_PATH_OFFLINE_FIXTURE",
    "MotorReject",
    "RealPrPoolError",
    "RealPrPoolReport",
    "assert_not_motor_or_hybrid",
    "build_candidate_ledger_row",
    "candidate_from_merged_pr",
    "candidate_ledger_row_from_real_pr",
    "is_motor_or_hybrid_identity",
    "mine_live_merged_pr_pool",
    "mine_offline_real_pr_pool",
    "mine_real_pr_pool",
    "real_pr_seed_pool",
    "reject_hybrid_motor_discover_attempt",
    "sanitize_real_pr_api_candidate",
    "summary_seed_pool_for_select5",
    "write_candidates_jsonl",
]
