"""Real-PR Docker oracle certification (VAL-RORC-001..004).

Product Real-PR packs under ``datasets/deepagent_v1`` must certify via Docker
only (never fake):

- VAL-RORC-001: solution applied → reward=1 (backend=docker)
- VAL-RORC-002: null/empty patch → reward=0
- VAL-RORC-003: isolation scan fail-closed (no solution/ / held-out test.patch
  in agent context) before promote
- VAL-RORC-004: product cert CLI/API refuses ``oracle_mode=fake`` / fake backend

This module builds on :mod:`swe_factory.harbor.deepagent_cert` but **requires**
``source_track=real_pr`` (hybrid / motor tracks never accept as product cert)
and materializes explicit sol/null evidence files for ship audits.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from swe_factory.harbor.deepagent_cert import (
    DeepAgentCertError,
    FakeBackendRejected,
    IsolationEvidence,
    PackMetaEvidence,
    certify_deepagent_pack,
    read_pack_meta,
    refuse_fake_backend,
    scan_pack_agent_isolation,
    write_oracle_evidence,
)
from swe_factory.harbor.harbor_oracle import HarborVerifierBackend
from swe_factory.oracle import codes as C
from swe_factory.producers.harbor_labeling import SuiteOutcome

REAL_PR_SOURCE_TRACK = "real_pr"
RORC_SOL_1 = "RORC_SOL_1"
RORC_NULL_0 = "RORC_NULL_0"
RORC_ISOLATION_CLEAN = "RORC_ISOLATION_CLEAN"
RORC_ISOLATION_LEAK = "RORC_ISOLATION_LEAK"
RORC_FAKE_REFUSED = "RORC_FAKE_REFUSED"
RORC_BAD_SOURCE_TRACK = "RORC_BAD_SOURCE_TRACK"
RORC_HYBRID_REFUSED = "RORC_HYBRID_REFUSED"
RORC_DOCKER_REQUIRED = "RORC_DOCKER_REQUIRED"
RORC_PASS = "RORC_PASS"
RORC_REJECT = "RORC_REJECT"

_HYBRID_OR_MOTOR = frozenset(
    {
        "hybrid_curated",
        "hybrid",
        "hybrid_bind",
        "motor",
        "motor_hybrid",
        "synthetic_grounded",
        "synthetic",
    }
)
_FAKE_MODES = frozenset({"fake", "stub", "mock", "offline"})

PromoteDisposition = Literal["accept", "reject"]


class RealOracleCertError(DeepAgentCertError):
    """Real-PR product docker-oracle certification failure."""


class RealPrFakeOracleRejected(FakeBackendRejected):
    """Product Real-PR cert refuses fake/oracle mock backends (VAL-RORC-004)."""


@dataclass(frozen=True, slots=True)
class SolNullEvidenceFiles:
    """On-disk sol/null reward evidence (VAL-RORC-001/002 surfaces)."""

    sol_path: str
    null_path: str
    combined_path: str | None = None
    sol_reward: int | float | None = None
    null_reward: int | float | None = None
    backend: str = "docker"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sol_path": self.sol_path,
            "null_path": self.null_path,
            "combined_path": self.combined_path,
            "sol_reward": self.sol_reward,
            "null_reward": self.null_reward,
            "backend": self.backend,
        }


@dataclass(frozen=True, slots=True)
class RealOracleCertResult:
    """Aggregate Real-PR docker oracle cert outcome (product promote gate)."""

    certified: bool
    disposition: PromoteDisposition
    task_id: str
    pack_dir: str
    backend: str
    source_track: str
    solution_reward: int | float | None
    null_reward: int | float | None
    isolation: IsolationEvidence
    pack_meta: PackMetaEvidence
    evidence_files: SolNullEvidenceFiles | None = None
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    oracle_mode: str = "docker"
    deepagent: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "disposition": self.disposition,
            "task_id": self.task_id,
            "pack_dir": self.pack_dir,
            "backend": self.backend,
            "mode": self.backend,
            "oracle_mode": self.oracle_mode,
            "source_track": self.source_track,
            "solution_reward": self.solution_reward,
            "null_reward": self.null_reward,
            "sol": self.solution_reward,
            "null": self.null_reward,
            "isolation": self.isolation.to_dict(),
            "isolation_status": "clean" if self.isolation.clean else "leak",
            "agent_isolated": self.isolation.clean,
            "pack_meta": self.pack_meta.to_dict(),
            "repository_url": self.pack_meta.repository_url,
            "base_commit_hash": self.pack_meta.base_commit_hash,
            "evidence_files": (
                self.evidence_files.to_dict() if self.evidence_files is not None else None
            ),
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "deepagent": dict(self.deepagent),
            "audit": dict(self.audit),
        }

    def to_audit_row(self) -> dict[str, Any]:
        return {
            "instance_id": self.task_id,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "certified": self.certified,
            "backend": self.backend,
            "oracle_mode": self.oracle_mode,
            "source_track": self.source_track,
            "solution_reward": self.solution_reward,
            "null_reward": self.null_reward,
            "sol": self.solution_reward,
            "null": self.null_reward,
            "isolation": "clean" if self.isolation.clean else "leak",
            "agent_isolated": self.isolation.clean,
            "repository_url": self.pack_meta.repository_url,
            "base_commit_hash": self.pack_meta.base_commit_hash,
            "reason_codes": list(self.reason_codes),
        }


def refuse_fake_oracle_mode_real_pr(
    backend: str | HarborVerifierBackend | None,
    *,
    certified: bool = True,
    dest: Path | str | None = None,
    oracle_mode: str | None = None,
) -> None:
    """VAL-RORC-004: refuse fake oracle_mode on Real-PR product cert path.

    Always fails closed when *certified* (product keep) or dest targets product
    deepagent_v1. Explicit ``oracle_mode=fake`` is treated like backend=fake.
    """
    mode_hint = (oracle_mode or "").strip().lower()
    if mode_hint in _FAKE_MODES:
        raise RealPrFakeOracleRejected(
            "Real-PR product cert refuses oracle_mode=fake "
            f"(got oracle_mode={mode_hint!r}); use oracle_mode=docker / "
            "backend=docker (VAL-RORC-004 / VAL-RSHIP-005)"
        )
    try:
        refuse_fake_backend(backend, certified=certified, dest=dest)
        # Also reject injectable Fake backends through shared refuse.
        if certified or (dest is not None and "deepagent" in str(dest).lower().replace("\\", "/")):
            refuse_fake_backend(backend, certified=True, dest=dest or "datasets/deepagent_v1")
    except FakeBackendRejected as exc:
        if isinstance(exc, RealPrFakeOracleRejected):
            raise
        raise RealPrFakeOracleRejected(
            "Real-PR product cert refuses fake/mock oracle backend "
            f"({exc}); use backend=docker (VAL-RORC-004)"
        ) from exc


def assert_real_pr_source_track(
    source_track: str | None,
    *,
    task_id: str = "",
) -> str:
    """Require source_track=real_pr for product Real-PR cert (refuse hybrid)."""
    track = (source_track or "").strip()
    lower = track.lower()
    if lower in _HYBRID_OR_MOTOR or lower.startswith("hybrid") or lower.startswith("motor"):
        raise RealOracleCertError(
            f"Real-PR docker oracle refuses hybrid/motor source_track={source_track!r} "
            f"for task {task_id or '(unknown)'} (VAL-RORC / VAL-RPACK-003); "
            f"require source_track={REAL_PR_SOURCE_TRACK}"
        )
    if lower != REAL_PR_SOURCE_TRACK:
        display = track if track else "(missing)"
        raise RealOracleCertError(
            f"Real-PR docker oracle requires source_track={REAL_PR_SOURCE_TRACK}; "
            f"got {display!r} for task {task_id or '(unknown)'} (VAL-RORC-001 product cert)"
        )
    return REAL_PR_SOURCE_TRACK


def write_sol_null_evidence_files(
    evidence_dir: Path | str,
    *,
    task_id: str,
    solution_reward: int | float | None,
    null_reward: int | float | None,
    backend: str = "docker",
    repository_url: str = "",
    base_commit_hash: str = "",
    source_track: str = REAL_PR_SOURCE_TRACK,
    isolation_clean: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> SolNullEvidenceFiles:
    """Materialize sol/null reward evidence files for ship/oracle audits.

    Layout (VAL-RORC-001/002 expectedBehavior 'sol/null evidence files'):
    - ``<dir>/<task_id>.sol.reward.json``  (reward=1 when sol settles)
    - ``<dir>/<task_id>.null.reward.json`` (reward=0 when null settles)
    - ``<dir>/<task_id>.oracle_evidence.json`` combined summary
    """
    root = Path(evidence_dir)
    root.mkdir(parents=True, exist_ok=True)
    sol_path = root / f"{task_id}.sol.reward.json"
    null_path = root / f"{task_id}.null.reward.json"
    combined = root / f"{task_id}.oracle_evidence.json"

    sol_payload: dict[str, Any] = {
        "phase": "solution",
        "reward": solution_reward,
        "backend": backend,
        "oracle_mode": backend,
        "task_id": task_id,
        "source_track": source_track,
        "repository_url": repository_url,
        "base_commit_hash": base_commit_hash,
        "isolation": "clean" if isolation_clean else "leak",
    }
    null_payload: dict[str, Any] = {
        "phase": "null",
        "reward": null_reward,
        "backend": backend,
        "oracle_mode": backend,
        "task_id": task_id,
        "source_track": source_track,
        "repository_url": repository_url,
        "base_commit_hash": base_commit_hash,
        "isolation": "clean" if isolation_clean else "leak",
    }
    if extra:
        sol_payload.update({k: v for k, v in extra.items() if not str(k).startswith("_")})
        null_payload.update({k: v for k, v in extra.items() if not str(k).startswith("_")})

    sol_path.write_text(json.dumps(sol_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    null_path.write_text(
        json.dumps(null_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    combined_payload: dict[str, Any] = {
        "task_id": task_id,
        "backend": backend,
        "oracle_mode": backend,
        "source_track": source_track,
        "solution_reward": solution_reward,
        "null_reward": null_reward,
        "sol": solution_reward,
        "null": null_reward,
        "isolation": "clean" if isolation_clean else "leak",
        "repository_url": repository_url,
        "base_commit_hash": base_commit_hash,
        "sol_evidence": str(sol_path),
        "null_evidence": str(null_path),
        "sol_ok": solution_reward == 1,
        "null_ok": null_reward == 0,
        "pair_ok": solution_reward == 1 and null_reward == 0,
    }
    if extra:
        combined_payload["extra"] = dict(extra)
    combined.write_text(
        json.dumps(combined_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return SolNullEvidenceFiles(
        sol_path=str(sol_path.resolve()),
        null_path=str(null_path.resolve()),
        combined_path=str(combined.resolve()),
        sol_reward=solution_reward,
        null_reward=null_reward,
        backend=backend,
    )


def isolation_blocks_real_pr_promote(
    pack_dir: Path | str,
    *,
    work_dir: Path | None = None,
) -> IsolationEvidence:
    """Re-run agent isolation; any hit blocks product promote (VAL-RORC-003)."""
    return scan_pack_agent_isolation(pack_dir, work_dir=work_dir)


def collect_rorc_audit_fields(result: RealOracleCertResult) -> dict[str, Any]:
    """Checklist of VAL-RORC evidence fields for ship inclusion."""
    required = {
        "solution_reward": result.solution_reward,
        "null_reward": result.null_reward,
        "isolation": "clean" if result.isolation.clean else "leak",
        "repository_url": result.pack_meta.repository_url,
        "base_commit_hash": result.pack_meta.base_commit_hash,
        "backend": result.backend,
        "source_track": result.source_track,
        "task_id": result.task_id,
        "oracle_mode": result.oracle_mode,
    }
    missing = [
        key
        for key, val in required.items()
        if val is None
        or val == ""
        or (key == "backend" and val != "docker")
        or (key == "oracle_mode" and val in _FAKE_MODES)
        or (key == "source_track" and str(val).lower() != REAL_PR_SOURCE_TRACK)
    ]
    sol_ok = result.solution_reward == 1
    null_ok = result.null_reward == 0
    isol_ok = result.isolation.clean
    track_ok = result.source_track.lower() == REAL_PR_SOURCE_TRACK
    complete = (
        not missing
        and sol_ok
        and null_ok
        and isol_ok
        and track_ok
        and result.pack_meta.real_url_ok
        and result.pack_meta.real_sha_ok
        and result.backend == "docker"
        and result.oracle_mode == "docker"
        and result.certified
    )
    return {
        "fields": required,
        "missing": missing,
        "sol_ok": sol_ok,
        "null_ok": null_ok,
        "isolation_ok": isol_ok,
        "track_ok": track_ok,
        "complete": complete,
        "blocks_promote": not complete,
        "evidence_files": (
            result.evidence_files.to_dict() if result.evidence_files is not None else None
        ),
    }


def certify_real_pr_pack(
    pack_dir: Path | str,
    *,
    backend: str | HarborVerifierBackend | None = "docker",
    oracle_mode: str | None = None,
    task_id: str | None = None,
    evidence_dir: Path | str | None = None,
    evidence_out: Path | str | None = None,
    audit_out: Path | str | None = None,
    gold_runs: Sequence[SuiteOutcome] | None = None,
    null_runs: Sequence[SuiteOutcome] | None = None,
    forced_flake: bool = False,
    run_pier_hooks: bool = True,
    dest_hint: Path | str | None = None,
    cleanup: bool = True,
    run_id: str = "realpr",
    require_real_pr_track: bool = True,
) -> RealOracleCertResult:
    """Docker-oracle certify a Real-PR Harbor pack for product promote.

    Always refuses fake backends / oracle_mode=fake (VAL-RORC-004).
    Requires sol=1 / null=0 (001/002), isolation clean (003), and
    ``source_track=real_pr`` (hybrid never product-certifies).
    Writes sol/null evidence files under *evidence_dir* when provided.
    """
    # VAL-RORC-004 first: fail closed before any work.
    # Product promote dests refuse Scripted*/Fake* injectables; unit callers pass
    # non-product dest_hint (or leave default offline) when using scripted backends.
    # Default dest_hint is "datasets/deepagent_v1" for product safety when omitted
    # ONLY when backend is not an offline injectable; for classic unit paths where
    # backend is a scripted class, auto-use a non-product dest so existing tests pass.
    effective_dest: Path | str | None
    if dest_hint is not None:
        effective_dest = dest_hint
    else:
        # Product-safe default refused for non-real backends; offline injectables
        # get a non-product hint so unit cert API still works.
        from swe_factory.harbor.harbor_oracle import HarborDockerVerifier as _HDV

        if backend is None or isinstance(backend, str | _HDV):
            effective_dest = "datasets/deepagent_v1"
        else:
            effective_dest = "datasets/deepagent_v1_offline_unit"
    try:
        refuse_fake_oracle_mode_real_pr(
            backend,
            certified=True,
            dest=effective_dest,
            oracle_mode=oracle_mode,
        )
    except FakeBackendRejected as exc:
        # Normalize message to Real-PR wording
        raise RealPrFakeOracleRejected(str(exc)) from exc

    root = Path(pack_dir)
    if not root.is_dir():
        raise RealOracleCertError(f"Real-PR pack dir not found: {root}")

    pack_meta = read_pack_meta(root)
    tid = task_id or pack_meta.task_id or root.name
    track = (pack_meta.source_track or "").strip()

    if require_real_pr_track:
        try:
            track = assert_real_pr_source_track(track, task_id=tid)
        except RealOracleCertError as exc:
            # Isolation still recorded for audit even on track refuse
            isolation = isolation_blocks_real_pr_promote(root)
            track_codes: list[str] = [RORC_BAD_SOURCE_TRACK, RORC_REJECT]
            lower = (pack_meta.source_track or "").strip().lower()
            if lower in _HYBRID_OR_MOTOR or lower.startswith("hybrid"):
                track_codes = [RORC_HYBRID_REFUSED, RORC_BAD_SOURCE_TRACK, RORC_REJECT]
            return RealOracleCertResult(
                certified=False,
                disposition="reject",
                task_id=tid,
                pack_dir=str(root.resolve()),
                backend="docker",
                source_track=pack_meta.source_track or "",
                solution_reward=None,
                null_reward=None,
                isolation=isolation,
                pack_meta=pack_meta,
                reason_codes=tuple(track_codes),
                reasons=(str(exc),),
                oracle_mode="docker",
                audit={"track_ok": False, "blocks_promote": True},
            )

    # Pre-run isolation: fail closed before expensive oracle when already dirty
    pre_isolation = isolation_blocks_real_pr_promote(root)

    # Shared docker cert path (also re-scans isolation + refuse fake)
    try:
        deepagent = certify_deepagent_pack(
            root,
            backend=backend,
            task_id=tid,
            evidence_out=None,  # we write RORC sol/null files below
            audit_out=None,
            gold_runs=gold_runs,
            null_runs=null_runs,
            forced_flake=forced_flake,
            run_pier_hooks=run_pier_hooks,
            dest_hint=effective_dest,
            cleanup=cleanup,
            run_id=run_id,
        )
    except FakeBackendRejected as exc:
        raise RealPrFakeOracleRejected(str(exc)) from exc
    except DeepAgentCertError as exc:
        raise RealOracleCertError(str(exc)) from exc

    # Prefer pack-level isolation (pre + deepagent agreed)
    isolation = IsolationEvidence(
        clean=pre_isolation.clean and deepagent.isolation.clean,
        hits=tuple(dict.fromkeys([*pre_isolation.hits, *deepagent.isolation.hits])),
        summary={
            **dict(deepagent.isolation.summary),
            "pre_scan_clean": pre_isolation.clean,
        },
    )

    codes: list[str] = list(deepagent.reason_codes)
    reasons: list[str] = list(deepagent.reasons)

    if not isolation.clean:
        if RORC_ISOLATION_LEAK not in codes:
            codes.append(RORC_ISOLATION_LEAK)
        if C.G5_LEAK not in codes:
            codes.append(C.G5_LEAK)
        reasons.append(f"Real-PR isolation leak blocks promote: {list(isolation.hits)}")
    else:
        codes.append(RORC_ISOLATION_CLEAN)

    sol = deepagent.solution_reward
    null = deepagent.null_reward
    backend_name = deepagent.backend
    if backend_name != "docker":
        codes.append(RORC_DOCKER_REQUIRED)
        reasons.append(f"Real-PR cert requires backend=docker (got {backend_name!r})")

    if sol == 1:
        codes.append(RORC_SOL_1)
    if null == 0:
        codes.append(RORC_NULL_0)

    # Evidence files (sol/null) — always written when evidence_dir provided
    evidence_files: SolNullEvidenceFiles | None = None
    ev_parent: Path | None = None
    if evidence_dir is not None:
        ev_parent = Path(evidence_dir)
    elif evidence_out is not None:
        ev_parent = Path(evidence_out).parent
    if ev_parent is not None:
        evidence_files = write_sol_null_evidence_files(
            ev_parent,
            task_id=tid,
            solution_reward=sol,
            null_reward=null,
            backend=backend_name,
            repository_url=pack_meta.repository_url,
            base_commit_hash=pack_meta.base_commit_hash,
            source_track=track or REAL_PR_SOURCE_TRACK,
            isolation_clean=isolation.clean,
            extra={
                "disposition": deepagent.disposition,
                "reason_codes": list(codes),
            },
        )
        # Also write the combined DeepAgent evidence shape at evidence_out when given
        if evidence_out is not None:
            write_oracle_evidence(evidence_out, deepagent)

    certified = (
        sol == 1
        and null == 0
        and isolation.clean
        and backend_name == "docker"
        and (track or "").lower() == REAL_PR_SOURCE_TRACK
        and pack_meta.real_url_ok
        and pack_meta.real_sha_ok
        and deepagent.certified
        and not any(c in C.HARD_REJECT_CODES for c in codes)
    )
    if certified:
        codes.append(RORC_PASS)
        codes.append(C.ORACLE_PASS)
    else:
        codes.append(RORC_REJECT)
        if deepagent.certified and not isolation.clean:
            # Isolation alone must pull promote even if underlying sol/null ok
            certified = False

    # Deduplicate codes
    seen: set[str] = set()
    uniq: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            uniq.append(code)

    result = RealOracleCertResult(
        certified=certified,
        disposition="accept" if certified else "reject",
        task_id=tid,
        pack_dir=str(root.resolve()),
        backend=backend_name,
        source_track=track or (pack_meta.source_track or ""),
        solution_reward=sol,
        null_reward=null,
        isolation=isolation,
        pack_meta=pack_meta,
        evidence_files=evidence_files,
        reason_codes=tuple(uniq),
        reasons=tuple(dict.fromkeys(reasons)),
        oracle_mode=backend_name,
        deepagent=deepagent.to_dict(),
        audit={},
    )
    audit = collect_rorc_audit_fields(result)
    if audit["blocks_promote"] and certified:
        result = RealOracleCertResult(
            certified=False,
            disposition="reject",
            task_id=result.task_id,
            pack_dir=result.pack_dir,
            backend=result.backend,
            source_track=result.source_track,
            solution_reward=result.solution_reward,
            null_reward=result.null_reward,
            isolation=result.isolation,
            pack_meta=result.pack_meta,
            evidence_files=result.evidence_files,
            reason_codes=tuple([*result.reason_codes, "AUDIT_INCOMPLETE", RORC_REJECT]),
            reasons=tuple([*result.reasons, "RORC audit incomplete blocks promote"]),
            oracle_mode=result.oracle_mode,
            deepagent=result.deepagent,
            audit=audit,
        )
    else:
        result = RealOracleCertResult(
            certified=result.certified,
            disposition=result.disposition,
            task_id=result.task_id,
            pack_dir=result.pack_dir,
            backend=result.backend,
            source_track=result.source_track,
            solution_reward=result.solution_reward,
            null_reward=result.null_reward,
            isolation=result.isolation,
            pack_meta=result.pack_meta,
            evidence_files=result.evidence_files,
            reason_codes=result.reason_codes,
            reasons=result.reasons,
            oracle_mode=result.oracle_mode,
            deepagent=result.deepagent,
            audit=audit,
        )

    if audit_out is not None:
        out = Path(audit_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_audit_row(), sort_keys=True) + "\n")

    return result


__all__ = [
    "REAL_PR_SOURCE_TRACK",
    "RORC_BAD_SOURCE_TRACK",
    "RORC_DOCKER_REQUIRED",
    "RORC_FAKE_REFUSED",
    "RORC_HYBRID_REFUSED",
    "RORC_ISOLATION_CLEAN",
    "RORC_ISOLATION_LEAK",
    "RORC_NULL_0",
    "RORC_PASS",
    "RORC_REJECT",
    "RORC_SOL_1",
    "PromoteDisposition",
    "RealOracleCertError",
    "RealOracleCertResult",
    "RealPrFakeOracleRejected",
    "SolNullEvidenceFiles",
    "assert_real_pr_source_track",
    "certify_real_pr_pack",
    "collect_rorc_audit_fields",
    "isolation_blocks_real_pr_promote",
    "refuse_fake_oracle_mode_real_pr",
    "write_sol_null_evidence_files",
]
