"""Real-PR Pier certification adapter (VAL-RPIER-001..004).

Product Real-PR packs under ``datasets/deepagent_v1`` certify via Pier:

- VAL-RPIER-001: structural load of Real-PR pack (no schema/tree error)
- VAL-RPIER-002: oracle/solution agent reward=1 (prefer **real** Pier binary;
  scripted-only is not a full substitute when live Pier is available)
- VAL-RPIER-003: null/nop agent reward=0 on the same pack
- VAL-RPIER-004: refuse fake oracle_mode; Pier evidence never claims fake backend

Jobs default under ``/tmp/harbor-deepagent-jobs*``. Offline unit tests may inject
a Pier runner or pass ``allow_scripted_substitute=True`` (explicit flag required
to treat scripted rewards as a full Product smoke substitute when Pier is not
invokable). Without that flag, scripted cannot mark a Real-PR keep as Pier-certified
when real Pier is available-or-required on the product path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from swe_factory.harbor.deepagent_cert import (
    FakeBackendRejected,
    IsolationEvidence,
    PackMetaEvidence,
    read_pack_meta,
    scan_pack_agent_isolation,
)
from swe_factory.harbor.pier_cert import (
    DEFAULT_JOBS_ROOT,
    DEFAULT_PIER_BIN,
    PierCertError,
    PierCertResult,
    PierInvokeError,
    PierRunEvidence,
    PierRunner,
    ScriptedPierRunner,
    SubprocessPierRunner,
    append_pier_audit,
    certify_pier_pack,
    ensure_jobs_root,
    refuse_fake_oracle_mode,
    resolve_pier_bin,
    structural_load_pack,
    write_pier_evidence,
)
from swe_factory.harbor.real_oracle_cert import (
    REAL_PR_SOURCE_TRACK,
    RealOracleCertError,
    assert_real_pr_source_track,
    refuse_fake_oracle_mode_real_pr,
)
from swe_factory.oracle import codes as C

# AVL reason codes (VAL-RPIER-*)
RPIER_STRUCTURAL_OK = "RPIER_STRUCTURAL_OK"
RPIER_STRUCTURAL_FAIL = "RPIER_STRUCTURAL_FAIL"
RPIER_SOL_1 = "RPIER_SOL_1"
RPIER_SOL_NOT_1 = "RPIER_SOL_NOT_1"
RPIER_NULL_0 = "RPIER_NULL_0"
RPIER_NULL_NOT_0 = "RPIER_NULL_NOT_0"
RPIER_FAKE_REFUSED = "RPIER_FAKE_REFUSED"
RPIER_BAD_SOURCE_TRACK = "RPIER_BAD_SOURCE_TRACK"
RPIER_HYBRID_REFUSED = "RPIER_HYBRID_REFUSED"
RPIER_PIER_UNAVAILABLE = "RPIER_PIER_UNAVAILABLE"
RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE = "RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE"
RPIER_REAL_PATH = "RPIER_REAL_PATH"
RPIER_SCRIPTED_PATH = "RPIER_SCRIPTED_PATH"
RPIER_PASS = "RPIER_PASS"
RPIER_REJECT = "RPIER_REJECT"

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

PierPathClass = Literal["real", "scripted", "injected", "unavailable"]
PromoteDisposition = Literal["accept", "reject"]


class RealPierCertError(PierCertError):
    """Real-PR Pier product certification failure."""


class RealPierUnavailableError(RealPierCertError):
    """Live Pier binary not invokable and scripted substitute not explicitly allowed."""


class RealPierFakeOracleRejected(FakeBackendRejected):
    """Product Real-PR Pier cert refuses fake oracle_mode (VAL-RPIER-004)."""


@dataclass(frozen=True, slots=True)
class RealPierEvidenceFiles:
    """On-disk sol/null Pier reward evidence (VAL-RPIER-002/003 surfaces)."""

    sol_path: str | None
    null_path: str | None
    combined_path: str | None = None
    sol_reward: int | float | None = None
    null_reward: int | float | None = None
    pier_path_class: str = "real"
    jobs_root: str = str(DEFAULT_JOBS_ROOT)
    backend: str = "docker"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sol_path": self.sol_path,
            "null_path": self.null_path,
            "combined_path": self.combined_path,
            "sol_reward": self.sol_reward,
            "null_reward": self.null_reward,
            "pier_path_class": self.pier_path_class,
            "jobs_root": self.jobs_root,
            "backend": self.backend,
            "oracle_mode": self.backend,
        }


@dataclass(frozen=True, slots=True)
class RealPierCertResult:
    """Aggregate Real-PR Pier cert outcome (product smoke / ship gate)."""

    certified: bool
    disposition: PromoteDisposition
    task_id: str
    pack_dir: str
    jobs_root: str
    backend: str
    oracle_mode: str
    source_track: str
    structural_ok: bool
    solution_reward: int | float | None
    null_reward: int | float | None
    pier_path_class: PierPathClass
    isolation: IsolationEvidence
    pack_meta: PackMetaEvidence
    oracle_run: PierRunEvidence | None = None
    null_run: PierRunEvidence | None = None
    evidence_files: RealPierEvidenceFiles | None = None
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    pier: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "disposition": self.disposition,
            "task_id": self.task_id,
            "pack_dir": self.pack_dir,
            "jobs_root": self.jobs_root,
            "backend": self.backend,
            "oracle_mode": self.oracle_mode,
            "source_track": self.source_track,
            "structural_ok": self.structural_ok,
            "solution_reward": self.solution_reward,
            "null_reward": self.null_reward,
            "sol": self.solution_reward,
            "null": self.null_reward,
            "pier_path_class": self.pier_path_class,
            "oracle_path_class": self.pier_path_class,
            "isolation": self.isolation.to_dict(),
            "isolation_status": "clean" if self.isolation.clean else "leak",
            "agent_isolated": self.isolation.clean,
            "pack_meta": self.pack_meta.to_dict(),
            "repository_url": self.pack_meta.repository_url,
            "base_commit_hash": self.pack_meta.base_commit_hash,
            "oracle_run": self.oracle_run.to_dict() if self.oracle_run else None,
            "null_run": self.null_run.to_dict() if self.null_run else None,
            "sol_reward_path": (
                self.oracle_run.reward.path if self.oracle_run is not None else None
            ),
            "null_reward_path": (self.null_run.reward.path if self.null_run is not None else None),
            "evidence_files": (
                self.evidence_files.to_dict() if self.evidence_files is not None else None
            ),
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "pier": dict(self.pier),
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
            "structural_ok": self.structural_ok,
            "sol": self.solution_reward,
            "null": self.null_reward,
            "pier_path_class": self.pier_path_class,
            "isolation": "clean" if self.isolation.clean else "leak",
            "agent_isolated": self.isolation.clean,
            "repository_url": self.pack_meta.repository_url,
            "base_commit_hash": self.pack_meta.base_commit_hash,
            "jobs_root": self.jobs_root,
            "reason_codes": list(self.reason_codes),
            "oracle_reward_path": (
                self.oracle_run.reward.path if self.oracle_run is not None else None
            ),
            "null_reward_path": (self.null_run.reward.path if self.null_run is not None else None),
        }


def pier_is_available(
    pier_bin: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[bool, Path | None, str | None]:
    """Return (available, resolved_path, error_message)."""
    try:
        path = resolve_pier_bin(pier_bin, env=env)
        return True, path, None
    except PierInvokeError as exc:
        return False, None, str(exc)
    except PierCertError as exc:
        return False, None, str(exc)


def prefer_real_pier_runner(
    *,
    pier_bin: Path | str | None = None,
    injected: PierRunner | None = None,
    allow_scripted_substitute: bool = False,
    timeout_sec: float = 1800.0,
    env: Mapping[str, str] | None = None,
) -> tuple[PierRunner, PierPathClass, Path | None]:
    """Choose Roman pier path: prefer real binary; scripted only with flag.

    Product Real-PR pier smoke **must not** silently treat ScriptedPierRunner as
    a full substitute when live pier is unavailable, unless
    ``allow_scripted_substitute=True`` is explicitly set (unit / offline demos).

    Injected runners (tests) are classed ``injected`` and do not count as
    real pier, but may still certify for offline unit coverage when provided
    (caller controls existence of pier).
    """
    if injected is not None:
        # Explicit inject: unit tests / higher-level ship hooks. Class = injected.
        # Prefer real pier when binary exists AND no inject (handled before inject by
        # callers that only inject for offline tests).
        return injected, "injected", None

    available, path, err = pier_is_available(pier_bin, env=env)
    if available and path is not None:
        return (
            SubprocessPierRunner(pier_bin=path, timeout_sec=timeout_sec, env=env),
            "real",
            path,
        )

    if allow_scripted_substitute:
        return ScriptedPierRunner(), "scripted", None

    # Product path: fail closed — cannot pretend scripted is full substitute
    msg = (
        "Real-PR pier cert: live pier unavailable and allow_scripted_substitute "
        f"was not set. {err or 'pier binary not found'}. Install via "
        f"services pier_venv_ensure ({DEFAULT_PIER_BIN}) or set PIER_BIN. "
        "Scripted rewards cannot fully substitute for real pier smoke without "
        "an explicit --allow-scripted-substitute flag (VAL-RPIER-002)."
    )
    raise RealPierUnavailableError(msg)


def write_real_pier_sol_null_evidence(
    evidence_dir: Path | str,
    *,
    task_id: str,
    solution_reward: int | float | None,
    null_reward: int | float | None,
    sol_reward_path: str | None = None,
    null_reward_path: str | None = None,
    pier_path_class: str = "real",
    jobs_root: str = str(DEFAULT_JOBS_ROOT),
    backend: str = "docker",
    repository_url: str = "",
    base_commit_hash: str = "",
    source_track: str = REAL_PR_SOURCE_TRACK,
    isolation_clean: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> RealPierEvidenceFiles:
    """Materialize sol/null reward evidence for Pier Real-PR audits.

    Layout:
    - ``<dir>/<task_id>.sol.reward.json``  (reward=1)
    - ``<dir>/<task_id>.null.reward.json`` (reward=0)
    - ``<dir>/<task_id>.pier_evidence.json`` combined
    """
    root = Path(evidence_dir)
    root.mkdir(parents=True, exist_ok=True)
    sol_path = root / f"{task_id}.sol.reward.json"
    null_path = root / f"{task_id}.null.reward.json"
    combined = root / f"{task_id}.pier_evidence.json"

    sol_payload: dict[str, Any] = {
        "phase": "oracle",
        "agent": "oracle",
        "reward": solution_reward,
        "backend": backend,
        "oracle_mode": backend,
        "task_id": task_id,
        "source_track": source_track,
        "repository_url": repository_url,
        "base_commit_hash": base_commit_hash,
        "pier_path_class": pier_path_class,
        "jobs_root": jobs_root,
        "reward_source_path": sol_reward_path,
        "isolation": "clean" if isolation_clean else "leak",
    }
    null_payload: dict[str, Any] = {
        "phase": "null",
        "agent": "nop",
        "reward": null_reward,
        "backend": backend,
        "oracle_mode": backend,
        "task_id": task_id,
        "source_track": source_track,
        "repository_url": repository_url,
        "base_commit_hash": base_commit_hash,
        "pier_path_class": pier_path_class,
        "jobs_root": jobs_root,
        "reward_source_path": null_reward_path,
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
        "pier_path_class": pier_path_class,
        "jobs_root": jobs_root,
        "isolation": "clean" if isolation_clean else "leak",
        "repository_url": repository_url,
        "base_commit_hash": base_commit_hash,
        "sol_evidence": str(sol_path),
        "null_evidence": str(null_path),
        "sol_reward_path": sol_reward_path,
        "null_reward_path": null_reward_path,
        "sol_ok": solution_reward == 1 or solution_reward == 1.0,
        "null_ok": null_reward == 0 or null_reward == 0.0,
        "pair_ok": (solution_reward in (1, 1.0)) and (null_reward in (0, 0.0)),
        "fake_oracle": False,
    }
    if extra:
        combined_payload["extra"] = dict(extra)
    combined.write_text(
        json.dumps(combined_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return RealPierEvidenceFiles(
        sol_path=str(sol_path.resolve()),
        null_path=str(null_path.resolve()),
        combined_path=str(combined.resolve()),
        sol_reward=solution_reward,
        null_reward=null_reward,
        pier_path_class=pier_path_class,
        jobs_root=jobs_root,
        backend=backend,
    )


def refuse_fake_oracle_mode_real_pier(
    oracle_mode: str | None,
    *,
    certified: bool = True,
    pack_or_dest: Path | str | None = None,
) -> None:
    """VAL-RPIER-004: refuse fake on Real-PR Pier product cert path."""
    mode = (oracle_mode or "docker").strip().lower()
    if mode in _FAKE_MODES:
        raise RealPierFakeOracleRejected(
            "Real-PR Pier cert refuses oracle_mode=fake "
            f"(got oracle_mode={mode!r}); docker-backed pier required "
            "(VAL-RPIER-004 / VAL-RORC-004)"
        )
    try:
        refuse_fake_oracle_mode(mode, certified=certified, pack_or_dest=pack_or_dest)
        refuse_fake_oracle_mode_real_pr(
            "docker" if mode not in _FAKE_MODES else mode,
            certified=True,
            dest=pack_or_dest or "datasets/deepagent_v1",
            oracle_mode=mode,
        )
    except FakeBackendRejected as exc:
        if isinstance(exc, RealPierFakeOracleRejected):
            raise
        raise RealPierFakeOracleRejected(
            "Real-PR Pier cert refuses fake/mock oracle mode "
            f"({exc}); use oracle_mode=docker (VAL-RPIER-004)"
        ) from exc


def certify_real_pier_pack(
    pack_dir: Path | str,
    *,
    runner: PierRunner | None = None,
    jobs_root: Path | str | None = None,
    task_id: str | None = None,
    oracle_mode: str = "docker",
    run_oracle: bool = True,
    run_null: bool = True,
    run_load_smoke: bool = True,
    force_build: bool = True,
    pier_bin: Path | str | None = None,
    evidence_out: Path | str | None = None,
    evidence_dir: Path | str | None = None,
    audit_out: Path | str | None = None,
    isolation: IsolationEvidence | None = None,
    n_concurrent: int = 1,
    timeout_sec: float = 1800.0,
    require_real_pr_track: bool = True,
    allow_scripted_substitute: bool = False,
    prefer_real_pier: bool = True,
    dest_hint: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> RealPierCertResult:
    """Pier-certify one Real-PR Harbor pack (VAL-RPIER-001..004).

    Prefers live Pier subprocess when available. Scripted runners cannot mark
    product Real-PR smoke fully certified without ``allow_scripted_substitute``.
    Injected runners (unit tests) are accepted for offline coverage when the
    caller supplies them, but the result records ``pier_path_class=injected``
    and only counts as full product certify when rewards match and track is
    real_pr **and** (path is real **or** allow_scripted_substitute / injected
    with explicit offline intent).

    Product certification (``certified=True``) for **live ship smoke** requires:

    - structural load OK
    - sol reward == 1 and null reward == 0
    - isolation clean, source_track=real_pr, oracle_mode=docker (no fake)
    - pier_path_class is ``real`` **or** explicit ``allow_scripted_substitute``
      (scripted/injected offline demos). Without that flag, when pier is
      unavailable, this raises :class:`RealPierUnavailableError`.
    """
    # VAL-RPIER-004 first
    try:
        refuse_fake_oracle_mode_real_pier(
            oracle_mode,
            certified=True,
            pack_or_dest=dest_hint or pack_dir or "datasets/deepagent_v1",
        )
    except FakeBackendRejected as exc:
        raise RealPierFakeOracleRejected(str(exc)) from exc

    root = Path(pack_dir)
    if not root.is_dir():
        raise RealPierCertError(f"Real-PR pack dir not found: {root}")

    pack_meta = read_pack_meta(root)
    tid = task_id or pack_meta.task_id or root.name
    track = (pack_meta.source_track or "").strip()

    jobs = ensure_jobs_root(jobs_root)
    codes: list[str] = []
    reasons: list[str] = []

    if require_real_pr_track:
        try:
            track = assert_real_pr_source_track(track, task_id=tid)
        except RealOracleCertError as exc:
            isolation_ev = isolation or scan_pack_agent_isolation(root)
            track_codes = [RPIER_BAD_SOURCE_TRACK, RPIER_REJECT]
            lower = (pack_meta.source_track or "").strip().lower()
            if lower in _HYBRID_OR_MOTOR or lower.startswith("hybrid"):
                track_codes = [RPIER_HYBRID_REFUSED, RPIER_BAD_SOURCE_TRACK, RPIER_REJECT]
            return RealPierCertResult(
                certified=False,
                disposition="reject",
                task_id=tid,
                pack_dir=str(root.resolve()),
                jobs_root=str(jobs),
                backend="docker",
                oracle_mode="docker",
                source_track=pack_meta.source_track or "",
                structural_ok=False,
                solution_reward=None,
                null_reward=None,
                pier_path_class="unavailable",
                isolation=isolation_ev,
                pack_meta=pack_meta,
                reason_codes=tuple(track_codes),
                reasons=(str(exc),),
                audit={"track_ok": False, "blocks_promote": True},
            )

    # Prefer real pier over scripted when no runner is injected (product path).
    # Explicit inject (unit tests / ship offline hooks) is respected as-is;
    # scripted/injected paths cannot full-certify without allow_scripted_substitute.
    path_class: PierPathClass
    active: PierRunner
    resolved_bin: Path | None = None

    if runner is not None:
        active = runner
        if isinstance(runner, ScriptedPierRunner):
            path_class = "scripted"
            if not allow_scripted_substitute:
                codes.append(RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE)
                reasons.append(
                    "scripted pier runner provided without "
                    "allow_scripted_substitute; not a full Real-PR "
                    "Pier substitute when used as sole cert path "
                    "(VAL-RPIER-002)"
                )
        else:
            path_class = "injected"
            if not allow_scripted_substitute:
                codes.append(RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE)
                reasons.append(
                    "injected pier runner without allow_scripted_substitute "
                    "is not counted as full Real-PR pier smoke (VAL-RPIER-002)"
                )
    elif prefer_real_pier:
        try:
            active, path_class, resolved_bin = prefer_real_pier_runner(
                pier_bin=pier_bin,
                injected=None,
                allow_scripted_substitute=allow_scripted_substitute,
                timeout_sec=timeout_sec,
                env=env,
            )
        except RealPierUnavailableError as exc:
            isolation_ev = isolation or scan_pack_agent_isolation(root)
            pier_ready = structural_load_pack(
                root,
                run_load_smoke=run_load_smoke,
                pier_job_prefix=str(jobs),
            )
            structural_ok = pier_ready.structural_ok and pier_ready.required_relpaths_ok
            unavail_codes = [
                RPIER_PIER_UNAVAILABLE,
                RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE,
                RPIER_REJECT,
            ]
            if structural_ok:
                unavail_codes.insert(0, RPIER_STRUCTURAL_OK)
            else:
                unavail_codes.insert(0, RPIER_STRUCTURAL_FAIL)
            result_unavail = RealPierCertResult(
                certified=False,
                disposition="reject",
                task_id=tid,
                pack_dir=str(root.resolve()),
                jobs_root=str(jobs),
                backend="docker",
                oracle_mode=(oracle_mode or "docker").strip().lower(),
                source_track=track or REAL_PR_SOURCE_TRACK,
                structural_ok=structural_ok,
                solution_reward=None,
                null_reward=None,
                pier_path_class="unavailable",
                isolation=isolation_ev,
                pack_meta=pack_meta,
                reason_codes=tuple(unavail_codes),
                reasons=(str(exc),),
                pier={"pier_available": False, "error": str(exc)},
                audit={"pier_available": False, "blocks_promote": True},
            )
            if evidence_dir is not None or evidence_out is not None:
                parent = (
                    Path(evidence_dir)
                    if evidence_dir is not None
                    else Path(str(evidence_out)).parent
                )
                ev_files = write_real_pier_sol_null_evidence(
                    parent,
                    task_id=tid,
                    solution_reward=None,
                    null_reward=None,
                    pier_path_class="unavailable",
                    jobs_root=str(jobs),
                    repository_url=pack_meta.repository_url,
                    base_commit_hash=pack_meta.base_commit_hash,
                    source_track=track or REAL_PR_SOURCE_TRACK,
                    isolation_clean=isolation_ev.clean,
                    extra={"error": str(exc)},
                )
                result_unavail = RealPierCertResult(
                    certified=False,
                    disposition="reject",
                    task_id=result_unavail.task_id,
                    pack_dir=result_unavail.pack_dir,
                    jobs_root=result_unavail.jobs_root,
                    backend=result_unavail.backend,
                    oracle_mode=result_unavail.oracle_mode,
                    source_track=result_unavail.source_track,
                    structural_ok=result_unavail.structural_ok,
                    solution_reward=None,
                    null_reward=None,
                    pier_path_class="unavailable",
                    isolation=result_unavail.isolation,
                    pack_meta=result_unavail.pack_meta,
                    evidence_files=ev_files,
                    reason_codes=result_unavail.reason_codes,
                    reasons=result_unavail.reasons,
                    pier=result_unavail.pier,
                    audit=result_unavail.audit,
                )
            if audit_out is not None:
                out = Path(audit_out)
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result_unavail.to_audit_row(), sort_keys=True) + "\n")
            # Hard error: cannot mark scripted as full substitute without flag
            raise RealPierUnavailableError(str(exc)) from None
    else:
        # prefer_real_pier=False → scripted offline path (still needs allow for cert)
        active = ScriptedPierRunner()
        path_class = "scripted"
        if not allow_scripted_substitute:
            codes.append(RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE)
            reasons.append(
                "prefer_real_pier=False without allow_scripted_substitute; "
                "scripted path recorded but not full product certify"
            )

    if path_class == "real":
        codes.append(RPIER_REAL_PATH)
    elif path_class == "scripted":
        codes.append(RPIER_SCRIPTED_PATH)

    # Run shared pier cert core (reuse parsers/runners evidence)
    try:
        pier_result: PierCertResult = certify_pier_pack(
            root,
            runner=active,
            jobs_root=jobs,
            task_id=tid,
            oracle_mode=oracle_mode,
            run_oracle=run_oracle,
            run_null=run_null,
            run_load_smoke=run_load_smoke,
            force_build=force_build,
            pier_bin=pier_bin or resolved_bin,
            evidence_out=None,  # write Real-PR shape below
            audit_out=None,
            isolation=isolation,
            n_concurrent=n_concurrent,
            timeout_sec=timeout_sec,
        )
    except FakeBackendRejected as exc:
        raise RealPierFakeOracleRejected(str(exc)) from exc
    except PierCertError as exc:
        raise RealPierCertError(str(exc)) from exc

    codes.extend(list(pier_result.reason_codes))
    reasons.extend(list(pier_result.reasons))

    sol = pier_result.oracle_run.reward.reward if pier_result.oracle_run else None
    null = pier_result.null_run.reward.reward if pier_result.null_run else None
    structural_ok = pier_result.structural_ok
    isol = pier_result.isolation

    if structural_ok:
        codes.append(RPIER_STRUCTURAL_OK)
    else:
        codes.append(RPIER_STRUCTURAL_FAIL)

    if sol in (1, 1.0):
        codes.append(RPIER_SOL_1)
    elif run_oracle:
        codes.append(RPIER_SOL_NOT_1)
    if null in (0, 0.0):
        codes.append(RPIER_NULL_0)
    elif run_null:
        codes.append(RPIER_NULL_NOT_0)

    # Full product certify: real pier always; scripted/injected only with flag.
    full_path_ok = path_class == "real" or (
        allow_scripted_substitute and path_class in {"scripted", "injected"}
    )

    certified = (
        pier_result.certified
        and full_path_ok
        and structural_ok
        and (sol in (1, 1.0) if run_oracle else True)
        and (null in (0, 0.0) if run_null else True)
        and isol.clean
        and (track or "").lower() == REAL_PR_SOURCE_TRACK
        and pier_result.backend == "docker"
        and (oracle_mode or "docker").strip().lower() not in _FAKE_MODES
        and pack_meta.real_url_ok
        and pack_meta.real_sha_ok
        and not any(c in C.HARD_REJECT_CODES for c in codes)
    )
    if certified:
        codes.append(RPIER_PASS)
    else:
        codes.append(RPIER_REJECT)

    seen: set[str] = set()
    uniq: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            uniq.append(code)

    # Evidence under jobs / evidence_dir
    evidence_files: RealPierEvidenceFiles | None = None
    ev_parent: Path | None = None
    if evidence_dir is not None:
        ev_parent = Path(evidence_dir)
    elif evidence_out is not None:
        ev_parent = Path(evidence_out).parent
    else:
        # Default: write under jobs root so evidence stays under /tmp jobs
        ev_parent = jobs / "evidence"

    if ev_parent is not None:
        evidence_files = write_real_pier_sol_null_evidence(
            ev_parent,
            task_id=tid,
            solution_reward=sol,
            null_reward=null,
            sol_reward_path=(
                pier_result.oracle_run.reward.path if pier_result.oracle_run else None
            ),
            null_reward_path=(pier_result.null_run.reward.path if pier_result.null_run else None),
            pier_path_class=path_class,
            jobs_root=str(jobs),
            backend=pier_result.backend,
            repository_url=pack_meta.repository_url,
            base_commit_hash=pack_meta.base_commit_hash,
            source_track=track or REAL_PR_SOURCE_TRACK,
            isolation_clean=isol.clean,
            extra={
                "disposition": "accept" if certified else "reject",
                "reason_codes": list(uniq),
                "structural_ok": structural_ok,
            },
        )

    if evidence_out is not None:
        # Full historical pier_result dump; Real-PR aggregate written after result.
        write_pier_evidence(evidence_out, pier_result)

    result = RealPierCertResult(
        certified=certified,
        disposition="accept" if certified else "reject",
        task_id=tid,
        pack_dir=str(root.resolve()),
        jobs_root=str(jobs),
        backend=pier_result.backend,
        oracle_mode=(oracle_mode or "docker").strip().lower(),
        source_track=track or (pack_meta.source_track or ""),
        structural_ok=structural_ok,
        solution_reward=sol,
        null_reward=null,
        pier_path_class=path_class,
        isolation=isol,
        pack_meta=pack_meta,
        oracle_run=pier_result.oracle_run,
        null_run=pier_result.null_run,
        evidence_files=evidence_files,
        reason_codes=tuple(uniq),
        reasons=tuple(dict.fromkeys(reasons)),
        pier=pier_result.to_dict(),
        audit={
            "pier_path_class": path_class,
            "full_path_ok": full_path_ok,
            "sol_ok": sol in (1, 1.0),
            "null_ok": null in (0, 0.0),
            "structural_ok": structural_ok,
            "blocks_promote": not certified,
            "resolved_pier_bin": str(resolved_bin) if resolved_bin else None,
            "allow_scripted_substitute": allow_scripted_substitute,
        },
    )

    # Write aggregate real pier evidence when evidence_out given
    if evidence_out is not None:
        outp = Path(evidence_out)
        if outp.suffix == ".json":
            real_path = outp.with_name(
                outp.stem + ".real_pier.json" if not outp.stem.endswith(".real_pier") else outp.name
            )
            real_path.parent.mkdir(parents=True, exist_ok=True)
            real_path.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    if audit_out is not None:
        append_pier_audit(audit_out, pier_result)
        # also append real pier row
        out = Path(audit_out)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_audit_row(), sort_keys=True) + "\n")

    return result


__all__ = [
    "DEFAULT_JOBS_ROOT",
    "DEFAULT_PIER_BIN",
    "REAL_PR_SOURCE_TRACK",
    "RPIER_BAD_SOURCE_TRACK",
    "RPIER_FAKE_REFUSED",
    "RPIER_HYBRID_REFUSED",
    "RPIER_NULL_0",
    "RPIER_NULL_NOT_0",
    "RPIER_PASS",
    "RPIER_PIER_UNAVAILABLE",
    "RPIER_REAL_PATH",
    "RPIER_REJECT",
    "RPIER_SCRIPTED_NOT_FULL_SUBSTITUTE",
    "RPIER_SCRIPTED_PATH",
    "RPIER_SOL_1",
    "RPIER_SOL_NOT_1",
    "RPIER_STRUCTURAL_FAIL",
    "RPIER_STRUCTURAL_OK",
    "PierPathClass",
    "PromoteDisposition",
    "RealPierCertError",
    "RealPierCertResult",
    "RealPierEvidenceFiles",
    "RealPierFakeOracleRejected",
    "RealPierUnavailableError",
    "certify_real_pier_pack",
    "pier_is_available",
    "prefer_real_pier_runner",
    "refuse_fake_oracle_mode_real_pier",
    "write_real_pier_sol_null_evidence",
]
