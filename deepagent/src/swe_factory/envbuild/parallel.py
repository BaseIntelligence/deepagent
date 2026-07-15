"""Bounded parallel envbuild for M9 scale ≥70.

Enforces the architecture concurrency ceiling (default 16, hard max 24)
via a process-wide semaphore. Jobs that would operate on off-limits docker
names are refused before any docker call.

Public contract:
- ``clamp_envbuild_workers(n)`` → workers in [1, MAX]
- ``parallel_envbuild(recipes, build_fn, max_workers=…)`` → ordered results
- Never touches ``mission-test-pg`` / ``challenge-prism*`` / ``acproxy``
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from swe_factory.envbuild.hygiene import (
    CONCURRENCY_HINT,
    MAX_CONCURRENT_ENVBUILD_JOBS,
    HygieneError,
    assert_safe_container_name,
    is_off_limits_name,
    is_owned_container_name,
    require_disk_for_envbuild,
)
from swe_factory.envbuild.models import EnvRecipe

# Inline skip codes (avoid importing swe_factory.sources package from envbuild —
# sources.funnel / git_mine may import envbuild and would circular-import).
SKIP_DISK_GATE = "disk_gate"
SKIP_OFF_LIMITS_DOCKER = "off_limits_docker_refused"
SKIP_PARALLEL_CAP_WAIT_TIMEOUT = "parallel_cap_wait_timeout"


def _describe_skip(code: str) -> str:
    docs = {
        SKIP_DISK_GATE: "Free disk below envbuild fail-closed threshold.",
        SKIP_OFF_LIMITS_DOCKER: (
            "Docker op refused: name is off-limits (mission-test-pg / "
            "challenge-prism* / acproxy) or non-owned prefix."
        ),
        SKIP_PARALLEL_CAP_WAIT_TIMEOUT: (
            "Parallel envbuild semaphore wait exceeded budget (cap enforced)."
        ),
    }
    return docs.get(code, f"skip {code}")


class SkipReason:
    """Minimal skip row used by parallel envbuild (mirrors sources.skip_reasons)."""

    __slots__ = ("code", "detail", "stage", "repo", "candidate_id", "meta")

    def __init__(
        self,
        code: str,
        detail: str = "",
        stage: str = "envbuild",
        repo: str = "",
        candidate_id: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.stage = stage
        self.repo = repo
        self.candidate_id = candidate_id
        self.meta = dict(meta or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "stage": self.stage,
            "repo": self.repo,
            "candidate_id": self.candidate_id,
            "meta": dict(self.meta),
            "documentation": _describe_skip(self.code),
        }


def describe_skip_reason(code: str) -> str:
    return _describe_skip(code)


T = TypeVar("T")

# Hard ceiling matches architecture AGENTS.md band ≤16–24.
HARD_MAX_ENVBUILD_WORKERS = 24
DEFAULT_ENVBUILD_WORKERS = MAX_CONCURRENT_ENVBUILD_JOBS  # 16

# Process-wide semaphore so nested callers cannot oversubscribe.
_global_slots: threading.BoundedSemaphore | None = None
_global_slots_lock = threading.Lock()
_active_workers = 0
_active_lock = threading.Lock()
_peak_active = 0


def clamp_envbuild_workers(
    requested: int | None, *, default: int = DEFAULT_ENVBUILD_WORKERS
) -> int:
    """Bound worker count to [1, HARD_MAX] with default from hygiene ceiling."""
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
    # Soft prefer hygiene default when caller asks for max band without care
    if n > MAX_CONCURRENT_ENVBUILD_JOBS and n <= HARD_MAX_ENVBUILD_WORKERS:
        # still allow up to hard max, but document via return only
        pass
    return n


def get_global_envbuild_semaphore(
    *,
    max_workers: int | None = None,
) -> threading.BoundedSemaphore:
    """Lazy process-wide semaphore sized to the clamped worker ceiling."""
    global _global_slots
    ceiling = clamp_envbuild_workers(max_workers)
    with _global_slots_lock:
        if _global_slots is None:
            _global_slots = threading.BoundedSemaphore(ceiling)
        return _global_slots


def reset_global_envbuild_semaphore_for_tests() -> None:
    """Test helper: drop the process-wide semaphore (not for product runtime)."""
    global _global_slots, _active_workers, _peak_active
    with _global_slots_lock:
        _global_slots = None
    with _active_lock:
        _active_workers = 0
        _peak_active = 0


def active_envbuild_jobs() -> int:
    """Current number of holder slots (approx for tests/metrics)."""
    with _active_lock:
        return _active_workers


def peak_envbuild_jobs() -> int:
    """Peak concurrent holders observed since last reset."""
    with _active_lock:
        return _peak_active


def _track_enter() -> None:
    global _active_workers, _peak_active
    with _active_lock:
        _active_workers += 1
        if _active_workers > _peak_active:
            _peak_active = _active_workers


def _track_exit() -> None:
    global _active_workers
    with _active_lock:
        _active_workers = max(0, _active_workers - 1)


@dataclass(frozen=True, slots=True)
class ParallelEnvJob:
    """One envbuild unit of work."""

    job_id: str
    recipe: EnvRecipe
    container_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "repo_id": self.recipe.repo_id,
            "base_commit": self.recipe.base_commit,
            "language": self.recipe.language,
            "container_name": self.container_name,
            "meta": dict(self.meta),
        }


@dataclass
class ParallelEnvResult:
    """Outcome of one parallel job."""

    job_id: str
    ok: bool
    result: Any = None
    error: str = ""
    skip_reason: SkipReason | None = None
    duration_s: float = 0.0
    worker_slot_wait_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ok": self.ok,
            "error": self.error,
            "duration_s": self.duration_s,
            "worker_slot_wait_s": self.worker_slot_wait_s,
            "skip_reason": self.skip_reason.to_dict() if self.skip_reason else None,
            "result": (
                self.result.to_dict()
                if self.result is not None and hasattr(self.result, "to_dict")
                else self.result
            ),
        }


@dataclass
class ParallelEnvReport:
    """Aggregate report for a parallel envbuild batch."""

    max_workers: int
    hard_max_workers: int = HARD_MAX_ENVBUILD_WORKERS
    hygiene_ceiling: int = MAX_CONCURRENT_ENVBUILD_JOBS
    concurrency_hint: str = CONCURRENCY_HINT
    job_count: int = 0
    ok_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    peak_active: int = 0
    results: list[ParallelEnvResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "hard_max_workers": self.hard_max_workers,
            "hygiene_ceiling": self.hygiene_ceiling,
            "concurrency_hint": self.concurrency_hint,
            "job_count": self.job_count,
            "ok_count": self.ok_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "peak_active": self.peak_active,
            "parallelism_bounded": self.max_workers <= self.hard_max_workers,
            "notes": list(self.notes),
            "results": [r.to_dict() for r in self.results],
            "skip_reasons": {
                (r.skip_reason.code if r.skip_reason else "error"): 1
                for r in self.results
                if not r.ok
            },
        }


def assert_job_docker_safe(container_name: str | None) -> None:
    """Refuse off-limits / non-owned names before any docker op."""
    name = (container_name or "").strip()
    if not name:
        return
    if is_off_limits_name(name):
        raise HygieneError(
            f"refusing parallel envbuild on off-limits container {name!r}: "
            f"{describe_skip_reason(SKIP_OFF_LIMITS_DOCKER)}"
        )
    if not is_owned_container_name(name):
        # Own/assert path also rejects off-limits
        assert_safe_container_name(name)


def _suggested_container_name(job: ParallelEnvJob) -> str:
    if job.container_name:
        return job.container_name
    slug = (job.recipe.repo_id or "repo").replace("/", "-")[:40]
    return f"sdf-envbuild-{slug}-{job.job_id[:8]}"


def run_envbuild_job(
    job: ParallelEnvJob,
    build_fn: Callable[[EnvRecipe], Any],
    *,
    semaphore: threading.BoundedSemaphore | None = None,
    acquire_timeout_s: float = 600.0,
    enforce_disk_gate: bool = False,
) -> ParallelEnvResult:
    """Acquire a slot, refuse off-limits, call *build_fn*, release slot."""
    sem = semaphore or get_global_envbuild_semaphore()
    container = _suggested_container_name(job)
    wait_t0 = time.monotonic()
    acquired = sem.acquire(timeout=max(0.1, float(acquire_timeout_s)))
    wait_s = time.monotonic() - wait_t0
    if not acquired:
        return ParallelEnvResult(
            job_id=job.job_id,
            ok=False,
            error="timed out waiting for envbuild concurrency slot",
            skip_reason=SkipReason(
                code=SKIP_PARALLEL_CAP_WAIT_TIMEOUT,
                detail=f"waited {wait_s:.2f}s for semaphore",
                stage="envbuild",
                repo=job.recipe.repo_id,
                candidate_id=job.job_id,
            ),
            worker_slot_wait_s=wait_s,
        )

    _track_enter()
    t0 = time.monotonic()
    try:
        try:
            assert_job_docker_safe(container)
        except HygieneError as exc:
            return ParallelEnvResult(
                job_id=job.job_id,
                ok=False,
                error=str(exc),
                skip_reason=SkipReason(
                    code=SKIP_OFF_LIMITS_DOCKER,
                    detail=str(exc),
                    stage="envbuild",
                    repo=job.recipe.repo_id,
                    candidate_id=job.job_id,
                    meta={"container_name": container},
                ),
                duration_s=time.monotonic() - t0,
                worker_slot_wait_s=wait_s,
            )
        if enforce_disk_gate:
            try:
                require_disk_for_envbuild()
            except HygieneError as exc:
                return ParallelEnvResult(
                    job_id=job.job_id,
                    ok=False,
                    error=str(exc),
                    skip_reason=SkipReason(
                        code=SKIP_DISK_GATE,
                        detail=str(exc),
                        stage="envbuild",
                        repo=job.recipe.repo_id,
                        candidate_id=job.job_id,
                    ),
                    duration_s=time.monotonic() - t0,
                    worker_slot_wait_s=wait_s,
                )
        try:
            outcome = build_fn(job.recipe)
        except Exception as exc:  # noqa: BLE001 — surface per-job failure
            return ParallelEnvResult(
                job_id=job.job_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
                worker_slot_wait_s=wait_s,
            )
        ok = True
        error_msg = ""
        success_attr = getattr(outcome, "success", None)
        if success_attr is not None:
            ok = bool(success_attr)
            if not ok:
                reason_attr = getattr(outcome, "reason", None)
                error_msg = str(reason_attr) if reason_attr else "build reported failure"
        return ParallelEnvResult(
            job_id=job.job_id,
            ok=ok,
            result=outcome,
            error=error_msg,
            duration_s=time.monotonic() - t0,
            worker_slot_wait_s=wait_s,
        )
    finally:
        _track_exit()
        sem.release()


def parallel_envbuild(
    jobs: Sequence[ParallelEnvJob],
    build_fn: Callable[[EnvRecipe], Any],
    *,
    max_workers: int | None = None,
    acquire_timeout_s: float = 600.0,
    enforce_disk_gate: bool = False,
) -> ParallelEnvReport:
    """Run envbuild jobs with bounded parallelism (never above hard max).

    Results are ordered to match *jobs* input order.
    """
    workers = clamp_envbuild_workers(max_workers)
    # Fresh process semaphore sized to this batch's clamp when unset
    sem = get_global_envbuild_semaphore(max_workers=workers)
    report = ParallelEnvReport(
        max_workers=workers,
        job_count=len(jobs),
        notes=[
            CONCURRENCY_HINT,
            f"clamped max_workers={workers} (hard_max={HARD_MAX_ENVBUILD_WORKERS})",
            "off-limits docker names refused before build_fn",
        ],
    )
    if not jobs:
        report.notes.append("empty job list")
        return report

    # Pre-filter off-limits container names so we never schedule damage work.
    prepared: list[tuple[int, ParallelEnvJob]] = []
    early: dict[int, ParallelEnvResult] = {}
    for idx, job in enumerate(jobs):
        container = _suggested_container_name(job)
        try:
            assert_job_docker_safe(container)
            prepared.append((idx, job))
        except HygieneError as exc:
            early[idx] = ParallelEnvResult(
                job_id=job.job_id,
                ok=False,
                error=str(exc),
                skip_reason=SkipReason(
                    code=SKIP_OFF_LIMITS_DOCKER,
                    detail=str(exc),
                    stage="envbuild",
                    repo=job.recipe.repo_id,
                    candidate_id=job.job_id,
                    meta={"container_name": container},
                ),
            )

    results_by_idx: dict[int, ParallelEnvResult] = dict(early)

    def _run(pair: tuple[int, ParallelEnvJob]) -> tuple[int, ParallelEnvResult]:
        i, j = pair
        return i, run_envbuild_job(
            j,
            build_fn,
            semaphore=sem,
            acquire_timeout_s=acquire_timeout_s,
            enforce_disk_gate=enforce_disk_gate,
        )

    if prepared:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_run, pair) for pair in prepared]
            for fut in concurrent.futures.as_completed(futs):
                idx, res = fut.result()
                results_by_idx[idx] = res

    ordered: list[ParallelEnvResult] = []
    for idx in range(len(jobs)):
        maybe = results_by_idx.get(idx)
        if maybe is None:
            ordered.append(
                ParallelEnvResult(
                    job_id=jobs[idx].job_id,
                    ok=False,
                    error="missing result",
                )
            )
            report.fail_count += 1
            continue
        result: ParallelEnvResult = maybe
        ordered.append(result)
        if result.ok:
            report.ok_count += 1
        elif result.skip_reason is not None:
            report.skip_count += 1
        else:
            report.fail_count += 1

    report.results = ordered
    report.peak_active = peak_envbuild_jobs()
    # Boundedness assertion for report consumers
    if report.peak_active > HARD_MAX_ENVBUILD_WORKERS:
        report.notes.append(
            f"WARNING: peak_active={report.peak_active} exceeded hard max "
            f"{HARD_MAX_ENVBUILD_WORKERS} (should be impossible with semaphore)"
        )
    return report


def parallel_envbuild_recipes(
    recipes: Sequence[EnvRecipe],
    build_fn: Callable[[EnvRecipe], Any],
    *,
    max_workers: int | None = None,
    job_id_prefix: str = "env",
    acquire_timeout_s: float = 600.0,
    enforce_disk_gate: bool = False,
) -> ParallelEnvReport:
    """Convenience wrapper: wrap recipes as jobs and run bounded parallel build."""
    jobs = [
        ParallelEnvJob(
            job_id=f"{job_id_prefix}-{i:04d}-{r.repo_id.replace('/', '-')[:32]}",
            recipe=r,
            container_name=f"sdf-envbuild-{i:04d}",
        )
        for i, r in enumerate(recipes)
    ]
    return parallel_envbuild(
        jobs,
        build_fn,
        max_workers=max_workers,
        acquire_timeout_s=acquire_timeout_s,
        enforce_disk_gate=enforce_disk_gate,
    )


__all__ = [
    "DEFAULT_ENVBUILD_WORKERS",
    "HARD_MAX_ENVBUILD_WORKERS",
    "ParallelEnvJob",
    "ParallelEnvReport",
    "ParallelEnvResult",
    "active_envbuild_jobs",
    "assert_job_docker_safe",
    "clamp_envbuild_workers",
    "get_global_envbuild_semaphore",
    "parallel_envbuild",
    "parallel_envbuild_recipes",
    "peak_envbuild_jobs",
    "reset_global_envbuild_semaphore_for_tests",
    "run_envbuild_job",
]
