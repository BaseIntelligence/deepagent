"""DeepSWE-grade product eval: Pier + mini-swe-agent + Harbor verifier (serial).

This is the **only** path allowed to claim DeepSWE-grade hardness fidelity
(``fidelity=pier_miniswe_harbor``). It is NOT the host-soft L2 script and NOT
``swe-factory panel --live`` never-solve soft solver (VAL-DEVAL-001/006).

Wave protocol (VAL-DEVAL-001..007):
1. Load Harbor product packs from ``datasets/deepswe_v1/tasks/*``.
2. Preflight dual-truth (oracle/solution reward=1, nop/null reward=0) when enabled.
3. Run Pier ``-a mini-swe-agent`` per model serial (``n_concurrent=1``) on exact
   OpenRouter ids ``x-ai/grok-4.5`` and ``moonshotai/kimi-k2.6``.
4. Harvest verifier ``reward.json``; compute pass@k and band rules.
5. Ledger reserve/settle under ``hard_stop_usd`` (default $300).
6. Write ``datasets/panel_deepswe_*/report.json`` with fidelity + spend + models.

Offline unit tests inject a mocked pier invoker; live path shells pier.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from swe_factory.accounting import AccountingError, BudgetLedger, default_ledger_path
from swe_factory.config import DEFAULT_PANEL_MODELS
from swe_factory.harbor.pier_cert import (
    DEFAULT_PIER_BIN,
    PierRunner,
    SubprocessPierRunner,
    parse_pier_job_result,
    resolve_pier_bin,
)
from swe_factory.panel.band import (
    BandDecision,
    BandFilterConfig,
    classify_band,
    compute_discrimination,
    compute_pass_at_k,
)
from swe_factory.panel.runner import (
    PANEL_SCAFFOLD_AGENT,
    PANEL_SCAFFOLD_NAME,
    PANEL_SCAFFOLD_RUNTIME,
    assert_required_panel_models,
    discover_real_pr_panel_keeps,
)

# Product fidelity tag (never reuse never-solve / soft L2 as DeepSWE).
DEEPSWE_EVAL_FIDELITY = "pier_miniswe_harbor"
DEEPSWE_EVAL_STAGE = "deepswe-eval-pier-miniswe"
DEFAULT_HARD_STOP_USD = Decimal("300")
DEFAULT_N_CONCURRENT = 1
DEFAULT_EVAL_K = 1
# Multi-turn mini-swe worst-case reserve per trial (ledger hard-stops at 300).
DEFAULT_TRIAL_RESERVE_USD = Decimal("25.00")
DEFAULT_JOBS_ROOT = Path("/tmp/harbor-deepswe-jobs")
DEFAULT_PRODUCT_ROOT = Path("datasets/deepswe_v1")
DEFAULT_OUT_ROOT = Path("datasets/panel_deepswe_eval")

# Preferred dual-truth Python product packs for first 5-pack canary (order matters).
PREFERRED_PACK_IDS: tuple[str, ...] = (
    "realpr-itemadapter-101",
    "realpr-click-3645",
    "realpr-attrs-1323",
    "realpr-httpx-3672",
    "realpr-packaging-1120",
)

# Exact OpenRouter model pair (VAL-DEVAL-003). Prefix ``openrouter/`` is allowed
# on the pier -m flag without substitution to a different model family.
DEEPSWE_EVAL_MODELS: tuple[str, ...] = tuple(DEFAULT_PANEL_MODELS)


class DeepSWEEvalError(RuntimeError):
    """Unrecoverable DeepSWE-grade eval configuration or path failure."""


class DeepSWEBudgetStop(DeepSWEEvalError):
    """Hard stop: remaining ledger budget cannot fund another trial."""

    def __init__(
        self,
        message: str,
        *,
        remaining_usd: Decimal,
        need_usd: Decimal,
    ) -> None:
        super().__init__(message)
        self.remaining_usd = remaining_usd
        self.need_usd = need_usd


@dataclass(frozen=True, slots=True)
class TrialReward:
    """One mini-swe / pier trial reward (harvested, never invented)."""

    pack_id: str
    model: str
    index: int
    reward: float | int | None
    solved: bool
    job_dir: str | None
    reward_path: str | None
    physical_call_id: str
    cost_usd: Decimal
    exit_code: int | None = None
    errors: tuple[str, ...] = ()
    invented_reward: bool = False
    openrouter_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "model": self.model,
            "index": self.index,
            "reward": self.reward,
            "solved": self.solved,
            "job_dir": self.job_dir,
            "reward_path": self.reward_path,
            "physical_call_id": self.physical_call_id,
            "cost_usd": format(self.cost_usd, "f"),
            "exit_code": self.exit_code,
            "errors": list(self.errors),
            "invented_reward": self.invented_reward,
            "openrouter_model": self.openrouter_model or self.model,
        }


@dataclass(frozen=True, slots=True)
class ModelPackStats:
    pack_id: str
    model: str
    k: int
    solves: int
    pass_at_k: float
    trials: tuple[TrialReward, ...]
    completed_trials: int
    incomplete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "model": self.model,
            "k": self.k,
            "solves": self.solves,
            "pass_at_k": self.pass_at_k,
            "completed_trials": self.completed_trials,
            "incomplete": self.incomplete,
            "trials": [t.to_dict() for t in self.trials],
        }


@dataclass(frozen=True, slots=True)
class PackPreflight:
    pack_id: str
    pack_path: str
    ok: bool
    solution_reward: float | int | None
    null_reward: float | int | None
    errors: tuple[str, ...] = ()
    mode: str = "skipped"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "pack_path": self.pack_path,
            "ok": self.ok,
            "solution_reward": self.solution_reward,
            "null_reward": self.null_reward,
            "errors": list(self.errors),
            "mode": self.mode,
        }


@dataclass
class PackEvalResult:
    pack_id: str
    pack_path: str
    models: list[ModelPackStats]
    decision: BandDecision | None
    preflight: PackPreflight | None
    total_cost_usd: Decimal
    budget_stop: bool = False
    complete: bool = True
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "pack_path": self.pack_path,
            "models": [m.to_dict() for m in self.models],
            "decision": self.decision.to_dict() if self.decision else None,
            "preflight": self.preflight.to_dict() if self.preflight else None,
            "total_cost_usd": format(self.total_cost_usd, "f"),
            "budget_stop": self.budget_stop,
            "complete": self.complete,
            "stop_reason": self.stop_reason,
        }


@dataclass
class DeepSWEEvalReport:
    """Durable product eval report written under datasets/panel_deepswe_*."""

    fidelity: str
    models: list[str]
    n_concurrent: int
    k: int
    hard_stop_usd: Decimal
    product_root: str
    pier_bin: str | None
    agent: str
    scaffold: str
    runtime: str
    n_packs_requested: int
    n_packs_scored: int
    n_packs_preflight_ok: int
    pack_results: list[PackEvalResult]
    total_spend_usd: Decimal
    remaining_usd: Decimal
    budget_stop: bool
    stop_reason: str | None
    invented_rewards: bool
    wall_s: float
    preflight_enabled: bool
    offline: bool
    preferred_pack_ids: list[str] = field(default_factory=list)
    host_mem_before: dict[str, float] = field(default_factory=dict)
    host_mem_after: dict[str, float] = field(default_factory=dict)
    ledger_path: str | None = None
    out_dir: str | None = None
    jobs_dir: str | None = None
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity": self.fidelity,
            "models": list(self.models),
            "n_concurrent": self.n_concurrent,
            "k": self.k,
            "hard_stop_usd": format(self.hard_stop_usd, "f"),
            "product_root": self.product_root,
            "pier_bin": self.pier_bin,
            "agent": self.agent,
            "scaffold": self.scaffold,
            "runtime": self.runtime,
            "n_packs_requested": self.n_packs_requested,
            "n_packs_scored": self.n_packs_scored,
            "n_packs_preflight_ok": self.n_packs_preflight_ok,
            "pack_results": [p.to_dict() for p in self.pack_results],
            "total_spend_usd": format(self.total_spend_usd, "f"),
            "spend_usd": format(self.total_spend_usd, "f"),
            "remaining_usd": format(self.remaining_usd, "f"),
            "budget_stop": self.budget_stop,
            "stop_reason": self.stop_reason,
            "invented_rewards": self.invented_rewards,
            "wall_s": self.wall_s,
            "preflight_enabled": self.preflight_enabled,
            "offline": self.offline,
            "preferred_pack_ids": list(self.preferred_pack_ids),
            "host_mem_before": dict(self.host_mem_before),
            "host_mem_after": dict(self.host_mem_after),
            "ledger_path": self.ledger_path,
            "out_dir": self.out_dir,
            "jobs_dir": self.jobs_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            # Honesty markers
            "pass_at_k_present": True,
            "never_solve_panel": False,
            "soft_l2_panel": False,
        }


class MiniSweInvoker(Protocol):
    """Callable protocol for one pack × model mini-swe pier trial."""

    def __call__(
        self,
        *,
        pack_path: Path,
        pack_id: str,
        model: str,
        jobs_dir: Path,
        index: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Return reward/solved/job_dir/reward_path/exit_code/errors/cost_usd."""
        ...


def normalize_model_id(model: str) -> str:
    """Strip optional ``openrouter/`` prefix for canonical report keys."""
    mid = model.strip()
    if mid.startswith("openrouter/"):
        mid = mid[len("openrouter/") :]
    return mid


def openrouter_model_flag(model: str) -> str:
    """Build pier ``-m`` value with openrouter/ prefix when missing."""
    mid = model.strip()
    if not mid:
        raise DeepSWEEvalError("model must be non-empty")
    if mid.startswith("openrouter/"):
        return mid
    return f"openrouter/{mid}"


def read_host_mem_gib() -> dict[str, float]:
    """Best-effort host memory snapshot in GiB (fails open to zeros)."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return {"total": 0.0, "used": 0.0, "available": 0.0}
    values: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            kib = float(parts[0])
        except ValueError:
            continue
        values[key] = kib / (1024.0 * 1024.0)
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    used = max(0.0, total - available)
    return {"total": round(total, 2), "used": round(used, 2), "available": round(available, 2)}


def reclaim_jobs_dirs(
    roots: Sequence[Path | str] | None = None,
    *,
    max_age_s: float | None = None,
) -> list[str]:
    """Remove **our** harbor/pier job workdirs under known prefixes.

    Empty roots deletes nothing. When ``max_age_s`` is None, listed roots are
    removed wholesale if they match harbor-deepswe job prefixes.
    """
    candidates: list[Path] = []
    if roots:
        candidates.extend(Path(r) for r in roots)
    else:
        tmp = Path("/tmp")
        if tmp.is_dir():
            for child in tmp.iterdir():
                name = child.name
                if child.is_dir() and (
                    name.startswith("harbor-deepswe-jobs")
                    or name.startswith("harbor-deepswe-panel")
                    or name.startswith("harbor-deepswe-eval")
                ):
                    candidates.append(child)
    removed: list[str] = []
    now = time.time()
    for path in candidates:
        try:
            if not path.exists():
                continue
            if max_age_s is not None:
                age = now - path.stat().st_mtime
                if age < max_age_s:
                    continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
        except OSError:
            continue
    return removed


def resolve_eval_models(models: Sequence[str] | None = None) -> tuple[str, ...]:
    """Normalize + validate exact Grok+Kimi pair (allows openrouter/ prefix)."""
    if models is None:
        raw = list(DEEPSWE_EVAL_MODELS)
    else:
        raw = [normalize_model_id(m) for m in models if m and str(m).strip()]
    if not raw:
        raise DeepSWEEvalError("models must be non-empty")
    # Map normalized back through required set.
    try:
        return assert_required_panel_models(raw)
    except Exception as exc:  # noqa: BLE001 - surface as DeepSWEEvalError
        raise DeepSWEEvalError(str(exc)) from exc


def _pack_sort_key(pack_id: str) -> tuple[int, str]:
    try:
        idx = PREFERRED_PACK_IDS.index(pack_id)
    except ValueError:
        idx = 10_000
    return (idx, pack_id)


def load_product_packs(
    product_root: Path | str,
    *,
    max_packs: int | None = None,
    pack_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Load product Harbor packs from datasets/deepswe_v1 (or override)."""
    root = Path(product_root)
    if not root.exists():
        raise DeepSWEEvalError(f"product root missing: {root}")
    keeps = discover_real_pr_panel_keeps([root])
    if not keeps:
        # Fall back to bare task dirs without instruction (still Pier-loadable).
        tasks = root / "tasks"
        if tasks.is_dir():
            for child in sorted(tasks.iterdir()):
                if child.is_dir() and (child / "task.toml").is_file():
                    keeps.append(
                        {
                            "task_id": child.name,
                            "problem_statement": f"DeepSWE pack {child.name}",
                            "pack_path": str(child.resolve()),
                            "pack_id": child.name,
                            "source_track": "real_pr",
                        }
                    )
    if pack_ids:
        wanted = {p.strip() for p in pack_ids if p and p.strip()}
        keeps = [k for k in keeps if k["task_id"] in wanted or k["pack_id"] in wanted]
        missing = wanted - {k["task_id"] for k in keeps}
        if missing:
            raise DeepSWEEvalError(f"requested pack_ids not found: {sorted(missing)}")
    keeps.sort(key=lambda k: _pack_sort_key(str(k["task_id"])))
    if max_packs is not None:
        if max_packs < 0:
            raise DeepSWEEvalError("max_packs must be >= 0")
        keeps = keeps[:max_packs]
    return keeps


def _reward_solved(reward: float | int | None) -> bool:
    if reward is None:
        return False
    try:
        return float(reward) >= 1.0 - 1e-9
    except (TypeError, ValueError):
        return False


def harvest_miniswe_cost_usd(job_root: Path | str | None) -> Decimal:
    """Best-effort OpenRouter cost from mini-swe trajectory final_metrics.

    Never invents spend: missing trajectory or missing total_cost_usd → 0.
    Does not affect reward/solved (those come only from reward.json).
    """
    if job_root is None:
        return Decimal("0")
    root = Path(job_root)
    if not root.exists():
        return Decimal("0")
    # Prefer agent trajectory next to trial, then any *trajectory*.json under root.
    candidates: list[Path] = []
    if root.is_file() and root.name.endswith(".json"):
        candidates.append(root)
    else:
        candidates.extend(sorted(root.rglob("*trajectory*.json")))
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        if not isinstance(data, dict):
            continue
        metrics = data.get("final_metrics")
        if not isinstance(metrics, dict):
            continue
        raw = metrics.get("total_cost_usd")
        if raw is None:
            raw = metrics.get("cost_usd")
        if raw is None:
            continue
        try:
            cost = Decimal(str(raw))
        except Exception:  # noqa: BLE001
            continue
        if cost < 0:
            continue
        return cost
    return Decimal("0")


def preflight_pack_oracle_nop(
    pack_path: Path | str,
    *,
    pack_id: str | None = None,
    jobs_dir: Path | str = DEFAULT_JOBS_ROOT,
    pier_runner: PierRunner | None = None,
    pier_bin: Path | str | None = None,
    timeout_s: float = 900.0,
    offline_stub: PackPreflight | None = None,
) -> PackPreflight:
    """Dual-truth preflight: solution/oracle=1 and nop/null=0 (VAL-DEVAL-002).

    When ``offline_stub`` is provided (unit tests), no pier process is started.
    """
    path = Path(pack_path)
    pid = (pack_id or path.name).strip()
    if offline_stub is not None:
        return offline_stub
    if not path.is_dir():
        return PackPreflight(
            pack_id=pid,
            pack_path=str(path),
            ok=False,
            solution_reward=None,
            null_reward=None,
            errors=(f"pack dir missing: {path}",),
            mode="error",
        )
    jobs = Path(jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    runner: PierRunner
    if pier_runner is not None:
        runner = pier_runner
        mode = "injected"
    else:
        try:
            bin_path = resolve_pier_bin(pier_bin)
        except Exception as exc:  # noqa: BLE001
            return PackPreflight(
                pack_id=pid,
                pack_path=str(path),
                ok=False,
                solution_reward=None,
                null_reward=None,
                errors=(f"pier unavailable: {exc}",),
                mode="unavailable",
            )
        runner = SubprocessPierRunner(pier_bin=bin_path, timeout_sec=timeout_s)
        mode = "live-pier"

    errors: list[str] = []
    sol_reward: float | int | None = None
    null_reward: float | int | None = None
    try:
        sol_ev = runner.run(
            pack_dir=path,
            agent="oracle",
            jobs_dir=jobs,
            job_name=f"pre-sol-{pid}",
            n_concurrent=1,
        )
        sol_reward = sol_ev.reward.reward
        if not _reward_solved(sol_reward):
            errors.append(f"oracle/solution reward expected 1, got {sol_reward!r}")
        nop_ev = runner.run(
            pack_dir=path,
            agent="nop",
            jobs_dir=jobs,
            job_name=f"pre-nop-{pid}",
            n_concurrent=1,
        )
        null_reward = nop_ev.reward.reward
        if null_reward is None or float(null_reward) != 0.0:
            errors.append(f"nop/null reward expected 0, got {null_reward!r}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"preflight failed: {exc}")
        return PackPreflight(
            pack_id=pid,
            pack_path=str(path),
            ok=False,
            solution_reward=sol_reward,
            null_reward=null_reward,
            errors=tuple(errors),
            mode=mode,
        )
    ok = (
        not errors
        and _reward_solved(sol_reward)
        and null_reward is not None
        and float(null_reward) == 0.0
    )
    return PackPreflight(
        pack_id=pid,
        pack_path=str(path),
        ok=ok,
        solution_reward=sol_reward,
        null_reward=null_reward,
        errors=tuple(errors),
        mode=mode,
    )


def _default_live_miniswe_invoke(
    *,
    pack_path: Path,
    pack_id: str,
    model: str,
    jobs_dir: Path,
    index: int,
    timeout_s: float,
    pier_bin: Path | None = None,
) -> dict[str, Any]:
    """Shell pier run -a mini-swe-agent -m <openrouter/model> -p pack -n 1."""
    try:
        bin_path = resolve_pier_bin(pier_bin)
    except Exception as exc:  # noqa: BLE001
        return {
            "reward": None,
            "solved": False,
            "job_dir": None,
            "reward_path": None,
            "exit_code": None,
            "errors": (f"pier binary missing: {exc}",),
            "cost_usd": Decimal("0"),
            "ok": False,
        }
    jobs_dir.mkdir(parents=True, exist_ok=True)
    model_flag = openrouter_model_flag(model)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    job_name = f"eval-{pack_id}-{safe_model}-k{index}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    cmd = (
        str(bin_path),
        "run",
        "-a",
        PANEL_SCAFFOLD_AGENT,
        "-m",
        model_flag,
        "-p",
        str(pack_path),
        "-n",
        "1",
        "-k",
        "1",
        "-o",
        str(jobs_dir),
        "--job-name",
        job_name,
        "-y",
    )
    env = os.environ.copy()
    env.setdefault("MSWEA_MODEL_NAME", model_flag)
    env.setdefault("OPENROUTER_MODEL", model_flag)
    job_dir = jobs_dir / job_name
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        # Partial job trees often already contain trajectory spend; harvest for ledger.
        timeout_job_dir = str(job_dir) if job_dir.exists() else str(jobs_dir / job_name)
        cost_usd = harvest_miniswe_cost_usd(job_dir if job_dir.exists() else None)
        return {
            "reward": None,
            "solved": False,
            "job_dir": timeout_job_dir,
            "reward_path": None,
            "exit_code": None,
            "errors": (f"pier timeout after {timeout_s}s",),
            "cost_usd": cost_usd,
            "ok": False,
            "command": list(cmd),
            "invented_reward": False,
        }
    except OSError as exc:
        return {
            "reward": None,
            "solved": False,
            "job_dir": None,
            "reward_path": None,
            "exit_code": None,
            "errors": (f"pier spawn failed: {exc}",),
            "cost_usd": Decimal("0"),
            "ok": False,
            "command": list(cmd),
            "invented_reward": False,
        }

    # Pier may nest under timestamp; prefer exact job_name tree, else jobs_dir.
    search_root = job_dir if job_dir.exists() else jobs_dir
    evidence = parse_pier_job_result(search_root, agent="mini-swe-agent")
    if job_dir.exists():
        evidence = parse_pier_job_result(job_dir, agent="mini-swe-agent")
    reward = evidence.reward
    errors: list[str] = list(evidence.errors)
    if proc.returncode != 0:
        errors.append(f"pier exit_code={proc.returncode}")
    resolved_job: str | None = (
        str(job_dir)
        if job_dir.exists()
        else (str(search_root) if Path(search_root).exists() else None)
    )
    # In-path settle truth: harvest real mini-swe OpenRouter spend from trajectory
    # final_metrics (never invent; missing metrics → 0). Not reward.json.
    cost_usd = harvest_miniswe_cost_usd(resolved_job)
    return {
        "reward": reward,
        "solved": _reward_solved(reward),
        "job_dir": resolved_job,
        "reward_path": evidence.path,
        "exit_code": proc.returncode,
        "errors": tuple(errors),
        "cost_usd": cost_usd,
        "ok": proc.returncode == 0 and reward is not None,
        "command": list(cmd),
        "invented_reward": False,
    }


def mocked_miniswe_invoker(
    matrix: dict[str, dict[str, list[bool]]],
    *,
    job_root: Path | str | None = None,
) -> MiniSweInvoker:
    """Build an offline invoker from pack_id → model → list[bool solves].

    Writes a real ``reward.json`` under a mock job dir so harvest paths stay honest.
    Never invents a solve that is not in the matrix.

    Note: cost_usd is a fixed $0.01 offline stub for cheap unit paths. Live-path
    spend truth is covered by :func:`trajectory_backed_miniswe_invoker` and
    :func:`_default_live_miniswe_invoke` + :func:`harvest_miniswe_cost_usd`.
    """
    counters: dict[tuple[str, str], int] = {}
    root = Path(job_root) if job_root is not None else Path("/tmp/harbor-deepswe-eval-mock")
    root.mkdir(parents=True, exist_ok=True)

    def _invoke(
        *,
        pack_path: Path,
        pack_id: str,
        model: str,
        jobs_dir: Path,
        index: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        del pack_path, timeout_s
        mid = normalize_model_id(model)
        key = (pack_id, mid)
        row = (matrix.get(pack_id) or {}).get(mid) or (matrix.get(pack_id) or {}).get(model)
        if row is None:
            return {
                "reward": None,
                "solved": False,
                "job_dir": None,
                "reward_path": None,
                "exit_code": 1,
                "errors": (f"mock matrix missing pack={pack_id!r} model={mid!r}",),
                "cost_usd": Decimal("0.01"),
                "ok": False,
                "invented_reward": False,
            }
        pos = counters.get(key, 0)
        counters[key] = pos + 1
        if pos >= len(row):
            return {
                "reward": None,
                "solved": False,
                "job_dir": None,
                "reward_path": None,
                "exit_code": 1,
                "errors": (f"mock matrix exhausted for {pack_id}/{mid}",),
                "cost_usd": Decimal("0.01"),
                "ok": False,
                "invented_reward": False,
            }
        solved = bool(row[pos])
        reward_val: int = 1 if solved else 0
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_dir = jobs_dir / f"mock-{pack_id}-{mid.replace('/', '_')}-k{index}-{pos}"
        verifier = job_dir / "trial" / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        reward_path = verifier / "reward.json"
        reward_path.write_text(
            json.dumps({"reward": reward_val}, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "reward": reward_val,
            "solved": solved,
            "job_dir": str(job_dir),
            "reward_path": str(reward_path),
            "exit_code": 0,
            "errors": (),
            "cost_usd": Decimal("0.01"),
            "ok": True,
            "invented_reward": False,
        }

    return _invoke


def trajectory_backed_miniswe_invoker(
    matrix: dict[str, dict[str, list[bool]]],
    *,
    cost_usd_per_trial: Decimal | float | str,
    job_root: Path | str | None = None,
) -> MiniSweInvoker:
    """Invoker that writes fake trajectory.final_metrics.total_cost_usd and harvests it.

    Mirrors live path settle truth without calling pier/OpenRouter. Used so hard_stop
    unit tests do not depend solely on the offline $0.01 mock cost stub.
    Never invents rewards: solve/reward only come from the matrix / reward.json.
    """
    fixed_cost = Decimal(str(cost_usd_per_trial))
    if fixed_cost < 0:
        raise DeepSWEEvalError("cost_usd_per_trial must be >= 0")
    counters: dict[tuple[str, str], int] = {}
    root = Path(job_root) if job_root is not None else Path("/tmp/harbor-deepswe-eval-traj")
    root.mkdir(parents=True, exist_ok=True)

    def _invoke(
        *,
        pack_path: Path,
        pack_id: str,
        model: str,
        jobs_dir: Path,
        index: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        del pack_path, timeout_s
        mid = normalize_model_id(model)
        key = (pack_id, mid)
        row = (matrix.get(pack_id) or {}).get(mid) or (matrix.get(pack_id) or {}).get(model)
        if row is None:
            return {
                "reward": None,
                "solved": False,
                "job_dir": None,
                "reward_path": None,
                "exit_code": 1,
                "errors": (f"trajectory matrix missing pack={pack_id!r} model={mid!r}",),
                "cost_usd": Decimal("0"),
                "ok": False,
                "invented_reward": False,
            }
        pos = counters.get(key, 0)
        counters[key] = pos + 1
        if pos >= len(row):
            return {
                "reward": None,
                "solved": False,
                "job_dir": None,
                "reward_path": None,
                "exit_code": 1,
                "errors": (f"trajectory matrix exhausted for {pack_id}/{mid}",),
                "cost_usd": Decimal("0"),
                "ok": False,
                "invented_reward": False,
            }
        solved = bool(row[pos])
        reward_val: int = 1 if solved else 0
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_dir = jobs_dir / f"traj-{pack_id}-{mid.replace('/', '_')}-k{index}-{pos}"
        agent = job_dir / "trial" / "agent"
        verifier = job_dir / "trial" / "verifier"
        agent.mkdir(parents=True, exist_ok=True)
        verifier.mkdir(parents=True, exist_ok=True)
        traj_path = agent / "trajectory.json"
        traj_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "final_metrics": {
                        "total_prompt_tokens": 1000,
                        "total_completion_tokens": 100,
                        "total_cost_usd": float(fixed_cost),
                        "total_steps": 3,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        reward_path = verifier / "reward.json"
        reward_path.write_text(
            json.dumps({"reward": reward_val}, indent=2) + "\n", encoding="utf-8"
        )
        # Same harvest function as live invoker (do not hard-code cost on return).
        cost_usd = harvest_miniswe_cost_usd(job_dir)
        return {
            "reward": reward_val,
            "solved": solved,
            "job_dir": str(job_dir),
            "reward_path": str(reward_path),
            "exit_code": 0,
            "errors": (),
            "cost_usd": cost_usd,
            "ok": True,
            "invented_reward": False,
        }

    return _invoke


def _decision_from_model_stats(
    per_model: dict[str, tuple[int, int]],
) -> BandDecision:
    """Classify band from model → (solves, k)."""
    per_pass = {m: compute_pass_at_k(s, k) for m, (s, k) in per_model.items()}
    total_solves = sum(s for s, _k in per_model.values())
    total_trials = sum(k for _s, k in per_model.values())
    disc = compute_discrimination(per_pass)
    return classify_band(
        per_model_pass_at_k=per_pass,
        total_solves=total_solves,
        total_trials=total_trials,
        discrimination=disc,
        config=BandFilterConfig(),
    )


def run_deepswe_eval(
    *,
    product_root: Path | str = DEFAULT_PRODUCT_ROOT,
    out_dir: Path | str = DEFAULT_OUT_ROOT,
    max_packs: int | None = 5,
    pack_ids: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    k: int = DEFAULT_EVAL_K,
    n_concurrent: int = DEFAULT_N_CONCURRENT,
    hard_stop_usd: float | str | Decimal = DEFAULT_HARD_STOP_USD,
    reserve_usd: float | str | Decimal = DEFAULT_TRIAL_RESERVE_USD,
    jobs_dir: Path | str = DEFAULT_JOBS_ROOT,
    pier_bin: Path | str | None = None,
    ledger: BudgetLedger | None = None,
    ledger_path: Path | str | None = None,
    preflight: bool = True,
    preflight_stubs: dict[str, PackPreflight] | None = None,
    invoker: MiniSweInvoker | None = None,
    offline: bool = False,
    trial_timeout_s: float = 3600.0,
    reclaim: bool = True,
    skip_preflight_fail: bool = True,
) -> DeepSWEEvalReport:
    """Run DeepSWE-grade pier mini-swe serial eval and write report.json.

    Offline tests pass ``offline=True`` + a mocked ``invoker`` (and optional
    ``preflight_stubs``). Live calls leave invoker None (uses pier binary).
    """
    if k <= 0:
        raise DeepSWEEvalError(f"k must be positive; got {k}")
    if int(n_concurrent) != 1:
        raise DeepSWEEvalError(
            f"n_concurrent must be 1 for DeepSWE pier/docker serial eval; got {n_concurrent}"
        )
    models_t = resolve_eval_models(models)
    hard_stop = Decimal(str(hard_stop_usd))
    reserve = Decimal(str(reserve_usd))
    if hard_stop <= 0:
        raise DeepSWEEvalError("hard_stop_usd must be > 0")
    if reserve <= 0:
        raise DeepSWEEvalError("reserve_usd must be > 0")

    product = Path(product_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = Path(jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)

    if reclaim:
        reclaim_jobs_dirs([jobs], max_age_s=None)

    keeps = load_product_packs(product, max_packs=max_packs, pack_ids=pack_ids)
    mem_before = read_host_mem_gib()
    started = datetime.now(UTC).isoformat()
    t0 = time.monotonic()

    if ledger is None:
        lpath = Path(ledger_path) if ledger_path is not None else (out / "ledger.jsonl")
        # Wave hard-stop is the eval cap (default $300), independent of global $600.
        ledger = BudgetLedger(
            lpath,
            cap_usd=hard_stop,
            worst_case_cost_usd=reserve,
            run_id=f"deepswe-eval-{uuid.uuid4().hex[:8]}",
        )
    else:
        lpath = ledger.path

    pier_bin_s: str | None = None
    if offline:
        # Offline must not require pier binary.
        if invoker is None:
            raise DeepSWEEvalError("offline eval requires an injected mini-swe invoker")
        invoke_fn = invoker
    else:
        try:
            pier_bin_s = str(resolve_pier_bin(pier_bin))
        except Exception:
            pier_bin_s = str(pier_bin) if pier_bin is not None else str(DEFAULT_PIER_BIN)

        if invoker is not None:
            invoke_fn = invoker
        else:
            bin_path = Path(pier_bin_s) if pier_bin_s else None

            def invoke_fn(
                *,
                pack_path: Path,
                pack_id: str,
                model: str,
                jobs_dir: Path,
                index: int,
                timeout_s: float,
            ) -> dict[str, Any]:
                return _default_live_miniswe_invoke(
                    pack_path=pack_path,
                    pack_id=pack_id,
                    model=model,
                    jobs_dir=jobs_dir,
                    index=index,
                    timeout_s=timeout_s,
                    pier_bin=bin_path,
                )

    pack_results: list[PackEvalResult] = []
    total_spend = Decimal("0")
    budget_stop = False
    stop_reason: str | None = None
    n_scored = 0
    n_preflight_ok = 0
    invented_any = False

    for keep in keeps:
        pack_id = str(keep["task_id"])
        pack_path = Path(str(keep["pack_path"]))
        pf: PackPreflight | None = None
        if preflight:
            stub = (preflight_stubs or {}).get(pack_id)
            if offline and stub is None:
                # Offline default: dual-truth passes without pier.
                stub = PackPreflight(
                    pack_id=pack_id,
                    pack_path=str(pack_path),
                    ok=True,
                    solution_reward=1,
                    null_reward=0,
                    mode="offline-stub",
                )
            pf = preflight_pack_oracle_nop(
                pack_path,
                pack_id=pack_id,
                jobs_dir=jobs / "_preflight",
                pier_bin=pier_bin,
                offline_stub=stub,
            )
            if pf.ok:
                n_preflight_ok += 1
            elif skip_preflight_fail:
                pack_results.append(
                    PackEvalResult(
                        pack_id=pack_id,
                        pack_path=str(pack_path),
                        models=[],
                        decision=None,
                        preflight=pf,
                        total_cost_usd=Decimal("0"),
                        budget_stop=False,
                        complete=False,
                        stop_reason="preflight_failed",
                    )
                )
                continue
            else:
                pack_results.append(
                    PackEvalResult(
                        pack_id=pack_id,
                        pack_path=str(pack_path),
                        models=[],
                        decision=None,
                        preflight=pf,
                        total_cost_usd=Decimal("0"),
                        budget_stop=False,
                        complete=False,
                        stop_reason="preflight_failed_hard",
                    )
                )
                break

        # Budget gate: need full matrix for this pack (models × k).
        need = reserve * Decimal(len(models_t) * k)
        remaining = ledger.remaining_usd()
        if remaining < need or ledger.has_unknown_billing():
            budget_stop = True
            stop_reason = (
                f"budget_stop: remaining_usd={format(remaining, 'f')} "
                f"< need {format(need, 'f')} before pack {pack_id!r}; "
                "no further paid pier trials; no invented rewards"
            )
            pack_results.append(
                PackEvalResult(
                    pack_id=pack_id,
                    pack_path=str(pack_path),
                    models=[],
                    decision=None,
                    preflight=pf,
                    total_cost_usd=Decimal("0"),
                    budget_stop=True,
                    complete=False,
                    stop_reason=stop_reason,
                )
            )
            break

        model_stats: list[ModelPackStats] = []
        pack_cost = Decimal("0")
        pack_budget_stop = False
        pack_incomplete = False
        per_model_sk: dict[str, tuple[int, int]] = {}
        stop_mid = False

        for model in models_t:
            trials: list[TrialReward] = []
            solves = 0
            for i in range(k):
                rem = ledger.remaining_usd()
                if rem < reserve or ledger.has_unknown_billing():
                    pack_budget_stop = True
                    budget_stop = True
                    pack_incomplete = True
                    stop_reason = (
                        f"budget_stop: remaining_usd={format(rem, 'f')} "
                        f"< reserve {format(reserve, 'f')} mid pack {pack_id!r} "
                        f"model={model!r} trial={i}"
                    )
                    stop_mid = True
                    break
                try:
                    physical = ledger.reserve(
                        stage=DEEPSWE_EVAL_STAGE,
                        task_id=pack_id,
                        model=model,
                        reserved_cost_usd=reserve,
                    )
                except AccountingError as exc:
                    pack_budget_stop = True
                    budget_stop = True
                    pack_incomplete = True
                    stop_reason = f"budget_stop: reserve failed: {exc}"
                    stop_mid = True
                    break

                inv = invoke_fn(
                    pack_path=pack_path,
                    pack_id=pack_id,
                    model=model,
                    jobs_dir=jobs,
                    index=i,
                    timeout_s=trial_timeout_s,
                )
                reward = inv.get("reward")
                if isinstance(reward, bool):
                    reward = 1 if reward else 0
                if reward is not None and not isinstance(reward, (int, float)):
                    reward = None
                solved = bool(inv.get("solved")) if "solved" in inv else _reward_solved(reward)
                if solved:
                    solves += 1
                cost_raw = inv.get("cost_usd", Decimal("0"))
                try:
                    cost = Decimal(str(cost_raw))
                except Exception:  # noqa: BLE001
                    cost = Decimal("0")
                if cost < 0:
                    cost = Decimal("0")
                if cost > reserve:
                    # Cap settle at reserved (accounting integrity).
                    cost = reserve
                invented = bool(inv.get("invented_reward", False))
                if invented:
                    invented_any = True
                errors = tuple(inv.get("errors") or ())
                trial = TrialReward(
                    pack_id=pack_id,
                    model=model,
                    index=i,
                    reward=reward if isinstance(reward, (int, float)) else None,
                    solved=solved,
                    job_dir=str(inv["job_dir"]) if inv.get("job_dir") else None,
                    reward_path=str(inv["reward_path"]) if inv.get("reward_path") else None,
                    physical_call_id=physical,
                    cost_usd=cost,
                    exit_code=inv.get("exit_code")
                    if isinstance(inv.get("exit_code"), int)
                    else None,
                    errors=errors if isinstance(errors, tuple) else tuple(errors),
                    invented_reward=invented,
                    openrouter_model=openrouter_model_flag(model),
                )
                trials.append(trial)
                pack_cost += cost
                total_spend += cost
                try:
                    ledger.settle(
                        physical,
                        cost_usd=cost,
                        status="success" if inv.get("ok", True) or reward is not None else "error",
                        usage={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    )
                except AccountingError as exc:
                    pack_budget_stop = True
                    budget_stop = True
                    pack_incomplete = True
                    stop_reason = f"budget_stop: settle failed: {exc}"
                    stop_mid = True
                    break

            completed = len(trials)
            incomplete = completed < k
            if incomplete:
                pack_incomplete = True
            pass_k = compute_pass_at_k(solves, k) if k > 0 else 0.0
            per_model_sk[model] = (solves, k)
            model_stats.append(
                ModelPackStats(
                    pack_id=pack_id,
                    model=model,
                    k=k,
                    solves=solves,
                    pass_at_k=pass_k,
                    trials=tuple(trials),
                    completed_trials=completed,
                    incomplete=incomplete,
                )
            )
            if stop_mid:
                break

        decision: BandDecision | None = None
        complete = not pack_incomplete and all(not m.incomplete for m in model_stats)
        if model_stats and not pack_incomplete:
            # Only band-classify full matrices.
            if all(not m.incomplete for m in model_stats):
                decision = _decision_from_model_stats(per_model_sk)
                complete = True
                n_scored += 1
            else:
                complete = False
        elif model_stats and not pack_budget_stop:
            # Partial model set without budget stop still counts as incomplete.
            complete = False
        if complete and not pack_budget_stop and decision is None and model_stats:
            decision = _decision_from_model_stats(per_model_sk)
            n_scored += 1
            complete = True

        pack_results.append(
            PackEvalResult(
                pack_id=pack_id,
                pack_path=str(pack_path),
                models=model_stats,
                decision=decision,
                preflight=pf,
                total_cost_usd=pack_cost,
                budget_stop=pack_budget_stop,
                complete=complete,
                stop_reason=stop_reason if pack_budget_stop else None,
            )
        )
        if pack_budget_stop:
            break

    finished = datetime.now(UTC).isoformat()
    wall = time.monotonic() - t0
    mem_after = read_host_mem_gib()
    remaining_after = ledger.remaining_usd()
    if remaining_after < 0:
        remaining_after = Decimal("0")

    report = DeepSWEEvalReport(
        fidelity=DEEPSWE_EVAL_FIDELITY,
        models=list(models_t),
        n_concurrent=1,
        k=k,
        hard_stop_usd=hard_stop,
        product_root=str(product.resolve()) if product.exists() else str(product),
        pier_bin=pier_bin_s,
        agent=PANEL_SCAFFOLD_AGENT,
        scaffold=PANEL_SCAFFOLD_NAME,
        runtime=PANEL_SCAFFOLD_RUNTIME,
        n_packs_requested=len(keeps),
        n_packs_scored=n_scored,
        n_packs_preflight_ok=n_preflight_ok,
        pack_results=pack_results,
        total_spend_usd=total_spend,
        remaining_usd=remaining_after,
        budget_stop=budget_stop,
        stop_reason=stop_reason,
        invented_rewards=invented_any,
        wall_s=round(wall, 3),
        preflight_enabled=preflight,
        offline=offline,
        preferred_pack_ids=list(PREFERRED_PACK_IDS),
        host_mem_before=mem_before,
        host_mem_after=mem_after,
        ledger_path=str(lpath),
        out_dir=str(out.resolve()),
        jobs_dir=str(jobs.resolve()) if jobs.exists() else str(jobs),
        started_at=started,
        finished_at=finished,
    )
    write_eval_report(out, report)
    with contextlib.suppress(Exception):
        ledger.write_summary_json(out / "ledger_summary.json")
    return report


def write_eval_report(out_dir: Path | str, report: DeepSWEEvalReport) -> Path:
    """Write report.json under out_dir; return path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "DEEPSWE_EVAL_FIDELITY",
    "DEEPSWE_EVAL_MODELS",
    "DEEPSWE_EVAL_STAGE",
    "DEFAULT_EVAL_K",
    "DEFAULT_HARD_STOP_USD",
    "DEFAULT_JOBS_ROOT",
    "DEFAULT_N_CONCURRENT",
    "DEFAULT_OUT_ROOT",
    "DEFAULT_PRODUCT_ROOT",
    "DEFAULT_TRIAL_RESERVE_USD",
    "PREFERRED_PACK_IDS",
    "DeepSWEBudgetStop",
    "DeepSWEEvalError",
    "DeepSWEEvalReport",
    "MiniSweInvoker",
    "ModelPackStats",
    "PackEvalResult",
    "PackPreflight",
    "TrialReward",
    "default_ledger_path",
    "harvest_miniswe_cost_usd",
    "load_product_packs",
    "mocked_miniswe_invoker",
    "normalize_model_id",
    "openrouter_model_flag",
    "preflight_pack_oracle_nop",
    "read_host_mem_gib",
    "reclaim_jobs_dirs",
    "resolve_eval_models",
    "run_deepswe_eval",
    "trajectory_backed_miniswe_invoker",
    "write_eval_report",
    "_default_live_miniswe_invoke",
]
