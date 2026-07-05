#!/usr/bin/env python
"""Self-budgeted, restart-safe background harvest driver for ``forge build --pilot``.

The m6-pilot-build harvest sweeps MANY borderline candidates (the difficulty
ladder x seeds x the modular band-producing repos boltons/jmespath, plus the
mutation-adequate ``qs`` pr_mirror entries) and lets the UNCHANGED band filter
select the in-band keeps. Because the harvest needs many hours and prior worker
sessions died at the session-length limit before shipping a keep, the compute is
run here as a DETACHED background OS process that outlives the worker session.

Guarantees implemented here:

* **Self-budgeted.** A hard global ceiling of ``$HARVEST_BUDGET_USD`` (default
  1400) OR ``$HARVEST_KEEP_TARGET`` shipped keeps (default 18, absolute cap 30),
  whichever first, enforced INSIDE this process so an unsupervised run can never
  overspend.
* **Budget continuity across restarts.** On startup the prior cumulative
  ``spend_usd`` / ``batches_done`` are read from the progress file and the global
  budget CONTINUES from there (never reset to 0). An interrupted in-flight
  batch's reserved estimate is committed to the running total and the batch is
  skipped, so the global ceiling holds across every restart combined.
* **Incremental keep materialization.** Each batch runs :func:`run_pilot`, whose
  checkpoint machinery ships every band-keep the instant it is found. After each
  batch the keeps are merged into the canonical ``results/pilot_keeps/`` tree and
  the ``dataset.jsonl`` / ``dataset.parquet`` are rebuilt from the shipped set, so
  an interruption never loses a shipped keep.
* **No gate/band loosening.** The band filter (``band_high``) and every oracle
  gate run exactly as configured in the pipeline; this driver only chooses the
  candidate supply and the budget/stop policy.

Gold-eval and the benchmark report are intentionally NOT run here -- they are the
finalization step the monitoring worker performs once over the full shipped set.

Secrets are never printed: only counts, costs, verdicts, and reject reasons are
logged; the API key / GitHub token never reach any surface.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_DIR = REPO_ROOT / "results" / "pilot_keeps"
CANON_TASKS = OUT_DIR / "tasks"
BATCHES_DIR = OUT_DIR / "batches"
PROGRESS = OUT_DIR / "harvest_progress.json"
LOG = OUT_DIR / "harvest.log"

BUDGET_USD = float(os.environ.get("HARVEST_BUDGET_USD", "1400"))
KEEP_TARGET = int(os.environ.get("HARVEST_KEEP_TARGET", "18"))
KEEP_HARD_CAP = int(os.environ.get("HARVEST_KEEP_HARD_CAP", "30"))
BATCH_SIZE = int(os.environ.get("HARVEST_BATCH_SIZE", "16"))
CANDIDATE_CONCURRENCY = int(os.environ.get("HARVEST_CANDIDATE_CONCURRENCY", "8"))
ROLLOUT_CONCURRENCY = int(os.environ.get("HARVEST_ROLLOUT_CONCURRENCY", "4"))
K = int(os.environ.get("HARVEST_K", "6"))
RESERVE_PER_CAND = float(os.environ.get("HARVEST_RESERVE_PER_CAND", "6.0"))
SEEDS = int(os.environ.get("HARVEST_SEEDS", "44"))

# Hard-band rungs only: ~25% of hard-rung oracle-passers land in-band vs ~17%
# overall (m6-band-supply), so restricting to the large-symbol needles maximizes
# the in-band rate per (paid) candidate. faults stays 2 (the count that clears
# the oracle cleanly); difficulty is centered on the fault-LOCATE axis.
RUNGS: tuple[dict[str, object], ...] = (
    {"faults": 2, "min_symbol_lines": 20, "prefer": "largest"},
    {"faults": 2, "min_symbol_lines": 30, "prefer": "largest"},
)
# Two amplifier generators so the kept set can span >=2 generators (VAL-CROSS-005).
STRUCT_GENERATORS = ("bug_combination", "multi_file")
# Modular band-producing repos (pure-logic; jmespath never hangs, boltons is the
# best band producer). cachetools/socket/threading targets are avoided.
STRUCT_REPO_IDS = ("mahmoud/boltons", "jmespath/jmespath.py")
# Mutation-adequate pr_mirror entries for JS + generator/language diversity.
PR_MIRROR_REPO_IDS = ("ljharb/qs#555", "ljharb/qs#559")


def _load_env_fallback() -> None:
    """Ensure forge credentials are present (launcher exports them; fallback here)."""
    if os.environ.get("TEACHER_LLM_API_KEY"):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        with LOG.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def read_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text())
        except Exception:
            return {}
    return {}


#: Last full progress state (without the volatile heartbeat fields), so the
#: periodic heartbeat can refresh ``heartbeat_ts`` mid-batch without disturbing
#: the budget/keep state machine -- a future worker polling across sessions can
#: then tell "alive but mid-batch" from "hung".
_last_progress: dict = {}


def write_progress(data: dict) -> None:
    global _last_progress
    _last_progress = dict(data)
    out = dict(data)
    out["heartbeat_ts"] = time.time()
    out["heartbeat_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = PROGRESS.with_name(PROGRESS.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, PROGRESS)


async def _heartbeat(interval: float = 30.0) -> None:
    """Refresh ``heartbeat_ts`` on the last full progress state every ``interval``s."""
    while True:
        await asyncio.sleep(interval)
        if _last_progress:
            try:
                write_progress(_last_progress)
            except Exception:
                pass


def _canonical_task_ids() -> set[str]:
    if not CANON_TASKS.is_dir():
        return set()
    return {d.name for d in CANON_TASKS.iterdir() if d.is_dir()}


def merge_and_rebuild() -> tuple[int, dict[str, int], dict[str, int]]:
    """Merge every batch's shipped keeps into the canonical tree, rebuild datasets.

    Driven by each batch's ``dataset.jsonl`` (the committed-keep record written by
    the checkpoint AFTER the workspace), so a canonical task always has a matching
    dataset row -- guaranteeing ``tasks/ == jsonl == parquet`` (VAL-CROSS-004).
    Idempotent: a keep already present in the canonical tree is skipped, and a
    re-materialization of the same candidate (same deterministic id) never dups.
    Returns (shipped_count, generator_breakdown, language_breakdown).
    """
    from swe_forge.export.jsonl import export_jsonl, import_jsonl
    from swe_forge.export.parquet import export_parquet

    seen: dict[str, object] = {}
    for batch_jsonl in sorted(BATCHES_DIR.glob("batch-*/dataset.jsonl")):
        batch_tasks = batch_jsonl.parent / "tasks"
        try:
            rows = import_jsonl(batch_jsonl)
        except Exception:
            continue
        for task in rows:
            tid = task.id
            if tid in seen:
                continue
            src = batch_tasks / tid
            if not src.is_dir():
                continue
            dest = CANON_TASKS / tid
            if not dest.exists():
                try:
                    shutil.copytree(src, dest)
                except Exception:
                    continue
            seen[tid] = task

    records = [seen[tid] for tid in sorted(seen)]
    export_jsonl(records, OUT_DIR / "dataset.jsonl", append=False)
    export_parquet(records, OUT_DIR / "dataset.parquet")

    gen_bd: dict[str, int] = {}
    lang_bd: dict[str, int] = {}
    try:
        from swe_forge.forge.report import load_task_provenances

        for prov in load_task_provenances(CANON_TASKS):
            gen_bd[prov.generator] = gen_bd.get(prov.generator, 0) + 1
            lang_bd[prov.language] = lang_bd.get(prov.language, 0) + 1
    except Exception:
        pass
    return len(records), gen_bd, lang_bd


def build_plans() -> list:
    """Weighted, interleaved candidate plan list (seed-outer for early diversity)."""
    from swe_forge.forge.pilot import CandidatePlan
    from swe_forge.forge.sources import build_source_registry

    src = build_source_registry()
    by_id = {r.repo_id: r for r in src.specs()}
    struct_repos = [by_id[rid] for rid in STRUCT_REPO_IDS if rid in by_id]
    pr_repos = [by_id[rid] for rid in PR_MIRROR_REPO_IDS if rid in by_id]

    plans: list = []
    for seed in range(SEEDS):
        for repo in struct_repos:
            for generator in STRUCT_GENERATORS:
                for rung in RUNGS:
                    plans.append(
                        CandidatePlan(
                            repo=repo,
                            generator=generator,
                            seed=seed,
                            params=dict(rung),
                        )
                    )
        # Periodic pr_mirror attempts (deterministic candidate; stochastic gates).
        if seed % 3 == 0:
            for repo in pr_repos:
                plans.append(
                    CandidatePlan(
                        repo=repo,
                        generator=repo.preferred_generator,
                        seed=seed,
                        params=repo.pr_params(),
                    )
                )
    return plans


async def run_batch(batch_idx: int, batch_plans: list, validate_models: bool):
    from swe_forge.forge.pilot import PilotConfig, run_pilot

    batch_out = BATCHES_DIR / f"batch-{batch_idx:04d}"
    config = PilotConfig(
        plans=batch_plans,
        out_dir=batch_out,
        k=K,
        concurrency=ROLLOUT_CONCURRENCY,
        candidate_concurrency=CANDIDATE_CONCURRENCY,
        validate_models=validate_models,
        run_gold_eval=False,
        write_report=False,
    )
    return await run_pilot(config, handle_signals=False)


async def amain() -> int:
    _load_env_fallback()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANON_TASKS.mkdir(parents=True, exist_ok=True)
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    prog = read_progress()
    spend = float(prog.get("spend_usd", 0.0))
    batches_done = int(prog.get("batches_done", 0))
    candidates_done = int(prog.get("candidates_done", 0))
    reserved = float(prog.get("reserved_usd", 0.0))
    in_flight = prog.get("in_flight_batch", None)

    # A reserve left on disk means the prior process died mid-batch: commit the
    # reserve to the running total (conservative -- never under-count) and skip
    # that interrupted batch so the global ceiling holds across restarts.
    if reserved > 0 and in_flight is not None:
        spend += reserved
        batches_done = max(batches_done, int(in_flight) + 1)
        reserved = 0.0

    shipped, gen_bd, lang_bd = merge_and_rebuild()
    log(
        f"startup pid={os.getpid()} resumed spend=${spend:.2f} "
        f"batches_done={batches_done} shipped={shipped} gens={gen_bd} langs={lang_bd} "
        f"budget=${BUDGET_USD:.0f} keep_target={KEEP_TARGET}"
    )

    plans = build_plans()
    log(f"plan list: {len(plans)} candidates")

    loop = asyncio.get_running_loop()
    state: dict[str, object] = {"stop": False, "task": None}

    def _on_signal() -> None:
        state["stop"] = True
        task = state["task"]
        if task is not None:
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    heartbeat_task = asyncio.ensure_future(_heartbeat())
    final_status = "plans_exhausted"
    validated = False

    while True:
        shipped = len(_canonical_task_ids())
        if state["stop"]:
            final_status = "stopped"
            break
        if shipped >= min(KEEP_TARGET, KEEP_HARD_CAP):
            final_status = "target_reached"
            break
        if spend + RESERVE_PER_CAND * BATCH_SIZE > BUDGET_USD:
            final_status = "budget_exhausted"
            break

        start = batches_done * BATCH_SIZE
        batch_plans = plans[start : start + BATCH_SIZE]
        if not batch_plans:
            final_status = "plans_exhausted"
            break

        reserved = RESERVE_PER_CAND * len(batch_plans)
        write_progress(
            {
                "status": "running",
                "pid": os.getpid(),
                "logfile": str(LOG),
                "progress_file": str(PROGRESS),
                "spend_usd": round(spend, 4),
                "reserved_usd": round(reserved, 4),
                "in_flight_batch": batches_done,
                "shipped_keeps": shipped,
                "candidates_done": candidates_done,
                "batches_done": batches_done,
                "generator_breakdown": gen_bd,
                "language_breakdown": lang_bd,
                "budget_usd": BUDGET_USD,
                "keep_target": KEEP_TARGET,
            }
        )
        log(
            f"batch {batches_done}: {len(batch_plans)} candidates "
            f"(cumulative spend=${spend:.2f}, shipped={shipped})"
        )

        batch_task = asyncio.ensure_future(
            run_batch(batches_done, batch_plans, validate_models=not validated)
        )
        state["task"] = batch_task
        try:
            outcome = await batch_task
        except asyncio.CancelledError:
            spend += reserved
            reserved = 0.0
            shipped, gen_bd, lang_bd = merge_and_rebuild()
            log(f"batch {batches_done} cancelled (SIGTERM); keeps preserved, stopping")
            final_status = "stopped"
            break
        except BaseException as exc:  # noqa: BLE001 - a batch failure must not overspend
            spend += reserved
            reserved = 0.0
            batches_done += 1
            shipped, gen_bd, lang_bd = merge_and_rebuild()
            log(f"batch failed: {type(exc).__name__}: {exc} -- skipping to next")
            write_progress(
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "logfile": str(LOG),
                    "progress_file": str(PROGRESS),
                    "spend_usd": round(spend, 4),
                    "reserved_usd": 0.0,
                    "in_flight_batch": None,
                    "shipped_keeps": shipped,
                    "candidates_done": candidates_done,
                    "batches_done": batches_done,
                    "generator_breakdown": gen_bd,
                    "language_breakdown": lang_bd,
                    "budget_usd": BUDGET_USD,
                    "keep_target": KEEP_TARGET,
                }
            )
            continue
        finally:
            state["task"] = None

        validated = True
        cost = float(outcome.usage.total_cost)
        spend += cost
        reserved = 0.0
        candidates_done += len(batch_plans)
        batches_done += 1
        shipped, gen_bd, lang_bd = merge_and_rebuild()
        c = outcome.counts
        log(
            f"batch {batches_done - 1} done: cost=${cost:.2f} tokens={outcome.usage.total_tokens} "
            f"funnel[sourced={c.sourced} env={c.env_built} synth={c.synthesized} "
            f"oracle_pass={c.oracle_pass} keep={c.calibration_keep}] "
            f"| cumulative spend=${spend:.2f} shipped={shipped} gens={gen_bd} langs={lang_bd}"
        )
        write_progress(
            {
                "status": "running",
                "pid": os.getpid(),
                "logfile": str(LOG),
                "progress_file": str(PROGRESS),
                "spend_usd": round(spend, 4),
                "reserved_usd": 0.0,
                "in_flight_batch": None,
                "shipped_keeps": shipped,
                "candidates_done": candidates_done,
                "batches_done": batches_done,
                "generator_breakdown": gen_bd,
                "language_breakdown": lang_bd,
                "budget_usd": BUDGET_USD,
                "keep_target": KEEP_TARGET,
            }
        )

    shipped, gen_bd, lang_bd = merge_and_rebuild()
    write_progress(
        {
            "status": final_status,
            "pid": os.getpid(),
            "logfile": str(LOG),
            "progress_file": str(PROGRESS),
            "spend_usd": round(spend, 4),
            "reserved_usd": 0.0,
            "in_flight_batch": None,
            "shipped_keeps": shipped,
            "candidates_done": candidates_done,
            "batches_done": batches_done,
            "generator_breakdown": gen_bd,
            "language_breakdown": lang_bd,
            "budget_usd": BUDGET_USD,
            "keep_target": KEEP_TARGET,
            "finished_ts": time.time(),
        }
    )
    heartbeat_task.cancel()
    log(
        f"HARVEST FINISHED status={final_status} spend=${spend:.2f} "
        f"shipped={shipped} gens={gen_bd} langs={lang_bd} candidates_done={candidates_done}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
