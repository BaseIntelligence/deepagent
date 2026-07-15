"""V1 ship harvest: expand certified keeps to the 20–50 band under budget.

Pipeline reuses ``run_micro_keep`` with diversification across modular allowlist
seeds (python / js / go) and both mutation kinds. Prefers synthetic_grounded
multi-file tasks with pinned SHAs. Real PR track is attempted when network/
token allow; under-supply is reported explicitly in ``report.md``.

Artifacts under ``datasets/v1/``:
  tasks.jsonl, tasks/, report.md, gate_audit.jsonl, ledger_summary.json
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from swe_factory.accounting import BudgetLedger, default_ledger_path
from swe_factory.config import FactorySettings, load_settings
from swe_factory.export.jsonl import read_tasks_jsonl
from swe_factory.export.workspace import ExportError, write_export_bundle
from swe_factory.pipeline.micro_keep import MicroKeepResult, run_micro_keep
from swe_factory.schema import SourceTrack, TaskRecord
from swe_factory.sources.allowlist import SeedRepo, get_seed
from swe_factory.sources.clone import is_immutable_sha

DEFAULT_TARGET_KEEPS = 20
DEFAULT_MAX_KEEPS = 50
DEFAULT_MAX_ATTEMPTS = 120
DEFAULT_PANEL_K = 2
DEFAULT_ATTEMPT_CAP_USD = Decimal("12")  # per micro-keep attempt micro-cap
DEFAULT_GLOBAL_SOFT_STOP_USD = Decimal("550")  # leave headroom under hard 600
DEFAULT_PANEL_NEED_USD = Decimal("6.0")  # biggest worst-case k=2 * 2 models * 1.5


@dataclass(frozen=True, slots=True)
class HarvestPlanItem:
    seed_id: str
    mutation: str
    diversification_index: int
    language: str
    prefer_stems: tuple[str, ...] = ()
    exclude_stems: tuple[str, ...] = ()
    note: str = ""


@dataclass
class AttemptRecord:
    index: int
    seed_id: str
    mutation: str
    diversification_index: int
    ok: bool
    is_keep: bool
    escalated: bool
    reason: str
    instance_id: str | None
    source_track: str | None
    language: str | None
    spend_exact_usd: str
    panel_rule: str | None = None
    panel_pass_at_k: float | None = None
    stage_funnel: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed_id": self.seed_id,
            "mutation": self.mutation,
            "diversification_index": self.diversification_index,
            "ok": self.ok,
            "is_keep": self.is_keep,
            "escalated": self.escalated,
            "reason": self.reason,
            "instance_id": self.instance_id,
            "source_track": self.source_track,
            "language": self.language,
            "spend_exact_usd": self.spend_exact_usd,
            "panel_rule": self.panel_rule,
            "panel_pass_at_k": self.panel_pass_at_k,
            "stage_funnel": dict(self.stage_funnel),
        }


@dataclass
class ShipV1Result:
    ok: bool
    keep_count: int
    target_keeps: int
    max_keeps: int
    out_dir: Path
    tasks_jsonl: Path | None
    report_path: Path | None
    gate_audit_path: Path | None
    ledger_summary_path: Path | None
    attempts: list[AttemptRecord]
    funnel: dict[str, Any]
    languages: dict[str, int]
    source_tracks: dict[str, int]
    spend_total_usd: str
    remaining_usd: str
    under_cap: bool
    under_supply_reasons: list[str]
    seeded_from: list[str]
    spot_checks: list[dict[str, Any]]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "keep_count": self.keep_count,
            "target_keeps": self.target_keeps,
            "max_keeps": self.max_keeps,
            "out_dir": str(self.out_dir),
            "tasks_jsonl": str(self.tasks_jsonl) if self.tasks_jsonl else None,
            "report_path": str(self.report_path) if self.report_path else None,
            "gate_audit_path": str(self.gate_audit_path) if self.gate_audit_path else None,
            "ledger_summary_path": (
                str(self.ledger_summary_path) if self.ledger_summary_path else None
            ),
            "attempts": [a.to_dict() for a in self.attempts],
            "funnel": dict(self.funnel),
            "languages": dict(self.languages),
            "source_tracks": dict(self.source_tracks),
            "spend_total_usd": self.spend_total_usd,
            "remaining_usd": self.remaining_usd,
            "under_cap": self.under_cap,
            "under_supply_reasons": list(self.under_supply_reasons),
            "seeded_from": list(self.seeded_from),
            "spot_checks": list(self.spot_checks),
            "reason": self.reason,
        }


def _boltons_stem_batches() -> list[tuple[str, ...]]:
    """Non-overlapping prefer batches so retries dig different modules."""
    stems = [
        "mathutils",
        "strutils",
        "iterutils",
        "dictutils",
        "listutils",
        "setutils",
        "formatutils",
        "funcutils",
        "timeutils",
        "urlutils",
        "typeutils",
        "statsutils",
        "cacheutils",
        "tableutils",
        "tbutils",
        "namedutils",
        "fileutils",
        "ioutils",
        "jsonutils",
        "socketutils",
        "gcutils",
        "ecoutils",
        "debugutils",
    ]
    batches: list[tuple[str, ...]] = []
    for i in range(0, len(stems), 2):
        batches.append(tuple(stems[i : i + 2]))
    return batches


def default_harvest_plan(
    *,
    prefer_python: bool = True,
    include_js: bool = True,
    include_go: bool = True,
    max_items: int = DEFAULT_MAX_ATTEMPTS,
    boltons_extra_rounds: int = 2,
) -> list[HarvestPlanItem]:
    """Diversified modular plan: boltons-first python, then cachetools/js/go."""
    items: list[HarvestPlanItem] = []
    batches = _boltons_stem_batches()
    # High-yield synthetic path first: boltons function_removal then multi_fault.
    if prefer_python:
        for round_i in range(max(1, int(boltons_extra_rounds))):
            for batch_i, batch in enumerate(batches):
                div = batch_i + round_i * len(batches)
                # Rotate stems further each round so pair order flips.
                stems = tuple(list(batch)[::-1]) if round_i % 2 else batch
                items.append(
                    HarvestPlanItem(
                        seed_id="python_boltons",
                        mutation="function_removal",
                        diversification_index=div,
                        language="python",
                        prefer_stems=stems,
                        note=f"boltons FR r{round_i} d{div}",
                    )
                )
        for round_i in range(max(1, int(boltons_extra_rounds))):
            for batch_i, batch in enumerate(batches):
                div = batch_i + round_i * len(batches)
                stems = tuple(list(batch)[::-1]) if round_i % 2 else batch
                items.append(
                    HarvestPlanItem(
                        seed_id="python_boltons",
                        mutation="multi_fault",
                        diversification_index=div + 50 * round_i,
                        language="python",
                        prefer_stems=stems,
                        note=f"boltons MF r{round_i} d{div}",
                    )
                )
        # Triple-stem prefer rows for more multi_fault motors
        triple = (
            ("mathutils", "strutils", "dictutils"),
            ("iterutils", "listutils", "setutils"),
            ("timeutils", "urlutils", "formatutils"),
            ("tableutils", "jsonutils", "fileutils"),
            ("statsutils", "typeutils", "cacheutils"),
            ("namedutils", "tbutils", "funcutils"),
            ("ioutils", "socketutils", "gcutils"),
            ("ecoutils", "debugutils", "strutils"),
        )
        for i, stems in enumerate(triple):
            items.append(
                HarvestPlanItem(
                    seed_id="python_boltons",
                    mutation="function_removal",
                    diversification_index=200 + i,
                    language="python",
                    prefer_stems=stems,
                    note="boltons FR triple",
                )
            )
            items.append(
                HarvestPlanItem(
                    seed_id="python_boltons",
                    mutation="multi_fault",
                    diversification_index=300 + i,
                    language="python",
                    prefer_stems=stems,
                    note="boltons MF triple",
                )
            )
        for i in range(12):
            items.append(
                HarvestPlanItem(
                    seed_id="python_cachetools",
                    mutation="function_removal" if i % 2 == 0 else "multi_fault",
                    diversification_index=i,
                    language="python",
                    note="cachetools diversified",
                )
            )
    if include_js:
        for seed_id in ("js_qs", "js_validator"):
            for i in range(8):
                items.append(
                    HarvestPlanItem(
                        seed_id=seed_id,
                        mutation="function_removal" if i % 2 == 0 else "multi_fault",
                        diversification_index=i,
                        language="javascript",
                        note=f"{seed_id} synth",
                    )
                )
    if include_go:
        for seed_id in ("go_uuid", "go_cast"):
            for i in range(8):
                items.append(
                    HarvestPlanItem(
                        seed_id=seed_id,
                        mutation="function_removal" if i % 2 == 0 else "multi_fault",
                        diversification_index=i,
                        language="go",
                        note=f"{seed_id} synth",
                    )
                )
    # Round-robin soften language dominance slightly after first python wave.
    # (kept order is already boltons-first — intentional yield bias)
    return items[:max_items]


def _load_task_from_export(export_dir: Path) -> TaskRecord | None:
    jsonl = export_dir / "tasks.jsonl"
    if not jsonl.is_file():
        return None
    tasks = read_tasks_jsonl(jsonl)
    return tasks[0] if tasks else None


def _ingest_existing_seed(
    seed_export: Path,
    *,
    kept_ids: set[str],
    kept_tasks: list[TaskRecord],
    broken_map: dict[str, Path],
    seeded_from: list[str],
    gate_audit_out: Path,
) -> TaskRecord | None:
    """Import a previously certified keep into the V1 set if still valid."""
    task = _load_task_from_export(seed_export)
    if task is None:
        return None
    # Panel hardness + multi-file + track required.
    if task.panel is None or task.panel.pass_at_k is None:
        return None
    pk = float(task.panel.pass_at_k)
    if not (0.0 < pk <= 0.5):
        return None
    if task.panel.discrimination is None or float(task.panel.discrimination) < 1.0:
        return None
    track = (
        task.source_track.value if hasattr(task.source_track, "value") else str(task.source_track)
    )
    if track not in {SourceTrack.REAL_PR.value, SourceTrack.SYNTHETIC_GROUNDED.value}:
        return None
    if not is_immutable_sha(task.base_commit):
        return None
    gold_files = [
        line[6:].split("\t", 1)[0]
        for line in task.gold_patch.splitlines()
        if line.startswith("+++ b/")
    ]
    if len({f for f in gold_files if f and f != "/dev/null"}) < 2:
        # count via schema-friendly: still require multi-file hard floor
        from swe_factory.oracle.gates import count_files_in_patch

        if len(count_files_in_patch(task.gold_patch)) < 2:
            return None
    if task.instance_id in kept_ids:
        return None

    # Locate broken workspace if present under tasks/<id>/repo
    repo_ws = seed_export / "tasks" / task.instance_id / "repo"
    if not repo_ws.is_dir():
        # Some exports keep repo under tasks/<id>/repo only when write_export_bundle ran
        return None

    kept_ids.add(task.instance_id)
    kept_tasks.append(task)
    broken_map[task.instance_id] = repo_ws
    seeded_from.append(str(seed_export))

    # Append prior gate audit if any.
    for name in ("gate_audit.jsonl",):
        src = seed_export.parent / name
        if not src.is_file():
            src = seed_export / name
        if src.is_file():
            gate_audit_out.parent.mkdir(parents=True, exist_ok=True)
            with (
                gate_audit_out.open("a", encoding="utf-8") as fh,
                src.open(encoding="utf-8") as src_fh,
            ):
                for line in src_fh:
                    if line.strip():
                        fh.write(line if line.endswith("\n") else line + "\n")
    return task


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _render_ship_report(
    *,
    funnel: dict[str, Any],
    languages: dict[str, int],
    source_tracks: dict[str, int],
    hardness: list[dict[str, Any]],
    spend_total: str,
    remaining: str,
    under_cap: bool,
    keep_count: int,
    target: int,
    under_supply: Sequence[str],
    attempts_summary: Sequence[dict[str, Any]],
    spot_checks: Sequence[dict[str, Any]],
    seeded_from: Sequence[str],
) -> str:
    lines = [
        "# SWE Dataset Factory V1 — ship report",
        "",
        f"- generated_at: `{datetime.now(UTC).isoformat()}`",
        f"- certified_keeps: **{keep_count}** (target band 20–50; soft target {target})",
        f"- ledger total_commit_usd: **{spend_total}**",
        f"- ledger remaining_usd: **{remaining}**",
        f"- under_cap ($600): **{under_cap}**",
        "",
        "## Funnel",
        "",
    ]
    for key in sorted(funnel):
        lines.append(f"- {key}: {funnel[key]}")
    lines.extend(["", "## Languages", ""])
    for lang in ("python", "javascript", "go"):
        count = languages.get(lang, 0)
        lines.append(f"- {lang}: {count}")
    for lang, count in sorted(languages.items()):
        if lang not in {"python", "javascript", "go"}:
            lines.append(f"- {lang}: {count}")
    lines.extend(["", "## Source tracks", ""])
    for track in ("synthetic_grounded", "real_pr"):
        lines.append(f"- {track}: {source_tracks.get(track, 0)}")
    lines.extend(["", "## Hardness summary (keeps)", ""])
    if not hardness:
        lines.append("- (none)")
    else:
        pks = [h["pass_at_k"] for h in hardness if h.get("pass_at_k") is not None]
        discs = [h["discrimination"] for h in hardness if h.get("discrimination") is not None]
        lines.append(f"- keep_count: {len(hardness)}")
        if pks:
            lines.append(
                f"- pass@k min/mean/max: "
                f"{min(pks):.4f} / {sum(pks) / len(pks):.4f} / {max(pks):.4f}"
            )
        if discs:
            lines.append(
                f"- discrimination min/mean/max: "
                f"{min(discs):.4f} / {sum(discs) / len(discs):.4f} / {max(discs):.4f}"
            )
        for h in hardness[:50]:
            lines.append(
                f"- `{h.get('instance_id')}` lang={h.get('language')} "
                f"track={h.get('source_track')} "
                f"pass@k={h.get('pass_at_k')} disc={h.get('discrimination')} "
                f"rule={h.get('band_rule')}"
            )
    lines.extend(["", "## Under-supply / language honesty", ""])
    if under_supply:
        for reason in under_supply:
            lines.append(f"- {reason}")
    else:
        lines.append("- no under-supply notes")
    lines.extend(["", "## Seeded from prior keeps", ""])
    if seeded_from:
        for s in seeded_from:
            lines.append(f"- `{s}`")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Harness spot-checks", ""])
    if not spot_checks:
        lines.append("- (not run)")
    else:
        for sc in spot_checks:
            lines.append(
                f"- `{sc.get('instance_id')}` gold_resolve={sc.get('gold_resolve')} "
                f"null_resolve={sc.get('null_resolve')} backend={sc.get('backend')} "
                f"ok={sc.get('ok')}"
            )
    lines.extend(["", "## Attempt disposition sample", ""])
    for a in list(attempts_summary)[:40]:
        reason = str(a.get("reason") or "")[:120]
        lines.append(
            f"- #{a.get('index')} {a.get('seed_id')}/{a.get('mutation')}"
            f" div={a.get('diversification_index')} keep={a.get('is_keep')} "
            f"reason={reason}"
        )
    lines.append("")
    return "\n".join(lines)


def _spot_check_gold_null(
    tasks: Sequence[TaskRecord],
    broken_map: dict[str, Path],
    *,
    n: int = 3,
    backend: str = "local",
) -> list[dict[str, Any]]:
    """Spot-check gold=resolve and null=0 on up to n tasks.

    Prefer host-local soft apply (fast) for ship reporting; falls back to
    FakeOracle when docker not requested. No gate relaxation.
    """
    if not tasks:
        return []
    sample = list(tasks)
    random.shuffle(sample)
    sample = sample[: max(1, min(n, len(sample)))]
    results: list[dict[str, Any]] = []

    if backend == "fake":
        from swe_factory.harness.score import score_gold_and_null
        from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite

        runner = FakeOracleRunner(
            broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
            gold_runs=[ScriptedSuite(f2p_exits=[0], p2p_exits=[0])],
            null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        )
        for task in sample:
            ws = broken_map.get(task.instance_id)
            if ws is None or not Path(ws).is_dir():
                results.append(
                    {
                        "instance_id": task.instance_id,
                        "ok": False,
                        "error": "missing broken workspace",
                        "backend": "fake",
                    }
                )
                continue
            pair = score_gold_and_null(task=task, workspace=ws, runner=runner)
            results.append(
                {
                    "instance_id": task.instance_id,
                    "gold_resolve": pair.gold.resolve,
                    "null_resolve": pair.null.resolve,
                    "ok": pair.passed,
                    "backend": "fake",
                }
            )
        return results

    # Local: git-apply gold / empty and run F2P+P2P via host shell.

    for task in sample:
        ws = broken_map.get(task.instance_id)
        if ws is None or not Path(ws).is_dir():
            results.append(
                {
                    "instance_id": task.instance_id,
                    "ok": False,
                    "error": "missing broken workspace",
                    "backend": "local",
                }
            )
            continue
        gold_ok = _local_resolve(Path(ws), task, patch=task.gold_patch)
        null_ok = _local_resolve(Path(ws), task, patch="")
        results.append(
            {
                "instance_id": task.instance_id,
                "gold_resolve": gold_ok,
                "null_resolve": null_ok,
                "ok": bool(gold_ok) and (not null_ok),
                "backend": "local",
            }
        )
    return results


def _local_resolve(broken: Path, task: TaskRecord, *, patch: str) -> bool:
    """Apply patch on a temp copy and run F2P+P2P with install commands best-effort."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sdf-spot-") as tmp:
        dest = Path(tmp) / "ws"
        shutil.copytree(
            broken,
            dest,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", ".venv", "node_modules", ".pytest_cache"
            ),
        )
        if patch.strip():
            patch_path = Path(tmp) / "cand.patch"
            patch_path.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")
            apply = subprocess.run(
                ["git", "apply", "--unsafe-paths", "--whitespace=nowarn", str(patch_path)],
                cwd=str(dest),
                capture_output=True,
                text=True,
                check=False,
            )
            if apply.returncode != 0:
                # try with -p0 / reverse heuristics
                apply2 = subprocess.run(
                    ["patch", "-p1", "--forward", "--batch", "-i", str(patch_path)],
                    cwd=str(dest),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if apply2.returncode != 0:
                    return False
        # install if python package
        if (dest / "setup.py").exists() or (dest / "pyproject.toml").exists():
            subprocess.run(
                "pip install -q -e . && pip install -q pytest",
                cwd=str(dest),
                shell=True,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        for cmd in list(task.fail_to_pass) + list(task.pass_to_pass):
            proc = subprocess.run(
                cmd,
                cwd=str(dest),
                shell=True,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if proc.returncode != 0:
                return False
        return True


def run_ship_v1(
    *,
    out_dir: Path | str = Path("datasets/v1"),
    target_keeps: int = DEFAULT_TARGET_KEEPS,
    max_keeps: int = DEFAULT_MAX_KEEPS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    panel_k: int = DEFAULT_PANEL_K,
    settings: FactorySettings | None = None,
    ledger: BudgetLedger | None = None,
    plan: Sequence[HarvestPlanItem] | None = None,
    seed_exports: Sequence[Path | str] | None = None,
    live_panel: bool = True,
    use_docker_oracle: bool = True,
    soft_backend: str = "local",
    attempt_micro_cap_usd: Decimal = DEFAULT_ATTEMPT_CAP_USD,
    soft_stop_spend_usd: Decimal = DEFAULT_GLOBAL_SOFT_STOP_USD,
    work_root: Path | str | None = None,
    spot_check_n: int = 3,
    spot_backend: str = "local",
    try_real_pr: bool = True,
) -> ShipV1Result:
    """Expand factory until target_keeps certified keeps (capped) under budget."""
    settings = settings or load_settings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Keep harvest workspaces OUTSIDE the ship export tree so leak scan on
    # datasets/v1 cannot see candidate gold.patch files under work/.
    work = Path(work_root) if work_root else out.parent / f".harvest_work_{out.name}"
    work.mkdir(parents=True, exist_ok=True)
    # Drop any legacy in-tree harvest dir that would poison leak scans.
    legacy_work = out / "_harvest_work"
    if legacy_work.exists():
        shutil.rmtree(legacy_work, ignore_errors=True)
    gate_audit = out / "gate_audit.jsonl"
    progress_path = out / "harvest_progress.jsonl"

    if ledger is None:
        ledger_path = default_ledger_path(Path.cwd())
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = BudgetLedger(
            ledger_path,
            cap_usd=settings.budget_usd,
            worst_case_cost_usd=Decimal("1.50"),
            run_id="ship-v1",
        )

    kept_tasks: list[TaskRecord] = []
    kept_ids: set[str] = set()
    broken_map: dict[str, Path] = {}
    seeded_from: list[str] = []
    attempts: list[AttemptRecord] = []
    under_supply: list[str] = []
    funnel: dict[str, Any] = {
        "candidates_started": 0,
        "sources_ok": 0,
        "envbuild_ok": 0,
        "produce_ok": 0,
        "oracle_pass": 0,
        "oracle_fail": 0,
        "panel_run": 0,
        "panel_keep": 0,
        "panel_drop": 0,
        "export_ok": 0,
        "seeded_ingested": 0,
        "real_pr_attempts": 0,
        "real_pr_keeps": 0,
        "synth_keeps": 0,
    }

    # Seed from prior definitive micro-keep export(s) + resume current out if present
    seeds = list(seed_exports or [])
    default_seed = Path("datasets/micro_keep_boltons_fr2/export")
    if default_seed.is_dir() and default_seed not in seeds:
        seeds.insert(0, default_seed)
    # Resume prior ship export under the same out_dir when re-running.
    if (out / "tasks.jsonl").is_file() and out not in seeds:
        seeds.insert(0, out)
    for s in seeds:
        sp = Path(s)
        if not sp.is_dir():
            continue
        # Multi-task resume: load every valid keep from tasks.jsonl
        jsonl = sp / "tasks.jsonl"
        if jsonl.is_file():
            try:
                prior_tasks = read_tasks_jsonl(jsonl)
            except Exception:  # noqa: BLE001
                prior_tasks = []
            for task in prior_tasks:
                if task.instance_id in kept_ids:
                    continue
                # reuse ingest validation by writing through a temp single-task path
                repo_ws = sp / "tasks" / task.instance_id / "repo"
                meta_ws = out / "_meta" / "broken_cache" / task.instance_id
                if not repo_ws.is_dir() and meta_ws.is_dir():
                    repo_ws = meta_ws
                if not repo_ws.is_dir():
                    continue
                if task.panel is None or task.panel.pass_at_k is None:
                    continue
                pk = float(task.panel.pass_at_k)
                if not (0.0 < pk <= 0.5):
                    continue
                if task.panel.discrimination is None or float(task.panel.discrimination) < 1.0:
                    continue
                track = (
                    task.source_track.value
                    if hasattr(task.source_track, "value")
                    else str(task.source_track)
                )
                if track not in {
                    SourceTrack.REAL_PR.value,
                    SourceTrack.SYNTHETIC_GROUNDED.value,
                }:
                    continue
                if not is_immutable_sha(task.base_commit):
                    continue
                from swe_factory.oracle.gates import count_files_in_patch

                if len(count_files_in_patch(task.gold_patch)) < 2:
                    continue
                kept_ids.add(task.instance_id)
                kept_tasks.append(task)
                broken_map[task.instance_id] = repo_ws
                seeded_from.append(f"{sp}#{task.instance_id}")
                funnel["seeded_ingested"] += 1
                funnel["export_ok"] += 1
                funnel["panel_keep"] += 1
                funnel["oracle_pass"] += 1
                funnel["produce_ok"] += 1
                if track == SourceTrack.SYNTHETIC_GROUNDED.value:
                    funnel["synth_keeps"] += 1
                else:
                    funnel["real_pr_keeps"] += 1
                _append_jsonl(
                    progress_path,
                    {
                        "event": "seed_ingest",
                        "instance_id": task.instance_id,
                        "source": str(sp),
                    },
                )
            continue
        t = _ingest_existing_seed(
            sp,
            kept_ids=kept_ids,
            kept_tasks=kept_tasks,
            broken_map=broken_map,
            seeded_from=seeded_from,
            gate_audit_out=gate_audit,
        )
        if t is not None:
            funnel["seeded_ingested"] += 1
            funnel["export_ok"] += 1
            funnel["panel_keep"] += 1
            funnel["oracle_pass"] += 1
            funnel["produce_ok"] += 1
            funnel["synth_keeps"] += 1
            _append_jsonl(
                progress_path,
                {
                    "event": "seed_ingest",
                    "instance_id": t.instance_id,
                    "source": str(sp),
                },
            )

    harvest_plan = (
        list(plan)
        if plan is not None
        else default_harvest_plan(max_items=max_attempts, boltons_extra_rounds=3)
    )

    # Skip plan keys already attempted in prior harvest_progress.jsonl.
    prior_keys: set[tuple[str, str, int]] = set()
    if progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event"):
                continue
            if row.get("seed_id") and row.get("mutation") is not None:
                prior_keys.add(
                    (
                        str(row.get("seed_id")),
                        str(row.get("mutation")),
                        int(row.get("diversification_index") or 0),
                    )
                )
    if prior_keys:
        harvest_plan = [
            it
            for it in harvest_plan
            if (it.seed_id, it.mutation, it.diversification_index) not in prior_keys
        ]

    def remaining() -> Decimal:
        return ledger.remaining_usd()

    def total_commit() -> Decimal:
        return ledger.summary().total_commit_usd

    # Optional real_pr probe (non-fatal)
    if try_real_pr and len(kept_tasks) < target_keeps and remaining() > DEFAULT_PANEL_NEED_USD:
        try:
            rp_keep = _try_one_real_pr_keep(
                work_root=work / "real_pr",
                ledger=ledger,
                settings=settings,
                live_panel=live_panel,
                panel_k=panel_k,
                use_docker_oracle=use_docker_oracle,
                gate_audit=gate_audit,
            )
            funnel["real_pr_attempts"] += 1
            if rp_keep is not None:
                task, broken = rp_keep
                if task.instance_id not in kept_ids:
                    kept_ids.add(task.instance_id)
                    kept_tasks.append(task)
                    broken_map[task.instance_id] = broken
                    funnel["real_pr_keeps"] += 1
                    funnel["panel_keep"] += 1
                    funnel["export_ok"] += 1
                    funnel["oracle_pass"] += 1
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "real_pr_keep",
                            "instance_id": task.instance_id,
                        },
                    )
            else:
                under_supply.append(
                    "real_pr: no certified keep from initial live miner probe "
                    "(rate limit, multi-file+tests filter, oracle, or panel band)"
                )
        except Exception as exc:  # noqa: BLE001
            under_supply.append(f"real_pr track under-supply: {exc}")

    idx = 0
    for item in harvest_plan:
        if len(kept_tasks) >= max_keeps:
            break
        if len(kept_tasks) >= target_keeps and total_commit() >= soft_stop_spend_usd:
            # reached band and soft-stop spend
            break
        if len(kept_tasks) >= target_keeps:
            # target hit — stop harvesting more (user asked 20–50; 20 is enough)
            break
        if remaining() < DEFAULT_PANEL_NEED_USD:
            under_supply.append(
                f"budget soft-stop: remaining ${remaining()} < panel need "
                f"${DEFAULT_PANEL_NEED_USD} after {len(kept_tasks)} keeps"
            )
            break
        if total_commit() >= soft_stop_spend_usd and len(kept_tasks) >= max(1, target_keeps // 2):
            under_supply.append(
                f"soft budget ceiling ${soft_stop_spend_usd} reached with {len(kept_tasks)} keeps"
            )
            break

        # Validate seed is still immutable
        try:
            seed = get_seed(item.seed_id)
        except KeyError as exc:
            attempts.append(
                AttemptRecord(
                    index=idx,
                    seed_id=item.seed_id,
                    mutation=item.mutation,
                    diversification_index=item.diversification_index,
                    ok=False,
                    is_keep=False,
                    escalated=False,
                    reason=str(exc),
                    instance_id=None,
                    source_track=None,
                    language=item.language,
                    spend_exact_usd="0",
                )
            )
            idx += 1
            continue
        if not is_immutable_sha(seed.base_commit):
            under_supply.append(
                f"seed {item.seed_id} skipped: non-immutable base_commit {seed.base_commit!r}"
            )
            idx += 1
            continue

        attempt_out = (
            work / f"attempt_{idx:04d}_{item.seed_id}_{item.mutation}_{item.diversification_index}"
        )
        funnel["candidates_started"] += 1
        try:
            result: MicroKeepResult = run_micro_keep(
                out_dir=attempt_out,
                seed_id=item.seed_id,
                mutation=item.mutation,
                settings=settings,
                ledger=ledger,
                micro_cap_usd=attempt_micro_cap_usd,
                panel_k=panel_k,
                use_docker_oracle=use_docker_oracle,
                use_docker_envbuild=False,
                soft_backend=soft_backend,  # type: ignore[arg-type]
                live_panel=live_panel,
                require_immutable_sha=True,
                diversification_index=item.diversification_index,
                prefer_stems=list(item.prefer_stems) if item.prefer_stems else None,
                exclude_stems=list(item.exclude_stems) if item.exclude_stems else None,
            )
        except Exception as exc:  # noqa: BLE001
            rec = AttemptRecord(
                index=idx,
                seed_id=item.seed_id,
                mutation=item.mutation,
                diversification_index=item.diversification_index,
                ok=False,
                is_keep=False,
                escalated=False,
                reason=f"exception: {exc}",
                instance_id=None,
                source_track=None,
                language=item.language,
                spend_exact_usd="0",
            )
            attempts.append(rec)
            _append_jsonl(progress_path, rec.to_dict())
            idx += 1
            continue

        # Merge stage funnel (skip candidates_started — already counted above)
        for k, v in result.funnel.items():
            if k == "candidates_started":
                continue
            if k in funnel and isinstance(v, int):
                funnel[k] = int(funnel.get(k, 0)) + int(v)

        panel_rule = None
        panel_pk = None
        if result.panel and isinstance(result.panel, dict):
            dec = result.panel.get("decision") or result.panel
            if isinstance(dec, dict):
                panel_rule = dec.get("rule") or dec.get("band_rule")
                panel_pk = dec.get("frontier_pass_at_k")
                if panel_pk is None:
                    panel_pk = dec.get("pass_at_k")

        rec = AttemptRecord(
            index=idx,
            seed_id=item.seed_id,
            mutation=item.mutation,
            diversification_index=item.diversification_index,
            ok=result.ok,
            is_keep=result.is_keep,
            escalated=result.escalated,
            reason=result.reason,
            instance_id=result.instance_id,
            source_track=result.source_track,
            language=item.language,
            spend_exact_usd=format(result.spend_exact_usd, "f"),
            panel_rule=str(panel_rule) if panel_rule is not None else None,
            panel_pass_at_k=float(panel_pk) if panel_pk is not None else None,
            stage_funnel=dict(result.funnel),
        )
        attempts.append(rec)
        _append_jsonl(progress_path, rec.to_dict())

        # Copy gate audit rows
        attempt_audit = attempt_out / "gate_audit.jsonl"
        if attempt_audit.is_file():
            with (
                gate_audit.open("a", encoding="utf-8") as fh,
                attempt_audit.open(encoding="utf-8") as src,
            ):
                for line in src:
                    if line.strip():
                        fh.write(line if line.endswith("\n") else line + "\n")

        if (
            result.is_keep
            and result.task is not None
            and result.export_dir is not None
            and result.instance_id not in kept_ids
        ):
            kept_ids.add(result.task.instance_id)
            kept_tasks.append(result.task)
            # broken workspace from export layout or produce workdir
            exported_repo = Path(result.export_dir) / "tasks" / result.task.instance_id / "repo"
            if exported_repo.is_dir():
                broken_map[result.task.instance_id] = exported_repo
            funnel["synth_keeps"] += 1
            # Interim snapshot for durability
            _write_ship_bundle(
                out=out,
                tasks=kept_tasks,
                broken_map=broken_map,
                funnel=funnel,
                attempts=attempts,
                ledger=ledger,
                under_supply=under_supply,
                seeded_from=seeded_from,
                target_keeps=target_keeps,
                spot_checks=[],
                interim=True,
            )
        idx += 1

    # Language / track honesty
    languages = _count_languages(kept_tasks)
    source_tracks = _count_tracks(kept_tasks)
    for lang in ("python", "javascript", "go"):
        if languages.get(lang, 0) == 0:
            under_supply.append(
                f"language={lang}: zero certified keeps after budgeted synth harvest "
                f"(see funnel/attempts — modular seed sites may lack motorized F2P "
                f"or panel kept solve-none/all)"
            )
    if source_tracks.get("real_pr", 0) == 0:
        under_supply.append(
            "source_track=real_pr: zero certified keeps; synthetic_grounded carried "
            "yield. Explicit under-supply (PR filter/oracle/panel/API)."
        )
    if source_tracks.get("synthetic_grounded", 0) == 0:
        under_supply.append("source_track=synthetic_grounded: zero keeps (unexpected).")

    # Final export + spot checks
    spot_checks: list[dict[str, Any]] = []
    tasks_jsonl: Path | None = None
    report_path: Path | None = None
    ledger_summary_path: Path | None = None
    reason = ""
    ok = False

    if not kept_tasks:
        reason = "no certified keeps after harvest; honest empty ship refused"
        under_supply.append(reason)
        # Still write empty-ish report + ledger snapshot for auditability
        report_path = out / "report.md"
        report_path.write_text(
            _render_ship_report(
                funnel=funnel,
                languages=languages,
                source_tracks=source_tracks,
                hardness=[],
                spend_total=format(total_commit(), "f"),
                remaining=format(remaining(), "f"),
                under_cap=ledger.summary().under_cap,
                keep_count=0,
                target=target_keeps,
                under_supply=under_supply,
                attempts_summary=[a.to_dict() for a in attempts],
                spot_checks=[],
                seeded_from=seeded_from,
            ),
            encoding="utf-8",
        )
        ledger_summary_path = ledger.write_summary_json(out / "ledger_summary.json")
        return ShipV1Result(
            ok=False,
            keep_count=0,
            target_keeps=target_keeps,
            max_keeps=max_keeps,
            out_dir=out,
            tasks_jsonl=None,
            report_path=report_path,
            gate_audit_path=gate_audit if gate_audit.is_file() else None,
            ledger_summary_path=ledger_summary_path,
            attempts=attempts,
            funnel=funnel,
            languages=languages,
            source_tracks=source_tracks,
            spend_total_usd=format(total_commit(), "f"),
            remaining_usd=format(remaining(), "f"),
            under_cap=ledger.summary().under_cap,
            under_supply_reasons=under_supply,
            seeded_from=seeded_from,
            spot_checks=[],
            reason=reason,
        )

    if len(kept_tasks) > max_keeps:
        kept_tasks = kept_tasks[:max_keeps]

    try:
        spot_checks = _spot_check_gold_null(
            kept_tasks, broken_map, n=spot_check_n, backend=spot_backend
        )
        bundle_meta = _write_ship_bundle(
            out=out,
            tasks=kept_tasks,
            broken_map=broken_map,
            funnel=funnel,
            attempts=attempts,
            ledger=ledger,
            under_supply=under_supply,
            seeded_from=seeded_from,
            target_keeps=target_keeps,
            spot_checks=spot_checks,
            interim=False,
        )
        tasks_jsonl = bundle_meta["tasks_jsonl"]
        report_path = bundle_meta["report_path"]
        ledger_summary_path = bundle_meta["ledger_summary_path"]
    except Exception as exc:  # noqa: BLE001
        reason = f"final export failed: {exc}"
        return ShipV1Result(
            ok=False,
            keep_count=len(kept_tasks),
            target_keeps=target_keeps,
            max_keeps=max_keeps,
            out_dir=out,
            tasks_jsonl=None,
            report_path=None,
            gate_audit_path=gate_audit if gate_audit.is_file() else None,
            ledger_summary_path=None,
            attempts=attempts,
            funnel=funnel,
            languages=languages,
            source_tracks=source_tracks,
            spend_total_usd=format(total_commit(), "f"),
            remaining_usd=format(remaining(), "f"),
            under_cap=ledger.summary().under_cap,
            under_supply_reasons=under_supply + [reason],
            seeded_from=seeded_from,
            spot_checks=spot_checks,
            reason=reason,
        )

    keep_count = len(kept_tasks)
    ok = 20 <= keep_count <= 50 and ledger.summary().under_cap
    if keep_count < 20:
        reason = (
            f"partial ship: only {keep_count} certified keeps (<20). "
            "Budget superiority or funnel yield insufficient without gate relaxation."
        )
        ok = False
    elif not all(sc.get("ok") for sc in spot_checks) and spot_checks:
        reason = "ship artifacts written but harness spot-check failures present"
        # still success if keep band met: report honestly; VAL requires spot check pass
        ok = False
    else:
        reason = f"shipped {keep_count} certified keeps in band under cap"

    return ShipV1Result(
        ok=ok,
        keep_count=keep_count,
        target_keeps=target_keeps,
        max_keeps=max_keeps,
        out_dir=out,
        tasks_jsonl=tasks_jsonl,
        report_path=report_path,
        gate_audit_path=gate_audit if gate_audit.is_file() else None,
        ledger_summary_path=ledger_summary_path,
        attempts=attempts,
        funnel=funnel,
        languages=languages,
        source_tracks=source_tracks,
        spend_total_usd=format(total_commit(), "f"),
        remaining_usd=format(remaining(), "f"),
        under_cap=ledger.summary().under_cap,
        under_supply_reasons=under_supply,
        seeded_from=seeded_from,
        spot_checks=spot_checks,
        reason=reason,
    )


def _count_languages(tasks: Sequence[TaskRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        lang = str(t.language).lower()
        if lang in {"js", "ts", "typescript"}:
            lang = "javascript"
        out[lang] = out.get(lang, 0) + 1
    return out


def _count_tracks(tasks: Sequence[TaskRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        track = t.source_track.value if hasattr(t.source_track, "value") else str(t.source_track)
        out[track] = out.get(track, 0) + 1
    return out


def _write_ship_bundle(
    *,
    out: Path,
    tasks: Sequence[TaskRecord],
    broken_map: dict[str, Path],
    funnel: dict[str, Any],
    attempts: Sequence[AttemptRecord],
    ledger: BudgetLedger,
    under_supply: Sequence[str],
    seeded_from: Sequence[str],
    target_keeps: int,
    spot_checks: Sequence[dict[str, Any]],
    interim: bool,
) -> dict[str, Path]:
    if not tasks:
        raise ExportError("refusing empty ship bundle")

    out.mkdir(parents=True, exist_ok=True)
    meta = out / "_meta"
    meta.mkdir(parents=True, exist_ok=True)

    # Preserve durable harvest files that write_export_bundle would wipe.
    durable_names = (
        "gate_audit.jsonl",
        "harvest_progress.jsonl",
        "attempts.jsonl",
        "report.md",
        "ledger_summary.json",
        "ship_manifest.json",
    )
    preserved: dict[str, str] = {}
    for name in durable_names:
        p = out / name
        if p.is_file():
            preserved[name] = p.read_text(encoding="utf-8")
        meta_p = meta / name
        if meta_p.is_file() and name not in preserved:
            preserved[name] = meta_p.read_text(encoding="utf-8")

    # Copy broken repos into a stable location outside the wiped export tree.
    stable_broken: dict[str, Path] = {}
    broken_root = meta / "broken_cache"
    broken_root.mkdir(parents=True, exist_ok=True)
    for tid, src in broken_map.items():
        dest = broken_root / tid
        if Path(src).is_dir() and not dest.is_dir():
            shutil.copytree(
                src,
                dest,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", ".venv", "node_modules"
                ),
            )
        if dest.is_dir():
            stable_broken[tid] = dest
        elif Path(src).is_dir():
            stable_broken[tid] = Path(src)

    # Stage export so we only overlay product artifacts onto out/
    staging = meta / "export_staging"
    if staging.exists():
        shutil.rmtree(staging)
    bundle = write_export_bundle(
        tasks=list(tasks),
        out_dir=staging,
        broken_repos=stable_broken,
        require_clean_leak_scan=True,
        require_panel=True,
        overwrite=True,
    )

    # Overlay tasks.jsonl + tasks/ into out without erasing durable harvest logs.
    final_jsonl = out / "tasks.jsonl"
    final_jsonl.write_bytes(Path(bundle.tasks_jsonl).read_bytes())
    tasks_dest = out / "tasks"
    if tasks_dest.exists():
        shutil.rmtree(tasks_dest)
    shutil.copytree(staging / "tasks", tasks_dest)
    for extra in ("export_manifest.json",):
        src = staging / extra
        if src.is_file():
            shutil.copy2(src, out / extra)

    languages = _count_languages(tasks)
    source_tracks = _count_tracks(tasks)
    hardness = []
    for t in tasks:
        hardness.append(
            {
                "instance_id": t.instance_id,
                "language": t.language,
                "source_track": (
                    t.source_track.value
                    if hasattr(t.source_track, "value")
                    else str(t.source_track)
                ),
                "pass_at_k": t.panel.pass_at_k if t.panel else None,
                "discrimination": t.panel.discrimination if t.panel else None,
                "grok_4_5": t.panel.grok_4_5 if t.panel else None,
                "opus_4_8": t.panel.opus_4_8 if t.panel else None,
                "band_rule": (t.gate_proof or {}).get("panel", {}).get("band_rule")
                if isinstance(t.gate_proof, dict)
                else None,
            }
        )

    report_text = _render_ship_report(
        funnel=funnel,
        languages=languages,
        source_tracks=source_tracks,
        hardness=hardness,
        spend_total=format(ledger.summary().total_commit_usd, "f"),
        remaining=format(ledger.remaining_usd(), "f"),
        under_cap=ledger.summary().under_cap,
        keep_count=len(tasks),
        target=target_keeps,
        under_supply=under_supply,
        attempts_summary=[a.to_dict() for a in attempts],
        spot_checks=list(spot_checks),
        seeded_from=seeded_from,
    )
    report_path = out / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary_path = ledger.write_summary_json(out / "ledger_summary.json")

    (out / "ship_manifest.json").write_text(
        json.dumps(
            {
                "interim": interim,
                "keep_count": len(tasks),
                "instance_ids": [t.instance_id for t in tasks],
                "languages": languages,
                "source_tracks": source_tracks,
                "funnel": funnel,
                "under_supply": list(under_supply),
                "spot_checks": list(spot_checks),
                "spend_total_usd": format(ledger.summary().total_commit_usd, "f"),
                "remaining_usd": format(ledger.remaining_usd(), "f"),
                "under_cap": ledger.summary().under_cap,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "attempts.jsonl").write_text(
        "\n".join(json.dumps(a.to_dict(), sort_keys=True) for a in attempts)
        + ("\n" if attempts else ""),
        encoding="utf-8",
    )

    # Restore durable audit/progress if export path overwrote them earlier.
    for name, content in preserved.items():
        target = out / name
        if name in {"report.md", "ledger_summary.json", "ship_manifest.json", "attempts.jsonl"}:
            # freshly rewritten above
            continue
        if not target.is_file() or target.stat().st_size < len(content.encode("utf-8")):
            target.write_text(content, encoding="utf-8")
        (meta / name).write_text(content, encoding="utf-8")

    return {
        "tasks_jsonl": final_jsonl,
        "report_path": report_path,
        "ledger_summary_path": summary_path,
    }


def _try_one_real_pr_keep(
    *,
    work_root: Path,
    ledger: BudgetLedger,
    settings: FactorySettings,
    live_panel: bool,
    panel_k: int,
    use_docker_oracle: bool,
    gate_audit: Path,
) -> tuple[TaskRecord, Path] | None:
    """Best-effort single real_pr candidate through oracle+panel.

    Uses unauthenticated GitHub when no token (rate-limited). Failures return None.
    """
    from swe_factory.export.workspace import write_export_bundle
    from swe_factory.panel.band import hardness_dict_from_decision
    from swe_factory.panel.runner import REQUIRED_PANEL_MODELS, run_panel
    from swe_factory.panel.score_solver import local_pytest_soft_solver
    from swe_factory.producers.pr_miner import PrMineError, PrMiner
    from swe_factory.schema import EnvironmentMeta, PanelHardness
    from swe_factory.sources.clone import ensure_pinned_checkout
    from swe_factory.sources.github import GitHubClient

    work_root.mkdir(parents=True, exist_ok=True)
    try:
        gh = GitHubClient.from_env()
    except Exception:  # noqa: BLE001
        return None
    miner = PrMiner(client=gh, work_root=work_root / "mine")
    # Prefer modular python host for host soft solver compatibility.
    repos = ["mahmoud/boltons", "tkem/cachetools"]
    prs = []
    for repo in repos:
        try:
            prs = miner.list_candidate_prs(repo, max_scan=20, max_keep=3, language="python")
        except Exception:  # noqa: BLE001
            continue
        if prs:
            break
    if not prs:
        return None

    pr = prs[0]
    # Build a SeedRepo-like pin via allowlist or ad-hoc clone
    seed = SeedRepo(
        seed_id=f"realpr_{pr.repo.replace('/', '_')}",
        language="python",
        repo=pr.repo,
        base_commit=pr.base_commit,
        license=pr.license or "MIT",
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
    )
    try:
        checkout = ensure_pinned_checkout(
            seed,
            dest_root=work_root / "checkout",
            prefer_local=False,
        )
    except Exception:  # noqa: BLE001
        return None

    try:
        candidate = miner.produce(
            pr,
            base_workspace=checkout.path,
            run_stub_oracle=True,
        )
    except PrMineError:
        return None

    # real_pr gold applies on base tree; workspace is the base pin without agent break.
    # Oracle validates gold vs null on that workspace (base already has green tests).
    broken = Path(candidate.workspace) if candidate.workspace is not None else checkout.path

    # Oracle
    from swe_factory.oracle.docker_run import (
        FakeOracleRunner,
        OracleDockerRunner,
        ScriptedSuite,
    )
    from swe_factory.oracle.gates import append_gate_audit, run_certified_gates_for_task

    if use_docker_oracle:
        from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers

        runner: Any = OracleDockerRunner(
            docker=DockerCLI(),
            base_image="python:3.12-slim",
            install_commands=["pip install -q pytest", "pip install -e ."],
            command_timeout=180.0,
        )
        try:
            gates = run_certified_gates_for_task(
                candidate.task,
                workspace=broken,
                runner=runner,
                require_multi_file=True,
                dual_runs=2,
                check_null_patch=True,
                check_leak=True,
            )
        finally:
            remove_leftover_sdf_containers()
    else:
        runner = FakeOracleRunner(
            broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
            gold_runs=[
                ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
            ],
            null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        )
        gates = run_certified_gates_for_task(
            candidate.task,
            workspace=broken,
            runner=runner,
            require_multi_file=True,
        )

    append_gate_audit(
        gate_audit,
        gates,
        candidate.task.instance_id,
        extra={"source_track": "real_pr", "pipeline": "ship-v1"},
    )
    if not gates.passed:
        return None

    task = candidate.task.model_copy(
        update={
            "environment": EnvironmentMeta(
                image_digest=f"sha256:ship_real_pr_{pr.base_commit[:12]}"
            ),
            "gate_proof": {
                **(candidate.task.gate_proof or {}),
                **gates.to_gate_proof(),
            },
        }
    )

    if not live_panel:
        # Offline ship path injects band-pass hardness (tests only).
        task = task.model_copy(
            update={
                "panel": PanelHardness(
                    grok_4_5=0.0,
                    opus_4_8=0.5,
                    pass_at_k=0.25,
                    discrimination=2.0,
                )
            }
        )
        return task, Path(broken)

    # Live panel
    from swe_factory.openrouter import OpenRouterClient

    sol = local_pytest_soft_solver(
        broken_workspace=broken,
        fail_to_pass=list(task.fail_to_pass),
        pass_to_pass=list(task.pass_to_pass),
        install_commands=["pip install -q pytest", "pip install -e ."],
    )
    with OpenRouterClient.from_settings(settings) as chat_client:
        panel_result = run_panel(
            task_id=task.instance_id,
            problem_statement=task.problem_statement,
            ledger=ledger,
            client=chat_client,
            models=list(REQUIRED_PANEL_MODELS),
            k=panel_k,
            stage="hardness-panel",
            soft_solver=sol,
            reserve_usd=Decimal("1.50"),
            max_tokens=4096,
            allow_missing_cost_as_zero=False,
            temperature=0.2 if panel_k > 1 else 0.0,
        )
    if not panel_result.is_keep:
        return None
    h = panel_result.panel_hardness()
    task = task.model_copy(
        update={
            "panel": PanelHardness(
                grok_4_5=h.grok_4_5,
                opus_4_8=h.opus_4_8,
                pass_at_k=h.pass_at_k,
                discrimination=h.discrimination,
            ),
            "gate_proof": {
                **(task.gate_proof or {}),
                "panel": hardness_dict_from_decision(panel_result.decision),
            },
        }
    )
    # Materialize export workspace for broken tree retention
    exp = work_root / "export_real_pr"
    write_export_bundle(
        tasks=[task],
        out_dir=exp,
        broken_repos={task.instance_id: broken},
        require_clean_leak_scan=True,
        require_panel=True,
    )
    broken_export = exp / "tasks" / task.instance_id / "repo"
    return task, broken_export if broken_export.is_dir() else Path(broken)


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_KEEPS",
    "DEFAULT_TARGET_KEEPS",
    "AttemptRecord",
    "HarvestPlanItem",
    "ShipV1Result",
    "default_harvest_plan",
    "run_ship_v1",
]
