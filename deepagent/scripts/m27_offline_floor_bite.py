#!/usr/bin/env python3
"""M27b offline floor-bite against historical prod_hard_keep (read-only src).

Proves DeepSWE-median floors (files≥4, hunks≥14, added≥400, F2P≥5) + intrinsic
EASY_REQUEST drop thin dual-solve packs (qs-487 class) without deleting
historical pack trees and without any Hugging Face I/O.

Writes:
  datasets/panel_m27_floor_bite/
    snapshot_dispositions.json   — pre-gate inventory (packs still on disk)
    dispositions.json            — M27 gate decide_pack results
    drop_reasons.json            — drops only, with floor reason codes
    curation_report.json         — summary + keep/drop ids
    report.md                    — human-readable bite proof
    SUMMARY.md                   — short gate note

Usage (repo root):
  .venv/bin/python scripts/m27_offline_floor_bite.py
  .venv/bin/python scripts/m27_offline_floor_bite.py \\
      --src datasets/prod_hard_keep \\
      --scoreboard datasets/panel_prod_hard_m26_n5/scoreboard.json \\
      --out datasets/panel_m27_floor_bite
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_factory.pipeline.curate_prod_hard import (
    DEFAULT_MIN_F2P_NODES,
    curate_dispositions,
    list_pack_task_ids,
)
from swe_factory.pipeline.hardness_floors import (
    PRODUCT_MIN_ADDED_LINES,
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
    hardness_result_from_pack_dir,
)
from swe_factory.pipeline.intrinsic_difficulty import intrinsic_from_pack_dir

SCHEMA = "deepagent.m27_floor_bite.v1"
FEATURE_ID = "m27b-offline-curate-bite-thin-prod"
ASSERTION = "VAL-DMED-003"


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _scoreboard_lookup(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = blob.get("per_pack") or blob.get("packs") or []
    out: dict[str, dict[str, Any]] = {}
    models = blob.get("models") or []
    short = [str(m).rsplit("/", 1)[-1] for m in models if isinstance(m, str)]
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = row.get("pack_id") or row.get("task_id")
        if not pid:
            continue
        per: dict[str, float] = {}
        for k, v in row.items():
            if k in {
                "pack_id",
                "task_id",
                "id",
                "complete",
                "decision",
                "frontier",
                "frontier_pass_at_k",
            }:
                continue
            if str(k).endswith(("_solves", "_cost", "_trials")):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            per[str(k)] = fv
        dual = bool(per) and all(v >= 1.0 for v in per.values()) and len(per) >= 2
        # dual when the two primary shorts both exist and ==1
        if not dual and short:
            vals = [per[s] for s in short if s in per]
            dual = len(vals) >= 2 and all(v >= 1.0 for v in vals)
        out[str(pid)] = {
            "decision": row.get("decision"),
            "frontier": row.get("frontier"),
            "per_model_pass_at_k": per,
            "dual_model_solve": dual,
            "models": list(models),
        }
    return out


def snapshot_inventory(
    src: Path,
    *,
    scoreboard: Path | None = None,
) -> dict[str, Any]:
    """Read-only inventory of pack trees still on disk under *src*."""
    task_ids = list_pack_task_ids(src)
    sb = _scoreboard_lookup(scoreboard)
    packs: list[dict[str, Any]] = []
    for tid in task_ids:
        pack_dir = src / "tasks" / tid
        hard = hardness_result_from_pack_dir(pack_dir)
        try:
            intrinsic = intrinsic_from_pack_dir(pack_dir).to_dict()
        except Exception as exc:  # noqa: BLE001 — inventory only
            intrinsic = {"error": str(exc)}
        sb_row = sb.get(tid) or {}
        packs.append(
            {
                "task_id": tid,
                "pack_dir_exists": pack_dir.is_dir(),
                "floors": hard.to_dict(),
                "intrinsic": intrinsic,
                "scoreboard": sb_row,
                "tree_preserved": True,
            }
        )
    return {
        "schema": f"{SCHEMA}.snapshot",
        "feature_id": FEATURE_ID,
        "assertion": ASSERTION,
        "src": str(src),
        "scoreboard": str(scoreboard) if scoreboard else None,
        "snapshotted_at": _utc(),
        "n_packs": len(packs),
        "pack_ids": task_ids,
        "packs": packs,
        "note": (
            "Historical prod_hard_keep pack trees are snapshotted read-only. "
            "This report never deletes or moves them."
        ),
    }


def run_bite(
    *,
    src: Path,
    out: Path,
    scoreboard: Path | None,
    panel_report: Path | None,
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES,
) -> dict[str, Any]:
    """Run M27 gates via curate_dispositions (no materialize / no rmtree)."""
    out.mkdir(parents=True, exist_ok=True)

    snap = snapshot_inventory(src, scoreboard=scoreboard)
    _write_json(out / "snapshot_dispositions.json", snap)

    # Verify trees still present after snapshot (no mutation).
    surviving = list_pack_task_ids(src)
    if set(surviving) != set(snap["pack_ids"]):
        raise RuntimeError(
            f"pack tree mutation detected during offline bite: "
            f"before={snap['pack_ids']} after={surviving}"
        )

    dispositions = curate_dispositions(
        src,
        panel_report=panel_report if panel_report and panel_report.is_file() else None,
        scoreboard=scoreboard if scoreboard and scoreboard.is_file() else None,
        min_f2p_nodes=min_f2p_nodes,
        include_explicit_drops=False,  # pure floor/intrinsic bite; no name table
        drop_on_solve_all=False,  # M25: model solve-all is never sole drop
        apply_intrinsic=True,
    )
    disp_dicts = [d.to_dict() for d in dispositions]
    _write_json(out / "dispositions.json", {"dispositions": disp_dicts})

    # Enrich drop_reasons with full floor reason codes + structural stats.
    drop_reasons: dict[str, dict[str, Any]] = {}
    keep_ids: list[str] = []
    drop_ids: list[str] = []
    sb = _scoreboard_lookup(scoreboard)

    for d in dispositions:
        tid = d.task_id
        pack_dir = src / "tasks" / tid
        hard = hardness_result_from_pack_dir(pack_dir)
        try:
            intrinsic = intrinsic_from_pack_dir(pack_dir)
            intr_d = intrinsic.to_dict()
        except Exception as exc:  # noqa: BLE001
            intrinsic = None
            intr_d = {"error": str(exc)}
        sb_row = sb.get(tid) or {}

        row: dict[str, Any] = {
            "task_id": tid,
            "keep": d.keep,
            "reason_code": d.reason_code,
            "detail": d.detail,
            "alignment_ok": d.alignment_ok,
            "hardness_ok": d.hardness_ok,
            "dual_truth_ok": d.dual_truth_ok,
            "f2p_count": hard.f2p_count,
            "source_file_count": hard.source_file_count,
            "source_hunk_count": hard.source_hunk_count,
            "added_lines": hard.added_lines,
            "floor_reasons": list(hard.reasons),
            "floor_ok": hard.ok,
            "floor_primary": hard.reason_code,
            "intrinsic_class": intr_d.get("intrinsic_class"),
            "intrinsic_should_drop": (
                bool(getattr(intrinsic, "should_drop_hardness", False))
                if intrinsic is not None
                else None
            ),
            "intrinsic_reason_code": intr_d.get("reason_code"),
            "intrinsic_reasons": list(intr_d.get("signals") or intr_d.get("reasons") or []),
            "dual_model_solve": bool(sb_row.get("dual_model_solve")),
            "scoreboard_frontier": sb_row.get("frontier"),
            "scoreboard_per_model": sb_row.get("per_model_pass_at_k") or {},
            "panel_verdict": d.panel_verdict,
            "panel_rule": d.panel_rule,
            "tree_preserved": pack_dir.is_dir(),
            "meta": dict(d.meta),
        }

        if d.keep:
            keep_ids.append(tid)
        else:
            drop_ids.append(tid)
            drop_reasons[tid] = row

    floors = {
        "min_source_files": PRODUCT_MULTI_FILE_FLOOR,
        "min_source_hunks": PRODUCT_SOURCE_HUNK_FLOOR,
        "min_added_lines": PRODUCT_MIN_ADDED_LINES,
        "min_f2p_nodes": min_f2p_nodes,
        "band": "deepswe_median_m27",
    }

    thin_dual_solves = sorted(
        tid
        for tid, info in drop_reasons.items()
        if info.get("dual_model_solve")
        and (not info.get("floor_ok") or info.get("intrinsic_should_drop"))
    )

    qs_class = sorted(
        tid
        for tid in drop_ids
        if "qs-487" in tid
        or (
            drop_reasons.get(tid, {}).get("added_lines") is not None
            and int(drop_reasons[tid].get("added_lines") or 9999) <= 40
            and int(drop_reasons[tid].get("source_file_count") or 99) <= 2
        )
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "feature_id": FEATURE_ID,
        "assertion": ASSERTION,
        "ok": True,
        "src": str(src),
        "out": str(out),
        "scoreboard": str(scoreboard) if scoreboard else None,
        "panel_report": str(panel_report) if panel_report else None,
        "generated_at": _utc(),
        "floors": floors,
        "n_src": len(dispositions),
        "n_keep": len(keep_ids),
        "n_drop": len(drop_ids),
        "keep_ids": keep_ids,
        "drop_ids": drop_ids,
        "drop_reasons": drop_reasons,
        "thin_dual_solve_drops": thin_dual_solves,
        "qs_487_class_drops": qs_class,
        "qs_487_dropped": any("qs-487" in t for t in drop_ids),
        "historical_trees_preserved": all(
            (src / "tasks" / tid).is_dir() for tid in snap["pack_ids"]
        ),
        "historical_pack_ids_unchanged": set(surviving) == set(snap["pack_ids"]),
        "hf_change": False,
        "materialize_ran": False,
        "policy_notes": [
            "M27 floors: files≥4, hunks≥14, added≥400, F2P≥5.",
            "M25: dual-model solve-all is never the sole drop reason.",
            "High-confidence intrinsic EASY_REQUEST may drop (qs-487 class).",
            "Offline bite only: no materialize clean/rmtree of prod_hard_keep.",
            "No HF upload/pull in this feature.",
        ],
        "dispositions": disp_dicts,
    }

    # Fail-closed quality of the *proof*, not of residual N.
    if not report["qs_487_dropped"]:
        report["ok"] = False
        report["fail_reason"] = "expected realpr-qs-487 among drops under M27 floors"
    if not report["historical_trees_preserved"]:
        report["ok"] = False
        report["fail_reason"] = "historical pack tree missing after offline bite"
    if report["n_drop"] < 1:
        report["ok"] = False
        report["fail_reason"] = "expected at least one thin pack drop under M27 floors"

    _write_json(out / "drop_reasons.json", drop_reasons)
    _write_json(out / "curation_report.json", report)

    md = _render_report_md(report)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "SUMMARY.md").write_text(_render_summary_md(report), encoding="utf-8")

    return report


def _render_summary_md(r: dict[str, Any]) -> str:
    lines = [
        "# M27b offline floor bite — SUMMARY",
        "",
        f"- **ok:** `{r.get('ok')}`",
        f"- **assertion:** `{ASSERTION}`",
        f"- **src:** `{r.get('src')}` (read-only; trees preserved)",
        f"- **n_src / keep / drop:** {r.get('n_src')} / {r.get('n_keep')} / {r.get('n_drop')}",
        f"- **qs-487 dropped:** `{r.get('qs_487_dropped')}`",
        f"- **thin dual-solve drops:** `{', '.join(r.get('thin_dual_solve_drops') or []) or '(none)'}`",
        f"- **HF change:** `{r.get('hf_change')}`",
        f"- **materialize ran:** `{r.get('materialize_ran')}`",
        "",
        "## Floors (DeepSWE-median M27)",
        "",
        "```json",
        json.dumps(r.get("floors") or {}, indent=2),
        "```",
        "",
        "## Drop ids",
        "",
    ]
    for tid in r.get("drop_ids") or []:
        dr = (r.get("drop_reasons") or {}).get(tid) or {}
        lines.append(
            f"- `{tid}` — `{dr.get('reason_code')}` "
            f"(floor_primary=`{dr.get('floor_primary')}`, "
            f"added={dr.get('added_lines')}, files={dr.get('source_file_count')}, "
            f"hunks={dr.get('source_hunk_count')}, f2p={dr.get('f2p_count')}, "
            f"dual_solve={dr.get('dual_model_solve')})"
        )
    lines.append("")
    if r.get("keep_ids"):
        lines.append("## Keep ids (pass M27 floors + non-easy intrinsic)")
        lines.append("")
        for tid in r["keep_ids"]:
            lines.append(f"- `{tid}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_report_md(r: dict[str, Any]) -> str:
    lines = [
        "# M27b offline curate bite — thin prod_hard_keep under DeepSWE-median floors",
        "",
        f"**Feature:** `{FEATURE_ID}`  ",
        f"**Assertion:** `{ASSERTION}`  ",
        f"**Generated:** `{r.get('generated_at')}`  ",
        f"**ok:** `{r.get('ok')}`",
        "",
        "## Goal",
        "",
        "Prove new M27 product floors drop soft historical packs before live regen.",
        "Historical `datasets/prod_hard_keep` pack trees stay on disk (audit-only).",
        "No Hugging Face change in this step.",
        "",
        "## Floors applied",
        "",
        "| Floor | Default |",
        "|---|---|",
        f"| source files | ≥ **{r.get('floors', {}).get('min_source_files')}** |",
        f"| source hunks | ≥ **{r.get('floors', {}).get('min_source_hunks')}** |",
        f"| gold added lines | ≥ **{r.get('floors', {}).get('min_added_lines')}** |",
        f"| F2P nodes | ≥ **{r.get('floors', {}).get('min_f2p_nodes')}** |",
        "| dual-truth + alignment | required |",
        "| intrinsic high-conf EASY_REQUEST | drop |",
        "| model dual-solve alone | **never** sole drop (M25) |",
        "",
        f"**Source:** `{r.get('src')}`  ",
        f"**Scoreboard (annotate only):** `{r.get('scoreboard')}`  ",
        f"**n_src / keep / drop:** **{r.get('n_src')}** / **{r.get('n_keep')}** / **{r.get('n_drop')}**",
        "",
        f"- qs-487 dropped: **{r.get('qs_487_dropped')}**",
        f"- qs-487-class drops: `{', '.join(r.get('qs_487_class_drops') or [])}`",
        f"- thin dual-solve drops: `{', '.join(r.get('thin_dual_solve_drops') or [])}`",
        f"- historical trees preserved: **{r.get('historical_trees_preserved')}**",
        f"- HF change: **{r.get('hf_change')}**",
        f"- materialize ran: **{r.get('materialize_ran')}**",
        "",
        "## Drop reasons (each pack below floors / easy intrinsic)",
        "",
    ]
    for tid in r.get("drop_ids") or []:
        dr = (r.get("drop_reasons") or {}).get(tid) or {}
        lines.extend(
            [
                f"### `{tid}`",
                "",
                f"- **reason_code:** `{dr.get('reason_code')}`",
                f"- **floor_primary:** `{dr.get('floor_primary')}`",
                f"- **floor_reasons:** `{', '.join(dr.get('floor_reasons') or [])}`",
                f"- **detail:** {dr.get('detail')}",
                f"- **stats:** files={dr.get('source_file_count')} "
                f"hunks={dr.get('source_hunk_count')} "
                f"added={dr.get('added_lines')} "
                f"f2p={dr.get('f2p_count')}",
                f"- **intrinsic:** class=`{dr.get('intrinsic_class')}` "
                f"should_drop=`{dr.get('intrinsic_should_drop')}` "
                f"code=`{dr.get('intrinsic_reason_code')}`",
                f"- **dual_model_solve (M26 scoreboard):** `{dr.get('dual_model_solve')}` "
                f"frontier=`{dr.get('scoreboard_frontier')}` "
                f"per_model=`{dr.get('scoreboard_per_model')}`",
                f"- **tree_preserved:** `{dr.get('tree_preserved')}`",
                "",
            ]
        )

    if r.get("keep_ids"):
        lines.append("## Keeps (still pass M27 under offline gates)")
        lines.append("")
        lines.append(
            "Note: residual keep under historical N=10 is expected to be thin; "
            "live regen into `prod_hard_deepswe_med` is m27c."
        )
        lines.append("")
        for tid in r["keep_ids"]:
            lines.append(f"- `{tid}`")
        lines.append("")

    lines.extend(
        [
            "## Policy notes",
            "",
        ]
    )
    for n in r.get("policy_notes") or []:
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=Path("datasets/prod_hard_keep"),
        help="Historical product root (read-only; never cleaned)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/panel_m27_floor_bite"),
        help="Report directory (write-only artifact root)",
    )
    p.add_argument(
        "--scoreboard",
        type=Path,
        default=Path("datasets/panel_prod_hard_m26_n5/scoreboard.json"),
        help="M26 (or latest) dual-model scoreboard for dual-solve annotation",
    )
    p.add_argument(
        "--panel-report",
        type=Path,
        default=Path("datasets/panel_prod_hard_m26_n5/report.json"),
        help="Optional full panel report.json",
    )
    p.add_argument("--min-f2p-nodes", type=int, default=DEFAULT_MIN_F2P_NODES)
    p.add_argument("--json", action="store_true", help="Print curation_report JSON to stdout")
    args = p.parse_args(argv)

    if not (args.src / "tasks").is_dir():
        print(f"error: missing tasks under {args.src}", file=sys.stderr)
        return 2

    before_ids = list_pack_task_ids(args.src)
    report = run_bite(
        src=args.src,
        out=args.out,
        scoreboard=args.scoreboard if args.scoreboard.is_file() else None,
        panel_report=args.panel_report if args.panel_report.is_file() else None,
        min_f2p_nodes=args.min_f2p_nodes,
    )
    after_ids = list_pack_task_ids(args.src)
    if before_ids != after_ids:
        print("error: src pack ids changed during offline bite", file=sys.stderr)
        return 3

    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(
            f"m27 floor bite ok={report.get('ok')} "
            f"keep={report.get('n_keep')} drop={report.get('n_drop')} "
            f"qs487={report.get('qs_487_dropped')} "
            f"trees_ok={report.get('historical_trees_preserved')} "
            f"out={args.out}"
        )
        for tid in report.get("drop_ids") or []:
            dr = (report.get("drop_reasons") or {})[tid]
            print(
                f"  DROP {tid}: {dr.get('reason_code')} "
                f"floor={dr.get('floor_primary')} "
                f"added={dr.get('added_lines')} files={dr.get('source_file_count')} "
                f"hunks={dr.get('source_hunk_count')} f2p={dr.get('f2p_count')} "
                f"dual_solve={dr.get('dual_model_solve')}"
            )
        for tid in report.get("keep_ids") or []:
            print(f"  KEEP {tid}")

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
