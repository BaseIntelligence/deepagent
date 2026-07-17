"""Curate production hardness panel from test_n10 (M21c / VAL-DHARD-004).

Policy
------
* DROP misaligned packs (e.g. more-itertools-1136 and same-class).
* DROP solve-all / thin-F2P easy packs (e.g. charset-normalizer-715, rich-4070).
* KEEP panel hard keep-band **when** dual-truth + alignment + floors hold.
* INCLUDE legit hard solve-none (complex multi-file, dual-truth OK) such as
  attrs-1457 and packaging-1120 (model scoreout ≠ auto-drop).
* Never pad with fixtures. Residual after gates must be N≥5, else fail-closed
  so a later live re-mine can refill with new floors.

Writes ``datasets/prod_hard_keep`` (default) with full Harbor pack trees,
dual-truth retained, ``pack_manifest`` / ``report`` / ``PROVENANCE`` updated,
and ``drop_reasons`` documented on the corpus.

No Live Hub I/O here — callers re-upload via ``deepagent upload``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree
from swe_factory.pipeline.easy_detect import (
    EasyDetectReport,
    classify_and_force_drop,
    classify_scoreboard,
)
from swe_factory.pipeline.hardness_floors import (
    DEFAULT_MIN_F2P_NODES,
    hardness_result_from_pack_dir,
)
from swe_factory.pipeline.prompt_alignment import alignment_result_from_pack_dir

CURSOR_SCHEMA = "deepagent.prod_hard_curation.v1"
DEFAULT_SRC = Path("datasets/test_n10")
DEFAULT_OUT = Path("datasets/prod_hard_keep")
MIN_HARD_KEEP = 5

# Feature-level suggestion set (m21c). Residual after gates may be smaller.
NOMINAL_KEEP_CANDIDATES: frozenset[str] = frozenset(
    {
        "realpr-itemadapter-101",
        "realpr-attrs-1323",
        "realpr-httpx-3672",
        "realpr-more-itertools-943",
        "realpr-rich-3486",
        "realpr-attrs-1457",  # legit hard solve-none
        "realpr-packaging-1120",  # legit hard solve-none
    }
)

# Explicit drops (policy). Additional gate failures may drop more candidates.
EXPLICIT_DROP: dict[str, dict[str, str]] = {
    "realpr-more-itertools-1136": {
        "reason_code": "prompt_verifier_misalign",
        "detail": (
            "Prompt–verifier misalignment: instruction claims version/export-only "
            "and do-not-change-runtime while F2P/gold assert runtime behaviour "
            "(windowed/unique_everseen class)."
        ),
    },
    "realpr-charset-normalizer-715": {
        "reason_code": "solve_all_easy_policy_drop",
        "detail": (
            "Solve-all / easy class: frontier panel pass@k=1.0 (both models resolved) "
            "and thin F2P=1 below MIN_F2P_NODES floor."
        ),
    },
    "realpr-rich-4070": {
        "reason_code": "solve_all_easy_policy_drop",
        "detail": (
            "Solve-all / easy class: frontier panel pass@k=1.0 (both models resolved) "
            "and thin F2P=1 below MIN_F2P_NODES floor."
        ),
    },
}

# Anchor reason codes for transparency when gates drop nominal keep-band rows.
GATE_DROP_CODES = {
    "align": "prompt_verifier_misalign",
    "floors": "hardness_floors_refuse",
    "dual_truth": "dual_truth_fail",
    "tree": "pack_tree_incomplete",
}


class ProdHardCurationError(RuntimeError):
    """Fail-closed curation (under-yield, missing src, tree invalid)."""


@dataclass(frozen=True, slots=True)
class PackDisposition:
    task_id: str
    keep: bool
    reason_code: str
    detail: str
    alignment_ok: bool | None = None
    hardness_ok: bool | None = None
    dual_truth_ok: bool | None = None
    f2p_count: int | None = None
    source_hunk_count: int | None = None
    solution_reward: int | float | None = None
    null_reward: int | float | None = None
    panel_verdict: str | None = None
    panel_rule: str | None = None
    frontier_pass_at_k: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "keep": self.keep,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "alignment_ok": self.alignment_ok,
            "hardness_ok": self.hardness_ok,
            "dual_truth_ok": self.dual_truth_ok,
            "f2p_count": self.f2p_count,
            "source_hunk_count": self.source_hunk_count,
            "solution_reward": self.solution_reward,
            "null_reward": self.null_reward,
            "panel_verdict": self.panel_verdict,
            "panel_rule": self.panel_rule,
            "frontier_pass_at_k": self.frontier_pass_at_k,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class CurationResult:
    ok: bool
    src: str
    out: str
    keep_ids: tuple[str, ...]
    drop_ids: tuple[str, ...]
    dispositions: tuple[PackDisposition, ...]
    drop_reasons: dict[str, dict[str, Any]]
    pack_count: int
    min_keep: int = MIN_HARD_KEEP
    reasons: tuple[str, ...] = field(default_factory=tuple)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CURSOR_SCHEMA,
            "ok": self.ok,
            "src": self.src,
            "out": self.out,
            "keep_ids": list(self.keep_ids),
            "drop_ids": list(self.drop_ids),
            "pack_count": self.pack_count,
            "min_keep": self.min_keep,
            "drop_reasons": self.drop_reasons,
            "dispositions": [d.to_dict() for d in self.dispositions],
            "reasons": list(self.reasons),
            "meta": dict(self.meta),
            "assertions": ["VAL-DHARD-004"],
        }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def list_pack_task_ids(src: Path) -> list[str]:
    """List task ids under ``src/tasks`` (sorted)."""
    root = Path(src) / "tasks"
    if not root.is_dir():
        raise ProdHardCurationError(f"no tasks/ under {src}")
    ids = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not ids:
        raise ProdHardCurationError(f"empty tasks/ under {src}")
    return ids


def _manifest_pack_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    packs = manifest.get("packs") or []
    if not isinstance(packs, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for p in packs:
        if isinstance(p, dict) and p.get("task_id"):
            out[str(p["task_id"])] = dict(p)
    return out


def _dual_truth_ok(pack_row: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not pack_row:
        return False, "missing pack_manifest row"
    sol = pack_row.get("solution_reward")
    null = pack_row.get("null_reward")
    try:
        sol_f = float(sol) if sol is not None else None
        null_f = float(null) if null is not None else None
    except (TypeError, ValueError):
        return False, f"non-numeric dual-truth sol={sol!r} null={null!r}"
    if sol_f != 1.0 or null_f != 0.0:
        return False, f"dual-truth fail sol={sol_f} null={null_f} (need sol=1/null=0)"
    if pack_row.get("certified") is False:
        return False, "pack_manifest certified=false"
    return True, "sol=1 null=0"


def _panel_lookup(
    panel_report: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not panel_report:
        return {}
    rows = panel_report.get("pack_results") or []
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = row.get("pack_id") or row.get("task_id")
        if not pid:
            continue
        raw_dec = row.get("decision")
        dec: dict[str, Any] = raw_dec if isinstance(raw_dec, dict) else {}
        out[str(pid)] = {
            "verdict": dec.get("verdict"),
            "rule": dec.get("rule"),
            "frontier_pass_at_k": dec.get("frontier_pass_at_k"),
            "reason": dec.get("reason"),
            "per_model_pass_at_k": dec.get("per_model_pass_at_k"),
        }
    return out


def decide_pack(
    task_id: str,
    *,
    pack_dir: Path,
    pack_row: Mapping[str, Any] | None,
    panel_row: Mapping[str, Any] | None = None,
    force_drop: Mapping[str, Mapping[str, str]] | None = None,
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES,
) -> PackDisposition:
    """Decide keep/drop for one pack with gates + policy drops."""
    explicit = dict(force_drop or EXPLICIT_DROP)
    hunks = None
    sol = None
    null = None
    if pack_row:
        hunks = pack_row.get("source_hunk_count")
        sol = pack_row.get("solution_reward")
        null = pack_row.get("null_reward")

    panel_verdict = None
    panel_rule = None
    frontier = None
    if panel_row:
        panel_verdict = panel_row.get("verdict")
        panel_rule = panel_row.get("rule")
        frontier = panel_row.get("frontier_pass_at_k")

    # Explicit policy drops first (misalign/solve-all named in m21c).
    if task_id in explicit:
        info = explicit[task_id]
        align = alignment_result_from_pack_dir(pack_dir)
        hard = hardness_result_from_pack_dir(
            pack_dir,
            source_hunk_count=int(hunks) if isinstance(hunks, int) else None,
            min_f2p_nodes=min_f2p_nodes,
        )
        dual_ok, dual_detail = _dual_truth_ok(pack_row)
        return PackDisposition(
            task_id=task_id,
            keep=False,
            reason_code=str(info.get("reason_code") or "policy_drop"),
            detail=str(info.get("detail") or "explicit production drop"),
            alignment_ok=align.ok,
            hardness_ok=hard.ok,
            dual_truth_ok=dual_ok,
            f2p_count=hard.f2p_count,
            source_hunk_count=int(hunks) if isinstance(hunks, int) else hard.source_hunk_count,
            solution_reward=sol,
            null_reward=null,
            panel_verdict=str(panel_verdict) if panel_verdict is not None else None,
            panel_rule=str(panel_rule) if panel_rule is not None else None,
            frontier_pass_at_k=float(frontier) if isinstance(frontier, int | float) else None,
            meta={"explicit_policy_drop": True, "dual_detail": dual_detail},
        )

    missing = verify_pack_tree(pack_dir)
    if missing:
        return PackDisposition(
            task_id=task_id,
            keep=False,
            reason_code=GATE_DROP_CODES["tree"],
            detail=f"incomplete Harbor pack tree: missing={missing}",
            dual_truth_ok=False,
            solution_reward=sol,
            null_reward=null,
            panel_verdict=str(panel_verdict) if panel_verdict is not None else None,
            panel_rule=str(panel_rule) if panel_rule is not None else None,
        )

    dual_ok, dual_detail = _dual_truth_ok(pack_row)
    if not dual_ok:
        return PackDisposition(
            task_id=task_id,
            keep=False,
            reason_code=GATE_DROP_CODES["dual_truth"],
            detail=dual_detail,
            dual_truth_ok=False,
            solution_reward=sol,
            null_reward=null,
            panel_verdict=str(panel_verdict) if panel_verdict is not None else None,
            panel_rule=str(panel_rule) if panel_rule is not None else None,
        )

    align = alignment_result_from_pack_dir(pack_dir)
    hard = hardness_result_from_pack_dir(
        pack_dir,
        source_hunk_count=int(hunks) if isinstance(hunks, int) else None,
        min_f2p_nodes=min_f2p_nodes,
    )
    if not align.ok:
        return PackDisposition(
            task_id=task_id,
            keep=False,
            reason_code=align.reason_code or GATE_DROP_CODES["align"],
            detail=align.detail,
            alignment_ok=False,
            hardness_ok=hard.ok,
            dual_truth_ok=True,
            f2p_count=hard.f2p_count,
            source_hunk_count=int(hunks) if isinstance(hunks, int) else hard.source_hunk_count,
            solution_reward=sol,
            null_reward=null,
            panel_verdict=str(panel_verdict) if panel_verdict is not None else None,
            panel_rule=str(panel_rule) if panel_rule is not None else None,
            frontier_pass_at_k=float(frontier) if isinstance(frontier, int | float) else None,
            meta={"gate": "prompt_alignment"},
        )
    if not hard.ok:
        return PackDisposition(
            task_id=task_id,
            keep=False,
            reason_code=hard.reason_code or GATE_DROP_CODES["floors"],
            detail=hard.detail,
            alignment_ok=True,
            hardness_ok=False,
            dual_truth_ok=True,
            f2p_count=hard.f2p_count,
            source_hunk_count=int(hunks) if isinstance(hunks, int) else hard.source_hunk_count,
            solution_reward=sol,
            null_reward=null,
            panel_verdict=str(panel_verdict) if panel_verdict is not None else None,
            panel_rule=str(panel_rule) if panel_rule is not None else None,
            frontier_pass_at_k=float(frontier) if isinstance(frontier, int | float) else None,
            meta={"gate": "hardness_floors"},
        )

    # Optional solve-all panel verdict even if not named explicitly.
    if panel_rule == "solve-all" or (isinstance(frontier, int | float) and float(frontier) >= 1.0):
        return PackDisposition(
            task_id=task_id,
            keep=False,
            reason_code="solve_all_easy_policy_drop",
            detail=(
                f"panel rule={panel_rule!r} frontier_pass_at_k={frontier}; "
                "solve-all dropped from hardness promote (VAL-DHARD-003)."
            ),
            alignment_ok=True,
            hardness_ok=True,
            dual_truth_ok=True,
            f2p_count=hard.f2p_count,
            source_hunk_count=int(hunks) if isinstance(hunks, int) else hard.source_hunk_count,
            solution_reward=sol,
            null_reward=null,
            panel_verdict=str(panel_verdict) if panel_verdict is not None else None,
            panel_rule=str(panel_rule) if panel_rule is not None else None,
            frontier_pass_at_k=float(frontier) if isinstance(frontier, int | float) else None,
            meta={"gate": "anti_easy_panel"},
        )

    # Solve-none is **kept** when dual-truth + floors + alignment hold (m21c).
    keep_reason = "keep_hard_dual_truth"
    detail = (
        "production hardness keep: dual-truth sol=1/null=0, prompt–verifier aligned, "
        f"hardness floors ok (F2P≥{min_f2p_nodes}, multi-file/hunk floors). "
    )
    if panel_rule == "solve-none":
        keep_reason = "keep_legit_hard_solve_none"
        detail += (
            "Panel solve-none retained: complex multi-file dual-truth pack; "
            "model scoreout ≠ dataset drop (M21 policy)."
        )
    elif panel_verdict == "keep":
        keep_reason = "keep_hard_panel_band"
        detail += f"Panel keep-band rule={panel_rule!r} frontier_pass_at_k={frontier}."
    else:
        detail += f"Panel={panel_verdict!r}/{panel_rule!r}."

    return PackDisposition(
        task_id=task_id,
        keep=True,
        reason_code=keep_reason,
        detail=detail,
        alignment_ok=True,
        hardness_ok=True,
        dual_truth_ok=True,
        f2p_count=hard.f2p_count,
        source_hunk_count=int(hunks) if isinstance(hunks, int) else hard.source_hunk_count,
        solution_reward=sol,
        null_reward=null,
        panel_verdict=str(panel_verdict) if panel_verdict is not None else None,
        panel_rule=str(panel_rule) if panel_rule is not None else None,
        frontier_pass_at_k=float(frontier) if isinstance(frontier, int | float) else None,
        meta={
            "nominal_keep_candidate": task_id in NOMINAL_KEEP_CANDIDATES,
            "required_paths_checked": list(REQUIRED_PACK_RELPATHS),
        },
    )


def merge_force_drops(
    *tables: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    """Merge force-drop tables (later entries override earlier on same id)."""
    merged: dict[str, dict[str, str]] = {}
    for table in tables:
        if not table:
            continue
        for tid, info in table.items():
            merged[str(tid)] = {
                "reason_code": str(info.get("reason_code") or "policy_drop"),
                "detail": str(info.get("detail") or "policy drop"),
            }
    return merged


def force_drop_from_scoreboard(
    scoreboard: Mapping[str, Any] | Path | str,
    *,
    src: Path | str | None = None,
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES,
) -> tuple[dict[str, dict[str, str]], EasyDetectReport]:
    """Derive force_drop from panel/eval scoreboard (no hardcoded pack names).

    M24 / VAL-DEASY-002/005: EASY_SOLVE_ALL when both/all models pass@1=1.0.
    """
    pack_dirs: dict[str, Path] | None = None
    if src is not None:
        src_path = Path(src)
        tasks = src_path / "tasks"
        if tasks.is_dir():
            pack_dirs = {
                p.name: p for p in tasks.iterdir() if p.is_dir() and not p.name.startswith(".")
            }
    report, drops = classify_and_force_drop(
        scoreboard,
        pack_dirs=pack_dirs,
        min_f2p_nodes=min_f2p_nodes,
    )
    return drops, report


def curate_dispositions(
    src: Path | str,
    *,
    panel_report: Mapping[str, Any] | Path | None = None,
    force_drop: Mapping[str, Mapping[str, str]] | None = None,
    scoreboard: Mapping[str, Any] | Path | str | None = None,
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES,
    include_explicit_drops: bool = True,
) -> list[PackDisposition]:
    """Score every pack under *src*.

    When *scoreboard* is provided, dual-model solve-alls auto-populate force_drop
    via :mod:`easy_detect` (no hardcoded pack names; M24).
    """
    src_path = Path(src)
    task_ids = list_pack_task_ids(src_path)
    manifest_path = src_path / "pack_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    pack_idx = _manifest_pack_index(manifest if isinstance(manifest, dict) else {})

    panel_blob: Mapping[str, Any] | None
    if isinstance(panel_report, Path):
        panel_blob = _load_json(panel_report) if panel_report.is_file() else None
    else:
        panel_blob = panel_report
    # Scoreboard path may stand in for panel report when report.json not given.
    if panel_blob is None and scoreboard is not None and panel_report is None:
        # Scoreboard-only path: easy_detect still builds force_drop; panel
        # verdict lookup remains empty unless report is supplied.
        pass
    panel_idx = _panel_lookup(panel_blob)

    auto_drop: dict[str, dict[str, str]] = {}
    if scoreboard is not None:
        auto_drop, _easy = force_drop_from_scoreboard(
            scoreboard,
            src=src_path,
            min_f2p_nodes=min_f2p_nodes,
        )
    base_explicit = EXPLICIT_DROP if include_explicit_drops else {}
    merged_drop = merge_force_drops(base_explicit, auto_drop, force_drop)

    out: list[PackDisposition] = []
    for tid in task_ids:
        out.append(
            decide_pack(
                tid,
                pack_dir=src_path / "tasks" / tid,
                pack_row=pack_idx.get(tid),
                panel_row=panel_idx.get(tid),
                force_drop=merged_drop,
                min_f2p_nodes=min_f2p_nodes,
            )
        )
    return out


def _filter_manifest(
    manifest: dict[str, Any],
    *,
    keep_ids: Sequence[str],
    dispositions: Sequence[PackDisposition],
    out_rel: str,
) -> dict[str, Any]:
    keep_set = set(keep_ids)
    drop_reasons = {
        d.task_id: {
            "reason_code": d.reason_code,
            "detail": d.detail,
            "panel_verdict": d.panel_verdict,
            "panel_rule": d.panel_rule,
            "frontier_pass_at_k": d.frontier_pass_at_k,
            "f2p_count": d.f2p_count,
            "alignment_ok": d.alignment_ok,
            "hardness_ok": d.hardness_ok,
        }
        for d in dispositions
        if not d.keep
    }
    keep_meta = {d.task_id: d.to_dict() for d in dispositions if d.keep}

    new = dict(manifest)
    packs = [
        p
        for p in (manifest.get("packs") or [])
        if isinstance(p, dict) and p.get("task_id") in keep_set
    ]
    packs_sorted = sorted(packs, key=lambda p: str(p.get("task_id")))
    new["packs"] = packs_sorted
    n = len(packs_sorted)
    new["count"] = n
    new["pack_count"] = n
    new["ok"] = n >= MIN_HARD_KEEP
    new["product_dest"] = True
    new["live_generate_dest"] = False
    new["curated_hardness"] = True
    new["curation_schema"] = CURSOR_SCHEMA
    new["curation_source"] = str(manifest.get("product_surface") or "datasets/test_n10")
    new["product_surface"] = out_rel
    new["band"] = {"min": MIN_HARD_KEEP, "target": n, "max": n}
    # Identity map
    raw_identity = manifest.get("identity")
    identity: dict[str, Any] = raw_identity if isinstance(raw_identity, dict) else {}
    new["identity"] = {k: v for k, v in identity.items() if k in keep_set}
    new["drop_reasons"] = drop_reasons
    new["keep_dispositions"] = keep_meta
    new["dropped_count"] = len(drop_reasons)
    new["curated_at"] = _utc_now_iso()
    new["assertions"] = list(
        dict.fromkeys(
            list(manifest.get("assertions") or [])
            + ["VAL-DHARD-004", "VAL-DHARD-002", "VAL-DHARD-003"]
        )
    )
    # Harbor load list
    hls = (
        dict(manifest.get("harbor_load_smoke") or {})
        if isinstance(manifest.get("harbor_load_smoke"), dict)
        else {}
    )
    if hls:
        hls = dict(hls)
        hls["listed"] = sorted(keep_set)
        if packs_sorted:
            hls["task_id"] = packs_sorted[0].get("task_id")
            hls["task_short_name"] = packs_sorted[0].get("task_id")
        new["harbor_load_smoke"] = hls
    # Modes
    new["mode"] = "curate_prod_hard_keep"
    new["panel_note"] = (
        "Hardness curate from live panel + dual-truth. Solve-all/misalign dropped; "
        "legit hard solve-none retained when dual-truth/align/floors ok."
    )
    return new


def _filter_list_records(
    payload: Any,
    *,
    keep_ids: set[str],
    id_keys: Sequence[str] = ("task_id", "pack_id", "instance_id"),
) -> Any:
    if isinstance(payload, list):
        kept: list[Any] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            tid = None
            for k in id_keys:
                if row.get(k):
                    tid = str(row[k])
                    break
            if tid in keep_ids:
                kept.append(row)
        return kept
    if isinstance(payload, dict):
        out = dict(payload)
        # common shapes
        for key in ("records", "packs", "rows", "entries", "items"):
            if key in out and isinstance(out[key], list):
                out[key] = _filter_list_records(out[key], keep_ids=keep_ids, id_keys=id_keys)
        if "certified_count" in out:
            recs = out.get("records")
            if isinstance(recs, list):
                out["certified_count"] = len(recs)
        return out
    return payload


def _filter_jsonl(path: Path, *, keep_ids: set[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    kept: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        tid = row.get("task_id") or row.get("pack_id") or row.get("id")
        raw_fields = row.get("fields")
        fields: dict[str, Any] = raw_fields if isinstance(raw_fields, dict) else {}
        tid = tid or fields.get("task_id") or fields.get("pack_id")
        if tid is None:
            # keep non-pack summary rows (corpus-level)
            if row.get("accepted") is not None and not any(
                k in row for k in ("task_id", "pack_id")
            ):
                # filter task-scoped only; drop if fields mention dropped packs
                f_tid = fields.get("task_id") if isinstance(fields, dict) else None
                if f_tid and str(f_tid) not in keep_ids:
                    continue
                if f_tid and str(f_tid) in keep_ids:
                    kept.append(row)
                    continue
                # corpus-level audit line without pack id — keep
                kept.append(row)
            continue
        if str(tid) in keep_ids:
            kept.append(row)
    return kept


def _copy_evidence_dirs(
    src: Path,
    out: Path,
    *,
    keep_ids: set[str],
) -> None:
    for sub in ("evidence/docker", "evidence/pier"):
        s = src / sub
        if not s.is_dir():
            continue
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in s.iterdir():
            if not f.is_file():
                continue
            name = f.name
            # keep if any keep id is a filename prefix/token
            if any(kid in name for kid in keep_ids) or name in {
                "gate_audit.jsonl",
            }:
                if name == "gate_audit.jsonl":
                    rows = _filter_jsonl(f, keep_ids=keep_ids)
                    (d / name).write_text(
                        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8",
                    )
                else:
                    shutil.copy2(f, d / name)


def _render_provenance(
    *,
    keep_ids: Sequence[str],
    identity: Mapping[str, Mapping[str, Any]],
    drop_reasons: Mapping[str, Mapping[str, Any]],
    n: int,
) -> str:
    lines = [
        "# PROVENANCE — datasets/prod_hard_keep (M21c production hardness)",
        "",
        "Curated **real_pr** Harbor hardness packs only. Droplist documents ",
        "misalign + solve-all/easy removals from `datasets/test_n10`. Dual-truth ",
        "retained for every keep (sol=1 / null=0). No fixture pad.",
        "",
        f"**Product hardness certified N: {n}** (target ≥{MIN_HARD_KEEP})",
        "",
        "| pack_id | language | license | upstream_url | base_sha | source_track | pr |",
        "|---|---|---|---|---|---|---:|",
    ]
    for tid in keep_ids:
        idn = identity.get(tid) or {}
        lang = idn.get("language") or "python"
        lic = idn.get("license") or "?"
        url = idn.get("repository_url") or idn.get("upstream_label") or ""
        sha = idn.get("base_commit") or ""
        track = idn.get("source_track") or "real_pr"
        seed = idn.get("seed_id") or ""
        lines.append(f"| `{tid}` | {lang} | {lic} | {url} | `{sha}` | {track} | {seed} |")
    lines.extend(
        [
            "",
            "## Drop reasons (not product hardness N)",
            "",
            "| pack_id | reason_code | detail |",
            "|---|---|---|",
        ]
    )
    for tid, info in sorted(drop_reasons.items()):
        code = info.get("reason_code") or "?"
        detail = str(info.get("detail") or "").replace("|", "/").replace("\n", " ")
        lines.append(f"| `{tid}` | `{code}` | {detail} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Source wave: `datasets/test_n10` live-mine dual-truth packs.",
            "- Gates: prompt–verifier alignment, MIN_F2P≥3, ≥10 hunks, multi-file, dual-truth.",
            "- Solve-all class + misalign class dropped from hardness promote.",
            (
                "- Legit hard solve-none kept when dual-truth+floors+align hold "
                "(model scoreout ≠ drop)."
            ),
            "- Agent trees: public git clone@SHA; Docker oracle never `oracle_mode=fake`.",
            "- Fixtures / hybrid archives are never product N.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_report(
    *,
    keep_ids: Sequence[str],
    drop_reasons: Mapping[str, Mapping[str, Any]],
    keep_dispositions: Sequence[PackDisposition],
    n: int,
    src_rel: str,
    out_rel: str,
) -> str:
    lines = [
        "# DeepAgent production hardness ship report (M21c)",
        "",
        f"- Generated (UTC): `{_utc_now_iso()}`",
        f"- Source corpus: `{src_rel}`",
        f"- Curated product path: `{out_rel}`",
        "- Source track (product): **`real_pr` only**",
        f"- Certified hardness packs N: **{n}** (floor ≥{MIN_HARD_KEEP})",
        "- Docker oracle: dual-truth sol=1 / null=0 retained on every keep",
        "- Prompt–verifier alignment: required on every keep",
        f"- Hardness floors: F2P≥{DEFAULT_MIN_F2P_NODES}, multi-file, source hunks≥10",
        "- Status: `OK` — curated production hardness panel (no fixture pad)",
        "",
        "## Drop reasons",
        "",
        "| pack_id | reason_code | panel | detail |",
        "|---|---|---|---|",
    ]
    for tid, info in sorted(drop_reasons.items()):
        code = info.get("reason_code") or "?"
        panel = info.get("panel_rule") or info.get("panel_verdict") or "-"
        detail = str(info.get("detail") or "").replace("|", "/").replace("\n", " ")
        lines.append(f"| `{tid}` | `{code}` | {panel} | {detail} |")
    lines.extend(
        [
            "",
            "## Certified hardness keeps (real_pr)",
            "",
        ]
    )
    for d in keep_dispositions:
        lines.append(
            f"- `{d.task_id}` reason=`{d.reason_code}` f2p={d.f2p_count} "
            f"hunks={d.source_hunk_count} sol={d.solution_reward} null={d.null_reward} "
            f"panel={d.panel_rule}/{d.panel_verdict} frontier={d.frontier_pass_at_k}"
        )
    lines.extend(
        [
            "",
            "## Gates (no relaxation)",
            "",
            "- dual-truth HarborDocker sol=1 / null=0",
            "- prompt–verifier alignment fail-closed",
            f"- hardness floors F2P≥{DEFAULT_MIN_F2P_NODES}, multi-file≥2, hunks≥10",
            "- anti-easy: solve-all + thin F2P dropped",
            "- no fixture pad; hybrid never product N",
            "- re-upload HF BaseIntelligence/deepagent revision `test`",
            "",
            "## Assertions",
            "",
            "- VAL-DHARD-004 curated production N to HF test",
            "- VAL-DHARD-002 / VAL-DHARD-003 floors + anti-easy",
            "",
        ]
    )
    return "\n".join(lines)


def _render_product_readme(*, n: int, drop_n: int) -> str:
    return (
        "# datasets/prod_hard_keep — production hardness panel (M21c)\n\n"
        "Curated **source_track=real_pr** Harbor packs for hardness product.\n"
        f"Certified N: **{n}** (floor ≥{MIN_HARD_KEEP}). Dropped from test_n10: **{drop_n}** "
        "(misalign / solve-all / thin-F2P / failed gates).\n"
        "Dual-truth sol=1/null=0 retained. No fixture pad. "
        "See `drop_reasons` in `pack_manifest.json` and `PROVENANCE.md`.\n"
    )


def materialize_prod_hard_keep(
    src: Path | str = DEFAULT_SRC,
    out: Path | str = DEFAULT_OUT,
    *,
    panel_report: Mapping[str, Any] | Path | None = None,
    force_drop: Mapping[str, Mapping[str, str]] | None = None,
    scoreboard: Mapping[str, Any] | Path | str | None = None,
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES,
    min_keep: int = MIN_HARD_KEEP,
    clean_out: bool = True,
    include_explicit_drops: bool = True,
) -> CurationResult:
    """Curate *src* into *out* with drop_reasons + dual-truth keeps only.

    *scoreboard* (M24): panel/eval ``scoreboard.json`` or ``report.json`` auto-drops
    dual-model solve-alls (EASY_SOLVE_ALL) without hardcoding pack names.

    Raises ``ProdHardCurationError`` if residual keep count < *min_keep*
    (caller must re-mine with new floors — never fixture pad).
    """
    src_path = Path(src).resolve()
    out_path = Path(out)
    if not src_path.is_dir():
        raise ProdHardCurationError(f"src missing: {src_path}")

    dispositions = curate_dispositions(
        src_path,
        panel_report=panel_report,
        force_drop=force_drop,
        scoreboard=scoreboard,
        min_f2p_nodes=min_f2p_nodes,
        include_explicit_drops=include_explicit_drops,
    )
    keep_ids = tuple(sorted(d.task_id for d in dispositions if d.keep))
    drop_ids = tuple(sorted(d.task_id for d in dispositions if not d.keep))
    drop_reasons = {
        d.task_id: {
            "reason_code": d.reason_code,
            "detail": d.detail,
            "panel_verdict": d.panel_verdict,
            "panel_rule": d.panel_rule,
            "frontier_pass_at_k": d.frontier_pass_at_k,
            "f2p_count": d.f2p_count,
            "alignment_ok": d.alignment_ok,
            "hardness_ok": d.hardness_ok,
            "dual_truth_ok": d.dual_truth_ok,
        }
        for d in dispositions
        if not d.keep
    }

    if len(keep_ids) < min_keep:
        raise ProdHardCurationError(
            f"prod_hard_keep residual N={len(keep_ids)} < min_keep={min_keep}; "
            f"keep_ids={list(keep_ids)}; drop_ids={list(drop_ids)}. "
            "Re-mine with new floors until N≥5 (never fixture pad). VAL-DHARD-004."
        )

    if clean_out and out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    tasks_out = out_path / "tasks"
    tasks_out.mkdir(parents=True, exist_ok=True)

    # Copy keep pack trees (full dual-truth Harbor layout)
    for tid in keep_ids:
        s = src_path / "tasks" / tid
        d = tasks_out / tid
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)
        missing = verify_pack_tree(d)
        if missing:
            raise ProdHardCurationError(f"copied pack tree invalid for {tid}: missing={missing}")

    # Manifest
    manifest_src = src_path / "pack_manifest.json"
    manifest_in = _load_json(manifest_src) if manifest_src.is_file() else {"packs": []}
    if not isinstance(manifest_in, dict):
        raise ProdHardCurationError("pack_manifest.json must be an object")
    out_rel = str(out)
    filtered = _filter_manifest(
        manifest_in,
        keep_ids=keep_ids,
        dispositions=dispositions,
        out_rel=out_rel,
    )
    _write_json(out_path / "pack_manifest.json", filtered)

    # Corpus metadata
    _write_json(
        out_path / "drop_reasons.json",
        {
            "schema": CURSOR_SCHEMA,
            "source": str(src),
            "out": out_rel,
            "dropped_count": len(drop_reasons),
            "keep_ids": list(keep_ids),
            "drop_reasons": drop_reasons,
            "assertions": ["VAL-DHARD-004"],
            "generated_at": _utc_now_iso(),
        },
    )
    _write_json(
        out_path / "curation_report.json",
        {
            "schema": CURSOR_SCHEMA,
            "ok": True,
            "source": str(src),
            "out": out_rel,
            "pack_count": len(keep_ids),
            "min_keep": min_keep,
            "keep_ids": list(keep_ids),
            "drop_ids": list(drop_ids),
            "drop_reasons": drop_reasons,
            "dispositions": [d.to_dict() for d in dispositions],
            "assertions": ["VAL-DHARD-002", "VAL-DHARD-003", "VAL-DHARD-004"],
            "generated_at": _utc_now_iso(),
        },
    )

    keep_set = set(keep_ids)
    # Oracle / pier evidence aggregates
    for name in (
        "oracle_evidence.json",
        "pier_evidence.json",
        "ship_summary.json",
        "gate_audit_summary.json",
        "ledger_summary.json",
    ):
        p = src_path / name
        if not p.is_file():
            continue
        try:
            blob = _load_json(p)
        except json.JSONDecodeError:
            continue
        if name == "ship_summary.json" and isinstance(blob, dict):
            blob = dict(blob)
            blob["certified_count"] = len(keep_ids)
            blob["product_surface"] = out_rel
            blob["curated_hardness"] = True
            blob["drop_reasons"] = drop_reasons
            blob["keep_ids"] = list(keep_ids)
            if isinstance(blob.get("languages"), dict):
                # recompute naive python count
                blob["languages"] = {"python": len(keep_ids)}
            blob["ok"] = True
            blob["curated_at"] = _utc_now_iso()
        elif name in {"oracle_evidence.json", "pier_evidence.json"}:
            blob = _filter_list_records(blob, keep_ids=keep_set)
            if isinstance(blob, dict):
                blob["curated_hardness"] = True
                blob["product_surface"] = out_rel
        _write_json(out_path / name, blob)

    # gate_audit.jsonl + e2e drip (filtered)
    for name in ("gate_audit.jsonl", "e2e_drip.jsonl"):
        rows = _filter_jsonl(src_path / name, keep_ids=keep_set)
        if rows or (src_path / name).is_file():
            (out_path / name).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )

    _copy_evidence_dirs(src_path, out_path, keep_ids=keep_set)

    raw_filt_identity = filtered.get("identity")
    identity: Mapping[str, Mapping[str, Any]] = (
        raw_filt_identity if isinstance(raw_filt_identity, dict) else {}
    )
    (out_path / "PROVENANCE.md").write_text(
        _render_provenance(
            keep_ids=keep_ids,
            identity=identity,
            drop_reasons=drop_reasons,
            n=len(keep_ids),
        ),
        encoding="utf-8",
    )
    keep_disp = tuple(d for d in dispositions if d.keep)
    (out_path / "report.md").write_text(
        _render_report(
            keep_ids=keep_ids,
            drop_reasons=drop_reasons,
            keep_dispositions=keep_disp,
            n=len(keep_ids),
            src_rel=str(src),
            out_rel=out_rel,
        ),
        encoding="utf-8",
    )
    (out_path / "PRODUCT_README.md").write_text(
        _render_product_readme(n=len(keep_ids), drop_n=len(drop_ids)),
        encoding="utf-8",
    )

    return CurationResult(
        ok=True,
        src=str(src),
        out=out_rel,
        keep_ids=keep_ids,
        drop_ids=drop_ids,
        dispositions=tuple(dispositions),
        drop_reasons=drop_reasons,
        pack_count=len(keep_ids),
        min_keep=min_keep,
        reasons=("curated_ok",),
        meta={
            "min_f2p_nodes": min_f2p_nodes,
            "scoreboard": str(scoreboard) if scoreboard is not None else None,
            "explicit_drops": sorted(
                merge_force_drops(
                    EXPLICIT_DROP if include_explicit_drops else None,
                    force_drop,
                ).keys()
            ),
            "nominal_keep_candidates": sorted(NOMINAL_KEEP_CANDIDATES),
        },
    )


def curate_hardness_from_scoreboard(
    src: Path | str,
    out: Path | str,
    *,
    scoreboard: Mapping[str, Any] | Path | str,
    panel_report: Mapping[str, Any] | Path | None = None,
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES,
    min_keep: int = 0,
    clean_out: bool = True,
    include_explicit_drops: bool = False,
) -> CurationResult:
    """CLI-facing hardness curate driven by scoreboard auto-easy detection.

    Default *include_explicit_drops=False* so drops are scoreboard-driven only
    (werkzeug-class solve-alls without name hardcoding). Residual floor defaults
    to 0 for post-eval demote of an already-certified hardness set (M24 waves may
    land below 5 after drops; report honesty preserves remaining keeps).
    """
    return materialize_prod_hard_keep(
        src,
        out,
        panel_report=panel_report,
        scoreboard=scoreboard,
        min_f2p_nodes=min_f2p_nodes,
        min_keep=min_keep,
        clean_out=clean_out,
        include_explicit_drops=include_explicit_drops,
    )


__all__ = [
    "CURSOR_SCHEMA",
    "DEFAULT_OUT",
    "DEFAULT_SRC",
    "EXPLICIT_DROP",
    "MIN_HARD_KEEP",
    "NOMINAL_KEEP_CANDIDATES",
    "CurationResult",
    "PackDisposition",
    "ProdHardCurationError",
    "curate_dispositions",
    "curate_hardness_from_scoreboard",
    "decide_pack",
    "force_drop_from_scoreboard",
    "list_pack_task_ids",
    "materialize_prod_hard_keep",
    "merge_force_drops",
    "classify_scoreboard",
]
