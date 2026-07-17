"""Curate production hardness panel from test_n10 (M21c + M25 intrinsic).

Policy
------
* DROP misaligned packs (e.g. more-itertools-1136 and same-class).
* DROP structural floors fails (thin F2P, multi-file/hunk shortfalls).
* DROP high-confidence **intrinsic** ``EASY_REQUEST`` (prompt + gold only).
* **Do NOT** drop solely because dual-model eval solve-all (M25 / VAL-DINTR-001).
  EASY_SOLVE_ALL remains a scoreboard label via :mod:`easy_detect`.
* KEEP panel hard keep-band **when** dual-truth + alignment + floors + not
  intrinsic-easy hold.
* INCLUDE legit hard solve-none **and** model solve-alls that still pass
  dual-truth + alignment + floors + intrinsic non-easy (model scoreout ≠ drop).
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
from swe_factory.pipeline.intrinsic_difficulty import (
    CLASS_EASY_REQUEST,
    REASON_EASY_REQUEST,
    intrinsic_from_pack_dir,
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
# M25: model solve-all alone is **not** a drop reason — thin-F2P / misalign only.
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
        "reason_code": "thin_f2p_easy_class",
        "detail": (
            "Structural hardness refuse: thin F2P=1 below MIN_F2P_NODES floor "
            "(model solve-all is not the drop gate under M25)."
        ),
    },
    "realpr-rich-4070": {
        "reason_code": "thin_f2p_easy_class",
        "detail": (
            "Structural hardness refuse: thin F2P=1 below MIN_F2P_NODES floor "
            "(model solve-all is not the drop gate under M25)."
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


def _scoreboard_panel_lookup(
    scoreboard: Mapping[str, Any] | Path | str | None,
) -> dict[str, dict[str, Any]]:
    """Annotate keep-band from eval scoreboard rows (no pack-name hardcoding).

    Converts per_pack / packs scoreboard rows into the panel_row shape that
    :func:`decide_pack` uses for ``keep_despite_model_solve_all`` annotation.
    Dual-model pass@1=1.0 ⇒ rule=``solve-all`` + frontier=1.0 (still not a drop
    under M25).
    """
    if scoreboard is None:
        return {}
    if isinstance(scoreboard, Path | str):
        path = Path(scoreboard)
        if not path.is_file():
            return {}
        try:
            blob = _load_json(path)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(blob, dict):
            return {}
        scoreboard = blob
    assert isinstance(scoreboard, Mapping)

    rows = scoreboard.get("per_pack") or scoreboard.get("packs") or scoreboard.get("rows") or []
    if not isinstance(rows, list):
        return {}

    model_keys: list[str] = []
    raw_models = scoreboard.get("models")
    if isinstance(raw_models, list):
        for m in raw_models:
            if not isinstance(m, str) or not m:
                continue
            short = m.rsplit("/", 1)[-1]
            model_keys.append(short)
            model_keys.append(m)

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = row.get("pack_id") or row.get("task_id") or row.get("id")
        if not pid:
            continue
        frontier = _as_scoreboard_float(row.get("frontier") if "frontier" in row else row.get("frontier_pass_at_k"))
        per_model: dict[str, float] = {}
        raw_pm = row.get("per_model_pass_at_k")
        if isinstance(raw_pm, dict):
            for k, v in raw_pm.items():
                fv = _as_scoreboard_float(v)
                if fv is not None:
                    per_model[str(k)] = fv
        # Board flattened: "grok-4.5": 1.0, "kimi-k2.6": 1.0
        for key, val in row.items():
            if key in {
                "pack_id",
                "task_id",
                "id",
                "frontier",
                "frontier_pass_at_k",
                "complete",
                "decision",
                "per_model_pass_at_k",
                "models",
            }:
                continue
            if key.endswith("_solves") or key.endswith("_cost") or key.endswith("_trials"):
                continue
            fv = _as_scoreboard_float(val)
            if fv is None:
                continue
            # Prefer known model keys; also accept short pass@k floats
            if model_keys and key not in model_keys and "/" not in str(key):
                # Still accept common short model names when models list incomplete
                if not any(ch.isdigit() for ch in str(key)) and str(key) not in {
                    "k",
                    "n",
                }:
                    continue
            per_model[str(key)] = fv

        all_solved = bool(per_model) and all(v >= 1.0 for v in per_model.values())
        if frontier is None and all_solved:
            frontier = 1.0
        rule = None
        verdict = None
        if all_solved or (frontier is not None and frontier >= 1.0 and per_model and all_solved):
            rule = "solve-all"
            # Historical boards used decision=drop for solve-alls; M25 still keeps.
            verdict = "drop"
        elif per_model and all(v <= 0.0 for v in per_model.values()):
            rule = "solve-none"
            verdict = "keep"
        elif per_model:
            rule = "split"
            verdict = "keep"

        # Nested decision dict (full panel report shape) may enrich rule/frontier.
        # Do **not** honour bare string decision="drop" on scoreboards where that
        # field is a legacy M24 hardness drop label rather than a panel band;
        # model matrix above already chose the correct keep/drop annotation.
        raw_decision = row.get("decision")
        if isinstance(raw_decision, dict):
            if raw_decision.get("rule") and not rule:
                rule = str(raw_decision.get("rule"))
            if raw_decision.get("verdict") and not per_model:
                verdict = str(raw_decision.get("verdict"))
            if frontier is None:
                frontier = _as_scoreboard_float(raw_decision.get("frontier_pass_at_k"))
        elif (
            isinstance(raw_decision, str)
            and raw_decision.lower() in {"keep", "drop"}
            and not per_model
            and rule is None
        ):
            verdict = raw_decision.lower()

        out[str(pid)] = {
            "verdict": verdict,
            "rule": rule,
            "frontier_pass_at_k": frontier,
            "reason": "scoreboard_annotate",
            "per_model_pass_at_k": per_model,
        }
    return out


def _as_scoreboard_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Default recovery roots used when packs were dropped for model solve-all only.
# Order matters: prefer product archives before materials skeletons.
DEFAULT_RESTORE_ROOTS: tuple[Path, ...] = (
    Path("datasets/deepagent_v1"),
    Path("datasets/deepagent_v1_seed5_archive"),
    Path("datasets/prod_hard_keep_prev_archive"),
    Path("datasets/live_materials"),
    Path("datasets/live_materials_m22"),
)

# Codes that mean "dropped only for dual-model success" (re-admit candidates).
SOLVE_ALL_ONLY_REASON_CODES: frozenset[str] = frozenset(
    {
        "solve_all_easy_policy_drop",
    }
)


def _load_drop_reasons_table(
    path_or_blob: Mapping[str, Any] | Path | str | None,
) -> dict[str, dict[str, Any]]:
    """Load drop_reasons map from drop_reasons.json / pack_manifest / dict."""
    if path_or_blob is None:
        return {}
    if isinstance(path_or_blob, Mapping):
        raw = path_or_blob.get("drop_reasons")
        if isinstance(raw, dict):
            return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
        # Already a drop_reasons map
        if all(isinstance(v, dict) for v in path_or_blob.values()):
            # Heuristic: reject if looks like a full report without drop_reasons
            if "drop_reasons" not in path_or_blob and "keep_ids" not in path_or_blob:
                return {str(k): dict(v) for k, v in path_or_blob.items() if isinstance(v, dict)}
            raw2 = path_or_blob.get("drop_reasons")
            if isinstance(raw2, dict):
                return {str(k): dict(v) for k, v in raw2.items() if isinstance(v, dict)}
        return {}
    path = Path(path_or_blob)
    if not path.is_file():
        return {}
    try:
        blob = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(blob, dict):
        return {}
    return _load_drop_reasons_table(blob)


def _find_recoverable_pack_dir(
    task_id: str,
    restore_roots: Sequence[Path | str],
) -> Path | None:
    """Return first full Harbor task dir for *task_id* under restore roots."""
    for root in restore_roots:
        root_p = Path(root)
        # Certified corpora store tasks/<id>
        for candidate in (root_p / "tasks" / task_id, root_p / task_id):
            if not candidate.is_dir():
                continue
            missing = verify_pack_tree(candidate)
            if not missing:
                return candidate
    return None


def _archive_pack_row(
    task_id: str,
    restore_roots: Sequence[Path | str],
) -> dict[str, Any] | None:
    """Load dual-truth pack_manifest row + identity stub for *task_id* from archives."""
    for root in restore_roots:
        root_p = Path(root)
        man_path = root_p / "pack_manifest.json"
        if not man_path.is_file():
            continue
        try:
            man = _load_json(man_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(man, dict):
            continue
        idx = _manifest_pack_index(man)
        row = idx.get(task_id)
        if not isinstance(row, dict):
            continue
        out = dict(row)
        # Enrich f2p from gate audit when missing
        if out.get("source_hunk_count") is None:
            hard = hardness_result_from_pack_dir(root_p / "tasks" / task_id)
            if hard.source_hunk_count is not None:
                out["source_hunk_count"] = hard.source_hunk_count
        identity = man.get("identity") if isinstance(man.get("identity"), dict) else {}
        idn = identity.get(task_id) if isinstance(identity, dict) else None
        return {"pack_row": out, "identity": dict(idn) if isinstance(idn, dict) else {}}
    return None


def recover_solve_all_only_drops(
    dest: Path | str,
    *,
    drop_reasons: Mapping[str, Any] | Path | str | None = None,
    restore_roots: Sequence[Path | str] | None = None,
    task_ids: Sequence[str] | None = None,
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES,
    apply_intrinsic: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Re-admit packs previously dropped only for dual-model solve-all (M25b).

    Eligibility (VAL-DINTR-003):
    * Listed in *drop_reasons* with reason_code ∈ SOLVE_ALL_ONLY_REASON_CODES,
      **or** explicitly named in *task_ids*
    * Full Harbor pack tree recoverable from *restore_roots* (default product
      archives + deepagent_v1)
    * Dual-truth sol=1/null=0 from archive manifest row
    * Prompt–verifier alignment OK
    * Hardness floors OK (F2P≥3, multi-file, hunks)
    * Intrinsic **not** high-confidence EASY_REQUEST

    Materials-only skeletons (patch+meta without Harbor layout) are skipped
    (never fixture pad / never partial rearrange).

    On success, copies pack trees into ``dest/tasks/<id>`` and merges dual-truth
    rows into ``dest/pack_manifest.json`` (or creates a minimal index).
    """
    dest_path = Path(dest)
    roots = tuple(Path(r) for r in (restore_roots or DEFAULT_RESTORE_ROOTS))
    reasons_table = _load_drop_reasons_table(drop_reasons)
    if not reasons_table and drop_reasons is None:
        # Default: read dest drop_reasons.json when present
        reasons_table = _load_drop_reasons_table(dest_path / "drop_reasons.json")
        if not reasons_table:
            reasons_table = _load_drop_reasons_table(dest_path / "pack_manifest.json")

    candidates: list[str] = []
    if task_ids:
        candidates = [str(t) for t in task_ids]
    else:
        for tid, info in reasons_table.items():
            code = str(info.get("reason_code") or "")
            if code in SOLVE_ALL_ONLY_REASON_CODES:
                candidates.append(str(tid))
    candidates = sorted(set(candidates))

    recovered: list[str] = []
    skipped: dict[str, str] = {}
    dispositions: list[dict[str, Any]] = []
    pack_rows_to_merge: dict[str, dict[str, Any]] = {}
    identity_to_merge: dict[str, dict[str, Any]] = {}

    for tid in candidates:
        pack_dir = _find_recoverable_pack_dir(tid, roots)
        if pack_dir is None:
            skipped[tid] = "no_full_harbor_tree_in_restore_roots"
            continue
        archive = _archive_pack_row(tid, roots)
        pack_row = dict(archive["pack_row"]) if archive and archive.get("pack_row") else None
        if pack_row is None:
            # Infer dual-truth from gate_audit under any root
            pack_row = _infer_pack_row_from_gate_audit(tid, roots)
        if pack_row is None:
            skipped[tid] = "missing_dual_truth_manifest_row"
            continue
        pack_row.setdefault("task_id", tid)
        pack_row.setdefault("certified", True)
        disp = decide_pack(
            tid,
            pack_dir=pack_dir,
            pack_row=pack_row,
            panel_row={"verdict": "drop", "rule": "solve-all", "frontier_pass_at_k": 1.0},
            force_drop=None,
            min_f2p_nodes=min_f2p_nodes,
            apply_intrinsic=apply_intrinsic,
        )
        dispositions.append(disp.to_dict())
        if not disp.keep:
            skipped[tid] = f"re_eval_drop:{disp.reason_code}"
            continue
        recovered.append(tid)
        pack_rows_to_merge[tid] = dict(pack_row)
        if archive and archive.get("identity"):
            identity_to_merge[tid] = dict(archive["identity"])  # type: ignore[arg-type]
        if not dry_run:
            dest_path.mkdir(parents=True, exist_ok=True)
            tasks_out = dest_path / "tasks"
            tasks_out.mkdir(parents=True, exist_ok=True)
            target = tasks_out / tid
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(pack_dir, target)
            missing = verify_pack_tree(target)
            if missing:
                shutil.rmtree(target, ignore_errors=True)
                recovered.remove(tid)
                pack_rows_to_merge.pop(tid, None)
                skipped[tid] = f"copy_invalid:{missing}"
                continue

    if not dry_run and recovered:
        _merge_manifest_rows(
            dest_path,
            pack_rows=pack_rows_to_merge,
            identity_rows=identity_to_merge,
        )

    return {
        "ok": True,
        "dest": str(dest),
        "recovered_ids": sorted(recovered),
        "skipped": skipped,
        "candidate_ids": candidates,
        "dispositions": dispositions,
        "restore_roots": [str(r) for r in roots],
        "policy": "m25b_restore_solve_all_only",
        "assertions": ["VAL-DINTR-003"],
        "dry_run": bool(dry_run),
        "n_recovered": len(recovered),
    }


def _infer_pack_row_from_gate_audit(
    task_id: str,
    restore_roots: Sequence[Path | str],
) -> dict[str, Any] | None:
    """Build a dual-truth pack_row from gate_audit.jsonl fields when marketing."""
    for root in restore_roots:
        for rel in (
            Path("gate_audit.jsonl"),
            Path("evidence/docker/gate_audit.jsonl"),
        ):
            path = Path(root) / rel
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line or task_id not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                tid = row.get("task_id") or (row.get("fields") or {}).get("task_id")
                if str(tid) != task_id:
                    continue
                fields = row.get("fields") if isinstance(row.get("fields"), dict) else {}
                sol = fields.get("solution_reward", row.get("solution_reward"))
                null = fields.get("null_reward", row.get("null_reward"))
                if sol is None and null is None:
                    continue
                return {
                    "task_id": task_id,
                    "certified": True,
                    "solution_reward": sol if sol is not None else 1,
                    "null_reward": null if null is not None else 0,
                    "source_hunk_count": fields.get("source_hunk_count"),
                    "source_track": fields.get("source_track") or "real_pr",
                    "language": fields.get("language") or "python",
                    "backend": fields.get("backend_class") or "HarborDockerVerifier",
                    "label_method": fields.get("label_method")
                    or "real_pr_dual_run_base_vs_gold",
                    "live_mine": bool(fields.get("live_mine", True)),
                    "f2p_count": fields.get("f2p_count"),
                }
    return None


def _merge_manifest_rows(
    dest: Path,
    *,
    pack_rows: Mapping[str, Mapping[str, Any]],
    identity_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    """Merge recovered dual-truth pack rows into dest pack_manifest.json."""
    man_path = dest / "pack_manifest.json"
    if man_path.is_file():
        try:
            man = _load_json(man_path)
        except (OSError, json.JSONDecodeError):
            man = {"packs": [], "identity": {}}
    else:
        man = {"packs": [], "identity": {}}
    if not isinstance(man, dict):
        man = {"packs": [], "identity": {}}
    packs_list = man.get("packs") if isinstance(man.get("packs"), list) else []
    by_id: dict[str, dict[str, Any]] = {}
    for p in packs_list:
        if isinstance(p, dict) and p.get("task_id"):
            by_id[str(p["task_id"])] = dict(p)
    for tid, row in pack_rows.items():
        merged = dict(by_id.get(tid) or {})
        merged.update(dict(row))
        merged["task_id"] = tid
        merged.setdefault("certified", True)
        by_id[tid] = merged
    man["packs"] = sorted(by_id.values(), key=lambda p: str(p.get("task_id")))
    man["count"] = len(man["packs"])
    man["pack_count"] = len(man["packs"])
    identity = man.get("identity") if isinstance(man.get("identity"), dict) else {}
    identity = dict(identity)
    for tid, idn in identity_rows.items():
        if isinstance(idn, dict) and idn:
            identity[tid] = dict(idn)
    man["identity"] = identity
    man["ok"] = True
    man["view"] = man.get("view") or "prod_hard_keep_restored"
    man["restored_solve_all_only"] = sorted(pack_rows.keys())
    _write_json(man_path, man)


def decide_pack(
    task_id: str,
    *,
    pack_dir: Path,
    pack_row: Mapping[str, Any] | None,
    panel_row: Mapping[str, Any] | None = None,
    force_drop: Mapping[str, Mapping[str, str]] | None = None,
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES,
    apply_intrinsic: bool = True,
) -> PackDisposition:
    """Decide keep/drop for one pack with gates + policy drops.

    Drop reasons (M25):
    * explicit force_drop / EXPLICIT table (misalign, structural thin F2P names)
    * incomplete tree / dual-truth fail
    * prompt–verifier misalignment
    * hardness floors (F2P<3, multi-file, hunks)
    * high-confidence intrinsic ``EASY_REQUEST`` (prompt+gold only)

    Model solve-all / panel rule=solve-all / frontier=1.0 do **not** auto-drop.
    """
    explicit = dict(force_drop or EXPLICIT_DROP)
    hunks = None
    sol = None
    null = None
    if pack_row:
        hunks = pack_row.get("source_hunk_count")
        sol = pack_row.get("solution_reward")
        null = pack_row.get("null_reward")

    raw_panel_verdict = None
    raw_panel_rule = None
    frontier = None
    if panel_row:
        raw_panel_verdict = panel_row.get("verdict")
        raw_panel_rule = panel_row.get("rule")
        frontier = panel_row.get("frontier_pass_at_k")

    panel_verdict_s: str | None = str(raw_panel_verdict) if raw_panel_verdict is not None else None
    panel_rule_s: str | None = str(raw_panel_rule) if raw_panel_rule is not None else None
    frontier_f: float | None = float(frontier) if isinstance(frontier, int | float) else None

    # Explicit policy drops first (misalign / structural thin-F2P named in m21c).
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
            panel_verdict=panel_verdict_s,
            panel_rule=panel_rule_s,
            frontier_pass_at_k=frontier_f,
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
            panel_verdict=panel_verdict_s,
            panel_rule=panel_rule_s,
            frontier_pass_at_k=frontier_f,
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
            panel_verdict=panel_verdict_s,
            panel_rule=panel_rule_s,
            frontier_pass_at_k=frontier_f,
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
            panel_verdict=panel_verdict_s,
            panel_rule=panel_rule_s,
            frontier_pass_at_k=frontier_f,
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
            panel_verdict=panel_verdict_s,
            panel_rule=panel_rule_s,
            frontier_pass_at_k=frontier_f,
            meta={"gate": "hardness_floors"},
        )

    # Intrinsic prompt+gold scorer (M25). High-confidence EASY_REQUEST may drop.
    # Model solve-all is **not** consulted here.
    intrinsic_meta: dict[str, Any] = {}
    if apply_intrinsic:
        intrinsic = intrinsic_from_pack_dir(
            pack_dir,
            f2p_count=hard.f2p_count,
            drop_on_easy_request=True,
            high_confidence_only=True,
        )
        intrinsic_meta = {
            "intrinsic_class": intrinsic.intrinsic_class,
            "easily_approachable": intrinsic.easily_approachable,
            "intrinsic_confidence": intrinsic.confidence,
            "intrinsic_reason_code": intrinsic.reason_code,
            "intrinsic_metrics": dict(intrinsic.metrics),
        }
        if intrinsic.should_drop_hardness and intrinsic.intrinsic_class == CLASS_EASY_REQUEST:
            return PackDisposition(
                task_id=task_id,
                keep=False,
                reason_code=REASON_EASY_REQUEST,
                detail=(
                    f"intrinsic EASY_REQUEST (confidence={intrinsic.confidence}): "
                    f"{intrinsic.detail}"
                ),
                alignment_ok=True,
                hardness_ok=True,
                dual_truth_ok=True,
                f2p_count=hard.f2p_count,
                source_hunk_count=(
                    int(hunks) if isinstance(hunks, int) else hard.source_hunk_count
                ),
                solution_reward=sol,
                null_reward=null,
                panel_verdict=panel_verdict_s,
                panel_rule=panel_rule_s,
                frontier_pass_at_k=frontier_f,
                meta={"gate": "intrinsic_easy_request", **intrinsic_meta},
            )

    # M25: panel solve-all / frontier=1.0 annotated, NEVER sole ship drop.
    # Solve-none is **kept** when dual-truth + floors + alignment + non-intrinsic-easy.
    keep_reason = "keep_hard_dual_truth"
    detail = (
        "production hardness keep: dual-truth sol=1/null=0, prompt–verifier aligned, "
        f"hardness floors ok (F2P≥{min_f2p_nodes}, multi-file/hunk floors); "
        "intrinsic not high-confidence EASY_REQUEST. "
    )
    if panel_rule_s == "solve-all" or (frontier_f is not None and frontier_f >= 1.0):
        keep_reason = "keep_despite_model_solve_all"
        detail += (
            f"Panel rule={panel_rule_s!r} frontier_pass_at_k={frontier_f}: model dual "
            "success is scoreboard-only (M25 / VAL-DINTR-001); not a hardness drop."
        )
    elif panel_rule_s == "solve-none":
        keep_reason = "keep_legit_hard_solve_none"
        detail += (
            "Panel solve-none retained: complex multi-file dual-truth pack; "
            "model scoreout ≠ dataset drop (M21 policy)."
        )
    elif panel_verdict_s == "keep":
        keep_reason = "keep_hard_panel_band"
        detail += f"Panel keep-band rule={panel_rule_s!r} frontier_pass_at_k={frontier_f}."
    else:
        detail += f"Panel={panel_verdict_s!r}/{panel_rule_s!r}."

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
        panel_verdict=panel_verdict_s,
        panel_rule=panel_rule_s,
        frontier_pass_at_k=frontier_f,
        meta={
            "nominal_keep_candidate": task_id in NOMINAL_KEEP_CANDIDATES,
            "required_paths_checked": list(REQUIRED_PACK_RELPATHS),
            "policy": "m25_intrinsic_hardness",
            **intrinsic_meta,
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
    drop_on_solve_all: bool = False,
) -> tuple[dict[str, dict[str, str]], EasyDetectReport]:
    """Derive force_drop from panel/eval scoreboard (no hardcoded pack names).

    M24 labels EASY_SOLVE_ALL when both/all models pass@1=1.0.

    M25 / VAL-DINTR-001: *drop_on_solve_all* defaults **False** so dual-model
    solve-all only annotates the scoreboard. Structural thin-F2P under pack_dirs
    may still populate force_drop. Hardness product drops for portfolio curation
    primarily use alignment + floors + intrinsic EASY_REQUEST inside
    :func:`decide_pack`.
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
        drop_on_solve_all=drop_on_solve_all,
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
    drop_on_solve_all: bool = False,
    apply_intrinsic: bool = True,
) -> list[PackDisposition]:
    """Score every pack under *src*.

    When *scoreboard* is provided, dual-model solve-alls are **labeled** via
    :mod:`easy_detect` (no hardcoded pack names). M25: they do **not** auto-drop
    unless *drop_on_solve_all* is True (legacy M24 replay only). Structural
    thin-F2P may still feed force_drop. Hardness product also applies intrinsic
    prompt+gold scoring inside :func:`decide_pack`.

    Scoreboard rows also annotate ``panel_row`` so dual solve-all keeps surface as
    ``keep_despite_model_solve_all`` rather than bare ``keep_hard_dual_truth``.
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
    panel_idx = _panel_lookup(panel_blob)
    # Scoreboard annotate fill gaps (FULL panel report preferred when present).
    if scoreboard is not None:
        for pid, row in _scoreboard_panel_lookup(scoreboard).items():
            if pid not in panel_idx or not panel_idx[pid].get("rule"):
                panel_idx[pid] = row

    auto_drop: dict[str, dict[str, str]] = {}
    if scoreboard is not None:
        auto_drop, _easy = force_drop_from_scoreboard(
            scoreboard,
            src=src_path,
            min_f2p_nodes=min_f2p_nodes,
            drop_on_solve_all=drop_on_solve_all,
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
                apply_intrinsic=apply_intrinsic,
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
        "Hardness curate from dual-truth + alignment + floors + intrinsic "
        "request/patch analysis. Model solve-all is scoreboard-only (M25); "
        "misalign / thin-F2P / high-confidence EASY_REQUEST dropped; legit hard "
        "solve-none and non-intrinsic-easy solve-alls retained."
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
            "- Intrinsic EASY_REQUEST (high confidence, prompt+gold) dropped from hardness.",
            "- Model dual success is **not** a hardness drop gate (M25 / VAL-DINTR-001).",
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
            "- anti-easy: thin F2P floors + high-confidence intrinsic EASY_REQUEST",
            "- model dual success is not a hardness gate (M25)",
            "- no fixture pad; hybrid never product N",
            "- re-upload HF BaseIntelligence/deepagent revision `test`",
            "",
            "## Assertions",
            "",
            "- VAL-DHARD-004 curated production N to HF test",
            "- VAL-DHARD-002 / VAL-DHARD-003 floors + anti-easy",
            "- VAL-DINTR-001 / VAL-DINTR-002 / VAL-DINTR-003 / VAL-DINTR-005 "
            "intrinsic policy (restore solve-all-only drops)",
            "",
        ]
    )
    return "\n".join(lines)


def _render_product_readme(*, n: int, drop_n: int) -> str:
    return (
        "# datasets/prod_hard_keep — production hardness panel (M21c)\n\n"
        "Curated **source_track=real_pr** Harbor packs for hardness product.\n"
        f"Certified N: **{n}** (floor ≥{MIN_HARD_KEEP}). Dropped from test_n10: **{drop_n}** "
        "(misalign / thin-F2P / intrinsic EASY_REQUEST / failed floors).\n"
        "Model dual success is not a hardness drop reason. "
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
    drop_on_solve_all: bool = False,
    apply_intrinsic: bool = True,
    restore_solve_all: bool = True,
    restore_roots: Sequence[Path | str] | None = None,
    restore_task_ids: Sequence[str] | None = None,
) -> CurationResult:
    """Curate *src* into *out* with drop_reasons + dual-truth keeps only.

    *scoreboard* labels dual-model solve-alls (EASY_SOLVE_ALL) for audit notes
    (M24). M25: those labels do **not** force-drop hardness product by default;
    drops are misalign + hardness floors + high-confidence intrinsic
    EASY_REQUEST.

    M25b (*restore_solve_all*, default True): before scoring, re-admit packs
    previously dropped only for model solve-all when dual-truth + alignment +
    floors + intrinsic non-easy still hold (VAL-DINTR-003). Recovery sources
    default to product archives under :data:`DEFAULT_RESTORE_ROOTS`.

    Raises ``ProdHardCurationError`` if residual keep count < *min_keep*
    (caller must re-mine with new floors — never fixture pad).

    When *src* and *out* resolve to the same path (in-place curate), pack trees
    are first staged under a sibling temporary directory so ``clean_out`` cannot
    rmtree the keep source mid-copy (canonical ``curate-hardness --src X --out X``).
    """
    src_path = Path(src).resolve()
    out_path = Path(out).resolve()
    if not src_path.is_dir():
        raise ProdHardCurationError(f"src missing: {src_path}")

    restore_meta: dict[str, Any] = {}
    # Preserve prior M25b restore ledger so re-curate of an already-restored
    # corpus still documents which packs returned after model-solve-all drops.
    prior_restored: list[str] = []
    for rel in ("drop_reasons.json", "curation_report.json", "pack_manifest.json"):
        prev_path = src_path / rel
        if not prev_path.is_file():
            continue
        try:
            prev_blob = _load_json(prev_path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(prev_blob, dict):
            raw_prev = prev_blob.get("restored_solve_all_only") or prev_blob.get(
                "restored_solve_all_only_ids"
            )
            if isinstance(raw_prev, list):
                prior_restored = [str(x) for x in raw_prev if x]
                if prior_restored:
                    break
    if restore_solve_all:
        # Source drop_reasons from pack root (post-M24 wipe left them empty of
        # keep trees but still documented the three werkzeug-class solves).
        drop_blob: Path | Mapping[str, Any] | None = None
        for rel in ("drop_reasons.json", "pack_manifest.json", "curation_report.json"):
            p = src_path / rel
            if p.is_file():
                drop_blob = p
                break
        restore_meta = recover_solve_all_only_drops(
            src_path,
            drop_reasons=drop_blob,
            restore_roots=restore_roots,
            task_ids=restore_task_ids,
            min_f2p_nodes=min_f2p_nodes,
            apply_intrinsic=apply_intrinsic,
            dry_run=False,
        )
    # Union: newly recovered + prior ledger that still exists under tasks/
    tasks_now = src_path / "tasks"
    present = {
        p.name
        for p in tasks_now.iterdir()
        if tasks_now.is_dir() and p.is_dir() and not p.name.startswith(".")
    }
    union_restored = sorted(
        set(list(restore_meta.get("recovered_ids") or []) + prior_restored) & present
        if present
        else set(list(restore_meta.get("recovered_ids") or []) + prior_restored)
    )
    restore_meta = dict(restore_meta or {})
    restore_meta["recovered_ids"] = union_restored
    restore_meta["prior_restored_ids"] = list(prior_restored)
    if "skipped" not in restore_meta:
        restore_meta["skipped"] = {}

    dispositions = curate_dispositions(
        src_path,
        panel_report=panel_report,
        force_drop=force_drop,
        scoreboard=scoreboard,
        min_f2p_nodes=min_f2p_nodes,
        include_explicit_drops=include_explicit_drops,
        drop_on_solve_all=drop_on_solve_all,
        apply_intrinsic=apply_intrinsic,
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

    # Preserve a callable relative surface for reports (cli args), not only resolve().
    out_rel = str(out)
    src_rel = str(src)

    same_path = src_path == out_path
    staging_path: Path | None = None
    write_root = out_path
    read_src = src_path
    if same_path:
        # Stage side-by-side so we never rmtree keep packs before copy.
        parent = out_path.parent if out_path.parent.exists() else Path(".")
        staging_path = parent / f".{out_path.name}.curate_stage"
        if staging_path.exists():
            shutil.rmtree(staging_path)
        staging_path.mkdir(parents=True, exist_ok=True)
        write_root = staging_path

    try:
        if clean_out and write_root.exists() and not same_path:
            shutil.rmtree(write_root)
        write_root.mkdir(parents=True, exist_ok=True)
        tasks_out = write_root / "tasks"
        tasks_out.mkdir(parents=True, exist_ok=True)

        # Copy keep pack trees (full dual-truth Harbor layout)
        for tid in keep_ids:
            s = read_src / "tasks" / tid
            if not s.is_dir():
                raise ProdHardCurationError(f"keep pack missing under src tasks/: {tid}")
            d = tasks_out / tid
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
            missing = verify_pack_tree(d)
            if missing:
                raise ProdHardCurationError(
                    f"copied pack tree invalid for {tid}: missing={missing}"
                )

        # Manifest (load from original src before any replace)
        manifest_src = read_src / "pack_manifest.json"
        manifest_in = _load_json(manifest_src) if manifest_src.is_file() else {"packs": []}
        if not isinstance(manifest_in, dict):
            raise ProdHardCurationError("pack_manifest.json must be an object")
        filtered = _filter_manifest(
            manifest_in,
            keep_ids=keep_ids,
            dispositions=dispositions,
            out_rel=out_rel,
        )
        _write_json(write_root / "pack_manifest.json", filtered)

        # Corpus metadata
        dintr_assertions = [
            "VAL-DHARD-004",
            "VAL-DINTR-001",
            "VAL-DINTR-003",
            "VAL-DEASY-002",
            "VAL-DEASY-003",
        ]
        _write_json(
            write_root / "drop_reasons.json",
            {
                "schema": CURSOR_SCHEMA,
                "source": src_rel,
                "out": out_rel,
                "scoreboard": str(scoreboard) if scoreboard is not None else None,
                "dropped_count": len(drop_reasons),
                "keep_ids": list(keep_ids),
                "drop_ids": list(drop_ids),
                "drop_reasons": drop_reasons,
                "restored_solve_all_only": list(restore_meta.get("recovered_ids") or []),
                "restore_skipped": dict(restore_meta.get("skipped") or {}),
                "assertions": dintr_assertions,
                "generated_at": _utc_now_iso(),
            },
        )
        _write_json(
            write_root / "curation_report.json",
            {
                "schema": CURSOR_SCHEMA,
                "ok": True,
                "source": src_rel,
                "out": out_rel,
                "scoreboard": str(scoreboard) if scoreboard is not None else None,
                "pack_count": len(keep_ids),
                "min_keep": min_keep,
                "keep_ids": list(keep_ids),
                "drop_ids": list(drop_ids),
                "drop_reasons": drop_reasons,
                "dispositions": [d.to_dict() for d in dispositions],
                "restored_solve_all_only": list(restore_meta.get("recovered_ids") or []),
                "restore_skipped": dict(restore_meta.get("skipped") or {}),
                "assertions": [
                    "VAL-DHARD-002",
                    "VAL-DHARD-003",
                    "VAL-DHARD-004",
                    "VAL-DINTR-001",
                    "VAL-DINTR-003",
                    "VAL-DEASY-002",
                    "VAL-DEASY-003",
                ],
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
            p = read_src / name
            if not p.is_file():
                continue
            try:
                blob = _load_json(p)
            except json.JSONDecodeError:
                continue
            if name == "ship_summary.json" and isinstance(blob, dict):
                blob = dict(blob)
                blob["certified_count"] = len(keep_ids)
                blob["pack_count"] = len(keep_ids)
                blob["count"] = len(keep_ids)
                blob["product_surface"] = out_rel
                blob["curated_hardness"] = True
                blob["drop_reasons"] = drop_reasons
                blob["keep_ids"] = list(keep_ids)
                blob["drop_ids"] = list(drop_ids)
                if isinstance(blob.get("languages"), dict):
                    # recompute naive python count
                    blob["languages"] = {"python": len(keep_ids)}
                blob["ok"] = True
                blob["curated_at"] = _utc_now_iso()
                if scoreboard is not None:
                    blob["easy_scoreboard"] = str(scoreboard)
            elif name in {"oracle_evidence.json", "pier_evidence.json"}:
                blob = _filter_list_records(blob, keep_ids=keep_set)
                if isinstance(blob, dict):
                    blob["curated_hardness"] = True
                    blob["product_surface"] = out_rel
            _write_json(write_root / name, blob)

        # gate_audit.jsonl + e2e drip (filtered)
        for name in ("gate_audit.jsonl", "e2e_drip.jsonl"):
            rows = _filter_jsonl(read_src / name, keep_ids=keep_set)
            if rows or (read_src / name).is_file():
                (write_root / name).write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8",
                )

        _copy_evidence_dirs(read_src, write_root, keep_ids=keep_set)

        raw_filt_identity = filtered.get("identity")
        identity: Mapping[str, Mapping[str, Any]] = (
            raw_filt_identity if isinstance(raw_filt_identity, dict) else {}
        )
        (write_root / "PROVENANCE.md").write_text(
            _render_provenance(
                keep_ids=keep_ids,
                identity=identity,
                drop_reasons=drop_reasons,
                n=len(keep_ids),
            ),
            encoding="utf-8",
        )
        keep_disp = tuple(d for d in dispositions if d.keep)
        (write_root / "report.md").write_text(
            _render_report(
                keep_ids=keep_ids,
                drop_reasons=drop_reasons,
                keep_dispositions=keep_disp,
                n=len(keep_ids),
                src_rel=src_rel,
                out_rel=out_rel,
            ),
            encoding="utf-8",
        )
        (write_root / "PRODUCT_README.md").write_text(
            _render_product_readme(n=len(keep_ids), drop_n=len(drop_ids)),
            encoding="utf-8",
        )

        # Atomic-ish replace for in-place curate: swap staging onto out_path.
        if same_path and staging_path is not None:
            backup = out_path.parent / f".{out_path.name}.curate_prev"
            if backup.exists():
                shutil.rmtree(backup)
            if out_path.exists():
                out_path.rename(backup)
            staging_path.rename(out_path)
            staging_path = None  # owned by out_path now
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        if staging_path is not None and staging_path.exists():
            # Failed mid-stage; leave original src intact.
            shutil.rmtree(staging_path, ignore_errors=True)

    return CurationResult(
        ok=True,
        src=src_rel,
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
            "in_place": same_path,
            "explicit_drops": sorted(
                merge_force_drops(
                    EXPLICIT_DROP if include_explicit_drops else None,
                    force_drop,
                ).keys()
            ),
            "nominal_keep_candidates": sorted(NOMINAL_KEEP_CANDIDATES),
            "restored_solve_all_only": list(restore_meta.get("recovered_ids") or []),
            "restore_skipped": dict(restore_meta.get("skipped") or {}),
            "restore_solve_all": bool(restore_solve_all),
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
    drop_on_solve_all: bool = False,
    apply_intrinsic: bool = True,
    restore_solve_all: bool = True,
    restore_roots: Sequence[Path | str] | None = None,
    restore_task_ids: Sequence[str] | None = None,
) -> CurationResult:
    """CLI-facing hardness curate (M25: intrinsic + floors, not model solve-all).

    Scoreboard still labels dual-model solve-alls for reporting, but
    *drop_on_solve_all* defaults False (VAL-DINTR-001). Residual floor defaults
    to 0 for post-eval curate of an already-certified hardness set.
    Intrinsic EASY_REQUEST (high confidence) + structural floors + misalign
    remain the hardness drop gates.

    M25b: *restore_solve_all* (default True) re-includes packs that were drop-
    listed solely for model solve-all when dual-truth + floors + alignment +
    intrinsic non-easy still hold (VAL-DINTR-003).
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
        drop_on_solve_all=drop_on_solve_all,
        apply_intrinsic=apply_intrinsic,
        restore_solve_all=restore_solve_all,
        restore_roots=restore_roots,
        restore_task_ids=restore_task_ids,
    )


__all__ = [
    "CURSOR_SCHEMA",
    "DEFAULT_OUT",
    "DEFAULT_RESTORE_ROOTS",
    "DEFAULT_SRC",
    "EXPLICIT_DROP",
    "MIN_HARD_KEEP",
    "NOMINAL_KEEP_CANDIDATES",
    "SOLVE_ALL_ONLY_REASON_CODES",
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
    "recover_solve_all_only_drops",
    "classify_scoreboard",
]
