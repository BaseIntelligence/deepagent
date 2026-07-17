"""Panel/eval scoreboard easy **labels** (M24) + M25 intrinsic-policy flip.

Classify packs as **EASY_SOLVE_ALL** when **every** model in the open panel matrix
has ``pass_at_1 == 1.0`` (frontier pass@k ≥ 1.0 / solve-all). This is a
**reporting / scoreboard label only** under M25: ``should_drop_hardness`` is
``False`` by default for model solve-all (VAL-DINTR-001). Hardness drops come
from misalign, structural floors (F2P<3, …), and high-confidence intrinsic
``EASY_REQUEST`` (see :mod:`intrinsic_difficulty`) — never solely model outcomes.

Thin F2P still reuses :mod:`hardness_floors` reason codes and **does** set
``should_drop_hardness=True`` (structural floor, not model score).

Drive scoreboard labeling without hardcoding pack names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.pipeline.hardness_floors import (
    REASON_F2P_BELOW_FLOOR,
    REASON_HARDNESS_OK,
    REASON_SOLVE_ALL_EASY,
    REASON_THIN_F2P_EASY,
    HardnessFloorResult,
    check_product_hardness_floors,
    hardness_result_from_pack_dir,
)

#: Product / panel label for solve-all easy class (alias of hardness reason).
EASY_SOLVE_ALL: str = "EASY_SOLVE_ALL"
REASON_EASY_SOLVE_ALL: str = REASON_SOLVE_ALL_EASY

#: Stable drop / keep codes for hardness promote.
REASON_OK_HARDNESS: str = REASON_HARDNESS_OK
REASON_ONE_SIDED_DISCRIM: str = "one_sided_discrimination_keep"
REASON_NOT_EASY: str = "not_easy_pass_matrix"

EASY_DETECT_SCHEMA: str = "deepagent.easy_detect.v1"


@dataclass(frozen=True, slots=True)
class EasyDetectResult:
    """Per-pack easy classification for hardness promote."""

    pack_id: str
    reason_code: str
    should_drop_hardness: bool
    detail: str
    label: str | None = None
    frontier_pass_at_k: float | None = None
    per_model_pass_at_k: dict[str, float] = field(default_factory=dict)
    models_scored: tuple[str, ...] = field(default_factory=tuple)
    all_models_solved: bool | None = None
    f2p_count: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "reason_code": self.reason_code,
            "should_drop_hardness": self.should_drop_hardness,
            "detail": self.detail,
            "label": self.label,
            "frontier_pass_at_k": self.frontier_pass_at_k,
            "per_model_pass_at_k": dict(self.per_model_pass_at_k),
            "models_scored": list(self.models_scored),
            "all_models_solved": self.all_models_solved,
            "f2p_count": self.f2p_count,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class EasyDetectReport:
    """Batch classification over a scoreboard / panel report."""

    ok: bool
    results: tuple[EasyDetectResult, ...]
    drop_ids: tuple[str, ...]
    keep_ids: tuple[str, ...]
    source: str | None = None
    schema: str = EASY_DETECT_SCHEMA
    reasons: tuple[str, ...] = field(default_factory=tuple)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ok": self.ok,
            "source": self.source,
            "drop_ids": list(self.drop_ids),
            "keep_ids": list(self.keep_ids),
            "results": [r.to_dict() for r in self.results],
            "reasons": list(self.reasons),
            "meta": dict(self.meta),
            "assertions": [
                "VAL-DEASY-002",
                "VAL-DEASY-005",
                "VAL-DINTR-001",
            ],
        }

    def by_pack(self) -> dict[str, EasyDetectResult]:
        return {r.pack_id: r for r in self.results}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pass_is_one(value: float | None, *, eps: float = 1e-9) -> bool:
    if value is None:
        return False
    return value >= (1.0 - eps)


def extract_per_model_pass_at_k(
    row: Mapping[str, Any],
    *,
    models: Sequence[str] | None = None,
) -> dict[str, float]:
    """Normalize model → pass@1 / pass@k from a paint-scoreboard or panel row.

    Accepts shapes:

    * ``per_model_pass_at_k`` / ``pass_at_1`` maps
    * nested ``decision.per_model_pass_at_k``
    * scoreboard flat keys ``grok-4.5`` / ``kimi-k2.6`` / full OpenRouter ids
    * nested ``models`` list of ``{model, pass_at_k|pass_at_1}``
    """
    out: dict[str, float] = {}

    # Nested decision (panel report.pack_results)
    decision = row.get("decision")
    if isinstance(decision, Mapping):
        nested = decision.get("per_model_pass_at_k")
        if isinstance(nested, Mapping):
            for k, v in nested.items():
                f = _as_float(v)
                if f is not None:
                    out[str(k)] = f

    for key in ("per_model_pass_at_k", "pass_at_1", "per_model_pass_at_1"):
        blob = row.get(key)
        if isinstance(blob, Mapping):
            for k, v in blob.items():
                f = _as_float(v)
                if f is not None:
                    out[str(k)] = f

    models_list = row.get("models")
    if isinstance(models_list, list):
        for item in models_list:
            if not isinstance(item, Mapping):
                continue
            mid = item.get("model") or item.get("model_id") or item.get("name")
            if not mid:
                continue
            rate = (
                item.get("pass_at_1")
                if item.get("pass_at_1") is not None
                else item.get("pass_at_k")
            )
            f = _as_float(rate)
            if f is not None:
                out[str(mid)] = f

    # Scoreboard short keys (eval_deepagent compact rows)
    short_keys = (
        "grok-4.5",
        "kimi-k2.6",
        "x-ai/grok-4.5",
        "moonshotai/kimi-k2.6",
        "grok_4_5",
        "kimi_k2_6",
    )
    for sk in short_keys:
        if sk in row and not isinstance(row[sk], Mapping | list | dict):
            f = _as_float(row.get(sk))
            if f is not None:
                out[sk] = f

    # Prefer models declared on the scoreboard document when provided.
    if models:
        preferred: dict[str, float] = {}
        for m in models:
            m_str = str(m)
            if m_str in out:
                preferred[m_str] = out[m_str]
                continue
            # fuzzy match short suffix against full ids
            short = m_str.split("/")[-1]
            if short in out:
                preferred[m_str] = out[short]
                continue
            for key, val in out.items():
                if short in key or key in m_str or m_str in key:
                    preferred[m_str] = val
                    break
        if preferred:
            return preferred

    return out


def extract_frontier_pass_at_k(row: Mapping[str, Any]) -> float | None:
    """Pull frontier aggregate pass@k when present."""
    if "frontier" in row:
        f = _as_float(row.get("frontier"))
        if f is not None:
            return f
    if "frontier_pass_at_k" in row:
        f = _as_float(row.get("frontier_pass_at_k"))
        if f is not None:
            return f
    decision = row.get("decision")
    if isinstance(decision, Mapping) and "frontier_pass_at_k" in decision:
        return _as_float(decision.get("frontier_pass_at_k"))
    return None


def pack_id_from_row(row: Mapping[str, Any]) -> str | None:
    for key in ("pack_id", "task_id", "id", "pack"):
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def classify_pack_from_panel_row(
    row: Mapping[str, Any] | None,
    *,
    pack_id: str | None = None,
    models: Sequence[str] | None = None,
    f2p_node_ids: Sequence[Any] | None = None,
    min_f2p_nodes: int | None = None,
    require_all_models: bool = True,
    hardness: HardnessFloorResult | None = None,
    drop_on_solve_all: bool = False,
) -> EasyDetectResult:
    """Classify one pack from a scoreboard / panel row (M24 / M25).

    *EASY_SOLVE_ALL* when **all** models in the matrix have pass@1 == 1.0, or
    when frontier_pass_at_k ≥ 1.0 and the matrix is non-empty with every rate 1.0.

    M25 / VAL-DINTR-001: model solve-all is a **label only** —
    ``should_drop_hardness`` defaults to ``False`` (*drop_on_solve_all=False*).
    Pass ``drop_on_solve_all=True`` only for legacy M24 replay tests / audits.

    One-sided discrimination (exactly one model at 1.0, others <1) → keep
    (label, no hardness drop).

    Thin F2P is still drop-flagged when *f2p_node_ids* / *hardness* is provided
    (structural floor, independent of model outcomes).
    """
    row = dict(row or {})
    pid = pack_id or pack_id_from_row(row) or "?"
    per_model = extract_per_model_pass_at_k(row, models=models)
    frontier = extract_frontier_pass_at_k(row)
    model_names = tuple(per_model.keys())

    # Structural thin F2P first (stable with hardness_floors codes).
    hard = hardness
    if hard is None and f2p_node_ids is not None:
        hard = check_product_hardness_floors(
            f2p_node_ids=f2p_node_ids,
            min_f2p_nodes=min_f2p_nodes,
            require_hunk_floor=False,
            require_multi_file=False,
            require_f2p_floor=True,
        )
    if hard is not None and not hard.ok:
        thin_codes = {REASON_THIN_F2P_EASY, REASON_F2P_BELOW_FLOOR}
        if thin_codes.intersection(hard.reasons) or hard.reason_code in thin_codes:
            code = (
                REASON_THIN_F2P_EASY if REASON_THIN_F2P_EASY in hard.reasons else hard.reason_code
            )
            return EasyDetectResult(
                pack_id=pid,
                reason_code=code,
                should_drop_hardness=True,
                detail=hard.detail or "thin F2P structural easy class",
                label="THIN_F2P_EASY",
                frontier_pass_at_k=frontier,
                per_model_pass_at_k=per_model,
                models_scored=model_names,
                all_models_solved=None,
                f2p_count=hard.f2p_count,
                meta={"gate": "thin_f2p", "hardness_reasons": list(hard.reasons)},
            )

    if not per_model:
        # Frontier-only solve-all signal (legacy single aggregate) — label only by default.
        if frontier is not None and _pass_is_one(frontier):
            return EasyDetectResult(
                pack_id=pid,
                reason_code=REASON_SOLVE_ALL_EASY,
                should_drop_hardness=bool(drop_on_solve_all),
                detail=(
                    f"EASY_SOLVE_ALL (scoreboard label): frontier_pass_at_k={frontier} ≥ 1.0 "
                    f"(no per-model matrix); should_drop_hardness={bool(drop_on_solve_all)} "
                    f"(M25 default False — model outcomes are not a hardness gate; "
                    f"VAL-DINTR-001)"
                ),
                label=EASY_SOLVE_ALL,
                frontier_pass_at_k=frontier,
                per_model_pass_at_k={},
                models_scored=(),
                all_models_solved=True,
                f2p_count=hard.f2p_count if hard is not None else None,
                meta={
                    "gate": "frontier_only_solve_all",
                    "drop_on_solve_all": bool(drop_on_solve_all),
                    "role": "scoreboard_label",
                },
            )
        return EasyDetectResult(
            pack_id=pid,
            reason_code=REASON_NOT_EASY,
            should_drop_hardness=False,
            detail="no per-model pass matrix; cannot classify as solve-all",
            label=None,
            frontier_pass_at_k=frontier,
            per_model_pass_at_k={},
            models_scored=(),
            all_models_solved=None,
            f2p_count=hard.f2p_count if hard is not None else None,
            meta={"gate": "missing_matrix"},
        )

    solved_flags = {_pass_is_one(v) for v in per_model.values()}
    all_solved = all(_pass_is_one(v) for v in per_model.values())
    any_solved = any(_pass_is_one(v) for v in per_model.values())
    n_models = len(per_model)
    n_full = sum(1 for v in per_model.values() if _pass_is_one(v))

    # Primary: BOTH / all models at pass@1 == 1.0 — label only by default (M25).
    if require_all_models and n_models >= 1 and all_solved:
        return EasyDetectResult(
            pack_id=pid,
            reason_code=REASON_SOLVE_ALL_EASY,
            should_drop_hardness=bool(drop_on_solve_all),
            detail=(
                f"EASY_SOLVE_ALL (scoreboard label): all {n_models} panel model(s) "
                f"pass@1=1.0 "
                f"({', '.join(f'{k}={v}' for k, v in sorted(per_model.items()))}); "
                f"frontier={frontier}; should_drop_hardness={bool(drop_on_solve_all)} "
                f"(M25 default False — dual-model success is not a hardness drop gate; "
                f"VAL-DINTR-001; use intrinsic request+patch + floors + alignment)"
            ),
            label=EASY_SOLVE_ALL,
            frontier_pass_at_k=frontier if frontier is not None else 1.0,
            per_model_pass_at_k=per_model,
            models_scored=model_names,
            all_models_solved=True,
            f2p_count=hard.f2p_count if hard is not None else None,
            meta={
                "gate": "all_models_pass_at_1",
                "n_models": n_models,
                "drop_on_solve_all": bool(drop_on_solve_all),
                "role": "scoreboard_label",
            },
        )

    # Frontier ≥ 1.0 + full matrix: still label-only by default.
    if frontier is not None and _pass_is_one(frontier) and all_solved:
        return EasyDetectResult(
            pack_id=pid,
            reason_code=REASON_SOLVE_ALL_EASY,
            should_drop_hardness=bool(drop_on_solve_all),
            detail=(
                f"EASY_SOLVE_ALL (scoreboard label): frontier_pass_at_k={frontier} ≥ 1.0 "
                f"and full solve matrix; should_drop_hardness={bool(drop_on_solve_all)} "
                f"(M25 / VAL-DINTR-001)"
            ),
            label=EASY_SOLVE_ALL,
            frontier_pass_at_k=frontier,
            per_model_pass_at_k=per_model,
            models_scored=model_names,
            all_models_solved=True,
            f2p_count=hard.f2p_count if hard is not None else None,
            meta={
                "gate": "frontier_and_matrix",
                "drop_on_solve_all": bool(drop_on_solve_all),
                "role": "scoreboard_label",
            },
        )

    # One-sided discrimination: at least one full solve, not all → keep.
    if any_solved and not all_solved:
        return EasyDetectResult(
            pack_id=pid,
            reason_code=REASON_ONE_SIDED_DISCRIM,
            should_drop_hardness=False,
            detail=(
                f"one-sided discrimination: {n_full}/{n_models} models at pass@1=1.0; "
                "retains hardness signal (not solve-all)"
            ),
            label=None,
            frontier_pass_at_k=frontier,
            per_model_pass_at_k=per_model,
            models_scored=model_names,
            all_models_solved=False,
            f2p_count=hard.f2p_count if hard is not None else None,
            meta={"gate": "one_sided", "n_full": n_full, "n_models": n_models},
        )

    return EasyDetectResult(
        pack_id=pid,
        reason_code=REASON_NOT_EASY,
        should_drop_hardness=False,
        detail=(
            f"not easy: matrix={per_model} frontier={frontier} (solved_full={n_full}/{n_models})"
        ),
        label=None,
        frontier_pass_at_k=frontier,
        per_model_pass_at_k=per_model,
        models_scored=model_names,
        all_models_solved=False if solved_flags else None,
        f2p_count=hard.f2p_count if hard is not None else None,
        meta={"gate": "not_easy"},
    )


def _load_mapping(blob: Mapping[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(blob, Mapping):
        return dict(blob)
    path = Path(blob)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"scoreboard/report must be a JSON object; got {type(data)}")
    return data


def _iter_pack_rows(
    doc: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str] | None, str]:
    """Yield pack-level rows from scoreboard.json or report.json shapes."""
    models_raw = doc.get("models")
    models: list[str] | None = None
    if isinstance(models_raw, list) and models_raw and all(isinstance(x, str) for x in models_raw):
        models = [str(x) for x in models_raw]

    if isinstance(doc.get("per_pack"), list):
        rows = [dict(r) for r in doc["per_pack"] if isinstance(r, Mapping)]
        return rows, models, "scoreboard.per_pack"

    if isinstance(doc.get("pack_results"), list):
        rows = [dict(r) for r in doc["pack_results"] if isinstance(r, Mapping)]
        return rows, models, "report.pack_results"

    # Bare list under results/packs
    for key in ("results", "packs", "rows"):
        if isinstance(doc.get(key), list):
            rows = [dict(r) for r in doc[key] if isinstance(r, Mapping)]
            return rows, models, f"doc.{key}"

    return [], models, "empty"


def classify_scoreboard(
    scoreboard: Mapping[str, Any] | Path | str,
    *,
    pack_f2p: Mapping[str, Sequence[Any]] | None = None,
    pack_dirs: Mapping[str, Path | str] | None = None,
    min_f2p_nodes: int | None = None,
    models: Sequence[str] | None = None,
    drop_on_solve_all: bool = False,
) -> EasyDetectReport:
    """Classify every pack on a scoreboard or panel report (M24 + M25).

    Accepts:

    * ``datasets/.../scoreboard.json`` (``per_pack`` compact matrix)
    * ``datasets/.../report.json`` (``pack_results`` + nested decision)

    Optional thin-F2P enrichment via *pack_f2p* map or on-disk *pack_dirs*.

    M25: *drop_on_solve_all* defaults False so model solve-all only labels;
    structural thin-F2P still sets ``should_drop_hardness``.
    """
    source: str | None
    if isinstance(scoreboard, Path | str):
        source = str(scoreboard)
        doc = _load_mapping(scoreboard)
    else:
        source = None
        doc = _load_mapping(scoreboard)

    rows, doc_models, shape = _iter_pack_rows(doc)
    model_list = list(models) if models is not None else doc_models

    results: list[EasyDetectResult] = []
    for row in rows:
        pid = pack_id_from_row(row) or "?"
        f2p: Sequence[Any] | None = None
        hard: HardnessFloorResult | None = None
        if pack_f2p and pid in pack_f2p:
            f2p = pack_f2p[pid]
        elif pack_dirs and pid in pack_dirs:
            hard = hardness_result_from_pack_dir(
                pack_dirs[pid],
                min_f2p_nodes=min_f2p_nodes,
            )
            # Only apply thin F2P as easy drop when hardness fails on F2P; avoid
            # conflating multi-file/hunk shortfalls into panel detector.
            thin_ok = (
                hard.reason_code
                in {
                    REASON_THIN_F2P_EASY,
                    REASON_F2P_BELOW_FLOOR,
                }
                or REASON_THIN_F2P_EASY in hard.reasons
                or REASON_F2P_BELOW_FLOOR in hard.reasons
            )
            if hard.ok or not thin_ok:
                hard = None

        results.append(
            classify_pack_from_panel_row(
                row,
                pack_id=pid,
                models=model_list,
                f2p_node_ids=f2p,
                min_f2p_nodes=min_f2p_nodes,
                hardness=hard,
                drop_on_solve_all=drop_on_solve_all,
            )
        )

    drop_ids = tuple(sorted(r.pack_id for r in results if r.should_drop_hardness))
    keep_ids = tuple(sorted(r.pack_id for r in results if not r.should_drop_hardness))
    labeled_solve_all = tuple(sorted(r.pack_id for r in results if r.label == EASY_SOLVE_ALL))
    return EasyDetectReport(
        ok=True,
        results=tuple(results),
        drop_ids=drop_ids,
        keep_ids=keep_ids,
        source=source,
        reasons=("classified",),
        meta={
            "shape": shape,
            "n_rows": len(results),
            "n_drop": len(drop_ids),
            "n_keep": len(keep_ids),
            "n_easy_solve_all_labels": len(labeled_solve_all),
            "easy_solve_all_labels": list(labeled_solve_all),
            "drop_on_solve_all": bool(drop_on_solve_all),
            "models": list(model_list or []),
            "policy": "m25_intrinsic_hardness",
        },
    )


def force_drop_from_easy_report(
    report: EasyDetectReport,
) -> dict[str, dict[str, str]]:
    """Build ``curate_prod_hard`` force_drop table from easy-detect results.

    No hardcoded pack names — only ``should_drop_hardness`` entries.

    M25: dual-model solve-all no longer populates this table by default
    (should_drop_hardness=False for EASY_SOLVE_ALL labels). Thin F2P structural
    drops still appear when present.
    """
    out: dict[str, dict[str, str]] = {}
    for r in report.results:
        if not r.should_drop_hardness:
            continue
        out[r.pack_id] = {
            "reason_code": r.reason_code,
            "detail": r.detail,
        }
    return out


def classify_and_force_drop(
    scoreboard: Mapping[str, Any] | Path | str,
    **kwargs: Any,
) -> tuple[EasyDetectReport, dict[str, dict[str, str]]]:
    """Convenience for curate-hardness CLI / pipeline wire-up.

    kwargs include *drop_on_solve_all* (default False under M25).
    """
    report = classify_scoreboard(scoreboard, **kwargs)
    return report, force_drop_from_easy_report(report)


__all__ = [
    "EASY_DETECT_SCHEMA",
    "EASY_SOLVE_ALL",
    "REASON_EASY_SOLVE_ALL",
    "REASON_NOT_EASY",
    "REASON_OK_HARDNESS",
    "REASON_ONE_SIDED_DISCRIM",
    "EasyDetectReport",
    "EasyDetectResult",
    "classify_and_force_drop",
    "classify_pack_from_panel_row",
    "classify_scoreboard",
    "extract_frontier_pass_at_k",
    "extract_per_model_pass_at_k",
    "force_drop_from_easy_report",
    "pack_id_from_row",
]
