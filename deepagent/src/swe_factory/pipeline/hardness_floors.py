"""Product hardness floors + anti-easy policy (M21 / VAL-DHARD-002/003/005).

Fail-closed floors for product and live-generate dests:

* source hunks ≥ ``PRODUCT_SOURCE_HUNK_FLOOR`` (default **10**, existing)
* multi-file product sources ≥ ``PRODUCT_MULTI_FILE_FLOOR`` (default **2**)
* **F2P node count ≥ ``MIN_F2P_NODES`` (default **3**)** — refuse thin F2P=1

``MIN_F2P_NODES`` is overridable via env ``MIN_F2P_NODES`` /
``DEEPAGENT_MIN_F2P_NODES`` (positive int). Product defaults never soften below 3
unless the env/const is explicitly set; thin packs still need an engineering
opt-out flag to pass on offline fixtures.

Anti-easy policy (VAL-DHARD-003)
--------------------------------
* Solve-all class (frontier pass@k = 1.0) is **dropped** from hardness promote
  (see panel band filter + curated production drops).
* Thin F2P≈1 packs are refused on product/live_generate dests by the F2P floor
  (unless ``engineering_opt_out=True`` for offline fixture-only ships — never the
  product default).
* Misaligned prompt↔verifier packs are refused by the alignment gate (M21a).

Agent timeout-class failures remain harness OK and out of this gate's scope.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.producers.hard_filter import (
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
)

# ---------------------------------------------------------------------------
# Floors (const) + env resolution
# ---------------------------------------------------------------------------

#: Default minimum F2P nodes for product / live_generate hardness dests.
DEFAULT_MIN_F2P_NODES: int = 3

#: Env keys for MIN_F2P_NODES override (first positive wins).
_MIN_F2P_ENV_KEYS: tuple[str, ...] = (
    "MIN_F2P_NODES",
    "DEEPAGENT_MIN_F2P_NODES",
)

# Stable reason codes (gate_audit / drip / raise messages)
REASON_HARDNESS_OK: str = "hardness_floors_ok"
REASON_HARDNESS_SKIPPED: str = "hardness_floors_skipped_engineering"
REASON_F2P_BELOW_FLOOR: str = "f2p_nodes_below_floor"
REASON_THIN_F2P_EASY: str = "thin_f2p_easy_class"
REASON_SOURCE_HUNKS_BELOW_FLOOR: str = "source_hunks_below_floor"
REASON_MULTI_FILE_FLOOR: str = "multi_file_floor_rejected"
REASON_SOLVE_ALL_EASY: str = "solve_all_easy_policy_drop"
REASON_EMPTY_F2P: str = "empty_f2p_node_ids"

# Dest markers kept in sync with ship_real_pr / prompt_alignment honesty paths.
_PRODUCT_DEST_MARKERS: tuple[str, ...] = ("deepagent_v1",)
_LIVE_GENERATE_DEST_MARKERS: tuple[str, ...] = ("test_n10", "prod_hard_keep")


class ProductHardnessFloorRejected(RuntimeError):
    """Product/live_generate refuse when hardness floors fail (VAL-DHARD-002)."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = REASON_F2P_BELOW_FLOOR,
        result: HardnessFloorResult | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.result = result


@dataclass(frozen=True, slots=True)
class HardnessFloorResult:
    """Outcome of product hardness floor evaluation."""

    ok: bool
    reason_code: str
    detail: str
    f2p_count: int = 0
    source_file_count: int = 0
    source_hunk_count: int | None = None
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES
    min_source_hunks: int = PRODUCT_SOURCE_HUNK_FLOOR
    min_source_files: int = PRODUCT_MULTI_FILE_FLOOR
    reasons: tuple[str, ...] = field(default_factory=tuple)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "f2p_count": self.f2p_count,
            "source_file_count": self.source_file_count,
            "source_hunk_count": self.source_hunk_count,
            "min_f2p_nodes": self.min_f2p_nodes,
            "min_source_hunks": self.min_source_hunks,
            "min_source_files": self.min_source_files,
            "reasons": list(self.reasons),
            "meta": dict(self.meta),
        }


def resolve_min_f2p_nodes(
    *,
    override: int | None = None,
    env: Mapping[str, str] | None = None,
    default: int = DEFAULT_MIN_F2P_NODES,
) -> int:
    """Resolve MIN_F2P_NODES from override → env → default (≥1).

    Product policy default remains 3. Explicit override/env may raise the floor
    further; values <1 fall back to *default*.
    """
    if override is not None:
        try:
            n = int(override)
        except (TypeError, ValueError):
            n = default
        return n if n >= 1 else default

    env_map = env if env is not None else os.environ
    for key in _MIN_F2P_ENV_KEYS:
        raw = env_map.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            n = int(text)
        except ValueError:
            continue
        if n >= 1:
            return n
    return default if default >= 1 else DEFAULT_MIN_F2P_NODES


def is_hardness_enforced_dest(
    dest: Path | str | None,
    *,
    offline_only: bool = False,
    live_mine: bool = False,
    engineering_opt_out: bool = False,
) -> bool:
    """True when product hardness floors hard-refuse on this dest.

    Engineering opt-out (offline fixtures only) always skips enforcement.
    Offline-only paths skip unless *live_mine* forces honesty.
    """
    if engineering_opt_out:
        return False
    if offline_only and not live_mine:
        return False
    if dest is None:
        # Unknown dest: treat as product-conservative only when live_mine.
        return bool(live_mine)

    text = str(dest).replace("\\", "/").lower().rstrip("/")
    parts = [p for p in text.split("/") if p]

    # Exact product segment (same rules as ship_real_pr.is_product_deepagent_dest).
    if "deepagent_v1" in parts:
        leaf = parts[-1] if parts else ""
        if not (leaf.startswith("deepagent") and leaf != "deepagent_v1"):
            return True

    if any(part in _LIVE_GENERATE_DEST_MARKERS for part in parts):
        return True

    return bool(live_mine)


def _norm_f2p(ids: Sequence[Any] | None) -> list[str]:
    if not ids:
        return []
    return [str(x).strip() for x in ids if str(x).strip()]


def _norm_source_files(files: Sequence[str] | None) -> list[str]:
    if not files:
        return []
    out: list[str] = []
    for f in files:
        p = str(f).strip().replace("\\", "/")
        if not p:
            continue
        # Coarse test-path skip (product multi-file counts source only).
        low = p.lower()
        base = Path(p).name.lower()
        if (
            "/test/" in f"/{low}/"
            or "/tests/" in f"/{low}/"
            or base.startswith("test_")
            or base.endswith("_test.py")
            or base.endswith("_test.go")
            or base.endswith(".test.ts")
            or base.endswith(".test.js")
            or base.endswith("_test.rs")
        ):
            continue
        out.append(p)
    return out


def check_product_hardness_floors(
    *,
    f2p_node_ids: Sequence[Any] | None = None,
    source_files: Sequence[str] | None = None,
    source_hunk_count: int | None = None,
    min_f2p_nodes: int | None = None,
    min_source_hunks: int = PRODUCT_SOURCE_HUNK_FLOOR,
    min_source_files: int = PRODUCT_MULTI_FILE_FLOOR,
    panel_frontier_pass_at_k: float | None = None,
    require_hunk_floor: bool = True,
    require_multi_file: bool = True,
    require_f2p_floor: bool = True,
    env: Mapping[str, str] | None = None,
) -> HardnessFloorResult:
    """Evaluate product hardness floors (VAL-DHARD-002 / VAL-DHARD-003).

    Does not raise — callers use :func:`refuse_product_hardness_floors` for
    fail-closed product refuse.
    """
    min_f2p = resolve_min_f2p_nodes(override=min_f2p_nodes, env=env)
    f2p = _norm_f2p(f2p_node_ids)
    sources = _norm_source_files(source_files)
    # If only source_file_count is implied by raw count without filter, callers
    # may pass already-filtered product source file lists from materials.
    src_count = (
        len(sources) if sources else len([s for s in (source_files or ()) if str(s).strip()])
    )
    hunk: int | None
    try:
        hunk = int(source_hunk_count) if source_hunk_count is not None else None
    except (TypeError, ValueError):
        hunk = None

    reasons: list[str] = []
    details: list[str] = []
    meta: dict[str, Any] = {
        "min_f2p_nodes": min_f2p,
        "min_source_hunks": min_source_hunks,
        "min_source_files": min_source_files,
        "anti_easy_policy": "solve_all_and_thin_f2p_dropped_from_hardness_promote",
        "default_min_f2p_nodes": DEFAULT_MIN_F2P_NODES,
    }

    if require_f2p_floor:
        if not f2p:
            reasons.append(REASON_EMPTY_F2P)
            details.append("product hardness requires non-empty F2P node ids")
        elif len(f2p) < min_f2p:
            reasons.append(REASON_F2P_BELOW_FLOOR)
            details.append(
                f"f2p_count={len(f2p)} < MIN_F2P_NODES={min_f2p} "
                "(thin F2P refused on product hardness path)"
            )
            # Fingerprint easy class for refuse table / audit smoke.
            if len(f2p) <= 1:
                reasons.append(REASON_THIN_F2P_EASY)
                details.append("thin F2P≈1 classified as easy; anti-easy policy drop")

    # Only when source file list was provided (unknown → skip soft).
    if require_multi_file and source_files is not None and src_count < min_source_files:
        reasons.append(REASON_MULTI_FILE_FLOOR)
        details.append(f"product source files {src_count} < multi-file floor {min_source_files}")

    if require_hunk_floor and hunk is not None and hunk < min_source_hunks:
        reasons.append(REASON_SOURCE_HUNKS_BELOW_FLOOR)
        details.append(f"source_hunk_count={hunk} < product floor {min_source_hunks}")

    # Optional solve-all fingerprint when panel already known (structural
    # promote path). Live dual-model canary is optional; policy still applies
    # when a frontier aggregate of 1.0 is supplied.
    if panel_frontier_pass_at_k is not None:
        try:
            fr = float(panel_frontier_pass_at_k)
        except (TypeError, ValueError):
            fr = None
        if fr is not None and fr >= 1.0:
            reasons.append(REASON_SOLVE_ALL_EASY)
            details.append(
                f"frontier pass@k={fr} is solve-all; anti-easy policy drops from hardness promote"
            )
            meta["panel_frontier_pass_at_k"] = fr

    if reasons:
        # Prefer F2P reason code as primary (feature headline floor).
        primary = reasons[0]
        for preferred in (
            REASON_F2P_BELOW_FLOOR,
            REASON_THIN_F2P_EASY,
            REASON_SOLVE_ALL_EASY,
            REASON_SOURCE_HUNKS_BELOW_FLOOR,
            REASON_MULTI_FILE_FLOOR,
            REASON_EMPTY_F2P,
        ):
            if preferred in reasons:
                primary = preferred
                break
        return HardnessFloorResult(
            ok=False,
            reason_code=primary,
            detail="; ".join(details) if details else primary,
            f2p_count=len(f2p),
            source_file_count=src_count,
            source_hunk_count=hunk,
            min_f2p_nodes=min_f2p,
            min_source_hunks=min_source_hunks,
            min_source_files=min_source_files,
            reasons=tuple(dict.fromkeys(reasons)),
            meta=meta,
        )

    return HardnessFloorResult(
        ok=True,
        reason_code=REASON_HARDNESS_OK,
        detail=(
            f"product hardness floors ok "
            f"(f2p={len(f2p)}>={min_f2p}, sources={src_count}, hunks={hunk})"
        ),
        f2p_count=len(f2p),
        source_file_count=src_count,
        source_hunk_count=hunk,
        min_f2p_nodes=min_f2p,
        min_source_hunks=min_source_hunks,
        min_source_files=min_source_files,
        reasons=(REASON_HARDNESS_OK,),
        meta=meta,
    )


def refuse_product_hardness_floors(
    *,
    f2p_node_ids: Sequence[Any] | None = None,
    source_files: Sequence[str] | None = None,
    source_hunk_count: int | None = None,
    dest: Path | str | None = None,
    offline_only: bool = False,
    live_mine: bool = False,
    engineering_opt_out: bool = False,
    force: bool = False,
    min_f2p_nodes: int | None = None,
    panel_frontier_pass_at_k: float | None = None,
    require_hunk_floor: bool = True,
    require_multi_file: bool = True,
    require_f2p_floor: bool = True,
    task_id: str | None = None,
    env: Mapping[str, str] | None = None,
) -> HardnessFloorResult:
    """Fail-closed refuse when product hardness floors fail (VAL-DHARD-002/005).

    *engineering_opt_out*: only for offline fixture / engineering ships. Never
    the product default. When True, hard refuse is skipped and a skipped result
    is returned.

    *force*: enforce even on non-product dests (unit tests / explicit certify).
    """
    enforce = force or is_hardness_enforced_dest(
        dest,
        offline_only=offline_only,
        live_mine=live_mine,
        engineering_opt_out=engineering_opt_out,
    )
    result = check_product_hardness_floors(
        f2p_node_ids=f2p_node_ids,
        source_files=source_files,
        source_hunk_count=source_hunk_count,
        min_f2p_nodes=min_f2p_nodes,
        panel_frontier_pass_at_k=panel_frontier_pass_at_k,
        require_hunk_floor=require_hunk_floor,
        require_multi_file=require_multi_file,
        require_f2p_floor=require_f2p_floor,
        env=env,
    )
    if not enforce:
        return HardnessFloorResult(
            ok=True,
            reason_code=REASON_HARDNESS_SKIPPED,
            detail=(
                "hardness floors hard refuse skipped "
                f"(engineering_opt_out={engineering_opt_out}, offline_only={offline_only}, "
                f"dest={dest!r})"
            ),
            f2p_count=result.f2p_count,
            source_file_count=result.source_file_count,
            source_hunk_count=result.source_hunk_count,
            min_f2p_nodes=result.min_f2p_nodes,
            min_source_hunks=result.min_source_hunks,
            min_source_files=result.min_source_files,
            reasons=(REASON_HARDNESS_SKIPPED,),
            meta={**result.meta, "enforced": False},
        )
    if result.ok:
        return result

    label = task_id or "pack"
    raise ProductHardnessFloorRejected(
        f"product hardness floors refuse for {label}: "
        f"{result.reason_code}: {result.detail} "
        f"(VAL-DHARD-002/003/005; dest={dest})",
        reason_code=result.reason_code,
        result=result,
    )


def hardness_result_from_pack_dir(
    pack_dir: Path | str,
    *,
    source_hunk_count: int | None = None,
    source_files: Sequence[str] | None = None,
    min_f2p_nodes: int | None = None,
) -> HardnessFloorResult:
    """Convenience: read tests/config.json (+ optional solution file list)."""
    root = Path(pack_dir)
    f2p: list[str] = []
    cfg = root / "tests" / "config.json"
    if cfg.is_file():
        try:
            import json

            blob = json.loads(cfg.read_text(encoding="utf-8"))
            raw = blob.get("f2p_node_ids") or blob.get("fail_to_pass") or []
            if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
                f2p = [str(x) for x in raw]
        except Exception:  # noqa: BLE001 — best-effort meta only
            f2p = []

    files: list[str] = list(source_files or [])
    if not files:
        sol = root / "solution" / "solution.patch"
        if not sol.is_file():
            sol = root / "solution.patch"
        if sol.is_file():
            # Lightweight path scrape from unified-diff headers.
            import re

            text = sol.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"^diff --git a/(.+?) b/(.+)$", text, re.MULTILINE):
                files.append(m.group(2))
            if not files:
                for m in re.finditer(r"^\+\+\+ b/(.+)$", text, re.MULTILINE):
                    files.append(m.group(1))

    return check_product_hardness_floors(
        f2p_node_ids=f2p,
        source_files=files or None,
        source_hunk_count=source_hunk_count,
        min_f2p_nodes=min_f2p_nodes,
    )


def anti_easy_policy_summary() -> dict[str, Any]:
    """Documented anti-easy refuse table for reports / PRODUCT_HARDNESS.md sync."""
    return {
        "policy": "anti_easy_hardness_promote",
        "assertions": ["VAL-DHARD-002", "VAL-DHARD-003", "VAL-DHARD-005"],
        "floors": {
            "min_f2p_nodes": DEFAULT_MIN_F2P_NODES,
            "min_f2p_env_keys": list(_MIN_F2P_ENV_KEYS),
            "source_hunk_floor": PRODUCT_SOURCE_HUNK_FLOOR,
            "multi_file_floor": PRODUCT_MULTI_FILE_FLOOR,
        },
        "refuse_reason_codes": {
            REASON_F2P_BELOW_FLOOR: "F2P node count below MIN_F2P_NODES (default 3)",
            REASON_THIN_F2P_EASY: "thin F2P≈1 fingerprint (easy class)",
            REASON_EMPTY_F2P: "empty F2P node ids",
            REASON_SOURCE_HUNKS_BELOW_FLOOR: f"source hunks < {PRODUCT_SOURCE_HUNK_FLOOR}",
            REASON_MULTI_FILE_FLOOR: f"product sources < {PRODUCT_MULTI_FILE_FLOOR}",
            REASON_SOLVE_ALL_EASY: "panel frontier pass@k=1.0 solve-all dropped",
        },
        "engineering_opt_out": (
            "Explicit engineering_opt_out=True or offline_only dests may skip; "
            "never the product / live_generate default."
        ),
        "notes": [
            "Solve-all class dropped from hardness promote (panel band + curate).",
            "Thin F2P=1 packs refuse product dest by default.",
            "Prompt–verifier alignment (M21a) is a separate fail-closed gate.",
            "Agent timeout-class model failures remain harness OK.",
        ],
    }


__all__ = [
    "DEFAULT_MIN_F2P_NODES",
    "PRODUCT_MULTI_FILE_FLOOR",
    "PRODUCT_SOURCE_HUNK_FLOOR",
    "REASON_EMPTY_F2P",
    "REASON_F2P_BELOW_FLOOR",
    "REASON_HARDNESS_OK",
    "REASON_HARDNESS_SKIPPED",
    "REASON_MULTI_FILE_FLOOR",
    "REASON_SOLVE_ALL_EASY",
    "REASON_SOURCE_HUNKS_BELOW_FLOOR",
    "REASON_THIN_F2P_EASY",
    "HardnessFloorResult",
    "ProductHardnessFloorRejected",
    "anti_easy_policy_summary",
    "check_product_hardness_floors",
    "hardness_result_from_pack_dir",
    "is_hardness_enforced_dest",
    "refuse_product_hardness_floors",
    "resolve_min_f2p_nodes",
]
