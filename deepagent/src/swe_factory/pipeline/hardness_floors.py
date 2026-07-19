"""Product hardness floors + anti-easy policy (M21 / M27 DeepSWE-median).

Fail-closed floors for product and live-generate dests (VAL-DMED-001 / VAL-DMED-012):

* multi-file: files ≥ **4** **OR** (files ≥ **3** AND added ≥ **500** AND hunks ≥ **14**)
  (packaging-class hybrid admit; qs/thin 2-file still refuse)
* source hunks ≥ ``PRODUCT_SOURCE_HUNK_FLOOR`` (default **14**, DeepSWE p50)
* gold added lines ≥ ``PRODUCT_MIN_ADDED_LINES`` (default **400**, DeepSWE min≈438)
* F2P node count ≥ ``MIN_F2P_NODES`` / ``DEFAULT_MIN_F2P_NODES`` (default **5**)

``MIN_F2P_NODES`` is overridable via env ``MIN_F2P_NODES`` /
``DEEPAGENT_MIN_F2P_NODES`` (positive int). Product defaults never soften below
the DeepSWE-median band unless the env/const is explicitly set; thin packs
still need an engineering opt-out flag to pass on offline fixtures.

Anti-easy policy
----------------
* Thin structural packs refuse on product/live_generate dests (F2P / files /
  hunks / added floors).
* High-confidence intrinsic ``EASY_REQUEST`` may drop via curate
  (:mod:`intrinsic_difficulty`) — model dual-success is **not** a sole drop
  (M25).
* Misaligned prompt↔verifier packs are refused by the alignment gate (M21a).

Agent timeout-class failures remain harness OK and out of this gate's scope.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.producers.hard_filter import (
    PRODUCT_MULTI_FILE_FLOOR,
    PRODUCT_SOURCE_HUNK_FLOOR,
    count_unified_diff_hunks,
)

# ---------------------------------------------------------------------------
# Floors (const) + env resolution
# ---------------------------------------------------------------------------

#: Default minimum F2P nodes for product / live_generate hardness dests (M27).
DEFAULT_MIN_F2P_NODES: int = 5

#: Minimum gold solution.patch plus-lines (unified diff, excluding ``+++`` headers).
PRODUCT_MIN_ADDED_LINES: int = 400

#: Hybrid multi-file DeepSWE-min branch (VAL-DMED-012): files≥3 admits when
#: gold_added ≥ this floor **and** hunks ≥ PRODUCT_SOURCE_HUNK_FLOOR.
#: packaging-1120 class (3 files, added≈882, hunks=24) qualifies; thin 3-file does not.
PRODUCT_HYBRID_MIN_SOURCE_FILES: int = 3
PRODUCT_HYBRID_MIN_ADDED_LINES: int = 500

#: Env keys for MIN_F2P_NODES override (first positive wins).
_MIN_F2P_ENV_KEYS: tuple[str, ...] = (
    "MIN_F2P_NODES",
    "DEEPAGENT_MIN_F2P_NODES",
)

#: Env keys for PRODUCT_MIN_ADDED_LINES override (first positive wins).
_MIN_ADDED_ENV_KEYS: tuple[str, ...] = (
    "PRODUCT_MIN_ADDED_LINES",
    "DEEPAGENT_MIN_ADDED_LINES",
    "MIN_ADDED_LINES",
)

# Stable reason codes (gate_audit / drip / raise messages)
REASON_HARDNESS_OK: str = "hardness_floors_ok"
REASON_HARDNESS_SKIPPED: str = "hardness_floors_skipped_engineering"
REASON_F2P_BELOW_FLOOR: str = "f2p_nodes_below_floor"
REASON_THIN_F2P_EASY: str = "thin_f2p_easy_class"
REASON_SOURCE_HUNKS_BELOW_FLOOR: str = "source_hunks_below_floor"
REASON_MULTI_FILE_FLOOR: str = "multi_file_floor_rejected"
REASON_ADDED_LINES_BELOW_FLOOR: str = "gold_added_lines_below_floor"
REASON_SOLVE_ALL_EASY: str = "solve_all_easy_policy_drop"
REASON_EMPTY_F2P: str = "empty_f2p_node_ids"

# Dest markers kept in sync with ship_real_pr / prompt_alignment honesty paths.
_PRODUCT_DEST_MARKERS: tuple[str, ...] = ("deepagent_v1",)
_LIVE_GENERATE_DEST_MARKERS: tuple[str, ...] = (
    "test_n10",
    "prod_hard_keep",
    "prod_hard_deepswe_med",
)

_PLUS_LINE_RE = re.compile(r"(?m)^\+(?!\+\+ )")
_DIFF_GIT_RE = re.compile(r"(?m)^diff --git a/(.+?) b/")


class ProductHardnessFloorRejected(RuntimeError):
    """Product/live_generate refuse when hardness floors fail (VAL-DHARD-002 / VAL-DMED-001)."""

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
    added_lines: int | None = None
    min_f2p_nodes: int = DEFAULT_MIN_F2P_NODES
    min_source_hunks: int = PRODUCT_SOURCE_HUNK_FLOOR
    min_source_files: int = PRODUCT_MULTI_FILE_FLOOR
    min_added_lines: int = PRODUCT_MIN_ADDED_LINES
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
            "added_lines": self.added_lines,
            "min_f2p_nodes": self.min_f2p_nodes,
            "min_source_hunks": self.min_source_hunks,
            "min_source_files": self.min_source_files,
            "min_added_lines": self.min_added_lines,
            "reasons": list(self.reasons),
            "meta": dict(self.meta),
        }


def count_gold_added_lines(solution_patch: str | None) -> int:
    """Count gold solution.patch plus-lines (exclude ``+++`` file headers)."""
    if not solution_patch or not str(solution_patch).strip():
        return 0
    return len(_PLUS_LINE_RE.findall(str(solution_patch)))


def multi_file_floor_ok(
    *,
    source_files: int,
    added_lines: int | None = None,
    hunks: int | None = None,
    min_source_files: int = PRODUCT_MULTI_FILE_FLOOR,
    hybrid_min_files: int = PRODUCT_HYBRID_MIN_SOURCE_FILES,
    hybrid_min_added: int = PRODUCT_HYBRID_MIN_ADDED_LINES,
    hybrid_min_hunks: int = PRODUCT_SOURCE_HUNK_FLOOR,
) -> bool:
    """DeepSWE-min hybrid multi-file admit (VAL-DMED-012).

    True when ``source_files ≥ min_source_files`` (default 4) **or** the hybrid
    branch holds: ``files ≥ 3 AND added ≥ 500 AND hunks ≥ 14``.

    packaging-class large refactors (3 files, large gold) pass; qs-class thin
    2-file APIs and thin 3-file packs fail.
    """
    try:
        n_files = int(source_files)
    except (TypeError, ValueError):
        return False
    if n_files >= int(min_source_files):
        return True
    if n_files < int(hybrid_min_files):
        return False
    try:
        added = int(added_lines) if added_lines is not None else -1
    except (TypeError, ValueError):
        added = -1
    try:
        n_hunks = int(hunks) if hunks is not None else -1
    except (TypeError, ValueError):
        n_hunks = -1
    return added >= int(hybrid_min_added) and n_hunks >= int(hybrid_min_hunks)


def resolve_min_f2p_nodes(
    *,
    override: int | None = None,
    env: Mapping[str, str] | None = None,
    default: int = DEFAULT_MIN_F2P_NODES,
) -> int:
    """Resolve MIN_F2P_NODES from override → env → default (≥1).

    Product policy default is DeepSWE-median F2P≥5 (M27). Explicit override/env
    may raise or lower the floor; values <1 fall back to *default*.
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


def resolve_min_added_lines(
    *,
    override: int | None = None,
    env: Mapping[str, str] | None = None,
    default: int = PRODUCT_MIN_ADDED_LINES,
) -> int:
    """Resolve PRODUCT_MIN_ADDED_LINES from override → env → default (≥0)."""
    if override is not None:
        try:
            n = int(override)
        except (TypeError, ValueError):
            n = default
        return n if n >= 0 else default

    env_map = env if env is not None else os.environ
    for key in _MIN_ADDED_ENV_KEYS:
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
        if n >= 0:
            return n
    return default if default >= 0 else PRODUCT_MIN_ADDED_LINES


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


def _source_files_from_patch(solution_patch: str | None) -> list[str]:
    if not solution_patch:
        return []
    files = _DIFF_GIT_RE.findall(solution_patch)
    if not files:
        for line in solution_patch.splitlines():
            if line.startswith("+++ b/") and not line.startswith("+++ b/dev/null"):
                files.append(line[6:].strip())
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        f = f.strip()
        if not f or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def check_product_hardness_floors(
    *,
    f2p_node_ids: Sequence[Any] | None = None,
    source_files: Sequence[str] | None = None,
    source_hunk_count: int | None = None,
    solution_patch: str | None = None,
    added_lines: int | None = None,
    min_f2p_nodes: int | None = None,
    min_source_hunks: int = PRODUCT_SOURCE_HUNK_FLOOR,
    min_source_files: int = PRODUCT_MULTI_FILE_FLOOR,
    min_added_lines: int | None = None,
    panel_frontier_pass_at_k: float | None = None,
    require_hunk_floor: bool = True,
    require_multi_file: bool = True,
    require_f2p_floor: bool = True,
    require_added_floor: bool = True,
    env: Mapping[str, str] | None = None,
) -> HardnessFloorResult:
    """Evaluate product hardness floors (VAL-DMED-001 / VAL-DHARD-002/003).

    Does not raise — callers use :func:`refuse_product_hardness_floors` for
    fail-closed product refuse.
    """
    min_f2p = resolve_min_f2p_nodes(override=min_f2p_nodes, env=env)
    min_added = resolve_min_added_lines(override=min_added_lines, env=env)
    f2p = _norm_f2p(f2p_node_ids)
    sources = _norm_source_files(source_files)
    # Prefer explicit source_files; falls back to patch path scrape for counts.
    if not sources and solution_patch:
        sources = _norm_source_files(_source_files_from_patch(solution_patch))
    src_count = (
        len(sources) if sources else len([s for s in (source_files or ()) if str(s).strip()])
    )
    hunk: int | None
    try:
        hunk = int(source_hunk_count) if source_hunk_count is not None else None
    except (TypeError, ValueError):
        hunk = None
    if hunk is None and solution_patch is not None:
        hunk = count_unified_diff_hunks(solution_patch)

    added: int | None = None
    if added_lines is not None:
        try:
            added = int(added_lines)
        except (TypeError, ValueError):
            added = None
    if added is None and solution_patch is not None:
        added = count_gold_added_lines(solution_patch)

    reasons: list[str] = []
    details: list[str] = []
    meta: dict[str, Any] = {
        "min_f2p_nodes": min_f2p,
        "min_source_hunks": min_source_hunks,
        "min_source_files": min_source_files,
        "min_added_lines": min_added,
        "anti_easy_policy": "deepswe_median_structural_floors_and_intrinsic_easy",
        "default_min_f2p_nodes": DEFAULT_MIN_F2P_NODES,
        "product_min_added_lines": PRODUCT_MIN_ADDED_LINES,
        "band": "deepswe_median_m27",
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

    # Only when source file list was provided or scraped (unknown → skip soft).
    # Hybrid multi-file (VAL-DMED-012): files≥4 OR (files≥3 AND added≥500 AND hunks≥14).
    have_source_signal = source_files is not None or (solution_patch is not None and bool(sources))
    hybrid_admit = False
    if require_multi_file and have_source_signal:
        if multi_file_floor_ok(
            source_files=src_count,
            added_lines=added,
            hunks=hunk,
            min_source_files=min_source_files,
            hybrid_min_files=PRODUCT_HYBRID_MIN_SOURCE_FILES,
            hybrid_min_added=PRODUCT_HYBRID_MIN_ADDED_LINES,
            hybrid_min_hunks=min_source_hunks,
        ):
            if src_count < min_source_files:
                hybrid_admit = True
                meta["multi_file_hybrid_admit"] = True
                meta["multi_file_rule"] = "files_ge_4_or_hybrid_3"
        else:
            reasons.append(REASON_MULTI_FILE_FLOOR)
            details.append(
                f"product source files {src_count} < multi-file floor {min_source_files} "
                f"and hybrid branch failed "
                f"(need files≥{PRODUCT_HYBRID_MIN_SOURCE_FILES} AND "
                f"added≥{PRODUCT_HYBRID_MIN_ADDED_LINES} AND hunks≥{min_source_hunks}; "
                f"got added={added}, hunks={hunk})"
            )
    meta.setdefault("multi_file_rule", "files_ge_4_or_hybrid_3")
    meta["multi_file_hybrid_admit"] = bool(hybrid_admit)

    if require_hunk_floor and hunk is not None and hunk < min_source_hunks:
        reasons.append(REASON_SOURCE_HUNKS_BELOW_FLOOR)
        details.append(f"source_hunk_count={hunk} < product floor {min_source_hunks}")

    if require_added_floor and added is not None and added < min_added:
        reasons.append(REASON_ADDED_LINES_BELOW_FLOOR)
        details.append(
            f"gold_added_lines={added} < PRODUCT_MIN_ADDED_LINES={min_added} "
            "(DeepSWE-median gold size floor)"
        )

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
            REASON_ADDED_LINES_BELOW_FLOOR,
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
            added_lines=added,
            min_f2p_nodes=min_f2p,
            min_source_hunks=min_source_hunks,
            min_source_files=min_source_files,
            min_added_lines=min_added,
            reasons=tuple(dict.fromkeys(reasons)),
            meta=meta,
        )

    return HardnessFloorResult(
        ok=True,
        reason_code=REASON_HARDNESS_OK,
        detail=(
            f"product hardness floors ok "
            f"(f2p={len(f2p)}>={min_f2p}, sources={src_count}, hunks={hunk}, "
            f"added={added}>={min_added})"
        ),
        f2p_count=len(f2p),
        source_file_count=src_count,
        source_hunk_count=hunk,
        added_lines=added,
        min_f2p_nodes=min_f2p,
        min_source_hunks=min_source_hunks,
        min_source_files=min_source_files,
        min_added_lines=min_added,
        reasons=(REASON_HARDNESS_OK,),
        meta=meta,
    )


def refuse_product_hardness_floors(
    *,
    f2p_node_ids: Sequence[Any] | None = None,
    source_files: Sequence[str] | None = None,
    source_hunk_count: int | None = None,
    solution_patch: str | None = None,
    added_lines: int | None = None,
    dest: Path | str | None = None,
    offline_only: bool = False,
    live_mine: bool = False,
    engineering_opt_out: bool = False,
    force: bool = False,
    min_f2p_nodes: int | None = None,
    min_added_lines: int | None = None,
    panel_frontier_pass_at_k: float | None = None,
    require_hunk_floor: bool = True,
    require_multi_file: bool = True,
    require_f2p_floor: bool = True,
    require_added_floor: bool = True,
    task_id: str | None = None,
    env: Mapping[str, str] | None = None,
) -> HardnessFloorResult:
    """Fail-closed refuse when product hardness floors fail (VAL-DMED-001 / VAL-DHARD).

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
        solution_patch=solution_patch,
        added_lines=added_lines,
        min_f2p_nodes=min_f2p_nodes,
        min_added_lines=min_added_lines,
        panel_frontier_pass_at_k=panel_frontier_pass_at_k,
        require_hunk_floor=require_hunk_floor,
        require_multi_file=require_multi_file,
        require_f2p_floor=require_f2p_floor,
        require_added_floor=require_added_floor,
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
            added_lines=result.added_lines,
            min_f2p_nodes=result.min_f2p_nodes,
            min_source_hunks=result.min_source_hunks,
            min_source_files=result.min_source_files,
            min_added_lines=result.min_added_lines,
            reasons=(REASON_HARDNESS_SKIPPED,),
            meta={**result.meta, "enforced": False},
        )
    if result.ok:
        return result

    label = task_id or "pack"
    raise ProductHardnessFloorRejected(
        f"product hardness floors refuse for {label}: "
        f"{result.reason_code}: {result.detail} "
        f"(VAL-DMED-001/VAL-DHARD-002/003/005; dest={dest})",
        reason_code=result.reason_code,
        result=result,
    )


def hardness_result_from_pack_dir(
    pack_dir: Path | str,
    *,
    source_hunk_count: int | None = None,
    source_files: Sequence[str] | None = None,
    min_f2p_nodes: int | None = None,
    min_added_lines: int | None = None,
) -> HardnessFloorResult:
    """Convenience: read tests/config.json + solution.patch floors."""
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
    patch_text: str | None = None
    sol = root / "solution" / "solution.patch"
    if not sol.is_file():
        sol = root / "solution.patch"
    if sol.is_file():
        patch_text = sol.read_text(encoding="utf-8", errors="replace")
        if not files:
            files = _source_files_from_patch(patch_text)

    hunks = source_hunk_count
    if hunks is None and patch_text is not None:
        hunks = count_unified_diff_hunks(patch_text)

    return check_product_hardness_floors(
        f2p_node_ids=f2p,
        source_files=files or None,
        source_hunk_count=hunks,
        solution_patch=patch_text,
        min_f2p_nodes=min_f2p_nodes,
        min_added_lines=min_added_lines,
    )


def anti_easy_policy_summary() -> dict[str, Any]:
    """Documented anti-easy refuse table for reports / PRODUCT_HARDNESS.md sync."""
    return {
        "policy": "anti_easy_hardness_promote_deepswe_median",
        "assertions": [
            "VAL-DMED-001",
            "VAL-DMED-002",
            "VAL-DHARD-002",
            "VAL-DHARD-003",
            "VAL-DHARD-005",
        ],
        "floors": {
            "min_f2p_nodes": DEFAULT_MIN_F2P_NODES,
            "min_f2p_env_keys": list(_MIN_F2P_ENV_KEYS),
            "source_hunk_floor": PRODUCT_SOURCE_HUNK_FLOOR,
            "multi_file_floor": PRODUCT_MULTI_FILE_FLOOR,
            "hybrid_min_source_files": PRODUCT_HYBRID_MIN_SOURCE_FILES,
            "hybrid_min_added_lines": PRODUCT_HYBRID_MIN_ADDED_LINES,
            "min_added_lines": PRODUCT_MIN_ADDED_LINES,
            "min_added_env_keys": list(_MIN_ADDED_ENV_KEYS),
            "band": "deepswe_median_m27",
            "multi_file_rule": "files_ge_4_or_hybrid_3",
        },
        "refuse_reason_codes": {
            REASON_F2P_BELOW_FLOOR: (
                f"F2P node count below MIN_F2P_NODES (default {DEFAULT_MIN_F2P_NODES})"
            ),
            REASON_THIN_F2P_EASY: "thin F2P≈1 fingerprint (easy class)",
            REASON_EMPTY_F2P: "empty F2P node ids",
            REASON_SOURCE_HUNKS_BELOW_FLOOR: f"source hunks < {PRODUCT_SOURCE_HUNK_FLOOR}",
            REASON_MULTI_FILE_FLOOR: (
                f"product sources < {PRODUCT_MULTI_FILE_FLOOR} and hybrid "
                f"(files≥{PRODUCT_HYBRID_MIN_SOURCE_FILES}+added≥{PRODUCT_HYBRID_MIN_ADDED_LINES}"
                f"+hunks≥{PRODUCT_SOURCE_HUNK_FLOOR}) failed"
            ),
            REASON_ADDED_LINES_BELOW_FLOOR: (
                f"gold solution.patch plus-lines < {PRODUCT_MIN_ADDED_LINES}"
            ),
            REASON_SOLVE_ALL_EASY: "panel frontier pass@k=1.0 solve-all dropped",
        },
        "engineering_opt_out": (
            "Explicit engineering_opt_out=True or offline_only dests may skip; "
            "never the product / live_generate default."
        ),
        "notes": [
            "M27 DeepSWE-median: files≥4 OR (files≥3 & added≥500 & hunks≥14); "
            "hunks≥14, added≥400, F2P≥5 otherwise.",
            "packaging-1120 class (3 files, large added) hybrid-admits structural.",
            "qs/thin 2-file packs still refuse.",
            "Thin F2P / thin gold packs refuse product dest by default.",
            "Intrinsic EASY_REQUEST (prompt+gold) may drop via curate (M25/M27).",
            "Model dual-success alone never drops hardness (M25).",
            "Prompt–verifier alignment (M21a) is a separate fail-closed gate.",
            "Agent timeout-class model failures remain harness OK.",
        ],
    }


__all__ = [
    "DEFAULT_MIN_F2P_NODES",
    "PRODUCT_HYBRID_MIN_ADDED_LINES",
    "PRODUCT_HYBRID_MIN_SOURCE_FILES",
    "PRODUCT_MIN_ADDED_LINES",
    "PRODUCT_MULTI_FILE_FLOOR",
    "PRODUCT_SOURCE_HUNK_FLOOR",
    "REASON_ADDED_LINES_BELOW_FLOOR",
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
    "count_gold_added_lines",
    "hardness_result_from_pack_dir",
    "is_hardness_enforced_dest",
    "multi_file_floor_ok",
    "refuse_product_hardness_floors",
    "resolve_min_added_lines",
    "resolve_min_f2p_nodes",
]
