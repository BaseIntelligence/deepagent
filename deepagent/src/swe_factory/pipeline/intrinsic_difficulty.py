"""Intrinsic request+patch difficulty scoring (M25 / VAL-DINTR-002).

Score hardness from **agent-visible instruction** + **gold solution patch**
(+ optional F2P count / source file list) only. Model dual-success is **not**
an input — pair with :mod:`easy_detect` labels for scoreboard reporting without
implicit ``should_drop_hardness``.

Classes
-------
* ``EASY_REQUEST`` — short/thin contract request, tiny multi-file gold, low F2P,
  single-module touch, few constraints/outcomes. ``easily_approachable=True``.
* ``HARD_REQUEST`` — multi-module / large patch / multi-constraint agenda /
  large F2P. ``easily_approachable=False``.
* ``UNCERTAIN`` — mixed signals; default **keep** for hardness product
  (only high-confidence ``EASY_REQUEST`` is an optional drop reason).

Drop policy (curate): never model outcomes alone; only misalign, structural
floors, and high-confidence ``EASY_REQUEST``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.producers.hard_filter import count_unified_diff_hunks

INTRINSIC_SCHEMA: str = "deepagent.intrinsic_difficulty.v1"

CLASS_EASY_REQUEST: str = "EASY_REQUEST"
CLASS_HARD_REQUEST: str = "HARD_REQUEST"
CLASS_UNCERTAIN: str = "UNCERTAIN"

REASON_EASY_REQUEST: str = "intrinsic_easy_request"
REASON_HARD_REQUEST: str = "intrinsic_hard_request"
REASON_UNCERTAIN: str = "intrinsic_uncertain"

# Thresholds (unit-tested). Tuned for product "thin contract vs multi-outcome"
# discrimination without needing model scores.
EASY_INSTR_CHARS_MAX: int = 700
HARD_INSTR_CHARS_MIN: int = 1200
EASY_OUTCOME_MAX: int = 2
HARD_OUTCOME_MIN: int = 4
EASY_CONSTRAINT_MAX: int = 1
HARD_CONSTRAINT_MIN: int = 3
EASY_HUNK_MAX: int = 4
HARD_HUNK_MIN: int = 10
EASY_SOURCE_FILES_MAX: int = 2
HARD_SOURCE_FILES_MIN: int = 3
EASY_F2P_MAX: int = 2
HARD_F2P_MIN: int = 4
EASY_ADDED_LINES_MAX: int = 25
HARD_ADDED_LINES_MIN: int = 60
#: Require this many easy / hard structural signals for confident class.
EASY_SIGNAL_CONFIDENCE: int = 3
HARD_SIGNAL_CONFIDENCE: int = 3

_OUTCOME_LINE_RE = re.compile(
    r"(?im)^\s*(?:#{1,4}\s*)?(?:expected outcomes?|requirements?|acceptance|"
    r"deliverables?)\s*:?\s*$"
)
_CONSTRAINT_LINE_RE = re.compile(
    r"(?im)^\s*(?:#{1,4}\s*)?(?:constraints?|limits?|do not|must not|important)\b"
)
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*(?:\d+[\.\)]\s+|[-*]\s+)")
_DIFF_GIT_RE = re.compile(r"(?m)^diff --git a/(.+?) b/")
_PLUS_LINE_RE = re.compile(r"(?m)^\+(?!\+\+ )")
_MINUS_LINE_RE = re.compile(r"(?m)^-(?!-- )")


@dataclass(frozen=True, slots=True)
class IntrinsicDifficultyResult:
    """Prompt+gold structural difficulty score (no model outcomes)."""

    intrinsic_class: str
    easily_approachable: bool
    confidence: str  # high | medium | low
    should_drop_hardness: bool
    reason_code: str
    detail: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intrinsic_class": self.intrinsic_class,
            "easily_approachable": self.easily_approachable,
            "confidence": self.confidence,
            "should_drop_hardness": self.should_drop_hardness,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "meta": dict(self.meta),
            "schema": INTRINSIC_SCHEMA,
        }


def _count_section_items(instruction: str, header_re: re.Pattern[str]) -> int:
    """Count numbered/bulleted items under a matching header section."""
    lines = instruction.splitlines()
    count = 0
    in_section = False
    for line in lines:
        if header_re.search(line):
            in_section = True
            continue
        if in_section:
            # Stop at next markdown header of similar weight.
            if re.match(r"^\s*#{1,4}\s+\S", line) and not header_re.search(line):
                break
            if _NUMBERED_ITEM_RE.match(line):
                count += 1
            elif line.strip() == "":
                continue
            elif (
                count > 0
                and not line.lstrip().startswith(("-", "*", "#"))
                and not re.match(r"^\s*\d+[\.\)]", line)
            ):
                # Non-list prose ends a tight section when we already saw items.
                break
    return count


def _estimate_outcomes(instruction: str) -> int:
    n = _count_section_items(instruction, _OUTCOME_LINE_RE)
    if n > 0:
        return n
    # Fallback: top-level numbered requirements anywhere.
    return sum(1 for line in instruction.splitlines() if re.match(r"^\s*\d+[\.\)]\s+\S", line))


def _estimate_constraints(instruction: str) -> int:
    n = _count_section_items(instruction, _CONSTRAINT_LINE_RE)
    if n > 0:
        return n
    # Constraint-ish imperative bullets / sentences.
    hits = 0
    for line in instruction.splitlines():
        low = line.lower()
        if any(
            k in low
            for k in (
                "do not",
                "must not",
                "don't",
                "only touch",
                "keep public",
                "without breaking",
                "preserve",
                "must remain",
            )
        ):
            hits += 1
    return hits


def _source_files_from_patch(solution_patch: str) -> list[str]:
    files = _DIFF_GIT_RE.findall(solution_patch or "")
    # Prefer +++ b/ paths when present without diff --git (rare short patches).
    if not files:
        for line in (solution_patch or "").splitlines():
            if line.startswith("+++ b/") and not line.startswith("+++ b/dev/null"):
                files.append(line[6:].strip())
    # De-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        f = f.strip()
        if not f or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _module_roots(paths: Sequence[str]) -> set[str]:
    roots: set[str] = set()
    for p in paths:
        parts = Path(p).parts
        if not parts:
            continue
        # Skip empty / single-file at root — still count as repo-root module bucket.
        if len(parts) == 1:
            roots.add(parts[0].rsplit(".", 1)[0] or parts[0])
        else:
            roots.add(parts[0])
    return roots


def score_request_patch_difficulty(
    instruction: str | None,
    solution_patch: str | None,
    f2p_count: int | None = None,
    source_files: Sequence[str] | None = None,
    *,
    drop_on_easy_request: bool = True,
    high_confidence_only: bool = True,
) -> IntrinsicDifficultyResult:
    """Score whether the request+gold looks immediately approachable.

    Pure structural harmonic from prompt + patch (+F2P). Does **not** read
    model pass@k.

    Returns ``IntrinsicDifficultyResult`` with:
    * ``intrinsic_class``: EASY_REQUEST | HARD_REQUEST | UNCERTAIN
    * ``easily_approachable``: True only for EASY_REQUEST
    * ``should_drop_hardness``: True only for high-confidence EASY_REQUEST when
      *drop_on_easy_request* is True (never from model scores here)
    """
    instr = (instruction or "").strip()
    patch = (solution_patch or "").strip()
    files = list(source_files) if source_files is not None else _source_files_from_patch(patch)
    hunks = count_unified_diff_hunks(patch)
    added = len(_PLUS_LINE_RE.findall(patch))
    deleted = len(_MINUS_LINE_RE.findall(patch))
    instr_len = len(instr)
    outcomes = _estimate_outcomes(instr) if instr else 0
    constraints = _estimate_constraints(instr) if instr else 0
    n_files = len(files)
    modules = _module_roots(files)
    n_modules = len(modules)
    f2p = int(f2p_count) if f2p_count is not None else None

    easy_reasons: list[str] = []
    hard_reasons: list[str] = []

    if instr_len == 0:
        easy_reasons.append("empty_instruction")
    elif instr_len <= EASY_INSTR_CHARS_MAX:
        easy_reasons.append(f"short_instruction_chars={instr_len}<={EASY_INSTR_CHARS_MAX}")
    elif instr_len >= HARD_INSTR_CHARS_MIN:
        hard_reasons.append(f"long_instruction_chars={instr_len}>={HARD_INSTR_CHARS_MIN}")

    if outcomes <= EASY_OUTCOME_MAX:
        easy_reasons.append(f"few_outcomes={outcomes}<={EASY_OUTCOME_MAX}")
    if outcomes >= HARD_OUTCOME_MIN:
        hard_reasons.append(f"many_outcomes={outcomes}>={HARD_OUTCOME_MIN}")

    if constraints <= EASY_CONSTRAINT_MAX:
        easy_reasons.append(f"few_constraints={constraints}<={EASY_CONSTRAINT_MAX}")
    if constraints >= HARD_CONSTRAINT_MIN:
        hard_reasons.append(f"many_constraints={constraints}>={HARD_CONSTRAINT_MIN}")

    if hunks <= EASY_HUNK_MAX:
        easy_reasons.append(f"tiny_gold_hunks={hunks}<={EASY_HUNK_MAX}")
    if hunks >= HARD_HUNK_MIN:
        hard_reasons.append(f"large_gold_hunks={hunks}>={HARD_HUNK_MIN}")

    if n_files <= EASY_SOURCE_FILES_MAX and n_modules <= 1:
        easy_reasons.append(f"single_module_or_thin_files=files:{n_files}/modules:{n_modules}")
    if n_files >= HARD_SOURCE_FILES_MIN or n_modules >= HARD_SOURCE_FILES_MIN:
        hard_reasons.append(f"multi_module_touch=files:{n_files}/modules:{n_modules}")

    if f2p is not None:
        if f2p <= EASY_F2P_MAX:
            easy_reasons.append(f"low_f2p={f2p}<={EASY_F2P_MAX}")
        if f2p >= HARD_F2P_MIN:
            hard_reasons.append(f"large_f2p={f2p}>={HARD_F2P_MIN}")

    if added <= EASY_ADDED_LINES_MAX and deleted <= EASY_ADDED_LINES_MAX:
        easy_reasons.append(f"tiny_delta_lines=+{added}/-{deleted}")
    if added >= HARD_ADDED_LINES_MIN or deleted >= HARD_ADDED_LINES_MIN:
        hard_reasons.append(f"large_delta_lines=+{added}/-{deleted}")

    # empty / missing gold is a thin-request smell only when instruction also thin
    if not patch:
        easy_reasons.append("empty_solution_patch")

    n_easy = len(easy_reasons)
    n_hard = len(hard_reasons)
    metrics: dict[str, Any] = {
        "instruction_chars": instr_len,
        "outcomes": outcomes,
        "constraints": constraints,
        "source_file_count": n_files,
        "module_count": n_modules,
        "modules": sorted(modules),
        "hunk_count": hunks,
        "added_lines": added,
        "deleted_lines": deleted,
        "f2p_count": f2p,
        "easy_signal_count": n_easy,
        "hard_signal_count": n_hard,
    }

    # Classification: hard beats easy on equal-ish when both high (prefer keep).
    if n_hard >= HARD_SIGNAL_CONFIDENCE and n_hard > n_easy:
        cls = CLASS_HARD_REQUEST
        conf = "high" if n_hard >= HARD_SIGNAL_CONFIDENCE + 1 else "medium"
        detail = (
            f"HARD_REQUEST (confidence={conf}): multi-outcome / multi-module / large gold "
            f"signals={n_hard} dominate easy_signals={n_easy}"
        )
        return IntrinsicDifficultyResult(
            intrinsic_class=cls,
            easily_approachable=False,
            confidence=conf,
            should_drop_hardness=False,
            reason_code=REASON_HARD_REQUEST,
            detail=detail,
            reasons=tuple(hard_reasons),
            metrics=metrics,
            meta={"schema": INTRINSIC_SCHEMA, "drop_gate": "model_independent"},
        )

    if n_easy >= EASY_SIGNAL_CONFIDENCE and n_easy > n_hard:
        cls = CLASS_EASY_REQUEST
        conf = "high" if n_easy >= EASY_SIGNAL_CONFIDENCE + 1 and n_hard == 0 else "medium"
        if n_easy >= EASY_SIGNAL_CONFIDENCE + 1 and n_hard <= 1:
            conf = "high"
        approachable = True
        drop = bool(drop_on_easy_request) and (conf == "high" if high_confidence_only else True)
        # Redeem if strong hard signals coexist — never auto-easy then.
        if n_hard >= HARD_SIGNAL_CONFIDENCE:
            cls = CLASS_UNCERTAIN
            approachable = False
            drop = False
            conf = "medium"
            detail = (
                f"UNCERTAIN: easy_signals={n_easy} co-present with hard_signals={n_hard}; "
                "default keep for hardness product"
            )
            return IntrinsicDifficultyResult(
                intrinsic_class=cls,
                easily_approachable=False,
                confidence=conf,
                should_drop_hardness=False,
                reason_code=REASON_UNCERTAIN,
                detail=detail,
                reasons=tuple(easy_reasons + hard_reasons),
                metrics=metrics,
                meta={"schema": INTRINSIC_SCHEMA, "drop_gate": "model_independent"},
            )
        detail = (
            f"EASY_REQUEST (confidence={conf}): thin contract / tiny gold "
            f"signals={n_easy} hard_signals={n_hard}; "
            f"should_drop_hardness={drop} (intrinsic only, never model scores)"
        )
        return IntrinsicDifficultyResult(
            intrinsic_class=cls,
            easily_approachable=approachable,
            confidence=conf,
            should_drop_hardness=drop,
            reason_code=REASON_EASY_REQUEST,
            detail=detail,
            reasons=tuple(easy_reasons),
            metrics=metrics,
            meta={
                "schema": INTRINSIC_SCHEMA,
                "drop_gate": "intrinsic_easy_high_confidence" if drop else "label_only_no_drop",
            },
        )

    # Mixed / low signal → UNCERTAIN keep
    conf = "low" if (n_easy + n_hard) < 2 else "medium"
    detail = (
        f"UNCERTAIN (confidence={conf}): easy_signals={n_easy} hard_signals={n_hard}; "
        "keep unless other gates refuse (misalign / floors)"
    )
    return IntrinsicDifficultyResult(
        intrinsic_class=CLASS_UNCERTAIN,
        easily_approachable=False,
        confidence=conf,
        should_drop_hardness=False,
        reason_code=REASON_UNCERTAIN,
        detail=detail,
        reasons=tuple(easy_reasons + hard_reasons),
        metrics=metrics,
        meta={"schema": INTRINSIC_SCHEMA, "drop_gate": "model_independent"},
    )


def intrinsic_from_pack_dir(
    pack_dir: Path | str,
    *,
    f2p_count: int | None = None,
    drop_on_easy_request: bool = True,
    high_confidence_only: bool = True,
) -> IntrinsicDifficultyResult:
    """Load instruction.md + solution/solution.patch (+ config F2P) from a pack."""
    root = Path(pack_dir)
    instr = ""
    ip = root / "instruction.md"
    if ip.is_file():
        instr = ip.read_text(encoding="utf-8", errors="replace")
    patch = ""
    sp = root / "solution" / "solution.patch"
    if sp.is_file():
        patch = sp.read_text(encoding="utf-8", errors="replace")
    f2p = f2p_count
    if f2p is None:
        cfg = root / "tests" / "config.json"
        if cfg.is_file():
            import json

            try:
                blob = json.loads(cfg.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                blob = {}
            if isinstance(blob, dict):
                nodes = blob.get("f2p_node_ids") or blob.get("fail_to_pass") or []
                if isinstance(nodes, list):
                    f2p = len(nodes)
    return score_request_patch_difficulty(
        instr,
        patch,
        f2p_count=f2p,
        drop_on_easy_request=drop_on_easy_request,
        high_confidence_only=high_confidence_only,
    )


__all__ = [
    "CLASS_EASY_REQUEST",
    "CLASS_HARD_REQUEST",
    "CLASS_UNCERTAIN",
    "INTRINSIC_SCHEMA",
    "REASON_EASY_REQUEST",
    "REASON_HARD_REQUEST",
    "REASON_UNCERTAIN",
    "IntrinsicDifficultyResult",
    "intrinsic_from_pack_dir",
    "score_request_patch_difficulty",
]
