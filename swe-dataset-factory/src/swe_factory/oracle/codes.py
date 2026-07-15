"""Stable oracle gate reason codes (VAL-ORACLE-006).

These codes are the durable audit vocabulary for gate_audit.jsonl. Keep names
stable across versions; add new codes rather than renumbering or renaming.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# G1 — broken/base: every F2P must fail
# ---------------------------------------------------------------------------
G1_F2P_FAIL_OK: Final = "G1_F2P_FAIL_OK"
G1_F2P_NOT_FAILING: Final = "G1_F2P_NOT_FAILING"
G1_EMPTY_F2P: Final = "G1_EMPTY_F2P"
G1_F2P_PRESENT: Final = "G1_F2P_PRESENT"  # stub-only structural

# ---------------------------------------------------------------------------
# G2 — gold dual-run: F2P+P2P pass twice
# ---------------------------------------------------------------------------
G2_GOLD_DUAL_PASS: Final = "G2_GOLD_DUAL_PASS"
G2_GOLD_FAIL: Final = "G2_GOLD_FAIL"
G2_GOLD_PRESENT: Final = "G2_GOLD_PRESENT"  # stub-only
G2_EMPTY_GOLD: Final = "G2_EMPTY_GOLD"

# ---------------------------------------------------------------------------
# Flake (dual-run disagreement) — never export certified
# ---------------------------------------------------------------------------
G2_FLAKE: Final = "G2_FLAKE"
FLAKE_REJECT: Final = "FLAKE_REJECT"

# ---------------------------------------------------------------------------
# G3 — null/empty candidate never resolves
# Prefer distinct codes: eval infrastructure failure vs null patch that resolves.
# ---------------------------------------------------------------------------
G3_NULL_NOT_RESOLVE: Final = "G3_NULL_NOT_RESOLVE"
G3_NULL_RESOLVES: Final = "G3_NULL_RESOLVES"
G3_NULL_EVAL_ERROR: Final = "G3_NULL_EVAL_ERROR"
G3_NON_NULL_GOLD: Final = "G3_NON_NULL_GOLD"  # stub-only
G3_NULL_GOLD: Final = "G3_NULL_GOLD"  # stub-only empty gold

# ---------------------------------------------------------------------------
# G4 — multi-file hard floor
# ---------------------------------------------------------------------------
G4_MULTI_FILE_OK: Final = "G4_MULTI_FILE_OK"
G4_MULTI_FILE: Final = "G4_MULTI_FILE"

# ---------------------------------------------------------------------------
# G5 — agent mount mass leakage (gold path / golden body)
# ---------------------------------------------------------------------------
G5_LEAK_CLEAN: Final = "G5_LEAK_CLEAN"
G5_LEAK: Final = "G5_LEAK"

# ---------------------------------------------------------------------------
# G0 / prompt structural (shared with stub offline path)
# ---------------------------------------------------------------------------
G0_DIGEST_PRESENT: Final = "G0_DIGEST_PRESENT"
G0_MISSING_DIGEST: Final = "G0_MISSING_DIGEST"
G_PROMPT_PRESENT: Final = "G_PROMPT_PRESENT"
G_PROMPT_EMPTY: Final = "G_PROMPT_EMPTY"

# ---------------------------------------------------------------------------
# Aggregate dispositions
# ---------------------------------------------------------------------------
ORACLE_PASS: Final = "ORACLE_PASS"
ORACLE_REJECT: Final = "ORACLE_REJECT"
STUB_PASS: Final = "STUB_PASS"
STUB_REJECT: Final = "STUB_REJECT"

# Codes that always fail closed for certified path
HARD_REJECT_CODES: Final[frozenset[str]] = frozenset(
    {
        G1_F2P_NOT_FAILING,
        G1_EMPTY_F2P,
        G2_GOLD_FAIL,
        G2_EMPTY_GOLD,
        G2_FLAKE,
        FLAKE_REJECT,
        G3_NULL_RESOLVES,
        G3_NULL_EVAL_ERROR,
        G3_NULL_GOLD,
        G4_MULTI_FILE,
        G5_LEAK,
        G_PROMPT_EMPTY,
        G0_MISSING_DIGEST,
        ORACLE_REJECT,
        STUB_REJECT,
    }
)

__all__ = [
    "FLAKE_REJECT",
    "G0_DIGEST_PRESENT",
    "G0_MISSING_DIGEST",
    "G1_EMPTY_F2P",
    "G1_F2P_FAIL_OK",
    "G1_F2P_NOT_FAILING",
    "G1_F2P_PRESENT",
    "G2_EMPTY_GOLD",
    "G2_FLAKE",
    "G2_GOLD_DUAL_PASS",
    "G2_GOLD_FAIL",
    "G2_GOLD_PRESENT",
    "G3_NON_NULL_GOLD",
    "G3_NULL_EVAL_ERROR",
    "G3_NULL_GOLD",
    "G3_NULL_NOT_RESOLVE",
    "G3_NULL_RESOLVES",
    "G4_MULTI_FILE",
    "G4_MULTI_FILE_OK",
    "G5_LEAK",
    "G5_LEAK_CLEAN",
    "G_PROMPT_EMPTY",
    "G_PROMPT_PRESENT",
    "HARD_REJECT_CODES",
    "ORACLE_PASS",
    "ORACLE_REJECT",
    "STUB_PASS",
    "STUB_REJECT",
]
