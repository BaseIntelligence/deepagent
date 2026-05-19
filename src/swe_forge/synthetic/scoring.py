"""Scoring helpers for synthetic tasks."""

from __future__ import annotations


def estimate_patch_complexity(patch: str) -> float:
    """Estimate task complexity from patch size and touched files."""
    changed_lines = 0
    touched_files = 0
    for line in patch.splitlines():
        if line.startswith("diff --git") or line.startswith("--- a/"):
            touched_files += 1
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            changed_lines += 1

    line_score = min(0.7, changed_lines / 80)
    file_score = min(0.3, touched_files / 10)
    return round(max(0.05, line_score + file_score), 3)


def difficulty_label(score: float) -> str:
    """Map a 0..1 complexity score to a coarse difficulty label."""
    if score >= 0.65:
        return "hard"
    if score >= 0.3:
        return "medium"
    return "easy"
