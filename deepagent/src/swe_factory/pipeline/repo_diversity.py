"""Repo diversity helpers for product curate / generate (VAL-DCOV-003 / M28).

Policy: max **2** certified packs per upstream GitHub repository (normalized
``owner/name``). Prefer higher scores when capping; when scores tie, keep first
seen order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

DEFAULT_MAX_PACKS_PER_REPO = 2


def normalize_upstream_repo(value: str | None) -> str:
    """Normalize a repo string or GitHub URL to lowercase ``owner/name``.

    Returns empty string when the value cannot be parsed as owner/name.
    """
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    cleaned = raw.removesuffix(".git")
    if "://" in cleaned or cleaned.startswith("github.com/"):
        if cleaned.startswith("github.com/"):
            cleaned = "https://" + cleaned
        try:
            parsed = urlparse(cleaned)
            path = (parsed.path or "").strip("/")
        except ValueError:
            path = cleaned
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}".lower()
        return ""
    parts = [p for p in cleaned.split("/") if p]
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}".lower()
    return ""


def _item_repo(item: Mapping[str, Any]) -> str:
    for key in ("repo", "repository", "upstream_repo", "full_name"):
        val = item.get(key)
        norm = normalize_upstream_repo(str(val) if val is not None else "")
        if norm:
            return norm
    for key in ("repository_url", "html_url", "url"):
        val = item.get(key)
        norm = normalize_upstream_repo(str(val) if val is not None else "")
        if norm:
            return norm
    return ""


def _item_pack_id(item: Mapping[str, Any]) -> str:
    for key in ("pack_id", "task_id", "id", "instance_id"):
        val = item.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _item_score(item: Mapping[str, Any], score_key: str | None) -> float:
    if score_key and score_key in item:
        try:
            return float(item[score_key])
        except (TypeError, ValueError):
            return 0.0
    for key in ("score", "hardness_score", "source_hunk_count", "priority"):
        if key in item:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def apply_max_packs_per_repo(
    items: Sequence[Mapping[str, Any]],
    *,
    max_packs: int = DEFAULT_MAX_PACKS_PER_REPO,
    score_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cap items at ``max_packs`` per normalized upstream repo.

    Within a repo, keep the highest-scoring items (stable: original order for
    ties). Returns ``(kept, dropped)`` as plain dict copies.
    """
    if max_packs < 1:
        raise ValueError("max_packs must be >= 1")
    # Group indices by repo preserving input order for score ties via index.
    by_repo: dict[str, list[tuple[float, int, Mapping[str, Any]]]] = {}
    no_repo: list[tuple[int, Mapping[str, Any]]] = []
    for idx, item in enumerate(items):
        repo = _item_repo(item)
        if not repo:
            no_repo.append((idx, item))
            continue
        score = _item_score(item, score_key)
        by_repo.setdefault(repo, []).append((score, idx, item))

    kept: list[tuple[int, dict[str, Any]]] = []
    dropped: list[dict[str, Any]] = []

    for _repo, group in by_repo.items():
        # Highest score first; lower index (earlier) wins ties.
        ordered = sorted(group, key=lambda t: (-t[0], t[1]))
        for i, (_score, idx, item) in enumerate(ordered):
            row = dict(item)
            if i < max_packs:
                kept.append((idx, row))
            else:
                row.setdefault(
                    "diversity_drop_reason",
                    f"max_packs_per_repo>{max_packs}",
                )
                dropped.append(row)

    # Items without parseable repo pass through (caller may enforce separately).
    for idx, item in no_repo:
        kept.append((idx, dict(item)))

    kept_sorted = [row for _, row in sorted(kept, key=lambda t: t[0])]
    return kept_sorted, dropped


def select_diverse_pack_ids(
    pairs: Sequence[tuple[str, str]],
    *,
    max_packs: int = DEFAULT_MAX_PACKS_PER_REPO,
) -> list[str]:
    """Select pack ids from ``(pack_id, repo_or_url)`` pairs with per-repo cap.

    First-seen order within each repo is preserved (no score). Useful for
    generate funnels that walk candidates in priority order.
    """
    counts: Counter[str] = Counter()
    selected: list[str] = []
    for pack_id, repo_raw in pairs:
        pid = str(pack_id).strip()
        if not pid:
            continue
        repo = normalize_upstream_repo(repo_raw)
        if not repo:
            selected.append(pid)
            continue
        if counts[repo] >= max_packs:
            continue
        counts[repo] += 1
        selected.append(pid)
    return selected


def packs_per_repo_histogram(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Count packs per normalized upstream repo."""
    counts: Counter[str] = Counter()
    for item in items:
        repo = _item_repo(item)
        if repo:
            counts[repo] += 1
    return dict(sorted(counts.items()))


__all__ = [
    "DEFAULT_MAX_PACKS_PER_REPO",
    "apply_max_packs_per_repo",
    "normalize_upstream_repo",
    "packs_per_repo_histogram",
    "select_diverse_pack_ids",
]
