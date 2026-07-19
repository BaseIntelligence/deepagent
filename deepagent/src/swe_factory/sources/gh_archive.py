"""GH Archive bulk-event discover for real_pr candidates (VAL-DCOV-002 / M28).

Reads public GitHub Archive hourly JSON (or offline fixture JSONL / ``.json.gz``)
and emits durable candidate rows with ``discovery_path=gh_archive``.

This path is for **volume discovery without hammering** ``api.github.com/search``.
Materialize / file stats still use the SOCKS-proxied GitHub REST client when live.

Live download of ``https://data.gharchive.org/YYYY-MM-DD-H.json.gz`` is optional
and never required for unit tests. Oxylabs realtime page scrape is **not** used
here (may 401) and must not block densify.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from swe_factory.sources.real_pr_pool import (
    build_candidate_ledger_row,
    write_candidates_jsonl,
)

logger = logging.getLogger(__name__)

DISCOVERY_PATH_GH_ARCHIVE = "gh_archive"

# Stable complement to search|list_pulls product labels.
_EVENT_TYPE_PR = "PullRequestEvent"


def is_merged_pull_request_event(event: Mapping[str, Any]) -> bool:
    """True when event is a closed/merged PullRequestEvent with a merge signal."""
    if str(event.get("type") or "") != _EVENT_TYPE_PR:
        return False
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        return False
    action = str(payload.get("action") or "").lower()
    if action not in {"closed", "merged"}:
        return False
    pr = payload.get("pull_request")
    if not isinstance(pr, Mapping):
        return False
    # Explicit boolean wins (closed-unmerged still emits closed + often null merge sha).
    if pr.get("merged") is True:
        return True
    if pr.get("merged") is False:
        return False
    if pr.get("merged_at"):
        return True
    merge_sha = pr.get("merge_commit_sha")
    # Closed with a non-empty merge commit and no explicit merged=false.
    return isinstance(merge_sha, str) and bool(merge_sha.strip())


def _repo_full_name(event: Mapping[str, Any], pr: Mapping[str, Any]) -> str:
    base = pr.get("base")
    if isinstance(base, Mapping):
        base_repo = base.get("repo")
        if isinstance(base_repo, Mapping):
            full = base_repo.get("full_name")
            if isinstance(full, str) and "/" in full:
                return full.strip()
        # occasional string full_name on base
        full2 = base.get("full_name")
        if isinstance(full2, str) and "/" in full2:
            return full2.strip()
    repo_obj = event.get("repo")
    if isinstance(repo_obj, Mapping):
        name = repo_obj.get("name")
        if isinstance(name, str) and "/" in name:
            return name.strip()
    return ""


def _language_guess(pr: Mapping[str, Any]) -> str:
    base = pr.get("base")
    if isinstance(base, Mapping):
        base_repo = base.get("repo")
        if isinstance(base_repo, Mapping):
            lang = base_repo.get("language")
            if isinstance(lang, str) and lang.strip():
                return lang.strip().lower()
    return ""


def _base_sha(pr: Mapping[str, Any]) -> str:
    base = pr.get("base")
    if isinstance(base, Mapping):
        sha = base.get("sha")
        if isinstance(sha, str):
            return sha.strip()
    return ""


def event_to_candidate_row(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """Map a GH Archive PullRequestEvent to a candidates.jsonl-shaped row."""
    if not is_merged_pull_request_event(event):
        return None
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        return None
    pr = payload.get("pull_request")
    if not isinstance(pr, Mapping):
        return None
    repo = _repo_full_name(event, pr)
    if not repo or "/" not in repo:
        return None
    number_raw = pr.get("number") if pr.get("number") is not None else payload.get("number")
    try:
        number = int(number_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    merge_sha = pr.get("merge_commit_sha")
    merge_commit_sha = (
        str(merge_sha).strip() if isinstance(merge_sha, str) and merge_sha.strip() else None
    )
    base_sha = _base_sha(pr)
    language = _language_guess(pr)
    html_url = pr.get("html_url") if isinstance(pr.get("html_url"), str) else ""
    title = pr.get("title") if isinstance(pr.get("title"), str) else ""

    row = build_candidate_ledger_row(
        repo=repo,
        pr_number=number,
        base_sha=base_sha,
        language=language,
        license="",
        discovery_path=DISCOVERY_PATH_GH_ARCHIVE,
        source_hunk_count=0,
        source_file_count=0,
        test_file_count=0,
        disposition="candidate",
        product_n_evidence=True,
        engineering_only=False,
        extra={
            "merge_commit_sha": merge_commit_sha,
            "html_url": html_url or f"https://github.com/{repo}/pull/{number}",
            "title": title,
            "repository_url": f"https://github.com/{repo}",
            "event_id": str(event.get("id") or ""),
            "event_created_at": str(event.get("created_at") or ""),
        },
    )
    return row


def discover_from_gh_archive_lines(
    lines: Iterable[str],
    *,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Parse newline-delimited GH Archive JSON text into candidate rows.

    Dedupes on ``(repo, pr_number)``. Skips non-JSON / non-merged events silently.
    """
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for raw in lines:
        text = raw.strip() if isinstance(raw, str) else str(raw).strip()
        if not text:
            continue
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        row = event_to_candidate_row(event)
        if row is None:
            continue
        key = (str(row.get("repo") or ""), int(row.get("pr_number") or 0))
        if not key[0] or key[1] <= 0:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if max_candidates is not None and len(out) >= max_candidates:
            break
    return out


@contextmanager
def _open_archive_text(path: Path) -> Iterator[IO[str]]:
    """Open plain JSONL or gzip-compressed GH Archive hourly file as text."""
    p = Path(path)
    if p.suffix == ".gz" or p.name.endswith(".json.gz"):
        with gzip.open(p, mode="rt", encoding="utf-8", errors="replace") as fh:
            yield fh  # gzip text mode yields a text stream
    else:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            yield fh


def discover_from_gh_archive_path(
    path: Path | str,
    *,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Offline-first discover from a local GH Archive file (``.jsonl`` or ``.json.gz``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GH Archive path not found: {p}")
    with _open_archive_text(p) as fh:
        return discover_from_gh_archive_lines(fh, max_candidates=max_candidates)


def write_gh_archive_candidates_jsonl(
    rows: Sequence[Mapping[str, Any]],
    path: Path | str,
) -> Path:
    """Write candidates.jsonl (shared durable ledger writer)."""
    return write_candidates_jsonl(rows, Path(path))


__all__ = [
    "DISCOVERY_PATH_GH_ARCHIVE",
    "discover_from_gh_archive_lines",
    "discover_from_gh_archive_path",
    "event_to_candidate_row",
    "is_merged_pull_request_event",
    "write_gh_archive_candidates_jsonl",
]
