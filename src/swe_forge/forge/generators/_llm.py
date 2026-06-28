"""Shared helpers for the LLM-backed generators (``lm_authored``, ``pr_mirror``).

These generators follow the pipeline's governing principle - *the teacher
proposes, deterministic execution disposes*. The teacher authors a candidate edit
(a subtle single-function bug, or the inversion of a real merged PR); this module
holds the pieces both share: running the async teacher from sync generator code,
splicing a replacement block back into a source file, rejecting bug-signposting,
and recording teacher usage/cost in provenance.

No provider/brand string and no caching live here; the teacher client is built
lazily from :class:`~swe_forge.forge.config.ForgeSettings` only when a generator
actually runs, so building the generator registry never needs credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
from typing import Any, Awaitable, TypeVar

from swe_forge.forge.adapters.base import Symbol
from swe_forge.forge.teacher import LLMResult

_T = TypeVar("_T")

# Comment/marker terms that "signpost" a planted bug. An ``lm_authored`` edit
# carrying any of these is rejected: the fault must be subtle, not advertised.
SIGNPOST_TERMS: tuple[str, ...] = (
    "bug",
    "fixme",
    "todo",
    "hack",
    "xxx",
    "intentional",
    "intentionally",
    "broken",
    "deliberately",
    "on purpose",
    "not implemented",
    "placeholder",
    "vulnerab",
)
_SIGNPOST_RE = re.compile(
    "|".join(re.escape(term) for term in SIGNPOST_TERMS), re.IGNORECASE
)


def contains_signposting(text: str) -> bool:
    """Return ``True`` iff ``text`` contains a bug-signposting term."""
    return bool(_SIGNPOST_RE.search(text))


_FENCE_RE = re.compile(r"^```[^\n]*\n(?P<body>.*)\n```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Remove a single surrounding Markdown code fence, if present."""
    match = _FENCE_RE.match(text.strip())
    return match.group("body") if match else text


def extract_code_field(text: str, preferred_keys: tuple[str, ...]) -> str:
    """Pull a code string from a teacher reply, tolerant of how it is wrapped.

    Endpoints do not always honor the requested ``response_format`` key name, and
    some double-wrap the answer (a JSON string nested inside the requested field).
    This accepts the value under any of ``preferred_keys``, falls back to the sole
    string value of a single-field object, and unwraps nested JSON recursively. A
    surrounding Markdown code fence is always stripped.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    unfenced = _strip_code_fence(raw)
    with contextlib.suppress(json.JSONDecodeError):
        return _resolve_code(json.loads(unfenced), preferred_keys, 0)
    return unfenced


def _resolve_code(value: Any, keys: tuple[str, ...], depth: int) -> str:
    """Recursively resolve a code string from a parsed JSON value."""
    if depth > 4:
        return value if isinstance(value, str) else ""
    if isinstance(value, str):
        text = _strip_code_fence(value.strip())
        if text.startswith("{"):
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(text)
                if _looks_wrapped(parsed, keys):
                    return _resolve_code(parsed, keys, depth + 1)
        return text
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return _resolve_code(candidate, keys, depth + 1)
        string_values = [
            item for item in value.values() if isinstance(item, str) and item.strip()
        ]
        if len(string_values) == 1:
            return _resolve_code(string_values[0], keys, depth + 1)
    return ""


def _looks_wrapped(parsed: Any, keys: tuple[str, ...]) -> bool:
    """Return ``True`` iff ``parsed`` is a JSON object worth unwrapping further."""
    if not isinstance(parsed, dict):
        return False
    if any(isinstance(parsed.get(key), str) for key in keys):
        return True
    string_values = [item for item in parsed.values() if isinstance(item, str)]
    return len(string_values) == 1


def run_sync(coro: Awaitable[_T]) -> _T:
    """Run an async coroutine to completion from synchronous generator code.

    Uses :func:`asyncio.run` when no event loop is running; if one already is
    (e.g. the generator was driven from async code), the coroutine is run on a
    dedicated thread with its own loop so we never re-enter a running loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]

    result: dict[str, Any] = {}

    def _worker() -> None:
        result["value"] = asyncio.run(coro)  # type: ignore[arg-type]

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    return result["value"]


def _trailing_newline(block: bytes) -> bool:
    return block.endswith(b"\n")


def splice_symbol_lines(original: bytes, symbol: Symbol, new_block: str) -> bytes:
    """Replace ``symbol``'s 1-based inclusive line span with ``new_block``.

    Preserves the original block's trailing-newline state so the surrounding
    file is not corrupted (a function that is the last line of a file without a
    final newline stays that way). Only the symbol's own lines change, keeping
    the edit single-function by construction.
    """
    text = original.decode("utf-8")
    lines = text.splitlines(keepends=True)
    start = max(symbol.start_line - 1, 0)
    end = min(symbol.end_line, len(lines))
    if start >= end:
        raise ValueError("symbol line span is empty or out of range")

    original_block = "".join(lines[start:end]).encode("utf-8")
    replacement = new_block
    if _trailing_newline(original_block):
        if not replacement.endswith("\n"):
            replacement += "\n"
    else:
        replacement = replacement.rstrip("\n")

    spliced = "".join(lines[:start]) + replacement + "".join(lines[end:])
    return spliced.encode("utf-8")


def strip_trivia(text: str) -> str:
    """Return ``text`` with ALL whitespace and ``#``/``//`` line comments removed.

    Collapses to bare tokens, so two blocks that differ only in spacing/comments
    compare equal. Used by ``pr_mirror`` to confirm a teacher reversion matches
    the deterministic pre-PR content at the token level (the shipped patch is the
    deterministic inverse regardless).
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw
        for marker in ("#", "//"):
            idx = line.find(marker)
            if idx != -1:
                line = line[:idx]
        out.append("".join(line.split()))
    return "".join(out)


def normalize_code(text: str) -> str:
    """Return ``text`` with comments, trailing whitespace, and blank lines removed.

    Unlike :func:`strip_trivia`, LEADING INDENTATION and internal spacing are
    preserved, so a Python re-indentation (which changes control flow) is seen as
    a real change. Used by ``lm_authored`` to reject only comment/blank/trailing-
    whitespace churn while keeping genuine single-function edits.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw
        for marker in ("#", "//"):
            idx = line.find(marker)
            if idx != -1:
                line = line[:idx]
        line = line.rstrip()
        if not line.strip():
            continue
        out.append(line)
    return "\n".join(out)


def teacher_usage_details(result: LLMResult, model: str) -> dict[str, object]:
    """Build the provenance ``teacher`` record (model + usage + cost, no secrets)."""
    return {
        "model": model,
        "usage": result.usage.to_dict(),
        "cost": result.cost,
    }


def merge_usage(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate several teacher usage records into a single total."""
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cost = 0.0
    model = ""
    for record in records:
        usage = record.get("usage")
        if isinstance(usage, dict):
            for key in total:
                value = usage.get(key, 0)
                if isinstance(value, (int, float)):
                    total[key] += int(value)
        rec_cost = record.get("cost", 0.0)
        if isinstance(rec_cost, (int, float)):
            cost += float(rec_cost)
        if not model:
            model = str(record.get("model", ""))
    return {"model": model, "usage": total, "cost": cost, "calls": len(records)}
