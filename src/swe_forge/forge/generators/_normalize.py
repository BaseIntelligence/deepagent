"""Token-normalized behavior-change check shared by the generators.

A manufactured fault must change program *behavior*; it must never be a
whitespace-, comment-, or import-only edit (VAL-GEN-002). This module collapses
source to its behavior-bearing tokens - dropping blank lines, line comments, and
import statements, and removing all remaining whitespace - so two texts that
differ only in trivia or imports normalize equal.

The single-file round-trip chokepoint and the menu self-check use
:func:`is_behavior_changing` to reject any fault whose normalized original and
mutated forms match, so a non-behavior-changing candidate is never emitted.
"""

from __future__ import annotations

import re

_COMMENT_MARKERS = ("#", "//")

# Import statements across the supported languages: Python ``import x`` /
# ``from x import y``, JS/TS ``import ...`` / ``export ... from '...'``, and Go
# ``import ...``. ``require(...)`` (CommonJS) is matched separately.
_IMPORT_RE = re.compile(
    r"^(import\b|from\b.+\bimport\b|export\b.+\bfrom\b)",
)
_REQUIRE_RE = re.compile(r"\brequire\s*\(")


def _strip_line_comment(line: str) -> str:
    """Cut a trailing ``#``/``//`` line comment (best-effort, marker-based)."""
    for marker in _COMMENT_MARKERS:
        idx = line.find(marker)
        if idx != -1:
            line = line[:idx]
    return line


def _is_import_line(stripped: str) -> bool:
    return bool(_IMPORT_RE.match(stripped) or _REQUIRE_RE.search(stripped))


def normalize_behavior(text: str) -> str:
    """Return ``text`` reduced to its behavior-bearing tokens.

    Drops line comments, blank lines, and import statements, then removes all
    whitespace, so whitespace/comment/import-only differences collapse away.
    """
    tokens: list[str] = []
    for raw in text.splitlines():
        line = _strip_line_comment(raw)
        stripped = line.strip()
        if not stripped or _is_import_line(stripped):
            continue
        tokens.append("".join(stripped.split()))
    return "\n".join(tokens)


def is_behavior_changing(original: str, mutated: str) -> bool:
    """Return ``True`` iff ``original`` and ``mutated`` differ in behavior tokens."""
    return normalize_behavior(original) != normalize_behavior(mutated)
