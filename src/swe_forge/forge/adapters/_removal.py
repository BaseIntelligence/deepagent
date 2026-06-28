"""Function/feature body removal shared by every language adapter.

``function_removal`` deletes a function or method *body* while keeping its
signature byte-for-byte, then the gold oracle re-inserts the original body
exactly. This module performs the body excision behind one tree-sitter driver so
the generator stays language-agnostic: it locates the target function node (by
the symbol's start line), replaces its body with a language-appropriate stub that
keeps the file parseable, and returns the forward patch. The inverse (gold) patch
is the caller's ``make_patch(mutated -> original)``, so the round-trip restores
the original byte-for-byte by construction.

The kept signature is everything before the body node; for Python the body
``block`` excludes the ``def ...:`` header (which carries the trailing newline and
indentation), while for Go/JS the body ``block``/``statement_block`` includes its
braces, so the stub supplies its own braces there.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from swe_forge.forge.adapters._diff import make_patch
from swe_forge.forge.adapters._mutate import _language_for
from swe_forge.forge.adapters.base import ParseError, Patch, PathLike, Symbol

if TYPE_CHECKING:
    from tree_sitter import Node

# Function/method node types that own a removable body, per language.
_FUNCTION_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition"}),
    "javascript": frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
            "arrow_function",
            "function_expression",
            "function",
            "generator_function",
        }
    ),
    "go": frozenset({"function_declaration", "method_declaration"}),
}

# The grammar node type of the body to excise, per language.
_BODY_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"block"}),
    "javascript": frozenset({"statement_block"}),
    "go": frozenset({"block"}),
}

# The stub the body is replaced with. Python relies on the kept header providing
# the indentation; Go/JS bodies include braces so the stub provides its own.
_STUB_BODY: dict[str, bytes] = {
    "python": b"raise NotImplementedError",
    "javascript": b'{\n  throw new Error("not implemented");\n}',
    "go": b'{\n\tpanic("not implemented")\n}',
}

# Bodies whose meaningful content is one of these are already a stub/no-op, so
# removing them would not change behavior; such symbols are skipped.
_TRIVIAL_BODY_TEXT: frozenset[str] = frozenset(
    {"", "pass", "...", "raise NotImplementedError", 'panic("not implemented")'}
)


class RemovalError(ParseError):
    """Raised when a symbol's body cannot be located or removed."""


def _walk(node: Node) -> list[Node]:
    """Return every node in ``node``'s subtree (pre-order)."""
    out: list[Node] = []
    stack = [node]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(reversed(current.children))
    return out


def _find_function_node(
    nodes: list[Node], func_types: frozenset[str], symbol: Symbol
) -> Node | None:
    """Return the function node whose header starts on ``symbol.start_line``.

    Prefers the outermost (largest-span) function node beginning on that line so
    a nested helper sharing the line never shadows the targeted definition.
    """
    matches = [
        node
        for node in nodes
        if node.type in func_types and node.start_point[0] + 1 == symbol.start_line
    ]
    if not matches:
        return None
    matches.sort(key=lambda n: n.end_byte - n.start_byte, reverse=True)
    return matches[0]


def _meaningful_body_text(language: str, body_text: str) -> str:
    """Return the body's inner text for the triviality check (braces stripped)."""
    stripped = body_text.strip()
    if (
        language in ("go", "javascript")
        and stripped.startswith("{")
        and stripped.endswith("}")
    ):
        return stripped[1:-1].strip()
    return stripped


def remove_body(language: str, file: PathLike, symbol: Symbol) -> Patch:
    """Remove ``symbol``'s body (keeping its signature) and return the forward patch.

    Returns an empty :class:`Patch` (``diff == ""``) when the symbol has no
    removable/meaningful body or the stub equals the original. Raises
    :class:`ParseError` if the source (or the stubbed result) does not parse.
    """
    func_types = _FUNCTION_NODE_TYPES.get(language)
    body_types = _BODY_NODE_TYPES.get(language)
    stub = _STUB_BODY.get(language)
    if func_types is None or body_types is None or stub is None:
        raise RemovalError(f"unsupported removal language {language!r}")

    from tree_sitter import Parser

    path = Path(file)
    src = path.read_bytes()
    parser = Parser(_language_for(language, path))
    tree = parser.parse(src)
    if tree.root_node.has_error:
        raise ParseError(f"failed to parse {language} source {path}: syntax error")

    node = _find_function_node(_walk(tree.root_node), func_types, symbol)
    if node is None:
        return Patch("", ())
    body = node.child_by_field_name("body")
    if body is None or body.type not in body_types:
        return Patch("", ())

    body_text = src[body.start_byte : body.end_byte].decode("utf-8", "replace")
    if _meaningful_body_text(language, body_text) in _TRIVIAL_BODY_TEXT:
        return Patch("", ())

    modified = src[: body.start_byte] + stub + src[body.end_byte :]
    if modified == src:
        return Patch("", ())

    # The stubbed result must still parse, so the broken state is a clean
    # body-removed function rather than a syntax error.
    if parser.parse(modified).root_node.has_error:
        return Patch("", ())

    diff = make_patch(str(file), src, modified)
    if not diff.strip():
        return Patch("", ())
    return Patch(diff, (str(file),))
