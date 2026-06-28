"""Deterministic AST mutation engine shared by every language adapter.

Implements the three ``ast_mutation`` operator families behind one tree-sitter
driver so the per-language adapters stay thin:

* ``operator_swap``  - swap a binary arithmetic/logical/equality operator
  (``+`` <-> ``-``, ``*`` <-> ``/``, ``&&`` <-> ``||``, ``and`` <-> ``or``,
  ``==`` <-> ``!=``).
* ``off_by_one``     - tighten/loosen a comparison boundary (``<`` <-> ``<=``,
  ``>`` <-> ``>=``); when the target has no comparison, bump the first integer
  literal by one.
* ``branch_removal`` - force the first ``if`` condition to the language's false
  literal so its guarded branch becomes dead.

Each mutation is a *surgical* byte replacement of a single token/span inside the
target symbol, so all other bytes are preserved and the resulting patch
round-trips byte-for-byte. The mutated site is always the first applicable one in
source order, making the result deterministic. A target with no applicable site
yields an empty :class:`~swe_forge.forge.adapters.base.Patch` (diff ``""``) so the
caller can move on to another symbol/operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from swe_forge.forge.adapters._diff import make_patch
from swe_forge.forge.adapters.base import (
    MutationOp,
    ParseError,
    Patch,
    PathLike,
    Symbol,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Node

# Binary operator swaps (operator_swap): arithmetic, logical, equality.
_OPERATOR_SWAP: dict[str, str] = {
    "+": "-",
    "-": "+",
    "*": "/",
    "/": "*",
    "&&": "||",
    "||": "&&",
    "and": "or",
    "or": "and",
    "==": "!=",
    "!=": "==",
}
# Comparison boundary swaps (off_by_one).
_COMPARISON_SWAP: dict[str, str] = {
    "<": "<=",
    "<=": "<",
    ">": ">=",
    ">=": ">",
}


@dataclass(frozen=True)
class _LangSpec:
    """Per-language tree-sitter node vocabulary used by the mutation engine."""

    false_literal: str
    int_literal_types: frozenset[str]
    binary_parent_types: frozenset[str]
    if_node_types: frozenset[str]


_SPECS: dict[str, _LangSpec] = {
    "python": _LangSpec(
        false_literal="False",
        int_literal_types=frozenset({"integer"}),
        binary_parent_types=frozenset(
            {"binary_operator", "comparison_operator", "boolean_operator"}
        ),
        if_node_types=frozenset({"if_statement"}),
    ),
    "javascript": _LangSpec(
        false_literal="false",
        int_literal_types=frozenset({"number"}),
        binary_parent_types=frozenset({"binary_expression"}),
        if_node_types=frozenset({"if_statement"}),
    ),
    "go": _LangSpec(
        false_literal="false",
        int_literal_types=frozenset({"int_literal"}),
        binary_parent_types=frozenset({"binary_expression"}),
        if_node_types=frozenset({"if_statement"}),
    ),
}


def _language_for(language: str, path: Path) -> Language:
    """Return the tree-sitter ``Language`` for ``language`` (and ``path``'s extension)."""
    from tree_sitter import Language

    if language == "python":
        import tree_sitter_python as ts_py

        return Language(ts_py.language())
    if language == "go":
        import tree_sitter_go as ts_go

        return Language(ts_go.language())
    if language == "javascript":
        # Reuse the extension-routing used by parse_symbols (.ts/.tsx/.js).
        from swe_forge.forge.adapters._treesitter import _language_for as js_language

        return js_language(path)
    raise ParseError(f"no mutation grammar for language {language!r}")


def _line_offsets(src: bytes) -> list[int]:
    """Return the byte offset at which each 1-based source line starts."""
    offsets = [0]
    for index, byte in enumerate(src):
        if byte == 0x0A:  # newline
            offsets.append(index + 1)
    return offsets


def _span_bytes(src: bytes, symbol: Symbol) -> tuple[int, int]:
    """Return the ``[start, end)`` byte span covering ``symbol``'s line range."""
    offsets = _line_offsets(src)
    start = (
        offsets[symbol.start_line - 1] if symbol.start_line - 1 < len(offsets) else 0
    )
    if symbol.end_line < len(offsets):
        end = offsets[symbol.end_line]
    else:
        end = len(src)
    return start, end


def _walk(node: Node) -> list[Node]:
    """Return every node in ``node``'s subtree (pre-order)."""
    out: list[Node] = []
    stack = [node]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(reversed(current.children))
    return out


def _within(node: Node, start: int, end: int) -> bool:
    return node.start_byte >= start and node.end_byte <= end


def _text(node: Node) -> str:
    return node.text.decode("utf-8", "replace") if node.text is not None else ""


def _binary_token_swap(
    nodes: list[Node], span: tuple[int, int], spec: _LangSpec, mapping: dict[str, str]
) -> tuple[int, int, str] | None:
    """Return the first binary-operator token to swap per ``mapping`` (or ``None``)."""
    start, end = span
    sites = [
        node
        for node in nodes
        if _within(node, start, end)
        and node.child_count == 0
        and not node.is_named
        and node.parent is not None
        and node.parent.type in spec.binary_parent_types
        and _text(node) in mapping
    ]
    sites.sort(key=lambda n: n.start_byte)
    if not sites:
        return None
    token = sites[0]
    return token.start_byte, token.end_byte, mapping[_text(token)]


def _int_increment(
    nodes: list[Node], span: tuple[int, int], spec: _LangSpec
) -> tuple[int, int, str] | None:
    """Return the first decimal integer literal bumped by one (or ``None``)."""
    start, end = span
    for node in sorted(
        (
            n
            for n in nodes
            if _within(n, start, end) and n.type in spec.int_literal_types
        ),
        key=lambda n: n.start_byte,
    ):
        literal = _text(node)
        if literal.isdigit():
            return node.start_byte, node.end_byte, str(int(literal) + 1)
    return None


def _branch_removal(
    nodes: list[Node], span: tuple[int, int], spec: _LangSpec
) -> tuple[int, int, str] | None:
    """Return the edit that forces the first ``if`` condition to the false literal."""
    start, end = span
    ifs = sorted(
        (
            n
            for n in nodes
            if _within(n, start, end) and n.is_named and n.type in spec.if_node_types
        ),
        key=lambda n: n.start_byte,
    )
    for if_node in ifs:
        condition = if_node.child_by_field_name("condition")
        if condition is None:
            continue
        if _text(condition) == spec.false_literal:
            continue
        if condition.type == "parenthesized_expression":
            replacement = f"({spec.false_literal})"
        else:
            replacement = spec.false_literal
        return condition.start_byte, condition.end_byte, replacement
    return None


def _find_site(
    op: MutationOp, nodes: list[Node], span: tuple[int, int], spec: _LangSpec
) -> tuple[int, int, str] | None:
    """Dispatch to the site-finder for ``op`` (``None`` when not applicable)."""
    if op is MutationOp.OPERATOR_SWAP:
        return _binary_token_swap(nodes, span, spec, _OPERATOR_SWAP)
    if op is MutationOp.OFF_BY_ONE:
        site = _binary_token_swap(nodes, span, spec, _COMPARISON_SWAP)
        return site if site is not None else _int_increment(nodes, span, spec)
    if op is MutationOp.BRANCH_REMOVAL:
        return _branch_removal(nodes, span, spec)
    return None


def mutate_source(
    language: str, file: PathLike, symbol: Symbol, op: MutationOp
) -> Patch:
    """Apply ``op`` to ``symbol`` in ``file`` and return the forward mutation patch.

    Returns an empty :class:`Patch` (``diff == ""``) when ``op`` has no applicable
    site inside the symbol. Raises :class:`ParseError` if the source is malformed.
    """
    spec = _SPECS.get(language)
    if spec is None:
        raise ParseError(f"unsupported mutation language {language!r}")

    from tree_sitter import Parser

    path = Path(file)
    src = path.read_bytes()
    parser = Parser(_language_for(language, path))
    tree = parser.parse(src)
    if tree.root_node.has_error:
        raise ParseError(f"failed to parse {language} source {path}: syntax error")

    span = _span_bytes(src, symbol)
    nodes = _walk(tree.root_node)
    site = _find_site(op, nodes, span, spec)
    if site is None:
        return Patch("", ())

    start, end, replacement = site
    modified = src[:start] + replacement.encode("utf-8") + src[end:]
    diff = make_patch(str(file), src, modified)
    return Patch(diff, (str(file),))
