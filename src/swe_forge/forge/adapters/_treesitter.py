"""tree-sitter symbol extraction shared by the JS/TS adapter.

Parses a single JavaScript or TypeScript file with the grammar matching its
extension (``.ts``/``.mts``/``.cts`` -> TypeScript, ``.tsx`` -> TSX, everything
else -> JavaScript, which also accepts JSX) and returns the file's
function/method declarations as :class:`~swe_forge.forge.adapters.base.Symbol`
objects with 1-based inclusive line spans.

What counts as a symbol: top-level and nested function declarations, generator
functions, and arrow/function expressions bound to a ``const``/``let``/``var``
declarator or a class field (``"function"``), plus class methods, getters,
setters, generators, and the constructor (``"method"``). Object-literal
shorthand methods are intentionally excluded so the returned set matches the
real declarations a caller would mutate. Malformed source (the parse tree
reports an error) raises :class:`ParseError` rather than yielding partial junk.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from swe_forge.forge.adapters.base import ParseError, Symbol

if TYPE_CHECKING:
    from tree_sitter import Language, Node

# Extensions handled by the TypeScript grammars; everything else uses JavaScript.
_TS_EXTENSIONS = frozenset({".ts", ".mts", ".cts"})
_TSX_EXTENSIONS = frozenset({".tsx"})

_FUNCTION_DECL_TYPES = frozenset(
    {"function_declaration", "generator_function_declaration"}
)
_FUNCTION_VALUE_TYPES = frozenset(
    {"arrow_function", "function_expression", "function", "generator_function"}
)


def _language_for(path: Path) -> Language:
    """Return the tree-sitter ``Language`` matching ``path``'s extension."""
    from tree_sitter import Language

    suffix = path.suffix.lower()
    if suffix in _TSX_EXTENSIONS:
        import tree_sitter_typescript as ts_ts

        return Language(ts_ts.language_tsx())
    if suffix in _TS_EXTENSIONS:
        import tree_sitter_typescript as ts_ts

        return Language(ts_ts.language_typescript())
    import tree_sitter_javascript as ts_js

    return Language(ts_js.language())


def _line_span(node: Node) -> tuple[int, int]:
    """Return the 1-based inclusive ``(start_line, end_line)`` of ``node``."""
    return node.start_point[0] + 1, node.end_point[0] + 1


def _signature(node: Node, source: bytes, *, body_owner: Node | None = None) -> str:
    """Render the declaration header (up to the body) on a single line.

    ``node`` provides the start of the header; ``body_owner`` (when the value
    holding the body differs from ``node``, e.g. a declarator bound to an arrow
    function) provides where the body begins so the rendered signature stops
    before it.
    """
    owner = body_owner if body_owner is not None else node
    body = owner.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    if end < node.start_byte:
        end = node.end_byte
    text = source[node.start_byte : end].decode("utf-8", "replace")
    return " ".join(text.split())


def _name_of(node: Node) -> str | None:
    """Return the declared name via the grammar's ``name`` field, if any."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return name_node.text.decode("utf-8", "replace") if name_node.text else None


class _Collector:
    """Walk the parse tree and accumulate function/method symbols."""

    def __init__(self, file: str, source: bytes) -> None:
        self.file = file
        self.source = source
        self.symbols: list[Symbol] = []

    def _emit(self, name: str, kind: str, anchor: Node, header: Node) -> None:
        start, end = _line_span(anchor)
        self.symbols.append(
            Symbol(
                name=name,
                kind=kind,
                file=self.file,
                start_line=start,
                end_line=end,
                signature=_signature(header, self.source, body_owner=anchor),
            )
        )

    def visit(self, node: Node) -> None:
        handler: Callable[[Node], None] | None = None
        if node.type in _FUNCTION_DECL_TYPES:
            handler = self._function_declaration
        elif node.type == "method_definition":
            handler = self._method_definition
        elif node.type in ("variable_declarator", "public_field_definition"):
            handler = self._bound_value
        if handler is not None:
            handler(node)
        for child in node.children:
            self.visit(child)

    def _function_declaration(self, node: Node) -> None:
        name = _name_of(node)
        if name:
            self._emit(name, "function", node, node)

    def _method_definition(self, node: Node) -> None:
        # method_definition appears both in class bodies and object literals;
        # only class members are real methods we can locate/mutate.
        if node.parent is None or node.parent.type != "class_body":
            return
        name = _name_of(node)
        if name:
            self._emit(name, "method", node, node)

    def _bound_value(self, node: Node) -> None:
        value = node.child_by_field_name("value")
        if value is None or value.type not in _FUNCTION_VALUE_TYPES:
            return
        name = _name_of(node)
        if not name:
            return
        # A function bound to a class field is a method; anywhere else (const/
        # let/var) it is a function.
        in_class = node.type == "public_field_definition" or (
            node.parent is not None and node.parent.type == "field_definition"
        )
        kind = "method" if in_class else "function"
        self._emit(name, kind, value, node)


def parse_js_symbols(file: str | Path) -> list[Symbol]:
    """Parse a JS/TS file and return its function/method symbols.

    Raises :class:`ParseError` if the source is malformed (the parser reports an
    error node), so callers surface a clean parse error instead of crashing.
    """
    from tree_sitter import Parser

    path = Path(file)
    source = path.read_bytes()
    language = _language_for(path)
    parser = Parser(language)
    tree = parser.parse(source)
    if tree.root_node.has_error:
        raise ParseError(
            f"failed to parse {path.suffix or 'JS/TS'} source {path}: "
            "syntax error in source"
        )
    collector = _Collector(str(path), source)
    collector.visit(tree.root_node)
    return collector.symbols
