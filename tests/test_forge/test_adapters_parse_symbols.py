"""Unit tests for adapter ``parse_symbols`` across Python, JS/TS, and Go.

Covers the m2-ast contract (VAL-ENV-009..011): each adapter returns exactly the
ground-truth function/method declarations (none hallucinated or missed), every
``Symbol`` carries locating metadata (name + file + line span), a trivial file
yields an empty list, and malformed source raises a clean
:class:`ParseError` (no crash). The ``swe-forge forge parse-symbols`` CLI surface
is exercised too. Go tests require the host ``go`` toolchain and skip when it is
unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_forge.forge.adapters import (
    GoAdapter,
    JavaScriptAdapter,
    ParseError,
    PythonAdapter,
    Symbol,
)
from swe_forge.forge.adapters._goast import GoToolchainError, parse_go_symbols
from swe_forge.forge.cli import app as forge_app

runner = CliRunner()


def _go_available() -> bool:
    try:
        from swe_forge.forge.adapters._goast import _find_go

        _find_go()
        return True
    except GoToolchainError:
        return False


requires_go = pytest.mark.skipif(
    not _go_available(), reason="the Go toolchain is not available on this host"
)


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _names(symbols: list[Symbol]) -> set[str]:
    return {s.name for s in symbols}


def _by_name(symbols: list[Symbol]) -> dict[str, Symbol]:
    return {s.name: s for s in symbols}


# --------------------------------------------------------------------------- #
# Python (VAL-ENV-009/010/011)
# --------------------------------------------------------------------------- #
PY_SAMPLE = '''\
"""module docstring"""
import os

CONST = 1


def add(a, b):
    return a + b


async def fetch(url):
    return url


class Calculator:
    def multiply(self, a, b):
        return a * b

    @staticmethod
    def helper(x):
        return x
'''


class TestPython:
    def test_returns_exact_ground_truth(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.py", PY_SAMPLE)
        symbols = PythonAdapter().parse_symbols(f)
        assert _names(symbols) == {"add", "fetch", "multiply", "helper"}

    def test_kinds_classified(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.py", PY_SAMPLE)
        by = _by_name(PythonAdapter().parse_symbols(f))
        assert by["add"].kind == "function"
        assert by["fetch"].kind == "function"
        assert by["multiply"].kind == "method"
        assert by["helper"].kind == "method"

    def test_locating_metadata_points_at_definition(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.py", PY_SAMPLE)
        by = _by_name(PythonAdapter().parse_symbols(f))
        add = by["add"]
        assert add.file == str(f)
        assert add.start_line == 7
        assert add.end_line == 8
        # The recorded start line is the real `def` line in the source.
        lines = PY_SAMPLE.splitlines()
        assert lines[add.start_line - 1].lstrip().startswith("def add")

    def test_signature_recorded(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.py", PY_SAMPLE)
        by = _by_name(PythonAdapter().parse_symbols(f))
        assert by["fetch"].signature == "async def fetch(url)"
        assert by["multiply"].signature == "def multiply(self, a, b)"

    def test_nested_function_included_as_function(self, tmp_path: Path) -> None:
        src = "def outer():\n    def inner():\n        return 1\n    return inner\n"
        f = _write(tmp_path, "nested.py", src)
        by = _by_name(PythonAdapter().parse_symbols(f))
        assert set(by) == {"outer", "inner"}
        assert by["inner"].kind == "function"

    def test_decorated_span_excludes_decorator(self, tmp_path: Path) -> None:
        src = "import functools\n\n\n@functools.cache\ndef cached(x):\n    return x\n"
        f = _write(tmp_path, "dec.py", src)
        by = _by_name(PythonAdapter().parse_symbols(f))
        assert by["cached"].start_line == 5

    def test_empty_file_yields_empty_list(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "empty.py", "# just a comment\nX = 1\n")
        assert PythonAdapter().parse_symbols(f) == []

    def test_malformed_raises_parse_error(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "bad.py", "def broken(:\n    pass\n")
        with pytest.raises(ParseError) as exc:
            PythonAdapter().parse_symbols(f)
        assert "bad.py" in str(exc.value)


# --------------------------------------------------------------------------- #
# JavaScript (VAL-ENV-009/010/011)
# --------------------------------------------------------------------------- #
JS_SAMPLE = """\
function add(a, b) { return a + b; }
const multiply = (a, b) => a * b;
const sub = function (a, b) { return a - b; };
function* counter() { yield 1; }
class Animal {
  constructor(name) { this.name = name; }
  speak() { return this.name; }
  get label() { return 1; }
  static make() { return new Animal('x'); }
}
const obj = { skip() { return 1; } };
"""


class TestJavaScript:
    def test_returns_arrow_function_and_class_methods(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.js", JS_SAMPLE)
        symbols = JavaScriptAdapter().parse_symbols(f)
        # Object-literal shorthand "skip" is intentionally excluded.
        assert _names(symbols) == {
            "add",
            "multiply",
            "sub",
            "counter",
            "constructor",
            "speak",
            "label",
            "make",
        }

    def test_kinds(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.js", JS_SAMPLE)
        by = _by_name(JavaScriptAdapter().parse_symbols(f))
        assert by["add"].kind == "function"
        assert by["multiply"].kind == "function"
        assert by["sub"].kind == "function"
        assert by["counter"].kind == "function"
        assert by["speak"].kind == "method"
        assert by["label"].kind == "method"
        assert by["make"].kind == "method"

    def test_locating_metadata(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.js", JS_SAMPLE)
        by = _by_name(JavaScriptAdapter().parse_symbols(f))
        assert by["add"].file == str(f)
        assert by["add"].start_line == 1
        assert by["speak"].start_line == 7

    def test_empty_file_yields_empty_list(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "empty.js", "// comment only\nconst x = 1;\n")
        assert JavaScriptAdapter().parse_symbols(f) == []

    def test_malformed_raises_parse_error(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "bad.js", "function broken( {\n")
        with pytest.raises(ParseError) as exc:
            JavaScriptAdapter().parse_symbols(f)
        assert "bad.js" in str(exc.value)


# --------------------------------------------------------------------------- #
# TypeScript / TSX (VAL-ENV-009: "incl. TS syntax", no TS parse error)
# --------------------------------------------------------------------------- #
TS_SAMPLE = """\
export function add(a: number, b: number): number { return a + b; }
const greet = (name: string): string => `hi ${name}`;
class Box<T> {
  private value: T;
  constructor(v: T) { this.value = v; }
  get(): T { return this.value; }
  handler = (x: number): number => x + 1;
}
interface Shape { area(): number; }
type ID = string;
"""


class TestTypeScript:
    def test_ts_syntax_parses_without_error(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.ts", TS_SAMPLE)
        symbols = JavaScriptAdapter().parse_symbols(f)
        # Functions/methods only: interface/type declarations are not symbols,
        # and the class field arrow `handler` is a method.
        assert _names(symbols) == {"add", "greet", "constructor", "get", "handler"}

    def test_ts_kinds(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.ts", TS_SAMPLE)
        by = _by_name(JavaScriptAdapter().parse_symbols(f))
        assert by["add"].kind == "function"
        assert by["greet"].kind == "function"
        assert by["handler"].kind == "method"
        assert by["get"].kind == "method"

    def test_ts_signature_keeps_type_annotations(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.ts", TS_SAMPLE)
        by = _by_name(JavaScriptAdapter().parse_symbols(f))
        assert by["add"].signature == "function add(a: number, b: number): number"

    def test_tsx_parses(self, tmp_path: Path) -> None:
        src = (
            "export const App = (p: {title: string}) => <div>{p.title}</div>;\n"
            "function render(): JSX.Element { return <span/>; }\n"
        )
        f = _write(tmp_path, "ui.tsx", src)
        assert _names(JavaScriptAdapter().parse_symbols(f)) == {"App", "render"}

    def test_malformed_ts_raises_parse_error(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "bad.ts", "function broken( {\n")
        with pytest.raises(ParseError):
            JavaScriptAdapter().parse_symbols(f)


# --------------------------------------------------------------------------- #
# Go (VAL-ENV-009/010/011, incl. receiver methods)
# --------------------------------------------------------------------------- #
GO_SAMPLE = """\
package shapes

import "math"

type Circle struct{ R float64 }

func NewCircle(r float64) *Circle { return &Circle{R: r} }

func (c *Circle) Area() float64 {
\treturn math.Pi * c.R * c.R
}

func (c Circle) Perimeter() float64 {
\treturn 2 * math.Pi * c.R
}

func Add(a, b int) int {
\treturn a + b
}
"""


@requires_go
class TestGo:
    def test_returns_functions_and_receiver_methods(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "shapes.go", GO_SAMPLE)
        symbols = GoAdapter().parse_symbols(f)
        assert _names(symbols) == {"NewCircle", "Area", "Perimeter", "Add"}

    def test_receiver_methods_classified(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "shapes.go", GO_SAMPLE)
        by = _by_name(GoAdapter().parse_symbols(f))
        assert by["NewCircle"].kind == "function"
        assert by["Add"].kind == "function"
        assert by["Area"].kind == "method"
        assert by["Perimeter"].kind == "method"

    def test_locating_metadata(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "shapes.go", GO_SAMPLE)
        by = _by_name(GoAdapter().parse_symbols(f))
        area = by["Area"]
        assert area.file == str(f)
        assert area.start_line == 9
        assert area.end_line == 11
        lines = GO_SAMPLE.splitlines()
        assert "func (c *Circle) Area()" in lines[area.start_line - 1]

    def test_signature_includes_receiver(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "shapes.go", GO_SAMPLE)
        by = _by_name(GoAdapter().parse_symbols(f))
        assert by["Area"].signature == "func (c *Circle) Area() float64"

    def test_trivial_package_yields_empty_list(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "empty.go", "package empty\n\nvar X = 1\n")
        assert GoAdapter().parse_symbols(f) == []

    def test_malformed_raises_parse_error(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "bad.go", "package bad\n\nfunc broken( {\n")
        with pytest.raises(ParseError) as exc:
            GoAdapter().parse_symbols(f)
        assert "bad.go" in str(exc.value)

    def test_helper_binary_is_cached(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "shapes.go", GO_SAMPLE)
        # Two parses must not error; the second reuses the cached binary.
        first = parse_go_symbols(f)
        second = parse_go_symbols(f)
        assert _names(first) == _names(second)


# --------------------------------------------------------------------------- #
# CLI surface: `forge parse-symbols`
# --------------------------------------------------------------------------- #
class TestParseSymbolsCli:
    def test_python_json(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.py", PY_SAMPLE)
        result = runner.invoke(forge_app, ["parse-symbols", str(f), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["language"] == "python"
        names = {s["name"] for s in payload["symbols"]}
        assert names == {"add", "fetch", "multiply", "helper"}
        for s in payload["symbols"]:
            assert s["file"] == str(f)
            assert s["start_line"] >= 1
            assert s["end_line"] >= s["start_line"]

    def test_ts_resolves_javascript(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.ts", TS_SAMPLE)
        result = runner.invoke(forge_app, ["parse-symbols", str(f), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["language"] == "javascript"
        assert {s["name"] for s in payload["symbols"]} == {
            "add",
            "greet",
            "constructor",
            "get",
            "handler",
        }

    @requires_go
    def test_go_json(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "shapes.go", GO_SAMPLE)
        result = runner.invoke(forge_app, ["parse-symbols", str(f), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["language"] == "go"
        assert {s["name"] for s in payload["symbols"]} == {
            "NewCircle",
            "Area",
            "Perimeter",
            "Add",
        }

    def test_empty_file_exits_zero_with_empty_list(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "empty.py", "X = 1\n")
        result = runner.invoke(forge_app, ["parse-symbols", str(f), "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output)["symbols"] == []

    def test_malformed_exits_one_with_clean_reason(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "bad.py", "def broken(:\n")
        result = runner.invoke(forge_app, ["parse-symbols", str(f), "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["parse_error"] is True
        assert "bad.py" in payload["reason"]
        assert "Traceback" not in result.output

    def test_missing_file_errors(self, tmp_path: Path) -> None:
        result = runner.invoke(
            forge_app, ["parse-symbols", str(tmp_path / "nope.py"), "--json"]
        )
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_unknown_extension_requires_language(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "mystery.rb", "def x; end\n")
        result = runner.invoke(forge_app, ["parse-symbols", str(f), "--json"])
        assert result.exit_code == 1
        assert "cannot infer language" in result.output

    def test_language_override(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "snippet.txt", "def add(a, b):\n    return a + b\n")
        result = runner.invoke(
            forge_app,
            ["parse-symbols", str(f), "--language", "python", "--json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert {s["name"] for s in payload["symbols"]} == {"add"}

    def test_human_readable_output(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "sample.py", PY_SAMPLE)
        result = runner.invoke(forge_app, ["parse-symbols", str(f)])
        assert result.exit_code == 0
        assert "add" in result.output
        assert "multiply" in result.output
