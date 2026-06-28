"""Unit tests for the language-adapter interface and registry (offline).

Covers the M1 foundation invariants for the adapter layer:
- ``LanguageAdapter`` is an ABC defining all contracted members.
- ``AdapterRegistry`` registers adapters and selects one by ``detect()``.
- The Python / JS-TS / Go stubs exist, are registered, and raise
  ``NotImplementedError`` for their (not-yet-implemented) behavior.
- The shared value types (Symbol / Patch / MutantStats / MutationOp) behave.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from swe_forge.forge.adapters import (
    AdapterRegistry,
    DuplicateAdapterError,
    GoAdapter,
    JavaScriptAdapter,
    LanguageAdapter,
    MutantStats,
    MutationOp,
    NoAdapterFoundError,
    Patch,
    PythonAdapter,
    Symbol,
    build_default_registry,
)
from swe_forge.forge.adapters.base import PathLike


class FakeAdapter(LanguageAdapter):
    """A fully-implemented adapter used to exercise registry selection."""

    def __init__(self, name: str, *, matches: bool) -> None:
        self.name = name
        self._matches = matches

    def detect(self, repo_path: PathLike) -> bool:
        return self._matches

    def base_image(self) -> str:
        return f"{self.name}:latest"

    def install_commands(self, repo_path: PathLike) -> list[str]:
        return [f"install-{self.name}"]

    def test_command(self, selection: Sequence[str] | None = None) -> str:
        if selection:
            return f"test-{self.name} {' '.join(selection)}"
        return f"test-{self.name}"

    def parse_symbols(self, file: PathLike) -> list[Symbol]:
        return [
            Symbol(name="f", kind="function", file=str(file), start_line=1, end_line=2)
        ]

    def mutate_ast(self, file: PathLike, symbol: Symbol, op: MutationOp) -> Patch:
        return Patch(diff=f"--- {file}\n", files=(str(file),))

    def mutation_tool_run(
        self,
        image: str,
        repo_path: PathLike,
        *,
        paths: Sequence[str] | None = None,
        test_command: str | None = None,
    ) -> MutantStats:
        return MutantStats(total=4, killed=3, survived=1, tool=f"{self.name}-tool")

    def is_test_file(self, path: PathLike) -> bool:
        return str(path).startswith("test_")


STUB_ADAPTERS = (PythonAdapter, JavaScriptAdapter, GoAdapter)
# Behavior still pending later milestones (AST mutate, in-Docker mutation run).
# The build-time methods (detect/base_image/install_commands/test_command/
# is_test_file) are implemented and covered in test_adapters_concrete.py;
# parse_symbols is implemented and covered in test_adapters_parse_symbols.py.
UNIMPLEMENTED_METHODS = (
    "mutate_ast",
    "mutation_tool_run",
)


class TestLanguageAdapterABC:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            LanguageAdapter()  # type: ignore[abstract]

    def test_incomplete_subclass_is_not_instantiable(self) -> None:
        class Incomplete(LanguageAdapter):
            name = "incomplete"

            def detect(self, repo_path: PathLike) -> bool:
                return False

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_fully_implemented_subclass_is_usable(self) -> None:
        adapter = FakeAdapter("python", matches=True)
        assert adapter.detect("/repo") is True
        assert adapter.base_image() == "python:latest"
        assert adapter.install_commands("/repo") == ["install-python"]
        assert adapter.is_test_file("test_x.py") is True


class TestAdapterRegistry:
    def test_register_returns_adapter_and_tracks_order(self) -> None:
        registry = AdapterRegistry()
        py = FakeAdapter("python", matches=False)
        go = FakeAdapter("go", matches=False)

        assert registry.register(py) is py
        registry.register(go)

        assert len(registry) == 2
        assert registry.names() == ("python", "go")
        assert registry.adapters() == (py, go)
        assert list(registry) == [py, go]

    def test_register_rejects_duplicate_name(self) -> None:
        registry = AdapterRegistry()
        registry.register(FakeAdapter("python", matches=False))
        with pytest.raises(DuplicateAdapterError):
            registry.register(FakeAdapter("python", matches=True))

    def test_register_rejects_empty_name(self) -> None:
        registry = AdapterRegistry()
        with pytest.raises(ValueError):
            registry.register(FakeAdapter("", matches=True))

    def test_detect_selects_matching_adapter(self) -> None:
        registry = AdapterRegistry()
        registry.register(FakeAdapter("python", matches=False))
        go = registry.register(FakeAdapter("go", matches=True))

        assert registry.detect("/repo") is go
        assert registry.detect_optional("/repo") is go

    def test_detect_returns_first_match_in_registration_order(self) -> None:
        registry = AdapterRegistry()
        first = registry.register(FakeAdapter("javascript", matches=True))
        registry.register(FakeAdapter("go", matches=True))

        assert registry.detect("/repo") is first
        assert [a.name for a in registry.detect_all("/repo")] == ["javascript", "go"]

    def test_detect_raises_when_no_adapter_matches(self) -> None:
        registry = AdapterRegistry()
        registry.register(FakeAdapter("python", matches=False))
        with pytest.raises(NoAdapterFoundError):
            registry.detect("/repo")

    def test_detect_optional_returns_none_when_no_match(self) -> None:
        registry = AdapterRegistry()
        registry.register(FakeAdapter("python", matches=False))
        assert registry.detect_optional("/repo") is None
        assert registry.detect_all("/repo") == []

    def test_get_by_name_and_missing(self) -> None:
        registry = AdapterRegistry()
        py = registry.register(FakeAdapter("python", matches=False))
        assert registry.get("python") is py
        with pytest.raises(NoAdapterFoundError):
            registry.get("rust")


class TestDefaultRegistry:
    def test_default_registry_holds_the_three_stubs(self) -> None:
        registry = build_default_registry()
        assert registry.names() == ("python", "javascript", "go")

    def test_default_registry_instances_are_independent(self) -> None:
        first = build_default_registry()
        second = build_default_registry()
        assert first is not second
        assert first.get("python") is not second.get("python")


class TestStubAdapters:
    @pytest.mark.parametrize(
        ("adapter_cls", "expected_name"),
        [
            (PythonAdapter, "python"),
            (JavaScriptAdapter, "javascript"),
            (GoAdapter, "go"),
        ],
    )
    def test_stub_name(
        self, adapter_cls: type[LanguageAdapter], expected_name: str
    ) -> None:
        assert adapter_cls.name == expected_name
        assert adapter_cls().name == expected_name

    @pytest.mark.parametrize("adapter_cls", STUB_ADAPTERS)
    @pytest.mark.parametrize("method", UNIMPLEMENTED_METHODS)
    def test_pending_methods_raise_not_implemented(
        self, adapter_cls: type[LanguageAdapter], method: str
    ) -> None:
        adapter = adapter_cls()
        args: tuple[object, ...]
        if method == "mutate_ast":
            symbol = Symbol(
                name="f", kind="function", file="a.py", start_line=1, end_line=2
            )
            args = ("a.py", symbol, MutationOp.OPERATOR_SWAP)
        else:  # mutation_tool_run
            args = ("image:tag", "/repo")
        with pytest.raises(NotImplementedError):
            getattr(adapter, method)(*args)


class TestValueTypes:
    def test_symbol_fields(self) -> None:
        sym = Symbol(name="add", kind="function", file="m.py", start_line=3, end_line=5)
        assert (sym.name, sym.file, sym.start_line, sym.end_line) == (
            "add",
            "m.py",
            3,
            5,
        )
        assert sym.signature is None

    def test_patch_defaults(self) -> None:
        patch = Patch(diff="--- a\n+++ b\n")
        assert patch.files == ()

    def test_mutant_stats_kill_ratio(self) -> None:
        assert MutantStats(total=10, killed=8).kill_ratio == 0.8
        assert MutantStats(total=0, killed=0).kill_ratio == 0.0

    def test_mutation_op_members_are_distinct_strings(self) -> None:
        values = [op.value for op in MutationOp]
        assert len(values) == len(set(values))
        assert MutationOp.OPERATOR_SWAP.value == "operator_swap"
