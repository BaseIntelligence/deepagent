"""Language adapters: the :class:`LanguageAdapter` interface, registry, and stubs.

Public surface for the adapter layer. :func:`build_default_registry` returns a
fresh registry with the Python / JS-TS / Go stubs registered in a deterministic
order.
"""

from __future__ import annotations

from swe_forge.forge.adapters.base import (
    AdapterError,
    AdapterRegistry,
    DuplicateAdapterError,
    LanguageAdapter,
    MutantStats,
    MutationOp,
    NoAdapterFoundError,
    ParseError,
    Patch,
    PathLike,
    Symbol,
)
from swe_forge.forge.adapters.golang import GoAdapter
from swe_forge.forge.adapters.javascript import JavaScriptAdapter
from swe_forge.forge.adapters.python import PythonAdapter


def build_default_registry() -> AdapterRegistry:
    """Return a registry holding the Python, JS-TS, and Go adapter stubs."""
    registry = AdapterRegistry()
    registry.register(PythonAdapter())
    registry.register(JavaScriptAdapter())
    registry.register(GoAdapter())
    return registry


__all__ = [
    "AdapterError",
    "AdapterRegistry",
    "DuplicateAdapterError",
    "GoAdapter",
    "JavaScriptAdapter",
    "LanguageAdapter",
    "MutantStats",
    "MutationOp",
    "NoAdapterFoundError",
    "ParseError",
    "Patch",
    "PathLike",
    "PythonAdapter",
    "Symbol",
    "build_default_registry",
]
