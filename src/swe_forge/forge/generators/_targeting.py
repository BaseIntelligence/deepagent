"""Shared target discovery and single-file fault construction for generators.

Every Stage-2 generator manufactures a bug from a target symbol in a non-test
source file. The discovery rules (which files/extensions count, which
directories to skip), the deterministic site enumeration, and the
mutate-then-verify-round-trip dance are identical across generators, so they live
here and are reused by ``ast_mutation`` (single fault), ``multi_file`` and
``bug_combination`` (several independent faults).

Callers MUST run inside ``contextlib.chdir(repo_root)``: the adapter reads source
files by the repo-relative path, while the diff helpers operate on explicit
bytes and are cwd-independent.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from swe_forge.forge.adapters._diff import (
    PatchError,
    apply_multi_patch,
    apply_patch,
    make_multi_patch,
    make_patch,
)
from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutationOp,
    ParseError,
    Patch,
    Symbol,
)
from swe_forge.forge.generators._normalize import is_behavior_changing

# Source extensions considered per language when auto-discovering targets.
SOURCE_EXTENSIONS: dict[str, frozenset[str]] = {
    "python": frozenset({".py"}),
    "javascript": frozenset(
        {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
    ),
    "go": frozenset({".go"}),
}

# Directories never searched for auto-discovered targets.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        "node_modules",
        "vendor",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        "testdata",
    }
)

# The deterministic operator families tried, in this order, when mutating a site.
DEFAULT_OPS: tuple[MutationOp, ...] = (
    MutationOp.OPERATOR_SWAP,
    MutationOp.OFF_BY_ONE,
    MutationOp.BRANCH_REMOVAL,
)
OP_BY_NAME: dict[str, MutationOp] = {op.value: op for op in DEFAULT_OPS}


def sha256_bytes(data: bytes) -> str:
    """Return the hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def tool_versions() -> dict[str, str]:
    """Best-effort versions of the tree-sitter grammars the mutation depends on."""
    versions: dict[str, str] = {}
    for dist in ("tree-sitter", "tree-sitter-python"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return versions


def resolve_ops(op_hint: str | None) -> tuple[MutationOp, ...]:
    """Return the operator families to try, honoring an explicit ``op_hint``."""
    if op_hint is None:
        return DEFAULT_OPS
    if op_hint not in OP_BY_NAME:
        raise KeyError(op_hint)
    return (OP_BY_NAME[op_hint],)


def discover_source_files(repo_root: Path, adapter: LanguageAdapter) -> list[str]:
    """Return the repo-relative non-test source files for ``adapter``'s language."""
    extensions = SOURCE_EXTENSIONS.get(adapter.name, frozenset())
    found: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(repo_root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if adapter.is_test_file(rel.as_posix()):
            continue
        found.append(rel.as_posix())
    return found


def resolve_target_files(
    repo_root: Path, adapter: LanguageAdapter, file_hint: str | None
) -> list[str]:
    """Return the target file list: the pinned ``file_hint`` or auto-discovery.

    Raises :class:`FileNotFoundError` when ``file_hint`` names a missing file.
    """
    if file_hint is not None:
        target = repo_root / file_hint
        if not target.is_file():
            raise FileNotFoundError(file_hint)
        return [Path(file_hint).as_posix()]
    return discover_source_files(repo_root, adapter)


def parse_symbols_safe(adapter: LanguageAdapter, rel: str) -> list[Symbol]:
    """Parse ``rel``'s symbols, returning ``[]`` if the file does not parse."""
    try:
        return adapter.parse_symbols(rel)
    except ParseError:
        return []


@dataclass(frozen=True)
class SingleFileFault:
    """One verified, round-tripping, single-file fault.

    Pairs the targeted ``symbol`` (and ``op`` for an AST mutation, ``None`` for a
    body removal) with the forward ``mutation_patch`` and the ``original``/
    ``mutated`` bytes of ``rel``. Each fault is independently behavior-changing
    (``mutated != original``) and proven to round-trip byte-for-byte before
    construction.
    """

    rel: str
    symbol: Symbol
    op: MutationOp | None
    mutation_patch: str
    original: bytes
    mutated: bytes

    @property
    def oracle_patch(self) -> str:
        """The inverse patch that restores ``rel`` from ``mutated`` to ``original``."""
        return make_patch(self.rel, self.mutated, self.original)


def _verify_single_file_fault(
    rel: str, symbol: Symbol, op: MutationOp | None, forward: Patch
) -> SingleFileFault | None:
    """Verify a forward edit round-trips byte-for-byte; build the fault or ``None``."""
    if not forward.diff.strip():
        return None
    original = Path(rel).read_bytes()
    try:
        mutated = apply_patch(original, forward.diff, rel)
        if mutated == original:
            return None
        # Behavior gate: a whitespace/comment/import-only edit changes no
        # behavior and must never become a Candidate (VAL-GEN-002).
        if not is_behavior_changing(
            original.decode("utf-8", "replace"), mutated.decode("utf-8", "replace")
        ):
            return None
        oracle = make_patch(rel, mutated, original)
        if not oracle.strip():
            return None
        restored = apply_patch(mutated, oracle, rel)
    except PatchError:
        return None
    if sha256_bytes(restored) != sha256_bytes(original):
        return None
    return SingleFileFault(
        rel=rel,
        symbol=symbol,
        op=op,
        mutation_patch=forward.diff,
        original=original,
        mutated=mutated,
    )


def verify_forward_patch(
    rel: str, symbol: Symbol, forward: Patch
) -> SingleFileFault | None:
    """Verify a teacher-proposed forward edit round-trips; build the fault or ``None``.

    Reuses the shared single-file round-trip check (apply forward, derive the
    inverse oracle, re-apply, sha256 must match the pristine original) so an
    LLM-authored edit that does not apply or does not invert is disposed of here
    and never shipped. Must run with cwd at the repo root (the original bytes are
    re-read from ``rel`` on disk).
    """
    return _verify_single_file_fault(rel, symbol, None, forward)


def try_fault(
    adapter: LanguageAdapter, rel: str, symbol: Symbol, op: MutationOp
) -> SingleFileFault | None:
    """Attempt one AST mutation; return a verified fault or ``None`` if inapplicable.

    Returns ``None`` (rather than raising) when the operator has no site, the
    edit is a no-op, or a patch does not apply, so the caller can move on to the
    next candidate site. Must run with cwd at the repo root.
    """
    try:
        forward = adapter.mutate_ast(rel, symbol, op)
    except (ParseError, PatchError):
        return None
    return _verify_single_file_fault(rel, symbol, op, forward)


def try_removal_fault(
    adapter: LanguageAdapter, rel: str, symbol: Symbol
) -> SingleFileFault | None:
    """Attempt a body removal on ``symbol``; return a verified fault or ``None``.

    Mirrors :func:`try_fault` but excises the function body (signature kept).
    Must run with cwd at the repo root.
    """
    try:
        forward = adapter.remove_function_body(rel, symbol)
    except (ParseError, PatchError):
        return None
    return _verify_single_file_fault(rel, symbol, None, forward)


def first_fault_in_file(
    adapter: LanguageAdapter,
    rel: str,
    ops: tuple[MutationOp, ...],
    *,
    symbol_hint: str | None = None,
) -> SingleFileFault | None:
    """Return the first verified fault for ``rel`` over its symbols x ``ops``.

    Symbols are tried in parse order and operators in ``ops`` order, so selection
    is deterministic. Must run with cwd at the repo root.
    """
    for symbol in parse_symbols_safe(adapter, rel):
        if symbol_hint is not None and symbol.name != symbol_hint:
            continue
        for op in ops:
            fault = try_fault(adapter, rel, symbol, op)
            if fault is not None:
                return fault
    return None


def collect_distinct_file_faults(
    adapter: LanguageAdapter,
    files: list[str],
    ops: tuple[MutationOp, ...],
    *,
    seed: int,
    count: int,
) -> list[SingleFileFault]:
    """Collect up to ``count`` independent faults, each in a DISTINCT file.

    Files are shuffled deterministically by ``seed`` and the first applicable
    fault per file is taken, so the faults are independent (distinct files,
    distinct symbols, non-adjacent hunks) and each reverts on its own. Returns
    however many distinct-file faults were found (possibly fewer than ``count``);
    callers enforce their own minimum. Must run with cwd at the repo root.
    """
    ordered = list(files)
    random.Random(seed).shuffle(ordered)
    faults: list[SingleFileFault] = []
    for rel in ordered:
        if len(faults) >= count:
            break
        fault = first_fault_in_file(adapter, rel, ops)
        if fault is not None:
            faults.append(fault)
    return faults


@dataclass(frozen=True)
class CombinedPatches:
    """A verified multi-file forward/inverse patch pair over several faults."""

    mutation_patch: str
    oracle_patch: str


def build_combined_patches(faults: list[SingleFileFault]) -> CombinedPatches:
    """Combine per-file faults into one multi-file mutation+oracle pair, verified.

    Each fault must touch a distinct file. The combined mutation applies all
    faults at once and the combined oracle restores every file byte-for-byte;
    both directions are applied through ``git apply`` and checked here so a
    non-applying or non-inverting combination is never returned.

    Raises :class:`ValueError` if faults share a file or the round-trip fails.
    """
    rels = [fault.rel for fault in faults]
    if len(set(rels)) != len(rels):
        raise ValueError("combined faults must each touch a distinct file")

    originals = {fault.rel: fault.original for fault in faults}
    targets = {fault.rel: fault.mutated for fault in faults}

    mutation = make_multi_patch(
        [(fault.rel, fault.original, fault.mutated) for fault in faults]
    )
    oracle = make_multi_patch(
        [(fault.rel, fault.mutated, fault.original) for fault in faults]
    )
    if not mutation.strip() or not oracle.strip():
        raise ValueError("combined faults produced an empty patch")

    try:
        applied = apply_multi_patch(originals, mutation)
        restored = apply_multi_patch(applied, oracle)
    except PatchError as exc:
        raise ValueError(f"combined patch did not apply cleanly: {exc}") from exc

    for rel in originals:
        if sha256_bytes(applied[rel]) != sha256_bytes(targets[rel]):
            raise ValueError(f"combined mutation did not match the fault for {rel}")
        if sha256_bytes(restored[rel]) != sha256_bytes(originals[rel]):
            raise ValueError(f"combined oracle did not restore {rel} byte-for-byte")

    return CombinedPatches(mutation_patch=mutation, oracle_patch=oracle)
