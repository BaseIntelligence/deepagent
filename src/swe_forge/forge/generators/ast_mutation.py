"""The ``ast_mutation`` generator: deterministic, multi-language AST faults.

Manufactures a bug by applying one deterministic AST mutation (operator swap,
off-by-one, or branch removal) to a single target symbol via
``adapter.mutate_ast``. The resulting :class:`Candidate` carries the forward
``mutation_patch`` and its inverse gold ``oracle_patch``; the generator verifies
the round-trip (mutation then oracle restores the touched file byte-for-byte)
with real ``git apply`` before emitting, so a non-applying or non-inverting
result is never shipped.

Determinism: given the same repo, target hints, and ``seed`` the generator
selects the same (file, symbol, operator) and produces byte-identical patches.
When a target is not fully pinned it enumerates candidate sites in a stable order
and shuffles them with ``random.Random(seed)``, returning the first site that
yields a valid behavior-changing, round-tripping mutation.
"""

from __future__ import annotations

import contextlib
import hashlib
import random
from importlib import metadata
from pathlib import Path

from swe_forge.forge.adapters._diff import PatchError, apply_patch, make_patch
from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutationOp,
    ParseError,
    Symbol,
)
from swe_forge.forge.generators.base import (
    BugGenerator,
    GenerationError,
    GenerationRequest,
)
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    Provenance,
    require_green_baseline,
)

# The operator families this generator supports, tried in this order.
_OPS: tuple[MutationOp, ...] = (
    MutationOp.OPERATOR_SWAP,
    MutationOp.OFF_BY_ONE,
    MutationOp.BRANCH_REMOVAL,
)
_OP_BY_NAME: dict[str, MutationOp] = {op.value: op for op in _OPS}

# Source extensions considered per language when auto-discovering targets.
_SOURCE_EXTENSIONS: dict[str, frozenset[str]] = {
    "python": frozenset({".py"}),
    "javascript": frozenset(
        {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
    ),
    "go": frozenset({".go"}),
}
# Directories never searched for auto-discovered targets.
_SKIP_DIRS: frozenset[str] = frozenset(
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tool_versions() -> dict[str, str]:
    """Best-effort versions of the tools the mutation depends on."""
    versions: dict[str, str] = {}
    for dist in ("tree-sitter", "tree-sitter-python"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return versions


class AstMutationGenerator(BugGenerator):
    """Deterministic AST operator-fault generator (Python / JS-TS / Go)."""

    name = "ast_mutation"

    def generate(
        self, request: GenerationRequest, adapter: LanguageAdapter
    ) -> Candidate:
        if request.env_image is not None:
            require_green_baseline(request.env_image)

        repo_root = Path(request.repo_root).resolve()
        if not repo_root.is_dir():
            raise GenerationError(f"repo path is not a directory: {repo_root}")

        ops = self._resolve_ops(request.op)

        # All file/symbol reads and the emitted patch paths are repo-relative.
        with contextlib.chdir(repo_root):
            files = self._target_files(repo_root, adapter, request.file)
            sites = self._candidate_sites(adapter, files, request.symbol, ops)
            if not sites:
                raise GenerationError(
                    f"ast_mutation: no mutable symbol found for "
                    f"{self._describe_target(request)} in {repo_root}"
                )
            random.Random(request.seed).shuffle(sites)
            for rel, symbol, op in sites:
                candidate = self._try_site(request, adapter, rel, symbol, op)
                if candidate is not None:
                    return candidate

        raise GenerationError(
            f"ast_mutation: no applicable mutation produced a verified round-trip "
            f"for {self._describe_target(request)} in {repo_root}"
        )

    def _resolve_ops(self, op_hint: str | None) -> tuple[MutationOp, ...]:
        if op_hint is None:
            return _OPS
        if op_hint not in _OP_BY_NAME:
            raise GenerationError(
                f"ast_mutation: unknown operator {op_hint!r}; "
                f"choose one of {', '.join(_OP_BY_NAME)}"
            )
        return (_OP_BY_NAME[op_hint],)

    def _target_files(
        self, repo_root: Path, adapter: LanguageAdapter, file_hint: str | None
    ) -> list[str]:
        if file_hint is not None:
            target = repo_root / file_hint
            if not target.is_file():
                raise GenerationError(
                    f"ast_mutation: target file not found: {file_hint}"
                )
            return [Path(file_hint).as_posix()]
        return self._discover_files(repo_root, adapter)

    def _discover_files(self, repo_root: Path, adapter: LanguageAdapter) -> list[str]:
        extensions = _SOURCE_EXTENSIONS.get(adapter.name, frozenset())
        found: list[str] = []
        for path in sorted(repo_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            rel = path.relative_to(repo_root)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if adapter.is_test_file(rel.as_posix()):
                continue
            found.append(rel.as_posix())
        return found

    def _candidate_sites(
        self,
        adapter: LanguageAdapter,
        files: list[str],
        symbol_hint: str | None,
        ops: tuple[MutationOp, ...],
    ) -> list[tuple[str, Symbol, MutationOp]]:
        sites: list[tuple[str, Symbol, MutationOp]] = []
        for rel in files:
            try:
                symbols = adapter.parse_symbols(rel)
            except ParseError:
                continue
            for symbol in symbols:
                if symbol_hint is not None and symbol.name != symbol_hint:
                    continue
                for op in ops:
                    sites.append((rel, symbol, op))
        return sites

    def _try_site(
        self,
        request: GenerationRequest,
        adapter: LanguageAdapter,
        rel: str,
        symbol: Symbol,
        op: MutationOp,
    ) -> Candidate | None:
        original = Path(rel).read_bytes()
        try:
            forward = adapter.mutate_ast(rel, symbol, op)
        except (ParseError, PatchError):
            return None
        if not forward.diff.strip():
            return None
        try:
            mutated = apply_patch(original, forward.diff, rel)
            if mutated == original:
                return None
            oracle = make_patch(rel, mutated, original)
            if not oracle.strip():
                return None
            restored = apply_patch(mutated, oracle, rel)
        except PatchError:
            return None
        if _sha256(restored) != _sha256(original):
            return None

        provenance = Provenance(
            generator=self.name,
            seed=request.seed,
            language=adapter.name,
            tool_versions=_tool_versions(),
            details={
                "operator": op.value,
                "file": rel,
                "symbol": symbol.name,
                "symbol_kind": symbol.kind,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
                "original_sha256": _sha256(original),
                "mutated_sha256": _sha256(mutated),
            },
        )
        return Candidate(
            language=adapter.name,
            generator=self.name,
            target=CandidateTarget(files=(rel,), symbols=(symbol.name,)),
            mutation_patch=forward.diff,
            oracle_patch=oracle,
            difficulty_hint="low",
            provenance=provenance,
        )

    def _describe_target(self, request: GenerationRequest) -> str:
        parts = []
        if request.file:
            parts.append(f"file={request.file}")
        if request.symbol:
            parts.append(f"symbol={request.symbol}")
        if request.op:
            parts.append(f"op={request.op}")
        return ", ".join(parts) if parts else "any target"
