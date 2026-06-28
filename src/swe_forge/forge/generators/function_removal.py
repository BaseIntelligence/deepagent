"""The ``function_removal`` generator: delete a body, keep the signature.

Manufactures a bug by removing a whole function/method *body* (its signature is
preserved byte-for-byte) via ``adapter.remove_function_body``; the gold oracle
re-inserts the original body exactly. The :class:`Candidate` carries the forward
``mutation_patch`` and its inverse ``oracle_patch``, and the generator verifies
the round-trip (mutation then oracle restores the touched file byte-for-byte)
before emitting, so a non-applying or non-inverting result is never shipped.

Multi-language (Python / JS-TS / Go) through the adapter. Determinism: given the
same repo, target hints, and ``seed`` the generator selects the same (file,
symbol) and produces byte-identical patches; when a target is not fully pinned it
enumerates candidate sites in a stable order and shuffles them with
``random.Random(seed)``, returning the first removable body.
"""

from __future__ import annotations

import contextlib
import random
from pathlib import Path

from swe_forge.forge.adapters.base import LanguageAdapter, Symbol
from swe_forge.forge.generators._targeting import (
    SingleFileFault,
    parse_symbols_safe,
    resolve_target_files,
    sha256_bytes,
    tool_versions,
    try_removal_fault,
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


class FunctionRemovalGenerator(BugGenerator):
    """Body-removal fault generator (signature kept; Python / JS-TS / Go)."""

    name = "function_removal"

    def generate(
        self, request: GenerationRequest, adapter: LanguageAdapter
    ) -> Candidate:
        if request.env_image is not None:
            require_green_baseline(request.env_image)

        repo_root = Path(request.repo_root).resolve()
        if not repo_root.is_dir():
            raise GenerationError(f"repo path is not a directory: {repo_root}")

        with contextlib.chdir(repo_root):
            try:
                files = resolve_target_files(repo_root, adapter, request.file)
            except FileNotFoundError:
                raise GenerationError(
                    f"function_removal: target file not found: {request.file}"
                ) from None
            sites = self._candidate_sites(adapter, files, request.symbol)
            if not sites:
                raise GenerationError(
                    f"function_removal: no removable function found for "
                    f"{self._describe_target(request)} in {repo_root}"
                )
            random.Random(request.seed).shuffle(sites)
            for rel, symbol in sites:
                fault = try_removal_fault(adapter, rel, symbol)
                if fault is not None:
                    return self._build_candidate(request, adapter, fault)

        raise GenerationError(
            f"function_removal: no function body could be removed and restored "
            f"for {self._describe_target(request)} in {repo_root}"
        )

    def _candidate_sites(
        self, adapter: LanguageAdapter, files: list[str], symbol_hint: str | None
    ) -> list[tuple[str, Symbol]]:
        sites: list[tuple[str, Symbol]] = []
        for rel in files:
            for symbol in parse_symbols_safe(adapter, rel):
                if symbol_hint is not None and symbol.name != symbol_hint:
                    continue
                sites.append((rel, symbol))
        return sites

    def _build_candidate(
        self,
        request: GenerationRequest,
        adapter: LanguageAdapter,
        fault: SingleFileFault,
    ) -> Candidate:
        provenance = Provenance(
            generator=self.name,
            seed=request.seed,
            language=adapter.name,
            tool_versions=tool_versions(),
            details={
                "operation": "function_removal",
                "file": fault.rel,
                "symbol": fault.symbol.name,
                "symbol_kind": fault.symbol.kind,
                "signature": fault.symbol.signature or "",
                "start_line": fault.symbol.start_line,
                "end_line": fault.symbol.end_line,
                "original_sha256": sha256_bytes(fault.original),
                "mutated_sha256": sha256_bytes(fault.mutated),
            },
        )
        return Candidate(
            language=adapter.name,
            generator=self.name,
            target=CandidateTarget(files=(fault.rel,), symbols=(fault.symbol.name,)),
            mutation_patch=fault.mutation_patch,
            oracle_patch=fault.oracle_patch,
            difficulty_hint="medium",
            provenance=provenance,
        )

    def _describe_target(self, request: GenerationRequest) -> str:
        parts = []
        if request.file:
            parts.append(f"file={request.file}")
        if request.symbol:
            parts.append(f"symbol={request.symbol}")
        return ", ".join(parts) if parts else "any target"
