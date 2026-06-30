"""The ``multi_file`` generator: coordinated edits across >=2 files.

Manufactures a bug whose forward ``mutation_patch`` touches two or more distinct
non-test source files (one independent AST fault per file) and whose inverse
gold ``oracle_patch`` restores every one of them. The :class:`Candidate`'s
``target.files`` lists exactly the changed files. The generator verifies the
multi-file round-trip (mutation then oracle restores every touched file
byte-for-byte) before emitting, so a non-applying or non-inverting result is
never shipped.

Determinism: given the same repo and ``seed`` the generator shuffles the
discovered files deterministically and takes the first applicable fault per file,
so it selects the same files/symbols and produces byte-identical patches.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from swe_forge.forge.adapters.base import LanguageAdapter
from swe_forge.forge.generators._targeting import (
    SingleFileFault,
    build_combined_patches,
    collect_distinct_file_faults,
    discover_source_files,
    resolve_ops,
    sha256_bytes,
    tool_versions,
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

_MIN_FILES = 2


def _resolve_count(request: GenerationRequest, minimum: int) -> int:
    """Return how many distinct-file faults to merge (>= ``minimum``)."""
    raw = request.params.get("files", minimum)
    if isinstance(raw, bool):
        count = minimum
    elif isinstance(raw, int):
        count = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        count = int(raw)
    else:
        count = minimum
    return max(count, minimum)


class MultiFileGenerator(BugGenerator):
    """Coordinated multi-file fault generator (Python / JS-TS / Go)."""

    name = "multi_file"

    def generate(
        self, request: GenerationRequest, adapter: LanguageAdapter
    ) -> Candidate:
        if request.env_image is not None:
            require_green_baseline(request.env_image)

        repo_root = Path(request.repo_root).resolve()
        if not repo_root.is_dir():
            raise GenerationError(f"repo path is not a directory: {repo_root}")

        try:
            ops = resolve_ops(request.op)
        except KeyError:
            raise GenerationError(
                f"multi_file: unknown operator {request.op!r}"
            ) from None
        count = _resolve_count(request, _MIN_FILES)

        with contextlib.chdir(repo_root):
            files = discover_source_files(repo_root, adapter)
            if len(files) < _MIN_FILES:
                raise GenerationError(
                    f"multi_file: need >= {_MIN_FILES} distinct non-test source "
                    f"files, found {len(files)} in {repo_root}"
                )
            faults = collect_distinct_file_faults(
                adapter, files, ops, seed=request.seed, count=count
            )
            if len(faults) < _MIN_FILES:
                raise GenerationError(
                    f"multi_file: could not build >= {_MIN_FILES} round-tripping "
                    f"faults across distinct files in {repo_root} "
                    f"(found {len(faults)})"
                )
            return self._build_candidate(request, adapter, faults)

    def _build_candidate(
        self,
        request: GenerationRequest,
        adapter: LanguageAdapter,
        faults: list[SingleFileFault],
    ) -> Candidate:
        combined = build_combined_patches(faults)
        ordered = sorted(faults, key=lambda fault: fault.rel)
        files = tuple(fault.rel for fault in ordered)
        symbols = tuple(fault.symbol.name for fault in ordered)
        provenance = Provenance(
            generator=self.name,
            seed=request.seed,
            language=adapter.name,
            tool_versions=tool_versions(),
            details={
                "operation": "multi_file",
                "files": list(files),
                "edits": [
                    {
                        "file": fault.rel,
                        "symbol": fault.symbol.name,
                        "start_line": fault.symbol.start_line,
                        "end_line": fault.symbol.end_line,
                        "operator": fault.op.value if fault.op else "",
                        "original_sha256": sha256_bytes(fault.original),
                        "mutated_sha256": sha256_bytes(fault.mutated),
                    }
                    for fault in ordered
                ],
            },
        )
        return Candidate(
            language=adapter.name,
            generator=self.name,
            target=CandidateTarget(files=files, symbols=symbols),
            mutation_patch=combined.mutation_patch,
            oracle_patch=combined.oracle_patch,
            difficulty_hint="medium",
            provenance=provenance,
        )
