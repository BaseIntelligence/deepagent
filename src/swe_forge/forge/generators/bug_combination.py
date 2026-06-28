"""The ``bug_combination`` generator: merge >=2 independent faults.

Manufactures a harder multi-fault task by combining two or more distinct,
independently behavior-changing faults (each in its own file, so the hunks are
non-adjacent and across distinct symbols). The forward ``mutation_patch`` applies
every fault at once; the inverse gold ``oracle_patch`` restores all of them.

Provenance lists the constituent ``faults[]``, and for each fault records a
``single_fault_revert`` patch: applying just that one revert to the fully-broken
tree restores only that fault's file while the others stay broken, so reverting
any single fault alone still leaves >= 1 failing test (the
``bug_combination``-specific guarantee). Because each fault lives in a distinct
file, a single fault's revert is exactly that file's inverse patch.

Determinism: given the same repo and ``seed`` the generator shuffles the
discovered files deterministically and takes the first applicable fault per file,
so it selects the same faults and produces byte-identical patches.
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

_MIN_FAULTS = 2


def _resolve_count(request: GenerationRequest, minimum: int) -> int:
    """Return how many independent faults to merge (>= ``minimum``)."""
    raw = request.params.get("faults", minimum)
    if isinstance(raw, bool):
        count = minimum
    elif isinstance(raw, int):
        count = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        count = int(raw)
    else:
        count = minimum
    return max(count, minimum)


class BugCombinationGenerator(BugGenerator):
    """Multi-fault combination generator (difficulty amplifier; multi-language)."""

    name = "bug_combination"

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
                f"bug_combination: unknown operator {request.op!r}"
            ) from None
        count = _resolve_count(request, _MIN_FAULTS)

        with contextlib.chdir(repo_root):
            files = discover_source_files(repo_root, adapter)
            if len(files) < _MIN_FAULTS:
                raise GenerationError(
                    f"bug_combination: need >= {_MIN_FAULTS} distinct non-test "
                    f"source files to merge independent faults, found {len(files)} "
                    f"in {repo_root}"
                )
            faults = collect_distinct_file_faults(
                adapter, files, ops, seed=request.seed, count=count
            )
            if len(faults) < _MIN_FAULTS:
                raise GenerationError(
                    f"bug_combination: could not build >= {_MIN_FAULTS} independent "
                    f"round-tripping faults in {repo_root} (found {len(faults)})"
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

        # Each fault touches a distinct file, so reverting one fault alone is its
        # own file-level inverse patch applied to the fully-broken tree; the other
        # faults stay broken (>= 1 failing test remains).
        fault_records = [
            {
                "index": index,
                "file": fault.rel,
                "symbol": fault.symbol.name,
                "symbol_kind": fault.symbol.kind,
                "operator": fault.op.value if fault.op else "",
                "mutation_patch": fault.mutation_patch,
                "single_fault_revert": fault.oracle_patch,
                "original_sha256": sha256_bytes(fault.original),
                "mutated_sha256": sha256_bytes(fault.mutated),
            }
            for index, fault in enumerate(ordered)
        ]
        provenance = Provenance(
            generator=self.name,
            seed=request.seed,
            language=adapter.name,
            tool_versions=tool_versions(),
            details={
                "operation": "bug_combination",
                "fault_count": len(ordered),
                "files": list(files),
                "faults": fault_records,
            },
        )
        return Candidate(
            language=adapter.name,
            generator=self.name,
            target=CandidateTarget(files=files, symbols=symbols),
            mutation_patch=combined.mutation_patch,
            oracle_patch=combined.oracle_patch,
            difficulty_hint="high",
            provenance=provenance,
        )
