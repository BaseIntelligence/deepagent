"""Offline tests for the m6-band-supply difficulty amplifier (lever 3).

The m6-pilot-build measurement showed ``bug_combination`` difficulty against the
elite sealed panel is BIMODAL on the fault-COUNT knob (faults=2 -> solve-all;
faults=3 -> solve-none), so the keep band is a knife-edge on that axis. This
feature centers difficulty on the ORTHOGONAL fault-LOCATE axis: fix faults=2 and
vary the TARGET-SYMBOL SIZE so a subtle single fault lands in a larger (harder to
locate) or smaller (obvious) function. These tests cover the deterministic,
fully-offline machinery (no Docker, no live LLM):

* :func:`order_symbols_by_difficulty` filters by ``min_symbol_lines`` and orders
  by the ``prefer`` target (largest/smallest/parse), stable + deterministic;
* the ``bug_combination`` generator threads ``min_symbol_lines``/``prefer`` params
  and records them in provenance, still round-tripping byte-for-byte;
* ``build_pilot_plans`` expands each amplifier cell across the difficulty ladder
  so oracle-passing candidates spread across the spectrum (band populated, not a
  knife-edge) -- while never touching band_high or any gate.
"""

from __future__ import annotations

from pathlib import Path

from swe_forge.forge.adapters import PythonAdapter
from swe_forge.forge.adapters.base import Symbol
from swe_forge.forge.generators import BugCombinationGenerator, GenerationRequest
from swe_forge.forge.generators import MultiFileGenerator
from swe_forge.forge.generators._targeting import (
    PREFER_LARGEST,
    PREFER_PARSE,
    PREFER_SMALLEST,
    order_symbols_by_difficulty,
)
from swe_forge.forge.pilot import (
    AMPLIFIER_GENERATORS,
    DEFAULT_AMPLIFIER_LADDER,
    build_pilot_plans,
)
from swe_forge.forge.sources import build_source_registry


def _sym(name: str, start: int, end: int) -> Symbol:
    return Symbol(
        name=name, kind="function", file="m.py", start_line=start, end_line=end
    )


# --------------------------------------------------------------------------- #
# order_symbols_by_difficulty
# --------------------------------------------------------------------------- #
def test_order_prefers_largest_symbols_first() -> None:
    small = _sym("small", 1, 3)  # span 3
    big = _sym("big", 10, 60)  # span 51
    mid = _sym("mid", 70, 85)  # span 16
    ordered = order_symbols_by_difficulty([small, big, mid], prefer=PREFER_LARGEST)
    assert [s.name for s in ordered] == ["big", "mid", "small"]


def test_order_prefers_smallest_symbols_first() -> None:
    small = _sym("small", 1, 3)
    big = _sym("big", 10, 60)
    ordered = order_symbols_by_difficulty([small, big], prefer=PREFER_SMALLEST)
    assert [s.name for s in ordered] == ["small", "big"]


def test_order_filters_below_min_symbol_lines() -> None:
    small = _sym("small", 1, 3)  # span 3
    big = _sym("big", 10, 60)  # span 51
    ordered = order_symbols_by_difficulty(
        [small, big], min_symbol_lines=10, prefer=PREFER_LARGEST
    )
    assert [s.name for s in ordered] == ["big"]


def test_order_parse_is_stable_and_unfiltered_by_default() -> None:
    a = _sym("a", 1, 2)
    b = _sym("b", 3, 40)
    c = _sym("c", 41, 42)
    # Default (parse) preserves the input order and drops nothing.
    ordered = order_symbols_by_difficulty([a, b, c], prefer=PREFER_PARSE)
    assert [s.name for s in ordered] == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# bug_combination threads the difficulty params + records provenance
# --------------------------------------------------------------------------- #
def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# One tiny helper per file plus one LARGER function per file, so the difficulty
# selector has both a "small obvious" and a "large needle" target to choose from.
_PY_BIG = """\
def tiny(x):
    return x + 1


def large(x):
    total = 0
    for i in range(10):
        if i % 2 == 0:
            total += i
        else:
            total -= i
    if x > 0:
        total += x
    else:
        total -= x
    return total
"""


def _amplifier_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "alpha.py", _PY_BIG)
    _write(root, "beta.py", _PY_BIG.replace("tiny", "tiny2").replace("large", "large2"))
    return root


def test_bug_combination_records_difficulty_params(tmp_path: Path) -> None:
    repo = _amplifier_repo(tmp_path)
    candidate = BugCombinationGenerator().generate(
        GenerationRequest(
            repo_root=repo,
            seed=1,
            params={"faults": 2, "min_symbol_lines": 5, "prefer": "largest"},
        ),
        PythonAdapter(),
    )
    details = candidate.provenance.details
    assert details["prefer"] == "largest"
    assert details["min_symbol_lines"] == 5
    assert details["fault_count"] == 2


def test_bug_combination_largest_targets_the_big_symbols(tmp_path: Path) -> None:
    # With min_symbol_lines high enough to exclude the tiny helpers, the amplifier
    # must target the LARGE functions (the needle) in each file.
    repo = _amplifier_repo(tmp_path)
    candidate = BugCombinationGenerator().generate(
        GenerationRequest(
            repo_root=repo,
            seed=1,
            params={"faults": 2, "min_symbol_lines": 5, "prefer": "largest"},
        ),
        PythonAdapter(),
    )
    symbols = set(candidate.target.symbols)
    assert symbols <= {"large", "large2"}
    assert not (symbols & {"tiny", "tiny2"})


def test_bug_combination_smallest_can_target_tiny_symbols(tmp_path: Path) -> None:
    repo = _amplifier_repo(tmp_path)
    candidate = BugCombinationGenerator().generate(
        GenerationRequest(
            repo_root=repo,
            seed=1,
            params={"faults": 2, "min_symbol_lines": 0, "prefer": "smallest"},
        ),
        PythonAdapter(),
    )
    # smallest-first tries the tiny helpers first; at least one tiny is targeted.
    assert set(candidate.target.symbols) & {"tiny", "tiny2"}


def test_bug_combination_difficulty_params_still_round_trip(tmp_path: Path) -> None:
    from swe_forge.forge.adapters._diff import apply_multi_patch

    repo = _amplifier_repo(tmp_path)
    candidate = BugCombinationGenerator().generate(
        GenerationRequest(
            repo_root=repo,
            seed=2,
            params={"faults": 2, "min_symbol_lines": 5, "prefer": "largest"},
        ),
        PythonAdapter(),
    )
    originals = {rel: (repo / rel).read_bytes() for rel in candidate.target.files}
    applied = apply_multi_patch(originals, candidate.mutation_patch)
    restored = apply_multi_patch(applied, candidate.oracle_patch)
    # Mutation changes behavior; oracle restores byte-for-byte.
    assert any(applied[rel] != originals[rel] for rel in originals)
    assert restored == originals


def test_bug_combination_deterministic_for_fixed_params(tmp_path: Path) -> None:
    repo = _amplifier_repo(tmp_path)
    params = {"faults": 2, "min_symbol_lines": 5, "prefer": "largest"}
    first = BugCombinationGenerator().generate(
        GenerationRequest(repo_root=repo, seed=3, params=dict(params)), PythonAdapter()
    )
    second = BugCombinationGenerator().generate(
        GenerationRequest(repo_root=repo, seed=3, params=dict(params)), PythonAdapter()
    )
    assert first.mutation_patch == second.mutation_patch
    assert first.oracle_patch == second.oracle_patch


# --------------------------------------------------------------------------- #
# multi_file honours the SAME amplifier ladder (2nd in-band generator)
# --------------------------------------------------------------------------- #
def test_multi_file_records_difficulty_params(tmp_path: Path) -> None:
    repo = _amplifier_repo(tmp_path)
    candidate = MultiFileGenerator().generate(
        GenerationRequest(
            repo_root=repo,
            seed=1,
            params={"files": 2, "min_symbol_lines": 5, "prefer": "largest"},
        ),
        PythonAdapter(),
    )
    details = candidate.provenance.details
    assert details["prefer"] == "largest"
    assert details["min_symbol_lines"] == 5
    # A large-symbol multi_file is an amplifier -> hard-band calibration budget.
    assert candidate.difficulty_hint == "high"


def test_multi_file_largest_targets_the_big_symbols(tmp_path: Path) -> None:
    repo = _amplifier_repo(tmp_path)
    candidate = MultiFileGenerator().generate(
        GenerationRequest(
            repo_root=repo,
            seed=1,
            params={"files": 2, "min_symbol_lines": 5, "prefer": "largest"},
        ),
        PythonAdapter(),
    )
    symbols = set(candidate.target.symbols)
    assert symbols <= {"large", "large2"}
    assert not (symbols & {"tiny", "tiny2"})


def test_multi_file_defaults_preserve_medium_difficulty(tmp_path: Path) -> None:
    # With no amplifier params the generator keeps its historical medium hint.
    repo = _amplifier_repo(tmp_path)
    candidate = MultiFileGenerator().generate(
        GenerationRequest(repo_root=repo, seed=1, params={}),
        PythonAdapter(),
    )
    assert candidate.difficulty_hint == "medium"
    assert candidate.provenance.details["min_symbol_lines"] == 0


def test_multi_file_difficulty_params_still_round_trip(tmp_path: Path) -> None:
    from swe_forge.forge.adapters._diff import apply_multi_patch

    repo = _amplifier_repo(tmp_path)
    candidate = MultiFileGenerator().generate(
        GenerationRequest(
            repo_root=repo,
            seed=2,
            params={"files": 2, "min_symbol_lines": 5, "prefer": "largest"},
        ),
        PythonAdapter(),
    )
    originals = {rel: (repo / rel).read_bytes() for rel in candidate.target.files}
    applied = apply_multi_patch(originals, candidate.mutation_patch)
    restored = apply_multi_patch(applied, candidate.oracle_patch)
    assert any(applied[rel] != originals[rel] for rel in originals)
    assert restored == originals


# --------------------------------------------------------------------------- #
# build_pilot_plans expands the amplifier ladder
# --------------------------------------------------------------------------- #
def test_pilot_plans_expand_amplifier_across_the_ladder() -> None:
    registry = build_source_registry()
    plans = build_pilot_plans(registry=registry, seeds_per_cell=2)
    amp = [p for p in plans if p.generator in AMPLIFIER_GENERATORS]
    assert amp, "expected amplifier plans"
    # Every ladder rung's param set appears among the amplifier plans.
    seen = {tuple(sorted(p.params.items())) for p in amp}
    for rung in DEFAULT_AMPLIFIER_LADDER:
        assert tuple(sorted(rung.items())) in seen
    # All amplifier plans carry a faults=2 count (the count that clears the oracle
    # cleanly; difficulty is centered via symbol size, not the bimodal count knob).
    assert all(p.params.get("faults") == 2 for p in amp)


def test_pilot_plans_ladder_multiplies_amplifier_cell_count() -> None:
    registry = build_source_registry()
    seeds = 2
    with_ladder = build_pilot_plans(registry=registry, seeds_per_cell=seeds)
    flat = build_pilot_plans(
        registry=registry, seeds_per_cell=seeds, amplifier_ladder=[]
    )
    amp_with = [p for p in with_ladder if p.generator in AMPLIFIER_GENERATORS]
    amp_flat = [p for p in flat if p.generator in AMPLIFIER_GENERATORS]
    # The ladder expands each amplifier (generator, seed) cell into one plan per
    # rung; an empty ladder falls back to a single plan per cell.
    assert len(amp_with) == len(amp_flat) * len(DEFAULT_AMPLIFIER_LADDER)


def test_pilot_plans_non_amplifier_generators_unchanged() -> None:
    registry = build_source_registry()
    seeds = 2
    plans = build_pilot_plans(registry=registry, seeds_per_cell=seeds)
    # Non-amplifier structural generators keep one plan per (generator, seed) cell
    # with no injected difficulty params.
    for p in plans:
        if p.generator not in AMPLIFIER_GENERATORS and p.generator != "pr_mirror":
            assert p.params == {}
