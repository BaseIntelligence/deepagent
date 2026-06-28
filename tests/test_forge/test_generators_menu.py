"""Unit tests for the cross-generator self-validation menu (m3-gen-menu).

Covers the feature's ``fulfills`` set:

* VAL-GEN-001 - every generator emits a verifiable forward+inverse pair; the
  menu independently re-derives the round-trip (byte-for-byte restore) for all
  six.
* VAL-GEN-002 - the forward mutation is behavior-changing, never
  whitespace/comment/import-only; such an edit yields no Candidate.
* VAL-GEN-007 - a ``coverage[generator][language]`` matrix is recorded with no
  empty Python column (LLM-backed generators carry a cost-exemption flag).
* VAL-GEN-008 - the Candidate schema is complete for every generator with a
  known generator name.
* VAL-GEN-010 - invalid generation fails cleanly and emits NO Candidate.

Fully offline: the LLM-backed generators run with deterministic stubs and Docker
is not used (behavior change is checked at the token-normalized level).
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_forge.forge.adapters._diff import make_patch
from swe_forge.forge.adapters.base import Symbol
from swe_forge.forge.cli import app as forge_app
from swe_forge.forge.generators import (
    BugGenerator,
    GenerationRequest,
    build_coverage_matrix,
    evaluate_cell,
    evaluate_coverage,
    run_menu_selfcheck,
    schema_completeness,
    verify_candidate_roundtrip,
)
from swe_forge.forge.generators import menu as menu_mod
from swe_forge.forge.generators._normalize import (
    is_behavior_changing,
    normalize_behavior,
)
from swe_forge.forge.generators._targeting import verify_forward_patch
from swe_forge.forge.generators.menu import MenuCellSpec
from swe_forge.forge.models import (
    GENERATOR_NAMES,
    Candidate,
    CandidateTarget,
    Provenance,
)

runner = CliRunner()


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Behavior-change normalizer (VAL-GEN-002)
# --------------------------------------------------------------------------- #
def test_normalize_behavior_drops_whitespace_comments_imports() -> None:
    a = "import os\ndef f():\n    return 1  # note\n"
    b = "def f():\n\n        return 1\n"
    assert normalize_behavior(a) == normalize_behavior(b)


def test_is_behavior_changing_true_for_real_change() -> None:
    assert is_behavior_changing("return a + b", "return a - b")


@pytest.mark.parametrize(
    "original,mutated",
    [
        ("def f():\n    return 1\n", "def f():\n\n    return 1\n"),  # blank line
        ("def f():\n    return 1\n", "def f():\n    return 1  # x\n"),  # comment
        ("def f():\n    return 1\n", "import sys\ndef f():\n    return 1\n"),  # import
    ],
)
def test_is_behavior_changing_false_for_trivia(original: str, mutated: str) -> None:
    assert not is_behavior_changing(original, mutated)


def test_chokepoint_rejects_whitespace_only_forward_patch(tmp_path: Path) -> None:
    rel = "f.py"
    original = "def f():\n    return 1\n"
    _write(tmp_path, rel, original)
    mutated = "def f():\n\n    return 1\n"  # blank line only - no behavior change
    symbol = Symbol(name="f", kind="function", file=rel, start_line=1, end_line=2)
    forward = type(
        "P", (), {"diff": make_patch(rel, original.encode(), mutated.encode())}
    )()
    with contextlib.chdir(tmp_path):
        # The shared single-file chokepoint disposes a non-behavior-changing edit.
        assert verify_forward_patch(rel, symbol, forward) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Full menu self-check (VAL-GEN-001 / 007 / 008)
# --------------------------------------------------------------------------- #
def test_menu_selfcheck_passes_for_all_six_generators(tmp_path: Path) -> None:
    report = run_menu_selfcheck(tmp_path, seed=0)
    assert report.ok, report.reasons
    # Every generator is present in the matrix...
    assert set(report.coverage) == set(GENERATOR_NAMES)
    # ...and each cell that ran round-trips, is behavior-changing, schema-complete.
    assert report.cells
    for cell in report.cells:
        assert cell.ok, (cell.generator, cell.language, cell.reason)
        assert cell.roundtrip_ok
        assert cell.behavior_changing
        assert cell.schema_complete


def test_menu_each_candidate_roundtrips_byte_for_byte(tmp_path: Path) -> None:
    report = run_menu_selfcheck(tmp_path, seed=0)
    for cell in report.cells:
        assert cell.candidate is not None
        for rel, digests in cell.file_sha256.items():
            # The oracle restored every touched file to its original digest.
            assert digests["restored"] == digests["original"], (cell.generator, rel)
            assert digests["mutated"] != digests["original"]


def test_menu_coverage_no_empty_python_column(tmp_path: Path) -> None:
    report = run_menu_selfcheck(tmp_path, seed=0)
    for name in GENERATOR_NAMES:
        languages = report.coverage[name]
        assert "python" in languages, f"{name} missing a Python coverage entry"
        assert languages["python"]["ok"], f"{name} Python coverage not ok"


def test_menu_coverage_non_python_or_llm_exempt(tmp_path: Path) -> None:
    report = run_menu_selfcheck(tmp_path, seed=0)
    for name in GENERATOR_NAMES:
        languages = report.coverage[name]
        non_python_ok = any(
            lang != "python" and entry["ok"] for lang, entry in languages.items()
        )
        llm_backed = any(entry["llm_backed"] for entry in languages.values())
        assert non_python_ok or llm_backed, (
            f"{name} lacks non-Python coverage/exemption"
        )


def test_menu_llm_backed_generators_flagged(tmp_path: Path) -> None:
    report = run_menu_selfcheck(tmp_path, seed=0)
    for name in ("lm_authored", "pr_mirror"):
        assert report.coverage[name]["python"]["llm_backed"] is True
    for name in ("ast_mutation", "function_removal", "multi_file", "bug_combination"):
        assert report.coverage[name]["python"]["llm_backed"] is False


def test_menu_candidate_schema_complete_for_every_cell(tmp_path: Path) -> None:
    report = run_menu_selfcheck(tmp_path, seed=0)
    for cell in report.cells:
        assert cell.candidate is not None
        ok, reason = schema_completeness(cell.candidate)
        assert ok, (cell.generator, cell.language, reason)
        data = cell.candidate.to_dict()
        assert data["generator"] in GENERATOR_NAMES
        for key in (
            "language",
            "target",
            "mutation_patch",
            "oracle_patch",
            "difficulty_hint",
        ):
            assert data[key]


def test_menu_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    report_a = run_menu_selfcheck(tmp_path / "a", seed=5)
    report_b = run_menu_selfcheck(tmp_path / "b", seed=5)
    sha_a = {
        (c.generator, c.language): c.file_sha256
        for c in report_a.cells
        if not c.llm_backed  # deterministic generators only
    }
    sha_b = {
        (c.generator, c.language): c.file_sha256
        for c in report_b.cells
        if not c.llm_backed
    }
    assert sha_a == sha_b


# --------------------------------------------------------------------------- #
# Independent round-trip + schema re-verification helpers
# --------------------------------------------------------------------------- #
def test_verify_candidate_roundtrip_ok(tmp_path: Path) -> None:
    rel = "src/calc.py"
    original = "def add(a, b):\n    return a + b\n"
    _write(tmp_path, rel, original)
    mutated = "def add(a, b):\n    return a - b\n"
    candidate = Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=(rel,), symbols=("add",)),
        mutation_patch=make_patch(rel, original.encode(), mutated.encode()),
        oracle_patch=make_patch(rel, mutated.encode(), original.encode()),
        difficulty_hint="low",
        provenance=Provenance(generator="ast_mutation", seed=0, language="python"),
    )
    result = verify_candidate_roundtrip(tmp_path, candidate)
    assert result.ok
    assert result.behavior_changing
    assert result.file_sha256[rel]["restored"] == result.file_sha256[rel]["original"]


def test_verify_candidate_roundtrip_flags_non_inverting_oracle(tmp_path: Path) -> None:
    rel = "src/calc.py"
    original = "def add(a, b):\n    return a + b\n"
    _write(tmp_path, rel, original)
    mutated = "def add(a, b):\n    return a - b\n"
    forward = make_patch(rel, original.encode(), mutated.encode())
    candidate = Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=(rel,), symbols=("add",)),
        mutation_patch=forward,
        oracle_patch=forward,  # NOT the inverse: applying it again won't restore
        difficulty_hint="low",
        provenance=Provenance(generator="ast_mutation", seed=0, language="python"),
    )
    result = verify_candidate_roundtrip(tmp_path, candidate)
    assert not result.ok
    assert "oracle" in result.reason.lower()


def test_verify_candidate_roundtrip_flags_behavior_non_changing(tmp_path: Path) -> None:
    rel = "f.py"
    original = "def f():\n    return 1\n"
    _write(tmp_path, rel, original)
    mutated = "def f():\n\n    return 1\n"  # whitespace only
    candidate = Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=(rel,), symbols=("f",)),
        mutation_patch=make_patch(rel, original.encode(), mutated.encode()),
        oracle_patch=make_patch(rel, mutated.encode(), original.encode()),
        difficulty_hint="low",
        provenance=Provenance(generator="ast_mutation", seed=0, language="python"),
    )
    result = verify_candidate_roundtrip(tmp_path, candidate)
    assert not result.ok
    assert not result.behavior_changing


# --------------------------------------------------------------------------- #
# Failure path: a generator that fails self-validation (VAL-GEN-010)
# --------------------------------------------------------------------------- #
class _BadOracleGenerator(BugGenerator):
    """Emits a Candidate whose oracle does NOT restore (broken round-trip)."""

    name = "ast_mutation"

    def generate(self, request: GenerationRequest, adapter) -> Candidate:
        rel = "src/calc.py"
        original = (Path(request.repo_root) / rel).read_bytes()
        mutated = original.replace(b"a + b", b"a - b")
        forward = make_patch(rel, original, mutated)
        return Candidate(
            language="python",
            generator="ast_mutation",
            target=CandidateTarget(files=(rel,), symbols=("add",)),
            mutation_patch=forward,
            oracle_patch=forward,  # wrong direction -> will not restore
            difficulty_hint="low",
            provenance=Provenance(generator="ast_mutation", seed=0, language="python"),
        )


def _bad_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "src/calc.py", "def add(a, b):\n    return a + b\n")
    return root


def test_menu_cell_with_broken_roundtrip_fails(tmp_path: Path) -> None:
    spec = MenuCellSpec(
        _BadOracleGenerator(), "python", False, _bad_repo, {"file": "src/calc.py"}
    )
    cell = evaluate_cell(spec, tmp_path, seed=0)
    assert not cell.ok
    assert not cell.roundtrip_ok
    assert "oracle" in cell.reason.lower()


def test_menu_selfcheck_aborts_when_a_generator_fails(tmp_path: Path) -> None:
    spec = MenuCellSpec(
        _BadOracleGenerator(), "python", False, _bad_repo, {"file": "src/calc.py"}
    )
    report = run_menu_selfcheck(tmp_path, seed=0, specs=[spec])
    assert not report.ok
    assert any("ast_mutation" in r for r in report.reasons)


def test_evaluate_coverage_flags_missing_python_column() -> None:
    # A single non-Python ok cell leaves every generator's Python column empty.
    from swe_forge.forge.generators.menu import CellResult

    cell = CellResult(
        generator="ast_mutation",
        language="javascript",
        ok=True,
        llm_backed=False,
        reason="",
        files=("calc.js",),
        roundtrip_ok=True,
        behavior_changing=True,
        schema_complete=True,
    )
    ok, reasons = evaluate_coverage([cell])
    assert not ok
    assert any("empty Python column" in r for r in reasons)


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_cli_gen_menu_writes_coverage_and_candidates(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        forge_app, ["gen-menu", "--out", str(out), "--seed", "1", "--json"]
    )
    assert result.exit_code == 0, result.output
    coverage = json.loads((out / "coverage.json").read_text())
    assert coverage["ok"] is True
    # No empty Python column.
    for name in GENERATOR_NAMES:
        assert coverage["coverage"][name]["python"]["ok"]
    # Per-cell candidate artifacts exist for at least the Python cells.
    for name in GENERATOR_NAMES:
        cell_dir = out / "candidates" / f"{name}__python"
        assert (cell_dir / "candidate.json").is_file()
        assert (cell_dir / "mutation.patch").is_file()
        assert (cell_dir / "oracle.patch").is_file()


def test_cli_gen_menu_failure_writes_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_spec = MenuCellSpec(
        _BadOracleGenerator(), "python", False, _bad_repo, {"file": "src/calc.py"}
    )
    monkeypatch.setattr(menu_mod, "build_menu_cell_specs", lambda **_: [bad_spec])
    out = tmp_path / "out"
    result = runner.invoke(forge_app, ["gen-menu", "--out", str(out)])
    assert result.exit_code == 1
    assert "gen-menu" in result.output
    # No artifact written on the failure path.
    assert not (out / "coverage.json").exists()
    assert not (out / "candidates").exists()


def test_cli_gen_menu_no_go_flag(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        forge_app, ["gen-menu", "--out", str(out), "--no-go", "--json"]
    )
    assert result.exit_code == 0, result.output
    coverage = json.loads(result.output)
    # With --no-go, no Go cells are recorded.
    for languages in coverage["coverage"].values():
        assert "go" not in languages


def test_coverage_matrix_builder_groups_by_generator_then_language() -> None:
    from swe_forge.forge.generators.menu import CellResult

    cells = [
        CellResult(
            "ast_mutation", "python", True, False, "", ("a.py",), True, True, True
        ),
        CellResult(
            "ast_mutation", "javascript", True, False, "", ("a.js",), True, True, True
        ),
    ]
    matrix = build_coverage_matrix(cells)
    assert set(matrix["ast_mutation"]) == {"python", "javascript"}
    assert matrix["ast_mutation"]["python"]["ok"]
