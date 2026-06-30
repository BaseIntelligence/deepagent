"""Unit tests for the structural generators (m3-gen-structural).

Covers VAL-GEN-003 (``bug_combination``: >=2 independent faults, single-fault
revert still fails, full oracle restores), VAL-GEN-004 (``multi_file``: >=2
distinct files, ``target.files`` matches, full round-trip) and VAL-GEN-006
(``function_removal``: body removed/stubbed with the signature kept, exact
reconstruction across languages), plus the shared schema/round-trip/determinism
contract (VAL-GEN-001/002/008/009).

The tests are fully offline (no Docker, no live LLM). Round-trip is checked at the
byte level (sha256); behavior change is checked by importing the mutated module
and confirming the fault flips the function's observable result. The Go paths
need the host ``go`` toolchain for ``parse_symbols`` and skip without it; Python
and JS satisfy the >=2-language requirement on their own.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from swe_forge.forge.adapters import GoAdapter, JavaScriptAdapter, PythonAdapter
from swe_forge.forge.adapters._diff import apply_multi_patch, apply_patch
from swe_forge.forge.adapters._goast import GoToolchainError
from swe_forge.forge.cli import app as forge_app
from swe_forge.forge.generators import (
    BugCombinationGenerator,
    FunctionRemovalGenerator,
    GenerationError,
    GenerationRequest,
    MultiFileGenerator,
    build_default_generator_registry,
)
from swe_forge.forge.models import (
    GENERATOR_NAMES,
    BaselineNotGreenError,
    Candidate,
    EnvImage,
)

runner = CliRunner()
_LOAD_COUNTER = itertools.count()


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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _load_module(source: bytes, *, suffix: str = ".py") -> ModuleType:
    """Import ``source`` as a fresh throwaway module and return it."""
    import tempfile

    name = f"_forge_fixture_{next(_LOAD_COUNTER)}"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as handle:
        handle.write(source)
        path = handle.name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Fixtures: single-file (removal) and multi-file (combination) Python repos
# --------------------------------------------------------------------------- #
PY_CLASSIFY = """\
def classify(n):
    if n < 0:
        return "negative"
    if n == 0:
        return "zero"
    return "positive"
"""

JS_CLASSIFY = """\
function classify(n) {
  if (n < 0) {
    return "negative";
  }
  return "positive";
}
"""

GO_CLASSIFY = """\
package calc

func Classify(n int) string {
\tif n < 0 {
\t\treturn "negative"
\t}
\treturn "positive"
}
"""

PY_ALPHA = "def scale(x):\n    return x * 2\n"
PY_BETA = "def shift(x):\n    return x + 10\n"
PY_TEST = (
    "from alpha import scale\n"
    "from beta import shift\n\n"
    "def test_scale():\n    assert scale(3) == 6\n\n"
    "def test_shift():\n    assert shift(3) == 13\n"
)


def _removal_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "src/calc.py", PY_CLASSIFY)
    return root


def _multi_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "alpha.py", PY_ALPHA)
    _write(root, "beta.py", PY_BETA)
    _write(root, "test_mod.py", PY_TEST)
    return root


# --------------------------------------------------------------------------- #
# function_removal (VAL-GEN-006)
# --------------------------------------------------------------------------- #
def test_function_removal_python_roundtrips_and_keeps_signature(tmp_path: Path) -> None:
    repo = _removal_repo(tmp_path)
    gen = FunctionRemovalGenerator()
    request = GenerationRequest(repo_root=repo, seed=0, file="src/calc.py")
    candidate = gen.generate(request, PythonAdapter())

    assert candidate.generator == "function_removal"
    assert candidate.language == "python"
    assert candidate.target.files == ("src/calc.py",)
    assert candidate.provenance.details["operation"] == "function_removal"

    rel = "src/calc.py"
    original = (repo / rel).read_bytes()
    mutated = apply_patch(original, candidate.mutation_patch, rel)
    assert mutated != original
    # Signature line kept verbatim; body replaced by a stub.
    assert b"def classify(n):" in mutated
    assert b"raise NotImplementedError" in mutated
    assert b'return "negative"' not in mutated
    # Exact reconstruction (sha256 match).
    restored = apply_patch(mutated, candidate.oracle_patch, rel)
    assert _sha(restored) == _sha(original)


def test_function_removal_python_changes_behavior(tmp_path: Path) -> None:
    repo = _removal_repo(tmp_path)
    candidate = FunctionRemovalGenerator().generate(
        GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"), PythonAdapter()
    )
    rel = "src/calc.py"
    original = (repo / rel).read_bytes()
    mutated = apply_patch(original, candidate.mutation_patch, rel)

    assert _load_module(original).classify(-1) == "negative"
    broken = _load_module(mutated)
    with pytest.raises(NotImplementedError):
        broken.classify(-1)


def test_function_removal_javascript_roundtrips(tmp_path: Path) -> None:
    _write(tmp_path, "package.json", '{"name":"demo","version":"0.0.0"}')
    _write(tmp_path, "calc.js", JS_CLASSIFY)
    candidate = FunctionRemovalGenerator().generate(
        GenerationRequest(repo_root=tmp_path, seed=0, file="calc.js"),
        JavaScriptAdapter(),
    )
    assert candidate.language == "javascript"
    rel = "calc.js"
    original = (tmp_path / rel).read_bytes()
    mutated = apply_patch(original, candidate.mutation_patch, rel)
    assert b"function classify(n)" in mutated
    assert b'throw new Error("not implemented")' in mutated
    restored = apply_patch(mutated, candidate.oracle_patch, rel)
    assert _sha(restored) == _sha(original)


@requires_go
def test_function_removal_go_roundtrips(tmp_path: Path) -> None:
    _write(tmp_path, "go.mod", "module demo\n\ngo 1.22\n")
    _write(tmp_path, "calc.go", GO_CLASSIFY)
    candidate = FunctionRemovalGenerator().generate(
        GenerationRequest(repo_root=tmp_path, seed=0, file="calc.go"), GoAdapter()
    )
    assert candidate.language == "go"
    rel = "calc.go"
    original = (tmp_path / rel).read_bytes()
    mutated = apply_patch(original, candidate.mutation_patch, rel)
    assert b"func Classify(n int) string" in mutated
    assert b'panic("not implemented")' in mutated
    restored = apply_patch(mutated, candidate.oracle_patch, rel)
    assert _sha(restored) == _sha(original)


def test_function_removal_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    repo = _removal_repo(tmp_path)
    gen = FunctionRemovalGenerator()
    first = gen.generate(
        GenerationRequest(repo_root=repo, seed=7, file="src/calc.py"), PythonAdapter()
    )
    second = gen.generate(
        GenerationRequest(repo_root=repo, seed=7, file="src/calc.py"), PythonAdapter()
    )
    assert first.mutation_patch == second.mutation_patch
    assert first.oracle_patch == second.oracle_patch
    assert first.provenance.seed == 7


def test_function_removal_gates_on_green_baseline(tmp_path: Path) -> None:
    repo = _removal_repo(tmp_path)
    red = EnvImage(
        repo_id="demo",
        language="python",
        image_tag="demo:red",
        base_image="python:3.12-slim",
        commit="0" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command="python -m pytest",
        baseline_green=False,
        baseline_exit_code=1,
    )
    with pytest.raises(BaselineNotGreenError):
        FunctionRemovalGenerator().generate(
            GenerationRequest(
                repo_root=repo, seed=0, file="src/calc.py", env_image=red
            ),
            PythonAdapter(),
        )


def test_function_removal_no_removable_function_raises(tmp_path: Path) -> None:
    _write(tmp_path, "empty.py", "X = 1\nY = 2\n")
    with pytest.raises(GenerationError):
        FunctionRemovalGenerator().generate(
            GenerationRequest(repo_root=tmp_path, seed=0, file="empty.py"),
            PythonAdapter(),
        )


# --------------------------------------------------------------------------- #
# multi_file (VAL-GEN-004)
# --------------------------------------------------------------------------- #
def _count_diff_sections(patch: str) -> int:
    return patch.count("diff --git ")


def test_multi_file_touches_two_files_and_target_matches(tmp_path: Path) -> None:
    repo = _multi_repo(tmp_path)
    candidate = MultiFileGenerator().generate(
        GenerationRequest(repo_root=repo, seed=1), PythonAdapter()
    )
    assert candidate.generator == "multi_file"
    files = set(candidate.target.files)
    assert files == {"alpha.py", "beta.py"}
    assert "test_mod.py" not in files
    assert _count_diff_sections(candidate.mutation_patch) >= 2
    # Each edit records its enclosing-symbol line span (mutation-gate scoping).
    edits = candidate.provenance.details["edits"]
    assert all(
        isinstance(e["start_line"], int) and e["end_line"] >= e["start_line"] >= 1
        for e in edits
    )


def test_multi_file_roundtrips_every_file(tmp_path: Path) -> None:
    repo = _multi_repo(tmp_path)
    candidate = MultiFileGenerator().generate(
        GenerationRequest(repo_root=repo, seed=1), PythonAdapter()
    )
    originals = {rel: (repo / rel).read_bytes() for rel in candidate.target.files}
    mutated = apply_multi_patch(originals, candidate.mutation_patch)
    assert all(mutated[rel] != originals[rel] for rel in originals)
    restored = apply_multi_patch(mutated, candidate.oracle_patch)
    for rel in originals:
        assert _sha(restored[rel]) == _sha(originals[rel])


def test_multi_file_requires_two_source_files(tmp_path: Path) -> None:
    _write(tmp_path, "only.py", PY_ALPHA)
    _write(tmp_path, "test_only.py", "from only import scale\n")
    with pytest.raises(GenerationError, match="distinct"):
        MultiFileGenerator().generate(
            GenerationRequest(repo_root=tmp_path, seed=0), PythonAdapter()
        )


def test_multi_file_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    repo = _multi_repo(tmp_path)
    gen = MultiFileGenerator()
    first = gen.generate(GenerationRequest(repo_root=repo, seed=3), PythonAdapter())
    second = gen.generate(GenerationRequest(repo_root=repo, seed=3), PythonAdapter())
    assert first.mutation_patch == second.mutation_patch
    assert first.oracle_patch == second.oracle_patch


# --------------------------------------------------------------------------- #
# bug_combination (VAL-GEN-003)
# --------------------------------------------------------------------------- #
def test_bug_combination_encodes_two_distinct_faults(tmp_path: Path) -> None:
    repo = _multi_repo(tmp_path)
    candidate = BugCombinationGenerator().generate(
        GenerationRequest(repo_root=repo, seed=2), PythonAdapter()
    )
    assert candidate.generator == "bug_combination"
    faults = candidate.provenance.details["faults"]
    assert isinstance(faults, list) and len(faults) >= 2
    # Distinct symbols across distinct files; non-adjacent hunks.
    assert len({f["file"] for f in faults}) >= 2
    assert len({f["symbol"] for f in faults}) >= 2
    assert _count_diff_sections(candidate.mutation_patch) >= 2
    # Each fault records its enclosing-symbol line span so the mutation gate can
    # scope cosmic-ray to the changed region (m6-pilot-difficulty).
    for fault in faults:
        assert isinstance(fault["start_line"], int)
        assert isinstance(fault["end_line"], int)
        assert fault["end_line"] >= fault["start_line"] >= 1


def test_bug_combination_full_oracle_restores_byte_for_byte(tmp_path: Path) -> None:
    repo = _multi_repo(tmp_path)
    candidate = BugCombinationGenerator().generate(
        GenerationRequest(repo_root=repo, seed=2), PythonAdapter()
    )
    originals = {rel: (repo / rel).read_bytes() for rel in candidate.target.files}
    mutated = apply_multi_patch(originals, candidate.mutation_patch)
    restored = apply_multi_patch(mutated, candidate.oracle_patch)
    for rel in originals:
        assert _sha(restored[rel]) == _sha(originals[rel])


def test_bug_combination_single_fault_revert_leaves_a_fault(tmp_path: Path) -> None:
    repo = _multi_repo(tmp_path)
    candidate = BugCombinationGenerator().generate(
        GenerationRequest(repo_root=repo, seed=2), PythonAdapter()
    )
    originals = {rel: (repo / rel).read_bytes() for rel in candidate.target.files}
    broken = apply_multi_patch(originals, candidate.mutation_patch)
    faults = candidate.provenance.details["faults"]

    for fault in faults:
        rel = fault["file"]
        # Reverting just this fault restores its file...
        reverted = apply_patch(broken[rel], fault["single_fault_revert"], rel)
        assert _sha(reverted) == _sha(originals[rel])
        # ...while every other fault's file stays broken (>= 1 failing test).
        others = [other for other in candidate.target.files if other != rel]
        assert others
        assert all(broken[other] != originals[other] for other in others)


def test_bug_combination_single_fault_revert_changes_behavior(tmp_path: Path) -> None:
    repo = _multi_repo(tmp_path)
    candidate = BugCombinationGenerator().generate(
        GenerationRequest(repo_root=repo, seed=2), PythonAdapter()
    )
    originals = {rel: (repo / rel).read_bytes() for rel in candidate.target.files}
    broken = apply_multi_patch(originals, candidate.mutation_patch)

    # Both functions misbehave on the fully-broken tree.
    assert _load_module(broken["alpha.py"]).scale(3) != 6
    assert _load_module(broken["beta.py"]).shift(3) != 13
    # The gold oracle restores correct behavior for both.
    restored = apply_multi_patch(broken, candidate.oracle_patch)
    assert _load_module(restored["alpha.py"]).scale(3) == 6
    assert _load_module(restored["beta.py"]).shift(3) == 13


def test_bug_combination_requires_two_files(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "only.py",
        "def a(x):\n    return x + 1\ndef b(x):\n    return x - 1\n",
    )
    with pytest.raises(GenerationError, match="distinct"):
        BugCombinationGenerator().generate(
            GenerationRequest(repo_root=tmp_path, seed=0), PythonAdapter()
        )


# --------------------------------------------------------------------------- #
# Registry + schema completeness (VAL-GEN-001/008)
# --------------------------------------------------------------------------- #
def test_registry_holds_all_structural_generators() -> None:
    registry = build_default_generator_registry()
    for name in ("ast_mutation", "function_removal", "multi_file", "bug_combination"):
        assert name in registry.names()
        assert registry.get(name).name == name


@pytest.mark.parametrize(
    "build",
    [
        lambda repo: FunctionRemovalGenerator().generate(
            GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"),
            PythonAdapter(),
        ),
    ],
)
def test_removal_candidate_schema_complete(tmp_path: Path, build) -> None:
    repo = _removal_repo(tmp_path)
    candidate = build(repo)
    _assert_schema_complete(candidate)


@pytest.mark.parametrize("generator", [MultiFileGenerator(), BugCombinationGenerator()])
def test_multi_candidate_schema_complete(tmp_path: Path, generator) -> None:
    repo = _multi_repo(tmp_path)
    candidate = generator.generate(
        GenerationRequest(repo_root=repo, seed=1), PythonAdapter()
    )
    _assert_schema_complete(candidate)


def _assert_schema_complete(candidate: Candidate) -> None:
    data = candidate.to_dict()
    assert data["language"]
    assert data["generator"] in GENERATOR_NAMES
    assert data["target"]["files"]
    assert data["mutation_patch"].strip()
    assert data["oracle_patch"].strip()
    assert data["difficulty_hint"]
    # Serializable round-trip.
    restored = Candidate.from_dict(json.loads(json.dumps(data)))
    assert restored.to_dict() == data


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "generator,repo_builder",
    [
        ("function_removal", _removal_repo),
        ("multi_file", _multi_repo),
        ("bug_combination", _multi_repo),
    ],
)
def test_cli_generate_structural(tmp_path: Path, generator, repo_builder) -> None:
    repo = repo_builder(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        forge_app,
        [
            "generate",
            "--path",
            str(repo),
            "--generator",
            generator,
            "--seed",
            "1",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "candidate.json").is_file()
    assert (out / "mutation.patch").is_file()
    assert (out / "oracle.patch").is_file()
    payload = json.loads(result.output)
    assert payload["generator"] == generator
