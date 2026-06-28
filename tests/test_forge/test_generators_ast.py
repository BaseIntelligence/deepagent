"""Unit tests for the ``ast_mutation`` generator and adapter ``mutate_ast``.

Covers the m3-gen-ast contract (VAL-GEN-005, VAL-GEN-009) plus the Candidate
data-model and the ``swe-forge forge generate`` CLI surface:

* ``mutate_ast`` produces round-tripping, behavior-changing patches for
  operator-swap / off-by-one / branch-removal across Python, JS/TS, and Go.
* Applying the mutation then the oracle restores every touched file
  byte-for-byte (sha256 match) and each patch applies cleanly with ``git``.
* The generator is deterministic: the same repo+target+seed yields
  byte-identical ``mutation_patch``/``oracle_patch`` and records the seed.
* Invalid generation raises and the CLI writes NO candidate artifact.

These tests are fully offline (no Docker, no live LLM). The Go *generator* path
needs the host ``go`` toolchain for ``parse_symbols`` and skips without it; Go
``mutate_ast`` is exercised directly via tree-sitter (no toolchain needed).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_forge.forge.adapters import (
    GoAdapter,
    JavaScriptAdapter,
    PythonAdapter,
)
from swe_forge.forge.adapters._diff import apply_patch, make_patch
from swe_forge.forge.adapters._goast import GoToolchainError
from swe_forge.forge.adapters.base import MutationOp, Symbol
from swe_forge.forge.cli import app as forge_app
from swe_forge.forge.generators import (
    AstMutationGenerator,
    GenerationError,
    GenerationRequest,
    build_default_generator_registry,
)
from swe_forge.forge.models import (
    GENERATOR_NAMES,
    BaselineNotGreenError,
    Candidate,
    CandidateTarget,
    EnvImage,
    ModelError,
    Provenance,
)

runner = CliRunner()


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

_OPS = (
    MutationOp.OPERATOR_SWAP,
    MutationOp.OFF_BY_ONE,
    MutationOp.BRANCH_REMOVAL,
)

PY_SOURCE = """\
def add(a, b):
    if a < b:
        return a + b
    return a - 1
"""

JS_SOURCE = """\
function add(a, b) {
  if (a < b) {
    return a + b;
  }
  return a - 1;
}
"""

GO_SOURCE = """\
package calc

func Add(a, b int) int {
\tif a < b {
\t\treturn a + b
\t}
\treturn a - 1
}
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _roundtrip_ok(rel: str, original: bytes, mutation: str) -> bool:
    """Apply ``mutation`` then its inverse oracle; return True on byte-for-byte restore."""
    mutated = apply_patch(original, mutation, rel)
    assert mutated != original
    oracle = make_patch(rel, mutated, original)
    restored = apply_patch(mutated, oracle, rel)
    return _sha(restored) == _sha(original)


# --------------------------------------------------------------------------- #
# Adapter mutate_ast: per-language round-trip + behavior change (VAL-GEN-005)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", _OPS)
def test_python_mutate_ast_roundtrips(tmp_path: Path, op: MutationOp) -> None:
    rel = "calc.py"
    _write(tmp_path, rel, PY_SOURCE)
    adapter = PythonAdapter()
    with contextlib.chdir(tmp_path):
        (symbol,) = [s for s in adapter.parse_symbols(rel) if s.name == "add"]
        patch = adapter.mutate_ast(rel, symbol, op)
        assert patch.diff.strip(), f"{op} produced no mutation"
        assert patch.files == (rel,)
        original = Path(rel).read_bytes()
        assert _roundtrip_ok(rel, original, patch.diff)


@pytest.mark.parametrize("op", _OPS)
def test_javascript_mutate_ast_roundtrips(tmp_path: Path, op: MutationOp) -> None:
    rel = "calc.js"
    _write(tmp_path, rel, JS_SOURCE)
    adapter = JavaScriptAdapter()
    with contextlib.chdir(tmp_path):
        (symbol,) = [s for s in adapter.parse_symbols(rel) if s.name == "add"]
        patch = adapter.mutate_ast(rel, symbol, op)
        assert patch.diff.strip(), f"{op} produced no mutation"
        original = Path(rel).read_bytes()
        assert _roundtrip_ok(rel, original, patch.diff)


def test_typescript_mutate_ast_roundtrips(tmp_path: Path) -> None:
    rel = "calc.ts"
    _write(
        tmp_path,
        rel,
        "export function add(a: number, b: number) {\n  return a + b;\n}\n",
    )
    adapter = JavaScriptAdapter()
    with contextlib.chdir(tmp_path):
        (symbol,) = [s for s in adapter.parse_symbols(rel) if s.name == "add"]
        patch = adapter.mutate_ast(rel, symbol, MutationOp.OPERATOR_SWAP)
        assert patch.diff.strip()
        original = Path(rel).read_bytes()
        assert _roundtrip_ok(rel, original, patch.diff)


@pytest.mark.parametrize("op", _OPS)
def test_go_mutate_ast_roundtrips(tmp_path: Path, op: MutationOp) -> None:
    # Go mutate_ast uses tree-sitter-go (no Go toolchain needed); the symbol span
    # is supplied directly so parse_symbols' Go helper is not required here.
    rel = "calc.go"
    _write(tmp_path, rel, GO_SOURCE)
    adapter = GoAdapter()
    symbol = Symbol(name="Add", kind="function", file=rel, start_line=3, end_line=9)
    with contextlib.chdir(tmp_path):
        patch = adapter.mutate_ast(rel, symbol, op)
        assert patch.diff.strip(), f"{op} produced no mutation"
        original = Path(rel).read_bytes()
        assert _roundtrip_ok(rel, original, patch.diff)


def test_mutate_ast_no_site_returns_empty_patch(tmp_path: Path) -> None:
    rel = "noop.py"
    _write(tmp_path, rel, "def f():\n    return\n")
    adapter = PythonAdapter()
    with contextlib.chdir(tmp_path):
        (symbol,) = adapter.parse_symbols(rel)
        patch = adapter.mutate_ast(rel, symbol, MutationOp.OPERATOR_SWAP)
        assert patch.diff == ""
        assert patch.files == ()


# --------------------------------------------------------------------------- #
# Generator: Candidate emission, round-trip self-validation, schema
# --------------------------------------------------------------------------- #
def _generate(repo: Path, *, seed: int = 0, **kwargs: object) -> Candidate:
    adapter = PythonAdapter()
    gen = AstMutationGenerator()
    request = GenerationRequest(repo_root=repo, seed=seed, **kwargs)  # type: ignore[arg-type]
    return gen.generate(request, adapter)


def _py_repo(tmp_path: Path) -> Path:
    _write(tmp_path, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(tmp_path, "src/calc.py", PY_SOURCE)
    return tmp_path


def test_generator_emits_complete_candidate(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _generate(repo, file="src/calc.py", symbol="add", op="operator_swap")
    assert candidate.generator == "ast_mutation"
    assert candidate.generator in GENERATOR_NAMES
    assert candidate.language == "python"
    assert candidate.target.files == ("src/calc.py",)
    assert candidate.target.symbol == "add"
    assert candidate.mutation_patch.strip()
    assert candidate.oracle_patch.strip()
    assert candidate.difficulty_hint
    assert candidate.provenance.seed == 0
    assert candidate.provenance.details["operator"] == "operator_swap"


def test_generator_roundtrip_restores_byte_for_byte(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _generate(repo, file="src/calc.py", symbol="add")
    rel = "src/calc.py"
    original = (repo / rel).read_bytes()
    assert _roundtrip_ok(rel, original, candidate.mutation_patch)


def test_generator_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    first = _generate(repo, seed=42, file="src/calc.py")
    second = _generate(repo, seed=42, file="src/calc.py")
    assert first.mutation_patch == second.mutation_patch
    assert first.oracle_patch == second.oracle_patch
    assert first.provenance.seed == 42


def test_generator_auto_discovers_target(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _generate(repo, seed=1)
    assert candidate.target.files[0].endswith("calc.py")
    rel = candidate.target.files[0]
    original = (repo / rel).read_bytes()
    assert _roundtrip_ok(rel, original, candidate.mutation_patch)


def test_generator_unknown_op_raises(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    with pytest.raises(GenerationError, match="unknown operator"):
        _generate(repo, file="src/calc.py", op="not_a_real_op")


def test_generator_missing_symbol_raises(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    with pytest.raises(GenerationError):
        _generate(repo, file="src/calc.py", symbol="does_not_exist")


def test_generator_missing_file_raises(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    with pytest.raises(GenerationError, match="target file not found"):
        _generate(repo, file="src/missing.py")


def test_generator_gates_on_green_baseline(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
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
        _generate(repo, file="src/calc.py", env_image=red)


@requires_go
def test_generator_go_end_to_end(tmp_path: Path) -> None:
    _write(tmp_path, "go.mod", "module demo\n\ngo 1.22\n")
    _write(tmp_path, "calc.go", GO_SOURCE)
    gen = AstMutationGenerator()
    request = GenerationRequest(repo_root=tmp_path, seed=3, file="calc.go")
    candidate = gen.generate(request, GoAdapter())
    assert candidate.language == "go"
    rel = candidate.target.files[0]
    original = (tmp_path / rel).read_bytes()
    assert _roundtrip_ok(rel, original, candidate.mutation_patch)


def test_default_generator_registry_has_ast_mutation() -> None:
    registry = build_default_generator_registry()
    assert "ast_mutation" in registry.names()
    assert registry.get("ast_mutation").name == "ast_mutation"
    with pytest.raises(KeyError):
        registry.get("nope")


# --------------------------------------------------------------------------- #
# Data model: Candidate / Provenance / CandidateTarget
# --------------------------------------------------------------------------- #
def test_candidate_serialization_roundtrip() -> None:
    candidate = Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("a.py",), symbols=("f",)),
        mutation_patch="--- a/a.py\n+++ b/a.py\n",
        oracle_patch="--- a/a.py\n+++ b/a.py\n",
        difficulty_hint="low",
        provenance=Provenance(generator="ast_mutation", seed=5, language="python"),
    )
    restored = Candidate.from_dict(candidate.to_dict())
    assert restored.to_dict() == candidate.to_dict()
    assert restored.target.symbol == "f"
    assert restored.provenance.seed == 5


def test_candidate_rejects_unknown_generator() -> None:
    with pytest.raises(ModelError):
        Candidate(
            language="python",
            generator="mystery",
            target=CandidateTarget(files=("a.py",)),
            mutation_patch="x",
            oracle_patch="y",
            difficulty_hint="low",
            provenance=Provenance(generator="x", seed=0, language="python"),
        )


def test_candidate_rejects_empty_patch() -> None:
    with pytest.raises(ModelError):
        Candidate(
            language="python",
            generator="ast_mutation",
            target=CandidateTarget(files=("a.py",)),
            mutation_patch="",
            oracle_patch="y",
            difficulty_hint="low",
            provenance=Provenance(generator="x", seed=0, language="python"),
        )


def test_candidate_target_requires_files() -> None:
    with pytest.raises(ModelError):
        CandidateTarget(files=())


# --------------------------------------------------------------------------- #
# CLI surface (VAL-GEN-009 evidence path)
# --------------------------------------------------------------------------- #
def test_cli_generate_writes_artifacts_and_is_deterministic(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    args = [
        "generate",
        "--path",
        str(repo),
        "--file",
        "src/calc.py",
        "--symbol",
        "add",
        "--seed",
        "9",
        "--json",
    ]
    res1 = runner.invoke(forge_app, [*args, "--out", str(out1)])
    res2 = runner.invoke(forge_app, [*args, "--out", str(out2)])
    assert res1.exit_code == 0, res1.output
    assert res2.exit_code == 0, res2.output

    for out in (out1, out2):
        assert (out / "candidate.json").is_file()
        assert (out / "mutation.patch").is_file()
        assert (out / "oracle.patch").is_file()

    assert (out1 / "mutation.patch").read_bytes() == (
        out2 / "mutation.patch"
    ).read_bytes()
    assert (out1 / "oracle.patch").read_bytes() == (out2 / "oracle.patch").read_bytes()

    payload = json.loads(res1.output)
    assert payload["candidate"]["provenance"]["seed"] == 9
    assert payload["patch_sha256"]["mutation.patch"]


def test_cli_generate_failure_writes_no_candidate(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        forge_app,
        [
            "generate",
            "--path",
            str(repo),
            "--file",
            "src/calc.py",
            "--symbol",
            "missing",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert not (out / "candidate.json").exists()
