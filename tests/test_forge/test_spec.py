"""Unit tests for ``forge/spec.py`` (m3-spec: test-conditioned backtranslation).

Covers the GeneratedSpec contract assertions (offline; no Docker, no live LLM):

- VAL-GEN-013: ``problem_statement`` derives from the F2P failure trace (mentions
  the failing test's observable behavior), provenance records the trace as input,
  and no line is copied from the mutation/oracle patches.
- VAL-GEN-014: none of the three fields leak the oracle/implementation body, an
  ``oracle_patch`` hunk line, or the generator name; the interface is signatures
  only. A leaky author is rejected (emits nothing).
- VAL-GEN-015: ``requirements`` is non-empty and each item is traceable to a
  named F2P test; none quotes oracle code.
- VAL-GEN-016: ``interface_block`` lists real target signatures (found in the
  original source defs) and the Candidate target symbol(s) appear.
- VAL-GEN-017: a GeneratedSpec is emitted only alongside a valid Candidate; a
  non-round-tripping candidate produces NO spec.

Candidates are built with the real generators so the patches/round-trip and the
leak-scan inputs are genuine; the spec author is the offline template (or a
fake) so the deterministic machinery is exercised without the endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_forge.forge.adapters import JavaScriptAdapter, PythonAdapter
from swe_forge.forge.generators import GenerationRequest
from swe_forge.forge.generators.ast_mutation import AstMutationGenerator
from swe_forge.forge.generators.bug_combination import BugCombinationGenerator
from swe_forge.forge.generators.function_removal import FunctionRemovalGenerator
from swe_forge.forge.models import Candidate, GeneratedSpec, ModelError
from swe_forge.forge.spec import (
    AuthoredRequirement,
    AuthoredSpec,
    F2PTrace,
    FailingTest,
    SpecAuthoringContext,
    SpecError,
    TemplateSpecAuthor,
    _contract_signature,
    build_interface_block,
    generate_spec,
    scan_spec_for_leaks,
)

runner = CliRunner()

PY_CALC = (
    "def classify(n):\n"
    "    if n < 0:\n"
    '        return "negative"\n'
    "    if n == 0:\n"
    '        return "zero"\n'
    '    return "positive"\n'
)

JS_CALC = (
    "function classify(n) {\n"
    "  if (n < 0) {\n"
    '    return "negative";\n'
    "  }\n"
    '  return "positive";\n'
    "}\n"
)


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _py_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "src/calc.py", PY_CALC)
    return root


def _js_repo(root: Path) -> Path:
    _write(root, "package.json", '{"name":"demo","version":"0.0.0"}\n')
    _write(root, "calc.js", JS_CALC)
    return root


def _py_candidate(root: Path) -> Candidate:
    return AstMutationGenerator().generate(
        GenerationRequest(repo_root=root, seed=0, file="src/calc.py"), PythonAdapter()
    )


def _trace() -> F2PTrace:
    return F2PTrace(
        tests=(
            FailingTest(
                name="tests/test_calc.py::test_negative",
                file="tests/test_calc.py",
                message="AssertionError: classify(-5) should be 'negative'",
                expected="'negative'",
                observed="'positive'",
            ),
        ),
        raw="E   AssertionError: assert classify(-5) == 'negative'",
    )


# --------------------------------------------------------------------------- #
# GeneratedSpec model
# --------------------------------------------------------------------------- #
def test_generated_spec_requires_nonempty_fields() -> None:
    with pytest.raises(ModelError):
        GeneratedSpec(
            problem_statement="",
            requirements=["x"],
            interface_block="def f()",
            provenance=_minimal_provenance(),
        )
    with pytest.raises(ModelError):
        GeneratedSpec(
            problem_statement="do it",
            requirements=[],
            interface_block="def f()",
            provenance=_minimal_provenance(),
        )
    with pytest.raises(ModelError):
        GeneratedSpec(
            problem_statement="do it",
            requirements=["x"],
            interface_block="   ",
            provenance=_minimal_provenance(),
        )


def test_generated_spec_round_trips_to_dict() -> None:
    spec = GeneratedSpec(
        problem_statement="solve",
        requirements=["a", "b"],
        interface_block="def f(x)",
        provenance=_minimal_provenance(),
    )
    restored = GeneratedSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.problem_statement == "solve"
    assert restored.requirements == ["a", "b"]
    assert restored.interface_block == "def f(x)"


def _minimal_provenance():
    from swe_forge.forge.models import Provenance

    return Provenance(generator="ast_mutation", seed=0, language="python")


# --------------------------------------------------------------------------- #
# F2PTrace parsing
# --------------------------------------------------------------------------- #
def test_f2p_trace_from_dict_supports_tests_and_fail_to_pass() -> None:
    a = F2PTrace.from_dict({"tests": [{"name": "t::a", "expected": "1"}]})
    assert a.test_names() == ("t::a",)
    b = F2PTrace.from_dict({"fail_to_pass": ["t::b", "t::c"]})
    assert b.test_names() == ("t::b", "t::c")


def test_failing_test_requires_name() -> None:
    with pytest.raises(SpecError):
        FailingTest(name="")


# --------------------------------------------------------------------------- #
# VAL-GEN-016: interface block lists real signatures; target symbols appear
# --------------------------------------------------------------------------- #
def test_interface_block_lists_real_target_signatures(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    block = build_interface_block(candidate, repo, PythonAdapter())

    # The candidate target symbol appears in the block.
    assert candidate.target.symbol == "classify"
    names = {s.name for s in block.symbols}
    assert "classify" in names
    # Every listed signature is found in the original source defs (name + params).
    assert "def classify(n)" in block.text
    for sym in block.symbols:
        assert (
            sym.signature.replace(" ", "") in PY_CALC.replace(" ", "").replace(":", "")
            or sym.name in PY_CALC
        )


def test_interface_block_infers_symbols_when_target_symbols_empty(
    tmp_path: Path,
) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    # Simulate a generator that records files but no explicit symbols (pr_mirror).
    stripped = Candidate(
        language=candidate.language,
        generator="pr_mirror",
        target=type(candidate.target)(files=candidate.target.files, symbols=()),
        mutation_patch=candidate.mutation_patch,
        oracle_patch=candidate.oracle_patch,
        difficulty_hint=candidate.difficulty_hint,
        provenance=candidate.provenance,
    )
    block = build_interface_block(stripped, repo, PythonAdapter())
    # The touched function is inferred from the mutation patch hunks.
    assert any(s.name == "classify" for s in block.symbols)


def test_interface_block_cross_language_javascript(tmp_path: Path) -> None:
    repo = _js_repo(tmp_path)
    candidate = AstMutationGenerator().generate(
        GenerationRequest(repo_root=repo, seed=0, file="calc.js"), JavaScriptAdapter()
    )
    block = build_interface_block(candidate, repo, JavaScriptAdapter())
    assert any(s.name == "classify" for s in block.symbols)
    assert "classify" in block.text


# --------------------------------------------------------------------------- #
# Spec-leak fix: the interface block is a signature CONTRACT (param names +
# annotations) with default-VALUE expressions stripped, so a multi-fault
# structural candidate whose faulted symbol has a signature with default values
# (an implementation token that also appears as a source/patch line) is NOT
# rejected as a false `spec_leak`. The leak auditor stays intact (still rejects
# a genuine impl leak).
# --------------------------------------------------------------------------- #
# A boltons-like modular target: a function whose (multi-line) signature carries
# default-value expressions -- the exact shape that leaked before the fix.
PY_FORMATUTILS = (
    'def render_field(name="field",\n'
    '                 formatter=lambda value: "=" + repr(value),\n'
    "                 defaults={},\n"
    '                 prefix="> "):\n'
    "    total = 0\n"
    "    for ch in name:\n"
    "        total = total + 1\n"
    "    return prefix + str(total)\n"
)
PY_MATHUTILS = (
    "def clamp(value, lower, upper):\n"
    "    if value < lower:\n"
    "        return lower\n"
    "    if value > upper:\n"
    "        return upper\n"
    "    return value\n"
)


def _boltons_like_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "pkg/formatutils.py", PY_FORMATUTILS)
    _write(root, "pkg/mathutils.py", PY_MATHUTILS)
    return root


def _multi_fault_candidate(root: Path) -> Candidate:
    return BugCombinationGenerator().generate(
        GenerationRequest(repo_root=root, seed=0, params={"faults": 2}), PythonAdapter()
    )


def test_contract_signature_strips_default_values() -> None:
    sig = (
        "def render_field(name, formatter=lambda value: '=' + repr(value), "
        "defaults={}, prefix='> ')"
    )
    out = _contract_signature(sig)
    # Parameter NAMES are preserved (the public contract).
    for param in ("name", "formatter", "defaults", "prefix"):
        assert param in out
    # But no default-VALUE implementation token survives.
    for token in ("lambda", "repr(value)", "{}", "'> '", "="):
        assert token not in out


def test_contract_signature_keeps_annotations_and_return() -> None:
    # A ``=`` default is cut, but ``:`` annotations, ``==`` etc. and the return
    # annotation are preserved (never mistaken for a default).
    sig = "def f(x: int, y: Dict[str, int] = {}, z=0) -> bool"
    out = _contract_signature(sig)
    assert "x: int" in out
    assert "y: Dict[str, int]" in out
    assert "z" in out
    assert "-> bool" in out
    assert "= {}" not in out and "z=0" not in out


def test_contract_signature_is_noop_without_defaults() -> None:
    # Go-style / plain signatures (no defaults) are returned unchanged in effect.
    assert _contract_signature("def clamp(value, lower, upper)") == (
        "def clamp(value, lower, upper)"
    )
    assert _contract_signature("func Clamp(v, lo, hi int) int") == (
        "func Clamp(v, lo, hi int) int"
    )


def test_multi_fault_interface_block_has_no_default_value_leak(
    tmp_path: Path,
) -> None:
    repo = _boltons_like_repo(tmp_path)
    candidate = _multi_fault_candidate(repo)
    # The multi-fault candidate targets the default-carrying function.
    assert "render_field" in candidate.target.symbols
    block = build_interface_block(candidate, repo, PythonAdapter())
    # The faulted symbol is still exposed by name (the contract is not empty).
    assert any(s.name == "render_field" for s in block.symbols)
    assert "render_field" in block.text
    # No default-value implementation token appears in the interface text.
    for token in ("lambda value", "defaults={}", 'prefix="> "', "repr(value)"):
        assert token not in block.text
    # And the leak auditor now PASSES the interface block for this candidate.
    findings = scan_spec_for_leaks(
        "A clean problem statement describing the expected behavior.",
        ["A clean requirement grounded in a failing test."],
        block.text,
        candidate,
        block.signatures(),
    )
    interface_findings = [f for f in findings if f.startswith("interface_block")]
    assert interface_findings == []


def test_multi_fault_candidate_yields_spec(tmp_path: Path) -> None:
    repo = _boltons_like_repo(tmp_path)
    candidate = _multi_fault_candidate(repo)
    trace = F2PTrace(
        tests=tuple(
            FailingTest(
                name=f"tests/test_hidden.py::test_{sym}",
                file="tests/test_hidden.py",
                expected="the correct result",
                observed="a wrong result",
            )
            for sym in candidate.target.symbols
        )
    )
    # generate_spec no longer raises SpecError('spec leaks ...') for this
    # legitimate multi-fault structural candidate.
    spec = generate_spec(
        candidate, trace, repo, PythonAdapter(), author=TemplateSpecAuthor()
    )
    assert "render_field" in spec.interface_block
    # No default-value impl token leaked into any field.
    for token in ("lambda value", "defaults={}"):
        assert token not in spec.interface_block


def test_leak_auditor_still_rejects_genuine_impl_leak_in_interface(
    tmp_path: Path,
) -> None:
    # TRUE-POSITIVE preserved: if a real body/impl line of the faulted symbol is
    # injected into the interface block, the auditor STILL rejects it. The fix
    # removed the SOURCE of the false leak, not the check.
    repo = _boltons_like_repo(tmp_path)
    candidate = _multi_fault_candidate(repo)
    block = build_interface_block(candidate, repo, PythonAdapter())
    leaked_line = _patch_code_lines(candidate)[0]
    poisoned = block.text + "\n# " + leaked_line
    findings = scan_spec_for_leaks(
        "clean problem statement",
        ["clean requirement"],
        poisoned,
        candidate,
        block.signatures(),
    )
    assert any(f.startswith("interface_block") for f in findings)


# --------------------------------------------------------------------------- #
# VAL-GEN-013 + 015: backtranslation from the trace + grounded requirements
# --------------------------------------------------------------------------- #
def test_problem_statement_is_test_conditioned(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    trace = _trace()
    spec = generate_spec(
        candidate, trace, repo, PythonAdapter(), author=TemplateSpecAuthor()
    )

    # References the failing test's observable behavior.
    assert "tests/test_calc.py::test_negative" in spec.problem_statement
    assert "negative" in spec.problem_statement
    # Provenance records the F2P trace as the input (not the diff).
    details = spec.provenance.details
    assert details["f2p_tests"] == ["tests/test_calc.py::test_negative"]
    assert details["f2p_trace"]["tests"][0]["name"] == (
        "tests/test_calc.py::test_negative"
    )
    assert details["source"] == "test-conditioned backtranslation"
    # No line copied from the patches.
    for patch_line in _patch_code_lines(candidate):
        assert patch_line not in spec.problem_statement


def test_requirements_are_grounded_in_named_f2p_tests(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    trace = _trace()
    spec = generate_spec(
        candidate, trace, repo, PythonAdapter(), author=TemplateSpecAuthor()
    )

    assert spec.requirements  # non-empty
    traceability = spec.provenance.details["requirement_traceability"]
    assert traceability
    valid = set(trace.test_names())
    for entry in traceability:
        assert entry["test"] in valid
        assert entry["requirement"] in spec.requirements
    # None quotes oracle code.
    for patch_line in _patch_code_lines(candidate):
        for requirement in spec.requirements:
            assert patch_line not in requirement


def test_requirement_with_unknown_test_is_grounded_to_sole_test(
    tmp_path: Path,
) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    trace = _trace()

    def author(ctx: SpecAuthoringContext) -> AuthoredSpec:
        return AuthoredSpec(
            problem_statement="classify must return 'negative' for negatives.",
            requirements=(
                AuthoredRequirement(text="negatives map to 'negative'", test="bogus"),
            ),
            model="anthropic/test",
        )

    spec = generate_spec(candidate, trace, repo, PythonAdapter(), author=author)
    # With a single failing test, an unmatched reference grounds to it.
    traceability = spec.provenance.details["requirement_traceability"]
    assert traceability[0]["test"] == "tests/test_calc.py::test_negative"


def test_ungroundable_requirements_raise(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    trace = F2PTrace(tests=(FailingTest(name="t::a"), FailingTest(name="t::b")))

    def author(ctx: SpecAuthoringContext) -> AuthoredSpec:
        return AuthoredSpec(
            problem_statement="do the thing",
            requirements=(
                AuthoredRequirement(text="some behavior", test="not-a-real-test"),
            ),
            model="anthropic/test",
        )

    with pytest.raises(SpecError):
        generate_spec(candidate, trace, repo, PythonAdapter(), author=author)


# --------------------------------------------------------------------------- #
# VAL-GEN-014: leak scan
# --------------------------------------------------------------------------- #
def test_clean_spec_has_no_leaks(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    spec = generate_spec(
        candidate, _trace(), repo, PythonAdapter(), author=TemplateSpecAuthor()
    )
    fields = [
        spec.problem_statement,
        "\n".join(spec.requirements),
        spec.interface_block,
    ]
    # No generator name leaks into any field.
    for text in fields:
        assert "ast_mutation" not in text
    # The interface exposes signatures only (no statement bodies).
    assert "return" not in spec.interface_block


def test_leaky_problem_statement_is_rejected(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    leaked = _patch_code_lines(candidate)[0]

    def author(ctx: SpecAuthoringContext) -> AuthoredSpec:
        return AuthoredSpec(
            problem_statement=f"Implement it like: {leaked}",
            requirements=(
                AuthoredRequirement(
                    text="behave", test="tests/test_calc.py::test_negative"
                ),
            ),
            model="anthropic/test",
        )

    with pytest.raises(SpecError, match="leaks"):
        generate_spec(candidate, _trace(), repo, PythonAdapter(), author=author)


def test_leaky_requirement_is_rejected(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    leaked = _patch_code_lines(candidate)[0]

    def author(ctx: SpecAuthoringContext) -> AuthoredSpec:
        return AuthoredSpec(
            problem_statement="A clean description of expected behavior.",
            requirements=(
                AuthoredRequirement(
                    text=f"the code must read {leaked}",
                    test="tests/test_calc.py::test_negative",
                ),
            ),
            model="anthropic/test",
        )

    with pytest.raises(SpecError, match="leaks"):
        generate_spec(candidate, _trace(), repo, PythonAdapter(), author=author)


def test_generator_name_leak_is_rejected(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)

    def author(ctx: SpecAuthoringContext) -> AuthoredSpec:
        return AuthoredSpec(
            problem_statement="This was made by the ast_mutation generator.",
            requirements=(
                AuthoredRequirement(
                    text="behave", test="tests/test_calc.py::test_negative"
                ),
            ),
            model="anthropic/test",
        )

    with pytest.raises(SpecError, match="leaks"):
        generate_spec(candidate, _trace(), repo, PythonAdapter(), author=author)


# --------------------------------------------------------------------------- #
# Misc generate_spec guards
# --------------------------------------------------------------------------- #
def test_empty_trace_raises(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = _py_candidate(repo)
    with pytest.raises(SpecError):
        generate_spec(
            candidate,
            F2PTrace(tests=()),
            repo,
            PythonAdapter(),
            author=TemplateSpecAuthor(),
        )


def test_function_removal_candidate_yields_spec(tmp_path: Path) -> None:
    repo = _py_repo(tmp_path)
    candidate = FunctionRemovalGenerator().generate(
        GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"), PythonAdapter()
    )
    spec = generate_spec(
        candidate, _trace(), repo, PythonAdapter(), author=TemplateSpecAuthor()
    )
    assert "classify" in spec.interface_block
    # Even though the whole body is in the oracle patch, no body line leaks.
    for patch_line in _patch_code_lines(candidate):
        assert patch_line not in spec.interface_block
        assert patch_line not in spec.problem_statement


# --------------------------------------------------------------------------- #
# VAL-GEN-017: spec emitted only alongside a valid Candidate (CLI)
# --------------------------------------------------------------------------- #
def test_cli_spec_offline_emits_paired_spec(tmp_path: Path) -> None:
    from swe_forge.forge.cli import app

    repo = _py_repo(tmp_path / "repo")
    candidate = _py_candidate(repo)
    cand_path = tmp_path / "candidate.json"
    cand_path.write_text(json.dumps(candidate.to_dict()), encoding="utf-8")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_trace().to_dict()), encoding="utf-8")
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "spec",
            "--candidate",
            str(cand_path),
            "--path",
            str(repo),
            "--trace",
            str(trace_path),
            "--out",
            str(out_dir),
            "--offline",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    spec_file = out_dir / "spec.json"
    assert spec_file.is_file()
    spec = GeneratedSpec.from_dict(json.loads(spec_file.read_text()))
    assert spec.requirements
    assert "classify" in spec.interface_block


def test_cli_spec_rejects_non_round_tripping_candidate(tmp_path: Path) -> None:
    from swe_forge.forge.cli import app

    repo = _py_repo(tmp_path / "repo")
    candidate = _py_candidate(repo)
    # Corrupt the oracle patch so the round-trip no longer restores byte-for-byte.
    broken = Candidate(
        language=candidate.language,
        generator=candidate.generator,
        target=candidate.target,
        mutation_patch=candidate.mutation_patch,
        oracle_patch=candidate.mutation_patch,  # wrong inverse
        difficulty_hint=candidate.difficulty_hint,
        provenance=candidate.provenance,
    )
    cand_path = tmp_path / "candidate.json"
    cand_path.write_text(json.dumps(broken.to_dict()), encoding="utf-8")
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_trace().to_dict()), encoding="utf-8")
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "spec",
            "--candidate",
            str(cand_path),
            "--path",
            str(repo),
            "--trace",
            str(trace_path),
            "--out",
            str(out_dir),
            "--offline",
        ],
    )
    assert result.exit_code == 1
    # No spec artifact written for an invalid candidate.
    assert not (out_dir / "spec.json").exists()


def _patch_code_lines(candidate: Candidate) -> list[str]:
    """Non-trivial code lines from the candidate's patches (for leak assertions)."""
    out: list[str] = []
    for patch in (candidate.oracle_patch, candidate.mutation_patch):
        for line in patch.splitlines():
            if line[:3] in ("+++", "---") or line.startswith("@@"):
                continue
            if line and line[0] in "+-":
                body = line[1:].strip()
                if len(body) >= 6 and any(c.isalpha() for c in body):
                    out.append(body)
    return out
