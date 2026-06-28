"""Cross-generator self-validation menu (Stage 2 integration).

The six bug generators each self-validate their own round-trip, but the *menu*
adds an independent, cross-generator gate: it exercises every generator behind
the uniform :class:`~swe_forge.forge.generators.base.BugGenerator` interface,
re-derives each emitted Candidate's forward+inverse round-trip from the patches
alone (not trusting the generator), confirms the forward mutation is
behavior-changing (not whitespace/comment/import-only), checks the Candidate
schema is complete with a known generator name, and records a
``coverage[generator][language]`` matrix. If any generator fails its
self-validation the whole run aborts with an attributable reason and emits NO
artifact (the by-construction guarantee must hold for all six).

The two LLM-backed generators (``lm_authored``, ``pr_mirror``) run here with
deterministic offline stubs so the menu self-check is free, reproducible, and
exercises the real "deterministic execution disposes" machinery; they are marked
``llm_backed`` (a documented cost-exemption) and are required to cover Python but
exempt from the non-Python column. The deterministic generators cover Python plus
at least one of JS/TS or Go.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from swe_forge.forge.adapters import (
    GoAdapter,
    JavaScriptAdapter,
    LanguageAdapter,
    PythonAdapter,
)
from swe_forge.forge.adapters._diff import (
    PatchError,
    apply_multi_patch,
    make_patch,
)
from swe_forge.forge.generators._normalize import is_behavior_changing
from swe_forge.forge.generators._targeting import sha256_bytes
from swe_forge.forge.generators.ast_mutation import AstMutationGenerator
from swe_forge.forge.generators.base import (
    BugGenerator,
    GenerationError,
    GenerationRequest,
)
from swe_forge.forge.generators.bug_combination import BugCombinationGenerator
from swe_forge.forge.generators.function_removal import FunctionRemovalGenerator
from swe_forge.forge.generators.lm_authored import (
    AuthoredEdit,
    AuthoringContext,
    LmAuthoredGenerator,
)
from swe_forge.forge.generators.multi_file import MultiFileGenerator
from swe_forge.forge.generators.pr_mirror import (
    InversionProposal,
    MergedPullRequest,
    PrFileChange,
    PrInversionContext,
    PrMirrorGenerator,
)
from swe_forge.forge.models import (
    GENERATOR_NAMES,
    Candidate,
)

# Required schema keys for every emitted Candidate (VAL-GEN-008).
_REQUIRED_CANDIDATE_KEYS: tuple[str, ...] = (
    "language",
    "generator",
    "target",
    "mutation_patch",
    "oracle_patch",
    "difficulty_hint",
)


# --------------------------------------------------------------------------- #
# Independent round-trip + schema re-verification (does not trust the generator)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoundTripResult:
    """Outcome of re-deriving a Candidate's forward+inverse round-trip.

    ``ok`` requires every touched file to apply cleanly in both directions, the
    oracle to restore each file byte-for-byte (sha256 match), and the forward
    mutation to be behavior-changing. ``file_sha256`` maps each touched path to
    its ``{original, mutated, restored}`` digests (round-trip evidence).
    """

    ok: bool
    behavior_changing: bool
    reason: str
    files: tuple[str, ...]
    file_sha256: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "behavior_changing": self.behavior_changing,
            "reason": self.reason,
            "files": list(self.files),
            "file_sha256": {k: dict(v) for k, v in self.file_sha256.items()},
        }


def verify_candidate_roundtrip(
    repo_root: Path, candidate: Candidate
) -> RoundTripResult:
    """Re-verify ``candidate``'s forward+inverse round-trip from its patches alone.

    Reads the pristine bytes of every targeted file from ``repo_root``, applies
    the ``mutation_patch`` (must change every touched file and be behavior-
    changing), then applies the ``oracle_patch`` and requires every file restored
    byte-for-byte. Returns a :class:`RoundTripResult`; a non-applying or
    non-inverting patch yields ``ok=False`` with an attributable reason rather
    than raising.
    """
    files = tuple(candidate.target.files)
    if not files:
        return RoundTripResult(False, False, "candidate target lists no files", ())

    originals: dict[str, bytes] = {}
    for rel in files:
        path = repo_root / rel
        if not path.is_file():
            return RoundTripResult(
                False, False, f"targeted file is missing on disk: {rel}", files
            )
        originals[rel] = path.read_bytes()

    if not candidate.mutation_patch.strip() or not candidate.oracle_patch.strip():
        return RoundTripResult(False, False, "empty mutation or oracle patch", files)

    try:
        mutated = apply_multi_patch(originals, candidate.mutation_patch)
    except PatchError as exc:
        return RoundTripResult(
            False, False, f"mutation_patch did not apply cleanly: {exc}", files
        )

    behavior_changing = True
    for rel in files:
        if mutated.get(rel) == originals[rel]:
            return RoundTripResult(
                False, False, f"mutation left {rel} byte-identical (no-op)", files
            )
        if not is_behavior_changing(
            originals[rel].decode("utf-8", "replace"),
            mutated[rel].decode("utf-8", "replace"),
        ):
            behavior_changing = False

    try:
        restored = apply_multi_patch(mutated, candidate.oracle_patch)
    except PatchError as exc:
        return RoundTripResult(
            False,
            behavior_changing,
            f"oracle_patch did not apply cleanly: {exc}",
            files,
        )

    file_sha256: dict[str, dict[str, str]] = {}
    for rel in files:
        file_sha256[rel] = {
            "original": sha256_bytes(originals[rel]),
            "mutated": sha256_bytes(mutated[rel]),
            "restored": sha256_bytes(restored.get(rel, b"")),
        }

    for rel in files:
        if file_sha256[rel]["restored"] != file_sha256[rel]["original"]:
            return RoundTripResult(
                False,
                behavior_changing,
                f"oracle did not restore {rel} byte-for-byte",
                files,
                file_sha256,
            )

    if not behavior_changing:
        return RoundTripResult(
            False,
            False,
            "forward mutation is whitespace/comment/import-only (not behavior-changing)",
            files,
            file_sha256,
        )

    return RoundTripResult(True, True, "", files, file_sha256)


def schema_completeness(candidate: Candidate) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the Candidate schema completeness check."""
    data = candidate.to_dict()
    for key in _REQUIRED_CANDIDATE_KEYS:
        value = data.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, f"missing/empty field: {key}"
    if data["generator"] not in GENERATOR_NAMES:
        return False, f"unknown generator name: {data['generator']!r}"
    target = data.get("target")
    if not isinstance(target, dict) or not target.get("files"):
        return False, "target.files must be a non-empty list"
    return True, ""


# --------------------------------------------------------------------------- #
# Per-cell evaluation + aggregation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CellResult:
    """Self-validation outcome for one (generator, language) menu cell."""

    generator: str
    language: str
    ok: bool
    llm_backed: bool
    reason: str
    files: tuple[str, ...]
    roundtrip_ok: bool
    behavior_changing: bool
    schema_complete: bool
    difficulty_hint: str = ""
    file_sha256: dict[str, dict[str, str]] = field(default_factory=dict)
    candidate: Candidate | None = None

    def matrix_entry(self) -> dict[str, object]:
        """The compact ``coverage[generator][language]`` cell record."""
        return {
            "ok": self.ok,
            "llm_backed": self.llm_backed,
            "roundtrip_ok": self.roundtrip_ok,
            "behavior_changing": self.behavior_changing,
            "schema_complete": self.schema_complete,
            "files": list(self.files),
            "difficulty_hint": self.difficulty_hint,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MenuCellSpec:
    """How to exercise one generator on one language with an offline fixture."""

    generator: BugGenerator
    language: str
    llm_backed: bool
    build_repo: Callable[[Path], Path]
    request_kwargs: dict[str, object] = field(default_factory=dict)


def evaluate_cell(spec: MenuCellSpec, workdir: Path, seed: int) -> CellResult:
    """Run one generator on its fixture and independently self-validate the result."""
    repo_root = (workdir / spec.generator.name / spec.language).resolve()
    repo_root.mkdir(parents=True, exist_ok=True)
    spec.build_repo(repo_root)
    adapter = _adapter_for(spec.language)

    request = GenerationRequest(
        repo_root=repo_root,
        seed=seed,
        **spec.request_kwargs,  # type: ignore[arg-type]
    )
    try:
        candidate = spec.generator.generate(request, adapter)
    except GenerationError as exc:
        return _failed_cell(spec, f"generation failed: {exc}")
    except Exception as exc:  # defensive: a generator must not crash the menu
        return _failed_cell(spec, f"generation raised {type(exc).__name__}: {exc}")

    schema_ok, schema_reason = schema_completeness(candidate)
    rt = verify_candidate_roundtrip(repo_root, candidate)

    ok = schema_ok and rt.ok
    reason = ""
    if not schema_ok:
        reason = f"schema: {schema_reason}"
    elif not rt.ok:
        reason = f"roundtrip: {rt.reason}"

    return CellResult(
        generator=spec.generator.name,
        language=spec.language,
        ok=ok,
        llm_backed=spec.llm_backed,
        reason=reason,
        files=rt.files or tuple(candidate.target.files),
        roundtrip_ok=rt.ok,
        behavior_changing=rt.behavior_changing,
        schema_complete=schema_ok,
        difficulty_hint=candidate.difficulty_hint,
        file_sha256=rt.file_sha256,
        candidate=candidate,
    )


def _failed_cell(spec: MenuCellSpec, reason: str) -> CellResult:
    return CellResult(
        generator=spec.generator.name,
        language=spec.language,
        ok=False,
        llm_backed=spec.llm_backed,
        reason=reason,
        files=(),
        roundtrip_ok=False,
        behavior_changing=False,
        schema_complete=False,
    )


@dataclass
class MenuReport:
    """Aggregated cross-generator self-validation result + coverage matrix."""

    cells: list[CellResult]
    coverage: dict[str, dict[str, dict[str, object]]]
    ok: bool
    reasons: list[str]
    seed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "seed": self.seed,
            "generators": list(GENERATOR_NAMES),
            "coverage": self.coverage,
            "reasons": list(self.reasons),
            "cells": [
                {
                    "generator": c.generator,
                    "language": c.language,
                    "ok": c.ok,
                    "llm_backed": c.llm_backed,
                    "roundtrip_ok": c.roundtrip_ok,
                    "behavior_changing": c.behavior_changing,
                    "schema_complete": c.schema_complete,
                    "difficulty_hint": c.difficulty_hint,
                    "files": list(c.files),
                    "file_sha256": {k: dict(v) for k, v in c.file_sha256.items()},
                    "reason": c.reason,
                }
                for c in self.cells
            ],
        }


def build_coverage_matrix(
    cells: list[CellResult],
) -> dict[str, dict[str, dict[str, object]]]:
    """Build the ``coverage[generator][language]`` matrix from cell results."""
    matrix: dict[str, dict[str, dict[str, object]]] = {
        name: {} for name in GENERATOR_NAMES
    }
    for cell in cells:
        matrix.setdefault(cell.generator, {})[cell.language] = cell.matrix_entry()
    return matrix


def evaluate_coverage(
    cells: list[CellResult],
) -> tuple[bool, list[str]]:
    """Validate the menu coverage rules; return ``(ok, reasons)``.

    Rules: every cell that ran must self-validate; every generator must have a
    passing Python entry (no empty Python column); every generator must have a
    passing non-Python entry OR be ``llm_backed`` (cost-exempt).
    """
    reasons: list[str] = []
    for cell in cells:
        if not cell.ok:
            reasons.append(f"{cell.generator}[{cell.language}]: {cell.reason}")

    by_gen: dict[str, list[CellResult]] = {name: [] for name in GENERATOR_NAMES}
    for cell in cells:
        by_gen.setdefault(cell.generator, []).append(cell)

    for name in GENERATOR_NAMES:
        gen_cells = by_gen.get(name, [])
        python_ok = any(c.language == "python" and c.ok for c in gen_cells)
        if not python_ok:
            reasons.append(f"{name}: no passing Python coverage (empty Python column)")
        non_python_ok = any(c.language != "python" and c.ok for c in gen_cells)
        llm_backed = any(c.llm_backed for c in gen_cells)
        if not non_python_ok and not llm_backed:
            reasons.append(
                f"{name}: no non-Python coverage and not flagged llm_backed (exempt)"
            )

    return (not reasons), reasons


def run_menu_selfcheck(
    workdir: Path,
    *,
    seed: int = 0,
    specs: list[MenuCellSpec] | None = None,
) -> MenuReport:
    """Exercise the full generator menu and self-validate every emitted Candidate."""
    cell_specs = specs if specs is not None else build_menu_cell_specs()
    cells = [evaluate_cell(spec, workdir, seed) for spec in cell_specs]
    coverage = build_coverage_matrix(cells)
    ok, reasons = evaluate_coverage(cells)
    return MenuReport(cells=cells, coverage=coverage, ok=ok, reasons=reasons, seed=seed)


# --------------------------------------------------------------------------- #
# Adapters + default fixtures/specs
# --------------------------------------------------------------------------- #
def _adapter_for(language: str) -> LanguageAdapter:
    if language == "python":
        return PythonAdapter()
    if language == "javascript":
        return JavaScriptAdapter()
    if language == "go":
        return GoAdapter()
    raise ValueError(f"unsupported menu language: {language!r}")


def go_toolchain_available() -> bool:
    """Return ``True`` iff the host Go toolchain is usable for ``parse_symbols``."""
    try:
        from swe_forge.forge.adapters._goast import GoToolchainError, _find_go

        try:
            _find_go()
            return True
        except GoToolchainError:
            return False
    except Exception:
        return False


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# Fixture sources (small, deterministic, behavior-bearing).
_PY_SINGLE = (
    "def classify(n):\n"
    "    if n < 0:\n"
    '        return "negative"\n'
    "    if n == 0:\n"
    '        return "zero"\n'
    '    return "positive"\n'
)
_PY_ALPHA = "def scale(x):\n    return x * 2\n"
_PY_BETA = "def shift(x):\n    return x + 10\n"
_PY_ADD = "def add(a, b):\n    return a + b\n"

_JS_SINGLE = (
    "function classify(n) {\n"
    "  if (n < 0) {\n"
    '    return "negative";\n'
    "  }\n"
    '  return "positive";\n'
    "}\n"
)
_JS_ALPHA = "function scale(x) {\n  return x * 2;\n}\n"
_JS_BETA = "function shift(x) {\n  return x + 10;\n}\n"

_GO_SINGLE = (
    "package calc\n\n"
    "func Classify(n int) string {\n"
    "\tif n < 0 {\n"
    '\t\treturn "negative"\n'
    "\t}\n"
    '\treturn "positive"\n'
    "}\n"
)

_PR_HEAD = "def f():\n    return 2\n"
_PR_BASE = "def f():\n    return 1\n"


def _py_single_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "src/calc.py", _PY_SINGLE)
    return root


def _py_multi_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "alpha.py", _PY_ALPHA)
    _write(root, "beta.py", _PY_BETA)
    _write(root, "test_mod.py", "from alpha import scale\nfrom beta import shift\n")
    return root


def _py_add_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "src/calc.py", _PY_ADD)
    return root


def _py_pr_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "mod.py", _PR_HEAD)
    return root


def _js_single_repo(root: Path) -> Path:
    _write(root, "package.json", '{"name":"demo","version":"0.0.0"}\n')
    _write(root, "calc.js", _JS_SINGLE)
    return root


def _js_multi_repo(root: Path) -> Path:
    _write(root, "package.json", '{"name":"demo","version":"0.0.0"}\n')
    _write(root, "alpha.js", _JS_ALPHA)
    _write(root, "beta.js", _JS_BETA)
    return root


def _go_single_repo(root: Path) -> Path:
    _write(root, "go.mod", "module demo\n\ngo 1.22\n")
    _write(root, "calc.go", _GO_SINGLE)
    return root


# Deterministic offline stubs for the LLM-backed generators (cost-exemption).
def _stub_author(ctx: AuthoringContext) -> AuthoredEdit:
    """Flip the function's ``a + b`` to ``a - b`` (a subtle single-function bug)."""
    return AuthoredEdit(
        new_source=ctx.function_source.replace("a + b", "a - b"),
        model="stub/offline",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        cost=0.0,
    )


def _stub_merged_pr() -> MergedPullRequest:
    pr_diff = make_patch("mod.py", _PR_BASE.encode(), _PR_HEAD.encode())
    return MergedPullRequest(
        number=1,
        sha="a" * 40,
        repo="octocat/demo",
        files=[PrFileChange(path="mod.py", patch=pr_diff)],
        url="https://example.invalid/pull/1",
        title="stub merged PR",
    )


def _stub_pr_resolver(repo_root: Path, params: dict[str, object]) -> MergedPullRequest:
    return _stub_merged_pr()


def _stub_pr_inverter(ctx: PrInversionContext) -> InversionProposal:
    return InversionProposal(
        reverted={"mod.py": _PR_BASE},
        model="stub/offline",
        usage=[
            {
                "model": "stub/offline",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "cost": 0.0,
            }
        ],
    )


def build_menu_cell_specs(*, include_go: bool | None = None) -> list[MenuCellSpec]:
    """Build the default menu cell specs covering all six generators.

    Deterministic generators cover Python + JavaScript (and Go when the toolchain
    is available); the LLM-backed generators cover Python with offline stubs and
    are flagged ``llm_backed`` (cost-exempt from the non-Python column).
    """
    if include_go is None:
        include_go = go_toolchain_available()

    specs: list[MenuCellSpec] = [
        MenuCellSpec(
            AstMutationGenerator(),
            "python",
            False,
            _py_single_repo,
            {"file": "src/calc.py"},
        ),
        MenuCellSpec(
            AstMutationGenerator(),
            "javascript",
            False,
            _js_single_repo,
            {"file": "calc.js"},
        ),
        MenuCellSpec(
            FunctionRemovalGenerator(),
            "python",
            False,
            _py_single_repo,
            {"file": "src/calc.py"},
        ),
        MenuCellSpec(
            FunctionRemovalGenerator(),
            "javascript",
            False,
            _js_single_repo,
            {"file": "calc.js"},
        ),
        MenuCellSpec(MultiFileGenerator(), "python", False, _py_multi_repo, {}),
        MenuCellSpec(MultiFileGenerator(), "javascript", False, _js_multi_repo, {}),
        MenuCellSpec(BugCombinationGenerator(), "python", False, _py_multi_repo, {}),
        MenuCellSpec(
            BugCombinationGenerator(), "javascript", False, _js_multi_repo, {}
        ),
        MenuCellSpec(
            LmAuthoredGenerator(author=_stub_author),
            "python",
            True,
            _py_add_repo,
            {"file": "src/calc.py"},
        ),
        MenuCellSpec(
            PrMirrorGenerator(resolver=_stub_pr_resolver, inverter=_stub_pr_inverter),
            "python",
            True,
            _py_pr_repo,
            {},
        ),
    ]
    if include_go:
        specs.extend(
            [
                MenuCellSpec(
                    AstMutationGenerator(),
                    "go",
                    False,
                    _go_single_repo,
                    {"file": "calc.go"},
                ),
                MenuCellSpec(
                    FunctionRemovalGenerator(),
                    "go",
                    False,
                    _go_single_repo,
                    {"file": "calc.go"},
                ),
            ]
        )
    return specs


__all__ = [
    "CellResult",
    "MenuCellSpec",
    "MenuReport",
    "RoundTripResult",
    "build_coverage_matrix",
    "build_menu_cell_specs",
    "evaluate_cell",
    "evaluate_coverage",
    "go_toolchain_available",
    "run_menu_selfcheck",
    "schema_completeness",
    "verify_candidate_roundtrip",
]
