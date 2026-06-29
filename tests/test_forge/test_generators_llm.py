"""Unit tests for the LLM-backed generators (m3-gen-llm).

Covers VAL-GEN-011 (``lm_authored``: subtle single-function bug, no signposting,
round-trip sha256 match, target matches the edited symbol, teacher usage
recorded, and rejection of non-applying/non-inverting/over-broad teacher output)
and VAL-GEN-012 (``pr_mirror``: ``mutation_patch`` is the semantic inverse of a
real merged PR's diff on current code, round-trip match, PR number/sha recorded,
and rejection of non-inverting teacher output).

The teacher and the PR resolver are mocked so the deterministic logic (splice,
git round-trip, inversion gating) is exercised fully offline (no Docker, no live
LLM, no network). Behavior change is checked by importing the mutated module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
from pathlib import Path
from types import ModuleType

import pytest

from swe_forge.forge.adapters import PythonAdapter
from swe_forge.forge.adapters._diff import apply_multi_patch, apply_patch, make_patch
from swe_forge.forge.generators import (
    GenerationError,
    GenerationRequest,
    LmAuthoredGenerator,
    PrMirrorGenerator,
)
from swe_forge.forge.generators._llm import (
    extract_code_field,
    normalize_code,
    strip_trivia,
)
from swe_forge.forge.generators.lm_authored import AuthoredEdit, AuthoringContext
from swe_forge.forge.generators.pr_mirror import (
    InversionProposal,
    MergedPullRequest,
    PrFileChange,
    PrInversionContext,
    _parse_pr_files,
    _resolve_pr_number,
    _resolve_repo_slug,
    _wrap_file_diff,
)
from swe_forge.forge.models import BaselineNotGreenError, EnvImage

_LOAD_COUNTER = itertools.count()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _load_module(source: bytes) -> ModuleType:
    import tempfile

    name = f"_forge_llm_fixture_{next(_LOAD_COUNTER)}"
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as handle:
        handle.write(source)
        path = handle.name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PY_ADD = """\
def add(a, b):
    return a + b
"""


def _add_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "src/calc.py", PY_ADD)
    return root


def _red_env() -> EnvImage:
    return EnvImage(
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


# --------------------------------------------------------------------------- #
# lm_authored: a fake BugAuthor that performs a deterministic text edit.
# --------------------------------------------------------------------------- #
class _FakeAuthor:
    def __init__(self, transform, *, model: str = "anthropic/test") -> None:
        self._transform = transform
        self._model = model
        self.calls: list[AuthoringContext] = []

    def __call__(self, ctx: AuthoringContext) -> AuthoredEdit:
        self.calls.append(ctx)
        return AuthoredEdit(
            new_source=self._transform(ctx.function_source),
            model=self._model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            cost=0.001,
        )


def _subtle_bug(src: str) -> str:
    return src.replace("a + b", "a - b")


def test_lm_authored_subtle_single_function_bug(tmp_path: Path) -> None:
    repo = _add_repo(tmp_path)
    author = _FakeAuthor(_subtle_bug)
    candidate = LmAuthoredGenerator(author=author).generate(
        GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"), PythonAdapter()
    )

    assert candidate.generator == "lm_authored"
    assert candidate.language == "python"
    # Target matches the edited symbol.
    assert candidate.target.files == ("src/calc.py",)
    assert candidate.target.symbol == "add"

    rel = "src/calc.py"
    original = (repo / rel).read_bytes()
    mutated = apply_patch(original, candidate.mutation_patch, rel)
    assert mutated != original
    # Single-function: only the `add` body changed.
    assert b"def add(a, b):" in mutated
    assert b"return a - b" in mutated
    # Round-trip restores byte-for-byte (sha256 match).
    restored = apply_patch(mutated, candidate.oracle_patch, rel)
    assert _sha(restored) == _sha(original)
    # Behavior actually flips.
    assert _load_module(original).add(2, 3) == 5
    assert _load_module(mutated).add(2, 3) == -1

    # No bug-signposting in the added lines.
    added = "\n".join(
        line[1:]
        for line in candidate.mutation_patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    for marker in ("bug", "fixme", "intentional"):
        assert marker not in added.lower()

    # Teacher usage/cost recorded in provenance.
    teacher = candidate.provenance.details["teacher"]
    assert teacher["model"] == "anthropic/test"
    assert teacher["usage"]["total_tokens"] == 15
    assert teacher["cost"] == 0.001


def test_lm_authored_repo_left_pristine(tmp_path: Path) -> None:
    repo = _add_repo(tmp_path)
    before = (repo / "src/calc.py").read_bytes()
    LmAuthoredGenerator(author=_FakeAuthor(_subtle_bug)).generate(
        GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"), PythonAdapter()
    )
    assert (repo / "src/calc.py").read_bytes() == before


def test_lm_authored_noop_edit_is_rejected(tmp_path: Path) -> None:
    repo = _add_repo(tmp_path)
    author = _FakeAuthor(lambda src: src)
    with pytest.raises(GenerationError):
        LmAuthoredGenerator(author=author).generate(
            GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"),
            PythonAdapter(),
        )


def test_lm_authored_signposted_bug_is_rejected(tmp_path: Path) -> None:
    repo = _add_repo(tmp_path)
    author = _FakeAuthor(lambda src: src.replace("a + b", "a - b  # intentional bug"))
    with pytest.raises(GenerationError):
        LmAuthoredGenerator(author=author).generate(
            GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"),
            PythonAdapter(),
        )


def test_lm_authored_unparseable_output_is_rejected(tmp_path: Path) -> None:
    repo = _add_repo(tmp_path)
    author = _FakeAuthor(lambda src: "def add(a, b)\n    return a - b")
    with pytest.raises(GenerationError):
        LmAuthoredGenerator(author=author).generate(
            GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"),
            PythonAdapter(),
        )


def test_lm_authored_added_symbol_is_rejected(tmp_path: Path) -> None:
    repo = _add_repo(tmp_path)
    author = _FakeAuthor(
        lambda src: (
            "def add(a, b):\n    return a - b\n\n\ndef helper():\n    return 0\n"
        )
    )
    with pytest.raises(GenerationError):
        LmAuthoredGenerator(author=author).generate(
            GenerationRequest(repo_root=repo, seed=0, file="src/calc.py"),
            PythonAdapter(),
        )


def test_lm_authored_gates_on_green_baseline(tmp_path: Path) -> None:
    repo = _add_repo(tmp_path)
    author = _FakeAuthor(_subtle_bug)
    with pytest.raises(BaselineNotGreenError):
        LmAuthoredGenerator(author=author).generate(
            GenerationRequest(
                repo_root=repo, seed=0, file="src/calc.py", env_image=_red_env()
            ),
            PythonAdapter(),
        )
    # No teacher call happened (gate is checked first).
    assert author.calls == []


def test_lm_authored_no_target_function_raises(tmp_path: Path) -> None:
    _write(tmp_path, "empty.py", "X = 1\nY = 2\n")
    author = _FakeAuthor(_subtle_bug)
    with pytest.raises(GenerationError):
        LmAuthoredGenerator(author=author).generate(
            GenerationRequest(repo_root=tmp_path, seed=0, file="empty.py"),
            PythonAdapter(),
        )


# --------------------------------------------------------------------------- #
# pr_mirror: a fake resolver (a real-looking merged PR) + a fake inverter.
# --------------------------------------------------------------------------- #
PR_BASE = "def f():\n    return 1\n"
PR_HEAD = "def f():\n    return 2\n"  # the PR changed return 1 -> return 2


def _pr_repo(root: Path) -> Path:
    _write(root, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(root, "mod.py", PR_HEAD)  # current code = the PR's post-merge state
    return root


def _merged_pr() -> MergedPullRequest:
    pr_diff = make_patch("mod.py", PR_BASE.encode(), PR_HEAD.encode())
    return MergedPullRequest(
        number=42,
        sha="a" * 40,
        repo="octocat/demo",
        files=[PrFileChange(path="mod.py", patch=pr_diff)],
        url="https://github.com/octocat/demo/pull/42",
        title="Return 2 instead of 1",
    )


class _FakeResolver:
    def __init__(self, pr: MergedPullRequest) -> None:
        self._pr = pr

    def __call__(self, repo_root: Path, params: dict[str, object]) -> MergedPullRequest:
        return self._pr


class _FakeInverter:
    def __init__(self, reverted: dict[str, str]) -> None:
        self._reverted = reverted
        self.calls: list[PrInversionContext] = []

    def __call__(self, ctx: PrInversionContext) -> InversionProposal:
        self.calls.append(ctx)
        return InversionProposal(
            reverted=dict(self._reverted),
            model="anthropic/test",
            usage=[
                {
                    "model": "anthropic/test",
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                    },
                    "cost": 0.002,
                }
            ],
        )


def test_pr_mirror_inverts_a_real_pr(tmp_path: Path) -> None:
    repo = _pr_repo(tmp_path)
    resolver = _FakeResolver(_merged_pr())
    inverter = _FakeInverter({"mod.py": PR_BASE})
    candidate = PrMirrorGenerator(resolver=resolver, inverter=inverter).generate(
        GenerationRequest(repo_root=repo, seed=0), PythonAdapter()
    )

    assert candidate.generator == "pr_mirror"
    assert candidate.target.files == ("mod.py",)

    rel = "mod.py"
    current = (repo / rel).read_bytes()
    # mutation reverts current (head) back to the pre-PR (base) content.
    reverted = apply_multi_patch({rel: current}, candidate.mutation_patch)[rel]
    assert reverted == PR_BASE.encode()
    assert _load_module(current).f() == 2
    assert _load_module(reverted).f() == 1
    # Oracle reinstates the PR (round-trip byte-for-byte).
    restored = apply_multi_patch({rel: reverted}, candidate.oracle_patch)[rel]
    assert _sha(restored) == _sha(current)

    # PR ref + teacher usage recorded.
    details = candidate.provenance.details
    assert details["pr_number"] == 42
    assert details["pr_sha"] == "a" * 40
    assert details["repo"] == "octocat/demo"
    assert details["teacher"]["usage"]["total_tokens"] == 28


def test_pr_mirror_records_pr_test_files(tmp_path: Path) -> None:
    """A PR touching source + a test file records the test file as F2P input.

    The source file is reverted (the manufactured fault); the PR's own test file
    is NOT reverted but is recorded in provenance so the pilot can run it as the
    isolated F2P (it FAILS once the source regresses, PASSES on gold).
    """
    repo = _pr_repo(tmp_path)
    _write(repo, "test_mod.py", PR_HEAD)
    pr = MergedPullRequest(
        number=99,
        sha="c" * 40,
        repo="octocat/demo",
        files=[
            PrFileChange(
                path="mod.py",
                patch=make_patch("mod.py", PR_BASE.encode(), PR_HEAD.encode()),
            ),
            PrFileChange(
                path="test_mod.py",
                patch=make_patch("test_mod.py", PR_BASE.encode(), PR_HEAD.encode()),
            ),
        ],
    )
    candidate = PrMirrorGenerator(
        resolver=_FakeResolver(pr),
        inverter=_FakeInverter({"mod.py": PR_BASE}),
    ).generate(GenerationRequest(repo_root=repo, seed=0), PythonAdapter())

    # Only the source file is reverted; the test file is recorded, not mutated.
    assert candidate.target.files == ("mod.py",)
    assert candidate.provenance.details["test_files"] == ["test_mod.py"]


def test_pr_mirror_records_no_test_files_when_pr_has_none(tmp_path: Path) -> None:
    repo = _pr_repo(tmp_path)
    candidate = PrMirrorGenerator(
        resolver=_FakeResolver(_merged_pr()),
        inverter=_FakeInverter({"mod.py": PR_BASE}),
    ).generate(GenerationRequest(repo_root=repo, seed=0), PythonAdapter())
    assert candidate.provenance.details["test_files"] == []


def test_pr_mirror_mutation_is_semantic_inverse_of_pr_diff(tmp_path: Path) -> None:
    repo = _pr_repo(tmp_path)
    pr = _merged_pr()
    candidate = PrMirrorGenerator(
        resolver=_FakeResolver(pr), inverter=_FakeInverter({"mod.py": PR_BASE})
    ).generate(GenerationRequest(repo_root=repo, seed=0), PythonAdapter())

    rel = "mod.py"
    current = (repo / rel).read_bytes()
    # Reverse-applying the PR diff to current yields exactly the mutation target.
    via_pr = apply_patch(current, pr.files[0].patch, rel, reverse=True)
    via_mutation = apply_multi_patch({rel: current}, candidate.mutation_patch)[rel]
    assert via_mutation == via_pr


def test_pr_mirror_non_inverting_teacher_output_is_rejected(tmp_path: Path) -> None:
    repo = _pr_repo(tmp_path)
    inverter = _FakeInverter({"mod.py": "def f():\n    return 7\n"})
    with pytest.raises(GenerationError, match="non-inverting"):
        PrMirrorGenerator(
            resolver=_FakeResolver(_merged_pr()), inverter=inverter
        ).generate(GenerationRequest(repo_root=repo, seed=0), PythonAdapter())


def test_pr_mirror_drifted_file_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path
    _write(repo, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    # Current code no longer matches the PR's post-image, so the PR diff cannot
    # reverse-apply: nothing usable, reject cleanly.
    _write(repo, "mod.py", "def f():\n    return 99\n")
    with pytest.raises(GenerationError):
        PrMirrorGenerator(
            resolver=_FakeResolver(_merged_pr()),
            inverter=_FakeInverter({"mod.py": PR_BASE}),
        ).generate(GenerationRequest(repo_root=repo, seed=0), PythonAdapter())


def test_pr_mirror_skips_test_files(tmp_path: Path) -> None:
    repo = tmp_path
    _write(repo, "pyproject.toml", "[project]\nname='demo'\nversion='0'\n")
    _write(repo, "test_mod.py", PR_HEAD)
    pr_diff = make_patch("test_mod.py", PR_BASE.encode(), PR_HEAD.encode())
    pr = MergedPullRequest(
        number=7,
        sha="b" * 40,
        repo="octocat/demo",
        files=[PrFileChange(path="test_mod.py", patch=pr_diff)],
    )
    with pytest.raises(GenerationError):
        PrMirrorGenerator(
            resolver=_FakeResolver(pr),
            inverter=_FakeInverter({"test_mod.py": PR_BASE}),
        ).generate(GenerationRequest(repo_root=repo, seed=0), PythonAdapter())


def test_pr_mirror_gates_on_green_baseline(tmp_path: Path) -> None:
    repo = _pr_repo(tmp_path)
    inverter = _FakeInverter({"mod.py": PR_BASE})
    with pytest.raises(BaselineNotGreenError):
        PrMirrorGenerator(
            resolver=_FakeResolver(_merged_pr()), inverter=inverter
        ).generate(
            GenerationRequest(repo_root=repo, seed=0, env_image=_red_env()),
            PythonAdapter(),
        )
    assert inverter.calls == []


# --------------------------------------------------------------------------- #
# pr_mirror resolver helpers (offline)
# --------------------------------------------------------------------------- #
def test_resolve_pr_number_validates() -> None:
    assert _resolve_pr_number({"pr_number": 5}) == 5
    assert _resolve_pr_number({"pr_number": "12"}) == 12
    with pytest.raises(GenerationError):
        _resolve_pr_number({})
    with pytest.raises(GenerationError):
        _resolve_pr_number({"pr_number": 0})


def test_resolve_repo_slug_prefers_params(tmp_path: Path) -> None:
    assert _resolve_repo_slug(tmp_path, {"repo": "owner/name"}) == "owner/name"
    with pytest.raises(GenerationError):
        _resolve_repo_slug(tmp_path, {})


def test_wrap_file_diff_is_git_applyable() -> None:
    diff = _wrap_file_diff("mod.py", "@@ -1 +1 @@\n-old\n+new")
    assert diff.startswith("diff --git a/mod.py b/mod.py\n")
    assert "--- a/mod.py" in diff
    assert "+++ b/mod.py" in diff
    assert diff.endswith("\n")


def test_parse_pr_files_keeps_modified_text_files() -> None:
    payload = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-x\n+y"},
        {"filename": "b.py", "status": "added", "patch": "@@ -0,0 +1 @@\n+z"},
        {"filename": "c.bin", "status": "modified"},  # no patch (binary)
    ]
    changes = _parse_pr_files(payload)
    assert [c.path for c in changes] == ["a.py"]
    assert changes[0].patch.startswith("diff --git a/a.py b/a.py")


# --------------------------------------------------------------------------- #
# Shared teacher-reply parsing helpers (offline)
# --------------------------------------------------------------------------- #
def test_extract_code_field_handles_wrappings() -> None:
    keys = ("function", "source", "code")
    func = "def f():\n    return 1"
    plain = '{"source": "def f():\\n    return 1"}'
    # Some endpoints double-wrap: a JSON string nested in the requested field.
    nested = '{"function": "{\\"source\\": \\"def f():\\\\n    return 1\\"}"}'
    fenced = '```json\n{"function": "def f():\\n    return 1"}\n```'
    single = '{"answer": "def f():\\n    return 1"}'
    for payload in (plain, nested, fenced, single):
        assert extract_code_field(payload, keys) == func
    assert extract_code_field("", keys) == ""


def test_normalize_code_keeps_indentation_drops_trivia() -> None:
    indented = "def f():\n    if x:\n        return 1\n    return 2\n"
    reindented = "def f():\n    if x:\n        return 1\n        return 2\n"
    # A Python re-indentation is a real (control-flow) change, not trivia.
    assert normalize_code(indented) != normalize_code(reindented)
    # Comments / blank lines / trailing whitespace are trivia.
    with_trivia = "def f():  \n\n    return 1  # note\n"
    without = "def f():\n    return 1\n"
    assert normalize_code(with_trivia) == normalize_code(without)


def test_strip_trivia_collapses_all_whitespace() -> None:
    assert strip_trivia("def  f( ):\n    return   1\n") == strip_trivia(
        "def f():\n        return 1\n"
    )
