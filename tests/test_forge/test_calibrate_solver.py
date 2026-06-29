"""Unit tests for the calibration solver scaffold (m5-solver).

Offline coverage (no real Docker, no live LLM) of the solver's contract
assertions, driven through fakes:

- VAL-CAL-001: the solver runs a tool loop and captures a unified-diff patch per
  rollout, with per-rollout usage/cost recorded.
- VAL-CAL-003: a submitted patch is scored through the SAME Docker FAIL->PASS
  recipe; a solve requires every hidden test to pass AND P2P green - a
  P2P-regressing patch scores non-solve.
- VAL-CAL-004: submit-gating - only a finished, non-empty, applying patch is
  scored; over-budget (no finish), empty diff, and non-applying patches record
  ``solve = false`` cleanly without crashing the runner.
- VAL-CAL-005: the solver prompt exposes ONLY statement/requirements/interface,
  never the mutation/oracle patch text or hidden-test bodies.

The DockerSandbox + live teacher paths are exercised by the worker/user-testing
validator in real Docker (see this feature's manual verification).
"""

from __future__ import annotations

import asyncio
import logging
import os

import pytest

from swe_forge.forge.calibrate.solver import (
    REASON_APPLY_FAILED,
    REASON_EMPTY_PATCH,
    REASON_F2P_OR_P2P_FAILED,
    REASON_NOT_FINISHED,
    REASON_ROLLOUT_ERROR,
    AgenticSolver,
    RolloutOutcome,
    SolveScore,
    SolverContext,
    build_solver_prompt,
    capture_workspace_patch,
    run_solver_rollout,
    score_patch,
    solver_tools,
)
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    EnvImage,
    GeneratedSpec,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    HiddenTest,
    HiddenTestFile,
)
from swe_forge.forge.teacher import AgenticResult, NormalizedToolCall, Usage

P2P = "python -m pytest"
F2P = "python -m pytest tests/test_subtract.py"
# The solver's submitted fix (broken `a + b` -> `a - b`), as a unified diff.
SOLVER_PATCH = (
    "diff --git a/calc.py b/calc.py\n"
    "--- a/calc.py\n+++ b/calc.py\n"
    "@@ -3,4 +3,4 @@ def add(a, b):\n \n \n def subtract(a, b):\n"
    "-    return a + b\n+    return a - b\n"
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeExec:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class FakeSandbox:
    """Records every command and simulates a tiny filesystem.

    ``responses`` maps an EXACT command string to an exit code (default 0). A
    ``git ... diff --cached`` capture returns ``diff_output`` so a rollout's
    submitted patch can be exercised deterministically.
    """

    def __init__(
        self, responses: dict[str, int] | None = None, *, diff_output: str = ""
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.files: dict[str, str] = {}
        self._responses = responses or {}
        self.diff_output = diff_output

    async def run_command(
        self,
        cmd: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> FakeExec:
        self.calls.append({"cmd": cmd, "env": env, "timeout": timeout})
        if "diff --cached" in cmd:
            return FakeExec(exit_code=0, stdout=self.diff_output)
        return FakeExec(exit_code=self._responses.get(cmd, 0))

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def read_file(self, path: str) -> str:
        return self.files.get(path, "")


class FakeTeacher:
    """Replays a scripted tool sequence, accumulating usage/cost like the real one."""

    def __init__(
        self,
        script: list[tuple[str, dict[str, object]]],
        *,
        usage: Usage | None = None,
        cost: float = 0.0,
    ) -> None:
        self._script = script
        self._usage = usage or Usage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        self._cost = cost

    async def agentic_turn(
        self, messages, tools, tool_executor, *, max_turns, max_tokens
    ) -> AgenticResult:
        for idx, (name, args) in enumerate(self._script):
            call = NormalizedToolCall(id=str(idx), name=name, arguments=args)
            await tool_executor(call)
        return AgenticResult(
            text="done",
            turns=len(self._script) or 1,
            usage=self._usage,
            cost=self._cost,
        )


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _candidate() -> Candidate:
    return Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("calc.py",), symbols=("subtract",)),
        mutation_patch=(
            "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n"
            "@@ -3,4 +3,4 @@ def add(a, b):\n \n \n def subtract(a, b):\n"
            "-    return a - b\n+    return a + b\n"
        ),
        oracle_patch=(
            "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n"
            "@@ -3,4 +3,4 @@ def add(a, b):\n \n \n def subtract(a, b):\n"
            "-    return a + b\n+    return a - b\n"
        ),
        difficulty_hint="easy",
        provenance=Provenance(generator="ast_mutation", seed=1, language="python"),
    )


def _env_image() -> EnvImage:
    return EnvImage(
        repo_id="py-oracle",
        language="python",
        image_tag="swe-forge-env-py-oracle:abc123",
        base_image="python:3.12-slim",
        commit="0" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command=P2P,
        baseline_green=True,
        baseline_exit_code=0,
    )


def _spec() -> GeneratedSpec:
    return GeneratedSpec(
        problem_statement=(
            "subtract(a, b) returns the wrong value; calling subtract(5, 3) yields 8 "
            "but the difference 2 is expected."
        ),
        requirements=["subtract(a, b) must return the difference a - b."],
        interface_block="def subtract(a, b): ...",
        provenance=Provenance(generator="ast_mutation", seed=1, language="python"),
    )


def _oracle_report() -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=[F2P],
        pass_to_pass=[P2P],
        test_files=[
            OracleTestFile(
                path="tests/test_subtract.py",
                content="import calc\n\ndef test_subtract():\n    assert calc.subtract(5, 3) == 2\n",
                origin="provided",
            )
        ],
    )


def _recipe(sandbox: FakeSandbox) -> DockerOracleRecipe:
    cand = _candidate()
    return DockerOracleRecipe(
        sandbox,
        language="python",
        workspace_dir="/workspace/repo",
        mutation_patch=cand.mutation_patch,
        oracle_patch=cand.oracle_patch,
        p2p_command=P2P,
    )


def _base_tests() -> list[HiddenTest]:
    return [
        HiddenTest(
            test_id=F2P,
            files=(HiddenTestFile(path="tests/test_subtract.py", content="x"),),
            origin="provided",
        )
    ]


# --------------------------------------------------------------------------- #
# Prompt surface (VAL-CAL-005)
# --------------------------------------------------------------------------- #
def test_build_solver_prompt_contains_only_spec_surface() -> None:
    spec = _spec()
    prompt = build_solver_prompt(spec, "python")
    assert spec.problem_statement in prompt
    assert spec.requirements[0] in prompt
    assert spec.interface_block in prompt
    # Never the gold/oracle implementation nor the mutation diff text.
    cand = _candidate()
    assert "return a - b" not in prompt  # gold body must not leak
    assert "mutation_patch" not in prompt
    assert cand.oracle_patch not in prompt
    assert "@@" not in prompt  # no diff hunks


def test_solver_tools_exposes_four_tools() -> None:
    names = {t["function"]["name"] for t in solver_tools()}
    assert names == {"shell", "read_file", "write_file", "finish"}


# --------------------------------------------------------------------------- #
# Rollout: tool loop + patch capture (VAL-CAL-001)
# --------------------------------------------------------------------------- #
async def test_solver_captures_finish_patch_with_usage() -> None:
    sandbox = FakeSandbox(diff_output=SOLVER_PATCH)
    script: list[tuple[str, dict[str, object]]] = [
        ("read_file", {"path": "calc.py"}),
        (
            "write_file",
            {"path": "calc.py", "content": "def subtract(a, b):\n    return a - b\n"},
        ),
        ("finish", {"summary": "fixed subtract"}),
    ]
    solver = AgenticSolver(client=FakeTeacher(script, cost=0.002))  # type: ignore[arg-type]
    rollout = await solver.solve(
        SolverContext(spec=_spec(), language="python", sandbox=sandbox)
    )

    assert rollout.finished is True
    assert rollout.patch == SOLVER_PATCH
    assert "calc.py" in sandbox.files
    # per-rollout usage/cost recorded
    assert rollout.usage.total_tokens == 15
    assert rollout.cost == 0.002
    assert rollout.error is None


async def test_solver_over_budget_without_finish_is_unfinished() -> None:
    sandbox = FakeSandbox(diff_output=SOLVER_PATCH)
    # The model edits but never calls finish (exhausts its budget).
    script: list[tuple[str, dict[str, object]]] = [
        ("write_file", {"path": "calc.py", "content": "whatever"}),
        ("shell", {"command": "ls"}),
    ]
    solver = AgenticSolver(client=FakeTeacher(script))  # type: ignore[arg-type]
    rollout = await solver.solve(
        SolverContext(spec=_spec(), language="python", sandbox=sandbox)
    )
    assert rollout.finished is False
    assert rollout.patch == ""  # nothing submitted


async def test_solver_finish_with_no_edits_captures_empty_patch() -> None:
    sandbox = FakeSandbox(diff_output="")  # git diff --cached is empty
    script: list[tuple[str, dict[str, object]]] = [("finish", {})]
    solver = AgenticSolver(client=FakeTeacher(script))  # type: ignore[arg-type]
    rollout = await solver.solve(
        SolverContext(spec=_spec(), language="python", sandbox=sandbox)
    )
    assert rollout.finished is True
    assert rollout.patch == ""


async def test_solver_aborted_loop_is_clean_non_finish() -> None:
    class BoomTeacher:
        async def agentic_turn(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("endpoint unreachable")

    solver = AgenticSolver(client=BoomTeacher())  # type: ignore[arg-type]
    rollout = await solver.solve(
        SolverContext(spec=_spec(), language="python", sandbox=FakeSandbox())
    )
    assert rollout.finished is False
    assert rollout.patch == ""
    assert rollout.error is not None


# --------------------------------------------------------------------------- #
# Scoring via the shared recipe (VAL-CAL-003)
# --------------------------------------------------------------------------- #
async def test_score_patch_solve_requires_f2p_and_p2p_green() -> None:
    sandbox = FakeSandbox()  # all commands exit 0 by default
    score = await score_patch(_recipe(sandbox), SOLVER_PATCH, _base_tests())
    assert score.solved is True
    assert score.applied is True
    assert score.f2p_passed is True
    assert score.p2p_passed is True
    assert score.failing_test_ids == ()
    # scored on the BROKEN tree: the mutation patch was applied first
    applied_cmds = [
        str(c["cmd"]) for c in sandbox.calls if "git apply" in str(c["cmd"])
    ]
    assert any("mutation.patch" in c for c in applied_cmds)
    assert any("solver.patch" in c for c in applied_cmds)


async def test_score_patch_p2p_regression_is_non_solve() -> None:
    # F2P passes but the patch regresses the P2P/regression suite.
    sandbox = FakeSandbox(responses={P2P: 1})
    score = await score_patch(_recipe(sandbox), SOLVER_PATCH, _base_tests())
    assert score.solved is False
    assert score.f2p_passed is True
    assert score.p2p_passed is False
    assert P2P in score.failing_test_ids
    assert score.reason == REASON_F2P_OR_P2P_FAILED


async def test_score_patch_f2p_failure_is_non_solve() -> None:
    sandbox = FakeSandbox(responses={F2P: 1})
    score = await score_patch(_recipe(sandbox), SOLVER_PATCH, _base_tests())
    assert score.solved is False
    assert score.f2p_passed is False
    assert score.p2p_passed is True
    assert F2P in score.failing_test_ids


async def test_score_patch_enforces_full_test_files_set() -> None:
    # An extra hidden test (e.g. a differential survivor-killer) must also pass.
    extra = "python -m pytest tests/test_extra.py"
    tests = [
        *_base_tests(),
        HiddenTest(
            test_id=extra,
            files=(HiddenTestFile(path="tests/test_extra.py", content="y"),),
            origin="synthesized",
        ),
    ]
    sandbox = FakeSandbox(responses={extra: 1})  # the extra hidden test fails
    score = await score_patch(_recipe(sandbox), SOLVER_PATCH, tests)
    assert score.solved is False
    assert extra in score.failing_test_ids


async def test_score_patch_non_applying_is_non_solve() -> None:
    solver_apply = "git apply --whitespace=nowarn .swe_forge_solver/solver.patch"
    solver_apply_3way = (
        "git apply --3way --whitespace=nowarn .swe_forge_solver/solver.patch"
    )
    sandbox = FakeSandbox(responses={solver_apply: 1, solver_apply_3way: 1})
    score = await score_patch(_recipe(sandbox), SOLVER_PATCH, _base_tests())
    assert score.solved is False
    assert score.applied is False
    assert score.reason == REASON_APPLY_FAILED


async def test_score_patch_empty_is_non_solve() -> None:
    score = await score_patch(_recipe(FakeSandbox()), "   \n", _base_tests())
    assert score.solved is False
    assert score.empty is True
    assert score.reason == REASON_EMPTY_PATCH


# --------------------------------------------------------------------------- #
# Orchestration submit-gating (VAL-CAL-004)
# --------------------------------------------------------------------------- #
def _patch_rollout(monkeypatch, rollout) -> None:
    async def fake_rollout(*args, **kwargs):  # noqa: ANN002, ANN003
        return rollout

    monkeypatch.setattr(
        "swe_forge.forge.calibrate.solver._run_rollout_in_docker", fake_rollout
    )


def _forbid_scoring(monkeypatch) -> None:
    async def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("scoring must not run for a non-submitted patch")

    monkeypatch.setattr("swe_forge.forge.calibrate.solver._score_rollout", boom)


async def test_run_rollout_not_finished_non_solve_no_scoring(monkeypatch) -> None:
    from swe_forge.forge.calibrate.solver import SolverRollout

    _patch_rollout(
        monkeypatch,
        SolverRollout(patch="", finished=False, turns=20, usage=Usage(), cost=0.0),
    )
    _forbid_scoring(monkeypatch)
    outcome = await run_solver_rollout(
        _candidate(), _env_image(), _spec(), _oracle_report(), model="frontier/x"
    )
    assert outcome.solved is False
    assert outcome.score.reason == REASON_NOT_FINISHED
    assert outcome.model == "frontier/x"


async def test_run_rollout_empty_patch_non_solve_no_scoring(monkeypatch) -> None:
    from swe_forge.forge.calibrate.solver import SolverRollout

    _patch_rollout(
        monkeypatch,
        SolverRollout(patch="   ", finished=True, turns=3, usage=Usage(), cost=0.0),
    )
    _forbid_scoring(monkeypatch)
    outcome = await run_solver_rollout(
        _candidate(), _env_image(), _spec(), _oracle_report()
    )
    assert outcome.solved is False
    assert outcome.score.empty is True
    assert outcome.score.reason == REASON_EMPTY_PATCH


async def test_run_rollout_finished_patch_is_scored(monkeypatch) -> None:
    from swe_forge.forge.calibrate.solver import SolverRollout

    _patch_rollout(
        monkeypatch,
        SolverRollout(
            patch=SOLVER_PATCH,
            finished=True,
            turns=4,
            usage=Usage(total_tokens=42),
            cost=0.01,
        ),
    )

    async def fake_score(*args, **kwargs):  # noqa: ANN002, ANN003
        return SolveScore(
            solved=True,
            applied=True,
            empty=False,
            f2p_passed=True,
            p2p_passed=True,
        )

    monkeypatch.setattr("swe_forge.forge.calibrate.solver._score_rollout", fake_score)
    outcome = await run_solver_rollout(
        _candidate(), _env_image(), _spec(), _oracle_report(), model="frontier/x"
    )
    assert outcome.solved is True
    assert outcome.usage.total_tokens == 42
    assert outcome.cost == 0.01
    assert isinstance(outcome, RolloutOutcome)


async def test_run_rollout_error_is_clean_non_solve(monkeypatch) -> None:
    async def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("docker exploded")

    monkeypatch.setattr("swe_forge.forge.calibrate.solver._run_rollout_in_docker", boom)
    outcome = await run_solver_rollout(
        _candidate(), _env_image(), _spec(), _oracle_report()
    )
    assert outcome.solved is False
    assert outcome.score.reason.startswith(REASON_ROLLOUT_ERROR)
    assert outcome.error is not None


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def test_rollout_outcome_serializable() -> None:
    outcome = RolloutOutcome(
        model="frontier/x",
        patch=SOLVER_PATCH,
        finished=True,
        solved=True,
        score=SolveScore(
            solved=True, applied=True, empty=False, f2p_passed=True, p2p_passed=True
        ),
        turns=4,
        usage=Usage(total_tokens=42),
        cost=0.01,
    )
    data = outcome.to_dict()
    assert data["solved"] is True
    assert data["score"]["solved"] is True
    assert data["usage"]["total_tokens"] == 42
    assert data["patch_bytes"] == len(SOLVER_PATCH)


def test_solve_score_serializable() -> None:
    score = SolveScore(
        solved=False,
        applied=True,
        empty=False,
        f2p_passed=True,
        p2p_passed=False,
        failing_test_ids=(P2P,),
        reason=REASON_F2P_OR_P2P_FAILED,
    )
    data = score.to_dict()
    assert data["failing_test_ids"] == [P2P]
    assert data["reason"] == REASON_F2P_OR_P2P_FAILED


def test_require_green_baseline_precondition() -> None:
    from swe_forge.forge.models import BaselineNotGreenError

    red = _env_image()
    red.baseline_green = False
    red.baseline_exit_code = 1
    # run_solver_rollout enforces require_green_baseline before any Docker work.
    with pytest.raises(BaselineNotGreenError):
        asyncio.run(run_solver_rollout(_candidate(), red, _spec(), _oracle_report()))


# --------------------------------------------------------------------------- #
# No gold leak via .git history (VAL-CAL-005; AGENTS.md "No gold leak via .git
# history"). Proven for real against a temp git repo (no Docker, no LLM): the
# broken baseline must be a single ORPHAN/root commit and gold must be
# unrecoverable from history, while patch capture still works.
# --------------------------------------------------------------------------- #
_TEST_IDENT = "-c user.email=t@localhost -c user.name=t"
GOLD_CALC = "def subtract(a, b):\n    return a - b\n"
BROKEN_CALC = "def subtract(a, b):\n    return a + b\n"
_MUTATION_DIFF = (
    "diff --git a/calc.py b/calc.py\n"
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def subtract(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)


class _LocalExec:
    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class LocalGitSandbox:
    """A real-filesystem sandbox: runs shell/git in a temp workspace.

    Implements the :class:`SandboxProtocol` surface against a real directory so
    the git-history protection can be proven for real (single orphan commit,
    ``git show HEAD~1`` fails, gold unrecoverable) with no Docker and no LLM.
    """

    def __init__(self, root) -> None:  # noqa: ANN001
        self.root = root

    async def run_command(self, cmd, *, cwd=None, timeout=None, env=None) -> _LocalExec:  # noqa: ANN001
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(cwd or self.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(env or {})},
        )
        out, err = await proc.communicate()
        return _LocalExec(
            proc.returncode if proc.returncode is not None else 0,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    async def write_file(self, path: str, content: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")


def _history_candidate() -> Candidate:
    return Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("calc.py",), symbols=("subtract",)),
        mutation_patch=_MUTATION_DIFF,
        oracle_patch=(
            "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n def subtract(a, b):\n"
            "-    return a + b\n+    return a - b\n"
        ),
        difficulty_hint="easy",
        provenance=Provenance(generator="ast_mutation", seed=1, language="python"),
    )


async def _make_gold_repo(root) -> LocalGitSandbox:  # noqa: ANN001
    """A temp repo whose HEAD = the pinned GOLD commit (as an EnvImage checkout)."""
    sandbox = LocalGitSandbox(root)
    await sandbox.write_file("calc.py", GOLD_CALC)
    await sandbox.write_file("tests/test_calc.py", "import calc\n")
    init = await sandbox.run_command(
        f"git init -q && git {_TEST_IDENT} add -A && "
        f"git {_TEST_IDENT} commit -q -m gold"
    )
    assert init.exit_code == 0, init.stderr
    return sandbox


async def test_setup_broken_baseline_single_orphan_commit(tmp_path) -> None:
    from swe_forge.forge.calibrate.solver import _setup_broken_baseline

    sandbox = await _make_gold_repo(tmp_path)
    await _setup_broken_baseline(sandbox, _history_candidate(), timeout=60.0)

    log = await sandbox.run_command("git log --oneline")
    assert log.exit_code == 0
    assert len([ln for ln in log.stdout.splitlines() if ln.strip()]) == 1

    count = await sandbox.run_command("git rev-list --count HEAD")
    assert count.stdout.strip() == "1"

    # No parent: HEAD~1 / HEAD^ do not resolve.
    parent = await sandbox.run_command("git show HEAD~1")
    assert parent.exit_code != 0


async def test_setup_broken_baseline_gold_unrecoverable(tmp_path) -> None:
    from swe_forge.forge.calibrate.solver import _setup_broken_baseline

    sandbox = await _make_gold_repo(tmp_path)
    await _setup_broken_baseline(sandbox, _history_candidate(), timeout=60.0)

    # HEAD:<path> returns the BROKEN content, never gold.
    head = await sandbox.run_command("git show HEAD:calc.py")
    assert head.exit_code == 0
    assert "return a + b" in head.stdout
    assert "return a - b" not in head.stdout

    # Adversarial "mine the git history" probes all fail (no recoverable ancestor).
    for probe in (
        "git show HEAD~1:calc.py",
        "git checkout HEAD~1 -- calc.py",
        "git diff HEAD~1 HEAD",
        "git cat-file -p HEAD^",
    ):
        res = await sandbox.run_command(probe)
        assert res.exit_code != 0, f"{probe!r} unexpectedly succeeded"

    # Gold appears in no reachable commit, and the working tree is broken.
    log_all = await sandbox.run_command("git log --all -p")
    assert "return a - b" not in log_all.stdout
    assert (tmp_path / "calc.py").read_text() == BROKEN_CALC


async def test_capture_patch_excludes_stray_untracked_files(tmp_path, caplog) -> None:
    sandbox = await _make_gold_repo(tmp_path)
    from swe_forge.forge.calibrate.solver import _setup_broken_baseline

    await _setup_broken_baseline(sandbox, _history_candidate(), timeout=60.0)

    # Model restores gold (a tracked-source edit) and drops scratch files.
    await sandbox.write_file("calc.py", GOLD_CALC)
    await sandbox.write_file("scratch_notes.txt", "debugging\n")
    await sandbox.write_file("junk/out.log", "noise\n")

    with caplog.at_level(logging.WARNING):
        patch = await capture_workspace_patch(sandbox, timeout=60.0)

    assert patch.strip()
    assert "calc.py" in patch
    assert "return a - b" in patch  # the model's fix is captured
    assert "scratch_notes.txt" not in patch
    assert "out.log" not in patch
    # Stray files are warned about, not silently folded in.
    assert "untracked file" in caplog.text
    assert "scratch_notes.txt" in caplog.text


async def test_capture_patch_applies_and_restores_gold(tmp_path) -> None:
    from swe_forge.forge.calibrate.solver import _setup_broken_baseline

    sandbox = await _make_gold_repo(tmp_path)
    await _setup_broken_baseline(sandbox, _history_candidate(), timeout=60.0)

    await sandbox.write_file("calc.py", GOLD_CALC)
    patch = await capture_workspace_patch(sandbox, timeout=60.0)
    assert patch.strip()

    # Reset to the broken baseline, then the captured diff must apply cleanly.
    await sandbox.run_command("git reset --hard -q HEAD")
    assert (tmp_path / "calc.py").read_text() == BROKEN_CALC
    await sandbox.write_file(".captured.patch", patch)
    applied = await sandbox.run_command("git apply --whitespace=nowarn .captured.patch")
    assert applied.exit_code == 0, applied.stderr
    assert (tmp_path / "calc.py").read_text() == GOLD_CALC


async def test_setup_broken_baseline_reinitializes_history() -> None:
    # Fast, git-free structural guard: the broken baseline must be created by a
    # `.git` removal + `git init` + a SINGLE `broken-baseline` commit, never the
    # old "commit on top of the gold history" path (which leaked gold).
    from swe_forge.forge.calibrate.solver import _setup_broken_baseline

    sandbox = FakeSandbox()
    await _setup_broken_baseline(sandbox, _candidate(), timeout=30.0)
    cmds = [str(c["cmd"]) for c in sandbox.calls]

    assert any("git apply" in c and "mutation.patch" in c for c in cmds)
    commits = [c for c in cmds if "commit" in c]
    assert len(commits) == 1
    assert "git init" in commits[0]
    assert ".git" in commits[0]
    assert "broken-baseline" in commits[0]
    assert "forge-broken" not in " ".join(cmds)  # old child-commit message is gone


# --------------------------------------------------------------------------- #
# Build-artifact capture hardening (m6-calib-pyc-build-artifact-fix). An
# EnvImage's baseline run can leave build artifacts (CPython `__pycache__/*.pyc`,
# `*.egg-info`) in the tree. If the candidate repo ships no `.gitignore` they were
# TRACKED in the broken baseline; the model running code regenerated a `.pyc` and
# `git add -u` folded a stale binary diff into the captured patch -> scorer
# `git apply` failed -> a real solve mis-scored `apply_failed`. The fix ignores
# common artifacts BEFORE `git add -A`, proven for real against a temp git repo.
# --------------------------------------------------------------------------- #
async def test_setup_broken_baseline_ignores_build_artifacts(tmp_path) -> None:
    from swe_forge.forge.calibrate.solver import _setup_broken_baseline

    # Gold repo ships NO .gitignore; the baseline run then generated bytecode and
    # packaging metadata into the working tree (untracked at setup time).
    sandbox = await _make_gold_repo(tmp_path)
    await sandbox.write_file("__pycache__/calc.cpython-312.pyc", "STALE-BYTECODE-1")
    await sandbox.write_file("calc.egg-info/PKG-INFO", "Name: calc\n")

    await _setup_broken_baseline(sandbox, _history_candidate(), timeout=60.0)

    tracked = await sandbox.run_command("git ls-files")
    assert tracked.exit_code == 0
    # Artifacts never enter the orphan baseline commit.
    assert ".pyc" not in tracked.stdout
    assert "__pycache__" not in tracked.stdout
    assert "egg-info" not in tracked.stdout
    # Real source IS committed, alongside the .gitignore that excludes artifacts.
    assert "calc.py" in tracked.stdout
    assert ".gitignore" in tracked.stdout


async def test_capture_patch_excludes_regenerated_pyc(tmp_path) -> None:
    from swe_forge.forge.calibrate.solver import _setup_broken_baseline

    sandbox = await _make_gold_repo(tmp_path)  # no .gitignore
    # Artifact present before setup (as a baseline-run would leave it).
    await sandbox.write_file("__pycache__/calc.cpython-312.pyc", "STALE-BYTECODE-1")
    await _setup_broken_baseline(sandbox, _history_candidate(), timeout=60.0)

    # The model fixes the source (restores gold) AND running the code regenerates
    # a DIFFERENT `.pyc` -- exactly the corruption vector. The captured patch must
    # contain only the source edit, never the binary `.pyc`/`__pycache__` diff.
    await sandbox.write_file("calc.py", GOLD_CALC)
    await sandbox.write_file(
        "__pycache__/calc.cpython-312.pyc", "REGENERATED-BYTECODE-2"
    )

    patch = await capture_workspace_patch(sandbox, timeout=60.0)
    assert patch.strip()
    assert "calc.py" in patch
    assert "return a - b" in patch  # the legitimate source fix is captured
    assert ".pyc" not in patch
    assert "__pycache__" not in patch

    # The clean, source-only patch applies to the broken baseline (a SOLVE, not
    # `apply_failed`): reset to broken, apply, gold restored.
    await sandbox.run_command("git reset --hard -q HEAD")
    assert (tmp_path / "calc.py").read_text() == BROKEN_CALC
    await sandbox.write_file(".captured.patch", patch)
    applied = await sandbox.run_command("git apply --whitespace=nowarn .captured.patch")
    assert applied.exit_code == 0, applied.stderr
    assert (tmp_path / "calc.py").read_text() == GOLD_CALC


async def test_setup_broken_baseline_reinit_appends_artifact_ignores() -> None:
    # Fast, git-free structural guard: the re-init must append the build-artifact
    # ignore patterns to `.gitignore` BEFORE the single `git add -A`/commit.
    from swe_forge.forge.calibrate.solver import _setup_broken_baseline

    sandbox = FakeSandbox()
    await _setup_broken_baseline(sandbox, _candidate(), timeout=30.0)
    commit_cmd = next(str(c["cmd"]) for c in sandbox.calls if "commit" in str(c["cmd"]))
    assert ".gitignore" in commit_cmd
    for pattern in ("__pycache__/", "*.pyc", "*.egg-info/", "node_modules/"):
        assert pattern in commit_cmd
    # The ignore write must precede `git add -A`.
    assert commit_cmd.index(".gitignore") < commit_cmd.index("add -A")
