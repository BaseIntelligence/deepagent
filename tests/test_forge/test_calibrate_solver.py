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
    import asyncio

    from swe_forge.forge.models import BaselineNotGreenError

    red = _env_image()
    red.baseline_green = False
    red.baseline_exit_code = 1
    # run_solver_rollout enforces require_green_baseline before any Docker work.
    with pytest.raises(BaselineNotGreenError):
        asyncio.run(run_solver_rollout(_candidate(), red, _spec(), _oracle_report()))
