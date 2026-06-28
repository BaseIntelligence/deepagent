"""Unit tests for the establish gate (m4-establish).

Offline coverage (no real Docker, no live LLM) of the establish gate's contract
assertions, driven through fakes:

- VAL-ORACLE-001: broken FAILS F2P, gold makes every F2P PASS, the repo suite is
  green as P2P on both trees, and the ids transition fail->pass.
- VAL-ORACLE-002: when the fault is uncovered, the agentic synthesizer's proposals
  that fail-on-broken/pass-on-gold are recorded in ``test_files[]`` (origin
  ``synthesized``) with their ids in ``fail_to_pass[]``; a non-discriminating
  proposal yields ``no_f2p_established``.
- VAL-ORACLE-003: the three induced reject cases (no F2P transition / gold fails
  to fix an intended F2P / P2P not green on broken) each yield ``verdict==reject``
  with the attributable reason.
- The reusable Docker recipe bakes in the Python re-test determinism invariant
  (``PYTHONDONTWRITEBYTECODE=1`` + ``__pycache__``/``.pyc`` purge before each run)
  and toggles trees via the mutation (gold->broken) / oracle (broken->gold) patches.
- OracleReport (de)serialization + invariants.

The DockerSandbox + live teacher paths are exercised by the worker/user-testing
validator in real Docker (see this feature's manual verification).
"""

from __future__ import annotations

import pytest

from swe_forge.forge.adapters import PythonAdapter
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    EnvImage,
    ModelError,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.establish import (
    REASON_GOLD_NOT_FIX,
    REASON_GOLD_P2P_NOT_GREEN,
    REASON_NO_F2P,
    REASON_P2P_NOT_GREEN_ON_BROKEN,
    DockerOracleRecipe,
    HiddenTest,
    HiddenTestFile,
    SynthesisContext,
    TestRun,
    TreeState,
    build_establish_report,
    establish_oracle,
)
from swe_forge.forge.oracle.test_synth import AgenticTestSynthesizer
from swe_forge.forge.teacher import AgenticResult, NormalizedToolCall, Usage


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeExec:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class FakeSandbox:
    """Records every command (with env) and simulates a tiny filesystem."""

    def __init__(self, responses: dict[str, int] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.files: dict[str, str] = {}
        self._responses = responses or {}

    async def run_command(
        self,
        cmd: str,
        *,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> FakeExec:
        self.calls.append({"cmd": cmd, "env": env, "timeout": timeout})
        return FakeExec(exit_code=self._responses.get(cmd, 0))

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def read_file(self, path: str) -> str:
        return self.files.get(path, "")


class FakeRecipe:
    """A programmable :class:`RecipeProtocol` for orchestration tests."""

    def __init__(
        self,
        *,
        language: str = "python",
        p2p_command: str = "python -m pytest",
        p2p_gold: bool = True,
        p2p_broken: bool = True,
        outcomes: dict[str, tuple[bool, bool]] | None = None,
    ) -> None:
        self.language = language
        self.p2p_command = p2p_command
        self._p2p_gold = p2p_gold
        self._p2p_broken = p2p_broken
        # outcomes: test_id -> (fails_on_broken, passes_on_gold)
        self._outcomes = outcomes or {}
        self.state = TreeState.GOLD
        self.transitions: list[TreeState] = []
        self.written: list[str] = []
        self.removed: list[str] = []

    async def set_state(self, state: TreeState) -> None:
        if state != self.state:
            self.transitions.append(state)
            self.state = state

    async def run_p2p(self) -> TestRun:
        passed = self._p2p_gold if self.state == TreeState.GOLD else self._p2p_broken
        return TestRun(
            command=self.p2p_command, exit_code=0 if passed else 1, passed=passed
        )

    async def write_test(self, test: HiddenTest) -> None:
        self.written.append(test.test_id)

    async def remove_test(self, test: HiddenTest) -> None:
        self.removed.append(test.test_id)

    async def run_test(self, test: HiddenTest) -> TestRun:
        fails_on_broken, passes_on_gold = self._outcomes.get(test.test_id, (True, True))
        if self.state == TreeState.BROKEN:
            passed = not fails_on_broken
        else:
            passed = passes_on_gold
        return TestRun(
            command=test.test_id, exit_code=0 if passed else 1, passed=passed
        )


def _make_synth(tests: list[HiddenTest]):
    async def synth(ctx: SynthesisContext) -> list[HiddenTest]:
        return list(tests)

    return synth


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _candidate() -> Candidate:
    return Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("src/m.py",), symbols=("f",)),
        mutation_patch="--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-return 1\n+return 2\n",
        oracle_patch="--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-return 2\n+return 1\n",
        difficulty_hint="medium",
        provenance=Provenance(generator="ast_mutation", seed=7, language="python"),
    )


def _env_image() -> EnvImage:
    return EnvImage(
        repo_id="demo",
        language="python",
        image_tag="swe-forge-env-demo:abc123",
        base_image="python:3.12-slim",
        commit="0" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command="python -m pytest",
        baseline_green=True,
        baseline_exit_code=0,
    )


def _ctx(sandbox: FakeSandbox | None = None) -> SynthesisContext:
    sandbox = sandbox or FakeSandbox()
    cand = _candidate()
    recipe = DockerOracleRecipe(
        sandbox,
        language="python",
        workspace_dir="/workspace/repo",
        mutation_patch=cand.mutation_patch,
        oracle_patch=cand.oracle_patch,
        p2p_command="python -m pytest",
    )
    return SynthesisContext(
        candidate=cand,
        env_image=_env_image(),
        adapter=PythonAdapter(),
        recipe=recipe,
    )


# --------------------------------------------------------------------------- #
# OracleReport model
# --------------------------------------------------------------------------- #
def test_oracle_report_roundtrips() -> None:
    report = OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=["python -m pytest tests/test_x.py"],
        pass_to_pass=["python -m pytest"],
        test_files=[
            OracleTestFile(path="tests/test_x.py", content="...", origin="synthesized")
        ],
    )
    again = OracleReport.from_dict(report.to_dict())
    assert again.verdict == "pass"
    assert again.fail_to_pass == report.fail_to_pass
    assert again.test_files[0].path == "tests/test_x.py"
    assert again.test_files[0].origin == "synthesized"


def test_oracle_report_reject_requires_reason() -> None:
    with pytest.raises(ModelError):
        OracleReport(language="python", generator="ast_mutation", verdict="reject")


def test_oracle_report_pass_forbids_reasons() -> None:
    with pytest.raises(ModelError):
        OracleReport(
            language="python",
            generator="ast_mutation",
            verdict="pass",
            reasons=["should not be here"],
        )


def test_oracle_report_rejects_bad_mutant_counts() -> None:
    with pytest.raises(ModelError):
        OracleReport(
            language="python",
            generator="ast_mutation",
            verdict="pass",
            mutants_total=3,
            mutants_killed=5,
        )


# --------------------------------------------------------------------------- #
# Reusable recipe: determinism invariant + tree toggling
# --------------------------------------------------------------------------- #
async def test_recipe_sets_broken_via_mutation_then_gold_via_oracle() -> None:
    sandbox = FakeSandbox()
    cand = _candidate()
    recipe = DockerOracleRecipe(
        sandbox,
        language="python",
        workspace_dir="/workspace/repo",
        mutation_patch=cand.mutation_patch,
        oracle_patch=cand.oracle_patch,
        p2p_command="python -m pytest",
    )
    await recipe.set_state(TreeState.BROKEN)
    await recipe.set_state(TreeState.GOLD)

    applies = [c["cmd"] for c in sandbox.calls if "git apply" in str(c["cmd"])]
    assert any("mutation.patch" in a for a in applies)
    assert any("oracle.patch" in a for a in applies)
    # mutation (gold->broken) precedes oracle (broken->gold)
    assert next(i for i, a in enumerate(applies) if "mutation.patch" in a) < next(
        i for i, a in enumerate(applies) if "oracle.patch" in a
    )


async def test_recipe_python_determinism_invariant() -> None:
    sandbox = FakeSandbox()
    recipe = DockerOracleRecipe(
        sandbox,
        language="python",
        workspace_dir="/workspace/repo",
        mutation_patch="m",
        oracle_patch="o",
        p2p_command="python -m pytest",
    )
    test = HiddenTest(test_id="python -m pytest tests/test_x.py")
    await recipe.run_test(test)

    cmds = [str(c["cmd"]) for c in sandbox.calls]
    purge_idx = next(
        i for i, c in enumerate(cmds) if "__pycache__" in c and "*.pyc" in c
    )
    run_idx = next(i for i, c in enumerate(cmds) if c == test.test_id)
    assert purge_idx < run_idx  # pyc purge BEFORE the re-run
    run_call = next(c for c in sandbox.calls if c["cmd"] == test.test_id)
    assert run_call["env"] == {"PYTHONDONTWRITEBYTECODE": "1"}


async def test_recipe_no_pycache_purge_for_non_python() -> None:
    sandbox = FakeSandbox()
    recipe = DockerOracleRecipe(
        sandbox,
        language="go",
        workspace_dir="/workspace/repo",
        mutation_patch="m",
        oracle_patch="o",
        p2p_command="go test ./...",
    )
    await recipe.run_test(HiddenTest(test_id="go test ./pkg"))
    cmds = [str(c["cmd"]) for c in sandbox.calls]
    assert not any("__pycache__" in c for c in cmds)
    run_call = next(c for c in sandbox.calls if c["cmd"] == "go test ./pkg")
    assert run_call["env"] is None


async def test_recipe_apply_failure_raises() -> None:
    from swe_forge.forge.oracle.establish import EstablishError

    sandbox = FakeSandbox(
        responses={
            "git apply --whitespace=nowarn .swe_forge_oracle/mutation.patch": 1,
            "git apply --3way --whitespace=nowarn .swe_forge_oracle/mutation.patch": 1,
        }
    )
    recipe = DockerOracleRecipe(
        sandbox,
        language="python",
        workspace_dir="/workspace/repo",
        mutation_patch="m",
        oracle_patch="o",
        p2p_command="python -m pytest",
    )
    with pytest.raises(EstablishError):
        await recipe.set_state(TreeState.BROKEN)


# --------------------------------------------------------------------------- #
# establish_oracle: happy path (VAL-ORACLE-001)
# --------------------------------------------------------------------------- #
async def test_establish_happy_path_provided_test() -> None:
    f2p = "python -m pytest tests/test_x.py"
    recipe = FakeRecipe(outcomes={f2p: (True, True)})
    provided = [
        HiddenTest(
            test_id=f2p,
            files=(HiddenTestFile(path="tests/test_x.py", content="..."),),
            origin="provided",
        )
    ]
    outcome = await establish_oracle(recipe, provided_tests=provided)

    assert outcome.verdict == "pass"
    assert outcome.reasons == []
    assert outcome.fail_to_pass == [f2p]
    assert outcome.pass_to_pass == ["python -m pytest"]
    assert [tf.path for tf in outcome.test_files] == ["tests/test_x.py"]
    # ids transitioned fail (broken) -> pass (gold)
    assert outcome.details["provided_confirmations"][0]["fails_on_broken"] is True
    assert outcome.details["provided_confirmations"][0]["passes_on_gold"] is True


# --------------------------------------------------------------------------- #
# establish_oracle: synthesis (VAL-ORACLE-002)
# --------------------------------------------------------------------------- #
async def test_establish_synthesizes_discriminating_test() -> None:
    f2p = "python -m pytest tests/test_syn.py"
    recipe = FakeRecipe(outcomes={f2p: (True, True)})
    synth_test = HiddenTest(
        test_id=f2p,
        files=(HiddenTestFile(path="tests/test_syn.py", content="def test(): ..."),),
        origin="synthesized",
    )
    outcome = await establish_oracle(
        recipe,
        synthesizer=_make_synth([synth_test]),
        synthesis_context=_ctx(),
    )
    assert outcome.verdict == "pass"
    assert outcome.fail_to_pass == [f2p]
    assert outcome.test_files[0].origin == "synthesized"
    assert outcome.details["synthesized_proposed"] == 1


async def test_establish_synthesis_nondiscriminating_rejects_no_f2p() -> None:
    # Proposal passes on broken too (does not fail) -> not a transition.
    f2p = "python -m pytest tests/test_bad.py"
    recipe = FakeRecipe(outcomes={f2p: (False, True)})
    synth_test = HiddenTest(
        test_id=f2p,
        files=(HiddenTestFile(path="tests/test_bad.py", content="x"),),
        origin="synthesized",
    )
    outcome = await establish_oracle(
        recipe,
        synthesizer=_make_synth([synth_test]),
        synthesis_context=_ctx(),
    )
    assert outcome.verdict == "reject"
    assert outcome.fail_to_pass == []
    assert any(r.startswith(REASON_NO_F2P) for r in outcome.reasons)


# --------------------------------------------------------------------------- #
# establish_oracle: reject cases (VAL-ORACLE-003)
# --------------------------------------------------------------------------- #
async def test_establish_reject_no_f2p_when_none_provided() -> None:
    recipe = FakeRecipe()
    outcome = await establish_oracle(recipe)  # no provided, no synthesizer
    assert outcome.verdict == "reject"
    assert outcome.fail_to_pass == []
    assert any(r.startswith(REASON_NO_F2P) for r in outcome.reasons)


async def test_establish_reject_gold_not_fix() -> None:
    f2p = "python -m pytest tests/test_x.py"
    # fails on broken (good) but gold does not fix it (passes_on_gold=False)
    recipe = FakeRecipe(outcomes={f2p: (True, False)})
    provided = [
        HiddenTest(
            test_id=f2p,
            files=(HiddenTestFile(path="tests/test_x.py", content="x"),),
            origin="provided",
        )
    ]
    outcome = await establish_oracle(recipe, provided_tests=provided)
    assert outcome.verdict == "reject"
    assert any(r.startswith(REASON_GOLD_NOT_FIX) for r in outcome.reasons)


async def test_establish_reject_p2p_not_green_on_broken() -> None:
    recipe = FakeRecipe(p2p_gold=True, p2p_broken=False)
    f2p = "python -m pytest tests/test_x.py"
    provided = [HiddenTest(test_id=f2p, origin="provided")]
    outcome = await establish_oracle(recipe, provided_tests=provided)
    assert outcome.verdict == "reject"
    assert any(r.startswith(REASON_P2P_NOT_GREEN_ON_BROKEN) for r in outcome.reasons)
    # gate ordering: early P2P failure short-circuits, no F2P work recorded
    assert outcome.fail_to_pass == []
    assert "provided_confirmations" not in outcome.details


async def test_establish_reject_gold_p2p_not_green() -> None:
    recipe = FakeRecipe(p2p_gold=False, p2p_broken=True)
    outcome = await establish_oracle(recipe)
    assert outcome.verdict == "reject"
    assert any(r.startswith(REASON_GOLD_P2P_NOT_GREEN) for r in outcome.reasons)


# --------------------------------------------------------------------------- #
# build_establish_report
# --------------------------------------------------------------------------- #
async def test_build_establish_report_pass() -> None:
    f2p = "python -m pytest tests/test_x.py"
    recipe = FakeRecipe(outcomes={f2p: (True, True)})
    provided = [
        HiddenTest(
            test_id=f2p,
            files=(HiddenTestFile(path="tests/test_x.py", content="x"),),
            origin="provided",
        )
    ]
    outcome = await establish_oracle(recipe, provided_tests=provided)
    report = build_establish_report(_candidate(), outcome, env_image=_env_image())

    assert isinstance(report, OracleReport)
    assert report.verdict == "pass"
    assert report.fail_to_pass == [f2p]
    assert report.pass_to_pass == ["python -m pytest"]
    assert report.provenance is not None
    assert report.details["stage"] == "establish"
    assert report.details["env_image"] == "swe-forge-env-demo:abc123"
    # serializable
    assert OracleReport.from_dict(report.to_dict()).verdict == "pass"


async def test_build_establish_report_reject_has_reasons() -> None:
    recipe = FakeRecipe()
    outcome = await establish_oracle(recipe)
    report = build_establish_report(_candidate(), outcome, env_image=_env_image())
    assert report.verdict == "reject"
    assert report.reasons  # attributable, non-empty


# --------------------------------------------------------------------------- #
# AgenticTestSynthesizer (offline, fake teacher)
# --------------------------------------------------------------------------- #
class FakeTeacher:
    def __init__(self, script: list[tuple[str, dict[str, object]]]) -> None:
        self._script = script

    async def agentic_turn(
        self, messages, tools, tool_executor, *, max_turns, max_tokens
    ) -> AgenticResult:
        for idx, (name, args) in enumerate(self._script):
            call = NormalizedToolCall(id=str(idx), name=name, arguments=args)
            await tool_executor(call)
        return AgenticResult(text="done", turns=1, usage=Usage(), cost=0.0)


async def test_agentic_synthesizer_collects_submitted_test() -> None:
    sandbox = FakeSandbox()
    ctx = _ctx(sandbox)
    content = "import m\n\ndef test_f():\n    assert m.f() == 1\n"
    script: list[tuple[str, dict[str, object]]] = [
        ("write_file", {"path": "tests/test_f.py", "content": content}),
        ("submit_test", {"path": "tests/test_f.py"}),
    ]
    synth = AgenticTestSynthesizer(client=FakeTeacher(script))  # type: ignore[arg-type]
    proposals = await synth(ctx)

    assert len(proposals) == 1
    assert proposals[0].test_id == "python -m pytest tests/test_f.py"
    assert proposals[0].origin == "synthesized"
    assert proposals[0].files[0].content == content
    assert sandbox.files["tests/test_f.py"] == content


async def test_agentic_synthesizer_rejects_string_matching() -> None:
    sandbox = FakeSandbox()
    ctx = _ctx(sandbox)
    content = "def test_src():\n    assert 'return 1' in open('src/m.py').read()\n"
    script: list[tuple[str, dict[str, object]]] = [
        ("write_file", {"path": "tests/test_src.py", "content": content}),
        ("submit_test", {"path": "tests/test_src.py"}),
    ]
    synth = AgenticTestSynthesizer(client=FakeTeacher(script))  # type: ignore[arg-type]
    proposals = await synth(ctx)
    assert proposals == []


async def test_agentic_synthesizer_submit_requires_prior_write() -> None:
    sandbox = FakeSandbox()
    ctx = _ctx(sandbox)
    script: list[tuple[str, dict[str, object]]] = [
        ("submit_test", {"path": "tests/never_written.py"}),
    ]
    synth = AgenticTestSynthesizer(client=FakeTeacher(script))  # type: ignore[arg-type]
    proposals = await synth(ctx)
    assert proposals == []
