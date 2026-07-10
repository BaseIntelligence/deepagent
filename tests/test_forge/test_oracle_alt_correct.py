"""Unit tests for the alt-correct false-negative gate (m4-altcorrect).

Offline coverage (no real Docker, no live LLM) of the gate's contract assertions,
driven through a programmable :class:`AltCorrectRunner` fake, plus parser coverage
of the teacher-backed alternative generator through a fake teacher client:

- VAL-ORACLE-012: gold passes F2P+P2P; every genuinely-correct alternative is
  ACCEPTED by the suite; ``alt_correct_accepted == True``.
- VAL-ORACLE-013: a correct alternative the suite FAILS flags over-fit ->
  ``alt_correct_accepted == False`` and ``verdict == reject`` citing over-fit (the
  default), or a recorded relax action that drops the offending test when safe.

The real DockerSandbox + live-teacher paths are exercised by this feature's manual
verification and the user-testing validator in real Docker.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass

import pytest

from swe_forge.forge.adapters import PythonAdapter
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    EnvImage,
    FinalMutationEvidence,
    ModelError,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.alt_correct import (
    REASON_ALT_CORRECT_INVALID_TEACHER_PROPOSAL,
    REASON_ALT_CORRECT_GOLD_NOT_GREEN,
    REASON_ALT_CORRECT_OVERFIT,
    AltCorrectError,
    AltCorrectGenerationContext,
    DockerAltCorrectRunner,
    AltImpl,
    AltImplFile,
    AltScore,
    NullAltCorrectGenerator,
    assess_alt_correct,
    build_alt_correct_report,
    run_alt_correct_gate,
)
from swe_forge.forge.oracle.alt_correct_synth import TeacherAltCorrectGenerator
from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile, TestRun


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeAltCorrectRunner:
    """A programmable :class:`AltCorrectRunner`.

    ``alt_failures`` maps an alternative id to the set of hidden-test ids it
    FAILS (it is accepted once all of those are excluded). ``alt_p2p_failures``
    are alternative ids that fail the P2P/regression suite (not relaxable).
    ``stubborn_alts`` keep failing even when their tests are excluded (drives the
    defensive relax-insufficient path). ``gold_base_fails`` makes the initial
    gold suite run fail (the defensive gold-not-green path).
    """

    def __init__(
        self,
        *,
        alt_failures: dict[str, set[str]] | None = None,
        alt_p2p_failures: set[str] | None = None,
        alt_public_failures: set[str] | None = None,
        stubborn_alts: set[str] | None = None,
        gold_base_fails: bool = False,
        sources: dict[str, str] | None = None,
    ) -> None:
        self.language = "python"
        self._alt_failures = alt_failures or {}
        self._alt_p2p_failures = set(alt_p2p_failures or set())
        self._alt_public_failures = set(alt_public_failures or set())
        self._stubborn_alts = set(stubborn_alts or set())
        self._gold_base_fails = gold_base_fails
        self._sources = sources or {}
        self.gold_calls: list[tuple[str, ...]] = []
        self.alt_calls: list[tuple[str, tuple[str, ...]]] = []
        self.read_sources_calls = 0

    async def score_gold(self, exclude=()) -> AltScore:  # type: ignore[no-untyped-def]
        self.gold_calls.append(tuple(sorted(exclude)))
        if self._gold_base_fails:
            return AltScore(
                f2p_passed=False,
                p2p_passed=True,
                public_suite_passed=True,
                failing_test_ids=("base",),
            )
        return AltScore(f2p_passed=True, p2p_passed=True, public_suite_passed=True)

    async def score_alt(self, alt, exclude=()) -> AltScore:  # type: ignore[no-untyped-def]
        skip = set(exclude)
        self.alt_calls.append((alt.impl_id, tuple(sorted(skip))))
        p2p_failed = alt.impl_id in self._alt_p2p_failures
        if alt.impl_id in self._stubborn_alts:
            return AltScore(
                f2p_passed=False,
                p2p_passed=not p2p_failed,
                public_suite_passed=alt.impl_id not in self._alt_public_failures,
                failing_test_ids=("stubborn",),
            )
        active = self._alt_failures.get(alt.impl_id, set()) - skip
        failing = tuple(sorted(active))
        if p2p_failed:
            failing = (*failing, "p2p")
        return AltScore(
            f2p_passed=not active,
            p2p_passed=not p2p_failed,
            public_suite_passed=alt.impl_id not in self._alt_public_failures,
            failing_test_ids=failing,
        )

    async def read_sources(self) -> dict[str, str]:
        self.read_sources_calls += 1
        return dict(self._sources)


@dataclass
class _CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class _RecordingRecipe:
    """A Docker recipe seam that exposes the concrete command ordering."""

    def __init__(self, *, public_exit: int = 0) -> None:
        self.public_exit = public_exit
        self.p2p_command = "pytest -q -k filtered"
        self.calls: list[str] = []
        self.sandbox = self

    async def set_state(self, _state: object) -> None:
        pass

    async def purge_pycache(self) -> None:
        pass

    async def run_command(self, command: str, **_kwargs: object) -> _CommandResult:
        self.calls.append(command)
        return _CommandResult(self.public_exit)

    async def write_file(self, _path: str, _content: str) -> None:
        pass

    async def run_p2p(self) -> TestRun:
        self.calls.append(self.p2p_command)
        return TestRun(command=self.p2p_command, exit_code=0, passed=True)

    async def write_test(self, _test: HiddenTest) -> None:
        pass

    async def remove_test(self, _test: HiddenTest) -> None:
        pass

    async def run_test(self, test: HiddenTest) -> TestRun:
        self.calls.append(test.test_id)
        return TestRun(command=test.test_id, exit_code=0, passed=True)


class _RecordingDockerAltRunner(DockerAltCorrectRunner):
    def __init__(self, recipe: _RecordingRecipe) -> None:
        super().__init__(
            _candidate(),
            _env_image(),
            PythonAdapter(),
            base_tests=[_test("tests/hidden.py")],
            p2p_command=recipe.p2p_command,
            original_public_command="python -m pytest -q",
        )
        self._recording_recipe = recipe

    @contextlib.asynccontextmanager
    async def _recipe(self):  # type: ignore[override]
        yield self._recording_recipe


class FakeLLMResult:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeTeacher:
    """Minimal stand-in for TeacherClient.complete_text that records its prompt."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0
        self.last_prompt = ""
        self.last_system = ""

    async def complete_text(self, prompt, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_prompt = prompt
        self.last_system = kwargs.get("system", "")
        return FakeLLMResult(self._text)


def _alt(impl_id: str, content: str = "def f():\n    return 1\n") -> AltImpl:
    return AltImpl(
        impl_id=impl_id,
        files=(AltImplFile(path="src/m.py", content=content),),
        description=f"alt {impl_id}",
    )


def _test(path: str) -> HiddenTest:
    return HiddenTest(
        test_id=f"python -m pytest {path}",
        files=(HiddenTestFile(path=path, content=f"# test body {path}"),),
        origin="synthesized",
    )


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
        original_public_test_command="python -m pytest",
        baseline_green=True,
        baseline_exit_code=0,
    )


def _differential_report(
    test_files: list[OracleTestFile],
    *,
    fail_to_pass: list[str] | None = None,
) -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=fail_to_pass or ["python -m pytest tests/test_x.py"],
        pass_to_pass=["python -m pytest"],
        test_files=test_files,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        differential_pass=True,
        provenance=Provenance(generator="ast_mutation", seed=7, language="python"),
        details={"stage": "differential"},
    )


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #
def test_alt_impl_requires_files() -> None:
    with pytest.raises(ModelError):
        AltImpl(impl_id="a1", files=())


def test_alt_impl_requires_id() -> None:
    with pytest.raises(ModelError):
        AltImpl(impl_id="  ", files=(AltImplFile(path="x.py", content="y"),))


def test_alt_score_accepted() -> None:
    assert AltScore(f2p_passed=True, p2p_passed=True, public_suite_passed=True).accepted
    assert not AltScore(
        f2p_passed=False, p2p_passed=True, public_suite_passed=True
    ).accepted
    assert not AltScore(
        f2p_passed=True, p2p_passed=False, public_suite_passed=True
    ).accepted
    assert not AltScore(f2p_passed=True, p2p_passed=True).accepted


# --------------------------------------------------------------------------- #
# VAL-ORACLE-012: a genuinely-correct alternative is ACCEPTED
# --------------------------------------------------------------------------- #
async def test_all_alternatives_accepted_passes() -> None:
    runner = FakeAltCorrectRunner()  # no failures -> every alt accepted
    outcome = await assess_alt_correct(runner, [_alt("alt_1"), _alt("alt_2")])
    assert outcome.is_pass
    assert outcome.alt_correct_accepted is True
    assert outcome.alternatives_total == 2
    assert outcome.alternatives_accepted == 2
    assert outcome.rejected == []
    assert outcome.relaxed is False
    # gold scored once at base
    assert runner.gold_calls == [()]


async def test_no_alternatives_rejects_nonvacuously() -> None:
    runner = FakeAltCorrectRunner()
    outcome = await assess_alt_correct(runner, [])
    assert not outcome.is_pass
    assert outcome.alt_correct_accepted is False
    assert outcome.alternatives_total == 0
    assert runner.alt_calls == []


async def test_gold_not_green_rejects() -> None:
    runner = FakeAltCorrectRunner(gold_base_fails=True)
    outcome = await assess_alt_correct(runner, [_alt("alt_1")])
    assert outcome.verdict == "reject"
    assert outcome.alt_correct_accepted is False
    assert outcome.reasons[0].startswith(REASON_ALT_CORRECT_GOLD_NOT_GREEN)
    # never scored an alternative once gold failed
    assert runner.alt_calls == []


async def test_docker_runner_runs_original_public_suite_before_filtered_and_hidden() -> (
    None
):
    recipe = _RecordingRecipe()
    runner = _RecordingDockerAltRunner(recipe)

    score = await runner.score_alt(_alt("alt_1"))

    assert score.accepted is True
    assert recipe.calls == [
        "python -m pytest -q",
        "pytest -q -k filtered",
        "python -m pytest tests/hidden.py",
    ]
    assert score.hidden_test_exits == (("python -m pytest tests/hidden.py", 0),)


async def test_docker_runner_public_red_never_runs_filtered_or_hidden() -> None:
    recipe = _RecordingRecipe(public_exit=1)
    runner = _RecordingDockerAltRunner(recipe)

    score = await runner.score_alt(_alt("alt_1"))

    assert score.public_valid is False
    assert recipe.calls == ["python -m pytest -q"]
    assert score.hidden_test_exits == ()


# --------------------------------------------------------------------------- #
# VAL-ORACLE-013: an over-fit suite that FAILS a correct alternative -> reject
# --------------------------------------------------------------------------- #
async def test_overfit_alternative_rejects_by_default() -> None:
    runner = FakeAltCorrectRunner(
        alt_failures={"alt_2": {"python -m pytest tests/overfit.py"}}
    )
    outcome = await assess_alt_correct(runner, [_alt("alt_1"), _alt("alt_2")])
    assert not outcome.is_pass
    assert outcome.verdict == "reject"
    assert outcome.alt_correct_accepted is False
    assert outcome.alternatives_accepted == 1
    assert outcome.rejected == ["alt_2"]
    assert outcome.reasons[0].startswith(REASON_ALT_CORRECT_OVERFIT)
    assert "alt_2" in outcome.reasons[0]
    assert outcome.relaxed is False


@pytest.mark.parametrize("hidden_passed", [False, True])
async def test_public_red_alternative_is_invalid_teacher_proposal_not_overfit(
    hidden_passed: bool,
) -> None:
    """A public-red proposal never enters hidden-overfit or relaxation logic."""
    runner = FakeAltCorrectRunner(alt_public_failures={"alt_public_red"})
    if not hidden_passed:
        runner._alt_failures["alt_public_red"] = {"python -m pytest tests/hidden.py"}

    outcome = await assess_alt_correct(
        runner,
        [_alt("alt_public_red")],
        fail_to_pass=["python -m pytest tests/hidden.py"],
        relax=True,
    )

    assert outcome.verdict == "reject"
    assert outcome.alt_correct_accepted is False
    assert outcome.alternatives_accepted == 0
    assert outcome.rejected == ["alt_public_red"]
    assert outcome.reasons == [
        f"{REASON_ALT_CORRECT_INVALID_TEACHER_PROPOSAL}: "
        "alternative(s) ['alt_public_red'] failed the original unfiltered "
        "upstream/public suite"
    ]
    assert "relax" not in outcome.details
    audit = outcome.protected_audit
    assert audit["alternatives"]["alt_public_red"]["public"]["passed"] is False
    assert audit["alternatives"]["alt_public_red"]["hidden"] == []


async def test_public_green_hidden_red_remains_an_overfit_rejection() -> None:
    runner = FakeAltCorrectRunner(
        alt_failures={"alt_public_green": {"python -m pytest tests/overfit.py"}}
    )

    outcome = await assess_alt_correct(
        runner,
        [_alt("alt_public_green")],
        fail_to_pass=["python -m pytest tests/hidden.py"],
    )

    assert outcome.verdict == "reject"
    assert outcome.reasons[0].startswith(REASON_ALT_CORRECT_OVERFIT)
    assert (
        outcome.protected_audit["alternatives"]["alt_public_green"]["public"]["passed"]
        is True
    )


async def test_public_green_hidden_green_passes_with_private_patch_audit() -> None:
    alternative = _alt("alt_public_green", "def f():\n    return 0 + 1\n")
    outcome = await assess_alt_correct(
        FakeAltCorrectRunner(), [alternative], fail_to_pass=["hidden"]
    )

    assert outcome.verdict == "pass"
    protected = outcome.protected_audit["alternatives"]["alt_public_green"]
    assert (
        protected["proposal_sha256"]
        == hashlib.sha256(b"src/m.py\0def f():\n    return 0 + 1\n\0").hexdigest()
    )
    assert protected["patches"] == [
        {"path": "src/m.py", "content": "def f():\n    return 0 + 1\n"}
    ]
    assert protected["public"]["passed"] is True


async def test_zero_public_valid_alternatives_rejects_even_when_hidden_would_pass() -> (
    None
):
    runner = FakeAltCorrectRunner(
        alt_public_failures={"alt_1", "alt_2"},
    )

    outcome = await assess_alt_correct(runner, [_alt("alt_1"), _alt("alt_2")])

    assert outcome.verdict == "reject"
    assert outcome.alternatives_accepted == 0
    assert outcome.details["public_valid_alternatives"] == 0
    assert outcome.reasons[0].startswith(REASON_ALT_CORRECT_INVALID_TEACHER_PROPOSAL)


# --------------------------------------------------------------------------- #
# VAL-ORACLE-013: the recorded-relax branch (--relax)
# --------------------------------------------------------------------------- #
async def test_relax_drops_overfit_test_and_passes() -> None:
    overfit = "python -m pytest tests/overfit.py"
    runner = FakeAltCorrectRunner(alt_failures={"alt_1": {overfit}})
    outcome = await assess_alt_correct(
        runner,
        [_alt("alt_1")],
        fail_to_pass=["python -m pytest tests/test_x.py"],
        relax=True,
    )
    assert outcome.is_pass
    assert outcome.alt_correct_accepted is True
    assert outcome.relaxed is True
    assert outcome.relaxed_test_ids == [overfit]
    relax = outcome.details["relax"]
    assert relax["succeeded"] is True
    assert relax["relaxed_test_ids"] == [overfit]


async def test_relax_refuses_to_remove_last_f2p() -> None:
    f2p = "python -m pytest tests/test_x.py"
    runner = FakeAltCorrectRunner(alt_failures={"alt_1": {f2p}})
    outcome = await assess_alt_correct(
        runner, [_alt("alt_1")], fail_to_pass=[f2p], relax=True
    )
    assert outcome.verdict == "reject"
    assert outcome.alt_correct_accepted is False
    assert outcome.relaxed is False
    assert outcome.details["relax"]["reason"] == "would_remove_last_f2p"
    assert outcome.reasons[0].startswith(REASON_ALT_CORRECT_OVERFIT)


async def test_relax_does_not_apply_to_p2p_failure() -> None:
    runner = FakeAltCorrectRunner(alt_p2p_failures={"alt_1"})
    outcome = await assess_alt_correct(
        runner,
        [_alt("alt_1")],
        fail_to_pass=["python -m pytest tests/test_x.py"],
        relax=True,
    )
    assert outcome.verdict == "reject"
    assert outcome.alt_correct_accepted is False
    assert outcome.details["relax"]["reason"] == "p2p_failure_not_relaxable"
    assert outcome.reasons[0].startswith(REASON_ALT_CORRECT_OVERFIT)


async def test_relax_insufficient_rejects() -> None:
    runner = FakeAltCorrectRunner(stubborn_alts={"alt_1"})
    outcome = await assess_alt_correct(
        runner,
        [_alt("alt_1")],
        fail_to_pass=["python -m pytest tests/test_x.py"],
        relax=True,
    )
    assert outcome.verdict == "reject"
    assert outcome.alt_correct_accepted is False
    assert outcome.details["relax"]["reason"] == "relax_insufficient"
    assert "alt_1" in outcome.reasons[0]


# --------------------------------------------------------------------------- #
# build_alt_correct_report
# --------------------------------------------------------------------------- #
async def test_build_report_pass_sets_flag_and_carries_fields() -> None:
    runner = FakeAltCorrectRunner()
    outcome = await assess_alt_correct(runner, [_alt("alt_1"), _alt("alt_2")])
    prior = _differential_report(
        [OracleTestFile(path="tests/test_x.py", content="X", origin="synthesized")]
    )
    report = build_alt_correct_report(
        _candidate(), prior, outcome, env_image=_env_image()
    )
    assert report.verdict == "pass"
    assert report.alt_correct_accepted is True
    # prior gate fields carried forward
    assert report.differential_pass is True
    assert report.flakiness_runs == 3
    assert report.mutants_total == 10
    assert report.mutants_killed == 10
    assert report.fail_to_pass == ["python -m pytest tests/test_x.py"]
    assert [tf.path for tf in report.test_files] == ["tests/test_x.py"]
    assert report.details["alt_correct"]["alternatives_total"] == 2
    # serializable + reproducible
    again = OracleReport.from_dict(report.to_dict())
    assert again.alt_correct_accepted is True
    assert again.verdict == "pass"


async def test_public_alt_audit_round_trips_only_through_protected_serialization() -> (
    None
):
    alternative = _alt("alt_1", "def f():\n    return 0 + 1\n")
    outcome = await assess_alt_correct(FakeAltCorrectRunner(), [alternative])
    report = build_alt_correct_report(
        _candidate(),
        _differential_report(
            [OracleTestFile(path="tests/test_x.py", content="X", origin="synthesized")]
        ),
        outcome,
        env_image=_env_image(),
    )

    public = report.to_dict()
    assert "0 + 1" not in repr(public)
    assert "patches" not in repr(public)
    assert public["details"]["alt_correct"]["public_valid_alternatives"] == 1
    assert "public_suite_sha256" in public["details"]["alt_correct"]

    protected = report.to_protected_dict()
    audit = protected["protected_alt_correct_audit"]
    assert audit["alternatives"]["alt_1"]["patches"][0]["content"].endswith("0 + 1\n")
    restored = OracleReport.from_protected_dict(protected)
    assert restored.protected_alt_correct_audit == report.protected_alt_correct_audit


async def test_build_report_reject_carries_reason() -> None:
    runner = FakeAltCorrectRunner(
        alt_failures={"alt_1": {"python -m pytest tests/o.py"}}
    )
    outcome = await assess_alt_correct(runner, [_alt("alt_1")])
    prior = _differential_report([OracleTestFile(path="tests/test_x.py", content="X")])
    report = build_alt_correct_report(_candidate(), prior, outcome)
    assert report.verdict == "reject"
    assert report.alt_correct_accepted is False
    assert any(REASON_ALT_CORRECT_OVERFIT in r for r in report.reasons)
    # reject invariant survives (de)serialization
    assert OracleReport.from_dict(report.to_dict()).verdict == "reject"


async def test_build_report_relax_removes_dropped_test_files() -> None:
    overfit = "python -m pytest tests/overfit.py"
    f2p = "python -m pytest tests/test_x.py"
    runner = FakeAltCorrectRunner(alt_failures={"alt_1": {overfit}})
    outcome = await assess_alt_correct(
        runner, [_alt("alt_1")], fail_to_pass=[f2p, overfit], relax=True
    )
    assert outcome.relaxed is True
    prior = _differential_report(
        [
            OracleTestFile(path="tests/test_x.py", content="X"),
            OracleTestFile(path="tests/overfit.py", content="O"),
        ],
        fail_to_pass=[f2p, overfit],
    )
    base_tests = [_test("tests/test_x.py"), _test("tests/overfit.py")]
    report = build_alt_correct_report(
        _candidate(), prior, outcome, base_tests=base_tests
    )
    assert report.verdict == "pass"
    assert report.alt_correct_accepted is True
    # the over-fit F2P id and its test file are removed; the surviving F2P remains
    assert report.fail_to_pass == [f2p]
    assert [tf.path for tf in report.test_files] == ["tests/test_x.py"]


async def test_relaxation_invalidates_prior_final_mutation_evidence() -> None:
    overfit = "python -m pytest tests/overfit.py"
    f2p = "python -m pytest tests/test_x.py"
    runner = FakeAltCorrectRunner(alt_failures={"alt_1": {overfit}})
    outcome = await assess_alt_correct(
        runner, [_alt("alt_1")], fail_to_pass=[f2p, overfit], relax=True
    )
    prior = _differential_report(
        [
            OracleTestFile(path="tests/test_x.py", content="X"),
            OracleTestFile(path="tests/overfit.py", content="O"),
        ],
        fail_to_pass=[f2p, overfit],
    )
    prior.final_mutation_evidence = FinalMutationEvidence(
        suite_fingerprint="b" * 64,
        mutants_total=10,
        mutants_killed=10,
        threshold=0.8,
        tool="fake-tool",
    )

    report = build_alt_correct_report(
        _candidate(),
        prior,
        outcome,
        base_tests=[_test("tests/test_x.py"), _test("tests/overfit.py")],
    )

    assert report.final_mutation_evidence is None
    assert report.details["mutation_evidence_invalidated"] == {
        "stage": "alt_correct",
        "reason": "hidden_suite_changed",
    }


# --------------------------------------------------------------------------- #
# Null implementation + run-gate guard
# --------------------------------------------------------------------------- #
async def test_null_alt_correct_generator_proposes_nothing() -> None:
    ctx = AltCorrectGenerationContext(
        candidate=_candidate(), adapter=PythonAdapter(), gold_sources={}
    )
    assert await NullAltCorrectGenerator()(ctx) == []


async def test_run_gate_requires_passing_prior_report() -> None:
    reject_prior = OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="reject",
        reasons=["differential_indistinguishable_variant: ..."],
        fail_to_pass=[],
        pass_to_pass=["python -m pytest"],
    )
    with pytest.raises(AltCorrectError):
        await run_alt_correct_gate(
            _candidate(), _env_image(), reject_prior, adapter=PythonAdapter()
        )


async def test_run_gate_fails_closed_without_original_public_command() -> None:
    report = await run_alt_correct_gate(
        _candidate(),
        EnvImage(
            **{
                **_env_image().to_dict(),
                "original_public_test_command": "",
            }
        ),
        _differential_report([]),
        adapter=PythonAdapter(),
    )

    assert report.verdict == "reject"
    assert report.alt_correct_accepted is False
    assert report.reasons[0].startswith("alt_correct_public_suite_unavailable")


# --------------------------------------------------------------------------- #
# Teacher-backed generator parsing (fake client; no live LLM)
# --------------------------------------------------------------------------- #
async def test_teacher_alt_generator_parses_blocks() -> None:
    text = (
        "Here are alternatives:\n"
        "```python\ndef f():\n    result = 1\n    return result\n```\n"
        "```python\ndef f():\n    return 1 + 0\n```\n"
    )
    gen = TeacherAltCorrectGenerator(client=FakeTeacher(text))  # type: ignore[arg-type]
    ctx = AltCorrectGenerationContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        gold_sources={"src/m.py": "def f():\n    return 1\n"},
        interface_block="def f() -> int",
        num_alternatives=2,
    )
    alts = await gen(ctx)
    assert [a.impl_id for a in alts] == ["alt_1", "alt_2"]
    assert all(a.files[0].path == "src/m.py" for a in alts)
    assert alts[0].files[0].content.startswith("def f():")


async def test_teacher_alt_generator_skips_block_equal_to_gold() -> None:
    gold = "def f():\n    return 1\n"
    text = f"```python\n{gold}```\n```python\ndef f():\n    return 0 + 1\n```\n"
    teacher = FakeTeacher(text)
    gen = TeacherAltCorrectGenerator(client=teacher)  # type: ignore[arg-type]
    ctx = AltCorrectGenerationContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        gold_sources={"src/m.py": gold},
        interface_block="def f() -> int",
        num_alternatives=2,
    )
    alts = await gen(ctx)
    # the block identical to gold is dropped; only the differently-written one remains
    assert len(alts) == 1
    assert "0 + 1" in alts[0].files[0].content
    # the published Interface is pinned into the teacher prompt
    assert "def f() -> int" in teacher.last_prompt


async def test_teacher_alt_generator_replaces_only_target_region() -> None:
    gold = (
        "# leading context\n"
        "class Example:\n"
        "    def f(self):\n"
        "        return 1\n"
        "\n"
        "# trailing context\n"
    )
    candidate = Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("src/m.py",), symbols=("Example.f",)),
        mutation_patch="forward",
        oracle_patch="inverse",
        difficulty_hint="medium",
        provenance=Provenance(
            generator="ast_mutation",
            seed=7,
            language="python",
            details={
                "constituents": [{"file": "src/m.py", "start_line": 3, "end_line": 4}]
            },
        ),
    )
    teacher = FakeTeacher(
        "```python\n    def f(self):\n        value = 1\n        return value\n```"
    )
    generator = TeacherAltCorrectGenerator(client=teacher)  # type: ignore[arg-type]

    alternatives = await generator(
        AltCorrectGenerationContext(
            candidate=candidate,
            adapter=PythonAdapter(),
            gold_sources={"src/m.py": gold},
            interface_block="def f(self) -> int",
        )
    )

    assert len(alternatives) == 1
    assert alternatives[0].files[0].content == gold.replace(
        "return 1", "value = 1\n        return value"
    )


async def test_teacher_alt_generator_no_sources_returns_empty() -> None:
    gen = TeacherAltCorrectGenerator(client=FakeTeacher("```\nx\n```"))  # type: ignore[arg-type]
    ctx = AltCorrectGenerationContext(
        candidate=_candidate(), adapter=PythonAdapter(), gold_sources={}
    )
    assert await gen(ctx) == []
