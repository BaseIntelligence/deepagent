"""Unit tests for the mutation-adequacy gate (m4-mutation).

Offline coverage (no real Docker, no live LLM) of the gate's contract assertions,
driven through a programmable :class:`MutationRunner` fake + a scripted synthesizer,
plus pure-parser coverage of each language tool's output:

- VAL-ORACLE-006: a language-correct tool runs; ``mutants_total > 0`` and
  ``mutants_killed`` are recorded consistent with the tool output; a suite killing
  >= threshold passes with no synthesis.
- VAL-ORACLE-007: an under-determined suite (kill ratio < threshold) triggers
  auto-synthesis; only proposals that reduce the survivor count are kept (each new
  test killed a previously-surviving mutant); survivors decrease across rounds and
  the final ratio reaches the threshold; new test ids are appended to ``test_files``.
- VAL-ORACLE-008: when the bounded loop cannot reach the threshold, the gate
  rejects with a reason citing the surviving mutants.

The real DockerSandbox + live-teacher paths are exercised by this feature's manual
verification and the user-testing validator in real Docker.
"""

from __future__ import annotations

import pytest

from swe_forge.forge.adapters import PythonAdapter
from swe_forge.forge.adapters._mutation_tools import (
    MutationToolError,
    ToolCounts,
    parse_cosmicray_report,
    parse_gomutesting,
    parse_stryker_json,
)
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    EnvImage,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile
from swe_forge.forge.oracle.mutation import (
    REASON_MUTATION_INADEQUATE,
    REASON_NO_MUTANTS,
    MutationError,
    MutationMeasurement,
    MutationSynthesisContext,
    NullMutationSynthesizer,
    assess_mutation,
    build_mutation_report,
    reconstruct_base_tests,
    run_mutation_gate,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeMutationRunner:
    """A programmable :class:`MutationRunner`.

    ``base_survivors`` is the set of surviving mutant ids when only the
    established suite runs; ``kill_map`` maps an extra test's path to the set of
    survivor ids that test kills. ``total`` is the (fixed) mutant population.
    """

    def __init__(
        self,
        *,
        total: int,
        base_survivors: set[str],
        kill_map: dict[str, set[str]] | None = None,
        sources: dict[str, str] | None = None,
        tool: str = "fake-tool",
    ) -> None:
        self.language = "python"
        self._total = total
        self._base = set(base_survivors)
        self._kill_map = kill_map or {}
        self._sources = sources or {}
        self._tool = tool
        self.measure_calls: list[tuple[str, ...]] = []
        self.read_sources_calls = 0

    async def measure(self, extra_tests):  # type: ignore[no-untyped-def]
        paths = tuple(f.path for t in extra_tests for f in t.files)
        self.measure_calls.append(paths)
        killed_ids: set[str] = set()
        for path in paths:
            killed_ids |= self._kill_map.get(path, set())
        survivors = self._base - killed_ids
        killed = self._total - len(survivors)
        return MutationMeasurement(
            total=self._total,
            killed=killed,
            tool=self._tool,
            survivors=tuple(sorted(survivors)),
        )

    async def read_sources(self) -> dict[str, str]:
        self.read_sources_calls += 1
        return dict(self._sources)


class ScriptedSynth:
    """Returns a pre-scripted list of proposals per round (1-indexed)."""

    def __init__(self, rounds: list[list[HiddenTest]]) -> None:
        self._rounds = rounds
        self.contexts: list[MutationSynthesisContext] = []

    async def __call__(self, ctx: MutationSynthesisContext) -> list[HiddenTest]:
        self.contexts.append(ctx)
        idx = ctx.round_index - 1
        return self._rounds[idx] if 0 <= idx < len(self._rounds) else []


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


def _template() -> MutationSynthesisContext:
    return MutationSynthesisContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        sources={},
        survivors=(),
        round_index=0,
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


def _flakiness_report(test_files: list[OracleTestFile]) -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=["python -m pytest tests/test_x.py"],
        pass_to_pass=["python -m pytest"],
        test_files=test_files,
        flakiness_runs=3,
        provenance=Provenance(generator="ast_mutation", seed=7, language="python"),
        details={"stage": "flakiness"},
    )


# --------------------------------------------------------------------------- #
# Pure parsers (VAL-ORACLE-006: counts consistent with tool output)
# --------------------------------------------------------------------------- #
_COSMIC_RAY_REPORT = """\
mod.py core/ReplaceComparisonOperator_Gt_NotEq 0
worker outcome: WorkerOutcome.NORMAL, test outcome: TestOutcome.SURVIVED
[job-id] 8a3a4275abf94590b05813dee29cf1fd
mod.py core/ReplaceComparisonOperator_Gt_Lt 0
worker outcome: WorkerOutcome.NORMAL, test outcome: TestOutcome.KILLED
[job-id] 7b1bd30cde9d413899c4752645b856f2
mod.py core/ReplaceComparisonOperator_Gt_GtE 0
worker outcome: WorkerOutcome.NORMAL, test outcome: TestOutcome.SURVIVED
[job-id] aeb55f1211ee48f39033b394dfea4321
mod.py core/NumberReplacer 1
worker outcome: WorkerOutcome.NORMAL, test outcome: TestOutcome.SURVIVED
total jobs: 18
complete: 18 (100.00%)
surviving mutants: 3 (16.67%)
"""


def test_parse_cosmicray_report_counts_and_survivors() -> None:
    counts = parse_cosmicray_report(_COSMIC_RAY_REPORT)
    assert counts.total == 18
    assert counts.killed == 15
    assert counts.survived == 3
    assert round(counts.kill_ratio, 4) == round(15 / 18, 4)
    assert len(counts.survivors) == 3
    assert any("ReplaceComparisonOperator_Gt_NotEq" in s for s in counts.survivors)


def test_parse_cosmicray_report_missing_total_raises() -> None:
    with pytest.raises(MutationToolError):
        parse_cosmicray_report("some unrelated output\n")


def test_parse_stryker_json_counts() -> None:
    report = """
    {
      "schemaVersion": "1.0",
      "files": {
        "src/calc.js": {
          "language": "javascript",
          "mutants": [
            {"id": "1", "mutatorName": "ArithmeticOperator", "status": "Killed",
             "location": {"start": {"line": 2}}},
            {"id": "2", "mutatorName": "ArithmeticOperator", "status": "Survived",
             "location": {"start": {"line": 3}}},
            {"id": "3", "mutatorName": "EqualityOperator", "status": "Timeout",
             "location": {"start": {"line": 4}}},
            {"id": "4", "mutatorName": "BlockStatement", "status": "NoCoverage",
             "location": {"start": {"line": 5}}},
            {"id": "5", "mutatorName": "X", "status": "CompileError",
             "location": {"start": {"line": 6}}}
          ]
        }
      }
    }
    """
    counts = parse_stryker_json(report)
    # Killed + Timeout = killed; Survived + NoCoverage = survived; CompileError excluded.
    assert counts.total == 4
    assert counts.killed == 2
    assert counts.survived == 2
    assert len(counts.survivors) == 2


def test_parse_stryker_json_invalid_raises() -> None:
    with pytest.raises(MutationToolError):
        parse_stryker_json("not json")


def test_parse_gomutesting_counts() -> None:
    output = """\
PASS "/tmp/x/calc.go.0" with checksum
FAIL "/tmp/x/calc.go.1"
PASS "/tmp/x/calc.go.2" with checksum
The mutation score is 0.667 (2 passed, 1 failed, 0 duration, 0 skipped)
"""
    counts = parse_gomutesting(output)
    assert counts.total == 3
    assert counts.killed == 2
    assert counts.survived == 1


def test_parse_gomutesting_missing_summary_raises() -> None:
    with pytest.raises(MutationToolError):
        parse_gomutesting("PASS something\n")


def test_toolcounts_rejects_killed_gt_total() -> None:
    with pytest.raises(MutationToolError):
        ToolCounts(total=2, killed=3)


# --------------------------------------------------------------------------- #
# MutationMeasurement math
# --------------------------------------------------------------------------- #
def test_measurement_ratio_and_survived() -> None:
    m = MutationMeasurement(total=10, killed=8, tool="x", survivors=("a", "b"))
    assert m.survived == 2
    assert m.kill_ratio == 0.8
    assert MutationMeasurement(total=0, killed=0).kill_ratio == 0.0


# --------------------------------------------------------------------------- #
# assess_mutation: adequate suite passes (VAL-ORACLE-006)
# --------------------------------------------------------------------------- #
async def test_adequate_suite_passes_without_synthesis() -> None:
    runner = FakeMutationRunner(total=10, base_survivors=set())
    synth = ScriptedSynth([])
    outcome = await assess_mutation(
        runner,
        synthesizer=synth,
        context_template=_template(),
        threshold=0.8,
    )
    assert outcome.is_pass
    assert outcome.mutants_total == 10
    assert outcome.mutants_killed == 10
    assert outcome.rounds == 0
    assert outcome.added_tests == []
    # measured exactly once (the baseline), no synthesis rounds
    assert runner.measure_calls == [()]
    assert synth.contexts == []


# --------------------------------------------------------------------------- #
# assess_mutation: under-determined -> synthesis (VAL-ORACLE-007)
# --------------------------------------------------------------------------- #
async def test_survivors_trigger_synthesis_until_threshold() -> None:
    runner = FakeMutationRunner(
        total=10,
        base_survivors={"s1", "s2", "s3"},  # killed 7 -> ratio 0.7 < 0.8
        kill_map={"tests/k1.py": {"s1", "s2"}},
    )
    synth = ScriptedSynth([[_test("tests/k1.py")]])
    outcome = await assess_mutation(
        runner,
        synthesizer=synth,
        context_template=_template(),
        threshold=0.8,
    )
    assert outcome.is_pass
    assert outcome.mutants_killed == 9  # killed s1,s2 -> only s3 survives
    assert outcome.mutants_total == 10
    assert outcome.rounds == 1
    assert [f.path for t in outcome.added_tests for f in t.files] == ["tests/k1.py"]
    assert outcome.survivors == ["s3"]
    # the synthesizer saw the surviving-mutant ids
    assert set(synth.contexts[0].survivors) == {"s1", "s2", "s3"}


async def test_non_reducing_proposal_is_discarded() -> None:
    runner = FakeMutationRunner(
        total=10,
        base_survivors={"s1", "s2", "s3"},
        kill_map={
            "tests/noop.py": set(),  # kills nothing
            "tests/good.py": {"s1", "s2", "s3"},  # kills all
        },
    )
    synth = ScriptedSynth([[_test("tests/noop.py"), _test("tests/good.py")]])
    outcome = await assess_mutation(
        runner,
        synthesizer=synth,
        context_template=_template(),
        threshold=0.8,
    )
    assert outcome.is_pass
    # only the survivor-reducing test is kept; the no-op is discarded
    assert [f.path for t in outcome.added_tests for f in t.files] == ["tests/good.py"]
    assert outcome.mutants_killed == 10


async def test_survivors_decrease_across_multiple_rounds() -> None:
    runner = FakeMutationRunner(
        total=10,
        base_survivors={"s1", "s2", "s3", "s4"},  # killed 6 -> 0.6
        kill_map={
            "tests/r1.py": {"s1", "s2"},
            "tests/r2.py": {"s3"},
        },
    )
    synth = ScriptedSynth([[_test("tests/r1.py")], [_test("tests/r2.py")]])
    outcome = await assess_mutation(
        runner,
        synthesizer=synth,
        context_template=_template(),
        threshold=0.9,  # need <=1 survivor
    )
    assert outcome.is_pass
    assert outcome.rounds == 2
    assert outcome.mutants_killed == 9  # only s4 survives
    assert outcome.survivors == ["s4"]
    # survivor counts strictly decreased across rounds
    rounds = outcome.details["rounds"]
    assert [r["survived_after"] for r in rounds] == [2, 1]


# --------------------------------------------------------------------------- #
# assess_mutation: unreachable threshold -> reject (VAL-ORACLE-008)
# --------------------------------------------------------------------------- #
async def test_unreachable_threshold_rejects_citing_survivors() -> None:
    runner = FakeMutationRunner(
        total=10,
        base_survivors={"s1", "s2", "s3", "s4", "s5"},  # killed 5 -> 0.5
        kill_map={
            "tests/r1.py": {"s1"},
            "tests/r2.py": {"s2"},
        },
    )
    synth = ScriptedSynth([[_test("tests/r1.py")], [_test("tests/r2.py")]])
    outcome = await assess_mutation(
        runner,
        synthesizer=synth,
        context_template=_template(),
        threshold=0.8,
        max_rounds=2,
    )
    assert not outcome.is_pass
    assert outcome.verdict == "reject"
    assert outcome.mutants_killed == 7  # killed s1,s2 -> 3 remain
    assert len(outcome.survivors) == 3
    assert outcome.reasons
    assert outcome.reasons[0].startswith(REASON_MUTATION_INADEQUATE)
    assert "surviving mutant" in outcome.reasons[0]


async def test_stuck_synthesis_breaks_early_and_rejects() -> None:
    runner = FakeMutationRunner(
        total=10,
        base_survivors={"s1", "s2", "s3"},  # 0.7
        kill_map={"tests/noop.py": set()},
    )
    synth = ScriptedSynth([[_test("tests/noop.py")]])
    outcome = await assess_mutation(
        runner,
        synthesizer=synth,
        context_template=_template(),
        threshold=0.8,
        max_rounds=3,
    )
    assert outcome.verdict == "reject"
    assert outcome.rounds == 1  # stopped after no progress in round 1
    assert outcome.added_tests == []


async def test_below_threshold_without_synthesizer_rejects() -> None:
    runner = FakeMutationRunner(total=10, base_survivors={"s1", "s2", "s3"})
    outcome = await assess_mutation(
        runner,
        synthesizer=None,
        threshold=0.8,
    )
    assert outcome.verdict == "reject"
    assert outcome.rounds == 0
    assert outcome.reasons[0].startswith(REASON_MUTATION_INADEQUATE)


async def test_zero_mutants_rejects() -> None:
    runner = FakeMutationRunner(total=0, base_survivors=set())
    outcome = await assess_mutation(runner, threshold=0.8)
    assert outcome.verdict == "reject"
    assert outcome.mutants_total == 0
    assert outcome.reasons[0].startswith(REASON_NO_MUTANTS)


async def test_null_synthesizer_proposes_nothing() -> None:
    ctx = _template()
    assert await NullMutationSynthesizer()(ctx) == []


async def test_invalid_threshold_raises() -> None:
    runner = FakeMutationRunner(total=4, base_survivors=set())
    with pytest.raises(MutationError):
        await assess_mutation(runner, threshold=1.5)


# --------------------------------------------------------------------------- #
# build_mutation_report
# --------------------------------------------------------------------------- #
async def test_build_report_pass_appends_tests_and_sets_counts() -> None:
    runner = FakeMutationRunner(
        total=10,
        base_survivors={"s1", "s2", "s3"},
        kill_map={"tests/k1.py": {"s1", "s2", "s3"}},
    )
    synth = ScriptedSynth([[_test("tests/k1.py")]])
    outcome = await assess_mutation(
        runner, synthesizer=synth, context_template=_template(), threshold=0.8
    )
    prior = _flakiness_report(
        [OracleTestFile(path="tests/test_x.py", content="X", origin="synthesized")]
    )
    report = build_mutation_report(_candidate(), prior, outcome, env_image=_env_image())

    assert report.verdict == "pass"
    assert report.mutants_total == 10
    assert report.mutants_killed == 10
    # established + synthesized survivor-killing test both present
    assert [tf.path for tf in report.test_files] == ["tests/test_x.py", "tests/k1.py"]
    # establish/flakiness fields carried forward
    assert report.flakiness_runs == 3
    assert report.fail_to_pass == ["python -m pytest tests/test_x.py"]
    assert (
        "flakiness" not in report.details or report.details.get("stage") != "mutation"
    )
    assert report.details["mutation"]["final"]["kill_ratio"] == 1.0
    # serializable + reproducible
    again = OracleReport.from_dict(report.to_dict())
    assert again.mutants_killed == 10
    assert again.verdict == "pass"


async def test_build_report_reject_carries_reason() -> None:
    runner = FakeMutationRunner(total=10, base_survivors={"s1", "s2", "s3"})
    outcome = await assess_mutation(runner, synthesizer=None, threshold=0.8)
    prior = _flakiness_report([OracleTestFile(path="tests/test_x.py", content="X")])
    report = build_mutation_report(_candidate(), prior, outcome)
    assert report.verdict == "reject"
    assert report.reasons
    assert any(REASON_MUTATION_INADEQUATE in r for r in report.reasons)
    assert report.mutants_total == 10
    assert report.mutants_killed == 7
    # reject invariant survives (de)serialization
    assert OracleReport.from_dict(report.to_dict()).verdict == "reject"


# --------------------------------------------------------------------------- #
# reconstruct_base_tests
# --------------------------------------------------------------------------- #
def test_reconstruct_base_tests_skips_empty_bodies() -> None:
    files = [
        OracleTestFile(path="tests/a.py", content="A", origin="synthesized"),
        OracleTestFile(path="tests/b.py", content="", origin="provided"),
    ]
    tests = reconstruct_base_tests(files)
    assert [t.files[0].path for t in tests] == ["tests/a.py"]
    assert tests[0].origin == "synthesized"


# --------------------------------------------------------------------------- #
# run_mutation_gate guards (offline; before any Docker use)
# --------------------------------------------------------------------------- #
async def test_run_gate_requires_passing_prior_report() -> None:
    reject_prior = OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="reject",
        reasons=["flakiness_nondeterministic_p2p: ..."],
        fail_to_pass=[],
        pass_to_pass=["python -m pytest"],
    )
    with pytest.raises(MutationError):
        await run_mutation_gate(
            _candidate(), _env_image(), reject_prior, adapter=PythonAdapter()
        )
