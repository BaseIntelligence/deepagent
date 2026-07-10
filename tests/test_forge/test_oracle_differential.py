"""Unit tests for the differential-vs-gold gate (m4-differential).

Offline coverage (no real Docker, no live LLM) of the gate's contract assertions,
driven through a programmable :class:`DifferentialRunner` fake + a scripted
synthesizer, plus parser coverage of the teacher-backed variant generator /
survivor-killing synthesizer through a fake teacher client:

- VAL-ORACLE-009: gold passes F2P+P2P; every plausible-wrong variant fails >=1
  test; ``differential_pass == True``.
- VAL-ORACLE-010: a wrong variant that survives the suite is killed by an added
  separating test (teacher proposes -> execution confirms it fails the variant and
  passes gold), then ``differential_pass == True``.
- VAL-ORACLE-011: an indistinguishable survivor after bounded strengthening ->
  ``differential_pass == False`` and ``verdict == reject`` citing the variant.

The real DockerSandbox + live-teacher paths are exercised by this feature's manual
verification and the user-testing validator in real Docker.
"""

from __future__ import annotations

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
from swe_forge.forge.oracle.differential import (
    REASON_DIFFERENTIAL_GOLD_NOT_GREEN,
    REASON_DIFFERENTIAL_SURVIVOR,
    DifferentialError,
    DifferentialSynthesisContext,
    NullVariantGenerator,
    NullVariantSynthesizer,
    Variant,
    VariantFile,
    VariantGenerationContext,
    VariantScore,
    assess_differential,
    build_differential_report,
    reconstruct_suite_tests,
    run_differential_gate,
)
from swe_forge.forge.oracle.differential_synth import (
    DifferentialKillSynthesizer,
    TeacherVariantGenerator,
)
from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeDifferentialRunner:
    """A programmable :class:`DifferentialRunner`.

    ``base_killed`` are the variant ids that already FAIL the established suite
    (no extra tests). ``kill_map`` maps a variant id to the set of extra-test
    paths that (additionally) kill it. ``gold_breakers`` are extra-test paths that
    would break gold (so the gate must discard them). ``gold_base_fails`` makes the
    initial gold suite run fail (the defensive gold-not-green path).

    ``discardable`` are inherited SYNTHESIZED discriminators (later-gate
    survivor-killers, not established F2P) the gate may drop; ``gold_red_ids`` are
    the test ids among them that are RED on gold (a mis-modeled discriminator) --
    excluding all of them from a gold run makes gold green again (unless
    ``protected_gold_red`` also fails, which models a genuine gold-not-green that
    no discard can fix).
    """

    def __init__(
        self,
        *,
        base_killed: set[str] | None = None,
        kill_map: dict[str, set[str]] | None = None,
        gold_breakers: set[str] | None = None,
        gold_base_fails: bool = False,
        sources: dict[str, str] | None = None,
        discardable: list[HiddenTest] | None = None,
        gold_red_ids: set[str] | None = None,
        protected_gold_red: bool = False,
    ) -> None:
        self.language = "python"
        self._base_killed = set(base_killed or set())
        self._kill_map = kill_map or {}
        self._gold_breakers = set(gold_breakers or set())
        self._gold_base_fails = gold_base_fails
        self._sources = sources or {}
        self.discardable_tests = tuple(discardable or ())
        self._gold_red_ids = set(gold_red_ids or set())
        self._protected_gold_red = protected_gold_red
        self.gold_calls: list[tuple[str, ...]] = []
        self.gold_exclude_calls: list[tuple[str, ...]] = []
        self.variant_calls: list[tuple[str, tuple[str, ...]]] = []
        self.read_sources_calls = 0

    def _active_red(self, exclude: set[str]) -> tuple[str, ...]:
        """Red-on-gold base discriminators that are NOT excluded."""
        return tuple(sorted(self._gold_red_ids - exclude))

    async def score_gold(self, extra_tests, *, exclude=()) -> VariantScore:  # type: ignore[no-untyped-def]
        paths = tuple(f.path for t in extra_tests for f in t.files)
        excluded = set(exclude)
        self.gold_calls.append(paths)
        self.gold_exclude_calls.append(tuple(sorted(excluded)))
        if self._gold_base_fails and not paths:
            return VariantScore(
                f2p_passed=False, p2p_passed=True, failing_test_ids=("base",)
            )
        breakers = tuple(p for p in paths if p in self._gold_breakers)
        active_red = self._active_red(excluded)
        protected = ("__protected__",) if self._protected_gold_red else ()
        failing = breakers + active_red + protected
        return VariantScore(
            f2p_passed=not failing,
            p2p_passed=True,
            failing_test_ids=failing,
        )

    async def score_variant(self, variant, extra_tests, *, exclude=()) -> VariantScore:  # type: ignore[no-untyped-def]
        paths = {f.path for t in extra_tests for f in t.files}
        excluded = set(exclude)
        self.variant_calls.append((variant.variant_id, tuple(sorted(paths))))
        killed = variant.variant_id in self._base_killed or bool(
            paths & self._kill_map.get(variant.variant_id, set())
        )
        active_red = self._active_red(excluded)
        # A red-on-gold discriminator fails on the variant too (it fails on
        # everything), so it would spuriously "kill" the variant if not excluded.
        return VariantScore(
            f2p_passed=not killed and not active_red,
            p2p_passed=True,
            failing_test_ids=("killer",) if (killed or active_red) else (),
        )

    async def read_sources(self) -> dict[str, str]:
        self.read_sources_calls += 1
        return dict(self._sources)


class ScriptedVariantSynth:
    """Returns a pre-scripted list of proposals per round (1-indexed)."""

    def __init__(self, rounds: list[list[HiddenTest]]) -> None:
        self._rounds = rounds
        self.contexts: list[DifferentialSynthesisContext] = []

    async def __call__(self, ctx: DifferentialSynthesisContext) -> list[HiddenTest]:
        self.contexts.append(ctx)
        idx = ctx.round_index - 1
        return self._rounds[idx] if 0 <= idx < len(self._rounds) else []


class FakeLLMResult:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeTeacher:
    """Minimal stand-in for TeacherClient.complete_text."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def complete_text(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return FakeLLMResult(self._text)


def _test(path: str) -> HiddenTest:
    return HiddenTest(
        test_id=f"python -m pytest {path}",
        files=(HiddenTestFile(path=path, content=f"# test body {path}"),),
        origin="synthesized",
    )


def _variant(vid: str, content: str = "def f():\n    return 2\n") -> Variant:
    return Variant(
        variant_id=vid,
        files=(VariantFile(path="src/m.py", content=content),),
        description=f"wrong {vid}",
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


def _template() -> DifferentialSynthesisContext:
    return DifferentialSynthesisContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        gold_sources={},
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


def _mutation_report(test_files: list[OracleTestFile]) -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=["python -m pytest tests/test_x.py"],
        pass_to_pass=["python -m pytest"],
        test_files=test_files,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        provenance=Provenance(generator="ast_mutation", seed=7, language="python"),
        details={"stage": "mutation"},
    )


# --------------------------------------------------------------------------- #
# Variant / VariantScore value types
# --------------------------------------------------------------------------- #
def test_variant_requires_files() -> None:
    with pytest.raises(ModelError):
        Variant(variant_id="v1", files=())


def test_variant_requires_id() -> None:
    with pytest.raises(ModelError):
        Variant(variant_id="  ", files=(VariantFile(path="x.py", content="y"),))


def test_variant_score_passes_suite() -> None:
    assert VariantScore(f2p_passed=True, p2p_passed=True).passes_suite
    assert not VariantScore(f2p_passed=False, p2p_passed=True).passes_suite
    assert not VariantScore(f2p_passed=True, p2p_passed=False).passes_suite


# --------------------------------------------------------------------------- #
# VAL-ORACLE-009: gold accepted, every wrong variant rejected
# --------------------------------------------------------------------------- #
async def test_all_variants_killed_passes() -> None:
    runner = FakeDifferentialRunner(base_killed={"variant_1", "variant_2"})
    outcome = await assess_differential(
        runner,
        [_variant("variant_1"), _variant("variant_2")],
        synthesizer=ScriptedVariantSynth([]),
        context_template=_template(),
    )
    assert outcome.is_pass
    assert outcome.differential_pass is True
    assert outcome.variants_total == 2
    assert outcome.variants_killed == 2
    assert outcome.survivors == []
    assert outcome.rounds == 0
    # gold scored once at base; no strengthening read of sources
    assert runner.gold_calls == [()]
    assert runner.read_sources_calls == 0


async def test_no_variants_rejects_nonvacuously() -> None:
    runner = FakeDifferentialRunner()
    outcome = await assess_differential(runner, [], context_template=_template())
    assert not outcome.is_pass
    assert outcome.differential_pass is False
    assert outcome.variants_total == 0
    assert outcome.variants_killed == 0


# --------------------------------------------------------------------------- #
# VAL-ORACLE-010: a survivor is killed by an added separating test
# --------------------------------------------------------------------------- #
async def test_survivor_killed_by_strengthening() -> None:
    runner = FakeDifferentialRunner(
        base_killed={"variant_1"},  # one already killed
        kill_map={"variant_2": {"tests/k.py"}},  # survivor killed by the new test
    )
    synth = ScriptedVariantSynth([[_test("tests/k.py")]])
    outcome = await assess_differential(
        runner,
        [_variant("variant_1"), _variant("variant_2")],
        synthesizer=synth,
        context_template=_template(),
    )
    assert outcome.is_pass
    assert outcome.differential_pass is True
    assert outcome.variants_killed == 2
    assert outcome.survivors == []
    assert outcome.rounds == 1
    assert [f.path for t in outcome.added_tests for f in t.files] == ["tests/k.py"]
    # the synthesizer saw exactly the surviving variant
    assert [v.variant_id for v in synth.contexts[0].survivors] == ["variant_2"]


async def test_strengthening_discards_test_that_breaks_gold() -> None:
    runner = FakeDifferentialRunner(
        kill_map={"variant_1": {"tests/bad.py", "tests/good.py"}},
        gold_breakers={"tests/bad.py"},  # would break gold -> must be discarded
    )
    synth = ScriptedVariantSynth([[_test("tests/bad.py"), _test("tests/good.py")]])
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=synth,
        context_template=_template(),
    )
    assert outcome.is_pass
    # only the gold-preserving separating test is kept
    assert [f.path for t in outcome.added_tests for f in t.files] == ["tests/good.py"]


async def test_strengthening_discards_non_separating_test() -> None:
    runner = FakeDifferentialRunner(
        kill_map={"variant_1": {"tests/kill.py"}},  # only this one separates
    )
    synth = ScriptedVariantSynth([[_test("tests/noop.py"), _test("tests/kill.py")]])
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=synth,
        context_template=_template(),
    )
    assert outcome.is_pass
    assert [f.path for t in outcome.added_tests for f in t.files] == ["tests/kill.py"]


async def test_multiple_rounds_separate_survivors() -> None:
    runner = FakeDifferentialRunner(
        kill_map={
            "variant_1": {"tests/r1.py"},
            "variant_2": {"tests/r2.py"},
        },
    )
    synth = ScriptedVariantSynth([[_test("tests/r1.py")], [_test("tests/r2.py")]])
    outcome = await assess_differential(
        runner,
        [_variant("variant_1"), _variant("variant_2")],
        synthesizer=synth,
        context_template=_template(),
    )
    assert outcome.is_pass
    assert outcome.rounds == 2
    assert outcome.variants_killed == 2
    rounds = outcome.details["rounds"]
    assert [r["survivors_after"] for r in rounds] == [["variant_2"], []]


# --------------------------------------------------------------------------- #
# VAL-ORACLE-011: unresolvable survivor -> reject, differential_pass=False
# --------------------------------------------------------------------------- #
async def test_unresolvable_survivor_rejects() -> None:
    runner = FakeDifferentialRunner(
        kill_map={"variant_1": set()},  # no test can separate it from gold
    )
    synth = ScriptedVariantSynth([[_test("tests/x.py")], [_test("tests/y.py")]])
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=synth,
        context_template=_template(),
        max_rounds=2,
    )
    assert not outcome.is_pass
    assert outcome.verdict == "reject"
    assert outcome.differential_pass is False
    assert outcome.survivors == ["variant_1"]
    assert outcome.reasons
    assert outcome.reasons[0].startswith(REASON_DIFFERENTIAL_SURVIVOR)
    assert "variant" in outcome.reasons[0]


async def test_survivor_without_synthesizer_rejects() -> None:
    runner = FakeDifferentialRunner()  # variant_1 survives, no synthesizer
    outcome = await assess_differential(
        runner, [_variant("variant_1")], synthesizer=None
    )
    assert outcome.verdict == "reject"
    assert outcome.differential_pass is False
    assert outcome.rounds == 0
    assert outcome.reasons[0].startswith(REASON_DIFFERENTIAL_SURVIVOR)


async def test_stuck_strengthening_breaks_early_and_rejects() -> None:
    runner = FakeDifferentialRunner(kill_map={"variant_1": set()})
    synth = ScriptedVariantSynth([[_test("tests/noop.py")]])
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=synth,
        context_template=_template(),
        max_rounds=3,
    )
    assert outcome.verdict == "reject"
    assert outcome.rounds == 1  # stopped after no progress in round 1
    assert outcome.added_tests == []


async def test_gold_not_green_rejects() -> None:
    runner = FakeDifferentialRunner(gold_base_fails=True)
    outcome = await assess_differential(
        runner, [_variant("variant_1")], context_template=_template()
    )
    assert outcome.verdict == "reject"
    assert outcome.differential_pass is False
    assert outcome.reasons[0].startswith(REASON_DIFFERENTIAL_GOLD_NOT_GREEN)
    # never scored a variant once gold failed
    assert runner.variant_calls == []


# --------------------------------------------------------------------------- #
# m6-differential-yield: mis-modeled synthesized discriminator is DISCARDED, not
# counted as a differential_gold_not_green reject of the candidate.
# --------------------------------------------------------------------------- #
async def test_red_on_gold_synthesized_discriminator_is_discarded_not_rejected() -> (
    None
):
    # An inherited SYNTHESIZED discriminator (a mutation-gate survivor-killer,
    # NOT an established F2P) is red on gold because the teacher mis-modeled the
    # target. The gate must DISCARD it and proceed -- never emit
    # differential_gold_not_green -- and still separate the real variant.
    bad = _test("tests/mut_survivor_k.py")
    runner = FakeDifferentialRunner(
        base_killed={"variant_1"},
        discardable=[bad],
        gold_red_ids={bad.test_id},
    )
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=ScriptedVariantSynth([]),
        context_template=_template(),
    )
    assert outcome.is_pass
    assert outcome.differential_pass is True
    assert not any(
        r.startswith(REASON_DIFFERENTIAL_GOLD_NOT_GREEN) for r in outcome.reasons
    )
    # the mis-modeled discriminator's file is recorded as discarded
    assert outcome.discarded_test_paths == ["tests/mut_survivor_k.py"]
    assert outcome.details["discarded_discriminators"] == ["tests/mut_survivor_k.py"]


async def test_discarded_discriminator_pruned_from_report_test_files() -> None:
    bad = _test("tests/mut_survivor_k.py")
    runner = FakeDifferentialRunner(
        base_killed={"variant_1"},
        discardable=[bad],
        gold_red_ids={bad.test_id},
    )
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=ScriptedVariantSynth([]),
        context_template=_template(),
    )
    prior = _mutation_report(
        [
            OracleTestFile(path="tests/test_x.py", content="X", origin="provided"),
            OracleTestFile(
                path="tests/mut_survivor_k.py", content="K", origin="synthesized"
            ),
        ]
    )
    report = build_differential_report(_candidate(), prior, outcome)
    assert report.verdict == "pass"
    # the red-on-gold synthesized discriminator is pruned so the shipped suite
    # stays gold=100%; the established F2P is retained.
    assert [tf.path for tf in report.test_files] == ["tests/test_x.py"]
    assert report.provenance.details["discarded_discriminators"] == [
        "tests/mut_survivor_k.py"
    ]


async def test_pruning_invalidates_prior_final_mutation_evidence() -> None:
    bad = _test("tests/mut_survivor_k.py")
    runner = FakeDifferentialRunner(
        base_killed={"variant_1"},
        discardable=[bad],
        gold_red_ids={bad.test_id},
    )
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=ScriptedVariantSynth([]),
        context_template=_template(),
    )
    prior = _mutation_report(
        [
            OracleTestFile(path="tests/test_x.py", content="X", origin="provided"),
            OracleTestFile(
                path="tests/mut_survivor_k.py", content="K", origin="synthesized"
            ),
        ]
    )
    prior.final_mutation_evidence = FinalMutationEvidence(
        suite_fingerprint="a" * 64,
        mutants_total=10,
        mutants_killed=10,
        threshold=0.8,
        tool="fake-tool",
    )

    report = build_differential_report(_candidate(), prior, outcome)

    assert report.final_mutation_evidence is None
    assert report.details["mutation_evidence_invalidated"] == {
        "stage": "differential",
        "reason": "hidden_suite_changed",
    }


async def test_genuine_gold_not_green_still_rejects_after_discards() -> None:
    # gold is still red after discarding every discardable discriminator -> the
    # failure is a PROTECTED test (established F2P / P2P): a genuine
    # differential_gold_not_green, still a real reject (case B untouched).
    bad = _test("tests/mut_survivor_k.py")
    runner = FakeDifferentialRunner(
        discardable=[bad],
        gold_red_ids={bad.test_id},
        protected_gold_red=True,
    )
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=ScriptedVariantSynth([]),
        context_template=_template(),
    )
    assert outcome.verdict == "reject"
    assert outcome.differential_pass is False
    assert outcome.reasons[0].startswith(REASON_DIFFERENTIAL_GOLD_NOT_GREEN)
    # never scored a variant once gold could not be made green
    assert runner.variant_calls == []


async def test_indistinguishable_variant_still_rejects_with_discarded_discriminator() -> (
    None
):
    # Even after discarding a mis-modeled discriminator, a genuinely
    # indistinguishable variant must STILL reject (the gate's purpose is intact).
    bad = _test("tests/mut_survivor_k.py")
    runner = FakeDifferentialRunner(
        discardable=[bad],
        gold_red_ids={bad.test_id},
        kill_map={"variant_1": set()},  # no valid test can separate it from gold
    )
    synth = ScriptedVariantSynth([[_test("tests/x.py")], [_test("tests/y.py")]])
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=synth,
        context_template=_template(),
        max_rounds=2,
    )
    assert outcome.verdict == "reject"
    assert outcome.differential_pass is False
    assert outcome.reasons[0].startswith(REASON_DIFFERENTIAL_SURVIVOR)
    assert outcome.survivors == ["variant_1"]
    # the mis-modeled discriminator was still discarded (not the cause of reject)
    assert outcome.discarded_test_paths == ["tests/mut_survivor_k.py"]


async def test_multiple_discriminators_only_red_ones_discarded() -> None:
    # Two inherited discriminators; only one is red on gold. The gate discards the
    # minimal set (just the red one) so the still-valid one keeps guarding.
    good = _test("tests/mut_good_k.py")
    bad = _test("tests/mut_bad_k.py")
    runner = FakeDifferentialRunner(
        base_killed={"variant_1"},
        discardable=[good, bad],
        gold_red_ids={bad.test_id},
    )
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=ScriptedVariantSynth([]),
        context_template=_template(),
    )
    assert outcome.is_pass
    assert outcome.discarded_test_paths == ["tests/mut_bad_k.py"]


async def test_synthesizer_set_without_template_raises() -> None:
    runner = FakeDifferentialRunner()  # variant survives -> would enter loop
    with pytest.raises(DifferentialError):
        await assess_differential(
            runner,
            [_variant("variant_1")],
            synthesizer=ScriptedVariantSynth([[_test("tests/k.py")]]),
            context_template=None,
        )


# --------------------------------------------------------------------------- #
# build_differential_report
# --------------------------------------------------------------------------- #
async def test_build_report_pass_appends_tests_and_sets_flag() -> None:
    runner = FakeDifferentialRunner(kill_map={"variant_1": {"tests/k.py"}})
    synth = ScriptedVariantSynth([[_test("tests/k.py")]])
    outcome = await assess_differential(
        runner,
        [_variant("variant_1")],
        synthesizer=synth,
        context_template=_template(),
    )
    prior = _mutation_report(
        [OracleTestFile(path="tests/test_x.py", content="X", origin="synthesized")]
    )
    report = build_differential_report(
        _candidate(), prior, outcome, env_image=_env_image()
    )
    assert report.verdict == "pass"
    assert report.differential_pass is True
    # established + separating test both present
    assert [tf.path for tf in report.test_files] == ["tests/test_x.py", "tests/k.py"]
    # mutation/flakiness fields carried forward
    assert report.flakiness_runs == 3
    assert report.mutants_total == 10
    assert report.mutants_killed == 10
    assert report.fail_to_pass == ["python -m pytest tests/test_x.py"]
    assert report.details["differential"]["final"]["variants_killed"] == 1
    # serializable + reproducible
    again = OracleReport.from_dict(report.to_dict())
    assert again.differential_pass is True
    assert again.verdict == "pass"


async def test_build_report_reject_carries_reason() -> None:
    runner = FakeDifferentialRunner()  # survivor, no synthesizer
    outcome = await assess_differential(
        runner, [_variant("variant_1")], synthesizer=None
    )
    prior = _mutation_report([OracleTestFile(path="tests/test_x.py", content="X")])
    report = build_differential_report(_candidate(), prior, outcome)
    assert report.verdict == "reject"
    assert report.differential_pass is False
    assert any(REASON_DIFFERENTIAL_SURVIVOR in r for r in report.reasons)
    # reject invariant survives (de)serialization
    assert OracleReport.from_dict(report.to_dict()).verdict == "reject"


# --------------------------------------------------------------------------- #
# reconstruct_suite_tests
# --------------------------------------------------------------------------- #
def test_reconstruct_suite_tests_uses_f2p_command_and_builds_others() -> None:
    adapter = PythonAdapter()
    files = [
        OracleTestFile(path="tests/test_x.py", content="X", origin="provided"),
        OracleTestFile(path="tests/mut_k.py", content="K", origin="synthesized"),
        OracleTestFile(path="tests/empty.py", content="", origin="synthesized"),
    ]
    tests = reconstruct_suite_tests(
        adapter, ["python -m pytest tests/test_x.py"], files
    )
    # F2P file keeps its exact command; mutation-added file gets a built command;
    # empty-body file is skipped.
    assert tests[0].test_id == "python -m pytest tests/test_x.py"
    assert tests[0].origin == "provided"
    assert "tests/mut_k.py" in tests[1].test_id
    assert [t.files[0].path for t in tests] == ["tests/test_x.py", "tests/mut_k.py"]


# --------------------------------------------------------------------------- #
# Null implementations
# --------------------------------------------------------------------------- #
async def test_null_variant_generator_and_synth_propose_nothing() -> None:
    gen_ctx = VariantGenerationContext(
        candidate=_candidate(), adapter=PythonAdapter(), gold_sources={}
    )
    assert await NullVariantGenerator()(gen_ctx) == []
    assert await NullVariantSynthesizer()(_template()) == []


# --------------------------------------------------------------------------- #
# run_differential_gate guards (offline; before any Docker use)
# --------------------------------------------------------------------------- #
async def test_run_gate_requires_passing_prior_report() -> None:
    reject_prior = OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="reject",
        reasons=["mutation_adequacy_below_threshold: ..."],
        fail_to_pass=[],
        pass_to_pass=["python -m pytest"],
    )
    with pytest.raises(DifferentialError):
        await run_differential_gate(
            _candidate(), _env_image(), reject_prior, adapter=PythonAdapter()
        )


# --------------------------------------------------------------------------- #
# Teacher-backed generator / synthesizer parsing (fake client; no live LLM)
# --------------------------------------------------------------------------- #
async def test_teacher_variant_generator_parses_blocks() -> None:
    text = (
        "Here are variants:\n"
        "```python\ndef f():\n    return 2\n```\n"
        "```python\ndef f():\n    return 0\n```\n"
    )
    gen = TeacherVariantGenerator(client=FakeTeacher(text))  # type: ignore[arg-type]
    ctx = VariantGenerationContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        gold_sources={"src/m.py": "def f():\n    return 1\n"},
        num_variants=3,
    )
    variants = await gen(ctx)
    assert [v.variant_id for v in variants] == ["variant_1", "variant_2"]
    assert all(v.files[0].path == "src/m.py" for v in variants)
    assert variants[0].files[0].content.startswith("def f():")


async def test_teacher_variant_generator_skips_block_equal_to_gold() -> None:
    gold = "def f():\n    return 1\n"
    text = f"```python\n{gold}```\n```python\ndef f():\n    return 9\n```\n"
    gen = TeacherVariantGenerator(client=FakeTeacher(text))  # type: ignore[arg-type]
    ctx = VariantGenerationContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        gold_sources={"src/m.py": gold},
        num_variants=3,
    )
    variants = await gen(ctx)
    # the block identical to gold is dropped; only the divergent one remains
    assert len(variants) == 1
    assert "return 9" in variants[0].files[0].content


async def test_teacher_variant_generator_no_sources_returns_empty() -> None:
    gen = TeacherVariantGenerator(client=FakeTeacher("```\nx\n```"))  # type: ignore[arg-type]
    ctx = VariantGenerationContext(
        candidate=_candidate(), adapter=PythonAdapter(), gold_sources={}
    )
    assert await gen(ctx) == []


async def test_differential_kill_synthesizer_builds_test() -> None:
    text = "```python\ndef test_f():\n    from m import f\n    assert f() == 1\n```"
    synth = DifferentialKillSynthesizer(client=FakeTeacher(text))  # type: ignore[arg-type]
    ctx = DifferentialSynthesisContext(
        candidate=_candidate(),
        adapter=PythonAdapter(),
        gold_sources={"src/m.py": "def f():\n    return 1\n"},
        survivors=(_variant("variant_1"),),
        round_index=1,
    )
    proposals = await synth(ctx)
    assert len(proposals) == 1
    assert proposals[0].files[0].path == "test_swe_forge_diff_1.py"
    assert "assert f() == 1" in proposals[0].files[0].content


async def test_differential_kill_synthesizer_no_survivors_returns_empty() -> None:
    synth = DifferentialKillSynthesizer(client=FakeTeacher("```\nx\n```"))  # type: ignore[arg-type]
    assert await synth(_template()) == []
