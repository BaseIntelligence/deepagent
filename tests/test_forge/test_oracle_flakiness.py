"""Unit tests for the flakiness gate (m4-flakiness).

Offline coverage (no real Docker) of the determinism gate's contract assertions,
driven through a programmable recipe + a fresh-container factory fake:

- VAL-ORACLE-004: the established F2P+P2P validation runs >=3 times in fresh
  throwaway containers; a deterministic suite yields identical pass/fail verdicts
  across all runs and ``OracleReport.flakiness_runs`` records the count (>=3).
- VAL-ORACLE-005: a nondeterministic F2P test (varying verdict across runs) is
  dropped from ``fail_to_pass``/``test_files``; if dropping removes the last F2P
  the candidate is rejected with a flakiness reason; a nondeterministic
  P2P/regression suite is rejected with a flakiness reason (it cannot be dropped).

The real DockerSandbox path is exercised by this feature's manual verification and
the user-testing validator in real Docker.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    EnvImage,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.establish import (
    HiddenTest,
    HiddenTestFile,
    TestRun,
    TreeState,
    build_establish_report,
)
from swe_forge.forge.oracle.flakiness import (
    MIN_FLAKINESS_RUNS,
    REASON_FLAKY_LAST_F2P,
    REASON_FLAKY_P2P,
    FlakinessError,
    assess_flakiness,
    build_flakiness_report,
    reconstruct_hidden_tests,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class ScriptedRecipe:
    """A programmable RecipeProtocol representing ONE fresh-container run.

    ``p2p`` is ``(gold_passed, broken_passed)``; ``outcomes`` maps a test id to
    ``(fails_on_broken, passes_on_gold)`` for this run. Exit codes are derived
    from the pass/fail verdict so evidence is populated.
    """

    def __init__(
        self,
        *,
        language: str = "python",
        p2p_command: str = "python -m pytest",
        p2p: tuple[bool, bool] = (True, True),
        outcomes: dict[str, tuple[bool, bool]] | None = None,
    ) -> None:
        self.language = language
        self.p2p_command = p2p_command
        self._p2p_gold, self._p2p_broken = p2p
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
        passed = (
            (not fails_on_broken) if self.state == TreeState.BROKEN else passes_on_gold
        )
        return TestRun(
            command=test.test_id, exit_code=0 if passed else 1, passed=passed
        )


class CountingFactory:
    """Yields pre-built recipes in order; counts how many fresh recipes were used."""

    def __init__(self, recipes: list[ScriptedRecipe]) -> None:
        self._recipes = recipes
        self.calls = 0

    def __call__(self):  # type: ignore[no-untyped-def]
        @asynccontextmanager
        async def cm():  # type: ignore[no-untyped-def]
            recipe = self._recipes[self.calls]
            self.calls += 1
            yield recipe

        return cm()


def _f2p(test_id: str, path: str | None = None) -> HiddenTest:
    files = (HiddenTestFile(path=path, content="..."),) if path else ()
    return HiddenTest(test_id=test_id, files=files, origin="synthesized")


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


def _establish_report(
    fail_to_pass: list[str], test_files: list[OracleTestFile]
) -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        fail_to_pass=fail_to_pass,
        pass_to_pass=["python -m pytest"],
        test_files=test_files,
        provenance=Provenance(generator="ast_mutation", seed=7, language="python"),
        details={"stage": "establish"},
    )


# --------------------------------------------------------------------------- #
# reconstruct_hidden_tests
# --------------------------------------------------------------------------- #
def test_reconstruct_maps_commands_to_test_files_by_path() -> None:
    f2p = ["python -m pytest tests/test_a.py", "python -m pytest tests/test_b.py"]
    files = [
        OracleTestFile(path="tests/test_a.py", content="A", origin="synthesized"),
        OracleTestFile(path="tests/test_b.py", content="B", origin="provided"),
    ]
    tests = reconstruct_hidden_tests(f2p, files)
    assert [t.test_id for t in tests] == f2p
    assert tests[0].files[0].path == "tests/test_a.py"
    assert tests[0].files[0].content == "A"
    assert tests[1].origin == "provided"


def test_reconstruct_command_without_matching_file_has_no_files() -> None:
    tests = reconstruct_hidden_tests(["python -m pytest tests/existing.py"], [])
    assert tests[0].files == ()


# --------------------------------------------------------------------------- #
# assess_flakiness: deterministic pass (VAL-ORACLE-004)
# --------------------------------------------------------------------------- #
async def test_deterministic_suite_passes_and_records_runs() -> None:
    f2p_id = "python -m pytest tests/test_x.py"
    tests = [_f2p(f2p_id, "tests/test_x.py")]
    recipes = [ScriptedRecipe(outcomes={f2p_id: (True, True)}) for _ in range(3)]
    factory = CountingFactory(recipes)

    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )

    assert outcome.verdict == "pass"
    assert outcome.reasons == []
    assert outcome.flakiness_runs == 3
    assert outcome.fail_to_pass == [f2p_id]
    assert outcome.dropped_test_ids == []
    # fresh container per run
    assert factory.calls == 3
    # identical exit codes across runs (evidence)
    per_test = outcome.details["per_test"][f2p_id]
    assert per_test["broken_exit_codes"] == [1, 1, 1]
    assert per_test["gold_exit_codes"] == [0, 0, 0]


async def test_runs_are_clamped_up_to_minimum() -> None:
    f2p_id = "python -m pytest tests/test_x.py"
    tests = [_f2p(f2p_id, "tests/test_x.py")]
    recipes = [ScriptedRecipe(outcomes={f2p_id: (True, True)}) for _ in range(5)]
    factory = CountingFactory(recipes)

    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=1
    )
    assert outcome.flakiness_runs == MIN_FLAKINESS_RUNS
    assert factory.calls == MIN_FLAKINESS_RUNS


async def test_more_runs_than_minimum_are_honored() -> None:
    f2p_id = "python -m pytest tests/test_x.py"
    tests = [_f2p(f2p_id, "tests/test_x.py")]
    recipes = [ScriptedRecipe(outcomes={f2p_id: (True, True)}) for _ in range(5)]
    factory = CountingFactory(recipes)

    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=5
    )
    assert outcome.flakiness_runs == 5
    assert factory.calls == 5


async def test_empty_f2p_raises() -> None:
    factory = CountingFactory([])
    with pytest.raises(FlakinessError):
        await assess_flakiness(
            factory, f2p_tests=[], p2p_command="python -m pytest", runs=3
        )


# --------------------------------------------------------------------------- #
# assess_flakiness: flaky F2P -> drop (VAL-ORACLE-005)
# --------------------------------------------------------------------------- #
async def test_flaky_f2p_dropped_when_sound_f2p_remains() -> None:
    flaky = "python -m pytest tests/test_flaky.py"
    sound = "python -m pytest tests/test_sound.py"
    tests = [_f2p(flaky, "tests/test_flaky.py"), _f2p(sound, "tests/test_sound.py")]
    # The flaky test passes-on-gold in runs 0 and 2 but FAILS on gold in run 1.
    recipes = [
        ScriptedRecipe(outcomes={flaky: (True, True), sound: (True, True)}),
        ScriptedRecipe(outcomes={flaky: (True, False), sound: (True, True)}),
        ScriptedRecipe(outcomes={flaky: (True, True), sound: (True, True)}),
    ]
    factory = CountingFactory(recipes)

    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )

    assert outcome.verdict == "pass"
    assert outcome.dropped_test_ids == [flaky]
    assert outcome.fail_to_pass == [sound]
    assert outcome.details["per_test"][flaky]["flaky"] is True
    assert outcome.details["per_test"][flaky]["gold_exit_codes"] == [0, 1, 0]
    assert outcome.details["per_test"][sound]["flaky"] is False


async def test_flaky_on_broken_tree_is_also_detected() -> None:
    flaky = "python -m pytest tests/test_flaky.py"
    sound = "python -m pytest tests/test_sound.py"
    tests = [_f2p(flaky, "tests/test_flaky.py"), _f2p(sound, "tests/test_sound.py")]
    # The flaky test fails-on-broken in runs 0,2 but PASSES on broken in run 1.
    recipes = [
        ScriptedRecipe(outcomes={flaky: (True, True), sound: (True, True)}),
        ScriptedRecipe(outcomes={flaky: (False, True), sound: (True, True)}),
        ScriptedRecipe(outcomes={flaky: (True, True), sound: (True, True)}),
    ]
    factory = CountingFactory(recipes)

    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )
    assert outcome.verdict == "pass"
    assert outcome.dropped_test_ids == [flaky]
    assert outcome.fail_to_pass == [sound]


# --------------------------------------------------------------------------- #
# assess_flakiness: flaky F2P -> reject (dropping removes last F2P)
# --------------------------------------------------------------------------- #
async def test_flaky_last_f2p_rejects() -> None:
    flaky = "python -m pytest tests/test_only.py"
    tests = [_f2p(flaky, "tests/test_only.py")]
    recipes = [
        ScriptedRecipe(outcomes={flaky: (True, True)}),
        ScriptedRecipe(outcomes={flaky: (True, False)}),
        ScriptedRecipe(outcomes={flaky: (True, True)}),
    ]
    factory = CountingFactory(recipes)

    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )
    assert outcome.verdict == "reject"
    assert outcome.fail_to_pass == []
    assert outcome.dropped_test_ids == [flaky]
    assert any(r.startswith(REASON_FLAKY_LAST_F2P) for r in outcome.reasons)


# --------------------------------------------------------------------------- #
# assess_flakiness: flaky P2P -> reject (cannot drop the regression suite)
# --------------------------------------------------------------------------- #
async def test_flaky_p2p_rejects() -> None:
    f2p_id = "python -m pytest tests/test_x.py"
    tests = [_f2p(f2p_id, "tests/test_x.py")]
    # P2P on gold flips: passes in runs 0,2 but fails in run 1.
    recipes = [
        ScriptedRecipe(p2p=(True, True), outcomes={f2p_id: (True, True)}),
        ScriptedRecipe(p2p=(False, True), outcomes={f2p_id: (True, True)}),
        ScriptedRecipe(p2p=(True, True), outcomes={f2p_id: (True, True)}),
    ]
    factory = CountingFactory(recipes)

    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )
    assert outcome.verdict == "reject"
    assert outcome.flaky_p2p is True
    assert any(r.startswith(REASON_FLAKY_P2P) for r in outcome.reasons)


# --------------------------------------------------------------------------- #
# build_flakiness_report
# --------------------------------------------------------------------------- #
async def test_build_report_pass_preserves_f2p_and_sets_runs() -> None:
    f2p_id = "python -m pytest tests/test_x.py"
    tests = [_f2p(f2p_id, "tests/test_x.py")]
    recipes = [ScriptedRecipe(outcomes={f2p_id: (True, True)}) for _ in range(3)]
    factory = CountingFactory(recipes)
    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )

    establish = _establish_report(
        [f2p_id], [OracleTestFile(path="tests/test_x.py", content="...")]
    )
    report = build_flakiness_report(
        _candidate(), establish, outcome, env_image=_env_image()
    )

    assert isinstance(report, OracleReport)
    assert report.verdict == "pass"
    assert report.flakiness_runs == 3
    assert report.fail_to_pass == [f2p_id]
    assert [tf.path for tf in report.test_files] == ["tests/test_x.py"]
    assert report.details["flakiness"]["runs"] == 3
    # serializable + reproducible field set
    assert OracleReport.from_dict(report.to_dict()).flakiness_runs == 3


async def test_build_report_drops_flaky_test_file() -> None:
    flaky = "python -m pytest tests/test_flaky.py"
    sound = "python -m pytest tests/test_sound.py"
    tests = [_f2p(flaky, "tests/test_flaky.py"), _f2p(sound, "tests/test_sound.py")]
    recipes = [
        ScriptedRecipe(outcomes={flaky: (True, True), sound: (True, True)}),
        ScriptedRecipe(outcomes={flaky: (True, False), sound: (True, True)}),
        ScriptedRecipe(outcomes={flaky: (True, True), sound: (True, True)}),
    ]
    factory = CountingFactory(recipes)
    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )

    establish = _establish_report(
        [flaky, sound],
        [
            OracleTestFile(path="tests/test_flaky.py", content="F"),
            OracleTestFile(path="tests/test_sound.py", content="S"),
        ],
    )
    report = build_flakiness_report(_candidate(), establish, outcome)

    assert report.verdict == "pass"
    assert report.fail_to_pass == [sound]
    # the flaky test's file is dropped; the sound one is kept
    assert [tf.path for tf in report.test_files] == ["tests/test_sound.py"]


async def test_build_report_reject_carries_flakiness_reason() -> None:
    flaky = "python -m pytest tests/test_only.py"
    tests = [_f2p(flaky, "tests/test_only.py")]
    recipes = [
        ScriptedRecipe(outcomes={flaky: (True, True)}),
        ScriptedRecipe(outcomes={flaky: (True, False)}),
        ScriptedRecipe(outcomes={flaky: (True, True)}),
    ]
    factory = CountingFactory(recipes)
    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )

    establish = _establish_report(
        [flaky], [OracleTestFile(path="tests/test_only.py", content="O")]
    )
    report = build_flakiness_report(_candidate(), establish, outcome)
    assert report.verdict == "reject"
    assert report.reasons
    assert any("flakiness" in r for r in report.reasons)
    assert report.flakiness_runs == 3
    # de/serialization holds the reject invariant
    assert OracleReport.from_dict(report.to_dict()).verdict == "reject"


# --------------------------------------------------------------------------- #
# integration with the establish report builder (shape parity)
# --------------------------------------------------------------------------- #
async def test_flakiness_extends_an_establish_report() -> None:
    from swe_forge.forge.oracle.establish import EstablishOutcome

    f2p_id = "python -m pytest tests/test_x.py"
    establish_outcome = EstablishOutcome(
        verdict="pass",
        reasons=[],
        fail_to_pass=[f2p_id],
        pass_to_pass=["python -m pytest"],
        test_files=[OracleTestFile(path="tests/test_x.py", content="...")],
    )
    establish = build_establish_report(
        _candidate(), establish_outcome, env_image=_env_image()
    )

    tests = [_f2p(f2p_id, "tests/test_x.py")]
    recipes = [ScriptedRecipe(outcomes={f2p_id: (True, True)}) for _ in range(3)]
    factory = CountingFactory(recipes)
    outcome = await assess_flakiness(
        factory, f2p_tests=tests, p2p_command="python -m pytest", runs=3
    )
    report = build_flakiness_report(
        _candidate(), establish, outcome, env_image=_env_image()
    )
    assert report.verdict == "pass"
    assert report.flakiness_runs == 3
    # establish stage details preserved alongside flakiness details
    assert "flakiness" in report.details
