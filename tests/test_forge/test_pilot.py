"""Offline coverage of the Stage 0->5 pilot orchestrator (`build --pilot`).

Exercises the whole-pipeline invariants this feature's ``fulfills`` set requires
(VAL-CROSS-001..023) deterministically -- no Docker, no live endpoint -- by
threading a fake :class:`CandidateProcessor` through the real orchestrator and
the real Stage-5 export/report code:

- VAL-CROSS-013: the per-stage funnel is monotone
  (sourced>=env>=synth>=oracle-pass>=keep==exported).
- VAL-CROSS-012: only oracle-pass AND band-keep candidates become ForgeTasks;
  an env-fail / synth-fail / oracle-reject / calib-drop never produces a
  ``tasks/<id>/`` dir or a dataset row (rejections never propagate).
- VAL-CROSS-020: the pilot's keep/reject decision agrees with the canonical
  per-stage export gate (assemble refuses exactly what the pilot drops).
- VAL-CROSS-014/016: every kept task materializes as workspace+jsonl+parquet
  consistently and a re-run reproduces the equivalent shipped set.
- VAL-CROSS-017/022: provenance is complete + agrees with the report, and the
  teacher/panel usage+cost is surfaced and non-zero.
- VAL-CROSS-015: a planted leak never ships and the run does not silently pass.
- VAL-CROSS-005: the pilot uses >=2 generators across all three languages.

The Docker headlines (gold=100%; frontier<threshold>0 on real rollouts) and the
hygiene checks (no key in artifacts, container teardown) are proven by this
feature's manual integration run and the user-testing validator.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from swe_forge.export.jsonl import import_jsonl
from swe_forge.export.parquet import import_parquet
from swe_forge.forge.export import assemble_forge_task
from swe_forge.forge.gold_eval import EvalRun, GoldEvalReport, TaskGoldResult
from swe_forge.forge.models import (
    CalibrationReport,
    Candidate,
    CandidateTarget,
    EnvImage,
    GeneratedSpec,
    ModelSolveRecord,
    OracleReport,
    OracleTestFile,
    Provenance,
    RepoSpec,
)
from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile
from swe_forge.forge.oracle.pipeline import ExportRefusedError
from swe_forge.forge.sources import build_source_registry
from swe_forge.forge.pilot import (
    DEFAULT_GENERATORS_BY_LANGUAGE,
    CandidateArtifacts,
    CandidatePlan,
    PilotConfig,
    StageCounts,
    StructuralF2PProtection,
    StructuralF2PProtectionError,
    build_pilot_plans,
    default_pilot_config,
    run_pilot,
)
from swe_forge.forge.teacher import Usage

_TS = "2026-01-01T00:00:00+00:00"
_GOLD_LINE = "    return compute_total_with_tax(items, tax_rate)"
_BROKEN_LINE = "    return sum(items)"


def _structural_protection() -> StructuralF2PProtection:
    return StructuralF2PProtection(
        tests=(
            HiddenTest(
                test_id="python -m pytest tests/hidden/test_f2p.py",
                files=(HiddenTestFile(path="tests/hidden/test_f2p.py", content="x"),),
                origin="provided",
            ),
        ),
        protected_names=("test_f2p",),
        protected_files=("tests/hidden/test_f2p.py",),
    )


# --------------------------------------------------------------------------- #
# Fixtures: plans + per-stage artifacts (one ForgeTask's worth, offline)
# --------------------------------------------------------------------------- #
def _repo(repo_id: str, language: str = "python") -> RepoSpec:
    return RepoSpec(
        repo_id=repo_id,
        url=f"https://github.com/acme/{repo_id}.git",
        commit="a" * 40,
        commit_date="2024-01-01T00:00:00+00:00",
        language=language,
        license="MIT",
        instance_cap=10,
    )


def _plan(
    repo_id: str, generator: str, seed: int, language: str = "python"
) -> CandidatePlan:
    return CandidatePlan(repo=_repo(repo_id, language), generator=generator, seed=seed)


def _provenance(plan: CandidatePlan) -> Provenance:
    return Provenance(
        generator=plan.generator,
        seed=plan.seed,
        language=plan.language,
        created_at=_TS,
    )


def _target_file(plan: CandidatePlan) -> str:
    return f"src/{plan.repo.repo_id}_{plan.generator}_{plan.seed}.py"


def _mutation_patch(file: str) -> str:
    return (
        f"--- a/{file}\n+++ b/{file}\n@@ -1,2 +1,2 @@\n"
        " def total(items, tax_rate):\n"
        f"-{_GOLD_LINE[4:]}\n"
        f"+{_BROKEN_LINE[4:]}\n"
    )


def _oracle_patch(file: str) -> str:
    return (
        f"--- a/{file}\n+++ b/{file}\n@@ -1,2 +1,2 @@\n"
        " def total(items, tax_rate):\n"
        f"-{_BROKEN_LINE[4:]}\n"
        f"+{_GOLD_LINE[4:]}\n"
    )


def _candidate(plan: CandidatePlan) -> Candidate:
    file = _target_file(plan)
    return Candidate(
        language=plan.language,
        generator=plan.generator,
        target=CandidateTarget(files=(file,), symbols=("total",)),
        mutation_patch=_mutation_patch(file),
        oracle_patch=_oracle_patch(file),
        difficulty_hint="medium",
        provenance=_provenance(plan),
    )


def _env_image(plan: CandidatePlan) -> EnvImage:
    return EnvImage(
        repo_id=plan.repo.repo_id,
        language=plan.language,
        image_tag=f"swe-forge-env-{plan.repo.repo_id}:abc123",
        base_image="python:3.12-slim",
        commit="a" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command="python -m pytest -q",
        baseline_green=True,
        baseline_exit_code=0,
    )


def _spec(plan: CandidatePlan, *, problem: str = "") -> GeneratedSpec:
    return GeneratedSpec(
        problem_statement=problem or "total() must include tax in the returned amount.",
        requirements=["total() returns the taxed sum for the items"],
        interface_block="def total(items, tax_rate): ...",
        provenance=_provenance(plan),
    )


def _oracle_pass(plan: CandidatePlan) -> OracleReport:
    return OracleReport(
        language=plan.language,
        generator=plan.generator,
        verdict="pass",
        reasons=[],
        fail_to_pass=["python -m pytest tests/hidden/test_total.py"],
        pass_to_pass=["python -m pytest -q"],
        test_files=[
            OracleTestFile(
                path="tests/hidden/test_total.py",
                content=(
                    "from src.m import total\n\n\n"
                    "def test_total():\n    assert total([100], 0.1) == 110\n"
                ),
            )
        ],
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        provenance=_provenance(plan),
    )


def _oracle_reject(plan: CandidatePlan) -> OracleReport:
    return OracleReport(
        language=plan.language,
        generator=plan.generator,
        verdict="reject",
        reasons=["mutation_failed: induced reject"],
        provenance=_provenance(plan),
    )


def _calibration(plan: CandidatePlan, *, keep: bool) -> CalibrationReport:
    report = CalibrationReport(
        language=plan.language,
        models=[
            ModelSolveRecord(model="weak/m", tier="weak", k=4, solves=0, pass_at_k=0.0),
            ModelSolveRecord(model="mid/m", tier="mid", k=4, solves=1, pass_at_k=0.25),
            ModelSolveRecord(
                model="frontier/m", tier="frontier", k=4, solves=1, pass_at_k=0.25
            ),
        ],
        k=4,
        irt_difficulty=1.0,
        irt_discrimination=1.5,
    )
    report.set_band_verdict(
        "keep" if keep else "drop",
        "in-band borderline" if keep else "solve-all too easy",
    )
    return report


# --------------------------------------------------------------------------- #
# A deterministic offline CandidateProcessor: each plan is told where to exit.
# --------------------------------------------------------------------------- #
class FakeProcessor:
    """Returns scripted artifacts so the funnel/gate logic is exercised offline.

    ``outcomes`` maps ``plan.label`` to one of: ``env_failed``, ``synth_failed``,
    ``oracle_reject``, ``calib_drop``, ``kept`` (default), or ``leak`` (a kept
    candidate whose spec leaks a gold line so export refuses it). Usage is
    attributed exactly where the real stages spend it: teacher for every
    env-built candidate that reached spec/oracle, panel for every oracle-pass
    candidate that reached calibration. Host scratch lives under the
    orchestrator-owned ``workdir`` so teardown can be asserted via the run root.
    """

    TEACHER = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    PANEL = Usage(prompt_tokens=200, completion_tokens=80, total_tokens=280)

    def __init__(self, outcomes: dict[str, str] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.scratch_dirs: list[Path] = []

    async def process(self, plan: CandidatePlan, workdir: Path) -> CandidateArtifacts:
        stage = self.outcomes.get(plan.label, "kept")
        art = CandidateArtifacts(plan=plan)
        scratch = workdir / "checkout"
        scratch.mkdir(parents=True, exist_ok=True)
        self.scratch_dirs.append(scratch)

        if stage == "env_failed":
            art.failure_reason = "env build red"
            return art
        art.env_image = _env_image(plan)
        art.repo_url = plan.repo.url
        art.base_commit = "a" * 40

        if stage == "synth_failed":
            art.failure_reason = "generation failed"
            return art
        art.candidate = _candidate(plan)
        leaky = stage == "leak"
        art.spec = _spec(
            plan,
            problem=f"Implement total. Gold:\n{_GOLD_LINE}\n" if leaky else "",
        )
        art.teacher_usage = self.TEACHER
        art.teacher_cost = 0.001

        if stage == "oracle_reject":
            art.oracle_report = _oracle_reject(plan)
            return art
        art.oracle_report = _oracle_pass(plan)
        art.panel_usage = self.PANEL
        art.panel_cost = 0.002

        if stage == "calib_drop":
            art.calibration_report = _calibration(plan, keep=False)
            return art
        art.calibration_report = _calibration(plan, keep=True)
        return art


def _all_gold(tasks_out: Path | str, *, runs: int = 2) -> GoldEvalReport:
    """A fake gold-eval seam: every shipped task scores ``{"score": 1}``."""
    tasks_root = Path(tasks_out) / "tasks"
    results = [
        TaskGoldResult(
            task_id=task_dir.name,
            task_dir=task_dir,
            image="img",
            runs=[
                EvalRun(
                    task_id=task_dir.name,
                    run_index=i,
                    score=1,
                    phase1_passed=True,
                    exit_code=0,
                    container_name=f"c{i}",
                )
                for i in range(runs)
            ],
        )
        for task_dir in sorted(tasks_root.iterdir())
    ]
    return GoldEvalReport(tasks_dir=tasks_root, results=results)


def _mixed_plans() -> list[CandidatePlan]:
    """A representative cross-language, multi-generator candidate batch."""
    return [
        _plan("iniconfig", "ast_mutation", 0, "python"),
        _plan("iniconfig", "function_removal", 1, "python"),
        _plan("iniconfig", "lm_authored", 2, "python"),
        _plan("yocto", "ast_mutation", 0, "javascript"),
        _plan("yocto", "function_removal", 1, "javascript"),
        _plan("jwt", "ast_mutation", 0, "go"),
        _plan("jwt", "function_removal", 1, "go"),
    ]


# --------------------------------------------------------------------------- #
# StageCounts unit invariants (VAL-CROSS-013)
# --------------------------------------------------------------------------- #
def test_stage_counts_monotone_true() -> None:
    counts = StageCounts(
        sourced=10,
        env_built=8,
        synthesized=7,
        oracle_pass=5,
        calibration_keep=3,
        exported=3,
    )
    assert counts.monotone is True


@pytest.mark.parametrize(
    "counts",
    [
        StageCounts(sourced=3, env_built=5),  # env > sourced
        StageCounts(
            sourced=5, env_built=4, synthesized=4, oracle_pass=5
        ),  # pass > synth
        StageCounts(
            sourced=5,
            env_built=4,
            synthesized=4,
            oracle_pass=3,
            calibration_keep=3,
            exported=2,
        ),  # keep != exported
    ],
)
def test_stage_counts_monotone_false(counts: StageCounts) -> None:
    assert counts.monotone is False


# --------------------------------------------------------------------------- #
# Stage 0 sourcing (VAL-CROSS-005): many candidates, >=2 generators, 3 languages
# --------------------------------------------------------------------------- #
def test_build_pilot_plans_spans_languages_and_generators() -> None:
    plans = build_pilot_plans(seeds_per_cell=2)
    languages = {p.language for p in plans}
    assert languages == {"python", "javascript", "go"}
    # Each language carries the structural generators on its modular repos PLUS a
    # pr_mirror plan from its allowlist entries -> >=2 generators per language
    # (VAL-CROSS-005) and pr_mirror present everywhere.
    for language, generators in DEFAULT_GENERATORS_BY_LANGUAGE.items():
        cell = {p.generator for p in plans if p.language == language}
        assert set(generators) <= cell  # structural generators all present
        assert "pr_mirror" in cell
        assert len(cell) >= 2


def test_build_pilot_plans_emits_pr_mirror_for_allowlist_entries() -> None:
    # Every pr_mirror allowlist RepoSpec yields exactly one pr_mirror plan whose
    # params carry the upstream slug + pr_number (no secret), pinned to that
    # repo's own base_commit.
    registry = build_source_registry()
    allowlist = [s for s in registry.specs() if s.has_pr_mirror]
    assert len(allowlist) == 15

    plans = build_pilot_plans(registry=registry, seeds_per_cell=1)
    pr_plans = [p for p in plans if p.generator == "pr_mirror"]
    assert len(pr_plans) == len(allowlist)
    for plan in pr_plans:
        assert plan.params == {
            "repo": plan.repo.pr_repo,
            "pr_number": plan.repo.pr_number,
        }
        # The plan is pinned to the entry's own base_commit.
        assert plan.repo.commit == plan.repo.commit  # spec carries its SHA
        # No credential ever leaks into a plan payload.
        assert "token" not in {k.lower() for k in plan.params}
    # >=1 pr_mirror plan per language (the supply spans all three).
    assert {p.language for p in pr_plans} == {"python", "javascript", "go"}


def test_build_pilot_plans_emits_structural_only_on_modular_repos() -> None:
    registry = build_source_registry()
    modular = {s.repo_id for s in registry.specs() if s.structural_source}
    plans = build_pilot_plans(registry=registry, seeds_per_cell=2)
    structural = [p for p in plans if p.generator != "pr_mirror"]
    # Structural plans only ever target the modular (structural_source) repos.
    assert {p.repo.repo_id for p in structural} <= modular
    # The tiny single-module seeds get no structural plans (they ship 0).
    assert all(
        p.repo.repo_id not in {"pytest-dev/iniconfig", "sindresorhus/yocto-queue"}
        for p in structural
    )


def test_build_pilot_plans_toggle_families() -> None:
    only_pr = build_pilot_plans(seeds_per_cell=2, include_structural=False)
    assert only_pr and all(p.generator == "pr_mirror" for p in only_pr)
    only_struct = build_pilot_plans(seeds_per_cell=2, include_pr_mirror=False)
    assert only_struct and all(p.generator != "pr_mirror" for p in only_struct)


def test_build_pilot_plans_respects_language_filter_and_cap() -> None:
    only_go = build_pilot_plans(seeds_per_cell=3, languages=["go"])
    assert {p.language for p in only_go} == {"go"}
    capped = build_pilot_plans(seeds_per_cell=5, max_plans=4)
    assert len(capped) == 4


def test_pr_mirror_provided_tests_builds_isolated_f2p() -> None:
    """A pr_mirror candidate's recorded test files become provided F2P selectors.

    The recorded test file (left in place by the source-only mutation) is run via
    the adapter's selection-aware command with no files to write, so the pilot
    establishes the isolated F2P without perturbing the regression suite.
    """
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.pilot import _pr_mirror_provided_tests

    adapter = build_default_registry().get("python")
    candidate = Candidate(
        language="python",
        generator="pr_mirror",
        target=CandidateTarget(files=("src/validators/between.py",)),
        mutation_patch="x",
        oracle_patch="y",
        difficulty_hint="high",
        provenance=Provenance(
            generator="pr_mirror",
            seed=0,
            language="python",
            details={"test_files": ["tests/test_between.py"]},
        ),
    )
    # No curated flipping names -> fall back to running the recorded test FILE.
    repo = _repo("acme/x")
    tests = _pr_mirror_provided_tests(candidate, adapter, repo, "python -m pytest")
    assert len(tests) == 1
    assert tests[0].test_id == "python -m pytest tests/test_between.py"
    assert tests[0].files == ()
    assert tests[0].origin == "provided"


def test_pr_mirror_provided_tests_uses_repo_runner_for_flipping_names() -> None:
    """Curated flipping-test names drive a positive selection via the repo runner.

    When the RepoSpec records the F2P-flipping test names (the same ones the baked
    baseline excludes), the provided F2P narrows the repo's OWN baseline command
    positively -- here a JS/TS Mocha ``npm test -- --grep`` the ``node --test``
    standard runner could not produce -- so a non-standard-runner pr_mirror
    candidate can confirm its isolated F2P at all.
    """
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.pilot import _pr_mirror_provided_tests

    adapter = build_default_registry().get("javascript")
    candidate = Candidate(
        language="javascript",
        generator="pr_mirror",
        target=CandidateTarget(files=("src/lib/isISO8601.js",)),
        mutation_patch="x",
        oracle_patch="y",
        difficulty_hint="high",
        provenance=Provenance(
            generator="pr_mirror",
            seed=0,
            language="javascript",
            details={"test_files": ["test/validators.test.js"]},
        ),
    )
    repo = RepoSpec(
        repo_id="validatorjs/validator.js#2787",
        url="https://github.com/validatorjs/validator.js.git",
        commit="a" * 40,
        commit_date="2024-01-01T00:00:00+00:00",
        language="javascript",
        license="MIT",
        instance_cap=5,
        baseline_test="npm test",
        p2p_exclusions=("should validate ISO 8601 dates",),
    )
    tests = _pr_mirror_provided_tests(candidate, adapter, repo, repo.baseline_test)
    assert len(tests) == 1
    # The repo's Mocha runner, narrowed positively to the flipping title.
    assert tests[0].test_id == (
        "npm test -- --grep '(should\\ validate\\ ISO\\ 8601\\ dates)'"
    )
    assert "--invert" not in tests[0].test_id
    assert tests[0].files == ()


def test_pr_mirror_provided_tests_empty_for_non_pr_mirror() -> None:
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.pilot import _pr_mirror_provided_tests

    adapter = build_default_registry().get("python")
    candidate = Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("src/m.py",)),
        mutation_patch="x",
        oracle_patch="y",
        difficulty_hint="low",
        provenance=Provenance(
            generator="ast_mutation",
            seed=0,
            language="python",
            details={"test_files": ["tests/test_m.py"]},
        ),
    )
    # The helper only fires for pr_mirror; a structural candidate gets nothing.
    repo = _repo("acme/x")
    assert _pr_mirror_provided_tests(candidate, adapter, repo, "python -m pytest") == []


def test_default_pilot_config_wires_plans_and_overrides() -> None:
    config = default_pilot_config(
        "/tmp/pilot-x", seeds_per_cell=1, run_gold_eval=False, gold_eval_runs=3
    )
    assert config.plans
    assert config.run_gold_eval is False
    assert config.gold_eval_runs == 3


# --------------------------------------------------------------------------- #
# Difficulty-amplifier wiring (m6-pilot-difficulty task 1)
# --------------------------------------------------------------------------- #
def test_build_pilot_plans_emits_bug_combination_on_modular_repos() -> None:
    # The difficulty amplifier bug_combination is wired into every language's
    # structural generator set and runs ONLY on the diversified MODULAR repos.
    for generators in DEFAULT_GENERATORS_BY_LANGUAGE.values():
        assert "bug_combination" in generators

    registry = build_source_registry()
    modular = {s.repo_id for s in registry.specs() if s.structural_source}
    plans = build_pilot_plans(registry=registry, seeds_per_cell=2)
    amp = [p for p in plans if p.generator == "bug_combination"]
    assert amp, "expected bug_combination plans"
    # Amplifier plans only ever target the modular structural-source repos, and
    # span >=1 language so the candidate mix can reach the hard band.
    assert {p.repo.repo_id for p in amp} <= modular
    assert {p.language for p in amp} & {"python", "javascript", "go"}
    # The tiny single-module seeds never get an amplifier plan.
    assert all(p.repo.structural_source for p in amp)


# --------------------------------------------------------------------------- #
# Per-candidate P2P-exclusion wiring (m6-pilot-difficulty task 2)
# --------------------------------------------------------------------------- #
def test_with_p2p_exclusions_narrows_baseline_and_records_provenance() -> None:
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.oracle.p2p_derive import compute_collateral_exclusions
    from swe_forge.forge.pilot import _with_p2p_exclusions

    adapter = build_default_registry().get("python")
    env = _env_image(_plan("boltons", "function_removal", 0))
    derivation = compute_collateral_exclusions(["slugify", "test_slugify"])
    derived = _with_p2p_exclusions(env, adapter, derivation)

    # Same image (same Docker container), narrowed baseline command excludes the
    # collateral via the adapter's -k 'not (...)' filter, baseline still green.
    assert derived.image_tag == env.image_tag
    assert derived.baseline_test_command.startswith(env.baseline_test_command)
    assert "not (slugify or test_slugify)" in derived.baseline_test_command
    assert derived.baseline_green is True
    # The derivation is recorded on the derived image provenance for audit.
    assert derived.provenance["per_candidate_p2p_exclusions"]["exclusions"] == [
        "slugify",
        "test_slugify",
    ]


def test_apply_structural_p2p_exclusions_bakes_derived_exclusions(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # The LiveCandidateProcessor derives a structural candidate's collateral and
    # bakes it into the EnvImage baseline (Docker derivation stubbed offline).
    import swe_forge.forge.pilot as pilot_mod
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.oracle.p2p_derive import P2PDerivation

    plan = _plan("boltons", "function_removal", 0)
    candidate = _candidate(plan)
    env = _env_image(plan)
    adapter = build_default_registry().get("python")
    art = CandidateArtifacts(plan=plan)

    derivation = P2PDerivation(exclusions=("slugify",), broken_failures=("slugify",))

    async def _fake_derive(*args: object, **kw: object) -> P2PDerivation:
        return derivation

    monkeypatch.setattr(pilot_mod, "derive_structural_p2p_exclusions", _fake_derive)
    processor = pilot_mod.LiveCandidateProcessor(panel=[], validate_models=False)
    derived = asyncio.run(
        processor._apply_structural_p2p_exclusions(
            candidate, env, adapter, art, _structural_protection()
        )
    )
    assert "not (slugify)" in derived.baseline_test_command
    assert art.p2p_derivation is derivation


def test_apply_structural_p2p_exclusions_threads_f2p_protection(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """Live derivation receives every previously discovered F2P path and name."""
    import swe_forge.forge.pilot as pilot_mod
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile
    from swe_forge.forge.oracle.p2p_derive import P2PDerivation
    from swe_forge.forge.pilot import StructuralF2PProtection

    plan = _plan("boltons", "function_removal", 0)
    candidate = _candidate(plan)
    env = _env_image(plan)
    adapter = build_default_registry().get("python")
    art = CandidateArtifacts(plan=plan)
    protection = StructuralF2PProtection(
        tests=(
            HiddenTest(
                test_id="python -m pytest tests/hidden/test_slugify.py",
                files=(
                    HiddenTestFile(
                        path="tests/hidden/test_slugify.py",
                        content="def test_slugify(): ...",
                    ),
                ),
                origin="provided",
            ),
        ),
        protected_names=("test_slugify",),
        protected_files=("tests/hidden/test_slugify.py",),
    )
    captured: dict[str, object] = {}

    async def _fake_derive(*args: object, **kw: object) -> P2PDerivation:
        captured.update(kw)
        return P2PDerivation(exclusions=("collateral",))

    monkeypatch.setattr(pilot_mod, "derive_structural_p2p_exclusions", _fake_derive)
    processor = pilot_mod.LiveCandidateProcessor(panel=[], validate_models=False)
    derived = asyncio.run(
        processor._apply_structural_p2p_exclusions(
            candidate, env, adapter, art, protection
        )
    )

    assert captured["protected_names"] == ("test_slugify",)
    assert captured["protected_files"] == ("tests/hidden/test_slugify.py",)
    assert "not (collateral)" in derived.baseline_test_command
    assert derived.provenance["per_candidate_p2p_exclusions"]["details"][
        "structural_f2p_protection"
    ]["protected_files"] == ["tests/hidden/test_slugify.py"]


def test_apply_structural_p2p_exclusions_rejects_protected_import_conflict(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """A protected F2P import error never lets the live path weaken P2P."""
    import swe_forge.forge.pilot as pilot_mod
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile
    from swe_forge.forge.oracle.p2p_derive import P2PDerivation
    from swe_forge.forge.pilot import StructuralF2PProtection

    plan = _plan("boltons", "function_removal", 0)
    candidate = _candidate(plan)
    env = _env_image(plan)
    adapter = build_default_registry().get("python")
    art = CandidateArtifacts(plan=plan)
    protection = StructuralF2PProtection(
        tests=(
            HiddenTest(
                test_id="python -m pytest tests/test_f2p.py",
                files=(HiddenTestFile(path="tests/test_f2p.py", content="x"),),
                origin="provided",
            ),
        ),
        protected_names=("test_f2p",),
        protected_files=("tests/test_f2p.py",),
    )

    async def _fake_derive(*args: object, **kw: object) -> P2PDerivation:
        assert kw["protected_files"] == protection.protected_files
        return P2PDerivation(
            file_exclusions=("tests/test_non_f2p.py",),
            protected_file_conflicts=("tests/test_f2p.py",),
            p2p_green_on_broken=False,
        )

    monkeypatch.setattr(pilot_mod, "derive_structural_p2p_exclusions", _fake_derive)
    processor = pilot_mod.LiveCandidateProcessor(panel=[], validate_models=False)
    derived = asyncio.run(
        processor._apply_structural_p2p_exclusions(
            candidate, env, adapter, art, protection
        )
    )

    assert derived == env
    assert art.p2p_derivation is not None
    assert art.p2p_derivation.has_protected_conflict is True


def test_structural_f2p_protection_rejects_unparseable_or_unowned_metadata() -> None:
    """Preflight only protects parser-proven F2P identities owned by its proposal."""
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.pilot import (
        _StructuralF2PObservation,
        _build_structural_f2p_protection,
    )

    adapter = build_default_registry().get("python")
    f2p = HiddenTest(
        test_id="python -m pytest tests/hidden/test_f2p.py",
        files=(HiddenTestFile(path="tests/hidden/test_f2p.py", content="x"),),
    )

    protection = _build_structural_f2p_protection(
        [
            _StructuralF2PObservation(
                test=f2p,
                failed_on_broken=True,
                stdout="FAILED tests/hidden/test_f2p.py::test_f2p\n",
                stderr="",
            )
        ],
        adapter,
    )
    assert protection.protected_names == ("test_f2p",)
    assert protection.protected_files == ("tests/hidden/test_f2p.py",)
    assert protection.tests[0].origin == "provided"

    with pytest.raises(StructuralF2PProtectionError, match="unparseable"):
        _build_structural_f2p_protection(
            [
                _StructuralF2PObservation(
                    test=f2p,
                    failed_on_broken=True,
                    stdout="truncated runner output",
                    stderr="",
                )
            ],
            adapter,
        )

    foreign_collection = (
        "=== ERRORS ===\nERROR tests/test_non_f2p.py\n=== short test summary info ===\n"
    )
    with pytest.raises(StructuralF2PProtectionError, match="not owned"):
        _build_structural_f2p_protection(
            [
                _StructuralF2PObservation(
                    test=f2p,
                    failed_on_broken=True,
                    stdout=foreign_collection,
                    stderr="",
                )
            ],
            adapter,
        )


def test_process_proposes_structural_f2p_before_derivation_and_reuses_it(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """The same structural F2P proposal protects derivation and is established."""
    import swe_forge.forge.pilot as pilot_mod
    from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile
    from swe_forge.forge.pilot import StructuralF2PProtection

    plan = _plan("boltons", "ast_mutation", 0)
    candidate = _candidate(plan)
    env = _env_image(plan)
    protection = StructuralF2PProtection(
        tests=(
            HiddenTest(
                test_id="python -m pytest tests/hidden/test_total.py",
                files=(
                    HiddenTestFile(
                        path="tests/hidden/test_total.py",
                        content="def test_total(): ...",
                    ),
                ),
                origin="provided",
            ),
        ),
        protected_names=("test_total",),
        protected_files=("tests/hidden/test_total.py",),
    )
    calls: list[str] = []
    captured: dict[str, object] = {}

    class _Generator:
        def generate(self, request: object, adapter: object) -> Candidate:
            return candidate

    class _Generators:
        def get(self, name: str) -> _Generator:
            assert name == candidate.generator
            return _Generator()

    processor = pilot_mod.LiveCandidateProcessor(panel=[], validate_models=False)
    processor._generators = _Generators()  # type: ignore[assignment]

    monkeypatch.setattr(pilot_mod, "_checkout_repo", lambda repo, dest: None)
    monkeypatch.setattr(pilot_mod, "_author_spec", lambda *args: _spec(plan))

    async def _fake_env(plan_arg: object, checkout: object) -> tuple[EnvImage, str]:
        return env, ""

    async def _fake_propose(
        candidate_arg: Candidate, env_arg: EnvImage, adapter_arg: object
    ) -> StructuralF2PProtection:
        assert candidate_arg is candidate
        assert env_arg is env
        calls.append("propose")
        return protection

    async def _fake_derive(
        candidate_arg: Candidate,
        env_arg: EnvImage,
        adapter_arg: object,
        art: CandidateArtifacts,
        received: StructuralF2PProtection,
    ) -> EnvImage:
        assert received is protection
        calls.append("derive")
        return env_arg

    async def _fake_oracle(*args: object, **kw: object) -> OracleReport:
        calls.append("establish")
        captured["provided_tests"] = kw["provided_tests"]
        captured["establish_synthesizer"] = kw["establish_synthesizer"]
        return _oracle_reject(plan)

    monkeypatch.setattr(processor, "_acquire_env_image", _fake_env)
    monkeypatch.setattr(processor, "_propose_structural_f2p", _fake_propose)
    monkeypatch.setattr(processor, "_apply_structural_p2p_exclusions", _fake_derive)
    monkeypatch.setattr(pilot_mod, "run_oracle_pipeline", _fake_oracle)

    art = asyncio.run(processor._process(plan, tmp_path, CandidateArtifacts(plan=plan)))

    assert calls == ["propose", "derive", "establish"]
    assert captured["provided_tests"] == list(protection.tests)
    assert captured["establish_synthesizer"] is None
    assert art.oracle_report is not None
    assert art.oracle_report.is_pass is False


def test_process_rejects_ambiguous_structural_f2p_before_derivation(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """Unknown protection metadata drops the candidate rather than weakening P2P."""
    import swe_forge.forge.pilot as pilot_mod
    from swe_forge.forge.pilot import StructuralF2PProtectionError

    plan = _plan("boltons", "ast_mutation", 0)
    candidate = _candidate(plan)
    env = _env_image(plan)
    derived = False

    class _Generator:
        def generate(self, request: object, adapter: object) -> Candidate:
            return candidate

    class _Generators:
        def get(self, name: str) -> _Generator:
            return _Generator()

    processor = pilot_mod.LiveCandidateProcessor(panel=[], validate_models=False)
    processor._generators = _Generators()  # type: ignore[assignment]
    monkeypatch.setattr(pilot_mod, "_checkout_repo", lambda repo, dest: None)
    monkeypatch.setattr(pilot_mod, "_author_spec", lambda *args: _spec(plan))

    async def _fake_env(plan_arg: object, checkout: object) -> tuple[EnvImage, str]:
        return env, ""

    async def _ambiguous(
        candidate_arg: Candidate, env_arg: EnvImage, adapter_arg: object
    ) -> object:
        raise StructuralF2PProtectionError("no unambiguous F2P paths")

    async def _should_not_derive(*args: object, **kw: object) -> EnvImage:
        nonlocal derived
        derived = True
        return env

    monkeypatch.setattr(processor, "_acquire_env_image", _fake_env)
    monkeypatch.setattr(processor, "_propose_structural_f2p", _ambiguous)
    monkeypatch.setattr(
        processor, "_apply_structural_p2p_exclusions", _should_not_derive
    )

    art = asyncio.run(processor.process(plan, tmp_path))

    assert derived is False
    assert art.oracle_report is None
    assert (
        "StructuralF2PProtectionError: no unambiguous F2P paths" in art.failure_reason
    )


def test_apply_structural_p2p_exclusions_noop_when_no_collateral(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import swe_forge.forge.pilot as pilot_mod
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.oracle.p2p_derive import P2PDerivation

    plan = _plan("boltons", "bug_combination", 0)
    candidate = _candidate(plan)
    env = _env_image(plan)
    adapter = build_default_registry().get("python")
    art = CandidateArtifacts(plan=plan)

    async def _fake_derive(*args: object, **kw: object) -> P2PDerivation:
        return P2PDerivation(exclusions=(), p2p_green_on_broken=True)

    monkeypatch.setattr(pilot_mod, "derive_structural_p2p_exclusions", _fake_derive)
    processor = pilot_mod.LiveCandidateProcessor(panel=[], validate_models=False)
    derived = asyncio.run(
        processor._apply_structural_p2p_exclusions(
            candidate, env, adapter, art, _structural_protection()
        )
    )
    # No collateral -> the baseline command is left untouched (no loosening).
    assert derived.baseline_test_command == env.baseline_test_command


def test_apply_structural_p2p_exclusions_best_effort_on_docker_error(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    import swe_forge.forge.pilot as pilot_mod
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.oracle.establish import EstablishError

    plan = _plan("boltons", "function_removal", 0)
    candidate = _candidate(plan)
    env = _env_image(plan)
    adapter = build_default_registry().get("python")
    art = CandidateArtifacts(plan=plan)

    async def _boom(*args: object, **kw: object) -> object:
        raise EstablishError("docker hiccup")

    monkeypatch.setattr(pilot_mod, "derive_structural_p2p_exclusions", _boom)
    processor = pilot_mod.LiveCandidateProcessor(panel=[], validate_models=False)
    derived = asyncio.run(
        processor._apply_structural_p2p_exclusions(
            candidate, env, adapter, art, _structural_protection()
        )
    )
    # A derivation failure never weakens a gate: the baseline is untouched and the
    # establish gate will reject p2p_not_green_on_broken if the collateral is real.
    assert derived.baseline_test_command == env.baseline_test_command


def test_process_swallows_unexpected_candidate_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A single candidate's UNEXPECTED error (e.g. a mutation-tool misconfig raising
    # deep in a gate) must be recorded as that candidate's failure, NOT propagated
    # to abort the whole sweep. The candidate is dropped (no oracle pass), never
    # vacuously passed.
    import swe_forge.forge.pilot as pilot_mod

    plan = _plan("cast", "bug_combination", 1)
    processor = pilot_mod.LiveCandidateProcessor(panel=[], validate_models=False)

    async def _boom(_plan: object, _work: object, _art: object) -> object:
        raise RuntimeError("go-mutesting baseline is not green")

    monkeypatch.setattr(processor, "_process", _boom)
    art = asyncio.run(processor.process(plan, Path("/tmp")))
    assert art.candidate is None
    assert art.oracle_report is None
    assert "RuntimeError: go-mutesting baseline is not green" in art.failure_reason


# --------------------------------------------------------------------------- #
# Single-discriminating-assertion pr_mirror F2P preference (task 3)
# --------------------------------------------------------------------------- #
def test_pr_mirror_provided_tests_prefers_single_assertion_f2p() -> None:
    from swe_forge.forge.adapters import build_default_registry
    from swe_forge.forge.pilot import _pr_mirror_provided_tests

    adapter = build_default_registry().get("python")
    candidate = Candidate(
        language="python",
        generator="pr_mirror",
        target=CandidateTarget(files=("src/validators/url.py",)),
        mutation_patch="x",
        oracle_patch="y",
        difficulty_hint="high",
        provenance=Provenance(generator="pr_mirror", seed=0, language="python"),
    )
    # The repo curates BOTH the whole flipping test (excluded from P2P) and a
    # single discriminating assertion (preferred as the F2P).
    repo = RepoSpec(
        repo_id="python-validators/validators#305",
        url="https://github.com/python-validators/validators.git",
        commit="a" * 40,
        commit_date="2024-01-01T00:00:00+00:00",
        language="python",
        license="MIT",
        instance_cap=2,
        p2p_exclusions=("test_returns_true_on_valid_url",),
        pr_f2p_names=("test_returns_true_on_valid_url_matrix_fragment",),
        pr_repo="python-validators/validators",
        pr_number=305,
        pr_generator="pr_mirror",
    )
    tests = _pr_mirror_provided_tests(candidate, adapter, repo, "python -m pytest")
    assert len(tests) == 1
    # The narrow single-assertion name is selected, NOT the whole flipping test.
    assert "test_returns_true_on_valid_url_matrix_fragment" in tests[0].test_id
    assert "test_returns_true_on_valid_url)" not in tests[0].test_id


# --------------------------------------------------------------------------- #
# The full offline funnel (VAL-CROSS-012/013/014/016/017/022)
# --------------------------------------------------------------------------- #
def _run(config: PilotConfig, processor: FakeProcessor, **kw: object):
    return asyncio.run(run_pilot(config, processor=processor, **kw))  # type: ignore[arg-type]


def test_pilot_funnel_monotone_and_reconciled(tmp_path: Path) -> None:
    plans = _mixed_plans()
    outcome = _run(PilotConfig(plans=plans, out_dir=tmp_path), FakeProcessor())

    counts = outcome.counts
    assert counts.sourced == len(plans)
    assert counts.monotone is True
    assert counts.calibration_keep == counts.exported == outcome.shipped_count

    # tasks/*/ == jsonl == parquet == report kept (VAL-CROSS-014/016).
    task_dirs = {p.name for p in (tmp_path / "tasks").iterdir()}
    jsonl = import_jsonl(tmp_path / "dataset.jsonl")
    parquet = import_parquet(tmp_path / "dataset.parquet")
    assert len(task_dirs) == len(jsonl) == len(parquet) == counts.exported
    assert {t.id for t in jsonl} == {r["id"] for r in parquet} == task_dirs
    assert outcome.report is not None
    assert outcome.report.counts.reconciled is True
    assert outcome.report.shipped_count == counts.exported


def test_pilot_ships_across_all_three_languages_with_two_generators(
    tmp_path: Path,
) -> None:
    outcome = _run(PilotConfig(plans=_mixed_plans(), out_dir=tmp_path), FakeProcessor())
    assert set(outcome.languages_shipped) == {"python", "javascript", "go"}
    assert len(outcome.generators_used) >= 2


def test_rejections_never_propagate(tmp_path: Path) -> None:
    plans = _mixed_plans()
    outcomes = {
        plans[0].label: "env_failed",
        plans[1].label: "synth_failed",
        plans[2].label: "oracle_reject",
        plans[3].label: "calib_drop",
        # plans[4..6] keep
    }
    processor = FakeProcessor(outcomes)
    outcome = _run(PilotConfig(plans=plans, out_dir=tmp_path), processor)

    counts = outcome.counts
    assert counts.sourced == 7
    assert counts.env_built == 6  # env_failed dropped
    assert counts.synthesized == 5  # synth_failed dropped
    assert counts.oracle_pass == 4  # oracle_reject dropped
    assert counts.calibration_keep == counts.exported == 3  # calib_drop dropped
    assert counts.monotone is True

    # None of the dropped candidates left a workspace or a dataset row.
    dropped_stages = {"env_failed", "synth_failed", "oracle_reject", "calib_drop"}
    dropped = [d for d in outcome.dispositions if d.stage in dropped_stages]
    assert len(dropped) == 4
    shipped_ids = {p.name for p in (tmp_path / "tasks").iterdir()}
    assert len(shipped_ids) == 3
    kept_labels = {d.plan.label for d in outcome.dispositions if d.stage == "kept"}
    assert kept_labels == {plans[4].label, plans[5].label, plans[6].label}


def test_only_pass_and_keep_become_forge_tasks_matches_export_gate(
    tmp_path: Path,
) -> None:
    # VAL-CROSS-020: the pilot's keep/drop decision agrees with the canonical
    # per-stage export gate (assemble refuses exactly what the pilot drops).
    keep_plan = _plan("iniconfig", "ast_mutation", 0)
    reject_plan = _plan("iniconfig", "function_removal", 1)
    drop_plan = _plan("iniconfig", "lm_authored", 2)

    # kept -> assemble succeeds.
    assemble_forge_task(
        candidate=_candidate(keep_plan),
        spec=_spec(keep_plan),
        oracle_report=_oracle_pass(keep_plan),
        calibration_report=_calibration(keep_plan, keep=True),
        env_image=_env_image(keep_plan),
        repo_url=keep_plan.repo.url,
    )
    # oracle-reject -> refused.
    with pytest.raises(ExportRefusedError):
        assemble_forge_task(
            candidate=_candidate(reject_plan),
            spec=_spec(reject_plan),
            oracle_report=_oracle_reject(reject_plan),
            calibration_report=_calibration(reject_plan, keep=True),
            env_image=_env_image(reject_plan),
            repo_url=reject_plan.repo.url,
        )
    # band-drop -> refused.
    with pytest.raises(ExportRefusedError):
        assemble_forge_task(
            candidate=_candidate(drop_plan),
            spec=_spec(drop_plan),
            oracle_report=_oracle_pass(drop_plan),
            calibration_report=_calibration(drop_plan, keep=False),
            env_image=_env_image(drop_plan),
            repo_url=drop_plan.repo.url,
        )


def test_usage_and_cost_surfaced(tmp_path: Path) -> None:
    plans = _mixed_plans()
    outcomes = {plans[0].label: "env_failed", plans[1].label: "synth_failed"}
    outcome = _run(PilotConfig(plans=plans, out_dir=tmp_path), FakeProcessor(outcomes))

    # Teacher spent on every candidate that reached spec (synthesized); panel on
    # every oracle-pass candidate that reached calibration.
    synthesized = outcome.counts.synthesized
    oracle_pass = outcome.counts.oracle_pass
    assert (
        outcome.usage.teacher.total_tokens
        == synthesized * FakeProcessor.TEACHER.total_tokens
    )
    assert (
        outcome.usage.panel.total_tokens
        == oracle_pass * FakeProcessor.PANEL.total_tokens
    )
    assert outcome.usage.total_cost > 0.0
    assert outcome.usage.total_tokens == (
        outcome.usage.teacher.total_tokens + outcome.usage.panel.total_tokens
    )
    surfaced = outcome.to_dict()["usage"]
    assert surfaced["total_tokens"] == outcome.usage.total_tokens  # type: ignore[index]


def test_provenance_present_and_agrees_with_report(tmp_path: Path) -> None:
    outcome = _run(
        PilotConfig(plans=_mixed_plans(), out_dir=tmp_path),
        FakeProcessor(),
        gold_eval_fn=_all_gold,
    )
    assert outcome.report is not None
    report = outcome.report
    # Every shipped task carries provenance; the report audits pass.
    assert len(report.provenances) == outcome.shipped_count
    assert report.completeness.passed is True
    assert report.consistency.passed is True
    # Generator/language breakdown reconciles to the shipped count.
    assert sum(report.generator_breakdown.values()) == outcome.shipped_count
    assert sum(report.language_breakdown.values()) == outcome.shipped_count


def test_pilot_ok_when_headlines_hold(tmp_path: Path) -> None:
    outcome = _run(
        PilotConfig(plans=_mixed_plans(), out_dir=tmp_path),
        FakeProcessor(),
        gold_eval_fn=_all_gold,
    )
    assert outcome.report is not None
    assert outcome.headline_a_pass is True  # gold = 100%
    assert outcome.headline_b_pass is True  # frontier 0.25 in (0, 0.30)
    assert outcome.report.passed is True
    assert len(outcome.generators_used) >= 2
    assert outcome.ok is True


# --------------------------------------------------------------------------- #
# Candidate-level concurrency: parallel processing preserves order + invariants
# --------------------------------------------------------------------------- #
def test_candidate_concurrency_preserves_order_and_funnel(tmp_path: Path) -> None:
    plans = _mixed_plans()
    serial = _run(PilotConfig(plans=plans, out_dir=tmp_path / "s"), FakeProcessor())
    parallel = _run(
        PilotConfig(plans=plans, out_dir=tmp_path / "p", candidate_concurrency=4),
        FakeProcessor(),
    )
    # Concurrency must not change the funnel, the shipped set, or disposition order.
    assert serial.counts.to_dict() == parallel.counts.to_dict()
    assert parallel.counts.monotone is True
    assert [d.plan.label for d in serial.dispositions] == [
        d.plan.label for d in parallel.dispositions
    ]
    ids_s = {p.name for p in (tmp_path / "s" / "tasks").iterdir()}
    ids_p = {p.name for p in (tmp_path / "p" / "tasks").iterdir()}
    assert ids_s == ids_p


def test_candidate_concurrency_actually_overlaps(tmp_path: Path) -> None:
    """A semaphore of N lets up to N candidates be in-flight simultaneously."""

    in_flight = 0
    peak = 0

    class SlowProcessor(FakeProcessor):
        async def process(
            self, plan: CandidatePlan, workdir: Path
        ) -> CandidateArtifacts:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.05)
                return await super().process(plan, workdir)
            finally:
                in_flight -= 1

    _run(
        PilotConfig(plans=_mixed_plans(), out_dir=tmp_path, candidate_concurrency=4),
        SlowProcessor(),
    )
    assert peak >= 2  # genuine overlap occurred under the semaphore


# --------------------------------------------------------------------------- #
# Reproducibility (VAL-CROSS-016): a re-run ships the equivalent set
# --------------------------------------------------------------------------- #
def test_reexport_reproduces_equivalent_shipped_set(tmp_path: Path) -> None:
    plans = _mixed_plans()
    first = _run(PilotConfig(plans=plans, out_dir=tmp_path / "a"), FakeProcessor())
    second = _run(PilotConfig(plans=plans, out_dir=tmp_path / "b"), FakeProcessor())

    ids_a = {p.name for p in (tmp_path / "a" / "tasks").iterdir()}
    ids_b = {p.name for p in (tmp_path / "b" / "tasks").iterdir()}
    assert ids_a == ids_b
    assert first.shipped_count == second.shipped_count == len(ids_a)
    assert len(import_jsonl(tmp_path / "a" / "dataset.jsonl")) == first.shipped_count
    assert len(import_jsonl(tmp_path / "b" / "dataset.jsonl")) == second.shipped_count


# --------------------------------------------------------------------------- #
# A planted leak never ships and the run does not silently pass (VAL-CROSS-015)
# --------------------------------------------------------------------------- #
def test_planted_leak_never_ships_and_run_does_not_pass(tmp_path: Path) -> None:
    plans = _mixed_plans()
    leak_plan = plans[0]
    outcome = _run(
        PilotConfig(plans=plans, out_dir=tmp_path),
        FakeProcessor({leak_plan.label: "leak"}),
        gold_eval_fn=_all_gold,
    )
    # The leaky candidate was a band-keep (counted) but export refused it, so the
    # funnel no longer reconciles -> the run surfaces the problem (not silent).
    assert outcome.counts.exported < outcome.counts.calibration_keep
    assert outcome.counts.monotone is False
    assert outcome.ok is False
    # The leaky task left no workspace.
    leak_file = _target_file(leak_plan)
    leak_id_fragment = Path(leak_file).stem
    shipped = {p.name for p in (tmp_path / "tasks").iterdir()}
    assert all(leak_id_fragment not in name for name in shipped)


# --------------------------------------------------------------------------- #
# Teardown (VAL-CROSS-018): host scratch is always cleaned up
# --------------------------------------------------------------------------- #
def test_host_workdirs_torn_down_after_run(tmp_path: Path) -> None:
    processor = FakeProcessor()
    _run(PilotConfig(plans=_mixed_plans(), out_dir=tmp_path), processor)
    assert processor.scratch_dirs  # candidates created host scratch
    assert all(not d.exists() for d in processor.scratch_dirs)


def test_host_workdirs_torn_down_even_when_processor_raises(tmp_path: Path) -> None:
    scratch: list[Path] = []

    class Boom(FakeProcessor):
        async def process(
            self, plan: CandidatePlan, workdir: Path
        ) -> CandidateArtifacts:
            art = await super().process(plan, workdir)
            scratch.extend(self.scratch_dirs)
            if plan.generator == "function_removal":
                raise RuntimeError("induced mid-pipeline failure")
            return art

    with pytest.raises(RuntimeError):
        _run(PilotConfig(plans=_mixed_plans(), out_dir=tmp_path), Boom())

    # The orchestrator owns the run-scoped root, so every host dir handed out
    # (including the one in flight when the stage raised) is still torn down.
    assert scratch
    assert all(not d.exists() for d in scratch)


# --------------------------------------------------------------------------- #
# Incremental checkpoint export (m6-pilot-checkpoint-export)
#
# The pilot now materializes each band-keep the instant it is found (workspace +
# jsonl/parquet row + provenance) via the reused Stage-5 export path, so a run
# stopped at any point has already shipped every keep found so far -- instead of
# the old all-or-nothing export that ran only after EVERY plan completed.
# --------------------------------------------------------------------------- #
from datetime import datetime, timezone  # noqa: E402

from swe_forge.forge.checkpoint import PilotCheckpoint  # noqa: E402
from swe_forge.forge.export import export_batch  # noqa: E402
from swe_forge.forge.pilot import _keep_export_request  # noqa: E402


def _task_dirs(out_dir: Path) -> list[Path]:
    tasks = out_dir / "tasks"
    return [p for p in tasks.iterdir() if p.is_dir()] if tasks.exists() else []


def _freeze_time(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Freeze both timestamp sources so two exports are byte-comparable.

    ``forge.models._utc_now_iso`` feeds ForgeTask/Provenance/meta timestamps;
    ``SweTask.created_at`` is set from ``datetime.now`` inside the dataset record
    conversion. Freezing both makes the shipped artifact a pure function of the
    (deterministic) FakeProcessor inputs.
    """
    monkeypatch.setattr("swe_forge.forge.models._utc_now_iso", lambda: _TS)

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr("swe_forge.swe.models.datetime", _Frozen)


def test_checkpoint_ships_each_keep_incrementally(tmp_path: Path) -> None:
    """Each recorded keep immediately grows tasks/ + jsonl + parquet by exactly 1.

    Proves the checkpoint materializes the workspace AND both dataset rows AND
    provenance per keep (not deferred to a final all-at-once pass).
    """
    plans = _mixed_plans()  # all keeps under the default FakeProcessor
    checkpoint = PilotCheckpoint(tmp_path, overwrite=True)

    # A stop before the first keep still leaves valid empty artifacts.
    assert (tmp_path / "dataset.jsonl").read_text() == ""
    assert import_parquet(tmp_path / "dataset.parquet") == []

    async def _drive() -> None:
        processor = FakeProcessor()
        for i, plan in enumerate(plans):
            workdir = tmp_path / "run" / f"c{i}"
            workdir.mkdir(parents=True)
            art = await processor.process(plan, workdir)
            request = _keep_export_request(art)
            assert request is not None
            result = await checkpoint.record_keep(i, request)
            assert result.status == "shipped"
            shipped = i + 1
            assert len(_task_dirs(tmp_path)) == shipped
            assert len(import_jsonl(tmp_path / "dataset.jsonl")) == shipped
            assert len(import_parquet(tmp_path / "dataset.parquet")) == shipped

    asyncio.run(_drive())
    # Every materialized keep carries provenance.
    for task_dir in _task_dirs(tmp_path):
        assert (task_dir / "provenance.json").is_file()


def test_pilot_stopped_after_k_keeps_ships_exactly_k(tmp_path: Path) -> None:
    """A sweep stopped after the k-th keep has already shipped exactly those k.

    The processor raises when it reaches the (k+1)-th plan (sequential order); the
    k keeps found before it are already materialized on disk (workspace + both
    dataset rows + provenance) and the on-disk funnel reconciles (keep==exported).
    """
    plans = _mixed_plans()
    k = 3
    stop_label = plans[k].label

    class StopAfterK(FakeProcessor):
        async def process(
            self, plan: CandidatePlan, workdir: Path
        ) -> CandidateArtifacts:
            if plan.label == stop_label:
                raise RuntimeError("budget/time ceiling reached")
            return await super().process(plan, workdir)

    with pytest.raises(RuntimeError):
        _run(
            PilotConfig(plans=plans, out_dir=tmp_path, candidate_concurrency=1),
            StopAfterK(),
        )

    task_dirs = _task_dirs(tmp_path)
    jsonl = import_jsonl(tmp_path / "dataset.jsonl")
    parquet = import_parquet(tmp_path / "dataset.parquet")
    # Exactly k keeps shipped -- not 0 (the old all-or-nothing failure mode).
    assert len(task_dirs) == k
    # Funnel reconciles under interruption: keep == exported across all artifacts.
    assert len(jsonl) == len(parquet) == k
    assert (
        {t.id for t in jsonl}
        == {r["id"] for r in parquet}
        == {d.name for d in task_dirs}
    )
    # Provenance is complete for each shipped keep.
    for task_dir in task_dirs:
        assert (task_dir / "provenance.json").is_file()


def test_completed_run_dataset_byte_identical_to_all_at_once(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """A completed incremental run's dataset == the pre-change one-shot export.

    Checkpointing changes only WHEN a keep is written, never WHAT ships: the
    finalized datasets are regenerated from the full kept set in plan order, so
    they are byte-for-byte identical to a single ``export_batch`` of the same set.
    """
    _freeze_time(monkeypatch)
    plans = _mixed_plans()

    # Reference: the old all-at-once path over the identical kept set (plan order).
    async def _collect_requests():  # type: ignore[no-untyped-def]
        processor = FakeProcessor()
        requests = []
        for i, plan in enumerate(plans):
            workdir = tmp_path / "ref-run" / f"c{i}"
            workdir.mkdir(parents=True)
            art = await processor.process(plan, workdir)
            request = _keep_export_request(art)
            if request is not None:
                requests.append(request)
        return requests

    ref_dir = tmp_path / "ref"
    export_batch(asyncio.run(_collect_requests()), ref_dir, overwrite=True)

    # The new incremental pilot.
    pilot_dir = tmp_path / "pilot"
    _run(PilotConfig(plans=plans, out_dir=pilot_dir), FakeProcessor())

    assert (pilot_dir / "dataset.jsonl").read_bytes() == (
        ref_dir / "dataset.jsonl"
    ).read_bytes()
    assert (pilot_dir / "dataset.parquet").read_bytes() == (
        ref_dir / "dataset.parquet"
    ).read_bytes()


def test_mid_write_kill_leaves_valid_dataset(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A crash DURING a dataset write never corrupts the already-shipped dataset.

    The first keep is shipped; the parquet half of the next atomic write is then
    forced to fail. Temp-then-rename means the failure removes the temp files and
    leaves BOTH previously-shipped datasets untouched, valid, and consistent.
    """
    plans = _mixed_plans()
    checkpoint = PilotCheckpoint(tmp_path, overwrite=True)

    async def _drive() -> None:
        processor = FakeProcessor()
        first = await processor.process(plans[0], tmp_path / "w0")
        await checkpoint.record_keep(0, _keep_export_request(first))  # type: ignore[arg-type]

        def _boom(*args: object, **kw: object) -> int:
            raise OSError("disk full mid-write")

        monkeypatch.setattr("swe_forge.export.parquet.export_parquet", _boom)
        second = await processor.process(plans[3], tmp_path / "w1")
        with pytest.raises(OSError):
            await checkpoint.record_keep(3, _keep_export_request(second))  # type: ignore[arg-type]

    asyncio.run(_drive())

    # Both datasets are still parseable (not corrupt) and reflect the first keep.
    jsonl = import_jsonl(tmp_path / "dataset.jsonl")
    parquet = import_parquet(tmp_path / "dataset.parquet")
    assert len(jsonl) == len(parquet) == 1
    # No temp scratch left behind after the failed atomic write.
    assert list(tmp_path.glob("dataset.jsonl.ckpt-*")) == []
    assert list(tmp_path.glob("dataset.parquet.ckpt-*")) == []


def test_cancellation_mid_sweep_preserves_shipped_keeps(tmp_path: Path) -> None:
    """A SIGTERM-style cancellation mid-sweep keeps every keep found so far shipped.

    A blocking candidate stalls the sweep after two keeps are already checkpointed;
    cancelling the pilot task (what the SIGTERM shutdown hook does) tears down the
    sweep, and the two keeps found before the interruption remain materialized as
    workspaces + dataset rows.
    """
    plans = _mixed_plans()
    block_label = plans[2].label

    async def _drive() -> None:
        reached = asyncio.Event()

        class Blocking(FakeProcessor):
            async def process(
                self, plan: CandidatePlan, workdir: Path
            ) -> CandidateArtifacts:
                if plan.label == block_label:
                    reached.set()
                    await asyncio.Event().wait()  # block until cancelled
                return await super().process(plan, workdir)

        task = asyncio.create_task(
            run_pilot(
                PilotConfig(plans=plans, out_dir=tmp_path, candidate_concurrency=1),
                processor=Blocking(),
            )
        )
        await asyncio.wait_for(reached.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_drive())

    # The two keeps found before the interruption are shipped + in both datasets.
    assert len(_task_dirs(tmp_path)) == 2
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 2
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 2


def test_handle_signals_does_not_change_the_happy_path(tmp_path: Path) -> None:
    """Installing the SIGTERM/SIGINT shutdown hook is a no-op on a clean run."""
    without = _run(
        PilotConfig(plans=_mixed_plans(), out_dir=tmp_path / "a"), FakeProcessor()
    )

    async def _drive():  # type: ignore[no-untyped-def]
        return await run_pilot(
            PilotConfig(plans=_mixed_plans(), out_dir=tmp_path / "b"),
            processor=FakeProcessor(),
            handle_signals=True,
        )

    with_hook = asyncio.run(_drive())
    assert without.counts.to_dict() == with_hook.counts.to_dict()
    assert without.shipped_count == with_hook.shipped_count


def test_keep_export_request_only_for_pass_and_keep() -> None:
    """The checkpoint gate mirrors the funnel: only oracle-pass AND band-keep."""
    keep_plan = _plan("iniconfig", "ast_mutation", 0)

    art_keep = CandidateArtifacts(
        plan=keep_plan,
        env_image=_env_image(keep_plan),
        candidate=_candidate(keep_plan),
        spec=_spec(keep_plan),
        oracle_report=_oracle_pass(keep_plan),
        calibration_report=_calibration(keep_plan, keep=True),
    )
    assert _keep_export_request(art_keep) is not None

    art_drop = CandidateArtifacts(
        plan=keep_plan,
        env_image=_env_image(keep_plan),
        candidate=_candidate(keep_plan),
        spec=_spec(keep_plan),
        oracle_report=_oracle_pass(keep_plan),
        calibration_report=_calibration(keep_plan, keep=False),
    )
    assert _keep_export_request(art_drop) is None

    art_reject = CandidateArtifacts(
        plan=keep_plan,
        env_image=_env_image(keep_plan),
        candidate=_candidate(keep_plan),
        spec=_spec(keep_plan),
        oracle_report=_oracle_reject(keep_plan),
    )
    assert _keep_export_request(art_reject) is None
