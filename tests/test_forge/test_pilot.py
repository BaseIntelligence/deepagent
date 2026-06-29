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
from swe_forge.forge.oracle.pipeline import ExportRefusedError
from swe_forge.forge.pilot import (
    DEFAULT_GENERATORS_BY_LANGUAGE,
    CandidateArtifacts,
    CandidatePlan,
    PilotConfig,
    StageCounts,
    build_pilot_plans,
    default_pilot_config,
    run_pilot,
)
from swe_forge.forge.teacher import Usage

_TS = "2026-01-01T00:00:00+00:00"
_GOLD_LINE = "    return compute_total_with_tax(items, tax_rate)"
_BROKEN_LINE = "    return sum(items)"


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
        problem_statement=problem
        or "total() must include tax in the returned amount.",
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
        sourced=10, env_built=8, synthesized=7, oracle_pass=5, calibration_keep=3, exported=3
    )
    assert counts.monotone is True


@pytest.mark.parametrize(
    "counts",
    [
        StageCounts(sourced=3, env_built=5),  # env > sourced
        StageCounts(sourced=5, env_built=4, synthesized=4, oracle_pass=5),  # pass > synth
        StageCounts(sourced=5, env_built=4, synthesized=4, oracle_pass=3, calibration_keep=3, exported=2),  # keep != exported
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
    for language, generators in DEFAULT_GENERATORS_BY_LANGUAGE.items():
        cell = {p.generator for p in plans if p.language == language}
        assert cell == set(generators)
        assert len(cell) >= 2


def test_build_pilot_plans_respects_language_filter_and_cap() -> None:
    only_go = build_pilot_plans(seeds_per_cell=3, languages=["go"])
    assert {p.language for p in only_go} == {"go"}
    capped = build_pilot_plans(seeds_per_cell=5, max_plans=4)
    assert len(capped) == 4


def test_default_pilot_config_wires_plans_and_overrides() -> None:
    config = default_pilot_config(
        "/tmp/pilot-x", seeds_per_cell=1, run_gold_eval=False, gold_eval_runs=3
    )
    assert config.plans
    assert config.run_gold_eval is False
    assert config.gold_eval_runs == 3


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
    assert outcome.usage.teacher.total_tokens == synthesized * FakeProcessor.TEACHER.total_tokens
    assert outcome.usage.panel.total_tokens == oracle_pass * FakeProcessor.PANEL.total_tokens
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
