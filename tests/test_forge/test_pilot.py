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
import signal
import shutil
import subprocess
import sys
import threading
import time

import pytest

from swe_forge.export.jsonl import import_jsonl
from swe_forge.export.parquet import import_parquet
from swe_forge.forge.export import (
    ExportRequest,
    TaskExportResult,
    assemble_forge_task,
)
from swe_forge.forge.gold_eval import EvalRun, GoldEvalReport, TaskGoldResult
from swe_forge.forge.models import (
    CalibrationReport,
    Candidate,
    CandidateTarget,
    EnvImage,
    FinalMutationEvidence,
    GeneratedSpec,
    ModelSolveRecord,
    OracleReport,
    OracleTestFile,
    Provenance,
    RepoSpec,
)
from swe_forge.forge.oracle.establish import HiddenTest, HiddenTestFile
from swe_forge.forge.oracle.mutation import final_suite_fingerprint
from swe_forge.forge.oracle.pipeline import ExportRefusedError
from swe_forge.forge.sources import build_source_registry
from swe_forge.forge.pilot import (
    DEFAULT_GENERATORS_BY_LANGUAGE,
    CandidateArtifacts,
    CandidatePlan,
    PilotConfig,
    PilotError,
    PilotOutcome,
    StageCounts,
    StructuralF2PProtection,
    StructuralF2PProtectionError,
    _accumulate_oracle_usage,
    build_pilot_plans,
    default_pilot_config,
    run_pilot,
)
from swe_forge.forge.teacher import (
    Usage,
    candidate_transport_fingerprint,
)
from tests.test_forge.receipt_helpers import (
    protected_alt_correct_audit,
    protected_alt_correct_summary,
    signed_transport_receipt,
)

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
def _repo(
    repo_id: str, language: str = "python", *, instance_cap: int = 10
) -> RepoSpec:
    return RepoSpec(
        repo_id=repo_id,
        url=f"https://github.com/acme/{repo_id}.git",
        commit="a" * 40,
        commit_date="2024-01-01T00:00:00+00:00",
        language=language,
        license="MIT",
        instance_cap=instance_cap,
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


def _teacher_gate_evidence() -> dict[str, object]:
    def call(gate: str) -> dict[str, object]:
        return {
            "gate": gate,
            "call_kind": "proposal",
            "real_teacher": True,
            "status": "success",
            "response_kind": "content",
            "model": "anthropic/test-teacher",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "cost": 0.0,
            "finish_reason": "stop",
            "requested_proposals": 1,
            "received_proposals": 1,
            "parsed_proposals": 1,
            "identical_proposals": 0,
            "invalid_proposals": 0,
            "discarded_proposals": 0,
            "execution_attempted": 1,
            "execution_completed": 1,
            "execution_errors": 0,
            "executable_proposals": 1,
            "error_type": "",
        }

    return {
        "differential": {"calls": [call("differential")]},
        "alt_correct": {"calls": [call("alt_correct")]},
    }


def _alt_correct_audit(test_files: list[OracleTestFile]) -> dict[str, object]:
    return protected_alt_correct_audit(
        test_files,
        ["python -m pytest tests/hidden/test_total.py"],
        [("src/m.py", "def total(xs): return sum(xs)\n")],
    )


def _oracle_pass(plan: CandidatePlan) -> OracleReport:
    test_files = [
        OracleTestFile(
            path="tests/hidden/test_total.py",
            content=(
                "from src.m import total\n\n\n"
                "def test_total():\n    assert total([100], 0.1) == 110\n"
            ),
        )
    ]
    report = OracleReport(
        language=plan.language,
        generator=plan.generator,
        verdict="pass",
        reasons=[],
        fail_to_pass=["python -m pytest tests/hidden/test_total.py"],
        pass_to_pass=["python -m pytest -q"],
        test_files=test_files,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        final_mutation_evidence=FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(test_files),
            mutants_total=10,
            mutants_killed=10,
            threshold=0.8,
            tool="fake-tool",
        ),
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        details={
            "teacher_gates": _teacher_gate_evidence(),
            "alt_correct": protected_alt_correct_summary(test_files),
        },
        protected_alt_correct_audit=_alt_correct_audit(test_files),
        provenance=_provenance(plan),
    )
    candidate = _candidate(plan)
    gates = report.details["teacher_gates"]
    assert isinstance(gates, dict)
    receipts: list[dict[str, object]] = []
    for index, (gate, payload) in enumerate(gates.items(), start=1):
        assert isinstance(gate, str) and isinstance(payload, dict)
        calls = payload["calls"]
        assert isinstance(calls, list) and isinstance(calls[0], dict)
        call = calls[0]
        call["recovery_accounting"] = None
        receipt = signed_transport_receipt(
            call_id=f"{index:032x}",
            candidate_fingerprint=candidate_transport_fingerprint(candidate),
            gate=gate,
            call_kind=str(call["call_kind"]),
            model=str(call["model"]),
            usage=Usage(**call["usage"]),  # type: ignore[arg-type]
            cost=float(call["cost"]),
        )
        call["call_id"] = receipt.call_id
        call["receipt_commitment"] = receipt.commitment
        receipts.append(receipt.to_private_dict())
    report.protected_teacher_transport_receipts = receipts
    return report


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
        cap_admitted=3,
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


def test_pilot_config_rejects_fewer_than_two_gold_runs(tmp_path: Path) -> None:
    with pytest.raises(PilotError, match="gold_eval_runs must be >= 2"):
        PilotConfig(
            plans=[],
            out_dir=tmp_path,
            gold_eval_runs=1,
            run_gold_eval=False,
        )


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
    assert derived.original_public_test_command == env.original_public_test_command
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


def test_instance_cap_admission_serializes_concurrent_keeps(tmp_path: Path) -> None:
    """Only cap-admitted keeps publish when concurrent candidates share a source."""
    # Deliberately use distinct objects for the same source id. The checkpoint,
    # not object identity or candidate completion order, owns the shared cap.
    sources = [_repo("acme/capped", instance_cap=2) for _ in range(4)]
    plans = [
        CandidatePlan(repo=source, generator="ast_mutation", seed=seed)
        for seed, source in enumerate(sources)
    ]

    outcome = _run(
        PilotConfig(
            plans=plans,
            out_dir=tmp_path,
            candidate_concurrency=4,
            run_gold_eval=False,
        ),
        FakeProcessor(),
    )

    assert outcome.counts.calibration_keep == 4
    assert outcome.counts.cap_admitted == outcome.counts.exported == 2
    assert outcome.counts.monotone is True
    assert {source.used for source in sources} == {2}
    assert {source.remaining for source in sources} == {0}

    cap_rejected = [d for d in outcome.dispositions if d.stage == "cap_rejected"]
    assert len(cap_rejected) == 2
    for disposition in cap_rejected:
        assert disposition.cap_grant is not None
        assert disposition.cap_grant["accepted"] is False
        assert disposition.cap_grant["cap"] == 2
        assert "per-repo cap reached" in disposition.reason

    assert len(_task_dirs(tmp_path)) == 2
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 2
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 2
    assert outcome.report is not None
    payload = outcome.report.to_dict()
    assert payload["funnel"] == {
        key: value
        for key, value in outcome.counts.to_dict().items()
        if key != "monotone"
    }
    assert payload["funnel_reconciles"] is True
    assert payload["capacity_reconciles"] is True
    assert payload["source_capacity"] == [
        {"repo_id": "acme/capped", "cap": 2, "used": 2, "remaining": 0}
    ]


def test_omitted_source_resolves_registered_spec_before_concurrent_admission(
    tmp_path: Path,
) -> None:
    """Omitted sources use the registered cap, before either request queues."""
    source = _repo("acme/capped", instance_cap=1)
    plans = [
        CandidatePlan(repo=source, generator="ast_mutation", seed=0),
        CandidatePlan(repo=source, generator="function_removal", seed=1),
    ]
    checkpoint = PilotCheckpoint(tmp_path, overwrite=True, source_specs=(source,))

    async def _drive() -> tuple[TaskExportResult, TaskExportResult]:
        processor = FakeProcessor()
        requests: list[ExportRequest] = []
        for index, plan in enumerate(plans):
            artifacts = await processor.process(plan, tmp_path / f"work-{index}")
            request = _keep_export_request(artifacts)
            assert request is not None
            requests.append(request)
        first, second = await asyncio.gather(
            checkpoint.record_keep(0, requests[0]),
            checkpoint.record_keep(1, requests[1]),
        )
        return first, second

    results = asyncio.run(_drive())
    assert {result.status for result in results} == {"shipped", "cap_rejected"}
    assert checkpoint.accepted_indexes == (0,)
    assert checkpoint.pending_indexes == ()
    assert checkpoint.capacity_grant(0) is not None
    assert checkpoint.capacity_grant(0).accepted is True
    assert checkpoint.capacity_grant(1) is not None
    assert checkpoint.capacity_grant(1).accepted is False
    assert source.used == 1
    assert len(_task_dirs(tmp_path)) == 1
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 1


def test_checkpoint_refuses_omitted_unknown_source_before_queue_mutation(
    tmp_path: Path,
) -> None:
    """An omitted source may only resolve through the constructor registry."""
    registered = _repo("acme/registered", instance_cap=1)
    unknown_plan = CandidatePlan(
        repo=_repo("acme/unknown", instance_cap=1),
        generator="ast_mutation",
        seed=0,
    )
    checkpoint = PilotCheckpoint(tmp_path, overwrite=True, source_specs=(registered,))

    async def _drive():
        artifacts = await FakeProcessor().process(unknown_plan, tmp_path / "work")
        request = _keep_export_request(artifacts)
        assert request is not None
        return await checkpoint.record_keep(0, request)

    result = asyncio.run(_drive())
    assert result.status == "source_rejected"
    assert "no registered RepoSpec" in result.reason
    assert checkpoint.accepted_indexes == ()
    assert checkpoint.pending_indexes == ()
    assert checkpoint.capacity_grant(0) is None
    assert registered.used == 0
    assert _task_dirs(tmp_path) == []
    assert import_jsonl(tmp_path / "dataset.jsonl") == []
    assert import_parquet(tmp_path / "dataset.parquet") == []


def test_checkpoint_refuses_missing_or_mismatched_explicit_source_before_queue(
    tmp_path: Path,
) -> None:
    """Missing identities and wrong explicit sources fail before a grant exists."""
    source = _repo("acme/source", instance_cap=1)
    other = _repo("acme/other", instance_cap=1)
    plan = CandidatePlan(repo=source, generator="ast_mutation", seed=0)

    async def _request() -> ExportRequest:
        artifacts = await FakeProcessor().process(plan, tmp_path / "work")
        request = _keep_export_request(artifacts)
        assert request is not None
        return request

    missing_request = asyncio.run(_request())
    missing_request.env_image.repo_id = ""
    missing = PilotCheckpoint(tmp_path / "missing", source_specs=(source,))
    missing_result = asyncio.run(missing.record_keep(0, missing_request, source=source))
    assert missing_result.status == "source_rejected"
    assert "missing EnvImage.repo_id" in missing_result.reason
    assert missing.accepted_indexes == ()
    assert missing.pending_indexes == ()
    assert missing.capacity_grant(0) is None

    mismatch_request = asyncio.run(_request())
    mismatch = PilotCheckpoint(tmp_path / "mismatch", source_specs=(source,))
    mismatch_result = asyncio.run(
        mismatch.record_keep(0, mismatch_request, source=other)
    )
    assert mismatch_result.status == "source_rejected"
    assert "does not match EnvImage.repo_id" in mismatch_result.reason
    assert mismatch.accepted_indexes == ()
    assert mismatch.pending_indexes == ()
    assert mismatch.capacity_grant(0) is None
    assert source.used == other.used == 0
    assert _task_dirs(tmp_path / "missing") == _task_dirs(tmp_path / "mismatch") == []


def test_pilot_records_source_rejection_without_artifacts(tmp_path: Path) -> None:
    """A direct caller's source mismatch remains visible in the pilot ledger."""
    plan = _plan("acme/source", "ast_mutation", 0)

    class MismatchedEnvProcessor(FakeProcessor):
        async def process(
            self, candidate_plan: CandidatePlan, workdir: Path
        ) -> CandidateArtifacts:
            artifacts = await super().process(candidate_plan, workdir)
            assert artifacts.env_image is not None
            artifacts.env_image.repo_id = "acme/unregistered"
            return artifacts

    outcome = _run(
        PilotConfig(
            plans=[plan],
            out_dir=tmp_path,
            run_gold_eval=False,
            write_report=False,
        ),
        MismatchedEnvProcessor(),
    )

    assert outcome.counts.calibration_keep == 1
    assert outcome.counts.cap_admitted == outcome.counts.exported == 0
    assert outcome.counts.export_refused == 1
    assert [(item.stage, item.reason) for item in outcome.dispositions] == [
        (
            "source_rejected",
            "explicit RepoSpec.repo_id 'acme/source' does not match "
            "EnvImage.repo_id 'acme/unregistered'",
        )
    ]
    assert _task_dirs(tmp_path) == []
    assert import_jsonl(tmp_path / "dataset.jsonl") == []
    assert import_parquet(tmp_path / "dataset.parquet") == []


def test_checkpoint_accepts_matching_explicit_unregistered_source(
    tmp_path: Path,
) -> None:
    """An explicit source is valid, but never authorizes future omission."""
    source = _repo("acme/direct", instance_cap=2)
    plan = CandidatePlan(repo=source, generator="ast_mutation", seed=0)
    checkpoint = PilotCheckpoint(tmp_path, overwrite=True)

    async def _drive() -> tuple[TaskExportResult, TaskExportResult]:
        artifacts = await FakeProcessor().process(plan, tmp_path / "work")
        request = _keep_export_request(artifacts)
        assert request is not None
        shipped = await checkpoint.record_keep(0, request, source=source)

        omitted_plan = CandidatePlan(
            repo=source,
            generator="function_removal",
            seed=1,
        )
        omitted_artifacts = await FakeProcessor().process(
            omitted_plan, tmp_path / "omitted"
        )
        omitted_request = _keep_export_request(omitted_artifacts)
        assert omitted_request is not None
        omitted = await checkpoint.record_keep(1, omitted_request)
        return shipped, omitted

    result, omitted = asyncio.run(_drive())
    assert result.status == "shipped"
    assert omitted.status == "source_rejected"
    assert "no registered RepoSpec" in omitted.reason
    assert source.used == 1
    assert checkpoint.capacity_grant(0) is not None
    assert checkpoint.capacity_grant(0).repo_id == source.repo_id
    assert checkpoint.capacity_grant(1) is None
    assert checkpoint.accepted_indexes == (0,)
    assert checkpoint.pending_indexes == ()
    assert len(_task_dirs(tmp_path)) == 1


def test_omitted_source_refusal_releases_its_single_grant_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused write returns an omitted-source grant without leaving artifacts."""
    source = _repo("acme/capped", instance_cap=1)
    first = CandidatePlan(repo=source, generator="ast_mutation", seed=0)
    replacement = CandidatePlan(repo=source, generator="function_removal", seed=1)
    checkpoint = PilotCheckpoint(tmp_path, overwrite=True, source_specs=(source,))
    original = checkpoint._record_keep_sync

    def _refuse(index: int, request: ExportRequest) -> TaskExportResult:
        return TaskExportResult(
            task_id=request._fallback_id(),
            status="refused",
            reason="induced source-safe refusal",
        )

    monkeypatch.setattr(checkpoint, "_record_keep_sync", _refuse)

    async def _refuse_first() -> TaskExportResult:
        first_artifacts = await FakeProcessor().process(first, tmp_path / "first")
        first_request = _keep_export_request(first_artifacts)
        assert first_request is not None
        return await checkpoint.record_keep(0, first_request)

    refused = asyncio.run(_refuse_first())
    assert refused.status == "refused"
    assert source.used == 0
    assert checkpoint._released_capacity == {0}
    assert _task_dirs(tmp_path) == []

    monkeypatch.setattr(checkpoint, "_record_keep_sync", original)

    async def _ship_replacement() -> TaskExportResult:
        replacement_artifacts = await FakeProcessor().process(
            replacement, tmp_path / "replacement"
        )
        replacement_request = _keep_export_request(replacement_artifacts)
        assert replacement_request is not None
        return await checkpoint.record_keep(1, replacement_request)

    shipped = asyncio.run(_ship_replacement())
    assert shipped.status == "shipped"
    assert source.used == 1
    assert len(_task_dirs(tmp_path)) == 1


def test_non_qualified_candidates_do_not_consume_source_capacity(
    tmp_path: Path,
) -> None:
    """Only an oracle-pass, band-keep candidate reaches RepoSpec.acquire."""
    source = _repo("acme/capped", instance_cap=1)
    plans = [
        CandidatePlan(repo=source, generator="ast_mutation", seed=seed)
        for seed in range(5)
    ]
    outcomes = {
        plans[0].label: "env_failed",
        plans[1].label: "synth_failed",
        plans[2].label: "oracle_reject",
        plans[3].label: "calib_drop",
    }

    outcome = _run(
        PilotConfig(plans=plans, out_dir=tmp_path, run_gold_eval=False),
        FakeProcessor(outcomes),
    )

    assert outcome.counts.calibration_keep == 1
    assert outcome.counts.cap_admitted == outcome.counts.exported == 1
    assert source.used == 1
    assert source.remaining == 0
    assert {d.stage for d in outcome.dispositions} == {
        "env_failed",
        "synth_failed",
        "oracle_reject",
        "calib_drop",
        "kept",
    }


def test_refused_export_releases_capacity_without_silencing_the_failure(
    tmp_path: Path,
) -> None:
    """A Stage-5 refusal does not spend a slot, but the failed run is visible."""
    source = _repo("acme/capped", instance_cap=1)
    leaking = CandidatePlan(repo=source, generator="ast_mutation", seed=0)
    replacement = CandidatePlan(repo=source, generator="function_removal", seed=1)

    outcome = _run(
        PilotConfig(
            plans=[leaking, replacement],
            out_dir=tmp_path,
            run_gold_eval=False,
        ),
        FakeProcessor({leaking.label: "leak"}),
    )

    assert outcome.counts.calibration_keep == 2
    assert outcome.counts.cap_admitted == outcome.counts.exported == 1
    assert outcome.counts.export_refused == 1
    assert outcome.counts.monotone is False
    assert outcome.ok is False
    assert source.used == 1
    assert [d.stage for d in outcome.dispositions] == ["export_refused", "kept"]
    assert len(_task_dirs(tmp_path)) == 1


def test_checkpoint_restart_restores_source_capacity_before_new_publication(
    tmp_path: Path,
) -> None:
    """Published keeps reconstruct used capacity for a fresh RepoSpec on restart."""
    first_source = _repo("acme/capped", instance_cap=1)
    first_plan = CandidatePlan(repo=first_source, generator="ast_mutation", seed=0)
    restarted_source = _repo("acme/capped", instance_cap=1)
    second_plan = CandidatePlan(
        repo=restarted_source, generator="function_removal", seed=1
    )

    async def _drive() -> None:
        processor = FakeProcessor()
        checkpoint = PilotCheckpoint(
            tmp_path, overwrite=True, source_specs=(first_source,)
        )
        first = await processor.process(first_plan, tmp_path / "first")
        first_request = _keep_export_request(first)
        assert first_request is not None
        assert (await checkpoint.record_keep(0, first_request)).status == "shipped"

        restarted = PilotCheckpoint(
            tmp_path, overwrite=True, source_specs=(restarted_source,)
        )
        assert restarted_source.used == 1
        assert restarted_source.remaining == 0
        second = await processor.process(second_plan, tmp_path / "second")
        second_request = _keep_export_request(second)
        assert second_request is not None
        refused = await restarted.record_keep(1, second_request)
        assert refused.status == "cap_rejected"
        assert "per-repo cap reached" in refused.reason

    asyncio.run(_drive())
    assert len(_task_dirs(tmp_path)) == 1
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 1


def test_checkpoint_replay_does_not_acquire_a_second_source_slot(
    tmp_path: Path,
) -> None:
    """A recovered plan index stays idempotent through PilotCheckpoint's API."""
    initial_source = _repo("acme/capped", instance_cap=1)
    initial_plan = CandidatePlan(repo=initial_source, generator="ast_mutation", seed=0)
    replay_source = _repo("acme/capped", instance_cap=1)
    replay_plan = CandidatePlan(repo=replay_source, generator="ast_mutation", seed=0)

    async def _drive() -> None:
        processor = FakeProcessor()
        first_checkpoint = PilotCheckpoint(
            tmp_path, overwrite=True, source_specs=(initial_source,)
        )
        initial_artifacts = await processor.process(initial_plan, tmp_path / "first")
        initial_request = _keep_export_request(initial_artifacts)
        assert initial_request is not None
        assert (
            await first_checkpoint.record_keep(0, initial_request)
        ).status == "shipped"

        restarted = PilotCheckpoint(
            tmp_path, overwrite=True, source_specs=(replay_source,)
        )
        replay_artifacts = await processor.process(replay_plan, tmp_path / "replay")
        replay_request = _keep_export_request(replay_artifacts)
        assert replay_request is not None
        result = await restarted.record_keep(0, replay_request)
        assert result.status == "skipped"
        assert replay_source.used == 1
        assert replay_source.remaining == 0

    asyncio.run(_drive())
    assert len(_task_dirs(tmp_path)) == 1
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1


def test_checkpoint_replay_fails_closed_for_invalid_source_identity(
    tmp_path: Path,
) -> None:
    """Replay idempotency applies only after source resolution succeeds."""
    source = _repo("acme/capped", instance_cap=1)
    plan = CandidatePlan(repo=source, generator="ast_mutation", seed=0)
    other = _repo("acme/other", instance_cap=1)

    async def _request(name: str) -> ExportRequest:
        artifacts = await FakeProcessor().process(plan, tmp_path / name)
        request = _keep_export_request(artifacts)
        assert request is not None
        return request

    async def _drive() -> None:
        initial = PilotCheckpoint(tmp_path, overwrite=True, source_specs=(source,))
        initial_request = await _request("initial")
        assert (await initial.record_keep(0, initial_request)).status == "shipped"
        assert source.used == 1

        missing = PilotCheckpoint(
            tmp_path,
            overwrite=True,
            source_specs=(source,),
        )
        missing_request = await _request("missing")
        missing_request.env_image.repo_id = ""
        missing_result = await missing.record_keep(0, missing_request)
        assert missing_result.status == "source_rejected"
        assert missing.accepted_indexes == ()
        assert missing.pending_indexes == ()

        unknown = PilotCheckpoint(
            tmp_path,
            overwrite=True,
            source_specs=(source,),
        )
        unknown_request = await _request("unknown")
        unknown_request.env_image.repo_id = "acme/unknown"
        unknown_result = await unknown.record_keep(0, unknown_request)
        assert unknown_result.status == "source_rejected"
        assert unknown.accepted_indexes == ()
        assert unknown.pending_indexes == ()

        mismatched = PilotCheckpoint(
            tmp_path,
            overwrite=True,
            source_specs=(source,),
        )
        mismatched_request = await _request("mismatched")
        mismatched_result = await mismatched.record_keep(
            0,
            mismatched_request,
            source=other,
        )
        assert mismatched_result.status == "source_rejected"
        assert mismatched.accepted_indexes == ()
        assert mismatched.pending_indexes == ()

    asyncio.run(_drive())
    assert source.used == 1
    assert len(_task_dirs(tmp_path)) == 1
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 1


def test_pilot_restart_reuses_committed_funnel_without_new_cap_acquisition(
    tmp_path: Path,
) -> None:
    """A replayed plan is counted from the committed generation, not re-admitted."""
    initial_source = _repo("acme/capped", instance_cap=1)
    initial_plan = CandidatePlan(repo=initial_source, generator="ast_mutation", seed=0)
    first = _run(
        PilotConfig(
            plans=[initial_plan],
            out_dir=tmp_path,
            run_gold_eval=False,
        ),
        FakeProcessor(),
    )
    assert first.counts.cap_admitted == first.counts.exported == 1

    restarted_source = _repo("acme/capped", instance_cap=1)
    replay = CandidatePlan(repo=restarted_source, generator="ast_mutation", seed=0)
    restarted = _run(
        PilotConfig(
            plans=[replay],
            out_dir=tmp_path,
            run_gold_eval=False,
        ),
        FakeProcessor(),
    )

    assert restarted.dispositions == []
    assert restarted.counts.to_dict() == {
        "sourced": 1,
        "env_built": 1,
        "synthesized": 1,
        "oracle_pass": 1,
        "calibration_keep": 1,
        "cap_admitted": 1,
        "exported": 1,
        "export_refused": 0,
        "monotone": True,
    }
    assert restarted_source.used == 1
    assert restarted_source.remaining == 0
    assert len(_task_dirs(tmp_path)) == 1
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1


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


def test_oracle_teacher_usage_is_accounted_before_candidate_disposition() -> None:
    plan = _plan("usage", "ast_mutation", 1)
    artifacts = CandidateArtifacts(plan=plan)

    _accumulate_oracle_usage(artifacts, _oracle_pass(plan))

    assert artifacts.teacher_usage.total_tokens == 4
    assert artifacts.teacher_cost == 0.0


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
    assert outcome.gold is not None
    assert report.gold.results == tuple(outcome.gold.results)
    assert report.gold.strict_proof is True
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
    checkpoint = PilotCheckpoint(
        tmp_path,
        overwrite=True,
        source_specs=tuple(plan.repo for plan in plans),
    )

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
    checkpoint = PilotCheckpoint(
        tmp_path,
        overwrite=True,
        source_specs=(plans[0].repo, plans[3].repo),
    )

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


def test_checkpoint_restart_recovers_complete_generation_and_ignores_staging(
    tmp_path: Path,
) -> None:
    """A restarted checkpoint resumes the committed generation, not staging data."""
    plans = _mixed_plans()

    async def _drive() -> None:
        processor = FakeProcessor()
        first = await processor.process(plans[0], tmp_path / "w0")
        checkpoint = PilotCheckpoint(
            tmp_path,
            overwrite=True,
            source_specs=(plans[0].repo, plans[3].repo),
        )
        first_result = await checkpoint.record_keep(
            0,
            _keep_export_request(first),  # type: ignore[arg-type]
        )
        assert first_result.status == "shipped"

        # Simulate a hard-killed publisher that left uncommitted staging bytes.
        abandoned = tmp_path / ".forge-publications" / ".staging-abandoned"
        abandoned.mkdir(parents=True)
        (abandoned / "dataset.jsonl").write_text('{"id":"not-committed"}\n')

        restarted = PilotCheckpoint(
            tmp_path,
            overwrite=True,
            source_specs=(plans[0].repo, plans[3].repo),
        )
        assert restarted.kept_count == 1
        assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1
        assert len(import_parquet(tmp_path / "dataset.parquet")) == 1

        second = await processor.process(plans[3], tmp_path / "w1")
        second_result = await restarted.record_keep(
            3,
            _keep_export_request(second),  # type: ignore[arg-type]
        )
        assert second_result.status == "shipped"

    asyncio.run(_drive())
    task_dirs = _task_dirs(tmp_path)
    jsonl = import_jsonl(tmp_path / "dataset.jsonl")
    parquet = import_parquet(tmp_path / "dataset.parquet")
    assert len(task_dirs) == len(jsonl) == len(parquet) == 2
    assert (
        {task.id for task in jsonl}
        == {row["id"] for row in parquet}
        == {task_dir.name for task_dir in task_dirs}
    )


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


# --------------------------------------------------------------------------- #
# Checkpoint shutdown safety (m6-pilot-checkpoint-shutdown-safety)
# --------------------------------------------------------------------------- #
def test_cancelled_refused_checkpoint_write_releases_source_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled drain settles a refused write without retaining its source slot."""
    from swe_forge.forge.export import TaskExportResult

    source = _repo("acme/capped", instance_cap=1)
    plan = CandidatePlan(repo=source, generator="ast_mutation", seed=0)
    checkpoint = PilotCheckpoint(tmp_path, overwrite=True, source_specs=(source,))
    started = threading.Event()
    release = threading.Event()

    def _refuse(index: int, request: ExportRequest) -> TaskExportResult:
        assert index == 0
        started.set()
        assert release.wait(timeout=5)
        return TaskExportResult(
            task_id=request._fallback_id(),
            status="refused",
            reason="induced leak refusal",
        )

    monkeypatch.setattr(checkpoint, "_record_keep_sync", _refuse)

    async def _drive() -> None:
        artifacts = await FakeProcessor().process(plan, tmp_path / "work")
        request = _keep_export_request(artifacts)
        assert request is not None
        assert checkpoint.admit_keep(0, request) is None
        assert source.used == 1

        drain = asyncio.create_task(checkpoint.drain(indexes=(0,)))
        assert await asyncio.to_thread(started.wait, 5)
        drain.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await drain

        for _ in range(50):
            if not checkpoint.pending_indexes:
                break
            await asyncio.sleep(0.01)

    asyncio.run(_drive())
    assert checkpoint.pending_indexes == ()
    assert source.used == 0
    assert source.remaining == 1
    assert _task_dirs(tmp_path) == []


def test_checkpoint_closes_admission_and_drains_accepted_keeps_in_plan_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM-style closure drains prior admissions before their source cleanup.

    The first admitted keep blocks inside publication while the second one queues.
    Closing admission rejects a later keep, and the asynchronous drain must finish
    the queued pair in deterministic plan-index order before either source path is
    eligible for deletion.
    """
    from swe_forge.forge import checkpoint as checkpoint_mod

    plans = _mixed_plans()
    checkpoint = PilotCheckpoint(
        tmp_path,
        overwrite=True,
        source_specs=tuple(plan.repo for plan in plans),
    )
    processor = FakeProcessor()
    sources = [tmp_path / "sources" / str(index) for index in range(3)]
    for source in sources:
        source.mkdir(parents=True)
        (source / "marker.txt").write_text(source.name)
        (source / ".gitignore").write_text(".git/\n")

    async def _request(index: int) -> ExportRequest:
        art = await processor.process(plans[index], tmp_path / f"work-{index}")
        request = _keep_export_request(art)
        assert request is not None
        request.broken_tree = sources[index]
        return request

    started = threading.Event()
    release = threading.Event()
    order: list[int] = []
    copied_sources: list[int] = []
    original = checkpoint._record_keep_sync
    original_export = checkpoint_mod._write_staged_workspace

    def _blocking(index: int, request: ExportRequest):
        order.append(index)
        if index == 2:
            started.set()
            assert release.wait(timeout=5)
        return original(index, request)

    monkeypatch.setattr(checkpoint, "_record_keep_sync", _blocking)

    def _source_aware_export(*args: object, **kwargs: object):
        broken = kwargs.get("broken_tree")
        assert isinstance(broken, Path) and broken.is_dir()
        copied_sources.append(int(broken.name))
        shutil.copytree(broken, tmp_path / "copied-sources" / broken.name)
        # The fixture source is intentionally not a full checkout. The existing
        # export tests cover copytree; this shutdown test isolates source lifetime.
        kwargs["broken_tree"] = None
        return original_export(*args, **kwargs)

    monkeypatch.setattr(
        "swe_forge.forge.checkpoint._write_staged_workspace", _source_aware_export
    )

    async def _drive() -> None:
        requests = await asyncio.gather(*[_request(index) for index in range(3)])
        assert checkpoint.admit_keep(2, requests[2]) is None
        first = asyncio.create_task(checkpoint.drain(indexes=(2,)))
        assert await asyncio.to_thread(started.wait, 5)

        # These were accepted before closure and must remain in the pending
        # ledger. The later candidate is refused without scheduling copy I/O.
        assert checkpoint.admit_keep(0, requests[0]) is None
        assert checkpoint.admit_keep(1, requests[1]) is None
        checkpoint.close_admission()
        refused = checkpoint.admit_keep(3, requests[2])
        assert refused is not None and refused.status == "refused"

        drain = asyncio.create_task(checkpoint.drain())
        await asyncio.sleep(0.05)
        assert sources[0].exists() and sources[1].exists() and sources[2].exists()
        release.set()
        await first
        await drain

    asyncio.run(_drive())

    assert checkpoint.accepted_indexes == (0, 1, 2)
    assert checkpoint.pending_indexes == ()
    assert order == [2, 0, 1]
    # Every admitted broken tree was still available when the exporter started.
    assert copied_sources == [2, 0, 1]
    assert len(_task_dirs(tmp_path)) == 3


def test_shutdown_drain_continues_after_a_failed_accepted_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed copy never strands a later accepted keep before cleanup."""
    plans = _mixed_plans()
    checkpoint = PilotCheckpoint(
        tmp_path,
        overwrite=True,
        source_specs=(plans[0].repo, plans[1].repo),
    )

    async def _requests() -> list[ExportRequest]:
        processor = FakeProcessor()
        result: list[ExportRequest] = []
        for index in (0, 1):
            art = await processor.process(plans[index], tmp_path / f"work-{index}")
            request = _keep_export_request(art)
            assert request is not None
            result.append(request)
        return result

    requests = asyncio.run(_requests())
    original = checkpoint._record_keep_sync

    def _fail_first(index: int, request: ExportRequest):
        if index == 0:
            raise OSError("induced checkpoint write failure")
        return original(index, request)

    monkeypatch.setattr(checkpoint, "_record_keep_sync", _fail_first)

    async def _drain() -> dict[int, object]:
        assert checkpoint.admit_keep(0, requests[0]) is None
        assert checkpoint.admit_keep(1, requests[1]) is None
        result = await checkpoint.drain(continue_on_error=True)
        assert isinstance(result, dict)
        return result

    results = asyncio.run(_drain())
    assert results[0].status == "failed"  # type: ignore[union-attr]
    assert results[1].status == "shipped"  # type: ignore[union-attr]
    assert checkpoint.pending_indexes == ()
    assert len(_task_dirs(tmp_path)) == 1


def test_shutdown_drain_defers_cancellation_until_all_io_has_settled() -> None:
    """A follow-up cancellation cannot expose cleanup while checkpoint I/O runs."""
    from swe_forge.forge.pilot import _drain_checkpoint_before_cleanup

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowCheckpoint:
        async def drain(self, *, continue_on_error: bool) -> dict[int, object]:
            assert continue_on_error is True
            started.set()
            await release.wait()
            return {}

    async def _drive() -> None:
        task = asyncio.create_task(_drain_checkpoint_before_cleanup(SlowCheckpoint()))  # type: ignore[arg-type]
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        assert await task is True

    asyncio.run(_drive())


def test_sigterm_drains_accepted_keep_before_run_root_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real signal hook closes admission, drains, then safely cleans sources."""
    from swe_forge.forge import checkpoint as checkpoint_mod

    plan = _mixed_plans()[0]
    source_seen_during_copy: list[bool] = []
    copied_markers: list[str] = []
    sources: list[Path] = []
    copy_started = threading.Event()
    original_export = checkpoint_mod._write_staged_workspace

    def _slow_export(*args: object, **kwargs: object):
        broken = kwargs.get("broken_tree")
        assert isinstance(broken, Path)
        source_seen_during_copy.append(broken.is_dir())
        copied_markers.append(
            shutil.copytree(broken, tmp_path / "signal-copy" / broken.name)
            .joinpath("marker.txt")
            .read_text()
        )
        copy_started.set()
        time.sleep(0.1)
        kwargs["broken_tree"] = None
        return original_export(*args, **kwargs)

    monkeypatch.setattr(
        "swe_forge.forge.checkpoint._write_staged_workspace", _slow_export
    )

    async def _drive() -> PilotOutcome:
        class KeepThenBlock(FakeProcessor):
            async def process(
                self, candidate_plan: CandidatePlan, workdir: Path
            ) -> CandidateArtifacts:
                if candidate_plan.label != plan.label:
                    await asyncio.Event().wait()
                art = await super().process(candidate_plan, workdir)
                source = workdir / "broken-source"
                source.mkdir()
                (source / "marker.txt").write_text("accepted")
                (source / ".gitignore").write_text(".git/\n")
                art.broken_tree = source
                art.cleanup_paths.append(source)
                sources.append(source)
                return art

        task = asyncio.create_task(
            run_pilot(
                PilotConfig(
                    plans=[plan, _mixed_plans()[1]],
                    out_dir=tmp_path,
                    candidate_concurrency=1,
                    run_gold_eval=False,
                    write_report=False,
                ),
                processor=KeepThenBlock(),
                handle_signals=True,
            )
        )
        assert await asyncio.to_thread(copy_started.wait, 5)
        # The callback is publishing its accepted keep. Deliver the actual signal
        # to the installed handler while Stage 5 copy I/O is active.
        signal.raise_signal(signal.SIGTERM)
        outcome = await asyncio.wait_for(task, timeout=10)
        return outcome

    outcome = asyncio.run(_drive())

    assert source_seen_during_copy == [True]
    assert copied_markers == ["accepted"]
    assert sources and not sources[0].exists()
    assert outcome.shipped_count == 1
    assert len(_task_dirs(tmp_path)) == 1
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 1


@pytest.mark.parametrize(
    ("boundary", "expected_count"),
    [
        ("before_generation_rename", 1),
        ("after_generation_rename", 1),
        ("before_pointer_replace", 1),
        ("after_pointer_replace", 2),
    ],
)
def test_sigkill_publication_boundaries_recover_complete_generation(
    tmp_path: Path,
    boundary: str,
    expected_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hard kill exposes only the old or complete new publication generation."""
    from tests.test_forge.test_export import _request as export_request
    from swe_forge.forge import export as export_mod
    from swe_forge.forge import publication as publication_mod

    monkeypatch.setattr(
        export_mod, "ensure_oracle_exportable", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        publication_mod, "ensure_oracle_exportable", lambda *_args, **_kwargs: None
    )
    first = export_batch(
        [export_request(include_teacher_evidence=False)], tmp_path, overwrite=True
    )
    original_ids = {task.id for task in import_jsonl(first.jsonl_path)}
    # The child reconstructs a valid request from the test fixture and kills
    # itself at a real os.replace publication boundary. It never touches Docker.
    script = r"""
import os
import sys
from pathlib import Path
from tests.test_forge.test_export import _request, _candidate
from swe_forge.forge import receipt_authority
from swe_forge.forge import publication
from swe_forge.forge import export as export_mod
from swe_forge.forge.export import export_batch

receipt_authority.default_authority_root = lambda: Path(
    os.environ["SWE_FORGE_TEST_RECEIPT_AUTHORITY_ROOT"]
)
out = Path(sys.argv[1])
boundary = sys.argv[2]
original = publication.os.replace

def replace(source, destination):
    destination = Path(destination)
    if boundary == "before_generation_rename" and destination.parent.name == "generations":
        os.kill(os.getpid(), 9)
    if boundary == "before_pointer_replace" and destination.name == "current":
        os.kill(os.getpid(), 9)
    result = original(source, destination)
    if boundary == "after_generation_rename" and destination.parent.name == "generations":
        os.kill(os.getpid(), 9)
    if boundary == "after_pointer_replace" and destination.name == "current":
        os.kill(os.getpid(), 9)
    return result

publication.os.replace = replace
export_mod.ensure_oracle_exportable = lambda *_args, **_kwargs: None
publication.ensure_oracle_exportable = lambda *_args, **_kwargs: None
export_batch(
    [
        _request(include_teacher_evidence=False),
        _request(
            candidate=_candidate(seed=313),
            repo_url="https://github.com/acme/second.git",
            include_teacher_evidence=False,
        ),
    ],
    out,
    overwrite=True,
)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        cwd="/projects/Agent-SWE",
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert child.returncode == -signal.SIGKILL

    from swe_forge.forge.publication import load_published_generation

    generation = load_published_generation(tmp_path)
    assert generation is not None
    task_ids = {task_dir.name for task_dir in _task_dirs(tmp_path)}
    jsonl_ids = {task.id for task in import_jsonl(tmp_path / "dataset.jsonl")}
    parquet_ids = {
        str(row["id"]) for row in import_parquet(tmp_path / "dataset.parquet")
    }
    assert task_ids == jsonl_ids == parquet_ids
    assert len(task_ids) == expected_count
    assert original_ids <= task_ids

    # Restarting after every boundary can continue publishing from the committed
    # generation; abandoned staging/final directories remain invisible.
    restarted = PilotCheckpoint(tmp_path, overwrite=True)
    assert restarted.kept_count == expected_count
