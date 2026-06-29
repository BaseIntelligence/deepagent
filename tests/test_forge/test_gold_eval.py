"""Headline A gold-eval verification path (VAL-EXPORT-009/010/011).

Offline tests inject a fake Docker runner so the aggregation + determinism logic
is exercised deterministically without Docker or the live endpoint:

- VAL-EXPORT-009: a final ``{"score": 1}`` plus ``Phase 1 PASSED`` is the
  per-task broken-fail + regression-green + gold proof; a Phase-1 abort or a
  gold-fail scores non-gold.
- VAL-EXPORT-010: gold == 100% across the shipped set is true iff every task's
  every run scored 1; one non-1 task breaks the aggregate.
- VAL-EXPORT-011: >=2 independent --rm runs must not flip; a 1->0 flip is
  reported non-deterministic.

The real ``docker run --rm`` path (a fully assembled task whose ``evaluate.sh``
clones a repo, applies the mutation+gold patches, and scores gold 1 twice in
fresh containers) is exercised by the ``integration``-marked test below, which
is deselected from the milestone gate.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

from swe_forge.forge import gold_eval as ge
from swe_forge.forge.gold_eval import (
    DockerExec,
    GoldEvalError,
    evaluate_task_gold,
    parse_score,
    phase1_passed,
    resolve_eval_image,
    resolve_tasks_root,
    run_gold_eval,
)

# --------------------------------------------------------------------------- #
# evaluate.sh stdout samples
# --------------------------------------------------------------------------- #
_GOLD_OUT = (
    "=== SWE-Forge Evaluator ===\n"
    "=== Phase 1: before gold patch ===\n"
    "Phase 1 PASSED\n"
    "=== Applying gold patch ===\n"
    "=== Phase 2: after gold patch ===\n"
    "=== RESULT: PASS ===\n"
    '{"score": 1}\n'
)
_BROKEN_FAIL_OUT = (
    "=== Phase 1: before gold patch ===\n"
    "FAIL: hidden test should FAIL before patch\n"
    "Phase 1 FAILED - aborting\n"
    '{"score": 0}\n'
)
_GOLD_FAIL_OUT = (
    "=== Phase 1: before gold patch ===\n"
    "Phase 1 PASSED\n"
    "=== Phase 2: after gold patch ===\n"
    "FAIL: pass_to_pass should PASS\n"
    "=== RESULT: FAIL ===\n"
    '{"score": 0}\n'
)


# --------------------------------------------------------------------------- #
# Fake runners + fixtures
# --------------------------------------------------------------------------- #
def _const_runner(stdout: str, exit_code: int = 0) -> ge.DockerRunner:
    def runner(task_dir, *, image, name, timeout, extra_args):  # type: ignore[no-untyped-def]
        return DockerExec(exit_code, stdout, "")

    return runner


def _sequence_runner(outputs: list[str]) -> ge.DockerRunner:
    calls = iter(outputs)

    def runner(task_dir, *, image, name, timeout, extra_args):  # type: ignore[no-untyped-def]
        return DockerExec(0, next(calls), "")

    return runner


def _per_task_runner(
    by_id: dict[str, str], default: str = _GOLD_OUT
) -> ge.DockerRunner:
    def runner(task_dir, *, image, name, timeout, extra_args):  # type: ignore[no-untyped-def]
        return DockerExec(0, by_id.get(Path(task_dir).name, default), "")

    return runner


def _make_task_dir(
    root: Path, task_id: str, *, image: str = "swe-forge-env:demo"
) -> Path:
    d = root / task_id
    d.mkdir(parents=True)
    (d / "evaluate.sh").write_text("#!/bin/bash\necho '{\"score\": 1}'\n")
    (d / "evaluate.sh").chmod(0o755)
    (d / "workspace.yaml").write_text(
        yaml.safe_dump(
            {"environment": {"image": image, "base_image": "python:3.12-slim"}}
        )
    )
    return d


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
def test_parse_score_returns_last_value() -> None:
    # An early abort prints {"score": 0}; the final score is the last match.
    assert parse_score('{"score": 0}\nlater\n{"score": 1}\n') == 1
    assert parse_score(_GOLD_OUT) == 1
    assert parse_score(_BROKEN_FAIL_OUT) == 0


def test_parse_score_none_when_absent() -> None:
    assert parse_score("no score here") is None
    assert parse_score("") is None


def test_phase1_passed_marker() -> None:
    assert phase1_passed(_GOLD_OUT) is True
    assert phase1_passed(_BROKEN_FAIL_OUT) is False


# --------------------------------------------------------------------------- #
# VAL-EXPORT-009: per-task broken-fail + regression-green + gold score 1
# --------------------------------------------------------------------------- #
def test_task_gold_when_all_runs_score_1(tmp_path: Path) -> None:
    d = _make_task_dir(tmp_path, "acme__ast__deadbeef0001")
    result = evaluate_task_gold(d, runs=2, runner=_const_runner(_GOLD_OUT))

    assert result.gold is True
    assert result.deterministic is True
    assert result.phase1_all is True  # Phase 1 PASSED = broken-fail + regression-green
    assert result.scores == [1, 1]
    assert result.final_score == 1


def test_task_not_gold_when_phase1_aborts(tmp_path: Path) -> None:
    d = _make_task_dir(tmp_path, "acme__ast__deadbeef0002")
    result = evaluate_task_gold(d, runs=2, runner=_const_runner(_BROKEN_FAIL_OUT))

    # Broken state did not fail / regression broke -> Phase 1 abort, not gold.
    assert result.gold is False
    assert result.phase1_all is False
    assert result.scores == [0, 0]


def test_task_not_gold_when_gold_phase_fails(tmp_path: Path) -> None:
    d = _make_task_dir(tmp_path, "acme__ast__deadbeef0003")
    result = evaluate_task_gold(d, runs=1, runner=_const_runner(_GOLD_FAIL_OUT))
    assert result.gold is False
    assert result.final_score == 0


# --------------------------------------------------------------------------- #
# VAL-EXPORT-011: determinism across >=2 fresh-container runs (no flip)
# --------------------------------------------------------------------------- #
def test_flip_across_runs_is_not_deterministic(tmp_path: Path) -> None:
    d = _make_task_dir(tmp_path, "acme__ast__deadbeef0004")
    result = evaluate_task_gold(
        d, runs=2, runner=_sequence_runner([_GOLD_OUT, _BROKEN_FAIL_OUT])
    )
    assert result.scores == [1, 0]
    assert result.deterministic is False
    assert result.gold is False


def test_runs_use_distinct_container_names(tmp_path: Path) -> None:
    d = _make_task_dir(tmp_path, "acme__ast__deadbeef0005")
    result = evaluate_task_gold(d, runs=3, runner=_const_runner(_GOLD_OUT))
    names = [r.container_name for r in result.runs]
    assert len(result.runs) == 3
    assert len(set(names)) == 3  # each fresh container is uniquely named
    assert all(n.startswith("swe-forge-goldeval-") for n in names)


# --------------------------------------------------------------------------- #
# VAL-EXPORT-010: gold == 100% across the entire shipped set
# --------------------------------------------------------------------------- #
def test_aggregate_all_gold(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    for i in range(3):
        _make_task_dir(tasks, f"acme__ast__id{i:012d}")
    report = run_gold_eval(tasks, runs=2, runner=_const_runner(_GOLD_OUT))

    assert report.shipped_count == 3
    assert report.gold_count == 3
    assert report.gold_rate == 1.0
    assert report.all_gold is True
    assert report.deterministic is True
    assert report.passed is True
    assert report.non_gold == []


def test_aggregate_one_non_gold_blocks_100pct(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    ids = [f"acme__ast__id{i:012d}" for i in range(3)]
    for tid in ids:
        _make_task_dir(tasks, tid)
    runner = _per_task_runner({ids[1]: _BROKEN_FAIL_OUT})
    report = run_gold_eval(tasks, runs=2, runner=runner)

    assert report.shipped_count == 3
    assert report.gold_count == 2
    assert report.all_gold is False
    assert report.passed is False
    assert [r.task_id for r in report.non_gold] == [ids[1]]


def test_aggregate_flip_blocks_pass(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _make_task_dir(tasks, "acme__ast__flip00000001")
    # First run 1, second run 0 -> the single task flips.
    report = run_gold_eval(
        tasks, runs=2, runner=_sequence_runner([_GOLD_OUT, _BROKEN_FAIL_OUT])
    )
    assert report.all_gold is False
    assert report.deterministic is False
    assert report.passed is False
    assert len(report.flipped) == 1


# --------------------------------------------------------------------------- #
# Discovery + image resolution + error handling
# --------------------------------------------------------------------------- #
def test_resolve_tasks_root_accepts_out_dir_with_tasks_subdir(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _make_task_dir(tasks, "acme__ast__id000000000001")
    # Pass the out_dir (parent of tasks/) -> resolves to tasks/.
    assert resolve_tasks_root(tmp_path) == tasks
    # Pass the tasks/ dir directly -> unchanged.
    assert resolve_tasks_root(tasks) == tasks


def test_resolve_eval_image_prefers_env_image_then_base(tmp_path: Path) -> None:
    d = _make_task_dir(tmp_path, "acme__ast__img00000001", image="env:built")
    assert resolve_eval_image(d) == "env:built"

    d2 = tmp_path / "nobuilt"
    d2.mkdir()
    (d2 / "evaluate.sh").write_text("#!/bin/bash\n")
    (d2 / "workspace.yaml").write_text(
        yaml.safe_dump({"environment": {"base_image": "node:22-slim"}})
    )
    assert resolve_eval_image(d2) == "node:22-slim"


def test_missing_image_raises(tmp_path: Path) -> None:
    d = tmp_path / "noimg"
    d.mkdir()
    (d / "evaluate.sh").write_text("#!/bin/bash\n")
    (d / "workspace.yaml").write_text(yaml.safe_dump({"environment": {}}))
    with pytest.raises(GoldEvalError):
        resolve_eval_image(d)


def test_missing_evaluate_script_raises(tmp_path: Path) -> None:
    d = tmp_path / "noscript"
    d.mkdir()
    (d / "workspace.yaml").write_text(yaml.safe_dump({"environment": {"image": "x:y"}}))
    with pytest.raises(GoldEvalError):
        evaluate_task_gold(d, runs=1, runner=_const_runner(_GOLD_OUT))


def test_empty_set_does_not_pass(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    report = run_gold_eval(tasks, runs=2, runner=_const_runner(_GOLD_OUT))
    assert report.shipped_count == 0
    assert report.all_gold is False  # nothing verified is not a pass
    assert report.passed is False


# --------------------------------------------------------------------------- #
# CLI wiring (offline: fake the default Docker runner)
# --------------------------------------------------------------------------- #
def test_cli_gold_eval_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from typer.testing import CliRunner

    from swe_forge.forge.cli import app

    tasks = tmp_path / "tasks"
    for i in range(2):
        _make_task_dir(tasks, f"acme__ast__cli{i:012d}")
    monkeypatch.setattr(ge, "run_evaluate_container", _const_runner(_GOLD_OUT))

    res = CliRunner().invoke(
        app, ["gold-eval", "--tasks-dir", str(tasks), "--runs", "2", "--json"]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["all_gold"] is True
    assert payload["shipped_count"] == 2
    assert payload["passed"] is True


def test_cli_gold_eval_fails_on_non_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from swe_forge.forge.cli import app

    tasks = tmp_path / "tasks"
    _make_task_dir(tasks, "acme__ast__clifail000001")
    monkeypatch.setattr(ge, "run_evaluate_container", _const_runner(_BROKEN_FAIL_OUT))

    res = CliRunner().invoke(app, ["gold-eval", "--tasks-dir", str(tasks)])
    assert res.exit_code == 1


# --------------------------------------------------------------------------- #
# VAL-EXPORT-009/010/011 end-to-end in Docker (real evaluate.sh, gold=1 twice)
# --------------------------------------------------------------------------- #
_FIXTURE_IMAGE = "swe-forge-goldeval-fixture:local"


def _git(args: list[str], cwd: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", *args], cwd=cwd, check=True, env=env, capture_output=True)


def _docker_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "version"], capture_output=True, check=False
            ).returncode
            == 0
        )
    except FileNotFoundError:
        return False


@pytest.mark.integration
def test_evaluate_sh_scores_gold_twice_in_docker(tmp_path: Path) -> None:
    """A fully assembled task's evaluate.sh scores gold 1 across 2 fresh --rm runs."""
    if not _docker_available():
        pytest.skip("docker not available")

    from swe_forge.forge.export import export_forge_task
    from swe_forge.forge.gold_eval import evaluate_task_gold
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
    )

    ts = "2026-01-01T00:00:00+00:00"

    # 1) A tiny gold repo: baseline test (P2P) + a taxed-total function (F2P target).
    work = tmp_path / "src-repo"
    (work / "mathlib").mkdir(parents=True)
    (work / "tests").mkdir()
    (work / "mathlib" / "__init__.py").write_text("")
    (work / "mathlib" / "calc.py").write_text(
        textwrap.dedent(
            """\
            def add(a, b):
                return a + b


            def total_with_tax(items, rate):
                return sum(items) * (100 + rate) // 100
            """
        )
    )
    (work / "tests" / "test_add.py").write_text(
        textwrap.dedent(
            """\
            from mathlib.calc import add


            def test_add():
                assert add(2, 3) == 5
            """
        )
    )
    _git(["init", "-q"], work)
    _git(["add", "-A"], work)
    _git(["commit", "-q", "-m", "gold"], work)
    gold_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # 2) Commit the broken variant (drop the tax) so git authors both diffs for us.
    (work / "mathlib" / "calc.py").write_text(
        textwrap.dedent(
            """\
            def add(a, b):
                return a + b


            def total_with_tax(items, rate):
                return sum(items)
            """
        )
    )
    _git(["add", "-A"], work)
    _git(["commit", "-q", "-m", "broken"], work)
    broken_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def _diff(from_ref: str, to_ref: str) -> str:
        out = subprocess.run(
            ["git", "diff", from_ref, to_ref],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return out if out.endswith("\n") else out + "\n"

    # Forward mutation (gold -> broken) and its inverse gold patch (broken -> gold).
    mutation_patch = _diff(gold_commit, broken_commit)
    oracle_patch = _diff(broken_commit, gold_commit)
    base_commit = gold_commit

    bare = tmp_path / "repo.git"
    _git(["clone", "--bare", "-q", str(work), str(bare)], tmp_path)

    candidate = Candidate(
        language="python",
        generator="ast_mutation",
        target=CandidateTarget(files=("mathlib/calc.py",), symbols=("total_with_tax",)),
        mutation_patch=mutation_patch,
        oracle_patch=oracle_patch,
        difficulty_hint="medium",
        provenance=Provenance(
            generator="ast_mutation", seed=1, language="python", created_at=ts
        ),
    )
    spec = GeneratedSpec(
        problem_statement="total_with_tax must apply the tax rate to the summed items.",
        requirements=["total_with_tax([100], 10) returns 110"],
        interface_block="def total_with_tax(items, rate): ...",
        provenance=Provenance(
            generator="ast_mutation", seed=1, language="python", created_at=ts
        ),
    )
    oracle = OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        reasons=[],
        fail_to_pass=["python -m pytest tests/hidden/test_tax.py -q"],
        pass_to_pass=["python -m pytest tests/test_add.py -q"],
        test_files=[
            OracleTestFile(
                path="tests/hidden/test_tax.py",
                content=(
                    "from mathlib.calc import total_with_tax\n\n\n"
                    "def test_tax():\n    assert total_with_tax([100], 10) == 110\n"
                ),
            )
        ],
        flakiness_runs=3,
        mutants_total=4,
        mutants_killed=4,
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        provenance=Provenance(
            generator="ast_mutation", seed=1, language="python", created_at=ts
        ),
    )
    models = [
        ModelSolveRecord(model="weak/m", tier="weak", k=4, solves=0, pass_at_k=0.0),
        ModelSolveRecord(model="mid/m", tier="mid", k=4, solves=1, pass_at_k=0.25),
        ModelSolveRecord(
            model="frontier/m", tier="frontier", k=4, solves=1, pass_at_k=0.25
        ),
    ]
    calibration = CalibrationReport(
        language="python",
        models=models,
        k=4,
        irt_difficulty=1.0,
        irt_discrimination=1.5,
    )
    calibration.set_band_verdict("keep", "in-band frontier + high discrimination")

    env_image = EnvImage(
        repo_id="mathlib",
        language="python",
        image_tag=_FIXTURE_IMAGE,
        base_image="python:3.12-slim",
        commit=base_commit,
        workspace_dir="/workspace/repo",
        install_commands=[],
        baseline_test_command="python -m pytest tests/test_add.py -q",
        baseline_green=True,
        baseline_exit_code=0,
    )

    from swe_forge.forge.export import assemble_forge_task

    task = assemble_forge_task(
        candidate=candidate,
        spec=spec,
        oracle_report=oracle,
        calibration_report=calibration,
        env_image=env_image,
        repo_url="/srv/repo.git",  # mounted into the container (see extra_args)
        base_commit=base_commit,
    )
    tasks_root = tmp_path / "tasks"
    result = export_forge_task(task, tasks_root, overwrite=True)
    assert result.status == "shipped", result.reason
    task_dir = result.path
    assert task_dir is not None

    # 3) Build a minimal EnvImage-like fixture image (git + pytest present).
    dockerfile = (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir pytest\n"
    )
    subprocess.run(
        ["docker", "build", "-t", _FIXTURE_IMAGE, "-"],
        input=dockerfile,
        text=True,
        check=True,
        capture_output=True,
    )

    # 4) Run the REAL evaluate.sh twice in fresh --rm containers -> gold 1 both times.
    gold = evaluate_task_gold(
        task_dir,
        runs=2,
        image=_FIXTURE_IMAGE,
        extra_args=["-v", f"{bare}:/srv/repo.git:ro"],
        timeout=600.0,
    )
    assert gold.gold is True, [r.stdout for r in gold.runs]
    assert gold.deterministic is True
    assert gold.phase1_all is True
    assert gold.scores == [1, 1]
