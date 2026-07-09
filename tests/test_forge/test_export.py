"""Offline coverage of Stage 5 export (assembly gate + workspace + datasets).

Exercises this feature's ``fulfills`` assertions deterministically without Docker
or the live endpoint:

- VAL-EXPORT-001/002/003/004: the fail-fast export gate (oracle pass AND band
  keep), refusal with no artifacts, and the qualified subset of a mixed batch.
- VAL-EXPORT-005/006/007/008/020: workspace contract, executable self-contained
  evaluate.sh, populated hidden tests, valid patch diffs, benchmark-only layout.
- VAL-EXPORT-012/013: jsonl+parquet one record per kept task, id-set equality,
  lossless round-trip, valid empty export.
- VAL-EXPORT-021/022: leak audit clean / a planted leak blocks shipping (incl.
  the .git history vector).
- VAL-EXPORT-023/024/025/026: deterministic+unique ids, idempotent overwrite,
  preserved-on-skip / no-partial-on-failure, robust 3-way apply in evaluate.sh.

The Docker headline (gold=100% via evaluate.sh) is exercised by this feature's
manual integration check and the user-testing validator.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from swe_forge.export.jsonl import import_jsonl
from swe_forge.export.parquet import import_parquet
from swe_forge.forge import export as export_mod
from swe_forge.forge.export import (
    ExportRequest,
    assemble_forge_task,
    audit_exported_workspace,
    audit_git_history,
    build_full_fail_to_pass,
    export_batch,
    export_forge_task,
    forge_task_id,
    reinit_orphan_git,
)
from swe_forge.forge.models import (
    CalibrationReport,
    Candidate,
    CandidateTarget,
    EnvImage,
    ExportGateError,
    FinalMutationEvidence,
    ForgeTask,
    GeneratedSpec,
    ModelSolveRecord,
    OracleReport,
    OracleTestFile,
    Provenance,
)
from swe_forge.forge.oracle.pipeline import ExportRefusedError
from swe_forge.forge.oracle.mutation import final_suite_fingerprint

_TS = "2026-01-01T00:00:00+00:00"
_GOLD_LINE = "    return compute_total_with_tax(items, tax_rate)"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _provenance() -> Provenance:
    return Provenance(
        generator="ast_mutation", seed=7, language="python", created_at=_TS
    )


def _candidate(*, generator: str = "ast_mutation", seed: int = 7) -> Candidate:
    return Candidate(
        language="python",
        generator=generator,
        target=CandidateTarget(files=("src/m.py",), symbols=("total",)),
        mutation_patch=(
            "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,2 +1,2 @@\n"
            " def total(items, tax_rate):\n"
            f"-{_GOLD_LINE[4:]}\n"
            "+    return sum(items)\n"
        ),
        oracle_patch=(
            "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,2 +1,2 @@\n"
            " def total(items, tax_rate):\n"
            "-    return sum(items)\n"
            f"+{_GOLD_LINE[4:]}\n"
        ),
        difficulty_hint="medium",
        provenance=Provenance(
            generator=generator, seed=seed, language="python", created_at=_TS
        ),
    )


def _env_image() -> EnvImage:
    return EnvImage(
        repo_id="demo-repo",
        language="python",
        image_tag="swe-forge-env-demo:abc123",
        base_image="python:3.12-slim",
        commit="a" * 40,
        workspace_dir="/workspace/repo",
        install_commands=["pip install -e ."],
        baseline_test_command="python -m pytest -q",
        baseline_green=True,
        baseline_exit_code=0,
    )


def _spec(*, problem: str = "") -> GeneratedSpec:
    return GeneratedSpec(
        problem_statement=problem or "total() must include tax in the returned amount.",
        requirements=["total() returns the taxed sum for the items"],
        interface_block="def total(items, tax_rate): ...",
        provenance=_provenance(),
    )


def _oracle_pass(*, extra_survivor: bool = False) -> OracleReport:
    test_files = [
        OracleTestFile(
            path="tests/hidden/test_total.py",
            content="from src.m import total\n\n\ndef test_total():\n    assert total([100], 0.1) == 110\n",
        )
    ]
    if extra_survivor:
        # A mutation/differential-gate survivor-killing test that lives ONLY in
        # test_files[] (not fail_to_pass) -- the export must still enforce it.
        test_files.append(
            OracleTestFile(
                path="tests/hidden/test_survivor.py",
                content="from src.m import total\n\n\ndef test_survivor():\n    assert total([0], 0.2) == 0\n",
                origin="synthesized",
            )
        )
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        reasons=[],
        fail_to_pass=["python -m pytest tests/hidden/test_total.py"],
        pass_to_pass=["python -m pytest -q"],
        test_files=test_files,
        flakiness_runs=3,
        mutants_total=10,
        mutants_killed=10,
        differential_pass=True,
        alt_correct_accepted=True,
        leak_audit="clean",
        final_mutation_evidence=FinalMutationEvidence(
            suite_fingerprint=final_suite_fingerprint(test_files),
            mutants_total=10,
            mutants_killed=10,
            threshold=0.8,
            tool="fake-tool",
        ),
        provenance=_provenance(),
    )


def _oracle_reject() -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="reject",
        reasons=["mutation_failed: induced reject"],
        provenance=_provenance(),
    )


def _calibration(*, keep: bool = True) -> CalibrationReport:
    models = [
        ModelSolveRecord(model="weak/m", tier="weak", k=4, solves=0, pass_at_k=0.0),
        ModelSolveRecord(model="mid/m", tier="mid", k=4, solves=1, pass_at_k=0.25),
        ModelSolveRecord(
            model="frontier/m", tier="frontier", k=4, solves=1, pass_at_k=0.25
        ),
    ]
    report = CalibrationReport(
        language="python",
        models=models,
        k=4,
        irt_difficulty=1.0,
        irt_discrimination=1.5,
    )
    report.set_band_verdict(
        "keep" if keep else "drop",
        "in-band frontier + high discrimination" if keep else "solve-all too easy",
    )
    return report


def _request(**overrides: object) -> ExportRequest:
    fields: dict[str, object] = {
        "candidate": _candidate(),
        "spec": _spec(),
        "oracle_report": _oracle_pass(),
        "calibration_report": _calibration(keep=True),
        "env_image": _env_image(),
        "repo_url": "https://github.com/acme/demo.git",
    }
    fields.update(overrides)
    return ExportRequest(**fields)  # type: ignore[arg-type]


def _task(**overrides: object) -> ForgeTask:
    request = _request(**overrides)
    return assemble_forge_task(
        candidate=request.candidate,
        spec=request.spec,
        oracle_report=request.oracle_report,
        calibration_report=request.calibration_report,
        env_image=request.env_image,
        repo_url=request.repo_url,
    )


# --------------------------------------------------------------------------- #
# VAL-EXPORT-023: deterministic + unique ids
# --------------------------------------------------------------------------- #
def test_task_id_is_deterministic() -> None:
    args = ("acme/demo", "ast_mutation", 7, ("src/m.py",), ("total",))
    assert forge_task_id(*args) == forge_task_id(*args)


def test_task_id_is_unique_per_target() -> None:
    base = ("acme/demo", "ast_mutation", 7, ("src/m.py",), ("total",))
    others = [
        ("acme/demo", "ast_mutation", 8, ("src/m.py",), ("total",)),
        ("acme/demo", "lm_authored", 7, ("src/m.py",), ("total",)),
        ("acme/demo", "ast_mutation", 7, ("src/n.py",), ("total",)),
        ("acme/other", "ast_mutation", 7, ("src/m.py",), ("total",)),
    ]
    ids = {forge_task_id(*base)} | {forge_task_id(*o) for o in others}
    assert len(ids) == 1 + len(others)


# --------------------------------------------------------------------------- #
# VAL-EXPORT-003: fail-fast assembly gate
# --------------------------------------------------------------------------- #
def test_assemble_refuses_oracle_reject() -> None:
    with pytest.raises(ExportRefusedError):
        assemble_forge_task(
            candidate=_candidate(),
            spec=_spec(),
            oracle_report=_oracle_reject(),
            calibration_report=_calibration(keep=True),
            env_image=_env_image(),
            repo_url="https://github.com/acme/demo.git",
        )


def test_assemble_refuses_calibration_drop() -> None:
    with pytest.raises(ExportRefusedError):
        assemble_forge_task(
            candidate=_candidate(),
            spec=_spec(),
            oracle_report=_oracle_pass(),
            calibration_report=_calibration(keep=False),
            env_image=_env_image(),
            repo_url="https://github.com/acme/demo.git",
        )


def test_assemble_refuses_stale_final_mutation_evidence() -> None:
    report = _oracle_pass()
    report.test_files.append(
        OracleTestFile(path="tests/hidden/test_later.py", content="assert True\n")
    )

    with pytest.raises(ExportRefusedError, match="final mutation evidence"):
        assemble_forge_task(
            candidate=_candidate(),
            spec=_spec(),
            oracle_report=report,
            calibration_report=_calibration(keep=True),
            env_image=_env_image(),
            repo_url="https://github.com/acme/demo.git",
        )


def test_assemble_refuses_multifault_without_constituent_metadata_or_proof() -> None:
    report = _oracle_pass()
    report.generator = "multi_file"

    with pytest.raises(ExportRefusedError, match="multifault"):
        assemble_forge_task(
            candidate=_candidate(generator="multi_file"),
            spec=_spec(),
            oracle_report=report,
            calibration_report=_calibration(keep=True),
            env_image=_env_image(),
            repo_url="https://github.com/acme/demo.git",
        )


def test_assemble_accepts_nondefault_final_mutation_threshold() -> None:
    report = _oracle_pass()
    evidence = report.final_mutation_evidence
    assert evidence is not None
    report.final_mutation_evidence = FinalMutationEvidence(
        suite_fingerprint=evidence.suite_fingerprint,
        mutants_total=evidence.mutants_total,
        mutants_killed=evidence.mutants_killed,
        threshold=0.9,
        tool=evidence.tool,
    )

    task = assemble_forge_task(
        candidate=_candidate(),
        spec=_spec(),
        oracle_report=report,
        calibration_report=_calibration(keep=True),
        env_image=_env_image(),
        repo_url="https://github.com/acme/demo.git",
    )

    assert task.oracle_report.final_mutation_evidence.threshold == 0.9


def test_direct_export_refuses_evidence_mismatched_after_assembly(
    tmp_path: Path,
) -> None:
    task = _task()
    # Defend the actual write boundary too, in case a task object is mutated or
    # deserialized after its initial assembly check.
    task.oracle_report.test_files.append(
        OracleTestFile(path="tests/hidden/test_later.py", content="assert True\n")
    )

    result = export_forge_task(task, tmp_path / "tasks")

    assert result.status == "refused"
    assert "final mutation evidence" in result.reason
    assert not (tmp_path / "tasks" / task.task_id).exists()


def test_exported_test_bytes_match_final_mutation_fingerprint(tmp_path: Path) -> None:
    report = _oracle_pass()
    report.test_files[0].content = "assert True"
    report.final_mutation_evidence = FinalMutationEvidence(
        suite_fingerprint=final_suite_fingerprint(report.test_files),
        mutants_total=report.mutants_total,
        mutants_killed=report.mutants_killed,
        threshold=0.8,
        tool="fake-tool",
    )

    result = export_batch([_request(oracle_report=report)], tmp_path)
    task_dir = result.shipped[0].path
    assert task_dir is not None
    exported = task_dir / "tests" / report.test_files[0].path
    assert exported.read_bytes() == b"assert True\n"
    assert report.final_mutation_evidence.suite_fingerprint == final_suite_fingerprint(
        [
            OracleTestFile(
                path=report.test_files[0].path,
                content=exported.read_text(),
            )
        ]
    )


def test_forgetask_direct_construction_enforces_gate() -> None:
    with pytest.raises(ExportGateError):
        ForgeTask(
            task_id="x",
            repo="acme/demo",
            repo_url="https://github.com/acme/demo.git",
            base_commit="a" * 40,
            language="python",
            generator="ast_mutation",
            candidate=_candidate(),
            spec=_spec(),
            oracle_report=_oracle_pass(),
            calibration_report=_calibration(keep=False),
            env_image=_env_image(),
            install_commands=["pip install -e ."],
            fail_to_pass=["python -m pytest tests/hidden/test_total.py"],
            pass_to_pass=["python -m pytest -q"],
            provenance=_provenance(),
        )


# --------------------------------------------------------------------------- #
# VAL-EXPORT-001/005/006/007/008/020: qualified task -> full workspace
# --------------------------------------------------------------------------- #
def test_qualified_task_exports_full_workspace(tmp_path: Path) -> None:
    result = export_batch([_request()], tmp_path)
    assert len(result.shipped) == 1
    task_dir = result.shipped[0].path
    assert task_dir is not None and task_dir.is_dir()

    # VAL-EXPORT-005: required files present + evaluate.sh executable.
    for name in ("workspace.yaml", "patch.diff", "deletion_patch.diff", "evaluate.sh"):
        assert (task_dir / name).is_file(), name
    tests_dir = task_dir / "tests"
    assert tests_dir.is_dir() and any(tests_dir.rglob("*.py"))  # VAL-EXPORT-007
    mode = (task_dir / "evaluate.sh").stat().st_mode & 0o777
    assert mode == 0o755

    evaluate = (task_dir / "evaluate.sh").read_text()
    # Self-contained + robust 3-way apply (VAL-EXPORT-005/026).
    assert "git clone" in evaluate and "git checkout" in evaluate
    assert evaluate.count("git apply --3way") >= 2
    assert "patch.diff" in evaluate and "deletion_patch.diff" in evaluate

    # VAL-EXPORT-008: patch diffs non-empty and end in newline.
    for name in ("patch.diff", "deletion_patch.diff"):
        content = (task_dir / name).read_text()
        assert content.strip() and content.endswith("\n")

    # VAL-EXPORT-006: workspace.yaml carries the task contract.
    import yaml

    data = yaml.safe_load((task_dir / "workspace.yaml").read_text())
    assert data["task_id"] == task_dir.name
    assert data["repo"]["url"] == "https://github.com/acme/demo.git"
    assert data["repo"]["base_commit"] == "a" * 40
    assert data["language"] == "python"
    assert data["install"]["commands"]
    assert data["tests"]["fail_to_pass"] and data["tests"]["pass_to_pass"]
    assert data["synthetic"]["deletion_patch_file"] == "deletion_patch.diff"
    assert data["synthetic"]["strategy"] == "ast_mutation"
    assert (
        data["meta"]["final_mutation_suite_fingerprint"]
        == _oracle_pass().final_mutation_evidence.suite_fingerprint
    )

    provenance = json.loads((task_dir / "provenance.json").read_text())
    assert (
        provenance["details"]["final_mutation_suite_fingerprint"]
        == _oracle_pass().final_mutation_evidence.suite_fingerprint
    )

    # VAL-EXPORT-020: solution/tests under forge_path, repo under repo_path.
    assert data["environment"]["repo_path"] == "/workspace/repo"
    assert data["environment"]["forge_path"] == "/workspace/forge"
    assert data["solution"]["path"] == "/workspace/forge"
    assert not data["solution"]["path"].startswith(data["environment"]["repo_path"])


def test_full_test_files_enforced_in_workspace_and_evaluate(tmp_path: Path) -> None:
    # A survivor-killing test lives ONLY in test_files[] (not fail_to_pass).
    request = _request(oracle_report=_oracle_pass(extra_survivor=True))
    result = export_batch([request], tmp_path)
    task_dir = result.shipped[0].path
    assert task_dir is not None

    # Shipped tests/ contains BOTH hidden tests.
    shipped = {p.name for p in (task_dir / "tests").rglob("*.py")}
    assert {"test_total.py", "test_survivor.py"} <= shipped

    # evaluate.sh + workspace.yaml enforce the FULL set, not just the original F2P.
    evaluate = (task_dir / "evaluate.sh").read_text()
    assert "tests/hidden/test_total.py" in evaluate
    assert "tests/hidden/test_survivor.py" in evaluate

    adapter_f2p = build_full_fail_to_pass(
        __import__("swe_forge.forge.adapters", fromlist=["build_default_registry"])
        .build_default_registry()
        .get("python"),
        request.oracle_report.fail_to_pass,
        request.oracle_report.test_files,
    )
    assert any("test_survivor.py" in cmd for cmd in adapter_f2p)


# --------------------------------------------------------------------------- #
# VAL-EXPORT-002/004: refusal + mixed batch
# --------------------------------------------------------------------------- #
def test_unqualified_tasks_refused_with_no_artifacts(tmp_path: Path) -> None:
    reject = _request(oracle_report=_oracle_reject())
    drop = _request(
        candidate=_candidate(seed=99), calibration_report=_calibration(keep=False)
    )
    result = export_batch([reject, drop], tmp_path)

    assert result.shipped == []
    assert len(result.refused) == 2
    assert list((tmp_path / "tasks").iterdir()) == []
    # Empty datasets exist and are valid.
    assert (tmp_path / "dataset.jsonl").read_text() == ""
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 0


def test_mixed_batch_ships_only_qualified_subset(tmp_path: Path) -> None:
    good = _request()
    reject = _request(candidate=_candidate(seed=11), oracle_report=_oracle_reject())
    drop = _request(
        candidate=_candidate(seed=22), calibration_report=_calibration(keep=False)
    )
    result = export_batch([good, reject, drop], tmp_path)

    assert len(result.shipped) == 1
    assert len(result.refused) == 2
    shipped_dirs = list((tmp_path / "tasks").iterdir())
    assert len(shipped_dirs) == 1
    # One refused id never appears anywhere.
    refused_ids = {r.task_id for r in result.refused}
    assert all(d.name not in refused_ids for d in shipped_dirs)


# --------------------------------------------------------------------------- #
# VAL-EXPORT-012/013: datasets one record per kept task, round-trip
# --------------------------------------------------------------------------- #
def test_datasets_one_record_per_task_with_idset_equality(tmp_path: Path) -> None:
    good_a = _request()
    good_b = _request(
        candidate=_candidate(seed=2), repo_url="https://github.com/acme/two.git"
    )
    export_batch([good_a, good_b], tmp_path)

    task_dirs = {p.name for p in (tmp_path / "tasks").iterdir()}
    jsonl_tasks = import_jsonl(tmp_path / "dataset.jsonl")
    parquet_rows = import_parquet(tmp_path / "dataset.parquet")

    assert len(task_dirs) == 2
    assert len(jsonl_tasks) == 2
    assert len(parquet_rows) == 2

    jsonl_ids = {t.id for t in jsonl_tasks}
    parquet_ids = {r["id"] for r in parquet_rows}
    assert jsonl_ids == parquet_ids == task_dirs

    # Lossless round-trip of list + map fields.
    row = parquet_rows[0]
    assert isinstance(row["install_config"], dict)
    assert isinstance(row["meta"], dict)
    assert isinstance(row["fail_to_pass"], list) and row["fail_to_pass"]
    task = next(t for t in jsonl_tasks if t.id == row["id"])
    assert task.fail_to_pass == row["fail_to_pass"]
    assert task.pass_to_pass == row["pass_to_pass"]
    assert task.install_config == row["install_config"]
    assert task.meta == row["meta"]


def test_empty_export_writes_valid_empty_artifacts(tmp_path: Path) -> None:
    result = export_batch([], tmp_path)
    assert result.shipped == []
    assert (tmp_path / "dataset.jsonl").read_text() == ""
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 0
    assert list((tmp_path / "tasks").iterdir()) == []


# --------------------------------------------------------------------------- #
# VAL-EXPORT-021/022: leak audit clean / planted leak blocks shipping
# --------------------------------------------------------------------------- #
def test_leak_audit_clean_on_exported_tree(tmp_path: Path) -> None:
    result = export_batch([_request()], tmp_path)
    task_dir = result.shipped[0].path
    assert task_dir is not None
    audit = audit_exported_workspace(
        task_dir,
        oracle_patch=_candidate().oracle_patch,
        test_files=_oracle_pass().test_files,
    )
    assert audit.passed is True
    assert audit.findings == []
    assert audit.risk_score == 0.0


def test_planted_leak_blocks_shipping(tmp_path: Path) -> None:
    # The spec leaks a verbatim gold line -> it lands in workspace.yaml prompt.
    leaky = _request(spec=_spec(problem=f"Implement total. Gold:\n{_GOLD_LINE}\n"))
    result = export_batch([leaky], tmp_path)

    assert result.shipped == []
    assert len(result.refused) == 1
    assert result.refused[0].status == "refused"
    assert any("oracle" in f.lower() for f in result.refused[0].leak_findings)
    assert list((tmp_path / "tasks").iterdir()) == []


def test_planted_forbidden_artifact_detected(tmp_path: Path) -> None:
    result = export_batch([_request()], tmp_path)
    task_dir = result.shipped[0].path
    assert task_dir is not None
    (task_dir / "solution.patch").write_text("the gold solution\n")
    audit = audit_exported_workspace(
        task_dir,
        oracle_patch=_candidate().oracle_patch,
        test_files=_oracle_pass().test_files,
    )
    assert audit.passed is False
    assert audit.findings


# --------------------------------------------------------------------------- #
# .git history vector (AGENTS.md "No gold leak via .git history")
# --------------------------------------------------------------------------- #
def test_git_history_vector_detects_and_orphan_reinit_clears(tmp_path: Path) -> None:
    repo = tmp_path / "task" / "repo"
    repo.mkdir(parents=True)
    git = ["git", "-C", str(repo)]
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    (repo / "m.py").write_text("GOLD\n")
    subprocess.run([*git, "add", "-A"], check=True, env=env)
    subprocess.run([*git, "commit", "-q", "-m", "gold"], check=True, env=env)
    (repo / "m.py").write_text("BROKEN\n")
    subprocess.run([*git, "add", "-A"], check=True, env=env)
    subprocess.run([*git, "commit", "-q", "-m", "broken"], check=True, env=env)

    # Two commits -> gold recoverable via HEAD~1.
    findings = audit_git_history(tmp_path / "task")
    assert findings and "git_history" in findings[0]

    # Re-init to a single orphan commit -> clean.
    reinit_orphan_git(repo)
    assert audit_git_history(tmp_path / "task") == []
    rev = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD~1"],
        capture_output=True,
        text=True,
    )
    assert rev.returncode != 0  # git show HEAD~1 must fail


# --------------------------------------------------------------------------- #
# VAL-EXPORT-024/025: idempotent overwrite / skip / no-partial-on-failure
# --------------------------------------------------------------------------- #
def test_reexport_overwrite_is_idempotent(tmp_path: Path) -> None:
    requests = [_request()]
    first = export_batch(requests, tmp_path, overwrite=True)
    second = export_batch(requests, tmp_path, overwrite=True)

    ids_first = {p.name for p in (tmp_path / "tasks").iterdir()}
    ids_second = {p.name for p in (tmp_path / "tasks").iterdir()}
    assert ids_first == ids_second
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 1
    assert len(first.kept) == len(second.kept) == 1


def test_existing_dir_without_overwrite_is_skipped(tmp_path: Path) -> None:
    task = _task()
    tasks_root = tmp_path / "tasks"
    first = export_forge_task(task, tasks_root, overwrite=True)
    assert first.status == "shipped"
    sentinel = first.path / "SENTINEL"  # type: ignore[union-attr]
    sentinel.write_text("keep me")

    second = export_forge_task(task, tasks_root, overwrite=False)
    assert second.status == "skipped"
    assert sentinel.read_text() == "keep me"  # preserved, not half-overwritten


def test_failed_midwrite_leaves_no_partial_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task()
    tasks_root = tmp_path / "tasks"

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected mid-write failure")

    monkeypatch.setattr(export_mod, "_write_evaluate", _boom)
    result = export_forge_task(task, tasks_root, overwrite=True)

    assert result.status == "failed"
    assert not (tasks_root / task.task_id).exists()
    # No leftover temp dirs either.
    assert list(tasks_root.iterdir()) == []


def test_forgetask_round_trips_through_dict() -> None:
    task = _task()
    restored = ForgeTask.from_dict(task.to_dict())
    assert restored.task_id == task.task_id
    assert restored.fail_to_pass == task.fail_to_pass
    assert restored.oracle_report.verdict == "pass"
    assert restored.calibration_report.band_verdict == "keep"


# --------------------------------------------------------------------------- #
# Transactional publication safety (m6-export-publication-safety)
# --------------------------------------------------------------------------- #
def _set_hidden_test_path(report: OracleReport, path: str) -> None:
    """Keep the fixture's final-suite evidence valid after replacing its path."""
    original = report.test_files[0]
    report.test_files[0] = OracleTestFile(
        path=path, content=original.content, origin=original.origin
    )
    evidence = report.final_mutation_evidence
    assert evidence is not None
    report.final_mutation_evidence = FinalMutationEvidence(
        suite_fingerprint=final_suite_fingerprint(report.test_files),
        mutants_total=evidence.mutants_total,
        mutants_killed=evidence.mutants_killed,
        threshold=evidence.threshold,
        tool=evidence.tool,
    )


@pytest.mark.parametrize("path", ["/tmp/forge-escape.py", "../forge-escape.py"])
def test_unsafe_hidden_test_paths_refuse_before_any_workspace_write(
    tmp_path: Path, path: str
) -> None:
    report = _oracle_pass()
    outside = tmp_path / "forge-escape.py"
    _set_hidden_test_path(report, str(outside) if path.startswith("/") else path)

    result = export_batch([_request(oracle_report=report)], tmp_path / "out")

    assert len(result.refused) == 1
    assert result.refused[0].status in ("refused", "failed")
    assert not outside.exists()
    assert not (tmp_path / "out" / "tasks" / result.refused[0].task_id).exists()


def test_evaluator_quotes_a_canonical_hidden_path_with_spaces(tmp_path: Path) -> None:
    report = _oracle_pass()
    path = "tests/hidden/test total.py"
    _set_hidden_test_path(report, path)
    result = export_batch([_request(oracle_report=report)], tmp_path)

    task_dir = result.shipped[0].path
    assert task_dir is not None
    script = (task_dir / "evaluate.sh").read_text(encoding="utf-8")
    assert "'tests/hidden/test total.py'" in script
    assert "python -m pytest 'tests/hidden/test total.py'" in script


def test_identical_duplicate_task_ids_ship_exactly_one_artifact_row(
    tmp_path: Path,
) -> None:
    result = export_batch([_request(), _request()], tmp_path, overwrite=True)

    assert [entry.status for entry in result.results] == ["shipped", "deduplicated"]
    assert len(result.kept) == 1
    assert len(list((tmp_path / "tasks").iterdir())) == 1
    assert len(import_jsonl(tmp_path / "dataset.jsonl")) == 1
    assert len(import_parquet(tmp_path / "dataset.parquet")) == 1


def test_conflicting_duplicate_task_id_aborts_without_mutating_output(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    first = _request(task_id="duplicate-task")
    second = _request(
        task_id="duplicate-task",
        candidate=_candidate(seed=999),
    )

    with pytest.raises(export_mod.ExportError, match="conflicting duplicate task_id"):
        export_batch([first, second], out, overwrite=True)

    assert not out.exists()


def test_failed_generation_keeps_the_prior_complete_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swe_forge.forge import publication

    first = export_batch([_request()], tmp_path, overwrite=True)
    before = publication.load_published_generation(tmp_path)
    assert before is not None
    before_ids = {task.id for task in import_jsonl(first.jsonl_path)}

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected generation validation failure")

    monkeypatch.setattr(publication, "_validate_staged_generation", _boom)
    with pytest.raises(OSError, match="injected generation validation failure"):
        export_batch(
            [
                _request(),
                _request(
                    candidate=_candidate(seed=8),
                    repo_url="https://github.com/acme/two.git",
                ),
            ],
            tmp_path,
            overwrite=True,
        )

    monkeypatch.undo()
    after = publication.load_published_generation(tmp_path)
    assert after is not None
    assert after.generation_id == before.generation_id
    assert {task.id for task in import_jsonl(tmp_path / "dataset.jsonl")} == before_ids
    assert not list((tmp_path / ".forge-publications").glob(".staging-*"))
