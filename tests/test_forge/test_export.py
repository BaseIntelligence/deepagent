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
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from swe_forge.export.jsonl import import_jsonl
from swe_forge.export.parquet import import_parquet
from swe_forge.forge import export as export_mod
from swe_forge.forge.gold_eval import discover_task_dirs
from swe_forge.forge.export import (
    ExportRequest,
    assemble_forge_task,
    audit_exported_workspace,
    audit_git_history,
    build_full_fail_to_pass,
    direct_protected_alt_correct_audit_path,
    direct_protected_teacher_receipts_path,
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
from swe_forge.forge.publication import (
    PublicationError,
    load_published_generation,
    protected_alt_correct_audit_path,
    protected_teacher_receipts_path,
)
from swe_forge.forge.teacher import Usage, candidate_transport_fingerprint
from tests.test_forge.receipt_helpers import (
    protected_alt_correct_audit,
    protected_alt_correct_summary,
    signed_transport_receipt,
)

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


def _alt_correct_audit(
    test_files: list[OracleTestFile] | None = None,
) -> dict[str, object]:
    if test_files is None:
        test_files = [
            OracleTestFile(
                path="tests/hidden/test_total.py",
                content=(
                    "from src.m import total\n\n\n"
                    "def test_total():\n    assert total([100], 0.1) == 110\n"
                ),
            )
        ]
    test_ids = [
        f"python -m pytest {shlex.quote(test_file.path)}" for test_file in test_files
    ]
    return protected_alt_correct_audit(
        test_files,
        test_ids,
        [("src/m.py", "def total(items, tax_rate):\n    return sum(items)\n")],
    )


def _attach_transport_receipts(report: OracleReport, candidate: Candidate) -> None:
    """Make exported pass fixtures use private concrete-transport authority."""
    gates = report.details.get("teacher_gates")
    if not isinstance(gates, dict):
        return
    receipts: list[dict[str, object]] = []
    for gate, payload in gates.items():
        if not isinstance(gate, str) or not isinstance(payload, dict):
            continue
        calls = payload.get("calls")
        if not isinstance(calls, list):
            continue
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or call.get("real_teacher") is not True:
                continue
            call.setdefault("recovery_accounting", None)
            usage = call.get("usage")
            if not isinstance(usage, dict):
                continue
            receipt = signed_transport_receipt(
                call_id=f"{len(receipts) + 1:032x}",
                candidate_fingerprint=candidate_transport_fingerprint(candidate),
                gate=gate,
                call_kind=str(call["call_kind"]),
                model=str(call["model"]),
                usage=Usage(
                    prompt_tokens=int(usage["prompt_tokens"]),
                    completion_tokens=int(usage["completion_tokens"]),
                    total_tokens=int(usage["total_tokens"]),
                ),
                cost=float(call["cost"]),
            )
            call["call_id"] = receipt.call_id
            call["receipt_commitment"] = receipt.commitment
            receipts.append(receipt.to_private_dict())
    report.protected_teacher_transport_receipts = receipts


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
    report = OracleReport(
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
        details={
            "teacher_gates": _teacher_gate_evidence(),
            "alt_correct": protected_alt_correct_summary(test_files),
        },
        protected_alt_correct_audit=_alt_correct_audit(test_files),
    )
    if extra_survivor:
        audit = report.protected_alt_correct_audit
        assert isinstance(audit, dict)
        hidden = [
            {
                "test_id": "python -m pytest tests/hidden/test_total.py",
                "exit_code": 0,
            },
            {
                "test_id": "python -m pytest tests/hidden/test_survivor.py",
                "exit_code": 0,
            },
        ]
        audit["gold"]["hidden"] = hidden  # type: ignore[index]
        audit["alternatives"]["alt_1"]["hidden"] = list(hidden)  # type: ignore[index]
    return report


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
    include_teacher_evidence = bool(overrides.pop("include_teacher_evidence", True))
    fields: dict[str, object] = {
        "candidate": _candidate(),
        "spec": _spec(),
        "oracle_report": _oracle_pass(),
        "calibration_report": _calibration(keep=True),
        "env_image": _env_image(),
        "repo_url": "https://github.com/acme/demo.git",
    }
    fields.update(overrides)
    report = fields["oracle_report"]
    if include_teacher_evidence:
        _attach_transport_receipts(
            report,  # type: ignore[arg-type]
            fields["candidate"],  # type: ignore[arg-type]
        )
    else:
        assert isinstance(report, OracleReport)
        report.details.pop("teacher_gates", None)
        report.protected_teacher_transport_receipts = []
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
    candidate = _candidate()
    _attach_transport_receipts(report, candidate)
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
        candidate=candidate,
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
    report.protected_alt_correct_audit = _alt_correct_audit(report.test_files)
    report.details["alt_correct"] = protected_alt_correct_summary(report.test_files)

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


def test_direct_export_is_discoverable_without_its_internal_store(
    tmp_path: Path,
) -> None:
    """Generic task discovery sees only the direct workspace, not its metadata."""
    task = _task()
    tasks_root = tmp_path / "tasks"

    result = export_forge_task(task, tasks_root, overwrite=True)

    assert result.shipped
    assert result.path is not None
    assert discover_task_dirs(tasks_root) == [result.path]
    assert not (tasks_root / ".forge-task-publications").exists()


def test_direct_export_retains_private_alt_correct_audit_outside_workspace(
    tmp_path: Path,
) -> None:
    task = _task()
    tasks_root = tmp_path / "tasks"

    result = export_forge_task(task, tasks_root, overwrite=True)

    assert result.shipped
    store = export_mod._direct_store_root(tasks_root, task.task_id)
    audits = sorted((store / ".protected-audit").glob("*.json"))
    assert len(audits) == 1
    assert audits[0].stat().st_mode & 0o777 == 0o600
    assert audits[0].parent.stat().st_mode & 0o777 == 0o700
    raw_proposal = task.oracle_report.protected_alt_correct_audit["alternatives"][  # type: ignore[index]
        "alt_1"
    ]["patches"][0]["content"]
    assert all(
        raw_proposal not in path.read_text(encoding="utf-8", errors="ignore")
        for path in result.path.rglob("*")  # type: ignore[union-attr]
        if path.is_file()
    )


def test_batch_export_refuses_malformed_private_alt_audit_before_writing(
    tmp_path: Path,
) -> None:
    request = _request()
    audit = request.oracle_report.protected_alt_correct_audit
    assert isinstance(audit, dict)
    audit["version"] = True

    result = export_batch([request], tmp_path)

    assert result.shipped == []
    assert len(result.refused) == 1
    assert "alt_correct" in result.refused[0].reason
    assert list((tmp_path / "tasks").iterdir()) == []


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


def _workspace_snapshot(workspace: Path) -> dict[str, bytes]:
    """Return the complete visible workspace bytes for overwrite assertions."""
    return {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }


def test_direct_export_roots_do_not_share_private_generations(tmp_path: Path) -> None:
    """Same-id workspaces under sibling task roots remain independently selected."""
    task = _task()
    first_root = tmp_path / "tasks-one"
    second_root = tmp_path / "tasks-two"
    first = export_forge_task(task, first_root, overwrite=True)
    assert first.shipped
    assert first.path is not None
    before = _workspace_snapshot(first.path)

    assert export_forge_task(task, second_root, overwrite=True).shipped
    successor = _task()
    successor.spec.problem_statement = "A root-local successor."
    second = export_forge_task(successor, second_root, overwrite=True)

    assert second.shipped
    assert _workspace_snapshot(first.path) == before


@pytest.mark.parametrize(
    "failure",
    ["workspace_write", "validation", "audit", "generation_rename", "pointer_switch"],
)
def test_direct_overwrite_failure_preserves_prior_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Every direct-publication failure keeps the selected workspace unchanged."""
    tasks_root = tmp_path / "tasks"
    predecessor = _task()
    first = export_forge_task(predecessor, tasks_root, overwrite=True)
    assert first.shipped
    assert first.path is not None
    before = _workspace_snapshot(first.path)

    successor = _task()
    successor.spec.problem_statement = "A distinct replacement workspace."

    if failure == "workspace_write":

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("injected workspace-write failure")

        monkeypatch.setattr(export_mod, "_write_evaluate", _boom)
    elif failure == "validation":

        def _validation_boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("injected validation failure")

        monkeypatch.setattr(export_mod, "_validate_staged_workspace", _validation_boom)
    elif failure == "audit":

        def _audit_boom(*_args: object, **_kwargs: object) -> object:
            raise OSError("injected audit failure")

        monkeypatch.setattr(export_mod, "audit_exported_workspace", _audit_boom)
    else:
        original_replace = export_mod.os.replace

        def _replace(source: object, destination: object) -> None:
            target = Path(destination)
            if (
                failure == "generation_rename" and target.parent.name == "generations"
            ) or (failure == "pointer_switch" and target.name == "current"):
                raise OSError(f"injected {failure} failure")
            original_replace(source, destination)

        monkeypatch.setattr(export_mod.os, "replace", _replace)

    result = export_forge_task(successor, tasks_root, overwrite=True)

    assert result.status == "failed"
    assert first.path.is_dir()
    assert _workspace_snapshot(first.path) == before


def test_direct_overwrite_rejects_unmanaged_legacy_destinations(
    tmp_path: Path,
) -> None:
    """Never delete or migrate legacy files/directories during direct overwrite."""
    task = _task()
    tasks_root = tmp_path / "tasks"
    legacy_dir = tasks_root / task.task_id
    legacy_dir.mkdir(parents=True)
    marker = legacy_dir / "do-not-delete"
    marker.write_text("legacy bytes", encoding="utf-8")

    skipped = export_forge_task(task, tasks_root, overwrite=False)
    assert skipped.status == "skipped"
    assert marker.read_text(encoding="utf-8") == "legacy bytes"

    result = export_forge_task(task, tasks_root, overwrite=True)

    assert result.status == "failed"
    assert marker.read_text(encoding="utf-8") == "legacy bytes"
    assert not (tasks_root / ".forge-task-publications").exists()

    marker.unlink()
    legacy_dir.rmdir()
    legacy_file = tasks_root / task.task_id
    legacy_file.write_text("legacy file bytes", encoding="utf-8")
    result = export_forge_task(task, tasks_root, overwrite=True)

    assert result.status == "failed"
    assert legacy_file.read_text(encoding="utf-8") == "legacy file bytes"
    assert not (tasks_root / ".forge-task-publications").exists()


def test_direct_export_rejects_corrupted_or_cross_task_pointer(tmp_path: Path) -> None:
    """A managed facade accepts only its own validated immutable generation."""
    tasks_root = tmp_path / "tasks"
    first = _task()
    first_result = export_forge_task(first, tasks_root, overwrite=True)
    assert first_result.shipped

    other = _task(candidate=_candidate(seed=73))
    other_result = export_forge_task(other, tasks_root, overwrite=True)
    assert other_result.shipped

    current = (tasks_root / first.task_id).resolve(
        strict=True
    ).parent.parent / "current"
    current.unlink()
    os.symlink(f"../{other.task_id}/generations/not-a-generation", current)

    result = export_forge_task(first, tasks_root, overwrite=False)

    assert result.status == "failed"
    assert "invalid target" in result.reason
    assert not list(current.parent.glob(".staging-*"))


def test_direct_export_refuses_a_truncated_selected_generation(tmp_path: Path) -> None:
    """A selected generation must satisfy the full workspace contract on restart."""
    task = _task()
    tasks_root = tmp_path / "tasks"
    assert export_forge_task(task, tasks_root, overwrite=True).shipped
    selected = (tasks_root / task.task_id).resolve(strict=True)
    (selected / "patch.diff").unlink()

    result = export_forge_task(task, tasks_root, overwrite=False)

    assert result.status == "failed"
    assert "missing required files: patch.diff" in result.reason


def test_direct_export_rejects_unmanaged_generation_store_symlink(
    tmp_path: Path,
) -> None:
    """Incomplete private stores cannot redirect a first direct publication."""
    task = _task()
    tasks_root = tmp_path / "tasks"
    store = export_mod._direct_store_root(tasks_root, task.task_id)
    store.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, store / "generations")

    result = export_forge_task(task, tasks_root, overwrite=True)

    assert result.status == "failed"
    assert "generations store is invalid" in result.reason
    assert not list(outside.iterdir())


def test_direct_export_rejects_symlinked_store_ancestor(tmp_path: Path) -> None:
    """A symlinked private-store ancestor cannot redirect direct publication."""
    task = _task()
    tasks_root = tmp_path / "tasks"
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / ".forge-task-publications")

    result = export_forge_task(task, tasks_root, overwrite=True)

    assert result.status == "failed"
    assert "store ancestor is invalid" in result.reason
    assert not list(outside.iterdir())


def test_direct_export_rejects_unsafe_task_id_before_writing(tmp_path: Path) -> None:
    """Task IDs cannot traverse from a direct task-root into another path."""
    task = _task()
    task.task_id = "../escaped"

    result = export_forge_task(task, tmp_path / "tasks", overwrite=True)

    assert result.status == "failed"
    assert "unsafe export task id" in result.reason
    assert not (tmp_path / "escaped").exists()


def test_batch_export_refuses_unsafe_task_id_before_staging(tmp_path: Path) -> None:
    """Batch and checkpoint assembly reject traversal before a stage is opened."""
    out_dir = tmp_path / "out"

    result = export_batch([_request(task_id="../../escaped")], out_dir)

    assert len(result.refused) == 1
    assert "unsafe export task id" in result.refused[0].reason
    assert not (tmp_path / "escaped").exists()
    assert not (out_dir / "tasks" / "escaped").exists()


def test_direct_overwrite_reports_success_after_committed_pointer_switch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-rename error acknowledges the already-selected complete successor."""
    tasks_root = tmp_path / "tasks"
    predecessor = _task()
    assert export_forge_task(predecessor, tasks_root, overwrite=True).shipped
    successor = _task()
    successor.spec.problem_statement = "Pointer-selected successor."
    original_replace = export_mod.os.replace

    def _replace(source: object, destination: object) -> None:
        result = original_replace(source, destination)
        if Path(destination).name == "current":
            raise OSError("reported after successful pointer replacement")
        return result

    monkeypatch.setattr(export_mod.os, "replace", _replace)
    result = export_forge_task(successor, tasks_root, overwrite=True)

    assert result.shipped
    assert result.path is not None
    assert (
        b"Pointer-selected successor." in (result.path / "workspace.yaml").read_bytes()
    )


@pytest.mark.parametrize(
    ("boundary", "expects_successor"),
    [
        ("before_generation_rename", False),
        ("after_generation_rename", False),
        ("before_pointer_replace", False),
        ("after_pointer_replace", True),
    ],
)
def test_sigkill_direct_overwrite_keeps_a_complete_selected_workspace(
    tmp_path: Path, boundary: str, expects_successor: bool
) -> None:
    """Direct publication exposes only an old or complete new workspace after SIGKILL."""
    tasks_root = tmp_path / "tasks"
    predecessor = _task()
    first = export_forge_task(predecessor, tasks_root, overwrite=True)
    assert first.shipped
    assert first.path is not None
    old_snapshot = _workspace_snapshot(first.path)

    script = r"""
import os
import signal
import sys
from pathlib import Path

from tests.test_forge.test_export import _task
from swe_forge.forge import receipt_authority
from swe_forge.forge import export as export_mod
from swe_forge.forge.export import export_forge_task

receipt_authority.default_authority_root = lambda: Path(
    os.environ["SWE_FORGE_TEST_RECEIPT_AUTHORITY_ROOT"]
)
tasks_root = Path(sys.argv[1])
boundary = sys.argv[2]
original_replace = export_mod.os.replace

def replace(source, destination):
    target = Path(destination)
    if boundary == "before_generation_rename" and target.parent.name == "generations":
        os.kill(os.getpid(), signal.SIGKILL)
    if boundary == "before_pointer_replace" and target.name == "current":
        os.kill(os.getpid(), signal.SIGKILL)
    result = original_replace(source, destination)
    if boundary == "after_generation_rename" and target.parent.name == "generations":
        os.kill(os.getpid(), signal.SIGKILL)
    if boundary == "after_pointer_replace" and target.name == "current":
        os.kill(os.getpid(), signal.SIGKILL)
    return result

export_mod.os.replace = replace
# This subprocess deliberately has no private authority after exec.  Bypass
# receipt validation only in this transaction-boundary probe; authority
# validity is exercised independently in this process and must fail closed
# after a real authority restart.
export_mod.ensure_oracle_exportable = lambda *_args, **_kwargs: None
successor = _task(include_teacher_evidence=False)
successor.spec.problem_statement = "A distinct replacement workspace."
export_forge_task(successor, tasks_root, overwrite=True)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tasks_root), boundary],
        cwd="/projects/Agent-SWE",
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert child.returncode == -signal.SIGKILL

    visible = tasks_root / predecessor.task_id
    assert visible.is_dir()
    current_snapshot = _workspace_snapshot(visible)
    if expects_successor:
        assert (
            b"A distinct replacement workspace." in current_snapshot["workspace.yaml"]
        )
    else:
        assert current_snapshot == old_snapshot

    # A new process ignores abandoned stage directories and reads the selected
    # complete workspace rather than any incomplete staging bytes.
    resumed = export_forge_task(predecessor, tasks_root, overwrite=False)
    # The transaction-only child deliberately bypassed authority validation.
    # If it selected that un-attested successor, a fresh verifier must reject
    # it rather than treating a prior receipt as reusable.
    assert resumed.status == ("failed" if expects_successor else "skipped")
    assert _workspace_snapshot(visible) == current_snapshot


def test_forgetask_round_trips_through_dict() -> None:
    task = _task()
    restored = ForgeTask.from_dict(task.to_dict())
    assert restored.task_id == task.task_id
    assert restored.fail_to_pass == task.fail_to_pass
    assert restored.oracle_report.verdict == "pass"
    assert restored.calibration_report.band_verdict == "keep"


def test_alt_correct_private_audit_persists_without_leaking_to_exports(
    tmp_path: Path,
) -> None:
    result = export_batch([_request()], tmp_path)
    generation = load_published_generation(tmp_path)
    assert generation is not None
    audit_path = protected_alt_correct_audit_path(
        generation.root, result.shipped[0].task_id
    )
    assert audit_path.is_file()
    assert audit_path.stat().st_mode & 0o777 == 0o600
    private_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    raw_proposal = private_audit["alternatives"]["alt_1"]["patches"][0]["content"]
    assert raw_proposal == "def total(items, tax_rate):\n    return sum(items)\n"

    public_files = [
        path
        for path in generation.root.rglob("*")
        if path.is_file() and ".protected-audit" not in path.parts
    ]
    assert public_files
    assert all(
        raw_proposal.rstrip() not in path.read_text(encoding="utf-8", errors="ignore")
        for path in public_files
    )
    assert raw_proposal.rstrip() not in json.dumps(
        generation.entries[0].task.to_dict(), sort_keys=True
    )


def test_published_generation_privileged_loader_rehydrates_alt_correct_audit(
    tmp_path: Path,
) -> None:
    result = export_batch([_request()], tmp_path)

    generation = load_published_generation(tmp_path)

    assert generation is not None
    task = generation.entries[0].task
    assert task.task_id == result.shipped[0].task_id
    assert task.oracle_report.protected_alt_correct_audit == _alt_correct_audit()
    # Rehydration is in-memory only. The manifest remains a public-safe
    # serialization without the protected evidence.
    manifest = (generation.root / "manifest.json").read_text(encoding="utf-8")
    raw_proposal = _alt_correct_audit()["alternatives"]["alt_1"]["patches"][0][
        "content"
    ]
    assert raw_proposal not in manifest
    assert "protected_alt_correct_audit" not in manifest


def test_transport_receipts_are_private_mode_0600_and_absent_from_public_export(
    tmp_path: Path,
) -> None:
    result = export_batch([_request()], tmp_path)
    generation = load_published_generation(tmp_path)
    assert generation is not None

    receipt_path = protected_teacher_receipts_path(
        generation.root, result.shipped[0].task_id
    )
    assert receipt_path.is_file()
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    signature = json.loads(receipt_path.read_text(encoding="utf-8"))[0]["signature"]
    public_files = [
        path
        for path in generation.root.rglob("*")
        if path.is_file() and ".protected-audit" not in path.parts
    ]
    assert public_files
    assert all(
        signature not in path.read_text(encoding="utf-8", errors="ignore")
        for path in public_files
    )


@pytest.mark.parametrize("corruption", ["missing", "altered", "wrong_key", "replayed"])
def test_direct_export_rejects_invalid_protected_teacher_receipts(
    tmp_path: Path, corruption: str
) -> None:
    task = _task()
    tasks_root = tmp_path / "tasks"
    first = export_forge_task(task, tasks_root)
    assert first.shipped
    assert first.path is not None
    receipt_path = direct_protected_teacher_receipts_path(first.path)
    assert receipt_path is not None

    if corruption == "missing":
        receipt_path.unlink()
    else:
        receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
        if corruption == "altered":
            receipts[0]["model"] = "anthropic/other"
        elif corruption == "wrong_key":
            receipts[0]["issuer_key_id"] = "0" * 32
        else:
            receipts.append(dict(receipts[0]))
        receipt_path.write_text(json.dumps(receipts), encoding="utf-8")

    replay = export_forge_task(task, tasks_root)
    assert replay.status == "failed"
    assert "protected teacher receipts" in replay.reason


@pytest.mark.parametrize("corruption", ["missing", "malformed", "mismatched"])
def test_direct_export_rejects_invalid_protected_alt_audit(
    tmp_path: Path, corruption: str
) -> None:
    task = _task()
    tasks_root = tmp_path / "tasks"
    first = export_forge_task(task, tasks_root)
    assert first.shipped
    assert first.path is not None
    audit_path = direct_protected_alt_correct_audit_path(first.path)
    assert audit_path is not None

    if corruption == "missing":
        audit_path.unlink()
    elif corruption == "malformed":
        audit_path.write_text("{broken", encoding="utf-8")
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["final_suite"]["suite_fingerprint"] = "c" * 64
        audit_path.write_text(json.dumps(audit), encoding="utf-8")

    replay = export_forge_task(task, tasks_root)
    assert replay.status == "failed"
    assert "protected alt-correct audit" in replay.reason


@pytest.mark.parametrize("corruption", ["missing", "altered", "wrong_key", "replayed"])
def test_publication_rejects_missing_altered_or_replayed_transport_receipts(
    tmp_path: Path, corruption: str
) -> None:
    result = export_batch([_request()], tmp_path)
    generation = load_published_generation(tmp_path)
    assert generation is not None
    receipt_path = protected_teacher_receipts_path(
        generation.root, result.shipped[0].task_id
    )
    if corruption == "missing":
        receipt_path.unlink()
    else:
        receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
        if corruption == "altered":
            receipts[0]["model"] = "anthropic/other"
        elif corruption == "wrong_key":
            receipts[0]["issuer_key_id"] = "0" * 32
        else:
            receipts.append(dict(receipts[0]))
        receipt_path.write_text(json.dumps(receipts), encoding="utf-8")

    with pytest.raises(PublicationError, match="protected teacher receipts"):
        load_published_generation(tmp_path)


@pytest.mark.parametrize(
    "corruption",
    ["missing", "malformed", "duplicate_key", "mismatched", "patch_mismatch"],
)
def test_hardened_publication_refuses_missing_or_invalid_private_alt_audit(
    tmp_path: Path, corruption: str
) -> None:
    result = export_batch([_request()], tmp_path)
    generation = load_published_generation(tmp_path)
    assert generation is not None
    audit_path = protected_alt_correct_audit_path(
        generation.root, result.shipped[0].task_id
    )
    if corruption == "missing":
        audit_path.unlink()
    elif corruption == "malformed":
        audit_path.write_text("{broken", encoding="utf-8")
    elif corruption == "duplicate_key":
        audit_path.write_text(
            '{"version": 1, "version": 1, "alternatives": {}}',
            encoding="utf-8",
        )
    elif corruption == "mismatched":
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["original_public_suite_sha256"] = "c" * 64
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["alternatives"]["alt_1"]["patches"][0]["content"] = "tampered"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(PublicationError, match="protected alt-correct audit"):
        load_published_generation(tmp_path)


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
    report.protected_alt_correct_audit = _alt_correct_audit(report.test_files)
    report.details["alt_correct"] = protected_alt_correct_summary(report.test_files)


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
    # Batch uses the private writer below its generation stage, not the direct
    # task-scoped publisher.  Nested public stores would make generation
    # validation/recovery depend on multiple independent pointers.
    from swe_forge.forge.publication import load_published_generation

    generation = load_published_generation(tmp_path)
    assert generation is not None
    assert not (generation.tasks_dir / ".forge-task-publications").exists()


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


def test_expected_current_generation_cas_preserves_an_intervening_publication(
    tmp_path: Path,
) -> None:
    """A terminal writer cannot replace a generation selected after its preflight."""
    first = export_batch([_request()], tmp_path, overwrite=True)
    expected = load_published_generation(tmp_path)
    assert expected is not None

    intervening = export_batch(
        [
            _request(
                candidate=_candidate(seed=8),
                repo_url="https://github.com/acme/intervening.git",
            )
        ],
        tmp_path,
        overwrite=True,
    )

    with pytest.raises(
        export_mod.ExportError,
        match="expected current generation .* does not match",
    ):
        export_batch(
            [],
            tmp_path,
            overwrite=True,
            expected_current_generation_id=expected.generation_id,
        )

    selected = load_published_generation(tmp_path)
    assert selected is not None
    assert selected.generation_id != expected.generation_id
    assert {task.id for task in import_jsonl(intervening.jsonl_path)} == {
        entry.task.task_id for entry in selected.entries
    }
    assert first.kept
