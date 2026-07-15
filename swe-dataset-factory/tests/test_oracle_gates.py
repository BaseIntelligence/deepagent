"""Unit tests for certified oracle gates G1–G5 (VAL-ORACLE-001..006).

Uses FakeOracleRunner — no Docker daemon required.
Stable reason codes must remain the audit vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path

from swe_factory.oracle import codes as C
from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite
from swe_factory.oracle.gates import (
    GateResult,
    append_gate_audit,
    count_files_in_patch,
    evaluate_multi_file_floor,
    evaluate_stub_gate_fields,
    run_certified_gates,
    run_stub_gates,
)
from swe_factory.schema import EnvironmentMeta, SourceTrack, TaskRecord


def _multi_gold() -> str:
    return (
        "diff --git a/demo_pkg/math_ops.py b/demo_pkg/math_ops.py\n"
        "--- a/demo_pkg/math_ops.py\n"
        "+++ b/demo_pkg/math_ops.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def add(a: int, b: int) -> int:\n"
        "-    return a - b\n"
        "+    return a + b\n"
        "\n"
        "diff --git a/demo_pkg/text_ops.py b/demo_pkg/text_ops.py\n"
        "--- a/demo_pkg/text_ops.py\n"
        "+++ b/demo_pkg/text_ops.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def reverse_words(text: str) -> str:\n"
        "-    return text\n"
        '+    return " ".join(reversed(text.split()))\n'
    )


def _single_gold() -> str:
    return "diff --git a/only.py b/only.py\n--- a/only.py\n+++ b/only.py\n@@ -1 +1 @@\n-x\n+y\n"


def _task(**kwargs: object) -> TaskRecord:
    base = {
        "instance_id": "t__oracle_unit",
        "source_track": SourceTrack.SYNTHETIC_GROUNDED,
        "repo": "fixtures/tiny_offline",
        "base_commit": "abc123",
        "language": "python",
        "problem_statement": "fix two modules",
        "fail_to_pass": ["python -m pytest tests/test_math.py -q"],
        "pass_to_pass": ["python -m pytest tests/test_ok.py -q"],
        "gold_patch": _multi_gold(),
        "environment": EnvironmentMeta(image_digest="sha256:unit"),
        "license": "MIT",
    }
    base.update(kwargs)
    return TaskRecord.model_validate(base)


def _passing_runner() -> FakeOracleRunner:
    return FakeOracleRunner(
        broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        gold_runs=[
            ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
            ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
        ],
        null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
    )


def test_count_files_multi_file_gold() -> None:
    files = count_files_in_patch(_multi_gold())
    assert len(files) >= 2
    assert "demo_pkg/math_ops.py" in files
    assert "demo_pkg/text_ops.py" in files


def test_g4_multi_file_floor_reject_single() -> None:
    ok, files, code, _ = evaluate_multi_file_floor(_single_gold())
    assert not ok
    assert code == C.G4_MULTI_FILE
    assert len(files) < 2


def test_certified_pass_path(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    runner = _passing_runner()
    result = run_certified_gates(
        gold_patch=_multi_gold(),
        fail_to_pass=["pytest f2p"],
        pass_to_pass=["pytest p2p"],
        problem_statement="fix it",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        agent_workspace=tmp_path / "agent_empty",
        check_leak=True,
    )
    assert result.passed is True
    assert result.mode == "certified"
    assert C.G1_F2P_FAIL_OK in result.reason_codes
    assert C.G2_GOLD_DUAL_PASS in result.reason_codes
    assert C.G3_NULL_NOT_RESOLVE in result.reason_codes
    assert C.G4_MULTI_FILE_OK in result.reason_codes
    assert C.G5_LEAK_CLEAN in result.reason_codes
    assert C.ORACLE_PASS in result.reason_codes
    assert runner.cleaned is True


def test_g1_rejects_when_f2p_pass_on_broken(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    runner = FakeOracleRunner(
        broken=ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),  # F2P wrongly green
        gold_runs=[ScriptedSuite([0], [0]), ScriptedSuite([0], [0])],
        null=ScriptedSuite([1], [0]),
    )
    result = run_certified_gates(
        gold_patch=_multi_gold(),
        fail_to_pass=["pytest f2p"],
        pass_to_pass=["pytest p2p"],
        problem_statement="fix",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        check_leak=False,
    )
    assert result.passed is False
    assert C.G1_F2P_NOT_FAILING in result.reason_codes
    assert C.ORACLE_REJECT in result.reason_codes


def test_g2_rejects_when_gold_does_not_resolve(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    runner = FakeOracleRunner(
        broken=ScriptedSuite([1], [0]),
        gold_runs=[ScriptedSuite([1], [0]), ScriptedSuite([1], [0])],
        null=ScriptedSuite([1], [0]),
    )
    result = run_certified_gates(
        gold_patch=_multi_gold(),
        fail_to_pass=["pytest f2p"],
        pass_to_pass=["pytest p2p"],
        problem_statement="fix",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        check_leak=False,
    )
    assert result.passed is False
    assert C.G2_GOLD_FAIL in result.reason_codes


def test_flake_reject_on_gold_dual_mismatch(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    runner = FakeOracleRunner(
        broken=ScriptedSuite([1], [0]),
        gold_runs=[
            ScriptedSuite([0], [0]),  # pass
            ScriptedSuite([1], [0]),  # flake on second
        ],
        null=ScriptedSuite([1], [0]),
    )
    result = run_certified_gates(
        gold_patch=_multi_gold(),
        fail_to_pass=["pytest f2p"],
        pass_to_pass=["pytest p2p"],
        problem_statement="fix",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        check_leak=False,
    )
    assert result.passed is False
    assert C.G2_FLAKE in result.reason_codes
    assert C.FLAKE_REJECT in result.reason_codes
    assert C.ORACLE_REJECT in result.reason_codes


def test_g3_null_patch_must_not_resolve(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    # Null resolves = F2P all green without gold — gate must reject
    runner = FakeOracleRunner(
        broken=ScriptedSuite([1], [0]),
        gold_runs=[ScriptedSuite([0], [0]), ScriptedSuite([0], [0])],
        null=ScriptedSuite([0], [0]),  # null wrongly resolves
    )
    result = run_certified_gates(
        gold_patch=_multi_gold(),
        fail_to_pass=["pytest f2p"],
        pass_to_pass=["pytest p2p"],
        problem_statement="fix",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        check_leak=False,
    )
    assert result.passed is False
    assert C.G3_NULL_RESOLVES in result.reason_codes


def test_g4_reject_single_file_before_docker(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    runner = _passing_runner()
    result = run_certified_gates(
        gold_patch=_single_gold(),
        fail_to_pass=["pytest f2p"],
        pass_to_pass=["pytest p2p"],
        problem_statement="fix",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        check_leak=False,
    )
    assert result.passed is False
    assert C.G4_MULTI_FILE in result.reason_codes
    # Fake runner never exercised (G4 is structural first)
    assert runner._gold_i == 0


def test_g5_leak_rejects_when_gold_file_present(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "gold.patch").write_text(_multi_gold(), encoding="utf-8")
    runner = _passing_runner()
    result = run_certified_gates(
        gold_patch=_multi_gold(),
        fail_to_pass=["pytest f2p"],
        pass_to_pass=["pytest p2p"],
        problem_statement="fix",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        agent_workspace=agent,
        check_leak=True,
    )
    assert result.passed is False
    assert C.G5_LEAK in result.reason_codes


def test_gate_audit_jsonl_stable_codes(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    runner = _passing_runner()
    result = run_certified_gates(
        gold_patch=_multi_gold(),
        fail_to_pass=["pytest f2p"],
        pass_to_pass=[],
        problem_statement="fix",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        check_leak=False,
    )
    audit = tmp_path / "gate_audit.jsonl"
    append_gate_audit(audit, result, "t__audit", extra={"stage": "oracle"})
    row = json.loads(audit.read_text(encoding="utf-8").strip())
    assert row["instance_id"] == "t__audit"
    assert row["disposition"] == "accept"
    assert row["mode"] == "certified"
    assert isinstance(row["reason_codes"], list)
    for code in row["reason_codes"]:
        assert isinstance(code, str) and code
    # Stable names expected
    assert "G1_F2P_FAIL_OK" in row["reason_codes"]
    assert "G2_GOLD_DUAL_PASS" in row["reason_codes"]
    assert "G3_NULL_NOT_RESOLVE" in row["reason_codes"]
    assert row["stage"] == "oracle"


def test_stub_gates_still_work_via_task() -> None:
    task = _task()
    gate: GateResult = run_stub_gates(task)
    assert gate.passed
    assert gate.mode == "stub_offline"


def test_stable_code_constants_exported() -> None:
    # Audit vocabulary must not silently rename
    assert C.G1_F2P_FAIL_OK == "G1_F2P_FAIL_OK"
    assert C.G2_GOLD_DUAL_PASS == "G2_GOLD_DUAL_PASS"
    assert C.G2_FLAKE == "G2_FLAKE"
    assert C.FLAKE_REJECT == "FLAKE_REJECT"
    assert C.G3_NULL_NOT_RESOLVE == "G3_NULL_NOT_RESOLVE"
    assert C.G4_MULTI_FILE == "G4_MULTI_FILE"
    assert C.G5_LEAK == "G5_LEAK"


def test_empty_f2p_structural_reject(tmp_path: Path) -> None:
    # Certified path short-circuits before runner for empty F2P
    ws = tmp_path / "repo"
    ws.mkdir()
    runner = _passing_runner()
    result = run_certified_gates(
        gold_patch=_multi_gold(),
        fail_to_pass=[],
        problem_statement="fix",
        image_digest="sha256:x",
        workspace=ws,
        runner=runner,
        check_leak=False,
    )
    assert result.passed is False
    assert C.G1_EMPTY_F2P in result.reason_codes


def test_stub_evaluate_fields_rejects_empty_f2p() -> None:
    result = evaluate_stub_gate_fields(
        gold_patch=_multi_gold(),
        fail_to_pass=[],
        problem_statement="x",
        image_digest="sha256:x",
    )
    assert result.passed is False
    assert C.G1_EMPTY_F2P in result.reason_codes
