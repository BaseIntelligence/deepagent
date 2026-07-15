"""Oracle validity gates (mechanical G0–G5)."""

from swe_factory.oracle.codes import HARD_REJECT_CODES
from swe_factory.oracle.docker_run import (
    FakeOracleRunner,
    OracleDockerError,
    OracleDockerRunner,
    ScriptedSuite,
    SuiteOutcome,
)
from swe_factory.oracle.gates import (
    GateResult,
    append_gate_audit,
    count_files_in_patch,
    evaluate_stub_gate_fields,
    run_certified_gates,
    run_certified_gates_for_task,
    run_stub_gates,
)

__all__ = [
    "HARD_REJECT_CODES",
    "FakeOracleRunner",
    "GateResult",
    "OracleDockerError",
    "OracleDockerRunner",
    "ScriptedSuite",
    "SuiteOutcome",
    "append_gate_audit",
    "count_files_in_patch",
    "evaluate_stub_gate_fields",
    "run_certified_gates",
    "run_certified_gates_for_task",
    "run_stub_gates",
]
