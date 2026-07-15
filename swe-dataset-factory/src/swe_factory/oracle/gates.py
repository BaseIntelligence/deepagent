"""Mechanical oracle gates G0–G5.

Two paths:
- ``run_stub_gates`` / ``evaluate_stub_gate_fields`` — offline structural wiring
  (no Docker; used by offline-fixture demo).
- ``run_certified_gates`` — live mechanical G1–G5 for cert-path evaluation
  (Docker or injectable backend; dual gold run + flake reject + audit codes).

Stable reason codes live in :mod:`swe_factory.oracle.codes` (VAL-ORACLE-006).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.oracle import codes as C
from swe_factory.oracle.docker_run import (
    OracleDockerError,
    OracleRunnerBackend,
    SuiteOutcome,
    scan_agent_workspace_leak,
)
from swe_factory.schema import TaskRecord

# Unified-diff-ish file headers: "diff --git a/path b/path" or "+++ b/path"
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_PLUS_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+)$", re.MULTILINE)

# Minimum source files for V1 hard set (G4)
MULTI_FILE_FLOOR: int = 2
GOLD_DUAL_RUNS: int = 2


def count_files_in_patch(gold_patch: str) -> list[str]:
    """Return unique file paths touched by a unified-style gold patch."""
    files: list[str] = []
    seen: set[str] = set()
    for match in _DIFF_GIT_RE.finditer(gold_patch):
        path = match.group(2).strip()
        if path and path not in seen and path != "/dev/null":
            seen.add(path)
            files.append(path)
    if not files:
        for match in _PLUS_FILE_RE.finditer(gold_patch):
            path = match.group(1).strip()
            if path.startswith("b/"):
                path = path[2:]
            if path and path not in seen and path != "/dev/null":
                seen.add(path)
                files.append(path)
    return files


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of running stub or certified oracle gates on a candidate."""

    passed: bool
    reason_codes: tuple[str, ...]
    reasons: tuple[str, ...]
    multi_file: bool
    files_touched: int
    mode: str = "stub_offline"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.passed

    def to_gate_proof(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "passed": self.passed,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "multi_file": self.multi_file,
            "files_touched": self.files_touched,
            "details": dict(self.details),
        }

    def to_audit_row(self, instance_id: str) -> dict[str, Any]:
        return {
            "instance_id": instance_id,
            "disposition": "accept" if self.passed else "reject",
            "reason_codes": list(self.reason_codes),
            "mode": self.mode,
            "multi_file": self.multi_file,
            "files_touched": self.files_touched,
        }


def append_gate_audit(
    path: Path | str,
    result: GateResult,
    instance_id: str,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append one JSONL gate audit row (VAL-ORACLE-006)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = result.to_audit_row(instance_id)
    if extra:
        row.update(extra)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out


# ---------------------------------------------------------------------------
# Stub path (offline fixture wiring)
# ---------------------------------------------------------------------------


def evaluate_stub_gate_fields(
    *,
    gold_patch: str,
    fail_to_pass: list[str],
    problem_statement: str,
    image_digest: str,
    require_multi_file: bool = True,
) -> GateResult:
    """Evaluate structural G1–G4 style checks offline without Docker/LLM.

    Stub semantics (wiring proof only):
    - G1: F2P list non-empty (proxy for "broken fails" blueprint present)
    - G2: gold patch multi-hunk / non-empty (proxy for gold dual-run shape)
    - G3: gold is non-empty so null is distinct
    - G4: multi-file hard floor when require_multi_file
    """
    codes: list[str] = []
    reasons: list[str] = []
    files = count_files_in_patch(gold_patch)
    multi = len(files) >= MULTI_FILE_FLOOR

    f2p = [c.strip() for c in fail_to_pass if str(c).strip()]
    if not f2p:
        codes.append(C.G1_EMPTY_F2P)
        reasons.append("fail_to_pass is empty; broken-state F2P blueprint missing")
    else:
        codes.append(C.G1_F2P_PRESENT)
        reasons.append(f"G1 stub: {len(f2p)} fail_to_pass command(s) present")

    gold = gold_patch.strip()
    if not gold:
        codes.append(C.G2_EMPTY_GOLD)
        reasons.append("gold_patch empty; gold dual-run cannot pass")
        codes.append(C.G3_NULL_GOLD)
        reasons.append("null/empty patch would already be the gold; scores undistinguished")
    else:
        codes.append(C.G2_GOLD_PRESENT)
        reasons.append("G2 stub: gold patch present for dual-run path")
        codes.append(C.G3_NON_NULL_GOLD)
        reasons.append("G3 stub: gold is non-empty so null patch is distinct")

    if not problem_statement.strip():
        codes.append(C.G_PROMPT_EMPTY)
        reasons.append("problem_statement empty")
    else:
        codes.append(C.G_PROMPT_PRESENT)
        reasons.append("agent prompt present")

    if not image_digest.strip():
        codes.append(C.G0_MISSING_DIGEST)
        reasons.append("environment.image_digest missing")
    else:
        codes.append(C.G0_DIGEST_PRESENT)
        reasons.append("environment.image_digest present (stub G0)")

    if require_multi_file:
        if multi:
            codes.append(C.G4_MULTI_FILE_OK)
            reasons.append(f"G4 multi-file floor satisfied ({len(files)} files)")
        else:
            codes.append(C.G4_MULTI_FILE)
            reasons.append(
                f"G4 multi-file floor failed: gold touches {len(files)} file(s): {files}"
            )

    hard_rejects = {
        C.G1_EMPTY_F2P,
        C.G2_EMPTY_GOLD,
        C.G3_NULL_GOLD,
        C.G4_MULTI_FILE,
        C.G_PROMPT_EMPTY,
        C.G0_MISSING_DIGEST,
    }
    passed = not any(c in hard_rejects for c in codes)
    if passed:
        codes.append(C.STUB_PASS)
        reasons.append("stub gates accept candidate")
    else:
        codes.append(C.STUB_REJECT)
        reasons.append("stub gates reject candidate")

    return GateResult(
        passed=passed,
        reason_codes=tuple(codes),
        reasons=tuple(reasons),
        multi_file=multi,
        files_touched=len(files),
        mode="stub_offline",
        details={"files": files, "f2p_count": len(f2p)},
    )


def run_stub_gates(task: TaskRecord, *, require_multi_file: bool = True) -> GateResult:
    """Run stub gates against a TaskRecord (no Docker, no providers)."""
    return evaluate_stub_gate_fields(
        gold_patch=task.gold_patch,
        fail_to_pass=list(task.fail_to_pass),
        problem_statement=task.problem_statement,
        image_digest=task.environment.image_digest,
        require_multi_file=require_multi_file,
    )


# ---------------------------------------------------------------------------
# Certified path (mechanical G1–G5)
# ---------------------------------------------------------------------------


def evaluate_multi_file_floor(
    gold_patch: str, *, min_files: int = MULTI_FILE_FLOOR
) -> tuple[bool, list[str], str, str]:
    """G4: return (ok, files, reason_code, reason_text)."""
    files = count_files_in_patch(gold_patch)
    if len(files) >= min_files:
        return True, files, C.G4_MULTI_FILE_OK, f"G4 multi-file floor ok ({len(files)} files)"
    return (
        False,
        files,
        C.G4_MULTI_FILE,
        f"G4 multi-file floor failed: gold touches {len(files)} file(s): {files}",
    )


def evaluate_leak_gate(
    *,
    agent_workspace: Path | None,
    gold_patch: str,
) -> tuple[bool, list[str], str, str]:
    """G5: agent mount must not leak gold patch content."""
    findings = scan_agent_workspace_leak(agent_workspace, gold_patch=gold_patch)
    if findings:
        return False, findings, C.G5_LEAK, f"G5 leak scan failed: {findings[0]}"
    return True, findings, C.G5_LEAK_CLEAN, "G5 agent mount leak scan clean"


def _suite_summary(suite: SuiteOutcome) -> dict[str, Any]:
    return {
        "phase": suite.phase,
        "f2p_exits": [c.exit_code for c in suite.f2p],
        "p2p_exits": [c.exit_code for c in suite.p2p],
        "all_f2p_failed": suite.all_f2p_failed(),
        "all_f2p_passed": suite.all_f2p_passed(),
        "all_p2p_passed": suite.all_p2p_passed(),
        "resolve": suite.resolve(),
        "patch_applied": suite.patch_applied,
    }


def run_certified_gates(
    *,
    gold_patch: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str] = (),
    problem_statement: str = "",
    image_digest: str = "",
    workspace: Path | str,
    runner: OracleRunnerBackend,
    agent_workspace: Path | str | None = None,
    require_multi_file: bool = True,
    dual_runs: int = GOLD_DUAL_RUNS,
    check_null_patch: bool = True,
    check_leak: bool = True,
    min_files: int = MULTI_FILE_FLOOR,
) -> GateResult:
    """Certified-path mechanical gates G1–G5 (VAL-ORACLE-001..006).

    Enforces in order:
    - Structural G0/prompt/F2P non-empty
    - G4 multi-file floor on gold
    - G1 every F2P fails on broken workspace
    - G2 gold applied dual-run: all F2P+P2P pass both times; mismatch → flake reject
    - G3 null/empty patch does not resolve
    - G5 optional agent workspace leak scan

    Always attempts ``runner.cleanup()`` in a finally block.
    """
    codes: list[str] = []
    reasons: list[str] = []
    details: dict[str, Any] = {}
    f2p = [c.strip() for c in fail_to_pass if str(c).strip()]
    p2p = [c.strip() for c in pass_to_pass if str(c).strip()]
    gold = gold_patch
    ws = Path(workspace)
    agent_ws = Path(agent_workspace) if agent_workspace is not None else None

    try:
        if not f2p:
            codes.append(C.G1_EMPTY_F2P)
            reasons.append("fail_to_pass is empty")
            codes.append(C.ORACLE_REJECT)
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=False,
                files_touched=0,
                mode="certified",
                details=details,
            )

        if not gold.strip():
            codes.append(C.G2_EMPTY_GOLD)
            reasons.append("gold_patch empty")
            codes.append(C.ORACLE_REJECT)
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=False,
                files_touched=0,
                mode="certified",
                details=details,
            )

        if not problem_statement.strip():
            codes.append(C.G_PROMPT_EMPTY)
            reasons.append("problem_statement empty")
        else:
            codes.append(C.G_PROMPT_PRESENT)
            reasons.append("agent prompt present")

        if not image_digest.strip():
            codes.append(C.G0_MISSING_DIGEST)
            reasons.append("environment.image_digest missing")
        else:
            codes.append(C.G0_DIGEST_PRESENT)
            reasons.append("environment.image_digest present")

        multi_ok, files, g4_code, g4_reason = evaluate_multi_file_floor(
            gold, min_files=min_files if require_multi_file else 1
        )
        codes.append(g4_code)
        reasons.append(g4_reason)
        details["files"] = files
        multi = len(files) >= MULTI_FILE_FLOOR

        if require_multi_file and not multi_ok:
            codes.append(C.ORACLE_REJECT)
            reasons.append("oracle reject: multi-file floor")
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=multi,
                files_touched=len(files),
                mode="certified",
                details=details,
            )

        # G1: broken F2P must all fail
        try:
            broken = runner.run_broken(workspace=ws, fail_to_pass=f2p, pass_to_pass=p2p)
        except OracleDockerError as exc:
            codes.append(C.G1_F2P_NOT_FAILING)
            reasons.append(f"G1 broken evaluation failed: {exc}")
            codes.append(C.ORACLE_REJECT)
            details["error"] = str(exc)
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=multi,
                files_touched=len(files),
                mode="certified",
                details=details,
            )
        details["broken"] = _suite_summary(broken)
        if not broken.all_f2p_failed():
            codes.append(C.G1_F2P_NOT_FAILING)
            reasons.append(
                "G1 fail: not every F2P fails on broken state "
                f"(exits={[c.exit_code for c in broken.f2p]})"
            )
            codes.append(C.ORACLE_REJECT)
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=multi,
                files_touched=len(files),
                mode="certified",
                details=details,
            )
        codes.append(C.G1_F2P_FAIL_OK)
        reasons.append(f"G1 ok: all {len(f2p)} F2P fail on broken state")

        # G2: gold dual-run
        gold_runs: list[SuiteOutcome] = []
        signatures: list[tuple[Any, ...]] = []
        try:
            for i in range(max(1, dual_runs)):
                suite = runner.run_with_patch(
                    workspace=ws,
                    patch=gold,
                    fail_to_pass=f2p,
                    pass_to_pass=p2p,
                    phase=f"gold_{i + 1}",
                )
                gold_runs.append(suite)
                signatures.append(suite.outcome_signature())
        except OracleDockerError as exc:
            codes.append(C.G2_GOLD_FAIL)
            reasons.append(f"G2 gold evaluation failed: {exc}")
            codes.append(C.ORACLE_REJECT)
            details["error"] = str(exc)
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=multi,
                files_touched=len(files),
                mode="certified",
                details=details,
            )
        details["gold_runs"] = [_suite_summary(s) for s in gold_runs]
        details["gold_signatures"] = [list(s) for s in signatures]

        if len(set(signatures)) > 1:
            codes.append(C.G2_FLAKE)
            codes.append(C.FLAKE_REJECT)
            reasons.append(f"flake: gold dual-run outcome mismatch signatures={signatures}")
            codes.append(C.ORACLE_REJECT)
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=multi,
                files_touched=len(files),
                mode="certified",
                details=details,
            )

        if not all(s.resolve() for s in gold_runs):
            codes.append(C.G2_GOLD_FAIL)
            reasons.append(
                "G2 gold dual-run failed strict resolve "
                f"(summaries={[_suite_summary(s) for s in gold_runs]})"
            )
            codes.append(C.ORACLE_REJECT)
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=multi,
                files_touched=len(files),
                mode="certified",
                details=details,
            )
        codes.append(C.G2_GOLD_DUAL_PASS)
        reasons.append(f"G2 ok: gold resolved on {len(gold_runs)} independent runs")

        # G3: null/empty patch must not resolve
        if check_null_patch:
            try:
                null_suite = runner.run_with_patch(
                    workspace=ws,
                    patch="",
                    fail_to_pass=f2p,
                    pass_to_pass=p2p,
                    phase="null",
                )
            except OracleDockerError as exc:
                # Distinct from G3_NULL_RESOLVES: infrastructure/eval failure,
                # not proof that the null patch actually scores as resolve=true.
                codes.append(C.G3_NULL_EVAL_ERROR)
                reasons.append(f"G3 null evaluation error (not a resolve verdict): {exc}")
                codes.append(C.ORACLE_REJECT)
                details["error"] = str(exc)
                details["g3_kind"] = "eval_error"
                return GateResult(
                    passed=False,
                    reason_codes=tuple(codes),
                    reasons=tuple(reasons),
                    multi_file=multi,
                    files_touched=len(files),
                    mode="certified",
                    details=details,
                )
            details["null"] = _suite_summary(null_suite)
            if null_suite.resolve():
                codes.append(C.G3_NULL_RESOLVES)
                reasons.append("G3 fail: null/empty patch resolves (F2P all pass)")
                codes.append(C.ORACLE_REJECT)
                details["g3_kind"] = "resolves"
                return GateResult(
                    passed=False,
                    reason_codes=tuple(codes),
                    reasons=tuple(reasons),
                    multi_file=multi,
                    files_touched=len(files),
                    mode="certified",
                    details=details,
                )
            codes.append(C.G3_NULL_NOT_RESOLVE)
            reasons.append("G3 ok: null/empty patch does not resolve")
            details["g3_kind"] = "not_resolve"

        # G5: optional leak scan on agent-visible mount
        if check_leak:
            leak_ok, findings, g5_code, g5_reason = evaluate_leak_gate(
                agent_workspace=agent_ws, gold_patch=gold
            )
            codes.append(g5_code)
            reasons.append(g5_reason)
            details["leak_findings"] = findings
            if not leak_ok:
                codes.append(C.ORACLE_REJECT)
                return GateResult(
                    passed=False,
                    reason_codes=tuple(codes),
                    reasons=tuple(reasons),
                    multi_file=multi,
                    files_touched=len(files),
                    mode="certified",
                    details=details,
                )

        # Structural fail-closed codes that may have been recorded earlier
        if C.G_PROMPT_EMPTY in codes or C.G0_MISSING_DIGEST in codes:
            codes.append(C.ORACLE_REJECT)
            reasons.append("oracle reject: structural prerequisites missing")
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=multi,
                files_touched=len(files),
                mode="certified",
                details=details,
            )

        if any(c in C.HARD_REJECT_CODES for c in codes):
            codes.append(C.ORACLE_REJECT)
            reasons.append("oracle reject: hard-reject code present")
            return GateResult(
                passed=False,
                reason_codes=tuple(codes),
                reasons=tuple(reasons),
                multi_file=multi,
                files_touched=len(files),
                mode="certified",
                details=details,
            )

        codes.append(C.ORACLE_PASS)
        reasons.append("certified oracle gates accept candidate")
        return GateResult(
            passed=True,
            reason_codes=tuple(codes),
            reasons=tuple(reasons),
            multi_file=multi,
            files_touched=len(files),
            mode="certified",
            details=details,
        )
    finally:
        with contextlib.suppress(Exception):
            runner.cleanup()


def run_certified_gates_for_task(
    task: TaskRecord,
    *,
    workspace: Path | str,
    runner: OracleRunnerBackend,
    agent_workspace: Path | str | None = None,
    require_multi_file: bool = True,
    dual_runs: int = GOLD_DUAL_RUNS,
    check_null_patch: bool = True,
    check_leak: bool = True,
) -> GateResult:
    """Run certified gates using TaskRecord fields."""
    return run_certified_gates(
        gold_patch=task.gold_patch,
        fail_to_pass=list(task.fail_to_pass),
        pass_to_pass=list(task.pass_to_pass),
        problem_statement=task.problem_statement,
        image_digest=task.environment.image_digest,
        workspace=workspace,
        runner=runner,
        agent_workspace=agent_workspace,
        require_multi_file=require_multi_file,
        dual_runs=dual_runs,
        check_null_patch=check_null_patch,
        check_leak=check_leak,
    )


__all__ = [
    "GOLD_DUAL_RUNS",
    "MULTI_FILE_FLOOR",
    "GateResult",
    "append_gate_audit",
    "count_files_in_patch",
    "evaluate_leak_gate",
    "evaluate_multi_file_floor",
    "evaluate_stub_gate_fields",
    "run_certified_gates",
    "run_certified_gates_for_task",
    "run_stub_gates",
]
