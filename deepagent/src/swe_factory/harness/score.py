"""Harness scoring: gold resolves, null does not (VAL-HARNESS-001).

Wraps the oracle Docker evaluation backend (or injectable FakeOracleRunner)
and exposes a simple resolve metric: all F2P pass AND all P2P pass after the
candidate patch is applied.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_factory.oracle.docker_run import (
    OracleDockerError,
    OracleRunnerBackend,
    SuiteOutcome,
)
from swe_factory.schema import TaskRecord


class HarnessError(RuntimeError):
    """Unrecoverable harness scoring failure."""


@dataclass(frozen=True, slots=True)
class HarnessScoreResult:
    """Single candidate evaluation result."""

    instance_id: str
    label: str
    resolve: bool
    score: float
    f2p_exits: tuple[int, ...] = ()
    p2p_exits: tuple[int, ...] = ()
    patch_applied: bool = False
    phase: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "resolve": self.resolve,
            "score": self.score,
            "f2p_exits": list(self.f2p_exits),
            "p2p_exits": list(self.p2p_exits),
            "patch_applied": self.patch_applied,
            "phase": self.phase,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class GoldNullPair:
    """Gold + null rescoring summary (VAL-HARNESS-001)."""

    instance_id: str
    gold: HarnessScoreResult
    null: HarnessScoreResult

    @property
    def passed(self) -> bool:
        """True only when gold resolves and null does not."""
        return self.gold.resolve is True and self.null.resolve is False

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "passed": self.passed,
            "gold": self.gold.to_dict(),
            "null": self.null.to_dict(),
        }


def _from_suite(
    *,
    instance_id: str,
    label: str,
    suite: SuiteOutcome,
) -> HarnessScoreResult:
    resolved = suite.resolve()
    return HarnessScoreResult(
        instance_id=instance_id,
        label=label,
        resolve=resolved,
        score=1.0 if resolved else 0.0,
        f2p_exits=tuple(c.exit_code for c in suite.f2p),
        p2p_exits=tuple(c.exit_code for c in suite.p2p),
        patch_applied=suite.patch_applied,
        phase=suite.phase,
        details={
            "all_f2p_passed": suite.all_f2p_passed(),
            "all_p2p_passed": suite.all_p2p_passed(),
            "container_name": suite.container_name,
        },
    )


def score_candidate(
    *,
    task: TaskRecord,
    workspace: Path | str,
    patch: str,
    runner: OracleRunnerBackend,
    label: str = "candidate",
    fail_to_pass: Sequence[str] | None = None,
    pass_to_pass: Sequence[str] | None = None,
) -> HarnessScoreResult:
    """Score one candidate patch against a task workspace.

    Does **not** call ``runner.cleanup()``; callers that own the runner
    lifecycle (e.g. ``score_gold_and_null``) should cleanup.
    """
    ws = Path(workspace)
    if not ws.is_dir():
        raise HarnessError(f"workspace not found: {ws}")
    f2p = list(fail_to_pass if fail_to_pass is not None else task.fail_to_pass)
    p2p = list(pass_to_pass if pass_to_pass is not None else task.pass_to_pass)
    if not f2p:
        raise HarnessError(f"task {task.instance_id!r} has empty fail_to_pass")

    try:
        suite = runner.run_with_patch(
            workspace=ws,
            patch=patch,
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            phase=label,
        )
    except OracleDockerError as exc:
        raise HarnessError(f"score_candidate failed for {label}: {exc}") from exc

    return _from_suite(instance_id=task.instance_id, label=label, suite=suite)


def score_gold_and_null(
    *,
    task: TaskRecord,
    workspace: Path | str,
    runner: OracleRunnerBackend,
    cleanup: bool = True,
) -> GoldNullPair:
    """Rescore gold (must resolve) and null/empty patch (must not resolve).

    VAL-HARNESS-001: gold resolve=true, null resolve=false.
    Always attempts runner.cleanup() when ``cleanup`` is True.
    """
    try:
        gold = score_candidate(
            task=task,
            workspace=workspace,
            patch=task.gold_patch,
            runner=runner,
            label="gold",
        )
        null = score_candidate(
            task=task,
            workspace=workspace,
            patch="",
            runner=runner,
            label="null",
        )
        return GoldNullPair(instance_id=task.instance_id, gold=gold, null=null)
    finally:
        if cleanup:
            with contextlib.suppress(Exception):
                runner.cleanup()


__all__ = [
    "GoldNullPair",
    "HarnessError",
    "HarnessScoreResult",
    "score_candidate",
    "score_gold_and_null",
]
