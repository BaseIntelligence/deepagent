"""Stage 4 calibration: difficulty + discrimination over a model panel.

Calibration measures how hard a manufactured task is by letting a panel of solver
models attempt it and scoring how often they succeed, through the SAME Docker
FAIL->PASS primitives the oracle gates use. This package starts with the agentic
solver scaffold (:mod:`swe_forge.forge.calibrate.solver`), which runs a bounded
tool loop on the broken tree + spec surface and scores the submitted patch.
"""

from swe_forge.forge.calibrate.solver import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    REASON_APPLY_FAILED,
    REASON_EMPTY_PATCH,
    REASON_F2P_OR_P2P_FAILED,
    REASON_NOT_FINISHED,
    REASON_ROLLOUT_ERROR,
    REASON_SCORE_ERROR,
    AgenticSolver,
    DockerPatchScorer,
    RolloutOutcome,
    SolveScore,
    SolverContext,
    SolverError,
    SolverRollout,
    build_solver_prompt,
    capture_workspace_patch,
    run_solver_rollout,
    score_patch,
    solver_tools,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TURNS",
    "REASON_APPLY_FAILED",
    "REASON_EMPTY_PATCH",
    "REASON_F2P_OR_P2P_FAILED",
    "REASON_NOT_FINISHED",
    "REASON_ROLLOUT_ERROR",
    "REASON_SCORE_ERROR",
    "AgenticSolver",
    "DockerPatchScorer",
    "RolloutOutcome",
    "SolveScore",
    "SolverContext",
    "SolverError",
    "SolverRollout",
    "build_solver_prompt",
    "capture_workspace_patch",
    "run_solver_rollout",
    "score_patch",
    "solver_tools",
]
