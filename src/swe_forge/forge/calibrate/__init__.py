"""Stage 4 calibration: difficulty + discrimination over a model panel.

Calibration measures how hard a manufactured task is by letting a panel of solver
models attempt it and scoring how often they succeed, through the SAME Docker
FAIL->PASS primitives the oracle gates use. This package starts with the agentic
solver scaffold (:mod:`swe_forge.forge.calibrate.solver`), which runs a bounded
tool loop on the broken tree + spec surface and scores the submitted patch, then
the panel runner (:mod:`swe_forge.forge.calibrate.runner`), which issues ``k``
independent rollouts per model and records the per-model pass@k.
"""

from swe_forge.forge.calibrate.irt import (
    DEFAULT_TIER_ABILITIES,
    DIFFICULTY_MAX,
    DISCRIMINATION_MAX,
    IrtError,
    IrtFit,
    build_calibration_report,
    fit_irt,
    pass_at_k,
    to_solve_records,
)
from swe_forge.forge.calibrate.runner import (
    DEFAULT_BUDGET,
    CalibrationRun,
    CalibrationRunnerError,
    ModelCalibration,
    RolloutBudget,
    RolloutFn,
    ValidatorFn,
    compute_pass_at_k,
    run_panel_calibration,
    suppress_litellm_async_warning,
)
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
    "DEFAULT_BUDGET",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_TIER_ABILITIES",
    "DIFFICULTY_MAX",
    "DISCRIMINATION_MAX",
    "REASON_APPLY_FAILED",
    "REASON_EMPTY_PATCH",
    "REASON_F2P_OR_P2P_FAILED",
    "REASON_NOT_FINISHED",
    "REASON_ROLLOUT_ERROR",
    "REASON_SCORE_ERROR",
    "AgenticSolver",
    "CalibrationRun",
    "CalibrationRunnerError",
    "DockerPatchScorer",
    "IrtError",
    "IrtFit",
    "ModelCalibration",
    "RolloutBudget",
    "RolloutFn",
    "RolloutOutcome",
    "SolveScore",
    "SolverContext",
    "SolverError",
    "SolverRollout",
    "ValidatorFn",
    "build_calibration_report",
    "build_solver_prompt",
    "capture_workspace_patch",
    "compute_pass_at_k",
    "fit_irt",
    "pass_at_k",
    "run_panel_calibration",
    "run_solver_rollout",
    "score_patch",
    "solver_tools",
    "suppress_litellm_async_warning",
    "to_solve_records",
]
