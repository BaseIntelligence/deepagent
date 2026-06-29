"""Stage 4 difficulty calibration: pass@k + a 2-parameter IRT fit.

The panel runner (:mod:`swe_forge.forge.calibrate.runner`) measures, for each
solver model, how many of ``k`` independent rollouts solved the task. This module
turns that per-model/per-rollout *solve matrix* into the two numbers that decide
whether a task is *hard but discriminating*:

* ``pass_at_k`` -- the documented per-model estimator ``solves / k`` (clamped to
  ``[0, 1]``). It is a correct nondecreasing function of ``solves`` at a fixed
  ``k``: zero solves yields exactly ``0.0`` and any solve yields ``> 0``
  (VAL-CAL-011). This is the single source of truth for pass@k; the runner
  imports it from here.
* A **2-parameter IRT** fit (``irt_difficulty`` + ``irt_discrimination``) over the
  solve matrix (VAL-CAL-012/013). We anchor each model's latent *ability* by its
  panel tier (``weak < mid < frontier``) and fit a one-covariate logistic model
  ``P(solve) = sigmoid(a * (theta - b))`` by penalized maximum likelihood. The
  slope ``a`` is the **discrimination** (how sharply solve-rate rises with ability
  -> a task only the strong tiers solve fits a steep slope -> high discrimination;
  a flat task fits a near-zero slope -> low discrimination), and ``b`` is the
  **difficulty** (the ability at which the solve probability is 0.5). Both are
  derived from the matrix, so a materially different matrix yields different
  params. Degenerate all-pass / all-fail matrices are handled without raising and
  routed to well-defined sentinels (very low / very high difficulty, zero
  discrimination).

The teacher LLM proposes; deterministic counting disposes: nothing here calls a
model. It consumes the recorded solves and produces numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from swe_forge.forge.models import (
    CalibrationReport,
    ModelSolveRecord,
    Provenance,
)

#: Latent ability anchored to each panel tier (weak < mid < frontier). The fit
#: estimates the item's difficulty/discrimination against these fixed abilities,
#: so "weak fails, frontier solves" maps to a steep (high-discrimination) slope.
DEFAULT_TIER_ABILITIES: dict[str, float] = {
    "weak": -1.0,
    "mid": 0.0,
    "frontier": 1.0,
}

#: Difficulty is reported on the ability scale and clamped to ``[-DIFFICULTY_MAX,
#: +DIFFICULTY_MAX]``; the sentinels for all-pass / all-fail use the edges.
DIFFICULTY_MAX: float = 6.0

#: Discrimination (the fitted slope) is clamped to ``[0, DISCRIMINATION_MAX]`` so
#: a near-separating matrix stays finite (and a reverse-separating matrix maps to
#: 0, i.e. "no useful discrimination").
DISCRIMINATION_MAX: float = 6.0

#: Ridge penalty applied during the fit. Tiny, so it barely perturbs a normal
#: fit, but it keeps the 2x2 Newton system invertible under (near-)perfect
#: separation instead of letting the coefficients run to infinity.
_RIDGE: float = 1e-3

#: Newton-Raphson (IRLS) iteration controls for the logistic fit.
_MAX_ITERATIONS: int = 100
_CONVERGENCE_TOL: float = 1e-9

#: Bound on the linear predictor before the logistic transform, to avoid
#: ``math.exp`` overflow when the slope grows large under separation.
_ETA_CLAMP: float = 30.0


class IrtError(ValueError):
    """Raised when the IRT fitter is given structurally invalid inputs."""


class SolveMatrixRow(Protocol):
    """Structural view of one model's solve record the fitter consumes.

    Both :class:`swe_forge.forge.models.ModelSolveRecord` and the runner's
    ``ModelCalibration`` satisfy this, so either can be fed to :func:`fit_irt` /
    :func:`build_calibration_report` without conversion.
    """

    @property
    def model(self) -> str: ...

    @property
    def tier(self) -> str: ...

    @property
    def k(self) -> int: ...

    @property
    def solves(self) -> int: ...


def pass_at_k(solves: int, k: int) -> float:
    """Per-model pass@k estimator ``solves / k`` clamped to ``[0, 1]``.

    The documented baseline estimator (VAL-CAL-011): a correct nondecreasing
    function of ``solves`` at fixed ``k`` where ``solves == 0`` gives exactly
    ``0.0`` and any ``solves >= 1`` gives ``> 0``. A degenerate ``k <= 0`` (no
    rollouts issued) is defined as ``0.0``.
    """
    if k <= 0 or solves <= 0:
        return 0.0
    return min(1.0, solves / k)


#: Back-compatible alias; the runner historically imported ``compute_pass_at_k``.
compute_pass_at_k = pass_at_k


def _sigmoid(eta: float) -> float:
    """Numerically stable logistic transform with a clamped argument."""
    if eta >= _ETA_CLAMP:
        return 1.0 - 1e-15
    if eta <= -_ETA_CLAMP:
        return 1e-15
    return 1.0 / (1.0 + math.exp(-eta))


@dataclass(frozen=True)
class IrtFit:
    """The fitted 2-parameter IRT result over a solve matrix.

    ``difficulty`` is the ability at which solve probability is 0.5 (higher =
    harder); ``discrimination`` is the slope of solve-rate against ability
    (higher = the task separates strong from weak models more sharply). ``degenerate``
    flags an all-pass / all-fail matrix routed to a sentinel; ``converged`` /
    ``iterations`` expose the fit's numerical state for provenance.
    """

    difficulty: float
    discrimination: float
    n_models: int
    n_trials: int
    total_solves: int
    converged: bool
    iterations: int
    degenerate: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "discrimination": self.discrimination,
            "n_models": self.n_models,
            "n_trials": self.n_trials,
            "total_solves": self.total_solves,
            "converged": self.converged,
            "iterations": self.iterations,
            "degenerate": self.degenerate,
        }


def _resolve_abilities(
    rows: Sequence[SolveMatrixRow],
    tier_abilities: dict[str, float],
) -> list[tuple[float, int, int]]:
    """Map rows to ``(ability, solves, trials)`` triples; validate tiers/counts."""
    triples: list[tuple[float, int, int]] = []
    for row in rows:
        if row.tier not in tier_abilities:
            raise IrtError(
                f"no ability anchor for tier {row.tier!r}; "
                f"known tiers: {sorted(tier_abilities)}"
            )
        if row.k < 0 or row.solves < 0 or row.solves > row.k:
            raise IrtError(
                f"invalid solve record for {row.model!r}: "
                f"solves={row.solves}, k={row.k}"
            )
        if row.k == 0:
            continue
        triples.append((float(tier_abilities[row.tier]), int(row.solves), int(row.k)))
    return triples


def fit_irt(
    rows: Sequence[SolveMatrixRow],
    *,
    tier_abilities: dict[str, float] | None = None,
) -> IrtFit:
    """Fit a 2-parameter IRT (difficulty + discrimination) over the solve matrix.

    Anchors each model's ability by its tier and fits the one-covariate logistic
    model ``P = sigmoid(a*(theta - b))`` by ridge-penalized Newton-Raphson. Returns
    the slope as ``discrimination`` and the 0.5-crossing ability as ``difficulty``,
    both clamped to their reporting range. All-pass / all-fail (and no-variation)
    matrices are routed to well-defined sentinels without raising.
    """
    abilities = tier_abilities or DEFAULT_TIER_ABILITIES
    triples = _resolve_abilities(rows, abilities)
    if not triples:
        raise IrtError("cannot fit IRT: the solve matrix has no rollouts")

    n_models = len(triples)
    n_trials = sum(n for _, _, n in triples)
    total_solves = sum(y for _, y, _ in triples)

    # Degenerate matrices: no information to fit a slope. Route to a sentinel
    # difficulty (all solved -> very easy; none solved -> very hard) with zero
    # discrimination, never raising (VAL-CAL-015).
    if total_solves == 0:
        return IrtFit(
            difficulty=DIFFICULTY_MAX,
            discrimination=0.0,
            n_models=n_models,
            n_trials=n_trials,
            total_solves=total_solves,
            converged=True,
            iterations=0,
            degenerate=True,
        )
    if total_solves == n_trials:
        return IrtFit(
            difficulty=-DIFFICULTY_MAX,
            discrimination=0.0,
            n_models=n_models,
            n_trials=n_trials,
            total_solves=total_solves,
            converged=True,
            iterations=0,
            degenerate=True,
        )

    thetas = [t for t, _, _ in triples]
    theta_spread = max(thetas) - min(thetas)
    if theta_spread == 0.0:
        # Only one ability level present: the slope is unidentifiable. Report zero
        # discrimination and a difficulty derived from the pooled solve-rate.
        rate = total_solves / n_trials
        logit = math.log(rate / (1.0 - rate))
        difficulty = _clamp(thetas[0] - logit, -DIFFICULTY_MAX, DIFFICULTY_MAX)
        return IrtFit(
            difficulty=difficulty,
            discrimination=0.0,
            n_models=n_models,
            n_trials=n_trials,
            total_solves=total_solves,
            converged=True,
            iterations=0,
            degenerate=False,
        )

    alpha, beta, iterations, converged = _newton_logistic(triples)

    discrimination = _clamp(beta, 0.0, DISCRIMINATION_MAX)
    # difficulty is the ability where P = 0.5, i.e. alpha + beta * b = 0.
    if abs(beta) < 1e-12:
        rate = total_solves / n_trials
        logit = math.log(rate / (1.0 - rate))
        difficulty = _clamp(-logit, -DIFFICULTY_MAX, DIFFICULTY_MAX)
    else:
        difficulty = _clamp(-alpha / beta, -DIFFICULTY_MAX, DIFFICULTY_MAX)

    return IrtFit(
        difficulty=difficulty,
        discrimination=discrimination,
        n_models=n_models,
        n_trials=n_trials,
        total_solves=total_solves,
        converged=converged,
        iterations=iterations,
        degenerate=False,
    )


def _newton_logistic(
    triples: Sequence[tuple[float, int, int]],
) -> tuple[float, float, int, bool]:
    """Ridge-penalized Newton-Raphson for ``logit(P) = alpha + beta*theta``.

    Each triple ``(theta, solves, trials)`` contributes ``trials`` Bernoulli
    observations at covariate ``theta``. Returns ``(alpha, beta, iterations,
    converged)``; the 2x2 normal equations are solved in closed form and the tiny
    ridge keeps the system invertible under (near-)separation.
    """
    alpha = 0.0
    beta = 0.0
    converged = False
    iteration = 0
    for iteration in range(1, _MAX_ITERATIONS + 1):
        g0 = -_RIDGE * alpha
        g1 = -_RIDGE * beta
        h00 = _RIDGE
        h01 = 0.0
        h11 = _RIDGE
        for theta, solves, trials in triples:
            p = _sigmoid(alpha + beta * theta)
            resid = solves - trials * p
            g0 += resid
            g1 += theta * resid
            w = trials * p * (1.0 - p)
            h00 += w
            h01 += w * theta
            h11 += w * theta * theta
        det = h00 * h11 - h01 * h01
        if det <= 0.0:
            break
        d_alpha = (g0 * h11 - g1 * h01) / det
        d_beta = (g1 * h00 - g0 * h01) / det
        alpha += d_alpha
        beta += d_beta
        alpha = _clamp(alpha, -50.0, 50.0)
        beta = _clamp(beta, -50.0, 50.0)
        if abs(d_alpha) < _CONVERGENCE_TOL and abs(d_beta) < _CONVERGENCE_TOL:
            converged = True
            break
    return alpha, beta, iteration, converged


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_solve_records(
    rows: Sequence[SolveMatrixRow],
) -> list[ModelSolveRecord]:
    """Project any solve-matrix rows to validated :class:`ModelSolveRecord`s.

    ``pass_at_k`` is (re)computed from ``solves``/``k`` via the single-source
    estimator, so the records carry the canonical pass@k regardless of how the
    input rows were built.
    """
    return [
        ModelSolveRecord(
            model=row.model,
            tier=row.tier,
            k=row.k,
            solves=row.solves,
            pass_at_k=pass_at_k(row.solves, row.k),
        )
        for row in rows
    ]


def build_calibration_report(
    language: str,
    rows: Sequence[SolveMatrixRow],
    *,
    k: int,
    difficulty_hint: str = "",
    tier_abilities: dict[str, float] | None = None,
    provenance: Provenance | None = None,
    details: dict[str, object] | None = None,
) -> CalibrationReport:
    """Assemble a :class:`CalibrationReport` from a panel solve matrix.

    Builds the per-model :class:`ModelSolveRecord`s (canonical pass@k), fits the
    2-parameter IRT, and returns the report with ``band_verdict == "pending"`` --
    the band filter (m5-filter) assigns the terminal keep/drop verdict. The IRT
    fit metadata is recorded under ``details["irt_fit"]`` for provenance.
    """
    records = to_solve_records(rows)
    fit = fit_irt(records, tier_abilities=tier_abilities)
    merged_details: dict[str, object] = dict(details or {})
    merged_details["irt_fit"] = fit.to_dict()
    merged_details["tier_abilities"] = dict(tier_abilities or DEFAULT_TIER_ABILITIES)
    return CalibrationReport(
        language=language,
        models=records,
        k=k,
        irt_difficulty=fit.difficulty,
        irt_discrimination=fit.discrimination,
        band_verdict="pending",
        difficulty_hint=difficulty_hint,
        provenance=provenance,
        details=merged_details,
    )


__all__ = [
    "DEFAULT_TIER_ABILITIES",
    "DIFFICULTY_MAX",
    "DISCRIMINATION_MAX",
    "IrtError",
    "IrtFit",
    "SolveMatrixRow",
    "build_calibration_report",
    "compute_pass_at_k",
    "fit_irt",
    "pass_at_k",
    "to_solve_records",
]
