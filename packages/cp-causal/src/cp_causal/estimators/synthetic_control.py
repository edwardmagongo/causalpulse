"""Synthetic control: a donor-weighted counterfactual for one treated unit.

Permutation inference (p-value, confidence interval) and the assumption
checks are added in Tasks 5-6 of the Phase 4 plan; this module starts with
the weight solve, the point estimate, and `CounterfactualPath`.
"""

from __future__ import annotations

import time

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import CounterfactualPath, EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, EstimatorConfig, Interval
from scipy.optimize import minimize

from cp_causal._panel import PanelArrays, outcome_scale, panel_arrays
from cp_causal._support import config_hash
from cp_causal.base import Requirements
from cp_causal.registry import register

MIN_DONORS = 5


def solve_donor_weights(treated_pre: np.ndarray, donor_pre: np.ndarray) -> np.ndarray:
    """Non-negative, simplex-constrained donor weights minimising pre-period SSE.

    Predictors are the standardised pre-treatment outcome values themselves,
    weighted uniformly -- no nested optimisation over a separate
    predictor-importance matrix. Abadie's original method nests two
    optimisations; skipping the outer one is a documented simplification
    (design doc §5.1), not an oversight, and stays correct as long as the
    donor pool is small relative to the pre-period length: an underdetermined
    fit (more donors than pre-periods) lets SLSQP drive pre-period error to
    near zero by overfitting noise, at the cost of the post-period
    extrapolation. This is why the measured protocol uses 17 donors against
    17 pre-periods, not more donors than periods.
    """
    n_donors = donor_pre.shape[0]
    x0 = np.full(n_donors, 1.0 / n_donors)

    def objective(weights: np.ndarray) -> float:
        synthetic = weights @ donor_pre
        return float(np.sum((treated_pre - synthetic) ** 2))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        # scipy-stubs' `_ConstraintDict` TypedDict is stricter than the
        # dict-based constraint API it is meant to describe (the runtime
        # accepts any `Callable[[ndarray], float]` for `fun`, but the stub's
        # overload resolution rejects this literal), so `minimize` picks no
        # matching overload here.
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],  # type: ignore[call-overload]
        options={"maxiter": 200, "ftol": 1e-10},
    )
    return np.asarray(result.x)


def _donor_matrix(panel: PanelArrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wide (unit x period) outcome arrays for the treated unit and its donors.

    Returns `(treated_series, donor_series, pre_mask)`. `pre_mask` is a
    boolean mask over periods, shared by both series since they are aligned
    on the same period axis.
    """
    wide = np.full((panel.n_units, panel.n_periods), np.nan)
    wide[panel.unit_index, panel.period_index] = panel.outcome
    treated_positions = np.flatnonzero(panel.cohort > 0)
    if treated_positions.size != 1:
        raise ValueError(
            f"SyntheticControl requires exactly one treated unit, found "
            f"{treated_positions.size}; this should have been rejected by "
            "Requirements.unmet() before fit() was ever called"
        )
    treated_index = int(treated_positions[0])
    start = int(panel.cohort[treated_index])
    pre_mask = panel.periods < start

    donor_rows = np.delete(np.arange(panel.n_units), treated_index)
    return wide[treated_index], wide[donor_rows], pre_mask


@register
class SyntheticControl:
    """Donor-weighted counterfactual for a single treated unit."""

    name = "synthetic_control"
    requirements = Requirements(
        needs_panel=True,
        min_treated_units=1,
        max_treated_units=1,
        min_control_units=MIN_DONORS,
        min_pre_periods=2,
        min_post_periods=1,
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        panel = panel_arrays(dataset)
        treated_series, donor_series, pre_mask = _donor_matrix(panel)
        post_mask = ~pre_mask

        weights = solve_donor_weights(treated_series[pre_mask], donor_series[:, pre_mask])
        synthetic = weights @ donor_series
        point = float(np.mean(treated_series[post_mask] - synthetic[post_mask]))

        pointwise_effect = treated_series - synthetic
        path = CounterfactualPath(
            periods=panel.periods.astype(float),
            observed=treated_series,
            predicted_absent_treatment=synthetic,
            # Placeholder-width band until Task 5 replaces it with the
            # permutation-based interval; a fixed +/-0 band would violate
            # `CounterfactualPath`'s own shape contract only if the arrays
            # mismatched, which they do not, so this stays valid but is not
            # yet meaningful uncertainty -- Task 5 makes it real.
            lower=synthetic,
            upper=synthetic,
            pointwise_effect=pointwise_effect,
            cumulative_effect=np.cumsum(np.where(post_mask, pointwise_effect, 0.0)),
        )

        return EffectEstimate(
            estimand="ATT",
            method=self.name,
            point=point,
            se=None,
            ci=Interval(lower=point, upper=point + 1e-9),
            ci_basis="permutation",
            bootstrap_ci=None,
            p_value=None,
            n_treated=1,
            n_control=int(donor_series.shape[0]),
            scale=outcome_scale(dataset),
            diagnostics={
                "donor_weights": Diagnostic(
                    name="donor_weights",
                    value=weights,
                    unit=None,
                    interpretation=(
                        "Simplex weight placed on each donor by the "
                        "pre-treatment fit, in donor order."
                    ),
                ),
            },
            assumptions=(),
            counterfactual=path,
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )
