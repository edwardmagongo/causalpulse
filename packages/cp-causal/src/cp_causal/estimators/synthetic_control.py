"""Synthetic control: a donor-weighted counterfactual for one treated unit.

Permutation inference (p-value, confidence interval) is implemented here via
`placebo_runs` and `permutation_p_value`. The assumption checks live in
`panel_checks.py`; `ridge_augmented_weights` is the fallback `fit` switches
to when `convex_hull_check` fails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import CounterfactualPath, EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, EstimatorConfig, Interval
from scipy.optimize import minimize

from cp_causal._panel import donor_matrix, outcome_scale, panel_arrays
from cp_causal._support import config_hash
from cp_causal.base import Requirements
from cp_causal.panel_checks import (
    convex_hull_check,
    donor_contamination_check,
    pre_treatment_fit_check,
    sufficient_pre_periods_check,
)
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


def ridge_augmented_weights(
    treated_pre: np.ndarray, donor_pre: np.ndarray, penalty: float = 1.0
) -> tuple[float, np.ndarray]:
    """Ridge-regularised weights with an intercept, for a convex-hull violation.

    A documented simplification of Ben-Michael/Feller/Rothstein's augmented
    synthetic control (design doc §5.4): fits an intercept plus L2-penalised,
    unconstrained donor coefficients directly, rather than the residual-
    correction form of the original method. Simplex weights cannot reproduce
    a value outside the donor pool's range by construction; an intercept plus
    unconstrained coefficients can. Measured (design doc §2 item 5), 300
    seeds on the convex-hull-violation scenario: plain simplex SC bias +1.21
    against ridge-augmented (penalty=1.0) bias +0.0223 -- and an earlier,
    no-intercept version of this ridge fit was *worse* than plain SC
    (bias -1.87 at the same penalty), which is why the intercept term here is
    load-bearing and not a minor refinement.
    """
    n_pre = donor_pre.shape[1]
    design = np.column_stack([np.ones(n_pre), donor_pre.T])
    gram = design.T @ design
    n_params = design.shape[1]
    ridge = penalty * np.eye(n_params)
    ridge[0, 0] = 0.0  # never shrink the intercept
    beta = np.linalg.solve(gram + ridge, design.T @ treated_pre)
    return float(beta[0]), np.asarray(beta[1:])


@dataclass(frozen=True, eq=False)
class PlaceboRuns:
    """One synthetic-control fit per unit: the real treated unit at index 0,
    then each donor playing "treated" in turn, with the remaining donors as
    its own donor pool.

    Backs three things from one computation, not three: the permutation
    p-value (`permutation_p_value`, over `rmspe_ratios`), the confidence
    interval (`SyntheticControl.fit`, over `point_estimates[1:]`), and the
    donor_contamination assumption check (Task 6, over `rmspe_ratios[1:]`) --
    a genuinely contaminated donor looks, from this placebo procedure's point
    of view, exactly like a unit whose post-period diverges from what its own
    donor pool predicts, which is what the ratio measures regardless of why
    the divergence happened.
    """

    point_estimates: np.ndarray
    rmspe_ratios: np.ndarray


def placebo_runs(
    treated_series: np.ndarray,
    donor_series: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
) -> PlaceboRuns:
    n_donors = donor_series.shape[0]
    points = np.empty(n_donors + 1)
    ratios = np.empty(n_donors + 1)
    for index in range(n_donors + 1):
        if index == 0:
            unit_series, pool = treated_series, donor_series
        else:
            unit_series = donor_series[index - 1]
            pool = np.delete(donor_series, index - 1, axis=0)
        weights = solve_donor_weights(unit_series[pre_mask], pool[:, pre_mask])
        synthetic = weights @ pool
        points[index] = float(np.mean(unit_series[post_mask] - synthetic[post_mask]))
        pre_rmspe = float(np.sqrt(np.mean((unit_series[pre_mask] - synthetic[pre_mask]) ** 2)))
        post_rmspe = float(np.sqrt(np.mean((unit_series[post_mask] - synthetic[post_mask]) ** 2)))
        ratios[index] = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf")
    return PlaceboRuns(point_estimates=points, rmspe_ratios=ratios)


def permutation_p_value(rmspe_ratios: np.ndarray) -> float:
    """Rank-based p-value: how extreme the real unit's ratio is among all units.

    Index 0 is the real treated unit (see `PlaceboRuns`). `rank` counts every
    unit, including the real one, whose ratio is at least as large -- so a
    real unit tied for most extreme gets `p = 1 / (n + 1)`, the smallest
    achievable p-value with `n` donors, matching Abadie's construction.
    """
    n = rmspe_ratios.size
    rank = int(np.sum(rmspe_ratios >= rmspe_ratios[0]))
    return rank / n


def _in_time_placebo(
    treated_series: np.ndarray, donor_series: np.ndarray, pre_mask: np.ndarray
) -> float:
    """A fake treatment date at the pre-period's midpoint; effect should be near zero.

    Splits the *real* pre-period in two: the first half as the placebo's own
    pre-period, the second half as its placebo post-period. A well-behaved
    fit should show a small effect here, since nothing was actually treated
    yet.
    """
    pre_periods = np.flatnonzero(pre_mask)
    if pre_periods.size < 4:
        return float("nan")
    midpoint = pre_periods.size // 2
    fake_pre = pre_periods[:midpoint]
    fake_post = pre_periods[midpoint:]
    weights = solve_donor_weights(treated_series[fake_pre], donor_series[:, fake_pre])
    synthetic = weights @ donor_series
    return float(np.mean(treated_series[fake_post] - synthetic[fake_post]))


def leave_one_donor_out_points(
    treated_series: np.ndarray,
    donor_series: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    reference_weights: np.ndarray,
    use_ridge: bool,
    weight_threshold: float = 0.05,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Point estimate with each influential donor excluded in turn.

    Only donors actually carrying non-trivial weight are worth re-fitting
    without -- dropping a zero-weight donor cannot move the estimate.
    `reference_weights` identifies which donors count as influential, and
    `use_ridge` picks which estimator form each leave-one-out refit uses --
    both must match whichever form produced the reported point (simplex
    weights, or the ridge-augmented fallback's donor coefficients). An
    earlier version always refit with plain simplex weights regardless of
    which form the reported point actually came from, so on a convex-hull
    violation the "swing" it reported was mostly the gap between the two
    estimator forms, not genuine donor influence -- caught by the final
    whole-branch review, not by any task-scoped test, because nothing
    exercised this diagnostic through `fit()` on a convex-hull-violation
    scenario.

    Returns the influential donor indices alongside their estimates, so a
    caller reporting the estimates can say which donor each one dropped.
    """
    influential = np.flatnonzero(np.abs(reference_weights) > weight_threshold)
    points: list[float] = []
    for donor_index in influential:
        remaining = np.delete(donor_series, donor_index, axis=0)
        if use_ridge:
            intercept, coefficients = ridge_augmented_weights(
                treated_series[pre_mask], remaining[:, pre_mask]
            )
            loo_synthetic = intercept + coefficients @ remaining
        else:
            loo_weights = solve_donor_weights(
                treated_series[pre_mask], remaining[:, pre_mask]
            )
            loo_synthetic = loo_weights @ remaining
        points.append(
            float(np.mean(treated_series[post_mask] - loo_synthetic[post_mask]))
        )
    return tuple(int(index) for index in influential), tuple(points)


def _leave_one_donor_out_max_swing(
    treated_series: np.ndarray,
    donor_series: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    full_point: float,
    reference_weights: np.ndarray,
    use_ridge: bool,
    weight_threshold: float = 0.05,
) -> float:
    """Largest change in the point estimate from dropping one influential donor.

    A summary of `leave_one_donor_out_points`, which does the refitting. The
    diagnostic reports this one number; `falsify` reports the whole set.
    """
    _, points = leave_one_donor_out_points(
        treated_series,
        donor_series,
        pre_mask,
        post_mask,
        reference_weights,
        use_ridge,
        weight_threshold,
    )
    return max((abs(point - full_point) for point in points), default=0.0)


@register
class SyntheticControl:
    """Donor-weighted counterfactual for a single treated unit."""

    name = "synthetic_control"
    requirements = Requirements(
        needs_panel=True,
        min_treated_units=1,
        max_treated_units=1,
        min_control_units=MIN_DONORS,
        # Matches sufficient_pre_periods_check's own minimum=10 below,
        # following every Phase 3 estimator's convention (DID 1/1, CS 1/1,
        # EventStudy 2/2: Requirements' floor always equals the check's own
        # declared minimum). An earlier version left this at a bare floor
        # of 2, which routed data as short as S5's 2 pre-periods onto this
        # estimator -- the exact underdetermined regime
        # solve_donor_weights's own docstring says its uniform-predictor
        # simplification stops being valid in. Found at final review.
        min_pre_periods=10,
        min_post_periods=1,
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        panel = panel_arrays(dataset)
        treated_series, donor_series, pre_mask = donor_matrix(panel)
        post_mask = ~pre_mask

        runs = placebo_runs(treated_series, donor_series, pre_mask, post_mask)
        p_value = permutation_p_value(runs.rmspe_ratios)

        weights = solve_donor_weights(treated_series[pre_mask], donor_series[:, pre_mask])
        synthetic_simplex = weights @ donor_series

        treated_pre_mean = float(treated_series[pre_mask].mean())
        donor_pre_means = donor_series[:, pre_mask].mean(axis=1)
        convex_hull = convex_hull_check(treated_pre_mean, donor_pre_means)

        use_ridge = convex_hull.status == "FAIL"
        if use_ridge:
            intercept, coefficients = ridge_augmented_weights(
                treated_series[pre_mask], donor_series[:, pre_mask]
            )
            synthetic = intercept + coefficients @ donor_series
            reference_weights = coefficients
        else:
            synthetic = synthetic_simplex
            reference_weights = weights

        point = float(np.mean(treated_series[post_mask] - synthetic[post_mask]))
        pointwise_effect = treated_series - synthetic
        # Measured from the *simplex* fit always, regardless of which form
        # produced `synthetic` above. `pre_treatment_fit_check`'s threshold
        # is calibrated against exactly this quantity (design doc §2 item 5:
        # S1 vs S3 *before* the ridge fallback) precisely because it answers
        # "could the donor weights find a good match on their own" -- a
        # question the ridge fallback exists to route around, not to
        # answer differently. Reading it from the post-fallback `synthetic`
        # instead (an earlier version of this code did) makes the check
        # unable to ever FAIL on the one scenario it was built to catch,
        # since the ridge fit is by design a good one.
        pre_rmspe = float(
            np.sqrt(np.mean((treated_series[pre_mask] - synthetic_simplex[pre_mask]) ** 2))
        )
        outcome_sd = float(treated_series[pre_mask].std(ddof=1))

        # A permutation-based interval from the placebo distribution's own
        # spread, following the same logic as the p-value: donors' point
        # estimates under "no true effect on them" are the reference
        # distribution `point`'s own uncertainty is judged against.
        placebo_spread = runs.point_estimates[1:]
        half_width = float(np.quantile(np.abs(placebo_spread), config.ci_level))
        ci = Interval(lower=point - half_width, upper=point + half_width, level=config.ci_level)

        in_time_placebo = _in_time_placebo(treated_series, donor_series, pre_mask)
        loo_swing = _leave_one_donor_out_max_swing(
            treated_series,
            donor_series,
            pre_mask,
            post_mask,
            point,
            reference_weights,
            use_ridge,
        )

        path = CounterfactualPath(
            periods=panel.periods.astype(float),
            observed=treated_series,
            predicted_absent_treatment=synthetic,
            lower=synthetic - half_width,
            upper=synthetic + half_width,
            pointwise_effect=pointwise_effect,
            cumulative_effect=np.cumsum(np.where(post_mask, pointwise_effect, 0.0)),
        )

        return EffectEstimate(
            estimand="ATT",
            method=self.name,
            point=point,
            se=None,
            ci=ci,
            ci_basis="permutation",
            bootstrap_ci=None,
            p_value=p_value,
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
                        "pre-treatment fit, in donor order. When convex_hull "
                        "below is FAIL, the reported point estimate and path "
                        "use the ridge-augmented fallback instead of these "
                        "weights directly."
                    ),
                ),
                "in_time_placebo_effect": Diagnostic(
                    name="in_time_placebo_effect",
                    value=in_time_placebo,
                    unit=None,
                    interpretation=(
                        "Estimated effect from a fake treatment date at the "
                        "pre-period's midpoint, where nothing was actually "
                        "treated. Should be small; a large value means the "
                        "method finds 'effects' even with no real "
                        "intervention."
                    ),
                ),
                "leave_one_donor_out_max_swing": Diagnostic(
                    name="leave_one_donor_out_max_swing",
                    value=loo_swing,
                    unit=None,
                    interpretation=(
                        "Largest change in the point estimate from dropping "
                        "any single donor holding more than 5% weight. A "
                        "large value means the result rests heavily on one "
                        "donor rather than a genuine combination."
                    ),
                ),
            },
            assumptions=(
                pre_treatment_fit_check(pre_rmspe, outcome_sd),
                convex_hull,
                donor_contamination_check(runs.rmspe_ratios[1:]),
                sufficient_pre_periods_check(panel, minimum=10),
            ),
            counterfactual=path,
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )
