"""Event-study difference in differences: one coefficient per event time."""

from __future__ import annotations

import time

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import AssumptionCheck, CounterfactualPath, EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, EstimatorConfig
from cp_statistics.covariance import cluster_robust_covariance, ols_fit, robust_covariance

from cp_causal._panel import (
    MIN_UNITS_PER_ARM,
    PanelArrays,
    bootstrap_panel_effect,
    cluster_count_diagnostic,
    clustered_vs_iid_diagnostic,
    outcome_scale,
    panel_arrays,
)
from cp_causal._support import (
    coefficient_se,
    config_hash,
    critical_value,
    degeneracy_diagnostic,
    normal_p_value,
    wald_interval,
)
from cp_causal.base import Requirements
from cp_causal.panel_checks import (
    absorbing_treatment_check,
    no_anticipation_check,
    no_concurrent_intervention_check,
    parallel_trends_check,
    sufficient_pre_periods_check,
)
from cp_causal.registry import register

# t - 1 is the omitted category: every coefficient is read relative to the
# period immediately before adoption.
REFERENCE_HORIZON = -1


def _event_time(panel: PanelArrays) -> np.ndarray:
    """Periods since adoption per row; a large negative for never-treated."""
    cohort = panel.cohort[panel.unit_index]
    horizon = panel.periods[panel.period_index] - cohort
    return np.where(cohort > 0, horizon, np.iinfo(np.int32).min)


def _event_design(
    panel: PanelArrays, horizons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit and period dummies plus one indicator per non-reference horizon.

    Never-treated units contribute only to the fixed effects, which is what
    makes them the comparison group.
    """
    n = panel.n_obs
    rows = np.arange(n)
    event = _event_time(panel)

    unit_dummies = np.zeros((n, panel.n_units - 1))
    keep = panel.unit_index > 0
    unit_dummies[rows[keep], panel.unit_index[keep] - 1] = 1.0

    period_dummies = np.zeros((n, panel.n_periods - 1))
    keep = panel.period_index > 0
    period_dummies[rows[keep], panel.period_index[keep] - 1] = 1.0

    estimated = horizons[horizons != REFERENCE_HORIZON]
    event_dummies = np.zeros((n, estimated.size))
    for column, horizon in enumerate(estimated):
        event_dummies[event == horizon, column] = 1.0

    design = np.column_stack(
        [np.ones(n), event_dummies, unit_dummies, period_dummies]
    )
    return design, estimated


def _horizons(panel: PanelArrays) -> np.ndarray:
    event = _event_time(panel)
    observed = sorted({int(h) for h in np.unique(event) if h > np.iinfo(np.int32).min})
    return np.asarray(observed, dtype=int)


def breakdown_value(
    point: float, se: float, ci_level: float, mean_horizon: float
) -> float:
    """Smallest linear pre-trend violation that would pull the interval to zero.

    **The linear-extrapolation form, not Rambachan-Roth honest confidence
    sets.** Honest confidence sets are a research-grade implementation and are
    not in this phase. What this computes: if the counterfactual trend had been
    off by a constant slope M per period, the estimate would shift by
    `M * mean_horizon`; the breakdown value is the smallest M that moves the
    interval bound onto zero. Zero means the conclusion is already not
    significant, so no violation at all is needed to overturn it.
    """
    if mean_horizon <= 0:
        return 0.0
    margin = abs(point) - critical_value(ci_level) * se
    return float(max(margin, 0.0) / mean_horizon)


@register
class EventStudy:
    """Leads and lags relative to adoption, with unit-clustered inference."""

    name = "event_study"
    requirements = Requirements(
        needs_panel=True,
        min_pre_periods=2,
        min_post_periods=1,
        min_treated_units=MIN_UNITS_PER_ARM,
        min_control_units=MIN_UNITS_PER_ARM,
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        panel = panel_arrays(dataset)

        horizons = _horizons(panel)
        design, estimated = _event_design(panel, horizons)
        fit = ols_fit(design, panel.outcome)
        covariance = cluster_robust_covariance(fit, panel.unit_index)

        # Coefficient columns run 1..estimated.size; the reference horizon is
        # pinned at exactly zero rather than estimated.
        coefficients = np.zeros(horizons.size)
        standard_errors = np.zeros(horizons.size)
        column_of: dict[int, int] = {
            int(h): 1 + i for i, h in enumerate(estimated)
        }
        for index, horizon in enumerate(horizons):
            column = column_of.get(int(horizon))
            if column is None:
                continue
            coefficients[index] = float(fit.beta[column])
            standard_errors[index] = coefficient_se(covariance, column)

        post = horizons >= 0
        pre = horizons < REFERENCE_HORIZON
        point = float(coefficients[post].mean()) if post.any() else float("nan")

        # Standard error of the average post coefficient, from the relevant
        # block of the clustered covariance rather than by treating the
        # coefficients as independent.
        post_columns = [column_of[int(h)] for h in horizons[post] if int(h) in column_of]
        weights = np.zeros(design.shape[1])
        if post_columns:
            weights[post_columns] = 1.0 / len(post_columns)
        se = float(np.sqrt(max(float(weights @ covariance @ weights), 0.0)))

        iid_covariance = robust_covariance(fit, kind="HC1")
        iid_se = float(np.sqrt(max(float(weights @ iid_covariance @ weights), 0.0)))

        pre_columns = [column_of[int(h)] for h in horizons[pre] if int(h) in column_of]
        lead_coefficients = np.asarray([float(fit.beta[c]) for c in pre_columns])
        lead_covariance = covariance[np.ix_(pre_columns, pre_columns)]

        def point_from(indices: np.ndarray) -> float:
            sub_units = np.unique(panel.unit_index[indices])
            ever = panel.cohort[panel.unit_index[indices]] > 0
            if (
                np.unique(panel.unit_index[indices][ever]).size < MIN_UNITS_PER_ARM
                or np.unique(panel.unit_index[indices][~ever]).size
                < MIN_UNITS_PER_ARM
            ):
                return float("nan")
            remapped = np.searchsorted(sub_units, panel.unit_index[indices])
            sub = PanelArrays(
                outcome=panel.outcome[indices],
                treatment=panel.treatment[indices],
                unit_index=remapped,
                period_index=panel.period_index[indices],
                units=sub_units,
                periods=panel.periods,
                # Sliced to the resampled units, matching `remapped`. Passing
                # the full-length cohort array here would misalign every
                # cohort lookup in the resample while still producing numbers.
                cohort=panel.cohort[sub_units],
            )
            sub_horizons = _horizons(sub)
            if not (sub_horizons >= 0).any():
                return float("nan")
            sub_design, sub_estimated = _event_design(sub, sub_horizons)
            resampled = ols_fit(sub_design, sub.outcome)
            columns = [
                1 + i for i, h in enumerate(sub_estimated) if h >= 0
            ]
            if not columns:
                return float("nan")
            return float(np.mean([resampled.beta[c] for c in columns]))

        bootstrap_result = bootstrap_panel_effect(point_from, panel, config)

        mean_horizon = float(horizons[post].mean() + 1) if post.any() else 0.0
        pre_slope, pre_slope_se = _pre_trend_slope(
            horizons[pre], coefficients[pre], lead_covariance
        )
        path = self._path(panel, horizons, coefficients, standard_errors, config)
        # The immediately-pre-adoption coefficient, for `no_anticipation_check`.
        # This is deliberately NOT the reference horizon itself: t - 1 is the
        # omitted category, pinned to exactly 0.0 by construction (see
        # `test_the_reference_period_is_omitted_and_pinned_at_zero`), so
        # reading it would make the check pass unconditionally with no signal
        # at all. The nearest *estimated* lead (t - 2, when present) is the
        # closest available proxy: anticipation showing up there is fixed by
        # moving the reference period back further, which is the remedy the
        # check text describes. `horizon_label` below is what keeps the
        # generated interpretation honest about which lead that actually is,
        # rather than unconditionally claiming to describe t - 1.
        pre_horizons = horizons[pre]
        if pre_horizons.size:
            anticipation_horizon = int(pre_horizons.max())
            anticipation_index = int(
                np.flatnonzero(horizons == anticipation_horizon)[0]
            )
            no_anticipation = no_anticipation_check(
                coefficient=float(coefficients[anticipation_index]),
                standard_error=float(standard_errors[anticipation_index]),
                horizon_label=(
                    "nearest estimated pre-treatment lead "
                    f"(t{anticipation_horizon}, since t{REFERENCE_HORIZON} "
                    "is the omitted reference period and cannot be estimated)"
                ),
            )
        else:
            no_anticipation = AssumptionCheck(
                name="no_anticipation",
                status="NOT_TESTABLE",
                statistic=None,
                threshold=None,
                interpretation=(
                    f"There is no estimated lead before t{REFERENCE_HORIZON}, "
                    "the omitted reference period, to test: with only a single "
                    "pre-treatment period observed, every pre-treatment row is "
                    "absorbed into the reference category and pinned to zero "
                    "by construction. This is not evidence that there was no "
                    "anticipation."
                ),
                remedy=(
                    "Extend the panel backwards so at least two pre-treatment "
                    "periods are observed."
                ),
            )

        treated_units = panel.cohort > 0
        return EffectEstimate(
            estimand="ATT",
            method=self.name,
            point=point,
            se=se,
            ci=wald_interval(point, se, config.ci_level),
            ci_basis="analytic",
            bootstrap_ci=bootstrap_result.interval,
            p_value=normal_p_value(point, se),
            n_treated=int(treated_units.sum()),
            n_control=int((~treated_units).sum()),
            scale=outcome_scale(dataset),
            diagnostics={
                "event_time_horizons": Diagnostic(
                    name="event_time_horizons",
                    value=horizons.astype(float),
                    unit="periods since adoption",
                    interpretation=(
                        "Event times the coefficients below are indexed by; "
                        f"{REFERENCE_HORIZON} is the omitted reference period."
                    ),
                ),
                "event_time_coefficients": Diagnostic(
                    name="event_time_coefficients",
                    value=coefficients,
                    unit=None,
                    interpretation=(
                        "Effect at each event time relative to the period before "
                        "adoption. Pre-treatment entries should be near zero if "
                        "trends were parallel."
                    ),
                ),
                "event_time_standard_errors": Diagnostic(
                    name="event_time_standard_errors",
                    value=standard_errors,
                    unit=None,
                    interpretation="Unit-clustered standard error per event time.",
                ),
                "pre_trend_slope": Diagnostic(
                    name="pre_trend_slope",
                    value=pre_slope,
                    unit="per period",
                    interpretation=(
                        f"Fitted slope through the pre-treatment coefficients: "
                        f"{pre_slope:+.4f} per period, 95% interval "
                        f"[{pre_slope - 1.96 * pre_slope_se:+.4f}, "
                        f"{pre_slope + 1.96 * pre_slope_se:+.4f}]. Magnitude and "
                        "precision reported separately, because the joint "
                        "p-value collapses them into one number: a tiny, "
                        "precisely-estimated pre-trend and a large, noisy one "
                        "can produce the same p-value and call for different "
                        "responses."
                    ),
                ),
                "pre_trend_slope_se": Diagnostic(
                    name="pre_trend_slope_se",
                    value=pre_slope_se,
                    unit="per period",
                    interpretation=(
                        "Standard error of the pre-trend slope, from the "
                        "clustered covariance of the lead coefficients it is "
                        "fitted through."
                    ),
                ),
                "pre_trend_breakdown_value": Diagnostic(
                    name="pre_trend_breakdown_value",
                    value=breakdown_value(point, se, config.ci_level, mean_horizon),
                    unit="per period",
                    interpretation=(
                        "Smallest per-period linear trend violation that would "
                        "pull the confidence interval onto zero. This is the "
                        "linear-extrapolation form, NOT a Rambachan-Roth honest "
                        "confidence set, which is not implemented here. Zero "
                        "means the estimate is already not significant."
                    ),
                ),
                "clustered_se_ratio": clustered_vs_iid_diagnostic(se, iid_se),
                "cluster_count": cluster_count_diagnostic(panel.n_units),
                "bootstrap_degenerate_replicates": degeneracy_diagnostic(
                    bootstrap_result.n_degenerate, min_arm_size=MIN_UNITS_PER_ARM
                ),
            },
            assumptions=(
                parallel_trends_check(lead_coefficients, lead_covariance),
                no_anticipation,
                absorbing_treatment_check(dataset),
                sufficient_pre_periods_check(panel, minimum=2),
                no_concurrent_intervention_check(),
            ),
            counterfactual=path,
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )

    def _path(
        self,
        panel: PanelArrays,
        horizons: np.ndarray,
        coefficients: np.ndarray,
        standard_errors: np.ndarray,
        config: EstimatorConfig,
    ) -> CounterfactualPath:
        """Observed and counterfactual treated-group means, period by period."""
        critical = critical_value(config.ci_level)
        treated_rows = panel.cohort[panel.unit_index] > 0
        event = _event_time(panel)

        observed = np.zeros(panel.n_periods)
        effect = np.zeros(panel.n_periods)
        error = np.zeros(panel.n_periods)
        for index in range(panel.n_periods):
            rows = treated_rows & (panel.period_index == index)
            observed[index] = float(panel.outcome[rows].mean()) if rows.any() else np.nan
            # The average effect at this calendar period is the mean of the
            # event-time coefficients the treated units are sitting at, among
            # only the units already at or past their own adoption (horizon
            # >= 0). Pre-adoption event times are deliberately excluded: a
            # unit that has not yet been treated has nothing to have caused
            # a divergence, so `effect` stays at its zero-initialised value
            # and `predicted_absent_treatment` equals `observed` exactly for
            # every (period, unit-group) combination before that unit's own
            # adoption -- which is also the calendar period before any cohort
            # has adopted, under staggered adoption.
            at_period = event[rows]
            post_adoption = at_period[at_period >= 0]
            matched = np.isin(horizons, post_adoption)
            if matched.any():
                effect[index] = float(coefficients[matched].mean())
                error[index] = float(standard_errors[matched].mean())

        predicted = observed - effect
        return CounterfactualPath(
            periods=panel.periods.astype(float),
            observed=observed,
            predicted_absent_treatment=predicted,
            lower=predicted - critical * error,
            upper=predicted + critical * error,
            pointwise_effect=effect,
            cumulative_effect=np.cumsum(effect),
        )


def _pre_trend_slope(
    horizons: np.ndarray, coefficients: np.ndarray, covariance: np.ndarray
) -> tuple[float, float]:
    """Least-squares slope through the pre-treatment coefficients, and its SE.

    The slope is a fixed linear combination `w` of the lead coefficients, so its
    variance is `w' V w` under the same clustered covariance the coefficients
    carry — not a fresh regression standard error, which would treat the leads
    as independent observations when they share control units.
    """
    if horizons.size < 2:
        return 0.0, 0.0
    centred = horizons.astype(float) - horizons.mean()
    denominator = float((centred**2).sum())
    if denominator == 0:
        return 0.0, 0.0
    weights = centred / denominator
    slope = float(weights @ coefficients)
    variance = float(weights @ covariance @ weights)
    return slope, float(np.sqrt(max(variance, 0.0)))
