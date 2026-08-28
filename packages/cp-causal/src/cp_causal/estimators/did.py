"""Difference-in-differences on panel data."""

from __future__ import annotations

import time

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import (
    CounterfactualPoint,
    EffectEstimate,
    FitMetadata,
)
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
from cp_causal.goodman_bacon import goodman_bacon_decomposition
from cp_causal.panel_checks import (
    absorbing_treatment_check,
    no_concurrent_intervention_check,
    sufficient_pre_periods_check,
)
from cp_causal.registry import register


def two_by_two_design(panel: PanelArrays, start: int) -> np.ndarray:
    """Intercept, ever-treated, post, and their interaction.

    Public because `cp_causal.sensitivity` refits this exact design to bound
    omitted-variable bias. A private copy there would be a second definition
    of the same matrix, free to drift from this one.

    The interaction coefficient is the DiD estimate. Computed through the
    saturated regression rather than a four-means formula so the standard error
    comes from the cluster-robust sandwich already under test in cp_statistics.
    """
    ever = (panel.cohort[panel.unit_index] > 0).astype(float)
    post = (panel.periods[panel.period_index] >= start).astype(float)
    return np.column_stack([np.ones(panel.n_obs), ever, post, ever * post])


@register
class DifferenceInDifferences:
    """The canonical 2x2 difference in differences, with unit-clustered inference."""

    name = "did_2x2"
    requirements = Requirements(
        needs_panel=True,
        needs_uniform_adoption=True,
        min_pre_periods=1,
        min_post_periods=1,
        min_treated_units=MIN_UNITS_PER_ARM,
        min_control_units=MIN_UNITS_PER_ARM,
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        panel = panel_arrays(dataset)

        adoption = panel.adoption_periods
        if len(adoption) != 1:
            raise ValueError(
                f"{self.name} requires a single adoption period; this panel has "
                f"{len(adoption)}: {list(adoption)}. Use two-way fixed effects "
                "or Callaway-Sant'Anna on staggered data."
            )
        start = adoption[0]

        design = two_by_two_design(panel, start)
        fit = ols_fit(design, panel.outcome)
        clusters = panel.unit_index
        covariance = cluster_robust_covariance(fit, clusters)

        point = float(fit.beta[3])
        se = coefficient_se(covariance, 3)
        iid_se = coefficient_se(robust_covariance(fit, kind="HC1"), 3)

        def point_from(indices: np.ndarray) -> float:
            resampled_units = np.unique(panel.unit_index[indices])
            ever = panel.cohort[panel.unit_index[indices]] > 0
            if (
                np.unique(panel.unit_index[indices][ever]).size < MIN_UNITS_PER_ARM
                or np.unique(panel.unit_index[indices][~ever]).size
                < MIN_UNITS_PER_ARM
                or resampled_units.size < 2
            ):
                return float("nan")
            sub = PanelArrays(
                outcome=panel.outcome[indices],
                treatment=panel.treatment[indices],
                unit_index=panel.unit_index[indices],
                period_index=panel.period_index[indices],
                units=panel.units,
                periods=panel.periods,
                cohort=panel.cohort,
            )
            resampled = ols_fit(two_by_two_design(sub, start), sub.outcome)
            return float(resampled.beta[3])

        bootstrap_result = bootstrap_panel_effect(point_from, panel, config)

        treated_units = panel.cohort > 0
        post_rows = panel.periods[panel.period_index] >= start
        treated_rows = treated_units[panel.unit_index]
        observed = float(panel.outcome[treated_rows & post_rows].mean())
        critical = critical_value(config.ci_level)

        return EffectEstimate(
            estimand="ATT",
            method=self.name,
            point=point,
            se=se,
            ci=wald_interval(point, se, config.ci_level),
            ci_basis="analytic",
            bootstrap_ci=bootstrap_result.interval,
            p_value=normal_p_value(point, se),
            # Units, not unit-periods: a unit-period count would inflate both by
            # the panel length, making every reported sample size wrong by the
            # same plausible-looking multiple.
            n_treated=int(treated_units.sum()),
            n_control=int((~treated_units).sum()),
            scale=outcome_scale(dataset),
            diagnostics={
                "clustered_se_ratio": clustered_vs_iid_diagnostic(se, iid_se),
                "cluster_count": cluster_count_diagnostic(panel.n_units),
                "bootstrap_degenerate_replicates": degeneracy_diagnostic(
                    bootstrap_result.n_degenerate, min_arm_size=MIN_UNITS_PER_ARM
                ),
            },
            assumptions=(
                absorbing_treatment_check(dataset),
                sufficient_pre_periods_check(panel, minimum=1),
                no_concurrent_intervention_check(),
            ),
            counterfactual=CounterfactualPoint(
                observed_mean=observed,
                predicted_absent_treatment=observed - point,
                lower=observed - point - critical * se,
                upper=observed - point + critical * se,
            ),
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )


def twfe_design(panel: PanelArrays) -> np.ndarray:
    """Intercept, treatment indicator, unit dummies, period dummies.

    Explicit dummies rather than two-way demeaning: demeaning is not exact on
    unbalanced panels, and exact agreement with an external reference is what
    this estimator is pinned against.

    One unit dummy and one period dummy are dropped against the intercept. That
    is the rank-deficiency guard: keeping all of them makes the design singular,
    and while lstsq would still return a minimum-norm solution, the covariance
    would be meaningless.
    """
    n = panel.n_obs
    rows = np.arange(n)

    unit_dummies = np.zeros((n, panel.n_units - 1))
    keep = panel.unit_index > 0
    unit_dummies[rows[keep], panel.unit_index[keep] - 1] = 1.0

    period_dummies = np.zeros((n, panel.n_periods - 1))
    keep = panel.period_index > 0
    period_dummies[rows[keep], panel.period_index[keep] - 1] = 1.0

    return np.column_stack(
        [np.ones(n), panel.treatment, unit_dummies, period_dummies]
    )


@register
class TwoWayFixedEffects:
    """Unit and period fixed effects with a treatment indicator.

    Accepts staggered adoption, and is biased under it when effects differ
    across cohorts: already-treated units then serve as controls with negative
    weights. Scenario P4 demonstrates that failure, and
    `goodman_bacon_decomposition` explains where it comes from.
    """

    name = "twfe"
    requirements = Requirements(
        needs_panel=True,
        min_pre_periods=1,
        min_post_periods=1,
        min_treated_units=MIN_UNITS_PER_ARM,
        min_control_units=MIN_UNITS_PER_ARM,
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        panel = panel_arrays(dataset)

        fit = ols_fit(twfe_design(panel), panel.outcome)
        covariance = cluster_robust_covariance(fit, panel.unit_index)

        point = float(fit.beta[1])
        # Only the treatment coefficient's own diagonal entry, never a blanket
        # sqrt over the diagonal: see `coefficient_se`.
        se = coefficient_se(covariance, 1)
        iid_se = coefficient_se(robust_covariance(fit, kind="HC1"), 1)

        def point_from(indices: np.ndarray) -> float:
            sub_units = np.unique(panel.unit_index[indices])
            ever = panel.cohort[panel.unit_index[indices]] > 0
            if (
                np.unique(panel.unit_index[indices][ever]).size < MIN_UNITS_PER_ARM
                or np.unique(panel.unit_index[indices][~ever]).size
                < MIN_UNITS_PER_ARM
                or sub_units.size < 2
            ):
                return float("nan")
            # Reindex units so the dummy block stays the size of the resample.
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
            resampled = ols_fit(twfe_design(sub), sub.outcome)
            return float(resampled.beta[1])

        bootstrap_result = bootstrap_panel_effect(point_from, panel, config)

        bacon = (
            goodman_bacon_decomposition(dataset)
            if len(panel.adoption_periods) > 1
            else None
        )

        treated_units = panel.cohort > 0
        treated_rows = panel.treatment == 1
        observed = float(panel.outcome[treated_rows].mean())
        critical = critical_value(config.ci_level)

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
                "clustered_se_ratio": clustered_vs_iid_diagnostic(se, iid_se),
                "cluster_count": cluster_count_diagnostic(panel.n_units),
                "n_adoption_periods": Diagnostic(
                    name="n_adoption_periods",
                    value=float(len(panel.adoption_periods)),
                    unit="periods",
                    interpretation=(
                        "Distinct adoption periods. With more than one and "
                        "effects that differ across cohorts, this estimator is "
                        "biased; run goodman_bacon_decomposition to see how much "
                        "of it rests on already-treated comparisons."
                    ),
                ),
                "bootstrap_degenerate_replicates": degeneracy_diagnostic(
                    bootstrap_result.n_degenerate, min_arm_size=MIN_UNITS_PER_ARM
                ),
                "already_treated_control_share": Diagnostic(
                    name="already_treated_control_share",
                    value=float(bacon.already_treated_share) if bacon else 0.0,
                    unit="share",
                    interpretation=(
                        "Share of this estimate resting on comparisons that use "
                        "already-treated units as controls. Those comparisons "
                        "carry the wrong sign when effects change over time, so "
                        "a large share here is the mechanism behind the bias, "
                        "not merely a warning about it."
                    ),
                ),
            },
            assumptions=(
                absorbing_treatment_check(dataset),
                sufficient_pre_periods_check(panel, minimum=1),
                no_concurrent_intervention_check(),
            ),
            counterfactual=CounterfactualPoint(
                observed_mean=observed,
                predicted_absent_treatment=observed - point,
                lower=observed - point - critical * se,
                upper=observed - point + critical * se,
            ),
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )
