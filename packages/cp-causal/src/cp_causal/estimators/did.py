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
from cp_core.values import EstimatorConfig
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
    no_concurrent_intervention_check,
    sufficient_pre_periods_check,
)
from cp_causal.registry import register


def _two_by_two_design(panel: PanelArrays, start: int) -> np.ndarray:
    """Intercept, ever-treated, post, and their interaction.

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

        design = _two_by_two_design(panel, start)
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
            resampled = ols_fit(_two_by_two_design(sub, start), sub.outcome)
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
