"""Analysis for randomised experiments.

Wald intervals for a difference of proportions are deliberately not used: they
undercover badly at the conversion rates experiments actually produce. Binary
outcomes get a Newcombe hybrid-score interval; continuous outcomes get Welch.

Prospective power calculations belong here eventually. Observed (post-hoc)
power does not and never will — it is a monotone function of the p-value and
tells you nothing the p-value did not.
"""

from __future__ import annotations

import time

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import (
    AssumptionCheck,
    CounterfactualPoint,
    EffectEstimate,
    FitMetadata,
)
from cp_core.values import Diagnostic, EstimatorConfig, Interval
from cp_statistics.proportions import newcombe_interval
from scipy import stats

from cp_causal._support import arrays, bootstrap_effect, config_hash
from cp_causal.base import Requirements
from cp_causal.registry import register


@register
class DifferenceInMeans:
    """Unadjusted difference between treatment arms."""

    name = "difference_in_means"
    requirements = Requirements(min_treated_units=2, min_control_units=2)

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        _, treatment, outcome = arrays(dataset)
        binary = dataset.schema.outcome_type == "binary"

        treated = outcome[treatment == 1]
        control = outcome[treatment == 0]
        point = float(treated.mean() - control.mean())

        if binary:
            interval, se, p_value, method_note = self._binary_inference(
                treated, control, config.ci_level
            )
        else:
            interval, se, p_value, method_note = self._continuous_inference(
                treated, control, config.ci_level
            )

        # Same coverage level as the analytic and bootstrap intervals, so a
        # caller setting ci_level does not get a counterfactual band silently
        # computed at a different confidence level than the rest of the result.
        counterfactual_critical = float(stats.norm.ppf(1 - (1 - config.ci_level) / 2))

        def point_from(indices: np.ndarray) -> float:
            w = treatment[indices]
            y = outcome[indices]
            if w.sum() == 0 or (1 - w).sum() == 0:
                return float("nan")
            return float(y[w == 1].mean() - y[w == 0].mean())

        return EffectEstimate(
            estimand="ATE",
            method=self.name,
            point=point,
            se=se,
            ci=interval,
            ci_basis="analytic",
            bootstrap_ci=bootstrap_effect(point_from, outcome.size, config),
            p_value=p_value,
            n_treated=int(treated.size),
            n_control=int(control.size),
            scale="percentage_point" if binary else "absolute",
            diagnostics={
                "interval_method": Diagnostic(
                    name="interval_method",
                    value=float(treated.size + control.size),
                    unit="observations",
                    interpretation=method_note,
                ),
                "treated_share": Diagnostic(
                    name="treated_share",
                    value=float(treated.size / outcome.size),
                    unit="proportion",
                    interpretation="Share of units assigned to treatment.",
                ),
            },
            assumptions=(
                AssumptionCheck(
                    name="randomised_assignment",
                    status="NOT_TESTABLE",
                    statistic=None,
                    threshold=None,
                    interpretation=(
                        "This estimator is unbiased only if treatment was randomly "
                        "assigned. Nothing in the data can establish that; it is a "
                        "claim about how the data came to exist."
                    ),
                    remedy=(
                        "Confirm the assignment procedure. If assignment was not "
                        "random, use an estimator that adjusts for covariates."
                    ),
                ),
            ),
            counterfactual=CounterfactualPoint(
                observed_mean=float(treated.mean()),
                predicted_absent_treatment=float(control.mean()),
                lower=float(
                    control.mean()
                    - counterfactual_critical * control.std(ddof=1) / np.sqrt(control.size)
                ),
                upper=float(
                    control.mean()
                    + counterfactual_critical * control.std(ddof=1) / np.sqrt(control.size)
                ),
            ),
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )

    @staticmethod
    def _binary_inference(
        treated: np.ndarray, control: np.ndarray, level: float
    ) -> tuple[Interval, float, float, str]:
        interval = newcombe_interval(
            successes_a=int(treated.sum()),
            trials_a=int(treated.size),
            successes_b=int(control.sum()),
            trials_b=int(control.size),
            level=level,
        )
        p_a = treated.mean()
        p_b = control.mean()
        se = float(
            np.sqrt(p_a * (1 - p_a) / treated.size + p_b * (1 - p_b) / control.size)
        )
        p_value = (
            float(2 * (1 - stats.norm.cdf(abs(p_a - p_b) / se))) if se > 0 else 1.0
        )
        note = "Newcombe hybrid-score interval for a difference of proportions."
        return interval, se, p_value, note

    @staticmethod
    def _continuous_inference(
        treated: np.ndarray, control: np.ndarray, level: float
    ) -> tuple[Interval, float, float, str]:
        result = stats.ttest_ind(treated, control, equal_var=False)
        se = float(
            np.sqrt(
                treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size
            )
        )
        point = float(treated.mean() - control.mean())
        critical = float(stats.t.ppf(1 - (1 - level) / 2, df=result.df))
        interval = Interval(
            lower=point - critical * se, upper=point + critical * se, level=level
        )
        return interval, se, float(result.pvalue), "Welch t-interval for a difference of means."
