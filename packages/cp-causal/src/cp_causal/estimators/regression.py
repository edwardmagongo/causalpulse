"""Covariate-adjusted estimation for randomised designs."""

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
from cp_core.values import Diagnostic, EstimatorConfig
from cp_statistics.covariance import ols_fit, robust_covariance

from cp_causal._support import (
    arrays,
    bootstrap_effect,
    config_hash,
    critical_value,
    degeneracy_diagnostic,
    normal_p_value,
    wald_interval,
)
from cp_causal.base import Requirements
from cp_causal.registry import register


def lin_design(covariates: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """Lin's design: intercept, treatment, centred covariates, interactions.

    Public because `cp_causal.sensitivity` refits this exact design to bound
    omitted-variable bias. A private copy there would be a second definition
    of the same matrix, free to drift from this one -- the failure Phase 7
    found when two estimators had each written out the same placebo formula.

    Centring is what makes the treatment coefficient the ATE. Without it the
    coefficient is the effect at X = 0, which is both a different quantity and
    a moving target under any rescaling of the covariates.
    """
    centred = covariates - covariates.mean(axis=0)
    return np.column_stack(
        [
            np.ones(treatment.size),
            treatment,
            centred,
            treatment[:, None] * centred,
        ]
    )


@register
class LinRegressionAdjustment:
    """Lin (2013) interaction-adjusted estimator with HC2 standard errors."""

    name = "lin_regression"
    requirements = Requirements(
        needs_covariates=True, min_treated_units=5, min_control_units=5
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        covariates, treatment, outcome = arrays(dataset)

        fit = ols_fit(lin_design(covariates, treatment), outcome)
        # HC2 rather than HC1: it divides by the leverage complement, which
        # behaves better when a few observations are influential.
        covariance = robust_covariance(fit, kind="HC2")
        point = float(fit.beta[1])
        se = float(np.sqrt(covariance[1, 1]))

        critical = critical_value(config.ci_level)
        interval = wald_interval(point, se, config.ci_level)
        p_value = normal_p_value(point, se)

        def point_from(indices: np.ndarray) -> float:
            w = treatment[indices]
            if w.sum() < 2 or (1 - w).sum() < 2:
                return float("nan")
            resampled = ols_fit(lin_design(covariates[indices], w), outcome[indices])
            return float(resampled.beta[1])

        bootstrap_result = bootstrap_effect(point_from, outcome.size, config)

        control_mean = float(outcome[treatment == 0].mean())
        return EffectEstimate(
            estimand="ATE",
            method=self.name,
            point=point,
            se=se,
            ci=interval,
            ci_basis="analytic",
            bootstrap_ci=bootstrap_result.interval,
            p_value=p_value,
            n_treated=int((treatment == 1).sum()),
            n_control=int((treatment == 0).sum()),
            scale="percentage_point"
            if dataset.schema.outcome_type == "binary"
            else "absolute",
            diagnostics={
                "r_squared": Diagnostic(
                    name="r_squared",
                    value=float(
                        1.0 - fit.residuals.var(ddof=0) / outcome.var(ddof=0)
                    ),
                    unit=None,
                    interpretation="Share of outcome variance explained by the design.",
                ),
                "n_parameters": Diagnostic(
                    name="n_parameters",
                    value=float(fit.k),
                    unit="parameters",
                    interpretation="Columns in the Lin design matrix.",
                ),
                "bootstrap_degenerate_replicates": degeneracy_diagnostic(
                    bootstrap_result.n_degenerate, min_arm_size=2
                ),
            },
            assumptions=(
                AssumptionCheck(
                    name="randomised_assignment",
                    kind="identification",
                    status="NOT_TESTABLE",
                    statistic=None,
                    threshold=None,
                    interpretation=(
                        "Lin adjustment removes finite-sample imbalance in the "
                        "covariates supplied; it does not correct for confounding "
                        "by anything unobserved. Randomisation is what licenses "
                        "the causal reading, and it cannot be checked from data."
                    ),
                    remedy=(
                        "Use a propensity or doubly-robust estimator if "
                        "assignment was not random."
                    ),
                ),
                AssumptionCheck(
                    name="outcome_model_specification",
                    kind="data_adequacy",
                    status="NOT_TESTABLE",
                    statistic=None,
                    threshold=None,
                    interpretation=(
                        "The adjustment is linear in the supplied covariates. Under "
                        "randomisation Lin stays consistent even when that linear "
                        "form is wrong, but its precision advantage shrinks."
                    ),
                    remedy=(
                        "Add interactions or transformed covariates if the "
                        "relationship is non-linear."
                    ),
                ),
            ),
            counterfactual=CounterfactualPoint(
                observed_mean=float(outcome[treatment == 1].mean()),
                predicted_absent_treatment=control_mean,
                lower=control_mean - critical * se,
                upper=control_mean + critical * se,
            ),
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )
