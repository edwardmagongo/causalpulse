"""Augmented inverse probability weighting (AIPW).

Consistent if either the propensity model or the outcome model is correct, so
a single modelling mistake does not invalidate the estimate. Cross-fitting is
mandatory rather than optional: in-sample nuisance predictions reintroduce the
overfitting bias the augmentation term exists to remove.
"""

from __future__ import annotations

import time
from typing import cast

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import CounterfactualPoint, EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, EstimatorConfig
from sklearn.linear_model import LinearRegression

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
from cp_causal.propensity import (
    PropensityFit,
    fit_propensity,
    overlap_checks,
    resolve_trim,
    standardised_mean_differences,
)
from cp_causal.registry import register


def _resolve_nuisance_covariates(
    dataset: CausalDataset, config: EstimatorConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Covariate matrices for the propensity and outcome nuisance models.

    Both default to `dataset.schema.covariates` — the analyst's declared
    adjustment set — so out of the box AIPW reads exactly what
    `DifferenceInMeans`, `LinRegressionAdjustment` and
    `InverseProbabilityWeighting` read on the same dataset. `config.options`
    may override either independently via `"propensity_covariates"` or
    `"outcome_covariates"` (each a tuple of column names).
    """
    schema = dataset.schema
    role_columns = {schema.unit, schema.treatment, schema.outcome}
    if schema.time is not None:
        role_columns.add(schema.time)
    if schema.cohort is not None:
        role_columns.add(schema.cohort)

    # Two declared columns are not structural roles but are just as
    # forbidden, and neither is caught by the "not present in the frame"
    # check because both are in the frame by construction. An auxiliary is
    # declared precisely because it may carry outcome information no
    # estimator may see -- §4.2 of the Phase 12 spec calls keeping it out of
    # the adjustment set a correctness requirement, enforced rather than
    # documented -- and a negative control outcome is chosen because the
    # treatment cannot affect it, which adjusting for it would destroy.
    # `CausalSchema` already refuses either as a `covariates` entry; this is
    # the same rule for the option-supplied path.
    forbidden: dict[str, str] = {name: "auxiliary" for name in schema.auxiliary_columns}
    if schema.negative_control_outcome is not None:
        forbidden[schema.negative_control_outcome] = "negative control outcome"

    def resolve(option_name: str) -> np.ndarray:
        raw = config.options.get(option_name)
        if raw is None:
            names: tuple[str, ...] = schema.covariates
        else:
            if isinstance(raw, str):
                raise ValueError(
                    f"options[{option_name!r}] is a bare string ({raw!r}); a "
                    f"tuple of column names is expected, e.g. ({raw!r},)"
                )
            names = cast("tuple[str, ...]", tuple(raw))  # type: ignore[arg-type]
            for name in names:
                if name not in dataset.frame.columns:
                    raise ValueError(
                        f"options[{option_name!r}] names column {name!r}, which "
                        "is not present in the frame"
                    )
                if name in role_columns:
                    raise ValueError(
                        f"options[{option_name!r}] names column {name!r}, which "
                        "is a role column and cannot be used as a covariate"
                    )
                if name in forbidden:
                    raise ValueError(
                        f"options[{option_name!r}] names column {name!r}, which "
                        f"is declared as a {forbidden[name]} and cannot be used "
                        "as a covariate"
                    )
        if not names:
            return np.empty((len(dataset.frame), 0))
        return dataset.frame[list(names)].to_numpy(dtype=float)

    return resolve("propensity_covariates"), resolve("outcome_covariates")


def _cross_fitted_outcome_models(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    folds: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions of E[Y|X, T=1] and E[Y|X, T=0].

    Each unit's prediction comes from arm-specific linear models trained on
    the other folds only, so the outcome model never sees the unit it
    predicts for — the same cross-fitting discipline `fit_propensity` already
    applies to the propensity model.
    """
    n = treatment.size
    order = rng.permutation(n)
    mu_treated = np.zeros(n)
    mu_control = np.zeros(n)

    for fold in range(folds):
        held_out = order[fold::folds]
        train = np.setdiff1d(order, held_out)
        train_treated = train[treatment[train] == 1]
        train_control = train[treatment[train] == 0]
        if train_treated.size < 2 or train_control.size < 2:
            raise ValueError(
                "cross-fitting produced a fold with too few units in one arm; "
                "reduce folds or supply more data"
            )
        mu_treated[held_out] = (
            LinearRegression()
            .fit(covariates[train_treated], outcome[train_treated])
            .predict(covariates[held_out])
        )
        mu_control[held_out] = (
            LinearRegression()
            .fit(covariates[train_control], outcome[train_control])
            .predict(covariates[held_out])
        )
    return mu_treated, mu_control


@register
class AugmentedIPW:
    """Cross-fitted AIPW with influence-function standard errors.

    Both nuisance models — the propensity model and the outcome regression —
    read `dataset.schema.covariates` by default, exactly like every other
    estimator in this package. `config.options` accepts two optional keys to
    override that per model:

    - `"propensity_covariates"`: a tuple of column names for the propensity
      model only.
    - `"outcome_covariates"`: a tuple of column names for the outcome
      regression only.
    - `"trim"`: the propensity-clipping threshold, in `[0, 0.5)`, defaulting
      to `POSITIVITY_THRESHOLD`. Read through the same `resolve_trim` that
      `InverseProbabilityWeighting` uses, so the two clip identically.

    Each supplied name must be a column present in the frame and must not be
    one of the schema's role columns (`unit`, `treatment`, `outcome`, `time`,
    `cohort`); a `ValueError` names the offending column and the option it
    came from otherwise.

    These options exist for two reasons. First, using different covariate
    sets for the two nuisance models is real analytic practice — an analyst
    may want a richer, more predictive set for the outcome regression than
    the identifying set used for the propensity model, or vice versa.
    Second, it is what makes double robustness testable without corrupting
    the default contract: a caller can degrade one nuisance model in
    isolation (e.g. `propensity_covariates` naming a strict subset) while
    leaving the other on the full declared covariate set, and AIPW should
    still recover the effect. Doing this by narrowing `schema.covariates`
    itself would cripple both nuisance models at once, since that schema is
    also what `arrays()` and every other estimator read — these options let
    the propensity and outcome covariate sets vary independently instead.
    """

    name = "aipw"
    requirements = Requirements(
        needs_covariates=True, min_treated_units=25, min_control_units=25
    )
    cross_fit_folds = 5

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        _, treatment, outcome = arrays(dataset)
        covariates, outcome_covariates = _resolve_nuisance_covariates(dataset, config)
        trim = resolve_trim(config.options)
        seed = 0 if config.random_seed is None else config.random_seed
        # One generator derived from the config's seed at the estimator boundary,
        # then threaded down. Helpers never seed their own generator, so every
        # random choice in this fit — including the cross-fitting fold split —
        # traces back to this root SeedSequence.
        rng = np.random.default_rng(np.random.SeedSequence(seed))

        scores, fitted = self._scores(
            covariates, outcome_covariates, treatment, outcome, rng, trim
        )
        point = float(scores.mean())
        # The influence function is the per-unit score itself, so its spread is
        # the standard error directly — no separate variance formula needed.
        se = float(scores.std(ddof=1) / np.sqrt(scores.size))

        critical = critical_value(config.ci_level)
        interval = wald_interval(point, se, config.ci_level)

        def point_from(indices: np.ndarray) -> float:
            w = treatment[indices]
            if w.sum() < 25 or (1 - w).sum() < 25:
                return float("nan")
            replicate_scores, _ = self._scores(
                covariates[indices],
                outcome_covariates[indices],
                w,
                outcome[indices],
                rng,
                trim,
            )
            return float(replicate_scores.mean())

        bootstrap_result = bootstrap_effect(point_from, outcome.size, config)

        weights = np.where(
            treatment == 1, 1.0 / fitted.scores, 1.0 / (1.0 - fitted.scores)
        )
        smd_before = standardised_mean_differences(covariates, treatment)
        smd_after = standardised_mean_differences(covariates, treatment, weights)
        control_mean = float(outcome[treatment == 1].mean()) - point

        return EffectEstimate(
            estimand="ATE",
            method=self.name,
            point=point,
            se=se,
            ci=interval,
            ci_basis="analytic",
            bootstrap_ci=bootstrap_result.interval,
            p_value=normal_p_value(point, se),
            n_treated=int((treatment == 1).sum()),
            n_control=int((treatment == 0).sum()),
            scale="percentage_point"
            if dataset.schema.outcome_type == "binary"
            else "absolute",
            diagnostics={
                "cross_fit_folds": Diagnostic(
                    name="cross_fit_folds",
                    value=float(self.cross_fit_folds),
                    unit="folds",
                    interpretation=(
                        "Each unit's nuisance predictions come from a model that "
                        "never saw it, so overfitting does not leak into the estimate."
                    ),
                ),
                "score_dispersion": Diagnostic(
                    name="score_dispersion",
                    value=float(scores.std(ddof=1)),
                    unit=None,
                    interpretation="Spread of the per-unit influence-function scores.",
                ),
                "n_trimmed": Diagnostic(
                    name="n_trimmed",
                    value=float(fitted.trimmed_mask.sum()),
                    unit="units",
                    interpretation="Units whose propensity score was clipped to the trim bounds.",
                ),
                "bootstrap_degenerate_replicates": degeneracy_diagnostic(
                    bootstrap_result.n_degenerate, min_arm_size=25
                ),
                "per_unit_propensity": Diagnostic(
                    name="per_unit_propensity",
                    value=fitted.scores,
                    unit="probability",
                    interpretation=(
                        "Estimated P(treated | covariates) per unit, after "
                        f"clipping to the trim bounds {fitted.trim_bounds}. These "
                        "are the scores the weights were built from, not an "
                        "untrimmed refit."
                    ),
                ),
                "smd_before": Diagnostic(
                    name="smd_before",
                    value=smd_before,
                    unit="standard deviations",
                    interpretation=(
                        "Per-covariate standardised mean difference between arms "
                        "before weighting, in schema covariate order."
                    ),
                ),
                "smd_after": Diagnostic(
                    name="smd_after",
                    value=smd_after,
                    unit="standard deviations",
                    interpretation=(
                        "The same comparison after weighting. The pair is what a "
                        "love plot shows; the gap between them is what the "
                        "weighting corrected."
                    ),
                ),
            },
            assumptions=overlap_checks(fitted.scores, treatment, smd_after),
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

    def _scores(
        self,
        covariates: np.ndarray,
        outcome_covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        rng: np.random.Generator,
        trim: float,
    ) -> tuple[np.ndarray, PropensityFit]:
        """Per-unit AIPW scores; their mean is the estimate.

        `covariates` drives the propensity model; `outcome_covariates` drives
        the outcome regression. Both are resolved by
        `_resolve_nuisance_covariates` before this is called, so they coincide
        (both equal to `schema.covariates`) unless `config.options` overrode
        one of them.

        `trim` is resolved once in `fit` and passed down rather than re-read
        here, so the main fit and every bootstrap replicate clip at the same
        threshold; re-reading it per replicate would be the same value by
        construction, but only by construction.

        The propensity fit is cross-fitted through `fit_propensity` itself
        (reused rather than reimplemented) and returned alongside the scores
        so the caller can build diagnostics from the same out-of-fold scores
        the estimate was built on, without a second, in-sample fit.
        """
        fitted = fit_propensity(
            covariates,
            treatment,
            cross_fit_folds=self.cross_fit_folds,
            rng=rng,
            trim=trim,
        )
        mu_treated, mu_control = _cross_fitted_outcome_models(
            outcome_covariates, treatment, outcome, self.cross_fit_folds, rng
        )
        e = fitted.scores
        scores = (
            mu_treated
            - mu_control
            + treatment * (outcome - mu_treated) / e
            - (1 - treatment) * (outcome - mu_control) / (1 - e)
        )
        return np.asarray(scores), fitted
