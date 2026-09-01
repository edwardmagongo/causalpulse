"""Imputing outcomes that went missing, and comparing that to dropping them.

Complete-case analysis and imputation are two answers to the same question,
and neither is free: dropping incomplete rows is unbiased only when values are
missing completely at random, while imputing them trusts a model of the very
values nobody observed. This module builds both and reports them side by side
rather than picking one on the caller's behalf.

Imputation is not a repair for data missing not at random. Nothing here
recovers a value whose absence depends on the value itself; that is what the
tipping-point bounds in `sensitivity` are for.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import (
    AssumptionCheck,
    EffectEstimate,
    FitMetadata,
    worst_status,
)
from cp_core.values import Diagnostic, EstimatorConfig, Interval
from cp_statistics.pooling import PooledEstimate, pool_rubin
from scipy import stats
from sklearn.linear_model import LogisticRegression

from cp_causal._support import config_hash
from cp_causal.registry import get_estimator

MAX_RESAMPLE_ATTEMPTS = 20
DEFAULT_IMPUTATIONS = 20

PANEL_IMPUTATION_REASON = (
    "Imputation here fills a missing outcome from a cross-sectional model, "
    "and {name} needs panel data, whose gaps are an attrition pattern over "
    "time rather than independent draws."
)
CONTINUOUS_IMPUTATION_REASON = (
    "The imputation model is logistic and the fill is a Bernoulli draw, "
    "which needs a binary outcome."
)
NO_MISSING_DATA_REASON = (
    "The dataset has no missing outcomes, so there is nothing to impute and "
    "complete-case analysis is the same analysis."
)
NO_STANDARD_ERROR_REASON = (
    "Rubin's rules pool a within-imputation variance, and {name} reports no "
    "standard error, so its several fits cannot be combined into one."
)

_DISAGREEMENT_NOTE = (
    "This check reached different verdicts across the imputed datasets, so "
    "it is reported at the most severe verdict any of them reached rather "
    "than whichever one the first completion happened to reach. Its "
    "statistic and threshold are dropped, because no single completion's "
    "numbers describe the pooled result."
)
_INCONSISTENT_PRESENCE_NOTE = (
    "This check was reported by only some of the imputed datasets' fits, "
    "which is the same kind of inconsistency as a disagreeing verdict, so "
    "it is reported as untested rather than as whichever completions "
    "happened to carry it."
)


def _imputation_design(dataset: CausalDataset) -> np.ndarray:
    """Treatment, covariates and any declared auxiliary, as one float matrix.

    The auxiliary is included here and nowhere else. That asymmetry is the
    whole point: a column the analysis model can also see carries no
    information the analysis model lacks, so imputation would gain nothing
    from it.
    """
    schema = dataset.schema
    frame = dataset.frame
    columns = [schema.treatment, *schema.covariates, *schema.auxiliary_columns]
    return np.column_stack([frame[column].to_numpy(dtype=float) for column in columns])


def impute_once(
    dataset: CausalDataset,
    rng: np.random.Generator,
    *,
    draw_parameters: bool,
) -> CausalDataset:
    """One completed copy, with every missing outcome drawn from a fitted model.

    `draw_parameters=True` bootstrap-resamples the observed rows before
    fitting, so each call sees a different imputation model. That is what
    makes a set of these draws proper multiple imputation rather than several
    reruns of one model's Bernoulli noise, and it is what puts real
    between-imputation variance into Rubin's rules.

    `draw_parameters=False` fits once at the point estimate and takes a single
    stochastic draw. That is single imputation, and it is offered so the cost
    of treating imputed values as observed can be measured rather than
    asserted. It draws a value rather than taking the most likely one, which
    would push every imputed unit to the majority outcome and bias the arm.
    """
    schema = dataset.schema
    frame = dataset.frame
    outcome = frame[schema.outcome].to_numpy(dtype=float)
    missing = np.isnan(outcome)
    if not missing.any():
        return dataset

    # Every column of `_imputation_design`, not just the auxiliary. Guarding
    # only the auxiliary left a gap in a covariate to reach sklearn, which
    # answers "Input X contains NaN" -- naming no column, from a library the
    # caller never invoked. That is the late, undiagnosable failure
    # `require_no_missing` exists to end, and it is reachable through
    # `sensitivity_report`, which hands `compare_missing_data` the raw
    # dataset by design. The treatment column needs no check here:
    # `CausalDataset` refuses a gap in it at construction.
    for column in schema.auxiliary_columns:
        if frame[column].isna().any():
            raise ValueError(
                f"auxiliary column {column!r} contains missing values. An "
                "auxiliary exists to carry information where the outcome does "
                "not, so there is no fallback for filling one."
            )
    for column in schema.covariates:
        if frame[column].isna().any():
            raise ValueError(
                f"covariate {column!r} contains missing values. The imputation "
                "model reads it alongside the treatment and any auxiliary, and "
                "this phase fills gaps in the outcome only, so there is nothing "
                "to fall back on for a gap in a covariate."
            )

    observed = np.flatnonzero(~missing)
    if np.unique(outcome[observed]).size < 2:
        raise ValueError(
            "the observed outcomes are all one class, so no logistic model "
            "can be fitted; filling the gaps with the only value ever seen "
            "would manufacture certainty rather than impute"
        )

    design = _imputation_design(dataset)
    rows = observed
    if draw_parameters:
        # A resample can land on a single class. Redraw rather than fall back
        # to the full sample, which would silently turn this draw into the
        # single-imputation path and shrink between-imputation variance.
        for _ in range(MAX_RESAMPLE_ATTEMPTS):
            candidate = rng.choice(observed, size=observed.size, replace=True)
            if np.unique(outcome[candidate]).size >= 2:
                rows = candidate
                break
        else:
            raise ValueError(
                f"no bootstrap resample of the observed outcomes contained "
                f"both classes in {MAX_RESAMPLE_ATTEMPTS} attempts; the "
                "observed outcomes are too close to constant to impute from"
            )

    model = LogisticRegression(max_iter=1000).fit(design[rows], outcome[rows])
    # `predict_proba` orders its columns by `classes_`, which is sorted, so
    # column 1 is P(outcome = 1) given the guard above that both classes are
    # present. Asserted rather than assumed: a silent reversal here would
    # invert every imputed value.
    if not np.array_equal(model.classes_, np.array([0.0, 1.0])):
        raise ValueError(f"unexpected outcome classes {model.classes_!r}")
    probability = model.predict_proba(design[missing])[:, 1]

    filled = outcome.copy()
    filled[missing] = (rng.random(probability.size) < probability).astype(float)
    return CausalDataset(
        frame=frame.assign(**{schema.outcome: filled}), schema=schema
    )


def _pool_assumptions(fits: list[EffectEstimate]) -> tuple[AssumptionCheck, ...]:
    """The assumption checks every completion agrees on; disagreements pooled
    to the worst verdict any completion reached.

    A check whose status differs across completions is reported at the most
    severe status seen, not as `NOT_TESTABLE`. Reporting it as untestable was
    this function's original rule and it was an UPGRADE in disguise:
    `score_validity` names `NOT_TESTABLE` checks in its rationale without
    moving the band, and `_SEVERITY` does not rank it, so a `WARN` eighteen
    completions out of twenty reached stopped counting anywhere downstream --
    measured at `MODERATE` against the `WEAK` every individual completion
    scored, and reachable with `FAIL` in place of `WARN`. Two meanings had
    been loaded onto one status: "cannot be answered from data", which the
    design ceiling already prices in, and "the answer varied", which nothing
    priced at all.

    The statistic and threshold are dropped on a disagreement even though the
    status survives, because they came from whichever completion happened to
    be first and no single completion's numbers describe the pooled result.

    One asymmetry is deliberate: a `PASS` on some completions and
    `NOT_TESTABLE` on others pools to `NOT_TESTABLE`. Nothing severe is
    hidden by that -- `PASS` is already the mildest rank -- and a check a
    completion could not answer is not a `PASS`.

    Absence is handled separately and still yields `NOT_TESTABLE`: a check
    present in some completions' assumption lists and absent from others has
    not been consistently tested at all, which is a different thing from a
    verdict that varied. It is neither silently dropped (which would hide it
    from a reader entirely) nor kept from whichever completions happened to
    carry it (which would report a partial answer as though it were a full
    one). In the estimators registered as of this writing, a fit's assumption
    checks are a fixed set of names determined by the estimator and config,
    never by which values happened to be imputed, so that branch does not
    fire in practice -- but nothing in the `Estimator` protocol guarantees
    it, and a defect there would be the same shape as the family predicates
    that shipped wrong classifications in earlier phases.

    The status-mismatch branch is not dead in the same way: on real fixtures,
    `causal_forest`'s `calibration_test` was observed to disagree across
    completions (`WARN` on some, `PASS` on others) at `m = 20`, on two of ten
    tried seeds, and now pools to `WARN`.
    """
    by_name: list[dict[str, AssumptionCheck]] = [
        {check.name: check for check in fit.assumptions} for fit in fits
    ]
    # An ordered set of every check name seen anywhere, first-fit order
    # first, so the pooled result matches the ordinary case's ordering.
    seen: dict[str, None] = {}
    for checks in by_name:
        for name in checks:
            seen[name] = None

    pooled: list[AssumptionCheck] = []
    for name in seen:
        present = [checks[name] for checks in by_name if name in checks]
        template = present[0]
        if len(present) < len(fits):
            pooled.append(
                replace(
                    template,
                    status="NOT_TESTABLE",
                    statistic=None,
                    threshold=None,
                    interpretation=(
                        f"{template.interpretation} {_INCONSISTENT_PRESENCE_NOTE}"
                    ),
                )
            )
        elif len({check.status for check in present}) > 1:
            statuses = [check.status for check in present]
            status = worst_status(statuses)
            if status == "PASS" and "NOT_TESTABLE" in statuses:
                # Nothing severe is being hidden -- PASS is already the
                # mildest rank -- but a check some completion could not
                # answer is not a PASS either.
                status = "NOT_TESTABLE"
            pooled.append(
                replace(
                    template,
                    status=status,
                    statistic=None,
                    threshold=None,
                    interpretation=f"{template.interpretation} {_DISAGREEMENT_NOTE}",
                )
            )
        else:
            pooled.append(template)
    return tuple(pooled)


def _pooling_diagnostics(pooled: PooledEstimate, m: int) -> dict[str, Diagnostic]:
    return {
        "imputations": Diagnostic(
            name="imputations",
            value=float(m),
            unit="datasets",
            interpretation=(
                "How many completed datasets the estimate was pooled from. "
                "More reduces the simulation noise Rubin's rules add, and "
                "does nothing about the uncertainty the missing data itself "
                "creates."
            ),
        ),
        "within_variance": Diagnostic(
            name="within_variance",
            value=pooled.within_variance,
            unit=None,
            interpretation=(
                "Average sampling variance the completed-data analyses "
                "reported for themselves, as if each completion were the "
                "observed data."
            ),
        ),
        "between_variance": Diagnostic(
            name="between_variance",
            value=pooled.between_variance,
            unit=None,
            interpretation=(
                "Variance of the point estimates across completions. This is "
                "the part the missing data is responsible for; it goes to "
                "zero as the missing fraction does."
            ),
        ),
        "proportion_variance_missing": Diagnostic(
            name="proportion_variance_missing",
            value=pooled.proportion_variance_missing,
            unit=None,
            interpretation=(
                "Share of the total variance attributable to the missing "
                "data -- Rubin's lambda. Near zero means the gaps cost this "
                "estimate little; near one means most of what is reported "
                "rests on the imputation model rather than on observed "
                "values. This is the quantity the degrees-of-freedom "
                "adjustment consumes, and it is not the fraction of missing "
                "information."
            ),
        ),
        "fraction_missing_information": Diagnostic(
            name="fraction_missing_information",
            value=pooled.fraction_missing_information,
            unit=None,
            interpretation=(
                "How much of the information about this estimate the missing "
                "data cost. Larger than the variance share above, because it "
                "carries a correction for the finite number of imputations: "
                "on this module's own worked example the two are 0.133 and "
                "0.125."
            ),
        ),
        "barnard_rubin_df": Diagnostic(
            name="barnard_rubin_df",
            value=pooled.degrees_of_freedom,
            unit="degrees of freedom",
            interpretation=(
                "Degrees of freedom of the t reference the interval uses, "
                "adjusted for the finite complete-data sample. Always below "
                "the degrees of freedom a complete dataset would have had."
            ),
        ),
    }


def multiple_impute(
    dataset: CausalDataset,
    estimator_name: str,
    config: EstimatorConfig,
    *,
    m: int = DEFAULT_IMPUTATIONS,
    seed: int = 0,
) -> EffectEstimate | None:
    """`m` completions, each refitted through the registry and pooled by Rubin's rules.

    Returns an ordinary `EffectEstimate` so the result reaches `compare`, the
    validity rubric and the language lint like any other, with
    `ci_basis="rubin"` naming the inference honestly rather than presenting a
    pooled quantity as an ordinary analytic or bootstrap interval.

    `None` when the estimator needs panel data, when the outcome is not
    binary, when nothing is missing, or when the estimator reports no
    standard error; `PANEL_IMPUTATION_REASON`, `CONTINUOUS_IMPUTATION_REASON`,
    `NO_MISSING_DATA_REASON` and `NO_STANDARD_ERROR_REASON` are the matching
    reasons a caller pairs with each case. The four conditions are checked
    separately rather than folded into one predicate, because a predicate
    that admits an estimator whose fields mean something else is how the
    same defect shipped in three consecutive phases of this project.

    The first three conditions are free -- they read the estimator's
    declared requirements and the dataset's own schema and missingness, none
    of which needs a fit. The fourth cannot be: whether an estimator reports
    a standard error is a property of its `fit()`, invisible without calling
    it, so it costs the full `m` refits before this function can return
    `None` on that account. In the estimators registered as of this writing
    that cost never actually bites -- every non-panel estimator either always
    reports a standard error or (like the two panel-only ones) is already
    excluded by the first condition -- so the honest cheapest check available
    is exactly this one: there is no signal available before fitting that
    would let this function skip refits it might still need.

    Costs `m` refits. Measured on a 600-unit dataset at `m = 20`: about
    0.30 seconds for `difference_in_means`, and about 19 seconds for
    `causal_forest`, whose 50-unit-per-arm minimum and per-tree bootstrap
    make it the slowest estimator this function can reach.
    """
    if m < 2:
        raise ValueError(
            "pooling needs at least two imputations; m must be at least two, "
            f"got {m}"
        )

    started = time.perf_counter()
    estimator_class = get_estimator(estimator_name)
    if estimator_class.requirements.needs_panel:
        return None
    if dataset.schema.outcome_type != "binary":
        return None
    if not dataset.frame[dataset.schema.outcome].isna().any():
        return None

    rng = np.random.default_rng(np.random.SeedSequence(seed))
    fits = [
        estimator_class().fit(
            impute_once(dataset, rng, draw_parameters=True), config
        )
        for _ in range(m)
    ]
    if any(fit.se is None for fit in fits):
        return None

    first = fits[0]
    # Two-arm degrees of freedom, uniform across estimators because the
    # `Estimator` protocol exposes no parameter count. For an adjusted
    # estimator this overstates the degrees of freedom by the number of
    # covariates, but the Barnard-Rubin adjustment's own sensitivity to this
    # input is tiny: on a 600-unit two-covariate dataset like this module's
    # own tests, moving `complete_data_df` from 598 (this uniform figure) to
    # 596 (the adjusted one, net of both covariates) moves the 97.5% t
    # quantile from 1.9639388980555033 to 1.963952263347006, a relative
    # difference of about 6.8e-6 -- immaterial next to the between-imputation
    # noise Rubin's rules already carry.
    complete_data_df = float(first.n_treated + first.n_control - 2)
    # `if fit.se is not None` is not a real filter here -- the guard above
    # already returned `None` for this exact `fits` list if any `se` were
    # `None`, so nothing is ever dropped. It exists only to narrow
    # `float | None` to `float` for mypy while keeping this list aligned
    # index-for-index with `[fit.point for fit in fits]`. Do not separate
    # this line from the guard above (e.g. by hoisting the filtered list to
    # a variable computed earlier, or by relaxing the guard to allow some
    # `None`s through) -- doing so could silently misalign points and SEs.
    pooled = pool_rubin(
        [fit.point for fit in fits],
        [fit.se for fit in fits if fit.se is not None],
        complete_data_df=complete_data_df,
    )

    critical = float(
        stats.t.ppf(1 - (1 - config.ci_level) / 2, pooled.degrees_of_freedom)
    )
    half_width = critical * pooled.se
    return EffectEstimate(
        estimand=first.estimand,
        method=estimator_name,
        point=pooled.point,
        se=pooled.se,
        ci=Interval(
            lower=pooled.point - half_width,
            upper=pooled.point + half_width,
            level=config.ci_level,
        ),
        ci_basis="rubin",
        # Deliberately absent: bootstrapping a Rubin-pooled quantity would
        # resample a statistic that is already an average over resampled
        # imputation models, a bootstrap of a bootstrap.
        bootstrap_ci=None,
        # Derivable from the same t reference, but a second inferential
        # quantity needing its own calibration. `is_significant` reads the
        # interval instead.
        p_value=None,
        n_treated=first.n_treated,
        n_control=first.n_control,
        scale=first.scale,
        diagnostics=_pooling_diagnostics(pooled, m),
        assumptions=_pool_assumptions(fits),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator=estimator_name,
            config_hash=config_hash(config),
            runtime_seconds=time.perf_counter() - started,
            random_seed=config.random_seed,
        ),
    )
