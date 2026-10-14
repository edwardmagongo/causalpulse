"""Hierarchical partially-pooled difference-in-differences.

Unit and time effects plus one treatment effect per adoption cohort, the
cohort effects drawn from a common population distribution so that a noisy
cohort is shrunk toward the others. Sampled with NUTS.

Non-centred throughout. With a handful of cohorts the funnel between
`sigma_tau` and the `tau_g` is acute, and a centred parameterisation is
expected to diverge -- the same reason Phase 5's regularised horseshoe is
non-centred.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from cp_core.dataset import CausalDataset
from cp_core.results import AssumptionCheck, EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, EstimatorConfig, Interval

from cp_causal._mcmc import MCMCConfig, run_nuts
from cp_causal._panel import PanelArrays, outcome_scale, panel_arrays
from cp_causal._support import config_hash
from cp_causal.base import Requirements
from cp_causal.bayesian_checks import (
    divergent_transitions_check,
    effective_sample_size_check,
    r_hat_check,
)
from cp_causal.panel_checks import (
    absorbing_treatment_check,
    no_anticipation_check,
    parallel_trends_check,
    placebo_leads,
)
from cp_causal.registry import register

# Exercises shapes and code paths only -- not a trustworthy posterior. Two
# chains, not one: numpyro.diagnostics.gelman_rubin hard-asserts >= 2.
HIER_FAST_MCMC_CONFIG = MCMCConfig(num_warmup=50, num_samples=50, num_chains=2)

# Measured (Task 7), not copied from another estimator. `target_accept_prob`
# is untouched -- still the `MCMCConfig` default of 0.95 -- because raising it
# to chase divergences away without saying so is exactly what this task was
# told not to do.
#
# Seed-0 fits at each scenario's own protocol (PANEL_N_UNITS/PANEL_N_PERIODS
# for P3/P4, HIER_N_UNITS/HIER_N_PERIODS for P7), reading max R-hat / min ESS
# / divergence rate straight off `r_hat_check`/`effective_sample_size_check`/
# `divergent_transitions_check`:
#   P3 (3 cohorts,  400x8):  R-hat 1.0073, min ESS  781, divergent 25/4000 (0.63%), ~26s
#   P4 (3 cohorts,  400x8):  R-hat 1.0020, min ESS  744, divergent  0/4000 (0.00%), ~80s
#   P7 (7 cohorts, 600x20):  R-hat 1.0068, min ESS  261, divergent  0/4000 (0.00%), ~332s
# All three clear R-hat <= 1.01 with effectively zero divergences. Of the
# smaller configs tried (all with 4 chains), 200/200, 500/500 and 700/700 each
# left at least one scenario over the R-hat bound at seed 0. 1000/1000 did
# NOT -- it passed all three at seed 0 (P3 1.0076, P4 1.0023, P7 1.0020) and
# was rejected only by the seed sweep below. Those are two different
# experiments and are kept apart deliberately: a single seed is what made
# 1000/1000 look adequate in the first place.
#
# Finding, not swept under the rug: P3's 3-cohort funnel is seed-sensitive at
# 1000 warmup / 1000 samples. Sweeping seeds 0-9 at that (smaller) config, one
# seed (7) produced R-hat 5.50 / ESS 2.07 / a 25.9% divergence rate -- a
# genuinely stuck chain, not a rounding error -- and two more (3, 9) landed
# just past 1.01 (1.016, 1.012). The same seeds all pass cleanly at 1500
# warmup (seed 7: R-hat 1.0042; seed 3: 1.0065; seed 9: 1.0071), which is why
# warmup was raised rather than samples alone. P7's 7 cohorts were far more
# stable across the same seeds at the smaller config (its own worst case,
# seed 7, was R-hat 1.0167 at 1000 warmup and 1.0062 at 1500) -- consistent
# with this module's docstring: fewer cohorts leave the across-cohort
# variance less identified and the non-centred funnel harder to traverse.
# Task 8's recovery harness should not assume this config guarantees a clean
# fit on every seed for a 3-cohort scenario; it still needs to check
# per-seed diagnostics rather than trusting the config alone.
HIER_PRODUCTION_MCMC_CONFIG = MCMCConfig(num_warmup=1500, num_samples=1000, num_chains=4)


def _model(
    outcome_std: np.ndarray,
    unit_index: np.ndarray,
    time_index: np.ndarray,
    cohort_index: np.ndarray,
    treated_mask: np.ndarray,
    exposure: np.ndarray,
    n_units: int,
    n_periods: int,
    n_cohorts: int,
    sigma_tau_prior_scale: float = 1.0,
) -> None:
    """One observation per (unit, period) row.

    The treatment effect on a treated row is `tau_g + delta * exposure`,
    where `exposure` counts periods since that unit adopted (0 in its first
    treated period) and is 0 on every untreated row.

    **Why the exposure term exists, measured rather than assumed.** Without
    it the model forces one constant effect per cohort. When the true effect
    instead grows with exposure, that misfit does not stay put: it is
    positively correlated with calendar period (later periods hold more
    heavily-exposed units), so `gamma_t` absorbs it, and the cohorts treated
    only in late periods lose the most. Measured on P4 at 15 seeds with the
    constant-only model: cohort effects estimated 3.36 / 2.03 / 0.49 against
    a truth of 3.50 / 2.50 / 1.50 -- the latest-adopting cohort short by
    two thirds -- giving a -17% ATT bias and 0/15 interval coverage, with
    R-hat and divergences clean on every seed. P3, whose effect is genuinely
    constant, was unbiased (+0.0002) under the same model, which is what
    identified the misspecification rather than the sampler as the cause.

    Note the signature: a *shared* slope, not one per cohort. This is the
    smallest extension that lets the effect move with exposure at all, and
    it deliberately stops short of the full cohort-by-exposure surface,
    which was considered and rejected on complexity grounds. It cannot
    represent a non-linear ramp or cohorts whose effects grow at different
    rates; `delta` shrinks toward 0 when the effect is constant, so the
    constant case is nested inside it.

    `cohort_index` is -1 for never-treated units; `treated_mask` is already
    False for every one of their rows, so the gathered value is never used.
    Gathering at a negative index does not error in JAX -- it silently
    reads *some* element (plain negative indexing wraps to the last one;
    the `jnp.clip` below maps it to index 0 instead) -- which is why the
    mask, not the index value, is what keeps a never-treated row from
    contributing a treatment effect.
    """
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    sigma_gamma = numpyro.sample("sigma_gamma", dist.HalfNormal(1.0))
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(1.0))

    z_alpha = numpyro.sample("z_alpha", dist.Normal(jnp.zeros(n_units), 1.0))
    z_gamma = numpyro.sample("z_gamma", dist.Normal(jnp.zeros(n_periods), 1.0))
    alpha = sigma_alpha * z_alpha
    gamma = sigma_gamma * z_gamma

    mu_tau = numpyro.sample("mu_tau", dist.Normal(0.0, 2.5))
    # `sigma_tau_prior_scale` exists so the pooling claim can be MEASURED
    # rather than asserted: partial pooling is this model's whole
    # justification, and with few cohorts a `sigma_tau` posterior that is
    # really just its prior would look identical to one the data informed.
    # Varying this scale and watching whether the posterior follows it is the
    # test. Default 1.0, so every other caller is unaffected.
    sigma_tau = numpyro.sample("sigma_tau", dist.HalfNormal(sigma_tau_prior_scale))
    z_g = numpyro.sample("z_g", dist.Normal(jnp.zeros(n_cohorts), 1.0))
    tau_g = numpyro.deterministic("tau_g", mu_tau + sigma_tau * z_g)

    # Effect growth per period of exposure, shared across cohorts. Weakly
    # informative on the standardised outcome scale, and centred on zero so
    # a genuinely constant effect is the prior's default rather than
    # something the data has to argue the model out of.
    delta = numpyro.sample("delta", dist.Normal(0.0, 0.5))

    safe_index = jnp.clip(jnp.asarray(cohort_index), 0, n_cohorts - 1)
    per_row = tau_g[safe_index] + delta * jnp.asarray(exposure)
    effect = jnp.where(jnp.asarray(treated_mask), per_row, 0.0)

    mean = alpha[jnp.asarray(unit_index)] + gamma[jnp.asarray(time_index)] + effect
    numpyro.sample("obs", dist.Normal(mean, sigma_y), obs=jnp.asarray(outcome_std))


# eq=False: holds ndarray fields, so a generated __eq__ would raise on the
# ambiguous truth value of an array rather than compare -- the same reason
# `PanelArrays` (this module's input) opts out of it.
@dataclass(frozen=True, eq=False)
class ModelInputs:
    """The exact arrays and scalars `_model` (and its priors) need.

    `outcome_std` is `outcome` standardised to zero mean and unit variance,
    since `_model`'s priors (`sigma_y ~ HalfNormal(1)`, `mu_tau ~ Normal(0,
    2.5)`, ...) are calibrated for a unit-scale outcome. `outcome_mean` and
    `outcome_sd` are retained so Task 5 can rescale `tau_g` back to the
    outcome's original units.
    """

    outcome_std: np.ndarray
    unit_index: np.ndarray
    time_index: np.ndarray
    cohort_index: np.ndarray
    treated_mask: np.ndarray
    n_units: int
    n_periods: int
    n_cohorts: int
    # Treated (unit, period) CELL count per cohort code, not treated UNIT
    # count. The overall ATT the recovery tests score against is the mean over
    # every treated cell, which implicitly weights each cohort by
    # units x post-periods. Weighting by units instead was measured on P4 to
    # give 2.5028 against its registered 2.8361 (the cell-mean measures
    # 2.8356): roughly 150 Monte
    # Carlo standard errors off, reported as estimator bias when it would
    # actually be this weighting error. See `test_cohort_cell_counts_...`.
    cohort_cell_counts: np.ndarray
    # Periods since adoption on each row: 0 in a unit's first treated period,
    # and 0 on every untreated row (where `_model` masks it away regardless).
    # Feeds the model's shared exposure slope.
    exposure: np.ndarray
    # Mean exposure across treated cells only. This is what turns a draw of
    # `delta` into its contribution to the aggregate ATT, and it must be the
    # treated-cell mean rather than the mean over all rows -- the untreated
    # zeros would otherwise dilute it.
    mean_treated_exposure: float
    outcome_mean: float
    outcome_sd: float


def build_model_inputs(panel: PanelArrays) -> ModelInputs:
    """Map a resolved panel into the arrays `_model` needs.

    Cohort codes: `panel.cohort` holds each unit's first-treated period (`0`
    for never-treated, per `CausalDataset.cohort_by_unit`'s convention). The
    distinct adoption periods are mapped to contiguous codes `0..n_cohorts-1`
    in ascending period order, so `cohort_cell_counts[i]` and the model's
    `tau_g[i]` refer to the same cohort. Never-treated units get code `-1`;
    `treated_mask` (taken directly from `panel.treatment`, the same column
    `cohort_by_unit` derived adoption from) is already False on every one of
    their rows, so `-1` is never used to index into `tau_g`.

    Small-cohort policy (spec section 5.2): a cohort with very few treated
    units is neither merged into a neighbouring cohort nor dropped. It is left
    to the model's own hierarchical shrinkage -- `tau_g = mu_tau + sigma_tau *
    z_g` pools every cohort toward `mu_tau`, and a cohort with little data
    simply shrinks harder, the same partial-pooling mechanism the module
    docstring describes. Merging or dropping would need an arbitrary
    threshold and would throw away exactly the data point the hierarchy
    exists to make usable; shrinkage does that job without one.
    """
    adoption_periods = panel.adoption_periods
    n_cohorts = len(adoption_periods)
    code_by_period = {period: code for code, period in enumerate(adoption_periods)}

    unit_cohort_code = np.array(
        [code_by_period.get(int(period), -1) for period in panel.cohort],
        dtype=np.int64,
    )
    cohort_index = unit_cohort_code[panel.unit_index]
    treated_mask = panel.treatment.astype(bool)

    cohort_cell_counts = np.array(
        [
            int(((cohort_index == code) & treated_mask).sum())
            for code in range(n_cohorts)
        ],
        dtype=np.int64,
    )

    outcome_mean = float(panel.outcome.mean())
    outcome_sd = float(panel.outcome.std())
    if outcome_sd == 0.0:
        # Degenerate (constant) outcome: nothing to standardise, and dividing
        # by zero would turn every value into nan/inf. Falls back to leaving
        # the (already-constant) outcome as-is rather than raising, matching
        # the defensive `+ 1e-9`-style guards elsewhere in this package
        # (e.g. `bsts.py`'s `treated_sd`).
        outcome_sd = 1.0
    outcome_std = (panel.outcome - outcome_mean) / outcome_sd

    # Periods since adoption, 0-based in the first treated period. Untreated
    # rows are forced to 0 rather than left negative: `_model` masks them out
    # anyway, but a large negative exposure multiplying `delta` inside the
    # unmasked branch would still be evaluated, and keeping it at 0 makes the
    # masked and unmasked arithmetic agree.
    adoption_of_row = np.asarray(panel.cohort)[panel.unit_index]
    exposure = np.where(
        treated_mask, np.asarray(panel.period_index) - adoption_of_row, 0
    ).astype(np.int64)
    mean_treated_exposure = (
        float(exposure[treated_mask].mean()) if treated_mask.any() else 0.0
    )

    return ModelInputs(
        outcome_std=outcome_std,
        unit_index=panel.unit_index,
        time_index=panel.period_index,
        cohort_index=cohort_index,
        treated_mask=treated_mask,
        n_units=panel.n_units,
        n_periods=panel.n_periods,
        n_cohorts=n_cohorts,
        cohort_cell_counts=cohort_cell_counts,
        exposure=exposure,
        mean_treated_exposure=mean_treated_exposure,
        outcome_mean=outcome_mean,
        outcome_sd=outcome_sd,
    )


def weighted_att_draws(
    tau_g_draws: np.ndarray,
    cohort_cell_counts: np.ndarray,
    outcome_sd: float,
    delta_draws: np.ndarray | None = None,
    mean_treated_exposure: float = 0.0,
) -> np.ndarray:
    """One treated-cell-weighted ATT per posterior draw, on the outcome's scale.

    `tau_g_draws` is `(n_draws, n_cohorts)`; `cohort_cell_counts` counts
    treated (unit, period) CELLS per cohort, in the same cohort order (see
    `ModelInputs.cohort_cell_counts`). `delta_draws` is the shared
    exposure-slope posterior, `(n_draws,)`.

    **The exposure term is part of the estimand, not a correction to it.**
    The model's effect on a treated row is `tau_g + delta * exposure`, so the
    average over treated cells is
    `sum_g w_g * tau_g + delta * mean(exposure over treated cells)`. Both
    terms are needed for the result to be the same quantity
    `true_att_overall` measures. Dropping the second silently reports the
    effect at zero exposure instead of the average effect, which on a
    ramping scenario is a large understatement -- that omission, in the form
    of a model with no exposure term at all, measured a -17% bias and 0/15
    coverage on P4.

    `delta_draws=None` reproduces the constant-effect arithmetic and exists
    so the weighting logic can be tested independently of the exposure term.

    Weighted by treated cells, and computed *within* each draw, for two
    separate reasons that are easy to conflate.

    The weighting fixes the estimand. `mu_tau`, the population mean, is the
    unweighted "typical cohort" effect. The target is the overall ATT, which
    averages over every treated (unit, period) cell -- so a cohort's weight is
    `units x post-periods`, and an earlier-adopting cohort counts for more
    than a later one of equal size. Weighting by treated units instead was
    measured on P4 to give 2.5028 against its registered 2.8361 (the cell
    mean measures 2.8356): roughly 150 Monte Carlo standard errors off, which
    would surface in the recovery harness as estimator bias when it is really
    a weighting error.

    Computing per draw fixes the interval. Averaging the draws first and
    weighting the summaries afterwards gives the same point estimate but a
    wrong credible interval, because it discards the joint posterior's
    cross-cohort correlation.

    The `outcome_sd` scale-back matches `ModelInputs.outcome_sd`, which is a
    population standard deviation (`ddof=0`), since that is what `_model`
    standardised the outcome by.
    """
    weights = cohort_cell_counts / cohort_cell_counts.sum()
    att = np.asarray(tau_g_draws @ weights)
    if delta_draws is not None:
        att = att + np.asarray(delta_draws) * mean_treated_exposure
    return att * outcome_sd


@dataclass(frozen=True)
class _PreTreatmentLeads:
    """Every pre-treatment placebo lead across every cohort, plus the single
    lead nearest its own cohort's adoption -- for `no_anticipation_check`.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    nearest_coefficient: float | None
    nearest_standard_error: float | None
    nearest_horizon_label: str


def pre_treatment_leads(panel: PanelArrays) -> _PreTreatmentLeads:
    """Placebo leads for `parallel_trends_check`/`no_anticipation_check`.

    The 2x2 placebo computation itself -- the same one `CallawaySantAnna`
    uses for its own placebo leads, including the per-cell independent-
    samples variance (not a single variance pooled across cells, which
    `CallawaySantAnna`'s own history measured as badly over-rejecting -- see
    `panel_checks.placebo_leads`'s docstring) -- lives in
    `panel_checks.placebo_leads`, so the two estimators cannot let this
    formula drift apart across two copies. Controls are always the
    never-treated union not-yet-treated, with no separate "control_group"
    option: `_model`'s own likelihood already leans on every not-yet-treated
    row to identify `gamma_t`, the same units this check treats as controls.

    On top of that shared computation, this picks the single lead nearest
    its own cohort's adoption (i.e. period `g - 2`, immediately before the
    omitted baseline `g - 1`) across all cohorts, generalising event-study's
    own "nearest estimated lead" convention to a multi-cohort panel where no
    single calendar period is "immediately before adoption" for every cohort
    at once -- a selection `CallawaySantAnna` has no equivalent of, which is
    why this function exists as a thin wrapper around `placebo_leads` rather
    than the estimator calling `placebo_leads` directly the way
    `CallawaySantAnna` does.

    Returns an empty `coefficients`/`covariance` pair, and `None` for the
    nearest lead, when no cohort has two pre-treatment periods -- the caller
    reports that as `NOT_TESTABLE`, not as a passing test.
    """
    keys, coefficients, covariance = placebo_leads(panel, "not_yet_treated")
    if coefficients.size == 0:
        return _PreTreatmentLeads(
            coefficients=coefficients,
            covariance=covariance,
            nearest_coefficient=None,
            nearest_standard_error=None,
            nearest_horizon_label="",
        )

    gaps = keys[:, 0] - keys[:, 1]
    nearest = int(np.argmin(gaps))
    nearest_cohort, nearest_period = int(keys[nearest, 0]), int(keys[nearest, 1])
    diagonal = np.diag(covariance)
    return _PreTreatmentLeads(
        coefficients=coefficients,
        covariance=covariance,
        nearest_coefficient=float(coefficients[nearest]),
        nearest_standard_error=float(np.sqrt(diagonal[nearest])),
        nearest_horizon_label=(
            f"period {nearest_period} placebo lead for the cohort adopting at "
            f"period {nearest_cohort} (its nearest pre-treatment period to the "
            "omitted baseline)"
        ),
    )


HIER_MIN_PRE_PERIODS = 1
HIER_MIN_POST_PERIODS = 1


@register
class HierarchicalDiD:
    """Hierarchical Bayesian difference-in-differences: one partially-pooled
    treatment effect per adoption cohort, reported as a treated-cell-weighted
    posterior ATT.

    **How many cohorts this wants, measured.** Partial pooling is the whole
    point of the model, and the across-cohort scale `sigma_tau` is what
    governs it. Refitting under a fourfold change in that parameter's prior
    scale moves its posterior by 1.4% at seven cohorts and by 81% at three
    (full prior-dominance would be 300%). So the pooling behaviour is
    genuinely estimated with seven cohorts and only about half-estimated with
    three; with a single cohort there is nothing across-cohort to identify and
    the diagnostic reports its prior. The headline ATT is far less sensitive
    than `sigma_tau` is -- P3's bias is -0.67% -- but a `sigma_tau` read as
    evidence that cohorts differ needs enough cohorts to mean it.

    **What it cannot represent.** Effects grow with exposure at ONE shared
    rate (`delta`). Cohorts whose effects grow at different rates, or grow
    non-linearly, are outside the model; the rejected
    cohort-by-exposure surface is where that would live.

    The credible interval is a posterior interval and is never a confidence
    interval. Its measured frequentist coverage is below nominal -- see the
    recovery harness, which records the number.
    """

    name = "hierarchical_did"
    requirements = Requirements(
        needs_panel=True,
        needs_covariates=False,
        min_pre_periods=HIER_MIN_PRE_PERIODS,
        min_post_periods=HIER_MIN_POST_PERIODS,
        min_treated_units=1,
        # The model's own likelihood leans on the never-treated to identify
        # each `gamma_t` -- see `_model`'s docstring -- and there is no
        # not-yet-treated fallback the way `CallawaySantAnna` has, so at
        # least one never-treated unit is a genuine requirement, not a
        # convention carried over from another estimator.
        min_control_units=1,
        # `_model`'s only outcome likelihood is `dist.Normal(mean, sigma_y)`
        # -- unconditionally Gaussian, with no Bernoulli/logit-link
        # alternative anywhere in this file. The default
        # `supports_outcome_types` of `{"binary", "continuous"}` left this
        # estimator eligible for a binary panel its own likelihood cannot
        # represent; measured on P6 ("Binary panel"), analytic coverage was
        # 0.01 against a nominal 0.95, worse than every peer DiD estimator on
        # the identical generated data. Restricted to continuous until a real
        # binary-outcome likelihood is built and validated on its own terms.
        supports_outcome_types=frozenset({"continuous"}),
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        panel = panel_arrays(dataset)
        inputs = build_model_inputs(panel)

        mcmc_config = config.options.get("mcmc_config", HIER_PRODUCTION_MCMC_CONFIG)
        if not isinstance(mcmc_config, MCMCConfig):
            raise TypeError(
                f"config.options['mcmc_config'] must be an MCMCConfig, got {type(mcmc_config)!r}"
            )
        prior_scale_raw = config.options.get("sigma_tau_prior_scale", 1.0)
        if isinstance(prior_scale_raw, bool) or not isinstance(
            prior_scale_raw, (int, float)
        ):
            raise TypeError(
                f"config.options['sigma_tau_prior_scale'] must be a positive "
                f"number, got {type(prior_scale_raw)!r}"
            )
        sigma_tau_prior_scale = float(prior_scale_raw)
        if sigma_tau_prior_scale <= 0.0:
            raise ValueError(
                f"config.options['sigma_tau_prior_scale'] must be positive, "
                f"got {sigma_tau_prior_scale}"
            )

        seed = 0 if config.random_seed is None else config.random_seed

        mcmc = run_nuts(
            _model,
            mcmc_config,
            seed,
            sigma_tau_prior_scale=sigma_tau_prior_scale,
            outcome_std=inputs.outcome_std,
            unit_index=inputs.unit_index,
            time_index=inputs.time_index,
            cohort_index=inputs.cohort_index,
            treated_mask=inputs.treated_mask,
            exposure=inputs.exposure,
            n_units=inputs.n_units,
            n_periods=inputs.n_periods,
            n_cohorts=inputs.n_cohorts,
        )

        samples = mcmc.get_samples()
        chain_samples = mcmc.get_samples(group_by_chain=True)
        diverging = mcmc.get_extra_fields(group_by_chain=True)["diverging"]

        tau_g_draws = np.asarray(samples["tau_g"])
        delta_draws = np.asarray(samples["delta"])
        att = weighted_att_draws(
            tau_g_draws,
            inputs.cohort_cell_counts,
            inputs.outcome_sd,
            delta_draws=delta_draws,
            mean_treated_exposure=inputs.mean_treated_exposure,
        )

        point = float(att.mean())
        lower_q, upper_q = (1 - config.ci_level) / 2, 1 - (1 - config.ci_level) / 2
        lo, hi = np.quantile(att, [lower_q, upper_q])
        ci = Interval(lower=float(lo), upper=float(hi), level=config.ci_level)
        # Posterior standard deviation of the weighted ATT draws, NOT a
        # sampling standard error: there is no sampling distribution here,
        # only a posterior. Reported because BSTS's own `se` field is
        # likewise the posterior sd, not because it plays the same role a
        # frequentist se does.
        se = float(att.std(ddof=1))

        p_effect_positive = float(np.mean(att > 0))

        mu_tau_mean = float(np.asarray(samples["mu_tau"]).mean()) * inputs.outcome_sd
        sigma_tau_mean = float(np.asarray(samples["sigma_tau"]).mean()) * inputs.outcome_sd
        tau_g_mean = tau_g_draws.mean(axis=0) * inputs.outcome_sd

        leads = pre_treatment_leads(panel)
        if leads.nearest_coefficient is None or leads.nearest_standard_error is None:
            no_anticipation: AssumptionCheck = AssumptionCheck(
                name="no_anticipation",
                kind="identification",
                status="NOT_TESTABLE",
                statistic=None,
                threshold=None,
                interpretation=(
                    "No cohort has two pre-treatment periods to compare, so "
                    "there is no placebo lead to test. This is not evidence "
                    "that there was no anticipation."
                ),
                remedy=(
                    "Extend the panel backwards so at least one cohort has "
                    "two pre-treatment periods observed."
                ),
            )
        else:
            no_anticipation = no_anticipation_check(
                coefficient=leads.nearest_coefficient,
                standard_error=leads.nearest_standard_error,
                horizon_label=leads.nearest_horizon_label,
            )

        treated_units = panel.cohort > 0
        return EffectEstimate(
            estimand="ATT",
            method=self.name,
            point=point,
            se=se,
            ci=ci,
            ci_basis="posterior",
            bootstrap_ci=None,
            p_value=None,
            n_treated=int(treated_units.sum()),
            n_control=panel.n_never_treated_units,
            scale=outcome_scale(dataset),
            diagnostics={
                "p_effect_positive": Diagnostic(
                    name="p_effect_positive",
                    value=p_effect_positive,
                    unit="probability",
                    interpretation=(
                        f"P(effect > 0) = {p_effect_positive:.3f} under the "
                        "posterior. A Bayesian probability statement, not a "
                        "p-value."
                    ),
                ),
                "mu_tau": Diagnostic(
                    name="mu_tau",
                    value=mu_tau_mean,
                    unit=None,
                    interpretation=(
                        "Posterior mean of the population (across-cohort) "
                        "treatment effect, rescaled to the outcome's original "
                        "units. The unweighted 'typical cohort' effect -- not "
                        "the treated-cell-weighted ATT this estimate reports "
                        "as its point."
                    ),
                ),
                "sigma_tau": Diagnostic(
                    name="sigma_tau",
                    value=sigma_tau_mean,
                    unit=None,
                    interpretation=(
                        "Posterior mean of the across-cohort standard "
                        "deviation of the treatment effect, rescaled to the "
                        "outcome's original units. Near zero means the "
                        "cohorts share essentially one effect; large means "
                        "they differ. HOW FAR THIS IS A FINDING ABOUT THE "
                        "DATA DEPENDS ON THE COHORT COUNT. Measured by "
                        "refitting under a fourfold change in this "
                        "parameter's own prior scale: at seven cohorts the "
                        "posterior moved 1.4% (data-driven), at three it "
                        "moved 81% against the 300% of full prior-dominance, "
                        "so roughly half of it is the prior. With a single "
                        "cohort there is no across-cohort variation to "
                        "identify at all and this reports the prior outright. "
                        "Read a large value at few cohorts with that in mind."
                    ),
                ),
                "delta": Diagnostic(
                    name="delta",
                    value=float(delta_draws.mean()) * inputs.outcome_sd,
                    unit=None,
                    interpretation=(
                        "Posterior mean of the shared effect growth per "
                        "period of exposure, in the outcome's original "
                        "units. Positive means the effect builds after "
                        "adoption. One slope for every cohort, so it cannot "
                        "describe cohorts whose effects grow at different "
                        "rates, nor a non-linear ramp."
                    ),
                ),
                "tau_g": Diagnostic(
                    name="tau_g",
                    value=tau_g_mean,
                    unit=None,
                    interpretation=(
                        "Posterior mean treatment effect for each adoption "
                        "cohort, in adoption-period order, rescaled to the "
                        "outcome's original units."
                    ),
                ),
            },
            assumptions=(
                r_hat_check(chain_samples),
                effective_sample_size_check(chain_samples),
                divergent_transitions_check(diverging),
                parallel_trends_check(leads.coefficients, leads.covariance),
                no_anticipation,
                absorbing_treatment_check(dataset),
            ),
            counterfactual=None,
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )
