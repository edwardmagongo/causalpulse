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

# Placeholder until Task 7 measures it. Task 7 replaces these numbers with
# ones measured against R-hat and divergences on P3/P4/P7.
HIER_PRODUCTION_MCMC_CONFIG = MCMCConfig(num_warmup=500, num_samples=500, num_chains=4)


def _model(
    outcome_std: np.ndarray,
    unit_index: np.ndarray,
    time_index: np.ndarray,
    cohort_index: np.ndarray,
    treated_mask: np.ndarray,
    n_units: int,
    n_periods: int,
    n_cohorts: int,
) -> None:
    """One observation per (unit, period) row.

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
    sigma_tau = numpyro.sample("sigma_tau", dist.HalfNormal(1.0))
    z_g = numpyro.sample("z_g", dist.Normal(jnp.zeros(n_cohorts), 1.0))
    tau_g = numpyro.deterministic("tau_g", mu_tau + sigma_tau * z_g)

    safe_index = jnp.clip(jnp.asarray(cohort_index), 0, n_cohorts - 1)
    effect = jnp.where(jnp.asarray(treated_mask), tau_g[safe_index], 0.0)

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
    # count. `SimulatedPanel.truth.true_att_overall` -- the number every
    # recovery test scores against -- is the mean over every treated cell,
    # which implicitly weights each cohort by units x post-periods. Weighting
    # by units instead was measured on P4 to give 2.5028 against the
    # registered 2.8361 (true_att_overall measures 2.8356): roughly 150 Monte
    # Carlo standard errors off, reported as estimator bias when it would
    # actually be this weighting error. See `test_cohort_cell_counts_...`.
    cohort_cell_counts: np.ndarray
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
        outcome_mean=outcome_mean,
        outcome_sd=outcome_sd,
    )


def weighted_att_draws(
    tau_g_draws: np.ndarray, cohort_cell_counts: np.ndarray, outcome_sd: float
) -> np.ndarray:
    """One treated-cell-weighted ATT per posterior draw, on the outcome's scale.

    `tau_g_draws` is `(n_draws, n_cohorts)`; `cohort_cell_counts` counts
    treated (unit, period) CELLS per cohort, in the same cohort order (see
    `ModelInputs.cohort_cell_counts`).

    Weighted by treated cells, and computed *within* each draw, for two
    separate reasons that are easy to conflate.

    The weighting fixes the estimand. `mu_tau`, the population mean, is the
    unweighted "typical cohort" effect. The target is
    `SimulatedPanel.truth.true_att_overall`, which averages over every treated
    (unit, period) cell -- so a cohort's weight is `units x post-periods`, and
    an earlier-adopting cohort counts for more than a later one of equal size.
    Weighting by treated units instead was measured on P4 to give 2.5028
    against the registered 2.8361 (`true_att_overall` itself measures
    2.8356): roughly 150 Monte Carlo standard errors off, which would surface
    in the recovery harness as estimator bias when it is really a weighting
    error.

    Computing per draw fixes the interval. Averaging the draws first and
    weighting the summaries afterwards gives the same point estimate but a
    wrong credible interval, because it discards the joint posterior's
    cross-cohort correlation.

    The `outcome_sd` scale-back matches `ModelInputs.outcome_sd`, which is a
    population standard deviation (`ddof=0`), since that is what `_model`
    standardised the outcome by.
    """
    weights = cohort_cell_counts / cohort_cell_counts.sum()
    return np.asarray(tau_g_draws @ weights) * outcome_sd


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
        seed = 0 if config.random_seed is None else config.random_seed

        mcmc = run_nuts(
            _model,
            mcmc_config,
            seed,
            outcome_std=inputs.outcome_std,
            unit_index=inputs.unit_index,
            time_index=inputs.time_index,
            cohort_index=inputs.cohort_index,
            treated_mask=inputs.treated_mask,
            n_units=inputs.n_units,
            n_periods=inputs.n_periods,
            n_cohorts=inputs.n_cohorts,
        )

        samples = mcmc.get_samples()
        chain_samples = mcmc.get_samples(group_by_chain=True)
        diverging = mcmc.get_extra_fields(group_by_chain=True)["diverging"]

        tau_g_draws = np.asarray(samples["tau_g"])
        att = weighted_att_draws(tau_g_draws, inputs.cohort_cell_counts, inputs.outcome_sd)

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
                        "they genuinely differ."
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
