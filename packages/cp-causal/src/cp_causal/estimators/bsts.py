"""Bayesian structural time series: a probabilistic counterfactual for one
treated unit.

Local linear trend (level and slope, both evolving as random walks) plus a
regularised-horseshoe regression on donor series, sampled via NUTS. Shares
its donor-pool data shape with `SyntheticControl` (`_panel.donor_matrix`) but
not its math or inference paradigm -- see the design doc
(`docs/specs/2026-08-16-causal-core-phase5-design.md`) for why they are
separate estimators.

Non-centered throughout. A first prototype pass with the textbook *centered*
horseshoe (`donor_coef ~ Normal(0, tau*lam_tilde)` sampled directly) produced
37 divergences and R-hat up to 1.09 on S1 seed 0 -- the classic horseshoe
funnel. Reparameterising `donor_coef` as `donor_coef_raw * tau * lam_tilde`
with `donor_coef_raw ~ Normal(0, 1)` dropped that to 10-13 divergences with
every R-hat <= 1.01 at the same protocol (design doc §2 item 7 prototype).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from cp_core.dataset import CausalDataset
from cp_core.results import CounterfactualPath, EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, EstimatorConfig, Interval
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS

from cp_causal._panel import donor_matrix, outcome_scale, panel_arrays
from cp_causal._support import config_hash
from cp_causal.base import Requirements
from cp_causal.bayesian_checks import (
    divergent_transitions_check,
    effective_sample_size_check,
    r_hat_check,
)
from cp_causal.panel_checks import sufficient_pre_periods_check
from cp_causal.registry import register

MIN_DONORS = 5  # matches SyntheticControl's floor; same donor-pool shape


@dataclass(frozen=True)
class MCMCConfig:
    """NUTS sampler settings. Two tiers ship below; everything about BSTS's
    estimation besides sampler effort is independent of which tier is used.
    """

    num_warmup: int
    num_samples: int
    num_chains: int
    target_accept_prob: float = 0.95


# Exercises the code path and array shapes only -- not a trustworthy
# posterior. Used by every fast/contract test in test_bsts.py. Two chains,
# not one: numpyro.diagnostics.gelman_rubin (r_hat_check's dependency) hard-
# asserts x.shape[0] >= 2, so a 1-chain config crashes the moment fit() calls
# r_hat_check -- found at Task 8, the first task to actually wire fit()
# through to the assumption checks (Tasks 3 and 6 each tested their own piece
# in isolation and never exercised this combination). Measured directly:
# 2 chains at this warmup/sample count cost no more than 1 (~2.5-2.9s either
# way, JIT compile dominates), so this is free, not a real speed tradeoff.
FAST_MCMC_CONFIG = MCMCConfig(num_warmup=50, num_samples=50, num_chains=2)

# Measured (design doc §2 item 7 prototype): clean R-hat (<=1.01) on S1/S2/S5
# at this configuration; ~1.5-2s per fit after compile, keeping a 400-seed
# recovery run under 15 minutes. Only the recovery harness (`slow`) uses it.
PRODUCTION_MCMC_CONFIG = MCMCConfig(num_warmup=500, num_samples=500, num_chains=4)


def _model(
    donor_pre_std: jnp.ndarray, treated_pre_std: jnp.ndarray | None, T_pre: int
) -> None:
    n_donors = donor_pre_std.shape[0]

    # Generic weakly-informative priors in *standardised* units -- the
    # estimator has no oracle access to the DGP's true noise/trend scale.
    sigma_level = numpyro.sample("sigma_level", dist.HalfNormal(0.5))
    sigma_slope = numpyro.sample("sigma_slope", dist.HalfNormal(0.2))
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(1.0))

    # Regularised horseshoe (Piironen & Vehtari), non-centered.
    tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
    c2 = numpyro.sample("c2", dist.InverseGamma(2.0, 1.0))
    lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n_donors)))
    lam_tilde = jnp.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
    donor_coef_raw = numpyro.sample(
        "donor_coef_raw", dist.Normal(jnp.zeros(n_donors), jnp.ones(n_donors))
    )
    donor_coef = numpyro.deterministic("donor_coef", donor_coef_raw * tau * lam_tilde)

    level0 = numpyro.sample("level0", dist.Normal(0.0, 1.0))
    slope0 = numpyro.sample("slope0", dist.Normal(0.0, 0.3))

    def transition(
        carry: tuple[jnp.ndarray, jnp.ndarray], t: jnp.ndarray
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        level, slope = carry
        eta = numpyro.sample("eta", dist.Normal(0.0, 1.0))
        zeta = numpyro.sample("zeta", dist.Normal(0.0, 1.0))
        new_level = level + slope + sigma_level * eta
        new_slope = slope + sigma_slope * zeta
        mu = new_level + jnp.dot(donor_coef, donor_pre_std[:, t])
        obs = None if treated_pre_std is None else treated_pre_std[t]
        numpyro.sample("obs", dist.Normal(mu, sigma_obs), obs=obs)
        return (new_level, new_slope), None

    scan(transition, (level0, slope0), jnp.arange(T_pre))


def _run_mcmc(
    donor_pre_std: np.ndarray,
    treated_pre_std: np.ndarray,
    T_pre: int,
    config: MCMCConfig,
    random_seed: int,
) -> MCMC:
    kernel = NUTS(_model, target_accept_prob=config.target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        # Explicit, not the "parallel" default: this process has one CPU
        # device (no `numpyro.set_host_device_count` call anywhere in the
        # codebase), so "parallel" would silently downgrade to sequential
        # anyway -- except it does so with a UserWarning, and this project's
        # pytest config (`filterwarnings = ["error"]`) turns that warning
        # into a test failure. Found immediately after FAST_MCMC_CONFIG grew
        # a second chain. Sequential is also what the FAST_MCMC_CONFIG timing
        # claim above assumes.
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        random.PRNGKey(random_seed),
        donor_pre_std=jnp.asarray(donor_pre_std),
        treated_pre_std=jnp.asarray(treated_pre_std),
        T_pre=T_pre,
    )
    return mcmc


@dataclass(frozen=True, eq=False)
class PosteriorSummary:
    """Posterior-derived quantities needed to build `CounterfactualPath` and
    `EffectEstimate`, already converted back into the outcome's original
    units.

    Separated from `_run_mcmc` so this arithmetic -- reconstructing the
    pre-period trend recursion, forecasting the post-period by mean
    continuation, and converting out of standardised units -- is testable
    against a synthetic posterior without paying for a real MCMC run.
    """

    mu_path: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    att_draws: np.ndarray
    donor_coef_mean: np.ndarray


def summarize_posterior(
    samples: Mapping[str, np.ndarray],
    donor_pre_std: np.ndarray,
    donor_post_std: np.ndarray,
    treated_post: np.ndarray,
    treated_mean: float,
    treated_sd: float,
    ci_level: float,
) -> PosteriorSummary:
    """Reconstruct the pre-period trend recursion from posterior draws and
    forecast the post-period by deterministic mean continuation
    (`level + h*slope`, no future process noise injected per horizon `h`).

    Because of that mean-continuation forecast, the post-period credible
    interval carries only parameter/state uncertainty as of the end of the
    pre-period -- it does NOT accumulate the random-walk innovation variance
    that a true posterior predictive interval would add at each additional
    horizon step. In other words, this returns the posterior of the
    conditional mean forecast, not a horizon-widening posterior predictive
    interval. This does not cause measurable under-coverage at this
    protocol's short 7-period post-window (measured 15/15 coverage on S1/S2),
    but the gap would grow with a longer post-period.
    """
    eta = np.asarray(samples["eta"])
    zeta = np.asarray(samples["zeta"])
    level = np.asarray(samples["level0"]).copy()
    slope = np.asarray(samples["slope0"]).copy()
    sigma_level = np.asarray(samples["sigma_level"])
    sigma_slope = np.asarray(samples["sigma_slope"])
    donor_coef = np.asarray(samples["donor_coef"])

    t_pre = eta.shape[1]
    n_post = donor_post_std.shape[1]
    n_draws = level.shape[0]

    mu_pre_std = np.empty((n_draws, t_pre))
    for t in range(t_pre):
        level = level + slope + sigma_level * eta[:, t]
        slope = slope + sigma_slope * zeta[:, t]
        mu_pre_std[:, t] = level + donor_coef @ donor_pre_std[:, t]

    # Deterministic mean-continuation forecast (no future process noise):
    # level_forecast[h] = level + h * slope, the local-linear-trend
    # point-forecast convention.
    horizons = np.arange(1, n_post + 1)
    level_forecast = level[:, None] + horizons[None, :] * slope[:, None]
    mu_post_std = level_forecast + donor_coef @ donor_post_std

    mu_pre = mu_pre_std * treated_sd + treated_mean
    mu_post = mu_post_std * treated_sd + treated_mean
    mu_path = np.concatenate([mu_pre.mean(axis=0), mu_post.mean(axis=0)])

    lower_q, upper_q = (1 - ci_level) / 2, 1 - (1 - ci_level) / 2
    all_draws = np.concatenate([mu_pre, mu_post], axis=1)
    lower = np.quantile(all_draws, lower_q, axis=0)
    upper = np.quantile(all_draws, upper_q, axis=0)

    att_draws = (treated_post[None, :] - mu_post).mean(axis=1)

    return PosteriorSummary(
        mu_path=mu_path,
        lower=lower,
        upper=upper,
        att_draws=att_draws,
        donor_coef_mean=donor_coef.mean(axis=0),
    )


@register
class BSTS:
    """Bayesian structural time series: a donor-weighted, probabilistic
    counterfactual for one treated unit."""

    name = "bsts"
    requirements = Requirements(
        needs_panel=True,
        min_treated_units=1,
        max_treated_units=1,
        min_control_units=MIN_DONORS,
        # Matches sufficient_pre_periods_check's own minimum=10 below,
        # following every Phase 3 estimator's convention (DID 1/1, CS 1/1,
        # EventStudy 2/2: Requirements' floor always equals the check's own
        # declared minimum). Also matches SyntheticControl's identical
        # min_pre_periods=10 -- same donor-pool data shape, same rationale:
        # routing data as short as S5's 2 pre-periods onto a donor-pool
        # estimator is the underdetermined regime this floor exists to keep
        # both estimators out of.
        min_pre_periods=10,
        min_post_periods=1,
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        panel = panel_arrays(dataset)
        treated_series, donor_series, pre_mask = donor_matrix(panel)
        post_mask = ~pre_mask

        treated_pre, treated_post = treated_series[pre_mask], treated_series[post_mask]
        donor_pre, donor_post = donor_series[:, pre_mask], donor_series[:, post_mask]

        donor_mean = donor_pre.mean(axis=1, keepdims=True)
        donor_sd = donor_pre.std(axis=1, keepdims=True) + 1e-9
        donor_pre_std = (donor_pre - donor_mean) / donor_sd
        donor_post_std = (donor_post - donor_mean) / donor_sd

        treated_mean = float(treated_pre.mean())
        treated_sd = float(treated_pre.std()) + 1e-9
        treated_pre_std = (treated_pre - treated_mean) / treated_sd

        mcmc_config = config.options.get("mcmc_config", PRODUCTION_MCMC_CONFIG)
        if not isinstance(mcmc_config, MCMCConfig):
            raise TypeError(
                f"config.options['mcmc_config'] must be an MCMCConfig, got {type(mcmc_config)!r}"
            )
        seed = 0 if config.random_seed is None else config.random_seed

        mcmc = _run_mcmc(donor_pre_std, treated_pre_std, treated_pre.shape[0], mcmc_config, seed)

        samples = mcmc.get_samples()
        chain_samples = mcmc.get_samples(group_by_chain=True)
        diverging = mcmc.get_extra_fields(group_by_chain=True)["diverging"]

        summary = summarize_posterior(
            samples,
            donor_pre_std,
            donor_post_std,
            treated_post,
            treated_mean,
            treated_sd,
            config.ci_level,
        )

        point = float(summary.att_draws.mean())
        lower_q, upper_q = (1 - config.ci_level) / 2, 1 - (1 - config.ci_level) / 2
        lo, hi = np.quantile(summary.att_draws, [lower_q, upper_q])
        ci = Interval(lower=float(lo), upper=float(hi), level=config.ci_level)

        practical_threshold_raw = config.options.get("practical_threshold", 0.0)
        if not isinstance(practical_threshold_raw, int | float):
            raise TypeError(
                f"options['practical_threshold'] must be a number, "
                f"got {type(practical_threshold_raw)!r}"
            )
        practical_threshold = float(practical_threshold_raw)
        p_effect_positive = float(np.mean(summary.att_draws > 0))
        p_effect_above_threshold = float(np.mean(summary.att_draws > practical_threshold))

        pointwise_effect = treated_series - summary.mu_path
        path = CounterfactualPath(
            periods=panel.periods.astype(float),
            observed=treated_series,
            predicted_absent_treatment=summary.mu_path,
            lower=summary.lower,
            upper=summary.upper,
            pointwise_effect=pointwise_effect,
            cumulative_effect=np.cumsum(np.where(post_mask, pointwise_effect, 0.0)),
        )

        return EffectEstimate(
            estimand="ATT",
            method=self.name,
            point=point,
            se=None,
            ci=ci,
            ci_basis="posterior",
            bootstrap_ci=None,
            p_value=None,
            n_treated=1,
            n_control=int(donor_series.shape[0]),
            scale=outcome_scale(dataset),
            diagnostics={
                "p_effect_positive": Diagnostic(
                    name="p_effect_positive",
                    value=p_effect_positive,
                    unit="probability",
                    interpretation=(
                        f"P(effect > 0) = {p_effect_positive:.3f} under the posterior. "
                        "A Bayesian probability statement, not a p-value."
                    ),
                ),
                "p_effect_above_threshold": Diagnostic(
                    name="p_effect_above_threshold",
                    value=p_effect_above_threshold,
                    unit="probability",
                    interpretation=(
                        f"P(effect > {practical_threshold:g}) = "
                        f"{p_effect_above_threshold:.3f} under the posterior."
                    ),
                ),
                "donor_coef": Diagnostic(
                    name="donor_coef",
                    value=summary.donor_coef_mean,
                    unit=None,
                    interpretation=(
                        "Posterior mean regression coefficient on each "
                        "standardised donor series, in donor order, under the "
                        "regularised horseshoe prior."
                    ),
                ),
            },
            assumptions=(
                r_hat_check(chain_samples),
                effective_sample_size_check(chain_samples),
                divergent_transitions_check(diverging),
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
