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

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS


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
    treated_pre: np.ndarray,
    treated_post: np.ndarray,
    treated_mean: float,
    treated_sd: float,
    ci_level: float,
) -> PosteriorSummary:
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
