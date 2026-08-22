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

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from cp_causal._mcmc import MCMCConfig

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
