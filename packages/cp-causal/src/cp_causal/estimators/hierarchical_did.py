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

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from cp_causal._mcmc import MCMCConfig
from cp_causal._panel import PanelArrays

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
