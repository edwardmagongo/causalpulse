"""NUTS sampler construction shared by this package's NumPyro estimators.

Holds only what is genuinely common: the settings dataclass and the sampler
wiring, including two settings whose reasons are easy to lose. Config
*instances* stay with the estimator that measured them -- a warmup count
tuned for a horseshoe time-series model says nothing about a hierarchical
panel model, and presenting one estimator's numbers from a shared module
would imply otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
from jax import random
from numpyro.infer import MCMC, NUTS


@dataclass(frozen=True)
class MCMCConfig:
    """NUTS sampler settings.

    `num_chains` must be at least 2 for any config whose samples reach
    `r_hat_check`: `numpyro.diagnostics.gelman_rubin` hard-asserts
    `x.shape[0] >= 2`, so a 1-chain config crashes the moment an estimator
    passes its samples to that check.
    """

    num_warmup: int
    num_samples: int
    num_chains: int
    target_accept_prob: float = 0.95


def run_nuts(
    model: Callable[..., None],
    config: MCMCConfig,
    random_seed: int,
    **model_kwargs: object,
) -> MCMC:
    """Build and run a NUTS sampler over `model`, returning the finished MCMC.

    `model_kwargs` are forwarded to `mcmc.run` unchanged, which is why this
    is generic where BSTS's original `_run_mcmc` was not -- that one named
    its own model's arguments in its signature.
    """
    kernel = NUTS(model, target_accept_prob=config.target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        # Explicit, not the "parallel" default: this process has one CPU
        # device (no `numpyro.set_host_device_count` call anywhere in the
        # codebase), so "parallel" silently downgrades to sequential anyway
        # -- except it does so with a UserWarning, and this project's pytest
        # config (`filterwarnings = ["error"]`) turns that warning into a
        # test failure.
        chain_method="sequential",
        progress_bar=False,
    )
    try:
        mcmc.run(random.PRNGKey(random_seed), **model_kwargs)
    finally:
        # Every call builds a new kernel, so JAX traces and compiles the
        # sampler again and keeps the result in caches that nothing empties. A
        # process that fits many models grows without bound. Measured on
        # 2026-10-06 in a Linux container, eight HierarchicalDiD fits in one
        # process: resident memory rose by about 50 to 130 MB a fit without
        # this call (785 MB after the first, 1,249 MB after the eighth) and by
        # about 10 MB a fit with it (790 MB, then 859 MB). On a 16 GB CI runner
        # the recovery tests' four worker processes used all of it in 80
        # minutes.
        #
        # It is not free. JAX reuses part of what it compiled even for a new
        # kernel, and this throws that away: a small model fitted eight times
        # took about 4 s a fit with the call and 1.5 to 2.3 s without it after
        # the first. A production fit runs for minutes, so the cost is small
        # beside it; a caller that fits thousands of small models would feel
        # it. In `finally`, because a fit that raises (the worker's job
        # timeout arrives as an exception inside it) has compiled as much.
        # The samples are arrays, not compiled code, and are untouched.
        jax.clear_caches()  # type: ignore[no-untyped-call]
    return mcmc
