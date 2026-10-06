"""`run_nuts` leaves no compiled sampler behind.

Each call compiles a new kernel, and JAX keeps what it compiles. A process
that fits model after model (the API's worker, a benchmark worker, a test
session) grew until it was killed: see the comment in `cp_causal._mcmc`.
Measuring memory here would be slow and would depend on the machine, so these
tests pin the mechanism instead: the caches are emptied once per run, after
sampling, and the samples outlive it.
"""

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
from cp_causal import _mcmc
from cp_causal._mcmc import MCMCConfig, run_nuts

_CONFIG = MCMCConfig(num_warmup=20, num_samples=20, num_chains=2)


def _model(y: np.ndarray) -> None:
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)


def test_each_run_empties_the_caches_once_after_sampling(monkeypatch):
    events: list[str] = []
    real_run = _mcmc.MCMC.run

    def recording_run(self, *args, **kwargs):
        real_run(self, *args, **kwargs)
        events.append("sampled")

    monkeypatch.setattr(_mcmc.MCMC, "run", recording_run)
    monkeypatch.setattr(_mcmc.jax, "clear_caches", lambda: events.append("cleared"))

    run_nuts(_model, _CONFIG, 0, y=np.array([0.5, 1.5, 1.0]))
    run_nuts(_model, _CONFIG, 1, y=np.array([0.5, 1.5, 1.0]))

    assert events == ["sampled", "cleared", "sampled", "cleared"]


def test_the_samples_survive_the_clearing_and_a_seed_still_fixes_them():
    y = np.array([0.5, 1.5, 1.0])

    first = np.asarray(run_nuts(_model, _CONFIG, 7, y=y).get_samples()["mu"])
    jax.clear_caches()
    second = np.asarray(run_nuts(_model, _CONFIG, 7, y=y).get_samples()["mu"])

    assert first.shape == (40,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
