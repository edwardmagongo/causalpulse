"""MCMC diagnostics for NumPyro-based estimators.

Not panel-specific: R-hat, effective sample size, and divergent transitions
are properties of any NUTS/MCMC fit, and Phase 6's hierarchical Bayesian DiD
model will need the same three checks BSTS needs here. Kept in their own
module for that reason, in the parameterised style `panel_checks.py`
established.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from cp_core.results import AssumptionCheck
from numpyro.diagnostics import gelman_rubin


def r_hat_check(samples: Mapping[str, np.ndarray], threshold: float = 1.01) -> AssumptionCheck:
    """Worst (maximum) R-hat across every sampled site.

    `samples` values are chain-separated arrays `(chains, draws, *event_shape)`
    -- `MCMC.get_samples(group_by_chain=True)`'s own output shape. The worst
    site determines whether the fit is trustworthy, not an arbitrary one: the
    same max-over-many reasoning `donor_contamination_check`
    (`panel_checks.py`) already applies to donors, applied here to parameters.
    Threshold 1.01 is the parent spec's own stated bound (design doc §5.8).
    """
    worst_name, worst = max(
        (
            (name, float(np.max(gelman_rubin(np.asarray(value)))))
            for name, value in samples.items()
        ),
        key=lambda item: item[1],
    )
    ok = worst <= threshold
    return AssumptionCheck(
        name="r_hat",
        status="PASS" if ok else "FAIL",
        statistic=worst,
        threshold=threshold,
        interpretation=(
            f"The worst R-hat across all sampled parameters is {worst:.4f} "
            f"({worst_name}). "
            + (
                "Chains agree with each other."
                if ok
                else "At least one parameter's chains have not converged."
            )
        ),
        remedy=(
            None
            if ok
            else "Increase warmup/samples, or reparameterise the offending site; "
            "do not trust this fit's point estimate or interval."
        ),
    )
