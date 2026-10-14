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
from numpyro.diagnostics import effective_sample_size, gelman_rubin


def r_hat_check(samples: Mapping[str, np.ndarray], threshold: float = 1.01) -> AssumptionCheck:
    """Worst (maximum) R-hat across every sampled site.

    `samples` values are chain-separated arrays `(chains, draws, *event_shape)`
    -- `MCMC.get_samples(group_by_chain=True)`'s own output shape. The worst
    site determines whether the fit is trustworthy, not an arbitrary one: the
    same max-over-many reasoning `donor_contamination_check`
    (`panel_checks.py`) already applies to donors, applied here to parameters.
    Threshold is 1.01.
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
        kind="estimation",
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


def effective_sample_size_check(
    samples: Mapping[str, np.ndarray], threshold: float = 200.0
) -> AssumptionCheck:
    """Worst (minimum) effective sample size across every sampled site.

    Threshold is a conservative floor, not a precisely calibrated one:
    prototype fits at this repository's production MCMC configuration
    measured minimum ESS in the 800+ range on clean fits and
    as low as 2.5 on a fit whose R-hat and divergence rate also independently
    flagged it as unreliable (S2 seed 9: worst R-hat 2.35, ESS 2.47,
    divergence rate 26.85%). 200 sits with wide margin below the former and
    above the latter; the recovery harness (`test_bsts_recovery.py`) is where
    this gets tightened against many seeds, the same two-stage process
    `donor_contamination_check`'s threshold went through in Phase 4.
    """
    worst_name, worst = min(
        (
            (name, float(np.min(effective_sample_size(np.asarray(value)))))
            for name, value in samples.items()
        ),
        key=lambda item: item[1],
    )
    ok = worst >= threshold
    return AssumptionCheck(
        name="effective_sample_size",
        kind="estimation",
        status="PASS" if ok else "FAIL",
        statistic=worst,
        threshold=threshold,
        interpretation=(
            f"The smallest effective sample size across all sampled "
            f"parameters is {worst:.1f} ({worst_name}). "
            + (
                "Enough independent draws to trust the posterior summary."
                if ok
                else "Too few independent draws behind at least one parameter's summary."
            )
        ),
        remedy=None if ok else "Increase num_samples, or investigate the poorly-mixing site.",
    )


def divergent_transitions_check(
    diverging: np.ndarray, threshold: float = 0.10
) -> AssumptionCheck:
    """Fraction of post-warmup draws NUTS flagged as divergent.

    A divergence is a direct symptom of the funnel geometry the regularised
    horseshoe exists to avoid, so this check is also
    an indirect check on that prior choice. Threshold 0.10 is measured
    on a prototype to separate clean donor-pool fits
    (`S1` mean divergence rate 4.22% over 15 seeds, `S2` mean 7.25%) from a
    fit independently flagged bad by R-hat and ESS (`S2` seed 9: 26.85%) --
    with real headroom, not a tight cut. Final calibration is the recovery
    harness's job, the same two-stage process `donor_contamination_check`
    went through in Phase 4.
    """
    diverging = np.asarray(diverging)
    rate = float(np.mean(diverging))
    ok = rate <= threshold
    return AssumptionCheck(
        name="divergent_transitions",
        kind="estimation",
        status="PASS" if ok else "FAIL",
        statistic=rate,
        threshold=threshold,
        interpretation=(
            f"{int(diverging.sum())} of {diverging.size} post-warmup draws "
            f"({rate:.1%}) were divergent transitions. "
            + (
                "Within tolerance."
                if ok
                else "The sampler could not traverse the posterior reliably here."
            )
        ),
        remedy=(
            None
            if ok
            else "Do not trust this fit's point estimate or interval; increase "
            "target_accept_prob, or treat this fit as unreliable."
        ),
    )
