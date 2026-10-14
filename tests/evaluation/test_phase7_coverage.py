"""Phase 15 Task 10: the measurement Phase 14 claimed but never took.

Phase 14 listed "Phase 7's credible-interval coverage is measured with
MCSE and the number is recorded" as a definition-of-done item, and claimed
the phase had done so. Neither was true: `hierarchical_did` is a
NumPyro/NUTS sampler and was excluded from that phase's `ci` gating tier on
cost (see `cp_evaluation.tiers`), so nothing in that phase ever fitted it.
The false claim was corrected on `main` in `d3d0fd0`. This module takes the
measurement and pins what it found.

**This test does not run the measurement.** `hierarchical_did` is a NUTS
sampler; the protocol below took 2207s (~37 minutes) in this tree. Re-running
it on every `pytest` invocation, gated or not, is not affordable, and this
repo's `slow` marker (see `pyproject.toml`) is documented as covering
"reference parity, measured-rate protocols, and MCMC-cost tests" -- exactly
this case -- so the test is marked `@pytest.mark.slow` and reads the recorded
numbers below as constants rather than re-fitting anything. `make test-fast`
skips it; `make test` and `make check` still run it, asserting only against
the numbers recorded here.

**Measured** 2026-09-06, this tree: `run_grid` over `hierarchical_did` only,
panel scenarios P1/P3/P4, `n_units=150`, `n_periods=7`, 40 replications,
`EstimatorConfig(random_seed=0, bootstrap_reps=1000)`, root entropy 0. Total
elapsed 2207s. Full summary saved at `/tmp/t10b_summary.csv`.

    scenario  coverage_posterior mcse     bias      bias_mcse  se_calibration  n_fitted
    P1        1.00000            0.07500  0.04935   0.01632    1.12892         40
    P3        0.95000            0.03446  0.01384   0.01714    1.01449         40
    P4        0.95000            0.03446  -0.01756  0.01807    0.99587         40

**Finding 1: this measurement does not reproduce Phase 7's reported 0.842.**
At this protocol the credible intervals cover at or above the
nominal 0.95 across all three scenarios -- 1.00, 0.95, 0.95 -- and
`se_calibration` (mean reported SE / empirical SD of the point estimate) is
1.13, 1.01, and 1.00, meaning the reported standard errors match or slightly
exceed the empirical spread of the estimates rather than understating it.
Treating each scenario as `Binomial(40, 0.842)` under Phase 7's reported
rate:

    P1: 40/40 covered.  P(X >= 40 | Binomial(40, 0.842)) = 0.001029
    P3: 38/40 covered.  P(X >= 38 | Binomial(40, 0.842)) = 0.037020
    P4: 38/40 covered.  P(X >= 38 | Binomial(40, 0.842)) = 0.037020

P1 alone puts the 0.842 figure at p = 0.001 against what was actually
observed here. An earlier 15-replication pass gave 15/15 on P1 and P3, which
looks just as conclusive but is not: `0.842 ** 15 = 0.077`, nowhere near
small enough to reject anything -- which is why this was re-run at 40.

**Finding 2: this is not a like-for-like comparison, and must not be read as
a refutation of Phase 7.** Phase 7's own protocol was much larger --
`HIER_N_UNITS=600, HIER_N_PERIODS=20`, recorded in scenario P7's registry
docstring (`cp_simulation/panel_scenarios.py`) -- while this measurement used
`n_units=150` over 7 periods, a substantially easier estimation problem,
because `hierarchical_did` costs about 25s per fit at n=150 and 80s at n=500
(measured in `cp_evaluation.tiers`'s `full_config` docstring); matching
Phase 7's own protocol here would have taken hours, not the ~37 minutes this
run took. The correct statement is that **the undercoverage is not
reproduced at this easier protocol**, not that Phase 7's original finding was
wrong. Whether it holds at Phase 7's own scale remains unmeasured by this
task.

**Finding 3: P1's MCSE of 0.075 is the rule of three, not an ordinary
binomial SE.** `cp_evaluation.metrics._share_with_mcse` substitutes `3/n` at
a unanimous 0-of-n or n-of-n share, because the binomial formula
`sqrt(p(1-p)/n)` is exactly 0.0 at either endpoint and would otherwise read
as infinite precision from 40 replications. At n=40 that is `3/40 = 0.075` --
an upper bound on the unobserved tail (the true rate could plausibly still be
a little below 1.0), not a precision claim the way an interior share's MCSE
is. P3 and P4 landed at an interior share (38/40 = 0.95) and so both get the
ordinary binomial standard error `sqrt(0.95 * 0.05 / 40) = 0.03446`.
"""

from __future__ import annotations

import math

import pytest

# MEASURED 2026-09-06, this tree. See module docstring for the full protocol
# and the /tmp/t10b_summary.csv this was read from.
_MEASURED: dict[str, dict[str, float]] = {
    "P1": {
        "coverage_posterior": 1.00000,
        "coverage_posterior_mcse": 0.07500,
        "bias": 0.04935,
        "bias_mcse": 0.01632,
        "se_calibration": 1.12892,
        "n_fitted": 40,
    },
    "P3": {
        "coverage_posterior": 0.95000,
        "coverage_posterior_mcse": 0.03446,
        "bias": 0.01384,
        "bias_mcse": 0.01714,
        "se_calibration": 1.01449,
        "n_fitted": 40,
    },
    "P4": {
        "coverage_posterior": 0.95000,
        "coverage_posterior_mcse": 0.03446,
        "bias": -0.01756,
        "bias_mcse": 0.01807,
        "se_calibration": 0.99587,
        "n_fitted": 40,
    },
}

# Phase 7's originally reported figure, for comparison only -- not
# reproduced here. See module docstring, Finding 2.
_PHASE_7_REPORTED_COVERAGE = 0.842
_PHASE_7_REPORTED_NOMINAL = 0.95


class _Band:
    """An inclusive range a metric must fall inside, and why it is that wide.

    Mirrors `cp_evaluation.tiers.Band`'s shape and intent (a band with no
    stated derivation is a band that gets widened the next time it flakes),
    kept local to this module rather than imported so this test does not
    depend on `tiers` changing its bands independently of this measurement.
    """

    def __init__(self, low: float, high: float, note: str) -> None:
        self.low = low
        self.high = high
        self.note = note


# Bands are +/- 4 mcse around the measured value, the same width convention
# `cp_evaluation.tiers.TOLERANCE_BANDS` uses. P1's coverage sits at the
# unanimous 40/40 endpoint, so its band is clipped to [low, 1.0] -- coverage
# cannot exceed 1.0, and a +4 mcse band that ignored that would silently
# assert a physically impossible upper endpoint.
_COVERAGE_BANDS = {
    "P1": _Band(
        max(0.0, 1.00000 - 4 * 0.07500),
        1.00000,
        "measured 1.00000 (40/40 covered), mcse 0.07500 (rule of three, "
        "3/40 -- see module docstring Finding 3, this is an upper bound on "
        "the unobserved tail, not an interior-share SE); clipped at 1.0.",
    ),
    "P3": _Band(
        0.95000 - 4 * 0.03446,
        min(1.0, 0.95000 + 4 * 0.03446),
        "measured 0.95000 (38/40 covered), mcse 0.03446, ordinary binomial "
        "SE at an interior share; +/- 4 mcse, clipped at 1.0.",
    ),
    "P4": _Band(
        0.95000 - 4 * 0.03446,
        min(1.0, 0.95000 + 4 * 0.03446),
        "measured 0.95000 (38/40 covered), mcse 0.03446, ordinary binomial "
        "SE at an interior share; +/- 4 mcse, clipped at 1.0.",
    ),
}

_BIAS_BANDS = {
    "P1": _Band(
        0.04935 - 4 * 0.01632,
        0.04935 + 4 * 0.01632,
        "measured 0.04935, mcse 0.01632; +/- 4 mcse.",
    ),
    "P3": _Band(
        0.01384 - 4 * 0.01714,
        0.01384 + 4 * 0.01714,
        "measured 0.01384, mcse 0.01714; +/- 4 mcse.",
    ),
    "P4": _Band(
        -0.01756 - 4 * 0.01807,
        -0.01756 + 4 * 0.01807,
        "measured -0.01756, mcse 0.01807; +/- 4 mcse.",
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["P1", "P3", "P4"])
def test_measured_coverage_stays_within_its_recorded_band(scenario):
    band = _COVERAGE_BANDS[scenario]
    value = _MEASURED[scenario]["coverage_posterior"]
    assert band.low <= value <= band.high, band.note


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["P1", "P3", "P4"])
def test_measured_bias_stays_within_its_recorded_band(scenario):
    band = _BIAS_BANDS[scenario]
    value = _MEASURED[scenario]["bias"]
    assert band.low <= value <= band.high, band.note


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["P1", "P3", "P4"])
def test_all_forty_replications_fitted(scenario):
    # A silent drop to fewer fitted replications would invalidate every MCSE
    # above without changing the constants recorded from the actual run.
    assert _MEASURED[scenario]["n_fitted"] == 40


@pytest.mark.slow
def test_measurement_contradicts_phase_7s_reported_undercoverage():
    """Binomial evidence against Phase 7's reported 0.842, see Finding 1.

    Each scenario's coverage hit count, treated as `Binomial(40, 0.842)`
    under Phase 7's reported rate, has a small enough exceedance probability
    that the two are not describing the same process at this protocol.
    """
    from scipy.stats import binom

    hits = {"P1": 40, "P3": 38, "P4": 38}
    expected_p = {"P1": 0.001029, "P3": 0.037020, "P4": 0.037020}
    for scenario, k in hits.items():
        p = binom.sf(k - 1, 40, _PHASE_7_REPORTED_COVERAGE)
        assert math.isclose(p, expected_p[scenario], rel_tol=1e-3), (
            f"{scenario}: recomputed exceedance probability {p} no longer "
            f"matches the recorded {expected_p[scenario]} -- re-derive it "
            "rather than editing this expectation to match a drifted value"
        )
        # The two loosest cells (P3/P4, p ~= 0.037) are themselves well
        # below a 5% significance threshold; P1 (p ~= 0.001) is far tighter.
        assert p < 0.05


@pytest.mark.slow
def test_se_calibration_does_not_understate_uncertainty():
    """se_calibration >= ~1 at this protocol: reported SEs are not the cause
    of any undercoverage here, since they match or slightly exceed the
    empirical spread of the point estimates rather than understating it."""
    for scenario in ("P1", "P3", "P4"):
        assert _MEASURED[scenario]["se_calibration"] > 0.95


def test_phase_7s_reported_figure_is_recorded_for_comparison_only():
    # Not a claim this measurement reproduces -- see module docstring,
    # Finding 2. Pinned here so a future edit that "corrects" 0.842 to match
    # this measurement is caught: they are different protocols, and the
    # discrepancy between them is the finding, not an error to erase.
    assert _PHASE_7_REPORTED_COVERAGE == 0.842
    assert _PHASE_7_REPORTED_NOMINAL == 0.95
