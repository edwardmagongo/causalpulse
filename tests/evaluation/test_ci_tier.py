"""The benchmark tier that gates the suite.

Bands are derived from a measured run and stated in MCSE units, so a band's
width is justified rather than chosen. Task 9's seeded-defect test is the
structural guarantee that these bands can fail: a band that cannot reject a
known-bad estimator is not a band.
"""

import pytest
from cp_evaluation.metrics import summarise
from cp_evaluation.run import run_grid
from cp_evaluation.tiers import CI_ESTIMATORS, CI_FIT_CONFIG, TOLERANCE_BANDS, ci_config
from scipy.stats import binom

# Absolute ceiling on the null-scenario false-positive rate, derived from the
# binomial distribution rather than picked by hand. `false_positive_rate <=
# 0.05 + 4 * mcse` alone is too loose to mean anything at this tier's 15
# replications: an observed rate of 0/15 has the rule-of-three MCSE floor of
# 3/15 (~0.2, see `_share_with_mcse` in metrics.py), so that bound alone
# evaluates to 0.05 + 4*0.2 = 0.85 -- it would only fail if 13 or more of the
# 15 null replications produced a false rejection.
#
# Instead: under a true false-positive rate of 5% (X ~ Binomial(n=15, p=0.05)),
# find the smallest count of false positives whose exceedance probability
# P(X >= k) is small enough (~1% or below) that a real 5% process will not
# flake this test. Computed with scipy.stats.binom:
#   P(X >= 3) = 0.0362147   (3.62%)  -- too flake-prone, k=3 is not enough
#   P(X >= 4) = 0.00546726  (0.55%) -- comfortably below 1%
# So k=4: the test should fail once 4 or more of 15 null replications
# reject, and pass for 3 or fewer -- i.e. the ceiling on the *rate* is
# (k - 1) / 15 = 3 / 15 = 0.2. A true 5% process clears this bar (3 or fewer
# false positives out of 15) more than 99.4% of the time, so unlike the 0.85
# MCSE-aware bound above, this ceiling actually tests something.
_NULL_FALSE_POSITIVE_EXCEEDANCE_K = 4
NULL_FALSE_POSITIVE_EXCEEDANCE_PROBABILITY = binom.sf(
    _NULL_FALSE_POSITIVE_EXCEEDANCE_K - 1, 15, 0.05
)
assert NULL_FALSE_POSITIVE_EXCEEDANCE_PROBABILITY <= 0.01, (
    "the chosen k no longer clears the ~1% flake target -- re-derive it"
)
NULL_FALSE_POSITIVE_RATE_CEILING = (_NULL_FALSE_POSITIVE_EXCEEDANCE_K - 1) / 15


@pytest.fixture(scope="module")
def summary():
    # ci_config() alone is only the axes; the estimator subset and the
    # bootstrap_reps=200 fit config TOLERANCE_BANDS was measured against are
    # separate module constants, exactly so a caller can't run the axes
    # against the wrong estimator list or bootstrap count by omitting them.
    frame = run_grid(ci_config(), estimators=CI_ESTIMATORS, fit_config=CI_FIT_CONFIG)
    return summarise(frame).set_index(["method", "scenario"])


def test_the_ci_tier_has_bands_to_check():
    # Deliberately does not request the `summary` fixture: this assertion is
    # about the committed constant, and requesting the fixture would run the
    # whole grid to check a dict is non-empty.
    assert TOLERANCE_BANDS, "the ci tier asserts nothing without committed bands"


@pytest.mark.parametrize("key", sorted(TOLERANCE_BANDS))
def test_every_committed_band_holds(key, summary):
    for band in TOLERANCE_BANDS[key]:
        value = summary.loc[key, band.metric]
        assert band.low <= value <= band.high, (
            f"{key} {band.metric} = {value}, outside [{band.low}, {band.high}]. {band.note}"
        )


def test_the_null_scenario_false_positive_rate_is_near_nominal(summary):
    # Design doc's falsifiability test of the whole system: under a scenario
    # with no effect, a 95% interval must exclude zero about 5% of the time.
    #
    # Two bounds are asserted, and the assertion is only as tight as the
    # tighter of the two:
    #   1. The MCSE-aware bound (0.05 + 4*mcse) stays meaningful if a future
    #      grid ever has a non-degenerate false_positive_rate_mcse.
    #   2. NULL_FALSE_POSITIVE_RATE_CEILING is a binomial-derived absolute
    #      ceiling (see module comment) that does not go slack just because
    #      the observed rate is 0/15 and mcse hits the rule-of-three floor --
    #      at today's grid that floor alone would let the MCSE-aware bound
    #      rise to 0.85, which only fails on total breakdown (13+ of 15).
    checked_any = False
    for method, scenario in summary.index:
        if scenario != "B":
            continue
        row = summary.loc[(method, scenario)]
        if row["false_positive_rate_n_used"] < 5:
            continue
        checked_any = True
        assert row["false_positive_rate"] <= 0.05 + 4 * row["false_positive_rate_mcse"]
        assert row["false_positive_rate"] <= NULL_FALSE_POSITIVE_RATE_CEILING, (
            f"{method}/{scenario} false_positive_rate="
            f"{row['false_positive_rate']} exceeds the binomial-derived "
            f"ceiling {NULL_FALSE_POSITIVE_RATE_CEILING} (see module comment "
            f"-- P(X>={_NULL_FALSE_POSITIVE_EXCEEDANCE_K}) = "
            f"{NULL_FALSE_POSITIVE_EXCEEDANCE_PROBABILITY:.6f} under a true "
            "5% rate at 15 replications)"
        )
    # The n_used guard above must not silently pass by skipping every group:
    # confirm at least one null-scenario method actually got checked.
    assert checked_any, "no null-scenario group had n_used >= 5 -- the guard skipped everything"


def test_every_group_reports_how_many_fits_it_actually_used(summary):
    # Guards against a band passing over an empty group.
    assert (summary["n_fitted"] + summary["n_ineligible"] + summary["n_refused"] > 0).all()


def test_ci_estimators_are_all_registered():
    # A typo'd name here would silently narrow the grid to fewer methods
    # than TOLERANCE_BANDS expects, and run_grid would raise loudly for an
    # unknown estimator -- but only when the tier actually runs. Cheaper to
    # catch it as a direct check on the constant.
    from cp_causal.registry import all_estimators

    unknown = set(CI_ESTIMATORS) - set(all_estimators())
    assert not unknown, f"CI_ESTIMATORS names unregistered estimators: {unknown}"


def test_every_band_key_is_covered_by_ci_estimators_and_scenarios():
    # TOLERANCE_BANDS is a hand-maintained constant; this pins it to the grid
    # it was measured against so a future edit to ci_config() or
    # CI_ESTIMATORS can't silently leave a band pointed at a (method,
    # scenario) pair the tier no longer produces.
    scenarios = {"A", "B", "D", "P1", "P2"}
    for method, scenario in TOLERANCE_BANDS:
        assert method in CI_ESTIMATORS, f"{method} is not in CI_ESTIMATORS"
        assert scenario in scenarios, f"{scenario} is not one of {sorted(scenarios)}"
