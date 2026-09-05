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
    # Checked against the MCSE the tier's own replication count supports
    # rather than a bare 0.05.
    for method, scenario in summary.index:
        if scenario != "B":
            continue
        row = summary.loc[(method, scenario)]
        if row["false_positive_rate_n_used"] < 5:
            continue
        assert row["false_positive_rate"] <= 0.05 + 4 * row["false_positive_rate_mcse"]


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
