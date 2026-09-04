"""LaLonde/NSW/CPS-1: the classic hard test of observational adjustment.

Scores every eligible cross-sectional estimator's NSW-treated-vs-CPS-1
observational estimate against the NSW-treated-vs-NSW-control experimental
benchmark. See tests/fixtures/real_data/lalonde.md for the data's source and
the causal question this answers.
"""

from pathlib import Path

import pandas as pd
import pytest
from cp_causal.registry import get_estimator
from cp_core.dataset import CausalDataset, CausalSchema
from cp_core.values import EstimatorConfig

FIXTURE = Path(__file__).parent.parent / "fixtures" / "real_data" / "lalonde.csv"
COVARIATES = ("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75")


def _schema() -> CausalSchema:
    return CausalSchema(
        unit="unit_id",
        treatment="treat",
        outcome="re78",
        covariates=COVARIATES,
        outcome_type="continuous",
    )


def _frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


def _experimental_benchmark(frame: pd.DataFrame) -> float:
    nsw = frame[frame.data_id == "Dehejia-Wahba Sample"]
    return float(nsw.loc[nsw.treat == 1, "re78"].mean() - nsw.loc[nsw.treat == 0, "re78"].mean())


def _observational_dataset(frame: pd.DataFrame) -> CausalDataset:
    treated = frame[(frame.data_id == "Dehejia-Wahba Sample") & (frame.treat == 1)]
    cps1 = frame[frame.data_id == "CPS1"]
    observational = pd.concat([treated, cps1], ignore_index=True)
    return CausalDataset(frame=observational, schema=_schema())


# Measured 2026-09-04, after the propensity scaling fix (commit f5f5c92).
# The experimental benchmark is a plain mean difference over 445 fixed rows, so
# it is exact; the four estimates are deterministic given `random_seed=0`.
#
#   experimental benchmark (NSW treated vs NSW control)   +1794.34
#   difference_in_means   -8497.52   gap -10291.86
#   ipw                   -6652.62   gap  -8446.96
#   lin_regression        -3680.60   gap  -5474.94
#   aipw                  -3487.35   gap  -5281.69
#
# The pre-registered expectation (design doc §4.4) was that the naive estimator
# would show a large, badly-signed gap and that adjustment would close some of
# it. Half of that held and half did not, and the half that did not is the
# finding worth recording rather than tuning away: adjustment closes between
# 18% (ipw) and 49% (aipw) of the naive gap, but **no estimator recovers the
# benchmark, and every one of them is still wrong in sign**. Not one of these
# four methods would tell an analyst that the NSW program raised earnings.
#
# That is the expected result rather than a defect in this codebase. It is
# LaLonde (1986)'s original point. Dehejia & Wahba (1999) recovered the
# experimental estimate using propensity-score *matching* on a restricted
# subsample, not inverse-probability weighting over all of CPS-1, and even that
# is contested (Smith & Todd 2005). These estimators are being asked to close a
# $12,000 baseline earnings gap between a long-term-unemployed training cohort
# and a general population sample; the overlap needed is not there.
EXPERIMENTAL_BENCHMARK = 1794.34
NAIVE_GAP = -10291.86
MEASURED_GAPS = {
    "difference_in_means": -10291.86,
    "lin_regression": -5474.94,
    "ipw": -8446.96,
    "aipw": -5281.69,
}
ADJUSTING = ("lin_regression", "ipw", "aipw")


def test_the_experimental_benchmark_is_what_the_fixture_documents():
    assert _experimental_benchmark(_frame()) == pytest.approx(EXPERIMENTAL_BENCHMARK, abs=0.01)


@pytest.mark.parametrize("estimator_name", list(MEASURED_GAPS))
def test_the_gap_against_the_experimental_benchmark_is_where_it_was_measured(estimator_name):
    # A 15% band around each measured gap. Wide enough to survive a solver or
    # BLAS change that moves a fit in the last digits, far narrower than the
    # differences between estimators it needs to distinguish (the closest pair,
    # lin_regression and aipw, sit $193 apart on a $5,000 gap -- so this band
    # does NOT separate those two, and is not meant to).
    frame = _frame()
    dataset = _observational_dataset(frame)
    estimate = get_estimator(estimator_name)().fit(dataset, EstimatorConfig(random_seed=0))

    gap = estimate.point - _experimental_benchmark(frame)

    assert gap == pytest.approx(MEASURED_GAPS[estimator_name], rel=0.15)


def test_the_naive_comparison_is_badly_wrong_in_both_size_and_sign():
    frame = _frame()
    estimate = (
        get_estimator("difference_in_means")()
        .fit(_observational_dataset(frame), EstimatorConfig(random_seed=0))
    )

    gap = estimate.point - _experimental_benchmark(frame)

    assert gap < -9000.0
    assert estimate.point < 0.0 < _experimental_benchmark(frame)


@pytest.mark.parametrize("estimator_name", ADJUSTING)
def test_adjustment_closes_part_of_the_naive_gap(estimator_name):
    frame = _frame()
    estimate = get_estimator(estimator_name)().fit(
        _observational_dataset(frame), EstimatorConfig(random_seed=0)
    )

    gap = estimate.point - _experimental_benchmark(frame)

    # Beats the naive comparison by at least $1,000. The smallest measured
    # improvement is ipw's, at $1,845.
    assert gap > NAIVE_GAP + 1000.0


@pytest.mark.parametrize("estimator_name", ADJUSTING)
def test_no_adjusting_estimator_recovers_the_experimental_benchmark(estimator_name):
    """The honest half of the finding, asserted so it cannot quietly change.

    If this test ever fails, the right response is to investigate, not to
    delete it -- an estimator in this codebase recovering LaLonde's
    experimental benchmark from CPS-1 would be a claim worth checking hard
    against the literature before believing.
    """
    frame = _frame()
    estimate = get_estimator(estimator_name)().fit(
        _observational_dataset(frame), EstimatorConfig(random_seed=0)
    )

    assert estimate.point < 0.0
    assert abs(estimate.point - _experimental_benchmark(frame)) > 3000.0
