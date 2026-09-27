import math

import numpy as np
import pytest
from cp_evaluation.cate import CATE_METRIC_COLUMNS, CateTruth, cate_row_metrics

_TRUTH = np.array([-0.05, 0.0, 0.1, 0.2, 0.4])


def _heterogeneous(values=_TRUTH) -> CateTruth:
    return CateTruth(values=values, heterogeneous=True)


def test_the_columns_are_the_three_the_design_names():
    assert CATE_METRIC_COLUMNS == ("cate_correlation", "cate_rmse", "cate_calibration_slope")


def test_perfect_recovery_scores_correlation_one_rmse_zero_slope_one():
    metrics = cate_row_metrics(_TRUTH.copy(), _heterogeneous())

    assert set(metrics) == set(CATE_METRIC_COLUMNS)
    assert metrics["cate_correlation"] == pytest.approx(1.0)
    assert metrics["cate_rmse"] == pytest.approx(0.0)
    assert metrics["cate_calibration_slope"] == pytest.approx(1.0)


def test_doubled_estimates_keep_correlation_and_halve_the_slope():
    # Over-dispersed estimates: the slope of truth on estimate is the shrinkage
    # a reader should apply, here exactly one half.
    metrics = cate_row_metrics(2.0 * _TRUTH, _heterogeneous())

    assert metrics["cate_correlation"] == pytest.approx(1.0)
    assert metrics["cate_calibration_slope"] == pytest.approx(0.5)
    assert metrics["cate_rmse"] == pytest.approx(float(np.sqrt(np.mean(_TRUTH**2))))


def test_a_constant_offset_moves_rmse_but_not_slope_or_correlation():
    metrics = cate_row_metrics(_TRUTH + 0.3, _heterogeneous())

    assert metrics["cate_rmse"] == pytest.approx(0.3)
    assert metrics["cate_calibration_slope"] == pytest.approx(1.0)
    assert metrics["cate_correlation"] == pytest.approx(1.0)


def test_a_constant_estimate_has_no_slope_or_correlation():
    # var(e) == 0: both are undefined. Must be NaN without a numpy warning,
    # which filterwarnings=error would turn into a failure.
    metrics = cate_row_metrics(np.full(_TRUTH.shape, 0.1), _heterogeneous())

    assert math.isnan(metrics["cate_calibration_slope"])
    assert math.isnan(metrics["cate_correlation"])
    assert np.isfinite(metrics["cate_rmse"])


def test_a_declared_constant_truth_is_scored_on_rmse_only():
    # NULL's realised truth has sd ~3e-17, not 0 (design section 1): the
    # declaration, not a variance test, is what turns correlation off.
    truth = CateTruth(values=np.full(5, 0.05) + np.array([0, 1e-17, 0, -1e-17, 0]),
                      heterogeneous=False)
    estimate = np.array([0.02, 0.07, 0.05, 0.11, 0.0])

    metrics = cate_row_metrics(estimate, truth)

    assert math.isnan(metrics["cate_correlation"])
    assert math.isnan(metrics["cate_calibration_slope"])
    assert metrics["cate_rmse"] == pytest.approx(
        float(np.sqrt(np.mean((estimate - 0.05) ** 2)))
    )


def test_a_wrong_length_estimate_raises():
    with pytest.raises(ValueError, match="shape"):
        cate_row_metrics(_TRUTH[:-1], _heterogeneous())


def test_a_scalar_estimate_raises():
    with pytest.raises(ValueError, match="shape"):
        cate_row_metrics(0.1, _heterogeneous())


def test_a_non_finite_estimate_raises():
    estimate = _TRUTH.copy()
    estimate[2] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        cate_row_metrics(estimate, _heterogeneous())
