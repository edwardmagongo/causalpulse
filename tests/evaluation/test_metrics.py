import numpy as np
import pandas as pd
import pytest
from cp_evaluation.metrics import MetricValue, accuracy_metrics
from cp_evaluation.run import ROW_COLUMNS


def _frame(points, truth, status="fitted", **overrides) -> pd.DataFrame:
    """A minimal frame carrying only what the metric under test reads."""
    n = len(points)
    row = dict.fromkeys(ROW_COLUMNS, np.nan)
    data = {k: [v] * n for k, v in row.items()}
    data["point"] = list(points)
    data["truth"] = [truth] * n if np.isscalar(truth) else list(truth)
    data["status"] = [status] * n
    data["method"] = ["m"] * n
    for key, value in overrides.items():
        data[key] = list(value) if isinstance(value, list) else [value] * n
    return pd.DataFrame(data, columns=list(ROW_COLUMNS))


def test_bias_is_the_mean_signed_error():
    frame = _frame([1.0, 2.0, 3.0], truth=2.0)

    result = accuracy_metrics(frame)

    assert result["bias"].value == pytest.approx(0.0)
    assert result["bias"].n_used == 3


def test_bias_mcse_is_the_standard_error_of_the_mean():
    # errors are -1, 0, +1 -> sd (ddof=1) = 1.0 -> mcse = 1/sqrt(3)
    frame = _frame([1.0, 2.0, 3.0], truth=2.0)

    assert accuracy_metrics(frame)["bias"].mcse == pytest.approx(1.0 / np.sqrt(3))


def test_rmse_and_mae_are_computed_from_the_same_errors():
    frame = _frame([1.0, 2.0, 3.0], truth=2.0)

    result = accuracy_metrics(frame)

    assert result["rmse"].value == pytest.approx(np.sqrt(2.0 / 3.0))
    assert result["mae"].value == pytest.approx(2.0 / 3.0)


def test_relative_bias_divides_by_truth():
    frame = _frame([2.0, 4.0], truth=2.0)  # bias = +1.0, truth = 2.0

    assert accuracy_metrics(frame)["relative_bias"].value == pytest.approx(0.5)


def test_relative_bias_is_nan_when_truth_is_zero_not_infinite():
    # Scenarios B and P2 are the null cells -- precisely the ones the
    # false-positive test needs -- and bias/0 is the natural formula there.
    frame = _frame([0.5, 1.5], truth=0.0)

    result = accuracy_metrics(frame)

    assert np.isnan(result["relative_bias"].value)
    assert not np.isinf(result["relative_bias"].value)


def test_non_fitted_rows_never_enter_a_metric():
    fitted = _frame([1.0, 3.0], truth=2.0)
    ineligible = _frame([np.nan, np.nan], truth=2.0, status="ineligible")
    frame = pd.concat([fitted, ineligible], ignore_index=True)

    result = accuracy_metrics(frame)

    assert result["bias"].n_used == 2
    assert result["bias"].value == pytest.approx(0.0)


def test_a_metric_over_no_usable_rows_reports_nan_and_n_used_zero():
    frame = _frame([np.nan], truth=2.0, status="refused")

    result = accuracy_metrics(frame)

    assert np.isnan(result["bias"].value)
    assert result["bias"].n_used == 0


def test_mcse_needs_at_least_two_observations():
    frame = _frame([1.0], truth=2.0)

    result = accuracy_metrics(frame)

    assert result["bias"].value == pytest.approx(-1.0)
    assert np.isnan(result["bias"].mcse)


def test_metric_value_is_a_plain_value_object():
    value = MetricValue(value=1.0, mcse=0.1, n_used=5)

    assert (value.value, value.mcse, value.n_used) == (1.0, 0.1, 5)


def test_rmse_mcse_is_nan_when_mean_squared_error_is_exactly_zero():
    # squared errors are all zero -> delta method divides by sqrt(0)
    frame = _frame([2.0, 2.0, 2.0], truth=2.0)

    result = accuracy_metrics(frame)

    assert result["rmse"].value == pytest.approx(0.0)
    assert np.isnan(result["rmse"].mcse)
