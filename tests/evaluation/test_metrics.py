import numpy as np
import pandas as pd
import pytest
from cp_evaluation.metrics import (
    MetricValue,
    accuracy_metrics,
    coverage_metrics,
    detection_metrics,
    se_calibration,
    sign_and_magnitude,
    summarise,
)
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


def _covered_frame(lowers, uppers, truth=1.0, **overrides) -> pd.DataFrame:
    frame = _frame([1.0] * len(lowers), truth=truth, **overrides)
    frame["ci_lower"] = list(lowers)
    frame["ci_upper"] = list(uppers)
    return frame


def test_analytic_coverage_is_the_share_of_intervals_containing_truth():
    frame = _covered_frame([0.0, 0.0, 2.0, 2.5], [2.0, 2.0, 3.0, 3.0], truth=1.0)

    result = coverage_metrics(frame)

    assert result["coverage_analytic"].value == pytest.approx(0.5)
    assert result["coverage_analytic"].n_used == 4


def test_coverage_mcse_is_binomial():
    frame = _covered_frame([0.0, 0.0, 2.0, 2.5], [2.0, 2.0, 3.0, 3.0], truth=1.0)

    # sqrt(p(1-p)/n) = sqrt(0.25/4) = 0.25
    assert coverage_metrics(frame)["coverage_analytic"].mcse == pytest.approx(0.25)


def test_unanimous_coverage_uses_the_rule_of_three_not_zero():
    # A 4-of-4 share has a binomial MCSE of exactly 0.0, which reads as
    # infinite precision from four observations. The rule of three gives
    # 3/n as the usable upper bound on the unobserved tail.
    frame = _covered_frame([0.0] * 4, [2.0] * 4, truth=1.0)

    result = coverage_metrics(frame)

    assert result["coverage_analytic"].value == pytest.approx(1.0)
    assert result["coverage_analytic"].mcse == pytest.approx(3.0 / 4.0)


def test_unanimous_non_coverage_also_uses_the_rule_of_three():
    frame = _covered_frame([5.0] * 4, [6.0] * 4, truth=1.0)

    result = coverage_metrics(frame)

    assert result["coverage_analytic"].value == pytest.approx(0.0)
    assert result["coverage_analytic"].mcse == pytest.approx(3.0 / 4.0)


def test_bootstrap_coverage_is_reported_separately_and_skips_missing_intervals():
    frame = _covered_frame([0.0, 0.0], [2.0, 2.0], truth=1.0)
    frame["bootstrap_ci_lower"] = [0.0, np.nan]
    frame["bootstrap_ci_upper"] = [2.0, np.nan]

    result = coverage_metrics(frame)

    assert result["coverage_bootstrap"].value == pytest.approx(1.0)
    assert result["coverage_bootstrap"].n_used == 1


def test_se_calibration_is_mean_reported_se_over_empirical_sd():
    frame = _frame([1.0, 2.0, 3.0], truth=2.0)
    frame["se"] = [1.0, 1.0, 1.0]

    # empirical sd of [1,2,3] with ddof=1 is 1.0, so the ratio is 1.0
    assert se_calibration(frame).value == pytest.approx(1.0)


def test_se_calibration_detects_an_understated_standard_error():
    frame = _frame([1.0, 2.0, 3.0], truth=2.0)
    frame["se"] = [0.5, 0.5, 0.5]

    assert se_calibration(frame).value == pytest.approx(0.5)


def test_se_calibration_excludes_rows_reporting_no_standard_error():
    frame = _frame([1.0, 2.0, 3.0, 4.0], truth=2.0)
    frame["se"] = [1.0, 1.0, 1.0, np.nan]

    result = se_calibration(frame)

    assert result.n_used == 3


def test_se_calibration_is_undefined_with_fewer_than_two_usable_rows():
    frame = _frame([1.0], truth=2.0)
    frame["se"] = [1.0]

    result = se_calibration(frame)

    assert np.isnan(result.value)
    assert result.n_used == 1
    # Too few usable rows for a standard error at all -- nan, not 0.0.
    assert np.isnan(result.mcse)


def test_se_calibration_mcse_is_bootstrapped_not_a_fixed_denominator_formula():
    # A constant reported se across replications, with point estimates that
    # vary. The old closed-form mcse, std(se, ddof=1) / sqrt(n) / empirical_sd,
    # has a zero numerator whenever se is constant, so it returned exactly
    # 0.0 here -- reading as infinite precision from a handful of rows, even
    # though the ratio's true sampling sd under these conditions is around
    # 2.7. The bootstrap propagates the denominator's own sampling
    # variability and must be finite and strictly positive instead.
    frame = _frame([1.0, 2.0, 3.0, 10.0, -4.0], truth=2.0)
    frame["se"] = [1.0, 1.0, 1.0, 1.0, 1.0]

    result = se_calibration(frame)

    assert np.isfinite(result.mcse)
    assert result.mcse > 0.0


def test_se_calibration_mcse_is_reproducible_for_the_same_seed():
    frame = _frame([1.0, 2.0, 3.0, 10.0, -4.0], truth=2.0)
    frame["se"] = [1.0, 1.0, 1.0, 1.0, 1.0]

    first = se_calibration(frame, seed=7)
    second = se_calibration(frame, seed=7)

    assert first.mcse == second.mcse


def test_mean_ci_width_is_reported():
    frame = _covered_frame([0.0, 1.0], [2.0, 5.0], truth=1.0)

    assert coverage_metrics(frame)["mean_ci_width"].value == pytest.approx(3.0)


def test_type_s_is_the_share_of_significant_estimates_with_the_wrong_sign():
    # truth is +1. Two significant estimates: one positive, one negative.
    frame = _covered_frame([0.5, -2.0], [1.5, -0.5], truth=1.0)
    frame["point"] = [1.0, -1.25]

    result = sign_and_magnitude(frame)

    assert result["type_s"].value == pytest.approx(0.5)
    assert result["type_s"].n_used == 2


def test_type_m_is_the_exaggeration_ratio_among_significant_estimates():
    # truth +1; significant estimates of magnitude 2 and 4 -> ratio 3.0
    frame = _covered_frame([1.5, 3.5], [2.5, 4.5], truth=1.0)
    frame["point"] = [2.0, 4.0]

    assert sign_and_magnitude(frame)["type_m"].value == pytest.approx(3.0)


def test_type_s_and_type_m_are_undefined_when_truth_is_zero():
    # A null cell has no true sign to get wrong and no magnitude to exaggerate.
    frame = _covered_frame([0.5, 1.5], [2.5, 3.5], truth=0.0)
    frame["point"] = [1.0, 2.0]

    result = sign_and_magnitude(frame)

    assert np.isnan(result["type_s"].value)
    assert np.isnan(result["type_m"].value)


def test_type_s_and_type_m_are_undefined_when_nothing_is_significant():
    frame = _covered_frame([-1.0, -1.0], [3.0, 3.0], truth=1.0)
    frame["point"] = [1.0, 1.0]

    result = sign_and_magnitude(frame)

    assert result["type_s"].n_used == 0
    assert np.isnan(result["type_m"].value)


def test_power_is_the_share_of_intervals_excluding_zero():
    frame = _covered_frame([0.5, -1.0, 0.2], [1.5, 3.0, 2.0], truth=1.0)

    assert detection_metrics(frame)["power"].value == pytest.approx(2.0 / 3.0)


def test_a_null_cell_reports_a_false_positive_rate_not_power():
    frame = _covered_frame([0.5, -1.0], [1.5, 3.0], truth=0.0)

    result = detection_metrics(frame)

    assert result["false_positive_rate"].value == pytest.approx(0.5)
    assert np.isnan(result["power"].value)


def test_summarise_groups_by_method_and_cell_identity():
    left = _frame([1.0, 2.0], truth=2.0)
    left["method"] = "a"
    left["scenario"] = "A"
    right = _frame([5.0, 6.0], truth=2.0)
    right["method"] = "b"
    right["scenario"] = "A"
    frame = pd.concat([left, right], ignore_index=True)

    summary = summarise(frame)

    assert set(summary["method"]) == {"a", "b"}
    assert "bias" in summary.columns
    assert "bias_mcse" in summary.columns
    assert "bias_n_used" in summary.columns


def test_summarise_carries_status_counts_so_absence_is_visible():
    fitted = _frame([1.0, 2.0], truth=2.0)
    fitted["method"] = "a"
    refused = _frame([np.nan], truth=2.0, status="refused")
    refused["method"] = "a"
    frame = pd.concat([fitted, refused], ignore_index=True)

    summary = summarise(frame)

    assert summary["n_fitted"].iloc[0] == 2
    assert summary["n_refused"].iloc[0] == 1
    assert summary["n_ineligible"].iloc[0] == 0
