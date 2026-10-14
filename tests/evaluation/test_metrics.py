import numpy as np
import pandas as pd
import pytest
from cp_evaluation.metrics import (
    MetricValue,
    accuracy_metrics,
    cate_metrics,
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
    overrides.setdefault("ci_basis", "analytic")
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


def test_each_non_analytic_basis_gets_its_own_coverage_column():
    frame = pd.concat(
        [
            _covered_frame([0.0, 5.0], [2.0, 6.0], truth=1.0, ci_basis="posterior"),
            _covered_frame([0.0], [2.0], truth=1.0, ci_basis="permutation"),
            _covered_frame([0.0, 0.0, 5.0], [2.0, 2.0, 6.0], truth=1.0, ci_basis="bootstrap"),
        ],
        ignore_index=True,
    )

    result = coverage_metrics(frame)

    assert result["coverage_posterior"].value == pytest.approx(0.5)
    assert result["coverage_posterior"].n_used == 2
    assert result["coverage_permutation"].value == pytest.approx(1.0)
    assert result["coverage_permutation"].n_used == 1
    assert result["coverage_bootstrap_se"].value == pytest.approx(2 / 3)
    assert result["coverage_bootstrap_se"].n_used == 3


def test_coverage_analytic_counts_only_analytic_rows():
    frame = pd.concat(
        [
            _covered_frame([0.0, 5.0], [2.0, 6.0], truth=1.0, ci_basis="analytic"),
            _covered_frame([0.0, 0.0], [2.0, 2.0], truth=1.0, ci_basis="posterior"),
        ],
        ignore_index=True,
    )

    result = coverage_metrics(frame)

    assert result["coverage_analytic"].value == pytest.approx(0.5)
    assert result["coverage_analytic"].n_used == 2


def test_a_basis_with_no_rows_reports_nan_and_n_used_zero():
    result = coverage_metrics(_covered_frame([0.0], [2.0], truth=1.0))

    for name in ("coverage_posterior", "coverage_permutation", "coverage_bootstrap_se"):
        assert np.isnan(result[name].value)
        assert result[name].n_used == 0


def test_bootstrap_ci_coverage_ignores_ci_basis():
    # coverage_bootstrap reads the separate bootstrap_ci field for any basis.
    frame = _covered_frame([0.0], [2.0], truth=1.0, ci_basis="bootstrap")
    frame["bootstrap_ci_lower"] = [0.0]
    frame["bootstrap_ci_upper"] = [2.0]

    result = coverage_metrics(frame)

    assert result["coverage_bootstrap"].n_used == 1
    assert result["coverage_bootstrap_se"].n_used == 1
    assert result["coverage_analytic"].n_used == 0


def test_mean_ci_width_still_pools_every_basis():
    frame = pd.concat(
        [
            _covered_frame([0.0], [2.0], truth=1.0, ci_basis="analytic"),
            _covered_frame([0.0], [4.0], truth=1.0, ci_basis="posterior"),
        ],
        ignore_index=True,
    )

    assert coverage_metrics(frame)["mean_ci_width"].value == pytest.approx(3.0)


def test_a_donor_pool_permutation_row_counts_in_permutation_not_analytic():
    # synthetic_control on a donor-pool cell: its placebo interval is permutation
    # coverage, never analytic, and a cross-section analytic row beside it stays
    # analytic only.
    donor = _covered_frame(
        [0.0, 0.0], [2.0, 2.0], truth=1.0, shape="donor_pool", ci_basis="permutation"
    )
    cross_section = _covered_frame(
        [5.0], [6.0], truth=1.0, shape="cross_section", ci_basis="analytic"
    )
    frame = pd.concat([donor, cross_section], ignore_index=True)

    result = coverage_metrics(frame)

    assert result["coverage_permutation"].value == pytest.approx(1.0)
    assert result["coverage_permutation"].n_used == 2
    assert result["coverage_analytic"].value == pytest.approx(0.0)
    assert result["coverage_analytic"].n_used == 1


def test_a_donor_pool_posterior_row_counts_in_posterior_not_analytic():
    # bsts on a donor-pool cell.
    donor = _covered_frame(
        [5.0, 0.0], [6.0, 2.0], truth=1.0, shape="donor_pool", ci_basis="posterior"
    )

    result = coverage_metrics(donor)

    assert result["coverage_posterior"].value == pytest.approx(0.5)
    assert result["coverage_posterior"].n_used == 2
    assert result["coverage_analytic"].n_used == 0
    assert result["coverage_permutation"].n_used == 0


def test_coverage_metrics_works_on_a_frame_with_no_shape_column():
    frame = _covered_frame([0.0, 0.0], [2.0, 0.5], truth=1.0, ci_basis="analytic").drop(
        columns="shape"
    )

    result = coverage_metrics(frame)

    assert "shape" not in frame.columns
    assert result["coverage_analytic"].value == pytest.approx(0.5)
    assert result["coverage_analytic"].n_used == 2
    assert result["coverage_permutation"].n_used == 0
    assert result["coverage_posterior"].n_used == 0


def test_hierarchical_dids_panel_posterior_ci_counts_as_posterior_not_analytic():
    # hierarchical_did's panel interval is a posterior credible interval, so it
    # belongs to coverage_posterior and never to coverage_analytic.
    frame = _covered_frame([0.0], [2.0], truth=1.0, shape="panel", ci_basis="posterior")

    result = coverage_metrics(frame)

    assert result["coverage_analytic"].n_used == 0
    assert result["coverage_posterior"].value == pytest.approx(1.0)
    assert result["coverage_posterior"].n_used == 1


def test_coverage_posterior_pools_panel_and_donor_pool_rows_it_is_handed():
    # The basis filter ignores shape by design. A caller pooling one method's
    # rows across shapes with by=[] gets both populations in one figure, which
    # is why E5 scopes hierarchical_did to panel rows itself before pooling.
    frame = pd.concat(
        [
            _covered_frame([0.0], [2.0], truth=1.0, shape="panel", ci_basis="posterior"),
            _covered_frame([5.0], [6.0], truth=1.0, shape="donor_pool", ci_basis="posterior"),
        ],
        ignore_index=True,
    )

    result = coverage_metrics(frame)

    assert result["coverage_posterior"].value == pytest.approx(0.5)
    assert result["coverage_posterior"].n_used == 2


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


def test_summarise_reports_the_mean_realised_treated_share_per_group():
    # A finding of the Phase 18 whole-branch review: the measured share -- the
    # mitigation for treatment_intercept being a logit-scale
    # parameter, not a quantity a reader thinks in -- must reach summarise's
    # own output, not just sit unread in the results frame.
    frame = _frame([1.0, 1.0], truth=1.0, realised_treated_share=[0.40, 0.60])

    summary = summarise(frame)

    assert summary["realised_treated_share"].iloc[0] == pytest.approx(0.50)
    assert summary["realised_treated_share_n_used"].iloc[0] == 2


def test_summarise_reports_no_realised_treated_share_for_a_panel_group():
    # Panel rows carry NaN for realised_treated_share (run.py's own
    # convention for every Phase 18 axis on a panel row, since the axis is
    # cross-sectional-only); the group-level mean must follow that same
    # absent-not-zero convention rather than reading as a spurious 0.0.
    frame = _frame([1.0, 1.0], truth=1.0)  # realised_treated_share defaults to NaN

    summary = summarise(frame)

    assert np.isnan(summary["realised_treated_share"].iloc[0])
    assert summary["realised_treated_share_n_used"].iloc[0] == 0


def test_summarise_handles_a_frame_with_no_realised_treated_share_column_at_all():
    # An older stored run (from before this phase added the column) has no
    # realised_treated_share column at all -- not even one full of NaN. This
    # is a different case from the panel-group test above, which drops it to
    # NaN but still has the column: `summarise`'s own
    # `if "realised_treated_share" in group.columns` branch reads straight
    # off column presence, and dropping the column here (rather than filling
    # it with NaN) is what actually reaches the `else` arm that returns
    # `_EMPTY` -- untested before this, since every other test's frame comes
    # from `_frame`, which always carries the column via `ROW_COLUMNS`.
    frame = _frame([1.0, 2.0], truth=2.0).drop(columns=["realised_treated_share"])

    summary = summarise(frame)

    assert np.isnan(summary["realised_treated_share"].iloc[0])
    assert np.isnan(summary["realised_treated_share_mcse"].iloc[0])
    assert summary["realised_treated_share_n_used"].iloc[0] == 0


def _se_calibration_frame(method: str, scenario: str) -> pd.DataFrame:
    # A constant reported se with varying points, same shape as the frame
    # used above to prove the bootstrap mcse is finite and positive.
    frame = _frame([1.0, 2.0, 3.0, 10.0, -4.0], truth=2.0)
    frame["se"] = [1.0, 1.0, 1.0, 1.0, 1.0]
    frame["method"] = method
    frame["scenario"] = scenario
    return frame


def test_summarise_gives_identical_groups_independent_bootstrap_seeds():
    # Two groups built from the exact same data but with different group
    # keys (different scenario). With the old shared seed=0, every group in
    # one summarise() call bootstraps identically-shaped data with the same
    # resample indices, so identical data would produce identical MCSEs --
    # an artifact of the shared stream, not a fact about the data. Deriving
    # each group's seed from its own identity must break that tie.
    left = _se_calibration_frame("a", "A")
    right = _se_calibration_frame("a", "B")
    frame = pd.concat([left, right], ignore_index=True)

    summary = summarise(frame)

    mcses = summary.set_index("scenario")["se_calibration_mcse"]
    assert mcses.loc["A"] != mcses.loc["B"]


def test_summarise_gives_different_data_different_mcse_too():
    left = _se_calibration_frame("a", "A")
    right = _frame([1.0, 5.0, 3.0, -8.0, 9.0], truth=2.0)
    right["se"] = [1.0, 1.0, 1.0, 1.0, 1.0]
    right["method"] = "a"
    right["scenario"] = "B"
    frame = pd.concat([left, right], ignore_index=True)

    summary = summarise(frame)

    mcses = summary.set_index("scenario")["se_calibration_mcse"]
    assert mcses.loc["A"] != mcses.loc["B"]


def test_summarise_is_reproducible():
    left = _se_calibration_frame("a", "A")
    right = _se_calibration_frame("b", "B")
    frame = pd.concat([left, right], ignore_index=True)

    first = summarise(frame)
    second = summarise(frame)

    pd.testing.assert_frame_equal(first, second)


def test_summarise_root_seed_changes_se_calibration_mcse():
    frame = _se_calibration_frame("a", "A")

    default_seed = summarise(frame)
    other_seed = summarise(frame, seed=12345)

    assert (
        default_seed["se_calibration_mcse"].iloc[0]
        != other_seed["se_calibration_mcse"].iloc[0]
    )


def test_summarise_does_not_pool_cells_that_differ_on_a_new_axis():
    # Phase 18's four axes are part of a cell's identity, not of its contents.
    # Two cells that differ only in `treatment_intercept` are different cells,
    # and a summary that pools them reports one bias averaged across the axis
    # -- which both hides the axis's effect and contaminates the baseline
    # cell's own number with the perturbed cells' rows. The grid varies these
    # axes against an anchor that IS a baseline cell (see
    # `grid._expand_cross_section`), so the pooling is not hypothetical: every
    # perturbed cell shares its baseline twin's shape/scenario/n_units/
    # n_periods/confounding_strength exactly.
    baseline = _frame([1.0, 1.0], truth=1.0, treatment_intercept=0.0)
    perturbed = _frame([5.0, 5.0], truth=1.0, treatment_intercept=1.5)
    frame = pd.concat([baseline, perturbed], ignore_index=True)

    summary = summarise(frame).set_index("treatment_intercept")

    assert len(summary) == 2
    # And the two groups keep their own numbers rather than meeting in the
    # middle: pooled, both would read as a bias of 2.0.
    assert summary.loc[0.0, "bias"] == pytest.approx(0.0)
    assert summary.loc[1.5, "bias"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "axis, value",
    [
        ("noise_scale", 0.9),
        ("effect_scale", 2.0),
        ("heterogeneity_scale", 0.0),
    ],
)
def test_summarise_separates_every_new_axis_not_just_the_first(axis, value):
    # One parametrised case per remaining axis: a fix that added only
    # `treatment_intercept` to the grouping would pass the test above and
    # still average the other three away.
    frame = pd.concat(
        [_frame([1.0], truth=1.0, **{axis: 0.5}), _frame([5.0], truth=1.0, **{axis: value})],
        ignore_index=True,
    )

    assert len(summarise(frame)) == 2


def _cate_rows(correlations, status="fitted") -> pd.DataFrame:
    records = []
    for seed, value in enumerate(correlations):
        row = dict.fromkeys(ROW_COLUMNS, np.nan)
        row.update(
            shape="cate", scenario="RICH", n_units=2000, seed=seed, method="causal_forest",
            status=status, reason="", estimand="ATE", truth=0.1, point=0.1,
            cate_correlation=value, cate_rmse=0.05, cate_calibration_slope=0.9,
            se=0.02, ci_lower=0.06, ci_upper=0.14, ci_basis="analytic",
            runtime_seconds=1.0,
        )
        records.append(row)
    return pd.DataFrame(records, columns=list(ROW_COLUMNS))


def test_cate_metrics_average_over_fitted_rows_with_an_mcse():
    metrics = cate_metrics(_cate_rows([0.8, 0.9, 0.7, 0.8]))

    assert metrics["cate_correlation"].value == pytest.approx(0.8)
    assert metrics["cate_correlation"].mcse == pytest.approx(
        float(np.std([0.8, 0.9, 0.7, 0.8], ddof=1) / 2.0)
    )
    assert metrics["cate_correlation"].n_used == 4
    assert metrics["cate_rmse"].value == pytest.approx(0.05)


def test_cate_metrics_skip_rows_that_carry_no_cate():
    frame = pd.concat(
        [_cate_rows([0.8, 0.9]), _cate_rows([np.nan, np.nan])], ignore_index=True
    )

    assert cate_metrics(frame)["cate_correlation"].n_used == 2


def test_a_frame_written_before_the_cate_columns_existed_reports_nothing():
    old = _cate_rows([0.8, 0.9]).drop(columns=["cate_correlation", "cate_rmse",
                                              "cate_calibration_slope"])

    metrics = cate_metrics(old)

    assert all(m.n_used == 0 and np.isnan(m.value) for m in metrics.values())


def test_summarise_carries_the_three_cate_columns():
    summary = summarise(_cate_rows([0.8, 0.9, 0.7]))

    for name in ("cate_correlation", "cate_rmse", "cate_calibration_slope"):
        for suffix in ("", "_mcse", "_n_used"):
            assert f"{name}{suffix}" in summary.columns
    assert summary.loc[0, "cate_correlation_n_used"] == 3
