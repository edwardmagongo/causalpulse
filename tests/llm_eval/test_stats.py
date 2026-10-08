"""`stats` against values computed by hand (shown in each test)."""

import math

import pytest
from cp_llm_eval.stats import clustered_rate, paired_difference, wilson, wilson_weighted


def test_a_clustered_rate_matches_the_hand_computed_cr1_form():
    # Clusters (n, d): (1, 4), (3, 4), (0, 2). Rate 4/10 = 0.4.
    # Residuals n - 0.4 d: -0.6, 1.4, -0.8; squares sum 0.36 + 1.96 + 0.64 = 2.96.
    # Variance 3/2 * 2.96 / 100 = 0.0444; se = 0.210713...
    rate = clustered_rate([(1, 4), (3, 4), (0, 2)])

    assert (rate.numerator, rate.denominator, rate.clusters) == (4, 10, 3)
    assert rate.rate == pytest.approx(0.4)
    assert rate.se == pytest.approx(math.sqrt(0.0444))
    assert rate.upper_bound is None


def test_identical_clusters_have_no_standard_error():
    # Every cluster at 1/4: each residual is 0, so the CR1 se is exactly 0,
    # which would read as certainty; it is reported as None instead.
    rate = clustered_rate([(1, 4), (1, 4), (1, 4)])

    assert rate.rate == pytest.approx(0.25)
    assert (rate.se, rate.upper_bound, rate.lower_bound) == (None, None, None)


def test_clusters_with_one_proportion_have_no_standard_error():
    # (1, 2) and (2, 4) are both 1/2: residuals 1 - 0.5*2 = 0 and 2 - 0.5*4 = 0.
    assert clustered_rate([(1, 2), (2, 4)]).se is None


def test_a_unanimous_rate_carries_the_mirrored_rule_of_three_and_no_standard_error():
    # 50 of 50 over 25 cases: lower bound 1 - 3/50 = 0.94.
    rate = clustered_rate([(2, 2)] * 25)

    assert (rate.numerator, rate.denominator, rate.clusters) == (50, 50, 25)
    assert (rate.rate, rate.se, rate.upper_bound) == (1.0, None, None)
    assert rate.lower_bound == pytest.approx(0.94)


def test_the_mirrored_rule_of_three_never_bounds_a_proportion_below_zero():
    assert clustered_rate([(2, 2)]).lower_bound == 0.0


def test_a_rate_that_is_neither_zero_nor_unanimous_has_no_lower_bound():
    assert clustered_rate([(1, 4), (3, 4)]).lower_bound is None


def test_a_zero_rate_carries_the_rule_of_three_and_no_standard_error():
    rate = clustered_rate([(0, 50), (0, 50)])

    assert (rate.rate, rate.se) == (0.0, None)
    assert rate.upper_bound == pytest.approx(0.03)


def test_one_cluster_has_no_standard_error():
    assert clustered_rate([(2, 10)]).se is None


def test_an_empty_cluster_is_not_counted():
    assert clustered_rate([(1, 4), (0, 0), (3, 4)]).clusters == 2


def test_a_rate_needs_a_unit():
    with pytest.raises(ValueError, match="at least one unit"):
        clustered_rate([(0, 0)])


def test_the_wilson_interval_matches_a_published_value():
    # 8 of 10 at 95%: (0.4902, 0.9433), as in Newcombe (1998), table I.
    interval = wilson(8, 10)

    assert interval.estimate == 0.8
    assert interval.lower == pytest.approx(0.4902, abs=1e-4)
    assert interval.upper == pytest.approx(0.9433, abs=1e-4)


def test_the_wilson_interval_stays_inside_zero_and_one():
    assert wilson(0, 5).lower == 0.0
    assert wilson(5, 5).upper == 1.0
    assert wilson(0, 5).upper > 0


def test_a_paired_difference_matches_the_hand_computed_mean_and_se():
    # Differences 0.25, -0.25, 0.5: mean 1/6; sample variance
    # ((1/12)^2 + (5/12)^2 + (4/12)^2) / 2 = (1 + 25 + 16) / 288 = 42/288;
    # se = sqrt(42/288 / 3).
    difference = paired_difference([(0.25, 0.5), (0.5, 0.25), (0.0, 0.5)])

    assert difference.mean == pytest.approx(1 / 6)
    assert difference.se == pytest.approx(math.sqrt(42 / 288 / 3))
    assert difference.lower < difference.mean < difference.upper
    assert difference.pairs == 3
    assert difference.degenerate is False


def test_a_paired_difference_with_one_difference_everywhere_is_degenerate():
    # Every case moves by 0.1: no spread, so se 0 and no interval to speak of.
    difference = paired_difference([(0.0, 0.1), (0.2, 0.3), (0.5, 0.6)])

    assert difference.se == 0.0
    assert difference.degenerate is True


def test_a_paired_difference_needs_two_pairs():
    with pytest.raises(ValueError, match="two pairs"):
        paired_difference([(0.0, 1.0)])


def test_the_rule_of_three_never_bounds_a_proportion_above_one():
    assert clustered_rate([(0, 2)]).upper_bound == 1.0


def test_the_weighted_wilson_interval_at_equal_weights_is_the_wilson_interval():
    # Kish n_eff of n equal weights is n, so p = 2/4 at n_eff 4 is wilson(2, 4):
    # the published [0.150, 0.850].
    weighted = wilson_weighted(0.5, 4.0)

    assert weighted.estimate == 0.5
    assert weighted.lower == pytest.approx(wilson(2, 4).lower)
    assert weighted.upper == pytest.approx(wilson(2, 4).upper)
    assert (weighted.lower, weighted.upper) == pytest.approx((0.150, 0.850), abs=5e-4)
    assert weighted.n == 4.0


def test_a_smaller_effective_sample_widens_the_weighted_interval():
    # p = 1/6 at n_eff 36/26 (weights 1 and 5) against the same p at n = 6.
    narrow, wide = wilson_weighted(1 / 6, 6.0), wilson_weighted(1 / 6, 36 / 26)

    assert wide.lower <= narrow.lower and wide.upper > narrow.upper


def test_a_weighted_proportion_needs_a_positive_effective_sample_and_a_proportion():
    with pytest.raises(ValueError):
        wilson_weighted(0.5, 0.0)
    with pytest.raises(ValueError):
        wilson_weighted(1.5, 4.0)
