"""`stats` against values computed by hand (shown in each test)."""

import math

import pytest
from cp_llm_eval.stats import clustered_rate, paired_difference, wilson


def test_a_clustered_rate_matches_the_hand_computed_cr1_form():
    # Clusters (n, d): (1, 4), (3, 4), (0, 2). Rate 4/10 = 0.4.
    # Residuals n - 0.4 d: -0.6, 1.4, -0.8; squares sum 0.36 + 1.96 + 0.64 = 2.96.
    # Variance 3/2 * 2.96 / 100 = 0.0444; se = 0.210713...
    rate = clustered_rate([(1, 4), (3, 4), (0, 2)])

    assert (rate.numerator, rate.denominator, rate.clusters) == (4, 10, 3)
    assert rate.rate == pytest.approx(0.4)
    assert rate.se == pytest.approx(math.sqrt(0.0444))
    assert rate.upper_bound is None


def test_identical_clusters_have_a_standard_error_of_zero():
    rate = clustered_rate([(1, 4), (1, 4), (1, 4)])

    assert rate.se == pytest.approx(0.0)


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


def test_a_paired_difference_needs_two_pairs():
    with pytest.raises(ValueError, match="two pairs"):
        paired_difference([(0.0, 1.0)])


def test_the_rule_of_three_never_bounds_a_proportion_above_one():
    assert clustered_rate([(0, 2)]).upper_bound == 1.0
