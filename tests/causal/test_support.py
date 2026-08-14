import numpy as np
import pytest
from cp_causal._support import (
    coefficient_se,
    critical_value,
    normal_p_value,
    wald_interval,
)
from scipy import stats


def test_critical_value_matches_the_normal_quantile():
    assert critical_value(0.95) == pytest.approx(1.959963984540054)
    assert critical_value(0.90) == pytest.approx(stats.norm.ppf(0.95))


def test_wald_interval_is_symmetric_about_the_point():
    interval = wald_interval(point=2.0, se=0.5, ci_level=0.95)

    assert interval.lower == pytest.approx(2.0 - 1.959963984540054 * 0.5)
    assert interval.upper == pytest.approx(2.0 + 1.959963984540054 * 0.5)
    assert interval.level == 0.95


def test_zero_standard_error_yields_a_degenerate_interval_and_p_value_one():
    # A zero SE is not an error: a perfectly-fitted arm can produce one. The
    # honest report is a point interval and no evidence against the null,
    # which is what the four Phase 2 estimators already do inline.
    interval = wald_interval(point=1.0, se=0.0, ci_level=0.95)

    assert interval.lower == interval.upper == 1.0
    assert normal_p_value(point=1.0, se=0.0) == 1.0


def test_p_value_matches_the_two_sided_normal_tail():
    assert normal_p_value(point=1.96, se=1.0) == pytest.approx(0.05, abs=1e-3)
    assert normal_p_value(point=0.0, se=1.0) == pytest.approx(1.0)


def test_coefficient_se_reads_one_diagonal_entry():
    covariance = np.array([[4.0, 0.1], [0.1, 9.0]])

    assert coefficient_se(covariance, 0) == pytest.approx(2.0)
    assert coefficient_se(covariance, 1) == pytest.approx(3.0)


def test_coefficient_se_rejects_a_negative_variance():
    # Reading one entry rather than sqrt-ing the whole diagonal is what keeps a
    # fixed-effect coefficient's floating-point noise from raising under
    # filterwarnings=["error"]. A negative variance on the *reported*
    # coefficient is a real problem and must not be silently clamped.
    covariance = np.array([[-1e-27, 0.0], [0.0, 4.0]])

    with pytest.raises(ValueError, match="negative variance"):
        coefficient_se(covariance, 0)

    assert coefficient_se(covariance, 1) == pytest.approx(2.0)


def test_negative_diagonal_entries_elsewhere_do_not_raise():
    # The motivating case: on a TWFE design the unit- and period-dummy
    # variances are near zero and come back at around -1e-27. Extracting only
    # the reported coefficient's entry is what makes that harmless.
    covariance = np.diag(np.array([4.0, 9.0, -1.7e-27, -8.0e-28]))

    assert coefficient_se(covariance, 1) == pytest.approx(3.0)
