"""Rubin's rules, against a worked example and three identities."""

import numpy as np
import pytest
from cp_statistics.pooling import pool_rubin

POINTS = [0.210, 0.198, 0.225, 0.203, 0.216]
SES = [0.031, 0.030, 0.032, 0.031, 0.030]


def test_the_worked_example():
    # Verified during design and cross-checked against
    # statsmodels.imputation.mice.MICE.combine's own arithmetic, which
    # computes cov_within + (1 + 1/m) * cov_between with np.cov's ddof=1 and
    # reports (1 + 1/m) * B / T under the FMI name -- that quantity is Rubin's
    # lambda (proportion_variance_missing) below, not the true FMI.
    #
    # True FMI cross-checked against the mice R package's pool.scalar.R,
    # which reports lambda and fmi as two separate diagnostics:
    #   r = lambda / (1 - lambda) = 0.125290 / 0.874710 = 0.143234...
    #   fmi = (r + 2 / (nu + 3)) / (r + 1)
    #       = (0.143234 + 2 / 225.3049) / 1.143234 = 0.133055
    pooled = pool_rubin(POINTS, SES, complete_data_df=1994.0)
    assert pooled.point == pytest.approx(0.210400, abs=1e-6)
    assert pooled.within_variance == pytest.approx(0.00094920, abs=1e-8)
    assert pooled.between_variance == pytest.approx(0.00011330, abs=1e-8)
    assert pooled.se**2 == pytest.approx(0.00108516, abs=1e-8)
    assert pooled.se == pytest.approx(0.032942, abs=1e-6)
    assert pooled.proportion_variance_missing == pytest.approx(0.125290, abs=1e-6)
    assert pooled.fraction_missing_information == pytest.approx(0.133055, abs=1e-6)
    assert pooled.degrees_of_freedom == pytest.approx(222.3049, abs=1e-4)


def test_the_degrees_of_freedom_never_exceed_the_complete_data_degrees_of_freedom():
    # An identity, not a tolerance. Barnard-Rubin adjusts DOWNWARD for a
    # finite complete-data sample; a value above the complete-data df means
    # the formula is wrong.
    rng = np.random.default_rng(0)
    for _ in range(500):
        m = int(rng.integers(2, 40))
        points = rng.normal(0.2, 0.02, size=m)
        ses = rng.uniform(0.01, 0.05, size=m)
        nu_com = float(rng.uniform(5, 5000))
        pooled = pool_rubin(points, ses, complete_data_df=nu_com)
        assert pooled.degrees_of_freedom <= nu_com


def test_it_recovers_the_1987_degrees_of_freedom_as_the_complete_data_sample_grows():
    # The limit the Barnard-Rubin adjustment is defined to recover:
    # (m - 1) / lambda**2 as nu_com -> inf.
    pooled = pool_rubin(POINTS, SES, complete_data_df=1e12)
    m = len(POINTS)
    expected = (m - 1) / pooled.proportion_variance_missing**2
    assert pooled.degrees_of_freedom == pytest.approx(expected, rel=1e-6)


def test_identical_estimates_do_not_divide_by_zero():
    # Reachable, not hypothetical: every imputation returns the same estimate
    # whenever nothing was missing. `(m - 1) / lam**2` would divide by zero.
    pooled = pool_rubin([0.2] * 5, [0.03] * 5, complete_data_df=1994.0)
    assert pooled.between_variance == 0.0
    assert pooled.proportion_variance_missing == 0.0
    assert pooled.fraction_missing_information == 0.0
    assert pooled.degrees_of_freedom == 1994.0
    assert pooled.se == pytest.approx(0.03, abs=1e-12)


def test_negative_standard_error_is_refused():
    # variances = standard_errors ** 2 is never negative, so a guard on
    # variances can never fire; a negative standard error must be caught
    # before squaring, not after.
    with pytest.raises(ValueError, match="non-negative"):
        pool_rubin([0.2, 0.3], [0.03, -0.01], complete_data_df=1994.0)


def test_all_zero_standard_errors_with_differing_points_do_not_divide_by_zero():
    # Every within-imputation standard error is exactly zero but the points
    # differ, so lambda is exactly 1.0 and df_observed's (1 - lambda) factor
    # is exactly zero. This must fall back to the finite Rubin (1987) df
    # rather than raising ZeroDivisionError or producing NaN/inf.
    points = [0.10, 0.20, 0.30, 0.40, 0.50]
    ses = [0.0, 0.0, 0.0, 0.0, 0.0]
    pooled = pool_rubin(points, ses, complete_data_df=1994.0)
    m = len(points)
    assert pooled.within_variance == 0.0
    assert pooled.between_variance > 0.0
    assert pooled.proportion_variance_missing == pytest.approx(1.0)
    assert pooled.fraction_missing_information == pytest.approx(1.0)
    assert np.isfinite(pooled.degrees_of_freedom)
    assert pooled.degrees_of_freedom == pytest.approx(m - 1)
    assert np.isfinite(pooled.se)


def test_a_single_imputation_is_refused():
    # m = 1 has no between-imputation variance at all -- var(ddof=1) on one
    # observation is undefined -- so single imputation must not be pooled. It
    # reports the analysis model's own standard error instead, which is
    # exactly the understatement the Task 10 harness measures.
    with pytest.raises(ValueError, match="at least two"):
        pool_rubin([0.2], [0.03], complete_data_df=1994.0)


def test_mismatched_input_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        pool_rubin([0.2, 0.3], [0.03], complete_data_df=1994.0)
