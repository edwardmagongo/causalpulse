"""Combining several estimates of one quantity into one, by Rubin's rules.

Multiple imputation produces several complete datasets and therefore several
estimates of the same thing. Averaging their points is the easy half; the
uncertainty is the half that matters, because the spread BETWEEN imputations
is itself information about how much the missing data cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PooledEstimate:
    """One estimate pooled from several, with its variance decomposed.

    `within_variance` is the average sampling variance the analyses reported
    for themselves; `between_variance` is the variance of their point
    estimates around their own mean.

    `proportion_variance_missing` is Rubin's lambda: `(1 + 1/m) * B / T`, the
    proportion of total variance attributable to the missing data. It is zero
    exactly when every imputation agreed, and it is the quantity the
    Barnard-Rubin degrees-of-freedom formula consumes below. It is *not* the
    fraction of missing information, despite the similar name in some other
    implementations (e.g. `statsmodels`' `MICEResults.frac_miss_info`, which
    reports lambda under the FMI name) -- see `fraction_missing_information`.

    `fraction_missing_information` is the actual fraction of missing
    information (FMI) in Rubin's sense, which carries an additional
    degrees-of-freedom correction on top of lambda:
    `r = lambda / (1 - lambda)`, `fmi = (r + 2 / (nu + 3)) / (r + 1)`, where
    `nu` is `degrees_of_freedom` below. On non-trivial examples the two
    quantities differ materially. Cross-checked against the `mice` R
    package's `pool.scalar.R`, which reports `lambda` and `fmi` as two
    separate diagnostics. At the edges: `lambda == 0.0` (no missing
    information at all) gives `fraction_missing_information == 0.0`, and
    `lambda == 1.0` gives `fraction_missing_information == 1.0` (the limit of
    the formula above as `r -> infinity`), rather than raising or producing
    NaN.
    """

    point: float
    se: float
    within_variance: float
    between_variance: float
    proportion_variance_missing: float
    fraction_missing_information: float
    degrees_of_freedom: float


def pool_rubin(
    points: Sequence[float],
    standard_errors: Sequence[float],
    complete_data_df: float,
) -> PooledEstimate:
    """Pool `m` estimates by Rubin's rules, with Barnard-Rubin degrees of freedom.

    `complete_data_df` is the degrees of freedom the analysis would have had
    with no missing data. It matters little -- moving it from 1994 to 1e12 on
    the worked example moves the 97.5% t quantile by 0.07% -- but it is what
    keeps the reported df below the complete-data df, which the 1987 formula
    does not guarantee and which is wrong on small samples.

    Special case: if every within-imputation standard error is exactly zero
    but the point estimates differ, all of the total variance is
    between-imputation, so `proportion_variance_missing` (lambda) is exactly
    1.0. The finite-sample Barnard-Rubin adjustment is undefined there (its
    `(1 - lambda)` factor would zero out `df_observed` and divide by zero),
    so this function falls back to the Rubin (1987) degrees of freedom,
    `(m - 1) / lambda**2`, which is finite and well-defined at `lambda == 1`.

    References: Rubin (1987), Barnard and Rubin (1999), Biometrika 86, 948-955.
    """
    if len(points) != len(standard_errors):
        raise ValueError(
            f"points and standard_errors must be the same length; got "
            f"{len(points)} and {len(standard_errors)}"
        )
    m = len(points)
    if m < 2:
        raise ValueError(
            "pooling needs at least two imputations; a single imputation has "
            "no between-imputation variance, and reporting one as though it "
            "did would invent precision"
        )

    standard_error_values = np.asarray(standard_errors, dtype=float)
    if np.any(standard_error_values < 0):
        raise ValueError("standard_errors must be non-negative")

    estimates = np.asarray(points, dtype=float)
    variances = standard_error_values**2

    point = float(estimates.mean())
    within = float(variances.mean())
    between = float(estimates.var(ddof=1))
    total = within + (1.0 + 1.0 / m) * between

    if total <= 0.0:
        # Every analysis reported a zero standard error and every imputation
        # agreed. Degenerate but not an error: a point interval is the honest
        # report, the same judgement `wald_interval` makes.
        return PooledEstimate(
            point=point,
            se=0.0,
            within_variance=within,
            between_variance=between,
            proportion_variance_missing=0.0,
            fraction_missing_information=0.0,
            degrees_of_freedom=complete_data_df,
        )

    lam = (1.0 + 1.0 / m) * between / total

    if lam <= 0.0:
        # Reachable whenever nothing was missing, so every imputation returned
        # the same estimate. `(m - 1) / lam**2` would divide by zero.
        degrees_of_freedom = complete_data_df
    elif lam >= 1.0:
        # Every within-imputation standard error is exactly zero but the
        # point estimates differ: all variance is between-imputation, so
        # lambda is exactly 1.0. `df_observed`'s `(1 - lam)` factor would be
        # zero, making `1.0 / df_observed` divide by zero. Fall back to the
        # Rubin (1987) degrees of freedom, finite and well-defined here.
        degrees_of_freedom = (m - 1) / lam**2
    else:
        df_old = (m - 1) / lam**2
        df_observed = (
            ((complete_data_df + 1.0) / (complete_data_df + 3.0))
            * complete_data_df
            * (1.0 - lam)
        )
        degrees_of_freedom = 1.0 / (1.0 / df_old + 1.0 / df_observed)

    if lam <= 0.0:
        fmi = 0.0
    elif lam >= 1.0:
        fmi = 1.0
    else:
        r = lam / (1.0 - lam)
        fmi = (r + 2.0 / (degrees_of_freedom + 3.0)) / (r + 1.0)

    return PooledEstimate(
        point=point,
        se=float(np.sqrt(total)),
        within_variance=within,
        between_variance=between,
        proportion_variance_missing=float(lam),
        fraction_missing_information=float(fmi),
        degrees_of_freedom=float(degrees_of_freedom),
    )
