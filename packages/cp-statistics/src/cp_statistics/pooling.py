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
    estimates around their own mean. `fraction_missing_information` is the
    share of the total attributable to the missing data, and is zero exactly
    when every imputation agreed.
    """

    point: float
    se: float
    within_variance: float
    between_variance: float
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

    estimates = np.asarray(points, dtype=float)
    variances = np.asarray(standard_errors, dtype=float) ** 2
    if np.any(variances < 0):
        raise ValueError("standard_errors must be non-negative")

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
            fraction_missing_information=0.0,
            degrees_of_freedom=complete_data_df,
        )

    fraction = (1.0 + 1.0 / m) * between / total

    if fraction <= 0.0:
        # Reachable whenever nothing was missing, so every imputation returned
        # the same estimate. `(m - 1) / fraction**2` would divide by zero.
        degrees_of_freedom = complete_data_df
    else:
        df_old = (m - 1) / fraction**2
        df_observed = (
            ((complete_data_df + 1.0) / (complete_data_df + 3.0))
            * complete_data_df
            * (1.0 - fraction)
        )
        degrees_of_freedom = 1.0 / (1.0 / df_old + 1.0 / df_observed)

    return PooledEstimate(
        point=point,
        se=float(np.sqrt(total)),
        within_variance=within,
        between_variance=between,
        fraction_missing_information=float(fraction),
        degrees_of_freedom=float(degrees_of_freedom),
    )
