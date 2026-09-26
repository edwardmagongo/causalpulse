"""Constructed result frames for the analysis tests.

Every row is built to a stated property (a hit rate, a bias, a runtime) so a test
can say what a pooled figure must be without running a fit. The interval on a
hitting row is centred on the truth and on a missing row lies wholly above it, so
coverage and the false-positive rate are exactly the requested fractions.
"""

import numpy as np
import pandas as pd
from cp_evaluation.grid import AXIS_BASELINES
from cp_evaluation.run import ROW_COLUMNS


def rows(
    method: str,
    scenario: str,
    n: int,
    *,
    truth: float = 1.0,
    bias: float = 0.0,
    hit_fraction: float = 1.0,
    half_width: float = 0.5,
    n_units: int = 500,
    runtime: float = 1.0,
    status: str = "fitted",
    panel: bool = False,
    donor_pool: bool = False,
    n_periods: float = 7,
    noise: float = 0.01,
    axes: dict[str, float] | None = None,
    ci_basis: str = "analytic",
) -> pd.DataFrame:
    """`n` rows for one (method, scenario, n_units); `hit_fraction` of them cover the truth.

    `noise` is the alternating error around `truth + bias`, so the estimator's own spread
    (and hence the MCSE of its bias) is set by it.
    """
    hits = round(n * hit_fraction)
    records = []
    for i in range(n):
        jitter = noise * (-1) ** i
        point = truth + bias + jitter
        centre = truth if i < hits else truth + 10.0
        record: dict[str, object] = dict.fromkeys(ROW_COLUMNS, np.nan)
        shape = "donor_pool" if donor_pool else ("panel" if panel else "cross_section")
        record.update(
            shape=shape,
            scenario=scenario,
            n_units=n_units,
            n_periods=n_periods if (panel or donor_pool) else np.nan,
            seed=i,
            method=method,
            status=status,
            reason="" if status == "fitted" else "not fitted",
            estimand="ATT" if (panel or donor_pool) else "ATE",
        )
        if not panel and not donor_pool:
            record.update(AXIS_BASELINES)
            record["realised_treated_share"] = 0.3
        if status == "fitted":
            record.update(
                truth=truth,
                point=point,
                se=half_width / 1.96,
                ci_lower=centre - half_width,
                ci_upper=centre + half_width,
                ci_basis=ci_basis,
                bootstrap_ci_lower=centre - half_width,
                bootstrap_ci_upper=centre + half_width,
                runtime_seconds=runtime,
            )
        for name, value in (axes or {}).items():
            record[name] = value
        records.append(record)
    return pd.DataFrame(records, columns=list(ROW_COLUMNS))


def frame(*parts: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True)
