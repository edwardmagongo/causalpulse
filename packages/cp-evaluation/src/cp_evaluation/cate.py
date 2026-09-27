"""Per-unit CATE recovery, reduced to three scalars per fit.

A CATE cell's estimator reports one effect per unit; the result frame holds one
row per fit. The reduction happens here, once per fitted row, so the frame,
the store and every reader stay scalar (design section 2, approach A).

The truth is always the response and never the regressor: the calibration slope
is the OLS slope of the true CATE on the estimated one. Regressing the other way
would fit a model to an estimator's own output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CATE_METRIC_COLUMNS: tuple[str, ...] = (
    "cate_correlation",
    "cate_rmse",
    "cate_calibration_slope",
)


@dataclass(frozen=True, eq=False)
class CateTruth:
    """A CATE cell's realised per-unit truth, and whether it is declared to vary.

    `heterogeneous` comes from `CATEScenarioSpec.heterogeneous`, not from the
    values: `NULL`'s realised truth has a standard deviation of ~3e-17
    (floating-point rounding of a constant), so no variance test separates it
    from a scenario whose truth genuinely varies without a tolerance chosen to
    make one scenario pass. eq=False because an array does not compare
    structurally.
    """

    values: np.ndarray
    heterogeneous: bool


def cate_row_metrics(estimated: object, truth: CateTruth) -> dict[str, float]:
    """Correlation, RMSE and calibration slope of one fit's per-unit CATE.

    Raises on an estimate of the wrong shape or with a non-finite entry: either
    is a defect in the estimator, and a NaN metric would hide it.

    RMSE is always computed. Correlation and slope are NaN when the truth is
    declared constant (they are undefined against a truth with no spread) or
    when the estimate is constant (a slope over zero variance is undefined).
    Both are decided before any numpy call that would warn.
    """
    e = np.asarray(estimated, dtype=float)
    t = np.asarray(truth.values, dtype=float)
    if e.ndim != 1 or e.shape != t.shape:
        raise ValueError(
            f"per_unit_cate has shape {e.shape}; the cell's truth has shape {t.shape}. "
            "A per-unit CATE must be one value per row of the dataset, in row order."
        )
    if not np.all(np.isfinite(e)):
        raise ValueError(
            f"per_unit_cate carries {int(np.sum(~np.isfinite(e)))} non-finite entries; "
            "scoring it would hide an estimator defect behind a NaN metric."
        )

    nan = float("nan")
    rmse = float(np.sqrt(np.mean((e - t) ** 2)))
    if not truth.heterogeneous:
        return {"cate_correlation": nan, "cate_rmse": rmse, "cate_calibration_slope": nan}

    e_centred = e - e.mean()
    t_centred = t - t.mean()
    var_e = float(np.mean(e_centred**2))
    var_t = float(np.mean(t_centred**2))
    if var_e == 0.0:
        return {"cate_correlation": nan, "cate_rmse": rmse, "cate_calibration_slope": nan}
    cov = float(np.mean(e_centred * t_centred))
    correlation = nan if var_t == 0.0 else cov / float(np.sqrt(var_e * var_t))
    return {
        "cate_correlation": correlation,
        "cate_rmse": rmse,
        "cate_calibration_slope": cov / var_e,
    }
