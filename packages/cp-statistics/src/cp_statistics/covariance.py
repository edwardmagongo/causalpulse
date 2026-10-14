"""Ordinary least squares with heteroskedasticity- and cluster-robust covariance.

Sandwich estimators all share the form `(X'X)^-1 M (X'X)^-1`; the estimators
differ only in how the meat matrix `M` is built. Keeping that structure explicit
makes the relationship between HC0 and the cluster-robust form legible, and it
is what the singleton-cluster equivalence test exercises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cp_statistics.linalg_safety import safe_lstsq, safe_symmetric_pinv

_HC_KINDS = frozenset({"HC0", "HC1", "HC2"})


# eq=False: this holds array-valued fields, so a generated __eq__ would raise
# on the ambiguous truth value of an array rather than compare. Identity
# equality is the honest semantics for a container of measurements.
@dataclass(frozen=True, eq=False)
class OLSFit:
    """The pieces of a least-squares fit needed by every sandwich estimator."""

    beta: np.ndarray
    residuals: np.ndarray
    xtx_inv: np.ndarray
    leverage: np.ndarray
    X: np.ndarray
    # Each column's largest absolute value (1 for a column of zeros), the divisor of the fit.
    scale: np.ndarray
    # The inverse of X'X for the scaled design; `xtx_inv` is this in the original units.
    scaled_xtx_inv: np.ndarray

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def k(self) -> int:
        return int(self.X.shape[1])


def ols_fit(X: np.ndarray, y: np.ndarray) -> OLSFit:
    """Fit OLS via least squares, retaining the quantities sandwiches need.

    Each column is scaled to a largest absolute value of 1 for the fit; every
    field of the result is in the original units except `scale` and
    `scaled_xtx_inv`, which describe the scaled design.

    Both linear-algebra calls go through `cp_statistics.linalg_safety`'s safe
    wrappers rather than calling `np.linalg.lstsq`/`np.linalg.pinv` directly --
    see that module's docstring for why: both raised `LinAlgError: SVD did not
    converge` on this platform, on two different real cells the benchmark
    study generated, on two well-posed (full-rank) matrices.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} entries"
        )

    # Each column divided by its largest absolute value, so that the fit does
    # not depend on the units a column is in. `pinv` cuts singular values
    # below 1e-15 of the largest, and `X'X` squares the spread of the columns'
    # scales: measured on LaLonde, 1975 earnings (re75) in cents instead of dollars gave
    # `lin_regression` an interval of [-4,480, -2,713] for [-9,360, 2,167].
    # The largest absolute value, not the Euclidean norm, because the norm
    # squares the values and overflows for exactly the large columns this is
    # for. A column of zeros keeps scale 1.
    largest = np.max(np.abs(X), axis=0) if X.shape[0] else np.ones(X.shape[1])
    scale = np.where(largest > 0, largest, 1.0)
    Xs = X / scale
    beta = safe_lstsq(Xs, y) / scale
    residuals = y - X @ beta
    scaled_xtx_inv = safe_symmetric_pinv(Xs.T @ Xs)
    xtx_inv = scaled_xtx_inv / np.outer(scale, scale)
    # `optimize=True` is not a micro-optimisation here. Without it einsum
    # evaluates this three-operand contraction as one naive O(n*k^2) C loop
    # with no BLAS; with it, it contracts pairwise (X @ xtx_inv, then a
    # row-wise dot) and the same arithmetic goes through matrix multiply.
    # Measured at the panel shape this project's TWFE/EventStudy fits produce
    # (n=3200, k=408): 617ms unoptimised against 4.1ms optimised, which was
    # 93% of the cost of an EventStudy fit and by far the largest single cost
    # in the test suite. The two agree to 3.3e-16 absolute / 2.5e-15 relative
    # -- summation order, nothing more.
    leverage = np.einsum("ij,jk,ik->i", Xs, scaled_xtx_inv, Xs, optimize=True)
    return OLSFit(
        beta=beta,
        residuals=residuals,
        xtx_inv=xtx_inv,
        leverage=leverage,
        X=X,
        scale=scale,
        scaled_xtx_inv=scaled_xtx_inv,
    )


def _sandwich(fit: OLSFit, meat: np.ndarray) -> np.ndarray:
    # `meat` is built on the scaled design; the covariance is unscaled last,
    # so a large column never enters a sum of squares in its own units.
    covariance = fit.scaled_xtx_inv @ meat @ fit.scaled_xtx_inv
    covariance = covariance / np.outer(fit.scale, fit.scale)
    return np.asarray((covariance + covariance.T) / 2.0)


def robust_covariance(fit: OLSFit, kind: str = "HC1") -> np.ndarray:
    """Heteroskedasticity-consistent covariance.

    HC0 is the asymptotic form; HC1 applies the degrees-of-freedom correction;
    HC2 divides by the leverage complement, which behaves better under
    influential observations.
    """
    if kind not in _HC_KINDS:
        raise ValueError(f"kind must be one of {sorted(_HC_KINDS)}, got {kind!r}")

    squared = fit.residuals**2
    if kind == "HC2":
        squared = squared / (1.0 - fit.leverage)

    scaled = fit.X / fit.scale
    meat = (scaled * squared[:, None]).T @ scaled
    covariance = _sandwich(fit, meat)
    if kind == "HC1":
        covariance = covariance * (fit.n / (fit.n - fit.k))
    return covariance


def cluster_robust_covariance(
    fit: OLSFit,
    clusters: np.ndarray,
    small_sample_correction: bool = True,
) -> np.ndarray:
    """Cluster-robust (CR0/CR1) covariance, allowing arbitrary within-cluster correlation.

    Ignoring clustering when errors are correlated within groups understates
    standard errors, often severely. The correction depends on the number of
    clusters, not the number of observations.
    """
    clusters = np.asarray(clusters)
    if clusters.shape[0] != fit.n:
        raise ValueError(
            f"clusters has length {clusters.shape[0]} but the fit has {fit.n} observations"
        )

    labels = np.unique(clusters)
    # A single cluster leaves the meat matrix rank one: the result is degenerate
    # whether or not the correction is applied, so the guard belongs outside it.
    n_clusters = int(labels.size)
    if n_clusters <= 1:
        raise ValueError("cluster-robust covariance requires at least two clusters")

    meat = np.zeros((fit.k, fit.k))
    for label in labels:
        mask = clusters == label
        score = (fit.X[mask] / fit.scale).T @ fit.residuals[mask]
        meat += np.outer(score, score)

    covariance = _sandwich(fit, meat)
    if small_sample_correction:
        correction = (n_clusters / (n_clusters - 1)) * ((fit.n - 1) / (fit.n - fit.k))
        covariance = covariance * correction
    return covariance
