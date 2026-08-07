"""Ordinary least squares with heteroskedasticity- and cluster-robust covariance.

Sandwich estimators all share the form `(X'X)^-1 M (X'X)^-1`; the estimators
differ only in how the meat matrix `M` is built. Keeping that structure explicit
makes the relationship between HC0 and the cluster-robust form legible, and it
is what the singleton-cluster equivalence test exercises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_HC_KINDS = frozenset({"HC0", "HC1", "HC2"})


@dataclass(frozen=True)
class OLSFit:
    """The pieces of a least-squares fit needed by every sandwich estimator."""

    beta: np.ndarray
    residuals: np.ndarray
    xtx_inv: np.ndarray
    leverage: np.ndarray
    X: np.ndarray

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def k(self) -> int:
        return int(self.X.shape[1])


def ols_fit(X: np.ndarray, y: np.ndarray) -> OLSFit:
    """Fit OLS via least squares, retaining the quantities sandwiches need."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} entries"
        )

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    return OLSFit(
        beta=beta, residuals=residuals, xtx_inv=xtx_inv, leverage=leverage, X=X
    )


def _sandwich(fit: OLSFit, meat: np.ndarray) -> np.ndarray:
    covariance = fit.xtx_inv @ meat @ fit.xtx_inv
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

    meat = (fit.X * squared[:, None]).T @ fit.X
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

    meat = np.zeros((fit.k, fit.k))
    for label in np.unique(clusters):
        mask = clusters == label
        score = fit.X[mask].T @ fit.residuals[mask]
        meat += np.outer(score, score)

    covariance = _sandwich(fit, meat)
    if small_sample_correction:
        n_clusters = int(np.unique(clusters).size)
        if n_clusters <= 1:
            raise ValueError("cluster-robust covariance requires at least two clusters")
        correction = (n_clusters / (n_clusters - 1)) * ((fit.n - 1) / (fit.n - fit.k))
        covariance = covariance * correction
    return covariance
