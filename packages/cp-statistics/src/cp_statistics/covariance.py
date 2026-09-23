"""Ordinary least squares with heteroskedasticity- and cluster-robust covariance.

Sandwich estimators all share the form `(X'X)^-1 M (X'X)^-1`; the estimators
differ only in how the meat matrix `M` is built. Keeping that structure explicit
makes the relationship between HC0 and the cluster-robust form legible, and it
is what the singleton-cluster equivalence test exercises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg as scipy_linalg

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

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def k(self) -> int:
        return int(self.X.shape[1])


def _lstsq_via_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares via QR with column pivoting (LAPACK `gelsy`), no SVD.

    Fallback for the rare `LinAlgError` `np.linalg.lstsq` raises on this
    platform. Found reproducing the benchmark study's P6 (binary panel)
    scenario: a bootstrap resample's event-study design, 1500x110, genuinely
    full column rank (`np.linalg.matrix_rank` == 110) and well conditioned
    (`cond` ~= 96), made `np.linalg.lstsq`'s SVD-based `gelsd` driver fail to
    converge -- reproduced deterministically (same cell, same seed), not a
    one-off. Both numpy and scipy resolve to the same underlying LAPACK on
    this machine (`numpy.show_config()`/`scipy.show_config()` both report
    Apple's Accelerate), so `scipy.linalg.lstsq`'s own default `gelsd` driver
    is not a safe fallback either: on the same matrix it does not raise, but
    its reported rank (1500, impossible for 110 columns) shows it returned a
    wrong answer rather than a genuine success -- the same bug, undetected
    rather than avoided. `gelsy` sidesteps it by construction (QR, not SVD),
    and its answer was cross-checked on the actual failing matrix against
    `gelss` (the older, non-divide-and-conquer SVD driver, and so a
    numerically independent method): the two agreed to five decimal places on
    the residual, which is what justifies trusting `gelsy` here, rather than
    trusting any driver that merely "did not raise".

    For a matrix that is *not* full column rank, `gelsy`'s solution is not
    guaranteed to be `lstsq`'s minimum-norm one -- the two agree only because
    the reproduced case is full rank. That compound case (rank-deficient and
    also SVD-non-convergent) has not been observed; if it occurs, this still
    returns a genuine least-squares fit, only possibly not the minimum-norm
    one `bootstrap_panel_effect`'s duplicated-unit note relies on.
    """
    beta, _, _, _ = scipy_linalg.lstsq(X, y, lapack_driver="gelsy")
    beta = np.asarray(beta)
    if not np.all(np.isfinite(beta)):
        raise np.linalg.LinAlgError(
            "QR fallback (gelsy) also failed to produce a finite solution"
        )
    return beta


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

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        beta = _lstsq_via_qr(X, y)
    residuals = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    # `optimize=True` is not a micro-optimisation here. Without it einsum
    # evaluates this three-operand contraction as one naive O(n*k^2) C loop
    # with no BLAS; with it, it contracts pairwise (X @ xtx_inv, then a
    # row-wise dot) and the same arithmetic goes through matrix multiply.
    # Measured at the panel shape this project's TWFE/EventStudy fits produce
    # (n=3200, k=408): 617ms unoptimised against 4.1ms optimised, which was
    # 93% of the cost of an EventStudy fit and by far the largest single cost
    # in the test suite. The two agree to 3.3e-16 absolute / 2.5e-15 relative
    # -- summation order, nothing more.
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X, optimize=True)
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

    labels = np.unique(clusters)
    # A single cluster leaves the meat matrix rank one: the result is degenerate
    # whether or not the correction is applied, so the guard belongs outside it.
    n_clusters = int(labels.size)
    if n_clusters <= 1:
        raise ValueError("cluster-robust covariance requires at least two clusters")

    meat = np.zeros((fit.k, fit.k))
    for label in labels:
        mask = clusters == label
        score = fit.X[mask].T @ fit.residuals[mask]
        meat += np.outer(score, score)

    covariance = _sandwich(fit, meat)
    if small_sample_correction:
        correction = (n_clusters / (n_clusters - 1)) * ((fit.n - 1) / (fit.n - fit.k))
        covariance = covariance * correction
    return covariance
