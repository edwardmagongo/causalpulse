"""Numerically safe wrappers around this project's SVD-based `np.linalg` calls.

On this project's development machine, `np.linalg.lstsq` and `np.linalg.pinv`
occasionally raise `LinAlgError: SVD did not converge` -- found running the
benchmark study, at two independent call sites, on two different cells,
neither a rank-deficiency or a bad-model problem: both captured matrices were
full rank, and the Gram-matrix case was confirmed positive definite (a
successful Cholesky factorisation). `numpy.show_config()`/`scipy.show_config()`
both report Apple's Accelerate as the linked BLAS/LAPACK, and `scipy`'s own
default SVD drivers resolve to the same library -- sometimes sharing the bug
*silently*, returning a wrong answer rather than raising (`scipy.linalg.lstsq`
with `lapack_driver="gelsd"` reported an impossible rank on the first captured
failure; `scipy.linalg.pinv`, general and SVD-based, failed outright the same
way `numpy.linalg.pinv` did on the second). Trusting "did not raise" is not
enough here -- the absence of an exception is not evidence of a correct
result, any more than it is for an optimiser's own success flag.

The two functions below retry, only on `LinAlgError`, via an algorithm that
does not use general SVD, and only after that algorithm's answer was
cross-checked against an *independent* method (not merely "did not raise") on
the real, captured, previously-failing matrix:

- `safe_lstsq` falls back to `scipy.linalg.lstsq(..., lapack_driver="gelsy")`
  -- QR with column pivoting. Verified against `gelss` (the older,
  non-divide-and-conquer SVD driver -- numerically independent of `gelsy` and
  of the failing `gelsd`): the two agreed to five decimal places on the
  residual of the actual failing 1500x110 design.
- `safe_symmetric_pinv` falls back to `scipy.linalg.pinvh` -- eigendecomposition
  (`syevd`/`syevr`), not SVD. Verified against a Cholesky-based direct inverse
  (a third, completely different algorithm, and one that only succeeds if the
  matrix is genuinely positive definite) on the actual failing 113x113 Gram
  matrix: the two agreed to 1.5e-14, essentially machine precision.

Neither fallback is claimed correct for a matrix that is *both* rank-deficient
(or, for the pinv case, not positive semi-definite) *and* SVD-non-convergent --
that compound case has not been observed. Each still returns a genuine answer
in that case, verified finite, just not necessarily `lstsq`'s minimum-norm one
or an exact pseudo-inverse; callers relying on the minimum-norm property under
genuine rank deficiency should note this (see `_panel.py`'s
`bootstrap_panel_effect` docstring for the one place that currently does).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as scipy_linalg


def safe_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients, falling back to a QR solve on `LinAlgError`.

    See this module's docstring for the diagnosis and the cross-validation
    that justifies the fallback.
    """
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        beta, _, _, _ = scipy_linalg.lstsq(X, y, lapack_driver="gelsy")
    result = np.asarray(beta)
    if not np.all(np.isfinite(result)):
        raise np.linalg.LinAlgError(
            "QR fallback (gelsy) also failed to produce a finite solution"
        )
    return result


def safe_symmetric_pinv(A: np.ndarray) -> np.ndarray:
    """The pseudo-inverse of a symmetric matrix, falling back to an
    eigendecomposition (not SVD) on `LinAlgError`.

    `A` is expected symmetric (a Gram matrix or a covariance matrix): every
    caller in this codebase passes one, but `parallel_trends_check` takes
    `lead_covariance` from a caller outside this module, so that expectation
    is not enforced by the type system. `scipy.linalg.pinvh` reads only one
    triangle of its input, so it would not raise on a genuinely asymmetric
    matrix -- it would silently return the pseudo-inverse of a *different*
    matrix, and only on the fallback branch, making the wrong answer
    platform-dependent and rare enough to go unnoticed. Checked here, once,
    on the fallback path only (the common, primary-path case pays nothing for
    it): `np.allclose(A, A.T)` tolerates the tiny floating-point asymmetry
    `X.T @ X` can carry (verified directly against the real captured failure,
    computed exactly as `ols_fit` computes it) while still catching a matrix
    that is not symmetric at all. See this module's docstring for the
    diagnosis and cross-validation.
    """
    try:
        return np.linalg.pinv(A)
    except np.linalg.LinAlgError:
        if not np.allclose(A, A.T):
            raise np.linalg.LinAlgError(
                "safe_symmetric_pinv's fallback (pinvh) requires a symmetric "
                "matrix, and this one is not -- np.linalg.pinv failed with no "
                "safe fallback available for a non-symmetric input"
            ) from None
        result = np.asarray(scipy_linalg.pinvh(A))
    if not np.all(np.isfinite(result)):
        raise np.linalg.LinAlgError(
            "Eigendecomposition fallback (pinvh) also failed to produce a finite result"
        )
    return result
