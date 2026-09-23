"""The safe `lstsq`/`pinv` wrappers, and the fallback branch each one adds.

Found running the benchmark study: `np.linalg.lstsq` and `np.linalg.pinv` each
raised `LinAlgError: SVD did not converge` on this platform, on two different
real cells, neither a rank-deficiency or a bad-model problem -- both captured
matrices were full rank, and the Gram-matrix case was confirmed positive
definite. See `cp_statistics.linalg_safety`'s module docstring for the full
diagnosis and the independent-method cross-validation that justifies each
fallback. These tests pin the fallback branches deterministically and
portably, by forcing the primary path to raise, rather than depending on it
actually failing on whatever machine runs this suite. The real, historically
failing cases are pinned separately, end to end, in
`tests/causal/test_event_study.py`.
"""

import cp_statistics.linalg_safety as linalg_safety_module
import numpy as np
import pytest
from cp_statistics.linalg_safety import safe_lstsq, safe_symmetric_pinv


def _always_raises_lstsq(X, y, rcond=None):
    raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")


def _always_raises_pinv(A, *args, **kwargs):
    raise np.linalg.LinAlgError("SVD did not converge")


# -- safe_lstsq ---------------------------------------------------------------


def test_safe_lstsq_matches_the_primary_path_when_it_works():
    x = np.linspace(0, 10, 50)
    X = np.column_stack([np.ones(50), x])
    y = 3.0 + 4.0 * x

    beta = safe_lstsq(X, y)

    assert beta[0] == pytest.approx(3.0, abs=1e-9)
    assert beta[1] == pytest.approx(4.0, abs=1e-9)


def test_safe_lstsq_falls_back_to_qr_when_lstsq_raises(monkeypatch):
    monkeypatch.setattr(np.linalg, "lstsq", _always_raises_lstsq)
    x = np.linspace(0, 10, 50)
    X = np.column_stack([np.ones(50), x])
    y = 3.0 + 4.0 * x

    beta = safe_lstsq(X, y)

    # The fallback recovers the same exact-fit coefficients the primary path
    # does -- the branch is invisible to a caller when it works correctly.
    assert beta[0] == pytest.approx(3.0, abs=1e-9)
    assert beta[1] == pytest.approx(4.0, abs=1e-9)


def test_safe_lstsq_raises_if_the_qr_fallback_is_also_non_finite(monkeypatch):
    def returns_garbage(X, y, lapack_driver=None):
        return np.full(X.shape[1], np.nan), None, None, None

    monkeypatch.setattr(np.linalg, "lstsq", _always_raises_lstsq)
    monkeypatch.setattr(linalg_safety_module.scipy_linalg, "lstsq", returns_garbage)
    x = np.linspace(0, 10, 20)
    X = np.column_stack([np.ones(20), x])
    y = 1.0 + 2.0 * x

    with pytest.raises(np.linalg.LinAlgError, match="QR fallback"):
        safe_lstsq(X, y)


# -- safe_symmetric_pinv -------------------------------------------------------


def _spd_matrix(n: int = 10, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)  # symmetric positive definite, well conditioned


def test_safe_symmetric_pinv_matches_the_primary_path_when_it_works():
    A = _spd_matrix()

    result = safe_symmetric_pinv(A)

    np.testing.assert_allclose(result, np.linalg.pinv(A), rtol=1e-10)
    # Moore-Penrose property: A @ pinv(A) @ A == A.
    np.testing.assert_allclose(A @ result @ A, A, atol=1e-8)


def test_safe_symmetric_pinv_falls_back_to_eigendecomposition_when_pinv_raises(
    monkeypatch,
):
    A = _spd_matrix()
    expected = np.linalg.pinv(A)  # computed before pinv is patched to always fail

    monkeypatch.setattr(np.linalg, "pinv", _always_raises_pinv)
    result = safe_symmetric_pinv(A)

    # The fallback still satisfies the defining property of a pseudo-inverse,
    # and agrees with what pinv would have given on the unpatched matrix.
    np.testing.assert_allclose(A @ result @ A, A, atol=1e-8)
    np.testing.assert_allclose(result, expected, rtol=1e-8)


def test_safe_symmetric_pinv_raises_if_the_eigendecomposition_fallback_is_also_non_finite(
    monkeypatch,
):
    def returns_garbage(A, *args, **kwargs):
        return np.full(A.shape, np.nan)

    monkeypatch.setattr(np.linalg, "pinv", _always_raises_pinv)
    monkeypatch.setattr(linalg_safety_module.scipy_linalg, "pinvh", returns_garbage)
    A = _spd_matrix()

    with pytest.raises(np.linalg.LinAlgError, match="Eigendecomposition fallback"):
        safe_symmetric_pinv(A)
