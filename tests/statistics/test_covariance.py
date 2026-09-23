import cp_statistics.covariance as covariance_module
import numpy as np
import pytest
from cp_statistics.covariance import (
    cluster_robust_covariance,
    ols_fit,
    robust_covariance,
)


def design(n: int = 200, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = 1.5 + 2.0 * x + rng.normal(size=n)
    return X, y


def test_ols_recovers_known_coefficients_without_noise():
    x = np.linspace(0, 10, 50)
    X = np.column_stack([np.ones(50), x])
    y = 3.0 + 4.0 * x

    fit = ols_fit(X, y)

    assert fit.beta[0] == pytest.approx(3.0, abs=1e-9)
    assert fit.beta[1] == pytest.approx(4.0, abs=1e-9)
    assert fit.n == 50
    assert fit.k == 2


def test_hc1_inflates_hc0_by_the_known_finite_sample_factor():
    X, y = design()
    fit = ols_fit(X, y)

    hc0 = robust_covariance(fit, kind="HC0")
    hc1 = robust_covariance(fit, kind="HC1")
    factor = fit.n / (fit.n - fit.k)

    np.testing.assert_allclose(hc1, hc0 * factor, rtol=1e-12)


def test_singleton_clusters_reduce_to_hc0():
    # With one observation per cluster the cluster-robust meat matrix is
    # algebraically identical to the HC0 meat matrix.
    X, y = design()
    fit = ols_fit(X, y)
    clusters = np.arange(fit.n)

    clustered = cluster_robust_covariance(fit, clusters, small_sample_correction=False)
    hc0 = robust_covariance(fit, kind="HC0")

    np.testing.assert_allclose(clustered, hc0, rtol=1e-10, atol=1e-14)


def test_within_cluster_correlation_inflates_standard_errors():
    # Errors constant within cluster: ignoring clustering understates uncertainty.
    rng = np.random.default_rng(11)
    n_clusters, per_cluster = 40, 25
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    x = np.repeat(rng.normal(size=n_clusters), per_cluster)
    shock = np.repeat(rng.normal(scale=2.0, size=n_clusters), per_cluster)
    n = n_clusters * per_cluster
    X = np.column_stack([np.ones(n), x])
    y = 1.0 + 0.5 * x + shock

    fit = ols_fit(X, y)
    naive_se = np.sqrt(np.diag(robust_covariance(fit, kind="HC1")))[1]
    clustered_se = np.sqrt(np.diag(cluster_robust_covariance(fit, clusters)))[1]

    assert clustered_se > 3.0 * naive_se


def test_cluster_covariance_is_invariant_to_cluster_label_values():
    X, y = design(n=120, seed=3)
    fit = ols_fit(X, y)
    clusters = np.repeat(np.arange(12), 10)
    relabelled = clusters * 1000 + 7

    original = cluster_robust_covariance(fit, clusters)
    renamed = cluster_robust_covariance(fit, relabelled)

    np.testing.assert_allclose(original, renamed, rtol=1e-12)


def test_covariance_matrices_are_symmetric_positive_semidefinite():
    X, y = design()
    fit = ols_fit(X, y)

    for matrix in (
        robust_covariance(fit, kind="HC0"),
        robust_covariance(fit, kind="HC2"),
        cluster_robust_covariance(fit, np.repeat(np.arange(20), 10)),
    ):
        np.testing.assert_allclose(matrix, matrix.T, rtol=1e-12)
        assert np.min(np.linalg.eigvalsh(matrix)) > -1e-12


def test_robust_covariance_rejects_unknown_kind():
    X, y = design()
    with pytest.raises(ValueError, match="kind"):
        robust_covariance(ols_fit(X, y), kind="HC9")


def test_cluster_covariance_rejects_length_mismatch():
    X, y = design(n=50)
    with pytest.raises(ValueError, match="length"):
        cluster_robust_covariance(ols_fit(X, y), np.arange(10))


@pytest.mark.parametrize("small_sample_correction", [True, False])
def test_cluster_covariance_rejects_a_single_cluster(small_sample_correction: bool):
    # One cluster leaves the meat matrix rank one, so the covariance is
    # degenerate regardless of the correction. Guarding only inside the
    # correction branch let the uncorrected call return that silently.
    X, y = design(n=50)
    fit = ols_fit(X, y)

    with pytest.raises(ValueError, match="at least two clusters"):
        cluster_robust_covariance(
            fit, np.zeros(fit.n, dtype=int),
            small_sample_correction=small_sample_correction,
        )


# -- the QR fallback for np.linalg.lstsq's rare LinAlgError -----------------
#
# Found running the benchmark study: a genuinely full-rank, well-conditioned
# bootstrap-resample design from scenario P6 made numpy's SVD-based lstsq
# (Apple's Accelerate LAPACK, `gelsd`) fail to converge. See `_lstsq_via_qr`'s
# docstring for the full diagnosis, including why scipy's own default driver
# is not a safe fallback either. These two tests pin the fallback branch
# itself, deterministically and portably (not depending on lstsq actually
# failing on whatever machine runs this suite); the real, historically
# failing case is pinned separately in
# tests/causal/test_event_study.py::test_the_p6_bootstrap_resample_that_once_broke_lstsq_now_fits.


def test_ols_fit_falls_back_to_qr_when_lstsq_raises(monkeypatch):
    def always_fails(X, y, rcond=None):
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")

    monkeypatch.setattr(np.linalg, "lstsq", always_fails)
    x = np.linspace(0, 10, 50)
    X = np.column_stack([np.ones(50), x])
    y = 3.0 + 4.0 * x

    fit = ols_fit(X, y)

    # The fallback recovers the same exact-fit coefficients the primary path
    # does (see test_ols_recovers_known_coefficients_without_noise) -- the
    # branch is invisible to a caller when it works correctly.
    assert fit.beta[0] == pytest.approx(3.0, abs=1e-9)
    assert fit.beta[1] == pytest.approx(4.0, abs=1e-9)


def test_ols_fit_raises_if_the_qr_fallback_is_also_non_finite(monkeypatch):
    def always_fails(X, y, rcond=None):
        raise np.linalg.LinAlgError("SVD did not converge in Linear Least Squares")

    def returns_garbage(X, y, lapack_driver=None):
        return np.full(X.shape[1], np.nan), None, None, None

    monkeypatch.setattr(np.linalg, "lstsq", always_fails)
    monkeypatch.setattr(covariance_module.scipy_linalg, "lstsq", returns_garbage)
    X, y = design()

    with pytest.raises(np.linalg.LinAlgError, match="QR fallback"):
        ols_fit(X, y)
