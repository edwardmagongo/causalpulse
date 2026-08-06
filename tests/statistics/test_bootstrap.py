import numpy as np
import pytest

from cp_statistics.bootstrap import bootstrap_interval


def test_iid_bootstrap_se_approximates_the_analytic_se_of_the_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=500)
    analytic_se = data.std(ddof=1) / np.sqrt(data.size)

    result = bootstrap_interval(
        np.mean, data, reps=2000, rng=np.random.default_rng(1)
    )

    assert result.standard_error == pytest.approx(analytic_se, rel=0.08)
    assert result.replicates.shape == (2000,)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    data = rng.normal(loc=1.0, size=300)

    result = bootstrap_interval(np.mean, data, reps=1000, rng=np.random.default_rng(3))

    assert result.interval.lower < data.mean() < result.interval.upper
    assert result.interval.level == 0.95


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    data = np.random.default_rng(4).normal(size=200)

    first = bootstrap_interval(np.mean, data, reps=500, rng=np.random.default_rng(99))
    second = bootstrap_interval(np.mean, data, reps=500, rng=np.random.default_rng(99))

    np.testing.assert_array_equal(first.replicates, second.replicates)


def test_cluster_bootstrap_is_wider_than_iid_under_within_cluster_correlation():
    rng = np.random.default_rng(5)
    clusters = np.repeat(np.arange(30), 20)
    data = np.repeat(rng.normal(scale=3.0, size=30), 20) + rng.normal(scale=0.1, size=600)

    iid = bootstrap_interval(np.mean, data, reps=1000, rng=np.random.default_rng(6))
    clustered = bootstrap_interval(
        np.mean,
        data,
        reps=1000,
        rng=np.random.default_rng(6),
        kind="cluster",
        clusters=clusters,
    )

    assert clustered.standard_error > 3.0 * iid.standard_error


def test_block_bootstrap_preserves_serial_structure():
    rng = np.random.default_rng(7)
    noise = rng.normal(size=400)
    series = np.zeros(400)
    for t in range(1, 400):
        series[t] = 0.9 * series[t - 1] + noise[t]

    iid = bootstrap_interval(np.mean, series, reps=1000, rng=np.random.default_rng(8))
    block = bootstrap_interval(
        np.mean,
        series,
        reps=1000,
        rng=np.random.default_rng(8),
        kind="block",
        block_size=40,
    )

    assert block.standard_error > 1.5 * iid.standard_error


def test_cluster_bootstrap_requires_cluster_labels():
    data = np.arange(10.0)
    with pytest.raises(ValueError, match="clusters"):
        bootstrap_interval(
            np.mean, data, reps=10, rng=np.random.default_rng(0), kind="cluster"
        )


def test_block_bootstrap_requires_block_size():
    data = np.arange(10.0)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_interval(
            np.mean, data, reps=10, rng=np.random.default_rng(0), kind="block"
        )


def test_basic_method_differs_from_percentile_on_a_skewed_statistic():
    rng = np.random.default_rng(9)
    data = rng.exponential(scale=2.0, size=200)

    percentile = bootstrap_interval(
        np.mean, data, reps=2000, rng=np.random.default_rng(10), method="percentile"
    )
    basic = bootstrap_interval(
        np.mean, data, reps=2000, rng=np.random.default_rng(10), method="basic"
    )

    assert percentile.interval.lower != pytest.approx(basic.interval.lower, abs=1e-9)


def test_block_bootstrap_rejects_zero_block_size():
    data = np.arange(10.0)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_interval(
            np.mean,
            data,
            reps=10,
            rng=np.random.default_rng(0),
            kind="block",
            block_size=0,
        )


def test_block_bootstrap_rejects_negative_block_size():
    data = np.arange(10.0)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_interval(
            np.mean,
            data,
            reps=10,
            rng=np.random.default_rng(0),
            kind="block",
            block_size=-5,
        )


def test_block_bootstrap_rejects_block_size_exceeding_data_length():
    data = np.arange(10.0)
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_interval(
            np.mean,
            data,
            reps=10,
            rng=np.random.default_rng(0),
            kind="block",
            block_size=20,
        )
