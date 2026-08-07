import numpy as np
from cp_simulation.scenarios import build_scenario


def test_same_root_seed_reproduces_identical_data():
    first = build_scenario("D").generate(n_units=2000, rng=np.random.default_rng(2026))
    second = build_scenario("D").generate(n_units=2000, rng=np.random.default_rng(2026))

    np.testing.assert_array_equal(
        first.dataset.frame.to_numpy(), second.dataset.frame.to_numpy()
    )
    np.testing.assert_array_equal(first.truth.true_cate, second.truth.true_cate)


def test_spawned_seeds_are_independent_across_replications():
    root = np.random.SeedSequence(12345)
    children = root.spawn(4)
    outcomes = [
        build_scenario("A")
        .generate(n_units=1000, rng=np.random.default_rng(child))
        .dataset.frame["outcome"]
        .to_numpy()
        for child in children
    ]

    for i in range(len(outcomes)):
        for j in range(i + 1, len(outcomes)):
            assert not np.array_equal(outcomes[i], outcomes[j])


def test_spawned_seeds_are_reproducible_by_index():
    def replication(index: int) -> np.ndarray:
        child = np.random.SeedSequence(777).spawn(8)[index]
        return (
            build_scenario("A")
            .generate(n_units=500, rng=np.random.default_rng(child))
            .dataset.frame["outcome"]
            .to_numpy()
        )

    np.testing.assert_array_equal(replication(5), replication(5))
