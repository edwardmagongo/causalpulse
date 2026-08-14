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


def test_panel_generation_is_reproducible_from_one_root_seed():
    from cp_simulation.panel_scenarios import build_panel_scenario

    def run():
        return build_panel_scenario("P4").generate(
            n_units=200, n_periods=8, rng=np.random.default_rng(4242)
        )

    first, second = run(), run()

    np.testing.assert_array_equal(
        first.dataset.frame.to_numpy(), second.dataset.frame.to_numpy()
    )
    np.testing.assert_array_equal(
        first.truth.true_effect_matrix, second.truth.true_effect_matrix
    )
    assert first.truth.true_att_gt == second.truth.true_att_gt


def test_spawned_seeds_give_independent_panels():
    from cp_simulation.panel_scenarios import build_panel_scenario

    children = np.random.SeedSequence(2026).spawn(3)
    outcomes = [
        build_panel_scenario("P1")
        .generate(n_units=100, n_periods=8, rng=np.random.default_rng(child))
        .dataset.frame["outcome"]
        .to_numpy()
        for child in children
    ]

    for i in range(len(outcomes)):
        for j in range(i + 1, len(outcomes)):
            assert not np.array_equal(outcomes[i], outcomes[j])
