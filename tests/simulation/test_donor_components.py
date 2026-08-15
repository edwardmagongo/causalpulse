"""Donor-pool DGP components: deterministic single-unit adoption and two
unit-effects variants that make the treated unit diverge from its donor pool
in a controlled way."""

from __future__ import annotations

import numpy as np
import pytest
from cp_simulation.donor_components import (
    ClusteredUnitEffects,
    LevelOffsetUnitEffects,
    SingleUnitAdoption,
)
from cp_simulation.panel_components import NormalUnitEffects


def test_single_unit_adoption_treats_exactly_one_unit():
    adoption = SingleUnitAdoption(start=18, unit_index=3)
    covariates = np.zeros((10, 1))
    slopes = np.zeros(10)
    rng = np.random.default_rng(0)

    cohorts = adoption.cohorts(covariates, slopes, n_periods=24, rng=rng)

    assert np.count_nonzero(cohorts) == 1
    assert cohorts[3] == 18
    assert not adoption.violates_parallel_trends()


def test_single_unit_adoption_rejects_an_out_of_range_index():
    adoption = SingleUnitAdoption(start=18, unit_index=99)
    covariates = np.zeros((10, 1))
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="unit_index"):
        adoption.cohorts(covariates, np.zeros(10), n_periods=24, rng=rng)


def test_clustered_unit_effects_gives_the_treated_unit_cluster_a_mean():
    effects = ClusteredUnitEffects(
        n_similar=9, mean_a=0.3, mean_b=-0.1, within_sd=0.001, treated_index=0
    )
    rng = np.random.default_rng(0)

    _, delta = effects.draw(18, rng)

    # within_sd=0.001 makes each cluster's own draws tight enough that a
    # single-seed check on cluster means is not noise-dominated.
    assert delta[0] == pytest.approx(0.3, abs=0.01)
    assert delta[:9].mean() == pytest.approx(0.3, abs=0.01)
    assert delta[9:].mean() == pytest.approx(-0.1, abs=0.01)


def test_clustered_unit_effects_rejects_a_treated_index_outside_cluster_a():
    with pytest.raises(ValueError, match="treated_index"):
        ClusteredUnitEffects(n_similar=3, mean_a=0.3, mean_b=-0.1, within_sd=0.01, treated_index=5)


def test_clustered_unit_effects_rejects_a_pool_with_no_room_for_cluster_b():
    effects = ClusteredUnitEffects(n_similar=18, mean_a=0.3, mean_b=-0.1, within_sd=0.01)
    with pytest.raises(ValueError, match="n_similar"):
        effects.draw(18, np.random.default_rng(0))


def test_level_offset_shifts_only_the_chosen_unit():
    base = NormalUnitEffects(level_sd=0.3, slope_sd=0.1)
    effects = LevelOffsetUnitEffects(base=base, unit_index=0, level_offset=5.0)
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)

    base_alpha, base_delta = base.draw(18, rng_a)
    offset_alpha, offset_delta = effects.draw(18, rng_b)

    assert offset_alpha[0] == pytest.approx(base_alpha[0] + 5.0)
    np.testing.assert_array_equal(offset_alpha[1:], base_alpha[1:])
    np.testing.assert_array_equal(offset_delta, base_delta)


def test_level_offset_reliably_places_the_unit_outside_a_17_donor_pool():
    # Measured (design doc §2 item 5): at level_offset=5.0, 300/300 seeds.
    # This is the check the estimator's convex_hull assumption check
    # ultimately relies on being reliable -- pinned here at the component
    # level, independent of the estimator.
    base = NormalUnitEffects(level_sd=0.3, slope_sd=0.1)
    effects = LevelOffsetUnitEffects(base=base, unit_index=0, level_offset=5.0)
    outside = 0
    for seed in range(100):
        alpha, _ = effects.draw(18, np.random.default_rng(seed))
        if alpha[0] > alpha[1:].max():
            outside += 1
    assert outside == 100
