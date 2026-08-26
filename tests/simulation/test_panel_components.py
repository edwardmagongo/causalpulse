import numpy as np
import pytest
from cp_simulation.panel_components import (
    ConstantDynamicEffect,
    LevelSelectedAdoption,
    LinearTimePath,
    NormalUnitEffects,
    RampDynamicEffect,
    StaggeredAdoption,
    TrendSelectedAdoption,
    UniformAdoption,
)


def test_zero_slope_dispersion_gives_every_unit_the_same_trend():
    # This is what makes parallel trends hold exactly by construction, so it is
    # the single most load-bearing property in the panel simulation.
    effects = NormalUnitEffects(level_sd=1.0, slope_sd=0.0)

    alpha, delta = effects.draw(200, np.random.default_rng(0))

    assert alpha.shape == delta.shape == (200,)
    np.testing.assert_array_equal(delta, np.zeros(200))
    assert alpha.std() > 0.5


def test_non_zero_slope_dispersion_produces_varying_trends():
    effects = NormalUnitEffects(level_sd=1.0, slope_sd=0.3)

    _, delta = effects.draw(2000, np.random.default_rng(0))

    assert delta.std() == pytest.approx(0.3, rel=0.1)


def test_linear_time_path_is_shared_by_every_unit():
    path = LinearTimePath(slope=0.1, shock_sd=0.0)

    gamma = path.values(5, np.random.default_rng(0))

    np.testing.assert_allclose(gamma, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_uniform_adoption_gives_one_adoption_period():
    adoption = UniformAdoption(start=5, share=0.5)
    covariates = np.zeros((400, 1))

    cohorts = adoption.cohorts(covariates, np.zeros(400), 8, np.random.default_rng(0))

    assert set(np.unique(cohorts)) == {0, 5}
    assert 0.4 < (cohorts > 0).mean() < 0.6


def test_staggered_adoption_spreads_units_across_cohorts():
    adoption = StaggeredAdoption(periods=(3, 5, 7), never_share=0.4)
    covariates = np.zeros((400, 1))

    cohorts = adoption.cohorts(covariates, np.zeros(400), 8, np.random.default_rng(0))

    assert set(np.unique(cohorts)) == {0, 3, 5, 7}
    assert (cohorts == 0).mean() == pytest.approx(0.4, abs=0.05)
    for period in (3, 5, 7):
        assert (cohorts == period).sum() > 50


def test_trend_selected_adoption_puts_steeper_units_in_earlier_cohorts():
    # This is the parallel-trends violation. If the ordering ever inverts, P5
    # stops violating anything and the scenario silently becomes P3.
    adoption = TrendSelectedAdoption(periods=(3, 5, 7), never_share=0.4)
    rng = np.random.default_rng(0)
    slopes = rng.normal(scale=0.3, size=400)

    cohorts = adoption.cohorts(np.zeros((400, 1)), slopes, 8, rng)

    early = slopes[cohorts == 3].mean()
    late = slopes[cohorts == 7].mean()
    never = slopes[cohorts == 0].mean()
    assert early > late > never


def test_level_selected_adoption_selects_on_the_unit_level():
    # Load-bearing for P6: link-function bias arises only when the two arms sit
    # at different points on the logistic curve.
    adoption = LevelSelectedAdoption(start=5)
    rng = np.random.default_rng(0)
    levels = rng.normal(size=400)

    cohorts = adoption.cohorts(np.zeros((400, 1)), levels, 8, rng, levels=levels)

    assert levels[cohorts == 5].mean() > levels[cohorts == 0].mean() + 0.8


def test_constant_effect_is_flat_over_exposure():
    effect = ConstantDynamicEffect(tau=1.0)
    cohorts = np.array([0, 3, 5])

    tau = effect.tau_matrix(cohorts, 6, np.zeros((3, 1)))

    assert tau.shape == (3, 6)
    np.testing.assert_array_equal(tau, np.ones((3, 6)))


def test_ramp_effect_grows_with_exposure_and_differs_across_cohorts():
    # The configuration that breaks two-way fixed effects: at any calendar
    # period an early cohort has a larger effect than a late one, so
    # already-treated units make bad controls.
    effect = RampDynamicEffect(tau=1.0)
    cohorts = np.array([3, 5])

    tau = effect.tau_matrix(cohorts, 6, np.zeros((2, 1)))

    # period 6 (index 5): cohort 3 has 4 periods of exposure, cohort 5 has 2
    assert tau[0, 5] == pytest.approx(4.0)
    assert tau[1, 5] == pytest.approx(2.0)


def test_adoption_rejects_a_start_outside_the_panel():
    adoption = UniformAdoption(start=12, share=0.5)

    with pytest.raises(ValueError, match="start"):
        adoption.cohorts(np.zeros((10, 1)), np.zeros(10), 8, np.random.default_rng(0))


def test_staggered_adoption_rejects_a_period_outside_the_panel():
    adoption = StaggeredAdoption(periods=(3, 5, 99), never_share=0.4)

    with pytest.raises(ValueError, match="periods"):
        adoption.cohorts(np.zeros((10, 1)), np.zeros(10), 8, np.random.default_rng(0))


def test_imposed_adoption_is_a_natural_experiment():
    assert UniformAdoption().provenance() == "natural_experiment"
    assert StaggeredAdoption().provenance() == "natural_experiment"


def test_adoption_selected_on_the_outcome_path_is_observational():
    # Both select on something the outcome process generates -- the unit's own
    # trend slope, or its own baseline level -- which is what makes them
    # observational regardless of whether parallel trends survives.
    assert TrendSelectedAdoption().provenance() == "observational"
    assert LevelSelectedAdoption().provenance() == "observational"
