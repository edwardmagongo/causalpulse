import dataclasses
from collections import Counter

import pytest
from cp_evaluation.grid import CrossSectionCell, expand_cells
from cp_evaluation.tiers import (
    CI_ESTIMATORS,
    FULL_ESTIMATORS,
    FULL_FIT_CONFIG,
    ci_config,
    full_config,
)
from cp_simulation.params import AXIS_NAMES, ScenarioParams
from cp_simulation.scenarios import SCENARIOS

# The four axes a cell is "at baseline" for, i.e. every axis except the one
# crossed against n_units. Derived from AXIS_NAMES for the same reason
# `grid._NEW_AXIS_NAMES` is: a fifth axis must not silently drop out of the
# baseline/one-at-a-time split these tests count.
_NEW_AXIS_NAMES = tuple(name for name in AXIS_NAMES if name != "confounding_strength")


def test_full_covers_every_cross_sectional_scenario_and_all_but_p7():
    from cp_simulation.panel_scenarios import PANEL_SCENARIOS
    from cp_simulation.scenarios import SCENARIOS

    config = full_config()

    assert set(config.cross_section.scenarios) == set(SCENARIOS)
    # P7 is deliberately excluded: it is generated at its own longer
    # protocol (HIER_N_UNITS=600, HIER_N_PERIODS=20, per
    # cp_simulation.panel_scenarios) and needs n_periods >= 16, but
    # PanelAxes applies one n_periods list to every scenario it carries.
    # Including P7 would force P1-P6 onto that far slower protocol just to
    # accommodate one scenario, so it is left out and every other panel
    # scenario is kept. See full_config's docstring for the measurement
    # that found this.
    assert set(config.panel.scenarios) == set(PANEL_SCENARIOS) - {"P7"}


def test_full_is_strictly_wider_than_ci():
    # If `full` were not a superset the two tiers would answer different
    # questions, and a band measured on one would say nothing about the other.
    assert len(expand_cells(full_config())) > len(expand_cells(ci_config()))


def test_full_uses_the_default_bootstrap_replications():
    # `ci` runs at 200 to fit a two-minute gate. `full` has no such budget and
    # should not inherit the compromise.
    assert FULL_FIT_CONFIG.bootstrap_reps == 1000


def test_full_includes_the_estimators_ci_had_to_drop():
    assert "causal_forest" in FULL_ESTIMATORS
    assert "hierarchical_did" in FULL_ESTIMATORS
    assert set(CI_ESTIMATORS) <= set(FULL_ESTIMATORS)


def test_full_includes_synthetic_control_and_bsts():
    assert "synthetic_control" in FULL_ESTIMATORS
    assert "bsts" in FULL_ESTIMATORS


def test_ci_includes_synthetic_control_but_not_bsts():
    # synthetic_control is SLSQP + permutation, the ci tier's cost class.
    # bsts is NumPyro/NUTS, the same cost class hierarchical_did is excluded
    # from ci for -- see the module docstring's measured numbers.
    assert "synthetic_control" in CI_ESTIMATORS
    assert "bsts" not in CI_ESTIMATORS


def test_both_tiers_cover_all_six_donor_pool_scenarios():
    from cp_simulation.donor_scenarios import DONOR_SCENARIOS

    assert ci_config().donor_pool is not None
    assert set(ci_config().donor_pool.scenarios) == set(DONOR_SCENARIOS)
    assert full_config().donor_pool is not None
    assert set(full_config().donor_pool.scenarios) == set(DONOR_SCENARIOS)


def test_adding_the_donor_pool_arm_does_not_perturb_any_existing_cell():
    # Grid-independence regression: standing up the third shape must not
    # change the identity, seed, or multiplicity of any pre-existing
    # cross-section/panel cell. Compares the full cell multisets (cells are
    # frozen dataclasses carrying their seed), not just counts, so a shifted
    # seed or a duplicated/dropped cell is caught.
    from cp_evaluation.grid import CrossSectionCell, PanelCell, expand_cells

    with_donor_pool = expand_cells(full_config())
    without_donor_pool = expand_cells(dataclasses.replace(full_config(), donor_pool=None))

    for shape in (CrossSectionCell, PanelCell):
        with_ = Counter(c for c in with_donor_pool if isinstance(c, shape))
        without = Counter(c for c in without_donor_pool if isinstance(c, shape))
        assert with_ == without
        assert sum(with_.values()) > 0


@pytest.mark.parametrize("name", FULL_ESTIMATORS)
def test_every_named_estimator_is_registered(name):
    from cp_causal.registry import all_estimators

    assert name in all_estimators()


def test_full_config_produces_the_designs_measured_68_cross_sectional_cells():
    # docs/specs/2026-09-13-causal-core-phase18-design.md section 12. Derived
    # from the enumeration rule by hand, per scenario, not read off the code:
    #
    #   |n_units| x (|confounding_strength| if declared else 1)
    #     + sum over declared new axes of (|values| - 1)
    #
    # with |n_units| = 2, |confounding_strength| = 3, and every new axis
    # carrying 3 values, so 2 non-baseline values each. D, G and H declare
    # confounding_strength; A, B, C and F do not.
    #
    #   scenario   declares cs   baseline      one-at-a-time   total
    #   A          no            2 x 1 =  2    3 x 2 = 6           8
    #   B          no            2 x 1 =  2    2 x 2 = 4           6
    #   C          no            2 x 1 =  2    3 x 2 = 6           8
    #   D          yes           2 x 3 =  6    3 x 2 = 6          12
    #   F          no            2 x 1 =  2    4 x 2 = 8          10
    #   G          yes           2 x 3 =  6    3 x 2 = 6          12
    #   H          yes           2 x 3 =  6    3 x 2 = 6          12
    #                                     26           42          68
    #
    # The one-at-a-time half is unchanged at 42; the baseline half falls from
    # 42 to 26 because A, B, C and F are no longer crossed against three
    # strengths they do not read. Counted at seed == 0 only: replications
    # multiplies every distinct cell identically, so it would inflate this
    # count without testing the enumeration itself.
    #
    # Both halves are asserted separately as well as summed: a bug that moved
    # 16 cells from one half to the other would leave the total at 68.
    config = full_config()

    distinct_cells = {
        cell
        for cell in expand_cells(config)
        if isinstance(cell, CrossSectionCell) and cell.seed == 0
    }
    default = ScenarioParams()
    baseline = {
        cell
        for cell in distinct_cells
        if all(getattr(cell, name) == getattr(default, name) for name in _NEW_AXIS_NAMES)
    }

    assert len(baseline) == 26
    assert len(distinct_cells) - len(baseline) == 42
    assert len(distinct_cells) == 68


def test_full_config_crosses_confounding_strength_only_where_it_is_declared():
    # The per-scenario shape behind the 26 above, checked against the registry
    # rather than restated: a scenario that declares the axis is crossed
    # against all three of full_config()'s strengths, and one that does not
    # contributes the single default strength. Without this, the count test
    # alone would pass on any rearrangement summing to 26.
    config = full_config()
    axes = config.cross_section
    assert axes is not None
    default = ScenarioParams()

    strengths: dict[str, set[float]] = {}
    for cell in expand_cells(config):
        if not isinstance(cell, CrossSectionCell):
            continue
        if any(getattr(cell, name) != getattr(default, name) for name in _NEW_AXIS_NAMES):
            continue
        strengths.setdefault(cell.scenario, set()).add(cell.confounding_strength)

    assert set(strengths) == set(axes.scenarios)
    for scenario, seen in strengths.items():
        if "confounding_strength" in SCENARIOS[scenario].axes:
            assert seen == set(axes.confounding_strength), scenario
        else:
            assert seen == {default.confounding_strength}, scenario
