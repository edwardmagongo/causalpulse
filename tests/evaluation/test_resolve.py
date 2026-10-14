import numpy as np
import pytest
from cp_core.dataset import CausalDataset
from cp_evaluation.grid import (
    CateCell,
    CrossSectionCell,
    DonorPoolCell,
    LongPanelCell,
    PanelCell,
    cell_rng,
)
from cp_evaluation.resolve import resolve_cell, truth_for
from cp_simulation.cate_scenarios import CATE_N_UNITS
from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
from cp_simulation.panel_scenarios import PANEL_PROTOCOLS, PANEL_SCENARIOS
from cp_simulation.params import ScenarioParams
from cp_simulation.scenarios import SCENARIOS


def test_a_cross_section_cell_resolves_to_a_dataset_of_the_right_size():
    cell = CrossSectionCell(scenario="D", n_units=400, confounding_strength=1.0, seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert isinstance(resolved.dataset, CausalDataset)
    assert len(resolved.dataset.frame) == 400


def test_a_panel_cell_resolves_to_a_panel_of_the_right_shape():
    cell = PanelCell(scenario="P1", n_units=100, n_periods=6, seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert len(resolved.dataset.frame) == 100 * 6


def test_resolution_is_reproducible():
    cell = CrossSectionCell(scenario="D", n_units=300, confounding_strength=1.0, seed=2)

    first = resolve_cell(cell, root_entropy=5).dataset.frame
    second = resolve_cell(cell, root_entropy=5).dataset.frame

    assert first.equals(second)


def test_a_cross_section_cell_carries_both_ate_and_att_truth():
    cell = CrossSectionCell(scenario="D", n_units=2000, confounding_strength=1.0, seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert set(resolved.truth) == {"ATE", "ATT"}
    assert np.isfinite(resolved.truth["ATE"])
    assert np.isfinite(resolved.truth["ATT"])


def test_a_panel_cell_carries_att_truth_only():
    cell = PanelCell(scenario="P1", n_units=200, n_periods=6, seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert set(resolved.truth) == {"ATT"}


def test_truth_for_returns_the_value_for_the_requested_estimand():
    cell = CrossSectionCell(scenario="D", n_units=2000, confounding_strength=1.0, seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    assert truth_for(resolved, "ATE") == resolved.truth["ATE"]
    assert truth_for(resolved, "ATT") == resolved.truth["ATT"]


def test_asking_a_panel_cell_for_ate_truth_raises_rather_than_defaulting():
    # A panel cell has no ATE truth. Falling back to the ATT value would
    # manufacture bias that is not there -- the same defect family as the
    # Phase 9 bsts bug and Phase 10's HIGH 2, where a quantity was built from
    # fields that did not mean, for that estimator, what the formula assumed.
    cell = PanelCell(scenario="P1", n_units=200, n_periods=6, seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    with pytest.raises(KeyError, match="ATE"):
        truth_for(resolved, "ATE")


def test_truth_is_measured_not_declared():
    # `declared_effect` is a label. Ground truth is measured from the realised
    # potential outcomes, and on a finite sample the two differ.
    cell = CrossSectionCell(scenario="D", n_units=500, confounding_strength=1.0, seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    declared = SCENARIOS["D"].declared_effect(ScenarioParams())
    assert resolved.truth["ATE"] != declared


def test_panel_truth_is_measured_not_declared():
    # P1's constant effect is added identically to every unit-period with no
    # sampling noise in the effect itself, so its measured ATT equals its
    # declared 1.0 exactly at any sample size -- that equality is a property
    # of the scenario, not evidence the pipeline reads `declared_effect`.
    # P4's declared 2.8361 is itself a measured, cohort-size-weighted average
    # (see its registry docstring), so a *different* finite sample draws
    # different cohort sizes and its measured overall ATT does not coincide
    # with the declared value. n_periods=8 is P4's minimum: its staggered
    # cohorts adopt at periods (3, 5, 7), which need at least one post-period
    # for the last cohort.
    cell = PanelCell(scenario="P4", n_units=100, n_periods=8, seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    declared = PANEL_SCENARIOS["P4"].declared_effect
    assert resolved.truth["ATT"] != declared


def test_a_donor_pool_cell_resolves_to_a_panel_of_the_registrys_own_shape():
    # DONOR_N_UNITS=18, DONOR_N_PERIODS=24 for S1 (donor_scenarios.py); not
    # caller-supplied, unlike PanelCell's n_units/n_periods.
    cell = DonorPoolCell(scenario="S1", seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert len(resolved.dataset.frame) == 18 * 24


def test_a_donor_pool_cell_carries_att_truth_only():
    cell = DonorPoolCell(scenario="S1", seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert set(resolved.truth) == {"ATT"}
    assert np.isfinite(resolved.truth["ATT"])


def test_s5_resolves_to_its_own_narrower_protocol():
    # S5 overrides n_periods=6 in DONOR_SCENARIOS (insufficient pre-periods,
    # deliberately); n_units stays DONOR_N_UNITS=18.
    cell = DonorPoolCell(scenario="S5", seed=0)

    resolved = resolve_cell(cell, root_entropy=1)

    assert len(resolved.dataset.frame) == 18 * 6


def test_s1_truth_equals_the_treated_units_hand_averaged_post_period_effects():
    # Independent: the treated unit's row of `true_effect_matrix` (per-cell
    # potential-outcome difference), averaged by hand over its post-adoption
    # cells, is compared with resolve_cell's truth["ATT"] and with S1's
    # constant tau=1.0. That is a different reduction from true_att_overall's
    # masked mean over the whole matrix.
    # NOT independent: the matrix and true_att_overall come from the same
    # generate() call (rebuilt here with the stream resolve_cell uses); this
    # does not re-derive the effect matrix itself.
    cell = DonorPoolCell(scenario="S1", seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    spec = DONOR_SCENARIOS["S1"]
    panel = build_donor_scenario("S1").generate(
        n_units=spec.n_units,
        n_periods=spec.n_periods,
        rng=cell_rng(cell, root_entropy=1),
    )
    truth = panel.truth
    treated_rows = np.flatnonzero(truth.true_cohorts > 0)
    assert len(treated_rows) == 1
    row = truth.true_effect_matrix[treated_rows[0]]
    post = row != 0.0
    assert post.sum() > 0
    hand_mean = float(row[post].mean())

    assert resolved.truth["ATT"] == pytest.approx(hand_mean)
    assert hand_mean == pytest.approx(1.0)
    assert len(truth.true_att_by_cohort) == 1
    only_cohort_att = next(iter(truth.true_att_by_cohort.values()))
    assert resolved.truth["ATT"] == pytest.approx(only_cohort_att)


def test_donor_pool_resolution_is_reproducible():
    cell = DonorPoolCell(scenario="S2", seed=1)

    first = resolve_cell(cell, root_entropy=5)
    second = resolve_cell(cell, root_entropy=5)

    assert first.dataset.frame.equals(second.dataset.frame)
    assert first.truth["ATT"] == second.truth["ATT"]


def test_s4_shocks_a_donor_column_but_leaves_truth_equal_to_its_base_dgps():
    # Identically seeded generators for S4 and its unwrapped `.base` PanelDGP:
    # same units and effects, so any difference is the shock alone.
    spec = DONOR_SCENARIOS["S4"]
    s4 = build_donor_scenario("S4")
    kwargs = {"n_units": spec.n_units, "n_periods": spec.n_periods}
    contaminated = s4.generate(rng=np.random.default_rng(123), **kwargs)
    base = s4.base.generate(rng=np.random.default_rng(123), **kwargs)

    assert contaminated.truth.true_att_overall == base.truth.true_att_overall

    cf = contaminated.dataset.frame
    bf = base.dataset.frame
    treated_unit = bf.loc[bf["cohort"] > 0, "unit_id"].iloc[0]
    donors = (bf["unit_id"] != treated_unit).to_numpy()
    c_out = cf["outcome"].to_numpy()
    b_out = bf["outcome"].to_numpy()
    differing = c_out != b_out

    assert differing.any()
    assert not (differing & ~donors).any()
    assert cf.loc[differing, "unit_id"].nunique() == 1


def test_asking_a_donor_pool_cell_for_ate_truth_raises_rather_than_defaulting():
    cell = DonorPoolCell(scenario="S1", seed=0)
    resolved = resolve_cell(cell, root_entropy=1)

    with pytest.raises(KeyError, match="ATE"):
        truth_for(resolved, "ATE")


def test_a_cate_cell_resolves_its_scenario_at_the_registry_size_with_per_unit_truth():
    resolved = resolve_cell(CateCell("RICH", 0), root_entropy=0)

    assert len(resolved.dataset.frame) == CATE_N_UNITS
    assert set(resolved.truth) == {"ATE", "ATT"}
    assert resolved.cate_truth is not None
    assert resolved.cate_truth.heterogeneous is True
    assert resolved.cate_truth.values.shape == (CATE_N_UNITS,)
    # The ATE truth is the mean of the same per-unit truth, not a second measurement.
    assert resolved.truth["ATE"] == pytest.approx(float(resolved.cate_truth.values.mean()))


def test_null_resolves_as_declared_constant():
    resolved = resolve_cell(CateCell("NULL", 0), root_entropy=0)

    assert resolved.cate_truth is not None
    assert resolved.cate_truth.heterogeneous is False


def test_only_a_cate_cell_carries_per_unit_truth():
    cross = resolve_cell(
        CrossSectionCell(scenario="F", n_units=300, confounding_strength=1.0, seed=0),
        root_entropy=0,
    )

    # F's DGP records true_cate too; the arm deliberately does not score it.
    assert cross.cate_truth is None


def test_a_long_panel_cell_resolves_at_its_protocol_with_att_truth():
    resolved = resolve_cell(LongPanelCell("P7", 0), root_entropy=0)
    n_units, n_periods = PANEL_PROTOCOLS["P7"]

    frame = resolved.dataset.frame
    assert frame[resolved.dataset.schema.unit].nunique() == n_units
    assert frame[resolved.dataset.schema.time].nunique() == n_periods
    assert set(resolved.truth) == {"ATT"}
    assert np.isfinite(resolved.truth["ATT"])
    assert resolved.cate_truth is None
