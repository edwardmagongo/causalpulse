import matplotlib.pyplot as plt
import numpy as np
import pytest
from cp_causal.registry import get_estimator
from cp_core.values import EstimatorConfig
from cp_figures.balance import (
    draw_love_plot,
    draw_overlap,
    love_plot_data,
    overlap_data,
)
from cp_simulation.scenarios import build_scenario
from matplotlib.collections import PathCollection

CONFIG = EstimatorConfig(random_seed=0, bootstrap_reps=50)


@pytest.fixture(scope="module")
def fitted():
    dataset = build_scenario("D").generate(
        n_units=600, rng=np.random.default_rng(0)
    ).dataset
    return dataset, get_estimator("ipw")().fit(dataset, CONFIG)


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def test_love_data_pairs_each_covariate_with_both_values(fitted):
    dataset, estimate = fitted

    data = love_plot_data(estimate, dataset.schema.covariates)

    assert data.covariates == tuple(dataset.schema.covariates)
    assert len(data.before) == len(data.covariates)
    assert len(data.after) == len(data.covariates)


def test_love_data_refuses_a_name_list_of_the_wrong_length(fitted):
    # The diagnostics are bare arrays. If the names do not line up, every
    # label attaches to the wrong bar and the figure is silently wrong.
    _, estimate = fitted

    with pytest.raises(ValueError, match="covariate"):
        love_plot_data(estimate, ("x1",))


def test_love_data_carries_the_threshold_from_the_project_not_a_literal(fitted):
    dataset, estimate = fitted

    data = love_plot_data(estimate, dataset.schema.covariates)

    assert data.threshold == pytest.approx(0.1)


def test_weighting_improves_balance_on_a_confounded_scenario(fitted):
    dataset, estimate = fitted

    data = love_plot_data(estimate, dataset.schema.covariates)

    assert max(abs(v) for v in data.after) < max(abs(v) for v in data.before)


def test_drawing_a_love_plot_labels_every_covariate(fitted, ax):
    # Order-sensitive: a reordering of ax.set_yticklabels relative to the
    # scatter rows (labels attaching to the wrong bars) is exactly the
    # hazard this task exists to close. Pure set membership (the brief's
    # original `any(name == label for label in labels)`) would pass under
    # that reordering; pairing each covariate to its row by index does not.
    dataset, estimate = fitted
    data = love_plot_data(estimate, dataset.schema.covariates)

    draw_love_plot(data, ax)

    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == list(data.covariates)


def test_drawing_a_love_plot_draws_the_threshold_line_on_both_sides(fitted, ax):
    # |SMD| is a two-sided criterion -- the plot needs a reference line at
    # both +threshold and -threshold, not just one. The filter below
    # isolates axvline-style lines (a single x value repeated across every
    # y sample) from anything else drawn on the axes; draw_love_plot keeps
    # its markers off Axes.lines (scatter/hlines, not plot) precisely so
    # this filter cannot accidentally catch a marker instead.
    dataset, estimate = fitted
    data = love_plot_data(estimate, dataset.schema.covariates)

    draw_love_plot(data, ax)

    positions = [line.get_xdata()[0] for line in ax.lines if len(set(line.get_xdata())) == 1]
    assert any(abs(p - 0.1) < 1e-9 for p in positions)
    assert any(abs(p - (-0.1)) < 1e-9 for p in positions)


def test_drawing_a_love_plot_scatters_before_and_after_at_their_own_values(fitted, ax):
    # A swap of the before/after markers in draw_love_plot's ax.scatter
    # calls passes every other test in this file (labels, threshold lines,
    # legend). Reach the actual drawn x-positions via ax.collections --
    # verified directly on matplotlib 3.11.1: ax.hlines contributes one
    # LineCollection and each ax.scatter call contributes one
    # PathCollection, appended in call order, so filtering to
    # PathCollection and taking them in order recovers "before" then
    # "after" unambiguously. On this fixture before=[0.847, 0.346] and
    # after=[-0.174, -0.001] (weighting narrows both covariates' SMD
    # substantially and flips sign on one), so a before/after swap changes
    # the asserted numbers rather than landing within tolerance by luck.
    dataset, estimate = fitted
    data = love_plot_data(estimate, dataset.schema.covariates)

    draw_love_plot(data, ax)

    path_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(path_collections) == 2
    before_x = [xy[0] for xy in path_collections[0].get_offsets()]
    after_x = [xy[0] for xy in path_collections[1].get_offsets()]

    assert before_x == pytest.approx(data.before)
    assert after_x == pytest.approx(data.after)


def test_overlap_data_splits_the_scores_by_arm(fitted):
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)

    data = overlap_data(estimate, treatment)

    assert len(data.treated_scores) == int(treatment.sum())
    assert len(data.control_scores) == int((1 - treatment).sum())


def test_overlap_data_refuses_a_treatment_vector_of_the_wrong_length(fitted):
    # per_unit_propensity is a bare array with no length check of its own.
    # A caller-supplied treatment vector of a different length would split
    # the scores at the wrong index silently rather than refuse.
    _, estimate = fitted

    with pytest.raises(ValueError, match="treatment"):
        overlap_data(estimate, np.array([0, 1, 1]))


def test_overlap_data_counts_zero_trimmed_units_on_a_fit_that_never_clips(fitted):
    # This scenario/seed pair (build_scenario("D"), n_units=600, seed=0) is
    # the one Task 3 proved clips zero of its 600 units: the raw logistic
    # scores land in [0.0267, 0.9723], strictly inside the estimator's
    # configured trim bounds. `n_trimmed` is read straight from the
    # estimator's own `n_trimmed` diagnostic, so on this fixture it must
    # honestly report zero, not a hardcoded/structural True.
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)

    data = overlap_data(estimate, treatment)

    assert data.n_trimmed == 0


def test_overlap_data_counts_trimmed_units_on_a_fit_that_does_clip():
    # Task 3's own fixture for exercising trimming (see
    # tests/causal/test_propensity_diagnostics.py): doubling the
    # confounding strength sharpens assignment enough that 8 of 200 units'
    # raw scores fall outside the trim bounds and get clipped for ipw.
    # Paired with test_overlap_data_counts_zero_trimmed_units_on_a_fit_that_never_clips
    # above, this is what makes n_trimmed a meaningful field at all -- a
    # field that is always the same value regardless of fixture could not
    # be told apart from a hardcoded constant.
    dataset = build_scenario("D", 2.0).generate(
        n_units=200, rng=np.random.default_rng(1)
    ).dataset
    estimate = get_estimator("ipw")().fit(dataset, CONFIG)
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)

    data = overlap_data(estimate, treatment)

    assert data.n_trimmed == 8


def test_overlap_data_derives_score_range_from_the_scores_actually_drawn(fitted):
    # cp_causal's estimators do not expose the propensity clip point as a
    # structured diagnostic (PropensityFit.trim_bounds is only interpolated
    # into a free-text interpretation string), and the clip point is
    # configurable per fit, so it cannot be assumed to equal
    # POSITIVITY_THRESHOLD. overlap_data instead reports the realized range
    # of the trimmed per_unit_propensity array itself -- the tightest bound
    # this particular fit actually produced. On a fit where nothing was
    # clipped (as here), that range sits strictly inside the estimator's
    # configured clip point rather than touching it, which is exactly what
    # this test pins: a hardcoded (0.01, 0.99) literal would fail it.
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)
    scores = np.asarray(estimate.diagnostics["per_unit_propensity"].value)

    data = overlap_data(estimate, treatment)

    assert data.score_range == (float(scores.min()), float(scores.max()))
    assert data.score_range != (0.01, 0.99)


def test_drawing_overlap_puts_both_arms_on_the_axes(fitted, ax):
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)

    draw_overlap(overlap_data(estimate, treatment), ax)

    assert ax.get_legend() is not None
    assert len(ax.get_legend().get_texts()) == 2


def test_drawing_overlap_draws_each_arms_scores_not_swapped(fitted, ax):
    # A swap of treated_scores/control_scores in draw_overlap's two ax.hist
    # calls passes every other test in this file (legend count, axis
    # label). Verified directly on matplotlib 3.11.1: each ax.hist(...,
    # bins=20) call appends exactly 20 Rectangle patches to ax.patches, in
    # call order -- so the first 20 patches back "treated" and the next 20
    # back "control". On this fixture the two arms' propensity scores are
    # distinguishable (treated mean 0.634 vs control mean 0.459, unequal
    # arm sizes 334/266), so recomputing each arm's histogram independently
    # and comparing per-bin heights against the drawn patches catches a
    # swap rather than passing by coincidence.
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)
    data = overlap_data(estimate, treatment)

    draw_overlap(data, ax)

    assert len(ax.patches) == 40
    treated_heights = [p.get_height() for p in ax.patches[:20]]
    control_heights = [p.get_height() for p in ax.patches[20:]]

    expected_treated, _ = np.histogram(data.treated_scores, bins=20)
    expected_control, _ = np.histogram(data.control_scores, bins=20)

    assert treated_heights == pytest.approx(expected_treated)
    assert control_heights == pytest.approx(expected_control)


def test_drawing_overlap_labels_the_axis_as_trimmed(fitted, ax):
    # The axis label is the reader-facing half of the trimmed-scores
    # hazard: untrimmed scores would display an overlap problem the
    # estimate did not actually face, so the axis must say the scores shown
    # are the trimmed ones.
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)

    draw_overlap(overlap_data(estimate, treatment), ax)

    assert "trim" in ax.get_xlabel().lower()


def test_drawing_leaves_pyplot_alone(fitted, ax):
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)
    before = plt.get_fignums()

    draw_overlap(overlap_data(estimate, treatment), ax)

    assert plt.get_fignums() == before
