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
    dataset, estimate = fitted
    data = love_plot_data(estimate, dataset.schema.covariates)

    draw_love_plot(data, ax)

    labels = [t.get_text() for t in ax.get_yticklabels()]
    for name in dataset.schema.covariates:
        assert any(name == label for label in labels)


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


def test_overlap_data_records_that_the_scores_are_trimmed(fitted):
    # Untrimmed scores would show an overlap problem the estimate never faced.
    #
    # This scenario/seed pair is the one Task 3 proved clips zero of its 600
    # units (see cp_causal's per_unit_propensity test history), so `trimmed`
    # cannot be validated here as an empirical "some score was clipped"
    # claim -- no fixture reachable from this test file could prove that
    # without duplicating Task 3's own scenario-sizing work. `trimmed`
    # instead records a structural fact about *which* diagnostic the scores
    # were read from: `per_unit_propensity` is documented, at its point of
    # production in cp_causal's estimators, to always hold the post-clip
    # scores the weights were actually built from, never an unclipped
    # refit. Reading that diagnostic is what `trimmed=True` asserts.
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)

    data = overlap_data(estimate, treatment)

    assert data.trimmed is True


def test_overlap_data_derives_trim_bounds_from_the_scores_actually_drawn(fitted):
    # cp_causal's estimators do not expose the propensity clip point as a
    # structured diagnostic (PropensityFit.trim_bounds is only interpolated
    # into a free-text interpretation string), so there is no typed field to
    # read it from. overlap_data instead reports the realized range of the
    # trimmed per_unit_propensity array itself -- the tightest bound this
    # particular fit actually produced. On a fit where nothing was clipped
    # (as here), that range sits strictly inside the estimator's configured
    # clip point rather than touching it, which is exactly what this test
    # pins: a hardcoded (0.01, 0.99) literal would fail it.
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)
    scores = np.asarray(estimate.diagnostics["per_unit_propensity"].value)

    data = overlap_data(estimate, treatment)

    assert data.trim_bounds == (float(scores.min()), float(scores.max()))
    assert data.trim_bounds != (0.01, 0.99)


def test_drawing_overlap_puts_both_arms_on_the_axes(fitted, ax):
    dataset, estimate = fitted
    treatment = dataset.frame[dataset.schema.treatment].to_numpy(dtype=int)

    draw_overlap(overlap_data(estimate, treatment), ax)

    assert ax.get_legend() is not None
    assert len(ax.get_legend().get_texts()) == 2


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
