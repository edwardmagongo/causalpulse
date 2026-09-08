import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cp_core.results import CounterfactualPath, EffectEstimate, FitMetadata
from cp_core.values import Interval
from cp_figures.effects import (
    counterfactual_path_data,
    draw_counterfactual_path,
    draw_forest_plot,
    forest_plot_data,
)


def _estimate(method="ipw", estimand="ATE", point=0.25, lo=0.05, hi=0.45, ci_basis="analytic"):
    return EffectEstimate(
        estimand=estimand, method=method, point=point, se=0.1,
        ci=Interval(lower=lo, upper=hi, level=0.95), ci_basis=ci_basis,
        bootstrap_ci=None, p_value=0.01, n_treated=100, n_control=100,
        scale="percentage_point", diagnostics={}, assumptions=(),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator=method, config_hash="h", runtime_seconds=0.1, random_seed=0
        ),
        subgroup_findings=(),
    )


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def test_forest_data_carries_one_row_per_estimate():
    data = forest_plot_data([_estimate("ipw"), _estimate("aipw", point=0.30)])

    assert data.methods == ("ipw", "aipw")
    assert data.points == (0.25, 0.30)
    assert data.lower == (0.05, 0.05)
    assert data.upper == (0.45, 0.45)


def test_forest_data_carries_the_estimand_per_row():
    # ATE and ATT are different quantities. A figure implying one shared
    # quantity is the defect family this project has hit three times.
    data = forest_plot_data([_estimate("ipw", "ATE"), _estimate("twfe", "ATT")])

    assert data.estimands == ("ATE", "ATT")


def test_forest_data_carries_the_interval_basis_per_row():
    data = forest_plot_data([_estimate()])

    assert data.bases == ("analytic",)


def test_forest_plot_refuses_an_estimate_without_an_interval(ax):
    # EffectEstimate.ci is typed Interval, not Interval | None, and
    # __post_init__ does not validate it -- dataclasses do not enforce field
    # types at runtime, so this construction succeeds despite the static
    # type. That is exactly the object require_interval exists to catch.
    bare = dataclasses.replace(_estimate(), ci=None)

    with pytest.raises(ValueError, match="without an interval"):
        forest_plot_data([bare])


def test_drawing_a_forest_plot_puts_the_points_on_the_axes(ax):
    data = forest_plot_data([_estimate("ipw"), _estimate("aipw", point=0.30)])

    draw_forest_plot(data, ax)

    # The error-bar container carries the points it was given.
    assert len(ax.containers) == 1
    drawn = np.asarray(ax.containers[0][0].get_xdata())
    assert np.allclose(sorted(drawn), sorted([0.25, 0.30]))


def test_drawing_a_forest_plot_labels_every_row_with_its_estimand(ax):
    # Presence checks (any(...)) cannot catch a mispaired zip: if the
    # drawing code paired estimand and method out of order, "ATE" and "ATT"
    # would both still be present somewhere in the label list even though
    # each row now names the wrong estimand for its method. Assert the
    # pairing itself, per row.
    data = forest_plot_data([_estimate("ipw", "ATE"), _estimate("twfe", "ATT")])

    draw_forest_plot(data, ax)

    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["ipw (ATE, analytic)", "twfe (ATT, analytic)"]
    for method, estimand, label in zip(data.methods, data.estimands, labels, strict=True):
        assert label == f"{method} ({estimand}, analytic)"


def test_drawing_a_forest_plot_labels_every_row_with_its_interval_basis(ax):
    # ci_basis distinguishes an analytic Wald interval from a bootstrap or
    # permutation one -- exactly as consequential to a cross-method
    # comparison as the estimand (see
    # test_drawing_a_forest_plot_labels_every_row_with_its_estimand above).
    # Order-sensitive, and checked pairwise rather than with a presence
    # check: a mispaired zip (row i's label carrying row j's basis) would
    # still leave every basis string present *somewhere* in the label list,
    # so `any(basis in label for ...)` could not catch a swap. Asserting the
    # exact label list, and then the pairing per row, can.
    data = forest_plot_data(
        [
            _estimate("ipw", "ATE", ci_basis="analytic"),
            _estimate("twfe", "ATT", ci_basis="bootstrap"),
            _estimate("aipw", "ATE", ci_basis="permutation"),
        ]
    )

    draw_forest_plot(data, ax)

    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [
        "ipw (ATE, analytic)",
        "twfe (ATT, bootstrap)",
        "aipw (ATE, permutation)",
    ]
    for method, estimand, basis, label in zip(
        data.methods, data.estimands, data.bases, labels, strict=True
    ):
        assert label == f"{method} ({estimand}, {basis})"


def test_drawing_a_forest_plot_draws_the_error_bars_spanning_the_interval(ax):
    # The lower_err/upper_err computation lives only inside draw_forest_plot
    # (by design), so the only way to catch a swapped xerr construction is
    # to inspect the drawn whisker geometry. Bounds are deliberately
    # asymmetric around each point -- a symmetric fixture would render
    # identically whether or not lower_err/upper_err were swapped.
    data = forest_plot_data(
        [
            _estimate("ipw", point=0.25, lo=0.05, hi=0.45),
            _estimate("twfe", point=0.30, lo=0.20, hi=0.33),
        ]
    )

    draw_forest_plot(data, ax)

    # ax.errorbar's ErrorbarContainer.lines is (points, caps, bar_LineCollections);
    # lines[2][0] is the horizontal-error LineCollection, whose segments are
    # [[x_min, y], [x_max, y]] per row, in input order (verified empirically
    # on matplotlib 3.11.1).
    segments = ax.containers[0].lines[2][0].get_segments()
    assert len(segments) == len(data.methods)
    for segment, lo, hi in zip(segments, data.lower, data.upper, strict=True):
        (x_min, _), (x_max, _) = segment
        assert x_min == pytest.approx(lo)
        assert x_max == pytest.approx(hi)


def test_drawing_returns_the_axes_it_was_given(ax):
    data = forest_plot_data([_estimate()])

    assert draw_forest_plot(data, ax) is ax


def test_drawing_does_not_touch_the_global_figure(ax):
    before = plt.get_fignums()
    draw_forest_plot(forest_plot_data([_estimate()]), ax)

    assert plt.get_fignums() == before


def _path():
    periods = np.arange(10)
    observed = np.linspace(10.0, 4.0, 10)
    predicted = np.linspace(10.0, 8.0, 10)
    return CounterfactualPath(
        periods=periods,
        observed=observed,
        predicted_absent_treatment=predicted,
        lower=predicted - 1.0,
        upper=predicted + 1.0,
        pointwise_effect=observed - predicted,
        cumulative_effect=np.cumsum(observed - predicted),
    )


def test_counterfactual_data_keeps_both_series_and_the_band():
    data = counterfactual_path_data(_path())

    assert np.allclose(data.observed, np.linspace(10.0, 4.0, 10))
    assert np.allclose(data.predicted, np.linspace(10.0, 8.0, 10))
    assert np.allclose(data.upper - data.lower, 2.0)


def test_counterfactual_band_is_labelled_as_the_counterfactuals_not_the_effects(ax):
    # The band is the uncertainty in the predicted-absent-treatment series.
    # Calling it the effect's interval would misstate what it shows.
    draw_counterfactual_path(counterfactual_path_data(_path()), ax)

    labels = " ".join(
        [t.get_text() for t in ax.get_legend().get_texts()]
    ).lower()
    assert "counterfactual" in labels
    assert "effect" not in labels


def test_drawing_a_counterfactual_path_draws_both_series(ax):
    draw_counterfactual_path(counterfactual_path_data(_path()), ax)

    assert len(ax.lines) >= 2
