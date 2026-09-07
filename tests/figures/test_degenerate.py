"""Degenerate inputs: the shapes a real analysis produces that a happy-path test never sees.

For every case here, the current implementation was checked empirically before this test was
written -- against real matplotlib and the real `cp_core` types, with warnings turned into errors
the same way the pytest suite runs -- and the acceptable behaviour it now pins is one of exactly
two: a clear refusal naming the problem, or a figure that draws correctly. A figure that reports
success while drawing nothing (or nothing that says what happened) is the outcome every test here
is written to catch, not merely to avoid raising on.

Three cases needed a fix (noted at each): `placebo_data` and `overlap_data` did not refuse an input
they cannot honestly draw, and `draw_counterfactual_path` silently produced a blank axes for a
single-period path (a markerless single point and a zero-width `fill_between` band are both
invisible, even though every artist call succeeds). The other five cases already behaved correctly
and are pinned as found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cp_core.results import (
    CounterfactualPath,
    EffectEstimate,
    FitMetadata,
    PlaceboTestResult,
    SpecificationCurve,
    SpecificationPoint,
)
from cp_core.values import Diagnostic, Interval
from cp_figures.balance import draw_love_plot, love_plot_data, overlap_data
from cp_figures.benchmark import draw_heatmap, heatmap_data
from cp_figures.effects import (
    counterfactual_path_data,
    draw_counterfactual_path,
    draw_forest_plot,
    forest_plot_data,
)
from cp_figures.panel import draw_event_study, event_study_data
from cp_figures.sensitivity import placebo_data
from matplotlib.collections import LineCollection, PolyCollection


@pytest.fixture
def ax():
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def _estimate(diagnostics=None, method="ipw", estimand="ATE", point=0.25, lo=0.05, hi=0.45):
    return EffectEstimate(
        estimand=estimand,
        method=method,
        point=point,
        se=0.1,
        ci=Interval(lower=lo, upper=hi, level=0.95),
        ci_basis="analytic",
        bootstrap_ci=None,
        p_value=0.01,
        n_treated=100,
        n_control=100,
        scale="percentage_point",
        diagnostics=diagnostics or {},
        assumptions=(),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator=method, config_hash="h", runtime_seconds=0.1, random_seed=0
        ),
        subgroup_findings=(),
    )


# --- An empty specification curve -------------------------------------------------------------


def test_an_empty_specification_curve_is_refused_before_it_reaches_cp_figures():
    # SpecificationCurve.__post_init__ (cp_core.results) already refuses this shape at
    # construction: "a specification curve must hold at least one refit". Checked empirically
    # before writing this test -- SpecificationCurve(specifications=(), original_index=0,
    # capped=False) never reaches specification_curve_data at all, because it cannot be built.
    # original_index=0 pointing into the empty tuple is caught by the same __post_init__'s second
    # check (`0 <= original_index < len(specifications)` is false when the tuple is empty), so
    # the "interesting part" the brief calls out is refused twice over, at the type boundary
    # cp_figures.sensitivity already sits behind. No change was needed in Task 7's module: the
    # decision was already implemented, just one layer below where the brief expected to find it.
    with pytest.raises(ValueError, match="at least one refit"):
        SpecificationCurve(specifications=(), original_index=0, capped=False)


def test_an_out_of_range_original_index_on_a_nonempty_curve_is_also_refused():
    # A companion case to the empty curve: a nonempty curve whose original_index still points
    # outside it. Same guard, different trigger, so it is worth pinning separately -- an
    # off-by-one in the bounds check could pass the empty-tuple case (0 <= 0 < 0 is false either
    # way) while missing this one.
    specs = (
        SpecificationPoint(
            specification={"trim": 0.01}, point=0.2, ci=Interval(lower=0.1, upper=0.3, level=0.95)
        ),
    )
    with pytest.raises(ValueError, match="original_index 1 is outside the 1 specifications"):
        SpecificationCurve(specifications=specs, original_index=1, capped=False)


# --- A single-estimate forest plot ------------------------------------------------------------


def test_a_single_estimate_forest_plot_still_draws_with_its_interval(ax):
    # Decision: draws correctly. forest_plot_data/draw_forest_plot have no code path that treats
    # one row specially, and a single-row errorbar with fmt="o" draws a visible point-and-whisker
    # the same as any row in a longer plot -- checked empirically before writing this test.
    data = forest_plot_data([_estimate(point=0.25, lo=0.05, hi=0.45)])

    draw_forest_plot(data, ax)

    assert len(ax.containers) == 1
    container = ax.containers[0]
    assert list(np.asarray(container[0].get_xdata())) == pytest.approx([0.25])
    # The whisker segment itself is drawn, not merely a bare point: lines[2] is the errorbar
    # container's bar-line collections (see test_effects.py's precedent), non-empty means a real
    # interval was drawn for this one row.
    segments = container.lines[2][0].get_segments()
    assert len(segments) == 1
    (x_min, _), (x_max, _) = segments[0]
    assert x_min == pytest.approx(0.05)
    assert x_max == pytest.approx(0.45)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["ipw (ATE)"]


# --- A heatmap whose every cell is absent -----------------------------------------------------


def _all_absent_summary():
    return pd.DataFrame(
        [
            {
                "method": "ipw",
                "scenario": "A",
                "bias": float("nan"),
                "n_fitted": 0,
                "n_ineligible": 15,
                "n_refused": 0,
            },
            {
                "method": "ipw",
                "scenario": "B",
                "bias": float("nan"),
                "n_fitted": 0,
                "n_ineligible": 15,
                "n_refused": 0,
            },
            {
                "method": "twfe",
                "scenario": "A",
                "bias": float("nan"),
                "n_fitted": 0,
                "n_ineligible": 15,
                "n_refused": 0,
            },
            {
                "method": "twfe",
                "scenario": "B",
                "bias": float("nan"),
                "n_fitted": 0,
                "n_ineligible": 15,
                "n_refused": 0,
            },
        ]
    )


def test_a_heatmap_with_every_cell_absent_draws_without_raising_or_a_scale(ax):
    # Decision: draws correctly -- an all-absent heatmap is a legitimate report ("nothing here
    # ran"), not an error. Checked empirically before writing this test, including that
    # ax.imshow on an entirely masked array raises no warning under the same
    # filterwarnings=error the real suite runs under (a masked-array normalisation edge case
    # that was worth checking rather than assuming).
    data = heatmap_data(_all_absent_summary(), metric="bias")
    assert bool(np.all(data.absent))

    draw_heatmap(data, ax)

    assert len(ax.images) == 1
    drawn = ax.images[0].get_array()
    assert isinstance(drawn, np.ma.MaskedArray)
    assert bool(np.ma.getmaskarray(drawn).all())
    # draw_heatmap never calls fig.colorbar; with every cell absent there is nothing on the
    # value scale to paint, and this pins that no colorbar axes gets added regardless -- a
    # future "only add a colorbar when there's something to show" branch would still need this
    # to hold for the all-absent case specifically.
    assert len(ax.figure.axes) == 1


# --- A placebo result with an empty placebo_effects array -------------------------------------


def test_placebo_data_refuses_zero_placebo_effects():
    # Decision: refuse. Checked empirically before writing this test: constructing
    # PlaceboTestResult(placebo_effects=np.array([]), real_effect=..., p_value=1.0) succeeds
    # (cp_causal.sensitivity._rank_p_value stacks the real effect alone against nothing and
    # ranks it against itself, so a zero-placebo run always produces p=1.0 no matter what the
    # real effect is), and drawing it with the pre-fix draw_placebo produced a histogram with 10
    # zero-height (invisible) bars next to a real-effect marker and a p-value that reads like an
    # ordinary result -- a figure that draws nothing while claiming success. placebo_data was
    # changed to refuse this shape instead; see cp_figures/sensitivity.py.
    result = PlaceboTestResult(placebo_effects=np.array([]), real_effect=0.25, p_value=1.0)

    with pytest.raises(ValueError, match="zero placebo effects"):
        placebo_data(result, statistic="absolute effect rank")


# --- An event study with no post-treatment horizons --------------------------------------------


def _event_study_no_post_treatment_horizons():
    def diag(name, values, unit):
        return Diagnostic(
            name=name, value=np.asarray(values, dtype=float), unit=unit, interpretation=name
        )

    return _estimate(
        method="event_study",
        diagnostics={
            "event_time_horizons": diag("event_time_horizons", [-3, -2, -1], "periods"),
            "event_time_coefficients": diag("event_time_coefficients", [0.1, 0.05, 0.0], "effect"),
            "event_time_standard_errors": diag(
                "event_time_standard_errors", [0.05, 0.04, 0.0], "effect"
            ),
        },
    )


def test_an_event_study_with_no_post_treatment_horizons_draws_correctly(ax):
    # Decision: draws correctly. event_study_data/draw_event_study identify the reference
    # horizon by its (coefficient, se) == (0.0, 0.0) pair, never by its sign or position relative
    # to zero, so an estimate whose horizons are all pre-treatment has no special code path to
    # fall into -- checked empirically before writing this test. Pins that this genuinely holds
    # for an all-negative-horizon estimate, not merely for the mixed pre/post fixture every other
    # event-study test in this package uses.
    estimate = _event_study_no_post_treatment_horizons()
    assert max(estimate.diagnostics["event_time_horizons"].value) < 0  # the premise itself

    data = event_study_data(estimate)
    assert data.reference_horizon == -1.0

    draw_event_study(data, ax)

    assert len(ax.containers) == 2
    estimated_container, reference_container = ax.containers
    estimated_x = sorted(np.asarray(estimated_container[0].get_xdata()))
    assert estimated_x == [-3.0, -2.0]
    # Every estimated (non-reference) horizon still carries a real error bar.
    assert len(estimated_container.lines[2][0].get_segments()) == 2
    # The reference horizon is still drawn, at zero, with no error bar.
    assert list(np.asarray(reference_container[0].get_xdata())) == [-1.0]
    assert len(reference_container.lines[2]) == 0


# --- A love plot with one covariate -------------------------------------------------------------


def test_a_love_plot_with_one_covariate_draws_correctly(ax):
    # Decision: draws correctly. love_plot_data's only refusal is a length mismatch between
    # covariate_names and the diagnostic arrays; a single covariate on both sides is a match of
    # length 1, and draw_love_plot's scatter/hlines calls have no minimum-length assumption --
    # checked empirically before writing this test.
    estimate = _estimate(
        diagnostics={
            "smd_before": Diagnostic(
                name="smd_before", value=np.array([0.5]), unit="smd", interpretation="x"
            ),
            "smd_after": Diagnostic(
                name="smd_after", value=np.array([0.05]), unit="smd", interpretation="x"
            ),
        }
    )

    data = love_plot_data(estimate, ["x1"])
    draw_love_plot(data, ax)

    assert [t.get_text() for t in ax.get_yticklabels()] == ["x1"]
    from matplotlib.collections import PathCollection

    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 2
    before_x, after_x = (s.get_offsets()[0][0] for s in scatters)
    assert before_x == pytest.approx(0.5)
    assert after_x == pytest.approx(0.05)


# --- A counterfactual path of length one --------------------------------------------------------


def _single_period_path():
    return CounterfactualPath(
        periods=np.array([0]),
        observed=np.array([10.0]),
        predicted_absent_treatment=np.array([9.0]),
        lower=np.array([8.0]),
        upper=np.array([10.0]),
        pointwise_effect=np.array([1.0]),
        cumulative_effect=np.array([1.0]),
    )


def test_a_counterfactual_path_of_length_one_draws_visibly(ax):
    # Decision: draws correctly, after a fix. Checked empirically before writing this test: the
    # pre-fix draw_counterfactual_path called plain ax.plot (no marker) for both series and
    # ax.fill_between for the band. For a single period, a markerless one-point line renders
    # nothing (a line needs two points to have a visible segment, and there is no marker to show
    # the lone point), and fill_between over one x-position fills a zero-width sliver -- also
    # invisible. Every artist call still succeeded and the legend still listed all three labels:
    # a figure that draws nothing while claiming success, for the whole plot. Fixed by adding
    # markers to both series and switching the band to ax.vlines when there is exactly one
    # period; see cp_figures/effects.py.
    data = counterfactual_path_data(_single_period_path())

    draw_counterfactual_path(data, ax)

    assert len(ax.lines) == 2
    for line in ax.lines:
        assert line.get_marker() == "o"
        assert list(line.get_xdata()) == [0]
    observed_line, predicted_line = ax.lines
    assert list(observed_line.get_ydata()) == [10.0]
    assert list(predicted_line.get_ydata()) == [9.0]

    # The band used vlines (a LineCollection with real vertical extent), not fill_between (a
    # zero-width PolyCollection for one x-position).
    assert len(ax.collections) == 1
    band = ax.collections[0]
    assert isinstance(band, LineCollection)
    assert not isinstance(band, PolyCollection)
    segments = band.get_segments()
    assert len(segments) == 1
    (x0, y0), (x1, y1) = segments[0]
    assert x0 == x1 == 0
    assert sorted((y0, y1)) == pytest.approx([8.0, 10.0])

    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts()).lower()
    assert "counterfactual" in labels


def test_a_counterfactual_path_of_two_or_more_periods_still_fills_a_band(ax):
    # Companion to the length-one fix: pins that the ordinary multi-period case still gets the
    # filled band (a PolyCollection), not vlines, so the len(periods) == 1 branch in
    # draw_counterfactual_path did not change behaviour for every other path in this package's
    # test suite.
    path = CounterfactualPath(
        periods=np.arange(3),
        observed=np.array([10.0, 9.0, 8.0]),
        predicted_absent_treatment=np.array([10.0, 9.5, 9.0]),
        lower=np.array([9.0, 8.5, 8.0]),
        upper=np.array([11.0, 10.5, 10.0]),
        pointwise_effect=np.array([0.0, -0.5, -1.0]),
        cumulative_effect=np.array([0.0, -0.5, -1.5]),
    )
    data = counterfactual_path_data(path)

    draw_counterfactual_path(data, ax)

    assert len(ax.collections) == 1
    assert isinstance(ax.collections[0], PolyCollection)


# --- An overlap figure where one arm is empty ---------------------------------------------------


def test_overlap_data_refuses_when_every_unit_is_treated():
    # Decision: refuse. Checked empirically before writing this test: the pre-fix overlap_data
    # happily split the scores into a nonempty treated array and an empty control array, and
    # draw_overlap's ax.hist(control_scores, ...) still added a "control" legend entry with zero
    # visible bars behind it -- a figure that looks like an ordinary (if narrow) two-arm
    # comparison rather than the total absence of a comparison group it actually is. Fixed by
    # having overlap_data refuse when either arm is empty; see cp_figures/balance.py.
    estimate = _estimate(
        diagnostics={
            "per_unit_propensity": Diagnostic(
                name="per_unit_propensity",
                value=np.array([0.5, 0.6, 0.7]),
                unit="probability",
                interpretation="x",
            ),
            "n_trimmed": Diagnostic(
                name="n_trimmed", value=0, unit="count", interpretation="x"
            ),
        }
    )
    treatment = np.array([1, 1, 1])

    with pytest.raises(ValueError, match="control arm has zero units"):
        overlap_data(estimate, treatment)


def test_overlap_data_refuses_when_every_unit_is_control():
    # The mirror of the all-treated case: names the other arm, so the message is not a
    # hardcoded "control" that would misreport an all-control input as the treated arm.
    estimate = _estimate(
        diagnostics={
            "per_unit_propensity": Diagnostic(
                name="per_unit_propensity",
                value=np.array([0.2, 0.3]),
                unit="probability",
                interpretation="x",
            ),
            "n_trimmed": Diagnostic(
                name="n_trimmed", value=0, unit="count", interpretation="x"
            ),
        }
    )
    treatment = np.array([0, 0])

    with pytest.raises(ValueError, match="treated arm has zero units"):
        overlap_data(estimate, treatment)
