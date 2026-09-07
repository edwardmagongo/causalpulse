import matplotlib.pyplot as plt
import numpy as np
import pytest
from cp_core.results import PlaceboTestResult, SpecificationCurve, SpecificationPoint
from cp_core.values import Interval
from cp_figures.sensitivity import (
    SpecificationCurveData,
    draw_placebo,
    draw_specification_curve,
    placebo_data,
    specification_curve_data,
)
from matplotlib.collections import PathCollection


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def _curve(capped=False):
    points = tuple(
        SpecificationPoint(
            specification={"trim": t},
            point=p,
            ci=Interval(lower=p - 0.1, upper=p + 0.1, level=0.95),
        )
        for t, p in ((0.01, 0.30), (0.02, 0.20), (0.05, 0.25))
    )
    return SpecificationCurve(specifications=points, original_index=1, capped=capped)


def _asymmetric_specification_curve_data(original_position=1):
    # Deliberately asymmetric lower/upper margins, none equal to each other:
    # a swapped (lower_err, upper_err) construction or a constant-margin
    # substitution both change the asserted numbers below.
    return SpecificationCurveData(
        points=(0.10, 0.25, 0.30),
        lower=(0.02, 0.20, 0.24),
        upper=(0.15, 0.28, 0.50),
        original_position=original_position,
        capped=False,
    )


def test_specification_data_is_sorted_by_point_estimate():
    data = specification_curve_data(_curve())

    assert list(data.points) == sorted(data.points)


def test_the_original_specification_is_tracked_through_the_sort():
    # original_index refers to the unsorted order. If the sort loses it, the
    # figure marks the wrong specification as the one actually reported.
    data = specification_curve_data(_curve())

    assert data.points[data.original_position] == pytest.approx(0.20)


def test_ties_in_point_estimate_do_not_break_original_tracking():
    # Two specifications land on the exact same point estimate. An
    # implementation that recovers original_position by *searching for the
    # original point's value* in the sorted list (rather than tracking the
    # original *index* through the sort) would find whichever tied entry
    # comes first and silently misreport which specification the caller's
    # original_index actually names -- distinguishable here only because the
    # two tied specifications carry different intervals.
    specs = (
        SpecificationPoint(
            specification={"trim": 0.01},
            point=0.20,
            ci=Interval(lower=0.05, upper=0.35, level=0.95),
        ),
        SpecificationPoint(
            specification={"trim": 0.02},
            point=0.20,
            ci=Interval(lower=0.15, upper=0.25, level=0.95),
        ),
        SpecificationPoint(
            specification={"trim": 0.03},
            point=0.10,
            ci=Interval(lower=0.00, upper=0.20, level=0.95),
        ),
    )
    curve = SpecificationCurve(specifications=specs, original_index=1, capped=False)

    data = specification_curve_data(curve)

    assert data.original_position == 2
    assert data.lower[data.original_position] == pytest.approx(0.15)
    assert data.upper[data.original_position] == pytest.approx(0.25)


def test_a_capped_curve_says_so():
    assert specification_curve_data(_curve(capped=True)).capped is True
    assert specification_curve_data(_curve(capped=False)).capped is False


def test_drawing_a_capped_curve_states_it_in_the_title(ax):
    # A capped curve drawn silently overstates how much of the specification
    # space was actually explored.
    draw_specification_curve(specification_curve_data(_curve(capped=True)), ax)

    assert "capped" in ax.get_title().lower()


def test_drawing_an_uncapped_curve_does_not_claim_capping(ax):
    draw_specification_curve(specification_curve_data(_curve(capped=False)), ax)

    assert "capped" not in ax.get_title().lower()


def test_drawing_draws_error_bars_spanning_the_stored_interval(ax):
    # lower_err/upper_err are computed only inside draw_specification_curve,
    # so the only way to catch a wrong magnitude or a swapped
    # (lower_err, upper_err) construction is to inspect the drawn whisker
    # geometry itself (see _asymmetric_specification_curve_data).
    data = _asymmetric_specification_curve_data()

    draw_specification_curve(data, ax)

    segments = ax.containers[0].lines[2][0].get_segments()
    assert len(segments) == len(data.points)
    for segment, lo, hi in zip(segments, data.lower, data.upper, strict=True):
        (_, y_min), (_, y_max) = segment
        assert y_min == pytest.approx(lo)
        assert y_max == pytest.approx(hi)


def test_drawing_marks_the_original_specification_distinctly(ax):
    # Step 3 of the brief requires the drawing to "mark the original" --
    # not merely carry original_position in the data. A figure that plotted
    # every specification identically would pass every other test here
    # while failing this requirement.
    data = _asymmetric_specification_curve_data(original_position=1)

    draw_specification_curve(data, ax)

    # Filtered to scatter's PathCollections specifically: `ax.errorbar` also
    # adds a LineCollection to `ax.collections` for the whiskers, which is
    # not part of what this test is checking.
    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 2
    other_points, original_point = scatters
    assert len(other_points.get_offsets()) == len(data.points) - 1

    original_offsets = original_point.get_offsets()
    assert len(original_offsets) == 1
    x, y = original_offsets[0]
    assert int(round(x)) == data.original_position
    assert y == pytest.approx(data.points[data.original_position])


def test_drawing_returns_the_axes_and_leaves_pyplot_alone(ax):
    before = plt.get_fignums()
    data = specification_curve_data(_curve())

    assert draw_specification_curve(data, ax) is ax
    assert plt.get_fignums() == before


def test_placebo_data_keeps_the_real_effect_and_the_distribution():
    result = PlaceboTestResult(
        placebo_effects=np.array([0.01, -0.02, 0.03, 0.00]),
        real_effect=0.25,
        p_value=0.02,
    )

    data = placebo_data(result, statistic="absolute effect rank")

    assert data.real_effect == pytest.approx(0.25)
    assert len(data.placebo_effects) == 4
    # Values, not just count: a bug that substituted a different array (or
    # dropped/reordered entries) of the same length would still pass a
    # length-only check.
    assert np.allclose(data.placebo_effects, [0.01, -0.02, 0.03, 0.00])
    assert data.p_value == pytest.approx(0.02)
    assert data.statistic == "absolute effect rank"


def test_placebo_data_requires_the_statistic_to_be_named():
    # p_value is not one statistic across producers -- three falsification
    # slots rank the absolute effect, the donor-pool slot uses Abadie's
    # RMSPE ratio rank -- and the type's own docstring says the two are not
    # comparable. A figure that implied one shared meaning would be wrong.
    result = PlaceboTestResult(placebo_effects=np.array([0.01]), real_effect=0.25, p_value=0.5)

    with pytest.raises(TypeError):
        placebo_data(result)


def test_placebo_data_rejects_the_statistic_as_a_positional_argument():
    # Keyword-only, not merely required: a positional call would still let a
    # caller pass a wrong-looking argument order by accident.
    result = PlaceboTestResult(placebo_effects=np.array([0.01]), real_effect=0.25, p_value=0.5)

    with pytest.raises(TypeError):
        placebo_data(result, "absolute effect rank")


def test_drawing_a_placebo_figure_names_the_statistic(ax):
    result = PlaceboTestResult(
        placebo_effects=np.array([0.01, -0.02, 0.03]), real_effect=0.25, p_value=0.02
    )

    draw_placebo(placebo_data(result, statistic="RMSPE ratio rank"), ax)

    text = (ax.get_title() + " " + ax.get_xlabel()).lower()
    assert "rmspe ratio rank" in text


def test_drawing_a_placebo_figure_names_a_different_statistic_differently(ax):
    # Pins the label to the *caller's* statistic, not a hardcoded string a
    # naive implementation might have baked in after seeing the test above.
    result = PlaceboTestResult(
        placebo_effects=np.array([0.01, -0.02, 0.03]), real_effect=0.25, p_value=0.02
    )

    draw_placebo(placebo_data(result, statistic="absolute effect rank"), ax)

    text = (ax.get_title() + " " + ax.get_xlabel()).lower()
    assert "absolute effect rank" in text
    assert "rmspe ratio rank" not in text


def test_drawing_a_placebo_figure_marks_the_real_effect(ax):
    result = PlaceboTestResult(
        placebo_effects=np.array([0.01, -0.02, 0.03]), real_effect=0.25, p_value=0.02
    )

    draw_placebo(placebo_data(result, statistic="absolute effect rank"), ax)

    verticals = [line.get_xdata()[0] for line in ax.lines if len(set(line.get_xdata())) == 1]
    assert any(abs(v - 0.25) < 1e-9 for v in verticals)


def test_drawing_a_placebo_figure_marks_a_different_real_effect_at_the_right_place(ax):
    # A fixed/hardcoded marker position would pass the test above too; move
    # the real effect and check the marker moves with it.
    result = PlaceboTestResult(
        placebo_effects=np.array([0.01, -0.02, 0.03]), real_effect=-0.40, p_value=0.02
    )

    draw_placebo(placebo_data(result, statistic="absolute effect rank"), ax)

    verticals = [line.get_xdata()[0] for line in ax.lines if len(set(line.get_xdata())) == 1]
    assert any(abs(v - (-0.40)) < 1e-9 for v in verticals)
    assert not any(abs(v - 0.25) < 1e-9 for v in verticals)


def test_drawing_placebo_returns_the_axes_and_leaves_pyplot_alone(ax):
    before = plt.get_fignums()
    result = PlaceboTestResult(
        placebo_effects=np.array([0.01, -0.02, 0.03]), real_effect=0.25, p_value=0.02
    )
    data = placebo_data(result, statistic="absolute effect rank")

    assert draw_placebo(data, ax) is ax
    assert plt.get_fignums() == before
