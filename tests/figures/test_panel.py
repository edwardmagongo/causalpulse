import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cp_core.results import EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, Interval
from cp_figures.panel import EventStudyData, draw_event_study, event_study_data
from scipy.stats import norm


# Independently computed rather than imported from `cp_figures.panel`: the
# point of this test is to check the module's formula against the same one
# the rest of the project uses (`cp_causal._support.critical_value`,
# `cp_statistics.proportions`), not against itself. A function of `level`,
# not a single constant, because event_study_data must build its margin from
# whatever level the fit actually used (EstimatorConfig.ci_level), not a
# fixed 95%.
def _critical_value(level):
    return float(norm.ppf(1 - (1 - level) / 2))


_CRITICAL_VALUE = _critical_value(0.95)


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def _event_study_estimate(
    horizons=(-3, -2, -1, 0, 1, 2),
    coefficients=(0.1, 0.05, 0.0, 0.4, 0.6, 0.7),
    errors=(0.05, 0.04, 0.0, 0.06, 0.07, 0.08),
    level=0.95,
):
    def diag(name, values, unit):
        return Diagnostic(
            name=name, value=np.asarray(values, dtype=float), unit=unit, interpretation=name
        )

    return EffectEstimate(
        estimand="ATT",
        method="event_study",
        point=0.55,
        se=0.07,
        ci=Interval(lower=0.41, upper=0.69, level=level),
        ci_basis="analytic",
        bootstrap_ci=None,
        p_value=0.001,
        n_treated=50,
        n_control=100,
        scale="absolute",
        diagnostics={
            "event_time_horizons": diag("event_time_horizons", horizons, "periods"),
            "event_time_coefficients": diag("event_time_coefficients", coefficients, "effect"),
            "event_time_standard_errors": diag(
                "event_time_standard_errors", errors, "effect"
            ),
        },
        assumptions=(),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator="event_study", config_hash="h", runtime_seconds=0.1, random_seed=0
        ),
        subgroup_findings=(),
    )


def test_it_reads_the_three_event_time_arrays():
    data = event_study_data(_event_study_estimate())

    assert np.allclose(data.horizons, [-3, -2, -1, 0, 1, 2])
    assert np.allclose(data.coefficients, [0.1, 0.05, 0.0, 0.4, 0.6, 0.7])


def test_intervals_are_built_from_the_standard_errors():
    # Real fit (see cp_causal.estimators.event_study.EventStudy.fit, verified
    # against panel scenario P3 at n_units=150, n_periods=7): the project's
    # own two-sided 95% normal critical value is
    # norm.ppf(0.975) = 1.9599639845400545, not the textbook-rounded 1.96 the
    # brief's draft used. Using the rounded literal here would pass by
    # coincidence at atol=1e-9 only because these fixture errors are small;
    # matching the project's actual formula is what makes this test mean
    # anything.
    data = event_study_data(_event_study_estimate())

    assert np.allclose(
        data.upper - data.coefficients,
        _CRITICAL_VALUE * np.array([0.05, 0.04, 0.0, 0.06, 0.07, 0.08]),
        atol=1e-9,
    )
    assert np.allclose(
        data.coefficients - data.lower,
        _CRITICAL_VALUE * np.array([0.05, 0.04, 0.0, 0.06, 0.07, 0.08]),
        atol=1e-9,
    )


def test_it_carries_the_estimates_own_ci_level_not_a_hardcoded_95_percent():
    # EstimatorConfig.ci_level is configurable and EventStudy.fit honours it
    # throughout (event_study.py:258, :319, :360) -- a fit run at 90% must
    # not be silently widened to 95% here. A test that only ever exercises
    # the default 95% level could not tell a hardcoded constant from a field
    # actually read off the estimate, so this fixture is deliberately at a
    # non-default level.
    data = event_study_data(_event_study_estimate(level=0.90))

    assert data.level == pytest.approx(0.90)


def test_intervals_at_a_non_default_level_use_that_levels_critical_value():
    # Paired with test_intervals_are_built_from_the_standard_errors above at
    # the default 95% level: the two together are what make the margin
    # meaningfully level-dependent rather than a constant that happens to
    # equal the 95% critical value regardless of what estimate.ci.level
    # says. norm.ppf(0.95) = 1.6448536269514722, measurably narrower than
    # norm.ppf(0.975) -- a fixed 95% critical value used here by mistake
    # would produce margins about 19% too wide, not merely off by rounding.
    errors = np.array([0.05, 0.04, 0.0, 0.06, 0.07, 0.08])
    data = event_study_data(_event_study_estimate(level=0.90))

    expected_margin = _critical_value(0.90) * errors
    assert np.allclose(data.upper - data.coefficients, expected_margin, atol=1e-9)
    assert np.allclose(data.coefficients - data.lower, expected_margin, atol=1e-9)
    # The margin actually drawn is measurably different from what the
    # (wrong) hardcoded 95% critical value would have produced.
    assert not np.allclose(
        data.upper - data.coefficients, _CRITICAL_VALUE * errors, atol=1e-9
    )


def test_it_identifies_the_reference_horizon():
    # Verified against a real event_study fit (panel scenario P3,
    # n_units=150, n_periods=7): EventStudy.fit initialises both the
    # coefficient and standard-error arrays with np.zeros and never writes
    # to the reference column, so the reference horizon is the one whose
    # coefficient and standard error are both an exact (0.0, 0.0) -- not an
    # approximation.
    data = event_study_data(_event_study_estimate())

    assert data.reference_horizon == -1


def test_the_reference_horizon_needs_both_the_coefficient_and_the_se_at_zero():
    # A genuinely null pre-period can land its coefficient on exactly zero
    # without being the reference: horizon -2 here has coefficient 0.0 but a
    # real (nonzero) standard error, i.e. it was estimated and happened to
    # come out flat. Only horizon -1, with *both* fields exactly zero, is
    # the un-estimated column. A rule that matched on the coefficient alone
    # would pick two horizons here (or the wrong one).
    data = event_study_data(
        _event_study_estimate(
            horizons=(-2, -1, 0, 1),
            coefficients=(0.0, 0.0, 0.4, 0.6),
            errors=(0.03, 0.0, 0.06, 0.07),
        )
    )

    assert data.reference_horizon == -1
    assert list(data.is_reference) == [False, True, False, False]
    # The null-but-estimated horizon (-2) still gets a real interval.
    index = list(data.horizons).index(-2)
    assert data.upper[index] - data.lower[index] > 0.0


def test_the_reference_horizon_is_flagged_rather_than_given_an_interval():
    data = event_study_data(_event_study_estimate())

    index = list(data.horizons).index(data.reference_horizon)
    assert data.is_reference[index] is True
    assert all(v is False for i, v in enumerate(data.is_reference) if i != index)


def test_an_estimate_with_no_confidence_interval_is_refused_not_an_attributeerror():
    # event_study_data reads only estimate.ci.level (see the module
    # docstring), but that read still needs estimate.ci to exist.
    # EffectEstimate.ci is typed Interval, not Interval | None, and
    # __post_init__ never checks it, so dataclasses.replace(..., ci=None)
    # constructs successfully despite the static type -- the same hazard
    # cp_figures.guards.require_interval catches for forest_plot_data.
    # Verified empirically before this guard existed: event_study_data(
    # dataclasses.replace(estimate, ci=None)) raised a bare
    # "AttributeError: 'NoneType' object has no attribute 'level'" instead
    # of a clear refusal naming the figure.
    bare = dataclasses.replace(_event_study_estimate(), ci=None)

    with pytest.raises(ValueError, match="without a confidence level"):
        event_study_data(bare)


def test_an_estimate_without_the_event_time_diagnostics_is_refused():
    estimate = _event_study_estimate()
    stripped = type(estimate)(
        **{
            **{f.name: getattr(estimate, f.name) for f in estimate.__dataclass_fields__.values()},
            "diagnostics": {},
        }
    )

    with pytest.raises(KeyError, match="event_time_horizons"):
        event_study_data(stripped)


def test_no_reference_horizon_is_refused():
    # A genuinely null pre-period could, in principle, produce a coefficient
    # of exactly zero -- but not also a standard error of exactly zero,
    # which only happens for the one column EventStudy.fit never estimates.
    # If neither is present, this input was not produced by an estimator
    # that pins a reference this way, and guessing would misdraw the figure.
    estimate = _event_study_estimate(
        horizons=(-2, -1, 0, 1),
        coefficients=(0.1, 0.05, 0.4, 0.6),
        errors=(0.05, 0.04, 0.06, 0.07),
    )

    with pytest.raises(ValueError, match="exactly one reference horizon"):
        event_study_data(estimate)


def test_more_than_one_reference_horizon_is_refused():
    # Two horizons both landing on an exact (0.0, 0.0) pair is ambiguous
    # about which one is the structural reference; refusing is safer than
    # arbitrarily picking the first match.
    estimate = _event_study_estimate(
        horizons=(-2, -1, 0, 1),
        coefficients=(0.0, 0.0, 0.4, 0.6),
        errors=(0.0, 0.0, 0.06, 0.07),
    )

    with pytest.raises(ValueError, match="exactly one reference horizon"):
        event_study_data(estimate)


def test_drawing_marks_the_reference_horizon_without_an_error_bar(ax):
    # Drawing a band around a structural zero invites a reader to test a
    # hypothesis that cannot be false. It is not enough to show the
    # reference horizon is absent from the estimated-horizons container --
    # that alone would also pass if the reference horizon were dropped from
    # the figure entirely. Pin both halves: the reference horizon is drawn
    # (at zero, with no error bar) *and* every estimated horizon carries one.
    data = event_study_data(_event_study_estimate())

    draw_event_study(data, ax)

    assert len(ax.containers) == 2
    estimated_container, reference_container = ax.containers

    estimated_x = sorted(np.asarray(estimated_container[0].get_xdata()))
    expected_estimated = sorted(
        horizon
        for horizon, is_ref in zip(data.horizons, data.is_reference, strict=True)
        if not is_ref
    )
    assert estimated_x == expected_estimated
    assert data.reference_horizon not in estimated_x
    # `lines[2]` is the errorbar container's bar-line collections (see
    # test_effects.py's precedent). Non-empty means every estimated horizon
    # in this container actually got an error bar, not just a point.
    assert len(estimated_container.lines[2]) > 0

    reference_x = list(np.asarray(reference_container[0].get_xdata()))
    reference_y = list(np.asarray(reference_container[0].get_ydata()))
    assert reference_x == [data.reference_horizon]
    assert reference_y == [0.0]
    # An empty bar-line collection is what proves "no error bar" -- not
    # merely that this point lives in a different container from the rest.
    assert len(reference_container.lines[2]) == 0


def _asymmetric_event_study_data():
    # Constructed directly rather than through event_study_data: that
    # function always builds `lower`/`upper` as a symmetric
    # `coefficient +/- critical_value * se`, so it can never produce the
    # asymmetric margins this test needs. Each estimated horizon's lower
    # and upper margins differ from each other (asymmetric within a
    # horizon) and from every other horizon's margins (not a shared
    # constant) -- so a swapped `(lower_err, upper_err)` construction and a
    # constant-margin substitution both change the asserted numbers.
    return EventStudyData(
        horizons=np.array([-2.0, -1.0, 0.0, 1.0]),
        coefficients=np.array([0.10, 0.0, 0.40, 0.60]),
        lower=np.array([0.02, 0.0, 0.10, 0.55]),
        upper=np.array([0.25, 0.0, 0.50, 0.63]),
        is_reference=(False, True, False, False),
        reference_horizon=-1.0,
        level=0.95,
    )


def test_drawing_draws_error_bars_spanning_the_stored_interval(ax):
    # lower_err/upper_err are computed only inside draw_event_study (by
    # design), so the only way to catch a wrong magnitude or a swapped
    # (lower_err, upper_err) construction is to inspect the drawn whisker
    # geometry itself. See _asymmetric_event_study_data for why the fixture
    # must be asymmetric: a symmetric fixture would render identically
    # whether or not lower_err and upper_err were swapped.
    data = _asymmetric_event_study_data()

    draw_event_study(data, ax)

    estimated_container = ax.containers[0]
    # `lines[2][0]` is the vertical-error LineCollection; each segment is
    # [[x, y_min], [x, y_max]] per point, in input order (verified
    # empirically on matplotlib 3.11.1, the same way test_effects.py
    # verified the horizontal-error case).
    segments = estimated_container.lines[2][0].get_segments()
    is_reference = np.asarray(data.is_reference, dtype=bool)
    estimated = ~is_reference
    expected_lower = data.lower[estimated]
    expected_upper = data.upper[estimated]

    assert len(segments) == int(np.sum(estimated))
    for segment, lo, hi in zip(segments, expected_lower, expected_upper, strict=True):
        (_, y_min), (_, y_max) = segment
        assert y_min == pytest.approx(lo)
        assert y_max == pytest.approx(hi)


def test_drawing_draws_a_zero_reference_line(ax):
    # Distinguished from the single-point reference marker (a length-1 line)
    # and the errorbar caps (one point per estimated horizon, so length 5
    # here) by checking for a length-2 line spanning the axes at y=0 -- what
    # `ax.axhline` draws.
    data = event_study_data(_event_study_estimate())

    draw_event_study(data, ax)

    reference_lines = [
        line
        for line in ax.get_lines()
        if len(line.get_xdata()) == 2 and set(line.get_ydata()) == {0.0}
    ]
    assert len(reference_lines) == 1


def test_drawing_names_the_interval_level_in_the_legend(ax):
    # An unlabelled band leaves a reader unable to say what was rendered
    # (design section 11). The legend must name the actual level drawn, not
    # just say "estimated" -- and must change when the level does, which is
    # what distinguishes this from a hardcoded "95%" string in the label.
    data_95 = event_study_data(_event_study_estimate(level=0.95))
    draw_event_study(data_95, ax)
    labels_95 = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("95%" in label for label in labels_95)

    figure, ax_90 = plt.subplots()
    try:
        data_90 = event_study_data(_event_study_estimate(level=0.90))
        draw_event_study(data_90, ax_90)
        labels_90 = [t.get_text() for t in ax_90.get_legend().get_texts()]
        assert any("90%" in label for label in labels_90)
        assert not any("95%" in label for label in labels_90)
    finally:
        plt.close(figure)


def test_drawing_does_not_round_a_high_level_up_to_100_percent(ax):
    # EstimatorConfig.ci_level is validated only as 0.0 < ci_level < 1.0
    # (packages/cp-core/src/cp_core/values.py), so 0.999 is a reachable
    # level. f"{0.999:.0%}" rounds to "100%", which claims a certainty the
    # fit was never actually run at -- a level of 0.999 is not 1.0. The
    # legend must say "99.9%", not "100%".
    data = event_study_data(_event_study_estimate(level=0.999))

    draw_event_study(data, ax)

    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("99.9%" in label for label in labels)
    assert not any("100%" in label for label in labels)


def test_drawing_returns_the_axes_and_leaves_pyplot_alone(ax):
    before = plt.get_fignums()
    data = event_study_data(_event_study_estimate())

    assert draw_event_study(data, ax) is ax
    assert plt.get_fignums() == before
