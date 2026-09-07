import matplotlib.pyplot as plt
import numpy as np
import pytest
from cp_core.results import EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, Interval
from cp_figures.panel import draw_event_study, event_study_data
from scipy.stats import norm

# Independently computed rather than imported from `cp_figures.panel`: the
# point of this test is to check the module's constant against the same
# formula the rest of the project uses (`cp_causal._support.critical_value`,
# `cp_statistics.proportions`), not against itself.
_CRITICAL_VALUE = float(norm.ppf(0.975))


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def _event_study_estimate(
    horizons=(-3, -2, -1, 0, 1, 2),
    coefficients=(0.1, 0.05, 0.0, 0.4, 0.6, 0.7),
    errors=(0.05, 0.04, 0.0, 0.06, 0.07, 0.08),
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
        ci=Interval(lower=0.41, upper=0.69, level=0.95),
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


def test_drawing_returns_the_axes_and_leaves_pyplot_alone(ax):
    before = plt.get_fignums()
    data = event_study_data(_event_study_estimate())

    assert draw_event_study(data, ax) is ax
    assert plt.get_fignums() == before
