import pytest

from cp_core.values import Diagnostic, EstimatorConfig, Interval


def test_interval_exposes_width_and_containment():
    interval = Interval(lower=0.021, upper=0.073)

    assert interval.level == 0.95
    assert interval.width == pytest.approx(0.052)
    assert interval.contains(0.05)
    assert not interval.contains(0.0)


def test_interval_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lower.*upper"):
        Interval(lower=0.5, upper=0.1)


def test_interval_rejects_out_of_range_level():
    with pytest.raises(ValueError, match="level"):
        Interval(lower=0.0, upper=1.0, level=1.5)


def test_interval_is_frozen():
    interval = Interval(lower=0.0, upper=1.0)
    with pytest.raises(AttributeError):
        interval.lower = 0.5  # type: ignore[misc]


def test_diagnostic_carries_interpretation():
    diagnostic = Diagnostic(
        name="pre_treatment_rmspe",
        value=0.031,
        unit="outcome units",
        interpretation="Pre-treatment fit is close; donor pool tracks the treated unit.",
    )

    assert diagnostic.name == "pre_treatment_rmspe"
    assert diagnostic.unit == "outcome units"


def test_estimator_config_defaults():
    config = EstimatorConfig()

    assert config.bootstrap_reps == 1000
    assert config.bootstrap_kind == "iid"
    assert config.ci_level == 0.95
    assert config.random_seed is None
    assert config.options == {}


def test_estimator_config_rejects_unknown_bootstrap_kind():
    with pytest.raises(ValueError, match="bootstrap_kind"):
        EstimatorConfig(bootstrap_kind="jackknife")
