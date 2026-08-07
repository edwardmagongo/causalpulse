import pytest
from cp_statistics.proportions import newcombe_interval, wilson_interval
from hypothesis import given
from hypothesis import strategies as st


def test_wilson_matches_published_reference():
    # Textbook reference case: 1 success in 10 trials, 95% level.
    interval = wilson_interval(successes=1, trials=10)

    assert interval.lower == pytest.approx(0.017876, abs=1e-6)
    assert interval.upper == pytest.approx(0.404150, abs=1e-6)


def test_wilson_stays_inside_the_unit_interval_at_zero_successes():
    interval = wilson_interval(successes=0, trials=20)

    assert interval.lower == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= interval.upper <= 1.0


def test_wilson_stays_inside_the_unit_interval_at_full_successes():
    interval = wilson_interval(successes=20, trials=20)

    assert 0.0 <= interval.lower <= 1.0
    assert interval.upper == pytest.approx(1.0, abs=1e-12)


def test_newcombe_matches_published_worked_example():
    # Newcombe (1998) method 10 worked example: 56/70 versus 48/80.
    interval = newcombe_interval(
        successes_a=56, trials_a=70, successes_b=48, trials_b=80
    )

    assert interval.lower == pytest.approx(0.052431, abs=1e-6)
    assert interval.upper == pytest.approx(0.333873, abs=1e-6)


def test_newcombe_is_antisymmetric_under_arm_swap():
    forward = newcombe_interval(successes_a=56, trials_a=70, successes_b=48, trials_b=80)
    reversed_ = newcombe_interval(successes_a=48, trials_a=80, successes_b=56, trials_b=70)

    assert reversed_.lower == pytest.approx(-forward.upper, abs=1e-12)
    assert reversed_.upper == pytest.approx(-forward.lower, abs=1e-12)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(successes=11, trials=10)


def test_wilson_rejects_zero_trials():
    with pytest.raises(ValueError, match="trials"):
        wilson_interval(successes=0, trials=0)


@given(
    trials=st.integers(min_value=1, max_value=500),
    fraction=st.floats(min_value=0.0, max_value=1.0),
)
def test_wilson_always_contains_the_point_estimate(trials: int, fraction: float):
    successes = int(round(fraction * trials))
    interval = wilson_interval(successes=successes, trials=trials)

    assert interval.lower <= successes / trials <= interval.upper
    assert 0.0 <= interval.lower <= interval.upper <= 1.0
