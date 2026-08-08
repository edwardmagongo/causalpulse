import numpy as np
import pytest
from cp_simulation.components import (
    BaselineSelectionAssignment,
    LogisticBaseline,
    PropensityAssignment,
    RandomAssignment,
    UnitGenerator,
)


def setup(n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = UnitGenerator(covariate_names=("x1", "x2")).generate(n, rng)
    covariates = frame[["x1", "x2"]].to_numpy()
    baseline = LogisticBaseline(intercept=-1.0, coefficients=(0.8, 0.5)).probabilities(
        covariates
    )
    return covariates, baseline, rng


def test_baseline_selection_correlates_treatment_with_baseline_outcome():
    covariates, baseline, rng = setup()

    treatment, propensity = BaselineSelectionAssignment(strength=8.0).assign(
        covariates, baseline, rng
    )

    assert np.corrcoef(treatment, baseline)[0, 1] > 0.2
    assert propensity.min() > 0.0 and propensity.max() < 1.0


def test_zero_strength_reduces_to_random_assignment():
    covariates, baseline, rng = setup()

    _, propensity = BaselineSelectionAssignment(strength=0.0).assign(
        covariates, baseline, rng
    )

    np.testing.assert_allclose(propensity, 0.5)


def test_share_sets_the_treated_fraction():
    # `share` is validated in __post_init__, so it must actually do something —
    # an earlier version accepted it and ignored it, silently giving 0.5 for
    # every value.
    covariates, baseline, _ = setup()

    means = [
        BaselineSelectionAssignment(strength=8.0, share=share)
        .assign(covariates, baseline, np.random.default_rng(1))[1]
        .mean()
        for share in (0.2, 0.5, 0.8)
    ]

    # Strictly increasing in `share`, and either side of 0.5 for the low and
    # high settings. Not asserted equal to `share`: with strength > 0 the mean
    # propensity is pulled toward 0.5 by expit's curvature, so it is only
    # approximately `share`. The exact case is pinned at strength 0 below.
    assert means[0] < means[1] < means[2]
    assert means[0] < 0.5 < means[2]


def test_zero_strength_gives_exactly_the_requested_share():
    covariates, baseline, rng = setup()

    _, propensity = BaselineSelectionAssignment(strength=0.0, share=0.3).assign(
        covariates, baseline, rng
    )

    np.testing.assert_allclose(propensity, 0.3)


def test_baseline_selection_reports_all_covariates_as_confounders():
    # Selection runs through the baseline, which every covariate feeds. An
    # analyst adjusting for the covariates can still identify the effect, so
    # reporting them is correct — reporting () would be the dangerous answer.
    assignment = BaselineSelectionAssignment(strength=8.0)

    assert assignment.confounders(("x1", "x2")) == ("x1", "x2")


def test_zero_strength_reports_no_confounders():
    assert BaselineSelectionAssignment(strength=0.0).confounders(("x1", "x2")) == ()


def test_existing_mechanisms_accept_but_ignore_the_baseline():
    covariates, baseline, rng = setup(n=5000)

    _, random_ps = RandomAssignment(share=0.5).assign(covariates, baseline, rng)
    _, prop_ps = PropensityAssignment(coefficients=(0.9, 0.7), strength=1.0).assign(
        covariates, baseline, rng
    )

    np.testing.assert_allclose(random_ps, 0.5)
    assert prop_ps.min() > 0.0


def test_negative_strength_is_rejected():
    with pytest.raises(ValueError, match="strength"):
        BaselineSelectionAssignment(strength=-1.0)
