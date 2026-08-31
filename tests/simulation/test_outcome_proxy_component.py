"""A proxy for the realised outcome, and the property that makes it useful."""

import numpy as np
import pytest
from cp_simulation.components import MissingnessMechanism, OutcomeProxy
from cp_simulation.scenarios import build_scenario


def test_noise_sd_must_be_positive():
    with pytest.raises(ValueError, match="noise_sd"):
        OutcomeProxy(noise_sd=0.0)


def test_the_proxy_column_reaches_the_frame_and_the_schema():
    dgp = build_scenario("D", outcome_proxy=OutcomeProxy(noise_sd=0.5))
    simulated = dgp.generate(n_units=500, rng=np.random.default_rng(0))
    assert "outcome_proxy" in simulated.dataset.frame.columns
    assert simulated.dataset.schema.auxiliary_columns == ("outcome_proxy",)
    # It is not a covariate, so no estimator can adjust for it.
    assert "outcome_proxy" not in simulated.dataset.schema.covariates
    assert "outcome_proxy" not in simulated.dataset.schema.required_columns


def test_the_proxy_is_observed_for_every_unit_including_the_missing_ones():
    # The point of the auxiliary is to carry information exactly where the
    # outcome does not. A proxy that went missing with the outcome would be
    # useless.
    dgp = build_scenario(
        "D",
        missingness=MissingnessMechanism(mechanism="MAR", rate=0.25, strength=2.0),
        outcome_proxy=OutcomeProxy(noise_sd=0.5),
    )
    simulated = dgp.generate(n_units=1000, rng=np.random.default_rng(0))
    frame = simulated.dataset.frame
    assert frame["outcome"].isna().any()
    assert not frame["outcome_proxy"].isna().any()


def test_the_proxy_carries_signal_beyond_treatment_and_covariates():
    # THE load-bearing property. A column correlated with the outcome only
    # THROUGH the covariates carries nothing an analysis model conditioning on
    # them lacks, and multiple imputation would gain nothing from it. Measured
    # as the correlation between the proxy and the outcome residual after
    # regressing the outcome on treatment and covariates.
    dgp = build_scenario("D", outcome_proxy=OutcomeProxy(noise_sd=0.5))
    simulated = dgp.generate(n_units=4000, rng=np.random.default_rng(0))
    frame = simulated.dataset.frame
    schema = simulated.dataset.schema
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame[schema.treatment].to_numpy(float),
            *[frame[c].to_numpy(float) for c in schema.covariates],
        ]
    )
    y = frame[schema.outcome].to_numpy(float)
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    correlation = float(np.corrcoef(residual, frame["outcome_proxy"].to_numpy(float))[0, 1])
    assert correlation > 0.5, correlation


def test_a_frame_without_a_proxy_is_byte_identical():
    # The Phase 10 and 11 precedent: an opt-in component must leave every
    # existing scenario producing exactly the frame it produced before. That
    # means every OTHER column's values, not just the proxy column's absence:
    # configuring a proxy must add a column and change nothing else. Compared
    # against a scenario WITHOUT a proxy at the same seed, rather than only
    # inspecting one frame in isolation, so a draw the proxy branch makes
    # before the shared columns are computed -- shifting them only when a
    # proxy is configured -- shows up as a mismatch here. (A draw added
    # unconditionally, on both branches alike, before this comparison begins,
    # would shift both frames identically and would NOT be caught by this
    # comparison; only pinning against independently known-correct values
    # could catch that shape of bug.)
    plain = build_scenario("D").generate(n_units=500, rng=np.random.default_rng(3))
    with_proxy = build_scenario("D", outcome_proxy=OutcomeProxy(noise_sd=0.5)).generate(
        n_units=500, rng=np.random.default_rng(3)
    )
    assert "outcome_proxy" not in plain.dataset.frame.columns
    assert plain.dataset.schema.auxiliary_columns == ()
    shared_columns = set(plain.dataset.frame.columns) & set(with_proxy.dataset.frame.columns)
    assert shared_columns == {"unit_id", "x1", "x2", "treated", "outcome"}
    for column in shared_columns:
        np.testing.assert_array_equal(
            plain.dataset.frame[column].to_numpy(),
            with_proxy.dataset.frame[column].to_numpy(),
            err_msg=f"column {column!r} differs when a proxy is configured",
        )


def test_the_proxy_does_not_disturb_the_missingness_mask():
    # The proxy is drawn AFTER the mask, from the saved complete vector, so
    # configuring one does not shift the random stream the mask draws from.
    # Without this the "with auxiliary" and "without auxiliary" arms of the
    # Task 10 harness would face different missing rows and the comparison
    # would be unpaired.
    mechanism = MissingnessMechanism(mechanism="MAR", rate=0.25, strength=2.0)
    without = build_scenario("D", missingness=mechanism).generate(
        n_units=1000, rng=np.random.default_rng(7)
    )
    with_proxy = build_scenario(
        "D", missingness=mechanism, outcome_proxy=OutcomeProxy(noise_sd=0.5)
    ).generate(n_units=1000, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(
        without.dataset.frame["outcome"].isna().to_numpy(),
        with_proxy.dataset.frame["outcome"].isna().to_numpy(),
    )


def test_the_component_changes_the_config_hash():
    # Two DGPs differing only by the proxy must not record the same
    # configuration hash, or a reproduction would silently rebuild the wrong
    # one.
    base = build_scenario("D")
    with_proxy = build_scenario("D", outcome_proxy=OutcomeProxy(noise_sd=0.5))
    assert with_proxy.config_hash != base.config_hash
