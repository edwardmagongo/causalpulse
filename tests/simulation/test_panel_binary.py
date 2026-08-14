import numpy as np
import pytest
from cp_simulation.components import UnitGenerator
from cp_simulation.panel_binary import BinaryPanelDGP
from cp_simulation.panel_components import (
    LevelSelectedAdoption,
    LinearTimePath,
    NormalUnitEffects,
    UniformAdoption,
)
from cp_simulation.panel_scenarios import PANEL_SCENARIOS, build_panel_scenario

N_UNITS, N_PERIODS, SEEDS = 800, 8, 120


def build(adoption, time_slope=0.1, intercept=-1.5, level_sd=0.8, tau_logit=0.45):
    return BinaryPanelDGP(
        unit_generator=UnitGenerator(covariate_names=("x1", "x2")),
        unit_effects=NormalUnitEffects(level_sd=level_sd, slope_sd=0.0),
        time_path=LinearTimePath(slope=time_slope),
        adoption=adoption,
        scenario_id="T",
        intercept=intercept,
        tau_logit=tau_logit,
    )


def two_by_two_bias(dgp, seeds=SEEDS):
    errors = []
    for seed in range(seeds):
        panel = dgp.generate(N_UNITS, N_PERIODS, np.random.default_rng(seed))
        frame = panel.dataset.frame
        ever = frame.groupby("unit_id")["treated"].transform("max").to_numpy()
        post = (frame["period"] >= 5).to_numpy().astype(float)
        y = frame["outcome"].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(post), ever, post, ever * post])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        errors.append(float(beta[3]) - panel.truth.true_att_overall)
    return float(np.mean(errors)), float(np.std(errors, ddof=1) / np.sqrt(seeds))


def test_the_outcome_is_binary_and_truth_is_on_the_probability_scale():
    panel = build(LevelSelectedAdoption(start=5)).generate(
        200, N_PERIODS, np.random.default_rng(0)
    )

    assert set(np.unique(panel.dataset.frame["outcome"])) <= {0.0, 1.0}
    assert panel.dataset.schema.outcome_type == "binary"
    # Measured on the probability scale, so it is not the logit-scale input.
    assert panel.truth.true_att_overall != pytest.approx(0.45, abs=0.2)
    assert 0.0 < panel.truth.true_att_overall < 0.2


def test_level_selected_adoption_produces_real_link_bias():
    bias, mcse = two_by_two_bias(build(LevelSelectedAdoption(start=5)))

    assert bias > 0.02, f"bias {bias:+.4f} (MCSE {mcse:.4f})"
    # Measured +0.0360 (MCSE 0.0020) at 800 units, 120 seeds, time_slope=0.1 --
    # the same slope every other panel scenario uses; no special parameter.
    # Headroom under the 0.02 threshold: (0.0360 - 0.02) / 0.0020 = 8.0 MCSE.
    assert bias / mcse > 4.0


def test_random_adoption_produces_no_link_bias():
    # The control condition. Without it, a P6 that measured nothing would look
    # exactly like a P6 that measured something. This is the ONE parameter that
    # is actually load-bearing here -- see the note below.
    bias, mcse = two_by_two_bias(build(UniformAdoption(start=5, share=0.5)))

    assert abs(bias) / mcse < 2.0, f"bias {bias:+.4f} (MCSE {mcse:.4f})"


def test_the_time_slope_is_load_bearing():
    # CORRECTED, 2026-08-14: an earlier version of this test asserted the
    # bias VANISHES at time_slope=0.1 (P1-P5's shared slope), on the theory
    # that a special time_slope=0.3 was required to reveal it. That theory was
    # measured against different unit-level parameters than the ones this
    # scenario ships with, and did not hold: at the actual parameters the bias
    # is non-monotonic in time_slope and changes sign across the range, and is
    # LARGER at 0.1 (+0.0360, 18.3 MCSE) than the discarded 0.3 (+0.0139, 7.0
    # MCSE) ever was. P6 now uses the shared slope -- no special parameter.
    #
    # What IS true, and what this test now guards: the bias requires
    # level-selected adoption. It is present at the shared slope, so a future
    # "simplification" that matches P1-P5's slope changes nothing (there is
    # nothing left to match). The one thing that must not change is the
    # adoption mechanism, which test_random_adoption_produces_no_link_bias
    # already guards. This test now asserts the bias SURVIVES at the shared
    # slope, so a regression that accidentally zeroes it out is caught.
    survives, survives_mcse = two_by_two_bias(
        build(LevelSelectedAdoption(start=5), time_slope=0.1), seeds=60
    )

    assert survives / survives_mcse > 4.0, (
        f"bias {survives:+.4f} (MCSE {survives_mcse:.4f})"
    )


def test_p6_is_registered_and_uses_the_shared_time_slope():
    spec = PANEL_SCENARIOS["P6"]
    dgp = build_panel_scenario("P6")

    assert spec.scenario_id == "P6"
    assert dgp.time_path.slope == 0.1
    assert isinstance(dgp.adoption, LevelSelectedAdoption)
