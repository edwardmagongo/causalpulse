"""The grid that runs in the suite, and the bands it must stay inside.

The `ci` tier is small enough to gate every change and large enough that its
bands mean something. Both halves are set by measurement rather than by
estimate -- the parent spec's "approximately 2,000 fits" predates any of this
code, and the numbers in TOLERANCE_BANDS come from a real run whose date and
values are recorded beside them.

**Estimator selection, and why each exclusion was made** (measured
2026-09-05, this tree, 2 cells: cross-sectional scenario D at n=500 and panel
scenario P1 at n=150x6):

    estimator             1000 reps   200 reps
    difference_in_means   0.08s       0.04s
    lin_regression        0.26s       0.08s
    ipw                    1.43s       0.30s
    aipw                   11.77s      2.30s
    did_2x2                0.59s       0.15s
    twfe                   5.25s       1.09s
    event_study            5.63s       1.18s
    callaway_santanna      0.47s       0.12s

`causal_forest` (an econml doubly-robust forest with internal cross-fitting)
and `hierarchical_did` (a NumPyro/NUTS sampler) were excluded before this
table was even built: both are documented as costing minutes per fit in this
tree, not seconds, which would blow the ~2 minute budget on their own. They
stay available in `full`, which has no such budget.

`synthetic_control` and `bsts` are excluded because they need a donor-pool
registry this phase does not build; `route()` marks them `ineligible` in
every cell of this grid, so including them would add zero fitted rows at
zero cost -- there is nothing wrong with listing them, but they contribute
nothing, so they are left out of `CI_ESTIMATORS` for the same reason
scenario H is left out of the axes below.

Scenario H (missing outcomes) is excluded from `ci_config()`'s axes because
every estimator refuses it (Phase 11's missing-data refusal): a scenario
where 100% of rows are `refused` contributes no fitted rows and no bands,
only cost.

`bootstrap_reps=200` (not the `EstimatorConfig` default of 1000) because
cost is linear in it -- `aipw` alone falls from 11.77s to 2.30s per 2 cells
-- and because `TOLERANCE_BANDS` below is measured at 200 reps specifically.
`full` may run at the default 1000; this tier's bands are valid only at 200,
since a different bootstrap count changes the analytic CI width and hence
`coverage_analytic` and `mean_ci_width`, though not `bias` itself (bias does
not depend on the bootstrap at all -- it is a property of the point
estimate). Widening the bootstrap count without re-measuring would silently
run bands measured under one configuration against rows produced under
another.

**Measured run** (2026-09-05, `ci_config()` above, `CI_FIT_CONFIG` below):
`run_grid` took 96.7s, produced 600 rows (300 fitted, 300 ineligible, 0
refused), and `summarise` took under 0.1s. Comfortably inside the ~2 minute
budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cp_core.values import EstimatorConfig

from cp_evaluation.grid import CrossSectionAxes, GridConfig, PanelAxes


@dataclass(frozen=True)
class Band:
    """An inclusive range a metric must fall inside, and why it is that wide.

    `note` is not decoration. A band with no stated derivation gets widened
    the next time it flakes, and a band that has been widened twice asserts
    nothing.
    """

    metric: str
    low: float
    high: float
    note: str


# The estimators measured and shipped in the `ci` tier. See the module
# docstring for why each excluded estimator (causal_forest, hierarchical_did,
# synthetic_control, bsts) is missing from this list.
CI_ESTIMATORS: tuple[str, ...] = (
    "difference_in_means",
    "lin_regression",
    "ipw",
    "aipw",
    "did_2x2",
    "twfe",
    "event_study",
    "callaway_santanna",
)

# Cost is linear in bootstrap_reps; 200 rather than the EstimatorConfig
# default of 1000 is what keeps this tier inside its ~2 minute budget, and is
# the setting TOLERANCE_BANDS was measured at. See the module docstring.
CI_FIT_CONFIG = EstimatorConfig(random_seed=0, bootstrap_reps=200)


def ci_config() -> GridConfig:
    """The gating grid: 3 cross-sectional scenarios, 2 panel scenarios, 15 reps.

    Scenario H is left off both axes (see module docstring). Run with
    `CI_ESTIMATORS` and `CI_FIT_CONFIG`, not the defaults `run_grid` would
    otherwise use -- `TOLERANCE_BANDS` is only valid against that combination.
    """
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "B", "D"), n_units=(500,), confounding_strength=(1.0,)
        ),
        panel=PanelAxes(scenarios=("P1", "P2"), n_units=(150,), n_periods=(6,)),
        replications=15,
    )


# MEASURED 2026-09-05, `ci_config()` run with `estimators=CI_ESTIMATORS,
# fit_config=CI_FIT_CONFIG`, 15 replications per cell (n_fitted=15
# throughout unless noted). Every entry states the measured value, its MCSE,
# and the band's width in MCSE units.
#
# Discriminating-power check (the reason two candidate bands below are
# missing rather than present): the design document requires that a band be
# narrow enough to fail. The reference defect is Task 9's `seeded_biased`
# estimator, which adds a constant +0.25 to the point estimate regardless of
# scale (`tests/evaluation/test_instrument.py`, `BIAS = 0.25`). Because that
# defect is an absolute additive constant rather than one scaled to an
# estimand's own size, whether a band can reject it is scale-dependent: 0.25
# is 5x the ~0.05 cross-sectional true effect (a huge, unmissable relative
# shift) but only 25% of panel scenario P1's ~1.0 true effect (a much
# smaller one) -- so "the band is narrower than 0.25" has to be checked
# arithmetically per cell, not assumed from one example.
#
# For every candidate cell the check is: does `measured_bias + 0.25` (and,
# symmetrically, `measured_bias - 0.25`) fall outside `[low, high]`? Where it
# does not, shipping the band would read as coverage while catching nothing.
#
#   method                scenario  bias      mcse      +-4 mcse band     0.25 shift lands
#   difference_in_means   A         0.0198    0.00928    [-0.017, 0.057]  outside -- ships
#   difference_in_means   D         0.1429    0.01077    [0.100, 0.186]   outside -- ships
#   lin_regression        A         0.0136    0.00747    [-0.016, 0.043]  outside -- ships
#   lin_regression        D         0.0197    0.01310    [-0.033, 0.072]  outside -- ships
#   ipw                   A         0.0136    0.00746    [-0.016, 0.043]  outside -- ships
#   ipw                   D         0.0192    0.01278    [-0.032, 0.070]  outside -- ships
#   aipw                  A         0.0128    0.00726    [-0.016, 0.042]  outside -- ships
#   aipw                  D         0.0158    0.01466    [-0.043, 0.074]  outside -- ships
#   did_2x2               P1        0.0416    0.05013    [-0.159, 0.242]  outside -- ships (close)
#   twfe                  P1        0.0416    0.05013    [-0.159, 0.242]  outside -- ships (close)
#   callaway_santanna     P1        0.0931    0.07539    [-0.209, 0.395]  INSIDE  -- REJECTED
#   event_study           P1        0.0931    0.07539    [-0.209, 0.395]  INSIDE  -- REJECTED
#
# `callaway_santanna` and `event_study` on P1 share the same bias and MCSE
# (both are built on the same 2x2 comparison for a single post-period
# panel). Their +-4 MCSE half-width is 0.3016, which is *wider* than the
# 0.25 defect itself: a bug that shifted either estimator's point estimate
# by exactly the amount Task 9 proves this instrument can catch elsewhere
# would land inside this band, undetected, in either direction. No bias
# band is committed for either cell. This is a real loss of coverage for
# these two estimators on P1 -- not a gap to paper over with a wider
# replication count inside this tier's time budget -- and `full` is where
# a tier with enough replications to narrow their MCSE belongs.
#
# The null scenarios (B, P2, true effect 0) are not exempted from this
# check even though "0.25 relative to a true effect of 0" has no ratio to
# speak of: the check is still the literal arithmetic above, and every null
# cell's MCSE is small enough (0.0069-0.0483) that a 0.25 shift escapes.
#
# `difference_in_means` on D is also the strongest qualitative signal in the
# table: bias 0.143 and coverage_analytic 0.0667 are the naive estimator
# visibly failing on a confounded scenario, exactly as it should. Its
# coverage_analytic gets its own band below, not because the check above
# requires it, but because a naive estimator quietly becoming
# well-calibrated would itself be a regression worth catching (it would mean
# something changed about scenario D or about how `difference_in_means` is
# routed to it).
TOLERANCE_BANDS: Mapping[tuple[str, str], tuple[Band, ...]] = {
    ("difference_in_means", "A"): (
        Band(
            "bias",
            -0.017308,
            0.056913,
            "measured 0.0198, mcse 0.00928; +/- 4 mcse. A 0.25 shift lands at "
            "0.27, outside the band.",
        ),
    ),
    ("difference_in_means", "B"): (
        Band(
            "bias",
            -0.039396,
            0.034682,
            "measured -0.00236, mcse 0.00926; +/- 4 mcse. A 0.25 shift lands "
            "at 0.25 or -0.25, outside the band.",
        ),
    ),
    ("difference_in_means", "D"): (
        Band(
            "bias",
            0.0998,
            0.185928,
            "measured 0.1429, mcse 0.01077; +/- 4 mcse. A 0.25 shift lands at "
            "0.39, outside the band.",
        ),
        Band(
            "coverage_analytic",
            -0.190958,
            0.324291,
            "measured 0.0667, mcse 0.06441; +/- 4 mcse. This is the naive "
            "estimator's known confounded-scenario failure, held in place: "
            "if it ever climbed above ~0.32 -- e.g. towards nominal 0.95 -- "
            "that would mean scenario D's confounding or the routing to "
            "difference_in_means changed, not that the estimator got "
            "better.",
        ),
    ),
    ("lin_regression", "A"): (
        Band(
            "bias",
            -0.016297,
            0.043503,
            "measured 0.01360, mcse 0.00747; +/- 4 mcse. A 0.25 shift lands "
            "at 0.26, outside the band.",
        ),
    ),
    ("lin_regression", "B"): (
        Band(
            "bias",
            -0.025962,
            0.030878,
            "measured 0.00246, mcse 0.00710; +/- 4 mcse. A 0.25 shift lands "
            "at 0.25 or -0.25, outside the band.",
        ),
    ),
    ("lin_regression", "D"): (
        Band(
            "bias",
            -0.032724,
            0.072062,
            "measured 0.01967, mcse 0.01310; +/- 4 mcse. A 0.25 shift lands "
            "at 0.27, outside the band.",
        ),
    ),
    ("ipw", "A"): (
        Band(
            "bias",
            -0.016254,
            0.043406,
            "measured 0.01358, mcse 0.00746; +/- 4 mcse. A 0.25 shift lands "
            "at 0.26, outside the band.",
        ),
    ),
    ("ipw", "B"): (
        Band(
            "bias",
            -0.02587,
            0.030817,
            "measured 0.00247, mcse 0.00709; +/- 4 mcse. A 0.25 shift lands "
            "at 0.25 or -0.25, outside the band.",
        ),
    ),
    ("ipw", "D"): (
        Band(
            "bias",
            -0.031903,
            0.070327,
            "measured 0.01921, mcse 0.01278; +/- 4 mcse. A 0.25 shift lands "
            "at 0.27, outside the band.",
        ),
    ),
    ("aipw", "A"): (
        Band(
            "bias",
            -0.016209,
            0.041837,
            "measured 0.01281, mcse 0.00726; +/- 4 mcse. A 0.25 shift lands "
            "at 0.26, outside the band.",
        ),
    ),
    ("aipw", "B"): (
        Band(
            "bias",
            -0.024697,
            0.030348,
            "measured 0.00283, mcse 0.00688; +/- 4 mcse. A 0.25 shift lands "
            "at 0.25 or -0.25, outside the band.",
        ),
    ),
    ("aipw", "D"): (
        Band(
            "bias",
            -0.042841,
            0.074476,
            "measured 0.01582, mcse 0.01466; +/- 4 mcse. A 0.25 shift lands "
            "at 0.27, outside the band.",
        ),
    ),
    ("did_2x2", "P1"): (
        Band(
            "bias",
            -0.158905,
            0.242162,
            "measured 0.04163, mcse 0.05013; +/- 4 mcse. A 0.25 shift lands "
            "at 0.29, outside the band by 0.049 -- the closer of the two "
            "surviving panel bands, but it survives.",
        ),
    ),
    ("did_2x2", "P2"): (
        Band(
            "bias",
            -0.119801,
            0.101956,
            "measured -0.00892, mcse 0.02772; +/- 4 mcse. A 0.25 shift lands "
            "at 0.24 or -0.26, outside the band.",
        ),
    ),
    ("twfe", "P1"): (
        Band(
            "bias",
            -0.158905,
            0.242162,
            "measured 0.04163, mcse 0.05013; +/- 4 mcse. Identical cell to "
            "did_2x2 on P1 -- both are built on the same 2x2 comparison for "
            "a single post-period panel. A 0.25 shift lands at 0.29, "
            "outside the band by 0.049.",
        ),
    ),
    ("twfe", "P2"): (
        Band(
            "bias",
            -0.119801,
            0.101956,
            "measured -0.00892, mcse 0.02772; +/- 4 mcse. A 0.25 shift lands "
            "at 0.24 or -0.26, outside the band.",
        ),
    ),
    ("callaway_santanna", "P2"): (
        Band(
            "bias",
            -0.212804,
            0.173901,
            "measured -0.01945, mcse 0.04834; +/- 4 mcse. A 0.25 shift lands "
            "at 0.23 or -0.27, outside the band.",
        ),
    ),
    ("event_study", "P2"): (
        Band(
            "bias",
            -0.212804,
            0.173901,
            "measured -0.01945, mcse 0.04834; +/- 4 mcse. A 0.25 shift lands "
            "at 0.23 or -0.27, outside the band.",
        ),
    ),
    # No entry for ("callaway_santanna", "P1") or ("event_study", "P1"):
    # see the discriminating-power table above. Their bias, 0.0931, is
    # itself the true P1 sampling behaviour of these two estimators at
    # n_units=150, 6 periods, 15 replications -- but their MCSE (0.07539) is
    # wide enough that a band built from it cannot reject a defect of the
    # size this phase already knows how to seed. Shipping it would read as
    # coverage while providing none.
}
