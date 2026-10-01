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

`causal_forest` (this repo's own honest causal forest, with hand-rolled
recursive splitting and internal cross-fitting) and `hierarchical_did` (a
NumPyro/NUTS sampler) were excluded before this
table was even built: both are documented as costing minutes per fit in this
tree, not seconds, which would blow the ~2 minute budget on their own. They
stay available in `full`, which has no such budget.

`bsts` is excluded from `CI_ESTIMATORS` for the same reason `causal_forest` and
`hierarchical_did` are: it is a NumPyro/NUTS sampler, the same cost class,
measured at ~17.5s/fit (`bsts.py`'s own `PRODUCTION_MCMC_CONFIG` docstring) --
enough to blow the ~2 minute budget on its own. `synthetic_control` (SLSQP +
permutation over ~17 donors) is excluded from `ci` on a different basis: it
has no committed `TOLERANCE_BANDS`, so in `ci` it adds cost for zero
assertions. MEASURED 2026-09-26, this tree: the `ci` tier took 338s with the
donor-pool arm (all six scenarios, 15 replications) and 257s without it, so
the arm's marginal cost was ~81s (+~32%), ~1.1s per fitted row over 75
fitted rows. The donor-pool arm therefore lives in `full`/`study` only;
`ci_config()` has no `donor_pool`, and its cell set is identical to what it
was before the donor-pool shape existed. Deriving `synthetic_control` bands
(with the same discriminating-power check as the other bands) is follow-up
work. Caveat: these 2026-09-26 timings were taken while another Claude
session ran a multi-hour CPU-heavy benchmark study on the same machine, so
they are likely inflated by contention and should be re-measured on an idle
machine. The per-fit figures quoted in this module are not comparable with
each other: ~1.1s per `synthetic_control` fitted row (ci, derived from the 81s
delta, `bootstrap_reps=200`), ~20.6s average per fitted donor row in
`full_config()`'s docstring (averaged over both estimators, at
`bootstrap_reps=1000`), and `bsts.py`'s ~17.5s/fit (a different environment
and config) differ in estimator mix, configuration and date.

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
budget. (Historical, and superseded: on 2026-09-26 the `ci` tier without any
donor-pool arm measured 257s, so the ~2 minute claim is stale independent of
the donor-pool arm.)

**Discriminating power is proportional to each arm's effect size, not
absolute.** The comment above `TOLERANCE_BANDS` checks every shipped band
against Task 9's `seeded_biased` instrument, a constant `+0.25` shift
(`tests/evaluation/test_instrument.py`, `BIAS = 0.25`) added to the point
estimate regardless of scale. That defect is not proportionally comparable
across arms: the cross-sectional scenarios' true effects are ~0.05, so 0.25
is a ~500% relative shift there, while panel scenario P1's true effect is
~1.0, so the same 0.25 is a ~25% relative shift there. "Rejects a 0.25
shift" therefore means something much easier in the panel arm than in the
cross-sectional arm.

Scaled to the same ~25%-of-effect severity that justified rejecting the two
panel bands (`callaway_santanna`/`event_study` on P1, see below), a
proportionally comparable cross-sectional defect is ~0.0125 (25% of the
~0.05 true effect). Every shipped cross-sectional bias band's +/-4-mcse
half-width (0.028-0.059, across scenarios A and D) is 2.2x-4.7x *wider*
than that. So at 15 replications, the cross-sectional bands catch gross
defects -- several times the true effect -- but not a proportionally subtle
one. The panel `did_2x2`/`twfe` bands on P1, by contrast, have a half-width
of 0.2005 (4 * mcse 0.05013, same absolute units as bias), which is
narrower than their own 25%-of-effect defect (0.25): they do pass the
proportional check.

This is a consequence of the ~2 minute runtime budget, not of the metric
itself: MCSE scales as `1 / sqrt(replications)`, so tightening the
cross-sectional bands to resolve a ~0.0125 defect would need roughly
`(2.2)**2 ~ 5x` (best case) to `(4.7)**2 ~ 22x` (worst case) today's 15
replications -- on the order of 75 to 330 replications per cell -- which
does not fit inside a ~2 minute gate. `full`, which carries no such
budget, is where that resolution belongs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from cp_core.values import EstimatorConfig

from cp_evaluation.grid import CateAxes, CrossSectionAxes, DonorPoolAxes, GridConfig, PanelAxes

# All six donor-pool scenarios, in full/study -- clean recovery
# (S0, S1) and adversarial ones (S2-S5) alike, the same mix A-H and P1-P7
# already are. Used by full/study only; ci carries no donor-pool arm, on
# measured cost -- see the module docstring.
_ALL_DONOR_POOL_SCENARIOS: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4", "S5")

# Every CATE scenario, sorted as `expand_cells` sorts them. NULL included: its
# RMSE measures invented heterogeneity (design decision 3).
_ALL_CATE_SCENARIOS: tuple[str, ...] = ("NOISY", "NULL", "RICH")


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

    No donor-pool arm: excluded on measured cost (see module docstring).
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
# "ships" below means only "rejects this absolute +/-0.25 shift" -- it is
# not a claim that the cross-sectional and panel arms are discriminating
# proportionally to the same degree. They are not: see the module
# docstring's "Discriminating power is proportional to each arm's effect
# size, not absolute" section for the proportional comparison (the
# cross-sectional bands resolve only a several-times-true-effect defect;
# the panel did_2x2/twfe bands on P1 also resolve a proportionally-scaled
# one).
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


# The estimators `ci` drops on cost are back: `full` runs on demand and has
# no two-minute budget. synthetic_control and bsts are both included here;
# neither is in `ci`: synthetic_control because it has no committed bands (its
# ~81s marginal cost is the measurement), bsts on its NUTS cost (see the module
# docstring).
FULL_ESTIMATORS: tuple[str, ...] = (
    "difference_in_means",
    "lin_regression",
    "ipw",
    "aipw",
    "causal_forest",
    "did_2x2",
    "twfe",
    "event_study",
    "callaway_santanna",
    "hierarchical_did",
    "synthetic_control",
    "bsts",
)

# The EstimatorConfig default. `ci`'s 200 is a concession to its two-minute
# runtime budget, and a tier run on demand should not inherit it.
FULL_FIT_CONFIG = EstimatorConfig(random_seed=0, bootstrap_reps=1000)


def full_config() -> GridConfig:
    """The full study grid: every registered estimator against a wide axis set,
    at `bootstrap_reps=1000`, run on demand rather than on every change.

    This grid was measured in this tree (2026-09-05/06), not copied from the
    parent spec's own §5 proposal, because running that proposal exposed three
    problems in it:

    1. **`n_periods=6` is invalid for four of the seven panel scenarios.**
       The spec proposes `periods in {6, 10}` uniformly. Probing each
       scenario's registry found the minimum viable `n_periods` is 5 for P1,
       P2, and P6, but 7 for P3, P4, and P5, and 16 for P7. At
       `n_periods=6`, P3/P4/P5 raise `ValueError: adoption periods must each
       lie in [2, 6] so that at least one pre-period and one post-period
       exist`. This tier therefore uses `n_periods=(7, 10)`, the smallest
       pair that is valid for every scenario it keeps.

    2. **Scenario P7 cannot share an axis with P1-P6 and is excluded.** Its
       own registry docstring documents that it is generated "at this
       scenario's own longer protocol: HIER_N_UNITS=600, HIER_N_PERIODS=20"
       (`cp_simulation/panel_scenarios.py`), and needs `n_periods >= 16`.
       `PanelAxes` carries one `n_periods` list applied to every scenario in
       it -- there is no per-scenario override -- so including P7 would force
       P1-P6 onto a 16+-period protocol they were not designed for, at a
       large multiple of their measured cost, purely to accommodate one
       scenario. P7 is dropped from `full_config()` instead; a reader
       wondering where it went should read this paragraph. `ci_config()`
       never surfaced this because it only ever used P1 and P2.

    3. **The spec's 50 replications, n_units=(150, 500), and n_periods=(6, 10)
       do not fit inside a reasonable on-demand budget.** Measured per-fit
       cost at `bootstrap_reps=1000` in this tree: most estimators are
       sub-second to ~2s/fit, but `aipw` is 6.47s, `causal_forest` 2.01s,
       `twfe` 11.92s, `event_study` 12.32s, and `hierarchical_did` (a NUTS
       sampler) is 24.8s/fit at panel n_units=150 and 79.7s/fit at
       n_units=500. Extrapolating the spec's proposed grid from these
       per-fit numbers puts one full run at roughly 35 hours, about 30 of
       them from the panel arm alone. Two axes were reduced in response:

       - **Panel `n_units=500` is dropped**, keeping only `n_units=(150,)`.
         `hierarchical_did` alone costs 79.7s/fit at n_units=500 against
         24.8s/fit at n_units=150 -- more than tripling the panel arm's cost
         for one axis value, on top of already dropping P7 and moving to
         `n_periods=(7, 10)`.
       - **Replications is 10, not the spec's 50.** This was chosen as the
         largest round number that kept the tier's runtime to a self-chosen
         few hours, not because a ceiling is owed to it -- see the "no
         runtime ceiling" paragraph below.

    **The four axes Phase 18 adds** (`treatment_intercept`, `noise_scale`,
    `effect_scale`, `heterogeneity_scale`) are set here at three measured
    points each -- a low, a baseline, and a high value. See
    `docs/specs/2026-09-13-causal-core-phase18-design.md` section 12 for the
    full measurement record, including the values rejected along the way.
    One value was respaced after the first measurement pass: `noise_scale`'s
    high point is **1.3**, not the 1.0 first adopted. 0.9 and 1.0 move
    residual outcome variation by +0.011 and +0.012 respectively -- a gap
    roughly a tenth of either one's gap from the 0.0 baseline -- so as a
    three-point axis they resolved two levels, not three, and the axis could
    show only *whether* an estimator degrades with noise rather than the
    shape of it. 1.3 moves it by +0.018, spacing the three points about
    evenly, while saturating the `[0.05, 0.85]` baseline clip at 8.06%/5.03%
    -- still well under the 16.1%/12.0% that had `noise_scale >= 2.0`
    rejected for reading as "mostly pinned at a clip bound" rather than as
    noisier.

    **A known caveat on `effect_scale`, disclosed rather than fixed.** Its
    low point of 0.5 moves measured `true_ate` by 0.0150 for scenario C and
    0.0193 for F -- 0.75x and 0.96x of the 0.02 floor
    `tests/simulation/test_axis_declarations.py` uses, i.e. below it. That
    floor exists to verify an axis is *wired*, and is measured against that
    test's own deliberately large `_PERTURBED_VALUE` of 3.0, not against any
    grid value; a 1.5pp effect on C is a legitimate thing for a benchmark to
    probe, not a degenerate cell. The design doc's section 12 records why
    0.5 shipped anyway and names the alternative considered (~0.3, which
    would clear the floor for both C and F but was not taken).

    Each axis is varied **one at a time** against the other three sitting at their
    `ScenarioParams` default (see `grid._expand_cross_section`), and only for
    the scenarios that declare it (`SCENARIOS[id].axes`), so this widens the
    cross-sectional arm from 26 to **68 cells**. Per scenario, the count is
    `|n_units| x (|confounding_strength| if declared else 1)` for the baseline
    block plus `(|values| - 1)` for each declared new axis:

        scenario   declares cs   baseline      one-at-a-time   total
        A          no            2 x 1 =  2    3 x 2 = 6           8
        B          no            2 x 1 =  2    2 x 2 = 4           6
        C          no            2 x 1 =  2    3 x 2 = 6           8
        D          yes           2 x 3 =  6    3 x 2 = 6          12
        F          no            2 x 1 =  2    4 x 2 = 8          10
        G          yes           2 x 3 =  6    3 x 2 = 6          12
        H          yes           2 x 3 =  6    3 x 2 = 6          12
                                          26           42          68

    Phase 18 itself shipped this as 42 to 84, because `confounding_strength`
    was then exempt from the declaration rule and all 7 scenarios were crossed
    against all 3 of its values. A follow-up closed that exemption (design doc
    section 12): A, B, C and F assign at random and never read the axis, so
    each now contributes one baseline strength per `n_units` rather than
    three, and both ends of the range drop by the same 16 cells. See section
    12 for the full derivation and the per-axis headroom/degenerate-value
    checks behind each chosen value.

    `ci_config()` is untouched by either change: its own four new-axis fields
    stay at their single-valued `CrossSectionAxes` default, and its
    `confounding_strength` axis is `(1.0,)` -- which is `ScenarioParams`' own
    default, so the strength a gated scenario collapses onto is the only
    strength it ever asked for. Every cell identity, and so every `cell_rng`
    stream, is what `TOLERANCE_BANDS` was measured against.

    There is no runtime ceiling this widening needs to clear: this tier
    "runs on demand and has no two-minute budget" (that budget belongs to
    `ci_config()`, and "a tier run on demand should not inherit it" -- see
    this function's own return value and `CI_FIT_CONFIG` above). It is
    reported for planning purposes only. **16 of the 42 new cells are
    directly timed at the values this tier actually ships** -- scenarios B
    (4 of 4), D (6 of 6) and F (6 of 8) -- at
    `n_units=2000`/`FULL_ESTIMATORS`/`FULL_FIT_CONFIG`, landing in an
    11.68s-12.75s band regardless of which axis or value was varied. The
    remaining 26 are extrapolated at the same 12.09s/cell average: scenarios
    A, C, G and H entirely (6 + 6 + 6 + 6 = 24 cells, never timed), plus F's
    two `noise_scale` cells, timed only at a superseded 0.5 and at the 1.0
    that 1.3 replaced. Counting those superseded runs would give 20; they
    are excluded because a timing at a value this tier does not ship is not
    a measurement of this tier. Extra cross-sectional cost: ~507.8s/replication
    (42 x 12.09s). Combined with the pre-existing arms (MEASURED 2026-09-06,
    below): ~189.7s (the baseline cross-sectional block, now 26 cells) +
    507.8s (42 new cells) + 419.1s (panel, unchanged) = ~1116.6s/replication,
    or ~3.10 hours at the 10 replications shipped here -- up from ~2.1 hours
    before this widening, and down from the ~3.56 hours Phase 18 shipped,
    which crossed all 7 scenarios against all 3 strengths.

    **Where ~189.7s comes from, and why it is not 353.4s x 26/42.** MEASURED
    2026-09-18, this tree, same `run_grid`/`FULL_ESTIMATORS`/`FULL_FIT_CONFIG`
    call as the 2026-09-06 run below, warm, after a discarded warm-up run: the
    26 kept cells took 227.1s and the 16 the gating removes took 196.0s, so
    the whole 42-cell block took 423.1s and the gating saves 46.3% of it. That
    machine is slower than the one the 353.4s below was measured on (423.1s
    for the identical 42 cells, a factor of 1.20), so what carries across is
    the **ratio**, not the absolute: 353.4s x (227.1 / 423.1) = ~189.7s, which
    keeps the whole projection on the 2026-09-06 basis the other two arms are
    stated in.

    The removed cells cost 12.25s each against 8.73s for the kept ones, so
    they are *more* expensive than the arm average -- which is why the design
    doc's first estimate of this saving (16 x the 8.4s/cell whole-arm average
    = ~135s/replication, ~22 minutes per run) understated it by about a fifth.
    The real figure is ~163.7s/replication, ~27 minutes across a
    10-replication run. Scenario H is part of the reason: all 6 of its cells
    are kept, since H declares `confounding_strength`, and none of its rows
    is a fit -- of its 60 rows, half are missing-data refusals and half are
    `ineligible` (the panel-only estimators routed away) -- so it costs almost
    nothing. It is not the whole reason: the kept block also holds 8 A/B/C/F
    baseline cells at the removed cells' 12.25s, which leaves D, G and H's 18
    cells to account for the remaining ~129s, ~7.2s each, and D and G alone
    would still average ~10.8s even if H were free. D and G are themselves
    cheaper per cell than the random-assignment scenarios the gating removed
    -- all 16 of which are fully-fitting cells.

    MEASURED 2026-09-06, this tree, `run_grid(config, estimators=list(
    FULL_ESTIMATORS), fit_config=FULL_FIT_CONFIG)` at `replications=1`, for
    the pre-widening 42-cell cross-sectional arm (7 scenarios x
    n_units=(500, 2000) x confounding_strength=(0.5, 1.0, 2.0)): 353.4s,
    420 rows (180 fitted, 210 ineligible, 30 refused -- the refusals are
    scenario H's missing outcomes, exactly as expected); the panel arm (6
    scenarios x n_units=(150,) x n_periods=(7, 10) = 12 cells) took 419.1s
    and produced 120 rows (54 fitted, 66 ineligible, 0 refused -- the
    ineligible rows are the five cross-sectional-only estimators routed
    away from every panel cell, plus a handful of panel-estimator/
    cell-shape mismatches). Neither arm raised. The 42 new cross-sectional
    cells this widening adds were not re-run through this exact harness;
    their cost is the extrapolation described above, not a fresh
    measurement of this exact `FULL_ESTIMATORS`/`FULL_FIT_CONFIG`
    combination -- see the design doc's section 12 for what was and was not
    directly timed.

    Scenario H (missing outcomes) is included, unlike in `ci_config()`: its
    rows are all refusals, which contributes no fitted rows to any band but
    is worth recording once in a full study even though `ci` leaves it out
    to save cost.

    The donor-pool arm adds all six scenarios (S0-S5). A cell does not include
    the estimator, so at this tier's 10 replications the arm is 60 cells (6
    scenarios x 10; the 800 -> 860 change in the grid's cell count).
    `fit_cell` writes one row per candidate estimator and `FULL_ESTIMATORS` has
    12 entries, so those 60 cells are 720 estimator rows (3,600 at
    `study_config()`'s 50, from 300 cells). Most are ineligible: three
    estimators are FITTED on donor cells, each eligible by its own
    `requirements`: synthetic_control, bsts and hierarchical_did. Per 10
    replications that is 50 synthetic_control + 50 bsts (both ineligible on S5,
    which has too few pre-periods) + 60 hierarchical_did (min_treated_units=1
    and one pre-period, so it fits on every donor cell including S5) = 160
    fitted rows (800 at 50 replications). The other nine estimators are
    ineligible on every donor cell (cross-section or panel-only designs). The
    hierarchical_did donor rows carry `ci_basis == "posterior"`, so they feed
    `coverage_posterior` (as bsts's do; synthetic_control's feed
    `coverage_permutation`), never `coverage_analytic`. The pre-registered
    expectations E5 and E7 are scoped to exclude donor rows.

    MEASURED 2026-09-26, this tree, `run_grid(DonorPoolAxes(scenarios=S0..S5)
    only, estimators=['synthetic_control', 'bsts'], fit_config=
    FULL_FIT_CONFIG)` at `replications=1`: 206.4s, 12 rows (10 fitted, 2
    ineligible, 0 refused). That run was restricted to those two estimators
    and EXCLUDES hierarchical_did. Both were `fitted` on S0-S4 and
    `ineligible` on S5, which has 2 pre-periods against the >=10 both
    require; the ineligible rows cost ~nothing, so the 206.4s is 10 fitted
    rows (5 scenarios x 2 estimators), ~20.6s per fitted row on average.
    hierarchical_did was NOT part of that measurement. Single reviewer runs of
    one hierarchical_did fit each, taken under contention and not in a quiet
    run, took S1 28.7s, S5 19.7s and S0 23.8s (~24s each).
    Caveat: this run, its extrapolations below, and the `ci` 338s/257s figures
    in the module docstring were all measured while another Claude session ran
    a multi-hour CPU-heavy benchmark on the same machine; wall-clock figures are
    likely inflated by contention and should be re-measured on an idle machine.
    The per-estimator split of the 206.4s was not timed, and ~20.6s is not
    comparable with the ci tier's per-cell figure (different estimator mix and
    `bootstrap_reps`). Only the 206.4s/12-row figure and the three single
    hierarchical_did fits are measured; the rest are extrapolations, assuming
    cost linear in replications, as for every other arm above. For the
    synthetic_control + bsts part: ~2,064s (~34 minutes) at this tier's 10
    replications, and ~10,320s (~2.9 hours) at `study_config()`'s 50. For
    hierarchical_did, an extrapolation from those single runs under
    contention: ~60 fits x ~24s ~= 24 minutes per full run and ~300 fits x
    ~24s ~= 2 hours per study run. All of this is on top of the cross-sectional
    and panel arms' own cost.

    The CATE arm (`CateAxes`, RICH/NULL/NOISY at `CATE_N_UNITS` = 2000) adds
    3 x 10 = 30 cells here and 150 at `study_config()`'s 50 replications. Each
    fits the five cross-sectional estimators; the seven panel and donor-pool
    estimators are ineligible. Only `causal_forest` reports a per-unit CATE, so
    only its rows carry `cate_correlation`/`cate_rmse`/`cate_calibration_slope`;
    the other four are scored on their ATE as on any cross-sectional cell.

    MEASURED 2026-09-28, this tree, idle machine: `run_grid` over the CATE arm
    alone at `replications=1`, every `FULL_ESTIMATORS` entry, `FULL_FIT_CONFIG`:
    44.4s, 36 rows (15 fitted, 21 ineligible). Per method (fitted seconds
    summed over the 3 cells): {'aipw': 25.1, 'causal_forest': 14.9,
    'difference_in_means': 0.1, 'ipw': 2.7, 'lin_regression': 1.3}.
    Extrapolated linearly in replications: ~444s at this tier's 10 and
    ~2,220s at `study_config()`'s 50. That is ~4.0% of `full`'s own
    documented ~3.10-hour total (~11,166s/10 replications, from this
    docstring's own paragraph above), well under the one-third threshold the
    design doc's section 12 sets for stopping to ask before committing to
    this tier's cost.
    """
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A", "B", "C", "D", "F", "G", "H"),
            n_units=(500, 2000),
            confounding_strength=(0.5, 1.0, 2.0),
            treatment_intercept=(-1.5, 0.0, 1.5),
            noise_scale=(0.0, 0.9, 1.3),
            effect_scale=(0.5, 1.0, 2.0),
            heterogeneity_scale=(0.0, 1.0, 2.0),
        ),
        panel=PanelAxes(
            scenarios=("P1", "P2", "P3", "P4", "P5", "P6"),
            n_units=(150,),
            n_periods=(7, 10),
        ),
        donor_pool=DonorPoolAxes(scenarios=_ALL_DONOR_POOL_SCENARIOS),
        cate=CateAxes(scenarios=_ALL_CATE_SCENARIOS),
        replications=10,
    )


# The replication count the benchmark study runs at. The parent design's own
# number (section 8.1 of the causal core spec), and the smallest that gives the
# coverage table a pooled MCSE narrow enough to expose gross undercoverage: at
# a coverage of 0.95 the binomial MCSE is 0.069 at 10 replications, 0.031 at 50
# and 0.022 at 100 (`docs/specs/2026-09-21-benchmark-study-design.md` section
# 1.2). `full` stays at 10; this is a separate tier, not a change to it.
STUDY_REPLICATIONS = 50


def study_config() -> GridConfig:
    """The study grid: `full_config()` at `STUDY_REPLICATIONS` replications.

    Defined by replacing one field rather than restating the axes, so the two
    tiers cannot drift: a widened axis in `full_config()` reaches the study
    with no second edit, and `tests/evaluation/test_study_tier.py` pins that
    the two are equal in everything but the replication count.
    """
    return replace(full_config(), replications=STUDY_REPLICATIONS)


def cate_study_config() -> GridConfig:
    """The CATE arm alone, at the study's replication count.

    Reads the arm from `full_config()` rather than restating it, so the two
    cannot drift, and `tests/evaluation/test_study_tier.py` pins that its cells
    are exactly `study_config()`'s CATE cells. A cell's generator is derived
    from its identity, so a run of this tier measures byte-identical data to
    the CATE part of a full study, at a fraction of the cost (design
    `docs/specs/2026-09-30-cate-report-section-design.md` §4).
    """
    return GridConfig(
        cross_section=None,
        panel=None,
        cate=full_config().cate,
        replications=STUDY_REPLICATIONS,
    )
