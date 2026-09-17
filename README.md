# CausalPulse

A causal inference and experimentation platform. Given an intervention and an outcome,
CausalPulse estimates what would have happened without the intervention, how large the effect
was, for whom it differed, and how much the conclusion should be trusted.

## Status

All seventeen phases of the causal core are implemented. The core is complete, with four items
deliberately left open — see the note at the end of this section.

**Phase 1** delivered the foundations: the workspace, data contracts, statistical primitives, and
a simulation framework whose data-generating processes carry known ground truth.

**Phase 2** delivered cross-sectional estimation: the result contracts, the estimator protocol and
registry, the design router, and four estimators — difference in means, Lin regression adjustment,
inverse probability weighting, and cross-fitted AIPW — each scored against known truth.

**Phase 3** delivered panel estimation: a panel data-generating process whose ground truth is
measured from realised potential outcomes rather than declared, six panel scenarios, and four
estimators — the canonical 2×2 difference-in-differences, two-way fixed effects, an event study
with a counterfactual path, and Callaway–Sant'Anna — plus the Goodman–Bacon decomposition that
explains *why* two-way fixed effects fails when it does. The 2×2, two-way fixed effects, and
Callaway–Sant'Anna are each scored against measured truth over 60 seeds at 400 units and 8 periods
on every scenario they are eligible for; the event study is scored the same way where its point
estimate and the scenario's measured ATT are the same estimand (see below), and two-way fixed
effects is additionally checked against `statsmodels` as an independent reference oracle.

**Phase 4** delivered synthetic control: a donor-pool data-generating process whose ground truth is
likewise measured rather than declared, six donor-pool scenarios (S0–S5), and `SyntheticControl` —
scored on a clean recovery scenario and a headline divergent-pre-trend scenario against a naive
donor-average baseline, with a ridge-augmented fallback for convex-hull violations, a
donor-contamination check, and permutation-based inference. `solve_donor_weights` is additionally
pinned against the reference implementation `pysyncon`.

**Phase 5** delivered the Bayesian structural time series counterfactual: a local linear trend plus
regularized-horseshoe donor regression fit with NUTS, three MCMC reliability diagnostics, and a
recovery harness over the same donor-pool scenarios synthetic control uses.

**Phase 6** delivered heterogeneous-effect estimation: an honest causal forest producing per-unit
CATE alongside an aggregate effect, three CATE scenarios whose per-unit ground truth is measured
rather than declared, and subgroup discovery behind three independent guards — an omnibus
calibration test, a minimum cell size, and Benjamini–Hochberg FDR control.

**Phase 7** delivered hierarchical Bayesian difference-in-differences: one partially-pooled
treatment effect per adoption cohort with a shared exposure trend, sampled with NUTS, reported as
a treated-cell-weighted posterior ATT. It adds a seventh panel scenario built specifically so the
across-cohort variance is identifiable, and it is checked three ways — against measured truth,
against two-way fixed effects in the degenerate single-cohort case, and against Callaway–Sant'Anna
on staggered adoption.

**Phase 8** delivered validity scoring and cross-method agreement: an ordinal Causal Validity
Score derived from an estimate's own assumption-check table and its design provenance, a
causal-language lint that checks the system never asserts more certainty than its own diagnostics
support, and a comparison layer that reports several methods' agreement without ever pooling their
estimates into one number.

**Phase 9** delivered the sensitivity suite: an E-value bound on unmeasured confounding, computed
exactly where the estimate's own predicted baseline makes it available and approximated via the
Chinn (2000) conversion otherwise; a Cinelli–Hazlett robustness value for four of the five
cross-sectional and panel point estimators; a specification curve across covariate subsets and
outlier/trim rules for `lin_regression`, `ipw`, and `aipw`; and four falsification techniques —
randomization inference, leave-one-donor-out, and hull-aware placebo distributions for both
donor-pool estimators — reported under one `SensitivityReport` and `FalsificationResult` type that
pairs every absent slot with its reason.

**Phase 10** delivered two more falsification techniques for cross-sectional estimators: a
negative-control outcome that treatment cannot affect by construction, and a
non-differential-misclassification correction for the treatment column that moves the estimate away
from zero, never toward it. Both are validated over 20 seeds against measured ground truth — the
negative control separates a confounded, unadjusted estimator (fires on 20 of 20 seeds) from an
adjusted one (0–1 of 20) — and both refuse outright on panel and donor-pool data, which need
different machinery.

**Phase 11** delivered missing-outcome handling for binary, cross-sectional estimators:
`arrays()`/`panel_arrays()` now refuse a dataset with a missing required column rather than failing
late with an unrelated error, complete-case analysis is offered as an explicit, named choice rather
than a silent default, and `tipping_point` reports Manski-style worst-case bounds and the smallest
adverse-fill fraction that would overturn a finding's significance and sign. Measured over 20 seeds,
the worst-case bounds always contained the complete-data estimate, and a real +5pp finding needed
about 25% adverse filling to break, against 0.7% for a null one.

**Phase 12** delivered imputation for the same binary, cross-sectional case: single and multiple
imputation of missing outcomes from a logistic model plus an optional auxiliary column, pooled by
Rubin's rules, and a `compare_missing_data` view that runs complete-case, single imputation,
multiple imputation, and Phase 11's tipping-point bounds side by side. Two pre-registered
expectations were falsified by measurement rather than confirmed: imputation with no auxiliary buys
nothing over complete-case (not the ~8% loss expected), and under MNAR it repairs about two-thirds
of the complete-case bias rather than leaving it unchanged, because this project's proxy is built
from the complete outcome and so carries real information about what is missing.

**Phase 13** delivered reproducibility and the project's first real (non-simulated) data:
`AnalysisManifest` serialises a multi-method run by reference — a regenerable simulated scenario or
a content-hashed file — rather than embedding the data, and `replay` refits from one and reproduces
bit-identical numbers outside three declared-nondeterministic provenance fields. Two vendored
real-data fixtures, LaLonde's NSW/CPS-1 job-training data and Abadie, Diamond & Hainmueller's
California Proposition 99 panel, are checked against their own published benchmarks. Fitting real
data for the first time surfaced two defects no simulated scenario could reach — a
non-scale-invariant propensity fit and a donor-weight solver that silently returned its own starting
point — both live since Phases 2 and 4 and both caught by checking a scale-invariance property
rather than a status flag.

**Phase 14** delivered the benchmark grid: a new `cp-evaluation` package, sitting above `cp-causal`
in the layers contract as the first package allowed to see both ground truth and the estimators, so
the ground-truth firewall stays closed to everything below it. `run_grid` fits ten of the project's
twelve estimators — cross-sectional and panel, not yet donor-pool — across the four DGP axes that
exist, and `summarise` reports bias, coverage, and calibration with Monte Carlo standard error
rather than a point number that cannot distinguish 94% from 95%. A gating `ci` tier runs in about
100 seconds; the reviews that shipped it found six defects, four of them in the plan's own draft
code, including a benchmark whose cells were not order-independent and a false-positive test that
could not fail.

**Phase 15** delivered the `causalpulse` CLI and result persistence: five commands (`simulate`,
`analyze`, `replay`, `bench run`, `bench report`) in a new `cp-cli` package sitting above
`cp-evaluation` in the layers contract, and Parquet-plus-JSON-sidecar persistence for a benchmark
run, with no cache — a stale cache entry would return a confidently wrong number with no signal.
This phase also corrected a false claim from Phase 14's own definition of done: Phase 7's
credible-interval coverage had never actually been measured, because `hierarchical_did` was excluded
from the gating tier on cost. Measured here at an easier protocol than Phase 7's own, the reported
undercoverage did not reproduce — stated as "not reproduced at this protocol," not as Phase 7 having
been wrong.

**Phase 16** delivered eight publication-quality matplotlib figures in a new `cp-figures` package:
counterfactual path, event study, love plot, propensity overlap, specification curve, placebo
distribution, cross-method forest plot, and benchmark heatmaps. Every figure is split into a pure
data function and a thin drawing function, and a shared guard rejects any attempt to plot an effect
estimate without its interval, enforced by construction rather than by review. Building the love
plot and propensity-overlap figures required recovering two diagnostics — per-unit propensity scores
and before/after covariate balance — that `ipw` and `aipw` had always computed and then discarded.
The package takes live result objects, not a CLI command: most of the eight figures need array data
that `estimates.csv` never persisted.

**Phase 17** delivered the evidence report generator, `cp-report`, the final piece of the causal
core. It renders four sections — the assumption table, the validity band with its rationale, the
cross-method comparison, and stated limitations — to Markdown or a single self-contained HTML
document with figures inlined as SVG, reusing Phase 16's data functions rather than the estimator
result types directly. The generator lints its own prose against `causal_language_lint` before
emitting anything and refuses to return a document that overclaims, naming the offending sentence,
term, and section. Like the figures, it takes live result objects rather than reading a stored
analysis: no CLI command currently produces a report, because the arrays most figures need were
never persisted.

The causal core is complete. Phase 17 recorded four items deliberately left open rather than
built. Three remain, after Phase 18 (below) closed the first: the CATE arm of the benchmark grid
(CATE correlation, RMSE, and calibration slope, which need their own cell shape and array-valued
diagnostics); the donor-pool arm of the benchmark grid (`synthetic_control` and `bsts`, which need a
donor-pool cell shape); and a residual from Phases 9 and 10 — `SyntheticControl`'s CI half-width and
`donor_contamination_check`'s threshold are still calibrated against the simplex placebo spread,
while `falsify`'s own placebo distribution is convex-hull-aware, and closing that gap is blocked on
the donor-pool benchmark arm for measurement. Each is recorded as a decision at Phase 16's kickoff,
not an oversight, and each would be its own phase if wanted.

**Phase 18** closed the first of those four items: the missing benchmark-grid axes.
`PropensityAssignment` gained an intercept and `LogisticBaseline` an additive noise scale, both
defaulting to today's behaviour exactly; `ScenarioParams` now carries all five cross-sectional
axes (the pre-existing `confounding_strength` plus treatment imbalance, noise level, effect size,
and heterogeneity), and a scenario declares which axes it actually responds to rather than being
crossed against every one blindly. `full_config`'s cross-sectional grid widens from 42 to 84 cells
to cover three measured points per new axis — a low, a baseline, and a high value, chosen by
generating real data rather than by estimate. See
`docs/specs/2026-09-13-causal-core-phase18-design.md` section 12 for the measured values, the
values considered and rejected, and the defects a whole-branch review found before any of it
shipped.

No CLI command currently produces an evidence report, either: the report generator built in Phase 17
takes live result objects, the same as the figures do, because the arrays six of the eight figures
need were never persisted to disk. A user who runs `causalpulse analyze` cannot turn its output into
a report without writing Python.

## Packages

| Package | Responsibility |
|---|---|
| `cp-core` | Value types, dataset contract, and the result contract. No algorithms. |
| `cp-statistics` | Proportion intervals, robust and cluster-robust covariance, bootstrap engines. |
| `cp-simulation` | Compositional data-generating processes carrying computed ground truth. |
| `cp-causal` | Cross-sectional and panel estimators, propensity diagnostics, panel assumption checks, design router. Forbidden from importing `cp-simulation`. |
| `cp-evaluation` | The benchmark grid: `run_grid` fits estimators across the DGP axes that exist and `summarise` reports bias, coverage, and calibration with Monte Carlo standard error. The first package allowed to import both `cp-simulation` and `cp-causal` together, so it sits above `cp-causal` in the layers contract and nothing below it may import it in turn. |
| `cp-figures` | Eight publication-quality matplotlib figures — counterfactual path, event study, love plot, propensity overlap, specification curve, placebo distribution, cross-method forest plot, benchmark heatmaps — each a pure data function plus a thin drawing function. Imports only `cp-core` and `cp-causal`. |
| `cp-report` | The evidence report generator: assumption table, validity band, cross-method comparison, and stated limitations, rendered to Markdown or one self-contained HTML file with figures inlined as SVG. Refuses to emit prose that fails `causal_language_lint`. |
| `cp-cli` | The `causalpulse` command-line entry point (`simulate`, `analyze`, `replay`, `bench run`, `bench report`) and Parquet-plus-JSON-sidecar result persistence. Sits at the top of the layers contract; nothing imports it. |

## Why the import boundary matters

An estimator that can reach ground truth can be accidentally tuned against it, and every
benchmark number it produces afterwards is meaningless. The separation is enforced by an
import-linter contract, not by convention.

## Ground truth is measured, not declared

Potential outcomes are constructed on the probability scale as `p0` and `p1 = clip(p0 + tau, 0, 1)`,
and the true effect is then measured as `mean(p1 - p0)`. The clip is not cosmetic: with a
baseline reaching 0.98, a nominal `tau` of 0.05 produces an actual effect of 0.046. Baseline
probabilities are therefore bounded to `[0.05, 0.85]`, and a test asserts that each scenario's
measured effect matches its declared label.

## What the estimators are checked against

Every estimator is scored against datasets whose true effect is known exactly. The check is not
only "does it get close to truth" but "does it fail where it should":

- Under randomisation, all four estimators recover the effect.
- Under confounding, the unadjusted difference in means stays badly biased — roughly +13
  percentage points against a true +5 — while the three adjusting estimators recover it. If the
  unadjusted estimator ever stopped being biased there, the scenario would have stopped being
  confounded and every other result would be meaningless.
- With a deliberately crippled propensity model that omits a confounder, inverse probability
  weighting becomes biased while AIPW stays close to truth. That is the double-robustness property,
  demonstrated rather than asserted.

### Where double robustness runs out

Double robustness protects an estimate when one of AIPW's two nuisance models — the propensity
model or the outcome model — is wrong, not when both are. Scenario G is the case where both are:
its true propensity depends on `expit(strength · (p0 − mean(p0)))`, where `p0` is itself a clipped
sigmoid of the covariates, so the true propensity is markedly nonlinear while AIPW fits it with a
linear logit; the outcome nuisance is plain OLS against that same nonlinear surface. Neither model
is correctly specified, and AIPW carries a small but real finite-sample bias as a result. Measured
at n=4000 over 40 seeds against a true effect of 0.05: AIPW +0.0170, Lin regression +0.0078, and
IPW −0.0031. IPW shares the same misspecified propensity model and stays essentially unbiased,
which confirms the bias is specific to AIPW's combination of nuisance models rather than to the
propensity model alone. The bias shrinks as `n` grows but does not vanish at the sample sizes
tested here.

### What the panel estimators are checked against

Six panel scenarios, each carrying ground truth measured from realised potential outcomes rather
than copied from an input parameter. Bias is the mean error over 60 seeds at 400 units and 8
periods, with the Monte Carlo standard error of that mean in brackets.

| Scenario | What it is | 2×2 DiD | TWFE | Callaway–Sant'Anna |
|---|---|---|---|---|
| P1 | Clean 2×2, parallel trends exact | −0.0143 (0.0088) | −0.0143 (0.0088) | −0.0064 (0.0155) |
| P2 | Null effect | −0.0143 (0.0088) | −0.0143 (0.0088) | −0.0064 (0.0155) |
| P3 | Staggered, homogeneous effects | n/a | +0.0047 (0.0084) | +0.0008 (0.0131) |
| P4 | Staggered, effects grow with exposure | n/a | **−0.9946 (0.0112)** | +0.0008 (0.0131) |
| P5 | Parallel trends violated | n/a | **+1.2704 (0.0111)** | **+1.6776 (0.0138)** |

2×2 DiD is n/a on P3–P5 because it requires uniform adoption; staggered adoption has no single
pre/post split for it to use. P1's Goodman–Bacon already-treated-as-control share is exactly 0.0000
across all 60 seeds — there is no earlier cohort to serve as a control, since adoption is uniform —
which is the mechanical reason P1 has nothing for two-way fixed effects to get wrong.

Two-way fixed effects and the canonical 2×2 agree to a measured **2.66e-14** on identical
uniform-adoption data — two independent implementations of the same estimand, which is a stronger
cross-check than either has alone.

### Where two-way fixed effects breaks down

P4 is the phase's headline. Effects grow with exposure, so at any calendar period an early cohort's
effect exceeds a late cohort's. Two-way fixed effects then uses already-treated units as controls
with negative weights, and its estimate is off by **−0.9946** (88.8 Monte Carlo standard errors
from zero) against a true ATT of **2.8361** — a 35% error. Callaway–Sant'Anna, which never uses an
already-treated unit as a control, is off by +0.0008 (0.1 MCSE from zero) — indistinguishable from
noise.

The Goodman–Bacon decomposition reports **where** that comes from: across the 60 seeds, a mean of
26.24% (minimum 23.63%) of the two-way fixed effects estimate on staggered data rests on
already-treated-as-control comparisons — available as a diagnostic on every staggered fit.

### Where Callaway–Sant'Anna breaks down

P5 exists so the suite does not present Callaway–Sant'Anna as magic. It assumes parallel trends as
much as two-way fixed effects does. When steeper-trending units adopt earlier, **both** are badly
biased against a true effect of 1.0 — TWFE by +1.2704 (114.8 MCSE from zero) and CS by +1.6776
(121.5 MCSE from zero). CS is the *worse* of the two here.

What the platform does provide is that the failure is announced: `parallel_trends` reports FAIL on
P5 for both the event study and Callaway–Sant'Anna, and the measured pre-trend gap is reported as a
diagnostic.

### Binary panels carry link-function bias

Parallel trends is a functional-form-dependent assumption. A binary panel generated as
`P(Y=1) = expit(alpha_i + gamma_t + tau·D_it)` has parallel trends on the **logit** scale and not on
the probability scale: the same time shift moves a unit at `p = 0.1` and a unit at `p = 0.5` by
different amounts.

Scenario P6 measures it, using the **same common time slope every other panel scenario uses** — not
a special parameter. (An earlier draft of the design tried to make the bias vanish with a bespoke
`time_slope = 0.3`; that claim did not survive implementation and was corrected during Task 6.) With
adoption selected on the baseline level, so the two arms sit at different points on the logistic
curve, a 2×2 DiD reporting percentage points is biased by **+0.0360** against a true effect of
0.1061 — 34%, at 18.3 Monte Carlo standard errors — measured at 800 units and 120 seeds. With random
adoption the arms are compositionally identical and the same measurement gives −0.0015 (0.8 MCSE),
indistinguishable from zero — confirming the bias requires selection on level, not the time path.

This is **not** bias arising from zero pre-trend signal. Parallel trends holds exactly on the
**logit** scale the data are generated on, and fails on the **probability** scale the 2×2 estimator
reports on — the link-function bias *is* that probability-scale differential trend, not a separate
phenomenon from it. Measured directly: `true_pre_trend_gap` on P6 is **+0.0091 per period** (over
120 seeds, ~456 standard errors of the mean from zero — essentially deterministic, not noise), and
with adoption at period 5 of an 8-period panel the pre/post centroid gap is roughly 4 periods, so
`0.0091 × 4 ≈ 0.0364` — close to the measured +0.0360 bias above. A reader should not conclude that
link-function bias can appear with zero pre-trend signal, or dismiss a real pre-trend warning on a
binary panel as "just the link function."

This is a property of applying a linear difference-in-differences to a non-linear outcome, not a
defect in the estimator. It is documented rather than hidden because binary conversion rates are
the motivating use case.

### What the pre-trend diagnostics do and do not do

- `parallel_trends` is a joint Wald test on the pre-treatment leads. **Failure to reject is not
  evidence that parallel trends holds** — the test has limited power, and the assumption concerns
  counterfactual post-treatment trends that no pre-treatment data can observe. With a single
  pre-period there are no leads at all and the status is `NOT_TESTABLE`, never `PASS`.
- The breakdown value is the **linear-extrapolation** form: the smallest per-period linear trend
  violation that would pull the interval onto zero. It is **not** a Rambachan–Roth honest confidence
  set, which is not implemented here.
- `no_concurrent_intervention` is always `NOT_TESTABLE`. Whether another change hit the treated
  units at the same time cannot be determined from the data at all.

### Two defects the panel recovery harness found

Building the harness that produced the table above (Task 13) found two real, previously-latent
defects, neither caught by any individual estimator's own tests:

- **`CallawaySantAnna`'s `parallel_trends` check was miscalibrated.** On a clean scenario with no
  violation, it reported FAIL on 80% of seeds against a nominal 5% — actively misleading, not
  merely imprecise. The cause: the placebo-lead covariance pooled one variance across three
  correlated placebo cells sharing the same baseline period instead of giving each cell its own
  analytic variance. Fixed to give each cell its own variance via the same two-independent-samples
  formula the real ATT(g,t) cells use; the false-positive rate measured 10% after the fix (still
  slightly above nominal, consistent with a chi-square approximation on correlated degrees of
  freedom, no longer a gross miscalibration), with power against a real violation unchanged.
- **The router silently allowed cross-sectional estimators to run on panel data.** `needs_panel`
  was enforced in only one direction — panel estimators were kept off cross-sectional data, but
  nothing kept cross-sectional estimators (difference in means, Lin regression, IPW, AIPW, all of
  which have no unit/period awareness at all) off panel data. Fitting one on a panel would silently
  treat every unit-period row as an independent observation. Fixed with the symmetric check.

## Synthetic control (Phase 4)

`SyntheticControl` is a donor-weighted counterfactual for a single treated unit: non-negative,
simplex-constrained weights on up to 17 donors, chosen to minimise pre-treatment fit error, with
the post-treatment mean gap between the treated unit and its synthetic twin as the point estimate.
It is scored against six donor-pool scenarios (S0–S5), each carrying ground truth measured from
realised potential outcomes the same way Phase 3's panel scenarios are, at a fixed protocol of 17
donors, 24 periods, adoption at period 18.

### What it is checked against

| Scenario | What it is | SC bias (seeds) |
|---|---|---|
| S1 | Clean recovery | −0.0324 (MCSE 0.0320), 400 seeds |
| S2 | Divergent pre-trend | +0.1736 (MCSE 0.0190), 400 seeds |

S2 is the phase's headline result. A naive equal-weighted donor average, scored on the same seeds
as the losing baseline, is biased by **+4.4167 (MCSE 0.0260)** — SC's +0.1736 is a dramatic
improvement over it, the same two-sided assertion pattern Phase 3's P4 test established (asserting
only SC's accuracy could quietly pass even if the naive comparison had stopped being wrong, so both
sides are checked). But +0.1736 is itself well clear of zero, not a near-zero recovery the way S1
or the panel estimators on their clean scenarios are. That is an honest property of single-treated-
unit synthetic control, not a defect in this implementation: there is no averaging across many
treated units the way the DiD family had available to it in Phase 3, so idiosyncratic donor-fit
error in one unit's synthetic path does not average out the way it does across a treated cohort.

### The convex-hull violation and the ridge-augmented fallback

Interpolation bias is synthetic control's principal failure mode: a simplex-constrained combination
of donors can only ever reproduce values inside their convex hull, so a treated unit outside that
hull cannot be matched by construction. S3 is built to trigger exactly this. The `convex_hull` check
fires on **300/300** seeds there. Plain simplex weights are badly biased as a result — **+1.21**
against S3's true effect — while the ridge-augmented fallback (an intercept plus L2-penalised,
unconstrained donor coefficients, which can reproduce values outside the donor pool's range) recovers
to **+0.0223 (MCSE 0.0270)** over 300 seeds, tolerance 0.15. `SyntheticControl.fit` switches to this
fallback automatically whenever `convex_hull` reports FAIL.

### The donor-contamination check, and a correction it needed

`donor_contamination_check` flags a donor whose own placebo RMSPE ratio is a robust statistical
outlier, reusing the same placebo procedure permutation inference already runs rather than a bespoke
test. Its real statistic is the **maximum** robust z-score across all 17 donors, not one
pre-specified donor's z-score — that has to be the maximum, since nobody knows in advance which
donor, if any, might be contaminated.

An earlier calibration of this check measured its threshold against one pre-specified donor's
z-score instead of the max, which found a false-positive rate of 5/100 at threshold 2.0. Deployed
against the real max-over-17-donors statistic the check actually uses, that same threshold produced
72/100 false positives on clean data — a multiple-comparisons gap between "is this one donor
extreme" and "is the most extreme of 17 donors extreme," caught by the recovery harness rather than
by the check's own unit tests, which had never measured the max statistic's real null distribution.
Recalibrated by measuring the max statistic's distribution directly (300 seeds): threshold **5.5**,
false-positive rate **7/100** on a clean donor pool, true-positive rate **96/100** against a donor
carrying its own **+5.0** hidden shock (also corrected up from +1.5, which was not reliably
detectable once the check was properly calibrated against the max statistic). This is exactly the
kind of measured-not-assumed correction this project's README already documents for other phases —
see "Two defects the panel recovery harness found" above for the earlier instance of the same
discipline.

### S5: insufficient pre-periods

S5 has only 2 actual pre-treatment periods. `sufficient_pre_periods_check(minimum=10)`, reused
unchanged from Phase 3, fires FAIL deterministically against it.

### A limitation worth stating plainly: S0 and the p-value floor

S0 has zero true effect, and measuring `SyntheticControl().fit(...).p_value < 0.05` over 100 seeds
gives **0/100** significant. That number is not evidence of good — or even conservative — 5%-level
calibration. It is a mathematical certainty at this donor-pool size, not a property of the data.
`permutation_p_value`'s minimum achievable value is `1 / (n_donors + 1) = 1/18 ≈ 0.0556`, since the
real treated unit always counts itself when ranking among the placebo donors — and that floor is
already above 0.05, so `p < 0.05` can never be true with 17 donors regardless of what the data look
like. This is the same 1/(N+1) floor Abadie's own permutation-inference construction has. A reader
should not conclude from "0/100 significant" that inference here is well-calibrated at the 5% level;
this donor count cannot express that at all, in either direction.

### `CounterfactualPath`: a deliberate convention divergence from `EventStudy`

`EventStudy`'s Phase 3 convention pins pre-treatment `predicted_absent_treatment` at exactly
`observed`, because its leads are pinned at zero by construction. `SyntheticControl` has no
equivalent guarantee: its pre-treatment `predicted_absent_treatment` is only as good as the donor
weights' fit, so pre-period `pointwise_effect` is small but genuinely nonzero — equal to the fit
residual, not zero. This is a deliberate, documented difference in what the two estimators' paths
mean during the pre-period, not a bug in either one.

### Donor weights use uniform predictor weighting, not Abadie's nested optimisation

`solve_donor_weights` matches the treated unit's standardised pre-treatment outcome path itself,
weighting every pre-period equally, rather than nesting a second, outer optimisation over a separate
predictor-importance matrix the way Abadie's original method does. This is a documented
simplification, not an oversight, and it stays correct as long as the donor pool is small relative
to the pre-period length — the measured protocol above uses 17 donors against 17 pre-periods, not
more donors than periods, specifically to avoid the underdetermined regime where this simplification
would let the solver overfit pre-period noise at the cost of post-period extrapolation.

### Reference parity: `pysyncon`

Unlike Callaway–Sant'Anna in Phase 3, which shipped without a pinned reference implementation,
synthetic control's donor-weight solve **is** pinned against `pysyncon`'s `Synth`: `solve_donor_weights`
matches its pre-treatment RMSPE within `rel=0.2` on S1, measured agreement of roughly 0.4% between the
two solvers — the 0.2 tolerance leaves about 50x headroom over that measurement for
environment-specific numerical drift (BLAS, scipy version) while still catching a solver that
regresses to a materially worse optimum. The comparison is on pre-treatment fit quality, not
elementwise weights, because the weight solve is a nonconvex-adjacent constrained optimisation with
potentially multiple near-optimal solutions when the pre-period is close to underdetermined — two
different SLSQP starting points and convergence paths can land on different vertices of a tied or
near-tied optimum. `pysyncon` is a dev-only dependency, marked `slow`, and never imported outside
`test_reference_parity.py`.

`pysyncon` also has a real import-time landmine worth recording: on a fresh, bytecode-cache-less
import, an invalid escape sequence in one of its submodules raises a `SyntaxWarning` that this
project's `filterwarnings = ["error"]` promotes to a hard `SyntaxError` — reproducible by clearing
`pysyncon`'s `__pycache__` and importing it under pytest's real config. The reference-parity test's
`_pysyncon()` helper works around it with a local, narrowly-scoped `SyntaxWarning` suppression around
the import only; nothing downstream of the import runs under anything but the project's normal
warnings-as-errors policy.

## BSTS (Phase 5)

A Bayesian structural time series counterfactual for the same donor-pool data shape synthetic
control uses: one treated unit, a donor pool, a long pre-period. Local linear trend (level and
slope, both random walks, sampled as explicit latents via NumPyro's `scan`) plus a
regularized-horseshoe regression on standardized donor series, fit with NUTS.

### What it is checked against

No reference-library parity dependency, unlike synthetic control's `pysyncon` pin: BSTS's
structural form is hand-specified and won't match any external library's model closely enough for
numerical parity to mean anything. Validated purely via the measured-ground-truth recovery harness
(`tests/causal/test_bsts_recovery.py`, `slow`) against the same `S0`-`S5` scenarios synthetic
control uses, plus three MCMC diagnostics (R-hat, effective sample size, divergent transitions,
`bayesian_checks.py`) that mark a fit unreliable rather than let a bad posterior be reported as a
clean finding.

### Two-tier MCMC configuration

`BSTS.fit()` defaults to `PRODUCTION_MCMC_CONFIG` -- a caller who supplies no
`options["mcmc_config"]` gets a real, trustworthy posterior, not a shortcut. `FAST_MCMC_CONFIG` (a
deliberately tiny chain that exercises code paths and array shapes only) is opt-in, used only by
this project's own fast contract tests; its own R-hat/effective-sample-size diagnostics would
reliably FAIL if relied on for a real result, which is exactly why it is never the default.

### Bayesian outputs, not p-values

`ci_basis="posterior"` always; `p_value` is always `None`. `P(effect > 0)` and `P(effect >
practical_threshold)` are Bayesian posterior probabilities, not p-values, and are reported through
`diagnostics` instead -- the same discipline that keeps a credible interval from ever being
described as a confidence interval anywhere in this codebase.

### `donor_matrix` is now shared

`SyntheticControl` and `BSTS` both extract their treated/donor wide arrays through
`cp_causal._panel.donor_matrix`, moved there from `synthetic_control.py` when BSTS needed the
identical extraction.

## Causal forest and CATE (Phase 6)

An honest causal forest. R-learner residualization (cross-fitted nuisance models via scikit-learn)
feeds a hand-rolled honest-splitting ensemble, which produces a per-unit CATE, an aggregate effect,
and subgroup findings guarded by an omnibus calibration test, a minimum cell size, and
Benjamini–Hochberg FDR control.

### The splits are hand-rolled, not scikit-learn's tree

This project's estimators are implemented from primitives, with scikit-learn restricted to nuisance
models — already the pattern for AIPW's propensity and outcome regressions. A causal forest's
splits *are* the causal quantity: they decide which units share a CATE estimate, which is not a
prediction problem. So recursive binary splitting, honest sample-splitting, ensemble aggregation,
and the infinitesimal-jackknife variance are all hand-written. scikit-learn fits only the
R-learner's residualization nuisance models.

### The omnibus calibration test uses out-of-bag CATE, not in-sample

Evaluated in-sample, the calibration test fired on a genuinely heterogeneous scenario *and* on a
null scenario with zero heterogeneity — an overfitting artifact, the same class of bias that honest
splitting and cross-fitting exist to prevent elsewhere in this estimator. Averaging only over the
trees that excluded a unit from their subsample entirely removed the false positive. The best
linear projection and subgroup discovery read the out-of-bag CATE for the same reason; only the
reported `per_unit_cate` diagnostic uses the full ensemble, because a point estimate wants every
tree available.

### The aggregate effect does not come from the forest's predictions

The headline `point` and `se` are the R-learner score on the cross-fitted residuals, not the mean
of the per-unit CATEs. The point estimates are near-identical; the intervals are not. Averaging the
per-unit infinitesimal-jackknife variances and dividing by `n` — the natural-looking formula, and
the one this phase's implementation plan specified — measured **0.40 coverage on the null scenario
and 0.35 on the rich one** against a nominal 0.95, because it treats 2000 per-unit CATEs built from
the same trees and the same residuals as independent draws. Applying the jackknife to the mean
prediction directly reaches 0.65 and 0.60. The residual score reaches 1.00 and 0.95.

Every fast test passed under the broken formula. Coverage is a *rate*, and only the multi-seed
recovery harness could see it.

### Reference parity: `econml`

Per-unit CATE is checked for rough agreement — not exact match — against `econml`'s
`CausalForestDML` on identical synthetic data. The two are independent implementations of the same
algorithm *family*, differing in splitting criterion, honesty scheme, and nuisance models; they do
not share an optimisation problem the way `SyntheticControl` and `pysyncon` do. Measured agreement
is 0.958, while against measured truth the two score 0.669 and 0.634 respectively — the pair track
each other far more closely than either tracks truth, which is what makes this a drift test rather
than a second, weaker recovery test. `econml` is a dev-only dependency, marked `slow`, and never
imported outside `tests/`.

## Hierarchical Bayesian DiD (Phase 7)

`HierarchicalDiD` gives each adoption cohort its own treatment effect, drawn from a common
population distribution so a noisy cohort is shrunk toward the others rather than trusted on its
own. Non-centred throughout: with a handful of cohorts the funnel between the across-cohort scale
and the cohort effects is severe, and a centred parameterisation diverges.

### The effect grows with exposure, and leaving that out was a real bug

The first version held one constant effect per cohort. That is not a harmless simplification when
the true effect ramps after adoption. The misfit is positively correlated with calendar period —
later periods hold more heavily-exposed units — so the time effects absorb it, and the cohorts
treated only in late periods lose the most.

Measured on P4, the ramping scenario, with the constant-only model:

| cohort (adopts) | 3 | 5 | 7 |
|---|---|---|---|
| true effect | 3.50 | 2.50 | 1.50 |
| estimated | 3.36 | 2.03 | 0.49 |

−17% ATT bias and **0 of 15 seeds** covering truth, with R-hat and divergences clean on every
seed. The obvious reading — shrinkage pulling the heavy cohorts toward the unweighted mean — is
wrong, and the numbers say so: shrinkage would pull 1.50 *up* and compress the spread, whereas
every cohort moved down and the spread widened. P3, whose effect is genuinely constant, was
unbiased under the identical model, which is what identified the misspecification rather than the
sampler.

The effect is now `tau_g + delta * exposure` with one slope shared across cohorts, and the
aggregate gains the matching term so it stays the same quantity the truth measures. P4's bias went
to −0.3% and coverage to 7 of 8; P3, the case that already worked, stayed unbiased. Deliberately
one slope, not one per cohort — cohorts growing at different rates, or non-linearly, are outside
this model.

### The headline is a treated-cell-weighted ATT, not the population mean

`mu_tau` is the unweighted "typical cohort" effect. The ATT averages over treated *cells*, so a
cohort's weight is units × post-periods and an earlier-adopting cohort counts for more. On P4 the
distinction is worth 2.5028 against 2.8356 — about 150 Monte Carlo standard errors, which the
recovery harness would have reported as estimator bias rather than the weighting error it is. The
weighted average is taken *within* each posterior draw; averaging draws first gives the same point
estimate and a wrong interval.

### How many cohorts the model actually wants

Partial pooling is the model's justification, so whether the across-cohort scale is estimated or
merely inherited from its prior is not a detail. Refitting under a fourfold change in that prior's
scale moves the posterior by **1.4% at seven cohorts** and **81% at three** (full prior-dominance
would be 300%). So the pooling behaviour is genuinely estimated with seven cohorts and about
half-estimated with three. The headline ATT is far less sensitive — P3's bias is −0.67% — but a
large across-cohort scale read as evidence that cohorts differ needs enough cohorts to mean it.
This is why Phase 7 added a seventh panel scenario rather than reusing the three-cohort ones.

### What it is checked against

Recovery against measured truth on six scenarios, plus two independent cross-checks:

| scenario | mean bias | coverage |
|---|---|---|
| P1 clean 2×2 | −0.0430 | 6/8 |
| P2 null | −0.0155 | 6/8 |
| P3 staggered, homogeneous | −0.0067 | 7/8 |
| P4 staggered, ramping | −0.0075 | 7/8 |
| P5 parallel trends violated | +2.1359 | 0/8 |
| P7 many cohorts | +0.0123 | 6/6 |

P5's large bias is correct: it violates parallel trends by construction, and every
difference-in-differences estimator here is biased on it. What matters is that it is not biased
*quietly* — `parallel_trends` FAILs on 8 of 8 seeds, and the test asserts both halves, so the
guard cannot rot while recovery elsewhere keeps passing.

With a single cohort the hierarchy has nothing to pool over and the estimate collapses onto
two-way fixed effects: largest difference 0.0277 over 8 seeds, with TWFE's point inside the
credible interval every time. On staggered adoption it agrees with Callaway–Sant'Anna — intervals
overlapping 8 of 8 on both P3 and P4, checked as overlap and magnitude rather than exact match,
since the two solve one problem by different means.

### The credible interval undercovers, and the number is recorded

Pooled coverage across the five recovering scenarios is 32/38 = **0.842 against a nominal 0.95**.
At 38 fits a true 0.95 carries a standard error of 0.035, so that gap is about three standard
errors — more than seed noise explains. A posterior credible interval carries no frequentist
coverage guarantee, so this is not a contradiction, and it is why `ci_basis` is `"posterior"` and
the interval is never called a confidence interval anywhere. But it should not be read as a 95%
frequentist interval. The harness's coverage floor is set to catch a collapse, not to police this
gap; closing it belongs to the calibration work still ahead.

R-hat also fails on a minority of seeds at the production configuration — 2 of 8 on P4, 3 of 6 on
P7 — and the harness asserts that as a rate rather than assuming every seed is clean. Bias stays
tight regardless of which seeds trip it, so this is diagnostic cleanliness rather than a bias
property.

## Validity scoring and cross-method agreement (Phase 8)

Every estimator returns its own table of assumption checks. Phase 8 turns that table into one
ordinal band — `STRONG` / `MODERATE` / `WEAK` / `INSUFFICIENT` — and adds a layer that compares
several fitted methods without collapsing them into one number.

### What the band is and is not

> This band scores observable diagnostics, not truth. It is not a probability that the effect is
> causal, and it cannot establish an assumption the data are unable to test.

That sentence — `VALIDITY_SCORE_LIMITS` — is carried on every `CausalValidityScore` as a property,
not a field a caller can omit or reword.

The band is capped by a **design-based ceiling** before any check is read: a randomised or
natural-experiment design can reach `STRONG`, an observational design cannot reach past
`MODERATE` no matter how clean its diagnostics are. Clean diagnostics on observational data mean
the checks that *can* run all passed, not that the untestable ones did too. Within that ceiling,
an `estimation`-kind failure (an unconverged sampler, for instance) drops the band straight to
`INSUFFICIENT` regardless of design — that failure means no usable number exists, which the design
ceiling has nothing to say about. An `identification`- or `data_adequacy`-kind failure caps the
band at `WEAK`; a warning of either kind drops it one step below the ceiling. The lowest cap wins
when several apply — a fit carrying both a `FAIL` and a `WARN` lands at `WEAK`, not
`INSUFFICIENT`, because the caps compose by minimum rather than chaining.

### Why provenance is declared, not inferred

The ceiling reads `CausalSchema.assignment_provenance` — `randomised`, `natural_experiment`, or
`observational` — a fact about how the data came to exist, not something derivable from which
estimator fit it. `difference_in_means` *assumes* randomisation; it does not establish it, so
reading the ceiling off the estimator would let any observational dataset claim a randomised
ceiling simply by being fit with the wrong method. The field defaults to `observational`, so
omitting it caps the band rather than raising it. Every simulation component declares its own
provenance the same way `confounders()` already does — as a fact about the mechanism, never
inferred from its type — so `PropensityAssignment` and `BaselineSelectionAssignment` correctly
report `randomised` at zero confounding strength and `observational` above it, agreeing with what
`confounders()` already says about the same mechanism.

### Why `NOT_TESTABLE` does not subtract

An assumption the data cannot test — unconfoundedness under observational assignment, for
instance — is named in the score's rationale but does not move the band on its own. The design
ceiling already prices that in: an observational design is capped at `MODERATE` precisely because
some of its identifying assumptions cannot be tested. Subtracting for `NOT_TESTABLE` on top of
that ceiling would count the same fact twice and read every honest observational analysis as
broken.

`lin_regression`'s `outcome_model_specification` check was originally shipped as an unconditional
`WARN` — a static caveat ("the adjustment is linear in the supplied covariates") that fires on
every fit regardless of the data, because nothing about it is computed from the fit. Since the
rubric drops the band one step for any `WARN`, this scored `lin_regression` strictly below
`difference_in_means` on identical randomised data, purely for shipping an honest caveat the
simpler estimator has no check for at all. Reclassified to `NOT_TESTABLE` — its correct status, by
the same test every other check here is held to: nothing tests whether the linear form is right
for this particular fit, so `WARN` was reporting a diagnostic result that was never actually
computed.

### The comparison layer does not average

`compare()` takes several fitted estimates of the same quantity and reports their spread, which
intervals overlap, and where their assumptions diverge — it never produces a pooled point
estimate. `MethodComparison` carries no `combined`/`pooled`/`mean` field; there is no number to
average even by accident. The methods share the same data and are not independent, so a pooled
estimate would have no defensible interpretation and a meta-analytic heterogeneity test would be
invalid on it. `compare()` refuses a mixed estimand, a mixed scale, fewer than two estimates, and
duplicate method names — comparing an ATE to an ATT, or a percentage-point effect to a log ratio,
is not two answers to one question. A method the validity score calls `INSUFFICIENT` is excluded
from the agreement verdict (its number is not usable, so letting it sit far from the others would
report a disagreement between one estimate and one non-estimate) but is still named in the
verdict text. When methods disagree, `assumption_diff` — which assumption each method relies on,
and what it found — is the output that matters; the interval spread alone does not say why.

### What the discrimination harness measured

The claim that the band tracks reliability is empirical, not just a property of the rubric's
rules, so it is measured against known truth rather than assumed. Across 20 seeds over every
cross-sectional, panel, and donor-pool scenario (1000 fits total, MCMC estimators excluded — their
bands come from convergence diagnostics, a different question from identification):

| band | fits |
|---|---|
| `STRONG` | 634 |
| `MODERATE` | 229 |
| `WEAK` | 137 |
| `INSUFFICIENT` | 0 (structurally absent here — needs an `estimation`-kind failure, which only the excluded MCMC convergence checks can produce) |

| bucket | n | coverage | MCSE |
|---|---|---|---|
| `MODERATE` or better | 863 | 0.8262 | 0.0129 |
| `WEAK` or worse | 137 | 0.5766 | 0.0422 |

The difference is 0.2496 against a combined MCSE of 0.0441 — the band-MODERATE-or-better bucket
covers truth about 25 percentage points more often, 2.8 times the asserted 2-MCSE floor. An
earlier attempt at 8 seeds, chosen first for runtime, measured a difference of 0.1120 against a
floor of 0.1228 — a **failing** margin, because the `WEAK`-or-worse bucket thins to 71 fits and its
MCSE grows accordingly. That result was not patched by loosening the assertion or picking a
seed count that happens to pass; the harness re-measured at 20 seeds, the count it actually
asserts at, and that is the number above.

Re-measured again after the `outcome_model_specification` reclassification above: 120 of 1000
fits moved, all of them `lin_regression`'s own runs (80 MODERATE→STRONG on randomised scenarios,
40 WEAK→MODERATE on observational ones). Headroom rose rather than fell — the 40 rows crossing the
`MODERATE`+/`WEAK`- boundary had their own coverage at 0.875, derived directly from the two
measured `WEAK`-or-worse counts (`(0.6441×177 − 0.5766×137) / 40 = 0.875` exactly) — which is why
the numbers above differ from an earlier measurement of this same harness.

**TWFE on P4 reads `STRONG`, its highest possible band, on the estimate this project's own Phase 3
section documents as off by −0.9946 against a true ATT of 2.8361 — 88.8 Monte Carlo standard
errors from zero, a 35% error.** This was pre-registered as the harness's most likely failure mode
before it was run: P4's staggered adoption with a ramping effect gives already-treated units
negative weight as controls, which is exactly what the Goodman–Bacon decomposition exists to
expose, and no assumption check in this phase tests for it. `parallel_trends` passes because
pre-treatment trends genuinely are parallel on P4 — the bias is a within-estimator weighting
artifact, not a violated identifying assumption. Because P4's adoption is imposed rather than
selected on the outcome, its design provenance is `natural_experiment`, whose ceiling is `STRONG`
outright — higher than the `MODERATE` the harness's own pre-registration expected, since nothing
here caps an unconfounded, badly-weighted estimate below the design ceiling. The validity score
is a check on observable diagnostics, exactly as `VALIDITY_SCORE_LIMITS` states; it is not a
check on estimator-specific weighting behaviour that only a decomposition like Goodman–Bacon can
see, and this is the sharpest concrete case of that limit found so far.

## Sensitivity: confounding, specification, falsification (Phase 9)

Three independent lines of attack on one finding, reported under a single `SensitivityReport`:
how strong an unmeasured confounder would have to be, whether the answer survives the analyst's
discretionary choices, and whether the finding fails tests it should fail if it were spurious.
They are independent on purpose. A clean confounding bound says nothing about whether the estimate
moves under a different covariate set, and neither says anything about whether a placebo period
looks the same. Reporting all three under one type is what stops a reader taking one clean answer
as the whole picture.

### Every absence names its reason

Each optional slot is paired with a `_reason` sibling, and the pairing is enforced at construction
rather than documented: a value present with a reason attached is refused just as firmly as a value
absent with none. An absence with no stated reason is the exact shape that let an unregistered
estimator ship silently in Phase 5, and every phase since has paid for closing that gap one place
at a time. `falsify` returns four techniques, each independently absent-able with its own reason,
because each is defined for a different family of estimator.

### The E-value's ratio is exact where the data carry one, approximated where they do not

`approximated` is a structural field, not prose a renderer could drop, because an approximated
E-value read as though it were the exact ratio-scale quantity is this phase's leading misuse risk.

| `estimate.scale` | ratio used | `approximated` |
|---|---|---|
| `log_ratio` | `exp(point)` — the estimate already is a log ratio | `False` |
| `percentage_point` with a usable `CounterfactualPoint` | `(predicted_absent_treatment + point) / predicted_absent_treatment` | `False` |
| `absolute`, or `percentage_point` with no usable counterfactual | `exp(0.91 × effect / outcome_sd)` | `True` |

The second row is why every cross-sectional scenario here gets an exact answer rather than a
converted one: their outcomes are binary, so the predicted untreated mean the estimators already
report turns the reported effect into a genuine risk ratio. Approximating a quantity that is
exactly available would have been strictly worse.

**The numerator is the baseline plus the reported effect, not the raw treated-arm mean, and the
difference is not cosmetic.** For `aipw`, `difference_in_means`, `did_2x2` and `callaway_santanna`
the counterfactual baseline is derived from the point, so the two forms are the same number. For
`lin_regression` they are not — `observed_mean` is the raw treated-arm mean while the point is the
Lin-*adjusted* ATE — and for `ipw` they are not either, since `observed_mean` is unweighted while
the baseline is Hájek-weighted. Shipping the raw arm ratio there computed an E-value for a
different estimate than the one being reported, always in the direction of overstating robustness:
measured on scenario G at `n_units=600`, seed 1, it gave **2.338** for an `ipw` point of 0.00077
(honest value 1.053) and **3.332** for a `lin_regression` point of 0.06251 (honest value 1.868).
`e_value_ci_bound` was already derived from the reported interval, so the two fields of one
`EValueResult` described two different estimates. Both now come from the same one.

A "usable" counterfactual is one with a positive baseline whose sum with the reported effect is
also positive — a ratio has to be positive to have an E-value at all. Either failure falls through
to the conversion row, which is why that row's `outcome_sd` requirement is not optional.

The `0.91` in the third row is load-bearing. It is the Chinn (2000) standardised-effect conversion
as used in VanderWeele–Ding's E-value guidance; plain `exp(d)` is a different and larger quantity,
and using it would overstate every E-value computed on that path. The conversion also assumes an
approximately normal, non-rare outcome — the *opposite* assumption from the rare-outcome derivation
the classical E-value is stated under — and `approximation_note` says so rather than leaving it
implicit.

### Cinelli–Hazlett covers four estimators, not five

`lin_regression`, `difference_in_means`, `did_2x2` and `twfe`. **`event_study` is deliberately
excluded**: its reported point is the mean of its post-treatment horizon coefficients, not one
coefficient, so the partial-R² quantity these bounds are read through has no single regressor to
attach to. The robustness value would still evaluate on that linear combination's t-statistic, but
shipping a number whose interpretation does not hold is worse than a stated absence.

Each bound refits the design the named estimator itself fitted, through that estimator's own design
function rather than a rebuilt copy — the failure Phase 7 found when two estimators had each
written out the same placebo formula. `difference_in_means` is the one exception, since it computes
a Welch t-test and has no design function; OLS on `[1, treatment]` is the same estimand written as
a regression, and a test pins the reconstructed coefficient against what the estimator actually
reports for all four.

**The t-statistic is the ordinary-least-squares one, and on `twfe` and `did_2x2` that ignores
clustering.** Those estimators report cluster-robust standard errors, but these bounds are derived
under homoskedastic OLS, so a clustered t-statistic would produce a number matching no quantity in
the derivation. The bound is correspondingly optimistic where clustering matters.

### The specification curve is three estimators wide, and the empty covariate subset never survives

`lin_regression`, `ipw` and `aipw`. `difference_in_means` refuses because it never reads covariates
— measured, it returns a bit-identical point for every subset, so its curve would be N copies of
one number, which reads as robustness rather than as an inapplicable analysis. `causal_forest`
refuses on cost: 4.56s per fit against `lin_regression`'s 0.02s. Both raise rather than returning
an empty curve, because "not built for this estimator yet" is a fact a direct caller should catch
immediately.

Candidate specifications are filtered through each estimator's own `Requirements.unmet()` rather
than a hard-coded rule, which is what drops the empty covariate subset: all three declare
`needs_covariates`, so the "full power set" the design called for is in practice every *non-empty*
subset. On the two-covariate cross-sectional scenarios that gives **6** specifications for
`lin_regression` (3 subsets × 2 outlier rules) and **18** for `ipw` and `aipw` (× 3 trim values).
The 256-specification cap cannot bind on any scenario in this repository — it is reachable only at
seven or more covariates — so it is exercised against a synthetic wide frame instead.

The trimming axis ships as `(0.01, 0.05, 0.10)`, running upward from `POSITIVITY_THRESHOLD` rather
than straddling it. The straddling version this phase's plan proposed, `(0.005, 0.01, 0.02)`, was
measured to clip zero rows on two real scenarios — an axis that moves nothing is not an axis, and
that was caught by measurement rather than asserted.

Leverage is computed once, from the full declared covariate set, so the outlier axis excludes the
same rows at every point. A per-subset rule would move two things at once and the axis would no
longer isolate the choice it names.

### Rosenbaum bounds are an intentional omission

They are built for matching estimators, and this codebase has none — propensity weighting,
regression, AIPW and causal forest, but no nearest-neighbour matching. Force-fitting an adapted
version would be a research judgment call rather than a mechanical port, so it is not attempted.
Recorded here as a decision, not left as a silent gap.

### Randomization inference detects a real effect 30% of the time, and that is recorded

Section 7.1's design pre-registered the expectation in both directions before the harness was run:
on a genuinely null scenario the real estimate should sit unremarkably inside the permutation
distribution, and on a scenario with a strong true effect it should sit in the tail, "near 1.0".
Measured over 20 seeds × 200 permutations at `n_units=800` on `lin_regression`:

| scenario | true effect | share of seeds at `p ≤ 0.05` | MCSE |
|---|---|---|---|
| B (null) | 0.0 | **0.050** | 0.049 |
| A (effect) | 0.05 | **0.300** | 0.102 |

**The null direction held almost exactly on nominal. The effect direction did not hold**, and it is
recorded rather than patched — the treatment Phase 6 gave TWFE-on-P4 and Phase 7 gave its
credible-interval undercoverage.

It is not a broken permutation, and the null row is the evidence: a permutation correlated with the
outcome would inflate the false-positive rate, and this one lands on 0.050 against a nominal 0.05.
It is real, modest power. Scenario A's effect of 0.05 sits at about 1.56 times the permutation
null's own spread (SD ≈ 0.032) at this sample size, and standard power arithmetic for a two-sided
test at α = 0.05 predicts P(Z > 1.96 − 1.56) ≈ 0.345 — consistent with the measured 0.300 well
inside its MCSE. The test was renamed for what it found and now brackets the rate from both sides
(0.10 ≤ rate ≤ 0.50) rather than asserting the pre-registered claim.

**The 20-seed protocol has limited power to catch a regression, and that limit is stated rather
than implied away.** At 20 seeds, an estimator whose true detection rate collapsed from 0.30 to
0.10 would still pass this test about 61% of the time, and one that fell to chance level (0.05)
about 26% of the time. Raising the seed count is what would fix that; it was not raised here
because every other threshold in the harness was measured under this protocol, and re-measuring
one number at a different protocol than its neighbours is this project's most frequently repeated
defect.

Leave-one-donor-out on the clean donor scenario S1, same 20 seeds: mean spread **0.2984** (MCSE
0.0494), max 1.0137 — a clean estimate should not hinge on any single donor, and the harness
asserts a ceiling of 0.40 on the mean.

### The donor-pool placebo distribution is convex-hull-aware, unit by unit

`falsify`'s `placebo_units` slot on `synthetic_control` puts every unit — the real treated one and
each donor playing "treated" in turn — through the same `fit_donor_pool_point` call, so each runs
its own `convex_hull` check against its own donor pool and takes the ridge-augmented fallback on
its own terms. The reported effect is then one draw from that distribution in the same estimator
form, rather than a hull-aware number plotted against a uniformly simplex-based reference.

An earlier version drew the placebo effects from `placebo_runs`, which is simplex throughout, and
the gap showed up exactly where the fallback exists. On S3, seed 0: the real effect is **+1.18353**
from the ridge form, against a simplex placebo distribution of mean **0.04255**, sd **0.67820**.
Refitted hull-aware, **2 of 17** donors fail their own hull check and switch — donor 14 moves from
−1.96713 to −0.85951, donor 11 from 0.51641 to 0.42087 — giving mean **0.10208**, sd **0.49954**.
On S1 the donors are the same series, so the placebo distribution moves identically; what differs
is that S1's treated unit stays inside its hull and its own effect is unchanged at **+1.41401**.

**The p-value did not change and was not meant to.** It stays the rank of the post-to-pre RMSPE
ratio over the simplex `placebo_runs` — Abadie's statistic, which `donor_contamination_check` is
calibrated against and which discounts a placebo unit that was never fitted well before treatment.
Only the effect distribution became hull-aware. `placebo_runs` itself is untouched: it still backs
the confidence interval and the contamination check with bit-identical output.

Both procedures iterate one `_unit_swaps` generator rather than each writing out "unit `i` treated,
the rest as its pool, real unit at index 0" for itself. Two independently written copies of that
convention drifting apart is a defect this branch already shipped twice, and one definition is what
stops a third.

The extra work is one hull check per placebo unit plus a ridge solve for each unit that fails it,
on top of a simplex solve that already dominates. Measured warm on seed 0, five calls:
`falsify(dataset, "synthetic_control", config)` goes from **0.0434s to 0.0793s** on S1 and from
**0.0356s to 0.0699s** on S3 — a factor of 1.8–2.0, under a doubling but not far under it.

### The donor-pool family gets a placebo-date test, not a stated absence

`falsify`'s `placebo_dates` slot was reported absent for every donor pool, with an accurate reason
attached to a wrong conclusion. The accurate part: `placebo_leads` builds its cells from a
within-group variance needing at least two units on each side, and a donor pool has exactly one
treated unit by construction, so it yields zero cells on S1's 17 pre-treatment periods as readily
as on S5's 2. The wrong part: "the panel technique does not apply here" is not the same claim as
"no placebo-date test applies here", and the slot was reporting the second while only the first had
been established.

The technique that does fit the design was already in the codebase in miniature. `SyntheticControl`
reports an `in_time_placebo_effect` diagnostic: refit the donor weights on the first half of the
pre-period, measure the gap over the second half, and expect roughly zero, since nothing was
treated yet. Sweeping that fake date across the whole pre-period instead of fixing it at the
midpoint turns one number into a distribution the real effect can be ranked against. On the donor
protocol — 24 periods, adoption at 18 — the pre-period is 17 periods, a window needs 2 to train on
and at least one left to measure over, so the sweep visits training lengths 2 through 16: **15
windows**.

**The floor of two training periods is not a style choice.** A window trained on a single period is
unidentified rather than merely weak: one equation over 17 simplex weights is satisfied exactly by a
whole face of solutions, and which one SLSQP lands on decides the reported gap. Measured on S1, the
sweep over training lengths 1–16 returns **−1.55847** at length 1 against a range of
**[−0.67421, +0.55785]** across every other length — not a draw from the same distribution, an
artefact of the solver.

**Both estimators run their own version, and that is the whole point.** Each synthetic-control
window goes through `fit_donor_pool_point`, so it runs its own convex-hull check over its own
training slice and takes the ridge-augmented fallback on its own terms — the same rule
`hull_aware_placebo_runs` follows in the section above, for the same reason. On S3 all 15 windows
take the ridge path, as the real fit does. `bsts` gets `bsts_in_time_placebo_sweep`, a full
posterior refit per window, with `real_effect` from the same `fit_bsts_counterfactual` pipeline
`BSTS.fit` reports its own point from. Reporting synthetic control's sweep under the `bsts` name
would have been the bug this branch already fixed once for `placebo_units`, in a slot that had not
yet been built when it was fixed.

Only the two slots *defined in terms of* the simplex weight vector stay deferred for `bsts`: a
donor swap needs weights to swap. An in-time placebo needs no weights at all — only that the
estimator can be refitted at an earlier date, which `bsts` can be.

**These placebo effects are not independent draws, and the docstrings say so where a caller meets
them.** Consecutive windows share almost their whole training slice — the length-9 window's
training periods are the length-8 window's plus one — and every window measures its gap over an
overlapping tail of the same pre-period. The rank is an honest ordering and `_rank_p_value` is
reused rather than replaced, but the resulting p-value must not be read as one from 15 independent
placebo experiments; the effective number of distinct looks is far lower and no correction here
recovers it. The windows are not identically distributed either: short training slices fit noise
the real 17-period fit would not, and short measurement slices average fewer periods. Both widen
the reference distribution, which biases the rank toward the null — conservative, but a real limit
on power.

Measured across 20 seeds, S1 (τ = 1.0) against S0 (the same DGP at τ = 0.0):

| | mean p | share on the 1/16 floor |
|---|---|---|
| S1, a real effect | **0.2000** (MCSE 0.0450) | **0.4500** (MCSE 0.1141) |
| S0, a genuine null | **0.6312** (MCSE 0.0586) | **0.0000** — 0 of 20, minimum p seen 0.2500 |

That pair is cleaner to compare than it first looks. The sweep reads only pre-treatment periods and
τ enters only after adoption, so **S0 and S1 produce byte-identical placebo distributions at every
one of the 20 seeds** — verified directly, not assumed. The entire difference between those two
rows is the real effect moving through a fixed reference distribution.

Note that 0.0625 is the smallest p-value this slot can return, so the "share at p ≤ 0.05" reading
used by the randomization-inference tests would be identically zero for both scenarios and would
measure nothing. The share landing *on* the floor is the analogous statistic.

**Cost, and the knob it forced.** A synthetic-control window is a simplex solve, so the sweep is
roughly 15 more `fit_donor_pool_point` calls on top of the ~37 that `placebo_runs` and
`hull_aware_placebo_runs` already do — about 40% more solves, and the measurement agrees: warm,
median of 10 calls over three blocks, `falsify` goes to **1.36×** its unswept cost on S1 and
**1.14×** on S3. (Ratios rather than absolute times, and measured in a separate worktree: they are
not comparable with the absolute figures in the section above, which were taken in a different
environment.) The spread between scenarios is real — SLSQP's iteration count is data-dependent.

`bsts` is a different order of cost entirely: each window is a fresh NUTS run at its own
pre-period length, which in JAX also means a fresh compile, and nothing is shared between windows.
Measured on S1 at `FAST_MCMC_CONFIG`, about **5s per window and 81.5s** for the uncapped sweep; at
`PRODUCTION_MCMC_CONFIG` the same sweep is a tens-of-minutes computation. Hence `n_placebo_dates`
on both `falsify` and `sensitivity_report`, following the precedent `n_placebo_units` already set
for the MCMC estimators: default to the full sweep, document what it costs, let a caller spend
deliberately. Capping is not free and says so — the rank p-value's floor is `1 / (n_windows + 1)`,
so a 3-window sweep cannot report below 0.25 however extreme the real effect is — and it keeps both
ends of the training-length range rather than sampling one end of a regime known to differ at each.

A pre-period too short to hold one window still reports a stated absence, now naming the length
that actually fired rather than the family: S5's 2 pre-treatment periods against the 3 needed. The
sweep returns empty before fitting anything, so an estimator that cannot run this test does not pay
a posterior to discover it.

### What is deferred, and why

- **Negative-control outcomes.** A genuine negative control needs an outcome column provably
  unaffected by treatment by construction. Improvising one from an existing covariate was
  considered and rejected: testing against a covariate is not testing against an outcome, and the
  difference matters for what the check can honestly claim.
- **Missing data** and **treatment misclassification.** Both need the sensitivity check validated
  against *measured* ground truth, and no component generates missing values or misclassifies
  treatment while keeping the true version recoverable for scoring. That is new data-generating
  work of the same shape as Phase 4's donor pool.
- **Panel randomization inference.** Permuting the treatment column on a panel would break the unit
  and period structure and the absorbing-adoption pattern. A correct version permutes adoption at
  the unit level, which is different machinery.
- **`placebo_units` and leave-one-donor-out for `bsts`.** `bsts` declares a single treated unit
  exactly as `synthetic_control` does, so `falsify` classifies both into the same family — but
  these two techniques are *defined in terms of* synthetic control's simplex donor-weight solve and
  the ridge-augmented fallback layered over it: a donor swap needs weights to swap. `bsts` has no
  such vector — its donors enter as state-space regressors under shrinkage priors — and a faithful
  version of either test would refit the full posterior once per donor. Both slots are reported
  absent with that reason rather than filled with another estimator's numbers, which is what the
  branch did before it was gated on `synthetic_control` by name.

  **This deferral no longer covers the whole family's falsification.** `placebo_dates` is not
  defined in terms of donor weights, so `bsts` fills it from its own posterior sweep — see "The
  donor-pool family gets a placebo-date test" above. What remains deferred is these two slots.
- **A measured rate for the `bsts` in-time placebo.** The slot is filled and its *shape* is pinned
  — that the numbers reported under the `bsts` name are `bsts`'s own, window by window — but its
  false-positive and detection rates are not measured, and that is a deferral with arithmetic
  behind it rather than an oversight. A rate needs seeds: 20 seeds over a null/effect pair is 160
  full NUTS runs. At `FAST_MCMC_CONFIG` that is about 13 minutes, but that config's own docstring
  says it exercises code paths and array shapes rather than producing a trustworthy posterior, so a
  rate measured there would characterise the sampler rather than the design — the exact thing
  `test_falsification_measured.py` exists to avoid. At `PRODUCTION_MCMC_CONFIG` the same protocol
  runs into hours. And a capped sweep puts the p-value floor at `1 / (n_windows + 1)`, leaving the
  statistic little resolution to detect a rate with. Recorded in that file's docstring rather than
  shipped as a weak claim.
- **Panel and donor-pool specification-curve axes** — time windows, control-group definitions,
  donor-set variants — matching how every prior phase shipped one estimator family's version of a
  capability before generalising.

## Negative controls and treatment misclassification (Phase 10)

Two more falsification techniques, both deferred out of Phase 9 because they need real data-
generating capability rather than a computation over an already-fitted estimate: a second outcome
that shares treatment's confounders but cannot be affected by treatment, and a treatment column
recorded with error while the true one stays available for scoring.

### What a negative-control finding means, and what it does not

`negative_control_test` refits the same estimator on a column that treatment cannot affect by
construction — `NegativeControlOutcome`'s `probabilities` never takes treatment as an argument, so
no configuration of it can make the column depend on treatment. **An estimate that excludes zero
there is evidence of residual confounding**: the only paths connecting treatment to that column run
through the confounders, so an association there is not an effect.

**A null result is a weaker claim than "no confounding."** It says this particular control found
none, which is not the same as there being none — a confounder the control doesn't share with the
real outcome would go undetected. `NegativeControlResult` carries no interpretation string of its
own (no report renderer exists yet in this codebase), so this caveat lives here and in the
function's own docstring rather than in a field a caller could drop.

### The misclassification correction moves away from the null, and cannot explain a finding away

`misclassification_curve` corrects an observed difference-of-means estimate for non-differential
treatment misclassification — a treatment column recorded with error, at rates that do not depend
on the outcome. Non-differential error attenuates toward the null, so **every corrected point in
the curve lies further from the null than the observed estimate, never closer.** This is the
phase's leading misuse risk: the correction can only say the true effect may be larger than
reported, never smaller, and never that a real finding was an artefact.

**The naive `(se + sp − 1)` attenuation factor is wrong except when the true and observed treated
shares happen to coincide.** With `se`/`sp` the sensitivity/specificity, `q` the *observed* treated
share, and `p` the *implied true* share solved from `q`, the correct factor is
`(se + sp − 1) · p(1 − p) / [q(1 − q)]` — the naive form drops the prevalence-ratio term, which
does not cancel except in the balanced, symmetric case. Verified against a simulated ground truth
of 4,000,000 units per row across four `(q, se, sp)` configurations, three of them asymmetric, all
recovering the true effect within 0.01. `q` is read from the fitted estimate's own
`n_treated`/`n_control`, never accepted as a caller-supplied parameter that could disagree with it
— the same two-values-that-must-agree failure shape as Phase 9's E-value defect.

**A rate pair whose implied correction would divide by almost nothing is dropped from the curve.**
The correction divides the point and both interval endpoints by the attenuation factor, and that
factor falls continuously to zero as the assumed rates approach the edge of what the observed
treated share admits. `MIN_ATTENUATION_FACTOR` puts a floor of 0.2 there — a corrected point may
be at most five times the observed one — and both `misclassification_curve` and
`MisclassificationPoint` enforce it, so no assembler of a curve can route around it. The floor was
set by measurement: across observed treated shares from 0.02 to 0.98 it keeps 88.0% of the default
grid's cells against 90.8% with no floor at all, and over the ordinary 0.2–0.8 band of treated
shares it costs nothing. Where no pair survives but the identity anchor, `sensitivity_report`
reports no curve and states that reason, rather than returning a one-point curve that would read as
"mismeasurement cannot move this finding".

**Only non-differential misclassification is modelled.** Differential misclassification — where the
error rates themselves depend on the true outcome — could manufacture an association where none
exists, which is a different and harder analysis than correcting an attenuation. This is an
intentional scope limit, not a gap: the tractable case is the one built, and the harder one is named
rather than approximated.

### What the measured harness found

All three of spec §7.3's pre-registered expectations held as stated. Measured 2026-08-29, 20 seeds
at `n_units=2000`:

| check | scenario | result |
|---|---|---|
| Negative control, randomised | A, four of the five covered estimators | fires 0.000–0.050 (nominal ≈ 0.05) |
| Negative control, confounded, unadjusted | D and G, `difference_in_means` | fires **1.000** (20 of 20 seeds) |
| Negative control, confounded, adjusted | D and G, `lin_regression`/`ipw`/`aipw` | fires 0.000–0.050 |
| Misclassification recovery (`se=0.9, sp=0.9`) | A, `difference_in_means`, true ATE 0.05000 | observed bias −0.01496; corrected bias −0.00619 (MCSE 0.00525) |
| Misclassification recovery (`se=0.8, sp=0.95`) | A, `difference_in_means`, true ATE 0.05000 | observed bias −0.01543; corrected bias −0.00494 (MCSE 0.00603) |

The separation on D and G is total, not a near-miss: every seed where `difference_in_means`
(which adjusts for nothing) fit the negative control, it found a spurious effect; every seed where
an adjusting estimator fit the same column on the same data, it did not. Two of these rates are
literal unanimous shares (20 of 20, or 0 of 20), which the binomial formula gives an MCSE of exactly
0.000 — a property of the formula at that share, not evidence of certainty, so the measured harness
uses rule-of-three reasoning for those thresholds rather than `mean ± MCSE` arithmetic. The
correction's residual bias sits under 1.2× its own Monte Carlo standard error at both rate
pairs — genuine recovery, not a partial one.

`causal_forest` is covered by `negative_control_test` but excluded from the measured harness on
runtime cost: the check is one extra full fit, and that fit is a measured 4.56s there against
milliseconds for the other four. The harness therefore measures `difference_in_means`,
`lin_regression`, `ipw` and `aipw`.

The harness asserts the **asymmetric** `se=0.8, sp=0.95` pair rather than the symmetric one.
At `se=sp=0.9` on scenario A the true and observed treated shares coincide, which is the one
configuration where the naive `(se + sp − 1)` factor equals the correct one — so an end-to-end
claim made only there would also have succeeded under the wrong formula.

### What is deferred, and why

- **Panel and donor-pool variants of both checks.** Both refuse a panel or donor-pool estimator by
  name, through `Requirements.needs_panel`, and `SensitivityReport` and `FalsificationResult` each
  carry the stated reason beside the empty slot. A negative control on panel data needs its own DGP
  machinery, since the control column has to share the panel's time structure as well as its
  confounders. The misclassification correction needs the treated share of the regressor, and on a
  panel the fitted estimate's arm counts are treated and control *units* while the regressor is the
  unit × period indicator — on the donor-pool estimators the treated count is a hardcoded 1, which is
  not a prevalence at all. Correcting with either number would return a figure that looks like an
  answer, so the check declines instead.
- **Differential misclassification.** Named above — a harder analysis that this phase's tractable,
  non-differential model does not attempt.
- **Missing data.** Moves to Phase 11, the last of §7.1's five original concerns.

## Missing outcomes: complete-case analysis and tipping-point bounds (Phase 11)

The last of §7.1's five original concerns. `CausalDataset` has always permitted missing outcomes,
but no estimator handled one — before this phase, a dataset with a missing outcome failed late and
differently per estimator (`ValueError: cannot convert float NaN to integer`,
`Interval bounds must be finite, got lower=nan`, `Input y contains NaN` from scikit-learn, or an
unrelated `LinAlgError` from a missing covariate), naming no column and suggesting no fix.

### Estimators now refuse missing data, and complete-case is opt-in, not automatic

`arrays()` and `panel_arrays()` — the two shared choke points every cross-sectional and
panel/donor-pool estimator fits through — now refuse a dataset with any missing value in a required
column, by name and by column, before any estimator sees it. Dropping the incomplete rows is offered
as a separate, explicit function, `complete_cases`, rather than something an estimator does on a
caller's behalf. **Complete-case analysis is unbiased when values are missing completely at random
(MCAR) and biased in general when they are not** — under MAR, where the chance of missingness
depends on observed covariates, or MNAR, where it depends on the outcome itself. Silently dropping
rows would make that assumption invisible; refusing forces a caller to choose it deliberately, and
the choice is recorded (`CompleteCaseRecord`, which rows were dropped and which column drove each).
Complete-case is a starting point offered for convenience, not a recommendation — the error message
that points to it says so, rather than presenting it as the fix.

### What a wide bound means

`tipping_point` reports the estimate under the two extreme corner fills — every missing outcome
filled with the value most opposed to the finding, then with the value most in its favour — and,
when the finding can be overturned at all, the smallest share of the missing outcomes that would
need filling adversely to remove its statistical significance. For `difference_in_means` and `ipw`
those corners are the Manski worst case outright, because each unit's weight keeps a constant sign
within its arm. For `lin_regression` they are not: its Lin-interacted coefficient weights some units
negatively, and flipping a single filled unit off the corner has been measured going past the
reported bound. The property that holds for every estimator the gate admits, and the one the tests
assert as an identity, is that the pair contains the complete-data estimate. **These bounds widen with the missing fraction, so
at high missingness they routinely span zero.** That is the honest answer the data admits, not a
failure of the method: a bound this wide says the data cannot settle the question, and the
measured harness below found exactly that at 25% missingness — the mean bound width was 0.485 on a
binary outcome whose full range is 1.0.

The bisection that finds the tipping fraction also checks more than significance: an adverse fill
that drives a finding through zero and out the other side, while its interval still excludes zero,
is a **reversal**, not survival, and `_conclusion_holds` requires the refit to agree in sign with
the original finding as well as remain significant. A significance-only check would report that
case as robust at the exact point it had been overturned.

### Only binary outcomes on cross-sectional data are covered

The worst-case fill needs a bounded outcome to fill the gaps with — 0 and 1 for a binary outcome —
so a continuous outcome has no worst case to compute, and `tipping_point` returns `None` with a
stated reason. Panel and donor-pool estimators are out of scope for the same reason Phase 10's two
checks stopped at cross-sectional data: attrition on a panel also removes a unit from periods it
would otherwise contribute to, which this phase's machinery does not model. Both `None` cases carry
their own reason string, checked by the language walk the same way Phase 9 and 10's reasons are.

### Imputation is Phase 12

This phase builds the missingness mechanism, complete-case analysis as an explicit baseline, and the
tipping-point bounds — two of the three pieces the parent spec's original "complete-case vs
imputation comparison, tipping-point analysis" row called for. **Imputation, and any comparison
between complete-case and imputed estimates, is Phase 12.** Tipping-point analysis needs no model
for the missing values, which is the point of it, so it does not depend on what Phase 12 builds.

### What the measured harness found

All four of spec §7.3's pre-registered expectations held as stated. Measured 2026-08-30, 20 seeds at
`n_units=2000`, `difference_in_means`, scenario D unless noted, MNAR at `rate=0.25, strength=2.0`:

| check | scenario | result |
|---|---|---|
| Complete-case bias, MCAR (`strength=0`) | D | mean gap −0.00189 (MCSE 0.00256) — the control, near zero |
| Complete-case bias, MNAR | D | mean gap −0.07699 (MCSE 0.00390) — clearly biased |
| MNAR bias exceeds MCAR bias by | D, paired | 0.07510 (combined MCSE 0.00467) |
| Worst-case bounds contain the complete-data estimate | D | 0 violations of 20; mean width 0.4850 |
| Tipping fraction, null finding | B | 0.00675 (MCSE 0.00423); 0 of 20 seeds survived full adverse fill |
| Tipping fraction, +5pp finding | D | 0.24645 (MCSE 0.01893); 0 of 20 seeds survived full adverse fill |

Neither B nor D survives its own full adverse fill in any of the 20 seeds at this missingness
severity — a 25%, strength-2.0 MNAR mechanism is a genuinely severe pattern, and it is plausible for
it to erase even a real +5pp effect entirely at full adversity. What discriminates the two scenarios
is not survival but **how much** adverse filling each takes: D's finding needs roughly 24.6% of its
missing outcomes filled against it before its significance breaks, against roughly 0.7% for a
finding with no real effect behind it in the first place — a difference of 0.240 against a combined
MCSE of 0.0194, ruling out the two being indistinguishable.

The bounds identity — that the worst-case interval always contains the complete-data estimate — is a
mathematical property of `difference_in_means` on a binary outcome (strictly monotonic in each
missing coordinate, so its extremes occur exactly at the two filled corners the code computes), not
a statistical claim, and is asserted with zero tolerance rather than headroom: one violation in
`test_the_bounds_always_contain_the_complete_data_estimate` would mean the bounds are wrong, not
that the tolerance needs widening.

### What is deferred, and why

- **Imputation and the complete-case-vs-imputation comparison.** Phase 12.
- **Continuous outcomes and panel data, for the tipping point.** Named above — no worst case to fill
  a continuous gap with, and panel attrition needs machinery this phase does not build.
- **A panel missingness mechanism.** `MissingnessMechanism` attaches to `CrossSectionDGP` only.

## Imputation, and where it stops helping (Phase 12)

The other two of the parent spec's original "complete-case vs imputation comparison, tipping-point
analysis" row. `impute_once` fills missing outcomes from a logistic model of treatment, covariates
and an optional declared `auxiliary_columns` (schema fields an imputation model may see but no
estimator may adjust for, since an auxiliary built from the outcome is a descendant of it);
`multiple_impute` draws `m` such completions, bootstrap-resampling the observed rows before each
fit, refits the requested estimator on every one, and pools by Rubin's rules (`cp_statistics.pooling`,
Barnard–Rubin degrees of freedom). `compare_missing_data` runs complete-case, single imputation,
multiple imputation and the Phase 11 tipping-point bounds on the same dataset and reports all four
together, agreeing or not, rather than picking one.

### Multiple imputation buys nothing without an auxiliary, and does not repair MNAR by itself

Both were pre-registered expectations going in, and both were falsified by measurement against the
shipped code — recorded as falsified rather than quietly rewritten, since a pre-registration that
gets edited after seeing the data stops being one:

- **MI with no auxiliary was expected to be ~8% *less* efficient than complete-case. Measured:
  statistically indistinguishable** (paired RMSE ratio 0.984, paired diff −7.1e-06 against an MCSE
  of 2.9e-05). This is the theoretically correct answer, not a surprising one in hindsight — Little
  (1992): imputing an outcome from information the analysis model already has gains nothing and
  loses nothing. The naive expectation that "any imputation helps" is the one to distrust.
- **Under MNAR, MI was expected to leave the complete-case gap roughly unchanged. Measured: it
  repairs about two thirds of it** and leaves a residual that is unambiguously nonzero. The
  mechanism the spec missed: this project's `OutcomeProxy` is built from the *complete* outcome, so
  it carries information about the missing values themselves, which makes MNAR partially ignorable
  *given the proxy*. Imputation is still not a general fix for MNAR — a proxy that carried no
  outcome information would not repair anything — but an auxiliary correlated with why data is
  missing changes what MNAR means for this estimate.

An auxiliary that genuinely predicts the missing values is a different story: **MI *with* one beats
complete-case, by a margin that shrinks as the auxiliary gets noisier** — paired diff −2.09e-04
(ratio −4.43 against its MCSE) at `noise_sd=0.25`, −1.24e-04 (ratio −2.73) at `noise_sd=0.5`; the
gradient continues to `noise_sd=1.0` but 40 seeds cannot separate it there. Single imputation is the
one arm that should not be trusted for its stated uncertainty: its SE calibration (mean reported SE
÷ empirical SD; 1.0 is honest) falls monotonically as the imputation gets less certain — 0.84 → 0.79
→ 0.75 across the same three noise levels — while multiple imputation stays close to complete-case's
0.94 throughout, because treating an imputed value as if it were observed is exactly the
overconfidence Rubin's rules exist to correct.

### What the measured harness found

Measured 2026-09-01, 40 seeds at `n_units=2000`, `m=20`, scenario D, `MissingnessMechanism(rate=0.25,
strength=2.0)`, `OutcomeProxy(noise_sd=0.5)` unless noted. Full tables in
`tests/causal/test_phase12_measured.py`.

| check | result |
|---|---|
| MAR gap, `difference_in_means`, complete-case | −0.04011 (MCSE 0.00232) — 17 MCSE from zero |
| MAR gap, same, `mi_aux` | −0.00056 (MCSE 0.00150) — indistinguishable from zero |
| MI-no-aux vs complete-case, `lin_regression`, paired RMSE ratio | 0.984 — indistinguishable |
| MI-aux vs complete-case, paired diff, `noise_sd=0.25` | −2.094e-04 (ratio −4.43) |
| MI-aux vs complete-case, paired diff, `noise_sd=0.5` | −1.243e-04 (ratio −2.73) |
| Single-imputation SE calibration, `noise_sd=0.25 / 0.5 / 1.0` | 0.8406 / 0.7927 / 0.7532 — monotone |
| MNAR gap, `difference_in_means`, complete-case → `mi_aux` | −0.07566 → −0.02551 (66% repaired, residual 14.5 MCSE from zero) |
| Between-imputation variance, `lin_regression`, `rate=0.05 / 0.10 / 0.25` | 1.10e-05 / 2.73e-05 / 7.46e-05 — rises with the missing rate |

### What is deferred, and why

- **Continuous outcomes and panel data.** `multiple_impute` and `compare_missing_data` share Phase
  11's `tipping_point` limits: the imputation model here is logistic, so only a binary outcome on
  cross-sectional data is covered.
- **A scaling policy for the imputation model.** `LogisticRegression`'s L2 penalty is applied to
  unstandardised features, so its regularisation is scale-dependent; rescaling `OutcomeProxy` by
  0.2×/5× moved `mi_aux`'s empirical SD and calibration by an amount too small to separate from
  noise at 20 seeds. Suggestive, not resolved — carried to a future phase rather than fixed on a
  measurement this uncertain.

## On scikit-learn

scikit-learn fits nuisance models only — the propensity model and the outcome regressions, which
are prediction problems. Every causal quantity is computed here: the weighting scheme, the
influence function, the cross-fitting structure, the standard errors, and the diagnostics.

## Development

```bash
make install    # sync the uv workspace
make check      # lint, type-check, import contracts, tests
make test-fast  # skip reference-parity tests
```
