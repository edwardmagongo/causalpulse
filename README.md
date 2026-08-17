# CausalPulse

A causal inference and experimentation platform. Given an intervention and an outcome,
CausalPulse estimates what would have happened without the intervention, how large the effect
was, for whom it differed, and how much the conclusion should be trusted.

## Status

Phases 1 through 3 of the causal core are implemented.

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

The Bayesian structural time series counterfactual, heterogeneous-effect estimation, the robustness
suite, and the benchmark harness are still ahead.

## Packages

| Package | Responsibility |
|---|---|
| `cp-core` | Value types, dataset contract, and the result contract. No algorithms. |
| `cp-statistics` | Proportion intervals, robust and cluster-robust covariance, bootstrap engines. |
| `cp-simulation` | Compositional data-generating processes carrying computed ground truth. |
| `cp-causal` | Cross-sectional and panel estimators, propensity diagnostics, panel assumption checks, design router. Forbidden from importing `cp-simulation`. |

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
