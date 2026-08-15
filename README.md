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
explains *why* two-way fixed effects fails when it does. Every estimator is scored against
measured truth over 60 seeds at 400 units and 8 periods, and two-way fixed effects is additionally
checked against `statsmodels` as an independent reference oracle.

Synthetic control, the Bayesian structural time series counterfactual, heterogeneous-effect
estimation, the robustness suite, and the benchmark harness are still ahead.

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
0.1061 — 34%, at 18.3 Monte Carlo standard errors — **with no confounding and no differential
pre-trend**, measured at 800 units and 120 seeds. With random adoption the arms are compositionally
identical and the same measurement gives −0.0015 (0.8 MCSE), indistinguishable from zero — confirming
the bias requires selection on level, not the time path.

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
