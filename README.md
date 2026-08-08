# CausalPulse

A causal inference and experimentation platform. Given an intervention and an outcome,
CausalPulse estimates what would have happened without the intervention, how large the effect
was, for whom it differed, and how much the conclusion should be trusted.

## Status

Phases 1 and 2 of the causal core are implemented.

**Phase 1** delivered the foundations: the workspace, data contracts, statistical primitives, and
a simulation framework whose data-generating processes carry known ground truth.

**Phase 2** delivered cross-sectional estimation: the result contracts, the estimator protocol and
registry, the design router, and four estimators — difference in means, Lin regression adjustment,
inverse probability weighting, and cross-fitted AIPW — each scored against known truth.

Panel methods (difference-in-differences, synthetic control), heterogeneous-effect estimation,
Bayesian analysis, the robustness suite, and the benchmark harness are still ahead.

## Packages

| Package | Responsibility |
|---|---|
| `cp-core` | Value types, dataset contract, and the result contract. No algorithms. |
| `cp-statistics` | Proportion intervals, robust and cluster-robust covariance, bootstrap engines. |
| `cp-simulation` | Compositional data-generating processes carrying computed ground truth. |
| `cp-causal` | Estimators, propensity diagnostics, design router. Forbidden from importing `cp-simulation`. |

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
