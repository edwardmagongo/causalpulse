# CausalPulse Benchmark Study Report

## Overview

10 estimators, 13 scenarios, 4000 cells, 50 replications per configuration (coverage MCSE ~= 0.031).

## Headline Findings

Met: 7 · Failed: 1 · Inconclusive: 0

| ID | Verdict | Explanation |
| --- | --- | --- |
| E1 | met | On scenario B, where treatment is randomised and the true effect is zero, the four estimators that assume randomisation or unconfoundedness reject a true null at close to the nominal 5% rate: difference_in_means 0.06, lin_regression 0.06, ipw 0.05 and aipw 0.06, each from 100 fits with a Monte Carlo standard error of about 0.02. |
| E2 | met | On the randomised scenario A, difference_in_means covers the true effect 95% of the time (100 fits, standard error 0.022), which is the nominal level. |
| E3 | met | On the confounded scenario D, difference_in_means is biased by 0.135 (standard error 0.003) and its interval covers the true effect only 6% of the time, while aipw is within 0.003 of the truth (standard error 0.004). Scenario G shows the same split: difference_in_means is biased by 0.167 and covers 2% of the time, while ipw, lin_regression and aipw stay within 0.014 of the truth and near 95% coverage. |
| E4 | met | On scenario P4, staggered adoption with effects that differ across cohorts, twfe is biased by -1.06 (standard error 0.036), a sign reversal, and its interval covered the true effect in none of the 100 fits. callaway_santanna is within 0.006 of the truth (standard error 0.014), and hierarchical_did is within 0.019 with 92% coverage; event_study is biased by 0.67 and covers 3% of the time. Separately, callaway_santanna's coverage sits below the nominal level on P1 to P4 and P6, between 0.85 and 0.91 with a standard error of about 0.03, even where its bias is small; this is a gap of about two standard errors and is best read as suggestive. |
| E5 | failed | This expectation failed: hierarchical_did's pooled credible-interval coverage is 0.75 (standard error 0.019, 500 fits over P1 to P5), below the 0.90 threshold. The shortfall comes from one scenario. On P1 to P4 its coverage is 0.97, 0.93, 0.92 and 0.92, and its bias is within 0.07 of the truth. On P5, a scenario built to violate parallel trends, its coverage is 0.00 and its bias is 2.19. That is a shared pattern, not a weakness of this estimator alone: on P5 twfe is biased by 1.33, callaway_santanna by 1.78 and event_study by 2.28, and each of the three covered the true effect in none of its 100 fits. The expectation pooled a scenario designed to defeat the assumption every difference-in-differences estimator relies on, so a pooled figure near 0.90 was never available to it. |
| E6 | met | On scenario H, where outcomes are missing, no estimator produced a fitted result. The five estimators eligible for cross-sectional data each refused all 100 fits, and the remaining estimators are ineligible for cross-sectional data. |
| E7 | met | hierarchical_did is the slowest estimator at a mean of 49.2 seconds per fit (standard error 1.3, 500 fits), about three and a half times aipw at 14.4 seconds and more than six times event_study at 7.7 seconds. |
| E8 | met | All 4,000 cells of the study are present with an intact stored result, and no unfinished results remain. |

## Figures

![Coverage of each estimator's primary interval, by method and scenario, pooled over each scenario's baseline slice. Lighter cells have higher coverage; grey cells have no fitted result.](coverage-of-each-estimator-s-primary-interval-by-method-and-scenario-pooled-over-each-scenario-s-baseline-slice-lighter-cells-have-higher-coverage-grey-cells-have-no-fitted-result-466cc15874.svg)
![bias on scenario D, varying confounding_strength one value at a time.](bias-on-scenario-d-varying-confounding-strength-one-value-at-a-time-c113f8f9e9.svg)

## Method Comparisons

### Bias by method and scenario

| method | A | B | C | D | F | G | H | P1 | P2 | P3 | P4 | P5 | P6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aipw | 0.001247 | -0.002599 | 0.0002535 | 0.002559 | -0.0004385 | 0.01379 |  |  |  |  |  |  |  |
| callaway_santanna |  |  |  |  |  |  |  | 0.02725 | -0.01088 | -0.01098 | 0.005729 | 1.784 | 0.02508 |
| causal_forest | 0.001233 | -0.002655 | 0.0003579 | 0.002519 | -0.0004617 | 0.002893 |  |  |  |  |  |  |  |
| did_2x2 |  |  |  |  |  |  |  | 0.01446 | 0.004129 |  |  |  | 0.03748 |
| difference_in_means | 0.001712 | -0.00216 | 0.0009873 | 0.1349 | -0.0004104 | 0.1672 |  |  |  |  |  |  |  |
| event_study |  |  |  |  |  |  |  | 0.02725 | -0.01088 | -0.01139 | 0.6698 | 2.275 | 0.02508 |
| hierarchical_did |  |  |  |  |  |  |  | 0.02381 | 0.06946 | -0.01036 | -0.0187 | 2.188 |  |
| ipw | 0.001445 | -0.002562 | 0.0005066 | 0.005298 | -0.0003317 | -0.003111 |  |  |  |  |  |  |  |
| lin_regression | 0.001475 | -0.002593 | 0.000487 | 0.00557 | -0.0003191 | 0.006618 |  |  |  |  |  |  |  |
| twfe |  |  |  |  |  |  |  | 0.01446 | 0.004129 | 0.002324 | -1.06 | 1.327 | 0.03748 |

### Coverage of each estimator's primary interval, by scenario

| method | interval | A | B | C | D | F | G | H | P1 | P2 | P3 | P4 | P5 | P6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aipw | analytic | 0.95 | 0.94 | 0.95 | 0.95 | 0.96 | 0.95 |  |  |  |  |  |  |  |
| callaway_santanna | bootstrap_se |  |  |  |  |  |  |  | 0.89 | 0.91 | 0.85 | 0.91 | 0 | 0.89 |
| causal_forest | analytic | 0.95 | 0.93 | 0.95 | 0.95 | 0.96 | 0.95 |  |  |  |  |  |  |  |
| did_2x2 | analytic |  |  |  |  |  |  |  | 0.95 | 0.92 |  |  |  | 0.92 |
| difference_in_means | analytic | 0.95 | 0.94 | 0.95 | 0.06 | 0.97 | 0.02 |  |  |  |  |  |  |  |
| event_study | analytic |  |  |  |  |  |  |  | 0.96 | 0.97 | 0.93 | 0.03 | 0 | 0.99 |
| hierarchical_did | posterior |  |  |  |  |  |  |  | 0.97 | 0.93 | 0.92 | 0.92 | 0 |  |
| ipw | analytic | 0.96 | 0.95 | 0.96 | 0.96 | 0.96 | 0.96 |  |  |  |  |  |  |  |
| lin_regression | analytic | 0.95 | 0.94 | 0.96 | 0.95 | 0.94 | 0.93 |  |  |  |  |  |  |  |
| twfe | analytic |  |  |  |  |  |  |  | 0.97 | 0.94 | 0.94 | 0 | 0 | 0.93 |

Each coverage value is how often the interval an estimator reports as its primary one contains the true effect, and the interval column names that interval's kind: analytic for standard-error intervals, posterior for the credible interval hierarchical_did reports, and bootstrap_se for the Wald interval callaway_santanna builds from a cluster-bootstrap standard error.

*Analytic and bootstrap interval coverage were also compared across the two cross-sectional sample sizes the grid varies; that comparison is the coverage__scenario_by_size table in the study's own output and is not reproduced here. It was not pre-registered as an expectation, and no direction is claimed.*

## Limitations

- synthetic_control: the donor-pool arm has no grid or metrics yet (a causal-core item still open)
- bsts: the donor-pool arm has no grid or metrics yet (a causal-core item still open)
- CATE metrics: the heterogeneity metrics (CATE correlation, RMSE, calibration slope) have no arm yet
- scenario P7: it needs n_periods >= 16 and cannot share a panel axis with P1 to P6
- hierarchical_did is excluded from scenario P6 (binary outcomes); its Gaussian likelihood cannot represent them (docs/specs/2026-09-21-benchmark-study-design.md §13.7/§13.10)