# Interpretation

## Summary

- This observational analysis reports 2 estimates of the average effect, and the comparison across methods labels them consistent.
- Under its own assumptions, difference_in_means suggests an effect of -8333, with an interval from -9484 to -7182 that excludes zero, whereas lin_regression suggests -3597, with an interval from -9360 to 2167 that includes zero.
- Both methods receive validity bands of MODERATE for difference_in_means and MODERATE for lin_regression, so the evidence may be read as suggestive rather than settled.

## Estimates

- difference_in_means estimates the ATE on the absolute scale at -8333, with a 95% interval from -9484 to -7182 and a direction labelled negative, which would reflect a treatment effect only if assignment were as good as random.
- For difference_in_means the interval excludes zero, with a p-value of < 0.001 from 185 treated and 16252 control units.
- lin_regression estimates the ATE on the absolute scale at -3597, with a 95% interval from -9360 to 2167 and a direction labelled negative, under the assumption that covariate adjustment plus as-good-as-random assignment removes confounding.
- For lin_regression the interval includes zero, with a p-value of 0.221 from 185 treated and 16252 control units.

## Validity

- difference_in_means receives a validity band of MODERATE against a ceiling of MODERATE, the ceiling being set by the observational design.
- The randomised_assignment check for difference_in_means has status NOT_TESTABLE, so the key identifying assumption for this method rests on how the data were collected rather than on the data themselves.
- lin_regression receives a validity band of MODERATE against a ceiling of MODERATE, again limited by the observational design.
- For lin_regression, the randomised_assignment check has status NOT_TESTABLE and the outcome_model_specification check has status NOT_TESTABLE.

## Agreement across methods

- Across 2 methods the comparison is labelled consistent, meaning the intervals overlap, which is consistent with a common effect under each method's assumptions.
- Point estimates in the comparison range from -8333 to -3597, with a descriptive dispersion of 3349 that is not a formal test of heterogeneity.
- Overlapping intervals do not imply identical conclusions, since only the difference_in_means interval excludes zero while the lin_regression interval includes zero.

## Caveats

- The count of estimator fits that raised an error is 0, but the identifying assumption checks for difference_in_means and lin_regression have statuses NOT_TESTABLE and NOT_TESTABLE, so neither estimate can be read as a treatment effect unless assignment was effectively random.
- The unadjusted difference_in_means estimate may carry confounding bias in observational data, and the gap between it and the covariate-adjusted lin_regression estimate may hint at such confounding.
- aipw, ipw and causal_forest carry router labels of eligible, eligible and eligible, yet no estimates from them appear here, even though they could adjust for confounding in observational data.
- The router marked did_2x2, twfe, event_study, callaway_santanna and hierarchical_did as not eligible, not eligible, not eligible, not eligible and not eligible because each requires panel data; dataset is a cross-section.
- The router also marked synthetic_control and bsts as not eligible and not eligible, since each requires panel data; dataset is a cross-section and each requires at most 1 treated unit(s); dataset has 185.

## Limits

This band scores observable diagnostics, not truth. It is not a probability that the effect is causal, and it cannot establish an assumption the data are unable to test.
