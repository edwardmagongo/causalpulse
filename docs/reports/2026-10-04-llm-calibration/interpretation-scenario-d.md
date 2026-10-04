# Interpretation

## Summary

- Under the assumptions of difference_in_means, the estimated effect would be 0.1693 on the percentage_point scale, with an interval from 0.07461 to 0.2589 that excludes zero.
- Under the assumptions of ipw, which adjusts for covariates, the estimate would be 0.06601, with an interval from -0.03947 to 0.1715 that includes zero.
- The comparison across methods is labelled consistent, but the data are observational, so neither result should be read as settled.

## Estimates

- difference_in_means targets the ATE and reports a point of 0.1693 with standard error 0.04736 and a 95% interval from 0.07461 to 0.2589, with p-value < 0.001.
- The direction for difference_in_means is labelled positive, based on 218 treated and 182 control units.
- ipw targets the ATE and reports a point of 0.06601 with standard error 0.05382 and a 95% interval from -0.03947 to 0.1715, with p-value 0.22.
- The direction for ipw is labelled positive, based on 218 treated and 182 control units.

## Validity

- difference_in_means is scored in the MODERATE band with a ceiling of MODERATE, which is set by the observational design.
- For difference_in_means, the randomised_assignment check is NOT_TESTABLE, which matters because this estimator would be unbiased only if treatment had been randomly assigned.
- ipw is scored in the WEAK band with a ceiling of MODERATE, the band sitting one step below the ceiling because of its positivity check with status WARN.
- For ipw, the covariate_balance check is PASS with a statistic of 0.09662 against a threshold of 0.1, while unconfoundedness is NOT_TESTABLE.

## Agreement across methods

- Across 2 methods, the comparison is labelled consistent, with point estimates ranging from 0.06601 to 0.1693.
- The descriptive spread across methods is 0.07305, which is not a formal test of heterogeneity, and overlapping intervals would be consistent with a common effect only under each method's own assumptions.

## Caveats

- For ipw, the positivity check has status WARN, as some units appear to lie outside the region where both arms are represented, so their contribution may rest on extrapolation and trimming to the common support could help.
- Because unconfoundedness is NOT_TESTABLE for ipw, unmeasured confounding could bias its estimate, and a sensitivity analysis would be advisable.
- Because the data are observational and randomised_assignment is NOT_TESTABLE for difference_in_means, its estimate could be biased if assignment was not random.
- The router judged aipw as eligible, and likewise lin_regression with status eligible and causal_forest with status eligible, yet no results for them appear in this analysis.
- Panel-data methods such as did_2x2 and synthetic_control were marked not eligible because the dataset is a cross-section.

## Limits

This band scores observable diagnostics, not truth. It is not a probability that the effect is causal, and it cannot establish an assumption the data are unable to test.
