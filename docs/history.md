# How CausalPulse was built

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
estimate and the scenario's measured ATT are the same estimand (see [Estimators](estimators.md)), and two-way fixed
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
100 seconds; the reviews that shipped it found six defects, four of them in the first draft
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
built, and all four are now closed: the missing grid axes (Phase 18, below), the donor-pool
benchmark arm (below), `SyntheticControl`'s CI half-width and `donor_contamination_check`'s
threshold recalibrated against the convex-hull-aware placebo spread `falsify` uses,
and the CATE arm of the benchmark grid. The CATE arm is a fourth cell shape over
the RICH, NULL and NOISY scenarios, scoring `causal_forest`'s per-unit CATE by correlation, RMSE and
calibration slope in the `full` and `study` tiers. `bench run --tier cate` runs that arm alone at
the study's 50 replications (150 cells, about 35 minutes), and `bench study-report --cate-run DIR`
adds its results to the research report as their own section.
Scenario P7 (many treatment cohorts) runs in the `full` and `study` tiers as a fixed-protocol arm at its
own 600-unit, 20-period protocol. `bench run --tier p7` runs it alone at 50 replications (about 2 hours with
4 workers), and `bench study-report --p7-run DIR` adds a "Scenario P7" section to the report.
`bench study-report --bundle PATH` also writes the study bundle, the same evidence as one JSON file for
the dashboard's benchmark explorer (see [Web dashboard](dashboard.md)); it needs `--format markdown`.

**Phase 18** closed the first of those four items: the missing benchmark-grid axes.
`PropensityAssignment` gained an intercept and `LogisticBaseline` an additive noise scale, both
defaulting to today's behaviour exactly; `ScenarioParams` now carries all five cross-sectional
axes (the pre-existing `confounding_strength` plus treatment imbalance, noise level, effect size,
and heterogeneity), and a scenario declares which axes it actually responds to rather than being
crossed against every one blindly. `full_config`'s cross-sectional grid widens from 26 to 68 cells
to cover three measured points per new axis — a low, a baseline, and a high value, chosen by
generating real data rather than by estimate. Phase 18 itself shipped that as 42 to 84, holding
`confounding_strength` exempt from the declaration rule as a named, tracked exemption; a follow-up
closed it, so the four randomly-assigning scenarios that never read that axis contribute one
baseline strength each instead of three, and both ends of the range fall by 16.

**Benchmark study, part 1** built what a study-grade run needs and the analysis that reads it.
`bench run --tier study` runs `full`'s grid at 50 replications into a directory of one Parquet shard
per cell plus a ledger, so an interrupted run resumes (`--resume`) and a finished one extends
(`--resume --replications N`); a parallel executor fits cells through the same `fit_cell` the serial
`run_grid` does. `bench study` turns a finished run into the headline tables, pooled over each
scenario's baseline slice with Monte Carlo standard errors, and scores eight expectations that were
written down before any result existed. That study run does not cover `synthetic_control`, `bsts`,
the CATE metrics or scenario P7 (the donor-pool arm below adds the first two to the tier definition,
not to the run already in progress). Results are not committed (`benchmarks/runs/` is ignored).

**The 4,000-cell run completed 2026-09-23** (ten estimators, thirteen scenarios). Seven of eight
pre-registered expectations were met. The eighth -- `hierarchical_did`'s pooled panel coverage not
below 0.90 -- failed at 0.625, and reading it per scenario rather than pooled found two different
things underneath one number: on P5 ("parallel trends violated"), every DiD-family estimator
collapses identically, exactly as that scenario is documented to do; on P6 ("binary panel"),
`hierarchical_did` alone collapses (0.01 coverage) while its peers stay near nominal (0.89-0.93) on
the same data -- traced to its NumPyro model's outcome likelihood being unconditionally Gaussian,
with nothing in its declared requirements restricting it away from binary outcomes. Recorded as a
finding; the fix is left for its own follow-up. The other seven expectations, including the
false-positive rate under the null scenario and confounding bias on the scenarios built to show it,
measured as expected.

**The donor-pool benchmark arm** closed the second of the four open causal-core items. The grid gained
a third cell shape, `DonorPoolCell`, which is just a scenario and a replication index: `DONOR_SCENARIOS`
(S0-S5) already fixes each scenario's size, so there is no size axis to cross. Ground truth reuses the
panel branch's `true_att_overall`, which for one treated unit is that unit's own post-period ATT.
`full_config()` and so `study_config()` now carry all six donor scenarios (800 to 860 cells at `full`,
4,000 to 4,300 at `study`); `ci_config()` deliberately does not, because `synthetic_control` has no
committed tolerance bands and the arm added about 81 seconds (338s against 257s) to the gating tier
for no assertion. On a donor cell the router makes three estimators eligible, not the two the arm was
designed around: `synthetic_control`, `bsts`, and `hierarchical_did`, whose own requirements allow a
single treated unit and one pre-period. On S5 (two pre-periods) only `hierarchical_did` is eligible.
The pre-registered E5 and E7 expectations were therefore scoped to panel and non-donor rows so their
populations did not change, and the rendered expectation says so when donor rows were excluded.
The whole-branch review found what six task-scoped reviews could not: the sharded store's run
identity never recorded the donor arm, so a `study` run would have reported "4000/4000 done" with
`synthetic_control` and `bsts` appearing only as ineligible rows and no mention in the coverage
document; `longest_first` raised on a donor cell; and the `hierarchical_did` eligibility above. Every
wall-clock figure in the tier docstrings was measured while another job ran on the machine and should
be re-measured on an idle one.

No CLI command currently produces an evidence report, either: the report generator built in Phase 17
takes live result objects, the same as the figures do, because the arrays six of the eight figures
need were never persisted to disk. A user who runs `causalpulse analyze` cannot turn its output into
a report without writing Python.
