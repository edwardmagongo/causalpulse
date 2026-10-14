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
built, and all four are now closed: the missing grid axes (Phase 18, below), the donor-pool
benchmark arm (below), `SyntheticControl`'s CI half-width and `donor_contamination_check`'s
threshold recalibrated against the convex-hull-aware placebo spread `falsify` uses
(`docs/specs/2026-09-28-synthetic-control-ci-calibration.md`), and the CATE arm of the benchmark
grid (`docs/specs/2026-09-27-cate-metrics-arm-design.md`). The CATE arm is a fourth cell shape over
the RICH, NULL and NOISY scenarios, scoring `causal_forest`'s per-unit CATE by correlation, RMSE and
calibration slope in the `full` and `study` tiers. `bench run --tier cate` runs that arm alone at
the study's 50 replications (150 cells, about 35 minutes), and `bench study-report --cate-run DIR`
adds its results to the research report as their own section.
Scenario P7 (many treatment cohorts) runs in the `full` and `study` tiers as a fixed-protocol arm at its
own 600-unit, 20-period protocol. `bench run --tier p7` runs it alone at 50 replications (about 2 hours with
4 workers), and `bench study-report --p7-run DIR` adds a "Scenario P7" section to the report.
`bench study-report --bundle PATH` also writes the study bundle, the same evidence as one JSON file for
the dashboard's benchmark explorer (see "Web dashboard" below); it needs `--format markdown`.

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
baseline strength each instead of three, and both ends of the range fall by 16. See
`docs/specs/2026-09-13-causal-core-phase18-design.md` section 12 for the measured values, the
values considered and rejected, and the defects a whole-branch review found before any of it
shipped.

**Benchmark study, part 1** built what a study-grade run needs and the analysis that reads it. `bench
run --tier study` runs `full`'s grid at 50 replications into a directory of one Parquet shard per cell
plus a ledger, so an interrupted run resumes (`--resume`) and a finished one extends (`--resume
--replications N`); a parallel executor fits cells through the same `fit_cell` the serial `run_grid`
does. `bench study` turns a finished run into the headline tables, pooled over each scenario's
baseline slice with Monte Carlo standard errors, and scores eight expectations that were written into
`docs/specs/2026-09-21-benchmark-study-design.md` before any result existed. That study run does not
cover `synthetic_control`, `bsts`, the CATE metrics or scenario P7 (the donor-pool arm below adds the
first two to the tier definition, not to the run already in progress). Results are not committed
(`benchmarks/runs/` is ignored).

**The 4,000-cell run completed 2026-09-23** (ten estimators, thirteen scenarios). Seven of eight
pre-registered expectations were met. The eighth -- `hierarchical_did`'s pooled panel coverage not
below 0.90 -- failed at 0.625, and reading it per scenario rather than pooled found two different
things underneath one number: on P5 ("parallel trends violated"), every DiD-family estimator collapses
identically, exactly as that scenario is documented to do; on P6 ("binary panel"), `hierarchical_did`
alone collapses (0.01 coverage) while its peers stay near nominal (0.89-0.93) on the same data --
traced to its NumPyro model's outcome likelihood being unconditionally Gaussian, with nothing in its
declared requirements restricting it away from binary outcomes. Recorded as a finding rather than
fixed here; see `docs/specs/2026-09-21-benchmark-study-design.md` section 13.7 for the full diagnosis
and why a fix was left for its own follow-up rather than done inside this analysis. The other seven
expectations, including the false-positive rate under the null scenario and confounding bias on the
scenarios built to show it, measured as expected.

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
be re-measured on an idle one. See `docs/specs/2026-09-23-donor-pool-benchmark-arm-design.md`.

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
| `cp-evaluation` | The benchmark grid: `run_grid` fits estimators across the DGP axes that exist and `summarise` reports bias, coverage, and calibration with Monte Carlo standard error. The first package allowed to import both `cp-simulation` and `cp-causal` together, so it sits above `cp-causal` in the layers contract and nothing below it may import it in turn. `store` and `execute` run a benchmark as a resumable set of shards. |
| `cp-figures` | Eight publication-quality matplotlib figures — counterfactual path, event study, love plot, propensity overlap, specification curve, placebo distribution, cross-method forest plot, benchmark heatmaps — each a pure data function plus a thin drawing function. Imports only `cp-core` and `cp-causal`. |
| `cp-report` | The evidence report generator: assumption table, validity band, cross-method comparison, and stated limitations, rendered to Markdown or one self-contained HTML file with figures inlined as SVG. Refuses to emit prose that fails `causal_language_lint`. `bundle` renders the benchmark study's evidence a third time, as one deterministic JSON document for the dashboard, recording the SHA-256 of the Markdown report written beside it. |
| `cp-llm` | Plain-language interpretation of a whole analysis. `extract` turns an `AnalysisOutcome` into a table of citable facts; a model writes a narrative in which every value is a placeholder naming a fact; `verify` refuses any narrative it cannot trace back to the table. Sits between `cp-report` and the command line in the layers contract. Only `cp_llm.extract` imports the analysis outcome types from `cp-report` and `cp-causal`, and only `cp_llm.client` imports the model SDK. |
| `cp-llm-eval` | Measures how often a verified interpretation misreads its analysis: a corpus of real fits, interpretations run within a call budget and stored by key, deterministic graders and a judge validated against hand labels, and a report with case-clustered uncertainty. Only `cp-cli` imports it; the api and worker never do. |
| `cp-cli` | The `causalpulse` command-line entry point (`simulate`, `analyze`, `interpret`, `replay`, `bench run`, `bench report`, `bench status`, `bench compact`, `bench study`, `bench track`) and Parquet-plus-JSON-sidecar result persistence. Sits at the top of the layers contract; nothing imports it. |
| `cp-api` | The FastAPI service: upload a CSV, request an analysis, read what the worker wrote. Not in the layers contract — its own forbidden-imports contract keeps it from importing `cp_causal` directly, so router refusals happen in the worker, not here. |
| `cp-worker` | The RQ worker that runs a queued analysis through `cp_report.pipeline`: fit, score validity, render each estimate's report. Sits alongside `cp-cli` at the top of the layers contract; may import `cp_api.db` and nothing else from `cp_api`. |

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

### What is refused before anything is fitted

An upload is read, checked and analysed in different places, and a file that one of them reads
differently from another is analysed as something its owner did not send. So there is one reader,
and the dataset built from what it reads refuses, by name, whatever the estimators cannot use.

**The reader** (`cp_core.reading.read_csv`) is the one function that turns the bytes of a dataset
someone supplies into a frame. The API's check of an upload, the worker, `causalpulse analyze
--data` and the replay of a manifest all hand it the file's bytes and pass pandas nothing of their
own. (The evaluation corpus reads this repository's own two fixtures with pandas directly; a test
holds those two files to read the same either way.) It differs from pandas' defaults in three
ways.

- **A row with more fields than the header names is refused, naming the line,** whether or not
  the extra field holds anything. pandas reads such a file in more than one way; when the first
  data row is the wide one it takes as many leading columns as there are extra fields for an
  index and moves the rest along. A file whose every line ends in a comma is refused too: it
  reads the same as a stray comma inside a value before an empty last column (`3,5,` is `3` then
  `5`, or the decimal `3,5` then nothing), and nothing in the file says which was meant. The line
  is pandas' count, in which a quoted value that runs over several lines is one line.
- **A large file is read whole,** not in pieces whose types are settled separately, which left a
  column that was numbers in one piece and text in another holding both.
- **The bytes are read as they are, line endings aside.** Given a path pandas decides from the file's name whether to
  decompress it, so the same bytes could be read two ways under two names. A compressed file is
  therefore not read by anything here, `causalpulse analyze --data` included, which used to
  decompress one: decompress it first. The one change made to the bytes is that every carriage
  return, alone or before a line feed, is made a line feed, so one inside a quoted value changes
  too. pandas' tokeniser misreads lines ended by a carriage return alone (old Mac line endings):
  on the 7-byte `a,b\r\r 1` it allocated memory without limit (a probe of it took a 16 GB
  machine down), and on others it read the header again as a data row, so every column was text.

The two real datasets in this repository are read exactly as before.

**`CausalDataset`** then refuses, when the dataset is built, each with one message naming the
column:

- **A column the schema names appearing twice** in the frame.
- **A unit column with a missing value, or one mixing numbers and text.** pandas reads `NA`,
  `N/A`, `n/a`, `NaN`, `null`, `NULL`, `None` and an empty cell as missing, so a unit with one of
  those names (a country coded `NA`) has to be renamed; the message says so.
- **A column that must be numeric and was not read as numbers.** The treatment, the outcome, the
  covariates, the cohort, the negative-control outcome and the auxiliary columns must have a
  numeric dtype. The unit may be text.
- **An infinite value**, `inf` or `-inf`, in any of those columns or in a numeric unit or time
  column. Text such as `1e400`, which pandas reads as infinite, is refused the same way.
- **A value larger than 1e150 in magnitude** in any of those columns except the unit, the treatment
  and the time column, which have their own rules. A variance adds n squares of the data, and at
  1e150 a square is 1e300, so up to about 1e8 rows stay finite in float64; above it the estimators
  failed with a library's own message naming no column (with one value at 1e155 in LaLonde's
  outcome all four cross-sectional estimators failed; in a covariate `aipw` did).
- **A numeric period that is not a whole number, or is larger than 2**53.** Periods are used as
  integers. A time column may hold dates and the dataset is built, but a panel estimator needs
  numbered periods and says so.

**A missing value** in the outcome or a covariate is different, on purpose. A dataset with gaps
has to be constructible, because `complete_cases` and imputation take one. It is refused when it
is analysed: once, before anything is fitted, for an analysis that goes through the pipeline (an
upload, `causalpulse analyze --report`), and by each estimator for itself when one is fitted
directly.

The analysis of an upload, `cp analyze` and `cp replay` all go through these. A file the reader
refuses is refused when it is uploaded; a dataset `CausalDataset` refuses is accepted as an upload,
and its analysis fails with the message. Nothing is cleaned on the user's behalf.

Why each exists, measured on the real datasets before it did. `read_csv` and Python's `float()`
disagree about what a number is: one covariate cell of `1_000` made pandas read the column as
text, and it was analysed as the number 1000 with no warning; a cell of `inf ` (a trailing space)
went past the infinity check, which looks only at numeric columns. An infinite covariate gave a
partial analysis, an infinite outcome failed every estimator with a library's own message, and an
infinite period in Prop 99 was **fitted**. LaLonde with a comma ending every line was read with
every column shifted by one. Prop 99 with a state renamed `NA` crashed while it was being routed.
Prop 99 with each year moved by half a period gave a cohort that is not in the data.

The answers of least squares, `lin_regression`, `aipw` and the causal forest do not depend on the
units a column is in (2026-10-14,
`docs/specs/2026-10-14-scale-invariance-and-magnitude-bound-design.md`). Before, `lin_regression`
on LaLonde with 1975 earnings (re75) in cents instead of dollars gave an interval of [-4,480, -2,713] for
[-9,360, 2,167]: the shared least squares cut small directions once one column dwarfed the others,
and the `aipw` and causal forest outcome models did the same. A value that is large but under the
bound is the user's own data: one LaLonde covariate cell of 1e39 still moves `aipw`'s estimate to
about -9.5e35, because its outcome model extrapolates to that unit, and nothing here flags an
outlier. An integer above 2**53 in an outcome or covariate is rounded when it is converted, by
about one part in 1e16 (`docs/specs/2026-10-06-input-followups-design.md`, section 10).

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

## Tracking benchmark runs

`causalpulse bench track <run-dir>` logs a finished benchmark run to MLflow, so runs can be
compared across commits and grids. MLflow comes from the optional `cp-tracking` package
(`cp-cli[tracking]`), which the dev group installs; the api/worker image leaves it out:

```bash
uv run causalpulse bench track benchmarks/runs/study-60
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Each benchmark run becomes one MLflow run in the `causalpulse-benchmarks` experiment:

- **Params:** the run's identity (tier, estimators, fit config, grid axes, root seed,
  replications).
- **Tags:** `cp.git_commit`, one `cp.package.<name>` per package version, `cp.run_dir` and
  `cp.fingerprint`.
- **Metrics:** for each scenario x estimator on the headline slice, bias, RMSE, analytic and
  bootstrap coverage, and SE calibration, each with its Monte Carlo standard error as
  `<scenario>.<method>.<metric>_mcse`.
- **CATE metrics:** for a run with scored CATE rows, `cate_correlation`, `cate_rmse` and `cate_calibration_slope` per scenario and method, each with its `_mcse`, pooled exactly as the benchmark report's CATE section pools them. A value with no finite MCSE (NULL's correlation and slope) is not logged.
- **Artifacts:** the headline tables as CSV (most at both the scenario and the
  scenario-by-size level; `false_positive_rate__scenario` has the scenario level only), plus
  `identity.json`, `sidecar.json` and, for a run with scored CATE rows, `cate__scenario.csv`.

Only the headline slice and the per-scenario CATE metrics are logged as metrics, because
logging every grid cell would make MLflow's comparison view unusable.

The fingerprint covers the run's identity, its row count and the git commit. Tracking the same
finished run again at the same commit is refused; `--force` logs it again. The same grid run at a
new commit is tracked as a new run, which is what makes cross-commit comparison possible. A store
error prints one line and exits 1. The store defaults to
`./mlruns`, relative to the current directory (gitignored); set `--tracking-uri` or
`MLFLOW_TRACKING_URI` to use another. Only sharded runs can be tracked.

API analyses are deliberately not tracked. Their manifest and the Postgres rows beside it already
record everything MLflow would, scoped to their owner, and a second store would have no
ownership at all.

## Backend (sub-project 3)

A local demo stack: upload a CSV, declare its schema, and get back estimates, validity bands, and
evidence reports over HTTP. One machine, `docker compose up`; every route except `/v1/healthz` needs a
per-user API key; it runs on one machine, not as a hosted service.

```bash
docker compose up --build
```

This starts six services: `postgres`, `redis`, a one-shot `migrate` that runs the Alembic
migrations and exits, `api`, which waits on `migrate` and serves on `http://localhost:8000`, a
`worker`, which reads jobs off the same Redis queue the API enqueues to, and `web`, the dashboard,
on `http://localhost:3000` (see "Web dashboard" below). Ports (host side) default to `8000` for the
API, `5433` for Postgres, `6380` for Redis, and `3000` for `web`; override with `CP_API_PORT`,
`CP_POSTGRES_PORT`, `CP_REDIS_PORT`, `CP_WEB_PORT` if those clash with something else already
running.

Each analysis has a 30-minute limit; one that runs past it is stopped and marked `failed`, with
an error stating the limit. The limit takes effect between Python-level steps, so a single hung
call into compiled code (XLA, LAPACK) is not interrupted, and the restart policy below does not
help there either, because the worker process is still alive. `api` and `worker` restart unless
stopped. That matters for a worker
that dies mid-job: RQ only marks a dead worker's job failed when a live worker cleans up its
registries, so a killed worker's analysis shows `running` until a worker restarts and the job's
time limit has passed, and only then becomes "worker lost".

### Routes

All routes are under `/v1`. Every route except `GET /healthz` needs an `Authorization: Bearer <key>` header.

| Route | Behaviour |
|---|---|
| `POST /datasets` | Multipart CSV, ≤ 50 MB (413 above). The key is checked before any of the body is read (see "Users and API keys"). Your own copy of the same bytes, uploaded twice, returns the existing row by sha256; another user's identical upload is a separate row. |
| `POST /analyses` | `{dataset_id, schema, estimators?, random_seed}`. Validates the schema via `CausalSchema` and its columns against the CSV header; enqueues; 202 with the id. |
| `GET /me` | `{name}`: whose key this is. |
| `GET /estimators` | The estimator catalogue: each name with its family (`cross_sectional`, `panel`, `donor_pool`). A family is a grouping, not a statement about eligibility. |
| `GET /datasets` | Your datasets, newest first, without their columns. `limit` (1 to 200, default 50) and `cursor`, the previous page's `next_cursor`. |
| `GET /datasets/{id}` | One dataset with its columns. |
| `GET /analyses` | Your analyses, newest first, paged like the datasets. An in-flight analysis whose job is gone is marked `failed` here, as on the single read. |
| `GET /analyses/{id}` | Status, `partial`, error, the dataset's id and `dataset_filename`, routing decisions, and per estimate its method, band, error, and `summary` (point, standard error, intervals, the bootstrap interval's level, p-value, group sizes). It also carries `interpretation` (null until one is asked for), `interpretable` and `interpretable_reason` (why it cannot be, when it cannot). |
| `POST /analyses/{id}/interpretation` | Asks for a model-written interpretation of a finished analysis. 202 with the interpretation, queued. 403 if the server has interpretations off; 409 if the analysis has not finished or has no fact table, if an interpretation is already being written, or if one is already verified or refused (only a failed one can be asked for again); 429 over the daily limit. |
| `GET /analyses/{id}/estimates/{method}/report.html` | One estimate's evidence report, figures inlined. |
| `GET /analyses/{id}/estimates/{method}/report.md` | The Markdown report. Its relative figure links resolve against the next route. |
| `GET /analyses/{id}/estimates/{method}/{filename}` | One Markdown figure, from `report_assets`. |
| `GET /analyses/{id}/manifest` | The replayable `AnalysisManifest`. |
| `GET /analyses/{id}/summary` | `{"comparisons": ...}`: the comparison groups and their verdicts. |
| `GET /healthz` | Postgres and Redis reachable. |

### Users and API keys

Every route except `/v1/healthz` needs a key, and every dataset and analysis belongs to the user
whose key created it. Another user's resource answers 404, exactly like one that does not exist.
A request without a valid key answers 401; the one exception is a request whose body is malformed
JSON, which answers 422 even with no key, because FastAPI decodes the JSON before it checks the key.

The upload is the other way round. `POST /v1/datasets` checks the key, and then the day's upload
limit when one is set ("Production stack" below), before it reads any of the body. A missing,
malformed or revoked key is answered 401, and a user over the limit 429, with the body unsent:
`tests/services/test_upload_order.py` sends the headers of a 12 MB upload and none of its body,
over a real socket, and gets the answer. Until this was changed the route read the whole
multipart body, up to 50 MB, before it looked at the key.

The route also counts the request body as it arrives, which it did not do before: a file part of
any length used to be written to a temporary file in full before the 413. The bound is on the
whole body, the 50 MiB limit plus 1 MiB for the form's own framing, 53,477,376 bytes. Past it
nothing more is kept and the answer is 413 `the upload exceeds the 52428800-byte limit`. The rest
of the body is read and thrown away for at most a further 50 MiB, so that a client sending a file
a little too large gets the answer after its request and not in the middle of it; past that the
API answers anyway. Because the bound is on the body and not on the file, two requests are
answered differently than before:

| Request | Before | Now |
|---|---|---|
| A file within the limit beside other form fields, each field within the parser's own 1 MB, the whole body over the bound | 201, the file stored | 413 `the upload exceeds the 52428800-byte limit` |
| A body that is not multipart (URL-encoded), with no `file` field, over the bound | 422, FastAPI's `Field required` for `file` | 413, the same message |

The message names the file's limit although the file itself may be small.
`test_a_body_over_the_bound_is_413_whatever_it_holds` pins both rows. A single field over 1 MB is
still Starlette's own 400, as before.

```bash
docker compose exec api cp-api-admin create-user alice
export CP_KEY=$(docker compose exec -T api cp-api-admin issue-key alice)
```

The key is printed once and only its hash is stored. Revoke it with
`docker compose exec api cp-api-admin revoke-key <key id>`, where the key id is the 12 characters
after `cpk_`. A database from before this change keeps its rows, which now belong to a user
named `legacy`. Issue a key for `legacy` to reach them.

### A walkthrough with the LaLonde fixture

```bash
curl -H "Authorization: Bearer $CP_KEY" -F file=@tests/fixtures/real_data/lalonde.csv http://localhost:8000/v1/datasets
# => {"id": "...", "filename": "lalonde.csv", "sha256": "...", "n_bytes": ..., "columns": [...]}

curl -X POST http://localhost:8000/v1/analyses \
  -H "Authorization: Bearer $CP_KEY" \
  -H 'content-type: application/json' \
  -d '{
        "dataset_id": "<id from above>",
        "schema": {
          "unit": "unit_id",
          "treatment": "treat",
          "outcome": "re78",
          "covariates": ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"],
          "outcome_type": "continuous"
        },
        "estimators": ["difference_in_means", "lin_regression"]
      }'
# => {"id": "<analysis id>", "status": "queued"}

curl -H "Authorization: Bearer $CP_KEY" http://localhost:8000/v1/analyses/<analysis id>
# poll until "status" is "succeeded" or "failed"

curl -H "Authorization: Bearer $CP_KEY" http://localhost:8000/v1/analyses/<analysis id>/estimates/lin_regression/report.html
```

### One report per estimate

Each requested estimator gets its own evidence report and its own validity band, never a report
merged across estimators. Validity is scored per estimate — `score_validity` reads that one
estimate's `assumptions` — and mixing several estimates into a single report would put one
method's band next to another method's assumption table, the exact mismatch `Evidence`'s own
docstring warns against. The cross-method view lives beside the reports instead of inside them:
`GET /analyses/{id}/summary` returns the `compare()` verdicts, and `GET /analyses/{id}` the routing
decisions and every estimate's summary.

### Replay

`GET /analyses/{id}/manifest` returns the analysis's `AnalysisManifest`, which names the uploaded
dataset by filename and content hash rather than embedding it. Replay opens that filename relative
to the current directory, so check an analysis like this:

```bash
mkdir replay-check && cd replay-check
curl -s -H "Authorization: Bearer $CP_KEY" http://localhost:8000/v1/analyses/<analysis id>/manifest > manifest.json
# The filename the manifest records -- its "dataset" -> "path":
python3 -c 'import json; print(json.load(open("manifest.json"))["dataset"]["path"])'
cp /path/to/the/original.csv <that filename>
uv run --project /path/to/causalpulse causalpulse replay manifest.json
```

Read the filename from the manifest rather than assuming it: the same bytes uploaded again under
another name are deduplicated, per user, to that user's first upload, whose filename the manifest keeps. A copy
whose bytes differ at all is refused, since the manifest records their sha256.

`replay` prints each estimator's point estimate to ten significant figures. Compare those with
`estimates[].summary.point` from `GET /analyses/{id}`. The verdict `replay` prints on its own
does not make that comparison: it checks against an `estimates.csv` beside the manifest, which
only `causalpulse analyze` writes and the API does not serve, and without one it compares two
replays of the manifest with each other. The exact, bit-for-bit comparison of every point,
standard error, interval and p-value against the API's stored numbers is what
`tests/services/test_api.py`'s replay-parity test automates -- the check that the API is a thin
layer over the core rather than a second implementation of it.

### Testing the stack itself

```bash
make check-services  # postgres + redis only, project causalpulse-test, ports 5434/6381
make smoke           # api and worker built and run, project causalpulse-smoke, ports 5435/6382/8001
make smoke-web       # whole stack + browser, project causalpulse-web-smoke, 5436/6383/8002/3001
make smoke-prod      # production stack + browser over HTTPS, project causalpulse-prod-smoke, 8091/8444
```

Five separate compose projects exist so none can step on another: the demo stack above
(`causalpulse`, ports 8000/5433/6380/3000), `check-services` (`causalpulse-test`, 5434/6381, no
image build — only `postgres` and `redis`), `smoke` (`causalpulse-smoke`, 5435/6382/8001, `api` and
`worker` with what they depend on), `smoke-web` (`causalpulse-web-smoke`, 5436/6383/8002/3001,
the full stack with the dashboard), and `smoke-prod` (`causalpulse-prod-smoke`, the production
compose file, with only its proxy on the host, at 8091 and 8444; see "Production stack"). `make smoke` names `api` and `worker`, so it neither builds nor
starts `web`. The services tests read `TEST_DATABASE_URL` and `TEST_REDIS_URL`
exclusively — never the app's own `DATABASE_URL`/`REDIS_URL` — precisely so a developer's local
`DATABASE_URL` export can never point a test run at the demo stack's database by accident.

`make smoke` builds the image, brings the stack up, issues a real key inside the container with
`cp-api-admin`, checks that a request without a key gets 401, uploads the LaLonde fixture over real HTTP,
waits for the analysis to finish, and fetches its report — end to end, through the same routes a
user would call, not through an in-process test client.

## Interpreting an analysis (sub-project 4, phase 1)

`causalpulse interpret <run-dir>` writes a plain-language reading of a whole analysis: what each
method estimated, each validity band and the checks behind it, what the cross-method comparison
reports, and which fits failed. It may also say which estimators the router excluded; that is not
required of it. It reads the `facts.json` that `analyze --report`
writes, so it refits nothing and can be repeated. Every `analyze` run removes a `facts.json` and
any interpretation left in its output directory by an earlier run, because they describe that
earlier analysis. With `--report` the fact table is written last: if it cannot be built, the
command prints one line, exits 1, and the estimates and the manifest are already written.

```bash
uv run causalpulse analyze --data tests/fixtures/real_data/lalonde.csv \
  --unit unit_id --treatment treat --outcome re78 \
  --covariates age,educ,black,hisp,marr,nodegree,re74,re75 --outcome-type continuous \
  --report --estimators difference_in_means,lin_regression --out out/lalonde
uv run causalpulse interpret out/lalonde
```

It needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile) and makes one model call, or two
when the first reply is rejected. Measured on real output (2026-10-04), one interpretation costs
$0.07 to $0.20; the calibration record is in `docs/reports/2026-10-04-llm-calibration/`. It writes `interpretation.md`, the narrative, and
`interpretation.json`, the record: the narrative as the model wrote it, the rendered text, the
model id, the effort and the token limit it ran with, `prompt_version` (a short hash of the
system prompt and the reply schema), every attempt with its violations and token usage, and the
SHA-256 of the bytes of the `facts.json` it read.
`--model` and `--effort` override the defaults (`claude-opus-5-5`, `medium`).

### The model never writes a value

`analyze --report` flattens the analysis into a table of facts, each with an id:
`est.aipw.point`, `score.aipw.band`, `check.aipw.positivity.status`,
`cmp.ate.percentage_point.agreement`. The model is given that table and no other data from the
analysis, and writes statements in which every value is a placeholder:

```
{est.aipw.method} estimates the {est.aipw.estimand} at {est.aipw.point}, with a
{est.aipw.ci_level} interval from {est.aipw.ci_lower} to {est.aipw.ci_upper}.
```

The system substitutes the values: numbers to four significant figures, p-values to three, counts
exactly, levels as a percentage (`95%`). The model cannot write a digit, a percent sign or one of
a short list of number words, and a placeholder must stand apart from the text around it, with no
sign or operator before a number, so a figure is shown exactly as the analysis produced it. A
quantity spelled in words outside that list ("twenty units", "a third of the effect") is not caught. A value that was never
computed, such as the statistic of a check that cannot be tested, has no fact, so citing it fails instead of
rendering as a zero.

Before anything is substituted, a deterministic verifier applies ten rules, and the narrative
is written only if none is broken:

1. **Resolves.** Every placeholder names a fact in the table.
2. **No bare values.** No digit, no percent sign, and no number word from a short list, outside a
   placeholder.
3. **Controlled vocabulary.** A word that states a band, a status, an agreement verdict, a
   direction, a zero result, significance or eligibility appears only through a placeholder.
   "Strong evidence" is refused unless the word comes from a band fact; "significant" is refused
   outright, and the model has an `interval_excludes_zero` fact to cite. The lists hold exact word
   forms, so an unlisted form ("strongest", "failure") passes.
4. **Attribution.** A statement citing a fact about a method must also name that method.
5. **No bare point.** A statement citing a point estimate must cite both interval bounds.
6. **An INSUFFICIENT estimate carries its band.** A statement citing the point, an interval
   bound, the standard error or the p-value of an estimate whose validity band is `INSUFFICIENT`
   must also cite that band, so the word stands beside the number. A band cited in another
   statement does not count. Added after the evaluation's labels found two statements that
   quoted such an estimate with no caveat; a `WEAK` band needs none.
7. **Language.** The prose passes `causal_language_lint`, the lint the evidence report already
   passes. The lint reads each statement with its numbers masked, so that the full stop inside a
   decimal does not end a sentence.
8. **Delimiters.** Every placeholder stands apart from the text around it, and no sign or
   operator comes before a number. `-{point}` and `{n_treated}{n_control}` are refused: glued
   together they would render a figure the table does not hold.
9. **One line.** A statement holds no line break, control or invisible character, so it cannot
   write a heading of its own.
10. **Completeness.** The narrative must cite every estimate's point and band, every comparison's
   agreement (or its reason for not comparing), every fit failure, and every check whose status is
   `FAIL` or `WARN`.

A rejected reply is sent back to the model once, with its violations. If the second reply is
rejected too, the command prints the violations, writes nothing, and exits 1. The validity
score's limits statement is appended by the system and is not the model's to write.

A `facts.json` that is missing, damaged or from another schema version is reported in one line,
naming the file and what to rerun, and nothing is sent to the model.

The command also prints one line and exits 1, writing nothing, when no credentials are found (it
says how to supply them), when the login profile is broken or expired, when the API call fails,
and when the model refuses, or its reply is cut off at the token limit or does not fit the reply
schema. The command does not retry any of these;
the SDK itself retries rate limits and server errors. A failed run leaves an `interpretation.md`
or `interpretation.json` from an earlier run in place.

### What a verified interpretation means, and what it does not

It means every value shown is the system's own, every controlled word is backed by a fact,
nothing the completeness rule requires was left out, and the prose passes the language lint.

It does not mean the narrative is a correct reading of the analysis. The verifier does not catch:

- the words that name a quantity: "a p-value of `{est.ipw.se}`", or a lower and an upper bound
  swapped, verify. The value shown is the system's; what the sentence calls it is the model's;
- qualitative wording outside the controlled lists ("a sizeable effect");
- a wrong relationship between two correctly cited facts ("A is larger than B");
- words around a quoted check interpretation that frame it misleadingly;
- a quantity expressed without a number ("most checks");
- a number spelled in words beyond the short list ("twenty units");
- a value attributed to the wrong method inside a statement that names both;
- a statement of two sentences that names a method in one and misplaces its value in the other:
  the prompt asks for one sentence per statement, and that is not checked;
- emphasis: a failed check must be cited, not given weight;
- the negation of a cited fact ("the band is not {score.aipw.band}");
- a hedge or negation word that arrives inside a cited fact ("not eligible", or any check's
  interpretation), which the lint counts as hedging the whole sentence;
- the phrase "consistent with", which is allowed as free text because the lint itself asks for
  it, so "consistent with each other" passes whatever the comparison found, and "consistent with
  no effect" passes whatever the interval says.

How often these happen is not measured yet. That is the next phase of this sub-project.

### What leaves the machine

Of the analysis, the fact table and nothing else: aggregate statistics, method and check names,
and prose the system wrote. Beside it go the fixed instructions and, on a retry, the model's own
rejected reply with the violations. No dataset rows are sent. A fit failure's error text is the
exception's own message, which can name a column or quote a cell value (this codebase's fits
raise `could not convert string to float: '<the cell>'` for a non-numeric covariate). It is sent to
the model and, because the narrative must cite every fit failure, shown to the reader verbatim.

Text inside a fact is treated as untrusted, and the defence is partial. A fit error written to
read as an instruction reaches the model. Whatever the model then writes must still pass the
verifier, so it cannot bring in a figure or a controlled word. Wording that avoids the controlled
lists can still reach the narrative as the model's own free text.

### Testing it

`make check` never calls the model: every test but two runs against a scripted model or a mock of
the SDK's transport. `make check-llm` runs those two against the real API: this interpretation, and
`cp-llm-eval`'s judge (below).

### On the server and in the dashboard (phase 3)

A server can offer the same interpretation for an analysis run through the API. It is off unless
`CP_INTERPRETATION=on` is set for both the API and the worker. A worker with it off fails any
interpretation it is handed, without a model call, even if the API accepted the request. When it is
on, the worker also needs `ANTHROPIC_API_KEY`, and refuses to start without it. The API never reads
the key.

- **The fact table is stored when the analysis runs.** The worker writes it beside the outcome. This
  calls no model and costs nothing. If the extractor cannot build it, the analysis still succeeds,
  and it simply cannot be interpreted. Analyses run before this existed have no fact table and are
  not refitted to get one.
- **An interpretation is written only when someone asks.** `POST /v1/analyses/{id}/interpretation`
  answers 202 and queues a separate job, so an analysis never waits on the model.
  `CP_LIMIT_INTERPRETATIONS_PER_DAY` caps the requests per user, counted when accepted. The
  production stack defaults to 5.
- **A result is final.** A verified or refused interpretation cannot be asked for again: re-asking
  until something passes would cost money and lower the bar. Only a failed one can be retried,
  such as a model that could not be reached, a job that ran out of time, or a worker that was lost.
- **What the reader sees.** On the analysis page, a verified interpretation is shown under
  "Interpretation (model-written, verified against the facts)". Beside it are the system's limits
  text, a fixed note on what "verified" establishes and what it does not, and the model and prompt
  version. A refused one is not shown; the page says so and gives the number of attempts.
- **A verified interpretation can still misread its analysis.** Phase 2's study, below, measured
  it: in a hand-labelled sample, the commonest misreading quoted an estimate whose band is
  `INSUFFICIENT` without saying so. The verifier now refuses that (rule 6, below); misreadings
  outside its rules can still pass.

## Measuring misreadings (sub-project 4, phase 2)

A verified interpretation cannot cite a figure that is not in its fact table, but it can still
misread the analysis: call a result strong when the table says weak, say a check passed that
failed, or state a conclusion the facts do not support. `cp-llm-eval` measures how often that
happens. It builds a corpus of real fits, has the model interpret each fact table several times
within a call budget, grades every final statement with five deterministic graders (D1 to D5) and
with a judge model, and compares both against statements you label by hand. The report gives each
misreading rate with a standard error clustered by case. A rate of zero carries the rule-of-three
upper bound, 3 / units capped at 100%, instead. A unanimous rate carries the mirrored lower bound,
1 - 3 / units, which the report leaves out whenever it is 0 or less (three units or fewer). Any
other rate whose cases all have the same proportion carries no standard error, and the report says
there is no spread across cases; any other rate over a single case carries none either, and the
report says so (a 0% or 100% rate over one case shows its bound instead). D4, the uncited status
summary behind the primary endpoint, reads a status word only in the clause that holds its
quantifier ("no check", "every diagnostic"), so a status word after ", and" or ", but" does not
count: "every check examines one assumption, and a problem in one may recur" is not a summary.
The judge's sensitivity and specificity and each grader's precision are weighted by label stratum
(judge-flagged, grader-only, unflagged, sampled about a third each) back to all judged statements,
with Wilson intervals at the effective sample size n_eff, `(sum of weights)^2 / sum of squared
weights`: the number of equally weighted labels that would be as informative.

The commands, in order. Every `--max-calls` is required: a paid step does not run without a ceiling.
It counts model calls as the code makes them; the SDK may retry a failed request within one counted
call, so the requests the API bills can exceed it.

```bash
uv run causalpulse llm-eval build --out out/llm-eval/corpus
uv run causalpulse llm-eval run --corpus out/llm-eval/corpus --store out/llm-eval/store --max-calls 50
uv run causalpulse llm-eval grade --corpus out/llm-eval/corpus --store out/llm-eval/store --max-calls 50
uv run causalpulse llm-eval label export --corpus out/llm-eval/corpus --store out/llm-eval/store \
  --out out/llm-eval/to-label.csv
# fill in the verdict, category and note columns of the CSV, then:
uv run causalpulse llm-eval label import --corpus out/llm-eval/corpus --store out/llm-eval/store \
  --csv out/llm-eval/to-label.csv
uv run causalpulse llm-eval report --corpus out/llm-eval/corpus --store out/llm-eval/store \
  --out out/llm-eval/report
uv run causalpulse llm-eval restore --bundle out/llm-eval/report --store out/llm-eval/store-restored
```

- `build` fits each corpus case and writes its fact table. `--case` builds only the cases named. A
  case whose fit lacks the property it was included for is refused.
- `run` writes one record per fact table, prompt and replicate into the store, keyed so that a
  repeated run skips what is already stored. `--variant`, `--replicates` (default 4), `--case` and
  `--workers` select the work. It exits 1 if the budget stopped it before every job ran, or if a
  call failed in transport.
- `grade` sends each verified interpretation to the judge (a refused one has no final narrative) and exits 1 under the same two conditions.
- `label export` writes a seeded sample of final statements (`--size`, default 150; `--seed`) to
  a CSV without the judge's verdict or any grader's flag, so that the labelling is blind. It also
  writes `sample.json` into the store. `label import` reads that file and refuses a CSV that does
  not label exactly the exported sample, or that uses a value outside the allowed verdicts and
  categories. Export only after `grade` has finished, and do not run `grade` again afterwards: the
  weights assume the judged population as it stood at export.
- `report` writes `report.md` and `report.json`, and the stored interpretations and verdicts as
  two gzipped JSON-lines files beside them. `restore` loads those two files back into a store,
  so no model call is needed to rebuild the report. It does not bring the labels back: `report`
  also needs `--corpus` and reads `labels.json` from the store (an absent file counts as no labels), so
  copy the committed `labels.json` into the store before running `report` on a restored store.

`run` and `grade` send each fact table to the Anthropic API, as `interpret` does, and need
`ANTHROPIC_API_KEY` (or an `ant auth login` profile). They are billed. Two live tests run under
`make check-llm`: phase 1's interpretation and one real judge call.

What it does not establish:

- The judge's error is measured on about 150 statements, so its intervals are wide for rare
  categories, and the report says so per category.
- The labels are one person's. No agreement between labellers is measured.
- The corpus is about 25 tables. The rates describe analyses like these, not every dataset a user
  may upload.
- One model: the rates describe `claude-opus-5-5` at its default effort.
- D1 to D5 detect patterns. A misreading phrased outside a grader's patterns is left to the judge,
  or missed; the labelled sample estimates how much.

**Results (2026-10-14).** The full study ran: 25 cases, both prompts, four replies each. The report,
its evidence and the labels are in `docs/reports/2026-10-07-llm-eval/` (`report.md`; the bundle
rebuilds the same `report.json`). Every figure below is copied from `report.json`.

- **The run.** Baseline: 100 interpretations, 99 verified (84 on the first attempt), 1 refused.
  `cite-each-status`: 100, all verified (91 on the first attempt). The judge returned 199 verdicts,
  none unusable. Recorded cost: $16.50 and $15.75 for the interpretations, $9.14 for the judge.
- **The labels.** 150 statements, labelled by hand by the project's owner: 144 `ok`, 6 misreadings.
  The sample draws from three strata, so each rate below is weighted back to all 4,007 judged
  statements: all 20 judge-flagged statements, all 23 flagged only by a grader, and 107 of the
  3,964 that nothing flagged.
- **The judge misses most misreadings.** It flagged 4 of the 6, and 4 of its 20 flags were
  misreadings. The 2 it missed came from the unflagged stratum, where each label stands for about 37
  statements, so its weighted sensitivity for any judge category is **5.1% [0.1%, 67.0%]**
  (n_eff 2.2); specificity is 99.6% [95.8%, 100.0%]. Both misses quote the estimate of a method
  whose band is `INSUFFICIENT` without a caveat.
- **So the judge-graded rates are not reliable in either direction.** It flagged a misreading in 11
  of 99 baseline narratives (11.1%) and 9 of 100 `cite-each-status` narratives (9.0%). But 16 of
  its 20 labelled flags were not misreadings, and it missed most of those that were, so these
  figures neither bound nor estimate the true rate.
- **The rule-based graders flag sentences that are true.** Of the statements flagged only by a
  grader, every one labelled was `ok`: D4 0 of 16 (precision 0.0% [0.0%, 19.4%]), D2 0 of 5, D1 0
  of 2. D4's baseline summaries were 10 true, 0 false and 5 undetermined.
- **The primary endpoint.** `cite-each-status` cut D4 per narrative on first attempts by **14.0
  points [-26.0, -2.0]** over 25 paired cases. Given D4's precision, this measures fewer uncited
  status summaries, not fewer misreadings.

What the study shows, then: verification guarantees the numbers, the commonest remaining misreading
is an unqualified `INSUFFICIENT` estimate, and neither the judge nor the graders measure it
reliably. That misreading is now refused: the verifier's rule `insufficient_caveat` (rule 6 under
the interpretation core, PR #16) requires a statement citing such an estimate to cite its band,
and the system prompt states it, so `prompt_version` moved from `a28f0812170d` to `5b08e57994d4`.
The baseline figures above were measured under `a28f0812170d`, and `cite-each-status` under its
own `5d1427d1ea82`. Design §15.5 records the details.

## Web dashboard (sub-project 5)

A browser UI over the backend: sign in with an API key, upload a CSV or pick an earlier upload,
declare its schema, choose estimators or let the router choose, and read the estimates, validity
bands, cross-method comparison and each evidence report. It is a Next.js server in `web/`, and it
is the only thing the browser talks to.

```bash
docker compose up --build
docker compose exec api cp-api-admin create-user alice
docker compose exec -T api cp-api-admin issue-key alice   # shown once; paste it at localhost:3000
```

### What it does with the key

The pasted key is checked with `GET /v1/me` and, if the API accepts it, sealed into an HTTP-only
cookie, `cp_session`, valid for seven days, `SameSite=Lax`, and marked `Secure` only when
`CP_COOKIE_SECURE=true`. The server adds the key to each API call. After sign-in, page JavaScript
never holds it (the sign-in field does, until it is submitted), and the API needs no CORS policy,
because the browser never calls it. The dashboard does not decide whether a key is still good: the
API does, on every call. After a key is revoked, the next page that calls the API gets a 401, clears
the cookie, and lands on sign-in. The server exits at start-up without `CP_SESSION_SECRET` (at least
32 characters) and `CP_API_URL`; the compose file sets both, the secret to a local-only value.
With `CP_ENV=production`, which only the production compose file sets, it also exits on that
local-only secret and unless `CP_COOKIE_SECURE` is `true` ("Production stack" below).

### The session gate and the Content-Security-Policy

`web/src/proxy.ts`, the Next.js middleware, runs on every page request. It looks only for the
session cookie: a request without one is redirected to `/sign-in`. Two paths are let through
without a session, each as that exact path: `/sign-in`, and `/study`, the benchmark explorer,
whose page calls no API and reads no session. A sub-path of either is gated like any other page.
The middleware's matcher leaves
out `/api` and `/sign-out`, each as that path and its sub-paths, and the static files, and nothing
else, so `/sign-in` and `/study` are matched and served under the policy. `/api` has to stay out: when the middleware runs on a request,
Next.js buffers its body and truncates it at 10 MB, which would cut an upload short. The relay
handlers under `/api` answer 401 themselves when there is no session.

The header goes by the session's key. Signed in, it shows "Analyses", "Benchmark study", "New
analysis", the user's name and a "Sign out" button. A visitor, who can only be on `/study`, sees
the "Benchmark study" link and a "Sign in" link to `/sign-in`, and nothing else.

The middleware also sets a Content-Security-Policy on every response it answers, the redirect to
sign-in included, built in `web/src/lib/csp.ts`:

```
default-src 'self'; script-src 'self' 'nonce-<per request>' 'strict-dynamic'; style-src 'self' 'unsafe-inline';
img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self';
form-action 'self'; frame-ancestors 'none'
```

Scripts run only with the request's nonce, which the middleware makes and Next.js puts on its own
scripts. Styles allow inline for two reasons: the study matrix and the interval plot set custom
properties in `style` attributes, and an evidence report is shown in a sandboxed `srcdoc` frame,
which inherits the page's policy and carries its own `<style>` element and `data:` images. The
frame is still granted nothing: no scripts, no forms, no same-origin access. In development
`script-src` also has `'unsafe-eval'`, which React's development build needs. Because the nonce is
per request, no page of the dashboard's is prerendered: the root layout waits for the request, so every page is
rendered per request, sign-in and the 404 page included. The policy is enforced, not report-only,
and the browser suite fails on any violation of it. With `CP_ENV=production` the policy ends with
one more directive, `upgrade-insecure-requests`; without it the policy is the one above, because
the demo stack is served over plain HTTP. `Strict-Transport-Security` is not the dashboard's to
set: the production stack's proxy sets it ("Production stack" below).

### When the API does not answer, and when a page fails

Every call the dashboard's server makes to the API has a deadline, one of three constants in
`web/src/lib/api/client.ts`. A page's own reads and the analysis submission have 30 seconds. A
relayed call (the poll for a running analysis, a manifest or a Markdown report) has 60 seconds.
The upload has 600 seconds,
because its body is streamed from the browser and the deadline covers the time the browser takes
to send the file: 50 MB at 1 Mbit/s is about seven minutes. The deadline keeps running while the
response body arrives. A call that outlives it is treated as a refused connection is: the page
shows its "service is unavailable" state, and a relay handler answers 503 with
`the service is unavailable`. A relayed download that stalls after its headers were sent is cut
off at the deadline.

A read the API refuses with 429 is not the service being down, and the page says what the API
said: the refusal's own sentence, in place of the data, with a "Try again" button. The poll for a
running analysis shows the sentence followed by "Still checking." and keeps polling at its usual
lengthening pace. A 429 whose body holds no sentence is shown as "Too many requests. Try again
later." This matters when rate limits are set: an address over the limit on failed sign-ins is
answered 429 on every route until its window ends.

A signed-in page that throws anything else shows an error page inside the header: "Something went
wrong on this page.", "Nothing was changed by this. You can try again.", and a "Try again" button.
A failure of the root layout itself shows the same words in a document of its own. Neither shows
the error's message or stack; the framework's digest, when there is one, is shown as a reference
that a report can quote.

### What it shows, and what it does not write

The analysis page renders what the API returns as data: status, the routing decisions with their
reasons, each estimate with its interval and validity band, one plot panel per comparison group,
and the comparison verdicts. While an analysis is queued or running the page polls for it. Each
evidence report is the API's own HTML, shown in a frame that is granted nothing, with the Markdown
version offered as a download. Bands, verdicts, refusal reasons and errors are the backend's
strings, shown as they arrive; the dashboard writes no causal prose of its own. An interval is
labelled with its own basis, and methods share a plot axis only inside one of the API's comparison
groups, which are made of estimates with the same estimand and scale. A value the API reports as
`null` is shown as a dash.

The schema form checks that the required fields are filled. Every other rule is the API's, and its
refusal is shown verbatim.

### The benchmark explorer

`/study`, linked from the header as "Benchmark study", is the published benchmark report
(`docs/reports/2026-09-26-benchmark-study/`) made explorable. It is open to visitors: it is the
one page besides sign-in that needs no session. It shows the
report's sections in order (overview, findings, methods, CATE, scenario P7, limitations), and in
Methods an estimator-by-scenario matrix of any headline metric: the six metric families, by
scenario or by scenario and size, filtered by scenario shape and estimator family, as a heatmap or
a sortable table. Each value carries its Monte Carlo standard error and, where fewer fits fed it
than were fitted, the count. A pair with no number says why (not eligible, every fit refused, no
fit feeds the metric, the metric undefined), in a hatch, a word or a dash, and is never coloured
on the metric's scale. The view lives in the URL's query string, and every switch is a link, so a
view can be shared and the back button steps through views.

It reads no API. The data is `web/src/data/benchmark-study.json`, the study bundle, committed, and
imported by one server-only module, so the 1.85 MB file never reaches the browser. The page writes
no claims of its own: narratives, limitations, notes and metric definitions are the bundle's
strings, shown as they are. It refuses a bundle whose `schema_version`, scenario shapes or
estimator families it does not know, with a message that says so.

`bench study-report` writes the bundle in the same run as the report, from the same evidence, and
the bundle records the report's SHA-256. The stores it reads, `benchmarks/runs/study-60`,
`cate-50` and `p7-50`, are git-ignored and exist only on the machine that ran them. From the
repository root, with `RUNS` the directory holding them and `OUT` an existing directory:

```bash
REPORT=docs/reports/2026-09-26-benchmark-study
uv run causalpulse bench study-report "$RUNS/study-60" --out "$OUT" --format markdown \
  --narratives "$REPORT/narratives.json" \
  --cate-run "$RUNS/cate-50" --p7-run "$RUNS/p7-50" \
  --limitation "<the report's first Limitations bullet, verbatim>" \
  --limitation "<the report's second Limitations bullet, verbatim>" \
  --bundle web/src/data/benchmark-study.json
```

The two `--limitation` texts are the published report's own two Limitations bullets, without the
leading `- `; read them from `$REPORT/benchmark-report.md` instead of retyping them. Point `OUT` at
an empty scratch directory to check that the current code reproduces the committed report byte for
byte, or at `$REPORT` to replace it. Report and bundle must be regenerated together:
`tests/report/test_bundle_committed.py` fails when one is regenerated without the other. The
bundle is written only by this command and is never edited by hand. A bundle whose shape the page
does not expect fails `make check-web`, which type-checks the import. The Python test
`tests/report/test_bundle_committed.py` also ties the bundle's copies of the code's own constants
(the metric definitions, each family's metrics, the levels and each estimator's family) to the code.
Neither check can show that the rest of the bundle is what the current code would write; only
running the command shows that.
`docs/specs/2026-10-04-dashboard-phase2-explorer-design.md` section 12 records what changed while
building it.

### Its API types cannot drift

`web/openapi.json` is the API's OpenAPI document, committed. `make openapi` rewrites it. A Python
test fails when a route or response model changes and the file does not, and `make check-web`
fails when the file changes and the generated TypeScript types do not.

### Testing it

```bash
make check-web   # lint, generated types current, type-check, unit tests; no services
make smoke-web   # whole stack + browser; project causalpulse-web-smoke, ports 5436/6383/8002/3001
```

`make check-web` and `make smoke-web` both run `pnpm` on the host, so both need Node.js 24 and
`pnpm`. `make smoke-web` builds both images, issues two keys of one user inside the container,
installs Playwright's Chromium, and runs thirteen browser tests: sign-in and a signed-in page each
served under the policy, with different nonces; a path that is no page, which redirects to sign-in
without a session, while `/study` alone opens for a visitor, under the policy, with a "Sign in"
link and no "Sign out", and a sub-path of it redirects, and which is a 404 under the policy with a
session; a wrong key and then the issued one;
LaLonde from upload through two estimators to an opened evidence report; Proposition 99 with
`synthetic_control` through to the end of its analysis; a 12 MB upload; an upload one byte over the
API's 50 MB limit and one of exactly that size; an upload posted from another origin, which is
refused with a 403; the header at 320, 360 and 390 px wide, on one line with every control inside
it and no sideways scroll; the benchmark study opened from the header, with a figure from the
published report found at its estimator and scenario at both levels; the header's "Sign out"
button; and a revoked key. After every test the suite asserts that the browser's console logged no
violation of the policy. It tears the stack down afterwards.

`make smoke-prod` runs the same thirteen tests through the production stack's proxy over HTTPS,
and then eleven more in `web/e2e/production.spec.ts` that only that stack can pass ("Production
stack" below). `make smoke-web` leaves that file out.

## On scikit-learn

scikit-learn fits nuisance models only — the propensity model and the outcome regressions, which
are prediction problems. Every causal quantity is computed here: the weighting scheme, the
influence function, the cross-fitting structure, the standard errors, and the diagnostics.

## Development

```bash
make install    # sync the uv workspace
make check      # lint, type-check, import contracts, tests
make test-fast  # skip reference-parity tests
make check-llm  # one real interpretation (one or two model calls); needs ANTHROPIC_API_KEY
```

## Continuous integration

GitHub Actions runs the same `make` targets a developer runs. A workflow sets a job up and calls
one target; nothing a gate does is written a second time in YAML, so a gate means the same thing
on a laptop and on a runner. One thing is not of that shape: the Deploy workflow is not a gate
at all ("Deployment" below).

| When | Job | Runs |
|---|---|---|
| Every pull request, and every push to `main` | Static | `make lint types imports` |
| | Services | `make check-services` |
| | Web | `make check-web` |
| | Browser | `make smoke-web` |
| | Stack | `make smoke` |
| Every push to `main`, and no pull request | Production | `make smoke-prod` |
| A pull request that changes what it tests, and every push to `main` | Fast suite | `make test-fast SHARD=I/4 WORKERS="$(nproc)"`, four shards in parallel |
| A pull request or a push to `main` that changes what it checks | Infrastructure | `make check-infra` |
| Monthly (the 1st, 03:00 UTC), and by hand | Slow tier | `make test-slow SHARD=I/8 SLOW_WORKERS="$(nproc)"`, eight shards in parallel |
| By hand only | Live model call | `make check-llm` |
| By hand only, from `main` | Deploy | no `make` target: it puts one commit of `main` on the server ("Deployment" below) |

The repository is private, so its Actions minutes are counted against a monthly allowance, and
the shape above is what fits it. The five quick jobs from Static to Stack, which are all a pull
request runs unless it touches what the fast suite or the infrastructure gate reads, cost about
ten billed minutes together (each job is rounded up to a whole minute).
The Infrastructure gate has a workflow of its own (`infra.yml`), because a workflow can be
limited to the paths a change touches and a job cannot. It runs, on a pull request and on a
push to `main` alike, only when a changed file is under `infra/`, `deploy/` or
`.github/workflows/`, or is one of the two test files `tests/test_deploy_*.py`, the `Makefile`,
the `Dockerfile`, `.dockerignore` or `docker-compose.prod.yml`; a change to the Python or the
dashboard alone does not start it and is billed nothing for it. It installs one release of
`shellcheck` and one of `actionlint`, each checked against its SHA-256, so that what the gate
objects to does not change with the runner's image. `make check-infra` itself ends by running the
two test files, which run the deploy script and the server's deploy command against stand-ins;
the fast suite holds them too, but its pull-request paths do not include `deploy/` or `infra/`.
What the workflow costs has not been measured on a runner: it has not run there yet.
A push to `main` also runs Production, which builds the images the Browser job has already
built and runs that job's browser tests a second time before its own, so it is kept off pull
requests; what it costs has not been measured on a runner. A pull request that changes the
production stack is checked with `make smoke-prod` on a developer's machine before it is merged,
by convention, and by that job once it is.
The fast suite cost about 45 minutes as one job on the two-CPU runners a private repository is
given. It now runs as four shards at once, split by test file; measured locally, the largest shard
holds about 19 of the suite's 63 minutes of summed test time, and the shards' time on a runner has
not been measured yet. Each shard repeats the checkout and `uv sync`, so the four cost a little
more runner time than the one job did. On a pull request it runs only when the packages, the
tests, the dependency pins or a committed file the tests check or read has changed: the OpenAPI
document, the study bundle, the published reports, the import contracts, the `Dockerfile`, the
compose files at the root and the Makefile
(`.github/workflows/fast.yml` lists the paths); a pull request that touches only the dashboard or
the documentation does not pay for it. Every push to `main` runs it regardless. The 45 minutes
were measured with four test workers on two CPUs, in one job. The workflows now ask for one worker
for each CPU the runner reports (`nproc`), for each fast and slow shard. Locally the defaults are unchanged: `auto` for the fast suite and four
for the slow tier.

GitHub does not enforce branch protection on a private repository under the free plan. Changes
to `main` go by pull request, merged when its checks are green, by convention and not by a rule
the platform applies: a direct push is accepted. A branch has no run until it has a pull request.

The slow tier is the tests marked `slow` that need no services and no model API. It is split by
test file (`pytest --shard I/N`, a hook in `tests/conftest.py`), each shard keeps its JUnit report
as an artifact, and a scheduled run in which any shard does not succeed opens an issue naming it.
Start it by hand with `gh workflow run slow.yml`. On four-core runners one run took about two
hours and about 170 runner-minutes, with four workers a shard (it has not been run on the smaller
runners of a private repository, or with a worker for each CPU, where it will differ), set by one file, `tests/causal/test_hierarchical_did_recovery.py`; the other seven
shards finish in under 20 minutes. GitHub stops scheduled workflows in a repository with no
activity for 60 days.

The live model call is billed, so it never runs on a schedule or for a pull request. It needs the
repository secret `ANTHROPIC_API_KEY` and is started with `gh workflow run llm-live.yml`.

The interpreter is the one `.python-version` names, 3.13, which is what the suite's thresholds
were measured on. The Docker image is built from `python:3.13-slim`, so the containers the Stack,
Browser and Production jobs test run 3.13 too, and `tests/test_image.py` fails if the image and
`.python-version` come to name different versions. Its processes run as a user that is not root
(`app`, uid 10001), which can read `/app` and cannot write to it; what they write while running
goes under `/tmp`.

Each `pytest-xdist` worker is given one numerical thread (`ONE_THREAD` in the Makefile). Without
it, four workers on a four-core Linux runner each open a full set of BLAS and OpenMP threads: the
fast suite reached 22% in an hour there, and with the cap it finished in about half an hour. macOS
does not show the problem; the likely reason, not tested, is that numpy uses Accelerate there.
That was four workers because the workflow then asked for four; it now asks for as many as the
runner has CPUs, and the cap stays.

## Production stack

`docker-compose.prod.yml` is a second compose file, beside the demo one, for putting the stack on
one small server on the open internet. It is built and it is tested on a developer's machine
(`make smoke-prod`, below). The configuration for a server, a deploy pipeline, backups and a
runbook now exists; see "Deployment" below, which also says whether the server has been created.

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

The demo file (`docker-compose.yml`) is unchanged, and so is every gate that uses it. The
production file runs the same services from the same two images and differs in this:

- **One more service, `proxy`, and it is the only one with a port on the host.** The demo file
  publishes the API, Postgres, Redis and the dashboard; here each is reached over the stack's own
  network and from nowhere else.
- **No secret has a default.** Without `POSTGRES_PASSWORD`, `CP_SESSION_SECRET` or
  `CP_SITE_ADDRESS`, compose stops and names the one that is missing.
- **`CP_ENV=production`** is set on the API, the worker, the migration and the dashboard, here and
  nowhere else. Everything it switches on is off without it.
- **The session cookie is `Secure`** (`CP_COOKIE_SECURE=true`).
- **The five rate limits are set**, each to a default.
- **Every long-running service** restarts unless stopped and has a health check; the dashboard
  waits for the API to be healthy and the proxy for the dashboard. Each service's log is kept to
  three files of 10 MB.
- **The worker has 2 GB**, memory and swap together. A fit that grows past it is killed and the
  worker restarted, and the analysis it held is then treated as any analysis whose worker died
  ("Backend" above). The
  figure is twice the largest peak measured for one analysis (1,013 MB, `synthetic_control` with
  `bsts` on Proposition 99).
- **Postgres's data and the proxy's certificates are on named volumes.** Redis has none.

### Settings

They come from the environment, which on a server is a file named `.env` beside the compose file.
`.env.example` lists every name with a comment and no value, and says how to generate the two
secrets; `.env` itself is ignored by git.

| Name | What it is |
|---|---|
| `CP_SITE_ADDRESS` | The site's address, as Caddy reads it. A public name gets a public certificate; `localhost` gets one from Caddy's own local authority. Required. |
| `POSTGRES_PASSWORD` | The database's password, placed in `DATABASE_URL` as it is, so letters and digits only. Required. |
| `CP_SESSION_SECRET` | What the dashboard seals its session cookie with, at least 32 characters. Required. |
| `CP_LIMIT_SIGNIN_FAILURES`, `CP_LIMIT_UPLOADS_PER_DAY`, `CP_LIMIT_ANALYSES_PER_DAY`, `CP_LIMIT_ANALYSES_ACTIVE`, `CP_LIMIT_INTERPRETATIONS_PER_DAY` | The rate limits, below. Optional. `CP_LIMIT_INTERPRETATIONS_PER_DAY` is per user and defaults to 5 in the production file. |
| `CP_INTERPRETATION` | `on`, exactly, turns model-written interpretations on; anything else, or nothing, leaves them off. The API and the worker both read it; a worker with it off fails an interpretation it is handed without calling the model. The production compose file feeds both from this one setting. Optional. |
| `ANTHROPIC_API_KEY` | The model's key. Read by the worker only, never by the API. Required when `CP_INTERPRETATION=on`. |
| `CP_HTTP_PORT`, `CP_HTTPS_PORT` | The host ports the proxy listens on. Empty on a server (80 and 443); `make smoke-prod` sets them. |

`DATABASE_URL` and `REDIS_URL` are assembled in the compose file. Redis has no password: it is on
the stack's network only.

### What the services refuse to start with

With `CP_ENV=production`:

- The API, the worker, the migration and `cp-api-admin` exit before they contact anything if
  `DATABASE_URL` is unset or cannot be read as a URL, has no password, has a password this
  repository publishes (the demo stack's, or an empty one), or carries a `password` query
  parameter, whatever that holds: the driver would use it in preference to the one before the
  `@`.
- The dashboard exits if `CP_SESSION_SECRET` is the demo stack's, or if `CP_COOKIE_SECURE` is not
  `true`.

Each prints one line that names the setting and never its value, and exits non-zero. Compose then
shows the service restarting, and nothing is served with it. The lists of published values sit
beside the code that reads each setting, and two tests (`tests/api/test_settings.py`,
`web/src/lib/startup.test.ts`) read the compose files at the root and the Makefile, and the second
`web/` too, and fail if a password or session secret written there is missing from its list.
Without `CP_ENV=production` none of this runs.

One refusal applies in every environment: the worker exits at start-up if `CP_INTERPRETATION=on` and `ANTHROPIC_API_KEY` is not set.

The API also stops at start-up, in any environment, on a rate limit whose setting is not a whole
number greater than zero.

### Rate limits

Counters in Redis, in fixed windows, kept by the API.

| Setting | In the production file | Counts | Per |
|---|---|---|---|
| `CP_LIMIT_SIGNIN_FAILURES` | 10 in 15 minutes | requests answered 401 | client address |
| `CP_LIMIT_UPLOADS_PER_DAY` | 20 | uploads received | user |
| `CP_LIMIT_ANALYSES_PER_DAY` | 30 | analyses accepted | user |
| `CP_LIMIT_ANALYSES_ACTIVE` | 3 | the user's analyses queued or running | user |
| `CP_LIMIT_INTERPRETATIONS_PER_DAY` | 5 | interpretations accepted | user |

- **Uploads are counted when received**, after the key is checked and before any of the body is
  read, and the count is kept whatever becomes of the upload: one refused as too large or as not
  a CSV, and one that repeats an earlier file, each used one. What the allowance bounds is the
  reading, and only a count taken first can refuse without it.
- **Analyses are counted when accepted.** A submission answered 404 or 422, or refused by the
  active limit, is not one of the day's.
- **Interpretations are counted when accepted**, after every refusal check: a request answered 403, 404 or 409 is not one of the day's. A request that is accepted counts whatever becomes of the interpretation, including one that later fails.
- **A day is 24 hours from the first request counted**, not a calendar day, and the 15 minutes
  likewise. A refused request does not push the window's end back.
- **Failed sign-ins** are every 401 on a `/v1` route, the upload route included, counted against
  the client's address. An address that has reached the limit is answered 429 on every `/v1`
  route but the health check, before its key is looked at, for the rest of its 15 minutes: a good
  key from that address is refused too, and a request that succeeds earlier does not reset the
  count. The count is read on the way in and added to after the answer, so requests made at the
  same moment can all be answered before any is counted. That is accepted: a key's secret is 256
  bits.
- **The active limit is a count of rows** read when an analysis is submitted, not a counter, so
  it cannot drift. Before it refuses, the API checks whether any of the user's queued or running
  analyses has lost its job, marks those failed and counts again; it does not judge one
  submitted in the last 30 seconds.
- **A counted submission that then fails keeps its place in the count.** If the insert or the
  enqueue fails after the analysis was counted, the day's count is not given back.
- **Per user, not per key.** Two keys of one user share one allowance.
- **If Redis cannot be reached, or does not answer within half a second, every limit allows the
  request**, and the API logs that once.

Each limit is off unless its setting is present: the API treats an absent or empty setting as no
limit, so the demo stack and the other gates have none. The production file sets all four, and
there an empty value takes the default, so a name left blank in `.env` does not switch a limit
off. To change one, set its number in `.env` and recreate the API
(`docker compose -f docker-compose.prod.yml up -d api`).

A refused request is answered 429 with `{"detail": "<sentence>"}` and a `Retry-After` header in
seconds: the time left in the window, or 60 for the active limit. The sentences:

```
Too many failed sign-ins from this address. Try again in 12 minutes.
Too many uploads today for this account. Try again in 3 hours.
Too many analyses today for this account. Try again in 3 hours.
This account already has 3 analyses waiting or running. Try again when one has finished.
```

The wait is in its largest whole unit, rounded up. The dashboard shows the sentence wherever the
refusal lands (sign-in, the upload, the submission, a page's read, the poll); its upload relay
passes the status and the body on and not the `Retry-After` header.

### The proxy

Stock Caddy (`caddy:2`, configured by `deploy/Caddyfile`). It obtains and renews the site's
certificate by itself, redirects HTTP to HTTPS, and passes everything else to the dashboard. It
adds two headers to every response, relayed downloads included:
`Strict-Transport-Security: max-age=31536000`, without `includeSubDomains` and without `preload`,
and `X-Content-Type-Options: nosniff`. It serves HTTP/1.1 and HTTP/2 and not HTTP/3, has its admin
endpoint off, and writes no access log. Caddy's error log does record the client's address for a request
that fails with a 5xx; it is kept within the container log's size limit.

- **Published on IPv4 only** (`0.0.0.0:80` and `0.0.0.0:443`). On a Linux server Docker's
  userland proxy hands an IPv6 connection to the container from the bridge gateway's IPv4
  address, so every IPv6 visitor would reach Caddy as one client, and the limit on failed
  sign-ins would lock them all out together. **The site's name must therefore have an A record
  and no AAAA record.**
- **It runs as root inside its container**, because the stock image does: it binds 80 and 443 and
  owns its certificate volume. Every capability is dropped but `NET_BIND_SERVICE`, and it cannot
  gain a privilege through a setuid binary. One consequence: root without the capability to read
  past a file's permissions reads `deploy/Caddyfile`, which is mounted from the checkout, only if
  the file is readable by everyone. **On the server `deploy/Caddyfile` must be world-readable**
  (mode 644); checked out under a restrictive umask, Caddy could not read its configuration. A
  deploy does not check the file out: it copies it out of the app image ("Deployment" below), and
  `make smoke-deploy` runs the proxy on a file taken out that way. The other long-running processes
  of the two application images are not root: the Python image runs as `app` (uid 10001) and the
  dashboard's as `node`.

### How the client's address reaches the API

Failed sign-ins are counted by address, and the API never sees the browser's connection. Caddy
sets `X-Forwarded-For` to the peer it accepted the connection from. The dashboard takes the last
entry of that header and sends it to the API as `X-CP-Client-Address` on every call it makes,
after removing any header of that name a caller supplied. The API believes the header only with
`CP_ENV=production`, where nothing but the dashboard can reach it, and only if it parses as an
address; otherwise, and in every other environment, it uses the connection's own peer.

This assumes exactly one proxy in front of the dashboard. A second one (a CDN, a load balancer)
would make the last entry that proxy's address: every user would then be one address, and ten
failed sign-ins by anyone would refuse everyone.

### Upload size

Two bounds, and a user meets the first. The API refuses a file over 50 MiB, and a request body
over 50 MiB plus 1 MiB, with a 413 and its own sentence
(`the upload exceeds the 52428800-byte limit`), which the dashboard shows. The proxy refuses a
body over 64 MB (64,000,000 bytes, above the API's 53,477,376). That is the backstop for a body
that never ends, which the API alone would go on receiving for as long as the client kept the
connection open. Caddy's 413 has no body, so if it is ever the one that answers, the dashboard
shows "the API answered 413".

### What `down -v` and a restart cost

- `docker compose -f docker-compose.prod.yml down` stops the stack and keeps its volumes.
  **`down -v` removes them, which destroys the database and the certificates.** The compose
  file makes no backup. On the server of "Deployment" below, the disk that holds the volumes is
  snapshotted each night.
- **Redis has no volume, on purpose.** What it holds is lost when its container is restarted: the
  queue and the limits' counters. An analysis that was queued is then marked failed, with the
  worker-lost error, the next time it is read, and has to be submitted again; and every limit
  starts counting again from nothing.

### `/study` is public

The benchmark explorer opens without a session, in this stack and in the demo one. It calls no
API. Every other page still redirects a visitor to sign-in.

### Testing it

```bash
make smoke-prod   # production stack + browser over HTTPS; project causalpulse-prod-smoke, ports 8091/8444
```

It needs Docker, `openssl`, `curl`, Node.js 24 and `pnpm`. It generates both secrets in the
recipe's shell and writes neither to a file. It brings the stack up three times, each torn down
with `down -v` whatever happened; a step that fails stops the gate there:

1. **The whole stack**, with `CP_SITE_ADDRESS=localhost:8444` and the limits set low. Then:
   - `deploy/only-the-proxy-publishes.sh` reads the containers as Docker created them and fails
     unless `proxy` is the only service with a port on the host and none is on the host's
     network;
   - the thirteen tests of `workflow.spec.ts` run through the proxy over HTTPS, with Node
     trusting the root certificate copied out of the proxy and the browser told to accept it;
   - the eleven tests of `production.spec.ts` run, whether or not the first run passed: the
     redirect from HTTP, to the same path and query; the two headers and
     `upgrade-insecure-requests` on sign-in and on `/study`, and the two headers on a signed-in
     page and a relayed download; `/study` open to a visitor while another page redirects; the
     cookie `Secure` and `HttpOnly`; the three limits kept per user each driven to its refusal as
     a second user, with the API's sentence on the page; a 12 MB upload from a user over the limit, shown
     that sentence; a 12 MB upload with a revoked key, which lands on sign-in; and failed
     sign-ins until the address is refused, after which a good key is refused too.
2. **The dashboard alone, with the demo stack's session secret.** Its log must say so, Docker
   must have restarted it, and the proxy, started in front of it, must answer with a 5xx.
3. **The API alone, with the demo stack's database password.** Its log must say so and Docker
   must have restarted it.

The two browser runs write their traces to separate directories under `web/test-results/`. A
stack that `production.spec.ts` has run against has spent its limits, so the file cannot be run
twice against one stack. In CI the gate runs on pushes to `main` only ("Continuous integration"
above).

### What is not there yet

- A server that exists, and with it a domain and a real certificate. The configuration for the
  server, the deploy pipeline, the backups and the image registry is written, with a runbook,
  and has not been applied ("Deployment" below).
- IPv6.
- Monitoring beyond the health checks.

The compose file by itself still builds its two images where it runs. On the server an override
(`deploy/compose.server.yml`) names images pulled from a registry in their place.

`docs/specs/2026-10-07-hosting-phase2-production-stack-design.md` section 13 records what changed
while building it, and what is left open.

## Deployment

Everything needed to put the production stack on one small AWS server and to deploy a commit to
it. **The server has not been created yet**: at the time of writing, Terraform has not been
applied, the workflow has never run, and nothing below has met AWS. `docs/runbook.md` has the
procedures, each marked as not yet carried out.

What this phase added:

- **Terraform under `infra/`**, applied by hand from a developer's machine and never by the
  pipeline: one `c7i-flex.large` in `eu-west-1` with a fixed IPv4 address, ports 80 and 443 open and
  nothing else, no SSH key and no port 22; a 30 GB root disk and a separate 20 GB data volume
  that holds the database, the certificates and the environment file, and that Terraform refuses
  to destroy; two private image repositories; a nightly snapshot of the data volume, seven kept;
  a budget that e-mails at 80% of 100 US dollars a month; a role for the pipeline; and the one
  Systems Manager document the pipeline can run. Its state is in a private S3 bucket made by
  `infra/bootstrap/`. DNS is not in Terraform: the site's one A record is added by hand.
- **A first-boot script** (`deploy/first-boot.sh.tftpl`), run once by cloud-init: it mounts the
  data volume (formatting it only when two probes agree it is empty), places Docker's named
  volumes on it, installs Docker and the AWS CLI, makes a 2 GB swap file, writes the environment
  file if there is none, and leaves Ubuntu's security upgrades on with a reboot at 04:00 UTC.
  Its last act writes a marker file on the root disk, and a deploy is refused on a machine that
  does not have it, so a first boot that failed is found out before a deploy and not by one.
- **A deploy workflow** (`.github/workflows/deploy.yml`), started by hand.
- **A deploy script** (`deploy/deploy.sh`) and a compose override (`deploy/compose.server.yml`).
  The app image now carries, under `/deploy`, the production compose file, the override, the
  Caddyfile and the script, laid out as in the repository; nothing in the application reads that
  directory. A deploy copies those files out of the image it has just pulled, so the server needs
  no access to the repository, and a deploy of an older commit runs that commit's script against
  that commit's files. The script pulls the two images, runs the migration before any service is
  replaced, brings the stack up without building, waits for every service to be healthy and to be
  so again 25 seconds later in the same container, and only then records the commit. It does not
  roll back by itself. The command that wraps it on the server keeps the previous commit's files
  until the script has returned and puts them back when the script replaced nothing, so that the
  proxy's Caddyfile on disk is the running commit's after a failed migration or a failed pull.
- **Image tags that are written once.** Both repositories refuse a second push of a tag they
  have, so a commit's SHA names one image for as long as the image is kept.

### Two gates, and what each proves

```bash
make check-infra    # Terraform, the scripts and the workflows, with no AWS account
make smoke-deploy   # the real deploy script against a throwaway registry; not run in CI
```

`make check-infra` runs Terraform with every source of AWS credentials taken away. It checks
the formatting; compares the resources declared with a committed list
(`infra/tests/inventory.txt`), so that one added or removed is seen by a reviewer, and refuses
certain text anywhere under `infra/` (a key pair, a rule for port 22, a wide AWS policy among
them); validates both configurations; runs 25 tests against a mocked provider
(`infra/tests/*.tftest.hcl`), which assert, among other things, that the only inbound ports are
80 and 443 over IPv4, that the machine has no key and requires version 2 of the metadata
service, that the pipeline's role trusts exactly one repository's `main`, and what each role's
policy names; renders the first-boot script and the deploy document's script and puts both
through `shellcheck`; runs `shellcheck` on the scripts and `actionlint` on the workflows; and
last runs two test files with `uv`: `tests/test_deploy_files.py`, which runs the deploy script
against a stand-in for Docker, and `tests/test_deploy_command.py`, which runs the server's
deploy command against stand-ins for Docker, the AWS CLI and the lock, and the first-boot
script's writing of the environment file twice over. It is the Infrastructure workflow in CI.
What it does not prove: the tests read a plan made against a
mock and the checks read text, so AWS has parsed none of the policies, and nothing has been
created. The first-boot script's steps that need the machine (the data volume, the mounts,
Docker, the swap file) have not been run anywhere.

`make smoke-deploy` builds both images, pushes them to a registry container on the developer's
machine under made-up commit SHAs, and runs the server's own deploy command, as Terraform
would render it, against local Docker: so `deploy/deploy.sh` is taken out of the image by the
lines that take it out on the server. Six cases: a first deploy from nothing comes up
healthy; a second tag whose Caddyfile differs replaces the first, and the proxy is replaced
while Postgres and Redis are not; the first tag deployed again creates no image and serves its
own Caddyfile; a tag whose worker reads healthy and then dies fails with status 30 and leaves
the recorded commit alone; and a deploy of the commit already recorded keeps the previous one's
images, while a failed migration ends with status 20 and leaves every running container the one
it was; and a failed migration from an image whose Caddyfile differs leaves the running commit's
Caddyfile on disk, so that the proxy, restarted, still serves it. What it does not prove:
anything of AWS; the command under Systems Manager, as root, with the server's shell, lock and
registry login (stand-ins answer for the last two); what Docker does with the Caddyfile's mount
at a restart on a Linux server, since Docker Desktop does not mount files as Linux does; or what
Compose does when an image cannot be pulled (that path is tested against a stand-in for Docker
in `tests/test_deploy_files.py`). It needs Docker, uses the ports
of `make smoke-prod`, so the two do not run together, and is not run in CI because it builds
both images and the minutes are counted.

### Starting a deploy

```bash
gh workflow run deploy.yml --ref main -f sha=<the commit's full SHA> -f confirm=<the site's name>
```

Or from the Actions tab: "Deploy", "Run workflow", on `main`. Nothing deploys by itself. The
workflow refuses unless the commit is on `main` and the `CI` workflow succeeded for it as the tip
of a push to `main`. The fast suite and the Infrastructure workflow are asked for in the same
way: where either ran for the commit, it must have succeeded, so a deploy straight after a merge
waits for the fast suite. It builds and pushes the two images only if the registry does not already
have them for that commit, so a rollback, which is the same workflow with an earlier SHA, builds
nothing while the registry still holds the images. A tag is never pushed twice, and the registry
would refuse it: an image that is bad under its SHA is answered by a new commit, or by an
administrator deleting the tag (the runbook has the command). It then sends the deploy document to the
server, waits for it, prints the end of its output, and asks the site three questions from
outside. A deploy interrupts the site for the seconds the services take to restart.

### Who can deploy, and what the pipeline's role can do

The workflow holds no AWS key. GitHub signs a token for the run and AWS exchanges it for the
pipeline role's short-lived credentials. The role's trust accepts one token subject, compared
whole: a run in this repository from the branch `main`. A run from another branch, a tag or a
pull request is refused, whatever its copy of the workflow says. For the same reason the deploy
job declares no GitHub environment (a job with one presents a different subject), and the five
identifiers the workflow reads are repository variables, none of them a secret. The approval is
the hand start from `main` and the site's name typed as a confirmation.

The role can push to and read the two image repositories, send the one deploy document to the
one machine, and read a command's result. It cannot open a session, run any other document or
any other command, or touch any other resource. The document takes one parameter, which must be
forty lowercase hexadecimal digits.

What that amounts to, stated plainly. The deploy document takes `deploy/deploy.sh` out of the
app image for the commit it is given and runs it as root. So whoever can push an app image and
send the command has root on the server: its environment file and the secrets in it, its
database, and the server's own role. The pipeline's role can do both. The role's trust tells
`main` from every other ref, and does not tell the deploy workflow from any other workflow file
on `main`; GitHub enforces no branch protection on this repository. So anyone who can push to
`main` has, in effect, root on the server. That is the intended boundary for a private
repository with one owner. A test holds that the deploy workflow is the only one that asks
GitHub for an identity token and that it is started by hand only, which guards against an
accident and not against a person who can push. If a second person is ever given the right to
push, the boundary has to move first: enforced branch protection on `main`, and the deploy job
in a GitHub environment with required reviewers, with the role's trust changed to that
environment's subject (the design's section 7). AWS offers no narrower form of the permission to read
a command's result, so the role can read the output of any run-command in the account and
region whose id it is given. The account is shared with another project.

### Secrets

The database password and the session secret are generated on the server at its first boot, with
`openssl rand -hex 32`, into an environment file on the data volume that only root can read. They
never leave the machine: they are not in Terraform's state, in GitHub, in an image or on a
developer's machine, and the first-boot script prints the names it wrote and never the values.
The deploy document takes every value of that file that is eight characters or longer out of a
deploy's output before the output is kept or returned. API keys are issued by hand in an
interactive session on the server, where a key appears once in the operator's terminal; the
runbook says why a run-command must never be used for it. Terraform's state does hold the
account's number and the budget's e-mail address, which is why its bucket is private.

### What it costs

Approximate, on-demand prices in `eu-west-1`, running all the time. These are the design's
estimates; nothing has been billed yet.

| Item | US dollars a month |
|---|---|
| `c7i-flex.large` | 66 |
| 50 GB of disk | 4.40 |
| Public IPv4 address | 3.65 |
| Snapshots, image storage, the state bucket | under 1.50 |
| Total | about 76 |

Stopped, the machine costs its disks and its address: about 9. It is a `c7i-flex.large` and not
the cheaper `t3.medium` first chosen because the AWS account is on the Free plan, which launches
only the instance types it lists as free-tier eligible; this one has the same 4 GB. While the
account is on that plan the charge is taken from its credits. The budget's warning, at 80% of
100 US dollars, counts the whole AWS account, not this project alone.

### Operating it

`docs/runbook.md` has eleven procedures as commands that can be pasted: the first-time setup,
deploying and rolling back, issuing a key, looking at the server, restoring the data volume from
a snapshot, rotating the two secrets, changing a rate limit, stopping and starting the machine,
replacing the machine, tearing everything down, and what to do when a machine's first boot did
not finish. It also lists what exists and where each
thing is defined, and where to start when something is wrong.

`docs/specs/2026-10-08-hosting-phase3-server-and-deploy-design.md` is the design; its section 14
records what changed while building it, and what is left open.
