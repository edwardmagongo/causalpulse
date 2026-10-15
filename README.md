# CausalPulse

[![CI](https://github.com/edwardmagongo/causalpulse/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/edwardmagongo/causalpulse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![Tests](https://img.shields.io/badge/Python%20tests-3%2C938-brightgreen)

A causal inference and experimentation platform. Given an intervention and an outcome,
CausalPulse estimates what would have happened without the intervention, how large the effect
was, for whom it differed, and how much the conclusion should be trusted — with the assumption
checks, robustness tests and evidence report behind every number.

The interesting part is not fitting the estimators — it is knowing when to believe them. Every
estimator is scored against simulated data whose true effect is *measured*, not declared,
including scenarios built specifically to break it. A pre-registered 4,000-cell benchmark says
where each one holds and where it fails. And everything that turns a number into words — the
validity score, the evidence report, the dashboard, the language-model summary — is checked so it
never claims more certainty than the diagnostics support.

## Highlights

- **Ground truth is measured, and the estimators cannot reach it.** A scenario's true effect is
  computed from its realised potential outcomes rather than taken from its label, and an
  import-linter contract forbids every estimator package from importing the simulation code, so
  no estimator can be tuned against the answer it is scored on. See
  [Architecture](#architecture).
- **Twelve estimators, each checked where it should fail as well as where it should succeed.**
  Two-way fixed effects is shown breaking under staggered adoption, with the Goodman–Bacon
  decomposition explaining why; synthetic control is pushed outside its donors' convex hull; the
  causal forest's subgroups must pass three independent guards. Reference parity is checked
  against `statsmodels`, `pysyncon` and `econml`. See [Estimators](#estimators).
- **A pre-registered benchmark whose one miss was a real bug.** Eight expectations were written
  down before any result existed; a 4,000-cell run over ten estimators and thirteen scenarios met
  seven. The eighth traced `hierarchical_did`'s collapse on binary outcomes to an outcome
  likelihood that was always Gaussian — since restricted to continuous outcomes. See
  [Benchmark study](#benchmark-study).
- **Real data found what simulation could not.** LaLonde's job-training data and the
  Proposition 99 panel, checked against their published benchmarks, surfaced a propensity fit
  that depended on a column's units and a donor-weight solver that reported success without ever
  leaving its starting point — both caught by checking a property the output owes, not a status
  flag. See [Estimators](#estimators).
- **Reports that refuse to overclaim.** The validity band is ordinal and derived from the
  estimate's own assumption checks; the cross-method comparison never pools estimates into one
  number; and the report generator lints its own prose and refuses to emit a sentence that
  asserts more than its diagnostics support. See [Validity and sensitivity](#validity-and-sensitivity).
- **A language-model summary that cannot invent a number — and a measured account of what it
  still gets wrong.** Every value in a narrative is a placeholder verified against a fact table.
  A 200-interpretation study with 150 hand-labelled statements found the commonest remaining
  misreading, turned it into a verifier rule, and reports its own judge's 5.1% sensitivity as the
  weakness it is. See [Interpretation](#interpretation).
- **3,938 Python tests, 451 dashboard unit tests and 24 browser tests**, plus gates that bring
  up the demo stack, the production stack behind TLS, and the real deploy script against a local
  registry. See [Tests](#tests).

## Table of Contents

- [Stack](#stack)
- [Running it](#running-it)
- [Architecture](#architecture)
- [Estimators](#estimators)
- [Validity and sensitivity](#validity-and-sensitivity)
- [Benchmark study](#benchmark-study)
- [Interpretation](#interpretation)
- [API and dashboard](#api-and-dashboard)
- [Tests](#tests)
- [Design decisions and limits](#design-decisions-and-limits)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [License](#license)

## Stack

Python 3.13 · uv workspace · NumPy · SciPy · pandas · scikit-learn · statsmodels · JAX ·
NumPyro · matplotlib · FastAPI · SQLAlchemy · Alembic · PostgreSQL 16 · Redis · RQ · Anthropic
API · Next.js 16 · React 19 · TypeScript · Vitest · Playwright · Docker · Caddy · Terraform · AWS
(EC2, ECR, Systems Manager) · GitHub Actions

## Running it

Requires Python 3.13 with [uv](https://docs.astral.sh/uv/), and Docker for the stack. The
dashboard needs Node.js 24 and pnpm.

```bash
make install     # sync the uv workspace
make check       # lint, types, import contracts, every test
make test-fast   # everything but the slow tier
```

The `causalpulse` command line covers the whole causal core:

```bash
# a simulated scenario, with its measured ground truth written beside it
uv run causalpulse simulate --scenario D --n-units 2000 --seed 1 --out d.csv

# fit LaLonde's job-training data and write estimates, a manifest and evidence reports
uv run causalpulse analyze --data tests/fixtures/real_data/lalonde.csv \
  --unit unit_id --treatment treat --outcome re78 \
  --covariates age,educ,black,hisp,marr,nodegree,re74,re75 --outcome-type continuous \
  --report --estimators difference_in_means,lin_regression --out out/lalonde

# refit from the manifest and confirm the numbers are bit-identical
uv run causalpulse replay out/lalonde/manifest.json

# the benchmark's gating tier, then its report
uv run causalpulse bench run --tier ci --out benchmarks/runs/ci
uv run causalpulse bench report benchmarks/runs/ci
```

The API, worker and dashboard run as one compose stack:

```bash
docker compose up --build                                  # API on :8000, dashboard on :3000
docker compose exec api cp-api-admin create-user alice
docker compose exec -T api cp-api-admin issue-key alice    # shown once; paste it at localhost:3000
```

## Architecture

```
cp-cli | cp-worker                     entry points: the command line and the queue worker
cp-llm-eval                            how often a verified summary misreads its analysis
cp-llm                                 fact table → model narrative → verifier
cp-report | cp-tracking                evidence reports; MLflow tracking (optional)
cp-figures                             eight figures, each a pure data function plus a drawing
cp-evaluation   ← cp-simulation        the benchmark grid: the one layer that sees ground truth
cp-causal                              estimators, diagnostics, sensitivity, the design router
cp-statistics                          intervals, robust covariance, bootstrap
cp-core                                data and result contracts, no algorithms

cp-api  →  Postgres, Redis             upload, request, read; never fits a model
```

Each package may import only the ones below it — an import-linter `layers` contract, checked in
CI. Six more contracts hold the boundaries that matter most:

| Contract | Why |
|---|---|
| No estimator package may import `cp-simulation` | An estimator that can reach ground truth can be tuned against it, and every benchmark number afterwards is meaningless. |
| The API never fits a model or renders a report | Requests stay fast and bounded; refusals that need the estimators happen in the worker. |
| The worker sees only `cp_api.db` and `cp_api.settings` | The worker shares the database, not the web layer. |
| Only `cp_llm.extract` reads result types | The model is given a fact table, never the objects it could misdescribe. |
| The API and worker never import tracking or the LLM evaluation harness | Offline research tools stay out of the served images. |

| Package | Responsibility |
|---|---|
| `cp-core` | Value types, the dataset contract (which refuses malformed data when a dataset is built, naming the column) and the result contract. |
| `cp-statistics` | Proportion intervals, robust and cluster-robust covariance, bootstrap engines. |
| `cp-simulation` | Compositional data-generating processes carrying measured ground truth. |
| `cp-causal` | Cross-sectional, panel, donor-pool, heterogeneous-effect and Bayesian estimators; assumption checks; sensitivity and falsification; the design router. |
| `cp-evaluation` | The benchmark grid: bias, coverage and calibration with Monte Carlo standard errors, run as a resumable set of shards. |
| `cp-figures` | Counterfactual path, event study, love plot, propensity overlap, specification curve, placebo distribution, forest plot, benchmark heatmaps. A shared guard refuses to plot an estimate without its interval. |
| `cp-report` | The evidence report — assumption table, validity band, cross-method comparison, limitations — as Markdown or one self-contained HTML file. |
| `cp-llm` | A plain-language interpretation whose every number is verified against a fact table. |
| `cp-llm-eval` | Measures how often a verified interpretation still misreads its analysis. |
| `cp-cli` | The `causalpulse` command, and Parquet-plus-JSON persistence for benchmark runs. |
| `cp-api` | The FastAPI service: per-user API keys, CSV upload, analysis requests, results. |
| `cp-worker` | The RQ worker that fits a queued analysis, scores it and renders its reports. |

scikit-learn fits nuisance models only — the propensity model and the outcome regressions, which
are prediction problems. Every causal quantity is computed here: the weighting scheme, the
influence function, the cross-fitting structure, the standard errors and the diagnostics.

## Estimators

| Design | Estimators |
|---|---|
| Cross-sectional | `difference_in_means`, `lin_regression`, `ipw`, `aipw` (cross-fitted) |
| Panel | `did_2x2`, `twfe`, `event_study`, `callaway_santanna`, `hierarchical_did` (Bayesian, NumPyro) |
| One treated unit against donors | `synthetic_control` (ridge-augmented outside the convex hull), `bsts` (Bayesian structural time series) |
| Heterogeneous effects | `causal_forest` (honest, with per-unit CATE and guarded subgroup discovery) |

A design router picks the estimators a dataset is eligible for, and refuses the rest with a
reason. Each estimator is scored against measured truth — the difference-in-differences family
over 60 seeds — on every scenario it is eligible for, including the ones built to make it fail: parallel trends
violated, staggered adoption with heterogeneous effects, a binary panel, a treated unit outside
its donors' convex hull, too few pre-periods.

Two real datasets are vendored and checked against their own published benchmarks: LaLonde's
NSW/CPS-1 job-training data and Abadie, Diamond & Hainmueller's California Proposition 99 panel.
An analysis is saved as a manifest that refers to its data by content hash, and `replay` refits
from it and reproduces bit-identical numbers.

What each estimator is checked against, where each breaks down, and the defects the checks found:
[`docs/estimators.md`](docs/estimators.md).

## Validity and sensitivity

- **A Causal Validity Score**, ordinal (`STRONG`, `MODERATE`, `WEAK`, `INSUFFICIENT`), derived
  from the estimate's own assumption-check table under a ceiling set by how the treatment was
  assigned: an observational design cannot rise above `MODERATE`, however clean its diagnostics.
  A check that cannot be run is `NOT_TESTABLE` and lowers nothing on its own. Every score carries
  the sentence saying it scores diagnostics, not truth.
- **Cross-method agreement without pooling.** Several methods' estimates are compared side by
  side and never averaged into one number.
- **Unmeasured confounding:** an E-value — exact where the estimate's own baseline allows, an
  approximation otherwise — and a Cinelli–Hazlett robustness value.
- **Specification and falsification:** a specification curve over covariate subsets and
  outlier rules; randomisation inference; leave-one-donor-out; placebo distributions that respect
  the convex hull; a negative-control outcome; a correction for a misclassified treatment column
  that can only move an estimate away from zero.
- **Missing outcomes:** complete-case analysis as an explicit, named choice rather than a silent
  default; Manski-style tipping-point bounds; single and multiple imputation pooled by Rubin's
  rules. Two pre-registered expectations about imputation were falsified by measurement, and the
  documentation says so.
- **A causal-language lint**, which the report generator runs over its own prose before emitting
  it — refusing any sentence that asserts more certainty than its diagnostics support — and which
  the test suite runs over every string literal in the packages.

Each in detail, with what the measured harnesses found:
[`docs/validity-and-sensitivity.md`](docs/validity-and-sensitivity.md).

## Benchmark study

`bench run --tier study` runs the full grid at 50 replications as a directory of Parquet shards
plus a ledger, so an interrupted run resumes and a finished one extends. `bench study` turns it
into headline tables pooled over each scenario's baseline slice, with Monte Carlo standard
errors, and scores expectations written down before any result existed.

The 4,000-cell run (ten estimators, thirteen scenarios) met seven of its eight pre-registered
expectations, including the false-positive rate under the null scenario and the confounding bias
on the scenarios built to show it. The eighth — `hierarchical_did`'s pooled panel coverage of at
least 0.90 — failed at 0.625. Read per scenario, that one number held two findings: on the
scenario where parallel trends are violated every DiD-family estimator collapses, as designed; on
the binary panel `hierarchical_did` alone collapsed (0.01 coverage, against 0.89–0.93 for its
peers on the same data), because its likelihood was Gaussian whatever the outcome. It is now
restricted to continuous outcomes.

Further arms score the donor-pool estimators, the causal forest's per-unit effects and a
many-cohort panel. The published report is in
[`docs/reports/2026-09-26-benchmark-study/`](docs/reports/2026-09-26-benchmark-study/), and
the dashboard's benchmark explorer serves the same evidence. Runs can be logged to MLflow:
[`docs/benchmarks.md`](docs/benchmarks.md).

## Interpretation

`causalpulse interpret <run-dir>` writes a plain-language reading of a whole analysis. The model
never writes a number: it writes placeholders that name facts in a table built from the analysis,
and a verifier refuses any narrative it cannot trace back to that table, any qualitative word
outside a controlled list, and any statement that quotes an `INSUFFICIENT` estimate without its
band. One interpretation costs about $0.07 to $0.20. On the server, each interpretation records
the prompt and verifier rules it was written under; when those change, an analysis can be
[interpreted again](docs/interpretation.md#interpreting-again-and-the-earlier-interpretations),
and its earlier interpretations are kept and listed below the new one.

`cp-llm-eval` measures what verification cannot: a narrative that cites only true figures but
still misreads them. The full study — 25 real analyses, two prompts, 200 interpretations, 4,007
judged statements, 150 of them labelled by hand — found that:

- the commonest misreading was quoting an `INSUFFICIENT` estimate without a caveat, now refused
  by the verifier;
- the judge model caught 4 of the 6 labelled misreadings but, weighted back to every statement,
  its sensitivity is 5.1% [0.1%, 67.0%] — so its rates neither bound nor estimate the true one;
- the rule-based graders flagged only sentences that were true.

Method, rates and caveats: [`docs/interpretation.md`](docs/interpretation.md). The study's
evidence and labels: [`docs/reports/2026-10-07-llm-eval/`](docs/reports/2026-10-07-llm-eval/).

## API and dashboard

The API is a FastAPI service over Postgres and an RQ queue: upload a CSV, declare its schema,
request an analysis, and read back estimates, validity bands, the cross-method comparison and
each evidence report. Every route but the health check needs a per-user API key, and a user sees
only their own datasets and analyses. A malformed upload is refused when the dataset is built,
naming the column.

The dashboard is a Next.js server that is the only thing the browser talks to: the API key is
sealed into an HTTP-only cookie after sign-in and never reaches page JavaScript. It covers upload,
schema declaration, estimator choice, results and the evidence reports, behind a nonce-based
Content-Security-Policy. `/study`, the benchmark explorer, is public.

Routes, keys, limits and the dashboard's session and failure handling:
[`docs/backend.md`](docs/backend.md) and [`docs/dashboard.md`](docs/dashboard.md).

## Tests

| Gate | What it covers |
|---|---|
| `make check` | Lint (ruff), strict mypy, the import contracts, and all 3,938 Python tests, with warnings as errors |
| `make test-fast` | The 3,858 tests outside the slow tier |
| `make test-slow` | The slow recovery harnesses, which score estimators over many seeds |
| `make check-services` | The API and worker tests, against a real Postgres and Redis |
| `make check-web` | The dashboard's lint, type checks and 451 unit tests |
| `make smoke` | The API and worker built as images and driven over HTTP |
| `make smoke-web` | 13 Playwright tests of the dashboard against that stack |
| `make smoke-prod` | The production stack behind TLS: 13 browser tests through the proxy, 11 more on its headers, limits and refusals, and a check that only the proxy has a port on the host |
| `make check-infra` | Terraform formatting, validation and mocked tests, an inventory of every resource, and the deploy scripts against stand-ins |
| `make smoke-deploy` | The real deploy script against a local registry: first deploy, upgrade, rollback, a broken image and a failed migration |
| `make check-llm` | One real interpretation; billed, so run by hand |

CI runs the same `make` targets, chosen by what a change touches so a private repository's
Actions minutes go where they matter: the quick gates on every pull request, the fast suite as
four parallel shards, the slow tier monthly. Details: [`docs/ci.md`](docs/ci.md).

## Design decisions and limits

- **No hosted demo.** The stack runs locally with one command; the production configuration and
  its deploy pipeline are complete and tested locally, but no server is running.
- **The benchmark's wall-clock figures were measured on a busy machine** and should be
  re-measured on an idle one before being quoted.
- **No sign-up, and no key issued over HTTP.** Users and keys are created from the command line
  by whoever runs the stack.
- **The language-model judge is not a reliable measure of misreadings**, as the study above
  shows; its rates are reported, labelled as such, and not used as a gate.
- **The rate limits assume one proxy in front.** A second proxy or a CDN would make every visitor
  look like the same address to the sign-in limit.
- **IPv4 only.** The production proxy is published on IPv4 so that every visitor keeps their own
  address for the sign-in limit.
- How CausalPulse was built, phase by phase, with what each phase found:
  [`docs/history.md`](docs/history.md).

## Configuration

The demo stack needs no settings. The production stack reads these from `.env`, which
[`.env.example`](.env.example) lists:

| Variable | Notes |
|---|---|
| `CP_SITE_ADDRESS` | The site's address, as Caddy reads it. A public name gets a public certificate. Required. |
| `POSTGRES_PASSWORD` | The database's password, letters and digits only. Required. |
| `CP_SESSION_SECRET` | What the dashboard seals its session cookie with, at least 32 characters. Required. |
| `CP_LIMIT_SIGNIN_FAILURES` | Failed sign-ins per client address in 15 minutes. Default 10. |
| `CP_LIMIT_UPLOADS_PER_DAY` | Uploads per user per day. Default 20. |
| `CP_LIMIT_ANALYSES_PER_DAY` | Analyses per user per day. Default 30. |
| `CP_LIMIT_ANALYSES_ACTIVE` | Analyses a user may have queued or running at once. Default 3. |
| `CP_LIMIT_INTERPRETATIONS_PER_DAY` | Interpretations per user per day. Default 5. |
| `CP_INTERPRETATION` | `on` turns model-written interpretations on; anything else leaves them off. |
| `ANTHROPIC_API_KEY` | The model's key, read by the worker only. Required when `CP_INTERPRETATION=on`. |

With `CP_ENV=production`, which only the production compose file sets, the API, the worker, the
migration and the dashboard each refuse to start on a secret this repository publishes. Full reference:
[`docs/production.md`](docs/production.md).

## Deployment

`docker-compose.prod.yml` runs the stack behind Caddy, which obtains and renews its own
certificate; only the proxy has a port on the host, and it adds HSTS and `nosniff` to every
response. Rate limits are counted in Redis and checked before an upload's body is read.

Terraform under [`infra/`](infra/) describes one small AWS server for it: a fixed IPv4 address,
only ports 80 and 443 open, no SSH, a separate data volume snapshotted nightly, two private image
repositories with tags that cannot be overwritten, and a GitHub OIDC role that can push those
images and run one Systems Manager command on that one instance — nothing else. The secrets are
generated on the server at first boot and never leave it. A deploy is started by hand for a
commit of `main` whose checks passed; the server pulls that commit's images, migrates before
replacing any service, and records the commit only after every service has stayed healthy.

**No server is running** — the configuration is written and checked, but nothing is live in AWS.
Costs, the trust boundary and the deploy gates: [`docs/deployment.md`](docs/deployment.md);
procedures: [`docs/runbook.md`](docs/runbook.md).

## License

[MIT](LICENSE)
