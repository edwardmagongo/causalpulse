# Tracking benchmark runs

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
