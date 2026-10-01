import json
import math
import re

import numpy as np
import pandas as pd
import pytest
from cp_evaluation.run import ROW_COLUMNS
from cp_tracking.payload import (
    CATE_TRACKED_METRICS,
    FINGERPRINT_TAG,
    TRACKED_METRICS,
    build_payload,
    fingerprint,
)

_KEY = re.compile(r"^[^.]+\.[^.]+\.(?P<metric>[a-z_]+)$")


def test_params_carry_the_runs_identity(loaded):
    frame, manifest, sidecar, run_dir = loaded

    payload = build_payload(frame, manifest, sidecar, run_dir)

    identity = manifest["identity"]
    assert payload.run_name == run_dir.name
    assert payload.params["tier"] == identity["tier"]
    assert payload.params["estimators"] == ",".join(identity["estimators"])
    assert payload.params["root_entropy"] == str(identity["root_entropy"])
    assert payload.params["replications"] == str(manifest["replications"])
    for field, value in identity["fit_config"].items():
        assert json.loads(payload.params[f"fit.{field}"]) == value
    for arm, axes in identity["grid"].items():
        if axes is None:
            continue
        for axis, value in axes.items():
            assert json.loads(payload.params[f"grid.{arm}.{axis}"]) == value


def test_tags_carry_provenance_and_the_fingerprint(loaded):
    frame, manifest, sidecar, run_dir = loaded

    payload = build_payload(frame, manifest, sidecar, run_dir)

    provenance = manifest["provenance"]
    assert payload.tags["cp.git_commit"] == (provenance["git_commit"] or "unknown")
    for name, version in provenance["package_versions"].items():
        assert payload.tags[f"cp.package.{name}"] == version
    assert payload.tags["cp.run_dir"] == str(run_dir.resolve())
    assert payload.tags[FINGERPRINT_TAG] == fingerprint(
        manifest["identity"], len(frame), provenance["git_commit"]
    )


def test_metric_keys_are_scenario_method_metric_and_each_has_its_mcse(loaded):
    payload = build_payload(*loaded)

    assert payload.metrics, "the tiny run should yield at least one headline metric"
    for key, value in payload.metrics.items():
        assert math.isfinite(value), key
        base = key.removesuffix("_mcse")
        match = _KEY.match(base)
        assert match and match["metric"] in TRACKED_METRICS, key
        assert base in payload.metrics and f"{base}_mcse" in payload.metrics, key


def test_cells_with_no_finite_value_are_skipped_not_logged_as_nan(loaded):
    # In the tiny grid's panel scenario P1, neither test estimator is eligible.
    payload = build_payload(*loaded)

    assert not any(key.startswith("P1.") for key in payload.metrics)
    assert any(key.startswith("A.") for key in payload.metrics)


def test_artifacts_hold_every_headline_table_and_the_identity(loaded):
    frame, manifest, sidecar, run_dir = loaded
    from cp_evaluation.study import headline_tables

    payload = build_payload(frame, manifest, sidecar, run_dir)

    assert set(payload.artifacts) == {
        *(f"{name}.csv" for name in headline_tables(frame)),
        "identity.json",
        "sidecar.json",
    }
    assert json.loads(payload.artifacts["identity.json"]) == manifest["identity"]


def test_the_fingerprint_is_stable_and_changes_with_the_rows(loaded):
    frame, manifest, _, _ = loaded
    identity = manifest["identity"]

    assert fingerprint(identity, len(frame), "abc") == fingerprint(identity, len(frame), "abc")
    assert fingerprint(identity, len(frame), "abc") != fingerprint(identity, len(frame) + 1, "abc")


def test_the_fingerprint_changes_with_the_git_commit(loaded):
    frame, manifest, _, _ = loaded
    identity = manifest["identity"]

    assert fingerprint(identity, len(frame), "abc") == fingerprint(identity, len(frame), "abc")
    assert fingerprint(identity, len(frame), "abc") != fingerprint(identity, len(frame), "def")
    assert fingerprint(identity, len(frame), "abc") != fingerprint(identity, len(frame), None)


def test_an_overlong_param_is_truncated_not_dropped(loaded):
    frame, manifest, sidecar, run_dir = loaded
    manifest = json.loads(json.dumps(manifest))
    manifest["identity"]["fit_config"]["options"] = {"blob": "x" * 7000}

    payload = build_payload(frame, manifest, sidecar, run_dir)

    value = payload.params["fit.options"]
    assert len(value) <= 6000 and value.endswith("[truncated; see identity.json]")


def test_an_overlong_estimator_list_is_truncated_not_dropped(loaded):
    frame, manifest, sidecar, run_dir = loaded
    manifest["identity"]["estimators"] = [f"estimator_{i:04d}" for i in range(1000)]

    payload = build_payload(frame, manifest, sidecar, run_dir)

    value = payload.params["estimators"]
    assert len(value) <= 6000 and value.endswith("[truncated; see identity.json]")


def test_the_fingerprint_ignores_key_order_and_sees_a_changed_value(loaded):
    frame, manifest, _, _ = loaded
    identity = manifest["identity"]
    reordered = dict(reversed(list(identity.items())))
    changed = {**identity, "root_entropy": f"{identity['root_entropy']}0"}

    assert list(reordered) != list(identity)
    base = fingerprint(identity, len(frame), "abc")
    assert fingerprint(reordered, len(frame), "abc") == base
    assert fingerprint(changed, len(frame), "abc") != base


def test_metric_values_match_the_headline_tables(loaded):
    from cp_evaluation.study import headline_tables

    frame, manifest, sidecar, run_dir = loaded
    payload = build_payload(frame, manifest, sidecar, run_dir)
    tables = headline_tables(frame)

    for table_name, metric in (
        ("accuracy__scenario", "bias"),
        ("coverage__scenario", "coverage_bootstrap"),
    ):
        table = tables[table_name]
        finite = table[metric].notna() & table[f"{metric}_mcse"].notna()
        rows = table[(table["scenario"] == "A") & finite]
        assert len(rows), f"no finite A.{metric} row to compare"
        row = rows.iloc[0]
        key = f"A.{row['method']}.{metric}"
        assert payload.metrics[key] == float(row[metric])
        assert payload.metrics[f"{key}_mcse"] == float(row[f"{metric}_mcse"])


def test_a_pair_with_a_finite_value_but_no_finite_mcse_is_skipped(loaded, monkeypatch):
    import pandas as pd

    frame, manifest, sidecar, run_dir = loaded
    nan = float("nan")
    tables = {
        "accuracy__scenario": pd.DataFrame(
            {
                "scenario": ["A", "A"],
                "method": ["good", "no_mcse"],
                "bias": [0.1, 0.2],
                "bias_mcse": [0.01, nan],
                "rmse": [0.3, 0.4],
                "rmse_mcse": [0.02, 0.03],
            }
        ),
        "coverage__scenario": pd.DataFrame(
            {
                "scenario": ["A"],
                "method": ["good"],
                "coverage_analytic": [0.9],
                "coverage_analytic_mcse": [0.01],
                "coverage_bootstrap": [0.8],
                "coverage_bootstrap_mcse": [0.01],
            }
        ),
        "se_calibration__scenario": pd.DataFrame(
            {
                "scenario": ["A"],
                "method": ["good"],
                "se_calibration": [1.0],
                "se_calibration_mcse": [0.1],
            }
        ),
    }
    monkeypatch.setattr("cp_tracking.payload.headline_tables", lambda _frame: tables)

    payload = build_payload(frame, manifest, sidecar, run_dir)

    assert payload.metrics["A.good.bias"] == 0.1 and payload.metrics["A.good.bias_mcse"] == 0.01
    assert "A.no_mcse.bias" not in payload.metrics
    assert "A.no_mcse.bias_mcse" not in payload.metrics
    assert payload.metrics["A.no_mcse.rmse"] == 0.4


def _with_cate_rows(frame, *, scenario="RICH", correlations=(0.8, 0.9, 0.7)):
    rows = []
    for seed, correlation in enumerate(correlations):
        row = dict.fromkeys(ROW_COLUMNS, np.nan)
        row.update(
            shape="cate", scenario=scenario, n_units=2000, seed=seed,
            method="causal_forest", status="fitted", reason="", estimand="ATE",
            truth=0.1, point=0.1, runtime_seconds=1.0, cate_correlation=correlation,
            cate_rmse=0.05 + 0.01 * seed, cate_calibration_slope=1.5,
        )
        rows.append(row)
    return pd.concat([frame, pd.DataFrame(rows, columns=list(ROW_COLUMNS))], ignore_index=True)


def test_cate_rows_are_logged_as_cate_metrics_with_their_mcse(loaded):
    frame, manifest, sidecar, run_dir = loaded

    payload = build_payload(_with_cate_rows(frame), manifest, sidecar, run_dir)

    key = "RICH.causal_forest.cate_correlation"
    assert payload.metrics[key] == pytest.approx(0.8)
    assert payload.metrics[f"{key}_mcse"] == pytest.approx(
        float(np.std([0.8, 0.9, 0.7], ddof=1) / np.sqrt(3))
    )
    for metric in CATE_TRACKED_METRICS:
        assert f"RICH.causal_forest.{metric}" in payload.metrics
        assert f"RICH.causal_forest.{metric}_mcse" in payload.metrics
    assert "cate__scenario.csv" in payload.artifacts
    assert "RICH" in payload.artifacts["cate__scenario.csv"]


def test_an_undefined_cate_metric_is_skipped_not_logged_as_nan(loaded):
    frame, manifest, sidecar, run_dir = loaded
    null = _with_cate_rows(frame, scenario="NULL", correlations=(np.nan, np.nan))
    null.loc[null["scenario"] == "NULL", "cate_calibration_slope"] = np.nan

    payload = build_payload(null, manifest, sidecar, run_dir)

    assert "NULL.causal_forest.cate_rmse" in payload.metrics
    assert "NULL.causal_forest.cate_correlation" not in payload.metrics
    assert "NULL.causal_forest.cate_calibration_slope" not in payload.metrics
    assert all(math.isfinite(v) for v in payload.metrics.values())


def test_cate_rows_with_none_scored_log_no_cate_metric_or_artifact(loaded):
    frame, manifest, sidecar, run_dir = loaded
    unscored = _with_cate_rows(frame)
    unscored[list(CATE_TRACKED_METRICS)] = np.nan

    payload = build_payload(unscored, manifest, sidecar, run_dir)

    assert not any(key.split(".")[-1].removesuffix("_mcse") in CATE_TRACKED_METRICS
                   for key in payload.metrics)
    assert "cate__scenario.csv" not in payload.artifacts


def test_a_run_without_cate_rows_logs_no_cate_metric_or_artifact(loaded):
    payload = build_payload(*loaded)

    assert not any(key.split(".")[-1].removesuffix("_mcse") in CATE_TRACKED_METRICS
                   for key in payload.metrics)
    assert "cate__scenario.csv" not in payload.artifacts


def test_adding_cate_rows_leaves_every_existing_ate_metric_unchanged(loaded):
    frame, manifest, sidecar, run_dir = loaded
    without = build_payload(frame, manifest, sidecar, run_dir)

    with_cate = build_payload(_with_cate_rows(frame), manifest, sidecar, run_dir)

    for key, value in without.metrics.items():
        assert with_cate.metrics[key] == value, key
