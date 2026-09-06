import numpy as np
import pandas as pd
import pytest
from cp_cli.io import (
    ESTIMATE_COLUMNS,
    estimates_frame,
    read_benchmark,
    write_benchmark,
    write_estimates,
)
from cp_core.results import EffectEstimate, FitMetadata
from cp_core.values import Interval
from cp_evaluation.run import ROW_COLUMNS


def _estimate(method="difference_in_means", se=0.1, bootstrap=True):
    return EffectEstimate(
        estimand="ATE",
        method=method,
        point=0.25,
        se=se,
        ci=Interval(lower=0.05, upper=0.45, level=0.95),
        ci_basis="analytic",
        bootstrap_ci=Interval(lower=0.06, upper=0.44, level=0.95) if bootstrap else None,
        p_value=0.01,
        n_treated=100,
        n_control=100,
        scale="percentage_point",
        diagnostics={},
        assumptions=(),
        counterfactual=None,
        fit_metadata=FitMetadata.capture(
            estimator=method, config_hash="h", runtime_seconds=0.5, random_seed=0
        ),
        subgroup_findings=(),
    )


def test_estimate_columns_reuse_the_benchmark_vocabulary():
    # Two phases will read both files. A person moving between an analysis and
    # a benchmark should not have to learn two names for the same quantity.
    shared = set(ESTIMATE_COLUMNS) & set(ROW_COLUMNS)
    assert {"method", "estimand", "point", "se", "ci_lower", "ci_upper", "ci_basis"} <= shared


def test_estimates_frame_has_one_row_per_estimate():
    frame = estimates_frame([_estimate("ipw"), _estimate("aipw")])

    assert list(frame.columns) == list(ESTIMATE_COLUMNS)
    assert list(frame.method) == ["ipw", "aipw"]
    assert frame.point.tolist() == [0.25, 0.25]


def test_a_missing_standard_error_becomes_nan_not_a_string():
    frame = estimates_frame([_estimate(se=None, bootstrap=False)])

    assert pd.isna(frame.se.iloc[0])
    assert pd.isna(frame.bootstrap_ci_lower.iloc[0])


def test_write_estimates_round_trips_through_csv(tmp_path):
    path = tmp_path / "estimates.csv"
    write_estimates([_estimate("ipw")], path)

    read_back = pd.read_csv(path)

    assert list(read_back.columns) == list(ESTIMATE_COLUMNS)
    assert read_back.point.iloc[0] == pytest.approx(0.25)


def _benchmark_frame():
    rows = []
    for scenario, point in (("A", 0.1), ("B", 0.2), ("A", 0.3)):
        row = dict.fromkeys(ROW_COLUMNS, np.nan)
        row.update(
            {
                "shape": "cross_section",
                "scenario": scenario,
                "n_units": 500,
                "seed": 0,
                "method": "difference_in_means",
                "status": "fitted",
                "reason": "",
                "estimand": "ATE",
                "truth": 0.05,
                "point": point,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=list(ROW_COLUMNS))


def test_benchmark_round_trips_through_parquet(tmp_path):
    frame = _benchmark_frame()
    write_benchmark(frame, tmp_path, {"tier": "ci"})

    read_back, sidecar = read_benchmark(tmp_path)

    assert sidecar["tier"] == "ci"
    assert len(read_back) == len(frame)
    assert set(read_back.columns) == set(frame.columns)


def test_the_partition_column_survives_as_a_plain_string(tmp_path):
    # partition_cols writes a directory tree and pyarrow reads the partition
    # key back as a categorical by default. Anything grouping on `scenario`
    # downstream would then behave differently than on the in-memory frame.
    write_benchmark(_benchmark_frame(), tmp_path, {})

    read_back, _ = read_benchmark(tmp_path)

    assert read_back.scenario.dtype == object
    assert sorted(read_back.scenario.unique()) == ["A", "B"]


def test_row_order_is_restored_rather_than_left_to_the_partition_layout(tmp_path):
    # Reading a partitioned dataset returns rows grouped by partition, not in
    # the order they were written. A caller comparing two frames would see a
    # spurious difference.
    frame = _benchmark_frame()
    write_benchmark(frame, tmp_path, {})

    read_back, _ = read_benchmark(tmp_path)

    key = ["shape", "scenario", "n_units", "seed", "method", "point"]
    pd.testing.assert_frame_equal(
        read_back.sort_values(key).reset_index(drop=True)[key],
        frame.sort_values(key).reset_index(drop=True)[key],
        check_dtype=False,
    )


def test_reading_a_directory_with_no_run_json_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="run.json"):
        read_benchmark(tmp_path)
