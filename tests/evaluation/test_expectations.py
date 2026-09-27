import numpy as np
import pandas as pd
from cp_evaluation.expectations import MIN_N_USED, evaluate_expectations, expectations_markdown
from cp_evaluation.run import ROW_COLUMNS
from cp_evaluation.store import RunStatus

from evaluation._study_frames import frame, rows

COMPLETE = RunStatus(expected=10, done=10, orphans=0, by_scenario={})


def _with_cate_rows(frame: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    cate_rows = []
    for seed in range(n):
        row = dict.fromkeys(ROW_COLUMNS, np.nan)
        row.update(shape="cate", scenario="RICH", n_units=2000, seed=seed,
                   method="causal_forest", status="fitted", reason="", estimand="ATE",
                   truth=0.1, point=0.1, runtime_seconds=1e6, cate_rmse=0.05)
        cate_rows.append(row)
    return pd.concat([frame, pd.DataFrame(cate_rows, columns=list(ROW_COLUMNS))], ignore_index=True)


def _verdicts(data: pd.DataFrame, status: RunStatus = COMPLETE) -> dict[str, str]:
    return {r.id: r.verdict for r in evaluate_expectations(data, status)}


FOUR = ("difference_in_means", "lin_regression", "ipw", "aipw")


def _passing() -> pd.DataFrame:
    """A frame on which every expectation E1 to E7 is met."""
    parts = [
        *(rows(m, "B", 100, truth=0.0, hit_fraction=0.95) for m in FOUR),
        rows("difference_in_means", "A", 100, hit_fraction=0.95),
        rows("difference_in_means", "D", 100, bias=0.5),
        rows("aipw", "D", 100, bias=0.02),
        rows("twfe", "P4", 100, bias=0.5, panel=True),
        rows("callaway_santanna", "P4", 100, bias=0.02, panel=True, ci_basis="bootstrap"),
        rows(
            "hierarchical_did",
            "P1",
            100,
            hit_fraction=0.95,
            panel=True,
            runtime=25.0,
            ci_basis="posterior",
        ),
        rows("difference_in_means", "H", 20, status="refused"),
        rows("difference_in_means", "H", 20, status="ineligible"),
    ]
    return frame(*parts)


def test_every_expectation_is_met_on_a_frame_built_to_meet_them():
    assert _verdicts(_passing()) == {f"E{i}": "met" for i in range(1, 9)}


def test_e1_fails_when_the_false_positive_rate_is_far_from_a_nominal_five_percent():
    data = frame(_passing(), rows("aipw", "B", 100, truth=0.0, hit_fraction=0.70))

    assert _verdicts(data)["E1"] == "failed"


def test_e1_is_inconclusive_below_the_minimum_number_of_rows():
    data = frame(*(rows(m, "B", MIN_N_USED - 1, truth=0.0, hit_fraction=0.95) for m in FOUR))

    assert _verdicts(data)["E1"] == "inconclusive"


def test_e2_fails_when_analytic_coverage_is_far_from_nominal():
    data = frame(rows("difference_in_means", "A", 100, hit_fraction=0.70))

    assert _verdicts(data)["E2"] == "failed"


def test_e3_fails_when_the_naive_estimator_is_not_the_more_biased_one():
    data = frame(
        rows("difference_in_means", "D", 100, bias=0.02),
        rows("aipw", "D", 100, bias=0.5),
    )

    assert _verdicts(data)["E3"] == "failed"


def test_e3_fails_when_the_naive_estimators_bias_is_within_noise():
    data = frame(
        rows("difference_in_means", "D", 100, bias=0.0),
        rows("aipw", "D", 100, bias=-0.0),
    )

    assert _verdicts(data)["E3"] == "failed"


def test_e3_fails_when_the_naive_estimators_larger_bias_is_within_noise():
    # 0.05 is larger than 0, but with an error spread of 1.0 over 100 rows its
    # MCSE is about 0.1, so it is not distinguishable from no bias at all.
    data = frame(
        rows("difference_in_means", "D", 100, bias=0.05, noise=1.0),
        rows("aipw", "D", 100, bias=0.0, noise=1.0),
    )

    assert _verdicts(data)["E3"] == "failed"


def test_e4_fails_when_twfe_is_not_the_more_biased_estimator():
    data = frame(
        rows("twfe", "P4", 100, bias=0.02, panel=True),
        rows("callaway_santanna", "P4", 100, bias=0.5, panel=True, ci_basis="bootstrap"),
    )

    assert _verdicts(data)["E4"] == "failed"


def test_e5_fails_only_when_coverage_is_clearly_below_ninety_percent():
    low = rows(
        "hierarchical_did",
        "P1",
        200,
        hit_fraction=0.70,
        panel=True,
        ci_basis="posterior",
    )
    borderline = rows(
        "hierarchical_did",
        "P1",
        200,
        hit_fraction=0.89,
        panel=True,
        ci_basis="posterior",
    )

    assert _verdicts(low)["E5"] == "failed"
    # 0.89 sits within 2 MCSE of 0.90 at 200 rows, so it is not a failure.
    assert _verdicts(borderline)["E5"] == "met"


def test_e6_fails_when_any_method_fitted_a_row_on_scenario_h():
    data = frame(_passing(), rows("aipw", "H", 5))

    assert _verdicts(data)["E6"] == "failed"


def test_e6_is_inconclusive_when_scenario_h_is_absent():
    data = _passing()

    assert _verdicts(data[data["scenario"] != "H"])["E6"] == "inconclusive"


def test_e7_fails_when_another_estimator_is_slower_than_hierarchical_did():
    data = frame(
        rows("hierarchical_did", "P1", 50, panel=True, runtime=2.0, ci_basis="posterior"),
        rows("twfe", "P1", 50, panel=True, runtime=30.0),
    )

    assert _verdicts(data)["E7"] == "failed"


def test_e5_ignores_donor_pool_rows_of_hierarchical_did():
    # coverage_posterior filters on ci_basis alone, so pooled over these donor rows too,
    # coverage would be 0.5 over 400 rows and E5 would fail.
    panel = rows("hierarchical_did", "P1", 200, hit_fraction=0.95, panel=True, ci_basis="posterior")
    donor = rows(
        "hierarchical_did", "S1", 200, hit_fraction=0.05, donor_pool=True, ci_basis="posterior"
    )

    alone = {r.id: r for r in evaluate_expectations(panel, COMPLETE)}["E5"]
    with_donor = {r.id: r for r in evaluate_expectations(frame(panel, donor), COMPLETE)}["E5"]

    assert alone.verdict == with_donor.verdict == "met"
    assert with_donor.measurements == alone.measurements
    assert with_donor.measurements[0].n_used == 200
    assert "coverage_posterior" in with_donor.measurements[0].label


def test_e5_is_inconclusive_when_hierarchical_did_has_only_donor_pool_rows():
    # Donor rows at 5% posterior coverage: pooled, they would fail E5 outright, so
    # inconclusive here means they were excluded, not that the column was empty.
    donor = rows(
        "hierarchical_did", "S1", 200, hit_fraction=0.05, donor_pool=True, ci_basis="posterior"
    )

    assert _verdicts(donor)["E5"] == "inconclusive"


def test_e5_reads_posterior_coverage_not_analytic():
    # hierarchical_did's interval is a posterior credible interval; a row mislabelled
    # analytic is not in coverage_posterior's population, so E5 has nothing to score.
    mislabelled = rows("hierarchical_did", "P1", 200, hit_fraction=0.95, panel=True)

    assert _verdicts(mislabelled)["E5"] == "inconclusive"


def test_e7_ranking_is_unchanged_by_donor_pool_rows():
    # A slow donor-pool bsts or synthetic_control fit would otherwise take rank 1 from
    # hierarchical_did; the ten-estimator ranking is over panel and cross-section rows only.
    base = frame(
        rows("hierarchical_did", "P1", 50, panel=True, runtime=25.0, ci_basis="posterior"),
        rows("twfe", "P1", 50, panel=True, runtime=2.0),
    )
    donor = frame(
        rows("bsts", "S1", 50, donor_pool=True, runtime=200.0, ci_basis="posterior"),
        rows(
            "synthetic_control", "S1", 50, donor_pool=True, runtime=90.0, ci_basis="permutation"
        ),
        rows("hierarchical_did", "S1", 50, donor_pool=True, runtime=1.0, ci_basis="posterior"),
    )

    with_donor = {r.id: r for r in evaluate_expectations(frame(base, donor), COMPLETE)}
    without = {r.id: r for r in evaluate_expectations(base, COMPLETE)}

    assert with_donor["E7"].verdict == without["E7"].verdict == "met"
    assert with_donor["E7"].measurements == without["E7"].measurements


def test_e7_is_inconclusive_on_a_donor_pool_only_frame():
    donor = frame(
        rows("bsts", "S1", 50, donor_pool=True, runtime=200.0, ci_basis="posterior"),
        rows("hierarchical_did", "S1", 50, donor_pool=True, runtime=1.0, ci_basis="posterior"),
    )

    assert _verdicts(donor)["E7"] == "inconclusive"


def test_e5_and_e7_notes_name_the_excluded_donor_pool_population():
    base = frame(
        rows("hierarchical_did", "P1", 50, panel=True, runtime=25.0, ci_basis="posterior"),
        rows("twfe", "P1", 50, panel=True, runtime=2.0),
    )
    donor = frame(
        rows("bsts", "S1", 30, donor_pool=True, ci_basis="posterior"),
        rows("hierarchical_did", "S1", 40, donor_pool=True, ci_basis="posterior"),
    )

    results = {r.id: r for r in evaluate_expectations(frame(base, donor), COMPLETE)}

    assert results["E5"].note == (
        "Pooled over panel rows only; 40 donor-pool hierarchical_did rows excluded."
    )
    assert results["E7"].note == (
        "Ranked over non-donor rows; 70 donor-pool rows excluded, "
        "fitted methods: bsts, hierarchical_did."
    )
    assert results["E5"].note in expectations_markdown(list(results.values()))


def test_e5_and_e7_are_identical_to_the_no_donor_output_when_the_frame_has_no_donor_rows():
    base = frame(
        rows("hierarchical_did", "P1", 50, panel=True, runtime=25.0, ci_basis="posterior"),
        rows("twfe", "P1", 50, panel=True, runtime=2.0),
    )

    results = {r.id: r for r in evaluate_expectations(base, COMPLETE)}

    for id_ in ("E5", "E7"):
        assert results[id_].note == ""
    assert results["E5"].rule == "failed only if pooled coverage_posterior + 2 MCSE < 0.90"
    assert results["E7"].rule == "rank 1 by mean runtime_seconds over each method's fitted rows"
    assert results["E5"].verdict == results["E7"].verdict == "met"


def test_e8_fails_on_remaining_cells_or_orphans():
    remaining = RunStatus(expected=10, done=9, orphans=0, by_scenario={})
    orphaned = RunStatus(expected=10, done=10, orphans=1, by_scenario={})

    assert _verdicts(_passing(), remaining)["E8"] == "failed"
    assert _verdicts(_passing(), orphaned)["E8"] == "failed"


def test_the_expectations_document_lists_each_verdict_and_its_measurements():
    text = expectations_markdown(evaluate_expectations(_passing(), COMPLETE))

    assert "8 met, 0 failed, 0 inconclusive." in text
    assert "## E1 - MET" in text
    assert "| measurement | value | MCSE | n |" in text


def test_e7_note_lists_only_methods_with_fitted_donor_rows():
    base = frame(
        rows("hierarchical_did", "P1", 50, panel=True, runtime=25.0, ci_basis="posterior"),
        rows("twfe", "P1", 50, panel=True, runtime=2.0),
    )
    donor = frame(
        rows("bsts", "S1", 10, donor_pool=True, ci_basis="posterior"),
        rows("twfe", "S5", 7, donor_pool=True, status="ineligible"),
        rows("aipw", "S5", 3, donor_pool=True, status="ineligible"),
    )

    results = {r.id: r for r in evaluate_expectations(frame(base, donor), COMPLETE)}

    assert results["E7"].note == (
        "Ranked over non-donor rows; 20 donor-pool rows excluded, fitted methods: bsts."
    )


def test_e7_is_unchanged_by_cate_arm_rows_with_a_huge_runtime():
    base = frame(
        rows("hierarchical_did", "P1", 50, panel=True, runtime=25.0, ci_basis="posterior"),
        rows("twfe", "P1", 50, panel=True, runtime=2.0),
    )

    with_cate = {r.id: r for r in evaluate_expectations(_with_cate_rows(base), COMPLETE)}
    without = {r.id: r for r in evaluate_expectations(base, COMPLETE)}

    assert with_cate["E7"].verdict == without["E7"].verdict
    assert with_cate["E7"].measurements == without["E7"].measurements
    assert with_cate["E7"].note.endswith("3 CATE-arm rows excluded.")
