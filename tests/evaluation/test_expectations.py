import pandas as pd
from cp_evaluation.expectations import MIN_N_USED, evaluate_expectations, expectations_markdown
from cp_evaluation.store import RunStatus

from evaluation._study_frames import frame, rows

COMPLETE = RunStatus(expected=10, done=10, orphans=0, by_scenario={})


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
        rows("callaway_santanna", "P4", 100, bias=0.02, panel=True),
        rows("hierarchical_did", "P1", 100, hit_fraction=0.95, panel=True, runtime=25.0),
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
        rows("callaway_santanna", "P4", 100, bias=0.5, panel=True),
    )

    assert _verdicts(data)["E4"] == "failed"


def test_e5_fails_only_when_coverage_is_clearly_below_ninety_percent():
    low = rows("hierarchical_did", "P1", 200, hit_fraction=0.70, panel=True)
    borderline = rows("hierarchical_did", "P1", 200, hit_fraction=0.89, panel=True)

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
        rows("hierarchical_did", "P1", 50, panel=True, runtime=2.0),
        rows("twfe", "P1", 50, panel=True, runtime=30.0),
    )

    assert _verdicts(data)["E7"] == "failed"


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
