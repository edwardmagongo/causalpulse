from cp_core.results import VALIDITY_SCORE_LIMITS, AssumptionCheck, CausalValidityScore
from cp_report.sections import assumption_table_data, validity_band_data


def _check(name="overlap", kind="identification", status="PASS", statistic=0.04,
           threshold=0.1, interpretation="Standardised mean differences are within threshold.",
           remedy=None):
    return AssumptionCheck(
        name=name, kind=kind, status=status, statistic=statistic,
        threshold=threshold, interpretation=interpretation, remedy=remedy,
    )


def _score(band="MODERATE", ceiling="STRONG"):
    return CausalValidityScore(
        band=band,
        ceiling=ceiling,
        rationale=("Design provenance is observational, so the band cannot exceed STRONG.",),
        checks=(_check(),),
    )


def test_the_assumption_table_carries_one_row_per_check():
    data = assumption_table_data((_check("overlap"), _check("balance")))

    assert data.names == ("overlap", "balance")


def test_a_not_testable_check_keeps_its_status():
    # Unconfoundedness cannot be tested from observational data. A report that
    # folded this into a pass rate would assert what the data cannot support.
    data = assumption_table_data(
        (_check("overlap", status="PASS"),
         _check("unconfoundedness", status="NOT_TESTABLE", statistic=None, threshold=None))
    )

    assert data.statuses == ("PASS", "NOT_TESTABLE")


def test_the_table_counts_not_testable_separately_from_failures():
    data = assumption_table_data(
        (_check("a", status="PASS"),
         _check("b", status="NOT_TESTABLE", statistic=None, threshold=None),
         _check("c", status="FAIL"))
    )

    assert data.n_passed == 1
    assert data.n_not_testable == 1
    assert data.n_failed == 1


def test_a_warned_check_is_counted_and_no_row_is_lost():
    # AssumptionStatus also includes WARN. A count that silently drops a
    # status is exactly the kind of defect this phase's reviews keep finding.
    data = assumption_table_data(
        (_check("a", status="PASS"),
         _check("b", status="WARN"),
         _check("c", status="FAIL"),
         _check("d", status="NOT_TESTABLE", statistic=None, threshold=None))
    )

    assert data.n_passed == 1
    assert data.n_warned == 1
    assert data.n_failed == 1
    assert data.n_not_testable == 1
    assert data.n_passed + data.n_warned + data.n_failed + data.n_not_testable == len(data.names)


def test_the_validity_band_carries_its_limits_from_the_constant():
    data = validity_band_data(_score())

    assert data.limits == VALIDITY_SCORE_LIMITS


def test_the_validity_band_carries_its_band_ceiling_and_rationale():
    data = validity_band_data(_score(band="WEAK", ceiling="MODERATE"))

    assert data.band == "WEAK"
    assert data.ceiling == "MODERATE"
    assert len(data.rationale) == 1


def test_the_assumption_tables_prose_fragments_carry_interpretations():
    data = assumption_table_data((_check(interpretation="Balance on age is adequate."),))

    assert any("Balance on age is adequate." in f for f in data.prose_fragments())


def test_the_assumption_tables_prose_fragments_carry_check_names():
    # A check named from a dataset column is exactly the interpolated text the
    # lint has never seen.
    data = assumption_table_data((_check(name="proven_uplift_balance"),))

    assert any("proven_uplift_balance" in f for f in data.prose_fragments())


def test_the_validity_bands_prose_fragments_carry_the_limits_and_rationale():
    data = validity_band_data(_score())

    joined = " ".join(data.prose_fragments())
    assert VALIDITY_SCORE_LIMITS in joined
    assert "Design provenance is observational" in joined
