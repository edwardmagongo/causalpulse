import hashlib
from pathlib import Path

import pytest
from cp_worker.execute import execute

from worker.data import scenario_d_upload


def test_execute_names_the_upload_by_its_own_hash():
    content, schema = scenario_d_upload()

    outcome = execute(content, "d.csv", schema, ["difference_in_means", "lin_regression"], 0)

    assert outcome.manifest.dataset.content_sha256 == hashlib.sha256(content).hexdigest()
    assert outcome.manifest.dataset.path == "d.csv"
    assert [r.estimate.method for r in outcome.reports] == [
        "difference_in_means", "lin_regression"
    ]


def test_an_ineligible_request_raises_the_routers_reasons_verbatim():
    content, schema = scenario_d_upload()

    with pytest.raises(ValueError, match="synthetic_control: requires panel data"):
        execute(content, "d.csv", schema, ["synthetic_control"], 0)


REAL_DATA = Path(__file__).parents[1] / "fixtures" / "real_data"
LALONDE = {
    "unit": "unit_id",
    "treatment": "treat",
    "outcome": "re78",
    "covariates": ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"],
    "outcome_type": "continuous",
}
PROP99 = {
    "unit": "state",
    "treatment": "treatment",
    "outcome": "cigsale",
    "covariates": [],
    "outcome_type": "continuous",
    "time": "year",
}
CROSS_SECTIONAL = ["difference_in_means", "lin_regression", "ipw", "aipw", "causal_forest"]


def with_first_cell(name: str, column: str, value: str) -> bytes:
    """The real CSV with `column` of its first data row written as `value`.

    The value is put in as text, the way an upload carries it, so what pandas
    makes of that text is part of what is tested.
    """
    lines = (REAL_DATA / name).read_text().split("\n")
    header, first = lines[0].split(","), lines[1].split(",")
    assert header.count(column) == 1 and len(first) == len(header)
    first[header.index(column)] = value
    return "\n".join([lines[0], ",".join(first), *lines[2:]]).encode()


@pytest.mark.parametrize("text", ["inf", "-inf", "Infinity", "1e400"])
def test_an_infinite_covariate_is_refused_once_by_column(text):
    content = with_first_cell("lalonde.csv", "re75", text)

    with pytest.raises(ValueError) as refusal:
        execute(content, "lalonde.csv", LALONDE, CROSS_SECTIONAL, 0)

    message = str(refusal.value)
    assert message.startswith("columns ['re75'] contain infinite values")
    # Before the check this was a partial analysis: one estimate, and four
    # estimators each failing with a library's own words.
    assert not any(name in message for name in CROSS_SECTIONAL)
    assert "every requested estimator failed" not in message
    assert "Input X" not in message and "array must not" not in message


def test_an_infinite_outcome_is_refused_by_column():
    content = with_first_cell("lalonde.csv", "re78", "inf")

    with pytest.raises(ValueError, match=r"columns \['re78'\] contain infinite values"):
        execute(content, "lalonde.csv", LALONDE, CROSS_SECTIONAL, 0)


def test_an_infinite_period_is_refused_where_it_used_to_be_fitted():
    # Before the check `hierarchical_did` returned an estimate from this panel.
    content = with_first_cell("prop99.csv", "year", "inf")

    with pytest.raises(ValueError, match=r"columns \['year'\] contain infinite values"):
        execute(content, "prop99.csv", PROP99, None, 0)


def test_a_huge_finite_covariate_still_gives_a_partial_analysis():
    # The dashboard's recorded partial fixture (web/scripts/capture-fixtures.mjs)
    # is made from exactly this request. If this fails, that fixture has lost
    # its source and needs another before it can be recorded again.
    content = with_first_cell("lalonde.csv", "re75", "1e200")

    outcome = execute(
        content, "lalonde-huge.csv", LALONDE, ["difference_in_means", "lin_regression"], 0
    )

    assert [report.estimate.method for report in outcome.reports] == ["difference_in_means"]
    assert [failure.method for failure in outcome.failures] == ["lin_regression"]
    assert outcome.failures[0].error


@pytest.mark.parametrize("column", ["re75", "re78"])
def test_an_infinity_written_with_a_trailing_space_is_refused(column):
    # pandas reads `inf ` as text, so the whole column is text and the finite
    # check passes over it; `float()` still makes an infinity of the cell.
    # Before the numeric-roles rule `re75` gave the partial analysis again.
    content = with_first_cell("lalonde.csv", column, "inf ")

    with pytest.raises(ValueError) as refusal:
        execute(content, "lalonde.csv", LALONDE, ["difference_in_means", "lin_regression"], 0)

    message = str(refusal.value)
    assert message.startswith(f"columns ['{column}'] must hold numbers")
    assert f"({column}: str)" in message
    assert "every requested estimator failed" not in message


def test_digit_grouping_in_a_covariate_is_refused_where_it_used_to_be_analysed():
    # `float("1_000")` is 1000.0, so this text column was fitted as numbers.
    content = with_first_cell("lalonde.csv", "re75", "1_000")

    with pytest.raises(ValueError, match=r"columns \['re75'\] must hold numbers"):
        execute(content, "lalonde.csv", LALONDE, ["difference_in_means", "lin_regression"], 0)


def test_stray_text_in_a_covariate_is_refused_once_by_column():
    content = with_first_cell("lalonde.csv", "re75", "abc")

    with pytest.raises(ValueError) as refusal:
        execute(content, "lalonde.csv", LALONDE, ["difference_in_means", "lin_regression"], 0)

    message = str(refusal.value)
    assert message.startswith("columns ['re75'] must hold numbers")
    assert "every requested estimator failed" not in message
    assert "could not convert" not in message


def test_a_text_unit_column_is_still_analysed():
    content = with_first_cell("lalonde.csv", "unit_id", "abc")

    outcome = execute(content, "lalonde.csv", LALONDE, ["difference_in_means"], 0)

    assert [report.estimate.method for report in outcome.reports] == ["difference_in_means"]


def lines_of(name: str) -> list[str]:
    return [line for line in (REAL_DATA / name).read_text().split("\n") if line]


@pytest.mark.parametrize("extra", [",", ",9"], ids=["a comma ending every line", "a field of data"])
def test_a_row_wider_than_the_header_is_refused_by_the_reader(extra):
    # Left to itself pandas takes the first column of such a file as an index
    # and shifts every other column by one. LaLonde was refused then only by
    # luck: its treatment column came to hold ages.
    rows = lines_of("lalonde.csv")
    content = "\n".join([rows[0], *(row + extra for row in rows[1:])]).encode()

    with pytest.raises(ValueError, match="line 2 of the CSV has 13 fields and its header names 12"):
        execute(content, "lalonde.csv", LALONDE, ["difference_in_means"], 0)


def test_a_stray_comma_before_an_empty_last_column_is_refused():
    # Every earnings value written with a decimal comma and an empty `notes`
    # column: pandas, told to keep the columns in place, read `re78` as its
    # whole part and put the rest under `notes`, and an estimate came back.
    rows = lines_of("lalonde.csv")
    at = rows[0].split(",").index("re78")
    rewritten = [rows[0] + ",notes"]
    for row in rows[1:]:
        cells = row.split(",")
        cells[at] = cells[at].replace(".", ",") if "." in cells[at] else cells[at] + ",0"
        rewritten.append(",".join(cells) + ",")

    with pytest.raises(ValueError, match="line 2 of the CSV has 14 fields and its header names 13"):
        execute("\n".join(rewritten).encode(), "lalonde.csv", LALONDE, ["difference_in_means"], 0)


def test_a_unit_pandas_reads_as_missing_is_refused_by_name():
    # Before the unit rule this panel crashed while it was being routed, on
    # a comparison of text with nan.
    content = (REAL_DATA / "prop99.csv").read_text().replace("Alabama", "NA").encode()

    with pytest.raises(ValueError, match="unit column 'state' contains missing values"):
        execute(content, "prop99.csv", PROP99, None, 0)


def test_a_fractional_period_is_refused_by_name():
    # Before the time rule the cohort was truncated to a period not in the data.
    rows = lines_of("prop99.csv")
    at = rows[0].split(",").index("year")
    shifted = [rows[0]]
    for row in rows[1:]:
        cells = row.split(",")
        cells[at] = f"{cells[at]}.5"
        shifted.append(",".join(cells))

    # One quick estimator is asked for, so that the run with the rule switched
    # off answers in seconds and not after the Bayesian fits.
    with pytest.raises(ValueError, match="time column 'year' must hold whole-numbered periods"):
        execute("\n".join(shifted).encode(), "prop99.csv", PROP99, ["synthetic_control"], 0)


def test_a_missing_covariate_is_refused_once_through_the_pipeline():
    content = with_first_cell("lalonde.csv", "re75", "")

    with pytest.raises(ValueError) as refusal:
        execute(content, "lalonde.csv", LALONDE, ["difference_in_means", "lin_regression"], 0)

    message = str(refusal.value)
    assert message.startswith("columns ['re75'] contain missing values")
    assert "every requested estimator failed" not in message
