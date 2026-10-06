"""Every reader of an upload goes through `cp_core.reading.read_csv`."""

import io
from pathlib import Path

import pandas as pd
import pytest
from cp_core.reading import MORE_FIELDS_THAN_HEADER, read_csv

REAL_DATA = Path(__file__).parents[1] / "fixtures" / "real_data"


def read(text: str, **kwargs) -> pd.DataFrame:
    return read_csv(io.BytesIO(text.encode()), **kwargs)


@pytest.mark.parametrize("name", ["lalonde.csv", "prop99.csv"])
def test_a_well_formed_file_is_read_exactly_as_pandas_reads_it(name):
    # A manifest written before this reader existed has to replay to the
    # same dataset, so a well-formed file must not change by a dtype.
    ours, pandas_own = read_csv(REAL_DATA / name), pd.read_csv(REAL_DATA / name)
    pd.testing.assert_frame_equal(ours, pandas_own, check_exact=True)


def test_a_comma_at_the_end_of_every_line_does_not_shift_the_columns():
    assert pd.read_csv(io.StringIO("a,b,c\n1,2,3,\n5,6,7,\n"))["a"].tolist() == [2, 6]
    frame = read("a,b,c\n1,2,3,\n5,6,7,\n")
    assert list(frame.columns) == ["a", "b", "c"]
    assert frame.to_numpy().tolist() == [[1, 2, 3], [5, 6, 7]]


@pytest.mark.parametrize(
    "text",
    ["a,b,c\n1,2,3,4\n5,6,7,8\n", "a,b,c\n1,2,3,4\n5,6,7\n", "a,b,c\n1,2,3,\n5,6,7,9\n"],
    ids=["every row", "the first row", "one row among trailing commas"],
)
def test_an_extra_field_holding_data_is_refused(text):
    with pytest.raises(ValueError) as refusal:
        read(text)
    assert str(refusal.value) == MORE_FIELDS_THAN_HEADER


def test_a_later_row_with_an_extra_field_is_still_pandas_own_error():
    with pytest.raises(pd.errors.ParserError, match="Expected 3 fields in line 3, saw 4"):
        read("a,b,c\n1,2,3\n5,6,7,8\n")


def test_an_empty_file_is_still_pandas_own_error():
    with pytest.raises(pd.errors.EmptyDataError):
        read("")


def test_a_row_with_too_few_fields_is_filled_with_a_missing_value():
    frame = read("a,b,c\n1,2\n5,6,7\n")
    assert frame["c"].isna().tolist() == [True, False]


def test_a_quoted_comma_is_a_value_and_not_a_field():
    assert read('a,b\n1,"x,y"\n')["b"].tolist() == ["x,y"]


def test_nrows_zero_gives_the_header_alone():
    frame = read("a,b\n1,2\n", nrows=0)
    assert list(frame.columns) == ["a", "b"]
    assert frame.empty


def test_a_large_column_of_numbers_and_text_is_read_as_text_throughout():
    # Left to itself pandas reads a file this size in pieces, and gives a
    # column that is numbers in one piece and text in another both types.
    rows = 300_000
    text = "unit,y\n" + "".join(f"{i},1.5\n" for i in range(rows)) + "abc,1.5\n"
    with pytest.warns(pd.errors.DtypeWarning):
        pandas_own = pd.read_csv(io.StringIO(text))
    assert {type(v) for v in pandas_own["unit"]} == {int, str}
    assert {type(v) for v in read(text)["unit"]} == {str}
