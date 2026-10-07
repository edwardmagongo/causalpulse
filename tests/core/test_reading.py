"""Every reader of a dataset goes through `cp_core.reading.read_csv`."""

import io
from pathlib import Path

import pandas as pd
import pytest
from cp_core.reading import read_csv

REAL_DATA = Path(__file__).parents[1] / "fixtures" / "real_data"


def read(text: str, **kwargs) -> pd.DataFrame:
    return read_csv(text.encode(), **kwargs)


def pandas_own(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


@pytest.mark.parametrize("name", ["lalonde.csv", "prop99.csv"])
def test_the_real_datasets_are_read_exactly_as_pandas_reads_them(name):
    # A manifest written before this reader existed has to replay to the
    # same dataset, so these files must not change by a dtype.
    content = (REAL_DATA / name).read_bytes()
    pd.testing.assert_frame_equal(
        read_csv(content), pd.read_csv(io.BytesIO(content)), check_exact=True
    )


@pytest.mark.parametrize(
    "text",
    [
        "a,b\n1,2\n3,4\n",
        'a,b\n1,"x,y"\n',
        'a,b\n1,"x\ny"\n2,z\n',
        "a,b\n\n1,2\n\n3,4\n",
        "\na,b\n1,2\n",
        "a,b\r\n1,2\r\n",
        "﻿a,b\n1,2\n",
        "a,b\n",
        "a\n1\n2\n",
        "a,a,b\n1,2,3\n",
        "a,,c\n1,2,3\n",
        "a,b,c\n1,2\n5,6,7\n",
        "a,b,c,\n1,2,3,\n5,6,7,\n",
    ],
    ids=[
        "plain", "a quoted comma", "a quoted line break", "blank lines", "a blank first line",
        "CRLF", "a byte-order mark", "a header alone", "one column", "a repeated name",
        "an empty name", "a short row", "a comma ending the header too",
    ],
)
def test_a_file_with_no_row_wider_than_its_header_is_read_as_pandas_reads_it(text):
    pd.testing.assert_frame_equal(read(text), pandas_own(text), check_exact=True)


@pytest.mark.parametrize(
    ("text", "line", "saw", "names"),
    [
        ("a,b,c\n1,2,3,4\n5,6,7,8\n", 2, 4, 3),
        ("a,b,c\n1,2,3,4\n5,6,7\n", 2, 4, 3),
        ("a,b,c\n1,2,3\n5,6,7,8\n", 3, 4, 3),
        ("a,b,c\n1,2,3,\n5,6,7,\n", 2, 4, 3),
        ("unit,treat,y,comment\n1,0,3,5,\n", 2, 5, 4),
        ("a,b,c\n1,2,3,,\n", 2, 5, 3),
        ("a,b\nr1,1,2\nr2,3,4\n", 2, 3, 2),
        ("a\n1,\n", 2, 2, 1),
        ("a;b;c\n1,5;2,5;3\n", 2, 3, 1),
        ("\na,b\n1,2,3\n", 3, 3, 2),
        ("a,b\r\n1,2,3\r\n", 2, 3, 2),
    ],
    ids=[
        "data on every row", "data on the first row", "data on a later row",
        "a comma ending every line", "a stray comma before an empty last column",
        "two commas ending a line", "row names with no header for them", "one column",
        "semicolons with decimal commas", "after a blank first line", "CRLF",
    ],
)
def test_a_row_wider_than_the_header_is_refused_naming_the_line(text, line, saw, names):
    with pytest.raises(ValueError) as refusal:
        read(text)
    assert str(refusal.value) == (
        f"line {line} of the CSV has {saw} fields and its header names {names}, so a value "
        "would be dropped or land in the wrong column. Check for an unquoted comma inside "
        "a value, or a comma at the end of the line."
    )


def test_the_two_readings_of_a_stray_comma_cannot_be_told_apart_which_is_why_both_are_refused():
    # `y` is 3 and `comment` is 5, with a comma ending the line; or `y` is the
    # decimal 3,5 and `comment` is empty. pandas, told to keep the columns in
    # place, takes the first reading and says nothing.
    text = "unit,treat,y,comment\n1,0,3,5,\n"
    assumed = pd.read_csv(io.StringIO(text), index_col=False)
    assert assumed[["y", "comment"]].to_numpy().tolist() == [[3, 5]]
    with pytest.raises(ValueError, match="line 2 of the CSV has 5 fields"):
        read(text)


def test_an_empty_file_is_still_pandas_own_error():
    with pytest.raises(pd.errors.EmptyDataError):
        read("")


def test_a_file_pandas_cannot_tokenise_is_still_pandas_own_error():
    with pytest.raises(pd.errors.ParserError, match="EOF inside string"):
        read('a,b\n"1,2\n')


def test_nrows_zero_gives_the_header_alone_and_refuses_no_row():
    frame = read("a,b\n1,2,3\n", nrows=0)
    assert list(frame.columns) == ["a", "b"]
    assert frame.empty


@pytest.mark.parametrize("wide_row", [1, 2, 3])
@pytest.mark.parametrize("nrows", [1, 2, 3, None])
def test_nrows_refuses_a_wide_row_it_reads_and_no_other(nrows, wide_row):
    text = table(2, 4, {wide_row: "1,1,1"})
    if nrows is None or wide_row <= nrows:
        with pytest.raises(ValueError, match=f"line {wide_row + 1} of the CSV has 3 fields"):
            read(text, nrows=nrows)
    else:
        assert read(text, nrows=nrows).shape == (nrows, 2)


def test_the_bytes_are_read_as_they_are_whatever_the_file_was_called(tmp_path):
    # Given a path, pandas decides from the name whether to decompress.
    path = tmp_path / "plain.csv.gz"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(Exception, match="gzip"):
        pd.read_csv(path)
    assert read_csv(path.read_bytes()).to_numpy().tolist() == [[1, 2]]


def test_a_large_column_of_numbers_and_text_is_read_as_text_throughout():
    # Left to itself pandas reads a file this size in pieces, and gives a
    # column that is numbers in one piece and text in another both types.
    rows = 300_000
    text = "unit,y\n" + "".join(f"{i},1.5\n" for i in range(rows)) + "abc,1.5\n"
    with pytest.warns(pd.errors.DtypeWarning):
        mixed = pandas_own(text)
    assert {type(v) for v in mixed["unit"]} == {int, str}
    assert {type(v) for v in read(text)["unit"]} == {str}


def table(width: int, rows: int, changed: dict[int, str]) -> str:
    """A CSV of `width` columns and `rows` data rows of ones, some rows replaced."""
    header = ",".join(f"c{i}" for i in range(width))
    plain = ",".join(["1"] * width)
    body = [changed.get(row, plain) for row in range(1, rows + 1)]
    return header + "\n" + "\n".join(body) + "\n"


# Where pandas, reading in pieces, starts a new piece: every 2**20 // width
# rows (rounded down to a power of two) when left to choose, and wherever a
# caller's `chunksize` puts it. A reader that read in pieces let the first row
# of each through at any width, and took a short one for the new width.
SHAPES = [(2, 450_000), (5, 300_000), (10, 150_000), (100, 20_000)]
ROWS = [1, 2, 3, 8_192, 8_193, 65_536, 65_537, 131_072, 131_073, 200_000, 200_001, 400_000]
PLACES = [(width, rows, row) for width, rows in SHAPES for row in (*ROWS, rows) if row <= rows]


@pytest.mark.parametrize("extra", [",", ",9", ",,"], ids=["empty", "data", "two empty"])
@pytest.mark.parametrize(("width", "rows", "row"), PLACES)
def test_a_wide_row_is_refused_wherever_it_falls(width, rows, row, extra):
    wide = ",".join(["1"] * width) + extra
    with pytest.raises(ValueError) as refusal:
        read(table(width, rows, {row: wide}))
    assert str(refusal.value).startswith(
        f"line {row + 1} of the CSV has {width + extra.count(',')} fields "
        f"and its header names {width}, "
    )


@pytest.mark.parametrize(("width", "rows", "row"), PLACES)
def test_a_short_row_is_filled_and_not_refused_wherever_it_falls(width, rows, row):
    short = ",".join(["1"] * (width - 1))
    frame = read(table(width, rows, {row: short}))
    assert frame.shape == (rows, width)
    assert frame.isna().to_numpy().sum() == 1
    assert pd.isna(frame.iloc[row - 1, width - 1])


def test_the_first_of_two_wide_rows_is_the_one_named_and_against_the_header():
    # The tokeniser compares a row with the one before it, so a reader that
    # let a wide row through named the next one, "against" the wide one.
    text = table(2, 200_010, {200_000: "1,1,1", 200_001: "1,1,1,1"})
    with pytest.raises(ValueError) as refusal:
        read(text)
    assert str(refusal.value).startswith(
        "line 200001 of the CSV has 3 fields and its header names 2, "
    )


def test_a_wide_first_row_is_refused_under_a_header_too_wide_to_read_in_one_piece():
    # At this width pandas, left to choose, reads one row to a piece.
    width = 400_000
    with pytest.raises(ValueError, match=f"line 2 of the CSV has {width + 1} fields"):
        read(table(width, 2, {1: ",".join(["1"] * width) + ","}))


def test_the_line_named_is_pandas_count_in_which_a_quoted_line_break_is_no_new_line():
    # Recorded so the message's "line" is not taken for an editor's: the wide
    # row is on the fourth line of this file.
    with pytest.raises(ValueError, match="line 3 of the CSV has 3 fields"):
        read('a,b\n1,"x\ny"\n2,3,4\n')
