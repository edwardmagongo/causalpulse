"""The one function that turns the bytes of a CSV dataset into a frame.

An upload is read in four places: when the API checks it, when the worker
analyses it, by `causalpulse analyze --data`, and when a manifest is replayed.
They have to agree, or the dataset that is analysed is not the one that was
checked, and a replay does not rebuild the dataset its manifest names. Each
of them hands `read_csv` the file's bytes and passes pandas no option of its
own. Bytes and not a path: given a path pandas decides from the file's name
whether to decompress it, so the same bytes under two names could be read two
ways.
"""

from __future__ import annotations

import io
import re

import pandas as pd

# What pandas says when a row is wider than it expects.
_TOO_WIDE = re.compile(r"Expected (\d+) fields in line (\d+), saw (\d+)")
_ROWS_PER_PIECE = 200_000


def _refusal_of_a_wide_row(exc: pd.errors.ParserError) -> ValueError | None:
    """pandas' complaint about a wide row, in this project's words, or None."""
    found = _TOO_WIDE.search(str(exc))
    if found is None:
        return None
    expected, line, saw = found.groups()
    return ValueError(
        f"line {line} of the CSV has {saw} fields and its header names {expected}, "
        "so a value would be dropped or land in the wrong column. Check for an "
        "unquoted comma inside a value, or a comma at the end of the line."
    )


def _count_fields(content: bytes, nrows: int | None) -> None:
    """Read the file with no header, in pieces that are thrown away.

    With no header the file's first line sets the number of fields, and
    pandas' own tokeniser refuses a wider row. That catches the first data
    row, which the read with a header lets through. It does not catch the
    first row of each later piece, which this read lets through in its turn;
    the read with a header refuses those. `read_csv` needs both.

    In pieces, because only the count matters and the whole file as text
    would be several times its size.
    """
    limit = None if nrows is None else nrows + 1
    pieces = pd.read_csv(
        io.BytesIO(content),
        header=None,
        dtype=str,
        na_filter=False,
        nrows=limit,
        chunksize=_ROWS_PER_PIECE,
    )
    with pieces:
        for _ in pieces:
            pass


def read_csv(content: bytes, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV the way every reader of a dataset must.

    A row with more fields than the header names is refused, whether or not
    the extra field holds anything. pandas left to itself reads such a file
    in more than one way: when the first data row is the wide one it takes
    the first column as an index and moves every other column along by one.
    Told not to (`index_col=False`), it keeps the columns in place and drops
    the extra field, with a warning in some cases and none in others. A comma
    at the end of every line reads the same to it as a stray comma inside a
    value followed by an empty last column, and nothing in the file says
    which was meant, so neither is accepted.

    The file is read twice, because neither read refuses every wide row by
    itself: see `_count_fields`. The line in the refusal is pandas' count,
    in which a quoted value that runs over several lines is one line.

    `low_memory=False`: pandas otherwise reads a large file in pieces and
    settles each piece's types separately, so a column that is numbers in one
    piece and text in another holds both.

    A row with fewer fields than the header is filled with missing values, as
    pandas does. A file pandas cannot read at all raises what pandas raises,
    for example `EmptyDataError`, `ParserError` or `UnicodeDecodeError`. The
    bytes are not decompressed, so a compressed file is one of those.
    """
    try:
        _count_fields(content, nrows)
        return pd.read_csv(io.BytesIO(content), nrows=nrows, index_col=False, low_memory=False)
    except pd.errors.ParserError as exc:
        refusal = _refusal_of_a_wide_row(exc)
        if refusal is None:
            raise
        raise refusal from exc
