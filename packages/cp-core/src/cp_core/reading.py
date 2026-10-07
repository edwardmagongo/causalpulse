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

# What pandas says when a row has more fields than the row before it.
_TOO_WIDE = re.compile(r"Expected (\d+) fields in line (\d+), saw (\d+)")


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


def read_csv(content: bytes, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV the way every reader of a dataset must.

    A row with more fields than the header names is refused, whether or not
    the extra field holds anything. A comma at the end of every line reads
    the same as a stray comma inside a value followed by an empty last
    column, and nothing in the file says which was meant, so neither is
    accepted.

    pandas' tokeniser compares each row with the one before it, except the
    first row after the header, which it lets through at any width; what
    pandas then makes of a wide first row depends on its options (an index
    taken from the leading columns, or the extra fields dropped). So the
    first two rows are read by themselves with no header, where the header
    is the row before and the first data row is compared with it; and then
    the whole file is read, where every later row is compared with a row
    already known to be no wider than the header. Both reads are made in one
    piece (`low_memory=False`): read in pieces, the first row of each piece
    is let through as well.

    That option also keeps a column's type from being settled piece by
    piece, which left a column that was numbers in one piece and text in
    another holding both.

    The line in the refusal is pandas' count, in which a quoted value that
    runs over several lines is one line. With `nrows=0` no row is looked at.

    A row with fewer fields than the header is filled with missing values, as
    pandas does. A file pandas cannot read at all raises what pandas raises,
    for example `EmptyDataError`, `ParserError` or `UnicodeDecodeError`. The
    bytes are not decompressed, and the compressed files tried were among
    those.
    """
    try:
        if nrows != 0:
            pd.read_csv(
                io.BytesIO(content),
                header=None,
                nrows=2,
                dtype=str,
                na_filter=False,
                low_memory=False,
            )
        return pd.read_csv(io.BytesIO(content), nrows=nrows, index_col=False, low_memory=False)
    except pd.errors.ParserError as exc:
        refusal = _refusal_of_a_wide_row(exc)
        if refusal is None:
            raise
        raise refusal from exc
