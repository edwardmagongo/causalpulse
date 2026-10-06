"""The one function that turns a CSV into a frame.

An upload is read in four places: when the API checks it, when the worker
analyses it, by `causalpulse analyze --data`, and when a manifest is replayed.
They have to agree, or the dataset that is analysed is not the one that was
checked, and a replay does not rebuild the dataset its manifest names. Each
of them calls `read_csv` here and passes pandas no option of its own.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import IO

import pandas as pd

MORE_FIELDS_THAN_HEADER = (
    "the CSV has rows with more fields than its header names, so values would be "
    "dropped or land in the wrong column. Check for an unquoted comma inside a value."
)


def read_csv(source: str | Path | IO[bytes], *, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV the way every reader of an upload must.

    `index_col=False`: left to itself pandas takes the first column as an
    index when the rows have one field more than the header, which shifts
    every column by one. A comma at the end of every line, which spreadsheets
    write, is enough to do it. With the option set the columns stay under
    their own names, and pandas warns when that drops a field holding data;
    the warning is raised here as a refusal, since the alternative is an
    analysis of a file that was not read whole.

    `low_memory=False`: pandas otherwise reads a large file in pieces and
    settles each piece's types separately, so a column that is numbers in one
    piece and text in another holds both.

    A file pandas cannot read at all raises what pandas raises
    (`EmptyDataError`, `ParserError`, `UnicodeDecodeError`).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        try:
            return pd.read_csv(source, nrows=nrows, index_col=False, low_memory=False)
        except pd.errors.ParserWarning as exc:
            raise ValueError(MORE_FIELDS_THAN_HEADER) from exc
