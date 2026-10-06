"""The one function that turns a CSV into a frame.

An upload is read in four places: when the API checks it, when the worker
analyses it, by `causalpulse analyze --data`, and when a manifest is replayed.
They have to agree, or the dataset that is analysed is not the one that was
checked, and a replay does not rebuild the dataset its manifest names. Each
of them calls `read_csv` here and passes pandas no option of its own.
"""

from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import IO

import pandas as pd

MORE_FIELDS_THAN_HEADER = (
    "the CSV has rows with more fields than its header names, so values would be "
    "dropped or land in the wrong column. Check for an unquoted comma inside a value."
)

# `warnings.catch_warnings` swaps the process's warning filters and puts them
# back, so two reads at once can leave the `error` filter installed for good,
# or take it away from under one another. The API calls this from a pool of
# threads; the lock makes this module's reads take turns.
_FILTERS = threading.Lock()


def read_csv(source: str | Path | IO[bytes], *, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV the way every reader of an upload must.

    `index_col=False`: left to itself pandas takes the first column as an
    index when the rows have one field more than the header, which shifts
    every column by one. A comma at the end of every line is enough to do it.
    With the option set the columns stay under their own names, and pandas
    warns when that drops a field holding data; the warning is raised here as
    a refusal, since the alternative is an analysis of a file that was not
    read whole.

    `low_memory=False`: pandas otherwise reads a large file in pieces and
    settles each piece's types separately, so a column that is numbers in one
    piece and text in another holds both.

    A file pandas cannot read at all raises what pandas raises, for example
    `EmptyDataError`, `ParserError` or `UnicodeDecodeError`. A parser warning
    about anything else is raised as itself, not relabelled.

    The lock covers this module only. Code elsewhere in the process that
    enters `warnings.catch_warnings` on another thread while a read is under
    way can still take the filter away for that read, and pandas then drops
    the field with a warning and no refusal. No code in this repository does
    that; whether a library the API imports does so on a request thread was
    looked for and not found, which is not the same as ruled out. The worker,
    which reads every upload again before analysing it, runs one job at a
    time in its process.
    """
    with _FILTERS, warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        try:
            return pd.read_csv(source, nrows=nrows, index_col=False, low_memory=False)
        except pd.errors.ParserWarning as exc:
            if "index_col=False" not in str(exc):
                raise
            raise ValueError(MORE_FIELDS_THAN_HEADER) from exc
