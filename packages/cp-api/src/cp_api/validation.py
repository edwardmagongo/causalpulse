"""What the API can refuse before queueing anything, using `cp_core` alone.

Routing needs `cp_causal`, which this package may not import, so an unknown
or ineligible estimator is the worker's to refuse (spec §11.3). Everything
here is a pure function of the request, which is what keeps it in the fast
test tier.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from cp_core.dataset import CausalSchema, schema_from_dict
from cp_core.reading import read_csv

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
FALLBACK_FILENAME = "upload.csv"
# The width of `datasets.filename`.
MAX_FILENAME_CHARS = 255


class RequestError(Exception):
    """A request the API refuses, with the HTTP status to refuse it with."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def safe_filename(raw: str | None) -> str:
    """The client's filename reduced to one bare path component.

    It becomes the manifest's fixture path, which replay opens relative to the
    current directory, so no separator, `.` or `..` may survive. Both `/` and
    `\\` are split on: `PurePath` on POSIX keeps a Windows path like
    `C:\\x\\y.csv` whole.
    """
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    if name in ("", ".", ".."):
        return FALLBACK_FILENAME
    return name[:MAX_FILENAME_CHARS]


def _parse(content: bytes, nrows: int | None = None) -> pd.DataFrame:
    try:
        return read_csv(io.BytesIO(content), nrows=nrows)
    # pandas' own failures to read a file (`EmptyDataError`, `ParserError`,
    # `UnicodeDecodeError`) are all `ValueError`s, as is the reader's refusal.
    except ValueError as exc:
        raise RequestError(422, f"the upload could not be read as CSV: {exc}") from exc


def too_large() -> RequestError:
    """The one refusal of an upload over the limit, whoever notices that it is."""
    return RequestError(413, f"the upload exceeds the {MAX_UPLOAD_BYTES}-byte limit")


def check_upload(content: bytes) -> tuple[str, ...]:
    if len(content) > MAX_UPLOAD_BYTES:
        raise too_large()
    if not content.strip():
        raise RequestError(422, "the upload is empty")
    frame = _parse(content)
    if frame.empty:
        raise RequestError(422, "the CSV has a header but no rows")
    return tuple(str(column) for column in frame.columns)


def csv_header(content: bytes) -> tuple[str, ...]:
    return tuple(str(column) for column in _parse(content, nrows=0).columns)


def check_analysis_request(
    schema: Mapping[str, Any], header: Sequence[str], estimators: Sequence[str] | None
) -> CausalSchema:
    if estimators is not None and not estimators:
        raise RequestError(
            422,
            "estimators, when given, must name at least one estimator; omit it to "
            "let the router choose",
        )
    try:
        parsed = schema_from_dict(schema)
    except (ValueError, TypeError) as exc:
        raise RequestError(422, f"invalid schema: {exc}") from exc
    present = set(header)
    wanted = (*parsed.required_columns, *parsed.auxiliary_columns)
    missing = sorted({column for column in wanted if column not in present})
    if missing:
        raise RequestError(
            422, f"the schema names column(s) the dataset does not have: {missing}"
        )
    return parsed
