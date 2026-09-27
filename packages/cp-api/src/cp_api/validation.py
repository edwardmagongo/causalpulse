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

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class RequestError(Exception):
    """A request the API refuses, with the HTTP status to refuse it with."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _parse(content: bytes, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content), nrows=nrows, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise RequestError(422, f"the upload could not be read as CSV: {exc}") from exc


def check_upload(content: bytes) -> tuple[str, ...]:
    if len(content) > MAX_UPLOAD_BYTES:
        raise RequestError(413, f"the upload exceeds the {MAX_UPLOAD_BYTES}-byte limit")
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
