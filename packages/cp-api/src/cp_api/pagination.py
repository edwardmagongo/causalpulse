"""Keyset cursors for the list routes.

A list is ordered newest first by `(created_at, id)`. The cursor names the
last row of a page, and the next page is every row that sorts after it. `id`
breaks ties: two rows can share a `created_at`, and a cursor on the time
alone would skip one of them or return it twice.

The cursor is opaque to the client: it is whatever `next_cursor` held.
"""

from __future__ import annotations

import base64
import binascii
import uuid
from datetime import datetime

from cp_api.validation import RequestError

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


def encode_cursor(created_at: datetime, row_id: uuid.UUID) -> str:
    raw = f"{created_at.isoformat()}|{row_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_cursor(raw: str) -> tuple[datetime, uuid.UUID]:
    """The `(created_at, id)` a cursor names, or a 422 for anything else."""
    try:
        text = base64.urlsafe_b64decode(raw.encode()).decode()
        moment, _, row_id = text.partition("|")
        created_at = datetime.fromisoformat(moment)
        if created_at.tzinfo is None:
            raise ValueError("a cursor's time carries its offset")
        return created_at, uuid.UUID(row_id)
    except (ValueError, binascii.Error) as exc:
        raise RequestError(422, "cursor is not one this API issued") from exc
