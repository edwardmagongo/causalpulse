import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from cp_api.pagination import decode_cursor, encode_cursor
from cp_api.validation import RequestError

MOMENT = datetime(2026, 10, 1, 12, 30, 15, 123456, tzinfo=UTC)
ROW = uuid.UUID("0f8fad5b-d9cb-469f-a165-70867728950e")


def test_a_cursor_round_trips_to_the_microsecond():
    assert decode_cursor(encode_cursor(MOMENT, ROW)) == (MOMENT, ROW)


def test_a_cursor_keeps_a_non_utc_offset_as_the_same_instant():
    # psycopg returns timestamps in the session's time zone, which need not be UTC.
    local = MOMENT.astimezone(timezone(timedelta(hours=3)))

    decoded, _ = decode_cursor(encode_cursor(local, ROW))

    assert decoded == MOMENT and decoded.utcoffset() == timedelta(hours=3)


def test_a_cursor_is_safe_in_a_query_string():
    assert set(encode_cursor(MOMENT, ROW)) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_="
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not base64 at all!",
        "bm90IGEgY3Vyc29y",  # "not a cursor"
        "MjAyNi0xMC0wMVQxMjozMDoxNSswMDowMHxub3QtYS11dWlk",  # a time, then "not-a-uuid"
        "MjAyNi0xMC0wMVQxMjozMDoxNXwwZjhmYWQ1Yi1kOWNiLTQ2OWYtYTE2NS03MDg2NzcyODk1MGU=",  # no offset
        "/w==",  # bytes that are not UTF-8
    ],
)
def test_anything_else_is_a_422(raw):
    with pytest.raises(RequestError) as refused:
        decode_cursor(raw)

    assert refused.value.status_code == 422
