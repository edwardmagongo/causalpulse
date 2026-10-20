"""Which entry of the server's list a request names, and what the analysis's
interpretations say of a request for each entry.

The request route and the analysis's `interpretation_options` both decide
through `_entry_refusal`, which asks `_interpretations_refusal` once the
analysis itself can be interpreted; `tests/services/test_interpretations.py`
runs both against PostgreSQL.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from cp_api.app import (
    ALREADY_REFUSED,
    ALREADY_VERIFIED,
    BEING_WRITTEN,
    CURRENT_INTERPRETER_VERSION,
    NAME_BOTH_OR_NEITHER,
    NOT_OFFERED,
    _chosen_entry,
    _interpretations_refusal,
    create_app,
)
from cp_api.db import Interpretation
from cp_api.models import InterpretationRequest
from cp_api.settings import InterpretationOption
from cp_api.validation import RequestError

A = InterpretationOption("claude-opus-5-5", "medium")
B = InterpretationOption("claude-sonnet-5-5", "low")
TWO = (A, B)
OLDER = "0123456789ab"
START = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _row(entry, status, version=CURRENT_INTERPRETER_VERSION, minutes=0):
    return Interpretation(
        id=uuid.uuid4(),
        status=status,
        job_id="interpretation-x",
        model=entry.model,
        effort=entry.effort,
        interpreter_version=version,
        created_at=START + timedelta(minutes=minutes),
    )


def _newest_first(*rows):
    return sorted(rows, key=lambda row: row.created_at, reverse=True)


# --- the entry a request names --------------------------------------------


@pytest.mark.parametrize(
    "body",
    [None, InterpretationRequest(), InterpretationRequest(model=None, effort=None)],
    ids=["no-body", "empty", "both-null"],
)
def test_a_request_naming_neither_gets_the_default(body):
    assert _chosen_entry(body, TWO) == A
    assert _chosen_entry(body, (B, A)) == B


def test_a_request_naming_a_listed_entry_gets_it():
    assert _chosen_entry(InterpretationRequest(model=B.model, effort=B.effort), TWO) == B
    assert _chosen_entry(InterpretationRequest(model=A.model, effort=A.effort), TWO) == A


@pytest.mark.parametrize(
    "body",
    [
        InterpretationRequest(model="claude-opus-5-5"),
        InterpretationRequest(effort="medium"),
        InterpretationRequest(model="claude-opus-5-5", effort=None),
        InterpretationRequest(model="", effort=None),
    ],
)
def test_a_request_naming_one_of_the_two_is_refused(body):
    with pytest.raises(RequestError) as refused:
        _chosen_entry(body, TWO)

    assert (refused.value.status_code, refused.value.message) == (422, NAME_BOTH_OR_NEITHER)


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        # A model on the list at another of its efforts, and the other way round.
        ("claude-opus-5-5", "low"),
        ("claude-sonnet-5-5", "medium"),
        ("claude-haiku-5-5", "medium"),
        ("claude-opus-5-5", "turbo"),
        ("Claude-Opus-5-5", "medium"),
        ("", ""),
    ],
)
def test_a_request_naming_an_entry_not_on_the_list_is_refused(model, effort):
    with pytest.raises(RequestError) as refused:
        _chosen_entry(InterpretationRequest(model=model, effort=effort), TWO)

    assert (refused.value.status_code, refused.value.message) == (422, NOT_OFFERED)


# --- what the interpretations say of each entry ---------------------------


def test_with_no_interpretation_every_entry_is_accepted():
    assert [_interpretations_refusal(entry, []) for entry in TWO] == [None, None]


@pytest.mark.parametrize("status", ["queued", "running"])
def test_one_pending_from_either_entry_refuses_every_entry(status):
    for pending_entry in TWO:
        rows = _newest_first(
            _row(A, "failed", minutes=1),
            _row(pending_entry, status, version=None, minutes=2),
        )

        assert [_interpretations_refusal(entry, rows) for entry in TWO] == [
            BEING_WRITTEN, BEING_WRITTEN,
        ]


def test_a_pending_one_refuses_even_when_it_is_not_the_newest():
    # Not reached through the route, which refuses while any is pending; the
    # rule is "any", not "the newest".
    rows = _newest_first(_row(A, "queued", version=None, minutes=1), _row(B, "failed", minutes=2))

    assert [_interpretations_refusal(entry, rows) for entry in TWO] == [
        BEING_WRITTEN, BEING_WRITTEN,
    ]


# Each kind of finished newest interpretation from an entry: what a request
# for that entry is answered, or None for accepted.
DECISIONS = [
    ("verified", CURRENT_INTERPRETER_VERSION, ALREADY_VERIFIED),
    ("verified", OLDER, None),
    ("verified", None, None),
    ("refused", CURRENT_INTERPRETER_VERSION, ALREADY_REFUSED),
    ("refused", OLDER, None),
    ("refused", None, None),
    ("failed", CURRENT_INTERPRETER_VERSION, None),
    ("failed", OLDER, None),
    ("failed", None, None),
]


@pytest.mark.parametrize(("status", "version", "refusal"), DECISIONS)
def test_an_entrys_own_newest_decides_for_it_and_not_for_the_other(status, version, refusal):
    rows = [_row(B, status, version)]

    assert _interpretations_refusal(B, rows) == refusal
    assert _interpretations_refusal(A, rows) is None


@pytest.mark.parametrize(("status", "version", "refusal"), DECISIONS)
def test_the_newest_from_the_entry_decides_whatever_the_other_entry_wrote_since(
    status, version, refusal
):
    # B's row is the analysis's newest, A's is older: A's own newest decides for A.
    for later in ("verified", "refused", "failed"):
        rows = _newest_first(_row(A, status, version, minutes=1), _row(B, later, minutes=2))

        assert _interpretations_refusal(A, rows) == refusal, later


def test_only_the_newest_from_the_entry_counts_not_an_older_one_from_it():
    # A verified one, then a failed one, from A: the failed one is A's newest.
    rows = _newest_first(_row(A, "verified", minutes=1), _row(A, "failed", minutes=2))
    assert _interpretations_refusal(A, rows) is None
    # The other way round: the verified one is newest, and final.
    rows = _newest_first(_row(A, "failed", minutes=1), _row(A, "verified", minutes=2))
    assert _interpretations_refusal(A, rows) == ALREADY_VERIFIED


def test_an_entry_is_both_its_model_and_its_effort():
    # A row from A's model at another effort, and one from A's effort with
    # another model: neither is from A.
    rows = _newest_first(
        _row(InterpretationOption(A.model, "high"), "verified", minutes=1),
        _row(InterpretationOption(B.model, A.effort), "refused", minutes=2),
    )

    assert _interpretations_refusal(A, rows) is None


def test_verified_from_one_then_the_other_refuses_both():
    rows = _newest_first(_row(A, "verified", minutes=1), _row(B, "verified", minutes=2))

    assert [_interpretations_refusal(entry, rows) for entry in TWO] == [
        ALREADY_VERIFIED, ALREADY_VERIFIED,
    ]


# --- the app's list --------------------------------------------------------


def test_the_app_refuses_to_start_on_a_malformed_setting(monkeypatch):
    # As the worker does: one line naming the setting and the entry, and no traceback.
    monkeypatch.setenv("CP_INTERPRETATION_OPTIONS", "claude-sonnet-5-5:low, claude-opus-5-5:turbo")

    with pytest.raises(SystemExit) as refused:
        create_app(sessions=object(), redis=object())

    message = str(refused.value)
    assert "CP_INTERPRETATION_OPTIONS" in message
    assert "'claude-opus-5-5:turbo'" in message
    assert "\n" not in message
    assert refused.value.__cause__ is None
    assert refused.value.__suppress_context__


def test_an_app_given_an_empty_list_is_a_mistake():
    with pytest.raises(ValueError, match="at least one entry"):
        create_app(sessions=object(), redis=object(), interpretation_options=())


def test_an_app_given_its_list_does_not_read_the_setting(monkeypatch):
    monkeypatch.setenv("CP_INTERPRETATION_OPTIONS", "claude-opus-5-5:turbo")

    create_app(sessions=object(), redis=object(), interpretation_options=TWO)
