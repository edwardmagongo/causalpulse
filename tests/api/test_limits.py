"""The limiter on its own: windows, settings, the client address, the sentences.

The Redis here is a small in-memory stand-in with a clock the test moves, so a
window ends without anything sleeping. What the limits do to real requests,
against a real Redis, is `tests/services/test_limits.py`.
"""

import asyncio
import contextlib
import logging
import socket
import threading
import time

import pytest
from cp_api import limits
from cp_api.limits import (
    Decision,
    FixedWindow,
    Limit,
    LimitExceededError,
    SigninFailures,
    client_address,
    limiter_redis,
    limits_from_environment,
)
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ExecAbortError, OutOfMemoryError, ReadOnlyError
from starlette.requests import Request
from starlette.responses import JSONResponse

THREE_A_MINUTE = Limit("uploads_per_day", 3, 60)


class _Pipeline:
    def __init__(self, store):
        self._store = store
        self._calls = []

    def incr(self, name):
        self._calls.append(lambda: self._store.incr(name))

    def expire(self, name, seconds, nx=False):
        self._calls.append(lambda: self._store.expire(name, seconds, nx=nx))

    def ttl(self, name):
        self._calls.append(lambda: self._store.ttl(name))

    def get(self, name):
        self._calls.append(lambda: self._store.get(name))

    def execute(self):
        self._store.round_trips += 1
        return [call() for call in self._calls]


class FakeRedis:
    """`INCR`, `EXPIRE ... NX`, `TTL` and `GET` as Redis answers them."""

    def __init__(self):
        self.now = 1_000.0
        self.round_trips = 0
        self._values = {}
        self._deadlines = {}

    def _drop_if_expired(self, name):
        deadline = self._deadlines.get(name)
        if deadline is not None and self.now >= deadline:
            del self._values[name], self._deadlines[name]

    def incr(self, name):
        self._drop_if_expired(name)
        self._values[name] = self._values.get(name, 0) + 1
        return self._values[name]

    def expire(self, name, seconds, nx=False):
        if name not in self._values or (nx and name in self._deadlines):
            return False
        self._deadlines[name] = self.now + seconds
        return True

    def ttl(self, name):
        self._drop_if_expired(name)
        if name not in self._values:
            return -2
        if name not in self._deadlines:
            return -1
        return round(self._deadlines[name] - self.now)

    def get(self, name):
        self._drop_if_expired(name)
        value = self._values.get(name)
        return None if value is None else str(value).encode()

    def pipeline(self, transaction=True):
        assert transaction, "the increment and its expiry must be one transaction"
        return _Pipeline(self)

    def keys(self):
        return sorted(self._values)


class DownRedis:
    def pipeline(self, transaction=True):
        raise RedisConnectionError("connection refused")


def test_a_window_fills_and_then_refuses():
    window = FixedWindow(FakeRedis())

    decisions = [window.hit("uploads:alice", THREE_A_MINUTE) for _ in range(4)]

    assert [d.allowed for d in decisions] == [True, True, True, False]


def test_retry_after_is_the_time_left_in_the_window():
    redis = FakeRedis()
    window = FixedWindow(redis)
    for _ in range(3):
        window.hit("uploads:alice", THREE_A_MINUTE)

    redis.now += 20

    assert window.hit("uploads:alice", THREE_A_MINUTE) == Decision(False, 40)


def test_a_refused_hit_does_not_push_the_end_of_the_window_back():
    redis = FakeRedis()
    window = FixedWindow(redis)
    for _ in range(3):
        window.hit("uploads:alice", THREE_A_MINUTE)

    redis.now += 59
    assert window.hit("uploads:alice", THREE_A_MINUTE) == Decision(False, 1)
    redis.now += 1

    assert window.hit("uploads:alice", THREE_A_MINUTE).allowed


def test_a_new_window_allows_again():
    redis = FakeRedis()
    window = FixedWindow(redis)
    for _ in range(4):
        window.hit("uploads:alice", THREE_A_MINUTE)

    redis.now += 60

    assert [window.hit("uploads:alice", THREE_A_MINUTE).allowed for _ in range(4)] == [
        True, True, True, False,
    ]  # fmt: skip


def test_retry_after_is_never_less_than_a_second():
    redis = FakeRedis()
    window = FixedWindow(redis)
    for _ in range(3):
        window.hit("uploads:alice", THREE_A_MINUTE)

    redis.now += 59.9  # Redis reports a TTL of 0 for the last fraction of a second.

    assert window.hit("uploads:alice", THREE_A_MINUTE) == Decision(False, 1)


def test_two_keys_do_not_share_a_window():
    window = FixedWindow(FakeRedis())
    for _ in range(4):
        window.hit("uploads:alice", THREE_A_MINUTE)

    assert window.hit("uploads:bob", THREE_A_MINUTE).allowed


def test_a_hit_is_one_round_trip():
    redis = FakeRedis()

    FixedWindow(redis).hit("uploads:alice", THREE_A_MINUTE)

    assert redis.round_trips == 1


def test_the_keys_carry_the_limiters_own_prefix():
    # rq keeps its queues in the same Redis, under `rq:`.
    redis = FakeRedis()

    FixedWindow(redis).hit("uploads:alice", THREE_A_MINUTE)

    assert redis.keys() == ["cp:limit:uploads:alice"]


def test_peek_reads_the_window_without_counting():
    redis = FakeRedis()
    window = FixedWindow(redis)

    assert window.peek("signin:10.0.0.1", THREE_A_MINUTE) == Decision(True, 0)
    for _ in range(3):
        window.hit("signin:10.0.0.1", THREE_A_MINUTE)
    redis.now += 15
    peeks = [window.peek("signin:10.0.0.1", THREE_A_MINUTE) for _ in range(5)]

    # Full is refused: the next one would be one too many.
    assert peeks == [Decision(False, 45)] * 5
    redis.now += 45
    assert window.peek("signin:10.0.0.1", THREE_A_MINUTE).allowed


def test_a_redis_that_cannot_be_reached_allows(monkeypatch, caplog):
    monkeypatch.setattr(limits, "_outage_logged", False)
    window = FixedWindow(DownRedis())

    with caplog.at_level(logging.WARNING, logger="cp_api.limits"):
        hits = [window.hit("uploads:alice", THREE_A_MINUTE) for _ in range(5)]
        peek = window.peek("uploads:alice", THREE_A_MINUTE)

    assert all(d.allowed for d in (*hits, peek))
    # Once per process, however many requests the outage lasts.
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "ConnectionError" in caplog.records[0].getMessage()


# What a Redis older than 7 answers, which will not run `EXPIRE ... NX` or the
# transaction it is in.
TOO_OLD = ExecAbortError("Transaction discarded because of previous errors.")


class RefusingRedis:
    """A Redis that is reached and will not run the limiter's command."""

    def __init__(self, error=TOO_OLD):
        self._error = error

    def pipeline(self, transaction=True):
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        TOO_OLD,
        OutOfMemoryError("command not allowed when used memory > 'maxmemory'."),
        ReadOnlyError("You can't write against a read only replica."),
    ],
)
def test_a_redis_that_refuses_the_command_allows_and_says_so_as_an_error(
    monkeypatch, caplog, error
):
    monkeypatch.setattr(limits, "_outage_logged", False)
    monkeypatch.setattr(limits, "_refusal_logged", False)
    window = FixedWindow(RefusingRedis(error))

    with caplog.at_level(logging.WARNING, logger="cp_api.limits"):
        hits = [window.hit("uploads:alice", THREE_A_MINUTE) for _ in range(5)]
        peek = window.peek("uploads:alice", THREE_A_MINUTE)

    assert all(d.allowed for d in (*hits, peek))
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    message = caplog.records[0].getMessage()
    assert "Redis refused the rate limiter's command" in message
    # The server's own words, and no guess at the cause: an old Redis is one of several.
    assert f"{type(error).__name__}: {error}" in message
    assert "Redis 7" not in message
    # Not the outage's message: Redis was reached.
    assert "could not be reached" not in message


def test_a_refused_command_and_an_outage_are_each_logged(monkeypatch, caplog):
    monkeypatch.setattr(limits, "_outage_logged", False)
    monkeypatch.setattr(limits, "_refusal_logged", False)

    with caplog.at_level(logging.WARNING, logger="cp_api.limits"):
        FixedWindow(RefusingRedis()).hit("uploads:alice", THREE_A_MINUTE)
        FixedWindow(DownRedis()).hit("uploads:alice", THREE_A_MINUTE)

    assert [record.levelno for record in caplog.records] == [logging.ERROR, logging.WARNING]


@pytest.fixture
def silent_redis_url():
    """A server that accepts every connection and never answers: a Redis that hangs."""
    listening = socket.socket()
    listening.bind(("127.0.0.1", 0))
    listening.listen(16)
    # `accept` gives up every tenth of a second to see whether the test is over.
    listening.settimeout(0.1)
    held, over = [], threading.Event()

    def accept():
        while not over.is_set():
            with contextlib.suppress(TimeoutError):
                held.append(listening.accept()[0])

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield f"redis://127.0.0.1:{listening.getsockname()[1]}/0"
    over.set()
    thread.join(5)
    listening.close()
    for connection in held:
        connection.close()


def test_the_limiters_client_gives_up_on_a_redis_that_does_not_answer(
    monkeypatch, silent_redis_url
):
    monkeypatch.setattr(limits, "_outage_logged", True)
    client = limiter_redis(silent_redis_url)
    window = FixedWindow(client)
    decisions = []

    def ask():
        decisions.append(window.peek("signin:10.0.0.1", THREE_A_MINUTE))
        decisions.append(window.hit("signin:10.0.0.1", THREE_A_MINUTE))

    # A watchdog, not a wait: a client with no timeout would never come back.
    thread = threading.Thread(target=ask, daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(4 * limits.LIMITER_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started

    assert decisions == [Decision(True, 0), Decision(True, 0)], (
        f"two calls had not both returned after {elapsed:.1f}s"
    )
    # Each call waited for an answer, so this was a timeout and not a refused connection.
    assert elapsed >= limits.LIMITER_TIMEOUT_SECONDS
    client.close()


def test_the_limiters_timeouts_are_a_second_or_less():
    assert 0 < limits.LIMITER_TIMEOUT_SECONDS <= 1
    kwargs = limiter_redis("redis://127.0.0.1:1/0").connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == limits.LIMITER_TIMEOUT_SECONDS
    assert kwargs["socket_connect_timeout"] == limits.LIMITER_TIMEOUT_SECONDS


class LockedRedis(FakeRedis):
    """`FakeRedis` for callers on several threads: one transaction at a time, as in Redis."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def pipeline(self, transaction=True):
        pipeline = super().pipeline(transaction)
        execute = pipeline.execute

        def locked():
            with self._lock:
                return execute()

        pipeline.execute = locked
        return pipeline


def test_failed_sign_ins_made_at_once_are_all_answered_before_any_is_counted(monkeypatch):
    """The sign-in limit is not exact for concurrent requests. This is accepted.

    The middleware reads the count on the way in and adds to it after a 401, so
    requests that are all in flight together are all let through to the key
    check. See `SigninFailures`. If a change makes this test fail by refusing
    some of them, the limit has become exact, at the price of a write to Redis
    on every request: decide that on purpose, then change this test.
    """
    monkeypatch.delenv("CP_ENV", raising=False)
    redis = LockedRedis()
    limit = Limit("signin_failures", 2, 900)
    sent = 8
    name = "cp:limit:signin:192.0.2.7"
    counts_when_answered = []

    async def main():
        arrived = 0
        everyone_is_in = asyncio.Event()

        async def refuses_the_key(scope, receive, send):
            # Every request is held here, past the limiter's read, until all are.
            nonlocal arrived
            arrived += 1
            if arrived == sent:
                everyone_is_in.set()
            await everyone_is_in.wait()
            counts_when_answered.append(redis.get(name))
            await JSONResponse({"detail": "unauthorized"}, status_code=401)(scope, receive, send)

        guarded = SigninFailures(refuses_the_key, FixedWindow(redis), limit)

        async def request():
            statuses = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                if message["type"] == "http.response.start":
                    statuses.append(message["status"])

            scope = {
                "type": "http", "method": "GET", "path": "/v1/me", "headers": [],
                "client": ("192.0.2.7", 40_000),
            }  # fmt: skip
            await guarded(scope, receive, send)
            return statuses[0]

        together = await asyncio.wait_for(
            asyncio.gather(*(request() for _ in range(sent))), timeout=30
        )
        return together, await request()

    together, afterwards = asyncio.run(main())

    # Eight guesses against a limit of two, none refused...
    assert together == [401] * sent
    # ...the first of them answered with nothing yet counted against the address...
    assert len(counts_when_answered) == sent
    assert counts_when_answered[0] is None
    # ...and each one counted afterwards, so the address is then refused.
    assert redis.get(name) == str(sent).encode()
    assert afterwards == 429


def test_no_setting_means_no_limit():
    read = limits_from_environment({})

    assert read == limits.Limits()
    assert (
        read.signin_failures, read.uploads_per_day, read.analyses_per_day, read.analyses_active
    ) == (None, None, None, None)  # fmt: skip


def test_an_empty_setting_disables_its_limit():
    read = limits_from_environment(
        {"CP_LIMIT_UPLOADS_PER_DAY": "", "CP_LIMIT_ANALYSES_ACTIVE": " "}
    )

    assert read == limits.Limits()


def test_the_settings_are_read_with_their_windows():
    read = limits_from_environment(
        {
            "CP_LIMIT_SIGNIN_FAILURES": "10",
            "CP_LIMIT_UPLOADS_PER_DAY": "20",
            "CP_LIMIT_ANALYSES_PER_DAY": "30",
            "CP_LIMIT_ANALYSES_ACTIVE": "3",
        }
    )

    assert read.signin_failures == Limit("signin_failures", 10, 900)
    assert read.uploads_per_day == Limit("uploads_per_day", 20, 86_400)
    assert read.analyses_per_day == Limit("analyses_per_day", 30, 86_400)
    assert read.analyses_active is not None
    assert (read.analyses_active.name, read.analyses_active.count) == ("analyses_active", 3)


def test_the_process_environment_is_what_is_read_by_default(monkeypatch):
    monkeypatch.setenv("CP_LIMIT_UPLOADS_PER_DAY", "7")
    monkeypatch.delenv("CP_LIMIT_ANALYSES_PER_DAY", raising=False)

    read = limits_from_environment()

    assert read.uploads_per_day == Limit("uploads_per_day", 7, 86_400)
    assert read.analyses_per_day is None


@pytest.mark.parametrize("value", ["0", "-1", "ten", "1.5", "3 a day", "+-2"])
@pytest.mark.parametrize(
    "setting",
    [
        "CP_LIMIT_SIGNIN_FAILURES",
        "CP_LIMIT_UPLOADS_PER_DAY",
        "CP_LIMIT_ANALYSES_PER_DAY",
        "CP_LIMIT_ANALYSES_ACTIVE",
    ],
)
def test_a_setting_that_is_not_a_positive_integer_stops_the_start(setting, value):
    with pytest.raises(SystemExit) as stopped:
        limits_from_environment({setting: value})

    assert setting in str(stopped.value)


def _request(peer=("192.0.2.7", 40_000), header=None):
    headers = [] if header is None else [(b"x-cp-client-address", header.encode())]
    return Request({"type": "http", "headers": headers, "client": peer})


def test_outside_production_the_address_is_the_connections(monkeypatch):
    monkeypatch.delenv("CP_ENV", raising=False)

    assert client_address(_request(header="203.0.113.9")) == "192.0.2.7"


def test_in_production_the_dashboards_header_is_believed(monkeypatch):
    monkeypatch.setenv("CP_ENV", "production")

    assert client_address(_request(header="203.0.113.9")) == "203.0.113.9"
    assert client_address(_request(header=" 203.0.113.9 , 10.0.0.1")) == "203.0.113.9"
    # One address has one spelling, so one window.
    assert client_address(_request(header="2001:DB8:0:0::1")) == "2001:db8::1"


@pytest.mark.parametrize("header", [None, "", "not an address", "203.0.113.9:443", ","])
def test_in_production_a_header_that_is_no_address_falls_back_to_the_connection(
    monkeypatch, header
):
    monkeypatch.setenv("CP_ENV", "production")

    assert client_address(_request(header=header)) == "192.0.2.7"


def test_a_connection_with_no_peer_address_has_a_fixed_key(monkeypatch):
    monkeypatch.delenv("CP_ENV", raising=False)

    assert client_address(_request(peer=None)) == limits.UNKNOWN_ADDRESS


@pytest.mark.parametrize(
    ("seconds", "wait"),
    [
        (1, "1 second"),
        (45, "45 seconds"),
        (60, "1 minute"),
        (61, "2 minutes"),
        (900, "15 minutes"),
        (3_600, "1 hour"),
        (3_601, "2 hours"),
        (86_400, "24 hours"),
    ],
)
def test_the_wait_is_said_in_the_largest_unit_and_never_rounded_down(seconds, wait):
    refusal = LimitExceededError(Limit("uploads_per_day", 20, 86_400), seconds)

    assert refusal.detail == f"Too many uploads today for this account. Try again in {wait}."


def test_every_limit_has_its_own_sentence():
    read = limits_from_environment(
        {
            "CP_LIMIT_SIGNIN_FAILURES": "10",
            "CP_LIMIT_UPLOADS_PER_DAY": "20",
            "CP_LIMIT_ANALYSES_PER_DAY": "30",
            "CP_LIMIT_ANALYSES_ACTIVE": "3",
        }
    )
    every = (
        read.signin_failures, read.uploads_per_day, read.analyses_per_day, read.analyses_active
    )  # fmt: skip

    details = [LimitExceededError(limit, 120).detail for limit in every]

    assert details == [
        "Too many failed sign-ins from this address. Try again in 2 minutes.",
        "Too many uploads today for this account. Try again in 2 minutes.",
        "Too many analyses today for this account. Try again in 2 minutes.",
        "This account already has 3 analyses waiting or running. "
        "Try again when one has finished.",
    ]


def test_a_refusal_is_a_429_with_the_wait_in_whole_seconds():
    response = LimitExceededError(Limit("uploads_per_day", 20, 86_400), 0).response()

    one = LimitExceededError(Limit("analyses_active", 1, 60), 60)
    assert one.detail == (
        "This account already has 1 analysis waiting or running. "
        "Try again when one has finished."
    )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.body == (
        b'{"detail":"Too many uploads today for this account. Try again in 1 second."}'
    )
