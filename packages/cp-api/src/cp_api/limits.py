"""Rate limits: fixed-window counters in Redis, each off unless its setting is present.

Four limits, read once when the app is built (`limits_from_environment`):

- failed sign-ins, counted by client address (`SigninFailures`, a middleware);
- uploads a day and analyses a day, counted by user (`FixedWindow.require`);
- analyses queued or running, which is a count of rows the app reads, not a
  counter here; this module holds only its setting and its sentence.

A window starts at the first request counted in it and ends `window_seconds`
later, when Redis expires the key. If Redis cannot be reached every limit
allows the request: the health check already reports the outage, and a limiter
that refused would turn a Redis blip into a site that refuses uploads. A Redis
that accepts a connection and then says nothing is the same outage, so the
limiter has a client of its own with short timeouts (`limiter_redis`).

This module imports nothing from `cp_api.app`, which imports it.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

from redis import Redis
from redis.exceptions import RedisError, ResponseError
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from cp_api.settings import is_production

logger = logging.getLogger(__name__)

# rq keeps its queues and jobs in the same Redis under `rq:`.
KEY_PREFIX = "cp:limit:"
CLIENT_ADDRESS_HEADER = "X-CP-Client-Address"
# The key for a connection that reports no peer address (a Unix socket, some
# test transports). All such connections share one window.
UNKNOWN_ADDRESS = "unknown"

# How long the limiter waits for Redis to accept a connection, and then for
# each answer. Redis is on the same network and answers a counter in well under
# a millisecond, so half a second is hundreds of times what a healthy one takes;
# and while Redis hangs every limited request waits this long, once or twice,
# before it is allowed, so it cannot be much longer.
LIMITER_TIMEOUT_SECONDS = 0.5

SIGNIN_WINDOW_SECONDS = 15 * 60
DAY_SECONDS = 24 * 60 * 60
# The active limit has no window: this is how long a refused client is told to wait.
ACTIVE_RETRY_SECONDS = 60

# Every sentence a refused client is shown, keyed by `Limit.name`. `{wait}` is
# the time left in the window; `{count}` and `{analyses}` are the limit itself.
_SENTENCES = {
    "signin_failures": "Too many failed sign-ins from this address. Try again in {wait}.",
    "uploads_per_day": "Too many uploads today for this account. Try again in {wait}.",
    "analyses_per_day": "Too many analyses today for this account. Try again in {wait}.",
    "interpretations_per_day": (
        "Too many interpretations today for this account. Try again in {wait}."
    ),
    "analyses_active": (
        "This account already has {count} {analyses} waiting or running. "
        "Try again when one has finished."
    ),
}

_outage_logged = False
_refusal_logged = False


def limiter_redis(url: str) -> Redis:
    """A client for the limiter alone, which gives up on a Redis that does not answer.

    Separate from the client the queue uses, so nothing the queue does is held
    to these timeouts. With the sign-in limit on, every `/v1` request begins
    with a read through this client; without a timeout of its own a Redis that
    hangs would hold each of them for as long as the library's default allows.
    """
    return Redis.from_url(
        url,
        socket_connect_timeout=LIMITER_TIMEOUT_SECONDS,
        socket_timeout=LIMITER_TIMEOUT_SECONDS,
    )


@dataclass(frozen=True)
class Limit:
    """At most `count` in any one window of `window_seconds`. `name` picks its sentence."""

    name: str
    count: int
    window_seconds: int


@dataclass(frozen=True)
class Limits:
    """The five limits; None is a limit that is switched off."""

    signin_failures: Limit | None = None
    uploads_per_day: Limit | None = None
    analyses_per_day: Limit | None = None
    analyses_active: Limit | None = None
    interpretations_per_day: Limit | None = None


@dataclass(frozen=True)
class Decision:
    """`retry_after` is the whole seconds left in the window, and 0 when allowed."""

    allowed: bool
    retry_after: int


def _setting(environ: Mapping[str, str], setting: str, name: str, window: int) -> Limit | None:
    raw = environ.get(setting, "").strip()
    if not raw:
        return None
    if not (raw.isascii() and raw.isdigit()) or int(raw) < 1:
        raise SystemExit(
            f"{setting} must be a whole number greater than zero, or empty to switch "
            "the limit off."
        )
    return Limit(name, int(raw), window)


def limits_from_environment(environ: Mapping[str, str] | None = None) -> Limits:
    """The limits the settings ask for. A setting that is absent or empty asks for none.

    Anything else that is not a positive integer stops the start, naming the setting.
    """
    source = os.environ if environ is None else environ
    return Limits(
        signin_failures=_setting(
            source, "CP_LIMIT_SIGNIN_FAILURES", "signin_failures", SIGNIN_WINDOW_SECONDS
        ),
        uploads_per_day=_setting(
            source, "CP_LIMIT_UPLOADS_PER_DAY", "uploads_per_day", DAY_SECONDS
        ),
        analyses_per_day=_setting(
            source, "CP_LIMIT_ANALYSES_PER_DAY", "analyses_per_day", DAY_SECONDS
        ),
        analyses_active=_setting(
            source, "CP_LIMIT_ANALYSES_ACTIVE", "analyses_active", ACTIVE_RETRY_SECONDS
        ),
        interpretations_per_day=_setting(
            source, "CP_LIMIT_INTERPRETATIONS_PER_DAY", "interpretations_per_day", DAY_SECONDS
        ),
    )


def _wait(seconds: int) -> str:
    """The wait in its largest whole unit, rounded up so it is never too short."""
    for size, unit in ((3600, "hour"), (60, "minute"), (1, "second")):
        if seconds >= size:
            number = -(-seconds // size)
            return f"{number} {unit}" if number == 1 else f"{number} {unit}s"
    return "1 second"


class LimitExceededError(Exception):
    """A request refused by a limit: 429, a plain sentence, and how long to wait."""

    def __init__(self, limit: Limit, retry_after: int) -> None:
        self.retry_after = max(1, retry_after)
        self.detail = _SENTENCES[limit.name].format(
            wait=_wait(self.retry_after),
            count=limit.count,
            analyses="analysis" if limit.count == 1 else "analyses",
        )
        super().__init__(self.detail)

    def response(self) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": self.detail},
            headers={"Retry-After": str(self.retry_after)},
        )


def _allow_through_outage(exc: RedisError) -> Decision:
    global _outage_logged, _refusal_logged
    if isinstance(exc, ResponseError):
        # Redis was reached and answered: it would not run the command. That is
        # a fault in the deployment, not an outage. One known cause is a Redis
        # older than 7, which has no `EXPIRE ... NX` (see `FixedWindow.hit`).
        # Others are a server out of memory that evicts nothing, a read-only
        # replica, and a key holding something that is not a count. The
        # message does not guess, and does not carry the exception's text:
        # redis-py adds the whole command to it, and a sign-in count's key
        # holds a client's address. The type name and the server's own log
        # say which it is.
        if not _refusal_logged:
            _refusal_logged = True
            logger.error(
                "Redis refused the rate limiter's command (%s), so the rate limits "
                "are counting nothing and allowing every request. The server's own "
                "log says why. This is logged once.",
                type(exc).__name__,
            )
        return Decision(True, 0)
    if not _outage_logged:
        _outage_logged = True
        logger.warning(
            "Redis could not be reached (%s), so the rate limits are allowing every "
            "request. This is logged once.",
            type(exc).__name__,
        )
    return Decision(True, 0)


class FixedWindow:
    """Counters that Redis expires. One round trip a call, and never an error."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def hit(self, key: str, limit: Limit) -> Decision:
        """Count one against `key` and say whether it was within the limit.

        `INCR`, `EXPIRE ... NX` and `TTL` in one `MULTI`: the count is taken and
        judged in a single atomic step, so two requests cannot both take the last
        place, and the key cannot be left without an expiry by a client that dies
        between two commands. `NX` sets the expiry only on the hit that opens the
        window, so later hits, refused ones included, do not push its end back.
        """
        name = KEY_PREFIX + key
        try:
            pipeline = self._redis.pipeline(transaction=True)
            pipeline.incr(name)
            # `EXPIRE ... NX` needs Redis 7. An older server refuses the whole
            # transaction, the increment with it, which is logged as an error.
            pipeline.expire(name, limit.window_seconds, nx=True)
            pipeline.ttl(name)
            count, _, ttl = pipeline.execute()
        except RedisError as exc:
            return _allow_through_outage(exc)
        if int(count) <= limit.count:
            return Decision(True, 0)
        return Decision(False, _seconds_left(int(ttl), limit))

    def peek(self, key: str, limit: Limit) -> Decision:
        """Whether `key` has already used its whole allowance, counting nothing."""
        name = KEY_PREFIX + key
        try:
            pipeline = self._redis.pipeline(transaction=True)
            pipeline.get(name)
            pipeline.ttl(name)
            count, ttl = pipeline.execute()
        except RedisError as exc:
            return _allow_through_outage(exc)
        if int(count or 0) < limit.count:
            return Decision(True, 0)
        return Decision(False, _seconds_left(int(ttl), limit))

    def require(self, key: str, limit: Limit | None) -> None:
        """Count one against `key`, and raise `LimitExceededError` if that is one too many."""
        if limit is None:
            return
        decision = self.hit(key, limit)
        if not decision.allowed:
            raise LimitExceededError(limit, decision.retry_after)


def _seconds_left(ttl: int, limit: Limit) -> int:
    # Redis answers 0 in a key's last fraction of a second and a negative number
    # for a key with no expiry; neither is something to tell a client to wait.
    if ttl < 0:
        return limit.window_seconds
    return max(1, ttl)


def client_address(request: HTTPConnection) -> str:
    """The address a failed sign-in is counted against.

    In production nothing but the dashboard can reach the API, and the dashboard
    passes the browser's address on in a header of its own; the first address in
    it is believed if it parses as one. Anywhere else the header is whatever the
    caller chose to send, so it is ignored and the connection's peer is used.
    """
    if is_production():
        claimed = request.headers.get(CLIENT_ADDRESS_HEADER, "").split(",", 1)[0].strip()
        try:
            return str(ipaddress.ip_address(claimed))
        except ValueError:
            pass
    return request.client.host if request.client is not None else UNKNOWN_ADDRESS


def _signin_guarded(path: str) -> bool:
    # The health check takes no key, so it has no sign-in to fail or to refuse.
    return path.startswith("/v1/") and path != "/v1/healthz"


class SigninFailures:
    """Counts each 401 against its client address, and refuses an address over the limit.

    An address that has used its allowance is answered 429 before the request
    reaches the app, so its key is never looked at; that check reads the count
    and adds nothing to it. A 401 is counted after the response has been sent.
    The Redis calls run in the thread pool, not on the event loop.

    The limit is not exact for requests made at the same time. The count is
    read on the way in and added to on the way out, so every request that
    arrives before the first 401 has been counted is let through: one address
    gets as many guesses in a window as it can hold open at once, not `count`.
    That is accepted. A key is a 12-character id and a 256-bit secret, which
    no such multiple makes guessable, and taking a place on the way in would
    make every request with a good key write to Redis.
    """

    def __init__(self, app: ASGIApp, window: FixedWindow, limit: Limit) -> None:
        self._app = app
        self._window = window
        self._limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _signin_guarded(scope["path"]):
            await self._app(scope, receive, send)
            return
        key = f"signin:{client_address(HTTPConnection(scope))}"
        # Read now, counted after the answer: see the class's note on requests
        # made at the same time.
        decision = await run_in_threadpool(self._window.peek, key, self._limit)
        if not decision.allowed:
            refusal = LimitExceededError(self._limit, decision.retry_after).response()
            await refusal(scope, receive, send)
            return
        status = 0

        async def watch(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        await self._app(scope, receive, watch)
        if status == 401:
            await run_in_threadpool(self._window.hit, key, self._limit)
