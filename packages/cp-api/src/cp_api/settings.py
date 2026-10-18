"""What a production deployment must not start with.

`CP_ENV=production` is set by the production compose file and by nothing else.
With it, the API, the worker, the migration and `cp-api-admin` call
`require_production_settings` before they contact anything, and exit if the
database password is one this repository publishes. Without it nothing here
does anything: the demo stack and the gates start as they always have.

This module imports no other module of `cp_api`, so the worker can call it
without loading the app.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy.engine import make_url

# Every database password that appears in this repository, and the empty one.
# `tests/api/test_settings.py` reads the demo compose file and the Makefile and
# fails if either carries a password that is missing here.
DEMO_DATABASE_PASSWORDS: frozenset[str] = frozenset({"", "causalpulse"})


def is_production() -> bool:
    return os.environ.get("CP_ENV") == "production"


def interpretation_enabled() -> bool:
    """Whether this server writes interpretations: `CP_INTERPRETATION=on` and nothing else.

    Read by the API, which offers them, and by the worker, which writes them
    and then needs the model's key. Off unless asked for, because each one is a
    paid call on the server's key.
    """
    return os.environ.get("CP_INTERPRETATION") == "on"


# The efforts a model can be asked for. `cp_llm.model.Effort` is the source;
# this module may not import `cp_llm`, so `tests/api/test_interpretation_options.py`
# pins the two together.
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

_DEFAULT_OPTIONS = "claude-opus-5-5:medium"
_MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]*$")


@dataclass(frozen=True)
class InterpretationOption:
    """One model at one effort, as the operator offers it."""

    model: str
    effort: str

    @property
    def label(self) -> str:
        return f"{self.model}, {self.effort}"


# The width of the `model` column an interpretation is stored in.
MODEL_ID_LIMIT = 64
_SHOWN_LIMIT = 80


def _shown(entry: str) -> str:
    """The entry as one line: cut when long, a non-printable character written as an escape."""
    cut = entry if len(entry) <= _SHOWN_LIMIT else entry[:_SHOWN_LIMIT] + "..."
    return "".join(
        c if c.isprintable() else c.encode("unicode_escape").decode("ascii") for c in cut
    )


def _refused_option(entry: str, why: str) -> ValueError:
    return ValueError(f"CP_INTERPRETATION_OPTIONS has a wrong entry '{_shown(entry)}': {why}.")


def interpretation_options(
    environ: Mapping[str, str] | None = None,
) -> tuple[InterpretationOption, ...]:
    """The models and efforts this server offers for an interpretation, in the operator's order.

    `CP_INTERPRETATION_OPTIONS` is a comma-separated list of `model:effort`;
    the first is the default. Absent or empty, it is `claude-opus-5-5:medium`.
    Whitespace around an entry, a model or an effort is ignored. A model is
    lower-case letters, digits, dots and hyphens, starting with a letter or a
    digit, at most `MODEL_ID_LIMIT` long; an effort is one of `EFFORTS`; no entry appears twice.

    Raises `ValueError` with one line naming the setting and the first wrong
    entry, so the API and the worker can refuse to start on it.
    """
    value = (os.environ if environ is None else environ).get("CP_INTERPRETATION_OPTIONS", "")
    if not value.strip():
        value = _DEFAULT_OPTIONS
    options: list[InterpretationOption] = []
    seen: set[InterpretationOption] = set()
    for raw in value.split(","):
        entry = raw.strip()
        if not entry:
            raise _refused_option(entry, "empty entry (a stray comma?)")
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) != 2:
            raise _refused_option(entry, "not model:effort")
        model, effort = parts
        if not _MODEL_NAME.fullmatch(model):
            raise _refused_option(
                entry,
                "the model must be lower-case letters, digits, dots and hyphens, "
                "starting with a letter or a digit",
            )
        if len(model) > MODEL_ID_LIMIT:
            raise _refused_option(entry, f"model id longer than {MODEL_ID_LIMIT} characters")
        if effort not in EFFORTS:
            raise _refused_option(entry, f"unknown effort, one of {', '.join(EFFORTS)}")
        option = InterpretationOption(model, effort)
        if option in seen:
            raise _refused_option(entry, "repeated")
        seen.add(option)
        options.append(option)
    return tuple(options)


_REFUSED = "CP_ENV=production does not start with it"


def require_production_settings() -> None:
    """Exit, in production, unless the database password is the deployment's own.

    The password has one place, before the `@`. A `password` query parameter
    is refused whatever it holds, because the driver uses it in preference to
    that one (measured with psycopg: `u:strong@h/db?password=demo` connects
    with `demo`), so a check that read only the first would pass a URL that
    connects on the second.

    Each message names the setting and never its value: it goes to a log.
    """
    if not is_production():
        return
    try:
        url = make_url(os.environ.get("DATABASE_URL", ""))
    except Exception:
        # Whatever the parser objects to (`ArgumentError` for most, a bare
        # `ValueError` for a port that is not a number), and without the
        # parser's own message, which may quote the URL.
        raise SystemExit(f"DATABASE_URL is unset or cannot be read as a URL. {_REFUSED}.") from None
    if any(key.lower() == "password" for key in url.query):
        raise SystemExit(
            f"DATABASE_URL carries a `password` query parameter. {_REFUSED}: "
            "put the password before the `@` and nowhere else."
        )
    if url.password is None or url.password in DEMO_DATABASE_PASSWORDS:
        raise SystemExit(
            f"DATABASE_URL has no password, or one published as a demo value. {_REFUSED}: "
            "set the deployment's own."
        )
