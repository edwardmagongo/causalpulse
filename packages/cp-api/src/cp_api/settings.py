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

from sqlalchemy.engine import make_url

# Every database password that appears in this repository, and the empty one.
# `tests/api/test_settings.py` reads the demo compose file and the Makefile and
# fails if either carries a password that is missing here.
DEMO_DATABASE_PASSWORDS: frozenset[str] = frozenset({"", "causalpulse"})


def is_production() -> bool:
    return os.environ.get("CP_ENV") == "production"


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
