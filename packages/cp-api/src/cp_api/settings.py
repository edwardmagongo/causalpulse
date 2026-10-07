"""What a production deployment must not start with.

`CP_ENV=production` is set by the production compose file and by nothing else.
With it, the API, the worker and the migration call
`require_production_settings` before they contact anything, and exit if the
database password is one this repository publishes. Without it nothing here
does anything: the demo stack and the gates start as they always have.

This module imports no other module of `cp_api`, so the worker can call it
without loading the app.
"""

from __future__ import annotations

import os

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

# Every database password that appears in this repository, and the empty one.
# `tests/api/test_settings.py` reads the demo compose file and the Makefile and
# fails if either carries a password that is missing here.
DEMO_DATABASE_PASSWORDS: frozenset[str] = frozenset({"", "causalpulse"})


def is_production() -> bool:
    return os.environ.get("CP_ENV") == "production"


def _database_password() -> str | None:
    """The password in `DATABASE_URL`, decoded; None if there is none to read."""
    try:
        return make_url(os.environ.get("DATABASE_URL", "")).password
    except ArgumentError:
        return None


def require_production_settings() -> None:
    """Exit, in production, unless the database password is the deployment's own.

    The message names the setting and never its value: it goes to a log.
    """
    if not is_production():
        return
    password = _database_password()
    if password is None or password in DEMO_DATABASE_PASSWORDS:
        raise SystemExit(
            "DATABASE_URL has no password, or one published as a demo value. "
            "CP_ENV=production does not start with it: set the deployment's own."
        )
