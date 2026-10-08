"""`cp-api-admin`: create users, issue keys, revoke keys.

It talks to the database directly (through `DATABASE_URL`) rather than over
HTTP, so there is no privileged network endpoint to protect. In the demo
stack: `docker compose exec api cp-api-admin issue-key alice`.

`issue-key` prints only the key on stdout, so a script can capture it; every
other message goes to stderr.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from cp_api.accounts import AccountError, create_user, issue_key, revoke_key
from cp_api.db import make_engine, make_sessionmaker
from cp_api.settings import require_production_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cp-api-admin", description="Manage CausalPulse API users and keys."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-user", help="Create a user.")
    create.add_argument("name")
    issue = commands.add_parser("issue-key", help="Issue a key; it is printed once.")
    issue.add_argument("name")
    revoke = commands.add_parser("revoke-key", help="Revoke a key by its id.")
    revoke.add_argument("key_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    require_production_settings()
    engine = make_engine()
    try:
        with make_sessionmaker(engine)() as session:
            if args.command == "create-user":
                create_user(session, args.name)
                session.commit()
                print(f"created user {args.name!r}", file=sys.stderr)
            elif args.command == "issue-key":
                key = issue_key(session, args.name)
                session.commit()
                print(key)
            else:
                revoke_key(session, args.key_id)
                session.commit()
                print(f"revoked key {args.key_id!r}", file=sys.stderr)
    except AccountError as exc:
        print(f"cp-api-admin: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


def entry() -> None:
    raise SystemExit(main())
