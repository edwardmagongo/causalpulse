"""Creating users, issuing keys, revoking keys: the one place any of it happens.

The `cp-api-admin` CLI and the test fixtures both go through these functions,
so neither carries its own copy. Callers own the transaction: these functions
flush, and never commit.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cp_api.auth import generate_key, hash_secret
from cp_api.db import ApiKey, User, utcnow

_MAX_NAME = 64


class AccountError(Exception):
    """A request the account functions refuse, with a message naming why."""


def _user_named(session: Session, name: str) -> User | None:
    return session.scalars(select(User).where(User.name == name)).one_or_none()


def create_user(session: Session, name: str) -> User:
    if not name.strip():
        raise AccountError("a user name must not be empty")
    if len(name) > _MAX_NAME:
        raise AccountError(f"a user name must be at most {_MAX_NAME} characters")
    if _user_named(session, name) is not None:
        raise AccountError(f"a user named {name!r} already exists")
    user = User(name=name)
    session.add(user)
    session.flush()
    return user


def issue_key(session: Session, name: str) -> str:
    """Issue a key for the named user and return it. It is never retrievable again."""
    user = _user_named(session, name)
    if user is None:
        raise AccountError(f"no user named {name!r}")
    key_id, secret, full_key = generate_key()
    session.add(ApiKey(id=key_id, user_id=user.id, secret_sha256=hash_secret(secret)))
    session.flush()
    return full_key


def revoke_key(session: Session, key_id: str) -> None:
    """Revoke a key. Revoking an already revoked key leaves its first revocation time."""
    key = session.get(ApiKey, key_id)
    if key is None:
        raise AccountError(f"no API key with id {key_id!r}")
    if key.revoked_at is None:
        key.revoked_at = utcnow()
        session.flush()
