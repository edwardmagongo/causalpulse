"""API keys: their format, and the request dependency that checks them.

A key is `cpk_<key_id>_<secret>`. The key id is sent in the clear and is the
`api_keys` primary key. Only a SHA-256 of the secret is stored, so a leaked
database cannot be replayed as working keys. The secret is random rather than
chosen by a person, so a fast hash is enough; a slow password hash would buy
nothing here.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import string
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from cp_api.db import ApiKey, User, utcnow
from cp_api.deps import SessionDep

KEY_PREFIX = "cpk_"
KEY_ID_LENGTH = 12
_KEY_ID_ALPHABET = string.ascii_letters + string.digits
# The key id never contains `_`, so the separator after it is unambiguous; the
# secret is everything after it and may itself contain `_` or `-`.
_KEY = re.compile(rf"{KEY_PREFIX}([A-Za-z0-9]{{{KEY_ID_LENGTH}}})_([A-Za-z0-9_-]+)")


def generate_key() -> tuple[str, str, str]:
    """Return `(key_id, secret, full_key)`. The full key is shown to its owner once."""
    key_id = "".join(secrets.choice(_KEY_ID_ALPHABET) for _ in range(KEY_ID_LENGTH))
    secret = secrets.token_urlsafe(32)
    return key_id, secret, f"{KEY_PREFIX}{key_id}_{secret}"


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


# Compared against when the key id is unknown, so that response timing does
# not reveal which key ids exist.
DUMMY_SHA256 = hash_secret("no key has this secret")


def parse_key(raw: str) -> tuple[str, str]:
    """Split a full key into `(key_id, secret)`, or raise `ValueError`."""
    match = _KEY.fullmatch(raw)
    if match is None:
        raise ValueError("not a CausalPulse API key")
    return match.group(1), match.group(2)


def secret_matches(secret: str, stored_sha256: str) -> bool:
    return hmac.compare_digest(hash_secret(secret), stored_sha256)


UNAUTHORIZED = "missing or invalid API key"


def _unauthorized() -> HTTPException:
    # One response for every failure, so it never says which check failed.
    return HTTPException(401, UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})


def current_user(
    session: SessionDep, authorization: Annotated[str | None, Header()] = None
) -> User:
    """The user whose key authenticated this request, or 401."""
    scheme, _, credentials = (authorization or "").partition(" ")
    if scheme.lower() != "bearer":
        raise _unauthorized()
    try:
        key_id, secret = parse_key(credentials.strip())
    except ValueError:
        raise _unauthorized() from None
    key = session.get(ApiKey, key_id)
    # Compare even for an unknown key id, so timing does not reveal which ids exist.
    matches = secret_matches(secret, key.secret_sha256 if key is not None else DUMMY_SHA256)
    if key is None or not matches or key.revoked_at is not None:
        raise _unauthorized()
    key.last_used_at = utcnow()
    session.commit()
    user = session.get(User, key.user_id)
    assert user is not None  # api_keys.user_id is a NOT NULL foreign key
    return user


# Module scope, like SessionDep: a local alias breaks under postponed annotations.
CurrentUser = Annotated[User, Depends(current_user)]
