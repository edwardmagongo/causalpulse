"""API keys: their format now, and the request dependency that checks them.

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
