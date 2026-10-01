import hashlib

import pytest
from cp_api.auth import (
    KEY_ID_LENGTH,
    KEY_PREFIX,
    generate_key,
    hash_secret,
    parse_key,
    secret_matches,
)


def test_a_generated_key_round_trips_through_parse_key():
    key_id, secret, full_key = generate_key()

    assert full_key == f"{KEY_PREFIX}{key_id}_{secret}"
    assert parse_key(full_key) == (key_id, secret)
    assert len(key_id) == KEY_ID_LENGTH and key_id.isalnum()


def test_secrets_containing_underscores_and_dashes_still_parse():
    # token_urlsafe produces both; a parser that split on every `_` would
    # reject roughly half of all real keys.
    keys = [generate_key() for _ in range(500)]
    assert any("_" in secret or "-" in secret for _, secret, _ in keys)
    for key_id, secret, full_key in keys:
        assert parse_key(full_key) == (key_id, secret)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "cpk_",
        "xyz_ABCDEFGHIJKL_secret",           # wrong prefix
        "cpk_ABCDEFGHIJK_secret",            # key id too short
        "cpk_ABCDEFGHIJKLM_secret",          # key id too long
        "cpk_ABCDEFGHIJ-L_secret",           # key id not alphanumeric
        "cpk_ABCDEFGHIJKL_",                 # empty secret
        "cpk_ABCDEFGHIJKL",                  # no secret at all
        "cpk_ABCDEFGHIJKL_sec ret",          # whitespace in the secret
        "Bearer cpk_ABCDEFGHIJKL_secret",    # the scheme is the caller's to strip
    ],
)
def test_malformed_keys_are_refused(raw):
    with pytest.raises(ValueError, match="not a CausalPulse API key"):
        parse_key(raw)


def test_hash_secret_is_hex_sha256():
    assert hash_secret("abc") == hashlib.sha256(b"abc").hexdigest()


def test_secret_matches_compares_against_the_stored_hash():
    _, secret, _ = generate_key()
    stored = hash_secret(secret)

    assert secret_matches(secret, stored)
    assert not secret_matches(secret + "x", stored)
