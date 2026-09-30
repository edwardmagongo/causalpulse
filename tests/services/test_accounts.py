import pytest
from cp_api.accounts import AccountError, create_user, issue_key, revoke_key
from cp_api.auth import hash_secret, parse_key
from cp_api.db import ApiKey, User


def test_issue_key_stores_only_the_hash_of_a_parseable_key(sessions):
    with sessions() as session:
        create_user(session, "alice")
        full_key = issue_key(session, "alice")
        session.commit()

    key_id, secret = parse_key(full_key)
    with sessions() as session:
        row = session.get(ApiKey, key_id)
        assert row is not None
        assert row.secret_sha256 == hash_secret(secret)
        assert secret not in row.secret_sha256
        assert session.get(User, row.user_id).name == "alice"
        assert row.revoked_at is None


def test_a_duplicate_user_name_is_refused(sessions):
    with sessions() as session:
        create_user(session, "alice")
        with pytest.raises(AccountError, match="already exists"):
            create_user(session, "alice")


@pytest.mark.parametrize("name", ["", "   ", "x" * 65])
def test_a_blank_or_overlong_name_is_refused(sessions, name):
    with sessions() as session, pytest.raises(AccountError):
        create_user(session, name)


def test_issuing_a_key_for_an_unknown_user_is_refused(sessions):
    with sessions() as session, pytest.raises(AccountError, match="no user named 'bob'"):
        issue_key(session, "bob")


def test_revoke_key_sets_revoked_at_once_and_is_idempotent(sessions):
    with sessions() as session:
        create_user(session, "alice")
        key_id, _ = parse_key(issue_key(session, "alice"))
        revoke_key(session, key_id)
        first = session.get(ApiKey, key_id).revoked_at
        revoke_key(session, key_id)
        assert first is not None
        assert session.get(ApiKey, key_id).revoked_at == first


def test_revoking_an_unknown_key_is_refused(sessions):
    with sessions() as session, pytest.raises(AccountError, match="no API key"):
        revoke_key(session, "AAAAAAAAAAAA")
