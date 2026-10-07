"""The start-up check that refuses demo secrets when `CP_ENV=production`."""

import re
from pathlib import Path

import pytest
from cp_api.settings import (
    DEMO_DATABASE_PASSWORDS,
    is_production,
    require_production_settings,
)
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parent.parent.parent

STRONG = "Zq7-vN2xLw9_TfB4hKc8RsJd"


def _url(password: str) -> str:
    return f"postgresql+psycopg://causalpulse:{password}@postgres:5432/causalpulse"


NO_PASSWORD = "postgresql+psycopg://causalpulse@postgres:5432/causalpulse"


def test_the_demo_password_and_the_empty_one_are_both_listed():
    assert {"", "causalpulse"} <= DEMO_DATABASE_PASSWORDS


def test_production_is_exactly_cp_env_production(monkeypatch):
    monkeypatch.delenv("CP_ENV", raising=False)
    assert not is_production()
    for value in ("", "development", "Production", "prod", "1", "true"):
        monkeypatch.setenv("CP_ENV", value)
        assert not is_production(), value
    monkeypatch.setenv("CP_ENV", "production")
    assert is_production()


@pytest.mark.parametrize("password", sorted(DEMO_DATABASE_PASSWORDS))
def test_each_demo_password_is_refused_in_production(monkeypatch, password):
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", _url(password))

    with pytest.raises(SystemExit) as refusal:
        require_production_settings()

    message = str(refusal.value)
    assert "DATABASE_URL" in message
    # The empty string is in every message; any other value must be in none.
    if password:
        assert password not in message


@pytest.mark.parametrize("password", sorted(DEMO_DATABASE_PASSWORDS))
def test_each_demo_password_is_accepted_without_cp_env(monkeypatch, password):
    monkeypatch.delenv("CP_ENV", raising=False)
    monkeypatch.setenv("DATABASE_URL", _url(password))

    require_production_settings()


def test_nothing_is_required_without_cp_env(monkeypatch):
    monkeypatch.delenv("CP_ENV", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    require_production_settings()


def test_a_strong_password_is_accepted_in_production(monkeypatch):
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", _url(STRONG))

    require_production_settings()


@pytest.mark.parametrize(
    "url",
    [NO_PASSWORD, _url(""), None, "", "not a url"],
    ids=["no-password", "empty-password", "unset", "empty-url", "unparseable"],
)
def test_a_url_with_no_password_is_refused_in_production(monkeypatch, url):
    monkeypatch.setenv("CP_ENV", "production")
    if url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", url)

    with pytest.raises(SystemExit) as refusal:
        require_production_settings()

    assert "DATABASE_URL" in str(refusal.value)


def test_a_demo_password_is_recognised_however_the_url_encodes_it(monkeypatch):
    # `%63` is `c`: the password is still the demo one. A check that split the
    # string instead of parsing the URL would let this through.
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", _url("%63ausalpulse"))

    with pytest.raises(SystemExit):
        require_production_settings()


def test_the_message_never_carries_the_url(monkeypatch):
    # A refused URL may be a real one with a mistake in it.
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://someone@db.internal.example/app")

    with pytest.raises(SystemExit) as refusal:
        require_production_settings()

    message = str(refusal.value)
    assert "someone" not in message and "db.internal.example" not in message


# --- where the check runs ---------------------------------------------------


def test_the_api_refuses_before_it_reads_anything_else(monkeypatch):
    from cp_api.app import create_app

    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", _url("causalpulse"))
    # Unset, so that getting as far as Redis would be a KeyError, not an exit.
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(SystemExit) as refusal:
        create_app()

    assert "DATABASE_URL" in str(refusal.value)


def test_an_app_given_its_sessions_reads_no_database_url(monkeypatch):
    # The tests and `cp-api-openapi` build the app this way; the check belongs
    # to the path that reads `DATABASE_URL`.
    from cp_api.app import create_app

    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    create_app(sessions=object(), redis=object())


def test_the_migration_refuses_before_it_connects(monkeypatch):
    from cp_api import migrate

    def _must_not_run(*args, **kwargs):
        raise AssertionError("the migration ran")

    monkeypatch.setattr(migrate, "upgrade", _must_not_run)
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", _url("causalpulse"))

    with pytest.raises(SystemExit) as refusal:
        migrate.main()

    assert "DATABASE_URL" in str(refusal.value)


# --- the list is complete ---------------------------------------------------

_DATABASE_URL = re.compile(r"postgresql(?:\+\w+)?://[^\s\"']+")
_POSTGRES_PASSWORD = re.compile(r"^\s*POSTGRES_PASSWORD:\s*(.*?)\s*$", re.MULTILINE)


def _passwords_in(name: str) -> list[str | None]:
    text = (ROOT / name).read_text(encoding="utf-8")
    passwords = [make_url(url).password for url in _DATABASE_URL.findall(text)]
    # `$(...)` and `${...}` are filled in when the file is used: a password
    # generated or read from the environment is not one the repository holds.
    return [password for password in passwords if not (password or "").startswith("$")]


def test_every_database_password_in_the_repository_is_on_the_list():
    # The files that configure a stack someone runs: the demo compose file and
    # the Makefile's gates. The README and `docs/` quote the same URLs, and are
    # not read: prose that mentions a value starts no service with it, and a
    # document may quote a value precisely to say it is refused.
    compose = _passwords_in("docker-compose.yml")
    makefile = _passwords_in("Makefile")
    server = _POSTGRES_PASSWORD.findall((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    # Each file is known to carry one; finding none means the pattern is stale.
    assert compose and makefile and server

    found = {*compose, *makefile, *(value.strip("\"'") for value in server)}
    found = {password for password in found if not (password or "").startswith("$")}
    assert None not in found
    assert found <= DEMO_DATABASE_PASSWORDS, found - DEMO_DATABASE_PASSWORDS
