"""The start-up check that refuses demo secrets when `CP_ENV=production`."""

import re
from pathlib import Path
from urllib.parse import unquote

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
    [
        NO_PASSWORD,
        _url(""),
        None,
        "",
        "not a url",
        # The parser raises a bare ValueError for this one, not its own error.
        f"postgresql+psycopg://causalpulse:{STRONG}@postgres:abc/causalpulse",
    ],
    ids=[
        "no-password",
        "empty-password",
        "unset",
        "empty-url",
        "unparseable",
        "port-not-a-number",
    ],
)
def test_a_url_with_no_password_or_that_cannot_be_read_exits_in_one_line(monkeypatch, url):
    monkeypatch.setenv("CP_ENV", "production")
    if url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", url)

    with pytest.raises(SystemExit) as refusal:
        require_production_settings()

    message = str(refusal.value)
    assert "DATABASE_URL" in message
    assert "\n" not in message and "Traceback" not in message
    assert STRONG not in message and "abc" not in message


@pytest.mark.parametrize("name", ["password", "PASSWORD"])
def test_a_demo_password_in_a_query_parameter_is_refused(monkeypatch, name):
    # The driver is handed the query parameter in preference to the password
    # before the `@`, so this URL connects with the demo one.
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", f"{_url(STRONG)}?{name}=causalpulse")

    with pytest.raises(SystemExit) as refusal:
        require_production_settings()

    message = str(refusal.value)
    assert "DATABASE_URL" in message and "query parameter" in message
    assert "causalpulse" not in message and STRONG not in message


def test_a_strong_password_in_both_places_is_refused_too(monkeypatch):
    # The rule is about the place, not the value: one place to read means one
    # place to check.
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", f"{_url(STRONG)}?password={STRONG}")

    with pytest.raises(SystemExit) as refusal:
        require_production_settings()

    assert STRONG not in str(refusal.value)


def test_a_password_only_in_a_query_parameter_is_refused(monkeypatch):
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", f"{NO_PASSWORD}?password={STRONG}")

    with pytest.raises(SystemExit):
        require_production_settings()


def test_other_query_parameters_are_accepted_in_production(monkeypatch):
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", f"{_url(STRONG)}?sslmode=require&connect_timeout=5")

    require_production_settings()


def test_a_password_query_parameter_is_accepted_without_cp_env(monkeypatch):
    monkeypatch.delenv("CP_ENV", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"{_url(STRONG)}?password=causalpulse")

    require_production_settings()


def test_the_driver_really_does_prefer_the_query_parameter():
    # The fact the rule rests on, read from the dialect this project connects
    # with. If a later SQLAlchemy stops doing this, the rule can be relaxed.
    url = make_url(f"{_url(STRONG)}?password=causalpulse")

    _, arguments = url.get_dialect()().create_connect_args(url)

    assert url.password == STRONG
    assert arguments["password"] == "causalpulse"


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


def test_the_admin_command_refuses_before_it_connects(monkeypatch):
    from cp_api import admin

    def _must_not_run(*args, **kwargs):
        raise AssertionError("an engine was made")

    monkeypatch.setattr(admin, "make_engine", _must_not_run)
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", _url("causalpulse"))

    with pytest.raises(SystemExit) as refusal:
        admin.main(["create-user", "alice"])

    assert "DATABASE_URL" in str(refusal.value)


# --- the list is complete ---------------------------------------------------
#
# The scanner below reads the files that configure a stack someone runs: every
# `docker-compose*.yml` at the root, and the Makefile's gates. The README and
# `docs/` quote the same URLs and are not read: prose that mentions a value
# starts no service with it, and a document may quote one to say it is refused.


class CannotTellError(AssertionError):
    """A value the scanner can neither read as a literal nor rule out as one."""


_BRACED = re.compile(r"\$\{(\w+)(?:(:?[-?+])(.*))?\}", re.DOTALL)
_MAKE_ASSIGNMENT = re.compile(r"^(\w+)[ \t]*(?:::?=|\?=|=)[ \t]*(.*?)[ \t]*$", re.MULTILINE)

# One piece of a value: `${...}`, `$(...)`, `$$(...)`, or a single character.
_PIECE = r"\$\{[^}]*\}|\$\$?\([^)]*\)"
_DATABASE_URL = re.compile(
    r"postgresql(?:\+\w+)?://"
    rf"(?:[^:@/\s\"']*(?::(?P<password>(?:{_PIECE}|[^@\s\"'])*))?@)?"
    r"(?P<rest>[^\s\"']*)"
)
_QUERY_PASSWORD = re.compile(rf"[?&]password=((?:{_PIECE}|[^&\s\"'])*)", re.IGNORECASE)
# `POSTGRES_PASSWORD: x` and `- POSTGRES_PASSWORD=x`, but not the name inside
# `${POSTGRES_PASSWORD:-x}`, which the URL pattern reads, nor `..._FILE`.
_SERVER_PASSWORD = re.compile(
    r"(?<![${\w])(?:POSTGRES_PASSWORD|PGPASSWORD)\b[\"']?[ \t]*[:=][ \t]*"
    rf"(\"[^\"\n]*\"|'[^'\n]*'|(?:{_PIECE}|[^\s\"'#])*)"
)


def make_variables(text: str) -> dict[str, str]:
    return {name: value for name, value in _MAKE_ASSIGNMENT.findall(text)}


def literal(token: str, variables: dict[str, str] | None = None) -> str | None:
    """The literal a value holds, or None if it holds none.

    `variables` is a Makefile's own, and its presence says the value came from
    a Makefile; without it the value came from a compose file.

    None is a reference the file does not fill in: `${NAME}`, `${NAME:?why}`,
    `$NAME`, a command's output. `${NAME:-x}` and `${NAME-x}` hold `x`: that is
    what the stack runs with when nobody sets `NAME`, which is how a new demo
    value would arrive. Anything else with a `$` in it raises `CannotTellError`.
    """
    if "$" not in token:
        return token
    if variables is not None:
        if token.startswith("$$"):
            return None  # the recipe's shell fills it in when the gate runs
        reference = re.fullmatch(r"\$[({](\w+)[)}]", token)
        if reference:
            value = variables.get(reference.group(1))
            if value is None or "$" in value:
                raise CannotTellError(f"{token} is not defined as a plain `NAME := value`")
            return value
        if re.fullmatch(r"\$\(\w+\s[^)]*\)", token):
            return None  # a function call, `$(shell ...)`: its output, not a literal
        raise CannotTellError(token)
    braced = _BRACED.fullmatch(token)
    if braced:
        _, operator, rest = braced.groups()
        if operator is None or operator in ("?", ":?"):
            return None
        if operator in ("-", ":-"):
            return literal(rest)
        raise CannotTellError(token)
    if re.fullmatch(r"\$\w+", token) or re.fullmatch(r"\$\$?\([^)]*\)", token):
        return None
    raise CannotTellError(token)


def database_passwords(text: str, *, makefile: bool = False) -> list[str]:
    """Every literal database password `text` holds, in the order it holds them.

    From a URL: the one before the `@` (a URL without one holds the empty
    password) and a `password` query parameter. From a compose file: what
    `POSTGRES_PASSWORD` is set to.
    """
    variables = make_variables(text) if makefile else None
    tokens: list[tuple[str, bool]] = []  # (value, percent-encoded)
    for match in _DATABASE_URL.finditer(text):
        tokens.append((match.group("password") or "", True))
        tokens.extend((value, True) for value in _QUERY_PASSWORD.findall(match.group("rest")))
    for value in _SERVER_PASSWORD.findall(text):
        quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"
        tokens.append((value[1:-1] if quoted else value, False))

    found = []
    for token, encoded in tokens:
        value = literal(token, variables)
        if value is not None:
            found.append(unquote(value) if encoded else value)
    return found


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # A compose file's mapping form, and its list form.
        ("      POSTGRES_PASSWORD: causalpulse", ["causalpulse"]),
        ('      POSTGRES_PASSWORD: "quoted value"  # a comment', ["quoted value"]),
        ("      - POSTGRES_PASSWORD=listed", ["listed"]),
        ('      - "POSTGRES_PASSWORD=listed"', ["listed"]),
        ("      PGPASSWORD: another", ["another"]),
        # A default is a literal; a reference with none is not.
        ("      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-newdemo}", ["newdemo"]),
        ("      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD-newdemo}", ["newdemo"]),
        ('      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD:-newdemo}"', ["newdemo"]),
        ("      POSTGRES_PASSWORD: ${OUTER:-${INNER:-nested}}", ["nested"]),
        ("      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}", []),
        ("      POSTGRES_PASSWORD: $POSTGRES_PASSWORD", []),
        ("      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?set a password first}", []),
        ("      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD?set a password first}", []),
        # Not a password: a path to one, and a variable passed through by name.
        ("      POSTGRES_PASSWORD_FILE: /run/secrets/db", []),
        ("      - POSTGRES_PASSWORD", []),
        # URLs.
        ("    DATABASE_URL: postgresql+psycopg://u:plain@postgres:5432/db", ["plain"]),
        ("    DATABASE_URL: postgresql://u:pl%61in@postgres/db", ["plain"]),
        ("    DATABASE_URL: postgresql+psycopg://u@postgres:5432/db", [""]),
        ("    DATABASE_URL: postgresql+psycopg://postgres:5432/db", [""]),
        (
            "    DATABASE_URL: postgresql+psycopg://u:${POSTGRES_PASSWORD:-newdemo}@postgres/db",
            ["newdemo"],
        ),
        ("    DATABASE_URL: postgresql+psycopg://u:${POSTGRES_PASSWORD}@postgres/db", []),
        (
            "    DATABASE_URL: postgresql://u:${POSTGRES_PASSWORD:?set a password}@postgres/db",
            [],
        ),
        (
            "    DATABASE_URL: postgresql://u:${POSTGRES_PASSWORD}@h/db?sslmode=x&password=inquery",
            ["inquery"],
        ),
        ("A: postgresql://u:one@h/db B: postgresql://u:two@h/db", ["one", "two"]),
    ],
)
def test_the_scanner_reads_a_compose_line(line, expected):
    assert database_passwords(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "      POSTGRES_PASSWORD: prefix${SUFFIX}",
        "      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:+alternative}",
        "    DATABASE_URL: postgresql://u:${A}${B}@postgres/db",
    ],
)
def test_the_scanner_fails_on_a_compose_value_it_cannot_read(line):
    with pytest.raises(CannotTellError):
        database_passwords(line)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ENV = TEST_DATABASE_URL=postgresql+psycopg://u:plain@localhost:5434/db", ["plain"]),
        # A variable defined plainly is followed, whichever assignment defines it.
        ("PW := followed\nENV = X=postgresql://u:$(PW)@localhost/db", ["followed"]),
        ("PW = followed\nENV = X=postgresql://u:${PW}@localhost/db", ["followed"]),
        ("PW ?= followed\nENV = X=postgresql://u:$(PW)@localhost/db", ["followed"]),
        # The recipe's shell fills these in when the gate runs.
        ("\tX=postgresql://u:$$PW@localhost/db uv run pytest", []),
        ("\tX=postgresql://u:$$(openssl rand -hex 16)@localhost/db uv run pytest", []),
        ("ENV = X=postgresql://u:$(shell openssl rand -hex 16)@localhost/db", []),
    ],
)
def test_the_scanner_reads_a_makefile(text, expected):
    assert database_passwords(text, makefile=True) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ENV = X=postgresql://u:$(PW)@localhost/db",  # defined nowhere
        "PW := $(shell openssl rand -hex 16)\nENV = X=postgresql://u:$(PW)@localhost/db",
        "PW := $(OTHER)\nOTHER := x\nENV = X=postgresql://u:$(PW)@localhost/db",
    ],
)
def test_the_scanner_fails_on_a_makefile_variable_it_cannot_follow(text):
    with pytest.raises(CannotTellError):
        database_passwords(text, makefile=True)


def _compose_files() -> list[Path]:
    return sorted([*ROOT.glob("docker-compose*.yml"), *ROOT.glob("docker-compose*.yaml")])


def test_every_database_password_in_the_repository_is_on_the_list():
    files = _compose_files()
    assert ROOT / "docker-compose.yml" in files

    found: dict[str, list[str]] = {
        path.name: database_passwords(path.read_text(encoding="utf-8")) for path in files
    }
    found["Makefile"] = database_passwords(
        (ROOT / "Makefile").read_text(encoding="utf-8"), makefile=True
    )

    # Both are known to carry one; finding none means the patterns are stale.
    assert found["docker-compose.yml"] and found["Makefile"]

    unlisted = {
        name: sorted(set(passwords) - DEMO_DATABASE_PASSWORDS)
        for name, passwords in found.items()
        if set(passwords) - DEMO_DATABASE_PASSWORDS
    }
    assert not unlisted, unlisted


def test_the_demo_compose_file_sets_the_password_in_both_of_its_places():
    # The URL the services read and the password the server is created with.
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert database_passwords(text) == ["causalpulse", "causalpulse"]
