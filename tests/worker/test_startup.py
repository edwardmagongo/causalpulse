"""`python -m cp_worker` refuses demo secrets in production before it starts."""

import pytest
from cp_worker import __main__ as entry


def test_the_worker_refuses_before_it_reaches_redis(monkeypatch):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("the worker was built")

    monkeypatch.setattr(entry, "make_worker", _must_not_run)
    monkeypatch.setenv("CP_ENV", "production")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://causalpulse:causalpulse@postgres:5432/causalpulse"
    )
    # Unset, so that getting as far as Redis would be a KeyError, not an exit.
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(SystemExit) as refusal:
        entry.main()

    assert "DATABASE_URL" in str(refusal.value)
