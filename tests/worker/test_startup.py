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


def test_interpretations_on_without_the_key_refuse_before_redis(monkeypatch):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("the worker was built")

    monkeypatch.setattr(entry, "make_worker", _must_not_run)
    monkeypatch.delenv("CP_ENV", raising=False)
    monkeypatch.setenv("CP_INTERPRETATION", "on")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    with pytest.raises(SystemExit) as refusal:
        entry.main()

    assert "ANTHROPIC_API_KEY" in str(refusal.value)


@pytest.mark.parametrize("setting", [None, "off", "ON", "yes"])
def test_only_the_exact_word_on_asks_for_the_key(monkeypatch, setting):
    from cp_worker.jobs import require_interpretation_settings

    if setting is None:
        monkeypatch.delenv("CP_INTERPRETATION", raising=False)
    else:
        monkeypatch.setenv("CP_INTERPRETATION", setting)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    require_interpretation_settings()


def test_the_refusal_never_prints_a_value(monkeypatch):
    from cp_worker.jobs import require_interpretation_settings

    monkeypatch.setenv("CP_INTERPRETATION", "on")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(SystemExit) as refusal:
        require_interpretation_settings()

    assert str(refusal.value).startswith("CP_INTERPRETATION=on, but ANTHROPIC_API_KEY is not set.")
