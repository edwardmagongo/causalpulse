"""Turning a statement's placeholders into the values they cite.

One formatting rule per fact kind, applied here and nowhere else, so a number
reads the same wherever it appears and the model has no say in its rounding.
"""

from __future__ import annotations

import re
from decimal import Decimal

from cp_llm.facts import Fact, FactTable

# Anything between one pair of braces. Deliberately wider than the fact-id
# grammar: `{est.aipw.point:.2f}` must be seen as a placeholder naming no fact,
# not skipped as ordinary text.
PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


class UnknownFactError(KeyError):
    """A placeholder names a fact the table does not hold."""


def placeholder_ids(statement: str) -> tuple[str, ...]:
    """The fact ids a statement cites, in order, repeats kept."""
    return tuple(match.group(1) for match in PLACEHOLDER.finditer(statement))


def free_text(statement: str) -> str:
    """The statement with every placeholder replaced by a space."""
    return PLACEHOLDER.sub(" ", statement)


def _significant(value: float, digits: int) -> str:
    """`value` to `digits` significant figures, never in exponent form.

    `format(12345.6, ".4g")` is `1.235e+04`, which is not prose. Rounding
    through that format and then writing the result positionally keeps the
    same digits.
    """
    rounded = float(format(value, f".{digits}g"))
    text = format(Decimal(repr(rounded)), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def format_fact(fact: Fact) -> str:
    value = fact.value
    if fact.kind == "count":
        return str(value)
    if fact.kind == "number":
        assert isinstance(value, float)
        return _significant(value, 4)
    if fact.kind == "level":
        assert isinstance(value, float)
        return f"{_significant(value * 100, 4)}%"
    if fact.kind == "p_value":
        assert isinstance(value, float)
        return "< 0.001" if value < 0.001 else _significant(value, 3)
    assert isinstance(value, str)
    return value


def substitute(statement: str, facts: FactTable) -> str:
    """`statement` with each placeholder replaced by its fact, formatted."""

    def _replace(match: re.Match[str]) -> str:
        fact = facts.get(match.group(1))
        if fact is None:
            raise UnknownFactError(match.group(1))
        return format_fact(fact)

    return PLACEHOLDER.sub(_replace, statement)
