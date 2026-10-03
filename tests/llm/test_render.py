import pytest
from cp_llm.facts import Fact, FactTable
from cp_llm.render import (
    UnknownFactError,
    format_fact,
    free_text,
    placeholder_ids,
    substitute,
)


def _fact(kind, value):
    return Fact("est.a.x", kind, value, None)


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("number", 0.123456, "0.1235"),
        ("number", -0.056254, "-0.05625"),
        ("number", 1794.34, "1794"),
        # Four significant figures, and never exponent notation.
        ("number", 12345.6, "12350"),
        ("number", 1.2345e-7, "0.0000001235"),
        ("number", 3.0, "3"),
        ("number", -0.0, "0"),
        ("number", -0.00001, "-0.00001"),
        ("count", 218, "218"),
        ("level", 0.95, "95%"),
        ("level", 0.9, "90%"),
        ("level", 0.995, "99.5%"),
        ("p_value", 0.3672, "0.367"),
        ("p_value", 0.001, "0.001"),
        ("p_value", 0.00099, "< 0.001"),
        ("label", "NOT_TESTABLE", "NOT_TESTABLE"),
        ("name", "difference_in_means", "difference_in_means"),
        ("text", "Checks warned (positivity).", "Checks warned (positivity)."),
    ],
)
def test_each_kind_has_one_format(kind, value, expected):
    assert format_fact(_fact(kind, value)) == expected


def test_placeholder_ids_keeps_order_and_repeats():
    statement = "{est.a.method} gives {est.a.point}; again {est.a.method}."

    assert placeholder_ids(statement) == ("est.a.method", "est.a.point", "est.a.method")


def test_a_format_spec_is_part_of_the_id_and_so_names_no_fact():
    assert placeholder_ids("{est.a.point:.2f}") == ("est.a.point:.2f",)


def test_free_text_is_the_statement_without_its_placeholders():
    assert free_text("{est.a.method} gives {est.a.point}.") == "  gives  ."


def test_substitute_replaces_every_placeholder_with_its_formatted_fact():
    facts = FactTable(
        (
            Fact("est.a.method", "name", "aipw", None),
            Fact("est.a.point", "number", 0.123456, "est.a.method"),
            Fact("est.a.ci_level", "level", 0.95, "est.a.method"),
        )
    )

    rendered = substitute(
        "{est.a.method} estimates {est.a.point} ({est.a.ci_level} interval).", facts
    )

    assert rendered == "aipw estimates 0.1235 (95% interval)."


def test_substitute_refuses_a_placeholder_that_names_no_fact():
    facts = FactTable((Fact("est.a.method", "name", "aipw", None),))

    with pytest.raises(UnknownFactError, match="est.a.power"):
        substitute("{est.a.method} has power {est.a.power}.", facts)


def test_substitute_uses_the_formatter_it_is_given():
    facts = FactTable(
        (
            Fact("est.a.method", "name", "aipw", None),
            Fact("est.a.point", "number", 0.123456, "est.a.method"),
        )
    )

    rendered = substitute(
        "{est.a.method} estimates {est.a.point}.", facts, lambda fact: fact.kind.upper()
    )

    assert rendered == "NAME estimates NUMBER."
