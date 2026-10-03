import json

import pytest
from cp_llm.facts import FACTS_SCHEMA_VERSION, Fact, FactTable


def _name(fact_id="est.a.method"):
    return Fact(fact_id, "name", "a", None)


@pytest.mark.parametrize("fact_id", ["point", "Est.a.point", "est..point", "est.a.", "est a.point"])
def test_an_id_must_be_dotted_lowercase_segments(fact_id):
    with pytest.raises(ValueError, match="not a fact id"):
        Fact(fact_id, "number", 1.0, None)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown kind"):
        Fact("est.a.point", "ratio", 1.0, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1.0, True, "3"])
def test_a_count_is_an_int_and_not_a_bool(value):
    with pytest.raises(ValueError, match="a count must be an int"):
        Fact("est.a.n_treated", "count", value, None)


@pytest.mark.parametrize("kind", ["number", "level", "p_value"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1, "0.5", True])
def test_a_float_kind_is_a_finite_float(kind, value):
    with pytest.raises(ValueError, match="must be a finite float"):
        Fact("est.a.point", kind, value, None)


@pytest.mark.parametrize("kind", ["label", "name", "text"])
@pytest.mark.parametrize("value", ["", "   ", 1.0])
def test_a_string_kind_is_a_non_empty_string(kind, value):
    with pytest.raises(ValueError, match="must be a non-empty string"):
        Fact("est.a.band", kind, value, None)


def test_a_duplicate_id_is_refused():
    with pytest.raises(ValueError, match="duplicate fact id: est.a.method"):
        FactTable((_name(), _name()))


def test_a_subject_must_name_a_name_fact_in_the_table():
    point = Fact("est.a.point", "number", 1.0, "est.a.method")

    with pytest.raises(ValueError, match="subject 'est.a.method' is not a name fact"):
        FactTable((point,))
    with pytest.raises(ValueError, match="subject 'est.a.se' is not a name fact"):
        FactTable(
            (
                Fact("est.a.se", "number", 0.1, None),
                Fact("est.a.point", "number", 1.0, "est.a.se"),
            )
        )
    assert FactTable((_name(), point)).get("est.a.point") is point


def test_get_returns_none_for_an_id_the_table_does_not_hold():
    assert FactTable((_name(),)).get("est.b.method") is None


def test_json_round_trips_exactly_and_keeps_an_integral_float_a_float():
    table = FactTable(
        (
            _name(),
            Fact("est.a.point", "number", 2.0, "est.a.method"),
            Fact("est.a.se", "number", 0.1 + 0.2, "est.a.method"),
            Fact("est.a.n_treated", "count", 7, "est.a.method"),
            Fact("score.a.band", "label", "WEAK", "est.a.method"),
        )
    )

    restored = FactTable.from_json(table.to_json())

    assert restored == table
    point = restored.get("est.a.point")
    assert point is not None and isinstance(point.value, float)
    count = restored.get("est.a.n_treated")
    assert count is not None and isinstance(count.value, int)


def test_json_is_strict_and_stable():
    table = FactTable((_name(), Fact("est.a.point", "number", 1.5, "est.a.method")))

    def _refuse(constant):
        raise AssertionError(f"non-standard JSON constant {constant}")

    payload = json.loads(table.to_json(), parse_constant=_refuse)

    assert payload["schema_version"] == FACTS_SCHEMA_VERSION
    assert payload["facts"][1] == {
        "id": "est.a.point", "kind": "number", "value": 1.5, "subject": "est.a.method",
    }
    # `sort_keys` puts a record's keys in alphabetical order; insertion order
    # (id, kind, value, subject) would put `value` before `subject`.
    text = table.to_json()
    record = text[text.index('"id": "est.a.point"') :]
    assert (
        record.index('"id"') < record.index('"kind"') < record.index('"subject"')
        < record.index('"value"')
    )


def test_another_schema_version_is_refused_with_the_mismatch():
    raw = json.dumps({"schema_version": FACTS_SCHEMA_VERSION + 1, "facts": []})

    with pytest.raises(ValueError, match="facts schema_version 2 is not 1") as excinfo:
        FactTable.from_json(raw)
    assert "rerun" not in str(excinfo.value)


_GOOD = {"id": "est.a.method", "kind": "name", "value": "a", "subject": None}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "not a JSON object"),
        ("text", "not a JSON object"),
        (None, "not a JSON object"),
        ({"schema_version": 1}, "no `facts` list"),
        ({"schema_version": 1, "facts": {"a": 1}}, "no `facts` list"),
        ({"schema_version": 1, "facts": ["x"]}, "fact 0 is not an object"),
        ({"schema_version": 1, "facts": [_GOOD, 3]}, "fact 1 is not an object"),
        *[
            (
                {"schema_version": 1, "facts": [_GOOD, {k: v for k, v in _GOOD.items() if k != f}]},
                f"fact 1 has no `{f}`",
            )
            for f in ("id", "kind", "value", "subject")
        ],
    ],
)
def test_a_structurally_bad_payload_is_a_value_error_that_says_what_is_wrong(payload, message):
    with pytest.raises(ValueError, match=message):
        FactTable.from_json(json.dumps(payload))


@pytest.mark.parametrize("raw", ["", '{"schema_version": 1, "facts": ['])
def test_invalid_json_is_a_value_error(raw):
    with pytest.raises(ValueError):
        FactTable.from_json(raw)


def _table_with(**overrides):
    record = {**_GOOD, **overrides}
    return json.dumps({"schema_version": 1, "facts": [_GOOD | {"id": "est.b.method"}, record]})


@pytest.mark.parametrize("fact_id", [5, None, [1]], ids=["int", "null", "list"])
def test_a_non_string_id_in_a_file_is_a_value_error(fact_id):
    with pytest.raises(ValueError, match="not a fact id"):
        FactTable.from_json(_table_with(id=fact_id))


@pytest.mark.parametrize("kind", [["name"], {}], ids=["list", "object"])
def test_an_unhashable_kind_in_a_file_is_a_value_error(kind):
    with pytest.raises(ValueError, match="unknown kind"):
        FactTable.from_json(_table_with(kind=kind))


@pytest.mark.parametrize("subject", [["x"], {}, 3], ids=["list", "object", "int"])
def test_a_non_string_subject_in_a_file_is_a_value_error(subject):
    with pytest.raises(ValueError, match="subject"):
        FactTable.from_json(_table_with(subject=subject))


@pytest.mark.parametrize("fact_id", [5, None, ["est.a.point"]], ids=["int", "null", "list"])
def test_a_fact_with_a_non_string_id_is_refused(fact_id):
    with pytest.raises(ValueError, match="not a fact id"):
        Fact(fact_id, "number", 1.0, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("subject", [3, ["x"], {}], ids=["int", "list", "object"])
def test_a_fact_with_a_non_string_subject_is_refused(subject):
    with pytest.raises(ValueError, match="est.a.point: subject"):
        Fact("est.a.point", "number", 1.0, subject)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["number", "level", "p_value"])
def test_an_integer_too_large_for_a_float_is_a_value_error(kind):
    raw = _table_with(id="est.a.point", kind=kind, value=10**400)

    with pytest.raises(ValueError, match="fact 1.*not a finite number"):
        FactTable.from_json(raw)
