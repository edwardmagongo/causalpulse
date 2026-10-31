"""`attribution` against the renderer's own output, and against planted defects.

Every expected value here is what `cp_llm.render.format_fact` writes for the
fact in question, since that is the form the matcher promises to recognise.
"""

import pytest
from cp_llm.facts import Fact, FactTable
from cp_llm.render import format_fact
from cp_llm_eval.attribution import (
    D0,
    D0_AMBIGUOUS,
    D0_ROUNDED,
    attribute,
    flags,
)


def _table(*facts):
    return FactTable(tuple(facts))


def _name(fact_id="est.aipw.method", value="aipw"):
    return Fact(fact_id, "name", value, None)


NAME = _name()
POINT = Fact("est.aipw.point", "number", 0.1349, NAME.id)
LEVEL = Fact("est.aipw.ci_level", "level", 0.95, NAME.id)
COUNT = Fact("data.n_units", "count", 4300, None)
P = Fact("est.aipw.p_value", "p_value", 0.0234, NAME.id)
TINY_P = Fact("est.other.p_value", "p_value", 0.0005, NAME.id)

TABLE = _table(NAME, POINT, LEVEL, COUNT, P)


def _outcomes(statement, facts=TABLE):
    return [(span.text, span.outcome, span.fact_ids) for span in attribute(statement, facts).spans]


# ------------------------------------------------- the renderer's own forms


def test_the_renderer_s_own_form_is_an_exact_match():
    # format_fact(POINT) is "0.1349": 0.1349 to four significant figures.
    assert format_fact(POINT) == "0.1349"

    assert _outcomes("The effect is 0.1349 on the outcome.") == [
        ("0.1349", "exact", ("est.aipw.point",))
    ]
    assert flags("The effect is 0.1349 on the outcome.", TABLE) == ()


def test_an_exact_span_is_rewritten_as_the_placeholder_it_cites():
    result = attribute("aipw estimates 0.1349 at 95%.", TABLE)

    assert result.reconstructed == "aipw estimates {est.aipw.point} at {est.aipw.ci_level}."


def test_a_level_matches_the_percentage_the_renderer_writes():
    assert format_fact(LEVEL) == "95%"

    assert _outcomes("a 95% interval") == [("95%", "exact", ("est.aipw.ci_level",))]


def test_a_count_matches_across_thousands_separators():
    # The renderer writes 4300; "4,300" is the same value, not a different one.
    assert format_fact(COUNT) == "4300"

    assert _outcomes("4,300 units") == [("4,300", "exact", ("data.n_units",))]
    assert _outcomes("4300 units") == [("4300", "exact", ("data.n_units",))]


def test_a_p_value_below_the_floor_matches_the_floor_form_however_it_is_spaced():
    table = _table(_name(), TINY_P)
    assert format_fact(TINY_P) == "< 0.001"

    assert _outcomes("p is < 0.001 here", table)[0][1] == "exact"
    assert _outcomes("p is <0.001 here", table)[0][1] == "exact"


def test_a_sentence_final_number_is_still_matched():
    # The decimal part is taken greedily, so the full stop must not end the span.
    assert _outcomes("The effect is 0.1349.") == [("0.1349", "exact", ("est.aipw.point",))]


def test_a_negative_number_is_matched_with_its_sign():
    negative = Fact("est.twfe.point", "number", -1.06, NAME.id)
    table = _table(NAME, negative)

    assert _outcomes("twfe reports -1.06 instead", table) == [
        ("-1.06", "exact", ("est.twfe.point",))
    ]


# ------------------------------------------------- the planted defects


def test_a_fabricated_value_is_unsupported():
    # 0.42 is nowhere in the table at any rounding.
    assert _outcomes("The effect is 0.42 on the outcome.") == [("0.42", "unsupported", ())]
    assert [f.category for f in flags("The effect is 0.42.", TABLE)] == [D0]


def test_more_digits_than_the_table_shows_is_unsupported():
    # The renderer writes POINT to four significant figures, so a reader of a
    # grounded narrative never sees a fifth. A model writing one is not
    # rounding the table's value, it is going past it.
    finer = Fact("est.aipw.point", "number", 0.12345, NAME.id)
    table = _table(NAME, finer)
    # 0.12345 sits just above the midpoint in binary, so four figures round up.
    assert format_fact(finer) == "0.1235"

    assert _outcomes("the effect is 0.12345", table) == [("0.12345", "unsupported", ())]


def test_a_value_belonging_to_another_fact_is_attributed_to_that_fact():
    # The matcher reads values, not intent: a number that is some other fact's
    # value is a citation of that other fact. Naive matching would call this a
    # citation of whichever fact the sentence is about; D5 is what catches the
    # mismatch, and it can only do so because the span is attributed here.
    other = Fact("est.ipw.point", "number", 0.25, _name("est.ipw.method", "ipw").id)
    table = _table(NAME, POINT, _name("est.ipw.method", "ipw"), other)

    result = attribute("aipw estimates 0.25 on the outcome.", table)

    assert result.spans[0].fact_ids == ("est.ipw.point",)
    assert result.spans[0].outcome == "exact"
    assert result.reconstructed == "aipw estimates {est.ipw.point} on the outcome."


# ------------------------------------------------- rounded, and why it is not D0


def test_a_coarser_rounding_is_reported_apart_from_an_unsupported_value():
    # 0.13 is 0.1349 to two significant figures: the quantity is right and the
    # digits are the model's choice. That is a milder defect than inventing a
    # number, so it must not land in the primary endpoint.
    assert _outcomes("about 0.13 on the outcome") == [
        ("0.13", "rounded", ("est.aipw.point",))
    ]
    assert [f.category for f in flags("about 0.13", TABLE)] == [D0_ROUNDED]
    assert D0 not in [f.category for f in flags("about 0.13", TABLE)]


def test_a_rounded_span_is_not_rewritten_as_a_citation():
    # It does not carry the table's value, so D1-D5 must see an uncited number.
    assert attribute("about 0.13 on the outcome", TABLE).reconstructed == (
        "about 0.13 on the outcome"
    )


def test_a_level_restated_as_a_proportion_or_in_words_is_rounded_not_unsupported():
    # "0.95" and a bare "95" (as in "95 percent") are the same quantity as 95%.
    assert _outcomes("a 0.95 interval") == [("0.95", "rounded", ("est.aipw.ci_level",))]
    assert _outcomes("a 95 percent interval") == [("95", "rounded", ("est.aipw.ci_level",))]


def test_a_percentage_cannot_round_to_a_plain_number():
    # Only a level renders with a %, so "13%" is not POINT at two figures.
    assert _outcomes("about 13% on the outcome") == [("13%", "unsupported", ())]


# ------------------------------------------------- ambiguity


def test_a_value_two_facts_share_is_ambiguous():
    twin = Fact("est.ipw.point", "number", 0.1349, _name("est.ipw.method", "ipw").id)
    table = _table(NAME, POINT, _name("est.ipw.method", "ipw"), twin)

    result = attribute("the estimate is 0.1349 here", table)

    assert result.spans[0].outcome == "ambiguous"
    assert set(result.spans[0].fact_ids) == {"est.aipw.point", "est.ipw.point"}
    assert [f.category for f in result.flags] == [D0_AMBIGUOUS]


def test_an_ambiguous_span_is_not_rewritten_as_a_citation():
    twin = Fact("est.ipw.point", "number", 0.1349, _name("est.ipw.method", "ipw").id)
    table = _table(NAME, POINT, _name("est.ipw.method", "ipw"), twin)

    assert attribute("the estimate is 0.1349 here", table).reconstructed == (
        "the estimate is 0.1349 here"
    )


def test_a_rounding_two_facts_share_is_ambiguous_not_rounded():
    near = Fact("est.ipw.point", "number", 0.1312, _name("est.ipw.method", "ipw").id)
    table = _table(NAME, POINT, _name("est.ipw.method", "ipw"), near)

    # 0.13 is both 0.1349 and 0.1312 at two significant figures.
    assert _outcomes("about 0.13", table)[0][1] == "ambiguous"


# ------------------------------------------------- text facts and placeholders


NOTE = Fact("check.aipw.positivity.detail", "text", "trimmed at 0.05 on both tails", NAME.id)


def test_a_number_inside_a_text_fact_is_quoted_not_invented():
    table = _table(NAME, NOTE)

    assert _outcomes("The fit was trimmed at 0.05 on both tails.", table) == [
        ("0.05", "quoted", ("check.aipw.positivity.detail",))
    ]
    # The table holds the number, so it is not D0; but the fact is the
    # sentence, not the number, so there is nothing to flag and nothing to
    # cite -- a placeholder for it would render the whole sentence back.
    assert flags("trimmed at 0.05", table) == ()
    assert attribute("trimmed at 0.05", table).reconstructed == "trimmed at 0.05"


def test_a_prose_fact_does_not_make_the_numbers_it_mentions_ambiguous():
    # The defect this separation exists to prevent: one sentence fact
    # mentioning several values used to match every one of them, so each
    # collided with the fact that actually holds it and the whole statement
    # came back ambiguous -- and an ambiguous span is never reconstructed.
    verdict = Fact("cmp.ate.raw.verdict", "text", "estimates range from 0.1349 to 0.25", None)
    other = Fact("est.ipw.point", "number", 0.25, _name("est.ipw.method", "ipw").id)
    table = _table(NAME, POINT, _name("est.ipw.method", "ipw"), other, verdict)

    assert _outcomes("aipw estimates 0.1349 and ipw 0.25.", table) == [
        ("0.1349", "exact", ("est.aipw.point",)),
        ("0.25", "exact", ("est.ipw.point",)),
    ]


# ------------------------------------------------- the subject resolves a tie


def test_a_shared_value_is_resolved_to_the_fact_whose_subject_is_named():
    # The comparison table restates each point estimate as an end of its
    # range, so the value belongs to two fact ids. The sentence is about ipw,
    # so that is the one it cites -- and now D1-D5 see a citation here.
    restated = Fact("cmp.ate.raw.point_low", "number", 0.1349, None)
    table = _table(NAME, POINT, restated)

    result = attribute("aipw estimates 0.1349 on the outcome.", table)

    assert result.spans[0].outcome == "exact"
    assert result.spans[0].fact_ids == ("est.aipw.point",)
    assert result.reconstructed == "aipw estimates {est.aipw.point} on the outcome."


def test_a_shared_value_stays_ambiguous_when_the_statement_names_neither_subject():
    restated = Fact("cmp.ate.raw.point_low", "number", 0.1349, None)
    table = _table(NAME, POINT, restated)

    result = attribute("the range starts at 0.1349.", table)

    assert result.spans[0].outcome == "ambiguous"
    assert [f.category for f in result.flags] == [D0_AMBIGUOUS]


def test_naming_both_subjects_resolves_nothing():
    # A guess here would hand D1-D5 a citation the model never made.
    twin = Fact("est.ipw.point", "number", 0.1349, _name("est.ipw.method", "ipw").id)
    table = _table(NAME, POINT, _name("est.ipw.method", "ipw"), twin)

    result = attribute("aipw and ipw both estimate 0.1349.", table)

    assert result.spans[0].outcome == "ambiguous"


def test_a_placeholder_s_own_id_is_not_scanned_for_numbers():
    # A raw statement still holds placeholders; `ci_level` must not yield a 5,
    # and `p_value` must not yield anything either.
    raw = "{est.aipw.method} reports {est.aipw.p_value} at {est.aipw.ci_level}."

    assert attribute(raw, TABLE).spans == ()
    assert attribute(raw, TABLE).reconstructed == raw


def test_a_statement_with_no_number_has_no_spans_and_is_unchanged():
    result = attribute("The design is a panel and no check raised a concern.", TABLE)

    assert result.spans == ()
    assert result.reconstructed == result.statement
    assert result.flags == ()


# ------------------------------------------------- several spans at once


def test_every_span_is_classified_and_the_reconstruction_keeps_their_order():
    statement = "aipw estimates 0.1349 at 95% over 4,300 units, but 0.42 elsewhere."

    result = attribute(statement, TABLE)

    assert [span.outcome for span in result.spans] == [
        "exact",
        "exact",
        "exact",
        "unsupported",
    ]
    assert result.reconstructed == (
        "aipw estimates {est.aipw.point} at {est.aipw.ci_level} over {data.n_units} units, "
        "but 0.42 elsewhere."
    )
    assert [f.category for f in result.flags] == [D0]


def test_the_flags_name_the_value_and_the_facts_behind_each_outcome():
    twin = Fact("est.ipw.point", "number", 0.1349, _name("est.ipw.method", "ipw").id)
    table = _table(NAME, POINT, _name("est.ipw.method", "ipw"), twin)

    unsupported = flags("the effect is 0.42", table)[0]
    ambiguous = flags("the effect is 0.1349", table)[0]

    assert "0.42" in unsupported.reason
    assert "est.aipw.point" in ambiguous.reason and "est.ipw.point" in ambiguous.reason


@pytest.mark.parametrize("statement", ["a third of the effect", "twenty units", "no units"])
def test_a_quantity_spelled_in_words_is_not_a_numeric_span(statement):
    # The same gap phase 1's verifier has, recorded rather than papered over.
    assert attribute(statement, TABLE).spans == ()


def test_a_number_the_table_writes_out_in_prose_is_not_called_a_rounding():
    # 0.05 is also 0.0499 at one significant figure, but the table shows
    # "0.05" verbatim, so charging the model with coarsening a value it could
    # have read is an accusation the table itself answers.
    near = Fact("est.aipw.p_value", "p_value", 0.0499, NAME.id)
    table = _table(NAME, near, NOTE)

    assert _outcomes("trimmed at 0.05 on both tails", table) == [
        ("0.05", "quoted", ("check.aipw.positivity.detail",))
    ]
    assert flags("trimmed at 0.05 on both tails", table) == ()
