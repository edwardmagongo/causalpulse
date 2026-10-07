import json

import pytest
from cp_llm_eval.deterministic import Flag
from cp_llm_eval.graded import GradedNarrative, GradedStatement, grade_records
from cp_llm_eval.judge import VERDICTS, Finding
from cp_llm_eval.labels import Label
from cp_llm_eval.report import (
    PRIMARY_ENDPOINT,
    _comparison,
    _difference_text,
    _rate_text,
    _validation,
    build_report,
    render_markdown,
    write_report,
)
from cp_llm_eval.runner import INTERPRETATIONS


@pytest.fixture(scope="module")
def graded(entry, stored):
    return grade_records(
        [entry], stored.records(INTERPRETATIONS), stored.records(VERDICTS), "claude-opus-5-5"
    )


@pytest.fixture(scope="module")
def report(entry, stored, graded):
    baseline = [n for n in graded if n.variant == "baseline" and n.replicate == 1][0]
    first, second = baseline.statements[0], baseline.statements[1]
    labels = [
        Label(first.statement_id, "misreading", "J1", ""),  # the judge flagged it: a hit
        Label(second.statement_id, "ok", "", ""),  # the judge did not: a correct pass
    ]
    return build_report(
        [entry], graded, labels, stored.records(INTERPRETATIONS), stored.records(VERDICTS)
    )


def test_the_run_counts_each_outcome_and_every_call(report):
    variants = report["run"]["variants"]

    assert variants["baseline"] | {"cost": 0} == {
        "interpretations": 2,
        "verified": 2,
        "refused": 0,
        "no_answer": 0,
        "verified_first_attempt": 2,
        "calls": 2,
        "cost": 0,
    }
    variant = variants["cite-each-status"]
    assert (variant["refused"], variant["calls"]) == (1, 3)
    assert variants["baseline"]["cost"] == pytest.approx(2 * (1000 * 4e-6 + 100 * 20e-6))
    assert report["run"]["judge"]["verdicts"] == 3


def test_a_judge_flag_on_one_statement_of_each_narrative_is_rated_both_ways(report, graded):
    rate = report["rates"]["baseline"]["J1"]
    statements = sum(len(n.statements) for n in graded if n.variant == "baseline")

    assert (rate["per_narrative"]["numerator"], rate["per_narrative"]["denominator"]) == (2, 2)
    assert (rate["per_statement"]["numerator"], rate["per_statement"]["denominator"]) == (
        2,
        statements,
    )


def test_a_category_never_seen_carries_the_rule_of_three(report):
    rate = report["rates"]["baseline"]["D4"]["per_narrative"]

    assert (rate["numerator"], rate["se"]) == (0, None)
    # 3 / 2 narratives, capped: two narratives bound nothing.
    assert rate["upper_bound"] == 1.0


def test_the_uncited_summary_is_counted_once_and_graded_true(report):
    assert report["rates"]["cite-each-status"]["D4"]["per_narrative"]["numerator"] == 1
    assert report["d4_truth"]["cite-each-status"] == {"true": 1, "false": 0, "undetermined": 0}


def test_first_attempts_include_the_refused_one(report):
    rate = report["first_attempt"]["cite-each-status"]["D4"]

    assert (rate["numerator"], rate["denominator"]) == (1, 2)


def test_the_judge_is_measured_against_the_labels(report):
    judge = report["validation"]["judge"]["J1"]

    assert report["validation"]["labelled"] == 2
    assert (judge["sensitivity"]["estimate"], judge["sensitivity"]["labelled"]) == (1.0, 1)
    assert (judge["specificity"]["estimate"], judge["specificity"]["labelled"]) == (1.0, 1)
    assert report["validation"]["judge"]["J2"]["sensitivity"] is None


def test_one_case_is_too_few_to_compare(report):
    (comparison,) = report["comparison"]

    assert comparison["primary_endpoint"] == PRIMARY_ENDPOINT
    assert comparison["primary"] is None


def _narrative(case_id, variant, replicate, d4_first):
    flags = ((Flag("D4", "summary", "true"),),) if d4_first else ((),)
    return GradedNarrative(
        interpretation=f"{case_id}-{variant}-{replicate}",
        case_id=case_id,
        design="cross_section",
        variant=variant,
        replicate=replicate,
        outcome="refused",
        attempts=2,
        statements=(),
        first_attempt=flags,
        judge_error=False,
    )


def test_the_primary_endpoint_is_the_paired_difference_in_first_attempt_d4():
    # Case a: baseline 2/2, variant 0/2 (difference -1). Case b: baseline 1/2,
    # variant 1/2 (difference 0). Mean -0.5; sd of (-1, 0) is sqrt(0.5); se 0.5.
    narratives = [
        _narrative("a", "baseline", 1, True),
        _narrative("a", "baseline", 2, True),
        _narrative("a", "v", 1, False),
        _narrative("a", "v", 2, False),
        _narrative("b", "baseline", 1, True),
        _narrative("b", "baseline", 2, False),
        _narrative("b", "v", 1, True),
        _narrative("b", "v", 2, False),
    ]

    comparison = _comparison(narratives, "v")

    assert comparison["primary"]["mean"] == pytest.approx(-0.5)
    assert comparison["primary"]["se"] == pytest.approx(0.5)
    assert comparison["primary"]["pairs"] == 2
    assert comparison["exploratory"]["D4"] is None


def test_the_markdown_states_every_section_and_no_missing_value(report):
    text = render_markdown(report)

    for heading in (
        "## The run",
        "## Final narratives: baseline",
        "## By design",
        "## First attempts",
        "## The graders against the labels",
        "## Comparison: cite-each-status minus baseline",
    ):
        assert heading in text
    assert "None" not in text
    assert PRIMARY_ENDPOINT in text


def test_the_report_is_written_as_markdown_and_json(report, tmp_path):
    markdown, data = write_report(report, tmp_path)

    assert markdown.read_text() == render_markdown(report)
    assert json.loads(data.read_text()) == json.loads(json.dumps(report))


def test_a_label_on_the_variants_statement_measures_the_judge_too(entry, stored, graded):
    variant = [n for n in graded if n.variant == "cite-each-status" and n.statements][0]
    labels = [Label(variant.statements[0].statement_id, "misreading", "J1", "")]

    report = build_report(
        [entry], graded, labels, stored.records(INTERPRETATIONS), stored.records(VERDICTS)
    )

    assert report["validation"]["labelled"] == 1


def _rate(**changes):
    rate = {
        "numerator": 34,
        "denominator": 276,
        "clusters": 25,
        "rate": 0.123,
        "se": 0.021,
        "upper_bound": None,
        "lower_bound": None,
    }
    return rate | changes


def test_every_rate_states_how_many_cases_it_spans():
    assert _rate_text(_rate()) == "12.3% (34/276, SE 2.1%, 25 cases)"
    assert _rate_text(_rate(clusters=1, se=None)) == "12.3% (34/276, 1 case)"
    zero = _rate(numerator=0, rate=0.0, se=0.0, upper_bound=0.011)
    assert _rate_text(zero) == "0% (at most 1.1%, 0/276, 25 cases)"
    assert _rate_text(zero | {"clusters": 1}) == "0% (at most 1.1%, 0/276, 1 case)"


def test_a_unanimous_rate_states_its_lower_bound_not_a_standard_error():
    unanimous = _rate(numerator=50, denominator=50, rate=1.0, se=None, lower_bound=0.94)

    assert _rate_text(unanimous) == "100% (at least 94.0%, 50/50, 25 cases)"


def test_a_difference_with_no_spread_states_it_has_no_interval():
    difference = {"mean": 0.0, "se": 0.0, "lower": 0.0, "upper": 0.0, "pairs": 7,
                  "degenerate": True}

    assert _difference_text(difference) == "+0.0 points (no spread across 7 cases; no interval)"


def _d4_precision(entry, stored, graded, category):
    flagged = [
        s
        for n in graded
        for s in n.statements
        if any(f.category == "D4" for f in s.deterministic)
    ]
    labels = [Label(flagged[0].statement_id, "misreading", category, "")]
    report = build_report(
        [entry], graded, labels, stored.records(INTERPRETATIONS), stored.records(VERDICTS)
    )
    precision = report["validation"]["graders"]["D4"]["precision"]
    return precision["estimate"], precision["labelled"]


def test_a_grader_is_credited_only_for_a_misreading_of_its_own_category(entry, stored, graded):
    assert _d4_precision(entry, stored, graded, "J1") == (0.0, 1)
    assert _d4_precision(entry, stored, graded, "D4") == (1.0, 1)


def test_a_variant_run_under_two_models_is_refused(entry, stored, graded):
    records = stored.records(INTERPRETATIONS)
    records[0] = {**records[0], "key": {**records[0]["key"], "model": "another-model"}}

    with pytest.raises(ValueError, match=records[0]["key"]["variant"]):
        build_report([entry], graded, [], records, stored.records(VERDICTS))


def _statement(position, judge=(), d4=False):
    return GradedStatement(
        statement_id=f"s{position}",
        interpretation="i",
        case_id="a",
        section="caveats",
        position=position,
        raw="r",
        rendered="r",
        deterministic=(Flag("D4", "summary", "true"),) if d4 else (),
        judge=judge,
    )


def test_the_judge_and_graders_are_weighted_by_stratum():
    # Population: 2 judge-flagged (stratum 0), 4 grader-only (1), 10 unflagged (2).
    # Labelled: both of stratum 0 (weight 2/2 = 1), two of stratum 1 (4/2 = 2),
    # two of stratum 2 (10/2 = 5).
    j1 = (Finding(category="J1", reason="r"),)
    statements = (
        _statement(0, j1, d4=True), _statement(1, j1),
        *(_statement(p, d4=True) for p in range(2, 6)),
        *(_statement(p) for p in range(6, 16)),
    )
    narrative = GradedNarrative(
        interpretation="i", case_id="a", design="cross_section", variant="baseline",
        replicate=1, outcome="verified", attempts=1, statements=statements,
        first_attempt=((),) * 16, judge_error=False,
    )
    labels = [
        Label("s0", "misreading", "J1", ""),  # flagged by the judge, w 1
        Label("s1", "ok", "", ""),  # flagged by the judge, w 1
        Label("s2", "misreading", "D4", ""),  # grader only, w 2
        Label("s3", "ok", "", ""),  # grader only, w 2
        Label("s6", "misreading", "J1", ""),  # missed by the judge, w 5
        Label("s7", "ok", "", ""),  # passed by the judge, w 5
    ]

    validation = _validation([narrative], labels)

    j1_figures = validation["judge"]["J1"]
    # Sensitivity: positives s0 (hit, w 1) and s6 (miss, w 5): 1/6, unweighted 1/2.
    # n_eff = (1 + 5)^2 / (1 + 25) = 36/26.
    sensitivity = j1_figures["sensitivity"]
    assert sensitivity["estimate"] == pytest.approx(1 / 6)
    assert sensitivity["n_eff"] == pytest.approx(36 / 26)
    assert sensitivity["labelled"] == 2
    # Specificity: negatives s1 (flagged, w 1), s2 and s3 (not, w 2 each),
    # s7 (not, w 5): (2 + 2 + 5) / 10 = 0.9, unweighted 3/4.
    specificity = j1_figures["specificity"]
    assert specificity["estimate"] == pytest.approx(0.9)
    assert specificity["n_eff"] == pytest.approx(100 / 34)
    assert specificity["labelled"] == 4
    # D4 precision: flagged s0 (J1, not D4: w 1), s2 (D4: w 2), s3 (ok: w 2):
    # 2/5, unweighted 1/3.
    precision = validation["graders"]["D4"]["precision"]
    assert precision["estimate"] == pytest.approx(0.4)
    assert precision["n_eff"] == pytest.approx(25 / 9)
    assert precision["labelled"] == 3
    assert validation["labelled"] == 6


def test_the_markdown_says_the_validation_is_weighted_by_stratum(report):
    text = render_markdown(report)

    assert "weighted by stratum" in text
    assert "n_eff" in text


def _first_attempt_narratives():
    # Cases a and b paired as in the primary-endpoint test; case c is baseline
    # only (not shared); in case d the variant has no first attempt (shared,
    # but not paired for the primary endpoint).
    narratives = [
        _narrative("a", "baseline", 1, True),
        _narrative("a", "v", 1, False),
        _narrative("b", "baseline", 1, True),
        _narrative("b", "v", 1, True),
        _narrative("c", "baseline", 1, True),
        _narrative("d", "baseline", 1, False),
    ]
    missing = _narrative("d", "v", 1, False)
    return narratives + [GradedNarrative(**{**missing.__dict__, "first_attempt": None})]


def test_the_report_counts_narratives_with_no_first_attempt(entry):
    report = build_report([entry], _first_attempt_narratives(), [], [], [])

    assert report["first_attempt_missing"] == {"baseline": 0, "v": 1}
    assert "with no first attempt, excluded from these figures: baseline 0, v 1" in (
        render_markdown(report)
    )


def test_each_comparison_states_how_many_shared_cases_it_paired(entry):
    report = build_report([entry], _first_attempt_narratives(), [], [], [])

    (comparison,) = report["comparison"]
    assert comparison["shared_cases"] == 3
    assert comparison["paired_cases"]["primary"] == 2
    assert comparison["paired_cases"]["D4"] == 0
    text = render_markdown(report)
    assert "2 of 3 shared cases" in text
    assert "| D4 | - | 0 of 3 shared cases |" in text


def test_statements_of_an_unjudged_narrative_do_not_enter_the_weights():
    # The same population and labels as above, plus a verified narrative the
    # judge could not grade: its statements are not in the population the
    # sample draws from, so the weights, and every figure, are unchanged.
    import dataclasses

    j1 = (Finding(category="J1", reason="r"),)
    statements = (
        _statement(0, j1, d4=True), _statement(1, j1),
        *(_statement(p, d4=True) for p in range(2, 6)),
        *(_statement(p) for p in range(6, 16)),
    )
    judged = GradedNarrative(
        interpretation="i", case_id="a", design="cross_section", variant="baseline",
        replicate=1, outcome="verified", attempts=1, statements=statements,
        first_attempt=((),) * 16, judge_error=False,
    )
    unjudged_statements = tuple(
        dataclasses.replace(
            _statement(p, judge=None, d4=p < 6), statement_id=f"u{p}", interpretation="u"
        )
        for p in range(20)
    )
    unjudged = dataclasses.replace(
        judged, interpretation="u", replicate=2, statements=unjudged_statements,
        first_attempt=((),) * 20, judge_error=True,
    )
    labels = [
        Label("s0", "misreading", "J1", ""),
        Label("s1", "ok", "", ""),
        Label("s2", "misreading", "D4", ""),
        Label("s3", "ok", "", ""),
        Label("s6", "misreading", "J1", ""),
        Label("s7", "ok", "", ""),
    ]

    assert not unjudged.judged
    assert _validation([judged, unjudged], labels) == _validation([judged], labels)
