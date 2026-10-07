import json

import pytest
from cp_llm_eval.deterministic import Flag
from cp_llm_eval.graded import GradedNarrative, grade_records
from cp_llm_eval.judge import VERDICTS
from cp_llm_eval.labels import Label
from cp_llm_eval.report import (
    PRIMARY_ENDPOINT,
    _comparison,
    _rate_text,
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
    assert (judge["sensitivity"]["estimate"], judge["sensitivity"]["n"]) == (1.0, 1)
    assert (judge["specificity"]["estimate"], judge["specificity"]["n"]) == (1.0, 1)
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
    }
    return rate | changes


def test_every_rate_states_how_many_cases_it_spans():
    assert _rate_text(_rate()) == "12.3% (34/276, SE 2.1%, 25 cases)"
    assert _rate_text(_rate(clusters=1, se=None)) == "12.3% (34/276, 1 case)"
    zero = _rate(numerator=0, rate=0.0, se=0.0, upper_bound=0.011)
    assert _rate_text(zero) == "0% (at most 1.1%, 0/276, 25 cases)"
    assert _rate_text(zero | {"clusters": 1}) == "0% (at most 1.1%, 0/276, 1 case)"


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
    return precision["estimate"], precision["n"]


def test_a_grader_is_credited_only_for_a_misreading_of_its_own_category(entry, stored, graded):
    assert _d4_precision(entry, stored, graded, "J1") == (0.0, 1)
    assert _d4_precision(entry, stored, graded, "D4") == (1.0, 1)


def test_a_variant_run_under_two_models_is_refused(entry, stored, graded):
    records = stored.records(INTERPRETATIONS)
    records[0] = {**records[0], "key": {**records[0]["key"], "model": "another-model"}}

    with pytest.raises(ValueError, match=records[0]["key"]["variant"]):
        build_report([entry], graded, [], records, stored.records(VERDICTS))
