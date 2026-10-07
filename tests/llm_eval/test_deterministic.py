"""The deterministic graders against the real scenario D fit (`facts`).

Each test states the real values it relies on first, so a refit that moved
them fails here, loudly, instead of passing for a reason that no longer holds.
"""

import pytest
from cp_llm_eval import deterministic
from cp_llm_eval.deterministic import (
    D1_UNPARSED,
    DETERMINISTIC_CATEGORIES,
    grade_narrative,
    grade_statement,
)
from llm.builders import good_narrative


def _value(facts, fact_id):
    fact = facts.get(fact_id)
    assert fact is not None, fact_id
    return fact.value


def _categories(statement, facts):
    return {flag.category for flag in grade_statement(statement, facts)}


def _statuses(facts, method):
    prefix = f"check.{method}."
    return sorted(
        str(f.value) for f in facts if f.id.startswith(prefix) and f.id.endswith(".status")
    )


@pytest.fixture(scope="module")
def known(facts):
    """The values the planted defects below are written against."""
    assert _value(facts, "est.difference_in_means.point") > _value(facts, "est.ipw.point")
    assert _value(facts, "score.ipw.band") == "WEAK"
    assert _value(facts, "score.ipw.ceiling") == "MODERATE"
    assert "WARN" in _statuses(facts, "ipw")
    assert set(_statuses(facts, "difference_in_means")) == {"NOT_TESTABLE"}
    assert str(_value(facts, "route.bsts.eligible")).startswith("not ")
    return facts


# One planted defect per grader, each caught by its own grader alone.
PLANTED = {
    "D1": (
        "The point for {est.ipw.method}, {est.ipw.point}, is larger than the point for "
        "{est.difference_in_means.method}, {est.difference_in_means.point}."
    ),
    "D2": "The band for {est.ipw.method} is not {score.ipw.band}.",
    "D3": "For {est.ipw.method} the p-value is {est.ipw.se}.",
    "D4": "No check for {est.ipw.method} raised a concern.",
    "D5": (
        "{est.difference_in_means.method} reads {est.difference_in_means.estimand}. "
        "Its band is {score.difference_in_means.band}."
    ),
}


@pytest.mark.parametrize("category", DETERMINISTIC_CATEGORIES)
def test_each_planted_defect_is_caught_by_its_own_grader_and_no_other(category, known):
    assert _categories(PLANTED[category], known) == {category}


@pytest.mark.parametrize("category", DETERMINISTIC_CATEGORIES)
def test_with_its_grader_removed_each_planted_defect_goes_uncounted(category, known, monkeypatch):
    remaining = {k: v for k, v in deterministic.GRADERS.items() if k != category}
    monkeypatch.setattr(deterministic, "GRADERS", remaining)

    assert _categories(PLANTED[category], known) == set()


def test_the_planted_defects_cover_every_grader():
    assert set(PLANTED) == set(deterministic.GRADERS) == set(DETERMINISTIC_CATEGORIES)


def test_phase_ones_verified_narrative_has_no_flag(known):
    grades = grade_narrative(good_narrative(), known)

    assert grades
    assert [g.flags for g in grades if g.flags] == []
    assert [g.position for g in grades] == list(range(len(grades)))


# D1 ------------------------------------------------------------------------


def test_a_true_comparative_is_not_flagged(known):
    statement = (
        "The point for {est.ipw.method}, {est.ipw.point}, is smaller than the point for "
        "{est.difference_in_means.method}, {est.difference_in_means.point}."
    )
    assert _categories(statement, known) == set()


def test_bands_are_compared_in_their_order(known):
    wrong = "For {est.ipw.method} the band {score.ipw.band} sits above {score.ipw.ceiling}."
    right = "For {est.ipw.method} the band {score.ipw.band} sits below {score.ipw.ceiling}."

    assert _categories(wrong, known) == {"D1"}
    assert _categories(right, known) == set()


def test_no_higher_than_is_read_as_at_most(known):
    statement = (
        "For {est.ipw.method} the band {score.ipw.band} is no higher than {score.ipw.ceiling}."
    )
    assert _categories(statement, known) == set()


def test_a_comparative_with_a_value_on_one_side_is_unparsed_not_flagged(known):
    statement = (
        "The {est.ipw.method} estimate is smaller at {est.ipw.point}, with an interval from "
        "{est.ipw.ci_lower} to {est.ipw.ci_upper}."
    )
    assert _categories(statement, known) == {D1_UNPARSED}


def test_a_comparative_with_one_cited_value_is_not_a_comparison(known):
    statement = "For {est.ipw.method} the band can go no higher than {score.ipw.ceiling}."
    assert _categories(statement, known) == set()


def test_a_number_compared_with_a_band_is_unparsed(known):
    statement = "For {est.ipw.method}, {est.ipw.point} is higher than {score.ipw.band}."
    assert _categories(statement, known) == {D1_UNPARSED}


# D2 ------------------------------------------------------------------------


def test_a_label_carrying_its_own_negation_is_exempt(known):
    statement = "The router marked {route.bsts.estimator} as {route.bsts.eligible}."
    assert _categories(statement, known) == set()


def test_a_negated_name_negates_nothing_the_table_states(known):
    statement = "No estimate from {route.aipw.estimator} appears among the results."
    assert _categories(statement, known) == set()


def test_a_negation_further_back_than_three_words_is_not_counted(known):
    statement = (
        "It is not the case that the evidence alone settles it, and the band for "
        "{est.ipw.method} is {score.ipw.band}."
    )
    assert _categories(statement, known) == set()


# D3 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "For {est.ipw.method} the p-value is {est.ipw.p_value}.",
        "For {est.ipw.method} the standard error of the estimate is {est.ipw.se}.",
        "For {est.ipw.method} the interval runs from {est.ipw.ci_lower} to {est.ipw.ci_upper}.",
        "For {est.ipw.method} the point estimate is {est.ipw.point}.",
        # Two label words: the nearer one says what the number is.
        "For {est.ipw.method} the interval of the point {est.ipw.point} is shown.",
    ],
)
def test_a_correctly_labelled_number_is_not_flagged(statement, known):
    assert _categories(statement, known) == set()


@pytest.mark.parametrize(
    "statement",
    [
        "For {est.ipw.method} the standard error is {est.ipw.point}.",
        "For {est.ipw.method} the lower bound is {est.ipw.point}.",
        "For {est.ipw.method} the point estimate is {est.ipw.ci_lower}.",
        "For {est.ipw.method} the interval runs from {est.ipw.ci_upper} to {est.ipw.ci_lower}.",
    ],
)
def test_a_mislabelled_number_or_a_backwards_range_is_flagged(statement, known):
    assert _categories(statement, known) == {"D3"}


# D4 ------------------------------------------------------------------------


def _d4(statement, facts):
    (flag,) = [f for f in grade_statement(statement, facts) if f.category == "D4"]
    return flag.truth


def test_an_uncited_summary_is_graded_true_or_false_against_the_named_methods_checks(known):
    assert _d4("No check for {est.ipw.method} raised a concern.", known) == "false"
    assert _d4("No check for {est.difference_in_means.method} raised a concern.", known) == "true"


def test_a_universal_untested_summary_is_read_against_the_statuses(known):
    dim = "Every check for {est.difference_in_means.method} is untested from the data."
    ipw = "Every check for {est.ipw.method} is untested from the data."

    assert _d4(dim, known) == "true"
    assert _d4(ipw, known) == "false"


def test_a_summary_with_no_recognised_predicate_is_undetermined(known):
    statement = "Every recorded check for {est.ipw.method} reads the same way."
    assert _d4(statement, known) == "undetermined"


def test_a_summary_that_cites_a_status_is_not_flagged(known):
    statement = (
        "No check for {est.ipw.method} other than {check.ipw.positivity.name} raised a "
        "concern, and it has status {check.ipw.positivity.status}."
    )
    assert _categories(statement, known) == set()


@pytest.mark.parametrize(
    ("statement", "truth"),
    [
        ("None of the five robustness checks for {est.ipw.method} raised a concern.", "false"),
        (
            "All three robustness checks for {est.difference_in_means.method} are untested "
            "from the data.",
            "true",
        ),
        ("Not a single check for {est.difference_in_means.method} raised a concern.", "true"),
        ("Not one of the checks for {est.ipw.method} raised a concern.", "false"),
    ],
)
def test_a_quantifier_with_words_before_its_noun_or_read_as_no_is_graded(
    statement, truth, known
):
    assert _d4(statement, known) == truth


# D5 ------------------------------------------------------------------------


def test_a_statement_of_one_sentence_cannot_misattribute_across_sentences(known):
    statement = (
        "{est.difference_in_means.method} reads {est.difference_in_means.estimand} and its "
        "band is {score.difference_in_means.band}."
    )
    assert _categories(statement, known) == set()


def test_a_method_named_in_the_same_sentence_is_attributed(known):
    statement = (
        "{est.ipw.method} reads {est.ipw.estimand}. "
        "{est.difference_in_means.method} has band {score.difference_in_means.band}."
    )
    assert _categories(statement, known) == set()


def test_a_negation_that_belongs_to_a_comparative_is_not_a_negated_value(known):
    # "can go no higher than" was written by the model during phase 1's calibration.
    statement = "For {est.ipw.method} the band can go no higher than {score.ipw.ceiling}."
    assert "D2" not in _categories(statement, known)


_CONTRACTIONS = ("don't", "can't", "won't", "couldn't", "wouldn't", "shouldn't", "hasn't")
_CONTRACTIONS += ("haven't", "isn't", "wasn't", "aren't", "weren't", "doesn't", "didn't")


@pytest.mark.parametrize("apostrophe", ["'", "’"], ids=["straight", "curly"])
@pytest.mark.parametrize("contraction", _CONTRACTIONS)
def test_every_contraction_negates_a_cited_value(known, contraction, apostrophe):
    word = contraction.replace("'", apostrophe)
    statement = f"The band for {{est.ipw.method}} {word} {{score.ipw.band}}."
    assert _categories(statement, known) == {"D2"}


def test_a_negated_universal_is_flagged_but_undetermined(known):
    statement = "Not every check for {est.ipw.method} passed."
    assert _d4(statement, known) == "undetermined"


def test_all_is_read_against_the_statuses(known):
    statement = "All checks for {est.difference_in_means.method} are untested from the data."
    assert _d4(statement, known) == "true"


def test_each_is_read_against_the_statuses(known):
    assert _d4("Each check for {est.ipw.method} passed.", known) == "false"


def test_a_negated_band_comparison_that_is_false_is_flagged(known):
    statement = (
        "For {est.ipw.method} the band {score.ipw.band} is no lower than {score.ipw.ceiling}."
    )
    assert _categories(statement, known) == {"D1"}
