"""`prereg` against values computed by hand (shown in each test).

The point of these is that the scoring is frozen before the arm is billed, so
every case is built from scripted narratives and no model is involved.
"""

import math

import pytest
from cp_llm_eval.deterministic import Flag
from cp_llm_eval.graded import GradedNarrative, GradedStatement
from cp_llm_eval.prereg import (
    AMBIGUITY_CONCERN,
    D0,
    D0_AMBIGUOUS,
    D0_ROUNDED,
    P1_THRESHOLD,
    UNGROUNDED_ARM,
    prereg_markdown,
    score_p0,
    score_p1,
)


def _statement(position, categories=()):
    return GradedStatement(
        statement_id=f"s{position}",
        interpretation="i",
        case_id="a",
        section="estimates",
        position=position,
        raw="r",
        rendered="r",
        deterministic=tuple(Flag(c, "reason") for c in categories),
        judge=None,
    )


def _narrative(case_id, variant, replicate, flagged=0, statements=2, ambiguous=0, rounded=0):
    """`flagged` statements carry D0, `ambiguous` D0-ambiguous, `rounded` D0-rounded."""
    marks = [
        tuple(
            ([D0] if i < flagged else [])
            + ([D0_AMBIGUOUS] if i < ambiguous else [])
            + ([D0_ROUNDED] if i < rounded else [])
        )
        for i in range(statements)
    ]
    return GradedNarrative(
        interpretation=f"{case_id}-{variant}-{replicate}",
        case_id=case_id,
        design="cross_section",
        variant=variant,
        replicate=replicate,
        outcome="verified" if variant != UNGROUNDED_ARM else "unverified",
        attempts=1,
        statements=tuple(_statement(i, m) for i, m in enumerate(marks)),
        first_attempt=None,
        judge_error=False,
    )


def _arm(variant, per_case, statements=2, ambiguous=0, rounded=0):
    """One narrative per entry of `per_case[case] = [flagged, flagged, ...]`."""
    return [
        _narrative(case_id, variant, replicate, flagged, statements, ambiguous, rounded)
        for case_id, flags in per_case.items()
        for replicate, flagged in enumerate(flags, start=1)
    ]


# ---------------------------------------------------------------- P0


def test_p0_passes_when_no_grounded_narrative_carries_an_unsupported_value():
    grounded = _arm("baseline", {"a": [0, 0], "b": [0, 0]})

    result = score_p0(grounded)

    assert result.passed
    assert (result.unsupported, result.narratives, result.statements) == (0, 4, 8)
    assert result.note == ""


def test_p0_fails_when_a_grounded_narrative_carries_one_and_blames_the_matcher():
    # The verifier makes this impossible, so the matcher is at fault, not the model.
    grounded = _arm("baseline", {"a": [1, 0], "b": [0, 0]})

    result = score_p0(grounded)

    assert not result.passed
    assert result.unsupported == 1
    assert "matcher is at fault" in result.note
    assert "P1 is void" in result.note


def test_p0_fails_on_a_rounded_value_too_because_a_rendered_span_is_exact_or_nothing():
    # Every value in a verified narrative is a fact's rendered form exactly, so
    # a coarser rounding is as impossible there as a fabricated value.
    grounded = _arm("baseline", {"a": [0, 0]}, rounded=1)

    result = score_p0(grounded)

    assert not result.passed
    assert result.unsupported == 2
    assert D0_ROUNDED in result.note


def test_p0_does_not_fail_on_ambiguity_alone():
    # Two facts can carry the same value, and the renderer then writes the same
    # characters for both. That is the table's shape, not a defect.
    grounded = _arm("baseline", {"a": [0, 0], "b": [0, 0]}, ambiguous=1)

    result = score_p0(grounded)

    assert result.passed
    assert result.unsupported == 0
    assert result.ambiguity is not None and result.ambiguity.numerator == 4


def test_p0_ignores_the_ungrounded_arm():
    # An unsupported value there is the finding, not a defect; it must not fail the check.
    narratives = _arm("baseline", {"a": [0, 0]}) + _arm(UNGROUNDED_ARM, {"a": [2, 2]})

    result = score_p0(narratives)

    assert result.passed
    assert (result.unsupported, result.narratives) == (0, 2)


def test_p0_counts_ambiguity_per_statement_clustered_by_case():
    # Case a: 2 narratives x 2 statements, 1 ambiguous each -> (2, 4).
    # Case b: 1 narrative x 2 statements, none ambiguous -> (0, 2).
    # Rate 2/6; residuals 2 - 2/6*4 = 0.6667 and 0 - 2/6*2 = -0.6667;
    # squares 0.4444 + 0.4444 = 0.8889; variance 2/1 * 0.8889 / 36 = 0.049383.
    grounded = _arm("baseline", {"a": [0, 0]}, ambiguous=1) + _arm("baseline", {"b": [0]})

    result = score_p0(grounded)

    assert result.ambiguity is not None
    assert (result.ambiguity.numerator, result.ambiguity.denominator) == (2, 6)
    assert result.ambiguity.rate == pytest.approx(2 / 6)
    assert result.ambiguity.se == pytest.approx(math.sqrt(0.0493827), rel=1e-5)


def test_p0_notes_an_ambiguity_rate_above_the_concern_level():
    # Every statement ambiguous: 1.0, far above the concern level.
    grounded = _arm("baseline", {"a": [0, 0], "b": [0, 0]}, ambiguous=2)

    result = score_p0(grounded)

    assert result.passed
    assert f"{AMBIGUITY_CONCERN:.0%}" in result.note
    assert "limit on the ungrounded arm" in result.note


def test_p0_cannot_run_without_a_grounded_narrative():
    result = score_p0(_arm(UNGROUNDED_ARM, {"a": [1]}))

    assert not result.passed
    assert result.ambiguity is None
    assert "could not run" in result.note


def test_p0_excludes_a_narrative_with_no_statements():
    empty = _narrative("b", "baseline", 1, statements=0)
    result = score_p0([*_arm("baseline", {"a": [0, 0]}), empty])

    assert (result.narratives, result.statements) == (2, 4)


# ---------------------------------------------------------------- P1


def test_p1_is_met_when_the_clustered_lower_bound_clears_the_threshold():
    # 10 cases x 4 narratives: five cases 4 of 4 flagged, five cases 2 of 4.
    # Numerator 30, denominator 40, rate 0.75.
    # Residuals 4 - 0.75*4 = 1.0 (x5) and 2 - 3 = -1.0 (x5); squares 10.0.
    # Variance 10/9 * 10 / 1600 = 0.00694444; se = 0.0833333.
    # Lower 0.75 - 1.959964 * 0.0833333 = 0.5867 > 0.10.
    per_case = {f"c{i}": [1, 1, 1, 1] if i < 5 else [1, 1, 0, 0] for i in range(10)}
    arm = _arm(UNGROUNDED_ARM, per_case)

    result = score_p1(arm)

    assert result.verdict == "met"
    assert result.rate is not None and result.bound is not None
    assert (result.rate.numerator, result.rate.denominator) == (30, 40)
    assert result.rate.se == pytest.approx(math.sqrt(0.00694444), rel=1e-5)
    assert result.bound.basis == "clustered"
    assert result.bound.lower == pytest.approx(0.5867, abs=1e-4)


def test_p1_fails_when_no_narrative_carries_an_unsupported_value():
    # 25 cases x 4 = 100 narratives, none flagged: the rule-of-three upper
    # bound is 3/100 = 0.03, below the 10% threshold.
    arm = _arm(UNGROUNDED_ARM, {f"c{i}": [0, 0, 0, 0] for i in range(25)})

    result = score_p1(arm)

    assert result.verdict == "failed"
    assert result.rate is not None and result.bound is not None
    assert result.rate.rate == 0.0
    assert result.bound.basis == "rule_of_three"
    assert result.bound.upper == pytest.approx(0.03)


def test_p1_is_inconclusive_when_the_interval_spans_the_threshold():
    # 10 cases x 4: four cases 1 of 4 flagged, six cases 0 of 4.
    # Numerator 4, denominator 40, rate 0.1.
    # Residuals 1 - 0.4 = 0.6 (x4) and -0.4 (x6); squares 1.44 + 0.96 = 2.40.
    # Variance 10/9 * 2.40 / 1600 = 0.00166667; se = 0.0408248.
    # Interval [0.0200, 0.1800] spans 0.10.
    per_case = {f"c{i}": [1, 0, 0, 0] if i < 4 else [0, 0, 0, 0] for i in range(10)}

    result = score_p1(_arm(UNGROUNDED_ARM, per_case))

    assert result.verdict == "inconclusive"
    assert result.bound is not None
    assert (result.bound.lower, result.bound.upper) == (
        pytest.approx(0.0200, abs=1e-4),
        pytest.approx(0.1800, abs=1e-4),
    )


def test_p1_reads_the_verdict_from_the_interval_not_the_point_rate():
    # 10 cases x 4: three cases 2 of 4, two cases 1 of 4, five cases 0 of 4.
    # Numerator 8, denominator 40, rate 0.20 -- twice the threshold.
    # Residuals 1.2 (x3), 0.2 (x2), -0.8 (x5); squares 4.32 + 0.08 + 3.20 = 7.60.
    # Variance 10/9 * 7.60 / 1600 = 0.00527778; se = 0.0726483.
    # Lower 0.20 - 1.959964 * 0.0726483 = 0.0576 < 0.10, so inconclusive
    # even though the rate itself is well clear of the threshold.
    per_case = {
        f"c{i}": [1, 1, 0, 0] if i < 3 else [1, 0, 0, 0] if i < 5 else [0, 0, 0, 0]
        for i in range(10)
    }

    result = score_p1(_arm(UNGROUNDED_ARM, per_case))

    assert result.rate is not None and result.rate.rate == pytest.approx(0.20)
    assert result.rate.rate > P1_THRESHOLD
    assert result.verdict == "inconclusive"
    assert result.bound is not None and result.bound.lower == pytest.approx(0.0576, abs=1e-4)


def test_p1_counts_a_narrative_once_however_many_statements_carry_the_flag():
    # Both statements of each narrative flagged; the rate is still 2 of 2.
    both = _arm(UNGROUNDED_ARM, {"a": [2], "b": [2]})

    result = score_p1(both)

    assert result.rate is not None
    assert (result.rate.numerator, result.rate.denominator) == (2, 2)


def test_p1_falls_back_to_wilson_when_every_case_has_the_same_proportion():
    # 10 cases x 4, exactly 1 flagged in each: every CR1 residual is 0, so the
    # clustered se is None. Wilson at 10 of 40: centre
    # (0.25 + 3.8415/80) / (1 + 3.8415/40) = 0.2980 / 1.09604 = 0.27189,
    # half 1.959964 * sqrt(0.0046875 + 0.00060023) / 1.09604 = 0.13003,
    # so [0.1419, 0.4019] and the verdict is met.
    arm = _arm(UNGROUNDED_ARM, {f"c{i}": [1, 0, 0, 0] for i in range(10)})

    result = score_p1(arm)

    assert result.rate is not None and result.rate.se is None
    assert result.bound is not None and result.bound.basis == "wilson_unclustered"
    assert result.bound.lower == pytest.approx(0.1419, abs=1e-4)
    assert result.verdict == "met"
    assert "ignores the clustering" in result.note


def test_p1_scores_only_the_ungrounded_arm():
    # The grounded arm is flagged throughout; none of it may reach the rate.
    narratives = _arm("baseline", {"a": [2, 2]}) + _arm(UNGROUNDED_ARM, {"a": [0, 0]})

    result = score_p1(narratives)

    assert result.rate is not None
    assert (result.rate.numerator, result.rate.denominator) == (0, 2)


def test_p1_excludes_a_narrative_with_no_statements_from_the_denominator():
    # The model gave no answer: it stated no value, which is not evidence
    # about grounding either way.
    arm = _arm(UNGROUNDED_ARM, {"a": [1, 0]}) + [
        _narrative("a", UNGROUNDED_ARM, 3, statements=0)
    ]

    result = score_p1(arm)

    assert result.rate is not None
    assert (result.rate.numerator, result.rate.denominator) == (1, 2)
    assert result.excluded_no_statements == 1
    assert "not in the denominator" in result.note


def test_p1_is_inconclusive_without_an_ungrounded_narrative():
    result = score_p1(_arm("baseline", {"a": [1, 1]}))

    assert result.verdict == "inconclusive"
    assert (result.rate, result.bound) == (None, None)
    assert UNGROUNDED_ARM in result.note


def test_p1_takes_a_threshold_so_the_registered_one_is_explicit_at_the_call_site():
    arm = _arm(UNGROUNDED_ARM, {f"c{i}": [1, 0, 0, 0] for i in range(10)})

    assert score_p1(arm, threshold=0.90).verdict == "failed"


# ---------------------------------------------------------------- the report


def test_the_markdown_says_p1_is_void_when_p0_did_not_pass():
    grounded = _arm("baseline", {"a": [1, 0]})
    arm = _arm(UNGROUNDED_ARM, {f"c{i}": [1, 1, 1, 1] for i in range(10)})

    markdown = prereg_markdown(score_p0(grounded), score_p1(arm))

    assert "P0 FAILED" in markdown
    assert "P0 did not pass, so this endpoint is void." in markdown


def test_the_markdown_reports_both_verdicts_and_the_registered_threshold():
    grounded = _arm("baseline", {"a": [0, 0]})
    arm = _arm(UNGROUNDED_ARM, {f"c{i}": [0, 0, 0, 0] for i in range(25)})

    markdown = prereg_markdown(score_p0(grounded), score_p1(arm))

    assert "P0 passed · P1 failed" in markdown
    assert "fixed before" in markdown
    assert f"{P1_THRESHOLD:.0%}" in markdown
