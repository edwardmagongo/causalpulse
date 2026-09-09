import pytest
from cp_report.language import ReportLanguageError, require_clean_prose


def test_clean_prose_is_accepted():
    require_clean_prose(
        [
            ("validity band", "The band scores observable diagnostics, not truth."),
            ("limitations", "If the parallel-trends assumption holds, the estimate is usable."),
        ]
    )


def test_an_unhedged_causal_claim_is_refused():
    with pytest.raises(ReportLanguageError):
        require_clean_prose([("limitations", "The treatment causes the observed lift.")])


def test_the_refusal_names_the_section_the_term_and_the_sentence():
    # A refusal saying only "the report failed the lint" would send a user
    # hunting through their own data for the sentence at fault.
    with pytest.raises(ReportLanguageError) as excinfo:
        require_clean_prose(
            [
                ("validity band", "The band scores observable diagnostics, not truth."),
                ("assumption table", "This proves unconfoundedness."),
            ]
        )

    message = str(excinfo.value)
    assert "assumption table" in message
    assert "proves" in message
    assert "This proves unconfoundedness." in message


def test_a_negated_absolute_is_accepted():
    # The honest form of this project's prose contains the banned words under
    # negation, which is exactly what the lint is built to allow.
    require_clean_prose(
        [("assumption table", "A balance table does not prove unconfoundedness.")]
    )


def test_interpolated_text_is_linted_too():
    # This is the risk the phase adds. Every string constant in the repository
    # is already scanned by tests/core/test_language.py; no covariate name
    # ever has been.
    covariate = "proven_uplift"

    with pytest.raises(ReportLanguageError):
        require_clean_prose([("assumption table", f"Balance on {covariate} is within threshold.")])


def test_a_hedge_spans_a_fragment_boundary_that_lacks_terminal_punctuation():
    # Neither fragment ends its sentence with punctuation, so under the real
    # "\n\n" separator they read as one continuous sentence and the hedge in
    # the first fragment covers the causal claim completed by the second --
    # exactly the shape of "If the parallel-trends assumption holds, the
    # policy causes a 2-point lift" split across two report sections. Under
    # a separator ending in a sentence terminator (". "), the fragments
    # would instead form two separate sentences, the hedge would no longer
    # reach the second, and this would wrongly raise. This pins design point
    # 1: the lint must run over the concatenation, not fragment by fragment.
    require_clean_prose(
        [
            ("validity band", "If the assumption holds, the estimate"),
            ("interpretation", "causes a measurable shift."),
        ]
    )


def test_every_violation_is_reported_not_only_the_first():
    # Fixing one and re-running to find the next is a bad loop to put a user in.
    with pytest.raises(ReportLanguageError) as excinfo:
        require_clean_prose(
            [
                ("assumption table", "This proves unconfoundedness."),
                ("limitations", "The rollout causes the lift."),
            ]
        )

    message = str(excinfo.value)
    assert "proves" in message
    assert "causes" in message
