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


def test_a_hedge_does_not_span_a_fragment_boundary_that_lacks_terminal_punctuation():
    # A fragment is a hard sentence boundary. An earlier version let a sentence
    # run across fragments, so a hedge -- or a negation -- in one fragment
    # covered a claim completed by the next. Negation and hedge words are
    # exactly what unrelated fragments contain ("cannot share a panel axis"),
    # so the leak licensed unhedged claims. Every real caller passes whole
    # prose fields, none of which is a sentence begun in one and finished in
    # another.
    with pytest.raises(ReportLanguageError):
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


def test_a_duplicate_sentence_is_attributed_to_its_own_section():
    # Two sections carrying the byte-identical offending sentence -- reused
    # boilerplate limitation text is the realistic case. A refusal that sends
    # a user to the wrong section is worse than one that says nothing, so
    # `document.find` searched from position 0 (which always returns the
    # leftmost occurrence) is not good enough: it would attribute both
    # violations to "validity band" and never mention "limitations" at all.
    with pytest.raises(ReportLanguageError) as excinfo:
        require_clean_prose(
            [
                ("validity band", "This proves unconfoundedness."),
                ("limitations", "This proves unconfoundedness."),
            ]
        )

    message = str(excinfo.value)
    assert "[validity band]" in message
    assert "[limitations]" in message


def test_a_duplicate_sentence_in_two_non_initial_fragments_is_attributed_correctly():
    # The residual from the first fix: two non-initial fragments both carry
    # the identical offending sentence. `_SENTENCE` sweeps the preceding
    # "\n\n" separator into every non-initial sentence match, so both
    # violations' raw `sentence` text (and their stripped text) are
    # byte-identical to each other -- unlike the two-fragment case above,
    # where the first fragment is initial and therefore has no swept-in
    # whitespace, no text-based comparison between consecutive violations can
    # tell these two occurrences apart. A fix that only changes *which* text
    # it compares (raw vs. stripped) does not close this: both are equal
    # either way.
    with pytest.raises(ReportLanguageError) as excinfo:
        require_clean_prose(
            [
                ("validity band", "The band scores observable diagnostics, not truth."),
                ("assumption table", "This proves unconfoundedness."),
                ("limitations", "This proves unconfoundedness."),
            ]
        )

    message = str(excinfo.value)
    assert "[assumption table]" in message
    assert "[limitations]" in message


def test_two_offending_terms_in_one_sentence_both_attribute_to_it():
    # Both violations come from the same `_SENTENCE` match, so they carry the
    # identical `LanguageViolation.start` -- attribution must not require
    # distinct positions to tell two terms in one sentence apart from two
    # occurrences of the same sentence. The offending sentence is deliberately
    # placed in the second section (not the first) so that a bug which
    # defaults to the first section on any ambiguity would be visibly wrong.
    with pytest.raises(ReportLanguageError) as excinfo:
        require_clean_prose(
            [
                ("validity band", "A clean opening sentence with no violations."),
                ("limitations", "This proves unconfoundedness and causes the lift."),
            ]
        )

    message = str(excinfo.value)
    assert message.count("[limitations]") == 2
    assert "[validity band]" not in message
    assert "proves" in message
    assert "causes" in message


def test_a_negation_in_an_unpunctuated_fragment_does_not_license_the_next_fragment():
    # The real `cp_evaluation.study.NOT_COVERED` reasons end without a period
    # and sit directly before caller-supplied limitation text in the benchmark
    # report. Read as one sentence, their "cannot" would excuse "proves".
    from cp_evaluation.study import NOT_COVERED

    fragments = [("Limitations", reason) for _, reason in NOT_COVERED]
    fragments.append(("Limitations", "This proves unconfoundedness."))

    with pytest.raises(ReportLanguageError) as excinfo:
        require_clean_prose(fragments)

    assert "'proves'" in str(excinfo.value)


def test_the_unpunctuated_offender_is_attributed_to_its_own_section():
    with pytest.raises(ReportLanguageError) as excinfo:
        require_clean_prose(
            [
                ("coverage", "these cannot share a panel axis"),
                ("limitations", "This proves unconfoundedness."),
            ]
        )

    message = str(excinfo.value)
    assert "[limitations]" in message
    assert "[coverage]" not in message


def test_an_unpunctuated_fragment_may_still_negate_its_own_term():
    require_clean_prose(
        [
            ("limitations", "a balance table does not prove unconfoundedness"),
            ("caption", "Overlap holds across arms."),
        ]
    )
