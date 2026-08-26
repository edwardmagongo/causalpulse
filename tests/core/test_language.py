"""The report-language lint, and the corpus scan that keeps it honest."""

import ast
import pathlib

import pytest
from cp_core.language import causal_language_lint

PACKAGES = pathlib.Path(__file__).resolve().parents[2] / "packages"
# The linter cannot be a subject of its own corpus scan. `_ABSOLUTE_TERMS` is a
# table whose *keys are the banned words* and whose values explain them, so the
# first scan reported 23 offences against it -- "prove", "guarantee", "cause"
# and the reason string "asserts proof; report evidence under stated
# assumptions instead". Those are a vocabulary, not a claim about a finding.
# The exclusion is exactly one file wide, deliberately.
LINT_MODULE = PACKAGES / "cp-core" / "src" / "cp_core" / "language.py"


@pytest.mark.parametrize(
    "text",
    [
        "This proves the intervention caused the lift.",
        "The effect is definitely positive.",
        "Randomisation guarantees an unbiased estimate.",
        "The result conclusively establishes causation.",
    ],
)
def test_lint_flags_unqualified_claims(text: str) -> None:
    assert causal_language_lint(text), text


@pytest.mark.parametrize(
    "text",
    [
        "A balance table does not prove unconfoundedness.",
        "Nothing in the data can prove that assignment was random.",
        "This provides evidence consistent with a causal effect under the stated assumptions.",
        "If the parallel-trends assumption holds, the policy causes a 2-point lift.",
        "Overlap is adequate.",
    ],
)
def test_lint_accepts_hedged_and_negated_prose(text: str) -> None:
    assert causal_language_lint(text) == (), text


def test_a_hedge_must_be_a_word_not_a_substring() -> None:
    # "if" appears inside "significant". A substring hedge test would let this
    # through, which would quietly disable the causal-verb rule almost
    # everywhere in this codebase.
    assert causal_language_lint("The rollout causes a significant lift.")


def test_establish_is_flagged_affirmed_and_accepted_negated() -> None:
    # The corpus scan is what added `establish` to the lexicon: it, not
    # "prove", is the verb this codebase reaches for when stating a limit, and
    # every shipped use of it is negated. Both directions are pinned so the
    # entry cannot be dropped as redundant with "prove". The affirmative case
    # deliberately contains no other banned lexeme -- an earlier draft wrote
    # "...was the cause", which the causal-verb rule flagged, so the test
    # passed with `establish` deleted from the lexicon entirely.
    violations = causal_language_lint("This establishes unconfoundedness.")
    assert [v.term for v in violations] == ["establishes"]
    assert (
        causal_language_lint(
            "No diagnostic computed from this data can establish that treatment "
            "assignment is independent of the potential outcomes."
        )
        == ()
    )


def test_a_negation_two_clauses_away_does_not_excuse_the_claim() -> None:
    # Reduced from the sentence that motivated clause-scoping the negation
    # check: cp_simulation.cate_scenarios described the NULL scenario with two
    # `no` cues inside a parenthetical aside, which were licensing "prove" in a
    # later clause. Sentence-scoped negation returns () here.
    violations = causal_language_lint(
        "This is the scenario the test must pass on -- reporting no reliable "
        "evidence and no subgroups -- the guard this phase exists to prove works."
    )
    assert [v.term for v in violations] == ["prove"]


def test_a_negation_after_the_term_does_not_excuse_it() -> None:
    assert causal_language_lint("This proves causation, and nothing else matters.")


def test_lint_reports_the_offending_sentence_not_the_whole_text() -> None:
    violations = causal_language_lint(
        "Overlap is adequate. This proves causation. Trends are parallel."
    )
    assert len(violations) == 1
    assert violations[0].sentence.strip() == "This proves causation."


def _message_literals(path: pathlib.Path) -> list[str]:
    """Every string constant in a file that is not a docstring.

    Docstrings are excluded deliberately: they are developer-facing prose about
    what the methods cannot establish, where "does not prove" is the correct
    thing to write. User-facing message strings are the target.
    """
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_the_corpus_scan_reaches_real_check_prose() -> None:
    """The scan must not be vacuously green because it reads nothing.

    `VALIDITY_SCORE_LIMITS` is the reason this module lives in cp-core at all:
    it is user-facing text in the bottom layer, and a linter downstream of it
    could not check it without breaking the layering contract. If the literal
    extractor ever stops reaching it, the corpus scan is worthless and this
    test says so before the scan reports a false all-clear.
    """
    literals = _message_literals(PACKAGES / "cp-core" / "src" / "cp_core" / "results.py")
    assert any("It is not a probability that the effect is causal" in text for text in literals)
    prose = [text for text in literals if len(text.split()) >= 6]
    assert len(prose) >= 5, prose


def test_no_shipped_message_string_makes_an_unqualified_causal_claim() -> None:
    offences = []
    for path in sorted(PACKAGES.rglob("*.py")):
        if path == LINT_MODULE:
            continue
        for literal in _message_literals(path):
            for violation in causal_language_lint(literal):
                offences.append(f"{path.name}: {violation.term} in {violation.sentence!r}")
    assert offences == [], "\n".join(offences)
