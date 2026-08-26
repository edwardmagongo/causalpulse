"""A lint over the language the system uses to describe causal findings.

Section 6.3 requires conditional phrasing and forbids asserting proof. The rule
that makes this hard is not the banned words -- it is that the honest form of
this codebase's prose *contains* those words under negation: "a balance table
does not prove unconfoundedness" is exactly what we want written. So both rule
classes are negation-aware, and a claim about causation is allowed when the
sentence carries a hedge.

The rules below were calibrated against every non-docstring string constant in
`packages/` outside this module (1090 literals, 1226 sentences) rather than
assumed, and the comments record the specific sentence that motivated each
adjustment. `tests/core/test_language.py` re-runs that scan on every commit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE = re.compile(r"[^.!?]*[.!?]|[^.!?]+$")

# Negation is scoped to the clause, not the sentence; see `_is_negated`.
_CLAUSE_BOUNDARY = re.compile(r"--|[,;:()–—]")

# Words that assert certainty. Flagged unless negated in the same clause.
_ABSOLUTE_TERMS: dict[str, str] = {
    "prove": "asserts proof; report evidence under stated assumptions instead",
    "proves": "asserts proof; report evidence under stated assumptions instead",
    "proven": "asserts proof; report evidence under stated assumptions instead",
    "proof": "asserts proof; report evidence under stated assumptions instead",
    "definitely": "asserts certainty the design cannot support",
    "definitively": "asserts certainty the design cannot support",
    "conclusively": "asserts certainty the design cannot support",
    "undeniably": "asserts certainty the design cannot support",
    "irrefutably": "asserts certainty the design cannot support",
    "guarantee": "asserts a guarantee no observational design provides",
    "guarantees": "asserts a guarantee no observational design provides",
    "guaranteed": "asserts a guarantee no observational design provides",
    # `establish` is not in the original list, and the corpus scan is what put
    # it there: "establish" -- not "prove" -- is the verb this codebase
    # actually reaches for when it states a limit. Every shipped use of it is
    # negated: propensity.py's unconfoundedness check writes "No diagnostic
    # computed from this data can establish that treatment assignment is
    # independent of the potential outcomes given these covariates", and
    # experimental.py's writes "Nothing in the data can establish that". A lint
    # that banned `prove` but not `establish` would pass the affirmative form
    # of the exact sentence this codebase is careful to negate.
    "establish": "asserts proof; report evidence under stated assumptions instead",
    "establishes": "asserts proof; report evidence under stated assumptions instead",
    "established": "asserts proof; report evidence under stated assumptions instead",
}

# Causal verbs. Allowed only with a hedge in the same sentence.
_CAUSAL_TERMS: tuple[str, ...] = ("cause", "causes", "caused", "causing")

_NEGATIONS: tuple[str, ...] = (
    "not",
    "never",
    "cannot",
    "nothing",
    "no",
    "without",
    "n't",
)

# Word-anchored, not a substring test. A bare `"if" in sentence` check matches
# inside "significant", "different" and "specific", which would hedge almost
# every sentence in this codebase and silently disable the rule.
_HEDGE = re.compile(
    r"\b(?:assum\w*|under|if|unless|whether|consistent with|evidence|would|"
    r"may|might|suggests?|provided|conditional|cannot|not|no)\b"
)


@dataclass(frozen=True)
class LanguageViolation:
    term: str
    sentence: str
    reason: str


def _words(fragment: str) -> list[str]:
    return re.findall(r"[a-z']+", fragment.lower())


def _is_negated(words: list[str], index: int) -> bool:
    """Whether a negation cue governs the term at `index` within its clause.

    Scoped to the words before the term rather than after it: "this proves
    causation, and nothing else matters" must still be flagged.

    Scoped to the clause rather than the whole sentence because the corpus
    showed sentence scope suppressing a firing it had no business suppressing.
    `cp_simulation.cate_scenarios` described the NULL scenario as "...the
    omnibus calibration test must correctly pass on -- reporting 'no reliable
    evidence of heterogeneity' and no subgroups -- the guard this phase exists
    to prove works": two `no` cues, both inside a parenthetical aside two
    clauses upstream, were silently licensing "prove". Clause scope was
    measured across the whole corpus before being adopted -- it changed exactly
    one verdict, that one, and the text was fixed rather than the rule relaxed.
    """
    return any(word in _NEGATIONS or word.endswith("n't") for word in words[:index])


def _is_hedged(sentence: str) -> bool:
    """Whether a hedge frames the sentence.

    Sentence-scoped, deliberately unlike `_is_negated`: a hedge is a frame over
    the whole statement, and the clause carrying it is usually not the clause
    carrying the claim -- "If the parallel-trends assumption holds, the policy
    causes a 2-point lift". A negation cue, by contrast, attaches to its verb.
    """
    return _HEDGE.search(sentence.lower()) is not None


def causal_language_lint(text: str) -> tuple[LanguageViolation, ...]:
    """Report every unqualified causal claim in `text`, sentence by sentence."""
    violations: list[LanguageViolation] = []
    for match in _SENTENCE.finditer(text):
        sentence = match.group()
        if not sentence.strip():
            continue
        hedged = _is_hedged(sentence)
        for clause in _CLAUSE_BOUNDARY.split(sentence):
            words = _words(clause)
            for index, word in enumerate(words):
                if word in _ABSOLUTE_TERMS and not _is_negated(words, index):
                    violations.append(LanguageViolation(word, sentence, _ABSOLUTE_TERMS[word]))
                elif word in _CAUSAL_TERMS and not hedged:
                    violations.append(
                        LanguageViolation(
                            word,
                            sentence,
                            "asserts causation without stating the assumptions it rests on",
                        )
                    )
    return tuple(violations)
