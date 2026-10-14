"""The refusal to accept a narrative that cannot be traced to its facts.

Ten rules. Nine read one statement at a time; the tenth reads the whole
narrative, because an omission is not in any one statement. Each is
registered by name so that a test can remove one and show its own planted
defect then passes -- no rule is covered only by another.

What these rules do not establish is stated in the
README: qualitative wording outside the controlled lists, a wrong
relationship between two correctly cited facts, and emphasis.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from cp_core.language import causal_language_lint

from cp_llm.facts import Fact, FactTable
from cp_llm.narrative import Narrative
from cp_llm.render import PLACEHOLDER, format_fact, free_text, placeholder_ids, substitute
from cp_llm.vocabulary import ALLOWED_PHRASES, CONTROLLED_WORDS, NUMBER_WORDS


@dataclass(frozen=True)
class Violation:
    """One broken rule. `section` and `statement` are empty for a narrative-wide rule."""

    rule: str
    section: str
    statement: str
    reason: str


_POINT = re.compile(r"est\.([a-z0-9_]+)\.point")
_REQUIRED = (
    re.compile(r"est\.[a-z0-9_]+\.point"),
    re.compile(r"score\.[a-z0-9_]+\.band"),
    re.compile(r"cmp\.[a-z0-9_]+\.[a-z0-9_]+\.(agreement|not_compared_reason)"),
    re.compile(r"fail\.[a-z0-9_]+\.error"),
)
_ESTIMATE_NUMBER = re.compile(r"est\.([a-z0-9_]+)\.(point|ci_lower|ci_upper|se|p_value)")
_CHECK_STATUS = re.compile(r"check\.[a-z0-9_]+\.[a-z0-9_]+\.status")
_REPORTABLE_STATUSES = frozenset({"FAIL", "WARN"})


def _words(statement: str) -> list[str]:
    text = free_text(statement).lower()
    for phrase in ALLOWED_PHRASES:
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)
    return re.findall(r"[a-z]+", text)


def _resolves(statement: str, facts: FactTable) -> Iterator[str]:
    remainder = free_text(statement)
    if "{" in remainder or "}" in remainder:
        yield "a brace that is not part of a placeholder"
    for fact_id in dict.fromkeys(placeholder_ids(statement)):
        if facts.get(fact_id) is None:
            yield f"placeholder {{{fact_id}}} names no fact in the table"


def _bare_value(statement: str, facts: FactTable) -> Iterator[str]:
    text = free_text(statement)
    if re.search(r"\d", text):
        yield "a digit outside a placeholder; every value must be cited, not written"
    if "%" in text:
        yield "a percent sign outside a placeholder; a level fact carries its own"
    for word in dict.fromkeys(_words(statement)):
        if word in NUMBER_WORDS:
            yield f"the number word {word!r} outside a placeholder"


def _vocabulary(statement: str, facts: FactTable) -> Iterator[str]:
    for word in dict.fromkeys(_words(statement)):
        if word in CONTROLLED_WORDS:
            yield f"{word!r} {CONTROLLED_WORDS[word]}"


def _attribution(statement: str, facts: FactTable) -> Iterator[str]:
    cited = set(placeholder_ids(statement))
    for fact_id in dict.fromkeys(placeholder_ids(statement)):
        fact = facts.get(fact_id)
        if fact is None or fact.subject is None or fact.subject in cited:
            continue
        yield (
            f"{{{fact_id}}} is about {{{fact.subject}}}, which this statement does not name"
        )


def _bare_point(statement: str, facts: FactTable) -> Iterator[str]:
    cited = set(placeholder_ids(statement))
    for fact_id in dict.fromkeys(placeholder_ids(statement)):
        match = _POINT.fullmatch(fact_id)
        if match is None:
            continue
        bounds = {f"est.{match.group(1)}.ci_lower", f"est.{match.group(1)}.ci_upper"}
        if not bounds <= cited:
            yield f"{{{fact_id}}} appears without both of its interval bounds"


def _insufficient_caveat(statement: str, facts: FactTable) -> Iterator[str]:
    """A number from an estimate whose band is INSUFFICIENT comes with that band.

    The caveat is the band itself, cited in the same statement, so the reader
    sees INSUFFICIENT beside the number: a hedge word could not be checked, and
    a band cited elsewhere in the narrative is not beside it.
    """
    cited = set(placeholder_ids(statement))
    methods = dict.fromkeys(
        match.group(1)
        for fact_id in placeholder_ids(statement)
        if (match := _ESTIMATE_NUMBER.fullmatch(fact_id)) is not None
    )
    for method in methods:
        band_id = f"score.{method}.band"
        band = facts.get(band_id)
        if band is not None and band.value == "INSUFFICIENT" and band_id not in cited:
            yield (
                f"{{est.{method}.method}}'s estimate is quoted without {{{band_id}}}, "
                "which is INSUFFICIENT"
            )


_NUMERIC_KINDS = frozenset({"number", "count", "level", "p_value"})


def _for_the_lint(fact: Fact) -> str:
    """A fact as the language lint should read it.

    The lint ends a sentence at every full stop, including the one inside a
    rendered number, so a hedge before `0.066` would not reach a verb after
    it. A numeric fact is therefore one token; a word-valued fact is
    substituted verbatim, because its words are what the lint must read.

    The token is a digit, not a letter: a letter glued to free text can spell
    a word the rendered text does not contain (`{n}o` would lint as "No" and
    `{n}ot` as "Not", hedging a claim that renders as "218o" or "218ot"). A
    digit cannot join letters into a word, and it is what a rendered number
    looks like to the lint, minus the full stop.
    """
    return "0" if fact.kind in _NUMERIC_KINDS else format_fact(fact)


def _language(statement: str, facts: FactTable) -> Iterator[str]:
    if any(facts.get(fact_id) is None for fact_id in placeholder_ids(statement)):
        # Nothing to render; `_resolves` has already reported why.
        return
    for violation in causal_language_lint(substitute(statement, facts, _for_the_lint)):
        yield f"{violation.term!r}: {violation.reason}"


_OPENERS = frozenset("([\"'\u201c\u2018")
_CLOSERS = frozenset(")]\"'\u201d\u2019")
_SENTENCE_PUNCTUATION = frozenset(".,;:!?")
_APOSTROPHES = frozenset("'\u2019")
_OPERATORS = frozenset("-\u2212\u2013+\u00b1\u00d7*/^~\u2248<>=")


def _may_follow_placeholder(after: str) -> bool:
    """Whether the text right after a placeholder keeps it a token of its own.

    A possessive, a closer or sentence punctuation may come between, but only
    if what follows it would itself be allowed here: `'s.` and `).` pass,
    `'stuff` and `)x` do not.
    """
    while after and not after[0].isspace():
        if after[0] in _APOSTROPHES and after[1:2] == "s":
            after = after[2:]
        elif after[0] in _CLOSERS or after[0] in _SENTENCE_PUNCTUATION:
            after = after[1:]
        else:
            return False
    return True


def _delimiters(statement: str, facts: FactTable) -> Iterator[str]:
    """A placeholder must stand alone, so what it renders next to cannot be misread.

    The other rules read the free text with each placeholder as a space, so
    what a placeholder is glued to is invisible to them: `-{point}` renders a
    minus the fact does not carry, `{n}{m}` one number from two, and
    `in{agreement}` a word the controlled list refuses.
    """
    reasons: list[str] = []
    for match in PLACEHOLDER.finditer(statement):
        fact_id = match.group(1)
        before, after = statement[: match.start()], statement[match.end() :]
        if before and not before[-1].isspace() and before[-1] not in _OPENERS:
            reasons.append(
                f"{{{fact_id}}} is glued to the text before it; put a space or an "
                "opening bracket before a placeholder"
            )
        if not _may_follow_placeholder(after):
            reasons.append(
                f"{{{fact_id}}} is glued to the text after it; put a space or sentence "
                "punctuation after a placeholder"
            )
        fact = facts.get(fact_id)
        if fact is None or fact.kind not in _NUMERIC_KINDS:
            continue
        stripped = before.rstrip()
        if stripped and stripped[-1] in _OPERATORS:
            reasons.append(
                f"a sign or operator before {{{fact_id}}}; the fact carries its own sign, "
                "and a range is written from one bound to the other"
            )
    yield from dict.fromkeys(reasons)


_INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


def _one_line(statement: str, facts: FactTable) -> Iterator[str]:
    """Refuse a character that is not ordinary text, read as written.

    A newline lets a statement write document structure (a heading of its own
    in the markdown); a zero-width character can split a controlled word so
    that no word list sees it.
    """
    for character in dict.fromkeys(statement):
        if unicodedata.category(character) in _INVISIBLE_CATEGORIES:
            yield (
                f"a control or invisible character (U+{ord(character):04X}); "
                "a statement is one line of ordinary text"
            )


def _completeness(narrative: Narrative, facts: FactTable) -> Iterator[str]:
    cited = {
        fact_id for _, statement in narrative.statements() for fact_id in placeholder_ids(statement)
    }
    for fact in facts:
        required = any(pattern.fullmatch(fact.id) for pattern in _REQUIRED) or (
            _CHECK_STATUS.fullmatch(fact.id) is not None
            and fact.value in _REPORTABLE_STATUSES
        )
        if required and fact.id not in cited:
            yield f"the narrative never cites {{{fact.id}}}"


StatementRule = Callable[[str, FactTable], Iterator[str]]
NarrativeRule = Callable[[Narrative, FactTable], Iterator[str]]

STATEMENT_RULES: dict[str, StatementRule] = {
    "resolves": _resolves,
    "bare_value": _bare_value,
    "vocabulary": _vocabulary,
    "attribution": _attribution,
    "bare_point": _bare_point,
    "insufficient_caveat": _insufficient_caveat,
    "language": _language,
    "delimiters": _delimiters,
    "one_line": _one_line,
}
NARRATIVE_RULES: dict[str, NarrativeRule] = {"completeness": _completeness}


def verify(narrative: Narrative, facts: FactTable) -> tuple[Violation, ...]:
    """Every rule the narrative breaks against `facts`; empty means verified."""
    violations = [
        Violation(rule=name, section=section, statement=statement, reason=reason)
        for section, statement in narrative.statements()
        for name, rule in STATEMENT_RULES.items()
        for reason in rule(statement, facts)
    ]
    violations.extend(
        Violation(rule=name, section="", statement="", reason=reason)
        for name, rule in NARRATIVE_RULES.items()
        for reason in rule(narrative, facts)
    )
    return tuple(violations)
