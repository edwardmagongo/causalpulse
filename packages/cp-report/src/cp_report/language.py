"""The report's refusal to emit prose that asserts more than it can support.

The parent design document requires the generated report to pass this
project's language lint. That lint has been in force since Phase 8 over every
non-docstring string constant under `packages/`, so the prose this package
hardcodes is already covered before a line of it is written.

What is new here is interpolated text. A covariate name, a scenario id or a
caller-supplied limitation reaches the reader as running prose and has never
been linted by anything. A dataset column named for a proven uplift is what
this refusal exists to catch.

The lint runs over the concatenation of every section's prose rather than
over each fragment alone. Each `interpretation` and `rationale` string in
this codebase was written to stand beside its own number; a claim that only
emerges when they are read in sequence is real, and linting fragments
separately would miss it by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

from cp_core.language import causal_language_lint

_SEPARATOR = "\n\n"


class ReportLanguageError(ValueError):
    """Raised when a report's prose would state a causal claim unqualified."""


def require_clean_prose(fragments: Sequence[tuple[str, str]]) -> None:
    """Refuse the report if any of its prose fails the language lint.

    Each fragment is `(section_name, text)`. The section name is carried only
    so a refusal can say where the offending sentence came from.
    """
    offsets: list[tuple[int, str]] = []
    position = 0
    for section, text in fragments:
        offsets.append((position, section))
        position += len(text) + len(_SEPARATOR)

    document = _SEPARATOR.join(text for _, text in fragments)
    violations = causal_language_lint(document)
    if not violations:
        return

    lines: list[str] = []
    # `causal_language_lint` walks sentences via `re.finditer`, which visits
    # matches strictly left to right, so `violations` already arrives in
    # document order. That makes a monotonic search cursor correct: search
    # for each sentence starting where the previous one was found, never from
    # position 0. Searching from 0 (as a plain `document.find(sentence)`
    # would) always returns the *leftmost* occurrence, so two sections that
    # happen to share byte-identical offending prose -- reused boilerplate
    # limitation text, a hedge phrasing repeated verbatim -- would both be
    # attributed to the first section, silently misattributing the second.
    #
    # A single sentence can also trip more than one term (e.g. "proves" and
    # "causes" in the same sentence), which emits multiple violations that
    # share the identical `sentence` string. The cursor must only advance
    # when the sentence changes, not on every violation: advancing the search
    # position on the second violation from the *same* sentence would search
    # past its own occurrence and could land on the next section's copy of
    # the same text instead, reintroducing the misattribution this cursor
    # exists to fix.
    #
    # "Same sentence" is judged on the *raw*, unstripped `violation.sentence`
    # -- not the stripped text used for searching below. Two violations from
    # one match share the literal same raw string, since `causal_language_lint`
    # computes `sentence = match.group()` once per match and reuses that
    # value for every violation the match produces. Two violations from
    # *different* matches with identical wording, by contrast, are told apart
    # by the same whitespace-sweeping behavior described below: only the
    # sentence at the very start of `document` has no separator swept into
    # its match, so a repeated sentence's second (and later) occurrence
    # always carries a leading `"\n\n"` its first occurrence lacks, making the
    # raw strings differ even though the stripped text is identical.
    cursor = 0
    current_sentence: str | None = None
    current_index = -1
    for violation in violations:
        # `_SENTENCE` matches greedily from wherever the previous sentence's
        # terminator left off, so a sentence immediately after a `_SEPARATOR`
        # join has that separator's leading whitespace folded into its match
        # text. Searching for the raw sentence would then land the offset
        # inside the *previous* fragment's span. Stripped, the search lands
        # on the sentence's actual first character.
        stripped = violation.sentence.strip()
        if violation.sentence != current_sentence:
            current_index = document.find(stripped, cursor)
            if current_index != -1:
                cursor = current_index + len(stripped)
            current_sentence = violation.sentence
        section = _section_at(offsets, current_index)
        lines.append(
            f"  [{section}] {violation.term!r}: {violation.reason}\n"
            f"    {stripped}"
        )
    raise ReportLanguageError(
        "refusing to emit a report whose prose asserts more than it can "
        "support:\n" + "\n".join(lines)
    )


def _section_at(offsets: Sequence[tuple[int, str]], index: int) -> str:
    """The section whose text contains `index`, or the first if unlocated."""
    # `offsets` is empty only when `fragments` is empty, in which case
    # `document` is also empty and `causal_language_lint` returns no
    # violations -- so `require_clean_prose` never reaches this function with
    # an empty `offsets`. The `"unknown"` fallback is defensive only; it is
    # not exercised by any test and is not reachable through the public
    # `require_clean_prose` entry point.
    if index < 0:
        return offsets[0][1] if offsets else "unknown"
    section = offsets[0][1] if offsets else "unknown"
    for start, name in offsets:
        if start > index:
            break
        section = name
    return section
