"""The report's refusal to emit prose that asserts more than it can support.

The parent design document requires the generated report to pass this
project's language lint. That lint has been in force since Phase 8 over every
non-docstring string constant under `packages/`, so the prose this package
hardcodes is already covered before a line of it is written.

What is new here is interpolated text. A covariate name, a scenario id, a
caller-supplied limitation or a figure caption reaches the reader as running
prose and has never been linted by anything. A dataset column named for a
proven uplift is what this refusal exists to catch, and a caption is
arguably the likeliest place for it: it reads like a sentence about what the
picture shows, which is exactly the shape an unhedged causal claim takes.

The lint runs over the concatenation of every section's prose, with a blank
line between fragments. `cp_core.language` treats a blank line as a sentence
boundary, so each fragment is its own sentence run even when it lacks terminal
punctuation: a negation or hedge in one fragment ("...cannot share a panel
axis") must not license an unhedged claim in the next. A fragment is never a
sentence begun in one section and finished in another.
"""

from __future__ import annotations

from collections.abc import Sequence

from cp_core.language import LanguageViolation, causal_language_lint

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

    lines = [_format_violation(offsets, violation) for violation in violations]
    raise ReportLanguageError(
        "refusing to emit a report whose prose asserts more than it can "
        "support:\n" + "\n".join(lines)
    )


def _format_violation(offsets: Sequence[tuple[int, str]], violation: LanguageViolation) -> str:
    """Render one violation, attributed to the section its sentence came from.

    `violation.start` is the offset of `violation.sentence` within `document`
    (`cp_core.language.LanguageViolation` guarantees
    `document[start:start + len(sentence)] == sentence`), so the section
    lookup is a direct position lookup -- no re-segmentation of `document` is
    needed here at all.

    One correction is still required: after a fragment ending in `.`, `!` or
    `?` the `_SEPARATOR` is not part of that match, so it leads the next
    sentence, and `start` (the raw match position) lands inside the *previous*
    fragment's offset span. The lookup therefore uses the position of the
    sentence's actual first character. (After an unpunctuated fragment the
    separator is the previous match's terminator and there is nothing to skip.)
    """
    stripped = violation.sentence.strip()
    leading_whitespace = len(violation.sentence) - len(violation.sentence.lstrip())
    section = _section_at(offsets, violation.start + leading_whitespace)
    return f"  [{section}] {violation.term!r}: {violation.reason}\n    {stripped}"


def _section_at(offsets: Sequence[tuple[int, str]], index: int) -> str:
    """The section whose text contains `index`."""
    # `offsets` is empty only when `fragments` is empty, in which case
    # `document` is also empty and `causal_language_lint` returns no
    # violations -- so `require_clean_prose` never reaches this function with
    # an empty `offsets`. The `"unknown"` fallback is defensive only.
    section = offsets[0][1] if offsets else "unknown"
    for start, name in offsets:
        if start > index:
            break
        section = name
    return section
