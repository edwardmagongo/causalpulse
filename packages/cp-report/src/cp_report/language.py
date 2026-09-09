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
    for violation in violations:
        # `_SENTENCE` matches greedily from wherever the previous sentence's
        # terminator left off, so a sentence immediately after a `_SEPARATOR`
        # join has that separator's leading whitespace folded into its match
        # text. Searching for the raw sentence would then land the offset
        # inside the *previous* fragment's span. Stripped, the search lands
        # on the sentence's actual first character.
        index = document.find(violation.sentence.strip())
        section = _section_at(offsets, index)
        lines.append(
            f"  [{section}] {violation.term!r}: {violation.reason}\n"
            f"    {violation.sentence.strip()}"
        )
    raise ReportLanguageError(
        "refusing to emit a report whose prose asserts more than it can "
        "support:\n" + "\n".join(lines)
    )


def _section_at(offsets: Sequence[tuple[int, str]], index: int) -> str:
    """The section whose text contains `index`, or the first if unlocated."""
    if index < 0:
        return offsets[0][1] if offsets else "unknown"
    section = offsets[0][1] if offsets else "unknown"
    for start, name in offsets:
        if start > index:
            break
        section = name
    return section
