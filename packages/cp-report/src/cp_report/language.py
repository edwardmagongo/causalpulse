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

import re
from collections.abc import Sequence

from cp_core.language import LanguageViolation, causal_language_lint

_SEPARATOR = "\n\n"

# Mirrors `cp_core.language._SENTENCE` exactly. Duplicated rather than
# imported -- it is private to that module, and `cp_core.language` must not
# change for this package's sake. Needed to independently recover each
# sentence match's *position* in `document`: `LanguageViolation` carries only
# `sentence` text, and text is not enough to attribute a violation to a
# section -- see the comment above `_attribute_violations` for why.
_SENTENCE = re.compile(r"[^.!?]*[.!?]|[^.!?]+$")


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

    lines = _attribute_violations(document, offsets, violations)
    raise ReportLanguageError(
        "refusing to emit a report whose prose asserts more than it can "
        "support:\n" + "\n".join(lines)
    )


def _attribute_violations(
    document: str,
    offsets: Sequence[tuple[int, str]],
    violations: Sequence[LanguageViolation],
) -> list[str]:
    """Attribute each violation to the section its sentence came from.

    An earlier version of this did the attribution by comparing
    `violation.sentence` text between consecutive violations: if it changed,
    search `document` for the new text and advance a cursor; if it repeated,
    assume it was another term from the *same* sentence and reuse the last
    position. That collapses two genuinely different cases that read
    identically as text:

    - several violations from one sentence (`causal_language_lint` emits one
      per offending term, and they must all attribute to that one section),
      and
    - two *different* sentences, in two *different* sections, that happen to
      carry byte-identical prose (reused boilerplate limitation text, a
      hedge phrasing repeated verbatim).

    Comparing raw (unstripped) `sentence` text told these apart only by
    accident: `_SENTENCE` sweeps the preceding `"\n\n"` separator into every
    *non-initial* sentence match, so only a sentence at the very start of
    `document` lacks it. Two non-initial sections sharing one sentence both
    get the identical swept prefix, so their raw text -- and their stripped
    text -- are equal too. No text comparison, raw or stripped, can
    distinguish that pair from "same sentence, second term": both look like
    "unchanged" to a text-equality check, so the naive fix reintroduces the
    exact misattribution it exists to close.

    What actually distinguishes the two cases is *position*, not text: one
    sentence occupies one span in `document`; a repeated sentence occupies
    two different spans. So this replays `causal_language_lint`'s own
    sentence segmentation over `document` via `_SENTENCE.finditer` (the
    identical pattern `causal_language_lint` uses internally), and for each
    match, re-lints just that match's text to find out how many violations
    it produced. Feeding a single `_SENTENCE` match's text back through
    `causal_language_lint` reproduces exactly the violations that match
    contributed to the original call: a match's text contains no terminator
    before its own (if any), so `_SENTENCE.finditer` on it in isolation
    yields exactly one match spanning the whole string, and `_is_hedged` /
    `_is_negated` are computed purely from that string. `violations` is
    consumed in that same document order (`re.finditer` visits left to
    right, and so does `causal_language_lint`'s own loop), so the first
    `count` remaining violations belong to the current match and the rest
    belong to later ones. This uses each match's real `match.start()`
    offset, so it never has to search for text at all -- there is no
    `document.find` call here, and consequently nothing that can return -1
    or otherwise fail to locate an occurrence.
    """
    lines: list[str] = []
    remaining = list(violations)
    for match in _SENTENCE.finditer(document):
        if not remaining:
            break
        text = match.group()
        stripped = text.strip()
        if not stripped:
            continue
        count = len(causal_language_lint(text))
        if count == 0:
            continue
        # `match.start()` is the start of the *raw* match, which -- for
        # every non-initial sentence -- includes the swept-in `_SEPARATOR`
        # leading whitespace (see the module-level comment on `_SENTENCE`
        # duplication). That position can land inside the *previous*
        # fragment's span rather than the current one's, so the section
        # lookup must use the position of the sentence's first real
        # character, not the match's raw start.
        real_start = match.start() + (len(text) - len(text.lstrip()))
        section = _section_at(offsets, real_start)
        for violation in remaining[:count]:
            lines.append(f"  [{section}] {violation.term!r}: {violation.reason}\n    {stripped}")
        remaining = remaining[count:]
    return lines


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
