"""Data for the report's assumption table, validity band, cross-method
comparison and limitations sections.

Each function returns exactly what its section will write -- names, statuses,
counts, prose -- and nothing about how a renderer lays it out. That split
keeps this module testable against structure rather than against rendered
strings, which break on every wording change.

Four hazards shaped the shapes below:

`NOT_TESTABLE` carries real weight. Unconfoundedness cannot be tested from
observational data, and `AssumptionCheck`'s own docstring says the system
exists to say exactly that rather than implying a balance table proved it.
Folding a `NOT_TESTABLE` row into a pass rate -- "3 of 4 checks passed" --
would assert something the data cannot support. So `AssumptionTableData`
counts `NOT_TESTABLE` on its own, separate from both `n_passed` and
`n_failed`. `WARN` gets the same treatment for a smaller reason: it is
neither a pass nor a failure, and folding it into either would silently
misreport a row a reader could otherwise see and judge themselves.

A validity band is never rendered without `VALIDITY_SCORE_LIMITS`. That
string is exposed as a property on `CausalValidityScore` (`score.limits`)
precisely so no call site can shorten, reword or drop it. `ValidityBandData`
reads `limits` from the score and takes no `limits` argument at all, so
there is no parameter a caller could pass a different string through.
`LimitationsData` reads `limits` the same way, for the same reason.

Estimators do not share an estimand -- cross-sectional ones report `ATE`,
panel ones report `ATT` -- and this project has already shipped the
confusion of treating them as one quantity three times. So `ComparisonData`
carries `estimands` per row rather than implying one shared quantity across
the table, the same way Phase 16's forest plot renders it per row.

A table of bare points is the uncertainty defect in a worse medium than a
chart: a number in a table reads as more authoritative than a dot on a plot.
`comparison_data` calls `require_interval` for every estimate, so a row
without an interval refuses to be built rather than rendering as a bare
point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cp_core.results import (
    AssumptionCheck,
    AssumptionKind,
    AssumptionStatus,
    CausalValidityScore,
    CIBasis,
    EffectEstimate,
    Estimand,
)
from cp_figures.guards import require_interval


@dataclass(frozen=True)
class AssumptionTableData:
    """One row per assumption check, plus the counts a reader might check.

    All the tuple fields are the same length as `names`, one entry per
    check, in the order the checks were given. `statistic`, `threshold` and
    `remedy` entries are `None` exactly when the check itself carries `None`
    there -- a check that was never computed must not read the same as one
    that was computed as zero, or a "not computed" figure would be silently
    misread as "computed as zero" downstream. See `AssumptionCheck` in
    `cp_core.results` for when each is `None` (`NOT_TESTABLE` checks have no
    `statistic` or `threshold`; `remedy` is only present when a check did
    not pass).

    The four counts -- `n_passed`, `n_warned`, `n_failed`, `n_not_testable`
    -- always sum to `len(names)`: every `AssumptionStatus` value lands in
    exactly one of them, so no row is silently dropped from the tally.
    `n_not_testable` is kept separate from `n_passed` and `n_failed` rather
    than folded into either, because a `NOT_TESTABLE` row is not evidence of
    a pass or a failure -- it is a statement that the data cannot speak to
    the assumption at all. Reporting "3 of 4 checks passed" over a row like
    that would claim something observational data cannot support.
    """

    names: tuple[str, ...]
    kinds: tuple[AssumptionKind, ...]
    statuses: tuple[AssumptionStatus, ...]
    statistics: tuple[float | None, ...]
    thresholds: tuple[float | None, ...]
    interpretations: tuple[str, ...]
    remedies: tuple[str | None, ...]
    n_passed: int
    n_warned: int
    n_failed: int
    n_not_testable: int

    def prose_fragments(self) -> tuple[str, ...]:
        """The running text a renderer will show the reader, for the lint.

        Included: each check's `name` (often drawn from a dataset column,
        e.g. a covariate) and `interpretation` (free text describing what
        was found) -- both arrive from a caller's data and have never been
        linted by anything. Excluded: `status` and `kind`, which are always
        one of a fixed, already-linted vocabulary (`PASS`/`WARN`/`FAIL`/
        `NOT_TESTABLE`, `identification`/`estimation`/`data_adequacy`), and
        the numeric `statistic`/`threshold` fields, which are not prose at
        all. `remedy` is included when present, for the same reason as
        `interpretation`: it is free text a caller supplied.
        """
        fragments = [
            f"{name}: {interpretation}"
            for name, interpretation in zip(self.names, self.interpretations, strict=True)
        ]
        fragments.extend(remedy for remedy in self.remedies if remedy is not None)
        return tuple(fragments)


@dataclass(frozen=True)
class ValidityBandData:
    """The ordinal validity band, its ceiling, and the rationale behind it.

    `limits` is always `score.limits` -- the `VALIDITY_SCORE_LIMITS`
    constant surfaced as a property on `CausalValidityScore` -- never a
    literal and never a caller-supplied argument, so a call site cannot
    shorten, reword or drop the statement that the band scores observable
    diagnostics rather than truth.
    """

    band: str
    ceiling: str
    rationale: tuple[str, ...]
    limits: str

    def prose_fragments(self) -> tuple[str, ...]:
        """The running text a renderer will show the reader, for the lint.

        Included: `limits` (the `VALIDITY_SCORE_LIMITS` constant text) and
        every `rationale` entry, which is free text a caller supplied to
        explain why the band landed where it did. Excluded: `band` and
        `ceiling`, both drawn from the fixed `ValidityBand` vocabulary
        (`STRONG`/`MODERATE`/`WEAK`/`INSUFFICIENT`) and already covered by
        the corpus scan.
        """
        return (self.limits, *self.rationale)


def assumption_table_data(checks: tuple[AssumptionCheck, ...]) -> AssumptionTableData:
    """Build the assumption table section's data from one check per row.

    See `AssumptionTableData` for why `NOT_TESTABLE` and `WARN` each get
    their own count rather than being folded into `n_passed` or `n_failed`.
    """
    n_passed = sum(1 for check in checks if check.status == "PASS")
    n_warned = sum(1 for check in checks if check.status == "WARN")
    n_failed = sum(1 for check in checks if check.status == "FAIL")
    n_not_testable = sum(1 for check in checks if check.status == "NOT_TESTABLE")

    return AssumptionTableData(
        names=tuple(check.name for check in checks),
        kinds=tuple(check.kind for check in checks),
        statuses=tuple(check.status for check in checks),
        statistics=tuple(check.statistic for check in checks),
        thresholds=tuple(check.threshold for check in checks),
        interpretations=tuple(check.interpretation for check in checks),
        remedies=tuple(check.remedy for check in checks),
        n_passed=n_passed,
        n_warned=n_warned,
        n_failed=n_failed,
        n_not_testable=n_not_testable,
    )


def validity_band_data(score: CausalValidityScore) -> ValidityBandData:
    """Build the validity band section's data from a `CausalValidityScore`.

    `limits` is read from `score.limits` -- see `ValidityBandData` for why
    that property, and never a literal or a caller argument, is the source.
    """
    return ValidityBandData(
        band=score.band,
        ceiling=score.ceiling,
        rationale=score.rationale,
        limits=score.limits,
    )


@dataclass(frozen=True)
class ComparisonData:
    """One row per estimate, for a table comparing methods against each other.

    Cross-sectional estimators report `ATE`; panel estimators report `ATT`.
    Those are different quantities, and this project has already shipped the
    confusion of treating them as one three times. So `estimands` is carried
    per row rather than assumed -- a renderer must show the reader which
    quantity each row is, not one heading over the whole table. `bases` is
    carried per row for the same reason: an analytic interval and a
    bootstrap interval were computed differently, and collapsing that into
    one column would hide it.

    All the tuple fields are the same length as `methods`, one entry per
    estimate, in the order the estimates were given. `lower` and `upper` are
    never `None` -- `comparison_data` calls `require_interval` for every
    estimate before a row is built, so a `ComparisonData` cannot exist for
    an estimate without an interval. A bare point in a table reads as more
    authoritative than a dot on a chart, which is why this guard exists here
    at all rather than only in `cp_figures`.
    """

    methods: tuple[str, ...]
    estimands: tuple[Estimand, ...]
    points: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    bases: tuple[CIBasis, ...]

    def prose_fragments(self) -> tuple[str, ...]:
        """The running text a renderer will show the reader, for the lint.

        Included: `methods`, arbitrary names that arrive from a caller's
        estimator configuration and have never been linted by anything.
        Excluded: `estimands` and `bases`, both drawn from the fixed
        `Estimand`/`CIBasis` vocabularies and already covered by the corpus
        scan, and `points`/`lower`/`upper`, which are numeric, not prose.
        """
        return tuple(self.methods)


@dataclass(frozen=True)
class LimitationsData:
    """The score's own limits, followed by whatever else the caller states.

    `limits` is always `score.limits` -- the same `VALIDITY_SCORE_LIMITS`
    constant `ValidityBandData` carries -- so a limitations section always
    leads with the one sentence this project requires next to every score,
    never a caller-supplied substitute. `statements` is the caller's tuple,
    unmodified and unreordered: additional limitations a report's author
    chose to state explicitly.
    """

    limits: str
    statements: tuple[str, ...]

    def prose_fragments(self) -> tuple[str, ...]:
        """The running text a renderer will show the reader, for the lint.

        Included: `limits`, for the same reason `ValidityBandData` includes
        it -- it is prose the renderer will show the reader, so it belongs
        in the same lint pass regardless of already living in source. And
        `statements`, the caller's own free text: the most likely place for
        an unhedged causal claim to arrive, since it is text a report's
        author wrote themselves rather than something already-classified.
        """
        return (self.limits, *self.statements)


def comparison_data(estimates: Sequence[EffectEstimate]) -> ComparisonData:
    """Build the cross-method comparison table's data, one row per estimate.

    Calls `require_interval` for every estimate before building a row --
    see `ComparisonData` for why a table of bare points is not permitted.
    The bounds written into `lower`/`upper` are read from the guard's
    return value, not from `estimate.ci` directly, so the call is load-
    bearing rather than decorative.
    """
    intervals = [
        require_interval(estimate, "cross-method comparison table") for estimate in estimates
    ]

    return ComparisonData(
        methods=tuple(estimate.method for estimate in estimates),
        estimands=tuple(estimate.estimand for estimate in estimates),
        points=tuple(estimate.point for estimate in estimates),
        lower=tuple(interval.lower for interval in intervals),
        upper=tuple(interval.upper for interval in intervals),
        bases=tuple(estimate.ci_basis for estimate in estimates),
    )


def limitations_data(
    score: CausalValidityScore, statements: tuple[str, ...]
) -> LimitationsData:
    """Build the limitations section's data, leading with the score's own limits.

    `limits` is read from `score.limits` -- see `LimitationsData` for why
    that property, never a literal, is the source. `statements` is the
    caller's tuple, carried through unmodified.
    """
    return LimitationsData(limits=score.limits, statements=statements)
