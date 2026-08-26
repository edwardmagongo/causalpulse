"""The Causal Validity Score: an ordinal band from an inspectable rubric.

The band scores observable diagnostics. It is not a probability that the effect
is causal, and no arrangement of clean diagnostics lifts an observational design
to the top band -- that is the design-based ceiling of section 6.2, and it is
the reason this function takes provenance as an argument rather than reading it
off the estimator.

Caps compose by minimum, not by chaining: a failed identification or
data-adequacy check caps the band at WEAK, and a warning caps it one step below
the design ceiling. A fit carrying both a FAIL and a WARN lands at WEAK -- the
lower of the two caps -- not at INSUFFICIENT, because INSUFFICIENT is reserved
for a check that says the computation itself did not succeed.
"""

from __future__ import annotations

from cp_core.dataset import AssignmentProvenance
from cp_core.results import (
    BAND_ORDER,
    AssumptionCheck,
    CausalValidityScore,
    EffectEstimate,
    ValidityBand,
)

_CEILING: dict[AssignmentProvenance, ValidityBand] = {
    "randomised": "STRONG",
    "natural_experiment": "STRONG",
    "observational": "MODERATE",
}

_ONE_STEP_DOWN: dict[ValidityBand, ValidityBand] = {
    "STRONG": "MODERATE",
    "MODERATE": "WEAK",
    "WEAK": "INSUFFICIENT",
    "INSUFFICIENT": "INSUFFICIENT",
}

_IDENTIFYING: tuple[str, ...] = ("identification", "data_adequacy")


def _named(checks: tuple[AssumptionCheck, ...]) -> str:
    return ", ".join(sorted(c.name for c in checks))


def score_validity(
    estimate: EffectEstimate, *, provenance: AssignmentProvenance
) -> CausalValidityScore:
    """Score one fitted estimate against its own assumption table."""
    checks = estimate.assumptions
    ceiling = _CEILING[provenance]
    rationale: list[str] = [
        f"Design provenance is {provenance}, so the band cannot exceed {ceiling}."
    ]

    unusable = tuple(c for c in checks if c.kind == "estimation" and c.status == "FAIL")
    if unusable:
        rationale.append(
            f"Estimation checks failed ({_named(unusable)}), so the computation did "
            "not succeed and the estimate is not usable at any band."
        )
        return CausalValidityScore(
            band="INSUFFICIENT",
            ceiling=ceiling,
            rationale=tuple(rationale),
            checks=checks,
        )

    caps: list[ValidityBand] = [ceiling]

    failed = tuple(c for c in checks if c.kind in _IDENTIFYING and c.status == "FAIL")
    if failed:
        caps.append("WEAK")
        rationale.append(
            f"Identification or data-adequacy checks failed ({_named(failed)}), "
            "which caps the band at WEAK."
        )

    warned = tuple(c for c in checks if c.status == "WARN")
    if warned:
        caps.append(_ONE_STEP_DOWN[ceiling])
        rationale.append(
            f"Checks warned ({_named(warned)}), which drops the band one step "
            f"below the {ceiling} ceiling."
        )

    untestable = tuple(c for c in checks if c.status == "NOT_TESTABLE")
    if untestable:
        # Named but not subtracted. The ceiling above already prices in the
        # fact that these assumptions cannot be tested; subtracting again here
        # would count the same fact twice and make every observational
        # analysis look broken.
        rationale.append(
            f"Not testable from data ({_named(untestable)}); recorded here and "
            "already reflected in the design ceiling, so it does not move the band."
        )

    band = min(caps, key=lambda candidate: BAND_ORDER[candidate])
    if band == ceiling and len(rationale) == 1:
        rationale.append("No check warned or failed, so the band sits at the ceiling.")

    return CausalValidityScore(
        band=band, ceiling=ceiling, rationale=tuple(rationale), checks=checks
    )
