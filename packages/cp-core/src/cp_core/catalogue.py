"""Every estimator's name and the design family the picker groups it under.

The API serves this list, and the API may not import the estimators
(`.importlinter`'s `api-cannot-fit` contract), so it cannot read their
registry. The list is declared here, where the API can see it, and
`tests/causal/test_catalogue.py` pins it to the registry and to each
estimator's own requirements.

A family is a grouping, not a statement about eligibility. Whether an
estimator can run on a given dataset is the router's decision, made from the
data, and an estimator can be eligible outside its family's usual shape.
"""

from __future__ import annotations

from typing import Literal

Family = Literal["cross_sectional", "panel", "donor_pool"]

# The order the catalogue lists families in.
FAMILY_ORDER: tuple[Family, ...] = ("cross_sectional", "panel", "donor_pool")

ESTIMATOR_FAMILIES: dict[str, Family] = {
    "aipw": "cross_sectional",
    "causal_forest": "cross_sectional",
    "difference_in_means": "cross_sectional",
    "ipw": "cross_sectional",
    "lin_regression": "cross_sectional",
    "callaway_santanna": "panel",
    "did_2x2": "panel",
    "event_study": "panel",
    "hierarchical_did": "panel",
    "twfe": "panel",
    "bsts": "donor_pool",
    "synthetic_control": "donor_pool",
}


def catalogue() -> list[tuple[str, Family]]:
    """`(name, family)` pairs, by family in `FAMILY_ORDER` and then by name."""
    return sorted(
        ESTIMATOR_FAMILIES.items(), key=lambda item: (FAMILY_ORDER.index(item[1]), item[0])
    )
