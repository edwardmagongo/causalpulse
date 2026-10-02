"""`cp_core.catalogue` is a hand-kept copy of the registry; this pins the copy.

Only a bare `import cp_causal` here, never an estimator module: importing one
directly would register it and hide an estimator the package forgot to import.
"""

import cp_causal  # noqa: F401  (registers every estimator)
from cp_causal.registry import all_estimators
from cp_core.catalogue import ESTIMATOR_FAMILIES, FAMILY_ORDER, catalogue


def _family(requirements) -> str:
    if not requirements.needs_panel:
        return "cross_sectional"
    return "donor_pool" if requirements.max_treated_units == 1 else "panel"


def test_the_catalogue_names_exactly_the_registered_estimators():
    assert set(ESTIMATOR_FAMILIES) == set(all_estimators())


def test_each_family_is_the_one_the_estimators_own_requirements_imply():
    derived = {name: _family(cls.requirements) for name, cls in all_estimators().items()}

    assert derived == ESTIMATOR_FAMILIES


def test_the_three_families_are_all_in_use():
    # Otherwise the derivation above could collapse to one branch and still agree.
    assert set(ESTIMATOR_FAMILIES.values()) == set(FAMILY_ORDER)


def test_the_catalogue_is_ordered_by_family_then_name():
    listed = catalogue()

    assert [name for name, _ in listed][:2] == ["aipw", "causal_forest"]
    assert [family for _, family in listed] == sorted(
        (family for _, family in listed), key=FAMILY_ORDER.index
    )
    assert len(listed) == len(ESTIMATOR_FAMILIES)
