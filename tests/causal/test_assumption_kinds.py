"""Every assumption check declares a kind, and the classification is pinned.

An exact mapping, not a spot check. The point of making `kind` required was
that a new check cannot appear unclassified; a test that asserts three of
seventeen would pass vacuously for the other fourteen, and a test that walks
whatever exists would pass vacuously for a check added with the wrong kind.

Read from the source with `ast` rather than by calling each check: exercising
all seventeen would mean constructing valid inputs for seventeen different
signatures, and the thing under test is a literal at the construction site.
"""

import ast
import pathlib

CP_CAUSAL = (
    pathlib.Path(__file__).resolve().parents[2]
    / "packages"
    / "cp-causal"
    / "src"
    / "cp_causal"
)

EXPECTED_KINDS = {
    "absorbing_treatment": "data_adequacy",
    "calibration_test": "data_adequacy",
    "convex_hull": "identification",
    "covariate_balance": "data_adequacy",
    "divergent_transitions": "estimation",
    "donor_contamination": "identification",
    "effective_sample_size": "estimation",
    "no_anticipation": "identification",
    "no_concurrent_intervention": "identification",
    "outcome_model_specification": "data_adequacy",
    "parallel_trends": "identification",
    "positivity": "identification",
    "pre_treatment_fit": "data_adequacy",
    "r_hat": "estimation",
    "randomised_assignment": "identification",
    "sufficient_pre_periods": "data_adequacy",
    "unconfoundedness": "identification",
}


def _declared_pairs() -> set[tuple[str, str]]:
    """Every (name, kind) literal pair passed to an AssumptionCheck call."""
    pairs = set()
    for path in sorted(CP_CAUSAL.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "AssumptionCheck"):
                continue
            supplied = {
                kw.arg: kw.value.value
                for kw in node.keywords
                if kw.arg in {"name", "kind"}
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            }
            assert set(supplied) == {"name", "kind"}, (
                f"{path.name}: AssumptionCheck built without literal name and kind. "
                "Both must be literals at the construction site so this test can see them."
            )
            pairs.add((supplied["name"], supplied["kind"]))
    return pairs


def test_the_declared_kinds_are_exactly_the_pinned_classification():
    declared = _declared_pairs()
    assert declared == set(EXPECTED_KINDS.items()), {
        "unpinned or misclassified": sorted(declared - set(EXPECTED_KINDS.items())),
        "pinned but no longer constructed": sorted(set(EXPECTED_KINDS.items()) - declared),
    }


def test_no_check_name_is_constructed_with_two_different_kinds():
    # `parallel_trends`, `no_anticipation` and `calibration_test` each have more
    # than one construction site. A name meaning `identification` in one file
    # and `estimation` in another would make the rubric's answer depend on which
    # branch produced the check.
    by_name: dict[str, set[str]] = {}
    for name, kind in _declared_pairs():
        by_name.setdefault(name, set()).add(kind)
    conflicting = {n: sorted(k) for n, k in by_name.items() if len(k) > 1}
    assert conflicting == {}, conflicting
