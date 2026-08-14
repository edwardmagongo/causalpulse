import pandas as pd
from cp_causal.base import Requirements
from cp_causal.registry import register
from cp_causal.router import eligible_estimators, route
from cp_core.dataset import CausalDataset, CausalSchema


def dataset(
    n_treated: int = 20,
    n_control: int = 20,
    covariates: tuple[str, ...] = ("x1",),
) -> CausalDataset:
    n = n_treated + n_control
    frame = pd.DataFrame(
        {
            "unit_id": range(n),
            "treated": [1] * n_treated + [0] * n_control,
            "outcome": [1, 0] * (n // 2),
            "x1": [float(i) for i in range(n)],
        }
    )
    return CausalDataset(
        frame=frame,
        schema=CausalSchema(
            unit="unit_id",
            treatment="treated",
            outcome="outcome",
            covariates=covariates,
            outcome_type="binary",
        ),
    )


def panel_dataset(start: int = 3, n_units: int = 4, n_periods: int = 4) -> CausalDataset:
    rows = []
    for unit in range(n_units):
        cohort = 0 if unit < 2 else start
        for period in range(1, n_periods + 1):
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "treated": int(cohort > 0 and period >= cohort),
                    "outcome": float(unit + period),
                }
            )
    frame = pd.DataFrame(rows)
    schema = CausalSchema(
        unit="unit_id",
        time="period",
        treatment="treated",
        outcome="outcome",
        covariates=(),
        outcome_type="continuous",
    )
    return CausalDataset(frame=frame, schema=schema)


class _CrossSectional:
    name = "cross_sectional_only"
    requirements = Requirements()

    def fit(self, dataset, config):  # pragma: no cover - not called here
        raise NotImplementedError


class _PanelOnly:
    name = "panel_only"
    requirements = Requirements(needs_panel=True)

    def fit(self, dataset, config):  # pragma: no cover - not called here
        raise NotImplementedError


class _NeedsManyUnits:
    name = "needs_many_units"
    requirements = Requirements(min_treated_units=1000)

    def fit(self, dataset, config):  # pragma: no cover - not called here
        raise NotImplementedError


CANDIDATES = {
    "cross_sectional_only": _CrossSectional,
    "panel_only": _PanelOnly,
    "needs_many_units": _NeedsManyUnits,
}


def test_router_reports_every_candidate_not_only_the_eligible_ones():
    decisions = route(dataset(), candidates=CANDIDATES)

    assert {d.estimator_name for d in decisions} == set(CANDIDATES)


def test_eligible_estimator_carries_no_reasons():
    decisions = {d.estimator_name: d for d in route(dataset(), candidates=CANDIDATES)}

    assert decisions["cross_sectional_only"].eligible is True
    assert decisions["cross_sectional_only"].reasons == ()


def test_excluded_estimator_states_why():
    # A shorter list of methods is less useful than an explained one.
    decisions = {d.estimator_name: d for d in route(dataset(), candidates=CANDIDATES)}
    panel = decisions["panel_only"]

    assert panel.eligible is False
    assert len(panel.reasons) == 1
    assert "panel" in panel.reasons[0].lower()


def test_multiple_unmet_requirements_are_all_reported():
    decisions = {
        d.estimator_name: d
        for d in route(dataset(n_treated=2, n_control=2), candidates=CANDIDATES)
    }

    assert len(decisions["needs_many_units"].reasons) == 1


class _FailsEveryRequirement:
    # Each of these four conditions produces its own distinct reason string in
    # `Requirements.unmet`: needs_panel (checked against schema.is_panel),
    # needs_covariates (checked against schema.covariates), min_treated_units,
    # and min_control_units. `supports_outcome_types` is left at its default
    # so the outcome_type check does not add a fifth reason, and it is not the
    # thing under test here.
    name = "fails_every_requirement"
    requirements = Requirements(
        needs_panel=True,
        needs_covariates=True,
        min_treated_units=1000,
        min_control_units=1000,
    )

    def fit(self, dataset, config):  # pragma: no cover - not called here
        raise NotImplementedError


def test_all_unmet_requirements_are_reported_not_just_the_first():
    # Unlike test_multiple_unmet_requirements_are_all_reported above (whose
    # estimator only ever fails one requirement on the dataset it's run
    # against), this drives an estimator that fails four requirements at once
    # on a single dataset, to actually verify the router reports all of them
    # rather than stopping at the first unmet one.
    cross_sectional_no_covariates = dataset(n_treated=2, n_control=2, covariates=())
    candidates = {"fails_every_requirement": _FailsEveryRequirement}

    decisions = {
        d.estimator_name: d
        for d in route(cross_sectional_no_covariates, candidates=candidates)
    }
    reasons = decisions["fails_every_requirement"].reasons

    assert len(reasons) == 4
    assert len({r.lower() for r in reasons}) == 4


def test_eligible_estimators_returns_sorted_names():
    assert eligible_estimators(dataset(), candidates=CANDIDATES) == (
        "cross_sectional_only",
    )


def test_decisions_are_ordered_by_name_for_stable_output():
    decisions = route(dataset(), candidates=CANDIDATES)

    assert [d.estimator_name for d in decisions] == sorted(CANDIDATES)


def test_route_with_no_candidates_arg_uses_the_live_registry():
    # Every other test in this file passes an explicit `candidates` stub, so
    # the default `candidates=None` path -- the one real callers use, which
    # reads `all_estimators()` from the live registry -- otherwise has zero
    # coverage. `_isolate_registry` (conftest.py, autouse) snapshots and
    # restores `cp_causal.registry._REGISTRY` around every test, so
    # registering here is safe and will not leak into other tests.
    @register
    class _RegisteredForRoutingTest:
        name = "registered_for_routing_test"
        requirements = Requirements()

        def fit(self, dataset, config):  # pragma: no cover - not called here
            raise NotImplementedError

    decisions = route(dataset())

    assert "registered_for_routing_test" in {d.estimator_name for d in decisions}


def test_route_with_no_candidates_arg_does_not_raise_on_empty_registry():
    # Pins that the default (`candidates=None`) path tolerates an empty or
    # minimal live registry rather than raising, even when nothing has been
    # registered beyond whatever production code registers at import time.
    decisions = route(dataset())

    assert isinstance(decisions, tuple)


def test_route_with_empty_candidates_dict_returns_empty_tuple():
    assert route(dataset(), candidates={}) == ()
    assert eligible_estimators(dataset(), candidates={}) == ()


def test_declared_pre_period_minimums_are_enforced_not_decorative():
    dataset_ = panel_dataset(start=2)

    unmet = Requirements(needs_panel=True, min_pre_periods=3).unmet(dataset_)

    assert any("pre-treatment period" in reason for reason in unmet)
    assert Requirements(needs_panel=True, min_pre_periods=1).unmet(dataset_) == ()
