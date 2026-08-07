import pandas as pd
from cp_causal.base import Requirements
from cp_causal.router import eligible_estimators, route
from cp_core.dataset import CausalDataset, CausalSchema


def dataset(n_treated: int = 20, n_control: int = 20) -> CausalDataset:
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
            covariates=("x1",),
            outcome_type="binary",
        ),
    )


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


def test_eligible_estimators_returns_sorted_names():
    assert eligible_estimators(dataset(), candidates=CANDIDATES) == (
        "cross_sectional_only",
    )


def test_decisions_are_ordered_by_name_for_stable_output():
    decisions = route(dataset(), candidates=CANDIDATES)

    assert [d.estimator_name for d in decisions] == sorted(CANDIDATES)
