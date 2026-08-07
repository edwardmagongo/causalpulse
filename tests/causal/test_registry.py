import pandas as pd
import pytest
from cp_causal.base import Requirements
from cp_causal.registry import all_estimators, get_estimator, register
from cp_core.dataset import CausalDataset, CausalSchema


def dataset(n_treated: int = 5, n_control: int = 5, covariates: bool = True) -> CausalDataset:
    n = n_treated + n_control
    frame = pd.DataFrame(
        {
            "unit_id": range(n),
            "treated": [1] * n_treated + [0] * n_control,
            "outcome": [1, 0] * (n // 2),
            "x1": [float(i) for i in range(n)],
        }
    )
    schema = CausalSchema(
        unit="unit_id",
        treatment="treated",
        outcome="outcome",
        covariates=("x1",) if covariates else (),
        outcome_type="binary",
    )
    return CausalDataset(frame=frame, schema=schema)


def test_requirements_met_returns_no_complaints():
    assert Requirements().unmet(dataset()) == ()


def test_panel_requirement_is_unmet_by_a_cross_section():
    unmet = Requirements(needs_panel=True).unmet(dataset())

    assert len(unmet) == 1
    assert "panel" in unmet[0].lower()


def test_covariate_requirement_is_unmet_without_covariates():
    unmet = Requirements(needs_covariates=True).unmet(dataset(covariates=False))

    assert len(unmet) == 1
    assert "covariate" in unmet[0].lower()


def test_arm_size_requirements_are_reported_separately():
    unmet = Requirements(min_treated_units=10, min_control_units=10).unmet(
        dataset(n_treated=2, n_control=2)
    )

    assert len(unmet) == 2


def test_outcome_type_requirement_is_unmet_for_unsupported_type():
    requirements = Requirements(supports_outcome_types=frozenset({"continuous"}))
    unmet = requirements.unmet(dataset())

    assert len(unmet) == 1
    assert "binary" in unmet[0]


def test_registry_round_trips_a_registered_estimator():
    @register
    class _Dummy:
        name = "dummy_round_trip"
        requirements = Requirements()

        def fit(self, dataset, config):  # pragma: no cover - not called here
            raise NotImplementedError

    assert get_estimator("dummy_round_trip") is _Dummy
    assert "dummy_round_trip" in all_estimators()


def test_registry_rejects_a_duplicate_name():
    @register
    class _First:
        name = "dummy_duplicate"
        requirements = Requirements()

        def fit(self, dataset, config):  # pragma: no cover - not called here
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):

        @register
        class _Second:
            name = "dummy_duplicate"
            requirements = Requirements()

            def fit(self, dataset, config):  # pragma: no cover - not called here
                raise NotImplementedError


def test_registry_rejects_an_unnamed_estimator():
    with pytest.raises(ValueError, match="name"):

        @register
        class _Unnamed:
            requirements = Requirements()

            def fit(self, dataset, config):  # pragma: no cover - not called here
                raise NotImplementedError


def test_unknown_estimator_name_is_rejected():
    with pytest.raises(KeyError, match="nonexistent"):
        get_estimator("nonexistent")


def test_all_estimators_returns_a_copy():
    # Callers must not be able to corrupt the registry process-wide.
    snapshot = all_estimators()
    snapshot["injected"] = object()  # type: ignore[assignment]

    assert "injected" not in all_estimators()
