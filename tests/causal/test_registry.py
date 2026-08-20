import subprocess
import sys

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


def test_registry_rejects_a_class_with_no_requirements():
    with pytest.raises(ValueError, match="requirements"):

        @register
        class _NoRequirements:
            name = "broken_no_requirements"

            def fit(self, dataset, config):  # pragma: no cover - not called here
                raise NotImplementedError


def test_registry_rejects_a_class_with_wrong_requirements_type():
    with pytest.raises(ValueError, match="requirements"):

        @register
        class _WrongRequirementsType:
            name = "broken_wrong_requirements_type"
            requirements = {}

            def fit(self, dataset, config):  # pragma: no cover - not called here
                raise NotImplementedError


def test_isolated_registry_restores_contents_on_exit(registry_isolation):
    # Pins the mechanism the autouse fixture is built from. Deliberately
    # self-contained: an earlier version of this used two top-level tests, one
    # registering and the next asserting absence, which passes vacuously when run
    # alone or reordered because the name was never registered in that process.
    before = all_estimators()

    with registry_isolation():

        @register
        class _Ephemeral:
            name = "isolation_probe"
            requirements = Requirements()

            def fit(self, dataset, config):  # pragma: no cover - not called here
                raise NotImplementedError

        assert "isolation_probe" in all_estimators()

    assert "isolation_probe" not in all_estimators()
    assert all_estimators() == before


def test_importing_cp_causal_registers_causal_forest_in_a_fresh_interpreter():
    """`import cp_causal` alone must register the causal forest.

    Run in a subprocess deliberately. The equivalent in-process assertion is
    self-fulfilling in two ways at once: `test_causal_forest.py` imports the
    estimator module directly, and inside a full-suite run that import has
    already happened by the time any assertion here executes -- so an
    `__init__.py` that had never listed `causal_forest` would still pass. That
    is exactly the defect Phase 5's final review found for `bsts`, and a fresh
    interpreter is the only form of this check that cannot be fooled by it.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cp_causal; from cp_causal.registry import all_estimators; "
            "print('causal_forest' in set(all_estimators()))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "True", completed.stderr
