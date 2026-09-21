"""A tiny grid and the estimator settings the store and executor tests share."""

import pandas as pd
from cp_core.values import EstimatorConfig
from cp_evaluation.grid import CrossSectionAxes, GridConfig, PanelAxes
from cp_evaluation.run import run_grid
from cp_evaluation.store import RunIdentity, RunStore

ESTIMATORS = ("difference_in_means", "lin_regression")
FIT_CONFIG = EstimatorConfig(random_seed=0, bootstrap_reps=20)


def tiny_config(replications: int = 4) -> GridConfig:
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(200,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=replications,
    )


def mixed_config(replications: int = 2) -> GridConfig:
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(200, 400), confounding_strength=(1.0,)
        ),
        panel=PanelAxes(scenarios=("P1",), n_units=(100,), n_periods=(6,)),
        replications=replications,
    )


def identity_for(config: GridConfig, *, root_entropy: int = 0) -> RunIdentity:
    return RunIdentity.of("tiny", config, ESTIMATORS, FIT_CONFIG, root_entropy)


def new_store(directory, config: GridConfig, *, root_entropy: int = 0) -> RunStore:
    return RunStore.create(
        directory, identity_for(config, root_entropy=root_entropy), config.replications
    )


def without_runtime(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows compare equal only up to wall clock: `runtime_seconds` is timing."""
    return frame.drop(columns="runtime_seconds").reset_index(drop=True)


def reference_frame(config: GridConfig) -> pd.DataFrame:
    return run_grid(config, estimators=ESTIMATORS, fit_config=FIT_CONFIG)
