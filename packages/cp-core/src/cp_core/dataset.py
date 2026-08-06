"""The universal input contract every estimator consumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OutcomeType = Literal["binary", "continuous"]


@dataclass(frozen=True)
class CausalSchema:
    """Describes which columns of a frame play which causal role."""

    unit: str
    treatment: str
    outcome: str
    covariates: tuple[str, ...]
    outcome_type: OutcomeType
    time: str | None = None
    treatment_start: object | None = None
    cohort: str | None = None

    @property
    def is_panel(self) -> bool:
        return self.time is not None

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = [self.unit, self.treatment, self.outcome, *self.covariates]
        if self.time is not None:
            columns.append(self.time)
        if self.cohort is not None:
            columns.append(self.cohort)
        return tuple(columns)


@dataclass(frozen=True)
class CausalDataset:
    """A long-format dataset paired with its causal schema.

    Validated on construction so that no estimator has to re-check structure.
    """

    frame: pd.DataFrame
    schema: CausalSchema

    def __post_init__(self) -> None:
        self._validate_columns_present()
        self._validate_treatment_is_binary()
        self._validate_outcome_type()
        self._validate_no_duplicate_observations()

    def _validate_columns_present(self) -> None:
        missing = [c for c in self.schema.required_columns if c not in self.frame.columns]
        if missing:
            raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    def _validate_treatment_is_binary(self) -> None:
        values = pd.unique(self.frame[self.schema.treatment].dropna())
        unexpected = set(values) - {0, 1, True, False}
        if unexpected:
            raise ValueError(
                f"treatment column {self.schema.treatment!r} must be binary; "
                f"found unexpected values {sorted(map(str, unexpected))}"
            )

    def _validate_outcome_type(self) -> None:
        if self.schema.outcome_type != "binary":
            return
        values = pd.unique(self.frame[self.schema.outcome].dropna())
        unexpected = set(values) - {0, 1, True, False}
        if unexpected:
            raise ValueError(
                f"outcome_type='binary' but column {self.schema.outcome!r} contains "
                f"non-binary values {sorted(map(str, unexpected))[:5]}"
            )

    def _validate_no_duplicate_observations(self) -> None:
        keys = [self.schema.unit]
        if self.schema.time is not None:
            keys.append(self.schema.time)
        if self.frame.duplicated(subset=keys).any():
            raise ValueError(
                f"frame contains duplicate observations for key {keys}; "
                "expected one row per unit" + (" per period" if self.schema.time else "")
            )

    @property
    def n_units(self) -> int:
        return int(self.frame[self.schema.unit].nunique())

    @property
    def n_treated_units(self) -> int:
        return self._units_by_arm(treated=True)

    @property
    def n_control_units(self) -> int:
        return self._units_by_arm(treated=False)

    def _units_by_arm(self, *, treated: bool) -> int:
        ever = self.frame.groupby(self.schema.unit)[self.schema.treatment].max()
        mask = ever.astype(bool) if treated else ~ever.astype(bool)
        return int(np.count_nonzero(mask))
