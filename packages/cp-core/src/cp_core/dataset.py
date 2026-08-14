"""The universal input contract every estimator consumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OutcomeType = Literal["binary", "continuous"]
_VALID_OUTCOME_TYPES: tuple[str, ...] = ("binary", "continuous")


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

    def __post_init__(self) -> None:
        # The annotation is erased at runtime, so a schema built from a config
        # file could carry `"Binary"` and quietly skip the binary-value check.
        if self.outcome_type not in _VALID_OUTCOME_TYPES:
            raise ValueError(
                f"outcome_type must be one of {list(_VALID_OUTCOME_TYPES)}, "
                f"got {self.outcome_type!r}"
            )

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


# eq=False: this holds array-valued fields, so a generated __eq__ would raise
# on the ambiguous truth value of an array rather than compare. Identity
# equality is the honest semantics for a container of measurements.
@dataclass(frozen=True, eq=False)
class CausalDataset:
    """A long-format dataset paired with its causal schema.

    Validated on construction so that no estimator has to re-check structure.
    """

    frame: pd.DataFrame
    schema: CausalSchema

    def __post_init__(self) -> None:
        self._validate_columns_present()
        self._validate_treatment_missing_values()
        self._validate_treatment_is_binary()
        self._validate_outcome_type()
        self._validate_time_column()
        self._validate_no_duplicate_observations()

    def _validate_columns_present(self) -> None:
        missing = [c for c in self.schema.required_columns if c not in self.frame.columns]
        if missing:
            raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    def _validate_treatment_missing_values(self) -> None:
        if self.frame[self.schema.treatment].isna().any():
            raise ValueError(
                f"treatment column {self.schema.treatment!r} contains missing values"
            )

    def _validate_treatment_is_binary(self) -> None:
        values = pd.unique(self.frame[self.schema.treatment].dropna())
        unexpected = set(values) - {0, 1}
        if unexpected:
            raise ValueError(
                f"treatment column {self.schema.treatment!r} must be binary; "
                f"found unexpected values {sorted(map(str, unexpected))}"
            )

    def _validate_outcome_type(self) -> None:
        if self.schema.outcome_type != "binary":
            return
        values = pd.unique(self.frame[self.schema.outcome].dropna())
        unexpected = set(values) - {0, 1}
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

    def _validate_time_column(self) -> None:
        """Structural check only: the time column must be orderable and complete.

        Panel *balance* and whether treatment is absorbing are deliberately not
        validated here. Both are legitimate real-data conditions — attrition
        produces unbalanced panels, and some interventions genuinely switch off
        — so they surface as properties an estimator checks and reports as an
        assumption failure. This mirrors rejecting nulls in the treatment column
        while permitting them in the outcome column: structural impossibilities
        are rejected, adverse data conditions are reported.
        """
        if self.schema.time is None:
            return
        column = self.frame[self.schema.time]
        if column.isna().any():
            raise ValueError(
                f"time column {self.schema.time!r} contains missing values"
            )
        if not (
            pd.api.types.is_numeric_dtype(column)
            or pd.api.types.is_datetime64_any_dtype(column)
        ):
            raise ValueError(
                f"time column {self.schema.time!r} must be orderable "
                f"(numeric or datetime), got dtype {column.dtype}"
            )

    def _require_panel(self) -> str:
        if self.schema.time is None:
            raise ValueError("dataset is not a panel; it declares no time column")
        return self.schema.time

    @property
    def periods(self) -> np.ndarray:
        """Sorted unique periods."""
        time = self._require_panel()
        return np.sort(np.asarray(pd.unique(self.frame[time])))

    @property
    def n_periods(self) -> int:
        return int(self.periods.size)

    @property
    def is_balanced(self) -> bool:
        """Whether every unit is observed in every period."""
        time = self._require_panel()
        counts = self.frame.groupby(self.schema.unit)[time].nunique()
        return bool((counts == self.n_periods).all())

    @property
    def treatment_is_absorbing(self) -> bool:
        """Whether no unit ever switches treatment back off.

        Reported rather than enforced. DiD is where it changes the answer, so
        the estimators check it; the contract layer only makes it observable.
        """
        time = self._require_panel()
        ordered = self.frame.sort_values([self.schema.unit, time])
        deltas = ordered.groupby(self.schema.unit)[self.schema.treatment].diff()
        return not bool((deltas < 0).any())

    def cohort_by_unit(self) -> pd.Series:
        """First treated period per unit; 0 for never-treated units.

        Derived from the treatment column rather than read from
        `schema.cohort`, so it works on real data carrying no cohort column and
        cannot disagree with the treatment it describes. `0` follows the
        convention in the DiD literature; the panel estimators guard against a
        collision by refusing a dataset whose periods are not strictly positive.
        """
        time = self._require_panel()
        treated_rows = self.frame[self.frame[self.schema.treatment] == 1]
        first = treated_rows.groupby(self.schema.unit)[time].min()
        units = pd.Index(np.sort(np.asarray(pd.unique(self.frame[self.schema.unit]))))
        return first.reindex(units).fillna(0)

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
