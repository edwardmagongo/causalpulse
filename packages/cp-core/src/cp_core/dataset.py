"""The universal input contract every estimator consumes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OutcomeType = Literal["binary", "continuous"]
_VALID_OUTCOME_TYPES: tuple[str, ...] = ("binary", "continuous")

AssignmentProvenance = Literal["randomised", "natural_experiment", "observational"]
_VALID_PROVENANCE: tuple[str, ...] = (
    "randomised",
    "natural_experiment",
    "observational",
)


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
    # How treatment came to be assigned. Not readable from the data -- it is a
    # claim about how the data came to exist, which is exactly what the
    # `randomised_assignment` check says cannot be tested. The default is the
    # conservative one so an analysis that never declares its provenance can
    # never reach the top validity band by omission.
    assignment_provenance: AssignmentProvenance = "observational"
    # A second outcome that treatment does not affect, used to detect residual
    # confounding. Declared rather than inferred, exactly like
    # `assignment_provenance`: which column is a negative control is a claim
    # about how the data came to exist, and no diagnostic computed from the
    # values can settle it.
    negative_control_outcome: str | None = None

    def __post_init__(self) -> None:
        # The annotation is erased at runtime, so a schema built from a config
        # file could carry `"Binary"` and quietly skip the binary-value check.
        if self.outcome_type not in _VALID_OUTCOME_TYPES:
            raise ValueError(
                f"outcome_type must be one of {list(_VALID_OUTCOME_TYPES)}, "
                f"got {self.outcome_type!r}"
            )
        if self.assignment_provenance not in _VALID_PROVENANCE:
            raise ValueError(
                f"assignment_provenance must be one of {list(_VALID_PROVENANCE)}, "
                f"got {self.assignment_provenance!r}"
            )
        if self.negative_control_outcome is not None:
            if self.negative_control_outcome == self.outcome:
                raise ValueError(
                    "negative_control_outcome must differ from outcome; a column "
                    "compared against itself would always read as clean"
                )
            if self.negative_control_outcome in self.covariates:
                raise ValueError(
                    f"negative_control_outcome {self.negative_control_outcome!r} is "
                    "also a covariate; it is downstream of treatment's confounders, "
                    "and adjusting for it would not be the same analysis"
                )
            # `outcome`/`covariates` were the only collisions checked until this
            # was caught by two independent task reviews in Phase 10: a
            # negative-control refit swaps the outcome column for this one and
            # refits the same estimator, so colliding with `unit`, `treatment`,
            # `time` or `cohort` would silently corrupt or trivialise that
            # refit rather than raise -- declaring `treatment` as the negative
            # control would regress treatment on itself.
            role_columns = {self.unit, self.treatment}
            if self.time is not None:
                role_columns.add(self.time)
            if self.cohort is not None:
                role_columns.add(self.cohort)
            if self.negative_control_outcome in role_columns:
                raise ValueError(
                    f"negative_control_outcome {self.negative_control_outcome!r} "
                    "collides with a role column (unit, treatment, time, or "
                    "cohort); a negative-control refit swaps in this column as "
                    "the outcome, and reusing a role column there would corrupt "
                    "or trivialise that refit rather than test anything"
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
        if self.negative_control_outcome is not None:
            columns.append(self.negative_control_outcome)
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


@dataclass(frozen=True)
class CompleteCaseRecord:
    """What a complete-case reduction removed, and which columns caused it.

    `dropped_by_column` counts the rows in which each column was individually
    missing. A row missing in two columns is counted under both, so these
    counts overlap and do not in general sum to
    `n_rows_before - n_rows_after`. Recorded per column anyway, because
    knowing which column drove the attrition is what tells an analyst whether
    the reduction is plausibly ignorable.
    """

    n_rows_before: int
    n_rows_after: int
    columns_with_missing: tuple[str, ...]
    dropped_by_column: Mapping[str, int]


def complete_cases(dataset: CausalDataset) -> tuple[CausalDataset, CompleteCaseRecord]:
    """Drop every row missing any required column, and record what went.

    Complete-case analysis is unbiased when values are missing completely at
    random and biased in general when they are not, so this is a modelling
    choice rather than a cleanup step. It is a separate function, and not
    something an estimator does on a caller's behalf, so that the choice is
    made deliberately and the record of it survives.

    Columns come from `schema.required_columns` rather than a list written out
    here, so a column added to the schema is considered without this function
    being edited.
    """
    frame = dataset.frame
    columns = list(dataset.schema.required_columns)
    per_column = {column: int(frame[column].isna().sum()) for column in columns}
    with_missing = tuple(column for column in columns if per_column[column])

    keep = frame[columns].notna().all(axis=1)
    kept = frame.loc[keep].reset_index(drop=True)
    if kept.empty:
        raise ValueError(
            f"no complete cases remain: all {len(frame)} rows are missing at "
            f"least one of {sorted(with_missing)}"
        )

    return (
        CausalDataset(frame=kept, schema=dataset.schema),
        CompleteCaseRecord(
            n_rows_before=int(len(frame)),
            n_rows_after=int(len(kept)),
            columns_with_missing=with_missing,
            dropped_by_column=per_column,
        ),
    )
