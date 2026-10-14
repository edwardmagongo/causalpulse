"""The universal input contract every estimator consumes."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

# The largest magnitude a numeric value may have. Float64 holds up to about
# 1.8e308, and a variance or a sum of squares adds n squares of the data: at
# 1e150 a square is 1e300, so up to about 1e8 rows stay finite. Measured on
# LaLonde on 2026-10-14, every estimator computed a finite result with one
# value at 1e150 and failed at 1e155 with a library's message naming no column.
MAGNITUDE_BOUND = 1e150

OutcomeType = Literal["binary", "continuous"]
_VALID_OUTCOME_TYPES: tuple[str, ...] = ("binary", "continuous")

AssignmentProvenance = Literal["randomised", "natural_experiment", "observational"]
_VALID_PROVENANCE: tuple[str, ...] = (
    "randomised",
    "natural_experiment",
    "observational",
)

# Why two primary roles may not share a column, keyed by the pair, so the
# refusal can say what would have gone wrong rather than only that the names
# must differ. `cohort` is absent: all four of its collisions fail the same
# way and share one reason, built where it is raised.
_ROLE_COLLISION_REASONS: dict[frozenset[str], str] = {
    frozenset({"unit", "treatment"}): (
        "treatment is binary, so the frame could hold at most two units, one "
        "per arm, and nothing would vary within an arm to estimate from"
    ),
    frozenset({"unit", "outcome"}): (
        "an identifier is a label rather than a measurement, and an effect on "
        "a label is not a quantity"
    ),
    frozenset({"treatment", "outcome"}): (
        "an estimator would regress the column on itself and report an "
        "identity as though it were an effect"
    ),
    frozenset({"unit", "time"}): (
        "every unit would sit in a period of its own, so no unit is observed "
        "twice and there is nothing to difference over time"
    ),
    frozenset({"treatment", "time"}): (
        "treatment would switch on for every unit at once with the period, "
        "leaving no untreated unit in a treated period to compare against"
    ),
    frozenset({"outcome", "time"}): (
        "the outcome would be the period itself, which no treatment moves"
    ),
}

# Why a primary role's column may not also be adjusted for.
_COVARIATE_COLLISION_REASONS: dict[str, str] = {
    "unit": (
        "an identifier is a label, and adjusting for its numeric value would "
        "treat an arbitrary numbering as a confounder"
    ),
    "treatment": (
        "the adjustment would be perfectly collinear with treatment, and the "
        "effect is not identified"
    ),
    "outcome": (
        "adjusting for the outcome explains it completely and leaves "
        "treatment nothing to account for"
    ),
    "time": (
        "the period is the index a panel estimator differences over, not a "
        "characteristic of a unit to adjust for"
    ),
    "cohort": (
        "cohort is fixed by when treatment started, so adjusting for it "
        "conditions on the treatment itself"
    ),
}


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
    # Columns an imputation model may use and no estimator may. Declared
    # rather than inferred, exactly like `negative_control_outcome`: whether a
    # column is a legitimate auxiliary is a claim about how the data came to
    # exist, and no diagnostic computed from the values can settle it.
    #
    # Deliberately NOT part of `required_columns`, which is a divergence from
    # `negative_control_outcome`. That property means "an estimator needs
    # this"; an auxiliary is a column no estimator may touch, so including it
    # would let the missing-data refusal reject a dataset over a column no fit
    # ever reads.
    auxiliary_columns: tuple[str, ...] = ()

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
        # Each primary role must name a column of its own. The negative
        # control and the auxiliaries below were checked against these roles
        # from the day each was added; the roles were never checked against
        # each other, so a schema naming one column as both treatment and
        # outcome constructed, fitted, and returned a number. Checked first
        # because the refusals below describe a column by the role it plays,
        # which presumes it plays only one.
        roles = {"unit": self.unit, "treatment": self.treatment, "outcome": self.outcome}
        if self.time is not None:
            roles["time"] = self.time
        if self.cohort is not None:
            roles["cohort"] = self.cohort
        for (first, column), (second, other) in itertools.combinations(roles.items(), 2):
            if column != other:
                continue
            if second == "cohort":
                # `_panel_placebo_units` writes placebo cohorts into this
                # column after it has written the placebo treatment, so a
                # shared column is not merely mislabelled: it is clobbered.
                reason = (
                    "cohort holds each unit's first treated period, and a "
                    "placebo refit that rewrites the cohort column would "
                    f"overwrite the {first} with it"
                )
            else:
                reason = _ROLE_COLLISION_REASONS[frozenset({first, second})]
            raise ValueError(f"{first} and {second} both name column {column!r}; {reason}")
        # The same rule `_resolve_nuisance_covariates` applies to covariates
        # supplied through `config.options`, for the declared adjustment set
        # every estimator reads by default.
        for role, column in roles.items():
            if column in self.covariates:
                raise ValueError(
                    f"covariates contains {column!r}, which is already the {role}; "
                    f"{_COVARIATE_COLLISION_REASONS[role]}"
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
        structural = {self.unit, self.treatment, self.outcome, *self.covariates}
        if self.negative_control_outcome is not None:
            structural.add(self.negative_control_outcome)

        # Check each auxiliary column for clashes and describe the actual role
        for aux_col in self.auxiliary_columns:
            if aux_col in structural:
                # Identify which role this column plays
                if aux_col == self.unit:
                    role = "unit"
                elif aux_col == self.treatment:
                    role = "treatment"
                elif aux_col == self.outcome:
                    role = "outcome"
                elif aux_col == self.negative_control_outcome:
                    role = "negative_control_outcome"
                elif aux_col in self.covariates:
                    role = "covariate"
                else:
                    role = "structural column"

                if role == "covariate":
                    raise ValueError(
                        f"auxiliary_columns contains {aux_col!r}, which is also a "
                        "covariate. An auxiliary is a column the imputation model may "
                        "use and no estimator may: a covariate that is also an "
                        "auxiliary would be adjusted for, and adjusting for a "
                        "descendant of the outcome biases every estimator."
                    )
                else:
                    raise ValueError(
                        f"auxiliary_columns contains {aux_col!r}, which is already "
                        f"the {role}. An auxiliary column cannot play multiple roles."
                    )
        if len(set(self.auxiliary_columns)) != len(self.auxiliary_columns):
            raise ValueError(
                f"auxiliary_columns contains duplicates: {self.auxiliary_columns}"
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


def schema_from_dict(data: Mapping[str, Any]) -> CausalSchema:
    """Build a `CausalSchema` from its JSON shape -- the one parser for it.

    JSON has no tuple type, so the two tuple-typed fields arrive as lists and
    are converted here. `covariates` is required and gets no default: a
    schema that omits it would silently mean "no confounders", and the
    estimates would answer a question the user did not ask.
    `auxiliary_columns` genuinely defaults to `()`.

    The manifest reader and the command line's `--schema` file parse this
    shape, and the backend API will too, so callers call this function rather
    than each keeping its own copy.
    """
    fields = dataclasses.fields(CausalSchema)
    unknown = sorted(set(data) - {f.name for f in fields})
    if unknown:
        raise ValueError(f"unknown schema field(s): {unknown}")
    required = {
        f.name
        for f in fields
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"schema is missing required field(s): {missing}")
    values = dict(data)
    for name in ("covariates", "auxiliary_columns"):
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, str):
            raise ValueError(
                f"{name} must be a list of column names, not a single string; "
                f"got {value!r}"
            )
        values[name] = tuple(value)
    values["auxiliary_columns"] = tuple(values.get("auxiliary_columns", ()))
    return CausalSchema(**values)


# The largest magnitude below which every whole number is exact as a float.
MAX_EXACT_PERIOD = 2**53


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
        self._validate_no_duplicated_labels()
        self._validate_unit_column()
        self._validate_numeric_roles()
        self._validate_finite_values()
        self._validate_magnitude()
        self._validate_treatment_missing_values()
        self._validate_treatment_is_binary()
        self._validate_outcome_type()
        self._validate_time_column()
        self._validate_no_duplicate_observations()

    def _validate_columns_present(self) -> None:
        missing = [c for c in self.schema.required_columns if c not in self.frame.columns]
        if missing:
            raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    def _validate_no_duplicated_labels(self) -> None:
        """Refuse a frame in which a column the schema names appears twice.

        `frame[label]` is then a frame and not a column, and what each later
        check and each estimator makes of that differs: an infinity under a
        duplicated covariate label used to go unseen, and a duplicated
        treatment raised pandas' own complaint about an ambiguous truth value.
        `read_csv` renames duplicates (`x`, `x.1`), so only a frame built in
        code has one. A duplicated label the schema does not name is left
        alone.
        """
        if self.frame.columns.is_unique:
            return
        named = dict.fromkeys((*self.schema.required_columns, *self.schema.auxiliary_columns))
        counts = self.frame.columns.value_counts()
        duplicated = sorted(column for column in named if counts.get(column, 0) > 1)
        if duplicated:
            raise ValueError(
                f"columns {duplicated} appear more than once in the frame; each column "
                "the schema names must appear once."
            )

    def _validate_unit_column(self) -> None:
        """Refuse a unit column with a gap, or one that mixes numbers and text.

        A missing unit is refused here for the reason a missing treatment or
        period is: a row that belongs to no unit is not an observation of
        anything. Before this a panel with one crashed while it was being
        routed, on a comparison of text with `nan`. The message lists what
        pandas reads as missing because that is how a real unit goes missing:
        a country coded `NA` is read as a gap.

        A unit column mixing numbers and text cannot be sorted, which every
        panel estimator does, and holds `5` and `'5'` as two units. Only a
        column of dtype `object` is scanned for that. A categorical column
        can mix them too and is not looked at.
        """
        name = self.schema.unit
        column = self.frame[name]
        if column.isna().any():
            raise ValueError(
                f"unit column {name!r} contains missing values. pandas reads NA, N/A, "
                "n/a, NaN, null, NULL, None and an empty cell as missing, so a unit "
                "with one of those names has to be renamed."
            )
        if column.dtype == object:
            is_text = column.map(lambda value: isinstance(value, str))
            if is_text.any() and not is_text.all():
                raise ValueError(
                    f"unit column {name!r} mixes numbers and text, so its values cannot "
                    "be ordered or compared. Make every unit a number or every unit text."
                )

    def _validate_numeric_roles(self) -> None:
        """Refuse a column that must be numeric and was not read as numbers.

        The treatment, the outcome, the covariates, the cohort, the
        negative-control outcome and the auxiliary columns are all used as
        numbers. `read_csv` and `float()` disagree about what a number is:
        `1_000`, or `inf` followed by a space, makes pandas read the column as
        text (in a large file, read in chunks, as an `object` column mixing
        numbers and text), and yet every cell converts when an estimator asks
        for floats. Before this rule such a column was analysed as numbers,
        and an infinity in it went past `_validate_finite_values`, which looks
        only at numeric dtypes.

        The unit is an identifier and may be text. The time column keeps its
        own rule, in `_validate_time_column`. An auxiliary column absent from
        the frame is passed over, since nothing here requires it to be present.

        Nothing is converted on the caller's behalf: the frame is left as it
        was read, so the bytes a manifest names still parse to this dataset.
        """
        schema = self.schema
        roles = [schema.treatment, schema.outcome, *schema.covariates]
        if schema.cohort is not None:
            roles.append(schema.cohort)
        if schema.negative_control_outcome is not None:
            roles.append(schema.negative_control_outcome)
        roles.extend(c for c in schema.auxiliary_columns if c in self.frame.columns)
        offending = sorted(
            column
            for column in dict.fromkeys(roles)
            if not pd.api.types.is_numeric_dtype(self.frame[column])
        )
        if offending:
            dtypes = ", ".join(f"{column}: {self.frame[column].dtype}" for column in offending)
            raise ValueError(
                f"columns {offending} must hold numbers, and were not read as numbers "
                f"({dtypes}). Every value has to be a plain number: look for stray "
                "text, a space after a value, or digit grouping such as 1_000 or 1,000."
            )

    def _validate_finite_values(self) -> None:
        """Refuse an infinite value in a numeric column the schema names.

        This runs before the treatment's own checks, so an infinite treatment
        value is reported as infinite and not as a third treatment level.

        A missing value in the outcome or a covariate is deliberately not
        refused here: `complete_cases` and imputation take a dataset with
        gaps, so one has to be constructible, and `require_no_missing` refuses
        it when an estimator is fitted. Nothing takes a dataset with an
        infinity. Before this check one in a covariate failed some estimators
        and not others, each with a library's own message naming no column,
        and one in a panel's time column was fitted and returned an estimate.

        Only columns with a numeric dtype are looked at. The roles that must be
        numeric have been through `_validate_numeric_roles` by now, so what is
        passed over is a unit or a time column that is not numeric, and an
        auxiliary column absent from the frame. A unit column of dtype `object`
        that holds only numbers is not inspected, so an infinity in one is not
        seen: the unit is used as a label and never as a number. A time column
        that is neither numeric nor dates is refused by `_validate_time_column`.
        """
        declared = (*self.schema.required_columns, *self.schema.auxiliary_columns)
        offending = sorted(
            column
            for column in dict.fromkeys(declared)
            if column in self.frame.columns
            and pd.api.types.is_numeric_dtype(self.frame[column])
            # Through float, with a gap as nan, so a nullable column is read
            # the same way as a plain one: a gap is not an infinity.
            and np.isinf(self.frame[column].to_numpy(dtype=float, na_value=np.nan)).any()
        )
        if offending:
            raise ValueError(
                f"columns {offending} contain infinite values (inf or -inf), and no "
                "estimator here can use them. Correct those values or remove those "
                "rows, and supply the data again."
            )

    def _validate_magnitude(self) -> None:
        """Refuse a value too large in magnitude to compute with.

        Every numeric column the schema names, except the unit, the treatment
        and the time column, which have stricter rules of their own and keep
        their messages: the unit is a label, the treatment is 0 or 1, and a
        period is a whole number no larger than 2**53. An infinity has been
        refused by `_validate_finite_values` before this runs; a missing value
        is not compared.
        """
        schema = self.schema
        own_rule = {schema.unit, schema.treatment, schema.time}
        declared = (*schema.required_columns, *schema.auxiliary_columns)
        found: dict[str, float] = {}
        for column in dict.fromkeys(declared):
            if column in own_rule or column not in self.frame.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.frame[column]):
                continue
            values = self.frame[column].to_numpy(dtype=float, na_value=np.nan)
            over = np.flatnonzero(np.abs(values) > MAGNITUDE_BOUND)
            if over.size:
                found[column] = float(values[over[0]])
        if found:
            offending = sorted(found)
            listed = ", ".join(f"{column}: {found[column]!r}" for column in offending)
            raise ValueError(
                f"columns {offending} hold values too large to compute with ({listed}); "
                "every value must be at most 1e150 in magnitude. Check the column's units, "
                "or correct those values, and supply the data again."
            )

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
        """Structural check only: the time column must be orderable, complete and exact.

        Panel *balance* and whether treatment is absorbing are deliberately not
        validated here. Both are legitimate real-data conditions — attrition
        produces unbalanced panels, and some interventions genuinely switch off
        — so they surface as properties an estimator checks and reports as an
        assumption failure. This mirrors rejecting nulls in the treatment column
        while permitting them in the outcome column: structural impossibilities
        are rejected, adverse data conditions are reported.

        A numeric period has to be a whole number no larger than 2**53. The
        panel code turns periods and cohorts into integers: a fractional
        period was truncated to one that is not in the data, and a period
        beyond the integer range became 9223372036854775807 with a warning.
        2**53 is where whole numbers stop being exact as floats. A column of
        dates is not checked here; it constructs, and the panel code says what
        it needs.
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
        if pd.api.types.is_numeric_dtype(column):
            # Integers are compared as integers: as floats, one just above the
            # limit rounds down onto it and would pass.
            if pd.api.types.is_integer_dtype(column):
                unusable = (column > MAX_EXACT_PERIOD) | (column < -MAX_EXACT_PERIOD)
            else:
                values = column.to_numpy(dtype=float)
                unusable = (values != np.floor(values)) | (np.abs(values) > MAX_EXACT_PERIOD)
            if unusable.any():
                found = column[unusable].iloc[0]
                # A numpy scalar prints as `np.float64(1989.5)`; one from another
                # backend may already be a plain number, with no `item`.
                shown = found.item() if hasattr(found, "item") else found
                raise ValueError(
                    f"time column {self.schema.time!r} must hold whole-numbered periods "
                    f"no larger than 2**53; found {shown!r}. Periods are used as "
                    "integers, so a fractional or very large value would be changed."
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
