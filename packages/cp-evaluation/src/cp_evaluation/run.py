"""Running a grid: every cell, every estimator, one row each.

The three statuses are the load-bearing idea. An estimator that could not run
in a cell is recorded as `ineligible` with the router's own reasons, and one
that refused the data is recorded as `refused` with the exception it raised.
Neither is dropped. Dropping them would compute every downstream metric over a
subset nobody chose and nobody can see: if `aipw` is ineligible below 25
treated units, its absence from the small-n cells has to read as *absent*
rather than as *fine*.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from cp_causal.registry import all_estimators, get_estimator
from cp_causal.router import route
from cp_core.values import EstimatorConfig
from cp_simulation.cate_scenarios import CATE_N_UNITS
from cp_simulation.donor_scenarios import DONOR_SCENARIOS

from cp_evaluation.cate import CATE_METRIC_COLUMNS, cate_row_metrics
from cp_evaluation.grid import (
    CateCell,
    Cell,
    CrossSectionCell,
    DonorPoolCell,
    GridConfig,
    PanelCell,
    expand_cells,
)
from cp_evaluation.resolve import ResolvedCell, resolve_cell, truth_for

if TYPE_CHECKING:
    from cp_causal.base import Estimator
    from cp_core.dataset import CausalDataset
    from cp_core.results import EffectEstimate

# The refusal `fit` raises when a column it needs carries missing values. Only
# this is caught; anything else propagates, because a benchmark that turns a
# crash into a row reports a number over the survivors and calls it a result.
#
# `ValueError` alone is not a reliable signal of that refusal: `cp_causal`
# also raises it for ordinary configuration and programming errors that have
# nothing to do with missing data -- `causal_forest.py` raises it for a
# non-positive tuning option, for instance. So "the documented refusal" is
# not determined by the exception type at all. A caught `ValueError` is
# classified as the refusal only when the dataset itself genuinely carries
# missing values in a column the schema requires
# (`_missing_a_required_column`, which mirrors the check `require_no_missing`
# performs before raising); otherwise it is a bug and is re-raised.
_REFUSAL = ValueError

ROW_COLUMNS: tuple[str, ...] = (
    "shape",
    "scenario",
    "n_units",
    "confounding_strength",
    "treatment_intercept",
    "noise_scale",
    "effect_scale",
    "heterogeneity_scale",
    "realised_treated_share",
    "n_periods",
    "seed",
    "method",
    "status",
    "reason",
    "estimand",
    "truth",
    "point",
    "se",
    "ci_lower",
    "ci_upper",
    "ci_basis",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "p_value",
    "n_treated",
    "n_control",
    "runtime_seconds",
    # Per-fit CATE recovery (cp_evaluation.cate). NaN on every row except a
    # fitted CATE-cell row whose estimate reports `per_unit_cate`.
    *CATE_METRIC_COLUMNS,
)


# The columns that hold text. A shard whose cells all left one of them null (an
# `ineligible` row has no `estimand` or `ci_basis`) is written and read back as
# an all-null column with no string type at all, and concatenating such a shard
# with others yields `object`. The frame `run_grid` builds from every row at once
# infers `str`. Readers of stored shards cast these back, so the two paths give
# the same frame rather than the same values in different dtypes.
STRING_COLUMNS: tuple[str, ...] = (
    "shape",
    "scenario",
    "method",
    "status",
    "reason",
    "estimand",
    "ci_basis",
)


def _missing_a_required_column(dataset: CausalDataset) -> bool:
    """Whether the dataset is missing values in a column its schema requires.

    Mirrors the check `cp_causal._support.require_no_missing` performs before
    raising, without raising itself: this is used to tell a genuine refusal
    apart from an ordinary `ValueError` that happens to be raised on the same
    dataset for an unrelated reason.
    """
    frame = dataset.frame
    return any(frame[column].isna().any() for column in dataset.schema.required_columns)


def _realised_treated_share(resolved: ResolvedCell) -> float:
    """The measured treated fraction of this cell's own generated dataset.

    Spec section 4.1: `treatment_intercept` sets a logit-scale intercept, but
    an intercept is not a quantity a reader of the results thinks in. This is
    measured from the realised data -- the same property
    `tests/simulation/test_axis_declarations.py::_treated_share` checks the
    axis against -- rather than read off `cell.treatment_intercept` or
    recomputed via `_treatment_share`'s sigmoid, either of which would be the
    *declared* mechanism relabelled, not a measured quantity.
    """
    frame = resolved.dataset.frame
    schema = resolved.dataset.schema
    return float(frame[schema.treatment].mean())


def _cell_columns(cell: Cell, resolved: ResolvedCell) -> dict[str, object]:
    """This cell's identity columns. Computed once per cell, not once per row."""
    if isinstance(cell, CrossSectionCell):
        return {
            "shape": "cross_section",
            "scenario": cell.scenario,
            "n_units": cell.n_units,
            "confounding_strength": cell.confounding_strength,
            "treatment_intercept": cell.treatment_intercept,
            "noise_scale": cell.noise_scale,
            "effect_scale": cell.effect_scale,
            "heterogeneity_scale": cell.heterogeneity_scale,
            "realised_treated_share": _realised_treated_share(resolved),
            "n_periods": np.nan,
            "seed": cell.seed,
        }
    if isinstance(cell, DonorPoolCell):
        # DonorPoolCell carries no n_units/n_periods of its own -- DONOR_SCENARIOS
        # fixes both per scenario (grid.py's DonorPoolCell docstring). Recorded
        # here for readability of the output frame, not because they vary.
        spec = DONOR_SCENARIOS[cell.scenario]
        return {
            "shape": "donor_pool",
            "scenario": cell.scenario,
            "n_units": spec.n_units,
            "confounding_strength": np.nan,
            "treatment_intercept": np.nan,
            "noise_scale": np.nan,
            "effect_scale": np.nan,
            "heterogeneity_scale": np.nan,
            "realised_treated_share": np.nan,
            "n_periods": spec.n_periods,
            "seed": cell.seed,
        }
    if isinstance(cell, PanelCell):
        return {
            "shape": "panel",
            "scenario": cell.scenario,
            "n_units": cell.n_units,
            "confounding_strength": np.nan,
            # The four axes Phase 18 adds, and the treated share measured from
            # them, are cross-sectional-only -- `ScenarioParams` reaches a
            # scenario through `build_scenario`, which the panel arm never
            # calls. Follows the same np.nan convention as
            # `confounding_strength` above, rather than inventing a second one.
            "treatment_intercept": np.nan,
            "noise_scale": np.nan,
            "effect_scale": np.nan,
            "heterogeneity_scale": np.nan,
            "realised_treated_share": np.nan,
            "n_periods": cell.n_periods,
            "seed": cell.seed,
        }

    if isinstance(cell, CateCell):
        # Size fixed by the registry (grid.py's CateCell docstring). A CATE
        # dataset is cross-sectional, so its treated share is measured as for
        # a CrossSectionCell; the axis columns are NaN because the arm varies
        # none of them.
        return {
            "shape": "cate",
            "scenario": cell.scenario,
            "n_units": CATE_N_UNITS,
            "confounding_strength": np.nan,
            "treatment_intercept": np.nan,
            "noise_scale": np.nan,
            "effect_scale": np.nan,
            "heterogeneity_scale": np.nan,
            "realised_treated_share": _realised_treated_share(resolved),
            "n_periods": np.nan,
            "seed": cell.seed,
        }

    raise TypeError(f"unhandled cell shape {type(cell).__name__}")


def _blank_row(
    cell_columns: dict[str, object], method: str, status: str, reason: str
) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(ROW_COLUMNS, np.nan)
    row.update(cell_columns)
    row.update({"method": method, "status": status, "reason": reason})
    return row


def _fitted_row(
    cell_columns: dict[str, object],
    method: str,
    estimate: EffectEstimate,
    resolved: ResolvedCell,
) -> dict[str, object]:
    bootstrap = estimate.bootstrap_ci
    row = _blank_row(cell_columns, method, "fitted", "")
    row.update(
        {
            "estimand": estimate.estimand,
            "truth": truth_for(resolved, estimate.estimand),
            "point": estimate.point,
            "se": np.nan if estimate.se is None else estimate.se,
            "ci_lower": estimate.ci.lower,
            "ci_upper": estimate.ci.upper,
            "ci_basis": estimate.ci_basis,
            "bootstrap_ci_lower": np.nan if bootstrap is None else bootstrap.lower,
            "bootstrap_ci_upper": np.nan if bootstrap is None else bootstrap.upper,
            "p_value": np.nan if estimate.p_value is None else estimate.p_value,
            "n_treated": estimate.n_treated,
            "n_control": estimate.n_control,
            "runtime_seconds": estimate.fit_metadata.runtime_seconds,
        }
    )
    per_unit = estimate.diagnostics.get("per_unit_cate")
    if resolved.cate_truth is not None and per_unit is not None:
        row.update(cate_row_metrics(per_unit.value, resolved.cate_truth))
    return row


def resolve_candidates(
    estimators: Sequence[str] | None,
) -> dict[str, type[Estimator]] | None:
    """The registry restricted to `estimators`, or None for every registered one.

    An unknown name is refused by name rather than silently dropped, matching
    `expand_cells`' refusal of an unknown scenario -- a typo'd estimator name
    should not quietly produce a benchmark that omits the method someone
    thought they were measuring.
    """
    registry = all_estimators()
    if estimators is None:
        return None
    unknown = sorted(set(estimators) - set(registry))
    if unknown:
        raise ValueError(f"unknown estimators {unknown}; available: {sorted(registry)}")
    return {name: registry[name] for name in estimators}


def fit_cell(
    cell: Cell,
    *,
    root_entropy: int,
    fit_config: EstimatorConfig,
    candidates: dict[str, type[Estimator]] | None,
) -> list[dict[str, object]]:
    """Every result row for one cell: one per candidate estimator.

    The per-cell body of `run_grid`, extracted so a parallel executor fits a
    cell through exactly the code the serial path does.
    """
    resolved = resolve_cell(cell, root_entropy)
    # Once per cell: these columns describe the cell and its generated
    # data, not the estimator reading it, and `realised_treated_share`
    # means a pass over the treatment column to produce.
    cell_columns = _cell_columns(cell, resolved)
    rows: list[dict[str, object]] = []
    for decision in route(resolved.dataset, candidates):
        method = decision.estimator_name
        if not decision.eligible:
            rows.append(
                _blank_row(cell_columns, method, "ineligible", "; ".join(decision.reasons))
            )
            continue
        try:
            estimate = get_estimator(method)().fit(resolved.dataset, fit_config)
        except _REFUSAL as exc:
            if not _missing_a_required_column(resolved.dataset):
                raise
            rows.append(_blank_row(cell_columns, method, "refused", str(exc)))
            continue
        rows.append(_fitted_row(cell_columns, method, estimate, resolved))
    return rows


# `confounding_strength` and the four axes Phase 18 adds sit right after
# `n_units`, matching where `grid.cell_sort_key` places them for
# cross-sectional cells. Without them, two cells that only differ in one
# of these -- which `expand_cells` assigns the same `seed` range
# independently for each value, including the one-at-a-time extra cells
# `_expand_cross_section` adds that share every other column and differ
# only in a single new axis -- tie on every other column, and pandas'
# stable sort would then leave their relative order wherever evaluation
# happened to visit them: the output would depend on the order cells
# were evaluated in, exactly what `_shuffle` exists to disprove. Listing
# them here makes the key total over the rows a grid can produce.
SORT_KEY: tuple[str, ...] = (
    "shape",
    "scenario",
    "n_units",
    "confounding_strength",
    "treatment_intercept",
    "noise_scale",
    "effect_scale",
    "heterogeneity_scale",
    "n_periods",
    "seed",
    "method",
)


def sort_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in the one order every reader of a result frame sees."""
    return frame.sort_values(list(SORT_KEY)).reset_index(drop=True)


def run_grid(
    config: GridConfig,
    *,
    root_entropy: int = 0,
    fit_config: EstimatorConfig | None = None,
    estimators: Sequence[str] | None = None,
    _shuffle: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Fit every eligible estimator in every cell, and record what happened.

    `estimators`, if given, restricts the run to those estimator names rather
    than every estimator in `cp_causal.registry.all_estimators()`. This
    changes which rows exist: `ineligible` counts are then relative to the
    requested set, not to the whole registry, so a run restricted to two
    estimators reports at most two rows per cell rather than one per
    registered estimator. An unknown name is refused by name (see
    `resolve_candidates`).

    `_shuffle` exists only so a test can prove the output does not depend on
    evaluation order; it permutes the order cells are visited and nothing else.
    Production callers never pass it.
    """
    cells = list(expand_cells(config))
    if _shuffle is not None:
        _shuffle.shuffle(cells)

    fit_config = EstimatorConfig(random_seed=0) if fit_config is None else fit_config
    candidates = resolve_candidates(estimators)

    rows: list[dict[str, object]] = []
    for cell in cells:
        rows.extend(
            fit_cell(
                cell, root_entropy=root_entropy, fit_config=fit_config, candidates=candidates
            )
        )

    frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
    return sort_result_frame(frame)
