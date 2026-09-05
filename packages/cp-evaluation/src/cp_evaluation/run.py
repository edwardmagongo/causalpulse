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

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from cp_causal.registry import get_estimator
from cp_causal.router import route
from cp_core.values import EstimatorConfig

from cp_evaluation.grid import Cell, CrossSectionCell, GridConfig, expand_cells
from cp_evaluation.resolve import ResolvedCell, resolve_cell, truth_for

if TYPE_CHECKING:
    from cp_core.results import EffectEstimate

# The refusal `fit` raises when a column it needs carries missing values. Only
# this is caught; anything else propagates, because a benchmark that turns a
# crash into a row reports a number over the survivors and calls it a result.
_REFUSAL = ValueError

ROW_COLUMNS: tuple[str, ...] = (
    "shape",
    "scenario",
    "n_units",
    "confounding_strength",
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
)


def _cell_columns(cell: Cell) -> dict[str, object]:
    if isinstance(cell, CrossSectionCell):
        return {
            "shape": "cross_section",
            "scenario": cell.scenario,
            "n_units": cell.n_units,
            "confounding_strength": cell.confounding_strength,
            "n_periods": np.nan,
            "seed": cell.seed,
        }
    return {
        "shape": "panel",
        "scenario": cell.scenario,
        "n_units": cell.n_units,
        "confounding_strength": np.nan,
        "n_periods": cell.n_periods,
        "seed": cell.seed,
    }


def _blank_row(cell: Cell, method: str, status: str, reason: str) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(ROW_COLUMNS, np.nan)
    row.update(_cell_columns(cell))
    row.update({"method": method, "status": status, "reason": reason})
    return row


def _fitted_row(
    cell: Cell, method: str, estimate: EffectEstimate, resolved: ResolvedCell
) -> dict[str, object]:
    bootstrap = estimate.bootstrap_ci
    row = _blank_row(cell, method, "fitted", "")
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
    return row


def run_grid(
    config: GridConfig,
    *,
    root_entropy: int = 0,
    fit_config: EstimatorConfig | None = None,
    _shuffle: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Fit every eligible estimator in every cell, and record what happened.

    `_shuffle` exists only so a test can prove the output does not depend on
    evaluation order; it permutes the order cells are visited and nothing else.
    Production callers never pass it.
    """
    cells = list(expand_cells(config))
    if _shuffle is not None:
        _shuffle.shuffle(cells)

    fit_config = EstimatorConfig(random_seed=0) if fit_config is None else fit_config

    rows: list[dict[str, object]] = []
    for cell in cells:
        resolved = resolve_cell(cell, root_entropy)
        for decision in route(resolved.dataset):
            method = decision.estimator_name
            if not decision.eligible:
                rows.append(
                    _blank_row(cell, method, "ineligible", "; ".join(decision.reasons))
                )
                continue
            try:
                estimate = get_estimator(method)().fit(resolved.dataset, fit_config)
            except _REFUSAL as exc:
                rows.append(_blank_row(cell, method, "refused", str(exc)))
                continue
            rows.append(_fitted_row(cell, method, estimate, resolved))

    frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
    sort_key = ["shape", "scenario", "n_units", "n_periods", "seed", "method"]
    return frame.sort_values(sort_key).reset_index(drop=True)
