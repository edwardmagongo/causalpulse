"""Callaway and Sant'Anna group-time average treatment effects.

Its own module rather than sharing `did.py`: the cohort-period bookkeeping is
substantial enough that combining it with the OLS-based estimators would produce
a file doing too much.

Every ATT(g, t) is a 2x2 comparison of cohort g against a control group, between
period t and the cohort's own baseline period g - 1. Aggregation over those
cells is a separate, explicit step, because the aggregation scheme changes what
the reported scalar means.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import CounterfactualPoint, EffectEstimate, FitMetadata
from cp_core.values import Diagnostic, EstimatorConfig

from cp_causal._panel import (
    MIN_UNITS_PER_ARM,
    PanelArrays,
    bootstrap_panel_effect,
    cluster_count_diagnostic,
    outcome_scale,
    panel_arrays,
)
from cp_causal._support import (
    config_hash,
    critical_value,
    degeneracy_diagnostic,
    normal_p_value,
    wald_interval,
)
from cp_causal.base import Requirements
from cp_causal.panel_checks import (
    absorbing_treatment_check,
    group_change,
    no_concurrent_intervention_check,
    parallel_trends_check,
    placebo_leads,
    sufficient_pre_periods_check,
    unit_means,
)
from cp_causal.registry import register

_AGGREGATIONS = ("simple", "dynamic", "group", "calendar")
_CONTROL_GROUPS = ("never_treated", "not_yet_treated")


def _option(config: EstimatorConfig, key: str, default: str, valid: tuple[str, ...]) -> str:
    value = config.options.get(key, default)
    if not isinstance(value, str) or value not in valid:
        raise ValueError(
            f"option {key!r} must be one of {list(valid)}, got {value!r}"
        )
    return value


class _Cells:
    """The ATT(g, t) grid, with each cell's weight and analytic standard error."""

    def __init__(self) -> None:
        self.keys: list[tuple[int, int]] = []
        self.values: list[float] = []
        self.weights: list[float] = []
        self.standard_errors: list[float] = []

    def add(
        self,
        cohort: int,
        period: int,
        value: float,
        weight: float,
        standard_error: float,
    ) -> None:
        self.keys.append((cohort, period))
        self.values.append(value)
        self.weights.append(weight)
        self.standard_errors.append(standard_error)

    @property
    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.keys, dtype=float).reshape(-1, 2),
            np.asarray(self.values, dtype=float),
            np.asarray(self.weights, dtype=float),
            np.asarray(self.standard_errors, dtype=float),
        )


def _group_mean(grid: np.ndarray, members: np.ndarray, column: int) -> float:
    if not members.any():
        return float("nan")
    values = grid[members, column]
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def compute_cells(
    panel: PanelArrays, control_group: str
) -> tuple[_Cells, np.ndarray]:
    """Every ATT(g, t) for t >= g, plus the mask of units used as controls.

    The baseline for cohort g is period g - 1, its last untreated period. A
    cohort adopting in the first observed period has no such baseline and is
    dropped: using period g as its own baseline would fabricate a zero effect.
    """
    grid = unit_means(panel)
    cells = _Cells()
    control_union = np.zeros(panel.n_units, dtype=bool)

    for cohort in panel.adoption_periods:
        baseline = panel.period_position(cohort - 1)
        if baseline < 0:
            continue
        members = panel.cohort == cohort
        if not members.any():
            continue

        for column, period in enumerate(panel.periods.astype(int)):
            if period < cohort:
                continue
            if control_group == "never_treated":
                controls = panel.cohort == 0
            else:
                # Not-yet-treated: units whose own adoption is still ahead of
                # this period, plus the never-treated. Excluding cohort g
                # itself is what keeps the comparison honest.
                controls = (panel.cohort == 0) | (
                    (panel.cohort > period) & (panel.cohort != cohort)
                )
            if not controls.any():
                continue

            treated_change, treated_var, n_treated = group_change(
                grid, members, column, baseline
            )
            control_change, control_var, n_control = group_change(
                grid, controls, column, baseline
            )
            if not np.isfinite(treated_change) or not np.isfinite(control_change):
                continue

            # The two arms are independent samples of unit-level changes, so
            # the variance of their difference is the sum of the two means'
            # variances. Analytic and cluster-robust, per design section 6.5;
            # the AGGREGATE standard error is bootstrap-based instead, because
            # cells share control units and are not independent of each other.
            standard_error = float(
                np.sqrt(treated_var / n_treated + control_var / n_control)
            )

            control_union |= controls
            cells.add(
                cohort=cohort,
                period=int(period),
                value=treated_change - control_change,
                weight=float(members.sum()),
                standard_error=standard_error,
            )

    return cells, control_union


def aggregate(cells: _Cells, scheme: str) -> tuple[float, Mapping[int, float]]:
    """Collapse the cell grid to a scalar, plus the scheme's own breakdown.

    `simple` weights every treated cell by its cohort size. The others group
    first and then average the group means, which is why they can disagree with
    `simple` even on the same cells — a difference of definition, not a defect.
    """
    keys, values, weights, _ = cells.as_arrays
    if values.size == 0:
        return float("nan"), {}

    if scheme == "simple":
        return float(np.average(values, weights=weights)), {}

    if scheme == "dynamic":
        grouping = (keys[:, 1] - keys[:, 0]).astype(int)
    elif scheme == "group":
        grouping = keys[:, 0].astype(int)
    else:
        grouping = keys[:, 1].astype(int)

    breakdown: dict[int, float] = {}
    for key in sorted(set(grouping.tolist())):
        mask = grouping == key
        breakdown[int(key)] = float(np.average(values[mask], weights=weights[mask]))

    overall = float(
        np.average(
            list(breakdown.values()),
            weights=[float((grouping == k).sum()) for k in breakdown],
        )
    )
    return overall, breakdown


@register
class CallawaySantAnna:
    """Group-time average treatment effects, robust to staggered adoption.

    Unlike two-way fixed effects, no already-treated unit is ever used as a
    control, so effects that differ across cohorts do not contaminate the
    estimate. It still assumes parallel trends, which is why scenario P5 breaks
    it too.
    """

    name = "callaway_santanna"
    requirements = Requirements(
        needs_panel=True,
        min_pre_periods=1,
        min_post_periods=1,
        min_treated_units=MIN_UNITS_PER_ARM,
        # Zero, not one. `Requirements.unmet` counts never-treated units, and
        # this estimator is explicitly designed to run without any: it falls
        # back to the not-yet-treated. Requiring one would make the router
        # exclude it from precisely the dataset its fallback exists for.
        min_control_units=0,
    )

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate:
        started = time.perf_counter()
        panel = panel_arrays(dataset)

        scheme = _option(config, "aggregation", "simple", _AGGREGATIONS)
        requested = _option(config, "control_group", "never_treated", _CONTROL_GROUPS)

        # The fallback is recorded rather than silent: the choice changes the
        # estimate, and two runs differing for an invisible reason is worse than
        # either answer.
        used = requested
        if requested == "never_treated" and panel.n_never_treated_units == 0:
            used = "not_yet_treated"

        cells, control_union = compute_cells(panel, used)
        if not cells.values:
            raise ValueError(
                "no group-time cells could be formed: every cohort lacks either "
                "a pre-treatment baseline period or an available control group"
            )
        point, breakdown = aggregate(cells, scheme)

        def point_from(indices: np.ndarray) -> float:
            sub_units = np.unique(panel.unit_index[indices])
            if sub_units.size < MIN_UNITS_PER_ARM:
                return float("nan")
            remapped = np.searchsorted(sub_units, panel.unit_index[indices])
            sub = PanelArrays(
                outcome=panel.outcome[indices],
                treatment=panel.treatment[indices],
                unit_index=remapped,
                period_index=panel.period_index[indices],
                units=sub_units,
                periods=panel.periods,
                cohort=panel.cohort[sub_units],
            )
            sub_cells, _ = compute_cells(sub, used)
            if not sub_cells.values:
                return float("nan")
            return aggregate(sub_cells, scheme)[0]

        bootstrap_result = bootstrap_panel_effect(point_from, panel, config)
        # Aggregate inference is bootstrap-based: the cells share control units,
        # so treating them as independent would understate the standard error.
        se = float(bootstrap_result.standard_error)

        keys, values, weights, cell_errors = cells.as_arrays
        treated_units = panel.cohort > 0
        # The units that actually served as comparison units, not the
        # never-treated count: under the not-yet-treated fallback the latter is
        # zero, and EffectEstimate requires n_control to be positive.
        n_control = int(control_union.sum())

        treated_rows = panel.treatment == 1
        observed = float(panel.outcome[treated_rows].mean())
        critical = critical_value(config.ci_level)

        # Keys (which (cohort, period) each cell belongs to) are not needed
        # here: HierarchicalDiD, `placebo_leads`'s other caller, uses them to
        # pick a nearest lead, but the joint Wald test below tests every lead
        # at once.
        _, pre_leads, pre_covariance = placebo_leads(panel, used)

        return EffectEstimate(
            estimand="ATT",
            method=self.name,
            point=point,
            se=se,
            ci=wald_interval(point, se, config.ci_level),
            # "bootstrap", not "analytic": `se` above IS the cluster-bootstrap
            # standard error, so labelling the interval analytic would
            # misdescribe where it came from. Per-cell standard errors are
            # analytic and reported separately as `att_gt_standard_errors`.
            ci_basis="bootstrap",
            bootstrap_ci=bootstrap_result.interval,
            p_value=normal_p_value(point, se),
            n_treated=int(treated_units.sum()),
            n_control=n_control,
            scale=outcome_scale(dataset),
            diagnostics={
                "att_gt_keys": Diagnostic(
                    name="att_gt_keys",
                    value=keys,
                    unit="(cohort, period)",
                    interpretation=(
                        "Cohort and period identifying each group-time cell "
                        "below, in the same order."
                    ),
                ),
                "att_gt_cells": Diagnostic(
                    name="att_gt_cells",
                    value=values,
                    unit=None,
                    interpretation=(
                        "ATT for each (cohort, period) cell, each a 2x2 "
                        "comparison against the control group between that "
                        "period and the cohort's last untreated period."
                    ),
                ),
                "att_gt_weights": Diagnostic(
                    name="att_gt_weights",
                    value=weights,
                    unit="units",
                    interpretation="Cohort size carried by each cell.",
                ),
                "att_gt_standard_errors": Diagnostic(
                    name="att_gt_standard_errors",
                    value=cell_errors,
                    unit=None,
                    interpretation=(
                        "Analytic, unit-clustered standard error for each cell, "
                        "from the spread of within-unit changes in the two arms. "
                        "The aggregate standard error is bootstrap-based "
                        "instead: cells share control units, so treating them "
                        "as independent would understate it."
                    ),
                ),
                "aggregate_keys": Diagnostic(
                    name="aggregate_keys",
                    value=np.asarray(sorted(breakdown), dtype=float),
                    unit=scheme,
                    interpretation=(
                        f"Keys of the {scheme!r} aggregation; empty for "
                        "'simple', which produces no breakdown."
                    ),
                ),
                "aggregate_values": Diagnostic(
                    name="aggregate_values",
                    value=np.asarray(
                        [breakdown[k] for k in sorted(breakdown)], dtype=float
                    ),
                    unit=None,
                    interpretation=f"Aggregated ATT per {scheme!r} key.",
                ),
                "control_group": Diagnostic(
                    name="control_group",
                    value=float(used == "not_yet_treated"),
                    unit=None,
                    interpretation=(
                        "Comparison units were the never-treated."
                        if used == "never_treated"
                        else (
                            "No never-treated units exist, so this fit fell back "
                            "to the not-yet-treated as controls. The choice "
                            "changes the estimate and is recorded for that "
                            "reason."
                        )
                    ),
                ),
                "n_group_time_cells": Diagnostic(
                    name="n_group_time_cells",
                    value=float(values.size),
                    unit="cells",
                    interpretation=(
                        "Group-time cells the aggregate rests on. A cohort with "
                        "no pre-treatment baseline period contributes none."
                    ),
                ),
                "cluster_count": cluster_count_diagnostic(panel.n_units),
                "bootstrap_degenerate_replicates": degeneracy_diagnostic(
                    bootstrap_result.n_degenerate, min_arm_size=MIN_UNITS_PER_ARM
                ),
            },
            assumptions=(
                parallel_trends_check(pre_leads, pre_covariance),
                absorbing_treatment_check(dataset),
                sufficient_pre_periods_check(panel, minimum=1),
                no_concurrent_intervention_check(),
            ),
            counterfactual=CounterfactualPoint(
                observed_mean=observed,
                predicted_absent_treatment=observed - point,
                lower=observed - point - critical * se,
                upper=observed - point + critical * se,
            ),
            fit_metadata=FitMetadata.capture(
                estimator=self.name,
                config_hash=config_hash(config),
                runtime_seconds=time.perf_counter() - started,
                random_seed=config.random_seed,
            ),
        )
