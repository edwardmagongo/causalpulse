"""Goodman-Bacon decomposition of a two-way fixed-effects estimate.

A diagnostic function rather than a registered estimator: it explains a TWFE
estimate rather than producing one of its own, and registering it would put it
in the router's candidate pool where it does not belong.

TWFE on staggered data is a weighted average of every available 2x2 comparison.
Some of those comparisons use *already-treated* units as controls, and when
effects change over time those comparisons carry the wrong sign. Reporting their
share is what turns "TWFE is biased here" from an assertion into a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cp_core.dataset import CausalDataset

from cp_causal._panel import PanelArrays, panel_arrays

ComponentKind = str


@dataclass(frozen=True)
class BaconComponent:
    """One 2x2 comparison inside a two-way fixed-effects estimate."""

    kind: ComponentKind
    treated_cohort: int
    control_cohort: int
    estimate: float
    weight: float


@dataclass(frozen=True)
class BaconDecomposition:
    components: tuple[BaconComponent, ...]
    already_treated_share: float
    weighted_average: float


def _cell_means(panel: PanelArrays) -> np.ndarray:
    grid = np.full((panel.n_units, panel.n_periods), np.nan)
    grid[panel.unit_index, panel.period_index] = panel.outcome
    return grid


def _window_mean(
    grid: np.ndarray, members: np.ndarray, columns: np.ndarray
) -> float:
    if not members.any() or not columns.any():
        return float("nan")
    block = grid[np.ix_(members, columns)]
    finite = block[np.isfinite(block)]
    return float(finite.mean()) if finite.size else float("nan")


def _two_by_two(
    grid: np.ndarray,
    treated: np.ndarray,
    control: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
) -> float:
    return (
        _window_mean(grid, treated, post) - _window_mean(grid, treated, pre)
    ) - (_window_mean(grid, control, post) - _window_mean(grid, control, pre))


def goodman_bacon_decomposition(dataset: CausalDataset) -> BaconDecomposition:
    """Decompose the TWFE estimate into its weighted 2x2 components.

    Weights follow Goodman-Bacon's variance form: a comparison contributes in
    proportion to the treatment variance it carries, which is the product of the
    group share and the within-window treatment variance.
    """
    panel = panel_arrays(dataset)
    grid = _cell_means(panel)
    periods = panel.periods.astype(int)
    cohorts = panel.adoption_periods
    never = panel.cohort == 0

    components: list[BaconComponent] = []
    raw_weights: list[float] = []

    def share(mask: np.ndarray) -> float:
        return float(mask.sum()) / float(panel.n_units)

    for cohort in cohorts:
        treated = panel.cohort == cohort
        pre = periods < cohort
        post = periods >= cohort
        if not pre.any() or not post.any():
            continue

        exposure = float(post.sum()) / float(periods.size)
        treated_variance = exposure * (1.0 - exposure)

        if never.any():
            estimate = _two_by_two(grid, treated, never, pre, post)
            if np.isfinite(estimate):
                weight = share(treated) * share(never) * treated_variance
                components.append(
                    BaconComponent(
                        kind="never_treated_as_control",
                        treated_cohort=cohort,
                        control_cohort=0,
                        estimate=estimate,
                        weight=weight,
                    )
                )
                raw_weights.append(weight)

        for other in cohorts:
            if other == cohort:
                continue
            control = panel.cohort == other
            if not control.any():
                continue

            if other > cohort:
                # The later cohort is still untreated over this window, so it
                # is a valid control: restrict to periods before it adopts.
                window = periods < other
                kind = "later_treated_as_control"
            else:
                # The earlier cohort is ALREADY treated over this window. This
                # is the comparison that carries a negative weight when effects
                # change over time, and the reason TWFE fails on P4.
                window = periods >= other
                kind = "already_treated_as_control"

            pre_window = pre & window
            post_window = post & window
            if not pre_window.any() or not post_window.any():
                continue

            estimate = _two_by_two(grid, treated, control, pre_window, post_window)
            if not np.isfinite(estimate):
                continue
            local_exposure = float(post_window.sum()) / float(window.sum())
            weight = (
                share(treated)
                * share(control)
                * local_exposure
                * (1.0 - local_exposure)
            )
            if weight <= 0:
                continue
            components.append(
                BaconComponent(
                    kind=kind,
                    treated_cohort=cohort,
                    control_cohort=other,
                    estimate=estimate,
                    weight=weight,
                )
            )
            raw_weights.append(weight)

    total = float(sum(raw_weights))
    if total <= 0:
        raise ValueError(
            "no 2x2 comparisons could be formed; the panel has no cohort with "
            "both a pre- and a post-treatment period"
        )

    normalised = tuple(
        BaconComponent(
            kind=c.kind,
            treated_cohort=c.treated_cohort,
            control_cohort=c.control_cohort,
            estimate=c.estimate,
            weight=c.weight / total,
        )
        for c in components
    )
    already = float(
        sum(c.weight for c in normalised if c.kind == "already_treated_as_control")
    )
    average = float(sum(c.weight * c.estimate for c in normalised))

    return BaconDecomposition(
        components=normalised,
        already_treated_share=already,
        weighted_average=average,
    )
