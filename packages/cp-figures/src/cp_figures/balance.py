"""The love plot and the propensity overlap.

Both figures display balance and propensity information rather than an
effect estimate, so `cp_figures.guards.require_interval` does not apply here
-- there is no interval to guard. Each carries its own hazard instead:

- The love plot pairs a bare-array diagnostic (`smd_before`/`smd_after`,
  Task 3's `cp_causal.estimators.weighting`/`doubly_robust` output) with
  covariate names supplied separately by the caller. The arrays carry no
  names of their own; if the caller's name list does not line up one-to-one
  with the arrays, every label attaches to the wrong bar and the figure is
  silently wrong. `love_plot_data` refuses rather than guess.
- The overlap figure must show the *trimmed* propensity scores the
  estimator actually weighted with (`per_unit_propensity`), not an
  unclipped refit -- an untrimmed distribution would display an overlap
  problem the estimate never actually faced. The axis label says so.

The |SMD| reference line is read from `cp_causal.propensity.BALANCE_THRESHOLD`
rather than a literal `0.1`, so this figure and the balance assumption check
in `cp_causal` cannot silently drift apart. This is a deliberate, accepted
deviation from Phase 16 design doc section 3, which says `cp_figures` "does
not need `cp_causal`" -- the layers contract (`cp_figures` sits above
`cp_causal`) permits the import, and it is the only way to avoid duplicating
the threshold literal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from cp_causal.propensity import BALANCE_THRESHOLD
from cp_core.results import EffectEstimate
from matplotlib.axes import Axes


@dataclass(frozen=True)
class LovePlotData:
    """Everything a love plot draws, one entry per covariate.

    `covariates` is carried alongside `before`/`after` rather than assumed
    to be the caller's own list, because `love_plot_data` reorders nothing
    -- the tuple recorded here is exactly the order the caller supplied it
    in, which `love_plot_data` has already checked lines up with the
    diagnostic arrays index for index (see the module docstring).
    """

    covariates: tuple[str, ...]
    before: tuple[float, ...]
    after: tuple[float, ...]
    threshold: float


def love_plot_data(estimate: EffectEstimate, covariate_names: Sequence[str]) -> LovePlotData:
    """Pair the before/after balance diagnostics with covariate names.

    Reads `smd_before` and `smd_after` by name and lets a `KeyError`
    propagate when either is absent: an estimate without them was not
    produced by a propensity-based estimator, and this function does not
    guess at a substitute.

    Both diagnostic arrays are in schema covariate order (see
    `cp_causal._support.arrays`, which builds the covariate matrix from
    `dataset.schema.covariates` in that order, and
    `cp_causal.propensity.standardised_mean_differences`, which operates on
    that same matrix column for column). `covariate_names` must be supplied
    in that same order and of the same length; `love_plot_data` checks the
    length rather than trusting the caller, because the diagnostics are bare
    arrays with no names of their own -- a mismatched list would silently
    attach every label to the wrong bar.

    `threshold` is read from `cp_causal.propensity.BALANCE_THRESHOLD` (see
    the module docstring), never a local literal, so this figure cannot
    disagree with the project's own balance assumption check.
    """
    before = np.asarray(estimate.diagnostics["smd_before"].value, dtype=float)
    after = np.asarray(estimate.diagnostics["smd_after"].value, dtype=float)
    names = tuple(covariate_names)

    if len(names) != len(before) or len(names) != len(after):
        raise ValueError(
            f"expected {len(before)} covariate names to match smd_before/smd_after, "
            f"got {len(names)}"
        )

    return LovePlotData(
        covariates=names,
        before=tuple(float(v) for v in before),
        after=tuple(float(v) for v in after),
        threshold=BALANCE_THRESHOLD,
    )


def draw_love_plot(data: LovePlotData, ax: Axes) -> Axes:
    """Draw one row per covariate: a before marker, an after marker, on `ax`.

    Markers are drawn with `ax.scatter` and the connecting segment with
    `ax.hlines`, neither of which appends to `Axes.lines` -- only the two
    `ax.axvline` calls below do. That keeps `Axes.lines` unambiguous for a
    caller (or a test) that wants to isolate the threshold reference lines
    from the markers.

    |SMD| is a two-sided criterion, so the reference line is drawn at both
    `+threshold` and `-threshold`, not just one.
    """
    y = np.arange(len(data.covariates))
    before = np.asarray(data.before)
    after = np.asarray(data.after)

    ax.hlines(y, before, after, colors="grey", linewidth=0.8, zorder=1)
    ax.scatter(before, y, marker="o", color="tab:red", label="before weighting", zorder=2)
    ax.scatter(after, y, marker="o", color="tab:blue", label="after weighting", zorder=2)

    ax.axvline(data.threshold, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(-data.threshold, color="grey", linewidth=0.8, linestyle="--")

    ax.set_yticks(list(y))
    ax.set_yticklabels(list(data.covariates))
    ax.set_xlabel("standardised mean difference")
    ax.legend()
    return ax


@dataclass(frozen=True, eq=False)
class OverlapData:
    """Everything a propensity-overlap figure draws.

    `eq=False` for the same reason as `CounterfactualPathData` in
    `cp_figures.effects`: `treated_scores`/`control_scores` are ndarrays,
    and a generated `__eq__` would raise on the ambiguous truth value of an
    array comparison rather than compare.

    `trimmed` records that `treated_scores`/`control_scores` were read from
    the `per_unit_propensity` diagnostic -- which, at every point it is
    produced in `cp_causal` (`weighting.py`, `doubly_robust.py`), is
    documented to always hold the post-clip scores the weights were
    actually built from, never an unclipped refit. It is a claim about
    *which* diagnostic these scores came from, not an empirical claim that
    this particular fit clipped at least one unit -- `overlap_data` has no
    way to test the latter, since `per_unit_propensity` never exposes an
    unclipped counterpart to compare against.

    `trim_bounds` is the realized `(min, max)` of the combined
    `per_unit_propensity` array. `cp_causal`'s estimators do not expose the
    propensity clip point as a structured diagnostic -- it is only
    interpolated into a free-text interpretation string
    (`PropensityFit.trim_bounds`, via `Diagnostic.interpretation`) -- so
    there is no typed field `overlap_data` can read it from. This realized
    range equals the estimator's configured clip point only where at least
    one unit landed exactly on that bound; otherwise it is merely the
    tightest range this fit actually produced. It is still an honest fact
    about the scores being drawn, and does not risk disagreeing with a
    non-default `trim` option the way a hardcoded default would.
    """

    treated_scores: np.ndarray
    control_scores: np.ndarray
    trim_bounds: tuple[float, float]
    trimmed: bool


def overlap_data(estimate: EffectEstimate, treatment: np.ndarray) -> OverlapData:
    """Split the trimmed propensity scores by treatment arm.

    Reads `per_unit_propensity` by name and lets a `KeyError` propagate when
    it is absent: an estimate without it was not produced by a
    propensity-based estimator, and this function does not guess at a
    substitute.

    Refuses when `treatment` does not have one entry per score: a mismatched
    length would split the scores at the wrong index silently rather than
    refuse.
    """
    scores = np.asarray(estimate.diagnostics["per_unit_propensity"].value, dtype=float)
    treatment = np.asarray(treatment)

    if treatment.shape[0] != scores.shape[0]:
        raise ValueError(
            f"treatment has {treatment.shape[0]} entries but per_unit_propensity "
            f"has {scores.shape[0]}"
        )

    return OverlapData(
        treated_scores=scores[treatment == 1],
        control_scores=scores[treatment == 0],
        trim_bounds=(float(scores.min()), float(scores.max())),
        trimmed=True,
    )


def draw_overlap(data: OverlapData, ax: Axes) -> Axes:
    """Draw both arms' propensity-score distributions on `ax`, with a legend.

    The x-axis label states the scores are trimmed (see `OverlapData.trimmed`
    and the module docstring): an untrimmed distribution would display an
    overlap problem the estimate did not actually face, so the label makes
    clear these are the scores the estimator weighted with, not a raw refit.
    """
    ax.hist(data.treated_scores, bins=20, alpha=0.5, label="treated")
    ax.hist(data.control_scores, bins=20, alpha=0.5, label="control")
    ax.set_xlabel("trimmed propensity score")
    ax.set_ylabel("count")
    ax.legend()
    return ax
