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

    `n_trimmed` is the count of units the estimator's own `n_trimmed`
    `Diagnostic` reports as clipped to the trim bounds (see
    `weighting.py`/`doubly_robust.py`, where it is computed as
    `fitted.trimmed_mask.sum()` on the same fit that produced
    `treated_scores`/`control_scores`). It is an empirical claim -- "this
    many units actually hit a bound" -- not a claim that the scores shown
    are the trimmed ones; that provenance claim is always true regardless of
    this count and belongs on the axis label in `draw_overlap`, not here.
    `n_trimmed` can legitimately be zero: on a fit with no extreme scores,
    reading `per_unit_propensity` is still correct, and this field says so
    honestly rather than claiming trimming occurred when it did not.

    `score_range` is the realized `(min, max)` of the combined
    `per_unit_propensity` array actually drawn. `cp_causal`'s estimators do
    not expose the propensity clip point itself as a structured
    diagnostic -- it is only interpolated into a free-text interpretation
    string (`PropensityFit.trim_bounds`, via `Diagnostic.interpretation`),
    and the clip point is configurable per fit (`EstimatorConfig.options`,
    read through `resolve_trim`), so it cannot be assumed to equal
    `POSITIVITY_THRESHOLD` either. There is no typed field `overlap_data`
    can read the configured clip point from, so this field deliberately does
    not claim to be that clip point -- it is named for what it actually is,
    the range of the scores on the axes, which equals the configured clip
    point only where at least one unit on each side landed exactly on that
    bound.
    """

    treated_scores: np.ndarray
    control_scores: np.ndarray
    score_range: tuple[float, float]
    n_trimmed: int


def overlap_data(estimate: EffectEstimate, treatment: np.ndarray) -> OverlapData:
    """Split the trimmed propensity scores by treatment arm.

    Reads `per_unit_propensity` by name and lets a `KeyError` propagate when
    it is absent: an estimate without it was not produced by a
    propensity-based estimator, and this function does not guess at a
    substitute.

    Refuses when `treatment` does not have one entry per score: a mismatched
    length would split the scores at the wrong index silently rather than
    refuse.

    Also refuses when either arm has zero units after the split -- e.g. a
    dataset where every unit was treated. `draw_overlap` histograms each arm
    independently; an empty arm would still add a legend entry for that arm
    while drawing no visible bars behind it, which reads as an ordinary
    (if narrow) distribution rather than the total absence of a comparison
    group that it actually is. There is no overlap to display when one side
    is empty, so this refuses rather than drawing that.

    `n_trimmed` is read from the estimator's own `n_trimmed` diagnostic
    (a `KeyError` propagates the same way as for `per_unit_propensity` if it
    is absent), not recomputed here, so it always reflects the same fit that
    produced `treated_scores`/`control_scores`.
    """
    scores = np.asarray(estimate.diagnostics["per_unit_propensity"].value, dtype=float)
    treatment = np.asarray(treatment)

    if treatment.shape[0] != scores.shape[0]:
        raise ValueError(
            f"treatment has {treatment.shape[0]} entries but per_unit_propensity "
            f"has {scores.shape[0]}"
        )

    treated_scores = scores[treatment == 1]
    control_scores = scores[treatment == 0]
    if treated_scores.size == 0 or control_scores.size == 0:
        empty_arm = "treated" if treated_scores.size == 0 else "control"
        raise ValueError(
            f"cannot draw an overlap figure: the {empty_arm} arm has zero units, "
            "so there is no overlap between arms to display"
        )

    return OverlapData(
        treated_scores=treated_scores,
        control_scores=control_scores,
        score_range=(float(scores.min()), float(scores.max())),
        n_trimmed=int(estimate.diagnostics["n_trimmed"].value),
    )


def draw_overlap(data: OverlapData, ax: Axes) -> Axes:
    """Draw both arms' propensity-score distributions on `ax`, with a legend.

    The x-axis label states the scores are trimmed (see the module
    docstring): an untrimmed distribution would display an overlap problem
    the estimate did not actually face, so the label makes clear these are
    the scores the estimator weighted with, not a raw refit. That claim is
    about provenance -- which diagnostic the scores were read from -- and is
    always true regardless of `OverlapData.n_trimmed`, which is a separate,
    empirical claim about how many units actually hit a bound on this fit.
    """
    ax.hist(data.treated_scores, bins=20, alpha=0.5, label="treated")
    ax.hist(data.control_scores, bins=20, alpha=0.5, label="control")
    ax.set_xlabel("trimmed propensity score")
    ax.set_ylabel("count")
    ax.legend()
    return ax
