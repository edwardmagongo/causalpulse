"""The specification curve and the placebo distribution.

Both figures show how an effect estimate behaves against a family of
alternatives that were not the one reported, and both carry a hazard this
module exists to close off:

- `SpecificationCurve.capped` records whether the search over alternative
  specifications was truncated before every combination was tried. A capped
  curve drawn exactly like an uncapped one overstates how much of the
  specification space was actually explored, so `draw_specification_curve`
  states it in the title whenever `capped` is `True`.
- `PlaceboTestResult.p_value` is not one statistic across every producer of
  that type, and the type's own docstring says so directly: of
  `FalsificationResult`'s three `PlaceboTestResult`-typed slots, two
  (`placebo_dates`, `randomization_inference`) rank the absolute effect
  against absolute placebo effects, while the third, `placebo_units` on a
  donor pool, carries Abadie's post-to-pre-period RMSPE ratio rank instead.
  Both read the same way on their own -- a rank p-value on `[0, 1]` -- but
  they are not comparable to each other as measurements of the same
  quantity.
  `placebo_data` therefore takes the statistic's name from its caller as a
  required, keyword-only argument with no default, so a caller cannot forget
  to say which one it is, and `draw_placebo` names it in the figure rather
  than implying one shared meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cp_core.results import PlaceboTestResult, SpecificationCurve
from matplotlib.axes import Axes


@dataclass(frozen=True)
class SpecificationCurveData:
    """Everything a specification-curve figure draws, sorted by point estimate.

    `original_position` locates, within this sorted order, the specification
    that `SpecificationCurve.original_index` names as the one actually
    reported elsewhere. It is tracked through the sort by index, not
    recovered afterwards by searching the sorted points for the original's
    value -- two specifications can land on the same point estimate, and a
    value search would then find whichever tied entry sorts first rather
    than the specific one the caller named.

    `capped` is carried through unchanged from `SpecificationCurve.capped`
    (see the module docstring) so `draw_specification_curve` can state it
    without a second read of the source result.
    """

    points: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    original_position: int
    capped: bool


def specification_curve_data(curve: SpecificationCurve) -> SpecificationCurveData:
    """Sort a specification curve's refits by point estimate, tracking the original.

    `SpecificationCurve.original_index` indexes `curve.specifications` in its
    original, unsorted order. This function sorts the *indices* by their
    point estimate (Python's sort is stable, so tied point estimates keep
    their original relative order) and then looks up where
    `original_index` landed in that sorted order -- rather than sorting the
    point values and searching for a match, which a tie would make
    ambiguous. See `SpecificationCurveData.original_position`.
    """
    order = sorted(
        range(len(curve.specifications)), key=lambda i: curve.specifications[i].point
    )
    original_position = order.index(curve.original_index)

    points = tuple(curve.specifications[i].point for i in order)
    lower = tuple(curve.specifications[i].ci.lower for i in order)
    upper = tuple(curve.specifications[i].ci.upper for i in order)

    return SpecificationCurveData(
        points=points,
        lower=lower,
        upper=upper,
        original_position=original_position,
        capped=curve.capped,
    )


def draw_specification_curve(data: SpecificationCurveData, ax: Axes) -> Axes:
    """Draw one point per specification, sorted by point estimate, with intervals.

    The specification actually reported elsewhere (`original_position`) is
    drawn with its own `scatter` call, in a distinct colour and a larger
    marker, so a reader can pick out the reported result among the
    alternatives rather than it being lost among peers -- the same concern
    that drives the forest plot's per-row estimand label in
    `cp_figures.effects`.

    The title states when the curve was capped (see the module docstring):
    a capped curve did not try every combination in its specification
    space, and drawing it exactly like an uncapped one would overstate how
    much of that space was actually explored.
    """
    x = np.arange(len(data.points))
    points = np.asarray(data.points)
    lower_err = points - np.asarray(data.lower)
    upper_err = np.asarray(data.upper) - points

    ax.errorbar(
        x,
        points,
        yerr=(lower_err, upper_err),
        fmt="none",
        ecolor="tab:blue",
        capsize=3,
    )

    is_original = np.arange(len(data.points)) == data.original_position
    other = ~is_original
    ax.scatter(
        x[other], points[other], color="tab:blue", zorder=3, label="alternative specification"
    )
    ax.scatter(
        x[is_original],
        points[is_original],
        color="tab:red",
        s=80,
        zorder=4,
        label="specification as reported",
    )

    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("specification (sorted by point estimate)")
    ax.set_ylabel("effect estimate")
    title = "specification curve"
    if data.capped:
        title += " (capped: not every combination in the specification space was tried)"
    ax.set_title(title)
    ax.legend()
    return ax


@dataclass(frozen=True, eq=False)
class PlaceboData:
    """Everything a placebo-distribution figure draws.

    `eq=False` because `placebo_effects` is an ndarray and a generated
    `__eq__` would raise on the ambiguous truth value of an array, the same
    reason `PlaceboTestResult` itself is `eq=False`.

    `statistic` names what `p_value` ranks (see the module docstring); it is
    carried alongside the numbers precisely because the numbers alone cannot
    say which statistic produced them.
    """

    placebo_effects: np.ndarray
    real_effect: float
    p_value: float
    statistic: str


def placebo_data(result: PlaceboTestResult, *, statistic: str) -> PlaceboData:
    """Pair a placebo test's real effect and distribution with the name of its statistic.

    `statistic` is required and keyword-only, with no default: `p_value` is
    not one statistic across every producer of `PlaceboTestResult` (see the
    module docstring), so there is no reasonable default to fall back on,
    and a caller must say which rank statistic this particular result's
    `p_value` is.
    """
    return PlaceboData(
        placebo_effects=np.asarray(result.placebo_effects, dtype=float),
        real_effect=result.real_effect,
        p_value=result.p_value,
        statistic=statistic,
    )


def draw_placebo(data: PlaceboData, ax: Axes) -> Axes:
    """Draw the placebo-effect distribution with the real effect marked, on `ax`.

    The x-axis label names `data.statistic` alongside the p-value, and the
    title repeats it: this figure is drawn for one specific rank statistic
    (see the module docstring), and a reader must not be left to assume the
    p-value means the same thing here as it did in some other placebo
    figure drawn from a different slot of `FalsificationResult`.
    """
    ax.hist(data.placebo_effects, color="tab:blue", alpha=0.7, label="placebo effects")
    ax.axvline(data.real_effect, color="tab:red", linewidth=2, label="real effect")
    ax.set_xlabel(f"effect ({data.statistic}, p = {data.p_value:.3f})")
    ax.set_ylabel("count")
    ax.set_title(f"placebo distribution -- {data.statistic}")
    ax.legend()
    return ax
