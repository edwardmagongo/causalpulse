"""Panel bookkeeping shared by every DiD-family estimator.

Written once and parameterised rather than copied into each estimator: Phase 2
shipped a diagnostic string that was wrong in three of four estimators purely
because the block had been copy-pasted, and four more estimators land here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import EffectScale
from cp_core.values import Diagnostic, EstimatorConfig
from cp_statistics.bootstrap import BootstrapResult, bootstrap_interval

# Every panel estimator's minimum units per arm, and every bootstrap resample's
# degeneracy threshold. One constant rather than four copies: Phase 2 shipped a
# diagnostic string that had drifted apart across four copied blocks, and a
# threshold is exactly the kind of value that drifts the same way.
MIN_UNITS_PER_ARM = 10


def outcome_scale(dataset: CausalDataset) -> EffectScale:
    """Effect scale implied by the schema's outcome type.

    Derived identically in every estimator so cross-method comparison stays
    sound: silently mixing percentage-point with absolute effects is a
    realistic way to reach a wrong conclusion.
    """
    return (
        "percentage_point" if dataset.schema.outcome_type == "binary" else "absolute"
    )


# eq=False: holds ndarray fields, so a generated __eq__ would raise on the
# ambiguous truth value of an array rather than compare.
@dataclass(frozen=True, eq=False)
class PanelArrays:
    """Long-format panel data with its index structure resolved once."""

    outcome: np.ndarray
    treatment: np.ndarray
    unit_index: np.ndarray
    period_index: np.ndarray
    units: np.ndarray
    periods: np.ndarray
    cohort: np.ndarray

    @property
    def n_obs(self) -> int:
        return int(self.outcome.size)

    @property
    def n_units(self) -> int:
        return int(self.units.size)

    @property
    def n_periods(self) -> int:
        return int(self.periods.size)

    @property
    def n_treated_units(self) -> int:
        return int(np.count_nonzero(self.cohort > 0))

    @property
    def n_never_treated_units(self) -> int:
        return int(np.count_nonzero(self.cohort == 0))

    @property
    def adoption_periods(self) -> tuple[int, ...]:
        return tuple(sorted({int(c) for c in self.cohort if c > 0}))

    def period_position(self, period: int) -> int:
        """Index of a period within `periods`, or -1 if it is not observed."""
        matches = np.flatnonzero(self.periods == period)
        return int(matches[0]) if matches.size else -1


def panel_arrays(dataset: CausalDataset) -> PanelArrays:
    """Resolve a panel dataset into aligned arrays and per-unit cohorts."""
    schema = dataset.schema
    if schema.time is None:
        raise ValueError("dataset is not a panel; it declares no time column")

    frame = dataset.frame.sort_values([schema.unit, schema.time])
    units = np.asarray(np.sort(np.unique(frame[schema.unit].to_numpy())))
    periods = np.asarray(np.sort(np.unique(frame[schema.time].to_numpy())))

    if not np.issubdtype(periods.dtype, np.number) or periods.min() <= 0:
        raise ValueError(
            "panel periods must be strictly positive; 0 is reserved as the "
            f"never-treated cohort sentinel, and this panel's periods start at "
            f"{periods.min()!r}"
        )

    unit_index = np.searchsorted(units, frame[schema.unit].to_numpy())
    period_index = np.searchsorted(periods, frame[schema.time].to_numpy())

    # Cohorts come from `CausalDataset.cohort_by_unit`, not from a second
    # derivation here. One definition of "first treated period" in the codebase
    # is the point: two would eventually disagree, and the disagreement would
    # show up as a plausible-looking estimate rather than an error. The reindex
    # onto `units` is what guarantees positional alignment with `unit_index`.
    cohort = (
        dataset.cohort_by_unit().reindex(units).fillna(0).to_numpy().astype(int)
    )

    return PanelArrays(
        outcome=frame[schema.outcome].to_numpy(dtype=float),
        treatment=frame[schema.treatment].to_numpy(dtype=float),
        unit_index=np.asarray(unit_index),
        period_index=np.asarray(period_index),
        units=units,
        periods=periods,
        cohort=cohort,
    )


def bootstrap_panel_effect(
    point_fn: Callable[[np.ndarray], float],
    panel: PanelArrays,
    config: EstimatorConfig,
) -> BootstrapResult:
    """Bootstrap a panel estimator by resampling whole units.

    **Clustering is forced here regardless of `config.bootstrap_kind`.** The
    config defaults to `"iid"`, and an iid bootstrap over panel rows resamples
    unit-periods independently, destroying within-unit dependence and
    understating the standard error — measured at 42% too small under an AR(1)
    error with rho = 0.8, and half-sized at rho = 0.95. That is the same
    magnitude as Phase 2's Hajek trap. Making the clustered path the only one a
    panel estimator can reach means a regression to the default cannot happen
    by omission.

    Duplicated units are *not* relabelled. Resampling a unit twice makes its
    dummy column appear twice, which is rank-deficient — but `np.linalg.lstsq`
    returns the minimum-norm solution and the treatment coefficient is
    unaffected: measured identical to 0.0000 across 40 draws against an
    explicitly relabelled resampler. The simpler path is therefore the correct
    one, and this note exists so the question is not reopened.
    """
    seed = 0 if config.random_seed is None else config.random_seed
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    def statistic(sample: np.ndarray) -> float:
        return point_fn(sample.astype(np.int64))

    return bootstrap_interval(
        statistic,
        np.arange(panel.n_obs, dtype=float),
        reps=config.bootstrap_reps,
        rng=rng,
        level=config.ci_level,
        kind="cluster",
        clusters=panel.unit_index,
    )


def clustered_vs_iid_diagnostic(clustered_se: float, iid_se: float) -> Diagnostic:
    """Report how much clustering widened the standard error.

    Surfaced in the result rather than only in a test, so the trap is
    observable to an analyst reading one fit.
    """
    ratio = float(clustered_se / iid_se) if iid_se > 0 else float("nan")
    return Diagnostic(
        name="clustered_se_ratio",
        value=ratio,
        unit="ratio",
        interpretation=(
            f"The unit-clustered standard error is {ratio:.2f}x the unclustered "
            "one. A ratio near 1 means little within-unit dependence; a large "
            "ratio means an unclustered interval would have been badly too "
            "narrow. The clustered value is the one reported."
        ),
    )


def cluster_count_diagnostic(n_units: int) -> Diagnostic:
    return Diagnostic(
        name="cluster_count",
        value=float(n_units),
        unit="clusters",
        interpretation=(
            f"Inference clusters on {n_units} units. Cluster-robust standard "
            "errors rely on the number of clusters, not the number of "
            "observations, and are unreliable below roughly 40."
        ),
    )


def donor_matrix(panel: PanelArrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wide (unit x period) outcome arrays for the treated unit and its donors.

    Shared by every donor-pool estimator (`SyntheticControl`, `BSTS`): built
    once so two estimators cannot let the same extraction logic drift apart,
    the same reasoning this module already applies to its other helpers.

    Returns `(treated_series, donor_series, pre_mask)`. `pre_mask` is a
    boolean mask over periods, shared by both series since they are aligned
    on the same period axis.
    """
    wide = np.full((panel.n_units, panel.n_periods), np.nan)
    wide[panel.unit_index, panel.period_index] = panel.outcome
    treated_positions = np.flatnonzero(panel.cohort > 0)
    if treated_positions.size != 1:
        raise ValueError(
            f"donor_matrix requires exactly one treated unit, found "
            f"{treated_positions.size}; this should have been rejected by "
            "Requirements.unmet() before fit() was ever called"
        )
    treated_index = int(treated_positions[0])
    start = int(panel.cohort[treated_index])
    pre_mask = panel.periods < start

    donor_rows = np.delete(np.arange(panel.n_units), treated_index)
    return wide[treated_index], wide[donor_rows], pre_mask


# A window trained on one period is not a weak fit, it is an unidentified one.
# On the simplex side, one equation over 17 donor weights is matched exactly by
# a whole face of solutions and which one the solver lands on decides the
# reported gap: measured on S1, the sweep over training lengths 1..16 gives
# -1.55847 at length 1 against a range of [-0.67421, +0.55785] over every other
# length. Two is also the floor `_in_time_placebo` already implied -- it
# declines below 4 pre-periods, and its midpoint split of 4 trains on 2. Shared
# by both donor-pool estimators so the two sweeps visit the same fake dates and
# their distributions stay comparable in length.
MIN_IN_TIME_TRAIN_PERIODS = 2


def split_pre_period(pre_mask: np.ndarray, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """A fake treatment date inside the pre-period, as two masks over all periods.

    The first `n_train` pre-treatment periods become the fake pre-period and
    every later pre-treatment period becomes the fake post-period. Both masks
    are subsets of `pre_mask` over the full period axis, so neither can reach
    a period after the real treatment date -- which is the whole point of an
    in-time placebo, and the one property that makes its effects a null
    distribution rather than a second look at the real effect.

    One definition of the split for every caller: `_in_time_placebo`, the
    synthetic-control sweep and the BSTS sweep all take their windows from
    here, so no two of them can disagree about which periods a fake date
    separates.
    """
    pre_positions = np.flatnonzero(pre_mask)
    fake_pre = np.zeros_like(pre_mask, dtype=bool)
    fake_post = np.zeros_like(pre_mask, dtype=bool)
    fake_pre[pre_positions[:n_train]] = True
    fake_post[pre_positions[n_train:]] = True
    return fake_pre, fake_post


def in_time_placebo_windows(
    pre_mask: np.ndarray,
    *,
    min_train_periods: int = MIN_IN_TIME_TRAIN_PERIODS,
    max_windows: int | None = None,
) -> tuple[int, ...]:
    """Training lengths the in-time placebo should visit, longest range first.

    Every length from `min_train_periods` up to one short of the pre-period's
    end, so each window keeps at least one period to measure a gap over.
    Returns empty when the pre-period cannot hold a single window, which is a
    real answer about the design rather than an error.

    `max_windows` caps the count for an estimator that pays a full posterior
    refit per window (see `bsts.bsts_in_time_placebo_sweep`). The cap keeps
    both ends of the range and spreads the rest evenly between them, so a
    capped sweep still spans short and long training windows rather than
    sampling one end of a regime that is known to behave differently at each.
    **A cap changes the rank p-value's floor**, which is `1 / (n_windows + 1)`
    -- four windows cannot report a p-value below 0.2 however extreme the real
    effect is.
    """
    lengths = range(min_train_periods, int(np.count_nonzero(pre_mask)))
    if max_windows is None or len(lengths) <= max_windows:
        return tuple(lengths)
    if max_windows <= 0:
        return ()
    if max_windows == 1:
        return (lengths[len(lengths) // 2],)
    picks = np.linspace(0, len(lengths) - 1, max_windows)
    return tuple(sorted({lengths[int(round(index))] for index in picks}))
