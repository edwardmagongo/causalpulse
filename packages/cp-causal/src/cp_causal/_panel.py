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
