"""The universal result contract every estimator produces.

One result type across all estimators is what makes cross-method comparison,
the benchmark harness, and later API serialisation nearly free. The fields that
look like bookkeeping — `scale`, `ci_basis`, `estimand` — are load-bearing:
silently mixing percentage-point with log-ratio effects, or a credible interval
with a confidence interval, is a realistic way to reach a wrong conclusion.
"""

from __future__ import annotations

import math
import platform as platform_module
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import numpy as np

from cp_core.values import Diagnostic, Interval

AssumptionStatus = Literal["PASS", "WARN", "FAIL", "NOT_TESTABLE"]
AssumptionKind = Literal["identification", "estimation", "data_adequacy"]
Estimand = Literal["ATE", "ATT", "CATE"]
CIBasis = Literal["analytic", "bootstrap", "permutation", "posterior", "rubin"]
EffectScale = Literal["absolute", "percentage_point", "log_ratio"]

_VALID_STATUSES: tuple[str, ...] = ("PASS", "WARN", "FAIL", "NOT_TESTABLE")
# What a failure of this check would mean. `estimation` is reserved for checks
# that say the computation itself did not succeed -- an unconverged sampler
# produces no usable estimate, whereas a violated identifying assumption
# produces a usable number that must not be read causally. The validity rubric
# treats those two failures differently, and it can only do so if the check
# that knows the answer is the thing that declares it.
_VALID_KINDS: tuple[str, ...] = ("identification", "estimation", "data_adequacy")
_VALID_ESTIMANDS: tuple[str, ...] = ("ATE", "ATT", "CATE")
_VALID_BASES: tuple[str, ...] = ("analytic", "bootstrap", "permutation", "posterior", "rubin")
_VALID_SCALES: tuple[str, ...] = ("absolute", "percentage_point", "log_ratio")

# Severity order for rolling up assumption checks. NOT_TESTABLE is deliberately
# absent: it records that a question cannot be answered from data, which is an
# honesty marker rather than a severity level. Ranking it would either hide a
# real FAIL or make every observational analysis look broken.
_SEVERITY: dict[str, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}


def worst_status(statuses: Iterable[AssumptionStatus]) -> AssumptionStatus:
    """The most severe of several assumption statuses.

    Public so that every caller rolling several checks up into one shares
    this ordering rather than defining a copy -- the pooling in
    `cp_causal.imputation` needs exactly this question answered across
    completions, and a second severity table would be free to drift.

    `NOT_TESTABLE` carries no rank (see `_SEVERITY` above), so anything
    rankable outranks it and it is returned only when nothing rankable was
    seen. An empty collection is `PASS`: no check objected.
    """
    observed = tuple(statuses)
    rankable = [status for status in observed if status in _SEVERITY]
    if not rankable:
        return "NOT_TESTABLE" if observed else "PASS"
    return max(rankable, key=lambda status: _SEVERITY[status])


_TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn")


@dataclass(frozen=True)
class AssumptionCheck:
    """A judgement about identification, as opposed to a measurement.

    `NOT_TESTABLE` carries real weight. Unconfoundedness cannot be tested from
    observational data, and the system says exactly that instead of implying a
    balance table proved it.
    """

    name: str
    kind: AssumptionKind
    status: AssumptionStatus
    statistic: float | None
    threshold: float | None
    interpretation: str
    remedy: str | None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {list(_VALID_KINDS)}, got {self.kind!r}"
            )
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"status must be one of {list(_VALID_STATUSES)}, got {self.status!r}"
            )


@dataclass(frozen=True)
class SubgroupFinding:
    """One reported subgroup from a heterogeneous-effects estimator, after
    surviving minimum-cell-size and multiple-testing correction.

    A dedicated type rather than a loosely-shaped dict, matching
    `AssumptionCheck`'s and `CounterfactualPath`'s precedent: `covariate`
    and `condition` name what defines the subgroup, `q_value` is the
    Benjamini-Hochberg-adjusted significance (not a raw p-value -- reporting
    many candidate subgroups without this correction is what "guards
    against fabricated subgroup findings" exists to prevent).
    """

    covariate: str
    condition: str
    n_units: int
    cate_estimate: float
    ci: Interval
    q_value: float

    def __post_init__(self) -> None:
        if self.n_units < 1:
            raise ValueError(f"n_units must be positive, got {self.n_units}")
        if not 0.0 <= self.q_value <= 1.0:
            raise ValueError(f"q_value must be in [0, 1], got {self.q_value}")


@dataclass(frozen=True, eq=False)
class FitMetadata:
    """What a fit was produced by, and what it would take to reproduce it.

    `config_hash`, `random_seed`, `package_versions` and `git_commit` are the
    reproduction inputs; `timestamp`, `platform` and `runtime_seconds` are
    provenance, and vary between two runs of the same fit.

    `eq=False` is load-bearing, not decoration: `frozen=True` with the default
    `eq=True` generates a `__hash__` over every field, and `package_versions` is
    a mapping, so `hash(instance)` would raise `TypeError: unhashable type`.
    Do not drop it without making every field hashable first.
    """

    estimator: str
    package_versions: Mapping[str, str]
    git_commit: str | None
    config_hash: str
    platform: str
    runtime_seconds: float
    timestamp: datetime
    random_seed: int | None

    @classmethod
    def capture(
        cls,
        estimator: str,
        config_hash: str,
        runtime_seconds: float,
        random_seed: int | None,
    ) -> FitMetadata:
        return cls(
            estimator=estimator,
            package_versions=package_versions(),
            git_commit=git_commit(),
            config_hash=config_hash,
            platform=platform_module.platform(),
            runtime_seconds=runtime_seconds,
            timestamp=datetime.now(UTC),
            random_seed=random_seed,
        )


def package_versions() -> dict[str, str]:
    """The installed version of every package tracked for reproduction.

    Public because more than `FitMetadata.capture` needs to record what
    produced a result -- `AnalysisManifest.capture` (`cp_core.manifest`) does
    the same for an analysis, and a benchmark run's own provenance sidecar
    does it for a stored benchmark. All such callers should read the one
    implementation rather than each growing (and drifting from) their own.
    """
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - environment dependent
            continue
    return versions


def git_commit() -> str | None:
    """Best-effort commit capture; absence is recorded rather than raising."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True)
class CounterfactualPoint:
    """A scalar counterfactual mean, for cross-sectional methods.

    Deliberately a different type from a time-series counterfactual path, so a
    scalar can never be rendered as a trajectory.
    """

    observed_mean: float
    predicted_absent_treatment: float
    lower: float
    upper: float


@dataclass(frozen=True, eq=False)
class CounterfactualPath:
    """A counterfactual trajectory, for methods that estimate one per period.

    A different type from `CounterfactualPoint` rather than a widened version of
    it, so a scalar counterfactual can never be rendered as a trajectory.

    `eq=False` because every field is an ndarray: a generated `__eq__` would
    raise on the ambiguous truth value of an array rather than compare.
    """

    periods: np.ndarray
    observed: np.ndarray
    predicted_absent_treatment: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    pointwise_effect: np.ndarray
    cumulative_effect: np.ndarray

    def __post_init__(self) -> None:
        fields = {
            "observed": self.observed,
            "predicted_absent_treatment": self.predicted_absent_treatment,
            "lower": self.lower,
            "upper": self.upper,
            "pointwise_effect": self.pointwise_effect,
            "cumulative_effect": self.cumulative_effect,
        }
        expected = self.periods.shape[0]
        wrong = {
            name: array.shape[0]
            for name, array in fields.items()
            if array.shape[0] != expected
        }
        if wrong:
            raise ValueError(
                f"every array must have the same length as `periods` ({expected}); "
                f"got {wrong}"
            )


@dataclass(frozen=True, eq=False)
class EffectEstimate:
    """The single result type returned by every estimator."""

    estimand: Estimand
    method: str
    point: float
    se: float | None
    ci: Interval
    ci_basis: CIBasis
    bootstrap_ci: Interval | None
    p_value: float | None
    n_treated: int
    n_control: int
    scale: EffectScale
    diagnostics: Mapping[str, Diagnostic]
    assumptions: tuple[AssumptionCheck, ...]
    counterfactual: CounterfactualPoint | CounterfactualPath | None
    fit_metadata: FitMetadata
    subgroup_findings: tuple[SubgroupFinding, ...] = ()

    def __post_init__(self) -> None:
        if self.estimand not in _VALID_ESTIMANDS:
            raise ValueError(
                f"estimand must be one of {list(_VALID_ESTIMANDS)}, got {self.estimand!r}"
            )
        if self.ci_basis not in _VALID_BASES:
            raise ValueError(
                f"ci_basis must be one of {list(_VALID_BASES)}, got {self.ci_basis!r}"
            )
        if self.scale not in _VALID_SCALES:
            raise ValueError(
                f"scale must be one of {list(_VALID_SCALES)}, got {self.scale!r}"
            )
        if self.se is not None and self.se < 0:
            raise ValueError(f"se must be non-negative, got {self.se}")
        if self.n_treated < 1:
            raise ValueError(f"n_treated must be positive, got {self.n_treated}")
        if self.n_control < 1:
            raise ValueError(f"n_control must be positive, got {self.n_control}")
        if self.p_value is not None and not (0.0 <= self.p_value <= 1.0):
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")

    @property
    def is_significant(self) -> bool:
        """Whether the reported interval excludes zero.

        Read off the interval rather than the p-value, so that the answer stays
        consistent for estimators whose inference is permutation- or
        posterior-based and carry no p-value at all.
        """
        return not self.ci.contains(0.0)

    @property
    def worst_assumption_status(self) -> AssumptionStatus:
        return worst_status(check.status for check in self.assumptions)


def estimate_summary(estimate: EffectEstimate) -> dict[str, str | float | int | None]:
    """One estimate as a flat, JSON-safe row.

    `EffectEstimate` itself has no JSON form -- it carries arrays,
    diagnostics and counterfactual paths. This is the part a table, a CSV
    row or a stored API record needs. Absent values are `None`, not `NaN`,
    because JSON has no `NaN`; numpy scalars are cast to Python ones for
    the same reason.
    """
    bootstrap = estimate.bootstrap_ci
    return {
        "method": estimate.method,
        "estimand": estimate.estimand,
        "scale": estimate.scale,
        "point": float(estimate.point),
        "se": None if estimate.se is None else float(estimate.se),
        "ci_lower": float(estimate.ci.lower),
        "ci_upper": float(estimate.ci.upper),
        "ci_level": float(estimate.ci.level),
        "ci_basis": estimate.ci_basis,
        "bootstrap_ci_lower": None if bootstrap is None else float(bootstrap.lower),
        "bootstrap_ci_upper": None if bootstrap is None else float(bootstrap.upper),
        "bootstrap_ci_level": None if bootstrap is None else float(bootstrap.level),
        "p_value": None if estimate.p_value is None else float(estimate.p_value),
        "n_treated": int(estimate.n_treated),
        "n_control": int(estimate.n_control),
        "runtime_seconds": float(estimate.fit_metadata.runtime_seconds),
    }


ValidityBand = Literal["STRONG", "MODERATE", "WEAK", "INSUFFICIENT"]

# Ordered, not just enumerated: the rubric composes several caps and takes the
# lowest, and the comparison layer excludes anything below WEAK. Public so
# those callers share one ordering rather than each defining a copy.
BAND_ORDER: dict[str, int] = {
    "INSUFFICIENT": 0,
    "WEAK": 1,
    "MODERATE": 2,
    "STRONG": 3,
}

# Section 6.3 requires this to accompany every rendering of the score. It is a
# constant surfaced as a property rather than a caller-supplied field so it
# cannot be shortened, reworded, or dropped at a call site.
VALIDITY_SCORE_LIMITS = (
    "This band scores observable diagnostics, not truth. It is not a "
    "probability that the effect is causal, and it cannot establish an "
    "assumption the data are unable to test."
)


@dataclass(frozen=True)
class CausalValidityScore:
    """An ordinal band, the ceiling that capped it, and the table behind it."""

    band: ValidityBand
    ceiling: ValidityBand
    rationale: tuple[str, ...]
    checks: tuple[AssumptionCheck, ...]

    def __post_init__(self) -> None:
        for label, value in (("band", self.band), ("ceiling", self.ceiling)):
            if value not in BAND_ORDER:
                raise ValueError(
                    f"{label} must be one of {sorted(BAND_ORDER, key=BAND_ORDER.__getitem__)}, "
                    f"got {value!r}"
                )
        if BAND_ORDER[self.band] > BAND_ORDER[self.ceiling]:
            raise ValueError(
                f"band {self.band} exceeds ceiling {self.ceiling}; the ceiling is a cap"
            )
        if not self.rationale:
            raise ValueError("rationale must not be empty; the rubric must be inspectable")

    @property
    def limits(self) -> str:
        return VALIDITY_SCORE_LIMITS


def _require_reason(value: object | None, reason: str | None, slot: str) -> None:
    """Exclusive-or between a result slot and the reason it is absent.

    An absence with no stated reason is exactly the shape that let an
    unregistered estimator ship silently in Phase 5, and a reason sitting
    next to a real result is a stale reason a reader cannot adjudicate. Both
    directions are refused here rather than left to a docstring, because a
    docstring cannot fail a build.
    """
    if value is None and not reason:
        raise ValueError(
            f"{slot} is absent, so {slot}_reason must say why; got {reason!r}"
        )
    if value is not None and reason is not None:
        raise ValueError(
            f"{slot} is present, so {slot}_reason must be None; got {reason!r}"
        )


@dataclass(frozen=True)
class EValueResult:
    """How strong an unmeasured confounder would have to be to explain a finding.

    `approximated` is True only when the risk ratio came from the
    standardised-effect conversion rather than from a ratio the data already
    carry. It is a structural field rather than prose so a renderer cannot
    drop it: an approximated E-value read as if it were the exact ratio-scale
    quantity is the misuse this field exists to make visible.
    """

    e_value: float
    e_value_ci_bound: float
    approximated: bool
    approximation_note: str | None

    def __post_init__(self) -> None:
        # E = RR + sqrt(RR*(RR-1)) on RR >= 1 is at least 1 by construction.
        # Below it means the conversion or the formula is wrong, not that the
        # finding is unusually fragile.
        if self.e_value < 1.0:
            raise ValueError(f"e_value must be at least 1, got {self.e_value}")
        if self.e_value_ci_bound < 1.0:
            raise ValueError(
                f"e_value_ci_bound must be at least 1, got {self.e_value_ci_bound}"
            )
        if self.approximated and not self.approximation_note:
            raise ValueError(
                "approximated=True requires an approximation_note stating the "
                "conversion and what it assumes"
            )


@dataclass(frozen=True)
class CinelliHazlettResult:
    """Omitted-variable bias bounds for one OLS coefficient.

    `robustness_value` is the share of residual variation a confounder would
    have to explain in *both* treatment and outcome to bring the estimate to
    zero; `robustness_value_alpha` is the weaker bar of merely making it
    statistically indistinguishable from zero. `degrees_of_freedom` is
    carried so both are reproducible by hand from the reported t-statistic.
    """

    robustness_value: float
    robustness_value_alpha: float
    partial_r2_treatment_outcome: float
    degrees_of_freedom: int

    def __post_init__(self) -> None:
        for label, value in (
            ("robustness_value", self.robustness_value),
            ("robustness_value_alpha", self.robustness_value_alpha),
            ("partial_r2_treatment_outcome", self.partial_r2_treatment_outcome),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {value}")
        if self.degrees_of_freedom < 1:
            raise ValueError(
                f"degrees_of_freedom must be positive, got {self.degrees_of_freedom}"
            )


@dataclass(frozen=True)
class ConfoundingSensitivity:
    """The two confounding-sensitivity checks for one estimate, side by side.

    `cinelli_hazlett` is `None` on non-OLS estimators it does not apply to;
    the paired reason is what keeps that absence from reading as an
    unregistered check rather than a deliberate scope limit.
    """

    e_value: EValueResult
    cinelli_hazlett: CinelliHazlettResult | None
    cinelli_hazlett_reason: str | None

    def __post_init__(self) -> None:
        _require_reason(
            self.cinelli_hazlett, self.cinelli_hazlett_reason, "cinelli_hazlett"
        )


@dataclass(frozen=True, eq=False)
class SpecificationPoint:
    """One refit, and the knobs that produced it.

    `eq=False` because `specification` is a Mapping: `frozen=True` with the
    default `eq=True` generates a `__hash__` over every field, and hashing a
    dict raises. Same reason as `FitMetadata`.
    """

    specification: Mapping[str, object]
    point: float
    ci: Interval


@dataclass(frozen=True)
class SpecificationCurve:
    """Every refit tried under alternative reasonable specifications.

    `original_index` points back at the specification actually reported
    elsewhere, so the curve can be rendered with that point highlighted
    rather than lost among peers; `capped` records whether the search space
    was truncated before every combination was tried.
    """

    specifications: tuple[SpecificationPoint, ...]
    original_index: int
    capped: bool

    def __post_init__(self) -> None:
        if not self.specifications:
            raise ValueError("a specification curve must hold at least one refit")
        if not 0 <= self.original_index < len(self.specifications):
            raise ValueError(
                f"original_index {self.original_index} is outside the "
                f"{len(self.specifications)} specifications; the analysis as run "
                "must appear in its own curve"
            )


@dataclass(frozen=True, eq=False)
class PlaceboTestResult:
    """A real effect against a distribution of effects that should be null.

    `p_value` is not one statistic across every producer of this type, and the
    field name cannot say which one it is. Three of `FalsificationResult`'s
    four slots rank the absolute effect against the absolute placebo effects.
    The fourth, `placebo_units` on a donor pool, carries Abadie's
    post-to-pre-period RMSPE ratio rank instead -- the standard inference for
    synthetic control, which discounts a placebo unit that was never fitted
    well before treatment. Both are rank p-values on the same scale, so both
    read the same way on their own; they are not comparable to each other as
    measurements of the same quantity.

    `real_effect` and `placebo_effects` are draws from one procedure in every
    slot, including that one: on the donor pool both are convex-hull-aware
    fits, each placebo unit running its own hull check against its own donor
    pool. `cp_causal.sensitivity.falsify` documents the p-value split with
    measured numbers, and what an earlier mixed-form version of the donor-pool
    placebo effects looked like.

    `eq=False` because `placebo_effects` is an ndarray and a generated
    `__eq__` would raise on the ambiguous truth value of an array.
    """

    placebo_effects: np.ndarray
    real_effect: float
    p_value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")


@dataclass(frozen=True)
class FalsificationResult:
    """Four independent placebo-style checks, each optional with a stated reason.

    None of the four applies to every estimator or every dataset shape (a
    single-donor synthetic control has no leave-one-donor-out set to compute,
    for instance), so each carries its own reason rather than one blanket
    field that cannot say which check was skipped or why.
    """

    placebo_dates: PlaceboTestResult | None
    placebo_dates_reason: str | None
    placebo_units: PlaceboTestResult | None
    placebo_units_reason: str | None
    leave_one_donor_out: tuple[float, ...] | None
    leave_one_donor_out_reason: str | None
    randomization_inference: PlaceboTestResult | None
    randomization_inference_reason: str | None
    negative_control: NegativeControlResult | None
    negative_control_reason: str | None

    def __post_init__(self) -> None:
        _require_reason(self.placebo_dates, self.placebo_dates_reason, "placebo_dates")
        _require_reason(self.placebo_units, self.placebo_units_reason, "placebo_units")
        _require_reason(
            self.leave_one_donor_out,
            self.leave_one_donor_out_reason,
            "leave_one_donor_out",
        )
        _require_reason(
            self.randomization_inference,
            self.randomization_inference_reason,
            "randomization_inference",
        )
        _require_reason(
            self.negative_control, self.negative_control_reason, "negative_control"
        )


@dataclass(frozen=True)
class TippingPoint:
    """How much the unobserved outcomes would have to differ to overturn a finding.

    The two bounds assume nothing about why values are missing: every missing
    outcome is filled with the value most opposed to the finding, and then
    with the value most in its favour. A wide interval is a property of the
    data rather than a failure of the method -- at high missingness the
    honest answer is that the data cannot settle the question.

    They are the estimate under those two CORNER fills, which is a weaker
    claim than the extreme over every possible fill. Filling one whole arm
    with a single value is extremal only if the estimator's weight on each
    unit's outcome has a constant sign within that arm -- true for
    `difference_in_means` and `ipw`, not true for `lin_regression`, and not
    established for `aipw` or `causal_forest`. `_worst_case_bounds` carries
    the measured counterexample. The property that holds for every estimator
    the gate admits, and the one the tests assert as an identity, is that
    this pair contains the complete-data estimate.

    `tipping_fraction` is the smallest share of the missing outcomes that,
    filled adversely, removes the finding's significance. It is `None` exactly
    when no share does, which is what `conclusion_survives_worst_case`
    records; the two must agree. `conclusion_survives_worst_case` inherits the
    same caveat: it records survival of the corner fill.
    """

    adverse_bound: float
    favourable_bound: float
    tipping_fraction: float | None
    n_missing_outcomes: int
    conclusion_survives_worst_case: bool

    def __post_init__(self) -> None:
        if self.adverse_bound > self.favourable_bound:
            raise ValueError(
                f"adverse_bound {self.adverse_bound} exceeds favourable_bound "
                f"{self.favourable_bound}; they are the two ends of one interval"
            )
        if self.n_missing_outcomes < 1:
            raise ValueError(
                "n_missing_outcomes must be positive; a tipping point needs "
                f"something unobserved to vary, got {self.n_missing_outcomes}"
            )
        if (
            self.tipping_fraction is not None
            and not 0.0 <= self.tipping_fraction <= 1.0
        ):
            raise ValueError(
                f"tipping_fraction must lie in [0, 1], got {self.tipping_fraction}"
            )
        # Exactly one of the two states, so a reader cannot be told both that
        # the finding survives and that a share of the missing data undoes it.
        if self.conclusion_survives_worst_case != (self.tipping_fraction is None):
            raise ValueError(
                "tipping_fraction must be None if and only if "
                "conclusion_survives_worst_case is True; got "
                f"{self.tipping_fraction!r} against "
                f"{self.conclusion_survives_worst_case}"
            )


def _sign(value: float) -> int:
    """-1, 0, or +1. Exact zero is its own class, not folded into either side.

    A risk difference of exactly 0.0 is reachable when treated and control
    rates tie on a discrete outcome, so this is not a hypothetical edge: three
    arms all at exactly 0.0 count as agreeing in sign, while `[+1, 0, -1]`
    counts as disagreeing, because 0 is neither `> 0` nor `< 0`. That is a
    choice, made explicitly here rather than left for a reader to infer from
    `arms_agree_in_sign`'s behaviour.
    """
    return (value > 0.0) - (value < 0.0)


@dataclass(frozen=True)
class MissingDataAnalysis:
    """Three answers to one question about gaps, reported side by side.

    Complete-case analysis drops the incomplete rows, which is unbiased only
    when values are missing completely at random. Single imputation fills them
    and then reports its uncertainty as though they had been observed.
    Multiple imputation fills them many times and carries the disagreement
    between fills into its interval. None of the three is the right answer in
    general, so all three are reported and none is selected here.

    The tipping point sits inside rather than beside them because it answers
    the same question without a model for the missing values, which is what
    makes it the honest fallback when the three arms disagree.
    """

    complete_case: EffectEstimate
    single_imputation: EffectEstimate
    multiple_imputation: EffectEstimate
    tipping_point: TippingPoint | None
    tipping_point_reason: str | None
    n_missing_outcomes: int
    fraction_missing: float
    arms_agree_in_sign: bool
    arms_agree_in_significance: bool

    @property
    def arms(self) -> tuple[EffectEstimate, EffectEstimate, EffectEstimate]:
        return (self.complete_case, self.single_imputation, self.multiple_imputation)

    def __post_init__(self) -> None:
        _require_reason(self.tipping_point, self.tipping_point_reason, "tipping_point")
        if self.n_missing_outcomes < 1:
            raise ValueError(
                "n_missing_outcomes must be positive; there is nothing to "
                f"compare complete-case against otherwise, got {self.n_missing_outcomes}"
            )
        if not 0.0 < self.fraction_missing <= 1.0:
            raise ValueError(
                f"fraction_missing must lie in (0, 1], got {self.fraction_missing}"
            )
        signs = {_sign(arm.point) for arm in self.arms}
        if self.arms_agree_in_sign != (len(signs) == 1):
            raise ValueError(
                f"arms_agree_in_sign={self.arms_agree_in_sign} disagrees with "
                f"the arms' point estimates {[arm.point for arm in self.arms]}"
            )
        significance = {arm.is_significant for arm in self.arms}
        if self.arms_agree_in_significance != (len(significance) == 1):
            raise ValueError(
                f"arms_agree_in_significance={self.arms_agree_in_significance} "
                "disagrees with the arms' intervals"
            )
        # Measured, not assumed: fitting each of the five eligible
        # cross-sectional estimators (difference_in_means, lin_regression,
        # ipw, aipw, causal_forest) on a complete-case reduction and on an
        # imputed completion of the same dataset -- across several
        # scenarios, seeds, and missingness rates -- `n_treated + n_control`
        # always equalled exactly the row count of the frame the estimator
        # was fitted on. None of the five drops rows internally (ipw/aipw
        # clip propensity weights rather than dropping units; causal_forest's
        # cross-fitting still counts every row into one fold or the other).
        # That licenses exact equality below rather than a weaker bound.
        single_total = self.single_imputation.n_treated + self.single_imputation.n_control
        multiple_total = (
            self.multiple_imputation.n_treated + self.multiple_imputation.n_control
        )
        if single_total != multiple_total:
            raise ValueError(
                "single_imputation and multiple_imputation are both fitted on "
                f"every row of the same dataset and must report the same total; "
                f"got {single_total} against {multiple_total}"
            )
        complete_case_total = self.complete_case.n_treated + self.complete_case.n_control
        expected_complete_case_total = single_total - self.n_missing_outcomes
        if complete_case_total != expected_complete_case_total:
            raise ValueError(
                f"complete_case reports {complete_case_total} units, but the "
                f"imputed arms' total ({single_total}) minus n_missing_outcomes "
                f"({self.n_missing_outcomes}) is {expected_complete_case_total}"
            )
        expected_fraction = self.n_missing_outcomes / single_total
        if not math.isclose(self.fraction_missing, expected_fraction, abs_tol=1e-9):
            raise ValueError(
                f"fraction_missing={self.fraction_missing} disagrees with "
                f"n_missing_outcomes/total = {self.n_missing_outcomes}/{single_total} "
                f"= {expected_fraction}"
            )


@dataclass(frozen=True)
class SensitivityReport:
    """How robust one finding is, across three independent lines of attack."""

    confounding: ConfoundingSensitivity
    specification_curve: SpecificationCurve | None
    specification_curve_reason: str | None
    falsification: FalsificationResult
    misclassification: MisclassificationCurve | None
    misclassification_reason: str | None
    missing_data: MissingDataAnalysis | None
    missing_data_reason: str | None

    def __post_init__(self) -> None:
        _require_reason(
            self.specification_curve,
            self.specification_curve_reason,
            "specification_curve",
        )
        _require_reason(
            self.misclassification, self.misclassification_reason, "misclassification"
        )
        _require_reason(self.missing_data, self.missing_data_reason, "missing_data")


@dataclass(frozen=True)
class NegativeControlResult:
    """An estimate on an outcome treatment does not affect.

    A non-null result is evidence of residual confounding: the only paths
    connecting treatment to this column run through the confounders. A null
    result is the weaker claim that this particular control found none, which
    is not the same as there being none, and the interpretation text says so.
    """

    outcome_column: str
    estimate: float
    ci: Interval
    excludes_null: bool
    estimator: str

    def __post_init__(self) -> None:
        # Two fields that must agree. Storing both is worth it -- a renderer
        # should not have to re-derive the judgement -- but a disagreement
        # would leave a reader no way to tell which one to believe.
        if self.excludes_null == self.ci.contains(0.0):
            raise ValueError(
                f"excludes_null={self.excludes_null} disagrees with the interval "
                f"[{self.ci.lower}, {self.ci.upper}]"
            )


# The correction divides the point estimate and both interval endpoints by
# the attenuation factor, so `1 / MIN_ATTENUATION_FACTOR` is the largest
# amplification a corrected point may carry. Chosen by measurement rather
# than taste: across observed treated shares from 0.02 to 0.98 in steps of
# 0.001, this floor keeps 88.0% of the default grid's (share, rate-pair)
# cells against 90.8% with no floor at all, and over the ordinary 0.2-0.8
# band of treated shares it costs nothing at all (100% kept either way). A
# stricter 0.5 would drop the retention to 80.1%, which is a curve most
# ordinary datasets would lose a fifth of.
#
# A positivity check alone is not enough. `1.0 - 0.90` is 0.09999999999999998
# rather than 0.1, so an observed share of exactly 0.10 lands a hair off the
# implied-true-share boundary instead of on it, and the default grid produced
# an attenuation of 3.1e-16 there -- a corrected point around 1e15, which a
# `> 0` guard admits. The floor bounds that and the wider near-boundary
# blow-up (10x at a share of 0.11, 5.4x at 0.12) with one invariant.
MIN_ATTENUATION_FACTOR = 0.2


@dataclass(frozen=True)
class MisclassificationPoint:
    """The effect implied at one assumed pair of misclassification rates."""

    sensitivity: float
    specificity: float
    corrected_point: float
    corrected_ci: Interval
    attenuation_factor: float

    def __post_init__(self) -> None:
        for label, value in (
            ("sensitivity", self.sensitivity),
            ("specificity", self.specificity),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {value}")
        # observed = true * attenuation_factor. At or below zero the observed
        # label carries no information about the true one and the correction
        # does not exist; just above zero it exists but divides by almost
        # nothing, which is the same defect wearing a finite number. Both are
        # refused here so an out-of-band point cannot be constructed at all,
        # whichever caller assembled it.
        if self.attenuation_factor < MIN_ATTENUATION_FACTOR:
            raise ValueError(
                f"attenuation_factor must be at least {MIN_ATTENUATION_FACTOR}, got "
                f"{self.attenuation_factor}: below that the correction divides by a "
                "number small enough that the corrected estimate reports the assumed "
                "rates rather than the data"
            )


@dataclass(frozen=True)
class MisclassificationCurve:
    """What the effect would have been under a range of assumed error rates.

    Under non-differential misclassification the observed estimate is
    attenuated toward the null, so every corrected point lies further from it
    than the observed one. This analysis cannot explain a finding away; it can
    only imply the true effect is larger than reported.
    """

    observed_point: float
    points: tuple[MisclassificationPoint, ...]
    observed_treated_share: float

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("a misclassification curve needs at least one point")
        if not 0.0 < self.observed_treated_share < 1.0:
            raise ValueError(
                "observed_treated_share must lie strictly in (0, 1), got "
                f"{self.observed_treated_share}"
            )
