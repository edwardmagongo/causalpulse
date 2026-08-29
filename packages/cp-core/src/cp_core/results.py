"""The universal result contract every estimator produces.

One result type across all estimators is what makes cross-method comparison,
the benchmark harness, and later API serialisation nearly free. The fields that
look like bookkeeping — `scale`, `ci_basis`, `estimand` — are load-bearing:
silently mixing percentage-point with log-ratio effects, or a credible interval
with a confidence interval, is a realistic way to reach a wrong conclusion.
"""

from __future__ import annotations

import platform as platform_module
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import numpy as np

from cp_core.values import Diagnostic, Interval

AssumptionStatus = Literal["PASS", "WARN", "FAIL", "NOT_TESTABLE"]
AssumptionKind = Literal["identification", "estimation", "data_adequacy"]
Estimand = Literal["ATE", "ATT", "CATE"]
CIBasis = Literal["analytic", "bootstrap", "permutation", "posterior"]
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
_VALID_BASES: tuple[str, ...] = ("analytic", "bootstrap", "permutation", "posterior")
_VALID_SCALES: tuple[str, ...] = ("absolute", "percentage_point", "log_ratio")

# Severity order for rolling up assumption checks. NOT_TESTABLE is deliberately
# absent: it records that a question cannot be answered from data, which is an
# honesty marker rather than a severity level. Ranking it would either hide a
# real FAIL or make every observational analysis look broken.
_SEVERITY: dict[str, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}

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
            package_versions=_package_versions(),
            git_commit=_git_commit(),
            config_hash=config_hash,
            platform=platform_module.platform(),
            runtime_seconds=runtime_seconds,
            timestamp=datetime.now(UTC),
            random_seed=random_seed,
        )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - environment dependent
            continue
    return versions


def _git_commit() -> str | None:
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
        rankable = [a for a in self.assumptions if a.status in _SEVERITY]
        if not rankable:
            return "NOT_TESTABLE" if self.assumptions else "PASS"
        return max(rankable, key=lambda a: _SEVERITY[a.status]).status


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

    On that same donor-pool slot, `real_effect` and `placebo_effects` are not
    two draws from one procedure either: the real effect is the convex-hull-
    aware fit the estimator itself reports, and the placebo effects are always
    simplex-based, since they exist to build the reference distribution the
    simplex fit is judged against. Where the hull check fails the two forms
    differ materially. `cp_causal.sensitivity.falsify` documents both facts
    with measured numbers.

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


@dataclass(frozen=True)
class SensitivityReport:
    """How robust one finding is, across three independent lines of attack."""

    confounding: ConfoundingSensitivity
    specification_curve: SpecificationCurve | None
    specification_curve_reason: str | None
    falsification: FalsificationResult

    def __post_init__(self) -> None:
        _require_reason(
            self.specification_curve,
            self.specification_curve_reason,
            "specification_curve",
        )
