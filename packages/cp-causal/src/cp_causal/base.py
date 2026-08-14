"""The contract every estimator implements.

`Requirements` does double duty: it stops an estimator being applied to data it
cannot handle, and it is the entire mechanism behind design selection. The
router does not need heuristics because each estimator already declares what it
needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from cp_core.dataset import CausalDataset
from cp_core.results import EffectEstimate
from cp_core.values import EstimatorConfig


@dataclass(frozen=True)
class Requirements:
    """What a dataset must provide for an estimator to be applicable."""

    needs_panel: bool = False
    min_pre_periods: int = 0
    min_post_periods: int = 0
    needs_covariates: bool = False
    min_treated_units: int = 1
    min_control_units: int = 1
    needs_uniform_adoption: bool = False
    supports_outcome_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"binary", "continuous"})
    )

    def unmet(self, dataset: CausalDataset) -> tuple[str, ...]:
        """Return one human-readable reason per unmet requirement.

        Returning reasons rather than a bare boolean is what lets the router
        explain why a method was excluded, which is more useful to a user than
        a shorter list of methods.
        """
        reasons: list[str] = []
        schema = dataset.schema

        if self.needs_panel and not schema.is_panel:
            reasons.append("requires panel data; dataset is a cross-section")
        if schema.is_panel and (
            self.needs_uniform_adoption or self.min_pre_periods or self.min_post_periods
        ):
            cohorts = dataset.cohort_by_unit()
            adopting = sorted({int(c) for c in cohorts if c > 0})
            if self.needs_uniform_adoption and len(adopting) > 1:
                reasons.append(
                    "requires uniform adoption (a single treatment start); "
                    f"dataset has {len(adopting)} adoption periods {adopting}"
                )
            if (self.min_pre_periods or self.min_post_periods) and adopting:
                earliest = adopting[0]
                periods = dataset.periods
                pre = int(np.count_nonzero(periods < earliest))
                post = int(np.count_nonzero(periods >= earliest))
                if pre < self.min_pre_periods:
                    reasons.append(
                        f"requires at least {self.min_pre_periods} pre-treatment "
                        f"period(s); dataset has {pre} before the earliest "
                        f"adoption at {earliest}"
                    )
                if post < self.min_post_periods:
                    reasons.append(
                        f"requires at least {self.min_post_periods} post-treatment "
                        f"period(s); dataset has {post}"
                    )
        if self.needs_covariates and not schema.covariates:
            reasons.append("requires covariates; dataset declares none")
        if dataset.n_treated_units < self.min_treated_units:
            reasons.append(
                f"requires at least {self.min_treated_units} treated units; "
                f"dataset has {dataset.n_treated_units}"
            )
        if dataset.n_control_units < self.min_control_units:
            reasons.append(
                f"requires at least {self.min_control_units} control units; "
                f"dataset has {dataset.n_control_units}"
            )
        if schema.outcome_type not in self.supports_outcome_types:
            reasons.append(
                f"does not support outcome_type {schema.outcome_type!r}; "
                f"supports {sorted(self.supports_outcome_types)}"
            )
        return tuple(reasons)


@runtime_checkable
class Estimator(Protocol):
    """Every estimator is a class with a name, requirements, and one method."""

    name: str
    requirements: Requirements

    def fit(self, dataset: CausalDataset, config: EstimatorConfig) -> EffectEstimate: ...
