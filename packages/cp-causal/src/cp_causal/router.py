"""Design selection.

The router carries no heuristics of its own. It asks each estimator whether the
dataset meets its declared requirements, which keeps the knowledge of what a
method needs next to the method itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from cp_core.dataset import CausalDataset

from cp_causal.base import Estimator
from cp_causal.registry import all_estimators


@dataclass(frozen=True)
class RoutingDecision:
    estimator_name: str
    eligible: bool
    reasons: tuple[str, ...]


def route(
    dataset: CausalDataset, candidates: dict[str, type[Estimator]] | None = None
) -> tuple[RoutingDecision, ...]:
    """Assess every candidate estimator against a dataset.

    Reports excluded methods with their reasons rather than silently dropping
    them: "synthetic control was excluded because the dataset is a
    cross-section" is more useful than a shorter list.
    """
    pool = all_estimators() if candidates is None else candidates
    decisions = []
    for name in sorted(pool):
        unmet = pool[name].requirements.unmet(dataset)  # type: ignore[attr-defined]
        decisions.append(
            RoutingDecision(estimator_name=name, eligible=not unmet, reasons=unmet)
        )
    return tuple(decisions)


def eligible_estimators(
    dataset: CausalDataset, candidates: dict[str, type[Estimator]] | None = None
) -> tuple[str, ...]:
    return tuple(d.estimator_name for d in route(dataset, candidates) if d.eligible)
