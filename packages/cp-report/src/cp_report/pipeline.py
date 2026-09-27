"""From a dataset to one evidence report per estimate.

Every piece already existed -- routing, fitting, validity scoring,
cross-method comparison, both renderers -- and nothing chained them: the
command line stopped at estimates, and `Evidence` was only ever built in
tests. `analyse` is that chain. The command line's `analyze --report` and
the backend worker both call it, so neither carries a copy.

Each estimator is fitted separately. `run_analysis` fits all of its
estimators in one expression, so one raising estimator would discard every
other fit. A failure here is recorded, not raised, and the manifest names
only the estimators that succeeded -- a manifest naming a failed estimator
is one `replay` could not honour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cp_causal.manifest import run_analysis
from cp_causal.router import RoutingDecision, route
from cp_core.dataset import CausalDataset
from cp_core.manifest import AnalysisManifest, DatasetReference
from cp_core.results import EffectEstimate
from cp_core.values import EstimatorConfig


@dataclass(frozen=True)
class FitFailure:
    """An estimator that raised, and its message exactly as raised."""

    method: str
    error: str


def select_estimators(
    dataset: CausalDataset, requested: Sequence[str] | None
) -> tuple[tuple[RoutingDecision, ...], tuple[str, ...]]:
    """Route the dataset, then settle which estimators to fit.

    `requested=None` means every estimator the router judges eligible. A
    request is refused whole, before anything is fitted, if it names an
    unknown or an ineligible estimator -- with the router's own reasons,
    unreworded.
    """
    decisions = route(dataset)
    by_name = {d.estimator_name: d for d in decisions}
    if requested is None:
        names = tuple(d.estimator_name for d in decisions if d.eligible)
        if not names:
            reasons = "; ".join(f"{d.estimator_name}: {', '.join(d.reasons)}" for d in decisions)
            raise ValueError(f"no registered estimator is eligible for this dataset ({reasons})")
        return decisions, names
    if not requested:
        raise ValueError(
            "a request must name at least one estimator; pass none at all to let "
            "the router choose"
        )
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"unknown estimator(s): {unknown}; registered: {sorted(by_name)}")
    names = tuple(dict.fromkeys(requested))
    ineligible = [by_name[name] for name in names if not by_name[name].eligible]
    if ineligible:
        reasons = "; ".join(f"{d.estimator_name}: {', '.join(d.reasons)}" for d in ineligible)
        raise ValueError(f"requested estimator(s) are not eligible for this dataset: {reasons}")
    return decisions, names


def fit_each(
    dataset: CausalDataset,
    reference: DatasetReference,
    names: Sequence[str],
    config: EstimatorConfig,
) -> tuple[tuple[EffectEstimate, ...], tuple[FitFailure, ...], AnalysisManifest]:
    """Fit each estimator on its own, so one failure cannot discard the rest."""
    estimates: list[EffectEstimate] = []
    succeeded: list[str] = []
    failures: list[FitFailure] = []
    for name in names:
        try:
            _, (estimate,) = run_analysis(dataset, reference, [(name, config)])
        except Exception as exc:
            failures.append(FitFailure(method=name, error=str(exc) or type(exc).__name__))
            continue
        estimates.append(estimate)
        succeeded.append(name)
    if not estimates:
        detail = "; ".join(f"{f.method}: {f.error}" for f in failures)
        raise ValueError(f"every requested estimator failed ({detail})")
    manifest = AnalysisManifest.capture(
        dataset=reference,
        schema=dataset.schema,
        estimators=[(name, config) for name in succeeded],
    )
    return tuple(estimates), tuple(failures), manifest
