"""Lint the text real fits actually produce, not just the literals in source.

Section 6.3's rule has to hold on the path a user sees. The source scan in
tests/core/test_language.py cannot see text assembled at run time from
fragments, which is exactly how the validity rationale and the comparison
verdict are built. This walks the objects instead.

Slow-marked: covering every registered estimator means fitting the two NUTS
estimators, and this exists to be thorough rather than fast.
"""

import dataclasses

import numpy as np
import pytest
from cp_causal.comparison import compare
from cp_causal.estimators.bsts import FAST_MCMC_CONFIG
from cp_causal.estimators.hierarchical_did import HIER_FAST_MCMC_CONFIG
from cp_causal.registry import all_estimators
from cp_causal.router import eligible_estimators
from cp_causal.validity import score_validity
from cp_core.language import causal_language_lint
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
from cp_simulation.panel_scenarios import build_panel_scenario
from cp_simulation.scenarios import build_scenario

# The MCMC estimators default to their PRODUCTION config -- 1500 warmup draws
# over 4 chains for HierarchicalDiD -- whenever `options["mcmc_config"]` is
# absent. This test checks prose, not recovery, so it must supply the fast
# configs explicitly. Omitting them would add hours for no extra coverage.
CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
MCMC_CONFIGS = {
    "bsts": FAST_MCMC_CONFIG,
    "hierarchical_did": HIER_FAST_MCMC_CONFIG,
}


def _config_for(name: str) -> EstimatorConfig:
    if name not in MCMC_CONFIGS:
        return CONFIG
    return dataclasses.replace(CONFIG, options={"mcmc_config": MCMC_CONFIGS[name]})


# Small on purpose: this test checks prose, not recovery, so every dataset is
# the smallest one that lets each estimator run at all. Both a uniform and a
# staggered panel are needed -- `did_2x2` declares `needs_uniform_adoption`, so
# a staggered panel alone would leave it out of the walk entirely.
DATASETS = {
    "cross_section": lambda: build_scenario("D").generate(
        n_units=400, rng=np.random.default_rng(0)
    ),
    "panel_uniform": lambda: build_panel_scenario("P1").generate(
        n_units=120, n_periods=8, rng=np.random.default_rng(0)
    ),
    "panel_staggered": lambda: build_panel_scenario("P3").generate(
        n_units=120, n_periods=8, rng=np.random.default_rng(0)
    ),
    # Sized from the scenario spec, like every other donor test: the donor
    # shape is 17 donors plus one treated unit over 24 periods, and inventing a
    # different one here would exercise a panel no other test uses.
    "donor": lambda: build_donor_scenario("S0").generate(
        DONOR_SCENARIOS["S0"].n_units,
        DONOR_SCENARIOS["S0"].n_periods,
        np.random.default_rng(0),
    ),
}


def _texts(estimate, score) -> list[str]:
    texts = []
    for assumption in estimate.assumptions:
        texts.append(assumption.interpretation)
        if assumption.remedy is not None:
            texts.append(assumption.remedy)
    for diagnostic in estimate.diagnostics.values():
        texts.append(diagnostic.interpretation)
    texts.extend(score.rationale)
    texts.append(score.limits)
    return texts


def _fit_everything():
    """Every registered estimator, on the first dataset shape it is eligible for."""
    fitted = {}
    for label, build in DATASETS.items():
        simulated = build()
        dataset = simulated.dataset
        for name in eligible_estimators(dataset):
            if name in fitted:
                continue
            estimator = all_estimators()[name]()
            estimate = estimator.fit(dataset, _config_for(name))
            fitted[name] = (
                label,
                estimate,
                score_validity(estimate, provenance=dataset.schema.assignment_provenance),
            )
    return fitted


@pytest.mark.slow
def test_every_registered_estimator_is_covered_by_the_language_walk():
    """The anti-drift guard, asserted separately so its failure is unambiguous.

    Without this, a future estimator's text escapes the lint entirely and the
    walk still passes -- the self-fulfilling shape that let an unregistered
    estimator ship in Phase 5.
    """
    fitted = _fit_everything()
    assert set(fitted) == set(all_estimators()), sorted(
        set(all_estimators()) - set(fitted)
    )


@pytest.mark.slow
def test_no_text_a_real_fit_produces_makes_an_unqualified_causal_claim():
    offences = []
    for name, (label, estimate, score) in sorted(_fit_everything().items()):
        for text in _texts(estimate, score):
            for violation in causal_language_lint(text):
                offences.append(f"{name} on {label}: {violation.term} in {violation.sentence!r}")
    assert offences == [], "\n".join(offences)


@pytest.mark.slow
def test_a_real_cross_method_verdict_passes_the_lint():
    simulated = build_panel_scenario("P3").generate(
        n_units=120, n_periods=8, rng=np.random.default_rng(0)
    )
    dataset = simulated.dataset
    names = [n for n in eligible_estimators(dataset) if n in {"twfe", "callaway_santanna"}]
    estimates = tuple(all_estimators()[n]().fit(dataset, _config_for(n)) for n in names)
    scores = tuple(
        score_validity(e, provenance=dataset.schema.assignment_provenance) for e in estimates
    )
    result = compare(estimates, scores)
    assert causal_language_lint(result.verdict) == (), result.verdict
