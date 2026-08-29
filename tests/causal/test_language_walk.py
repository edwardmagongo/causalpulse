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
from cp_causal.sensitivity import sensitivity_report
from cp_causal.validity import score_validity
from cp_core.language import causal_language_lint
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
from cp_simulation.panel_scenarios import build_panel_scenario
from cp_simulation.scenarios import DEFAULT_NEGATIVE_CONTROL, build_scenario

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
                dataset,
                estimate,
                score_validity(estimate, provenance=dataset.schema.assignment_provenance),
            )
    return fitted


def _report_texts(dataset, name, estimate) -> list[str]:
    """Every optional string a SensitivityReport carries, for one real fit.

    `n_permutations` and `n_placebo_units` are kept small -- like the rest of
    this walk, the point is reaching every reason string a report can
    produce, not measuring where a p-value lands. `outcome_sd` is the
    dataset's own outcome standard deviation: several estimators here report
    on an absolute scale with no counterfactual point for `e_value` to read a
    ratio from directly (twfe, event_study, hierarchical_did, the donor-pool
    pair), and without an `outcome_sd`, `sensitivity_report` would raise
    before reaching the prose this walk exists to check.
    """
    outcome_sd = float(dataset.frame[dataset.schema.outcome].std())
    report = sensitivity_report(
        dataset,
        name,
        estimate,
        _config_for(name),
        outcome_sd=outcome_sd,
        n_permutations=20,
        n_placebo_units=3,
        # Two windows, not the uncapped sweep. On the donor-pool pair this is
        # an in-time placebo per window, and on bsts each window is a full
        # NUTS run -- uncapped that is 81.5s of sampling for a walk whose
        # whole purpose is reaching prose, not measuring where a p-value
        # lands. Same reasoning as `n_placebo_units=3` above.
        n_placebo_dates=2,
        seed=1,
    )
    falsification = report.falsification
    optional_texts = (
        report.confounding.e_value.approximation_note,
        report.confounding.cinelli_hazlett_reason,
        report.specification_curve_reason,
        report.misclassification_reason,
        falsification.placebo_dates_reason,
        falsification.placebo_units_reason,
        falsification.leave_one_donor_out_reason,
        falsification.randomization_inference_reason,
        falsification.negative_control_reason,
    )
    return [text for text in optional_texts if text is not None]


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
    for name, (label, dataset, estimate, score) in sorted(_fit_everything().items()):
        texts = _texts(estimate, score) + _report_texts(dataset, name, estimate)
        for text in texts:
            for violation in causal_language_lint(text):
                offences.append(f"{name} on {label}: {violation.term} in {violation.sentence!r}")
    assert offences == [], "\n".join(offences)


@pytest.mark.slow
def test_a_declared_negative_control_reaches_the_walk_without_a_new_violation():
    """The two Phase 10 result types carry no string fields of their own --
    only their `_reason` siblings do, and those already run through every
    fit above regardless of whether a control is declared. What this test
    adds is the one path the rest of the walk cannot reach on its own: a
    dataset that actually declares a negative control, so `negative_control`
    is populated instead of `negative_control_reason`, and the report is
    built and linted without raising.
    """
    simulated = build_scenario("D", negative_control=DEFAULT_NEGATIVE_CONTROL).generate(
        n_units=400, rng=np.random.default_rng(0)
    )
    dataset = simulated.dataset
    estimate = all_estimators()["lin_regression"]().fit(dataset, CONFIG)
    texts = _texts(estimate, score_validity(
        estimate, provenance=dataset.schema.assignment_provenance
    )) + _report_texts(dataset, "lin_regression", estimate)
    offences = [
        f"lin_regression on D+negative_control: {violation.term} in {violation.sentence!r}"
        for text in texts
        for violation in causal_language_lint(text)
    ]
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
