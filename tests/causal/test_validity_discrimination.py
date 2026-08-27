"""Does the validity band actually track reliability?

A rule-based rubric is fully covered by the table tests in test_validity.py.
That the band *means* something is a different, empirical claim, and this
project's standard is that empirical claims get measured against known truth.

The claim tested is a rate, not an instance: coverage among fits banded
MODERATE or better against coverage among fits banded WEAK or worse. Every
threshold below was measured under exactly this protocol -- same seed count
(20), same scenarios, same estimators, through the shipped `score_validity`
rather than a prototype:

  band mix (20 seeds, 1000 fits, no MCMC estimators): STRONG=554,
  MODERATE=269, WEAK=177, INSUFFICIENT=0. INSUFFICIENT is structurally absent
  from this harness -- it requires an `estimation`-kind FAIL, which only the
  two excluded MCMC estimators' convergence checks can produce.

  MODERATE+: n=823, coverage=0.8238, MCSE=0.0133, mean|bias|=0.1209
  WEAK-:     n=177, coverage=0.6441, MCSE=0.0360, mean|bias|=0.4985
  difference = 0.1797; difference MCSE = sqrt(0.0133^2 + 0.0360^2) = 0.0384
  2 * difference MCSE = 0.0768; headroom = 0.1797 - 0.0768 = 0.1029 (2.3x)

A first attempt at 8 seeds (chosen for runtime before this file's threshold
was measured) gave difference=0.1120 against 2*MCSE=0.1228 -- a *failing*
margin, driven entirely by the WEAK- bucket thinning to n=71. Rather than
loosen the assertion or hand-pick a seed count that happens to pass, this
file uses 20 seeds: the same count Task 9 measured at, so no number here is
extrapolated from a different protocol than the one this test actually runs.

Slow-marked. MCMC estimators are excluded: they would multiply the runtime
and their bands are driven by convergence diagnostics rather than
identification, which is a different question.
"""

import collections

import numpy as np
import pytest
from cp_causal.registry import all_estimators
from cp_causal.router import eligible_estimators
from cp_causal.validity import score_validity
from cp_core.values import EstimatorConfig
from cp_simulation.donor_scenarios import DONOR_SCENARIOS, build_donor_scenario
from cp_simulation.panel_scenarios import build_panel_scenario
from cp_simulation.scenarios import build_scenario

CONFIG = EstimatorConfig(bootstrap_reps=25, random_seed=1)
MCMC = ("bsts", "hierarchical_did")

# Measured at this exact count by Task 9 step 5 (see module docstring). Not
# extrapolated from a different seed count -- the 8-seed attempt above is the
# reason this matters.
DISCRIMINATION_SEEDS = 20

CROSS_SECTION = ("A", "B", "C", "D", "F", "G")
PANEL = ("P1", "P2", "P3", "P4", "P5")
DONOR = ("S0", "S1", "S2")


def _rows() -> list[tuple[str, bool]]:
    rows: list[tuple[str, bool]] = []
    cases = (
        [("cross_section", s) for s in CROSS_SECTION]
        + [("panel", s) for s in PANEL]
        + [("donor", s) for s in DONOR]
    )
    for shape, scenario_id in cases:
        for seed in range(DISCRIMINATION_SEEDS):
            rng = np.random.default_rng(seed)
            if shape == "cross_section":
                simulated = build_scenario(scenario_id).generate(n_units=2000, rng=rng)
                truth = simulated.truth.true_ate
            elif shape == "panel":
                simulated = build_panel_scenario(scenario_id).generate(
                    n_units=400, n_periods=8, rng=rng
                )
                truth = simulated.truth.true_att_overall
            else:
                spec = DONOR_SCENARIOS[scenario_id]
                simulated = build_donor_scenario(scenario_id).generate(
                    spec.n_units, spec.n_periods, rng
                )
                truth = simulated.truth.true_att_overall
            dataset = simulated.dataset
            for name in eligible_estimators(dataset):
                if name in MCMC:
                    continue
                estimate = all_estimators()[name]().fit(dataset, CONFIG)
                score = score_validity(
                    estimate, provenance=dataset.schema.assignment_provenance
                )
                rows.append((score.band, estimate.ci.contains(truth)))
    return rows


def _coverage(
    rows: list[tuple[str, bool]], bands: tuple[str, ...]
) -> tuple[float, float, int]:
    covered = np.array([c for b, c in rows if b in bands], dtype=float)
    if covered.size == 0:
        return 0.0, 0.0, 0
    rate = float(covered.mean())
    return rate, float(np.sqrt(rate * (1 - rate) / covered.size)), covered.size


@pytest.fixture(scope="module")
def rows() -> list[tuple[str, bool]]:
    return _rows()


@pytest.mark.slow
def test_the_harness_produces_bands_on_both_sides_of_the_comparison(
    rows: list[tuple[str, bool]],
) -> None:
    """Checked first and separately, because it is the assumption everything
    below rests on. A harness where every fit lands in one band would make the
    discrimination test pass or fail for reasons unrelated to the rubric.

    Measured mix at 20 seeds: STRONG=554, MODERATE=269, WEAK=177,
    INSUFFICIENT=0 (1000 fits total). Floors below sit comfortably under both
    measured counts -- MODERATE+ would have to drop by ~27% and WEAK- by
    ~44% before either trips -- so ordinary run-to-run numerical drift
    (library version bumps, solver tie-breaking) cannot false-fail this, but
    the harness collapsing onto one band (the degenerate case section 6's own
    review warns about) still would.
    """
    mix = collections.Counter(band for band, _ in rows)
    assert sum(mix[b] for b in ("MODERATE", "STRONG")) >= 600
    assert sum(mix[b] for b in ("WEAK", "INSUFFICIENT")) >= 100


@pytest.mark.slow
def test_higher_bands_cover_truth_more_often(rows: list[tuple[str, bool]]) -> None:
    # Measured (20 seeds, 1000 fits, through the shipped score_validity):
    #   MODERATE+: n=823, coverage=0.8238, MCSE=0.0133
    #   WEAK-:     n=177, coverage=0.6441, MCSE=0.0360
    #   difference=0.1797, difference MCSE=sqrt(0.0133^2+0.0360^2)=0.0384
    #   asserted floor = 2*MCSE = 0.0768; headroom = 0.1029 (measured
    #   difference is 2.3x the floor).
    high, high_mcse, n_high = _coverage(rows, ("MODERATE", "STRONG"))
    low, low_mcse, n_low = _coverage(rows, ("WEAK", "INSUFFICIENT"))
    difference_mcse = float(np.sqrt(high_mcse**2 + low_mcse**2))
    assert n_high > 0 and n_low > 0
    assert high - low > 2 * difference_mcse, (
        f"coverage {high:.4f} (n={n_high}) vs {low:.4f} (n={n_low}); "
        f"difference {high - low:.4f} against 2 MCSE {2 * difference_mcse:.4f}"
    )
