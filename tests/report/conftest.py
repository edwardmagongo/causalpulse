"""Shared fixtures for `tests/report/`.

`evidence` builds one `Evidence` instance carrying one estimate, a score, two
checks, one limitation and one figure -- the smallest instance that exercises
every field `write_markdown_report` and `render_html_report` read, so
`test_html.py` and `test_markdown.py` can both depend on it without each
file constructing its own.
"""

from __future__ import annotations

import pytest
from cp_report.evidence import Evidence, ReportFigure

from report.builders import _check, _estimate, _score


def _draw(ax):
    ax.plot([0, 1], [0, 1])
    return ax


@pytest.fixture
def evidence() -> Evidence:
    # score.checks matches evidence.checks below -- see Evidence's docstring
    # for why the two fields can otherwise silently disagree.
    checks = (_check("overlap"), _check("balance", kind="estimation"))
    return Evidence(
        estimates=(_estimate(),),
        score=_score(checks=checks),
        checks=checks,
        limitations=("The donor pool is small.",),
        figures=(ReportFigure(caption="Effect estimates.", draw=_draw),),
    )


@pytest.fixture
def benchmark_evidence():
    from report.benchmark_builders import make_benchmark_evidence

    return make_benchmark_evidence()


@pytest.fixture
def d_run(tmp_path, monkeypatch):
    """A real, freshly fitted sharded run over the tiny D+P1 grid."""
    import cp_cli.commands.bench as bench_module
    from cp_cli.__main__ import main
    from evaluation._run_helpers import ESTIMATORS, FIT_CONFIG

    from report.benchmark_builders import tiny_grid_with_scenario_d

    monkeypatch.setitem(
        bench_module.TIERS, "study", (tiny_grid_with_scenario_d, ESTIMATORS, FIT_CONFIG)
    )
    run_dir = tmp_path / "run"
    assert main(["bench", "run", "--tier", "study", "--out", str(run_dir)]) == 0
    return run_dir


@pytest.fixture
def cate_run(tmp_path, monkeypatch):
    """A real, freshly fitted `cate`-tier run: two RICH and two NULL cells.

    `causal_forest` with 20 trees, so it fits in seconds and still reports a real
    per-unit CATE through the sharded executor's worker processes (a monkeypatched
    stub estimator would not reach them).
    """
    import cp_cli.commands.bench as bench_module
    from cp_cli.__main__ import main
    from cp_core.values import EstimatorConfig
    from cp_evaluation.grid import CateAxes, GridConfig

    def tiny_cate():
        return GridConfig(
            cross_section=None, panel=None, cate=CateAxes(scenarios=("RICH", "NULL")),
            replications=2,
        )

    monkeypatch.setitem(
        bench_module.TIERS,
        "cate",
        (
            tiny_cate,
            ("difference_in_means", "causal_forest"),
            EstimatorConfig(random_seed=0, bootstrap_reps=20, options={"n_trees": 20}),
        ),
    )
    run_dir = tmp_path / "cate"
    assert main(["bench", "run", "--tier", "cate", "--out", str(run_dir)]) == 0
    return run_dir


@pytest.fixture
def p7_run(tmp_path, monkeypatch):
    """A real `p7`-tier run: two P7 cells at the 600 by 20 protocol, fitted only by
    the two cheap panel estimators (hierarchical_did takes minutes a cell there)."""
    import cp_cli.commands.bench as bench_module
    from cp_cli.__main__ import main
    from cp_core.values import EstimatorConfig
    from cp_evaluation.grid import GridConfig, LongPanelAxes

    monkeypatch.setitem(
        bench_module.TIERS,
        "p7",
        (
            lambda: GridConfig(cross_section=None, panel=None,
                               long_panel=LongPanelAxes(scenarios=("P7",)), replications=2),
            ("callaway_santanna", "twfe"),
            EstimatorConfig(random_seed=0, bootstrap_reps=20),
        ),
    )
    run_dir = tmp_path / "p7"
    assert main(["bench", "run", "--tier", "p7", "--out", str(run_dir)]) == 0
    return run_dir


@pytest.fixture(scope="module")
def cross_section():
    """Scenario D at n=400: small, cross-sectional, and fits in well under a second
    for `difference_in_means` and `lin_regression`, whose assumption tables differ
    (one check against two) -- which is what lets a test tell their reports apart."""
    import numpy as np
    from cp_simulation.params import ScenarioParams
    from cp_simulation.scenarios import build_scenario

    return build_scenario("D", ScenarioParams()).generate(
        n_units=400, rng=np.random.default_rng(0)
    ).dataset
