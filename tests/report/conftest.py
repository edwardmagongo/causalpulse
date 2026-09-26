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
