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
    return Evidence(
        estimates=(_estimate(),),
        score=_score(),
        checks=(_check("overlap"), _check("balance", kind="estimation")),
        limitations=("The donor pool is small.",),
        figures=(ReportFigure(caption="Effect estimates.", draw=_draw),),
    )
