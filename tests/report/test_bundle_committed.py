"""The committed study bundle and the committed report were generated together.

The bundle records the SHA-256 of the Markdown report written in the same
run. If either file is regenerated without the other, this fails. It reads
no run store: the stores are not in the repository.

It also ties the bundle's copies of the code's own constants (the metric
definitions, the families' metrics, the levels, each estimator's family) to
the code, so a changed definition cannot leave a stale bundle behind with
every test green. It cannot show that the rest of the bundle is what the
current code would write. Only regenerating both from the stores shows that.
"""

import hashlib
import json
from pathlib import Path

from cp_core.catalogue import ESTIMATOR_FAMILIES
from cp_evaluation.study import LEVELS, METRIC_DEFINITIONS, TABLE_METRICS
from cp_report.bundle import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "web" / "src" / "data" / "benchmark-study.json"
REPORT = ROOT / "docs" / "reports" / "2026-09-26-benchmark-study" / "benchmark-report.md"


def _no_constants(token):
    raise AssertionError(f"the committed bundle holds the non-JSON token {token}")


def test_the_committed_bundle_was_written_with_the_committed_report():
    bundle = json.loads(BUNDLE.read_text(), parse_constant=_no_constants)

    assert bundle["report_sha256"] == hashlib.sha256(REPORT.read_bytes()).hexdigest(), (
        "regenerate the report and the bundle together: see the README's benchmark explorer "
        "section"
    )


def test_the_committed_bundle_is_the_schema_this_code_writes():
    assert json.loads(BUNDLE.read_text())["schema_version"] == SCHEMA_VERSION


def test_the_committed_bundle_carries_the_constants_the_code_declares_today():
    bundle = json.loads(BUNDLE.read_text())
    regenerate = (
        "the committed bundle no longer matches the code's constants: regenerate the report "
        "and the bundle together, see the README's benchmark explorer section"
    )

    assert bundle["metric_definitions"] == dict(METRIC_DEFINITIONS), regenerate
    assert bundle["family_metrics"] == {f: list(m) for f, m in TABLE_METRICS.items()}, regenerate
    assert bundle["levels"] == {level: list(columns) for level, columns in LEVELS.items()}, (
        regenerate
    )
    committed = {e["name"]: e["family"] for e in bundle["estimators"]}
    assert committed, regenerate
    assert committed == {name: ESTIMATOR_FAMILIES.get(name) for name in committed}, regenerate
