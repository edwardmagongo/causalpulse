"""The committed study bundle and the committed report were generated together.

The bundle records the SHA-256 of the Markdown report written in the same
run. If either file is regenerated without the other, this fails. It reads
no run store: the stores are not in the repository.

It cannot show that the committed bundle is what the current code would
write. Only regenerating both from the stores shows that.
"""

import hashlib
import json
from pathlib import Path

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
