"""Stored upload in, `AnalysisOutcome` out -- no database, no queue."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from typing import Any

from cp_core.dataset import CausalDataset, schema_from_dict
from cp_core.manifest import FixtureReference
from cp_core.reading import read_csv
from cp_report.pipeline import AnalysisOutcome, analyse


def execute(
    content: bytes,
    filename: str,
    schema: Mapping[str, Any],
    estimators: Sequence[str] | None,
    random_seed: int,
) -> AnalysisOutcome:
    """Analyse an uploaded CSV.

    The manifest names the upload by filename and by the hash of these exact
    bytes, recomputed here rather than trusted from the database, so
    `cp replay` against a copy of the file reproduces the numbers or refuses.
    """
    dataset = CausalDataset(
        # Replay parity depends on parsing these bytes exactly as
        # `cp_core.manifest.resolve_fixture_reference` does: both call
        # `cp_core.reading.read_csv` and pass pandas nothing of their own.
        frame=read_csv(io.BytesIO(content)),
        schema=schema_from_dict(schema),
    )
    reference = FixtureReference(path=filename, content_sha256=hashlib.sha256(content).hexdigest())
    return analyse(dataset, reference, estimators, random_seed=random_seed)
