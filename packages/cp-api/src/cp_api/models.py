"""The shapes the JSON routes answer with.

They exist for the OpenAPI document, which the dashboard's TypeScript types
are generated from (`cp_api.openapi`). Each model describes a payload the
route already built as a plain dict, so declaring it changes no response.

Stored values are typed only as far as this package controls them: `status`
is written by the API and the worker, and the catalogue's `family` comes from
`cp_core`. A validity band, an agreement or an interval basis is the core's
vocabulary, read back from JSON the worker stored, and stays a `str`: a value
the core adds later should reach the client, not fail the response.
"""

from __future__ import annotations

from typing import Literal

from cp_core.catalogue import Family
from pydantic import BaseModel

Status = Literal["queued", "running", "succeeded", "failed"]


class Me(BaseModel):
    name: str


class EstimatorEntry(BaseModel):
    name: str
    family: Family


class EstimatorCatalogue(BaseModel):
    estimators: list[EstimatorEntry]


class DatasetUploaded(BaseModel):
    id: str
    filename: str
    sha256: str
    n_bytes: int
    columns: list[str]


class DatasetItem(BaseModel):
    id: str
    filename: str
    sha256: str
    n_bytes: int
    created_at: str


class DatasetDetail(DatasetItem):
    columns: list[str]


class DatasetPage(BaseModel):
    items: list[DatasetItem]
    next_cursor: str | None


class AnalysisCreated(BaseModel):
    id: str
    status: Status


class RoutingDecision(BaseModel):
    estimator: str
    eligible: bool
    reasons: list[str]


class EstimateSummary(BaseModel):
    """`cp_core.results.estimate_summary`'s row, key for key."""

    method: str
    estimand: str
    scale: str
    point: float
    se: float | None
    ci_lower: float
    ci_upper: float
    ci_level: float
    ci_basis: str
    bootstrap_ci_lower: float | None
    bootstrap_ci_upper: float | None
    p_value: float | None
    n_treated: int
    n_control: int
    runtime_seconds: float


class Estimate(BaseModel):
    method: str
    band: str | None
    summary: EstimateSummary | None
    error: str | None


class AnalysisDetail(BaseModel):
    id: str
    dataset_id: str
    status: Status
    partial: bool
    error: str | None
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    routing: list[RoutingDecision]
    estimates: list[Estimate]


class AnalysisItem(BaseModel):
    id: str
    dataset_id: str
    dataset_filename: str
    status: Status
    partial: bool
    error: str | None
    requested_estimators: list[str] | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class AnalysisPage(BaseModel):
    items: list[AnalysisItem]
    next_cursor: str | None


class ComparisonGroup(BaseModel):
    """One `(estimand, scale)` group, compared or not.

    `ComparisonGroup.to_dict` in the report package writes one of two shapes:
    a compared group carries the four comparison fields, and a group with a
    single method carries `not_compared_reason` instead. The route excludes
    unset fields, so each group answers with exactly the keys it was stored
    with.
    """

    estimand: str
    scale: str
    methods: list[str]
    agreement: str | None = None
    verdict: str | None = None
    point_range: list[float] | None = None
    dispersion: float | None = None
    not_compared_reason: str | None = None


class Summary(BaseModel):
    comparisons: list[ComparisonGroup]
