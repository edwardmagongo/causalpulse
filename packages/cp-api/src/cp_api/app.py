"""The HTTP API: upload a CSV, request an analysis, read what the worker wrote.

Nothing here fits a model: `POST /v1/analyses` enqueues the worker's job by
its dotted path, a string, so this package never imports the worker. What
the API can refuse with `cp_core` alone it refuses here (see
`cp_api.validation`); the router's refusals are the worker's.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from cp_api import validation
from cp_api.auth import CurrentUser
from cp_api.db import (
    JOB_FUNCTION,
    JOB_TIMEOUT_SECONDS,
    QUEUE_NAME,
    Analysis,
    AnalysisSummary,
    Dataset,
    EstimateRow,
    RoutingDecisionRow,
    User,
    make_engine,
    make_sessionmaker,
    utcnow,
)
from cp_api.deps import SessionDep
from cp_api.validation import RequestError, check_analysis_request, check_upload, csv_header

WORKER_LOST = "worker lost: the job ended without recording a result"
# A Redis timeout can arrive after the server accepted the job, which a worker
# may then run anyway; this message is true either way.
QUEUE_DOWN = (
    "the job queue could not be reached, so the analysis could not be confirmed as "
    "queued; submit it again"
)
_LOST = {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELED}


class AnalysisRequest(BaseModel):
    # `schema` is the wire name; a field literally named `schema` would
    # shadow a BaseModel attribute and warn, which this suite treats as an error.
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: uuid.UUID
    causal_schema: dict[str, Any] = Field(alias="schema")
    estimators: list[str] | None = None
    # `analyses.random_seed` is a 32-bit Integer column.
    random_seed: int = Field(default=0, ge=0, lt=2**31)


def _iso(moment: datetime | None) -> str | None:
    return None if moment is None else moment.isoformat()


def _job_lost(redis: Redis, analysis_id: uuid.UUID) -> bool:
    try:
        job = Job.fetch(str(analysis_id), connection=redis)
    except NoSuchJobError:
        return True
    return job.get_status() in _LOST


def create_app(
    sessions: sessionmaker[Session] | None = None, redis: Redis | None = None
) -> FastAPI:
    app = FastAPI(title="CausalPulse")
    app.state.sessions = sessions if sessions is not None else make_sessionmaker(make_engine())
    queue_redis = redis if redis is not None else Redis.from_url(os.environ["REDIS_URL"])

    @app.exception_handler(RequestError)
    async def _refused(request: Request, exc: RequestError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    async def _unavailable(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": f"service unavailable: {type(exc).__name__}"}
        )

    app.add_exception_handler(OperationalError, _unavailable)
    app.add_exception_handler(RedisError, _unavailable)

    def _analysis_or_404(session: Session, analysis_id: uuid.UUID, user: User) -> Analysis:
        # Scoped to the caller: another user's analysis is exactly as absent as
        # one that does not exist, and gets the same message.
        analysis = session.scalars(
            select(Analysis).where(Analysis.id == analysis_id, Analysis.owner_id == user.id)
        ).one_or_none()
        if analysis is None:
            raise HTTPException(404, f"no analysis {analysis_id}")
        return analysis

    def _estimate_or_404(
        session: Session, analysis_id: uuid.UUID, method: str, user: User
    ) -> EstimateRow:
        _analysis_or_404(session, analysis_id, user)
        row = session.scalars(
            select(EstimateRow).where(
                EstimateRow.analysis_id == analysis_id, EstimateRow.method == method
            )
        ).one_or_none()
        if row is None:
            raise HTTPException(404, f"analysis {analysis_id} has no estimate for {method!r}")
        return row

    def _summary_or_404(
        session: Session, analysis_id: uuid.UUID, user: User
    ) -> AnalysisSummary:
        analysis = _analysis_or_404(session, analysis_id, user)
        summary = session.get(AnalysisSummary, analysis_id)
        if summary is None:
            raise HTTPException(
                404, f"analysis {analysis_id} has no results (status: {analysis.status})"
            )
        return summary

    @app.post("/v1/datasets", status_code=201)
    def upload_dataset(
        user: CurrentUser,
        response: Response,
        file: Annotated[UploadFile, File()],
        session: SessionDep,
    ) -> dict[str, Any]:
        content = file.file.read(validation.MAX_UPLOAD_BYTES + 1)
        columns = check_upload(content)
        sha256 = hashlib.sha256(content).hexdigest()
        # Deduplicated within the caller's own datasets only: the same bytes
        # from two users are two datasets, each visible to its owner alone.
        dataset = session.scalars(
            select(Dataset).where(Dataset.owner_id == user.id, Dataset.sha256 == sha256)
        ).one_or_none()
        if dataset is None:
            dataset = Dataset(
                owner_id=user.id,
                # A bare name only: it becomes the manifest's fixture path.
                filename=validation.safe_filename(file.filename),
                sha256=sha256,
                n_bytes=len(content),
                csv=content,
            )
            session.add(dataset)
            session.commit()
        else:
            response.status_code = 200
        return {
            "id": str(dataset.id),
            "filename": dataset.filename,
            "sha256": dataset.sha256,
            "n_bytes": dataset.n_bytes,
            "columns": list(columns),
        }

    @app.post("/v1/analyses", status_code=202)
    def create_analysis(
        user: CurrentUser, body: AnalysisRequest, session: SessionDep
    ) -> dict[str, Any]:
        dataset = session.scalars(
            select(Dataset).where(Dataset.id == body.dataset_id, Dataset.owner_id == user.id)
        ).one_or_none()
        if dataset is None:
            raise HTTPException(404, f"no dataset {body.dataset_id}")
        check_analysis_request(body.causal_schema, csv_header(dataset.csv), body.estimators)
        analysis = Analysis(
            owner_id=user.id,
            dataset_id=dataset.id,
            causal_schema=body.causal_schema,
            requested_estimators=body.estimators,
            random_seed=body.random_seed,
            status="queued",
        )
        session.add(analysis)
        # Committed before enqueueing, so a fast worker never looks for a row
        # that is not there yet.
        session.commit()
        try:
            Queue(QUEUE_NAME, connection=queue_redis).enqueue(
                JOB_FUNCTION,
                str(analysis.id),
                job_id=str(analysis.id),
                job_timeout=JOB_TIMEOUT_SECONDS,
            )
        except RedisError:
            analysis.status = "failed"
            analysis.error = QUEUE_DOWN
            analysis.finished_at = utcnow()
            session.commit()
            raise
        return {"id": str(analysis.id), "status": analysis.status}

    @app.get("/v1/analyses/{analysis_id}")
    def read_analysis(
        user: CurrentUser, analysis_id: uuid.UUID, session: SessionDep
    ) -> dict[str, Any]:
        analysis = _analysis_or_404(session, analysis_id, user)
        if analysis.status in ("queued", "running") and _job_lost(queue_redis, analysis_id):
            analysis.status = "failed"
            analysis.error = WORKER_LOST
            analysis.finished_at = utcnow()
            session.commit()
        decisions = session.scalars(
            select(RoutingDecisionRow)
            .where(RoutingDecisionRow.analysis_id == analysis_id)
            .order_by(RoutingDecisionRow.estimator_name)
        ).all()
        estimates = session.scalars(
            select(EstimateRow)
            .where(EstimateRow.analysis_id == analysis_id)
            .order_by(EstimateRow.id)
        ).all()
        return {
            "id": str(analysis.id),
            "dataset_id": str(analysis.dataset_id),
            "status": analysis.status,
            "partial": analysis.partial,
            "error": analysis.error,
            "created_at": _iso(analysis.created_at),
            "started_at": _iso(analysis.started_at),
            "finished_at": _iso(analysis.finished_at),
            "routing": [
                {"estimator": d.estimator_name, "eligible": d.eligible, "reasons": list(d.reasons)}
                for d in decisions
            ],
            "estimates": [
                {"method": e.method, "band": e.band, "summary": e.summary, "error": e.error}
                for e in estimates
            ],
        }

    @app.get("/v1/analyses/{analysis_id}/estimates/{method}/report.html")
    def report_html(
        user: CurrentUser, analysis_id: uuid.UUID, method: str, session: SessionDep
    ) -> HTMLResponse:
        row = _estimate_or_404(session, analysis_id, method, user)
        if row.report_html is None:
            raise HTTPException(404, f"{method} produced no report: {row.error}")
        return HTMLResponse(row.report_html)

    @app.get("/v1/analyses/{analysis_id}/estimates/{method}/report.md")
    def report_markdown(
        user: CurrentUser, analysis_id: uuid.UUID, method: str, session: SessionDep
    ) -> PlainTextResponse:
        row = _estimate_or_404(session, analysis_id, method, user)
        if row.report_markdown is None:
            raise HTTPException(404, f"{method} produced no report: {row.error}")
        return PlainTextResponse(row.report_markdown, media_type="text/markdown")

    # Declared after the two report routes, which FastAPI therefore matches first.
    @app.get("/v1/analyses/{analysis_id}/estimates/{method}/{filename}")
    def report_asset(
        user: CurrentUser,
        analysis_id: uuid.UUID,
        method: str,
        filename: str,
        session: SessionDep,
    ) -> Response:
        row = _estimate_or_404(session, analysis_id, method, user)
        assets = row.report_assets or {}
        if filename not in assets:
            raise HTTPException(404, f"{method}'s report has no figure {filename!r}")
        return Response(assets[filename], media_type="image/svg+xml")

    @app.get("/v1/analyses/{analysis_id}/manifest")
    def read_manifest(
        user: CurrentUser, analysis_id: uuid.UUID, session: SessionDep
    ) -> JSONResponse:
        return JSONResponse(_summary_or_404(session, analysis_id, user).manifest)

    @app.get("/v1/analyses/{analysis_id}/summary")
    def read_summary(
        user: CurrentUser, analysis_id: uuid.UUID, session: SessionDep
    ) -> dict[str, Any]:
        return {"comparisons": _summary_or_404(session, analysis_id, user).comparisons}

    @app.get("/v1/healthz")
    def healthz(session: SessionDep) -> dict[str, str]:
        session.execute(text("SELECT 1"))
        queue_redis.ping()
        return {"postgres": "ok", "redis": "ok"}

    return app
