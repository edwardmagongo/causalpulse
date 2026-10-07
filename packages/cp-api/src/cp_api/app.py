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
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

from cp_core.catalogue import catalogue
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from redis import Redis
from redis.exceptions import RedisError
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus
from sqlalchemy import func, select, text, tuple_
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from starlette.datastructures import FormData
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Message

from cp_api import models, validation
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
from cp_api.limits import (
    FixedWindow,
    LimitExceededError,
    Limits,
    SigninFailures,
    limits_from_environment,
)
from cp_api.pagination import DEFAULT_LIMIT, MAX_LIMIT, decode_cursor, encode_cursor
from cp_api.settings import require_production_settings
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


# Module scope, like SessionDep: a local alias breaks under postponed annotations.
Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT)]


def _job_lost(redis: Redis, analysis_id: uuid.UUID) -> bool:
    try:
        job = Job.fetch(str(analysis_id), connection=redis)
    except NoSuchJobError:
        return True
    return job.get_status() in _LOST


# `POST /v1/datasets` reads its own form (see `uploaded_file`), so FastAPI does
# not know the route has a body. This is what it wrote in the OpenAPI document
# while the route declared `file: UploadFile`, under the name it gave it; the
# dashboard's types are generated from the document.
UPLOAD_BODY_NAME = "Body_upload_dataset_v1_datasets_post"
UPLOAD_BODY_SCHEMA: dict[str, Any] = {
    "properties": {
        "file": {
            "contentMediaType": "application/octet-stream",
            "title": "File",
            "type": "string",
        }
    },
    "required": ["file"],
    "title": UPLOAD_BODY_NAME,
    "type": "object",
}
UPLOAD_REQUEST_BODY: dict[str, Any] = {
    "content": {
        "multipart/form-data": {"schema": {"$ref": f"#/components/schemas/{UPLOAD_BODY_NAME}"}}
    },
    "required": True,
}
# What a request body may hold besides the file itself: the multipart
# boundaries and the part's headers, generously.
FORM_OVERHEAD_BYTES = 1024 * 1024
_UPLOAD_FIELD = TypeAdapter(UploadFile)


def upload_allowance(request: Request, user: CurrentUser, session: SessionDep) -> User:
    """The caller, once the key and the day's upload limit have let the upload in.

    Taken before the upload is read and kept whatever becomes of it: what the
    allowance bounds is the reading and parsing, which a refused or repeated
    upload costs as well.
    """
    state = request.app.state
    state.limit_window.require(f"uploads:{user.id}", state.uploads_per_day)
    # Checking the key left a transaction open, and with it a pooled connection.
    # The body is read next and a slow client can take minutes over it, so the
    # connection goes back first; the route's own queries take one again.
    session.commit()
    return user


UploadingUser = Annotated[User, Depends(upload_allowance)]


async def _form(request: Request) -> FormData:
    """The request's form, read from a body that is refused if it is too long.

    Starlette's parser keeps a file part of any length in a temporary file,
    so the length is counted here as the body arrives. A body over the bound
    is read to its end and thrown away, so the client gets its 413 after
    sending the whole request, as it does for a file just over the limit.
    """
    bound = validation.MAX_UPLOAD_BYTES + FORM_OVERHEAD_BYTES
    seen = 0

    async def receive() -> Message:
        nonlocal seen
        message = await request.receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > bound:
                while message["type"] == "http.request" and message.get("more_body", False):
                    message = await request.receive()
                raise validation.too_large()
        return message

    try:
        return await Request(request.scope, receive).form()
    except (StarletteHTTPException, RequestError):
        raise
    except Exception as exc:
        # What FastAPI answers when it reads a declared body and cannot.
        raise HTTPException(400, "There was an error parsing the body") from exc


def _file_field(form: FormData) -> UploadFile:
    """The form's `file`, or the 422 FastAPI gave when the route declared the field."""
    value = form.get("file")
    if value is None or value == "":
        raise RequestValidationError(
            [{"type": "missing", "loc": ("body", "file"), "msg": "Field required", "input": None}]
        )
    try:
        return _UPLOAD_FIELD.validate_python(value)
    except ValidationError as exc:
        raise RequestValidationError(
            [
                {**error, "loc": ("body", "file", *error["loc"])}
                for error in exc.errors(include_url=False)
            ]
        ) from exc


async def uploaded_file(request: Request, user: UploadingUser) -> AsyncIterator[UploadFile]:
    """The uploaded file. Nothing of the body is read until `user` is resolved.

    FastAPI reads a body the route declares before it runs the route's
    dependencies, so a route declaring `file: UploadFile` would read up to
    50 MB before looking at the key. This reads the form itself, and depends
    on the key and the upload limit so that both come first.
    """
    form = await _form(request)
    try:
        yield _file_field(form)
    finally:
        await form.close()


# Module scope, like SessionDep: a local alias breaks under postponed annotations.
UploadedFile = Annotated[UploadFile, Depends(uploaded_file)]


def create_app(
    sessions: sessionmaker[Session] | None = None,
    redis: Redis | None = None,
    limits: Limits | None = None,
) -> FastAPI:
    if sessions is None:
        # The server's own start: uvicorn calls this factory before it listens.
        # The check sits on the path that reads `DATABASE_URL`, so an app given
        # its sessions (the tests, `cp-api-openapi`) is not asked for one.
        require_production_settings()
        sessions = make_sessionmaker(make_engine())
    app = FastAPI(title="CausalPulse")
    app.state.sessions = sessions
    queue_redis = redis if redis is not None else Redis.from_url(os.environ["REDIS_URL"])
    # Every limit is off unless its setting is present (see `cp_api.limits`).
    if limits is None:
        limits = limits_from_environment()
    window = FixedWindow(queue_redis)
    if limits.signin_failures is not None:
        app.add_middleware(SigninFailures, window=window, limit=limits.signin_failures)
    # Read by `upload_allowance`, which is at module scope as `get_session` is.
    app.state.limit_window = window
    app.state.uploads_per_day = limits.uploads_per_day
    analyses_per_day = limits.analyses_per_day
    analyses_active = limits.analyses_active

    @app.exception_handler(RequestError)
    async def _refused(request: Request, exc: RequestError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(LimitExceededError)
    async def _limited(request: Request, exc: LimitExceededError) -> JSONResponse:
        return exc.response()

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

    def _fail_if_lost(session: Session, analysis: Analysis) -> None:
        """Mark an in-flight analysis `failed` when its job is gone (spec §11.4).

        Shared by the single read and the list, so a dead job never shows as
        `running` in one and `failed` in the other.
        """
        if analysis.status in ("queued", "running") and _job_lost(queue_redis, analysis.id):
            analysis.status = "failed"
            analysis.error = WORKER_LOST
            analysis.finished_at = utcnow()
            session.commit()

    @app.get("/v1/me", response_model=models.Me)
    def read_me(user: CurrentUser) -> dict[str, Any]:
        return {"name": user.name}

    @app.get("/v1/estimators", response_model=models.EstimatorCatalogue)
    def list_estimators(user: CurrentUser) -> dict[str, Any]:
        return {
            "estimators": [{"name": name, "family": family} for name, family in catalogue()]
        }

    @app.get("/v1/datasets", response_model=models.DatasetPage)
    def list_datasets(
        user: CurrentUser,
        session: SessionDep,
        limit: Limit = DEFAULT_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        # Named columns, never the entity: `datasets.csv` holds up to 50 MB a row.
        query = select(
            Dataset.id, Dataset.filename, Dataset.sha256, Dataset.n_bytes, Dataset.created_at
        ).where(Dataset.owner_id == user.id)
        if cursor is not None:
            query = query.where(tuple_(Dataset.created_at, Dataset.id) < decode_cursor(cursor))
        rows = session.execute(
            query.order_by(Dataset.created_at.desc(), Dataset.id.desc()).limit(limit + 1)
        ).all()
        page = rows[:limit]
        return {
            "items": [
                {
                    "id": str(row.id),
                    "filename": row.filename,
                    "sha256": row.sha256,
                    "n_bytes": row.n_bytes,
                    "created_at": row.created_at.isoformat(),
                }
                for row in page
            ],
            "next_cursor": (
                encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit else None
            ),
        }

    @app.get("/v1/datasets/{dataset_id}", response_model=models.DatasetDetail)
    def read_dataset(
        user: CurrentUser, dataset_id: uuid.UUID, session: SessionDep
    ) -> dict[str, Any]:
        dataset = session.scalars(
            select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == user.id)
        ).one_or_none()
        if dataset is None:
            raise HTTPException(404, f"no dataset {dataset_id}")
        return {
            "id": str(dataset.id),
            "filename": dataset.filename,
            "sha256": dataset.sha256,
            "n_bytes": dataset.n_bytes,
            "created_at": dataset.created_at.isoformat(),
            "columns": list(csv_header(dataset.csv)),
        }

    @app.post(
        "/v1/datasets",
        status_code=201,
        response_model=models.DatasetUploaded,
        openapi_extra={"requestBody": UPLOAD_REQUEST_BODY},
    )
    def upload_dataset(
        user: CurrentUser,
        response: Response,
        # Not a declared body: the key and the limits are checked before it is read.
        file: UploadedFile,
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

    @app.get("/v1/analyses", response_model=models.AnalysisPage)
    def list_analyses(
        user: CurrentUser,
        session: SessionDep,
        limit: Limit = DEFAULT_LIMIT,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        query = (
            select(Analysis, Dataset.filename)
            .join(Dataset, Dataset.id == Analysis.dataset_id)
            .where(Analysis.owner_id == user.id)
        )
        if cursor is not None:
            query = query.where(
                tuple_(Analysis.created_at, Analysis.id) < decode_cursor(cursor)
            )
        rows = session.execute(
            query.order_by(Analysis.created_at.desc(), Analysis.id.desc()).limit(limit + 1)
        ).all()
        page = rows[:limit]
        # Only the page's in-flight rows, so the Redis reads are bounded by `limit`.
        for analysis, _ in page:
            _fail_if_lost(session, analysis)
        return {
            "items": [
                {
                    "id": str(analysis.id),
                    "dataset_id": str(analysis.dataset_id),
                    "dataset_filename": filename,
                    "status": analysis.status,
                    "partial": analysis.partial,
                    "error": analysis.error,
                    "requested_estimators": analysis.requested_estimators,
                    "created_at": analysis.created_at.isoformat(),
                    "started_at": _iso(analysis.started_at),
                    "finished_at": _iso(analysis.finished_at),
                }
                for analysis, filename in page
            ],
            "next_cursor": (
                encode_cursor(page[-1][0].created_at, page[-1][0].id)
                if len(rows) > limit
                else None
            ),
        }

    @app.post("/v1/analyses", status_code=202, response_model=models.AnalysisCreated)
    def create_analysis(
        user: CurrentUser, body: AnalysisRequest, session: SessionDep
    ) -> dict[str, Any]:
        if analyses_active is not None:
            # A count of rows, not a counter, so it cannot drift: an analysis
            # stops counting when its status changes, however that happens.
            active = session.scalar(
                select(func.count())
                .select_from(Analysis)
                .where(
                    Analysis.owner_id == user.id, Analysis.status.in_(("queued", "running"))
                )
            )
            if (active or 0) >= analyses_active.count:
                raise LimitExceededError(analyses_active, analyses_active.window_seconds)
        # After the active check, so a submission refused there is not one of the
        # day's; before validation, like an upload's.
        window.require(f"analyses:{user.id}", analyses_per_day)
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

    @app.get("/v1/analyses/{analysis_id}", response_model=models.AnalysisDetail)
    def read_analysis(
        user: CurrentUser, analysis_id: uuid.UUID, session: SessionDep
    ) -> dict[str, Any]:
        analysis = _analysis_or_404(session, analysis_id, user)
        _fail_if_lost(session, analysis)
        # One column, never the entity: `datasets.csv` holds up to 50 MB a row.
        filename = session.scalars(
            select(Dataset.filename).where(
                Dataset.id == analysis.dataset_id, Dataset.owner_id == user.id
            )
        ).one()
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
            "dataset_filename": filename,
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

    @app.get(
        "/v1/analyses/{analysis_id}/summary",
        response_model=models.Summary,
        # A group answers with the keys it was stored with (see `models.ComparisonGroup`).
        response_model_exclude_unset=True,
    )
    def read_summary(
        user: CurrentUser, analysis_id: uuid.UUID, session: SessionDep
    ) -> dict[str, Any]:
        return {"comparisons": _summary_or_404(session, analysis_id, user).comparisons}

    @app.get("/v1/healthz")
    def healthz(session: SessionDep) -> dict[str, str]:
        session.execute(text("SELECT 1"))
        queue_redis.ping()
        return {"postgres": "ok", "redis": "ok"}

    generated = app.openapi

    def openapi() -> dict[str, Any]:
        # FastAPI builds the document once and returns the same one afterwards.
        document = generated()
        document["components"]["schemas"].setdefault(UPLOAD_BODY_NAME, UPLOAD_BODY_SCHEMA)
        return document

    app.openapi = openapi  # type: ignore[method-assign]
    return app
