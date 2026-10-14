"""The HTTP API: upload a CSV, request an analysis, read what the worker wrote.

Nothing here fits a model: `POST /v1/analyses` enqueues the worker's job by
its dotted path, a string, so this package never imports the worker. What
the API can refuse with `cp_core` alone it refuses here (see
`cp_api.validation`); the router's refusals are the worker's.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import uuid
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timedelta
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
    INTERPRETATION_JOB_FUNCTION,
    INTERPRETATION_PENDING,
    JOB_FUNCTION,
    JOB_TIMEOUT_SECONDS,
    QUEUE_NAME,
    Analysis,
    AnalysisSummary,
    Dataset,
    EstimateRow,
    Interpretation,
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
    limiter_redis,
    limits_from_environment,
)
from cp_api.pagination import DEFAULT_LIMIT, MAX_LIMIT, decode_cursor, encode_cursor
from cp_api.settings import interpretation_enabled, require_production_settings
from cp_api.validation import RequestError, check_analysis_request, check_upload, csv_header

WORKER_LOST = "worker lost: the job ended without recording a result"
# A Redis timeout can arrive after the server accepted the job, which a worker
# may then run anyway; this message is true either way.
QUEUE_DOWN = (
    "the job queue could not be reached, so the analysis could not be confirmed as "
    "queued; submit it again"
)
_LOST = {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELED}
# How old an analysis must be before a submission's lost-job check will judge
# it (`create_analysis`). An analysis has a row before it has a job: the row is
# committed and then the job is enqueued, and in between it looks exactly like
# one whose job is lost. `created_at` is PostgreSQL's `now()`, which is when
# the submitting request's transaction began, so the span to cover is longer
# than commit to enqueue. It runs from just after that request's key was
# checked: reading the dataset (up to 50 MB), checking the request, the day's
# count (the limiter gives up on a Redis that hangs after about a second), the
# insert, the commit and the enqueue. None of these waits on a person, and
# half a minute allows for a loaded machine and for any difference between
# the database's clock and this process's, which `utcnow` reads. It costs
# little: a job that really is lost holds its place this long at most (give or take
# the difference between the API's clock and the database's), and a
# refused client is told to wait longer than that (`ACTIVE_RETRY_SECONDS`).
SUBMISSION_GRACE = timedelta(seconds=30)

# The model an interpretation is asked of, and how hard it thinks. Phase 1's
# defaults; recorded on the row, which is what the worker reads.
INTERPRETATION_MODEL = "claude-opus-5-5"
INTERPRETATION_EFFORT = "medium"
# What `POST /v1/analyses/{id}/interpretation` answers when it cannot accept.
INTERPRETATIONS_OFF = "Interpretation is not enabled on this server."
NOT_SUCCEEDED = "Only an analysis that has succeeded can be interpreted."
NO_FACTS = (
    "This analysis has no fact table, so it cannot be interpreted: it was run before "
    "interpretations were available, or its results could not be put into one. Run it "
    "again to try."
)
BEING_WRITTEN = "An interpretation of this analysis is already being written."
ALREADY_VERIFIED = "This analysis already has an interpretation."
ALREADY_REFUSED = "This analysis already has an interpretation that could not be verified."
# Each verifier rule in plain words, in the verifier's own order, for the
# reason a refused interpretation gives. `cp_llm.verify` names the rules; the
# API may not import it, so a test checks the two lists agree.
RULE_DESCRIPTIONS: dict[str, str] = {
    "resolves": "cited a fact the analysis does not have, or wrote a stray brace",
    "bare_value": "wrote a number in its own words instead of citing it",
    "vocabulary": (
        "stated a band, a check status, an agreement, a direction, significance, a zero result "
        "or eligibility in its own words instead of citing it"
    ),
    "attribution": "cited a method's result without naming the method",
    "bare_point": "gave an estimate without its interval",
    "insufficient_caveat": "quoted an estimate rated INSUFFICIENT without saying so",
    "language": "used causal wording the system does not allow",
    "delimiters": "ran a cited value into the words or signs around it",
    "one_line": "broke a statement across lines or used a hidden character",
    "completeness": (
        "left out a result it must report: an estimate, a band, a comparison, a fit failure, "
        "or a failed or warned check"
    ),
}


def refusal_reason(rules: Iterable[str]) -> str:
    """Why a refused interpretation was refused, from the rules its last attempt broke."""
    broken = set(rules)
    known = [rule for rule in RULE_DESCRIPTIONS if rule in broken]
    unknown = sorted(broken - set(RULE_DESCRIPTIONS))
    parts = [RULE_DESCRIPTIONS[rule] for rule in known]
    parts += [f"broke the verifier's rule {rule}" for rule in unknown]
    return f"The last attempt was refused because it {'; and it '.join(parts)}."


INTERPRETATION_QUEUE_DOWN = (
    "the job queue could not be reached, so the interpretation could not be confirmed as "
    "queued; ask for it again"
)


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
    return _job_id_lost(redis, str(analysis_id))


def _job_id_lost(redis: Redis, job_id: str) -> bool:
    try:
        job = Job.fetch(job_id, connection=redis)
    except NoSuchJobError:
        return True
    return job.get_status() in _LOST


def _interpretation_out(row: Interpretation) -> dict[str, Any]:
    """The row as `models.InterpretationDetail`: each field only for its status."""
    rendered = row.rendered if row.status == "verified" else None
    attempts = row.attempts or []
    rules = {violation["rule"] for violation in (attempts[-1]["violations"] if attempts else [])}
    return {
        "status": row.status,
        "model": row.model,
        "returned_model": row.returned_model,
        "prompt_version": row.prompt_version,
        "created_at": _iso(row.created_at),
        "started_at": _iso(row.started_at),
        "finished_at": _iso(row.finished_at),
        "sections": rendered["sections"] if rendered else None,
        "limits": rendered["limits"] if rendered else None,
        "notice": row.notice if row.status == "verified" else None,
        "attempts": len(attempts) if row.status == "refused" else None,
        "reason": (
            refusal_reason(rules)
            if row.status == "refused" and rules
            else None
        ),
        "error": row.error if row.status == "failed" else None,
    }


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
    # Deliberately before the body, and so before the file is known to be valid:
    # what an upload costs is the read, and only a count taken first can refuse
    # without it. An analysis is counted when it is accepted (`create_analysis`).
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
    is read on and thrown away, so the client gets its 413 after sending the
    whole request, as it does for a file just over the limit; but only for a
    further `MAX_UPLOAD_BYTES`. Past that this stops reading, after at most one
    message more, and the 413 is answered to a client that is still sending.
    That ends the request: no task, temporary file or database connection is
    left with it. It does not
    close the connection. uvicorn goes on taking what such a client sends,
    discarding it, for as long as the client keeps the connection open.

    In size the bound is the API's own: of any body the app reads the bound
    and as much again as a file may be, and keeps none of what is over the
    bound. In time the API has no bound: a body that arrives slowly is waited
    for as long as it keeps arriving, and a connection answered early is held
    as long as its client holds it. That bound is the deployment's. In
    production only the dashboard reaches the API, behind the proxy, and it
    gives an upload 600 seconds (`UPLOAD_DEADLINE_MS` in `web/`).
    """
    bound = validation.MAX_UPLOAD_BYTES + FORM_OVERHEAD_BYTES
    drained = bound + validation.MAX_UPLOAD_BYTES
    seen = 0

    async def receive() -> Message:
        nonlocal seen
        message = await request.receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > bound:
                while (
                    message["type"] == "http.request"
                    and message.get("more_body", False)
                    and seen <= drained
                ):
                    message = await request.receive()
                    seen += len(message.get("body", b""))
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
    interpretation: bool | None = None,
) -> FastAPI:
    if sessions is None:
        # The server's own start: uvicorn calls this factory before it listens.
        # The check sits on the path that reads `DATABASE_URL`, so an app given
        # its sessions (the tests, `cp-api-openapi`) is not asked for one.
        require_production_settings()
        sessions = make_sessionmaker(make_engine())
    app = FastAPI(title="CausalPulse")
    app.state.sessions = sessions
    if redis is None:
        queue_redis = Redis.from_url(os.environ["REDIS_URL"])
        # The limiter's own client, with short timeouts the queue's is not held to.
        limit_redis = limiter_redis(os.environ["REDIS_URL"])
    else:
        # A client handed in (the tests) serves both, with whatever timeouts it has.
        queue_redis = limit_redis = redis
    # Every limit is off unless its setting is present (see `cp_api.limits`).
    if limits is None:
        limits = limits_from_environment()
    window = FixedWindow(limit_redis)
    if limits.signin_failures is not None:
        app.add_middleware(SigninFailures, window=window, limit=limits.signin_failures)
    # Read by `upload_allowance`, which is at module scope as `get_session` is.
    app.state.limit_window = window
    app.state.uploads_per_day = limits.uploads_per_day
    analyses_per_day = limits.analyses_per_day
    analyses_active = limits.analyses_active
    interpretations_per_day = limits.interpretations_per_day
    # Whether this server writes interpretations (`CP_INTERPRETATION=on`).
    interpreting = interpretation_enabled() if interpretation is None else interpretation

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
        `running` in one and `failed` in the other; and by a submission the
        active limit is about to refuse, so a dead job does not hold a place.
        """
        if analysis.status in ("queued", "running") and _job_lost(queue_redis, analysis.id):
            analysis.status = "failed"
            analysis.error = WORKER_LOST
            analysis.finished_at = utcnow()
            session.commit()

    def _fail_interpretation_if_lost(session: Session, row: Interpretation) -> None:
        """Mark a pending interpretation `failed` when its job is gone, as for an analysis."""
        if row.status in INTERPRETATION_PENDING and _job_id_lost(queue_redis, row.job_id):
            row.status = "failed"
            row.error = WORKER_LOST
            row.finished_at = utcnow()
            session.commit()

    def _lock_analysis(session: Session, analysis_id: uuid.UUID) -> None:
        """Hold the analysis row's lock until the session's next commit or rollback."""
        session.execute(select(Analysis.id).where(Analysis.id == analysis_id).with_for_update())

    def _uninterpretable(session: Session, analysis: Analysis) -> str | None:
        """Why this analysis cannot be interpreted, or None if it can (interpretations on)."""
        if analysis.status != "succeeded":
            return NOT_SUCCEEDED
        has_facts = session.scalar(
            select(AnalysisSummary.facts.is_not(None)).where(
                AnalysisSummary.analysis_id == analysis.id
            )
        )
        return None if has_facts else NO_FACTS

    active_statuses = ("queued", "running")

    def _active_count(session: Session, user: User) -> int:
        count = session.scalar(
            select(func.count())
            .select_from(Analysis)
            .where(Analysis.owner_id == user.id, Analysis.status.in_(active_statuses))
        )
        return count or 0

    def _active_rows(session: Session, user: User) -> list[Analysis]:
        return list(
            session.scalars(
                select(Analysis).where(
                    Analysis.owner_id == user.id, Analysis.status.in_(active_statuses)
                )
            )
        )

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
        # A count of rows, not a counter, so it cannot drift: an analysis stops
        # counting when its status changes, however that happens.
        if (
            analyses_active is not None
            and _active_count(session, user) >= analyses_active.count
        ):
            # A job that died with its worker still counts until something
            # reads it. A client that only submits never does, so before
            # refusing, the caller's active rows are checked as a read would
            # check them, and counted again. If Redis cannot say whether the
            # jobs are lost, the rows stand. So does a row too new to judge:
            # another submission of the caller's, made at the same time, has
            # its row before it has its job (see `SUBMISSION_GRACE`). The
            # reads have no such exception.
            with contextlib.suppress(RedisError):
                settled = utcnow() - SUBMISSION_GRACE
                for active in _active_rows(session, user):
                    if active.created_at <= settled:
                        _fail_if_lost(session, active)
            if _active_count(session, user) >= analyses_active.count:
                raise LimitExceededError(analyses_active, analyses_active.window_seconds)
        dataset = session.scalars(
            select(Dataset).where(Dataset.id == body.dataset_id, Dataset.owner_id == user.id)
        ).one_or_none()
        if dataset is None:
            raise HTTPException(404, f"no dataset {body.dataset_id}")
        check_analysis_request(body.causal_schema, csv_header(dataset.csv), body.estimators)
        # Counted here, when nothing is left that can refuse the request, so a
        # submission answered 404, 422 or 429 above is not one of the day's.
        # Refuse, not fail: an insert or an enqueue that then fails keeps its
        # place in the day's count. That is rare (and a Redis that is down
        # fails the count open as well), and the place is neither given back
        # nor needs to be.
        # Counted and judged in one step (`FixedWindow.hit`), before the row
        # exists: one refused here leaves no row and no job.
        window.require(f"analyses:{user.id}", analyses_per_day)
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
        interpretation = session.get(Interpretation, analysis_id)
        if interpretation is not None:
            _fail_interpretation_if_lost(session, interpretation)
        reason = _uninterpretable(session, analysis) if interpreting else None
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
            "interpretation": (
                None if interpretation is None else _interpretation_out(interpretation)
            ),
            "interpretable": interpreting and reason is None,
            "interpretable_reason": reason,
        }

    @app.post(
        "/v1/analyses/{analysis_id}/interpretation",
        status_code=202,
        response_model=models.InterpretationDetail,
    )
    def request_interpretation(
        user: CurrentUser, analysis_id: uuid.UUID, session: SessionDep
    ) -> dict[str, Any]:
        analysis = _analysis_or_404(session, analysis_id, user)
        # Serialises requests for one analysis, so a double request cannot write two jobs.
        _lock_analysis(session, analysis_id)
        if not interpreting:
            raise RequestError(403, INTERPRETATIONS_OFF)
        reason = _uninterpretable(session, analysis)
        if reason is not None:
            raise RequestError(409, reason)
        row = session.get(Interpretation, analysis_id)
        if row is not None and row.status in INTERPRETATION_PENDING:
            # A job that died with its worker would otherwise hold the analysis
            # for good. A row too new to judge stands: its job may not be
            # enqueued yet (see `SUBMISSION_GRACE`).
            with contextlib.suppress(RedisError):
                if row.created_at <= utcnow() - SUBMISSION_GRACE:
                    _fail_interpretation_if_lost(session, row)
                    # That commit released the lock; take it again and read
                    # what a request that slipped in meanwhile left.
                    _lock_analysis(session, analysis_id)
                    session.refresh(row)
        if row is not None and row.status in INTERPRETATION_PENDING:
            raise RequestError(409, BEING_WRITTEN)
        if row is not None and row.status == "verified":
            raise RequestError(409, ALREADY_VERIFIED)
        if row is not None and row.status == "refused":
            raise RequestError(409, ALREADY_REFUSED)
        # Counted when nothing is left that can refuse it, as an analysis is.
        window.require(f"interpretations:{user.id}", interpretations_per_day)
        # A fresh job id each time, so a job asked for again never collides
        # with the failed one still recorded in the queue.
        job_id = f"interpretation-{analysis_id}-{uuid.uuid4().hex[:8]}"
        if row is None:
            row = Interpretation(analysis_id=analysis_id)
            session.add(row)
        row.status = "queued"
        row.job_id = job_id
        row.model = INTERPRETATION_MODEL
        row.effort = INTERPRETATION_EFFORT
        for cleared in (
            "returned_model", "prompt_version", "facts_sha256", "rendered", "notice",
            "attempts", "error", "started_at", "finished_at",
        ):
            setattr(row, cleared, None)
        row.created_at = func.now()
        # Committed before enqueueing, so a fast worker never looks for a row
        # that is not there yet.
        session.commit()
        session.refresh(row)
        try:
            Queue(QUEUE_NAME, connection=queue_redis).enqueue(
                INTERPRETATION_JOB_FUNCTION,
                str(analysis_id),
                job_id=job_id,
                job_timeout=JOB_TIMEOUT_SECONDS,
            )
        except RedisError:
            row.status = "failed"
            row.error = INTERPRETATION_QUEUE_DOWN
            row.finished_at = utcnow()
            session.commit()
            raise
        return _interpretation_out(row)

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
