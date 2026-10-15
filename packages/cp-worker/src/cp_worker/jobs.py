"""The RQ job: load a queued analysis, run it, write the outcome back.

Each status change is its own transaction. The job ends the analysis
`failed`, and itself returns normally, on each of three paths, and logs each
with its traceback:

- any exception from `execute` -- a router refusal, a dataset the core
  rejects, every estimator failing -- stored verbatim;
- an exception while storing the outcome, after rolling back whatever part
  of it was written, so no half-written result remains;
- the job's time limit (`cp_worker.worker.JobTimeLimitExceeded`), with a
  message stating the limit in seconds.

So RQ records these jobs as finished. The API's lost-worker check is left for a job that never
reached one of them: a worker killed mid-job, a job deleted from Redis, a database unreachable
even for the `failed` write, or an analysis id with no row.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, cast

from cp_api.db import (
    Analysis,
    AnalysisSummary,
    Dataset,
    EstimateRow,
    Interpretation,
    RoutingDecisionRow,
    make_engine,
    make_sessionmaker,
    utcnow,
)
from cp_api.settings import interpretation_enabled
from cp_core.results import estimate_summary
from cp_llm.extract import fact_table
from cp_llm.facts import FactTable
from cp_llm.interpret import (
    SECTION_HEADINGS,
    VERIFIED_MEANS,
    InterpretationRefusedError,
    ModelDidNotAnswerError,
    RenderedNarrative,
    interpret,
)
from cp_llm.model import Effort, ModelCallError, NarrativeModel
from cp_llm.prompt import prompt_version
from cp_llm.version import interpreter_version
from cp_report.pipeline import AnalysisOutcome
from rq import get_current_job
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cp_worker.execute import execute
from cp_worker.worker import JobTimeLimitExceeded

logger = logging.getLogger(__name__)

# Stored on an interpretation this worker will not write (`CP_INTERPRETATION` off).
WORKER_INTERPRETATIONS_OFF = (
    "interpretations are turned off on this server's worker, so this one was not written"
)


def facts_of(analysis_id: uuid.UUID, outcome: AnalysisOutcome) -> dict[str, Any] | None:
    """The outcome's fact table as JSON, or None if it cannot be built.

    The extractor refuses, for example, a name the table cannot hold. That
    costs this analysis its interpretation, never its result.
    """
    try:
        loaded = json.loads(fact_table(outcome).to_json())
        if not isinstance(loaded, dict):
            raise TypeError("a fact table must serialise to a JSON object")
    except Exception:
        logger.exception("analysis %s: its fact table could not be built", analysis_id)
        return None
    return loaded


def store_outcome(session: Session, analysis_id: uuid.UUID, outcome: AnalysisOutcome) -> None:
    for decision in outcome.decisions:
        session.add(
            RoutingDecisionRow(
                analysis_id=analysis_id,
                estimator_name=decision.estimator_name,
                eligible=decision.eligible,
                reasons=list(decision.reasons),
            )
        )
    for report in outcome.reports:
        session.add(
            EstimateRow(
                analysis_id=analysis_id,
                method=report.estimate.method,
                summary=estimate_summary(report.estimate),
                band=report.score.band,
                report_html=report.html,
                report_markdown=report.markdown,
                report_assets=dict(report.assets),
            )
        )
    for failure in outcome.failures:
        session.add(
            EstimateRow(analysis_id=analysis_id, method=failure.method, error=failure.error)
        )
    session.add(
        AnalysisSummary(
            analysis_id=analysis_id,
            manifest=json.loads(outcome.manifest.to_json()),
            comparisons=[group.to_dict() for group in outcome.comparisons],
            facts=facts_of(analysis_id, outcome),
        )
    )


def _fail(
    sessions: sessionmaker[Session], key: uuid.UUID, error: str, *, unless_finished: bool = False
) -> None:
    with sessions() as session:
        failed = session.get(Analysis, key)
        assert failed is not None
        if unless_finished and failed.status in ("succeeded", "failed"):
            return
        failed.status = "failed"
        failed.error = error
        failed.finished_at = utcnow()
        session.commit()


def _run(sessions: sessionmaker[Session], key: uuid.UUID) -> None:
    with sessions() as session:
        analysis = session.get(Analysis, key)
        if analysis is None:
            raise LookupError(f"no analysis {key}")
        dataset = session.get(Dataset, analysis.dataset_id)
        if dataset is None:
            raise LookupError(f"analysis {key} names a missing dataset")
        analysis.status = "running"
        analysis.started_at = utcnow()
        # A read racing the API's commit-then-enqueue gap can have marked this
        # analysis lost; it is running now, so that verdict no longer holds.
        analysis.error = None
        analysis.finished_at = None
        session.commit()
        content, filename = dataset.csv, dataset.filename
        schema = dict(analysis.causal_schema)
        requested = analysis.requested_estimators
        seed = analysis.random_seed

    try:
        outcome = execute(content, filename, schema, requested, seed)
    except Exception as exc:
        logger.exception("analysis %s failed", key)
        # Stored verbatim: a router refusal is the core's own message.
        _fail(sessions, key, str(exc) or type(exc).__name__)
        return

    with sessions() as session:
        try:
            store_outcome(session, key, outcome)
            done = session.get(Analysis, key)
            assert done is not None
            done.status = "succeeded"
            done.partial = outcome.partial
            done.finished_at = utcnow()
            session.commit()
        except Exception as exc:
            logger.exception("analysis %s ran, but storing its outcome failed", key)
            session.rollback()
            error = str(exc) or type(exc).__name__
        else:
            return
    _fail(sessions, key, error)


def run_analysis_job(analysis_id: str) -> None:
    key = uuid.UUID(analysis_id)
    engine = make_engine()
    sessions = make_sessionmaker(engine)
    try:
        try:
            _run(sessions, key)
        except JobTimeLimitExceeded as exc:
            logger.exception("analysis %s was stopped at its time limit", key)
            # The alarm is one-shot, so recording this cannot itself be cut off.
            # It can fire after an outcome was already committed, though, and
            # that outcome stands.
            _fail(sessions, key, str(exc), unless_finished=True)
    finally:
        engine.dispose()


def require_interpretation_settings() -> None:
    """Exit if interpretations are on and the model's key is not set.

    The message names the setting, never a value: it goes to a log.
    """
    if interpretation_enabled() and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "CP_INTERPRETATION=on, but ANTHROPIC_API_KEY is not set. The worker writes "
            "interpretations with it; set it, or turn interpretations off."
        )


def build_model(model: str, effort: str) -> NarrativeModel:
    """The one place a real model is built; a test replaces it."""
    # Imported here, so that only the interpretation job loads the SDK.
    from cp_llm.client import ClaudeModel

    return ClaudeModel(model=model, effort=cast(Effort, effort))


def facts_digest(facts: dict[str, Any]) -> str:
    """The SHA-256 of the stored fact table, in one canonical JSON form."""
    canonical = json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rendered_record(rendered: RenderedNarrative) -> dict[str, Any]:
    """The narrative as the dashboard shows it: each section under its heading, then the limits.

    Every section is kept, an empty one too, in phase 1's order.
    """
    return {
        "sections": [
            {"name": name, "heading": SECTION_HEADINGS[name], "statements": list(statements)}
            for name, statements in rendered.sections
        ],
        "limits": rendered.limits,
    }


def _resolve(session: Session, key: uuid.UUID) -> Interpretation | None:
    """The interpretation row a job argument names, locked, or None if it names none.

    The argument is the row's id. A job enqueued before interpretations had
    their own ids carried the analysis id instead; for one of those, the
    analysis's newest row stands in, but only while it is queued.
    """
    row = session.get(Interpretation, key, with_for_update=True)
    if row is not None:
        return row
    newest = session.scalars(
        select(Interpretation)
        .where(Interpretation.analysis_id == key)
        .order_by(Interpretation.created_at.desc(), Interpretation.id.desc())
        .limit(1)
        .with_for_update()
    ).first()
    if newest is None or newest.status != "queued":
        return None
    return newest


@dataclass
class _Progress:
    """What a job has established so far, for its failure path to record."""

    # Set once the job holds the row it runs; a job never writes to any other.
    row_id: uuid.UUID | None = None
    # Set before the model is called, and stored with every outcome.
    version: str | None = None


def _finish_interpretation(
    sessions: sessionmaker[Session], row_id: uuid.UUID, version: str, **fields: Any
) -> None:
    with sessions() as session:
        row = session.get(Interpretation, row_id)
        assert row is not None
        for name, value in fields.items():
            setattr(row, name, value)
        row.interpreter_version = version
        row.finished_at = utcnow()
        session.commit()


def _owns(row: Interpretation) -> bool:
    """Whether this job is the one the row is waiting on.

    A job that is not -- the row was failed (queue down, worker lost) and
    asked for again, so a newer job is queued or running -- must not make the
    model call: it is paid for, and the newer job makes it.
    """
    current = get_current_job()
    if row.status != "queued" or (current is not None and current.id != row.job_id):
        logger.warning(
            "interpretation job %s skipped: the row is %s, awaiting job %s",
            current.id if current is not None else "(direct call)", row.status, row.job_id,
        )
        return False
    return True


def _interpret(sessions: sessionmaker[Session], key: uuid.UUID, progress: _Progress) -> None:
    with sessions() as session:
        # Locked, so two jobs for one row cannot both pass the check below.
        row = _resolve(session, key)
        if row is None:
            logger.warning(
                "interpretation job for %s skipped: it names no interpretation, and no "
                "analysis whose newest interpretation is queued",
                key,
            )
            return
        if not _owns(row):
            return
        progress.row_id = row.id
        version = progress.version = interpreter_version()
        if not interpretation_enabled():
            # The API and the worker each read the setting; a worker that has
            # it off writes nothing, and makes no call, whatever the API accepted.
            row.status = "failed"
            row.error = WORKER_INTERPRETATIONS_OFF
            row.interpreter_version = version
            row.finished_at = utcnow()
            session.commit()
            return
        summary = session.get(AnalysisSummary, row.analysis_id)
        if summary is None or summary.facts is None:
            raise LookupError(f"analysis {row.analysis_id} has no fact table")
        row.status = "running"
        row.started_at = utcnow()
        row.error = None
        row.finished_at = None
        stored = dict(summary.facts)
        row.facts_sha256 = facts_digest(stored)
        session.commit()
        row_id, model_name, effort = row.id, row.model, row.effort

    facts = FactTable.from_json(json.dumps(stored))
    prompt = prompt_version()
    try:
        result = interpret(facts, build_model(model_name, effort))
    except InterpretationRefusedError as exc:
        _finish_interpretation(
            sessions, row_id, version, status="refused", prompt_version=prompt,
            attempts=[a.to_dict() for a in exc.attempts],
        )
    except ModelDidNotAnswerError as exc:
        _finish_interpretation(
            sessions, row_id, version, status="failed", prompt_version=prompt,
            attempts=[a.to_dict() for a in exc.attempts],
            error=f"the model gave no narrative (stop reason: {exc.stop_reason})",
        )
    except ModelCallError as exc:
        _finish_interpretation(sessions, row_id, version, status="failed", error=str(exc))
    else:
        _finish_interpretation(
            sessions, row_id, version, status="verified", prompt_version=result.prompt_version,
            returned_model=result.model, rendered=rendered_record(result.rendered),
            notice=VERIFIED_MEANS, attempts=[a.to_dict() for a in result.attempts],
        )


def run_interpretation_job(job_argument: str) -> None:
    """Write the interpretation row the argument names, or record why there is none.

    The argument is the row's id; an analysis id, from a job enqueued before
    rows had their own ids, is accepted for that analysis's newest row while it
    is queued. Ends normally on every path, as `run_analysis_job` does:
    anything unexpected is logged with its traceback and stored as `failed` on
    the row, once the job holds one.
    """
    key = uuid.UUID(job_argument)
    engine = make_engine()
    sessions = make_sessionmaker(engine)
    progress = _Progress()
    try:
        try:
            _interpret(sessions, key, progress)
            return
        except JobTimeLimitExceeded as exc:
            logger.exception("interpretation job for %s was stopped at its time limit", key)
            error = (
                f"the interpretation exceeded the {exc.seconds}-second job limit and was stopped"
            )
        except Exception as exc:
            logger.exception("interpretation job for %s failed", key)
            error = str(exc) or type(exc).__name__
        if progress.row_id is not None:
            # A version that could not be computed stays None, which is safe: a
            # failed row may be asked for again whatever its version.
            _fail_interpretation(sessions, progress.row_id, error, progress.version)
    finally:
        engine.dispose()


def _fail_interpretation(
    sessions: sessionmaker[Session], row_id: uuid.UUID, error: str, version: str | None
) -> None:
    """Record a failure, unless the row is gone or a result was already stored."""
    with sessions() as session:
        row = session.get(Interpretation, row_id)
        if row is None or row.status in ("verified", "refused", "failed"):
            return
        row.status = "failed"
        row.error = error
        row.interpreter_version = version
        row.finished_at = utcnow()
        session.commit()
