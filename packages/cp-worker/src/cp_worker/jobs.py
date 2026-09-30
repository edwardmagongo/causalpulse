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

So RQ records these jobs as finished. The API's lost-worker check (spec
§11.4) is left for a job that never reached one of them: a worker killed
mid-job, a job deleted from Redis, a database unreachable even for the
`failed` write, or an analysis id with no row.
"""

from __future__ import annotations

import json
import logging
import uuid

from cp_api.db import (
    Analysis,
    AnalysisSummary,
    Dataset,
    EstimateRow,
    RoutingDecisionRow,
    make_engine,
    make_sessionmaker,
    utcnow,
)
from cp_core.results import estimate_summary
from cp_report.pipeline import AnalysisOutcome
from sqlalchemy.orm import Session, sessionmaker

from cp_worker.execute import execute
from cp_worker.worker import JobTimeLimitExceeded

logger = logging.getLogger(__name__)


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
