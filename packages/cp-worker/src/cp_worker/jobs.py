"""The RQ job: load a queued analysis, run it, write the outcome back.

Each status change is its own transaction. Any exception from `execute` --
a router refusal, a dataset the core rejects, every estimator failing -- is
stored verbatim on the analysis, which is then `failed`; the job itself
returns normally, so RQ records it as finished and the API's lost-worker
check (spec §11.4) only ever fires for a job that died without a word.
"""

from __future__ import annotations

import json
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
from sqlalchemy.orm import Session

from cp_worker.execute import execute


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


def run_analysis_job(analysis_id: str) -> None:
    key = uuid.UUID(analysis_id)
    engine = make_engine()
    sessions = make_sessionmaker(engine)
    try:
        with sessions() as session:
            analysis = session.get(Analysis, key)
            if analysis is None:
                raise LookupError(f"no analysis {analysis_id}")
            dataset = session.get(Dataset, analysis.dataset_id)
            if dataset is None:
                raise LookupError(f"analysis {analysis_id} names a missing dataset")
            analysis.status = "running"
            analysis.started_at = utcnow()
            session.commit()
            content, filename = dataset.csv, dataset.filename
            schema = dict(analysis.causal_schema)
            requested = analysis.requested_estimators
            seed = analysis.random_seed

        try:
            outcome = execute(content, filename, schema, requested, seed)
        except Exception as exc:
            with sessions() as session:
                failed = session.get(Analysis, key)
                assert failed is not None
                failed.status = "failed"
                failed.error = str(exc) or type(exc).__name__
                failed.finished_at = utcnow()
                session.commit()
            return

        with sessions() as session:
            store_outcome(session, key, outcome)
            done = session.get(Analysis, key)
            assert done is not None
            done.status = "succeeded"
            done.partial = outcome.partial
            done.finished_at = utcnow()
            session.commit()
    finally:
        engine.dispose()
