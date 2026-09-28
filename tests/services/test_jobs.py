import hashlib
import json

import pytest
from cp_api.db import Analysis, AnalysisSummary, Dataset, EstimateRow, RoutingDecisionRow
from cp_causal.registry import get_estimator
from cp_worker.jobs import run_analysis_job
from sqlalchemy import select
from worker.data import scenario_d_upload

pytestmark = pytest.mark.services


def _queue(sessions, estimators):
    content, schema = scenario_d_upload()
    with sessions() as session:
        dataset = Dataset(
            filename="d.csv", sha256=hashlib.sha256(content).hexdigest(),
            n_bytes=len(content), csv=content,
        )
        session.add(dataset)
        session.flush()
        analysis = Analysis(
            dataset_id=dataset.id, causal_schema=schema, requested_estimators=estimators,
            random_seed=0, status="queued",
        )
        session.add(analysis)
        session.commit()
        return analysis.id


def test_a_job_stores_estimates_reports_routing_and_the_manifest(sessions, worker_env):
    analysis_id = _queue(sessions, ["difference_in_means", "lin_regression"])

    run_analysis_job(str(analysis_id))

    with sessions() as session:
        analysis = session.get(Analysis, analysis_id)
        assert analysis.status == "succeeded"
        assert analysis.partial is False
        assert analysis.started_at is not None and analysis.finished_at is not None
        rows = session.scalars(
            select(EstimateRow).where(EstimateRow.analysis_id == analysis_id)
            .order_by(EstimateRow.id)
        ).all()
        assert [r.method for r in rows] == ["difference_in_means", "lin_regression"]
        for row in rows:
            assert row.band in {"STRONG", "MODERATE", "WEAK", "INSUFFICIENT"}
            assert row.report_html.startswith("<!DOCTYPE html>")
            assert row.report_assets
            assert row.summary["method"] == row.method
        decisions = session.scalars(
            select(RoutingDecisionRow).where(RoutingDecisionRow.analysis_id == analysis_id)
        ).all()
        assert any(not d.eligible for d in decisions)
        summary = session.get(AnalysisSummary, analysis_id)
        assert summary.manifest["dataset"]["content_sha256"] == session.get(
            Dataset, analysis.dataset_id
        ).sha256
        assert json.dumps(summary.comparisons)


def test_a_refused_request_fails_the_analysis_with_the_core_message_verbatim(sessions, worker_env):
    analysis_id = _queue(sessions, ["synthetic_control"])

    run_analysis_job(str(analysis_id))

    with sessions() as session:
        analysis = session.get(Analysis, analysis_id)
        assert analysis.status == "failed"
        assert "synthetic_control: requires panel data" in analysis.error


def test_an_unknown_estimator_is_the_workers_to_refuse(sessions, worker_env):
    analysis_id = _queue(sessions, ["nope"])

    run_analysis_job(str(analysis_id))

    with sessions() as session:
        assert "unknown estimator(s): ['nope']" in session.get(Analysis, analysis_id).error


def test_one_failed_fit_makes_the_analysis_partial_not_failed(sessions, worker_env, monkeypatch):
    def boom(self, dataset, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(get_estimator("lin_regression"), "fit", boom)
    analysis_id = _queue(sessions, ["difference_in_means", "lin_regression"])

    run_analysis_job(str(analysis_id))

    with sessions() as session:
        analysis = session.get(Analysis, analysis_id)
        assert (analysis.status, analysis.partial) == ("succeeded", True)
        failed = session.scalars(
            select(EstimateRow).where(
                EstimateRow.analysis_id == analysis_id, EstimateRow.method == "lin_regression"
            )
        ).one()
        assert (failed.error, failed.report_html, failed.band) == ("boom", None, None)
