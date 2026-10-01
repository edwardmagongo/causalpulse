"""Writing a `TrackingPayload` to MLflow. The only module that imports MLflow.

Deduplicated by the payload's fingerprint tag: a finished benchmark run is
logged once unless `force` is given. Only a FINISHED run counts as logged: a
log that fails partway deletes its half-written run, and one killed outright
leaves a RUNNING run that the duplicate check ignores, so neither blocks a retry.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow.entities import Metric, Param  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from cp_tracking.payload import FINGERPRINT_TAG, TrackingPayload  # noqa: E402

EXPERIMENT = "causalpulse-benchmarks"
# MlflowClient.log_batch's per-call limits.
_PARAMS_PER_BATCH = 100
_METRICS_PER_BATCH = 1000


class AlreadyTracked(Exception):  # noqa: N818 -- a state ("already tracked"), not an error kind
    def __init__(self, run_id: str) -> None:
        super().__init__(f"already tracked as MLflow run {run_id}; pass --force to log it again")
        self.run_id = run_id


def default_tracking_uri(base: Path) -> str:
    """`sqlite:///<base>/mlruns/mlflow.db`. The directory is created here."""
    (base / "mlruns").mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{base / 'mlruns' / 'mlflow.db'}"


def artifact_root_for(tracking_uri: str) -> str | None:
    """Artifacts beside a SQLite store; the server's own default otherwise.

    Without this, MLflow puts a SQLite store's artifacts in `./mlartifacts`
    under whatever the working directory happens to be.
    """
    if not tracking_uri.startswith("sqlite:///"):
        return None
    # SQLAlchemy resolves a relative database path against the working directory.
    database = Path(tracking_uri.removeprefix("sqlite:///")).absolute()
    return (database.parent / "artifacts").as_uri()


def _experiment_id(client: MlflowClient, tracking_uri: str) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is not None:
        return str(experiment.experiment_id)
    return str(
        client.create_experiment(EXPERIMENT, artifact_location=artifact_root_for(tracking_uri))
    )


def log_payload(payload: TrackingPayload, tracking_uri: str, *, force: bool = False) -> str:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _experiment_id(client, tracking_uri)
    if not force:
        existing = client.search_runs(
            [experiment_id],
            filter_string=(
                f"tags.`{FINGERPRINT_TAG}` = '{payload.tags[FINGERPRINT_TAG]}'"
                " and attributes.status = 'FINISHED'"
            ),
            max_results=1,
        )
        if existing:
            raise AlreadyTracked(existing[0].info.run_id)
    run_id = str(
        client.create_run(experiment_id, run_name=payload.run_name, tags=payload.tags).info.run_id
    )
    try:
        params = [Param(k, v) for k, v in sorted(payload.params.items())]
        for start in range(0, len(params), _PARAMS_PER_BATCH):
            client.log_batch(run_id, params=params[start : start + _PARAMS_PER_BATCH])
        now = int(time.time() * 1000)
        metrics = [Metric(k, v, now, 0) for k, v in sorted(payload.metrics.items())]
        for start in range(0, len(metrics), _METRICS_PER_BATCH):
            client.log_batch(run_id, metrics=metrics[start : start + _METRICS_PER_BATCH])
        for name, text in sorted(payload.artifacts.items()):
            client.log_text(run_id, text, name)
        client.set_terminated(run_id)
    except BaseException:
        client.delete_run(run_id)
        raise
    return run_id
