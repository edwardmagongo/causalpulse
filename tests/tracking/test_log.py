import pytest
from cp_tracking.payload import FINGERPRINT_TAG, TrackingPayload


def _payload(fp="f" * 64, n_metrics=3):
    return TrackingPayload(
        run_name="study-60",
        params={"tier": "study", **{f"p{i}": str(i) for i in range(150)}},  # > one batch
        tags={FINGERPRINT_TAG: fp, "cp.git_commit": "abc"},
        metrics={f"A.ipw.m{i}": float(i) for i in range(n_metrics)},
        artifacts={"accuracy__scenario.csv": "method,scenario\nipw,A\n", "identity.json": "{}"},
    )


@pytest.fixture
def store(tmp_path):
    return f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"


def _client(uri):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


def test_a_payload_becomes_one_run_with_everything_on_it(store, tmp_path):
    from cp_tracking.log import EXPERIMENT, log_payload

    run_id = log_payload(_payload(n_metrics=1500), store)  # > one metric batch

    run = _client(store).get_run(run_id)
    assert run.info.run_name == "study-60"
    assert run.info.status == "FINISHED"
    assert run.data.params["tier"] == "study" and len(run.data.params) == 151
    assert len(run.data.metrics) == 1500 and run.data.metrics["A.ipw.m7"] == 7.0
    assert run.data.tags[FINGERPRINT_TAG] == "f" * 64
    names = {a.path for a in _client(store).list_artifacts(run_id)}
    assert names == {"accuracy__scenario.csv", "identity.json"}
    experiment = _client(store).get_experiment(run.info.experiment_id)
    assert experiment.name == EXPERIMENT
    # Artifacts land next to the database, never in the working directory.
    assert experiment.artifact_location.startswith((tmp_path / "mlruns").as_uri())


def test_the_same_fingerprint_is_refused_unless_forced(store):
    from cp_tracking.log import AlreadyTracked, log_payload

    first = log_payload(_payload(), store)
    with pytest.raises(AlreadyTracked) as refused:
        log_payload(_payload(), store)
    assert refused.value.run_id == first
    assert "pass --force" in str(refused.value)

    second = log_payload(_payload(), store, force=True)
    assert second != first


def test_a_different_fingerprint_is_a_new_run(store):
    from cp_tracking.log import log_payload

    assert log_payload(_payload("a" * 64), store) != log_payload(_payload("b" * 64), store)


def test_a_failed_log_leaves_no_run_to_block_a_retry(store, monkeypatch):
    from cp_tracking import log as log_module

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(log_module.MlflowClient, "log_text", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        log_module.log_payload(_payload(), store)
    monkeypatch.undo()

    assert log_module.log_payload(_payload(), store)  # not refused as "already tracked"


def test_the_default_store_is_mlruns_under_the_base_directory(tmp_path):
    from cp_tracking.log import artifact_root_for, default_tracking_uri

    uri = default_tracking_uri(tmp_path)

    assert uri == f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
    assert artifact_root_for(uri) == (tmp_path / "mlruns" / "artifacts").as_uri()
    assert artifact_root_for("http://mlflow.example:5000") is None
