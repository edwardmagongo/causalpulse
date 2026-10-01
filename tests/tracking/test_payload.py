import json
import math
import re

from cp_tracking.payload import FINGERPRINT_TAG, TRACKED_METRICS, build_payload, fingerprint

_KEY = re.compile(r"^[^.]+\.[^.]+\.(?P<metric>[a-z_]+)$")


def test_params_carry_the_runs_identity(loaded):
    frame, manifest, sidecar, run_dir = loaded

    payload = build_payload(frame, manifest, sidecar, run_dir)

    identity = manifest["identity"]
    assert payload.run_name == run_dir.name
    assert payload.params["tier"] == identity["tier"]
    assert payload.params["estimators"] == ",".join(identity["estimators"])
    assert payload.params["root_entropy"] == str(identity["root_entropy"])
    assert payload.params["replications"] == str(manifest["replications"])
    for field, value in identity["fit_config"].items():
        assert json.loads(payload.params[f"fit.{field}"]) == value
    for arm, axes in identity["grid"].items():
        if axes is None:
            continue
        for axis, value in axes.items():
            assert json.loads(payload.params[f"grid.{arm}.{axis}"]) == value


def test_tags_carry_provenance_and_the_fingerprint(loaded):
    frame, manifest, sidecar, run_dir = loaded

    payload = build_payload(frame, manifest, sidecar, run_dir)

    provenance = manifest["provenance"]
    assert payload.tags["cp.git_commit"] == (provenance["git_commit"] or "unknown")
    for name, version in provenance["package_versions"].items():
        assert payload.tags[f"cp.package.{name}"] == version
    assert payload.tags["cp.run_dir"] == str(run_dir.resolve())
    assert payload.tags[FINGERPRINT_TAG] == fingerprint(manifest["identity"], len(frame))


def test_metric_keys_are_scenario_method_metric_and_each_has_its_mcse(loaded):
    payload = build_payload(*loaded)

    assert payload.metrics, "the tiny run should yield at least one headline metric"
    for key, value in payload.metrics.items():
        assert math.isfinite(value), key
        base = key.removesuffix("_mcse")
        match = _KEY.match(base)
        assert match and match["metric"] in TRACKED_METRICS, key
        assert base in payload.metrics and f"{base}_mcse" in payload.metrics, key


def test_cells_with_no_finite_value_are_skipped_not_logged_as_nan(loaded):
    # In the tiny grid's panel scenario P1, neither test estimator is eligible.
    payload = build_payload(*loaded)

    assert not any(key.startswith("P1.") for key in payload.metrics)
    assert any(key.startswith("A.") for key in payload.metrics)


def test_artifacts_hold_every_headline_table_and_the_identity(loaded):
    frame, manifest, sidecar, run_dir = loaded
    from cp_evaluation.study import headline_tables

    payload = build_payload(frame, manifest, sidecar, run_dir)

    assert set(payload.artifacts) == {
        *(f"{name}.csv" for name in headline_tables(frame)),
        "identity.json",
        "sidecar.json",
    }
    assert json.loads(payload.artifacts["identity.json"]) == manifest["identity"]


def test_the_fingerprint_is_stable_and_changes_with_the_rows(loaded):
    frame, manifest, _, _ = loaded
    identity = manifest["identity"]

    assert fingerprint(identity, len(frame)) == fingerprint(identity, len(frame))
    assert fingerprint(identity, len(frame)) != fingerprint(identity, len(frame) + 1)


def test_an_overlong_param_is_truncated_not_dropped(loaded):
    frame, manifest, sidecar, run_dir = loaded
    manifest = json.loads(json.dumps(manifest))
    manifest["identity"]["fit_config"]["options"] = {"blob": "x" * 7000}

    payload = build_payload(frame, manifest, sidecar, run_dir)

    value = payload.params["fit.options"]
    assert len(value) <= 6000 and value.endswith("[truncated; see identity.json]")
