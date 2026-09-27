import dataclasses
import json

import pandas as pd
import pytest
from cp_core.values import EstimatorConfig
from cp_evaluation.grid import (
    CateAxes,
    CateCell,
    CrossSectionAxes,
    DonorPoolAxes,
    DonorPoolCell,
    GridConfig,
    cell_key,
    expand_cells,
)
from cp_evaluation.run import fit_cell, resolve_candidates
from cp_evaluation.store import (
    CELLS_DIR,
    LEDGER_NAME,
    MANIFEST_NAME,
    CorruptLedgerError,
    IncompleteRunError,
    LedgerEntry,
    RunIdentity,
    RunMismatchError,
    RunStore,
    grid_config_of,
    shard_path,
    write_shard,
)
from cp_evaluation.tiers import study_config

from evaluation._run_helpers import (
    ESTIMATORS,
    FIT_CONFIG,
    identity_for,
    mixed_config,
    new_store,
    reference_frame,
    tiny_config,
    without_runtime,
)


def _fill(store: RunStore) -> None:
    """Fit and record every expected cell, serially, without the executor."""
    candidates = resolve_candidates(ESTIMATORS)
    for key, cell in store.expected_cells().items():
        rows = fit_cell(
            cell,
            root_entropy=store.identity.root_entropy,
            fit_config=FIT_CONFIG,
            candidates=candidates,
        )
        store.record(write_shard(store.directory, key, rows))


# -- identity ---------------------------------------------------------------


def test_identities_that_differ_name_the_field():
    base = identity_for(tiny_config())

    other_entropy = identity_for(tiny_config(), root_entropy=7)
    other_fit = RunIdentity.of(
        "tiny", tiny_config(), ESTIMATORS, EstimatorConfig(random_seed=0, bootstrap_reps=99), 0
    )
    other_estimators = RunIdentity.of("tiny", tiny_config(), ("ipw",), FIT_CONFIG, 0)
    wider = dataclasses.replace(
        tiny_config(),
        cross_section=dataclasses.replace(tiny_config().cross_section, n_units=(200, 400)),
    )

    assert base.differences(base) == []
    assert base.differences(other_entropy) == ["root_entropy"]
    assert base.differences(other_fit) == ["fit_config.bootstrap_reps"]
    assert base.differences(other_estimators) == ["estimators"]
    assert base.differences(identity_for(wider)) == ["grid.cross_section.n_units"]


def test_the_replication_count_is_not_part_of_the_identity():
    assert identity_for(tiny_config(4)) == identity_for(tiny_config(400))


def test_estimator_order_is_not_part_of_the_identity():
    forward = RunIdentity.of("t", tiny_config(), ("ipw", "aipw"), FIT_CONFIG, 0)
    backward = RunIdentity.of("t", tiny_config(), ("aipw", "ipw"), FIT_CONFIG, 0)

    assert forward == backward


# -- creating, opening, extending -------------------------------------------


def test_a_created_run_reopens_with_the_same_identity_and_replications(tmp_path):
    created = new_store(tmp_path / "run", tiny_config(4))

    reopened = RunStore.open(tmp_path / "run")

    assert reopened.identity == created.identity
    assert reopened.replications == 4
    assert reopened.provenance["git_commit"] == created.provenance["git_commit"]


def test_the_manifest_rebuilds_the_grid_it_was_made_from(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(3))

    rebuilt = RunStore.open(tmp_path / "run").grid_config()

    assert rebuilt == tiny_config(3)
    assert [cell_key(c) for c in expand_cells(rebuilt)] == [
        cell_key(c) for c in expand_cells(store.grid_config())
    ]


def test_the_store_expects_every_cell_the_config_expands_to_donor_and_cate_arms_included(
    tmp_path,
):
    config = study_config()
    store = RunStore.create(
        tmp_path / "run",
        RunIdentity.of("study", config, ESTIMATORS, FIT_CONFIG, 0),
        config.replications,
    )

    expected = store.expected_cells()

    assert set(expected) == {cell_key(c) for c in expand_cells(config)}
    assert len(expected) == len(expand_cells(config)) == 4450
    assert any(isinstance(c, DonorPoolCell) for c in expected.values())
    assert any(isinstance(c, CateCell) for c in expected.values())


def test_a_grid_with_no_donor_arm_keeps_its_identity_free_of_a_donor_key():
    identity = identity_for(tiny_config(3))

    assert "donor_pool" not in identity.grid


def test_the_donor_arm_survives_the_identity_round_trip():
    config = dataclasses.replace(
        tiny_config(3), donor_pool=DonorPoolAxes(scenarios=("S0", "S3"))
    )
    identity = identity_for(config)

    assert identity.grid["donor_pool"] == {"scenarios": ["S0", "S3"]}
    assert grid_config_of(identity.grid, 3) == config


def _cross_only() -> GridConfig:
    return GridConfig(
        cross_section=CrossSectionAxes(
            scenarios=("A",), n_units=(200,), confounding_strength=(1.0,)
        ),
        panel=None,
        replications=2,
    )


def test_a_grid_without_a_cate_arm_keeps_the_identity_it_always_had():
    identity = RunIdentity.of("t", _cross_only(), ["ipw"], EstimatorConfig(), 0)

    assert set(identity.grid) == {"cross_section", "panel"}


def test_the_cate_arm_round_trips_through_the_identity():
    config = dataclasses.replace(_cross_only(), cate=CateAxes(scenarios=("RICH", "NULL")))
    identity = RunIdentity.of("t", config, ["causal_forest"], EstimatorConfig(), 0)

    assert identity.grid["cate"] == {"scenarios": ["RICH", "NULL"]}
    assert grid_config_of(identity.grid, replications=2) == config


def test_create_refuses_a_directory_that_already_holds_a_run(tmp_path):
    new_store(tmp_path / "run", tiny_config())

    with pytest.raises(FileExistsError, match="already holds a run"):
        new_store(tmp_path / "run", tiny_config())


def test_create_refuses_a_non_empty_directory_with_no_run(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "notes.txt").write_text("not a run")

    with pytest.raises(FileExistsError, match="not empty"):
        new_store(tmp_path / "run", tiny_config())


def test_open_refuses_a_directory_with_no_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match=MANIFEST_NAME):
        RunStore.open(tmp_path)


def test_continuing_under_a_different_identity_is_refused_and_names_the_field(tmp_path):
    store = new_store(tmp_path / "run", tiny_config())

    with pytest.raises(RunMismatchError, match="root_entropy"):
        store.check_identity(identity_for(tiny_config(), root_entropy=1))


def test_the_replication_target_can_grow_and_the_growth_persists(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(2))

    store.set_replications(5)

    assert RunStore.open(tmp_path / "run").replications == 5


def test_the_replication_target_cannot_shrink(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(5))

    with pytest.raises(ValueError, match="finished shards"):
        store.set_replications(2)


# -- shards and the ledger --------------------------------------------------


def test_a_written_shard_leaves_no_temporary_file_and_is_described_exactly(tmp_path):
    store = new_store(tmp_path / "run", tiny_config())
    (cell, *_) = expand_cells(tiny_config())
    key = cell_key(cell)

    entry = write_shard(store.directory, key, [])

    assert shard_path(store.directory, key).exists()
    assert list((store.directory / CELLS_DIR).glob("*.tmp")) == []
    assert entry.key == key and entry.rows == 0


def test_a_shard_never_appears_under_its_real_name_until_it_is_complete(tmp_path, monkeypatch):
    store = new_store(tmp_path / "run", tiny_config())
    (cell, *_) = expand_cells(tiny_config())
    key = cell_key(cell)

    def interrupted(source, destination):
        raise OSError("killed before the rename")

    monkeypatch.setattr("cp_evaluation.store.os.replace", interrupted)
    with pytest.raises(OSError, match="killed"):
        write_shard(store.directory, key, [])

    # The bytes exist only under the temporary name, which nothing reads.
    assert not shard_path(store.directory, key).exists()
    assert store.intact_entries() == {}


def test_a_recorded_cell_is_intact(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(1))
    _fill(store)

    assert set(store.intact_entries()) == set(store.expected_cells())


def test_a_shard_edited_after_it_was_recorded_stops_counting(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(2))
    _fill(store)
    victim = sorted(store.expected_cells())[0]
    with shard_path(store.directory, victim).open("ab") as handle:
        handle.write(b"tampered")

    assert victim not in store.intact_entries()
    assert store.status().remaining == 1


def test_a_shard_whose_row_count_disagrees_with_its_line_stops_counting(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(1))
    _fill(store)
    (key,) = store.expected_cells()
    real = store.intact_entries()[key]
    lied = LedgerEntry(key=key, rows=real.rows + 1, sha256=real.sha256)
    (store.directory / LEDGER_NAME).write_text(lied.to_line() + "\n")

    assert store.intact_entries() == {}


def test_a_missing_shard_is_not_done(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(2))
    _fill(store)
    victim = sorted(store.expected_cells())[0]
    shard_path(store.directory, victim).unlink()

    assert victim not in store.intact_entries()


def test_a_torn_final_ledger_line_is_dropped_not_fatal(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(2))
    _fill(store)
    with (store.directory / LEDGER_NAME).open("a") as handle:
        handle.write('{"key": "half a li')

    assert set(store.intact_entries()) == set(store.expected_cells())


def test_a_corrupt_line_that_is_not_the_last_is_an_error(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(2))
    _fill(store)
    lines = (store.directory / LEDGER_NAME).read_text().splitlines()
    (store.directory / LEDGER_NAME).write_text("\n".join(["not json", *lines]) + "\n")

    with pytest.raises(CorruptLedgerError, match=":1 "):
        store.intact_entries()


def test_a_shard_with_no_ledger_line_is_an_orphan_and_is_not_done(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(2))
    (cell, *_) = expand_cells(tiny_config(2))
    write_shard(store.directory, cell_key(cell), [])

    status = store.status()

    assert status.orphans == 1
    assert status.done == 0


def test_status_counts_done_remaining_and_per_scenario(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(3))
    before = store.status()
    _fill(store)
    after = store.status()

    assert (before.expected, before.done, before.remaining) == (3, 0, 3)
    assert (after.done, after.remaining, after.orphans) == (3, 0, 0)
    assert after.by_scenario == {"A": (3, 3)}


# -- reading ----------------------------------------------------------------


def test_an_incomplete_run_is_refused_unless_a_partial_read_is_asked_for(tmp_path):
    store = new_store(tmp_path / "run", tiny_config(3))

    with pytest.raises(IncompleteRunError, match="3 of 3"):
        store.read_frame()
    assert store.read_frame(partial=True).empty


def test_a_complete_run_reads_back_as_run_grid_would_have_returned_it(tmp_path):
    config = tiny_config(3)
    store = new_store(tmp_path / "run", config)
    _fill(store)

    frame = store.read_frame()

    pd.testing.assert_frame_equal(without_runtime(frame), without_runtime(reference_frame(config)))


def test_a_run_with_ineligible_cells_reads_back_with_the_dtypes_run_grid_gives(tmp_path):
    # A panel cell leaves `estimand` and `ci_basis` null for the estimators that
    # cannot run on it, so its shard holds an all-null text column. Read back and
    # concatenated, that is `object`; `run_grid` infers `str` for the same rows.
    config = mixed_config(1)
    store = new_store(tmp_path / "run", config)
    _fill(store)

    frame = store.read_frame()
    expected = reference_frame(config)

    assert frame.dtypes.astype(str).to_dict() == expected.dtypes.astype(str).to_dict()


def test_the_manifest_is_valid_json_with_a_schema_version(tmp_path):
    new_store(tmp_path / "run", tiny_config())

    manifest = json.loads((tmp_path / "run" / MANIFEST_NAME).read_text())

    assert manifest["schema"] == 1
    assert manifest["identity"]["tier"] == "tiny"
