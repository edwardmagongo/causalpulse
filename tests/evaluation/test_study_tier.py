import dataclasses

from cp_evaluation.grid import cell_key, expand_cells
from cp_evaluation.tiers import STUDY_REPLICATIONS, full_config, study_config


def test_the_study_tier_is_the_full_tier_at_fifty_replications():
    study, full = study_config(), full_config()

    assert STUDY_REPLICATIONS == 50
    assert study.replications == 50
    # Equal in every field but the replication count, so a widened `full` axis
    # reaches the study with no second edit.
    assert dataclasses.replace(study, replications=full.replications) == full


def test_the_study_grid_is_five_times_the_full_grid():
    # 890 cells at 10 replications is 89 configurations (80 cross-section/panel
    # plus the six donor-pool scenarios plus 3 CATE); 50 replications of each is 4,450.
    # Counted from the shipped grids rather than stated.
    assert len(expand_cells(full_config())) == 890
    assert len(expand_cells(study_config())) == 4450


def test_every_study_cell_has_its_own_key():
    keys = [cell_key(cell) for cell in expand_cells(study_config())]

    assert len(set(keys)) == len(keys) == 4450


def test_a_cell_key_does_not_depend_on_how_many_replications_were_asked_for():
    # What makes extending a run cheap: replication index 3 is the same cell
    # whether the grid asks for 5 replications or 50.
    small = {cell_key(c) for c in expand_cells(dataclasses.replace(full_config(), replications=5))}
    large = {cell_key(c) for c in expand_cells(dataclasses.replace(full_config(), replications=50))}

    assert small < large
    assert len(small) == 445
