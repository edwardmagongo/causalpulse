import dataclasses

from cp_evaluation.grid import CateCell, LongPanelCell, cell_key, expand_cells
from cp_evaluation.tiers import (
    STUDY_REPLICATIONS,
    cate_study_config,
    full_config,
    p7_study_config,
    study_config,
)


def test_the_study_tier_is_the_full_tier_at_fifty_replications():
    study, full = study_config(), full_config()

    assert STUDY_REPLICATIONS == 50
    assert study.replications == 50
    # Equal in every field but the replication count, so a widened `full` axis
    # reaches the study with no second edit.
    assert dataclasses.replace(study, replications=full.replications) == full


def test_the_study_grid_is_five_times_the_full_grid():
    # 900 cells at 10 replications is 90 configurations (80 cross-section/panel
    # plus the six donor-pool scenarios plus 3 CATE plus 1 P7); 50 replications of each is 4,500.
    # Counted from the shipped grids rather than stated.
    assert len(expand_cells(full_config())) == 900
    assert len(expand_cells(study_config())) == 4500


def test_every_study_cell_has_its_own_key():
    keys = [cell_key(cell) for cell in expand_cells(study_config())]

    assert len(set(keys)) == len(keys) == 4500


def test_a_cell_key_does_not_depend_on_how_many_replications_were_asked_for():
    # What makes extending a run cheap: replication index 3 is the same cell
    # whether the grid asks for 5 replications or 50.
    small = {cell_key(c) for c in expand_cells(dataclasses.replace(full_config(), replications=5))}
    large = {cell_key(c) for c in expand_cells(dataclasses.replace(full_config(), replications=50))}

    assert small < large
    assert len(small) == 450


def test_the_cate_tier_is_exactly_the_studys_cate_cells():
    # Identity-derived generators: equal cells means byte-identical data, so a
    # CATE-only run measures exactly what the CATE part of a full study would.
    cate_only = expand_cells(cate_study_config())
    from_study = tuple(c for c in expand_cells(study_config()) if isinstance(c, CateCell))

    assert cate_only == from_study
    assert len(cate_only) == 150


def test_the_cate_tier_carries_no_other_arm():
    config = cate_study_config()

    assert config.cross_section is None
    assert config.panel is None
    assert config.donor_pool is None
    assert config.cate == full_config().cate


def test_the_p7_tier_is_exactly_the_studys_p7_cells():
    p7 = expand_cells(p7_study_config())

    assert p7 == tuple(c for c in expand_cells(study_config()) if isinstance(c, LongPanelCell))
    assert len(p7) == 50
    assert (p7_study_config().cross_section, p7_study_config().panel) == (None, None)
    assert p7_study_config().donor_pool is None
    assert p7_study_config().cate is None
