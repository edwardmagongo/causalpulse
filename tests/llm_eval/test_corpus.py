import json
from pathlib import Path

import pytest
from cp_llm.facts import FACTS_FILENAME
from cp_llm_eval.corpus import (
    DIVERGENT,
    FAIL_STATUS,
    MANIFEST_FILENAME,
    SINGLE_ESTIMATE,
    Case,
    CorpusPropertyError,
    _cross_section,
    build,
    build_case,
    default_cases,
    load,
)

ROOT = Path(__file__).resolve().parents[2]


def _single(case_id: str = "t-single", requirements=(SINGLE_ESTIMATE,)) -> Case:
    return Case(
        case_id,
        "cross_section",
        "awkward",
        "one method",
        _cross_section("D"),
        estimators=("difference_in_means",),
        requirements=requirements,
    )


def test_a_case_whose_fit_lacks_its_property_is_refused():
    with pytest.raises(CorpusPropertyError, match="divergent"):
        build_case(_single(requirements=(DIVERGENT,)), ROOT)


def test_a_case_whose_fit_has_its_property_is_built():
    facts = build_case(_single(), ROOT)

    assert SINGLE_ESTIMATE.holds(facts)
    assert facts.get("est.difference_in_means.point") is not None


def test_build_writes_each_table_and_a_manifest_that_load_reads_back(tmp_path):
    entries = build([_single()], tmp_path, ROOT)

    assert (tmp_path / "t-single" / FACTS_FILENAME).exists()
    loaded = load(tmp_path)
    assert [e.case_id for e in loaded] == ["t-single"]
    assert loaded[0].facts == entries[0].facts
    assert loaded[0].facts_sha256 == entries[0].facts_sha256
    assert loaded[0].requirements == ("exactly one estimate",)
    assert loaded[0].estimators == ("difference_in_means",)


def test_load_refuses_a_table_changed_after_the_build(tmp_path):
    build([_single()], tmp_path, ROOT)
    path = tmp_path / "t-single" / FACTS_FILENAME
    path.write_text(path.read_text() + " ")

    with pytest.raises(ValueError, match="does not match the manifest"):
        load(tmp_path)


def test_no_manifest_is_written_when_a_case_is_refused(tmp_path):
    with pytest.raises(CorpusPropertyError):
        build([_single(), _single("t-bad", (DIVERGENT,))], tmp_path, ROOT)

    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_two_cases_with_one_id_are_refused(tmp_path):
    with pytest.raises(ValueError, match="share an id"):
        build([_single(), _single()], tmp_path, ROOT)


def test_the_manifest_records_each_cases_declaration(tmp_path):
    build([_single()], tmp_path, ROOT)

    record = json.loads((tmp_path / MANIFEST_FILENAME).read_text())["cases"][0]
    assert record["design"] == "cross_section"
    assert record["group"] == "awkward"
    assert len(record["facts_sha256"]) == 64


def test_the_default_corpus_is_the_designs_list():
    ids = [case.case_id for case in default_cases()]

    assert ids == [
        *(f"cross-{s}" for s in "ABCDFG"),
        "cross-H-complete",
        *(f"panel-P{i}" for i in range(1, 8)),
        *(f"donor-S{i}" for i in range(6)),
        "real-lalonde",
        "real-prop99",
        "awkward-fail-divergent",
        "awkward-divergent",
        "awkward-single",
    ]
    designs = {case.case_id: case.design for case in default_cases()}
    assert designs["real-prop99"] == "donor_pool"
    assert designs["real-lalonde"] == "cross_section"


def test_every_awkward_case_declares_a_property():
    awkward = [case for case in default_cases() if case.group == "awkward"]

    assert awkward
    assert all(case.requirements for case in awkward)
    assert FAIL_STATUS in awkward[0].requirements


@pytest.mark.slow
def test_every_default_case_builds_and_shows_its_property(tmp_path):
    # About 20 minutes: P7 at its own protocol alone takes about 7.
    entries = build(default_cases(), tmp_path, ROOT)

    assert len(entries) == len(default_cases())
