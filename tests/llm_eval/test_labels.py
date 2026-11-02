import csv

import pytest
from cp_llm_eval.deterministic import Flag
from cp_llm_eval.graded import GradedStatement, grade_records
from cp_llm_eval.judge import VERDICTS, Finding
from cp_llm_eval.labels import (
    COLUMNS,
    Label,
    LabelError,
    export_csv,
    import_csv,
    load,
    load_sample,
    sample,
    save,
    save_sample,
    stratum,
)
from cp_llm_eval.runner import INTERPRETATIONS


@pytest.fixture(scope="module")
def graded(entry, stored):
    return grade_records(
        [entry], stored.records(INTERPRETATIONS), stored.records(VERDICTS), "claude-opus-5-5"
    )


def test_the_sample_is_drawn_from_judged_final_statements_without_repeats(graded):
    chosen = sample(graded, size=10, seed=1)

    judged = {s.statement_id for n in graded if n.judged for s in n.statements}
    ids = [s.statement_id for s in chosen]
    assert len(ids) == len(set(ids)) == 10
    assert set(ids) <= judged


def test_every_stratum_is_represented_when_it_exists(graded):
    chosen = sample(graded, size=9, seed=0)

    assert any(s.judge for s in chosen)
    assert any(not s.judge and s.deterministic for s in chosen)
    assert any(not s.judge and not s.deterministic for s in chosen)


def test_a_small_stratum_gives_all_it_has_and_the_rest_is_filled(graded):
    deterministic_only = [
        s for n in graded if n.judged for s in n.statements if s.deterministic and not s.judge
    ]
    chosen = sample(graded, size=30, seed=0)

    assert len(deterministic_only) < 10
    assert {s.statement_id for s in deterministic_only} <= {s.statement_id for s in chosen}
    assert len(chosen) == min(30, sum(len(n.statements) for n in graded if n.judged))


def test_the_samples_row_order_does_not_group_the_strata(graded):
    strata = [stratum(s) for s in sample(graded, size=12, seed=0)]

    assert len(set(strata)) == 3
    assert strata != sorted(strata)
    assert strata != sorted(strata, reverse=True)


def test_the_sample_is_the_same_for_the_same_seed(graded):
    assert [s.statement_id for s in sample(graded, 12, seed=3)] == [
        s.statement_id for s in sample(graded, 12, seed=3)
    ]


def test_the_exported_file_shows_the_statement_and_its_facts_but_no_grade(graded, entry, tmp_path):
    chosen = [s for n in graded if n.judged for s in n.statements]
    path = tmp_path / "labels.csv"

    export_csv(chosen, {entry.case_id: entry.facts}, path)

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == COLUMNS
    assert [r["statement_id"] for r in rows] == [s.statement_id for s in chosen]
    assert all(r["verdict"] == r["category"] == r["note"] == "" for r in rows)
    assert all("{" not in r["statement"] for r in rows)
    assert any("est.ipw.point = " in r["cited_facts"] for r in rows)
    assert any("ipw: point " in r["context"] for r in rows)
    text = path.read_text()
    assert "J1" not in text and "D4" not in text


def _write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" for column in COLUMNS} | row)


def test_a_fully_labelled_file_is_imported(tmp_path):
    path = tmp_path / "l.csv"
    _write(
        path,
        [
            {"statement_id": "a", "verdict": "ok"},
            {"statement_id": "b", "verdict": "Misreading", "category": "j1", "note": "n"},
        ],
    )

    assert import_csv(path, ["a", "b"]) == [
        Label("a", "ok", "", ""),
        Label("b", "misreading", "J1", "n"),
    ]


def test_every_unusable_row_is_reported_at_once(tmp_path):
    path = tmp_path / "l.csv"
    _write(
        path,
        [
            {"statement_id": "zz", "verdict": "ok"},
            {"statement_id": "a", "verdict": ""},
            {"statement_id": "b", "verdict": "misreading"},
            {"statement_id": "c", "verdict": "ok", "category": "J2"},
            {"statement_id": "c", "verdict": "ok"},
        ],
    )

    with pytest.raises(LabelError) as caught:
        import_csv(path, ["a", "b", "c"])
    message = str(caught.value)
    expected = ("line 2: unknown", "line 3: verdict", "line 4: a misreading", "line 5", "line 6")
    for fragment in expected:
        assert fragment in message


def test_a_file_missing_a_sampled_statement_is_refused(tmp_path):
    path = tmp_path / "l.csv"
    _write(path, [{"statement_id": "a", "verdict": "ok"}])

    with pytest.raises(LabelError) as caught:
        import_csv(path, ["a", "b"])
    assert "1 sampled statement(s) not labelled" in str(caught.value)
    assert "'b'" in str(caught.value)


def test_the_sample_ids_round_trip_in_order(graded, tmp_path):
    chosen = sample(graded, size=9, seed=0)

    save_sample(chosen, tmp_path / "sub" / "sample.json")

    assert load_sample(tmp_path / "sub" / "sample.json") == [s.statement_id for s in chosen]


def test_saved_labels_load_back_and_none_saved_is_none(tmp_path):
    labels = [Label("a", "ok", "", ""), Label("b", "misreading", "D4", "x")]
    save(labels, tmp_path / "labels.json")

    assert load(tmp_path / "labels.json") == labels
    assert load(tmp_path / "absent.json") == []


# the D0 stratum -----------------------------------------------------------


def _graded_statement(deterministic=(), judge=None):
    return GradedStatement(
        statement_id="s",
        interpretation="i",
        case_id="a",
        section="summary",
        position=0,
        raw="r",
        rendered="r",
        deterministic=deterministic,
        judge=judge,
    )


def test_a_statement_carrying_a_d0_is_labelled_as_d0_even_when_the_judge_flagged_it():
    """D0 wins the tie, so the labeller sees the arm's primary endpoint.

    The judge flags far more statements than the matcher does, so without the
    priority a D0 statement would nearly always land in the judge's stratum
    and the endpoint would be measured on whatever was left over.
    """
    both = _graded_statement(
        deterministic=(Flag("D0", "no fact supports 7.7321"),),
        judge=(Finding(category="J1", reason="r"),),
    )

    assert stratum(both) == 0


def test_the_other_three_strata_keep_their_order():
    judged = _graded_statement(judge=(Finding(category="J1", reason="r"),))
    grader_only = _graded_statement(deterministic=(Flag("D4", "summary", "true"),))
    unflagged = _graded_statement()

    assert (stratum(judged), stratum(grader_only), stratum(unflagged)) == (1, 2, 3)


def test_a_rounded_or_ambiguous_value_is_not_the_d0_stratum():
    """Neither is a misreading, so neither earns the endpoint's stratum."""
    rounded = _graded_statement(deterministic=(Flag("D0-rounded", "r"),))
    ambiguous = _graded_statement(deterministic=(Flag("D0-ambiguous", "r"),))

    assert (stratum(rounded), stratum(ambiguous)) == (3, 3)


def test_the_sample_spans_all_four_strata_once_an_ungrounded_arm_is_in_the_store(
    entry, stored_three_arms
):
    graded = grade_records(
        [entry],
        stored_three_arms.records(INTERPRETATIONS),
        stored_three_arms.records(VERDICTS),
        "claude-opus-5-5",
    )

    chosen = sample(graded, size=40, seed=0)

    assert {stratum(s) for s in chosen} == {0, 1, 2, 3}
    # The D0 statements come from the ungrounded arm alone: on a grounded arm
    # every value was substituted from the table, so none can be unsupported.
    d0 = {s.interpretation for s in chosen if stratum(s) == 0}
    ungrounded = {n.interpretation for n in graded if n.variant == "ungrounded"}
    assert d0 and d0 <= ungrounded
