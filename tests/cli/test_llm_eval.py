"""`causalpulse llm-eval` end to end, on the fast single-method case, with no network.

`build_model` and `build_judge` are replaced, so the paid steps run against a
scripted model and judge. Everything else is the real command.
"""

import csv
import json
from pathlib import Path

import pytest
from cp_cli import __main__ as cli
from cp_cli.commands import llm_eval
from cp_llm.narrative import Narrative
from cp_llm_eval import judge as judge_module
from cp_llm_eval.judge import JudgeReply, StatementVerdict
from cp_llm_eval.runner import INTERPRETATIONS
from cp_llm_eval.store import Store, read_export
from llm.builders import _band, _estimate
from llm_eval.fakes import ReplyJudge, ReplyModel

SINGLE = Narrative(
    summary=["{cmp.ate.percentage_point.not_compared_reason}"],
    estimates=[_estimate("difference_in_means")],
    validity=[_band("difference_in_means")],
    agreement=[],
    caveats=[],
)


def _clean(n):
    return JudgeReply(statements=[StatementVerdict(number=i, findings=[]) for i in range(1, n + 1)])


@pytest.fixture
def scripted(monkeypatch):
    model, judge = ReplyModel([SINGLE]), ReplyJudge(_clean)
    monkeypatch.setattr(llm_eval, "build_model", lambda *_: model)
    monkeypatch.setattr(llm_eval, "build_judge", lambda *_: judge)
    return model, judge


def _exported(tmp_path, monkeypatch):
    """Build, run, grade and export a labelling sample; return (corpus, store, sample csv)."""
    monkeypatch.chdir(cli.__file__.rsplit("/packages/", 1)[0])
    corpus, store = tmp_path / "corpus", tmp_path / "store"
    common = ["--corpus", str(corpus), "--store", str(store)]

    assert cli.main(["llm-eval", "build", "--out", str(corpus), "--case", "awkward-single"]) == 0
    assert cli.main(["llm-eval", "run", *common, "--max-calls", "4", "--replicates", "2"]) == 0
    assert cli.main(["llm-eval", "grade", *common, "--max-calls", "2"]) == 0
    sample = tmp_path / "sample.csv"
    assert cli.main(["llm-eval", "label", "export", *common, "--out", str(sample)]) == 0
    return corpus, store, sample


def _read(sample):
    with sample.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write(sample, rows):
    with sample.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_run_grade_label_and_report(tmp_path, scripted, monkeypatch, capsys):
    corpus, store, sample = _exported(tmp_path, monkeypatch)
    out = tmp_path / "report"
    common = ["--corpus", str(corpus), "--store", str(store)]

    rows = _read(sample)
    for row in rows:
        row["verdict"] = "ok"
    _write(sample, rows)
    assert cli.main(["llm-eval", "label", "import", *common, "--csv", str(sample)]) == 0
    assert cli.main(["llm-eval", "report", *common, "--out", str(out)]) == 0

    model, judge = scripted
    assert (model.calls, judge.calls) == (2, 2)
    report = json.loads((out / "report.json").read_text())
    assert report["run"]["variants"]["baseline"]["verified"] == 2
    assert report["validation"]["labelled"] == len(rows) == 6
    assert len(read_export(out / "interpretations.jsonl.gz")) == 2
    assert len(read_export(out / "verdicts.jsonl.gz")) == 2
    assert "wrote" in capsys.readouterr().out

    # The committed bundle, restored into an empty store with the labels,
    # reports the same study without a call.
    again, out_again = tmp_path / "again", tmp_path / "report-again"
    assert cli.main(["llm-eval", "restore", "--bundle", str(out), "--store", str(again)]) == 0
    (again / "labels.json").write_bytes((store / "labels.json").read_bytes())
    args = ["--corpus", str(corpus), "--store", str(again), "--out", str(out_again)]
    assert cli.main(["llm-eval", "report", *args]) == 0
    assert (out_again / "report.json").read_text() == (out / "report.json").read_text()
    assert (model.calls, judge.calls) == (2, 2)


def test_a_rerun_pays_for_nothing_already_stored(tmp_path, scripted, monkeypatch):
    monkeypatch.chdir(cli.__file__.rsplit("/packages/", 1)[0])
    corpus, store = tmp_path / "corpus", tmp_path / "store"
    common = ["--corpus", str(corpus), "--store", str(store), "--max-calls", "4"]
    common += ["--replicates", "1"]
    cli.main(["llm-eval", "build", "--out", str(corpus), "--case", "awkward-single"])
    cli.main(["llm-eval", "run", *common])

    assert cli.main(["llm-eval", "run", *common]) == 0
    assert scripted[0].calls == 1


def test_a_run_the_budget_cuts_short_exits_non_zero(tmp_path, scripted, monkeypatch):
    monkeypatch.chdir(cli.__file__.rsplit("/packages/", 1)[0])
    corpus = tmp_path / "corpus"
    cli.main(["llm-eval", "build", "--out", str(corpus), "--case", "awkward-single"])
    common = ["--corpus", str(corpus), "--store", str(tmp_path / "store")]

    # The first reserves two calls and uses one; the second then needs 1 + 2 > 2.
    assert cli.main(["llm-eval", "run", *common, "--max-calls", "2", "--replicates", "2"]) == 1


def test_a_paid_step_without_a_ceiling_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["llm-eval", "run", "--corpus", "c", "--store", "s"])


def test_grade_without_a_ceiling_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["llm-eval", "grade", "--corpus", "c", "--store", "s"])


def test_a_grade_the_budget_cuts_short_exits_non_zero(tmp_path, scripted, monkeypatch):
    monkeypatch.chdir(cli.__file__.rsplit("/packages/", 1)[0])
    corpus = tmp_path / "corpus"
    cli.main(["llm-eval", "build", "--out", str(corpus), "--case", "awkward-single"])
    common = ["--corpus", str(corpus), "--store", str(tmp_path / "store")]
    assert cli.main(["llm-eval", "run", *common, "--max-calls", "4", "--replicates", "2"]) == 0

    # Two verified narratives to judge, room for one call.
    assert cli.main(["llm-eval", "grade", *common, "--max-calls", "1"]) == 1
    assert scripted[1].calls == 1


def test_an_unknown_case_is_refused(tmp_path, capsys):
    assert cli.main(["llm-eval", "build", "--out", str(tmp_path), "--case", "nope"]) == 1
    assert "unknown case" in capsys.readouterr().err


def test_an_import_missing_a_sampled_row_is_refused(tmp_path, scripted, monkeypatch, capsys):
    corpus, store, sample = _exported(tmp_path, monkeypatch)
    rows = _read(sample)
    for row in rows:
        row["verdict"] = "ok"
    dropped = rows.pop()["statement_id"]
    _write(sample, rows)
    common = ["--corpus", str(corpus), "--store", str(store)]
    capsys.readouterr()

    assert cli.main(["llm-eval", "label", "import", *common, "--csv", str(sample)]) == 1
    assert dropped in capsys.readouterr().err
    assert not (store / "labels.json").exists()


def test_an_import_with_no_export_is_refused(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "labelled.csv"
    csv_path.write_text("statement_id\n")
    common = ["--corpus", str(tmp_path / "corpus"), "--store", str(tmp_path / "store")]

    assert cli.main(["llm-eval", "label", "import", *common, "--csv", str(csv_path)]) == 1
    assert "label export" in capsys.readouterr().err


# the ungrounded arm, and the cost of a revised judge ----------------------


def _built(tmp_path, monkeypatch):
    monkeypatch.chdir(cli.__file__.rsplit("/packages/", 1)[0])
    corpus, store = tmp_path / "corpus", tmp_path / "store"
    assert cli.main(["llm-eval", "build", "--out", str(corpus), "--case", "awkward-single"]) == 0
    return ["--corpus", str(corpus), "--store", str(store)]


def test_the_ungrounded_arm_runs_and_is_counted_as_its_own_outcome(
    tmp_path, scripted, monkeypatch
):
    """It has no verifier, so it is neither verified nor refused, and is judged all the same."""
    common = _built(tmp_path, monkeypatch)
    out = tmp_path / "report"

    # 3, not 2: a grounded job reserves both of its attempts up front, while
    # the ungrounded arm reserves the one call it can make.
    assert cli.main([*(["llm-eval", "run"] + common), "--max-calls", "3", "--replicates", "1",
                     "--arm", "ungrounded", "--arm", "baseline"]) == 0
    assert cli.main([*(["llm-eval", "grade"] + common), "--max-calls", "2"]) == 0
    assert cli.main([*(["llm-eval", "report"] + common), "--out", str(out)]) == 0

    report = json.loads((out / "report.json").read_text())
    arm = report["run"]["variants"]["ungrounded"]
    assert (arm["interpretations"], arm["unverified"], arm["verified"]) == (1, 1, 0)
    assert report["denominators"]["ungrounded"]["can_refuse"] is False
    # Judged under the same prompt as the grounded arm, which is what makes
    # the J-rates differ by arm rather than by instrument.
    _model, judge = scripted
    assert judge.calls == 2


def test_the_older_variant_flag_still_names_an_arm(tmp_path, scripted, monkeypatch):
    common = _built(tmp_path, monkeypatch)

    assert cli.main([*(["llm-eval", "run"] + common), "--max-calls", "2", "--replicates", "1",
                     "--variant", "baseline"]) == 0

    store = Store(Path(common[3]))
    (record,) = store.records(INTERPRETATIONS)
    assert record["key"]["variant"] == "baseline"


def test_grade_refuses_to_bill_a_rejudge_until_it_is_asked_for(
    tmp_path, scripted, monkeypatch, capsys
):
    """A revised judge prompt invalidates every stored verdict by key.

    Without the guard, the next `grade` would quietly pay for all of them
    again -- 199 calls on the real study.
    """
    common = _built(tmp_path, monkeypatch)
    assert cli.main([*(["llm-eval", "run"] + common), "--max-calls", "2", "--replicates", "1"]) == 0
    assert cli.main([*(["llm-eval", "grade"] + common), "--max-calls", "1"]) == 0
    _model, judge = scripted
    assert judge.calls == 1

    monkeypatch.setattr(judge_module, "judge_prompt_version", lambda: "a-revised-judge")
    monkeypatch.setattr(llm_eval, "judge_prompt_version", lambda: "a-revised-judge")

    assert cli.main([*(["llm-eval", "grade"] + common), "--max-calls", "1"]) == 1
    message = capsys.readouterr().err
    assert "a-revised-judge" in message and "--rejudge" in message
    assert judge.calls == 1

    assert cli.main([*(["llm-eval", "grade"] + common), "--max-calls", "1", "--rejudge"]) == 0
    assert judge.calls == 2


def test_report_writes_the_pre_registered_endpoints_beside_it(tmp_path, scripted, monkeypatch):
    common = _built(tmp_path, monkeypatch)
    out = tmp_path / "report"

    assert cli.main([*(["llm-eval", "run"] + common), "--max-calls", "3", "--replicates", "1",
                     "--arm", "ungrounded", "--arm", "baseline"]) == 0
    assert cli.main([*(["llm-eval", "grade"] + common), "--max-calls", "2"]) == 0
    assert cli.main([*(["llm-eval", "report"] + common), "--out", str(out)]) == 0

    prereg = (out / "prereg.md").read_text()
    assert prereg.startswith("# Pre-registered endpoints")
    assert "P0 passed" in prereg
    assert "P1 " in prereg
