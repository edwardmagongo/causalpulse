"""`causalpulse llm-eval` end to end, on the fast single-method case, with no network.

`build_model` and `build_judge` are replaced, so the paid steps run against a
scripted model and judge. Everything else is the real command.
"""

import csv
import json

import pytest
from cp_cli import __main__ as cli
from cp_cli.commands import llm_eval
from cp_llm.narrative import Narrative
from cp_llm_eval.judge import JudgeReply, StatementVerdict
from cp_llm_eval.store import read_export
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
