import hashlib
import json

import cp_cli.commands.interpret as interpret_command
import pytest
from cp_cli.__main__ import main
from cp_core.results import VALIDITY_SCORE_LIMITS
from cp_llm.facts import FactTable
from cp_llm.model import ModelCallError
from llm.builders import ScriptedModel, good_narrative, swap


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """One real `analyze --report` run over the two methods `good_narrative` cites."""
    out = tmp_path_factory.mktemp("run")
    code = main(
        ["analyze", "--scenario", "D", "--n-units", "400", "--seed", "0", "--out", str(out),
         "--report", "--estimators", "difference_in_means,ipw"]
    )
    assert code == 0
    return out


def _use(monkeypatch, script):
    model = ScriptedModel(script)
    seen = []

    def build(name, effort):
        seen.append((name, effort))
        return model

    monkeypatch.setattr(interpret_command, "build_model", build)
    return model, seen


def test_analyze_report_writes_a_fact_table_interpret_can_read(run_dir):
    facts = FactTable.from_json((run_dir / "facts.json").read_text())

    assert facts.get("est.difference_in_means.point") is not None
    assert facts.get("est.ipw.point") is not None
    assert facts.get("cmp.ate.percentage_point.agreement") is not None


def test_interpret_writes_the_narrative_and_its_record(run_dir, monkeypatch, capsys):
    _, seen = _use(monkeypatch, [good_narrative()])

    code = main(["interpret", str(run_dir)])

    assert code == 0
    assert seen == [("claude-opus-5-5", "medium")]
    markdown = (run_dir / "interpretation.md").read_text()
    assert markdown.startswith("# Interpretation")
    assert markdown.rstrip().endswith(VALIDITY_SCORE_LIMITS)
    assert "{" not in markdown
    record = json.loads((run_dir / "interpretation.json").read_text())
    assert record["model"] == "scripted-model"
    assert len(record["attempts"]) == 1
    assert record["narrative"] == good_narrative().model_dump()
    assert record["facts_sha256"] == hashlib.sha256(
        (run_dir / "facts.json").read_bytes()
    ).hexdigest()
    assert "interpretation.md" in capsys.readouterr().out


def test_model_and_effort_reach_the_model_factory(run_dir, monkeypatch):
    _, seen = _use(monkeypatch, [good_narrative()])

    assert main(["interpret", str(run_dir), "--model", "claude-sonnet-5-5", "--effort", "low"]) == 0

    assert seen == [("claude-sonnet-5-5", "low")]


def test_a_refusal_prints_the_violations_writes_nothing_and_exits_one(
    run_dir, monkeypatch, capsys
):
    for name in ("interpretation.md", "interpretation.json"):
        (run_dir / name).unlink(missing_ok=True)
    band = good_narrative().validity[0]
    bare = swap(good_narrative(), "validity", 0, band + " That is 2 steps down.")
    _use(monkeypatch, [bare, bare])

    code = main(["interpret", str(run_dir)])

    assert code == 1
    assert not (run_dir / "interpretation.md").exists()
    assert not (run_dir / "interpretation.json").exists()
    err = capsys.readouterr().err
    assert "failed verification on all 2 attempt" in err
    assert "[bare_value]" in err
    assert "That is 2 steps down." in err


def test_a_model_that_does_not_answer_exits_one_with_its_stop_reason(
    run_dir, monkeypatch, capsys
):
    _use(monkeypatch, ["refusal"])

    assert main(["interpret", str(run_dir)]) == 1
    assert "stop reason: refusal" in capsys.readouterr().err


def test_an_unreachable_model_exits_one_with_one_line(run_dir, monkeypatch, capsys):
    class Unreachable:
        def complete(self, system, turns):
            raise ModelCallError("no API credentials were found; set ANTHROPIC_API_KEY")

    monkeypatch.setattr(interpret_command, "build_model", lambda name, effort: Unreachable())

    assert main(["interpret", str(run_dir)]) == 1
    assert capsys.readouterr().err.strip() == (
        "no API credentials were found; set ANTHROPIC_API_KEY"
    )


def test_a_directory_without_facts_says_what_to_run(tmp_path, monkeypatch, capsys):
    _use(monkeypatch, [good_narrative()])

    assert main(["interpret", str(tmp_path)]) == 1
    assert "run `causalpulse analyze --report` first" in capsys.readouterr().err


def test_an_unknown_effort_is_a_usage_error(run_dir):
    with pytest.raises(SystemExit) as excinfo:
        main(["interpret", str(run_dir), "--effort", "extreme"])
    assert excinfo.value.code == 2
