"""A real corpus entry for `tests/llm_eval/`.

Scenario D at n=400, seed 0, fitted by `difference_in_means` and `ipw`: the
same fit as `tests/llm/conftest.py`, so phase 1's `good_narrative` verifies
against it. Built through `cp_llm_eval.corpus.build`, so the entry is what the
harness itself would produce.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cp_llm.facts import FactTable
from cp_llm_eval.corpus import Case, CorpusEntry, _cross_section, build

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def entry(tmp_path_factory: pytest.TempPathFactory) -> CorpusEntry:
    case = Case(
        "t-d",
        "cross_section",
        "scenario",
        "scenario D, two methods",
        _cross_section("D"),
        estimators=("difference_in_means", "ipw"),
    )
    (built,) = build([case], tmp_path_factory.mktemp("corpus"), ROOT)
    return built


@pytest.fixture(scope="session")
def facts(entry: CorpusEntry) -> FactTable:
    return entry.facts


@pytest.fixture(scope="session")
def stored(entry: CorpusEntry, tmp_path_factory: pytest.TempPathFactory):
    """A small finished run of `entry`: two variants, two replicates, then judged.

    baseline r1 and r2 verify as phase 1's `good_narrative`; cite-each-status
    r1 is refused twice (a bare value); r2 verifies with one more caveat, an
    uncited status summary (D4) that phase 1's verifier lets through. The
    judge flags J1 on the first statement of every narrative.
    """
    from cp_llm_eval.judge import Finding, JudgeReply, StatementVerdict, judge_all, judge_jobs
    from cp_llm_eval.runner import INTERPRETATIONS, CallBudget, plan_jobs, run
    from cp_llm_eval.store import Store
    from cp_llm_eval.variants import BASELINE, CITE_EACH_STATUS
    from llm.builders import good_narrative, swap

    from llm_eval.fakes import D4_STATEMENT, ReplyJudge, ReplyModel

    good = good_narrative()
    bare = swap(good, "validity", 0, good.validity[0] + " That is 2 steps down.")
    data = good.model_dump()
    data["caveats"].append(D4_STATEMENT)
    summarised = type(good)(**data)

    store = Store(tmp_path_factory.mktemp("stored"))
    jobs = plan_jobs([entry], [BASELINE, CITE_EACH_STATUS], 2, "claude-opus-5-5", "medium")
    model = ReplyModel([good, good, bare, bare, summarised])
    run(jobs, lambda: model, store, CallBudget(20), workers=1)

    def reply(n):
        verdicts = [StatementVerdict(number=i, findings=[]) for i in range(1, n + 1)]
        verdicts[0] = StatementVerdict(number=1, findings=[Finding(category="J1", reason="r")])
        return JudgeReply(statements=verdicts)

    jobs = judge_jobs(store.records(INTERPRETATIONS), {entry.case_id: entry}, "claude-opus-5-5")
    judge_all(jobs, lambda: ReplyJudge(reply), store, CallBudget(20), workers=1)
    return store
