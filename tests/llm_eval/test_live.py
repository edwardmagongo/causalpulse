"""One real judge call. Run by `make check-llm`, never by `make check`.

Skipped unless `CP_LLM_LIVE=1`, whatever credentials are present, as phase 1's
live test is. It also answers whether the API
accepts the judge's reply schema, one entry per statement.
"""

import os

import pytest
from cp_llm.render import substitute
from cp_llm_eval.judge import JUDGE_SYSTEM, judge_message, reply_error
from llm.builders import good_narrative

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("CP_LLM_LIVE") != "1", reason="set CP_LLM_LIVE=1 to call the API"
    ),
]


def test_the_real_judge_returns_a_verdict_for_every_statement(facts):
    import anthropic
    from cp_llm_eval.judge import ClaudeJudge

    client = anthropic.Anthropic()
    print(f"\nbase_url={client.base_url}")
    statements = tuple(
        (section, substitute(text, facts)) for section, text in good_narrative().statements()
    )

    call = ClaudeJudge(client=client).judge(JUDGE_SYSTEM, judge_message(statements, facts))

    assert call.reply is not None, call.stop_reason
    assert reply_error(call.reply, len(statements)) is None
    print(f"judge={call.model} usage=({call.usage.input_tokens}, {call.usage.output_tokens})")
    for verdict in call.reply.statements:
        for finding in verdict.findings:
            print(f"  statement {verdict.number}: {finding.category} {finding.reason}")
