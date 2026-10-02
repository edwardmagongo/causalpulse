"""One call to the real API. Run by `make check-llm`, never by `make check`.

Skipped unless `CP_LLM_LIVE=1`, whatever credentials are present: a developer
with a key in their environment must not have the ordinary gate spend money.
"""

import os

import pytest
from cp_llm.interpret import interpret

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.environ.get("CP_LLM_LIVE") != "1", reason="set CP_LLM_LIVE=1 to call the API"
)
def test_the_real_model_writes_a_narrative_that_verifies(facts):
    from cp_llm.client import ClaudeModel

    result = interpret(facts, ClaudeModel())

    assert result.attempts[-1].violations == ()
    assert "{" not in result.rendered.to_markdown()
    usage = [(a.usage.input_tokens, a.usage.output_tokens) for a in result.attempts]
    print(f"\nmodel={result.model} attempts={len(result.attempts)} usage={usage}")
    for attempt in result.attempts[:-1]:
        for violation in attempt.violations:
            print(f"  first-attempt violation [{violation.rule}] {violation.reason}")
