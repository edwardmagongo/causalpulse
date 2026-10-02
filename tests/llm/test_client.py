"""The Claude adapter against a transport that answers locally.

No request leaves the process: the SDK is given an `httpx2.MockTransport`, so
what is checked is the request the adapter builds and how it reads a reply.
"""

import json
import os

import anthropic
import httpx2
import pytest
from cp_llm.client import ClaudeModel
from cp_llm.model import DEFAULT_EFFORT, DEFAULT_MODEL, MAX_TOKENS, ModelCallError, Turn

from llm.builders import good_narrative


class Wire:
    def __init__(self):
        self.requests = []
        self.replies = []

    def reply(self, text, stop_reason="end_turn", model="claude-opus-5-5"):
        self.replies.append(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": text},
                ],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 1234, "output_tokens": 56},
            }
        )

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        return httpx2.Response(200, json=self.replies.pop(0))


@pytest.fixture
def wire():
    return Wire()


@pytest.fixture
def model(wire):
    client = anthropic.Anthropic(
        api_key="not-a-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(wire)),
    )
    return ClaudeModel(client=client)


def test_the_request_constrains_the_reply_to_the_narrative_schema(model, wire):
    wire.reply(good_narrative().model_dump_json())

    model.complete("the system prompt", [Turn("user", "the facts")])

    (body,) = wire.requests
    assert body["model"] == DEFAULT_MODEL
    assert body["max_tokens"] == MAX_TOKENS
    assert body["system"] == "the system prompt"
    assert body["messages"] == [{"role": "user", "content": "the facts"}]
    assert body["output_config"]["effort"] == DEFAULT_EFFORT
    schema = body["output_config"]["format"]
    assert schema["type"] == "json_schema"
    assert set(schema["schema"]["required"]) == {
        "summary", "estimates", "validity", "agreement", "caveats",
    }
    assert schema["schema"]["additionalProperties"] is False
    # Thinking is left at the model's default, and no fallback model is named.
    assert "thinking" not in body
    assert "fallbacks" not in body


def test_a_finished_reply_is_parsed_and_its_usage_kept(model, wire):
    # A model name other than the one requested, so that recording the request
    # instead of the response would show.
    wire.reply(good_narrative().model_dump_json(), model="claude-opus-5-5-served")

    reply = model.complete("s", [Turn("user", "u")])

    assert reply.narrative == good_narrative()
    assert reply.stop_reason == "end_turn"
    assert reply.model == "claude-opus-5-5-served"
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (1234, 56)


def test_the_echo_continues_the_conversation_with_every_block_intact(model, wire):
    wire.reply(good_narrative().model_dump_json())
    wire.reply(good_narrative().model_dump_json())
    first = model.complete("s", [Turn("user", "u")])

    model.complete("s", [Turn("user", "u"), Turn("assistant", first.echo), Turn("user", "fix")])

    assistant = wire.requests[1]["messages"][1]
    assert assistant["role"] == "assistant"
    assert [block["type"] for block in assistant["content"]] == ["thinking", "text"]
    assert assistant["content"][0]["signature"] == "sig"
    assert assistant["content"][1]["text"] == good_narrative().model_dump_json()


@pytest.mark.parametrize(
    ("stop_reason", "text"),
    [("refusal", ""), ("max_tokens", '{"summary": ["cut off mid')],
)
def test_an_unfinished_reply_has_no_narrative_and_does_not_raise(
    model, wire, stop_reason, text
):
    # The text of such a reply is not the JSON that was asked for, so it is
    # never parsed: the stop reason is read first.
    wire.reply(text, stop_reason=stop_reason)

    reply = model.complete("s", [Turn("user", "u")])

    assert reply.narrative is None
    assert reply.stop_reason == stop_reason


@pytest.mark.parametrize("text", ['{"summary": "not a list"}', "not json"])
def test_a_reply_that_ends_normally_but_does_not_fit_the_schema_is_no_answer(
    model, wire, text
):
    wire.reply(text, stop_reason="end_turn")

    reply = model.complete("s", [Turn("user", "u")])

    assert reply.narrative is None
    assert reply.stop_reason == "invalid_reply"


def test_the_model_and_effort_are_the_callers(wire):
    client = anthropic.Anthropic(
        api_key="not-a-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(wire)),
    )
    wire.reply(good_narrative().model_dump_json())

    ClaudeModel(model="claude-sonnet-5-5", effort="low", client=client).complete(
        "s", [Turn("user", "u")]
    )

    assert wire.requests[0]["model"] == "claude-sonnet-5-5"
    assert wire.requests[0]["output_config"]["effort"] == "low"


def test_an_api_error_is_raised_as_a_model_call_error_with_the_sdks_as_its_cause(wire):
    def unauthorised(request):
        return httpx2.Response(
            401,
            json={
                "type": "error",
                "error": {"type": "authentication_error", "message": "invalid x-api-key"},
            },
        )

    client = anthropic.Anthropic(
        api_key="not-a-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(unauthorised)),
    )

    with pytest.raises(ModelCallError, match="AuthenticationError.*invalid x-api-key") as excinfo:
        ClaudeModel(client=client).complete("s", [Turn("user", "u")])

    assert isinstance(excinfo.value.__cause__, anthropic.AuthenticationError)


def test_a_connection_failure_is_a_model_call_error_with_the_sdks_as_its_cause():
    def refused(request):
        raise httpx2.ConnectError("refused")

    client = anthropic.Anthropic(
        api_key="not-a-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(refused)),
    )

    with pytest.raises(ModelCallError, match="APIConnectionError") as excinfo:
        ClaudeModel(client=client).complete("s", [Turn("user", "u")])

    assert isinstance(excinfo.value.__cause__, anthropic.APIConnectionError)


def test_an_sdk_error_that_is_not_an_api_error_is_mapped_too(model, monkeypatch):
    # A broken or expired `ant auth login` profile raises CredentialsError,
    # which is an AnthropicError but not an APIError.
    error = anthropic.CredentialsError("the profile has expired")

    def expired(**kwargs):
        raise error

    monkeypatch.setattr(model._client.messages, "create", expired)

    with pytest.raises(ModelCallError, match="CredentialsError.*has expired") as excinfo:
        model.complete("s", [Turn("user", "u")])

    assert excinfo.value.__cause__ is error


def test_a_client_that_cannot_be_built_is_mapped_where_it_is_first_used(monkeypatch):
    error = anthropic.CredentialsError("no profile")

    def cannot_build(*args, **kwargs):
        raise error

    monkeypatch.setattr(anthropic, "Anthropic", cannot_build)

    model = ClaudeModel()  # constructing the model builds no client

    with pytest.raises(ModelCallError, match="CredentialsError") as excinfo:
        model.complete("s", [Turn("user", "u")])
    assert excinfo.value.__cause__ is error


def test_the_default_client_is_built_once_and_reused(wire, monkeypatch):
    built = []
    real = anthropic.Anthropic

    def build(*args, **kwargs):
        client = real(
            api_key="not-a-key",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(wire)),
        )
        built.append(client)
        return client

    monkeypatch.setattr(anthropic, "Anthropic", build)
    wire.reply(good_narrative().model_dump_json())
    wire.reply(good_narrative().model_dump_json())
    model = ClaudeModel()

    model.complete("s", [Turn("user", "u")])
    model.complete("s", [Turn("user", "u")])

    assert len(built) == 1


def test_missing_credentials_say_how_to_supply_them(model, monkeypatch):
    # What the SDK raises with no key, token or profile: a TypeError, from
    # building the request headers. Reproduced by message, since whether a
    # machine has credentials is not something a test should depend on.
    def no_credentials(**kwargs):
        raise TypeError('"Could not resolve authentication method. Expected one of api_key"')

    monkeypatch.setattr(model._client.messages, "create", no_credentials)

    with pytest.raises(ModelCallError, match="set ANTHROPIC_API_KEY or run `ant auth login`"):
        model.complete("s", [Turn("user", "u")])


def test_any_other_type_error_is_a_bug_and_keeps_its_traceback(model, monkeypatch):
    def broken(**kwargs):
        raise TypeError("unexpected keyword argument")

    monkeypatch.setattr(model._client.messages, "create", broken)

    with pytest.raises(TypeError, match="unexpected keyword"):
        model.complete("s", [Turn("user", "u")])


@pytest.mark.skipif(os.environ.get("CP_LLM_LIVE") == "1", reason="a live run uses the real API")
def test_a_client_built_by_mistake_has_no_credentials_and_no_server_to_reach(monkeypatch):
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "ANTHROPIC_AUTH_TOKEN" not in os.environ
    assert str(anthropic.Anthropic(api_key="not-a-key").base_url).startswith("http://127.0.0.1:9")
