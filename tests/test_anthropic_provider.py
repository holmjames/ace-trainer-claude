"""Tests for agent/llm/anthropic_provider.py with a fake SDK client (no network, no key)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from examples.llm.providers import ProviderError

from agent.llm.anthropic_provider import AnthropicProvider, FallbackProvider, provider_from_env

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}
MESSAGES = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "{}"}]


def _response(text="{\"x\": 1}", stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class FakeClient:
    def __init__(self, *results):
        self.results = list(results)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request):
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _timeout():
    return anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def _status(code: int):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(code, request=request, json={"error": {"message": "secret-ish body"}})
    return anthropic.APIStatusError("boom", response=response, body=None)


def test_success_sends_schema_effort_and_cached_system_prompt():
    client = FakeClient(_response())
    provider = AnthropicProvider("claude-fable-5-1", client=client, timeout=40)

    answer = provider.complete_structured(MESSAGES, "x", SCHEMA)

    assert answer == {"x": 1}
    request = client.requests[0]
    assert request["model"] == "claude-fable-5-1"
    assert request["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"}
    assert request["system"][0]["text"] == "be terse"
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["messages"] == [{"role": "user", "content": "{}"}]
    assert "thinking" not in request  # Fable 5.1 thinks by default; sending a thinking config is a 400
    assert provider.last_usage["input_tokens"] == 10 and provider.last_latency_ms is not None


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_refusal_or_truncation_is_a_provider_error(stop_reason):
    provider = AnthropicProvider(client=FakeClient(_response(stop_reason=stop_reason)))
    with pytest.raises(ProviderError):
        provider.complete_structured(MESSAGES, "x", SCHEMA)


def test_non_object_json_is_a_provider_error():
    provider = AnthropicProvider(client=FakeClient(_response(text="[1,2]")))
    with pytest.raises(ProviderError, match="JSON object"):
        provider.complete_structured(MESSAGES, "x", SCHEMA)


def test_timeout_and_http_errors_become_provider_errors_without_the_body():
    provider = AnthropicProvider(client=FakeClient(_timeout(), _status(500)))
    with pytest.raises(ProviderError, match="timed out"):
        provider.complete_structured(MESSAGES, "x", SCHEMA)
    with pytest.raises(ProviderError) as info:
        provider.complete_structured(MESSAGES, "x", SCHEMA)
    assert "HTTP 500" in str(info.value) and "secret-ish" not in str(info.value)


def test_fallback_provider_moves_to_the_next_model_and_records_who_answered():
    slow = AnthropicProvider("fable", client=FakeClient(_timeout()))
    quick = AnthropicProvider("sonnet", client=FakeClient(_response(text=json.dumps({"x": 2}))))
    chain = FallbackProvider([slow, quick])

    assert chain.complete_structured(MESSAGES, "x", SCHEMA) == {"x": 2}
    assert chain.last_model == "sonnet"
    assert chain.last_errors and "timed out" in chain.last_errors[0]


def test_fallback_provider_raises_when_every_model_fails():
    chain = FallbackProvider([AnthropicProvider("a", client=FakeClient(_timeout())),
                              AnthropicProvider("b", client=FakeClient(_status(429)))])
    with pytest.raises(ProviderError, match="every model failed"):
        chain.complete_structured(MESSAGES, "x", SCHEMA)
    assert chain.last_model is None


def test_provider_from_env_requires_the_key_and_builds_the_chain(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        provider_from_env()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setenv("AGENT_MODEL", "claude-fable-5-1")
    monkeypatch.setenv("AGENT_FALLBACK_MODEL", "claude-sonnet-5-5")
    chain = provider_from_env()
    assert [p.model for p in chain._providers] == ["claude-fable-5-1", "claude-sonnet-5-5"]
    assert "sk-ant" not in repr(chain) and "sk-ant" not in repr(chain._providers[0])

    monkeypatch.setenv("AGENT_FALLBACK_MODEL", "")
    assert len(provider_from_env()._providers) == 1


def test_workspace_id_becomes_a_default_header_on_the_real_client(monkeypatch):
    provider = AnthropicProvider("claude-fable-5-1", workspace_id="wrkspc_test123")
    assert provider._client.default_headers["anthropic-workspace-id"] == "wrkspc_test123"
    assert "anthropic-workspace-id" not in AnthropicProvider("claude-fable-5-1").__dict__.get("_client").default_headers

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", " wrkspc_env456 ")
    chain = provider_from_env()
    assert all(p.workspace_id == "wrkspc_env456" for p in chain._providers)
