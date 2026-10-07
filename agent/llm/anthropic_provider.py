"""Claude as the agent's model, through the official ``anthropic`` SDK.

Plain-language version: a *provider* is a small object with one job — take a
list of chat messages and a JSON schema, ask the model, and hand back a Python
dict that matches the schema. The game code never talks to the vendor API
directly, so swapping or stacking models never touches game logic.

Two classes live here:

- ``AnthropicProvider`` — one model (e.g. Claude Fable 5.1) with a hard
  per-call timeout. It implements the starter's ``LLMProvider`` protocol
  (``examples/llm/providers.py``), so everything written against that protocol
  works unchanged.
- ``FallbackProvider`` — a chain of providers tried in order. If Fable is slow
  or the API errors, the same question goes to the next model. If every model
  fails, it raises ``ProviderError`` and the caller plays a code-only move.

Secrets: the SDK reads ``ANTHROPIC_API_KEY`` from the environment itself. The
key is never stored on these objects, never in ``repr``, never in logs or
error messages (only HTTP status codes and exception class names are kept).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import anthropic

from examples.llm.providers import LLMProvider, ProviderError

DEFAULT_MODEL = "claude-fable-5-1"
DEFAULT_FALLBACK_MODEL = "claude-sonnet-5-5"
DEFAULT_TIMEOUT_SECONDS = 40.0  # battle decisions have a 300 s clock; leave room for the fallback
DEFAULT_FALLBACK_TIMEOUT_SECONDS = 12.0
DEFAULT_EFFORT = "low"  # thinking is always on for Fable 5.1; effort controls how long it thinks
DEFAULT_MAX_TOKENS = 2000


class AnthropicProvider:
    """One Claude model answering in strict JSON (``output_config.format``)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.workspace_id = workspace_id or None
        if client is None:
            # max_retries=0: we own retries and fallback, so a slow call never
            # silently doubles its wall-clock time inside the SDK.
            # A personal key that spans several workspaces must name the
            # workspace on every request (anthropic-workspace-id header); a
            # key scoped to one workspace needs nothing extra.
            headers = {"anthropic-workspace-id": self.workspace_id} if self.workspace_id else None
            client = anthropic.Anthropic(timeout=timeout, max_retries=0, default_headers=headers)
        self._client = client
        self.last_latency_ms: int | None = None
        self.last_usage: dict | None = None

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model!r}, timeout={self.timeout}, effort={self.effort!r})"

    def complete_structured(self, messages: list[dict], schema_name: str, schema: dict) -> dict:
        system_text = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        chat = [m for m in messages if m.get("role") != "system"]
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": chat,
            "output_config": {
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.effort,
            },
        }
        if system_text:
            # The system prompt is identical every turn, so mark it cacheable:
            # repeated turns pay a fraction of its input price.
            request["system"] = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

        started = time.monotonic()
        try:
            response = self._client.messages.create(**request)
        except anthropic.APITimeoutError:
            raise ProviderError(f"{self.model}: timed out after {self.timeout:.0f}s") from None
        except anthropic.RateLimitError:
            raise ProviderError(f"{self.model}: rate limited (HTTP 429)") from None
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"{self.model}: request failed (HTTP {exc.status_code})") from None
        except anthropic.APIConnectionError:
            raise ProviderError(f"{self.model}: connection failed") from None
        finally:
            self.last_latency_ms = int((time.monotonic() - started) * 1000)

        usage = getattr(response, "usage", None)
        self.last_usage = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        }

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise ProviderError(f"{self.model}: the model declined to answer")
        if stop_reason == "max_tokens":
            raise ProviderError(f"{self.model}: answer cut off at max_tokens={self.max_tokens}")

        text = next((block.text for block in response.content if getattr(block, "type", None) == "text"), None)
        if text is None:
            raise ProviderError(f"{self.model}: response had no text block")
        try:
            data = json.loads(text)
        except ValueError:
            raise ProviderError(f"{self.model}: reply was not valid JSON") from None
        if not isinstance(data, dict):
            raise ProviderError(f"{self.model}: reply was not a JSON object")
        return data


class FallbackProvider:
    """Try each provider in order; the first valid answer wins.

    ``last_model`` records who answered the most recent call, so the decision
    log can show whether Fable, Sonnet, or neither produced a given move.
    """

    def __init__(self, providers: list[LLMProvider], *, log: Callable[[str], None] | None = None) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self._providers = list(providers)
        self._log = log or (lambda line: None)
        self.last_model: str | None = None
        self.last_latency_ms: int | None = None
        self.last_usage: dict | None = None
        self.last_errors: list[str] = []

    @property
    def model(self) -> str:
        return self._providers[0].model

    def __repr__(self) -> str:
        return f"FallbackProvider({[p.model for p in self._providers]!r})"

    def complete_structured(self, messages: list[dict], schema_name: str, schema: dict) -> dict:
        self.last_errors = []
        for provider in self._providers:
            try:
                answer = provider.complete_structured(messages, schema_name, schema)
            except ProviderError as exc:
                self.last_errors.append(str(exc))
                self._log(f"[llm] {schema_name}: {exc}; trying next provider")
                continue
            self.last_model = provider.model
            self.last_latency_ms = getattr(provider, "last_latency_ms", None)
            self.last_usage = getattr(provider, "last_usage", None)
            return answer
        self.last_model = None
        raise ProviderError("every model failed: " + " | ".join(self.last_errors))


def provider_from_env(*, log: Callable[[str], None] | None = None) -> FallbackProvider:
    """Build the Fable -> Sonnet chain from the environment (or ``.env``).

    Variables: ``ANTHROPIC_API_KEY`` (required), ``ANTHROPIC_WORKSPACE_ID``
    (only for a key that spans several workspaces), ``AGENT_MODEL``,
    ``AGENT_FALLBACK_MODEL`` (empty string disables the fallback model),
    ``AGENT_LLM_TIMEOUT``, ``AGENT_FALLBACK_TIMEOUT``, ``AGENT_EFFORT``.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to the starter's .env (never commit that file).")
    primary = os.environ.get("AGENT_MODEL") or DEFAULT_MODEL
    fallback = os.environ.get("AGENT_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
    effort = os.environ.get("AGENT_EFFORT") or DEFAULT_EFFORT
    timeout = float(os.environ.get("AGENT_LLM_TIMEOUT") or DEFAULT_TIMEOUT_SECONDS)
    fallback_timeout = float(os.environ.get("AGENT_FALLBACK_TIMEOUT") or DEFAULT_FALLBACK_TIMEOUT_SECONDS)

    workspace_id = (os.environ.get("ANTHROPIC_WORKSPACE_ID") or "").strip() or None
    providers: list[LLMProvider] = [AnthropicProvider(primary, timeout=timeout, effort=effort, workspace_id=workspace_id)]
    if fallback and fallback != primary:
        providers.append(AnthropicProvider(fallback, timeout=fallback_timeout, effort=effort, workspace_id=workspace_id))
    return FallbackProvider(providers, log=log)
