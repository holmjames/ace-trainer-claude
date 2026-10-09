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

import concurrent.futures
import json
import os
import re
import time
from typing import Any, Callable

import anthropic

from examples.llm.providers import LLMProvider, ProviderError

DEFAULT_MODEL = "claude-opus-5-5"  # chosen Oct 6 by self-play: not worse than Fable 5.1 over 40 games, half the p95 latency; Fable stays one .env line away
DEFAULT_FALLBACK_MODEL = "claude-sonnet-5-5"
DEFAULT_TIMEOUT_SECONDS = 18.0  # the platform auto-moves at 50 s per battle decision (420 s bank); p95 Opus latency is ~8 s at low effort. The agent passes a per-call deadline from the live clock; this is the ceiling.
DEFAULT_FALLBACK_TIMEOUT_SECONDS = 8.0  # whole chain worst case: 18+2 + 8+2 = 30 s
DEFAULT_EFFORT = "low"  # thinking is always on for Fable 5.1; effort controls how long it thinks
DEFAULT_MAX_TOKENS = 4000
HARD_TIMEOUT_GRACE_SECONDS = 2.0  # wall-clock guard on top of the SDK's own timeout  # thinking counts toward output; 2000 was cut off once in self-play
MIN_CALL_SECONDS = 3.0  # a model with less than this left of the chain's deadline is not asked
FALLBACK_RESERVE_SECONDS = 6.0  # what the first model leaves the next one when the chain has a deadline


_SECRET = re.compile(r"(sk-ant-|eak_live_|wrkspc_)[A-Za-z0-9_\-]+")


def _api_error_detail(exc: Any) -> str:
    """The API's own error type and message, trimmed and with anything key-shaped removed."""
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return ""
    text = f"{err.get('type', '')}: {str(err.get('message', ''))[:240]}"
    return " [" + _SECRET.sub("[REDACTED]", text) + "]"


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
        self.last_effort: str | None = None
        self.last_timeout: float | None = None

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model!r}, timeout={self.timeout}, effort={self.effort!r})"

    def _call_with_deadline(self, request: dict, limit: float):
        """Run the SDK call in a worker thread and give up at ``limit`` + grace.

        The SDK has its own timeout, but a socket that stops answering mid-read has been seen
        to hang far past it. The game clock does not care why. If the worker has not returned
        by the deadline we raise APITimeoutError and the caller falls back; the worker thread
        is left to finish or die on its own.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="anthropic-call")
        future = executor.submit(self._client.messages.create, **request)
        try:
            return future.result(timeout=limit + HARD_TIMEOUT_GRACE_SECONDS)
        except concurrent.futures.TimeoutError:
            raise anthropic.APITimeoutError(request=None) from None  # type: ignore[arg-type]
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def complete_structured(self, messages: list[dict], schema_name: str, schema: dict, *,
                            timeout: float | None = None, effort: str | None = None) -> dict:
        """One structured call. ``timeout`` (seconds) bounds this request instead of the instance default: the game
        clock is per decision, so the agent passes what it can still afford. ``effort`` overrides the instance's
        effort for this call (Team Preview and a battle turn may differ)."""
        system_text = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        chat = [m for m in messages if m.get("role") != "system"]
        limit = float(timeout) if timeout is not None else float(self.timeout)
        self.last_effort = effort or self.effort
        self.last_timeout = limit
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": chat,
            "output_config": {
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.last_effort,
            },
        }
        if timeout is not None:
            request["timeout"] = limit  # the SDK's per-request timeout (the instance default otherwise)
        if system_text:
            # The system prompt is identical every turn, so mark it cacheable:
            # repeated turns pay a fraction of its input price.
            request["system"] = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

        started = time.monotonic()
        try:
            response = self._call_with_deadline(request, limit)
        except anthropic.APITimeoutError:
            raise ProviderError(f"{self.model}: timed out after {limit:.0f}s") from None
        except anthropic.RateLimitError:
            raise ProviderError(f"{self.model}: rate limited (HTTP 429)") from None
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"{self.model}: request failed (HTTP {exc.status_code}){_api_error_detail(exc)}") from None
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
        self.last_attempts: list[dict] = []  # per model tried on the last call: model, effort, timeout, latency, outcome
        self._credit_warned = False

    @property
    def model(self) -> str:
        return self._providers[0].model

    @property
    def effort(self) -> str | None:
        return getattr(self._providers[0], "effort", None)

    def __repr__(self) -> str:
        return f"FallbackProvider({[p.model for p in self._providers]!r})"

    def complete_structured(self, messages: list[dict], schema_name: str, schema: dict, *,
                            deadline: float | None = None, effort: str | None = None) -> dict:
        """Try each provider in order within ``deadline`` seconds of wall clock for the whole chain: the first model
        gets the larger share but leaves the next one a usable slice (``FALLBACK_RESERVE_SECONDS``), a model with
        less than ``MIN_CALL_SECONDS`` left is not tried. Without a deadline each model keeps its own timeout."""
        self.last_errors = []
        self.last_attempts = []
        started = time.monotonic()
        for index, provider in enumerate(self._providers):
            timeout = None
            if deadline is not None:
                remaining = deadline - (time.monotonic() - started)
                if remaining < MIN_CALL_SECONDS:
                    self.last_errors.append(f"{provider.model}: not tried ({remaining:.1f}s left of the {deadline:.1f}s deadline)")
                    self.last_attempts.append({"model": provider.model, "outcome": "skipped", "left_s": round(remaining, 1)})
                    continue
                reserve = 0.0
                if index + 1 < len(self._providers):
                    reserve = min(float(getattr(self._providers[index + 1], "timeout", FALLBACK_RESERVE_SECONDS)), FALLBACK_RESERVE_SECONDS)
                timeout = min(float(getattr(provider, "timeout", remaining)), max(remaining - reserve, remaining * 0.7))
            call_started = time.monotonic()
            try:
                answer = provider.complete_structured(messages, schema_name, schema, timeout=timeout, effort=effort)
            except ProviderError as exc:
                self.last_errors.append(str(exc))
                self.last_attempts.append({"model": provider.model, "effort": effort or getattr(provider, "effort", None),
                                           "timeout_s": round(timeout, 1) if timeout is not None else getattr(provider, "timeout", None),
                                           "latency_ms": int((time.monotonic() - call_started) * 1000), "outcome": "error", "error": str(exc)})
                self._log(f"[llm] {schema_name}: {exc}; trying next provider")
                continue
            self.last_model = provider.model
            self.last_latency_ms = getattr(provider, "last_latency_ms", None)
            self.last_usage = getattr(provider, "last_usage", None)
            self.last_attempts.append({"model": provider.model, "effort": effort or getattr(provider, "effort", None),
                                       "timeout_s": round(timeout, 1) if timeout is not None else getattr(provider, "timeout", None),
                                       "latency_ms": self.last_latency_ms, "outcome": "ok", "usage": self.last_usage})
            return answer
        self.last_model = None
        if any("credit balance" in e.lower() for e in self.last_errors) and not self._credit_warned:
            self._credit_warned = True
            self._log("!!! ANTHROPIC CREDITS EXHAUSTED: every model call is failing with 'credit balance is too low'. "
                      "The agent is now playing CODE-ONLY moves. Add credits in the Anthropic Console (Plans & Billing) immediately. !!!")
        raise ProviderError("every model failed: " + " | ".join(self.last_errors))


PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
PROBE_MESSAGES = [{"role": "system", "content": "You answer a pre-flight connectivity check."},
                  {"role": "user", "content": 'Reply with exactly {"ok": true}.'}]


def probe(provider: Any) -> dict:
    """One tiny real call through the configured chain: the pre-flight proof that a model answers.

    Returns ``{"model": who answered, "latency_ms": ..., "errors": [what earlier models in the chain said]}``.
    Raises ``ProviderError`` when every model fails or the answer is not ``{"ok": true}``. Never includes a key:
    provider errors carry status codes and class names only."""
    started = time.monotonic()
    answer = provider.complete_structured(PROBE_MESSAGES, "preflight_probe", PROBE_SCHEMA)
    latency = int((time.monotonic() - started) * 1000)
    if not isinstance(answer, dict) or answer.get("ok") is not True:
        raise ProviderError(f"probe answer was {str(answer)[:80]!r}, expected {{'ok': True}}")
    return {
        "model": getattr(provider, "last_model", None) or getattr(provider, "model", None),
        "latency_ms": getattr(provider, "last_latency_ms", None) or latency,
        "errors": list(getattr(provider, "last_errors", None) or []),
    }


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
