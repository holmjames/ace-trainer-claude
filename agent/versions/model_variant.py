"""The full agent with a different judging model, for head-to-head model comparisons.

    AGENT_VARIANT_MODEL=claude-opus-5-5   python sim/harness.py --p1 agent.versions.model_variant --p2 fable
    AGENT_VARIANT_MODEL=claude-sonnet-5-5 python sim/harness.py --p1 agent.versions.model_variant --p2 fable

Everything else (code brain, prompts, fallback to Sonnet then code) is identical to agent/agent.py.
"""

from __future__ import annotations

import os

from agent.agent import PokemonAgent
from agent.llm.anthropic_provider import DEFAULT_FALLBACK_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS, AnthropicProvider, FallbackProvider


def create_agent() -> PokemonAgent:
    model = os.environ.get("AGENT_VARIANT_MODEL") or "claude-opus-5-5"
    workspace = (os.environ.get("ANTHROPIC_WORKSPACE_ID") or "").strip() or None
    effort = os.environ.get("AGENT_EFFORT") or "low"
    chain = [AnthropicProvider(model, timeout=DEFAULT_TIMEOUT_SECONDS, effort=effort, workspace_id=workspace)]
    if model != "claude-sonnet-5-5":
        chain.append(AnthropicProvider("claude-sonnet-5-5", timeout=DEFAULT_FALLBACK_TIMEOUT_SECONDS, effort=effort, workspace_id=workspace))
    return PokemonAgent(FallbackProvider(chain), version=f"model:{model}")
