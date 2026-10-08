"""Sparring partner: the starter's own LLM example agent (examples/llm_agent.py), driven by a Claude model.

Most other entrants will run that example more or less unchanged, with a general-purpose model and no damage
math. Playing it in the simulator is the closest thing to a real opponent that needs no tournament server:

    python sim/harness.py --games 10 --p1 code  --p2 agent.versions.sparring --seed 101   # our code brain vs it (free on our side)
    python sim/harness.py --games 10 --p1 fable --p2 agent.versions.sparring --seed 102   # the full Opus agent vs it

``SPARRING_MODEL`` picks the model (default Sonnet 5.5, roughly $0.30 a game); ``ANTHROPIC_API_KEY`` and, for a
personal key, ``ANTHROPIC_WORKSPACE_ID`` come from .env. Its decisions are logged by the harness like any player's,
so losses can be mined with scripts/loss_to_scenario.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from agent.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from examples.llm_agent import LLMAgent  # noqa: E402

DEFAULT_SPARRING_MODEL = "claude-sonnet-5-5"


def _quiet(message: str) -> None:
    if os.environ.get("SPARRING_VERBOSE"):
        print(f"[sparring] {message}", file=sys.stderr, flush=True)


def create_agent() -> LLMAgent:
    provider = AnthropicProvider(
        os.environ.get("SPARRING_MODEL") or DEFAULT_SPARRING_MODEL,
        timeout=float(os.environ.get("SPARRING_TIMEOUT") or 30.0),
        workspace_id=os.environ.get("ANTHROPIC_WORKSPACE_ID"),
    )
    agent = LLMAgent(provider, log=_quiet)
    agent._version = f"sparring:{provider.model}"  # shown in the harness tally
    return agent
