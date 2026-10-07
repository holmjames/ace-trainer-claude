"""Baseline: the same agent with NO model calls.

Draft, lineup and every battle turn come from the computed fallbacks alone (best
scored card, best scored lineup, top candidate turn). Putting this in one seat
and the full agent in the other measures exactly what Claude adds on top of the
code. It costs nothing to run.
"""

from __future__ import annotations

from agent.agent import PokemonAgent


def create_agent() -> PokemonAgent:
    return PokemonAgent(None, version="code-only")
