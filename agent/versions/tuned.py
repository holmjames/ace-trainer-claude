"""Code-only agent with tuning overrides from the environment, for self-play sweeps.

    AGENT_TUNE='{"protect_bonus_guaranteed": 70}' python sim/harness.py --p1 agent.versions.tuned --p2 code

See agent/pokemon/tuning.py for the parameter names and defaults.
"""

from __future__ import annotations

from agent.agent import PokemonAgent
from agent.pokemon import tuning


def create_agent() -> PokemonAgent:
    params = tuning.from_env()
    return PokemonAgent(None, version=tuning.label(params), params=params)
