"""Ablation: the code brain with a RANDOM draft. Measures how much the draft scorer is worth.

Everything else (lineup shortlist, turn sheet, candidate ranking) is unchanged and code-only.
"""

from __future__ import annotations

import random

from agent.agent import PokemonAgent


class RandomDraftAgent(PokemonAgent):
    def __init__(self) -> None:
        super().__init__(None, version="random-draft")
        self._rng = random.Random()

    def _draft(self, state, obs, log):
        pick = self._rng.choice(state.legal_actions)
        log.write("draft", state_version=state.state_version, card_id=pick.action_id, random=True)
        return pick, "Random draft pick (ablation)."


def create_agent() -> RandomDraftAgent:
    return RandomDraftAgent()
