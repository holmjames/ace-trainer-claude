"""Ablation: code draft + code lineup, but NAIVE battle turns (the starter's smoke move: first move,
first opposing target). Measures how much the turn sheet and candidate ranking are worth.
"""

from __future__ import annotations

from examples import smoke_agent

from agent.agent import PokemonAgent


class NaiveBattleAgent(PokemonAgent):
    def __init__(self) -> None:
        super().__init__(None, version="naive-battle")

    def _turn(self, action, state, obs, log):
        move = smoke_agent.choose_action(state, None)
        log.write("turn", state_version=state.state_version, turn=obs.get("turn"), payload=move, naive=True)
        return move, "Naive turn (ablation)."


def create_agent() -> NaiveBattleAgent:
    return NaiveBattleAgent()
