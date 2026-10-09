"""A search-bot-style counter-picking drafter, for the simulator.

Live, the Oct 7 opponent beat us 0-4 with a drafter that counter-picks in the open; its scorer and public reasons are
reproduced in agent/pokemon/counterpick.py (which the draft search also uses for the opponent's replies). This agent
drafts with it so our drafters can be measured against it for free. Team Preview and battle are our own code brain,
so a match against it isolates the draft. Its public reason is the same shape as the Oct 7 opponent's, which makes
simulator logs easy to compare with the live transcripts.
"""

from __future__ import annotations

from agent.agent import PokemonAgent
from agent.pokemon import draft as draft_rules
from agent.pokemon.counterpick import (  # noqa: F401  (re-exported: tests and sims import them from here)
    BEAT_MARGIN, BEAT_POINTS, INTIMIDATE_PROOF, ROLE_POINTS, ROLE_TAGS, SHUTDOWN_TOOLS, SUPPORT_JOBS,
    counter_beats, counter_score, counter_value, tool_works,
)


class CounterPickerAgent(PokemonAgent):
    """The code brain with a greedy counter-picking draft (see agent/pokemon/counterpick.py)."""

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(None, version="counter-picker", params=params)

    def _draft(self, state, obs, log):
        theirs = [c for c in self.memory.opp_cards.values() if c.get("moves")]
        best = None
        for index, action in enumerate(state.legal_actions):
            card = self.memory.pool_cards.get(draft_rules.card_id_of(action))
            if not card:
                continue
            score, reason = counter_score(card, theirs)
            if best is None or score > best[0]:
                best = (score, index, action, reason)
        if best is None:
            pick = state.legal_actions[0]
            reason = "first legal card"
        else:
            _, _, pick, reason = best
        card_id = draft_rules.card_id_of(pick)
        self.memory.record_pick(card_id)
        log.write("draft", state_version=state.state_version, card_id=card_id, reason=reason, counter_picker=True)
        return pick, reason


def create_agent() -> CounterPickerAgent:
    return CounterPickerAgent()
