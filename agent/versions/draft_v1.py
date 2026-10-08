"""The code brain with the ORIGINAL draft scorer and Team Preview (draft_v2 = lineup_v2 = 0), for A/B runs.

    python sim/harness.py --games 20 --p1 code --p2 agent.versions.draft_v1     # current defaults vs the old drafting
    python sim/draft_eval.py match v2l v1 --pairs 500                            # the paired version of the same question

Everything else (battle turns, tuning) is the current code. Kept so the counter-aware drafter and Team Preview
v2 can always be measured against what lost 0-4 on Oct 7.
"""

from __future__ import annotations

from agent.agent import PokemonAgent
from agent.pokemon import tuning


def create_agent() -> PokemonAgent:
    params = tuning.merged({"draft_v2": 0, "lineup_v2": 0})
    return PokemonAgent(None, version="draft-v1", params=params)
