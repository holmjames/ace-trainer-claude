"""The secret tie-break seed registry answers only for the match that registered it.

The live runtime plays every match in its own spawned worker. A long-lived process (the simulator's arms,
scripts/live_scenarios.py, the replay tools, this suite) plays many matches one after another; there draft._SEEDS,
keyed by id(memory.my_cards), handed a later match whose card dict took a freed dict's address the earlier match's
secret seed (Oct 10 state audit). The test below fails on 81c4591.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.pokemon import draft
from agent.pokemon.memory import MatchMemory


# -- the secret tie-break seed --------------------------------------------------------------------------------------


def test_a_recycled_card_dict_address_never_hands_a_new_match_the_old_matchs_seed(monkeypatch):
    """CPython gives a freed dict's address to the next dict it creates (4 and 6 matches of 30 in a tight loop in one
    process took an earlier match's seed, Oct 10 audit). Modelled here with an id() that puts every dict at one
    address: the registry must answer only for the very dict it registered."""
    monkeypatch.setattr(draft, "id", lambda obj: 0xADD, raising=False)
    draws = iter([111, 222])
    monkeypatch.setattr(draft, "secrets", SimpleNamespace(randbits=lambda bits: next(draws)))
    old = MatchMemory()
    assert draft.tiebreak_seed(old.my_cards, old) == 111
    new = MatchMemory()  # the next match: its card dict lands where the old one was
    assert draft.known_seed(new.my_cards) is None  # its Team Preview before any draft decision: no seed, as in a fresh process
    assert draft.tiebreak_seed(new.my_cards, new) == 222  # its first draft decision draws its own
    assert new.tiebreak_seed == 222 and draft.known_seed(new.my_cards) == 222
    assert draft.known_seed(old.my_cards) is None and old.tiebreak_seed == 111  # each match keeps its own on its memory
