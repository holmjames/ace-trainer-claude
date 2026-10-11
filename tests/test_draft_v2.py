"""Tests for the counter-aware drafter (draft_v2 = 1 in agent/pokemon/draft.py) on real catalog cards."""

from __future__ import annotations

import json
import time

import pytest

from altruagent.models import LegalAction

from agent.pokemon import data, draft, matchup, tuning
from agent.pokemon.memory import MatchMemory, species_key

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")
V2 = tuning.merged({"draft_v2": 1})

# The first live loss (match 0a2b7c2a): we drafted first from this pool.
POOL_0A2B = ["Amoonguss", "Archaludon", "Baxcalibur", "Dragonite", "Gholdengo", "Great Tusk", "Gyarados", "Hatterene",
             "Incineroar", "Indeedee-F", "Landorus-Therian", "Lucario", "Sableye", "Thundurus", "Ting-Lu", "Tornadus",
             "Tyranitar", "Wo-Chien"]


def memory_for(pool: list[str], mine: list[str] = (), theirs: list[str] = ()) -> MatchMemory:
    memory = MatchMemory()
    memory.my_seat_key = "me"
    for species in pool:
        card = CATALOG[species]
        memory.pool_cards[card["card_id"]] = card
    for species in mine:
        memory.my_cards[species_key(species)] = CATALOG[species]
    for species in theirs:
        memory.opp_cards[species_key(species)] = CATALOG[species]
    return memory


def offered(memory: MatchMemory) -> list[LegalAction]:
    drafted = {c["card_id"] for c in (*memory.my_cards.values(), *memory.opp_cards.values())}
    held = {data.to_id(c["item"]) for c in memory.my_cards.values()}
    return [LegalAction(f"draft_pick:{c['card_id']}", f"Draft {c['species']}", {}) for c in memory.pool_cards.values()
            if c["card_id"] not in drafted and data.to_id(c["item"]) not in held]


def test_pick_order_follows_the_snake():
    assert draft.pick_order(0, True) == ["us", "them", "them", "us", "us", "them", "them", "us", "us", "them", "them", "us"]
    assert draft.pick_order(1, False) == ["us", "us", "them", "them", "us", "us", "them", "them", "us", "us", "them"]
    assert draft.pick_order(11, True) == ["us"]


def test_v2_pick_is_legal_deterministic_and_fast():
    memory = memory_for(POOL_0A2B)
    legal = offered(memory)
    started = time.perf_counter()
    first = draft.choose_pick(legal, memory, V2)
    elapsed = time.perf_counter() - started
    second = draft.choose_pick(legal, memory, V2)
    assert first.card_id == second.card_id
    assert first.action in legal
    assert elapsed < 2.0, elapsed  # the clock is 15 s; aim well under 2 s
    assert any(n.startswith("v2:") for n in first.notes)


def test_v2_does_not_pass_the_card_that_beat_us():
    # Live (match 0a2b7c2a) our greedy scorer passed Gholdengo at picks 4 and 5 and then had no answer to it.
    # Same position: we hold Landorus-Therian, they hold Incineroar and Dragonite. Gholdengo must be ours by pick 5.
    memory = memory_for(POOL_0A2B, mine=["Landorus-Therian"], theirs=["Incineroar", "Dragonite"])
    pick4 = draft.choose_pick(offered(memory), memory, V2)
    memory.my_cards[species_key(pick4.species)] = CATALOG[pick4.species]
    pick5 = draft.choose_pick(offered(memory), memory, V2)
    assert "Gholdengo" in (pick4.species, pick5.species), (pick4.notes, pick5.notes)
    old = draft.choose_pick(offered(memory_for(POOL_0A2B, mine=["Landorus-Therian"], theirs=["Incineroar", "Dragonite"])),
                            memory_for(POOL_0A2B, mine=["Landorus-Therian"], theirs=["Incineroar", "Dragonite"]),
                            tuning.merged({"draft_v2": 0}))
    assert old.species == "Baxcalibur"  # what the old scorer did live


def test_v2_respects_the_item_clause_in_its_lookahead():
    # Two Focus Sash cards: holding one, the other is never offered to us, and the search must not plan to take it.
    pool = ["Garchomp", "Whimsicott", "Chien-Pao", "Incineroar", "Rillaboom", "Amoonguss", "Gholdengo", "Kingambit",
            "Flutter Mane", "Iron Hands", "Archaludon", "Dragonite", "Tornadus", "Sylveon", "Gyarados", "Ursaluna",
            "Porygon2", "Pelipper"]
    memory = memory_for(pool, mine=["Garchomp"], theirs=["Incineroar", "Rillaboom"])
    legal = offered(memory)
    assert not any("whimsicott" in a.action_id or "chien-pao" in a.action_id for a in legal)
    pick = draft.choose_pick(legal, memory, V2)
    assert pick.action in legal


def test_v2_last_pick_is_the_best_finisher():
    pool = POOL_0A2B
    mine = ["Landorus-Therian", "Baxcalibur", "Tornadus", "Lucario", "Great Tusk"]
    theirs = ["Incineroar", "Dragonite", "Archaludon", "Gholdengo", "Gyarados", "Indeedee-F"]
    memory = memory_for(pool, mine=mine, theirs=theirs)
    legal = offered(memory)
    pick = draft.choose_pick(legal, memory, V2)
    board = draft._board_for([CATALOG[s] for s in pool], V2["draft_card_w"], V2["draft_tier_adjust"])
    idx = lambda s: board.index(matchup.unit(CATALOG[s]).key)  # noqa: E731
    us, them = [idx(s) for s in mine], [idx(s) for s in theirs]
    search = draft.DraftSearch(board, four_eval=bool(V2["draft_four_eval"]), their_fours=int(V2["draft_their_fours"]),
                               robust=V2["lineup_v2_robust"], hole_w=V2["draft_hole_w"])
    # The last pick is the card that finishes the best four (the draft's own finished value, Oct 8), tie-break window aside.
    values = {draft.card_id_of(a): search.finished(us + [idx(CATALOG_BY_ID[draft.card_id_of(a)])], them, idx(CATALOG_BY_ID[draft.card_id_of(a)]))[0] for a in legal}
    assert matchup.win_probability(max(values.values())) - matchup.win_probability(values[pick.card_id]) <= V2["draft_tiebreak_eps"]


CATALOG_BY_ID = {c["card_id"]: c["species"] for c in CATALOG.values()}


def test_old_scorer_is_still_available_behind_the_flag():
    memory = memory_for(POOL_0A2B)
    v1 = draft.choose_pick(offered(memory), memory, tuning.merged({"draft_v2": 0}))
    assert not any(n.startswith("v2:") for n in v1.notes)


def test_v2_handles_unknown_cards_and_a_thin_pool():
    mystery = {"card_id": "x-1", "species": "Totally Made Up", "item": "Leftovers", "ability": "Pressure", "nature": "Serious",
               "evs": {}, "moves": ["Tackle", "Protect"]}
    memory = memory_for(["Garchomp", "Incineroar"])
    memory.pool_cards[mystery["card_id"]] = mystery
    legal = offered(memory)
    pick = draft.choose_pick(legal, memory, V2)
    assert pick.action in legal


def test_reply_depth_two_branches_their_snake_double_and_stays_fast():
    # Timed on the drafter this budget was set for: the per-card table off. Deploy #5 ships it (draft_card_adjust_w and
    # draft_card_adjust_them_w 0.5), and where Archaludon is offered, as here, its floor ranks the pick a second time
    # without it: ~1.2 s -> ~2.4 s at this unshipped depth on an idle machine. The shipped depth 1 with the table is
    # timed under the same 2 s in test_v2_pick_is_legal_deterministic_and_fast (~0.5 s).
    memory = memory_for(POOL_0A2B)
    started = time.perf_counter()
    pick = draft.choose_pick(offered(memory), memory, tuning.merged({"draft_v2": 1, "draft_reply_depth": 2,
                                                                     "draft_card_adjust_w": 0.0, "draft_card_adjust_them_w": 0.0}))
    assert time.perf_counter() - started < 2.0
    assert pick.action in offered(memory)


def test_frozen_v1_version_keeps_the_old_drafting():
    from agent.versions import draft_v1

    agent = draft_v1.create_agent()
    assert agent.params["draft_v2"] == 0 and agent.params["lineup_v2"] == 0


def test_a_failing_search_falls_back_to_the_old_scorer(monkeypatch):
    memory = memory_for(POOL_0A2B)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(draft, "choose_pick_v2", broken)
    pick = draft.choose_pick(offered(memory), memory, V2)
    assert pick.action in offered(memory)
    assert any("old scorer used" in n for n in pick.notes)
