"""Tests for the opponent-aware draft scorer (agent/pokemon/draft.py). Uses the vendored
Showdown tables in data/ (built by scripts/build_dex.py)."""

from __future__ import annotations

import time

import pytest

from altruagent.models import LegalAction

from agent.pokemon import data, draft
from agent.pokemon.memory import MatchMemory, species_key

pytestmark = pytest.mark.skipif(not (data.DATA_DIR / "pokedex.json").exists(), reason="run scripts/build_dex.py first")


def card(species, moves, *, item="Leftovers", ability="Pressure", nature="Serious", evs=None, card_id=None):
    return {"card_id": card_id or f"vgc-{data.to_id(species)}", "species": species, "item": item, "ability": ability,
            "nature": nature, "evs": evs or {}, "moves": moves}


RILLABOOM = card("Rillaboom", ["Wood Hammer", "Grassy Glide", "Fake Out", "U-turn"], ability="Grassy Surge", nature="Adamant")
INCINEROAR = card("Incineroar", ["Flare Blitz", "Knock Off", "Fake Out", "Parting Shot"], ability="Intimidate", nature="Careful")
GARCHOMP = card("Garchomp", ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"], ability="Rough Skin", nature="Jolly", evs={"spe": 252, "atk": 252})
GYARADOS = card("Gyarados", ["Waterfall", "Tera Blast", "Dragon Dance", "Protect"], ability="Intimidate", nature="Jolly")
URSHIFU = card("Urshifu-Rapid-Strike", ["Surging Strikes", "Close Combat", "Aqua Jet", "Detect"], ability="Unseen Fist", nature="Jolly")
AMOONGUSS = card("Amoonguss", ["Spore", "Rage Powder", "Pollen Puff", "Protect"], ability="Regenerator", nature="Calm")
FLUTTER = card("Flutter Mane", ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Protect"], ability="Protosynthesis", nature="Timid", evs={"spa": 252, "spe": 252})


def memory_with(*, mine=(), theirs=(), pool=()):
    memory = MatchMemory()
    memory.my_seat_key = "me"
    for c in (*mine, *theirs, *pool):
        memory.pool_cards[c["card_id"]] = c
    for c in mine:
        memory.my_cards[species_key(c["species"])] = c
    for c in theirs:
        memory.opp_cards[species_key(c["species"])] = c
    return memory


def legal(*cards):
    return [LegalAction(f"draft_pick:{c['card_id']}", f"Draft {c['species']}", {}) for c in cards]


def test_profile_reads_real_data_and_level_50_stats():
    p = draft.profile(GARCHOMP)
    assert p.types == ["dragon", "ground"]
    assert p.bst == 600
    assert p.stats["spe"] == 169  # base 102, 252 EVs, Jolly
    assert p.spread and p.protect and not p.priority
    assert [m["id"] for m in p.attacks] == ["earthquake", "dragonclaw", "rockslide"]


def test_hit_quality_rewards_super_effective_stab():
    rilla, gyara, urshifu, incin = draft.profile(RILLABOOM), draft.profile(GYARADOS), draft.profile(URSHIFU), draft.profile(INCINEROAR)
    assert draft.hit_quality(rilla, urshifu) == 3.0  # Wood Hammer: grass STAB (1.5) x2 into fighting/water
    assert draft.hit_quality(rilla, gyara) == 1.5  # water 2x, flying 0.5x -> neutral; STAB only
    assert draft.hit_quality(incin, gyara) < 1.0  # Flare Blitz is resisted; best is a neutral Knock Off (65 BP STAB)
    assert draft.hit_quality(draft.profile(AMOONGUSS), gyara) > 0  # Pollen Puff still hits


def test_defense_vs_opponents_actual_moves():
    rilla, incin = draft.profile(RILLABOOM), draft.profile(INCINEROAR)
    water_team = [draft.profile(GYARADOS), draft.profile(URSHIFU)]
    assert draft.defense_vs(rilla, water_team) > draft.defense_vs(incin, water_team)


def test_team_fit_helpers():
    team = [draft.profile(INCINEROAR)]
    assert "grass" in draft.new_offensive_types(draft.profile(RILLABOOM), team)
    assert draft.new_offensive_types(draft.profile(INCINEROAR), team) == []
    # Incineroar is weak to water/ground/fighting/rock; Garchomp (ice/dragon/fairy) shares none.
    assert draft.shared_weaknesses(draft.profile(GARCHOMP), team) == 0
    assert draft.shared_weaknesses(draft.profile(card("Arcanine", ["Flare Blitz"])), team) == 1


def test_against_a_water_roster_the_grass_attacker_is_picked():
    memory = memory_with(mine=(FLUTTER,), theirs=(GYARADOS, URSHIFU), pool=(RILLABOOM, INCINEROAR, GARCHOMP))
    pick = draft.choose_pick(legal(INCINEROAR, GARCHOMP, RILLABOOM), memory)
    assert pick.species == "Rillaboom", pick.notes


def test_first_pick_prefers_raw_quality_and_speed():
    memory = memory_with(pool=(AMOONGUSS, GARCHOMP, INCINEROAR))
    pick = draft.choose_pick(legal(AMOONGUSS, GARCHOMP, INCINEROAR), memory)
    assert pick.species == "Garchomp"


def test_support_is_valued_once_the_team_has_attackers():
    memory = memory_with(mine=(GARCHOMP, FLUTTER, URSHIFU), pool=(AMOONGUSS,))
    with_team, _ = draft.score_card(AMOONGUSS, memory)
    without_team, _ = draft.score_card(AMOONGUSS, memory_with(pool=(AMOONGUSS,)))
    assert with_team > without_team + 10


def test_unknown_species_still_scores_without_crashing():
    mystery = card("Totally Made Up", ["Tackle", "Unknown Move"])
    score, notes = draft.score_card(mystery, memory_with(pool=(mystery,)))
    assert isinstance(score, float)


def test_draft_is_deterministic_and_fast():
    pool = [card(s, ["Protect", "Tackle"], card_id=f"c{i}") for i, s in enumerate(
        ["Garchomp", "Gyarados", "Rillaboom", "Incineroar", "Amoonguss", "Flutter Mane", "Urshifu-Rapid-Strike",
         "Arcanine", "Pelipper", "Kingambit", "Gholdengo", "Dragonite", "Tornadus", "Landorus-Therian",
         "Whimsicott", "Indeedee-F", "Farigiraf", "Ting-Lu"])]
    memory = memory_with(mine=(FLUTTER,), theirs=(GYARADOS,), pool=pool)
    started = time.perf_counter()
    first = draft.choose_pick(legal(*pool), memory)
    second = draft.choose_pick(legal(*pool), memory)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert first.card_id == second.card_id
    assert elapsed_ms < 200, elapsed_ms
