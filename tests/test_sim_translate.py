"""Tests for sim/translate.py (pure functions) and, when Node + the engine are installed, one real
local game through sim/harness.py."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from sim import translate

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "sim" / "node_modules" / "pokemon-showdown"


def test_to_choice_covers_move_switch_pass_and_no_target():
    payload = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "fakeout", "target": 2},
               "slot_1": {"type": "switch", "species": "Amoonguss"}}
    assert translate.to_choice(payload) == "move fakeout 2, switch Amoonguss"
    assert translate.to_choice({"slot_0": {"type": "move", "move_id": "tailwind", "target": 0}, "slot_1": {"type": "pass"}}) == "move tailwind, pass"
    assert translate.to_choice({"slot_0": {"type": "move", "move_id": "helpinghand", "target": -2}, "slot_1": {"type": "move", "move_id": "protect"}}) == "move helpinghand -2, move protect"


def test_team_text_and_preview_order():
    cards = [{"species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "nature": "Careful", "evs": {"hp": 252, "spd": 252}, "moves": ["Fake Out", "Knock Off", "Flare Blitz", "Parting Shot"]},
             {"species": "Rillaboom", "item": "Assault Vest", "ability": "Grassy Surge", "nature": "Adamant", "evs": {"atk": 252}, "moves": ["Wood Hammer", "Fake Out", "U-turn", "Grassy Glide"]},
             {"species": "Garchomp", "item": "Life Orb", "ability": "Rough Skin", "nature": "Jolly", "evs": {}, "moves": ["Earthquake"]},
             {"species": "Amoonguss", "item": "Rocky Helmet", "ability": "Regenerator", "nature": "Calm", "evs": {}, "moves": ["Spore"]},
             {"species": "Whimsicott", "item": "Focus Sash", "ability": "Prankster", "nature": "Timid", "evs": {}, "moves": ["Tailwind"]},
             {"species": "Flutter Mane", "item": "Choice Specs", "ability": "Protosynthesis", "nature": "Timid", "evs": {}, "moves": ["Moonblast"]}]
    text = translate.team_text(cards[:2])
    assert "Incineroar @ Safety Goggles" in text and "EVs: 252 HP / 252 SpD" in text and "- Parting Shot" in text
    choice, ordered = translate.preview_choice(["garchomp", "amoonguss", "incineroar", "whimsicott"], ["whimsicott", "garchomp"], cards)
    assert choice == "team 1234"
    assert [c["species"] for c in ordered] == ["Whimsicott", "Garchomp", "Amoonguss", "Incineroar"]  # leads first


def test_draft_state_applies_item_clause_and_turn():
    pool = [{"card_id": "a", "species": "A", "item": "Leftovers"}, {"card_id": "b", "species": "B", "item": "Life Orb"}, {"card_id": "c", "species": "C", "item": "Leftovers"}]
    rosters = {"p1": [pool[0]], "p2": []}
    mine = translate.draft_state("s", me="p1", them="p2", pool=pool[1:], rosters=rosters, picks=[], first_drafter="p1", current_seat="p1", version=1)
    assert [a.action_id for a in mine.legal_actions] == ["draft_pick:b"]  # C's Leftovers clash with our roster
    theirs = translate.draft_state("s", me="p2", them="p1", pool=pool[1:], rosters=rosters, picks=[], first_drafter="p1", current_seat="p1", version=1)
    assert theirs.legal_actions == [] and theirs.is_current_actor is False


def test_battle_state_builds_platform_shaped_template():
    snap = {
        "turn": 3, "ended": False, "winner": None,
        "field": {"weather": "raindance", "terrain": None, "pseudo_weather": ["trickroom"]},
        "sides": {
            "p1": {"name": "A", "side_conditions": ["tailwind"],
                   "request": {"active": [{"moves": [{"id": "fakeout", "target": "normal", "pp": 10}, {"id": "tailwind", "target": "allySide", "pp": 5},
                                                     {"id": "helpinghand", "target": "adjacentAlly", "pp": 20, "disabled": False}]},
                                          {"moves": [{"id": "earthquake", "target": "allAdjacent", "pp": 10}, {"id": "protect", "target": "self", "pp": 10, "disabled": True}]}],
                               "side": {}},
                   "pokemon": [
                       {"species": "incineroar", "name": "Incineroar", "hp": 100, "maxhp": 202, "fainted": False, "active": True, "position": 0, "status": None, "item": "safetygoggles", "ability": "intimidate", "types": ["Fire", "Dark"], "base_stats": {}, "boosts": {"atk": -1}, "moves": ["fakeout"]},
                       {"species": "garchomp", "name": "Garchomp", "hp": 183, "maxhp": 183, "fainted": False, "active": True, "position": 1, "status": None, "item": "lifeorb", "ability": "roughskin", "types": ["Dragon", "Ground"], "base_stats": {}, "boosts": {}, "moves": ["earthquake"]},
                       {"species": "amoonguss", "name": "Amoonguss", "hp": 221, "maxhp": 221, "fainted": False, "active": False, "position": None, "status": None, "item": "rockyhelmet", "ability": "regenerator", "types": ["Grass", "Poison"], "base_stats": {}, "boosts": {}, "moves": []},
                       {"species": "whimsicott", "name": "Whimsicott", "hp": 0, "maxhp": 150, "fainted": True, "active": False, "position": None, "status": "fnt", "item": None, "ability": "prankster", "types": ["Grass", "Fairy"], "base_stats": {}, "boosts": {}, "moves": []},
                   ]},
            "p2": {"name": "B", "side_conditions": [], "request": {"wait": True},
                   "pokemon": [
                       {"species": "gyarados", "name": "Gyarados", "hp": 50, "maxhp": 170, "fainted": False, "active": True, "position": 0, "status": "par", "item": "sitrusberry", "ability": "intimidate", "types": ["Water", "Flying"], "base_stats": {}, "boosts": {}, "moves": []},
                       {"species": "pelipper", "name": "Pelipper", "hp": 0, "maxhp": 150, "fainted": True, "active": True, "position": 1, "status": "fnt", "item": None, "ability": "drizzle", "types": ["Water", "Flying"], "base_stats": {}, "boosts": {}, "moves": []},
                   ]},
        },
    }
    state = translate.battle_state("s", snap, "p1", 7)
    assert state is not None and state.phase == "moving"
    template = state.legal_actions[0].input["action"]
    slot0, slot1 = template["slots"]
    fake = slot0["options"][0]
    assert fake["targets"] == [1] and fake["target_options"][0]["species"] == "Gyarados"  # the fainted Pelipper is not a target
    assert slot0["options"][1]["targets"] == []  # Tailwind: no target
    assert slot0["options"][2]["targets"] == [-2] and slot0["options"][2]["target_options"][0]["side"] == "ally"  # Helping Hand -> slot 1
    assert [o["move_id"] for o in slot1["options"] if o["type"] == "move"] == ["earthquake"]  # disabled Protect dropped
    assert {o["species"] for o in slot1["options"] if o["type"] == "switch"} == {"Amoonguss"}  # fainted Whimsicott not switchable
    obs = state.raw["observation"]
    assert obs["weather"] == "raindance" and obs["fields"] == ["trickroom"] and obs["side_conditions"] == ["tailwind"]
    assert obs["team"]["p1: Incineroar"]["boosts"] == {"atk": -1} and obs["opponent_team"]["p2: Gyarados"]["status"] == "par"
    assert "moves" not in obs["opponent_team"]["p2: Gyarados"]  # opponent sets are not revealed by the observation
    assert translate.battle_state("s", snap, "p2", 1) is None  # waiting side has no decision


def test_force_switch_request_offers_switches_and_pass():
    snap = {"turn": 4, "ended": False, "winner": None, "field": {"weather": None, "terrain": None, "pseudo_weather": []},
            "sides": {"p1": {"name": "A", "side_conditions": [], "request": {"forceSwitch": [True, False], "side": {}},
                             "pokemon": [{"species": "a", "name": "A1", "hp": 0, "maxhp": 1, "fainted": True, "active": True, "position": 0, "types": [], "base_stats": {}, "boosts": {}, "moves": []},
                                         {"species": "b", "name": "B1", "hp": 1, "maxhp": 1, "fainted": False, "active": True, "position": 1, "types": [], "base_stats": {}, "boosts": {}, "moves": []},
                                         {"species": "c", "name": "C1", "hp": 1, "maxhp": 1, "fainted": False, "active": False, "position": None, "types": [], "base_stats": {}, "boosts": {}, "moves": []}]},
                      "p2": {"name": "B", "side_conditions": [], "request": {"wait": True}, "pokemon": []}}}
    template = translate.battle_state("s", snap, "p1", 2).legal_actions[0].input["action"]
    assert template["slots"][0]["options"] == [{"type": "switch", "species": "C1"}] and template["slots"][0]["force_switch"] is True
    assert template["slots"][1]["options"] == [{"type": "pass"}]


@pytest.mark.skipif(not shutil.which("node") or not ENGINE.exists(), reason="needs Node and `npm install` in sim/")
def test_one_full_local_game_runs_end_to_end(tmp_path, monkeypatch):
    from sim import harness

    monkeypatch.setenv("AGENT_LOG_DIR", str(tmp_path))
    bridge = harness.Bridge()
    try:
        rng = random.Random(5)
        players = {"p1": harness.make_player("code", rng)[0], "p2": harness.RandomPlayer(rng)}
        result = harness.play_game(bridge, "test-game", players, {"p1": "code", "p2": "random"}, rng, verbose=False)
    finally:
        bridge.close()
    assert result["turns"] >= 1 and result["rejected"] == {"p1": 0, "p2": 0}
    assert len(result["rosters"]["p1"]) == 6 and len(result["teams"]["p1"]) == 4
