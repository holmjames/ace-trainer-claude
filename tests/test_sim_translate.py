"""Tests for sim/translate.py (pure functions over hand-built snapshots) and, when Node + the engine are installed,
one real local game through sim/harness.py. The live-format contract itself is pinned by test_sim_live_parity.py."""

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
               "slot_1": {"type": "switch", "species": "amoonguss"}}
    assert translate.to_choice(payload) == "move fakeout 2, switch amoonguss"
    assert translate.to_choice({"slot_0": {"type": "move", "move_id": "tailwind", "target": 0}, "slot_1": {"type": "pass"}}) == "move tailwind, pass"
    assert translate.to_choice({"slot_0": {"type": "move", "move_id": "helpinghand", "target": -2}, "slot_1": {"type": "move", "move_id": "protect"}}) == "move helpinghand -2, move protect"


def test_team_text_and_preview_choice_over_the_full_six():
    cards = [{"species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "nature": "Careful", "evs": {"hp": 252, "spd": 252}, "moves": ["Fake Out", "Knock Off", "Flare Blitz", "Parting Shot"]},
             {"species": "Rillaboom", "item": "Assault Vest", "ability": "Grassy Surge", "nature": "Adamant", "evs": {"atk": 252}, "moves": ["Wood Hammer", "Fake Out", "U-turn", "Grassy Glide"]},
             {"species": "Garchomp", "item": "Life Orb", "ability": "Rough Skin", "nature": "Jolly", "evs": {}, "moves": ["Earthquake"]},
             {"species": "Amoonguss", "item": "Rocky Helmet", "ability": "Regenerator", "nature": "Calm", "evs": {}, "moves": ["Spore"]},
             {"species": "Whimsicott", "item": "Focus Sash", "ability": "Prankster", "nature": "Timid", "evs": {}, "moves": ["Tailwind"]},
             {"species": "Flutter Mane", "item": "Choice Specs", "ability": "Protosynthesis", "nature": "Timid", "evs": {}, "moves": ["Moonblast"]}]
    text = translate.team_text(cards[:2])
    assert "Incineroar @ Safety Goggles" in text and "EVs: 252 HP / 252 SpD" in text and "- Parting Shot" in text
    # Showdown gets all six (draft order); the lineup becomes a Team Preview choice over them, leads first
    choice, order = translate.preview_choice(["garchomp", "amoonguss", "incineroar", "whimsicott"], ["whimsicott", "garchomp"], cards)
    assert choice == "team 5341" and order == [4, 2, 3, 0]
    assert translate.preview_choice(["fluttermane", "rillaboom", "garchomp", "amoonguss"], ["fluttermane", "rillaboom"],
                                    ["incineroar", "rillaboom", "garchomp", "amoonguss", "whimsicott", "fluttermane"])[0] == "team 6234"


def test_draft_state_lists_every_card_but_only_offers_item_clause_legal_picks():
    pool = [{"card_id": "b", "species": "B", "item": "Life Orb"}, {"card_id": "c", "species": "C", "item": "Leftovers"}]
    rosters = {"seat-0": [{"card_id": "a", "species": "A", "item": "Leftovers"}], "seat-1": []}
    mine = translate.draft_state("s", me="seat-0", pool=pool, rosters=rosters, picks=[], first_drafter="seat-0", current_seat="seat-0", version=1)
    obs = mine.raw["observation"]
    assert obs["available_card_ids"] == ["b", "c"] == obs["unused_card_ids"]  # live lists C even though we cannot take it
    assert [a.action_id for a in mine.legal_actions] == ["draft_pick:b"]  # C's Leftovers clash with our roster
    assert mine.legal_actions[0].label == "Draft b" and mine.raw["current_actor"] == {"agent_id": "seat-0", "position": 0}
    assert obs["rosters"] == {"seat-0": [{"card_id": "a", "species": "A"}], "seat-1": []}
    assert obs["decision_timeout_seconds"] == 15.0 and "T" in obs["decision_deadline_at"]
    theirs = translate.draft_state("s", me="seat-1", pool=pool, rosters=rosters, picks=[], first_drafter="seat-0", current_seat="seat-0", version=1)
    assert theirs.legal_actions == [] and theirs.is_current_actor is False


# -- hand-built snapshots: Showdown requests + a per-player protocol log, as sim/bridge.js sends them --------------


def _mon(ident, details, condition, active, moves, item, ability):
    return {"ident": ident, "details": details, "condition": condition, "active": active, "stats": {}, "moves": moves,
            "baseAbility": ability, "item": item, "pokeball": "pokeball", "ability": ability, "terastallized": ""}


INCINEROAR = ("p1: Incineroar", "Incineroar, L50, M", ["fakeout", "knockoff", "partingshot", "flareblitz"], "safetygoggles", "intimidate")
GARCHOMP = ("p1: Garchomp", "Garchomp, L50, F", ["earthquake", "rockslide", "ironhead", "protect"], "focussash", "roughskin")
AMOONGUSS = ("p1: Amoonguss", "Amoonguss, L50, F", ["spore", "ragepowder", "pollenpuff", "sludgebomb"], "rockyhelmet", "regenerator")
WHIMSICOTT = ("p1: Whimsicott", "Whimsicott, L50, F", ["tailwind", "moonblast", "encore", "lightscreen"], "focussash", "prankster")
PELIPPER = ("p1: Pelipper", "Pelipper, L50, M", ["hurricane", "weatherball", "tailwind", "protect"], "damprock", "drizzle")
LANDORUS = ("p1: Landorus", "Landorus-Therian, L50, M", ["earthquake", "rockslide", "uturn", "protect"], "choicescarf", "intimidate")

HEADER = ["|init|battle", "|title|a vs. b", "|j|☆a", "|j|☆b", "|gametype|doubles", "|player|p1|a|266|", "|player|p2|b|169|", "|gen|9",
          "|tier|[Gen 9] VGC 2025 Reg I", "|clearpoke",
          "|poke|p1|Pelipper, L50, M|", "|poke|p1|Incineroar, L50, M|", "|poke|p1|Garchomp, L50, F|", "|poke|p1|Amoonguss, L50, F|",
          "|poke|p1|Whimsicott, L50, F|", "|poke|p1|Landorus-Therian, L50, M|",
          "|poke|p2|Gyarados, L50, M|", "|poke|p2|Torkoal, L50, F|", "|poke|p2|Gholdengo, L50|", "|poke|p2|Cresselia, L50, F|",
          "|poke|p2|Indeedee-F, L50, F|", "|poke|p2|Landorus-Therian, L50, M|", "|teampreview|4", "|", "|teamsize|p1|4", "|teamsize|p2|4", "|start"]

PREVIEW = {"teamPreview": True, "maxChosenTeamSize": 4, "side": {"name": "a", "id": "p1", "pokemon": [
    _mon(*PELIPPER[:2], "167/167", True, *PELIPPER[2:]), _mon(*INCINEROAR[:2], "202/202", True, *INCINEROAR[2:]),
    _mon(*GARCHOMP[:2], "183/183", False, *GARCHOMP[2:]), _mon(*AMOONGUSS[:2], "221/221", False, *AMOONGUSS[2:]),
    _mon(*WHIMSICOTT[:2], "153/153", False, *WHIMSICOTT[2:]), _mon(*LANDORUS[:2], "164/164", False, *LANDORUS[2:])]}}


def _snapshot(log, request):
    roster = [{"species": s, "base_species": b} for s, b in (("gyarados", "Gyarados"), ("torkoal", "Torkoal"), ("gholdengo", "Gholdengo"),
                                                             ("cresselia", "Cresselia"), ("indeedeef", "Indeedee"), ("landorustherian", "Landorus"))]
    return {"battle_tag": "battle-gen9vgc2025regi-9", "turn": 2, "ended": False, "winner": None,
            "sides": {"p1": {"name": "a", "request": request, "preview_request": PREVIEW, "roster": [], "pokemon": []},
                      "p2": {"name": "b", "request": {"wait": True}, "preview_request": None, "roster": roster, "pokemon": []}},
            "logs": {"p1": list(log), "p2": []}}


BATTLE_LOG = HEADER + [
    "|switch|p1a: Incineroar|Incineroar, L50, M|202/202", "|switch|p1b: Garchomp|Garchomp, L50, F|183/183",
    "|switch|p2a: Gyarados|Gyarados, L50, M|100/100", "|switch|p2b: Torkoal|Torkoal, L50, F|100/100",
    "|-weather|SunnyDay|[from] ability: Drought|[of] p2b: Torkoal", "|-ability|p1a: Incineroar|Intimidate|boost",
    "|-unboost|p2a: Gyarados|atk|1", "|-unboost|p2b: Torkoal|atk|1", "|-ability|p2a: Gyarados|Intimidate|boost",
    "|-unboost|p1a: Incineroar|atk|1", "|-unboost|p1b: Garchomp|atk|1", "|turn|1",
    "|", "|move|p2a: Gyarados|Waterfall|p1b: Garchomp", "|-damage|p1b: Garchomp|100/183",
    "|move|p1a: Incineroar|Knock Off|p2b: Torkoal", "|-damage|p2b: Torkoal|0 fnt", "|faint|p2b: Torkoal",
    "|move|p1b: Garchomp|Rock Slide|p2a: Gyarados|[spread] p2a", "|-damage|p2a: Gyarados|61/100",
    "|-enditem|p2a: Gyarados|Sitrus Berry|[eat]", "|-heal|p2a: Gyarados|86/100|[from] item: Sitrus Berry",
    "|-sidestart|p1: a|move: Tailwind", "|", "|-weather|SunnyDay|[upkeep]", "|upkeep", "|turn|2"]

TURN2_REQUEST = {"active": [
    {"moves": [{"move": "Fake Out", "id": "fakeout", "pp": 15, "maxpp": 16, "target": "normal", "disabled": True},
               {"move": "Knock Off", "id": "knockoff", "pp": 31, "maxpp": 32, "target": "normal", "disabled": False},
               {"move": "Parting Shot", "id": "partingshot", "pp": 32, "maxpp": 32, "target": "normal", "disabled": False},
               {"move": "Flare Blitz", "id": "flareblitz", "pp": 24, "maxpp": 24, "target": "normal", "disabled": False}], "canTerastallize": "Fire"},
    {"moves": [{"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "target": "allAdjacent", "disabled": False},
               {"move": "Rock Slide", "id": "rockslide", "pp": 15, "maxpp": 16, "target": "allAdjacentFoes", "disabled": False},
               {"move": "Iron Head", "id": "ironhead", "pp": 24, "maxpp": 24, "target": "normal", "disabled": False},
               {"move": "Protect", "id": "protect", "pp": 16, "maxpp": 16, "target": "self", "disabled": False}], "canTerastallize": "Dragon"}],
    "side": {"name": "a", "id": "p1", "pokemon": [
        _mon(*INCINEROAR[:2], "202/202", True, *INCINEROAR[2:]), _mon(*GARCHOMP[:2], "100/183", True, *GARCHOMP[2:]),
        _mon(*AMOONGUSS[:2], "221/221", False, *AMOONGUSS[2:]), _mon(*WHIMSICOTT[:2], "0 fnt", False, *WHIMSICOTT[2:])]}}


def test_battle_state_is_built_from_the_log_and_the_request_in_the_live_format():
    state = translate.battle_state("s", _snapshot(BATTLE_LOG, TURN2_REQUEST), "p1", 3, agent_id="seat-0")
    assert state is not None and state.phase == "moving" and state.state_version == 3
    obs = state.raw["observation"]
    assert (obs["turn"], obs["weather"], obs["side_conditions"], obs["fields"]) == (2, {"SUNNYDAY": 1}, {"TAILWIND": 1}, {})
    # our team: all six, roster order; the unbrought Pelipper keeps the Team Preview request's active flag
    assert list(obs["team"]) == ["p1: Pelipper", "p1: Incineroar", "p1: Garchomp", "p1: Amoonguss", "p1: Whimsicott", "p1: Landorus"]
    assert (obs["team"]["p1: Pelipper"]["active"], obs["team"]["p1: Pelipper"]["revealed"]) == (True, False)
    assert obs["team"]["p1: Landorus"]["species"] == "landorustherian" and obs["team"]["p1: Landorus"]["name"] == "Landorus"
    assert obs["team"]["p1: Garchomp"]["current_hp"] == 100 and obs["team"]["p1: Garchomp"]["max_hp"] == 183
    assert obs["team"]["p1: Incineroar"]["boosts"]["atk"] == -1 and obs["team"]["p1: Whimsicott"]["status"] == "FNT"
    # opponents: only those that appeared; HP in percent; items/abilities/moves only as revealed
    gyarados, torkoal = obs["opponent_team"]["p2: Gyarados"], obs["opponent_team"]["p2: Torkoal"]
    assert list(obs["opponent_team"]) == ["p2: Gyarados", "p2: Torkoal"]
    assert (gyarados["current_hp"], gyarados["max_hp"], gyarados["item"], gyarados["ability"], gyarados["moves"]) == (86, 100, None, "intimidate", ["waterfall"])
    assert (torkoal["item"], torkoal["ability"], torkoal["fainted"], torkoal["active"]) == ("unknown_item", None, True, True)
    assert obs["opponent_active_pokemon"][1] is None and obs["opponent_active_pokemon"][0]["species"] == "gyarados"
    assert obs["can_tera"] == [True, True] and obs["force_switch"] == [False, False]
    slot0, slot1 = state.legal_actions[0].input["action"]["slots"]
    assert slot0["active"]["species"] == "incineroar" and slot1["board_position"] == -2
    knock = slot0["options"][0]
    assert [o.get("move_id") for o in slot0["options"] if o["type"] == "move"] == ["knockoff", "partingshot", "flareblitz"]  # disabled Fake Out dropped
    assert knock["targets"] == [-2, 1, 2]  # the ally first, and the empty position stays listed
    assert knock["target_options"] == [{"target": -2, "side": "ally", "species": "garchomp"}, {"target": 1, "side": "opponent", "species": "gyarados"},
                                       {"target": 2, "side": "opponent", "species": None}]
    eq = next(o for o in slot1["options"] if o.get("move_id") == "earthquake")
    assert eq["targets"] == [0] and eq["target_options"] == [{"target": 0, "side": "none", "species": None}]  # spread: [0], not []
    assert next(o for o in slot1["options"] if o.get("move_id") == "protect")["targets"] == [0]
    assert [o for o in slot1["options"] if o["type"] == "switch"] == [{"type": "switch", "species": "amoonguss"}]  # fainted Whimsicott is not offered
    assert obs["available_moves"][1][0]["target"] == "ALL_ADJACENT" and obs["available_moves"][1][0]["max_pp"] == 16
    assert translate.battle_state("s", _snapshot(BATTLE_LOG, {"wait": True}), "p1", 1) is None  # waiting side has no decision


def test_force_switch_offers_switches_to_the_forced_slot_and_pass_to_the_other():
    request = {"forceSwitch": [True, False], "side": {"name": "a", "id": "p1", "pokemon": [
        _mon(*INCINEROAR[:2], "0 fnt", True, *INCINEROAR[2:]), _mon(*GARCHOMP[:2], "100/183", True, *GARCHOMP[2:]),
        _mon(*AMOONGUSS[:2], "221/221", False, *AMOONGUSS[2:]), _mon(*WHIMSICOTT[:2], "153/153", False, *WHIMSICOTT[2:])]}}
    log = BATTLE_LOG + ["|", "|move|p2a: Gyarados|Waterfall|p1a: Incineroar", "|-damage|p1a: Incineroar|0 fnt", "|faint|p1a: Incineroar", "|upkeep"]
    state = translate.battle_state("s", _snapshot(log, request), "p1", 4)
    obs = state.raw["observation"]
    template = state.legal_actions[0].input["action"]
    assert template["slots"][0]["options"] == [{"type": "switch", "species": "amoonguss"}, {"type": "switch", "species": "whimsicott"}]
    assert template["slots"][0]["force_switch"] is True and template["slots"][0]["active"] is None  # fainted: None
    assert template["slots"][1]["options"] == [{"type": "pass"}]
    assert obs["available_moves"] == [[], []] and obs["can_tera"] == [False, False]
    assert [[m["species"] for m in s] for s in obs["available_switches"]] == [["amoonguss", "whimsicott"]] * 2  # live lists them for both slots
    assert obs["team"]["p1: Incineroar"]["active"] is True  # a fainted Pokémon stays active until it is replaced, as live


def test_short_handed_force_switch_lets_one_slot_pass():
    """Both actives fainted, one reserve: Showdown needs one switch and one pass. (No live capture of this case yet.)"""
    request = {"forceSwitch": [True, True], "side": {"name": "a", "id": "p1", "pokemon": [
        _mon(*INCINEROAR[:2], "0 fnt", True, *INCINEROAR[2:]), _mon(*GARCHOMP[:2], "0 fnt", True, *GARCHOMP[2:]),
        _mon(*AMOONGUSS[:2], "221/221", False, *AMOONGUSS[2:]), _mon(*WHIMSICOTT[:2], "0 fnt", False, *WHIMSICOTT[2:])]}}
    template = translate.battle_state("s", _snapshot(BATTLE_LOG, request), "p1", 4).legal_actions[0].input["action"]
    for slot in template["slots"]:
        assert slot["options"] == [{"type": "switch", "species": "amoonguss"}, {"type": "pass"}]


def test_team_preview_from_the_request_and_the_poke_lines():
    snap = _snapshot(HEADER, PREVIEW)
    state = translate.preview_state("s", snap, "p1", agent_id="seat-0")
    obs = state.raw["observation"]
    assert [m["species"] for m in obs["your_roster"]] == ["pelipper", "incineroar", "garchomp", "amoonguss", "whimsicott", "landorustherian"]
    assert [m["active"] for m in obs["your_roster"]] == [True, True, False, False, False, False]
    opp = obs["opponent_roster"]
    assert [m["name"] for m in opp] == ["Gyarados", "Torkoal", "Gholdengo", "Cresselia", "Indeedee", "Landorus"]
    assert all(m["moves"] == [] and m["item"] == "unknown_item" and (m["current_hp"], m["max_hp"]) == (0, 0) for m in opp)
    assert [m["ability"] for m in opp] == [None, None, "goodasgold", "levitate", None, "intimidate"]  # single-ability species only
    assert state.legal_actions[0].input["action"]["roster"] == [m["species"] for m in obs["your_roster"]] and state.state_version == 0


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
    assert set(result["teams"]["p1"]) <= set(result["rosters"]["p1"])
