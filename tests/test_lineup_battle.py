"""Tests for the Team Preview shortlist (agent/pokemon/lineup.py) and the battle turn sheet
(agent/pokemon/battle.py). Uses the vendored Showdown tables in data/."""

from __future__ import annotations

import pytest

from agent.pokemon import battle, data, lineup
from agent.pokemon.memory import MatchMemory, species_key

pytestmark = pytest.mark.skipif(not (data.DATA_DIR / "pokedex.json").exists(), reason="run scripts/build_dex.py first")


def card(species, moves, *, item="Leftovers", ability="Pressure", nature="Serious", evs=None):
    return {"card_id": f"vgc-{data.to_id(species)}", "species": species, "item": item, "ability": ability,
            "nature": nature, "evs": evs or {}, "moves": moves}


MINE = [
    card("Rillaboom", ["Wood Hammer", "Grassy Glide", "Fake Out", "U-turn"], ability="Grassy Surge", nature="Adamant", evs={"atk": 252, "hp": 252}),
    card("Incineroar", ["Flare Blitz", "Knock Off", "Fake Out", "Parting Shot"], ability="Intimidate", nature="Careful"),
    card("Garchomp", ["Earthquake", "Dragon Claw", "Rock Slide", "Protect"], ability="Rough Skin", nature="Jolly", evs={"spe": 252, "atk": 252}),
    card("Flutter Mane", ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Protect"], ability="Protosynthesis", nature="Timid", evs={"spa": 252, "spe": 252}, item="Choice Specs"),
    card("Amoonguss", ["Spore", "Rage Powder", "Pollen Puff", "Protect"], ability="Regenerator", nature="Calm"),
    card("Whimsicott", ["Tailwind", "Moonblast", "Encore", "Protect"], ability="Prankster", nature="Timid"),
]
THEIRS = [
    card("Gyarados", ["Waterfall", "Tera Blast", "Dragon Dance", "Protect"], ability="Intimidate", nature="Jolly"),
    card("Urshifu-Rapid-Strike", ["Surging Strikes", "Close Combat", "Aqua Jet", "Detect"], ability="Unseen Fist", nature="Jolly", evs={"atk": 252, "spe": 252}),
    card("Kingambit", ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect"], ability="Defiant", nature="Adamant"),
    card("Pelipper", ["Hurricane", "Weather Ball", "Tailwind", "Protect"], ability="Drizzle", nature="Modest"),
    card("Dragonite", ["Extreme Speed", "Scale Shot", "Fire Punch", "Protect"], ability="Multiscale", nature="Adamant"),
    card("Gholdengo", ["Make It Rain", "Shadow Ball", "Nasty Plot", "Protect"], ability="Good as Gold", nature="Modest"),
]
ROSTER = [data.to_id(c["species"]) for c in MINE]


def memory():
    m = MatchMemory()
    for c in MINE:
        m.my_cards[species_key(c["species"])] = c
    for c in THEIRS:
        m.opp_cards[species_key(c["species"])] = c
    return m


# -- lineup --------------------------------------------------------------------------------------


def test_shortlist_returns_legal_lineups_in_server_spelling():
    m = memory()
    cands = lineup.shortlist(m.my_cards, m.opp_cards, ROSTER, top=3)
    assert len(cands) == 3
    for c in cands:
        assert len(set(c.bring)) == 4 and set(c.bring) <= set(ROSTER)
        assert len(set(c.leads)) == 2 and set(c.leads) <= set(c.bring)
    assert cands[0].score >= cands[1].score >= cands[2].score


def test_shortlist_brings_the_grass_answer_to_a_water_heavy_opponent():
    m = memory()
    best = lineup.shortlist(m.my_cards, m.opp_cards, ROSTER, top=1)[0]
    assert "rillaboom" in best.bring, best
    assert any("speed control" in n or "fake out" in n for n in best.notes)


def test_shortlist_handles_roster_spellings_and_unknown_species():
    m = memory()
    roster = ["Rillaboom", "Incineroar", "Garchomp", "Flutter-Mane", "Amoonguss", "Whimsicott"]
    best = lineup.shortlist(m.my_cards, m.opp_cards, roster, top=1)[0]
    assert set(best.bring) <= set(roster)  # server spelling preserved
    weird = lineup.shortlist({}, {}, ["a", "b", "c", "d", "e", "f"], top=2)
    assert len(weird) == 2 and len(weird[0].bring) == 4


# -- battle math ----------------------------------------------------------------------------------


def _mon(card_, *, side, position, hp=1.0, boosts=None, status=None):
    return battle.build_mon({"species": card_["species"], "current_hp_fraction": hp, "boosts": boosts or {}, "status": status},
                            card_, side=side, position=position)


def test_stage_multiplier():
    assert battle.stage_multiplier(0) == 1 and battle.stage_multiplier(1) == 1.5 and battle.stage_multiplier(2) == 2
    assert battle.stage_multiplier(-1) == pytest.approx(2 / 3)
    assert battle.stage_multiplier(9) == 4  # clamped at +6


def test_damage_percent_matches_a_known_calc():
    # 252+ Atk Rillaboom Wood Hammer vs 0 HP / 0 Def Gyarados: 178 Atk vs 99 Def at level 50.
    rilla = _mon(MINE[0], side="mine", position=0)
    gyara = _mon(THEIRS[0], side="theirs", position=1)
    low, high = battle.damage_percent(rilla, data.move_info("Wood Hammer") | {"id": "woodhammer"}, gyara)
    # By hand: Atk 194 (base 125, 252 EVs, Adamant), Def 99, HP 170. base = floor(floor(22*120*194/99)/50)+2 = 105;
    # x1.5 STAB, x1.0 type (water 2x, flying 0.5x), random 0.85..1.0 -> 133.9..157.5 of 170 HP.
    assert low == pytest.approx(105 * 0.85 * 1.5 / 170 * 100, abs=0.2)
    assert high == pytest.approx(105 * 1.0 * 1.5 / 170 * 100, abs=0.2)


def test_damage_respects_immunity_stab_and_items():
    chomp = _mon(MINE[2], side="mine", position=0)
    pelipper = _mon(THEIRS[3], side="theirs", position=1)
    assert battle.damage_percent(chomp, data.move_info("Earthquake") | {"id": "earthquake"}, pelipper) == (0.0, 0.0)  # Flying immune
    flutter = _mon(MINE[3], side="mine", position=1)  # Choice Specs
    plain = battle.build_mon({"species": "Flutter Mane"}, {**MINE[3], "item": "Leftovers"}, side="mine", position=1)
    specs = battle.damage_percent(flutter, data.move_info("Moonblast") | {"id": "moonblast"}, pelipper)
    no_specs = battle.damage_percent(plain, data.move_info("Moonblast") | {"id": "moonblast"}, pelipper)
    assert specs[1] > no_specs[1] * 1.4
    assert battle.damage_percent(chomp, data.move_info("Protect") | {"id": "protect"}, pelipper) is None


def test_speed_order_uses_real_stats_items_and_status():
    scarfed = battle.build_mon({"species": "Garchomp"}, {**MINE[2], "item": "Choice Scarf"}, side="mine", position=0)
    paralysed = _mon(MINE[3], side="mine", position=1, status="par")
    assert scarfed.stat("spe") == pytest.approx(169 * 1.5)
    assert paralysed.stat("spe") == pytest.approx(205 * 0.5)


# -- the sheet and candidates ---------------------------------------------------------------------


def _template():
    def move(move_id, targets, options):
        return {"type": "move", "move_id": move_id, "targets": targets, "target_options": options}
    opp = [{"target": 1, "side": "opponent", "species": "Gyarados"}, {"target": 2, "side": "opponent", "species": "Urshifu-Rapid-Strike"}]
    return {
        "type": "doubles_turn",
        "slots": [
            {"slot": 0, "active": "Rillaboom", "force_switch": False, "options": [
                move("woodhammer", [1, 2], opp), move("fakeout", [1, 2], opp), move("uturn", [1, 2], opp),
                {"type": "switch", "species": "amoonguss"}]},
            {"slot": 1, "active": "Garchomp", "force_switch": False, "options": [
                move("earthquake", [], []), move("rockslide", [], []), move("protect", [], []),
                {"type": "switch", "species": "amoonguss"}]},
        ],
    }


def _obs():
    return {
        "turn": 1, "weather": None, "field": {},
        "team": {"p1: Rillaboom": {"species": "Rillaboom", "active": True, "current_hp_fraction": 1.0},
                 "p1: Garchomp": {"species": "Garchomp", "active": True, "current_hp_fraction": 1.0}},
        "opponent_team": {"p2: Gyarados": {"species": "Gyarados", "active": True, "current_hp_fraction": 1.0},
                          "p2: Urshifu": {"species": "Urshifu-Rapid-Strike", "active": True, "current_hp_fraction": 0.5}},
    }


def test_turn_sheet_has_speed_order_damage_threats_and_valid_candidates():
    sheet = battle.build_sheet(_template(), _obs(), memory())
    order = [row["species"] for row in sheet.speed_order]
    assert order[0] in ("Urshifu-Rapid-Strike", "Garchomp")  # the two 252+ Spe Pokémon lead
    assert len(sheet.speed_order) == 4

    slot0 = next(e for e in sheet.our_options if e["slot"] == 0)
    wood = next(o for o in slot0["options"] if o["move"] == "woodhammer")
    urshifu_row = next(t for t in wood["targets"] if t["species"] == "Urshifu-Rapid-Strike")
    assert urshifu_row["ko"] == "guaranteed"  # 4x-effective STAB into a half-HP target

    assert any(h["move"] == "surgingstrikes" for t in sheet.threats for h in t["hits"])
    assert sheet.candidates and sheet.candidates[0]["name"] in ("best_attacks", "focus_fire")
    for c in sheet.candidates:
        for key in ("slot_0", "slot_1"):
            assert set(c[key]) == {"option", "target"}


def test_turn_sheet_degrades_gracefully_without_data():
    template = {"type": "doubles_turn", "slots": [
        {"slot": 0, "active": None, "options": [{"type": "pass"}]},
        {"slot": 1, "active": "Mysterymon", "options": [{"type": "move", "move_id": "zap", "targets": [1]}]}]}
    sheet = battle.build_sheet(template, {}, MatchMemory())
    assert sheet.speed_order == [] and sheet.threats == []
    assert isinstance(sheet.as_prompt(), dict)


def test_tailwind_and_trick_room_come_from_fields_and_side_conditions():
    obs = {**_obs(), "fields": ["Trick Room"], "side_conditions": ["Tailwind"], "opponent_side_conditions": []}
    sheet = battle.build_sheet(_template(), obs, memory())
    assert sheet.trick_room and any("Tailwind" in n for n in sheet.notes)
    rilla = next(r for r in sheet.speed_order if r["species"] == "Rillaboom")
    assert rilla["speed"] == pytest.approx(2 * 105, abs=1)  # base 85, 0 Spe EVs, Adamant -> 105, doubled by Tailwind
    # Under Trick Room the slowest goes first.
    assert sheet.speed_order == sorted(sheet.speed_order, key=lambda r: r["speed"])


def test_multihit_moves_count_every_hit():
    urshifu = _mon(THEIRS[1], side="theirs", position=1)
    flutter = _mon(MINE[3], side="mine", position=0)
    three_hits = battle.damage_percent(urshifu, data.move_info("Surging Strikes") | {"id": "surgingstrikes"}, flutter)
    one_hit = battle.damage_percent(urshifu, data.move_info("Aqua Jet") | {"id": "aquajet"}, flutter)
    assert three_hits[1] > 2.5 * one_hit[1]  # 3 x 25 BP (STAB) vs 1 x 40 BP (STAB)


def test_focus_sash_and_multiscale_change_ko_math():
    urshifu = _mon(THEIRS[1], side="theirs", position=1)
    sash_flutter = battle.build_mon({"species": "Flutter Mane", "current_hp_fraction": 1.0}, {**MINE[3], "item": "Focus Sash"}, side="mine", position=0)
    hurt_sash = battle.build_mon({"species": "Flutter Mane", "current_hp_fraction": 0.6}, {**MINE[3], "item": "Focus Sash"}, side="mine", position=0)
    aqua_jet = data.move_info("Aqua Jet") | {"id": "aquajet"}
    surging = data.move_info("Surging Strikes") | {"id": "surgingstrikes"}
    assert battle.damage_percent(urshifu, aqua_jet, sash_flutter)[1] <= 99.0  # single hit at full HP cannot KO through the sash
    assert battle.damage_percent(urshifu, surging, sash_flutter)[1] > 99.0  # multi-hit breaks the sash
    assert battle.damage_percent(urshifu, aqua_jet, hurt_sash)[1] > battle.damage_percent(urshifu, aqua_jet, sash_flutter)[1] * 1.3  # sash irrelevant when hurt
    dragonite = battle.build_mon({"species": "Dragonite", "current_hp_fraction": 1.0}, THEIRS[4], side="theirs", position=2)
    no_scale = battle.build_mon({"species": "Dragonite", "current_hp_fraction": 1.0}, {**THEIRS[4], "ability": "Inner Focus"}, side="theirs", position=2)
    flutter = _mon(MINE[3], side="mine", position=0)
    moonblast = data.move_info("Moonblast") | {"id": "moonblast"}
    assert battle.damage_percent(flutter, moonblast, dragonite)[1] == pytest.approx(battle.damage_percent(flutter, moonblast, no_scale)[1] / 2, rel=0.02)


def test_fake_out_setup_skips_covert_cloak_holders():
    cloak_memory = memory()
    cloak_memory.opp_cards[species_key("Gyarados")] = {**THEIRS[0], "item": "Covert Cloak"}
    cloak_memory.opp_cards[species_key("Urshifu-Rapid-Strike")] = {**THEIRS[1], "item": "Covert Cloak"}
    cloak_memory.fresh_active = ["rillaboom", "garchomp"]
    obs = {**_obs(), "turn": 1, "active_pokemon": [{"species": "Rillaboom"}, {"species": "Garchomp"}],
           "opponent_active_pokemon": [{"species": "Gyarados"}, {"species": "Urshifu-Rapid-Strike"}]}
    sheet = battle.build_sheet(_template(), obs, cloak_memory)
    assert all(c["name"] != "fake_out_setup" for c in sheet.candidates)


def test_armor_tail_on_either_opponent_blocks_fake_out_setup():
    m = memory()
    m.opp_cards[species_key("Gyarados")] = {**THEIRS[0], "ability": "Armor Tail"}
    m.fresh_active = ["rillaboom", "garchomp"]
    obs = {**_obs(), "turn": 1, "active_pokemon": [{"species": "Rillaboom"}, {"species": "Garchomp"}],
           "opponent_active_pokemon": [{"species": "Gyarados"}, {"species": "Urshifu-Rapid-Strike"}]}
    sheet = battle.build_sheet(_template(), obs, m)
    assert all(c["name"] != "fake_out_setup" for c in sheet.candidates)
