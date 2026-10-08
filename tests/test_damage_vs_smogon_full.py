"""Our damage estimates against Smogon's official calculator WITH abilities, items and field effects on.

The fixture (tests/fixtures/smogon_calc_full.json) comes from scripts/gen_smogon_fixture.js: every pool card
attacking every other with its real set under sampled weather, terrain, screens, stat stages, burn and a
third-party Ruin ability. Technician, the Ruin abilities, Guts, Booster Energy, Hadron Engine, Orichalcum
Pulse, terrain boosts, screens, Body Press, Foul Play, Weather Ball, Eruption, Facade, Heavy Slam, Knock Off
and always-crit moves all have to land within a few percent of max HP.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from agent.pokemon import battle, data

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smogon_calc_full.json"
pytestmark = pytest.mark.skipif(not FIXTURE.exists() or not (data.DATA_DIR / "pokedex.json").exists(),
                                reason="needs tests/fixtures/smogon_calc_full.json and data/ tables")


def _mon(spec: dict, *, side: str, position: int, hp: float = 1.0):
    card = {"species": spec["species"], "nature": spec["nature"], "evs": spec["evs"], "item": spec["item"], "ability": spec["ability"], "moves": []}
    summary = {"species": spec["species"], "current_hp_fraction": hp, "boosts": spec.get("boosts") or {}, "status": spec.get("status")}
    return battle.build_mon(summary, card, side=side, position=position)


def _field(case: dict, attacker, defender) -> battle.FieldState:
    f = case["field"]
    fs = battle.FieldState(weather=battle.normalize_weather(f.get("weather")), terrain=battle.normalize_terrain((f.get("terrain") or "") + "terrain"))
    fs.their_side = {c for c, on in (("reflect", f.get("defender_reflect")), ("lightscreen", f.get("defender_light_screen"))) if on}
    fs.actives = [attacker, defender]
    third = f.get("third_ruin")
    if third:
        fs.actives.append(_mon({"species": third["species"], "nature": "Serious", "evs": {}, "item": None, "ability": third["ability"]}, side="theirs", position=2))
    for mon in fs.actives:
        mon.boosted_stat = battle.paradox_boosted_stat(mon, fs)
    return fs


def _errors(fixture: Path = FIXTURE) -> list[tuple[float, dict]]:
    out = []
    for case in json.loads(fixture.read_text()):
        move = {**(data.move_info(case["move"]) or {}), "id": data.to_id(case["move"])}
        attacker = _mon(case["attacker"], side="mine", position=0, hp=case["attacker"].get("hp_fraction", 1.0))
        defender = _mon(case["defender"], side="theirs", position=1, hp=case["defender_hp_fraction"])
        if data.to_id(case["defender"]["item"]) == "focussash" and case["defender_hp_fraction"] >= 0.999:
            continue  # we cap a single hit at 99% through a full-HP sash on purpose
        fs = _field(case, attacker, defender)
        if data.to_id(case["attacker"]["ability"]) == "download":
            # Download fires on entry (+1 SpA if the foe's Def >= SpD, else +1 Atk); live, it shows up in the observed boosts.
            dstat = {k: defender.stats.get(k, 0) * battle.stage_multiplier(defender.boosts.get(k, 0)) for k in ("def", "spd")}
            key = "spa" if dstat["def"] >= dstat["spd"] else "atk"
            attacker.boosts[key] = min(6, attacker.boosts.get(key, 0) + 1)
        if data.to_id(case["defender"]["ability"]) == "download":
            # The calculator fires the defender's Download too (it matters for Foul Play, which uses the target's Attack).
            dstat = {k: attacker.stats.get(k, 0) * battle.stage_multiplier(attacker.boosts.get(k, 0)) for k in ("def", "spd")}
            key = "spa" if dstat["def"] >= dstat["spd"] else "atk"
            defender.boosts[key] = min(6, defender.boosts.get(key, 0) + 1)
        seed = data.to_id(case["defender"]["item"])
        if seed.endswith("seed") and fs.terrain and seed.startswith(fs.terrain):
            # The calculator assumes the seed has popped. In a live game the +1 shows up in the observed boosts
            # (and we only drop the Knock Off bonus), so mirror that here.
            defender.boosts[{"psychicseed": "spd", "mistyseed": "spd", "electricseed": "def", "grassyseed": "def"}[seed]] = 1
        est = battle.damage_percent(attacker, move, defender, field=fs, spread=case["spread_in_calc"])
        ref = case["damage"]
        if est is None:
            assert ref == [0, 0] or (move.get("base_power") or 0) == 0, case
            continue
        cur, max_hp = case["defender_cur_hp"], case["defender_max_hp"]
        # Anything past a KO is just a KO: compare both estimates clamped to the HP that is actually there.
        ours = (min(est[0] * cur / 100, cur), min(est[1] * cur / 100, cur))
        ref = (min(ref[0], cur), min(ref[1], cur))
        error = max(abs(ours[0] - ref[0]), abs(ours[1] - ref[1])) / max_hp * 100
        out.append((error, case))
    return out


def test_full_model_tracks_the_official_calculator():
    errors = _errors()
    assert len(errors) > 1000
    values = sorted(e for e, _ in errors)
    assert statistics.median(values) < 1.0
    assert values[int(len(values) * 0.9)] < 2.5
    worst, case = max(errors, key=lambda pair: pair[0])
    assert worst < 6.0, (worst, case["attacker"]["species"], case["move"], case["defender"]["species"], case["field"])


def test_specific_mechanics_match_the_calculator():
    """Each mechanic the old model lacked is exercised by at least one fixture case, and that case is close."""
    by_mech = {
        "technician": lambda c: c["attacker"]["ability"] == "Technician",
        "sword_of_ruin_attacker": lambda c: c["attacker"]["ability"] == "Sword of Ruin",
        "third_party_ruin": lambda c: bool(c["field"].get("third_ruin")),
        "guts_burn": lambda c: c["attacker"]["ability"] == "Guts" and c["attacker"].get("status") == "brn",
        "booster_offense": lambda c: c["attacker"]["species"] == "Raging Bolt",
        "hadron_engine_terrain": lambda c: c["attacker"]["ability"] == "Hadron Engine" and c["field"].get("terrain") == "Electric",
        "orichalcum_sun": lambda c: c["attacker"]["ability"] == "Orichalcum Pulse" and c["field"].get("weather") == "Sun",
        "screens": lambda c: c["field"].get("defender_reflect") or c["field"].get("defender_light_screen"),
        "body_press": lambda c: c["move"] == "Body Press",
        "foul_play": lambda c: c["move"] == "Foul Play",
        "weather_ball_rain": lambda c: c["move"] == "Weather Ball" and c["field"].get("weather") == "Rain",
        "eruption_low_hp": lambda c: c["move"] == "Eruption",
        "heavy_slam": lambda c: c["move"] in ("Heavy Slam", "Heat Crash"),
        "grassy_earthquake": lambda c: c["move"] == "Earthquake" and c["field"].get("terrain") == "Grassy",
        "psychic_expanding_force": lambda c: c["move"] == "Expanding Force" and c["field"].get("terrain") == "Psychic",
        "surging_strikes_crit": lambda c: c["move"] == "Surging Strikes",
        "population_bomb": lambda c: c["move"] == "Population Bomb",
        "minds_eye_hits_ghosts": lambda c: c["attacker"]["ability"] == "Mind's Eye" and "ghost" in (data.resolve_types({"species": c["defender"]["species"]}) or []),
        "charge_move_boost": lambda c: c["move"] in ("Meteor Beam", "Electro Shot"),
        "priority_blocked_by_psychic_terrain": lambda c: c["field"].get("terrain") == "Psychic" and (data.move_info(c["move"]) or {}).get("priority", 0) > 0,
    }
    errors = _errors()
    for name, pred in by_mech.items():
        subset = [e for e, c in errors if pred(c)]
        assert subset, f"no fixture case for {name}"
        assert max(subset) < 6.0, (name, max(subset))
