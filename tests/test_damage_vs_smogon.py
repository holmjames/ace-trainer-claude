"""Our damage estimates against Smogon's official calculator (@smogon/calc), 376 generated cases.

The fixture (tests/fixtures/smogon_calc.json) was produced by a small Node script with abilities set
to a neutral one, because our model does not simulate abilities. Variable-hit moves are skipped: the
calculator assumes 3 hits while we model the item (Loaded Dice -> 4.5). Everything else must land
within a few percent of max HP of the calculator's min and max.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from agent.pokemon import battle, data

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smogon_calc.json"
pytestmark = pytest.mark.skipif(not FIXTURE.exists() or not (data.DATA_DIR / "pokedex.json").exists(),
                                reason="needs tests/fixtures/smogon_calc.json and data/ tables")


def _mon(card: dict, *, side: str, position: int, hp: float):
    merged = {"species": card["species"], "nature": card["nature"], "evs": card["evs"], "item": card["item"], "ability": None, "moves": []}
    return battle.build_mon({"species": card["species"], "current_hp_fraction": hp}, merged, side=side, position=position)


def _errors() -> list[tuple[float, dict]]:
    out = []
    for case in json.loads(FIXTURE.read_text()):
        move = {**(data.move_info(case["move"]) or {}), "id": data.to_id(case["move"])}
        if isinstance(move.get("multihit"), list):
            continue
        attacker = _mon(case["attacker"], side="mine", position=0, hp=1.0)
        defender = _mon(case["defender"], side="theirs", position=1, hp=case["defender_hp_fraction"])
        est = battle.damage_percent(attacker, move, defender, spread=case["spread_in_calc"])
        if est is None:
            assert (move.get("base_power") or 0) == 0, case  # only status moves may be skipped
            continue
        cur, max_hp, ref = case["defender_cur_hp"], case["defender_max_hp"], case["no_ability_damage"]
        ours = (est[0] * cur / 100, est[1] * cur / 100)
        error = max(abs(ours[0] - ref[0]), abs(ours[1] - ref[1])) / max_hp * 100
        out.append((error, case))
    return out


def test_damage_estimates_track_the_official_calculator():
    errors = _errors()
    assert len(errors) > 300
    values = sorted(e for e, _ in errors)
    assert statistics.median(values) < 1.0
    assert values[int(len(values) * 0.9)] < 2.5
    worst, case = max(errors, key=lambda pair: pair[0])
    assert worst < 6.0, (worst, case["attacker"]["species"], case["move"], case["defender"]["species"])


def test_immunities_and_items_match_the_calculator():
    cases = json.loads(FIXTURE.read_text())
    immune = [c for c in cases if c["no_ability_damage"] == [0, 0]]
    for case in immune:
        move = {**(data.move_info(case["move"]) or {}), "id": data.to_id(case["move"])}
        if (move.get("base_power") or 0) == 0:
            continue  # fixed-damage moves (Ruination) have no base power; the calculator reports 0, we report "no estimate"
        est = battle.damage_percent(_mon(case["attacker"], side="mine", position=0, hp=1.0),
                                    move, _mon(case["defender"], side="theirs", position=1, hp=case["defender_hp_fraction"]),
                                    spread=case["spread_in_calc"])
        assert est == (0.0, 0.0), case
