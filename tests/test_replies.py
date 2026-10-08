"""Core lane, Part 1: the candidate generators and scorer fixes (D1f, D1i, D1j, D1l, the Ruination composition, unsafe
switch rows, the KO credit). Live positions replay a captured session through ONE code-only agent (the
tests/test_live_semantics.py pattern) and read the sheet and the played turn from its log; hand-built positions use the
catalog sets. Every damage number is the sheet's own (battle.damage_percent / hit_profile), never hand arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.pokemon import battle, data
from agent.pokemon.memory import MatchMemory, species_key
from agent.pokemon.tuning import DEFAULTS
from tests.test_live_semantics import replay, turn_row

# Sessions: the four Oct 7 losses (M1-M4), the Oct 8 practice win vs the search bot (O8) and vs a generic-LLM agent (O8B).
M1, M2, M3, M4, O8, O8B = "d9405f02", "1aef9227", "7ce71025", "0a2b7c2a", "bcbaabe2", "93a98b6f"
_ROWS: dict[str, list[dict]] = {}


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    """Replay a session once per module: ``rows(prefix)`` -> its log rows."""
    def get(prefix: str) -> list[dict]:
        if prefix not in _ROWS:
            _ROWS[prefix] = replay(prefix, tmp_path_factory.mktemp(prefix))[1]
        return _ROWS[prefix]
    return get


def candidates(row: dict) -> list[dict]:
    return row["turn_sheet"]["candidate_turns"]


def named(row: dict, name: str) -> list[dict]:
    return [c for c in candidates(row) if c["name"] == name]


def played(row: dict) -> dict:
    return row["payload"]


def move_of(payload: dict, slot: int) -> str | None:
    s = payload.get(f"slot_{slot}") or {}
    return s.get("move_id") if s.get("type") == "move" else None


# -- D1f: spread rows slipped past avoid_target --------------------------------------------------------------------


def test_spread_rows_no_longer_hide_finish_the_other(rows):
    # M3 turn 3: Hatterene's Dazzling Gleam KOs the 9% Whimsicott and Great Tusk holds Rock Slide (spread). The spread
    # row carried target 0, passed the avoid_target filter as Tusk's "best attack into the other foe", and the
    # "Gleam KOs Whimsicott, Headlong Rush into Incineroar" turn was never generated (REVIEW 4A).
    row = turn_row(rows(M3), 3)
    finish = named(row, "finish_the_other")
    assert finish, [c["name"] for c in candidates(row)]
    assert "dazzlinggleam already KOs whimsicott" in finish[0]["why"] and "headlongrush goes into incineroar" in finish[0]["why"]


def test_spread_rows_no_longer_hide_focus_fire(rows):
    # The Oct 8 LLM-agent game, turn 6: Waterfall + Icicle Spear into the 54% Grimmsnarl is the double that wins the
    # turn; Gyarados' Earthquake (spread) hid it and the sheet had one candidate.
    row = turn_row(rows(O8B), 6)
    focus = [c for c in named(row, "focus_fire") if "grimmsnarl" in c["why"]]
    assert focus, [c["why"] for c in candidates(row)]
    assert len(candidates(row)) >= 2


def test_best_attack_with_avoid_target_ignores_spread_rows():
    entry = {"options": [
        {"option": 0, "move": "rockslide", "priority": 0, "spread": True, "targets": [
            {"target": 0, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [300.0, 350.0], "ko": "guaranteed", "hp_frac": 0.09, "spread": True},
            {"target": 0, "species": "B", "side": "theirs", "damage_pct_of_current_hp": [30.0, 36.0], "ko": "no", "hp_frac": 1.0, "spread": True}]},
        {"option": 1, "move": "headlongrush", "priority": 0, "spread": False, "targets": [
            {"target": 1, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [400.0, 490.0], "ko": "guaranteed", "hp_frac": 0.09},
            {"target": 2, "species": "B", "side": "theirs", "damage_pct_of_current_hp": [40.0, 50.0], "ko": "no", "hp_frac": 1.0}]},
    ]}
    opt, row, _ = battle._best_attack(entry, avoid_target=1)
    assert opt["move"] == "headlongrush" and row["target"] == 2
    opt, row, _ = battle._best_attack(entry)  # no constraint: the spread summary carries its per-target rows
    assert row["target"] == 0 and row.get("rows") and {r["species"] for r in row["rows"]} == {"A", "B"}


# -- D1j: fake_out_setup's KO read and add()'s dedup ---------------------------------------------------------------


def test_a_spread_summary_is_not_a_single_target_ko():
    summary = {"target": 0, "species": "A + B", "side": "theirs", "damage_pct_of_current_hp": [110.0, 130.0], "rows": [
        {"target": 0, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [60.0, 70.0]},
        {"target": 0, "species": "B", "side": "theirs", "damage_pct_of_current_hp": [50.0, 60.0]}]}
    assert not battle.kos_a_foe(summary)
    summary["rows"][0]["damage_pct_of_current_hp"] = [100.0, 120.0]
    assert battle.kos_a_foe(summary)
    assert battle.kos_a_foe({"target": 1, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [101.0, 120.0]})
    assert not battle.kos_a_foe({"target": 1, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [90.0, 120.0]})
    assert battle.kos_a_foe({"target": 1, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [90.0, 120.0]}, min_chance=0.5)


def _catalog() -> dict[str, dict]:
    return {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())}


def test_duplicate_turns_keep_the_better_scored_name():
    """best_attacks and finish_the_other produce the same payload when one slot's best attack already KOs a foe and the
    other slot's best single-target hit is into the survivor: keep the higher-scoring (and better explained) row."""
    cards = _catalog()
    m = MatchMemory()
    m.my_cards["gyarados"] = cards["Gyarados"]
    m.my_cards["lucario"] = cards["Lucario"]
    m.opp_cards["fluttermane"] = cards["Flutter Mane"]
    m.opp_cards["incineroar"] = cards["Incineroar"]
    opp = [{"target": 1, "side": "opponent", "species": "Flutter Mane"}, {"target": 2, "side": "opponent", "species": "Incineroar"}]
    slots = [{"slot": 0, "active": "Gyarados", "options": [{"type": "move", "move_id": "waterfall", "targets": [1, 2], "target_options": opp}]},
             {"slot": 1, "active": "Lucario", "options": [{"type": "move", "move_id": "closecombat", "targets": [1, 2], "target_options": opp}]}]
    obs = {"turn": 4, "team": {"a": {"species": "Gyarados", "active": True, "current_hp_fraction": 1.0}, "b": {"species": "Lucario", "active": True, "current_hp_fraction": 1.0}},
           "opponent_team": {"c": {"species": "Flutter Mane", "active": True, "current_hp_fraction": 0.2}, "d": {"species": "Incineroar", "active": True, "current_hp_fraction": 1.0}},
           "active_pokemon": [{"species": "Gyarados"}, {"species": "Lucario"}], "opponent_active_pokemon": [{"species": "Flutter Mane"}, {"species": "Incineroar"}]}
    sheet = battle.build_sheet({"type": "doubles_turn", "slots": slots}, obs, m)
    payloads = [(c["slot_0"]["option"], c["slot_0"]["target"], c["slot_1"]["option"], c["slot_1"]["target"]) for c in sheet.candidates]
    assert len(payloads) == len(set(payloads)), payloads
    waterfall_fm = next(c for c in sheet.candidates if c["slot_0"] == {"option": 0, "target": 1} and c["slot_1"] == {"option": 0, "target": 2})
    assert waterfall_fm["name"] == "finish_the_other", sheet.candidates
