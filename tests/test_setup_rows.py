"""The redirect and Fake Out rows pair their move with the partner's speed control (Tailwind, Trick Room, Icy Wind,
Electroweb). A Tailwind already up on our side sets nothing, and a second Trick Room ENDS the room we set: neither is
the partner's setup then, and the partner attacks instead. Positions: the hand scenarios of scripts/scenarios.py with
our Tailwind or Trick Room up (code only)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.pokemon import battle, tuning

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("scenarios", ROOT / "scripts" / "scenarios.py")
scenarios = importlib.util.module_from_spec(_spec)
sys.modules["scenarios"] = scenarios
_spec.loader.exec_module(scenarios)
HAND = {s.name: s for s in scenarios.SCENARIOS + scenarios.HARD}
FAKE_OUT, REDIRECT = HAND["fake_out_the_scarf_urshifu"], HAND["redirect_to_free_specs_attacker"]
TORNADUS = dict(my_active=("amoonguss", "tornadus"), mine=["amoonguss", "tornadus", "incineroar", "garchomp"])
HATTERENE = dict(my_active=("amoonguss", "hatterene"), mine=["amoonguss", "hatterene", "incineroar", "garchomp"])


def rows(sc, tmp_path) -> list[dict]:
    """The candidate rows the code-only agent logs for scenario ``sc`` (scripts/scenarios.py run())."""
    scenarios.run(sc, code_only=True, log_dir=tmp_path)
    logged = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    return next(r for r in logged if r.get("kind") == "turn")["turn_sheet"]["candidate_turns"]


@pytest.mark.parametrize("name, sc, row, spent", [
    # turn 3: Incineroar just came in beside the Tornadus that set our Tailwind
    ("fake_out_tailwind_up", dataclasses.replace(FAKE_OUT, turn=3, my_fresh=("incineroar",), their_fresh=(), my_side=["Tailwind"]),
     "fake_out_setup", "tailwind"),
    ("redirect_tailwind_up", dataclasses.replace(REDIRECT, my_side=["Tailwind"], **TORNADUS), "redirect_to_protect_partner", "tailwind"),
    ("fake_out_our_trick_room", dataclasses.replace(FAKE_OUT, my_active=("incineroar", "hatterene"), turn=3, my_fresh=("incineroar",),
                                                    their_fresh=(), fields=["Trick Room"], mine=["incineroar", "hatterene", "garchomp", "amoonguss"]),
     "fake_out_setup", "trickroom"),
    ("redirect_our_trick_room", dataclasses.replace(REDIRECT, fields=["Trick Room"], **HATTERENE), "redirect_to_protect_partner", "trickroom"),
])
def test_the_partner_attacks_instead_of_a_spent_setup(name, sc, row, spent, tmp_path, monkeypatch):
    monkeypatch.setitem(tuning.DEFAULTS, "reply_mixture", 0.0)  # the default scorer's rows
    monkeypatch.setitem(tuning.DEFAULTS, "lineup_sim_budget_s", 0.0)
    found = [c for c in rows(sc, tmp_path) if c["name"] == row]
    assert found, name
    assert not [c for c in found if f"sets {spent}" in c["why"]], found
    assert all("attack" in c["why"] for c in found), found


@pytest.mark.parametrize("name, sc, row, against_us", [
    # our Tailwind up: Amoonguss 100 and Hatterene 88 outspeed Kingambit (71); a Trick Room would put them behind it
    ("redirect_room_under_our_tailwind", dataclasses.replace(REDIRECT, my_side=["Tailwind"], **HATTERENE), "redirect_to_protect_partner", "trickroom"),
    # our Trick Room up: Incineroar (80) moves before Urshifu (149); doubled (160) it would move after it
    ("fake_out_tailwind_under_our_room", dataclasses.replace(FAKE_OUT, turn=3, my_fresh=("incineroar",), their_fresh=(), fields=["Trick Room"]),
     "fake_out_setup", "tailwind"),
])
def test_the_partner_attacks_instead_of_a_setup_that_turns_the_speed_order_against_us(name, sc, row, against_us, tmp_path, monkeypatch):
    monkeypatch.setitem(tuning.DEFAULTS, "reply_mixture", 0.0)
    monkeypatch.setitem(tuning.DEFAULTS, "lineup_sim_budget_s", 0.0)
    found = [c for c in rows(sc, tmp_path) if c["name"] == row]
    assert found and all("attack" in c["why"] and f"sets {against_us}" not in c["why"] for c in found), (name, found)


def test_a_speed_setup_backfires_only_when_one_of_ours_would_lose_a_foe_it_outspeeds():
    def sheet(ours: list[int], foes: list[int], *, our_tailwind: bool = False, room: bool = False) -> battle.TurnSheet:
        s = battle.TurnSheet(trick_room=room, notes=["Our Tailwind is up (speed doubled)."] if our_tailwind else [])
        s.speed_order = [{"side": "mine", "speed": v} for v in ours] + [{"side": "theirs", "speed": v} for v in foes]
        return s
    ours_room = SimpleNamespace(moves_by_turn={"mine": {"hatterene": {2: "trickroom"}}, "theirs": {}})
    their_room = SimpleNamespace(moves_by_turn={"mine": {}, "theirs": {"farigiraf": {2: "trickroom"}}})
    assert battle.setup_backfires(sheet([100, 88], [149, 71], our_tailwind=True), "trickroom")  # ours outspeed Kingambit
    assert not battle.setup_backfires(sheet([60, 50], [149, 71], our_tailwind=True), "trickroom")  # still slower than both
    assert not battle.setup_backfires(sheet([100, 88], [149, 71]), "trickroom")  # no Tailwind of ours to undo
    assert battle.setup_backfires(sheet([80, 179], [149, 307], room=True), "tailwind", ours_room)  # 80 -> 160 passes 149
    assert not battle.setup_backfires(sheet([30, 40], [149, 307], room=True), "tailwind", ours_room)  # 60, 80: still first
    assert not battle.setup_backfires(sheet([80, 179], [149, 307], room=True), "tailwind", their_room)  # their room: as before
    assert not battle.setup_backfires(sheet([80], [149]), "icywind") and not battle.setup_backfires(sheet([80], [149]), "tailwind")


def test_a_tailwind_not_yet_up_is_still_the_redirect_partners_setup(tmp_path, monkeypatch):
    monkeypatch.setitem(tuning.DEFAULTS, "reply_mixture", 0.0)
    monkeypatch.setitem(tuning.DEFAULTS, "lineup_sim_budget_s", 0.0)
    found = [c for c in rows(dataclasses.replace(REDIRECT, **TORNADUS), tmp_path) if c["name"] == "redirect_to_protect_partner"]
    assert found and all("sets tailwind" in c["why"] for c in found), found


def test_only_our_own_trick_room_is_spent():
    sheet = battle.TurnSheet()
    sheet.trick_room = True
    ours = SimpleNamespace(moves_by_turn={"mine": {"hatterene": {2: "trickroom"}}, "theirs": {}})
    theirs = SimpleNamespace(moves_by_turn={"mine": {}, "theirs": {"farigiraf": {4: "trickroom"}}})
    assert battle.setup_spent(sheet, "trickroom", ours) and battle.setup_spent(sheet, "trickroom", None)  # no log: assume ours
    assert not battle.setup_spent(sheet, "trickroom", theirs)  # a second Trick Room ends THEIR room: still worth playing
    assert not battle.setup_spent(sheet, "tailwind", ours) and not battle.setup_spent(sheet, "icywind", ours)
    sheet.notes.append("Our Tailwind is up (speed doubled).")
    sheet.trick_room = False
    assert battle.setup_spent(sheet, "tailwind", ours) and not battle.setup_spent(sheet, "trickroom", ours)
