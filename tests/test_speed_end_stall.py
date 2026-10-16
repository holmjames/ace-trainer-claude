"""speed_end_stall: a Protect buys the turn in which our slot moves first, once THEIR speed advantage ends.

Live, Oct 15 (f73663d8) T5, Trick Room's last turn: our Baxcalibur (speed 152, the fastest, so last under the room) and
Archaludon (36%) against Dragonite (151, Loaded Dice Multiscale) and Urshifu-Rapid-Strike (149). Dragonite's Scale Shot
KOs Baxcalibur (113-173%) before it moves only because of the room; next turn Baxcalibur moves first and its Icicle
Spear (261-409%) KOs Dragonite. Baxcalibur had Protected the turn before, so the 1-in-3 Protect row (128.3) trailed
finish_the_other (146.6); we attacked, Baxcalibur fell first and the game was lost. With the key on, that slot's
Protect rows gain the key x its best same-priority attack into that foe, at the Protect's odds (0.33 ** streak)."""
import copy

import pytest

from agent.pokemon import battle, tuning
from test_gate_repeat_protect import READ
from test_hp_power_first import _state
from test_model_path import _answer, _position

MATCH = "f73663d8"
LIVE_ROWS = [("finish_the_other", 146.6), ("protect_threatened", 128.3), ("focus_fire", 79.4), ("best_attacks", 68.7)]
REASON = "Trick Room ends after this turn: next turn baxcalibur moves before dragonite"


@pytest.fixture(scope="module")
def t5():
    return _state(MATCH, 0, 5)


@pytest.fixture(scope="module")
def tw5():
    """d9405f02 T5: their Tailwind's last turn (Gholdengo 298, Dragonite 202 under it; our Garchomp 169, Cresselia 105)."""
    return _state("d9405f02", 0, 5)


def _sheet(state, key: float, obs: dict | None = None) -> "battle.TurnSheet":
    agent, template, live_obs = state
    return battle.build_sheet(template, obs or live_obs, copy.deepcopy(agent.memory), {**agent.params, "speed_end_stall": key})


def _scores(sheet) -> list[tuple]:
    return [(c["name"], c["score"]) for c in sheet.candidates]


def _rows(sheet) -> list[tuple]:
    return [(c["name"], c["slot_0"], c["slot_1"], c["score"], c["why"]) for c in sheet.candidates]


def _with(obs: dict, *, fields: dict | None = None, their_side: dict | None = None, spe: dict | None = None) -> dict:
    out = copy.deepcopy(obs)
    if fields is not None:
        out["fields"] = fields
    if their_side is not None:
        out["opponent_side_conditions"] = their_side
    for species, stage in (spe or {}).items():
        for entry in out["team"].values():
            if entry.get("species") == species or entry.get("name") == species:
                entry.setdefault("boosts", {})["spe"] = stage
    return out


def test_default_is_off():
    assert tuning.DEFAULTS["speed_end_stall"] == 0.0


def test_live_t5_off_is_the_live_rows(t5):
    sheet = _sheet(t5, 0.0)
    assert sheet.timers == {"trickroom": 1} and sheet.stall_reasons == []
    assert _scores(sheet) == LIVE_ROWS
    assert not any("after this turn" in n for n in sheet.notes)


def test_live_t5_on_the_protect_leads(t5):
    """On: Icicle Spear into Dragonite next turn is worth 181.5 (a sure KO of its biggest threat); at 1 in 3 the repeat
    Protect gains 59.9 and leads, the attack rows unchanged."""
    off, on = _sheet(t5, 0.0), _sheet(t5, 1.0)
    assert on.stall_reasons == [REASON]
    assert any(n.startswith(REASON + ", so a Protect from slot 0") and "1 in 3" in n for n in on.notes)
    assert _scores(on) == [("protect_threatened", 188.2), ("finish_the_other", 146.6), ("focus_fire", 79.4), ("best_attacks", 68.7)]
    top = on.candidates[0]
    assert top["slot_0"] == {"option": 3, "target": 0} and top["slot_1"] == {"option": 0, "target": 2}  # Protect + Body Press
    assert top["why"].endswith("; " + REASON)
    assert [r for r in _rows(on) if r[0] != "protect_threatened"] == [r for r in _rows(off) if r[0] != "protect_threatened"]


def test_the_stall_scales_with_the_key(t5):
    half = _sheet(t5, 0.5)
    assert _scores(half)[0] == ("protect_threatened", 158.3)


def test_trick_room_not_in_its_last_turn_changes_nothing(t5):
    obs = _with(t5[2], fields={"GRASSY_TERRAIN": 0, "TRICK_ROOM": 2})  # two turns left, counting this one
    off, on = _sheet(t5, 0.0, obs), _sheet(t5, 1.0, obs)
    assert on.timers == {"trickroom": 2} and on.stall_reasons == []
    assert _rows(on) == _rows(off) and _scores(off) == LIVE_ROWS


def test_a_slot_already_first_under_the_room_changes_nothing(t5):
    """Baxcalibur at -1 Speed (101): slower than both foes, it moves before them under the room and would not outspeed
    them without it; nothing lands first because of the room, so nothing is stalled for."""
    obs = _with(t5[2], spe={"Baxcalibur": -1})
    off, on = _sheet(t5, 0.0, obs), _sheet(t5, 1.0, obs)
    assert on.stall_reasons == [] and _rows(on) == _rows(off)


def test_their_tailwind_last_turn_live_d9405f02_t5(tw5):
    """Their Tailwind's last turn: Dragonite's Scale Shot (202 under it, 101 without) KOs Garchomp (169) first. Live the
    code attacked, Shadow Ball + Scale Shot KO'd Garchomp before it moved. On: Garchomp's fresh Protect leads."""
    off, on = _sheet(tw5, 0.0), _sheet(tw5, 1.0)
    assert off.timers == on.timers == {"their_tailwind": 1}
    assert off.stall_reasons == [] and on.stall_reasons == ["their Tailwind ends after this turn: next turn garchomp moves before dragonite"]
    assert _scores(off)[:2] == [("best_attacks", 36.5), ("protect_threatened", 31.4)]
    assert _scores(on)[:2] == [("protect_threatened", 63.6), ("best_attacks", 36.5)]


def test_their_tailwind_with_turns_left_changes_nothing(tw5):
    obs = _with(tw5[2], their_side={"TAILWIND": 3})  # this turn and next
    off, on = _sheet(tw5, 0.0, obs), _sheet(tw5, 1.0, obs)
    assert on.timers == {"their_tailwind": 2} and on.stall_reasons == [] and _rows(on) == _rows(off)


def test_a_slot_too_slow_even_without_their_tailwind_changes_nothing(tw5):
    """Garchomp at -2 Speed (84.5): slower than Dragonite (101) and Gholdengo (149) even once their Tailwind is gone."""
    obs = _with(tw5[2], spe={"Garchomp": -2})
    off, on = _sheet(tw5, 0.0, obs), _sheet(tw5, 1.0, obs)
    assert on.stall_reasons == [] and _rows(on) == _rows(off)


def test_the_gate_still_lets_the_models_attack_stand():
    """gate_repeat_protect: with the key on, row 0 is Baxcalibur's repeat Protect; the model's finish_the_other (146.6,
    more than the margin below 188.2) still plays, because the row 0 it would be sent back to rests on a repeat Protect."""
    agent, sheet, template, choice = _position(MATCH, 0, 5, params={"speed_end_stall": 1.0, "gate_repeat_protect": 1.0})
    rows = sheet.candidates
    assert rows[0]["name"] == "protect_threatened" and rows[0]["score"] == 188.2
    row = next(c["row"] for c in rows if c["name"] == "finish_the_other")
    meta: dict = {}
    value = agent._resolve_turn_answer(_answer(row), sheet, choice, READ, meta, template=template)
    assert meta.get("repeat_protect_gate") is True and "stands: the best row" in meta["resolution"]
    assert value == choice.build({"slot_0": rows[row]["slot_0"], "slot_1": rows[row]["slot_1"]})
    # ... and the model's pick of the Protect row plays as row 0
    meta = {}
    assert agent._resolve_turn_answer(_answer(0), sheet, choice, READ, meta, template=template) == \
        choice.build({"slot_0": rows[0]["slot_0"], "slot_1": rows[0]["slot_1"]})
