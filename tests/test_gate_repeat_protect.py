"""gate_repeat_protect: the row gate never sends the model back to a row 0 that rests on a repeat Protect.

Live, 189 model-decided turns: the gate overruled the model 9 times, and 4 of those overrules forced a Protect on a
streak (1 in 3) over the model's attack: 3ff60171 T6, 42ae2b89 T6, c63f19be T6 and 37a1ee8a T10. Three of those games
were lost. With the key on, a listed row the model picked plays when row 0 needs a repeat Protect and the model's row
does not; every other rejection is unchanged."""
import pytest

from test_model_path import _answer, _position

READ = {"top_column": {1: "attack", 2: "attack"}}


# 37a1ee8a T10 is a lone-foe endgame: lone_foe_split (on by default) already makes the double attack row 0 there, so
# the gate is exercised with it off (the deploy #8 rows).
BASE = {"37a1ee8a": {"lone_foe_split": 0.0}}


def _gate(prefix: str, turn: int, row_name: str, params: dict | None = None):
    agent, sheet, template, choice = _position(prefix, 0, turn, params={**BASE.get(prefix, {}), **(params or {})})
    rows = sheet.candidates
    row = next(c["row"] for c in rows if c["name"] == row_name)
    meta: dict = {}
    value = agent._resolve_turn_answer(_answer(row), sheet, choice, READ, meta, template=template)
    return rows, row, value, meta, choice


@pytest.mark.parametrize("prefix,turn", [("3ff60171", 6), ("37a1ee8a", 10)])
def test_on_the_models_attack_row_plays_over_a_repeat_protect(prefix, turn):
    rows, row, value, meta, choice = _gate(prefix, turn, "best_attacks", {"gate_repeat_protect": 1.0})
    assert row > 0 and "Protect used 1 turn(s) in a row already" in rows[0]["why"]
    assert meta.get("repeat_protect_gate") is True and "stands: the best row" in meta["resolution"]
    assert value == choice.build({"slot_0": rows[row]["slot_0"], "slot_1": rows[row]["slot_1"]})


@pytest.mark.parametrize("prefix,turn", [("3ff60171", 6), ("37a1ee8a", 10)])
def test_off_the_old_gate_sends_the_model_back_to_row_0(prefix, turn):
    rows, row, value, meta, choice = _gate(prefix, turn, "best_attacks", {"gate_repeat_protect": 0.0})
    assert "rejected" in meta["resolution"] and not meta.get("repeat_protect_gate")
    assert value == choice.build({"slot_0": rows[0]["slot_0"], "slot_1": rows[0]["slot_1"]})


def test_without_the_template_nothing_changes():
    agent, sheet, template, choice = _position("3ff60171", 0, 6, params={"gate_repeat_protect": 1.0})
    rows = sheet.candidates
    row = next(c["row"] for c in rows if c["name"] == "best_attacks")
    meta: dict = {}
    agent._resolve_turn_answer(_answer(row), sheet, choice, READ, meta)
    assert "rejected" in meta["resolution"]


def test_with_lone_foe_split_on_37a1ee8a_t10_needs_no_gate_at_all():
    agent, sheet, template, choice = _position("37a1ee8a", 0, 10)
    assert sheet.candidates[0]["name"] == "best_attacks"
