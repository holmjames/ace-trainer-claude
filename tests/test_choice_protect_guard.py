"""choice_protect_guard: a Protect from a slot holding an unlocked Choice item locks it into Protect.

Live, Oct 12 (56f63b32) T11: our Choice Scarf Landorus (24%) under Kingambit's LETHAL Sucker Punch, Cresselia beside it,
no bench. The sheet's note said the Protect LOCKS it, yet the Protect row was row 0 (67.8 against Earthquake + Moonblast's
22.8) and the model took it. They doubled Cresselia; on T12 Landorus could only Protect again, which failed, and the
game was lost. With the key on, a lock-Protect row is credited nothing for the Protect (the lock spends the turn it
buys) and stays only when it still leads every row where that slot makes another move by protect_bonus_possible; a
proposed Protect from that slot is refused when no row lists one, and the chosen row plays."""
import pytest

from examples.llm.base import InvalidChoice

from test_model_path import _answer, _index, _position

MATCH = "56f63b32"
READ = {"top_column": {1: "attack", 2: "attack"}}


def _protects(sheet, template, slot: int) -> list[dict]:
    protect = _index(template, slot, "protect")
    return [c for c in sheet.candidates if c[f"slot_{slot}"]["option"] == protect]


def test_t11_off_the_protect_row_leads_as_on_deploy_6():
    _, sheet, template, _ = _position(MATCH, 0, 11, {"choice_protect_guard": 0.0})
    assert sheet.choice_protect == {0}
    top = sheet.candidates[0]
    assert top["name"] == "protect_threatened" and top["score"] == 67.8
    assert top["slot_0"]["option"] == _index(template, 0, "protect") and top["slot_1"]["option"] == _index(template, 1, "moonblast")
    assert [c["name"] for c in sheet.candidates] == ["protect_threatened", "best_attacks"]


def test_t11_on_no_row_protects_the_scarf_holder():
    """Credited nothing for its Protect, the row is worth 22.8, level with Earthquake + Moonblast: not a wide lead, so it
    goes and the attack row is row 0. The note tells the model so."""
    _, sheet, template, _ = _position(MATCH, 0, 11)
    assert sheet.choice_protect == {0} and not _protects(sheet, template, 0)
    top = sheet.candidates[0]
    assert top["name"] == "best_attacks" and top["score"] == 22.8
    assert top["slot_0"]["option"] == _index(template, 0, "earthquake") and top["slot_1"]["option"] == _index(template, 1, "moonblast")
    assert any(n.startswith("Our landorustherian holds a Choice item: Protect LOCKS it") and "refused when no row does" in n for n in sheet.notes)


def test_t11_a_lock_protect_that_leads_widely_stays_credited_nothing_for_the_protect():
    """The exception, by its margin: with protect_bonus_possible at -1 the 22.8 clears 22.8 - 1, so the row stays, at
    22.8 (not 67.8), and says why."""
    _, sheet, template, _ = _position(MATCH, 0, 11, {"protect_bonus_possible": -1.0})
    (row,) = _protects(sheet, template, 0)
    assert row["score"] == 22.8 and "Choice item locks it into Protect: listed because" in row["why"]


def test_t12_a_holder_already_locked_into_protect_is_not_guarded():
    """T12: locked, Landorus is offered Protect alone; nothing else exists, so the guard has nothing to say."""
    _, sheet, _, _ = _position(MATCH, 0, 12)
    assert sheet.choice_protect == set()


@pytest.mark.parametrize("read", [None, ("attack", "attack")])
def test_t11_the_live_protect_proposed_as_an_override_is_refused_and_row_0_plays(read):
    """The live answer (Protect + Moonblast) as an override: refused, read or no read (the read matches the code's
    column), with E2's rejection line, and row 0 plays. A row number that is not a row makes it an invalid answer."""
    agent, sheet, template, choice = _position(MATCH, 0, 11)
    live = {"slot_0": {"option": _index(template, 0, "protect"), "target": 0},
            "slot_1": {"option": _index(template, 1, "moonblast"), "target": 1}}
    meta: dict = {}
    value = agent._resolve_turn_answer(_answer(0, override=live, prediction=read), sheet, choice, READ, meta)
    assert meta["resolution"] == ("override rejected: slot 0's Choice item would lock it into Protect (next turn only a 1-in-3 "
                                  "Protect or a switch, no attack); row 0 plays")
    top = sheet.candidates[0]
    assert value == choice.build({"slot_0": top["slot_0"], "slot_1": top["slot_1"]}) and value["slot_0"]["move_id"] == "earthquake"
    with pytest.raises(InvalidChoice):
        agent._resolve_turn_answer(_answer(5, override=live), sheet, choice, READ, {})


def test_t11_off_the_same_override_plays_as_the_row_it_is():
    agent, sheet, template, choice = _position(MATCH, 0, 11, {"choice_protect_guard": 0.0})
    live = {"slot_0": {"option": _index(template, 0, "protect"), "target": 0},
            "slot_1": {"option": _index(template, 1, "moonblast"), "target": 1}}
    meta: dict = {}
    value = agent._resolve_turn_answer(_answer(0, override=live), sheet, choice, READ, meta)
    assert meta["resolution"] == "override accepted (same as a row)" and value["slot_0"]["move_id"] == "protect"


def test_a_listed_lock_protect_row_lets_the_override_through():
    """When the rows keep a lock-Protect for that slot (the exception), a proposal that Protects it is judged by the
    margin as before: here it is that row."""
    agent, sheet, template, choice = _position(MATCH, 0, 11, {"protect_bonus_possible": -1.0})
    (row,) = _protects(sheet, template, 0)
    meta: dict = {}
    value = agent._resolve_turn_answer(_answer(0, override={"slot_0": row["slot_0"], "slot_1": row["slot_1"]}), sheet, choice, READ, meta)
    assert meta["resolution"].startswith("override accepted") and value["slot_0"]["move_id"] == "protect"
