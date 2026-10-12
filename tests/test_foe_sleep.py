"""foe_sleep: a sleeping (paralysed, frozen) foe's KO threat counts only as often as it acts this turn.

Live, Oct 12 (c63f19be): Torkoal (10%) Yawned Iron Hands on T3 and Incineroar on T4. On T6 Iron Hands had missed one
turn asleep (wakes 1 in 3) and Incineroar had just fallen asleep (cannot move), yet the code's top row Protected Torkoal
a second turn in a row (itself 1 in 3) against Iron Hands' Close Combat and won over the model's pick."""
import pytest

from agent.pokemon import tuning
from test_reliability_f import _position

MATCH = "c63f19be"


@pytest.mark.parametrize("key", [0.0, 1.0])
def test_t5_protect_from_the_awake_incineroar_stands(monkeypatch, key):
    """T5: Iron Hands fell asleep at the end of T4 (wake 0), but Incineroar is awake and its Knock Off KOs Torkoal
    first: the Protect row stays on top either way, now quoting the foe that can actually move."""
    monkeypatch.setitem(tuning.DEFAULTS, "foe_sleep", key)
    _, sheet, _, _ = _position(MATCH, 0, 5)
    assert sheet.foe_wake == {1: 0.0} and sheet.candidates[0]["name"] == "protect_threatened"
    assert ("incineroar's knockoff" if key else "ironhands's closecombat") in sheet.candidates[0]["why"]


def test_t6_attacks_into_two_sleeping_foes(monkeypatch):
    """T6: Iron Hands wakes 1 in 3, Incineroar not at all. Off: a second Protect leads (52.7 vs 37.8). On: Heat Wave +
    Trick Room leads and the Protect row's bonus is a third of a third."""
    monkeypatch.setitem(tuning.DEFAULTS, "foe_sleep", 0.0)
    _, sheet, _, _ = _position(MATCH, 0, 6)
    assert sheet.candidates[0]["name"] == "protect_threatened"
    monkeypatch.setitem(tuning.DEFAULTS, "foe_sleep", 1.0)
    _, sheet, _, _ = _position(MATCH, 0, 6)
    assert sheet.foe_wake == {1: 0.333, 2: 0.0}
    assert sheet.candidates[0]["name"] == "best_attacks" and "heatwave" in sheet.candidates[0]["why"]
