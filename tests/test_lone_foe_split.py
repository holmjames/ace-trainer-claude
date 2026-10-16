"""lone_foe_split: their ONE acting Pokémon's single-target hit lands on one of our slots, not both.

Live, Oct 15 (37a1ee8a) T10: our Iron Hands (40%, speed 73) and Arcanine (36%, speed 116) against their last Pokémon,
Basculegion (27%, speed 130). Its Wave Crash, Liquidation and +1 Aqua Jet can each KO one of ours before it moves, so
survival_factor priced BOTH slots dead and the double attack (Wild Charge 480-567% + Snarl 91-110%) scored 0.0; row 0 was
a second Protect in a row (1 in 3) and the gate played it over the model's double attack ("whichever survives finishes
it"). With the key on, each slot is lost only as often as Basculegion aims there."""
import pytest

from agent.pokemon import battle, data, tuning
from test_hp_power_first import _state
from test_reliability_f import _position

MATCH = "37a1ee8a"


@pytest.fixture(scope="module")
def t10():
    return _state(MATCH, 0, 10)


@pytest.fixture(scope="module")
def t9():
    return _state(MATCH, 0, 9)


def _sheet(state, key: float) -> "battle.TurnSheet":
    agent, template, obs = state
    return battle.build_sheet(template, obs, agent.memory, {**agent.params, "lone_foe_split": key})


def _rows(sheet) -> list[tuple]:
    return [(c["name"], c["slot_0"], c["slot_1"], c["score"]) for c in sheet.candidates]


def _survival(sheet, slot: int, P: dict) -> float:
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    return battle.survival_factor(slot, sheet, 0, rank, None, P)


def test_default_is_on():
    assert tuning.DEFAULTS["lone_foe_split"] == 1.0


def test_live_t10_off_is_deploy_8(t10):
    """Off: the rows logged live, the double attack at 0.0 under a second Protect."""
    sheet = _sheet(t10, 0.0)
    assert not sheet.lone_aim
    assert [(c["name"], c["score"]) for c in sheet.candidates] == [("protect_threatened", 14.9), ("best_attacks", 0.0)]
    assert _survival(sheet, 0, {**tuning.DEFAULTS, "lone_foe_split": 0.0}) == 0.0
    assert _survival(sheet, 1, {**tuning.DEFAULTS, "lone_foe_split": 0.0}) == 0.0


def test_live_t10_on_the_double_attack_leads(t10):
    """On: Basculegion's best hit is worth about the same into either (Iron Hands 40%, Arcanine 36%: within 10%), so it
    aims at each half the time; each slot acts half the time and the double attack that KOs it is row 0."""
    sheet = _sheet(t10, 1.0)
    assert sheet.lone_aim == {"position": 2, "aim": {0: 0.5, 1: 0.5}}
    assert _survival(sheet, 0, tuning.DEFAULTS) == pytest.approx(0.5)
    assert _survival(sheet, 1, tuning.DEFAULTS) == pytest.approx(0.5)
    assert [(c["name"], c["score"]) for c in sheet.candidates] == [("best_attacks", 55.6), ("protect_threatened", 49.7)]
    top = sheet.candidates[0]
    assert (top["slot_0"], top["slot_1"]) == ({"option": 2, "target": 2}, {"option": 2, "target": 0})
    assert "wildcharge" in top["why"] and "snarl" in top["why"]


def test_two_foes_acting_change_nothing(t9):
    """T9: Incineroar and Basculegion both act (four LETHAL warnings): the rows are the key-off rows exactly."""
    off, on = _sheet(t9, 0.0), _sheet(t9, 1.0)
    assert len(on.threats) == 2 and not on.lone_aim
    assert sum(w.startswith("LETHAL") for w in on.warnings) >= 1
    assert _rows(on) == _rows(off)


def test_a_lone_foes_spread_ko_counts_in_full(t10, monkeypatch):
    """A spread hit reaches both of ours: T10 with Basculegion's attacks swapped for Muddy Water (a spread KO on
    Arcanine before it moves) prices Arcanine's survival with the key on as off, and the rows do not change."""
    original = battle.build_mon

    def spread_only(summary, card, *, side, position, item_lost=False):
        mon = original(summary, card, side=side, position=position, item_lost=item_lost)
        if mon is not None and side == "theirs" and battle.species_key(mon.species) == "basculegion":
            mon.moves = [{"id": m, **data.move_info(m)} for m in ("muddywater", "protect")]
        return mon

    monkeypatch.setattr(battle, "build_mon", spread_only)
    off, on = _sheet(t10, 0.0), _sheet(t10, 1.0)
    hits = [h for t in on.threats for h in t["hits"]]
    assert hits and {h["move"] for h in hits} == {"muddywater"}
    assert any(h["into_slot"] == 1 and h["ko"] == "guaranteed" for h in hits)
    assert on.lone_aim  # the lone foe is there; its only attack is spread
    # Arcanine is lost whenever Muddy Water hits (85% accurate), not half the time
    assert _survival(on, 1, tuning.DEFAULTS) == _survival(off, 1, {**tuning.DEFAULTS, "lone_foe_split": 0.0}) == pytest.approx(0.15)
    assert _rows(on) == _rows(off)


@pytest.mark.parametrize("key", [0.0, 1.0])
def test_c63f19be_t5_the_lone_foe_aims_at_dragonite(monkeypatch, key):
    """Oct 12 (c63f19be) T5: Iron Hands fell asleep at the end of T4 (cannot move), so Incineroar acts alone. Its Knock
    Off KOs Torkoal (10%) first, but its best hit is worth more into Dragonite (0.8 there, 0.2 at Torkoal). Off: Torkoal
    Protects (row 0). On: both attack. Live, Torkoal Protected and Incineroar Flare Blitzed Dragonite."""
    monkeypatch.setitem(tuning.DEFAULTS, "lone_foe_split", key)
    _, sheet, _, _ = _position("c63f19be", 0, 5)
    assert sheet.foe_wake == {1: 0.0}
    if key:
        assert sheet.lone_aim == {"position": 2, "aim": {0: 0.2, 1: 0.8}}
        assert [(c["name"], c["score"]) for c in sheet.candidates[:2]] == [("best_attacks", 105.0), ("protect_threatened", 92.0)]
    else:
        assert not sheet.lone_aim
        assert [(c["name"], c["score"]) for c in sheet.candidates[:2]] == [("protect_threatened", 94.0), ("best_attacks", 49.0)]
