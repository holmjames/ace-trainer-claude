"""foe_aim_split: lone_foe_split for TWO of theirs acting: each foe's single-target hit lands on one of our slots.

Live, Oct 15 (3ff60171) T6: our Hatterene (41%, speed 44) and Gyarados (100%, speed 102) against Incineroar (4%, speed
80, a fresh Fake Out) and Iron Bundle (22%, speed 309). Incineroar's Flare Blitz / Knock Off and Iron Bundle's Hydro Pump
can KO Hatterene first, Iron Bundle's Freeze-Dry Gyarados, so survival_factor (the worst threat at a slot, counted as
certain) priced BOTH slots dead: the double spread attack (Dazzling Gleam + Earthquake, each KOs both foes) scored 0.0,
row 0 was a second Protect in a row (1 in 3) at 14.9 and the gate played it over the model's double attack; the Protect
failed and we lost. With the key on, each foe's KO counts at a slot as often as that foe aims there, the two foes as
independent events."""
import pytest

from agent.pokemon import battle, data, tuning
from test_hp_power_first import _state
from test_reliability_f import _position

MATCH = "3ff60171"


@pytest.fixture(scope="module")
def t6():
    return _state(MATCH, 0, 6)


def _sheet(state, key: float, **more) -> "battle.TurnSheet":
    agent, template, obs = state
    return battle.build_sheet(template, obs, agent.memory, {**agent.params, "foe_aim_split": key, **more})


def _rows(sheet) -> list[tuple]:
    return [(c["name"], c["slot_0"], c["slot_1"], c["score"]) for c in sheet.candidates]


def _survival(sheet, slot: int) -> float:
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    return battle.survival_factor(slot, sheet, 0, rank, None, tuning.DEFAULTS)


def test_default_is_off():
    assert tuning.DEFAULTS["foe_aim_split"] == 0.0


def test_live_t6_off_is_57a150f(t6):
    """Off: the rows logged live (and 57a150f's in the code-only replay): the double attack at 0.0 under the Protect."""
    sheet = _sheet(t6, 0.0)
    assert not sheet.foe_aim and not sheet.lone_aim
    assert [(c["name"], c["score"]) for c in sheet.candidates] == [
        ("protect_threatened", 14.9), ("best_attacks", 0.0), ("finish_the_other", 0.0), ("taunt_status", 0.0)]
    assert _survival(sheet, 0) == 0.0 and _survival(sheet, 1) == 0.0


def test_live_t6_on_the_double_attack_leads(t6):
    """On: Incineroar's best hit is worth most into Hatterene (0.8 there), Iron Bundle's into Gyarados (Freeze-Dry x4,
    0.8 there). Gyarados survives Iron Bundle 0.2 of the time and Incineroar's fresh Fake Out (0.2 at it) 0.8: 0.16.
    Hatterene: (1 - 0.8) for Incineroar x (1 - 0.2 x 0.8 Hydro Pump's accuracy) for Iron Bundle x 0.2 (the Fake Out
    aimed there 0.8). The double spread attack is row 0, finishing with Psychic next, the repeat Protect third."""
    sheet = _sheet(t6, 1.0)
    assert sheet.foe_aim == {1: {0: 0.8, 1: 0.2}, 2: {0: 0.2, 1: 0.8}} and not sheet.lone_aim
    assert _survival(sheet, 1) == pytest.approx(0.2 * 0.8)
    assert _survival(sheet, 0) == pytest.approx(0.2 * (1 - 0.2 * 0.8) * 0.2)
    assert [(c["name"], c["score"]) for c in sheet.candidates] == [
        ("best_attacks", 30.1), ("finish_the_other", 29.6), ("protect_threatened", 17.7), ("focus_fire", 12.8)]
    top = sheet.candidates[0]
    assert (top["slot_0"], top["slot_1"]) == ({"option": 1, "target": 0}, {"option": 1, "target": 0})
    assert "dazzlinggleam" in top["why"] and "earthquake" in top["why"]


def test_a_fresh_fake_out_lands_at_its_flinch_aim(t6):
    """Fake Out is one target: a fresh one's hit counts at a slot as often as the flinch is aimed there (flinch_risk)."""
    sheet = _sheet(t6, 1.0)
    assert sheet.flinch_risk == {0: 0.8, 1: 0.2}
    assert [battle._lone_aim(sheet, 1, {"move": "fakeout", "into_slot": n}) for n in (0, 1)] == pytest.approx([0.8, 0.2])
    assert battle._lone_aim(_sheet(t6, 0.0), 1, {"move": "fakeout", "into_slot": 1}) == 1.0


def test_two_foes_acting_off_pinned_to_57a150f():
    """37a1ee8a T9 (Incineroar and Basculegion both act): off, the rows 57a150f gives (the replay's rows)."""
    agent, template, obs = _state("37a1ee8a", 0, 9)
    off = battle.build_sheet(template, obs, agent.memory, {**agent.params, "foe_aim_split": 0.0})
    assert not off.foe_aim
    assert [(c["name"], c["score"]) for c in off.candidates] == [
        ("protect_threatened", 40.6), ("fake_out_setup", 29.1), ("finish_the_other", 10.6), ("focus_fire", 4.6)]


def test_a_lone_foe_is_lone_foe_split_exactly():
    """37a1ee8a T10 (Basculegion their last): both keys on give lone_foe_split's rows and survival exactly."""
    state = _state("37a1ee8a", 0, 10)
    lone, both = _sheet(state, 0.0, lone_foe_split=1.0), _sheet(state, 1.0, lone_foe_split=1.0)
    assert not both.foe_aim and both.lone_aim == lone.lone_aim == {"position": 2, "aim": {0: 0.5, 1: 0.5}}
    assert _rows(both) == _rows(lone)
    assert [_survival(both, n) for n in (0, 1)] == [_survival(lone, n) for n in (0, 1)]


def test_a_lone_foe_beside_a_sleeper_is_lone_foe_split_exactly(monkeypatch):
    """c63f19be T5: Iron Hands asleep with no chance to wake, Incineroar acts alone: two foes on the field, one acting;
    both keys on give lone_foe_split's rows (the sleeper's threat counts 0 either way)."""
    monkeypatch.setitem(tuning.DEFAULTS, "lone_foe_split", 1.0)
    monkeypatch.setitem(tuning.DEFAULTS, "foe_aim_split", 0.0)
    _, lone, _, _ = _position("c63f19be", 0, 5)
    monkeypatch.setitem(tuning.DEFAULTS, "foe_aim_split", 1.0)
    _, both, _, _ = _position("c63f19be", 0, 5)
    assert both.foe_wake == {1: 0.0} and not both.foe_aim and both.lone_aim == lone.lone_aim
    assert _rows(both) == _rows(lone)


def test_a_spread_ko_counts_in_full(t6, monkeypatch):
    """A spread hit reaches both of ours: T6 with Iron Bundle on Blizzard (a spread KO 'possible' on Hatterene before it
    moves) and Incineroar on Rock Slide (no KO): with the key on Iron Bundle is aimed, yet its Blizzard counts in full
    and survival and the rows are the key-off ones."""
    original = battle.build_mon
    moves = {"ironbundle": ("blizzard", "protect"), "incineroar": ("rockslide", "protect")}

    def spread_only(summary, card, *, side, position, item_lost=False):
        mon = original(summary, card, side=side, position=position, item_lost=item_lost)
        if mon is not None and side == "theirs" and battle.species_key(mon.species) in moves:
            mon.moves = [{"id": m, **data.move_info(m)} for m in moves[battle.species_key(mon.species)]]
        return mon

    monkeypatch.setattr(battle, "build_mon", spread_only)
    off, on = _sheet(t6, 0.0), _sheet(t6, 1.0)
    blizzard = next(h for t in on.threats for h in t["hits"] if h["move"] == "blizzard" and h["into_slot"] == 0)
    assert blizzard["ko"] == "possible" and 2 in on.foe_aim and on.foe_aim[2][0] < 1.0
    assert battle._lone_aim(on, 2, blizzard) == 1.0
    assert 0.0 < _survival(on, 0) == _survival(off, 0) < 1.0
    assert _rows(on) == _rows(off)


def test_a_partial_weight_pulls_each_aim_halfway_back_to_both_slots():
    """foe_aim_split 0.5: an aim of 0.8 / 0.2 becomes 0.9 / 0.6 (1 - 0.5 x (1 - aim)); 1.0 keeps the full split."""
    from agent.pokemon import battle as _b
    full, half = {}, {}
    for weight, out in ((1.0, full), (0.5, half)):
        P = {**tuning.DEFAULTS, "foe_aim_split": weight}
        sheet = _b.TurnSheet()
        class _T(dict):
            pass
        sheet.threats = [{"position": 1, "species": "x", "hits": []}]
        orig = _b._foe_aim
        _b._foe_aim = lambda t, ours, P: {0: 0.8, 1: 0.2}
        try:
            _b._foe_aim_split(sheet, {}, {1: None, 2: None}, {**P, "foe_sleep": 0.0})
        finally:
            _b._foe_aim = orig
        out.update(sheet.foe_aim.get(1, {}))
    assert full == {0: 0.8, 1: 0.2}
    assert half == pytest.approx({0: 0.9, 1: 0.6})
