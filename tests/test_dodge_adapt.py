"""dodge_adapt: once the opponent has dodged 2+ of our single-target attacks this match (Protected successfully or
switched out before the hit), a heavy single-target hit into a foe that can dodge counts x (1 - dodge_adapt x q).

The tally is read from the protocol log only (fresh memory each match, nothing keyed to who the opponent is). Live
position: 3666fb1d T8 (seat 0), where their Flutter Mane and Gyarados had dodged 6 of our 10 targeted attacks (Protect
on T1, T3, T5, T7; a switch out on T4)."""
import copy

import pytest

from agent.pokemon import battle, tuning
from agent.pokemon.memory import MatchMemory
from test_hp_power_first import _state

MATCH = "3666fb1d"


def _log(lines: list[str]) -> dict:
    """An observation carrying a protocol log, our side p1."""
    return {"team": {"p1: Iron Hands": {}, "p1: Flutter Mane": {}}, "protocol_log": lines}


START = [
    "|player|p1|us",
    "|player|p2|them",
    "|poke|p1|Iron Hands, L50",
    "|poke|p1|Flutter Mane, L50",
    "|poke|p2|Kingambit, L50",
    "|poke|p2|Indeedee-F, L50, F",
    "|poke|p2|Dondozo, L50",
    "|start",
    "|switch|p1a: Iron Hands|Iron Hands, L50|100/100",
    "|switch|p1b: Flutter Mane|Flutter Mane, L50|100/100",
    "|switch|p2a: Kingambit|Kingambit, L50|100/100",
    "|switch|p2b: Dondozo|Dondozo, L50|100/100",
]


def _memory(lines: list[str]) -> MatchMemory:
    memory = MatchMemory()
    memory.observe_protocol(_log(START + lines))
    return memory


def test_defaults_are_off():
    assert tuning.DEFAULTS["dodge_adapt"] == 0.0


def test_the_tally_reads_protects_and_switch_outs():
    memory = _memory([
        "|turn|1",  # Kingambit Protects our Close Combat: a dodge
        "|move|p2a: Kingambit|Protect|p2a: Kingambit",
        "|-singleturn|p2a: Kingambit|Protect",
        "|move|p1a: Iron Hands|Close Combat|p2a: Kingambit",
        "|-activate|p2a: Kingambit|move: Protect",
        "|move|p1b: Flutter Mane|Dazzling Gleam|p2a: Kingambit|[spread] p2a,p2b",  # a spread move: never counted
        "|turn|2",  # Kingambit switches out before our Close Combat lands: a dodge (the hit meets Indeedee)
        "|switch|p2a: Indeedee|Indeedee-F, L50, F|100/100",
        "|move|p1a: Iron Hands|Close Combat|p2a: Indeedee",
        "|-damage|p2a: Indeedee|20/100",
        "|turn|3",  # Indeedee takes the hit: aimed, no dodge
        "|move|p1a: Iron Hands|Drain Punch|p2a: Indeedee",
        "|-damage|p2a: Indeedee|0 fnt",
        "|faint|p2a: Indeedee",
        "|switch|p2a: Kingambit|Kingambit, L50|100/100",  # a replacement after a faint: no dodge
        "|turn|4",  # Kingambit's second Protect in a row fails: no dodge
        "|move|p2a: Kingambit|Protect||[still]",
        "|-fail|p2a: Kingambit",
        "|move|p1a: Iron Hands|Close Combat|p2a: Kingambit",
        "|-damage|p2a: Kingambit|10/100",
        "|move|p2b: Dondozo|Wave Crash|p1a: Iron Hands",
        "|turn|5",
    ])
    assert memory.dodges == [(1, "kingambit", "protect"), (2, "kingambit", "switch")]
    assert memory.targeted_attacks == [(1, "kingambit"), (2, "kingambit"), (3, "indeedeef"), (4, "kingambit")]
    assert memory.dodge_tally() == (2, 4)


def test_a_switch_after_our_move_is_no_dodge():
    """A switch after a move line (a pivot, a replacement) is not a dodge of the hit that already landed."""
    memory = _memory([
        "|turn|1",
        "|move|p1a: Iron Hands|Close Combat|p2a: Kingambit",
        "|-damage|p2a: Kingambit|40/100",
        "|move|p2b: Dondozo|Wave Crash|p1a: Iron Hands",
        "|switch|p2a: Indeedee|Indeedee-F, L50, F|100/100",
        "|turn|2",
    ])
    assert memory.dodge_tally() == (0, 1)


class _Foe:
    def __init__(self, species: str) -> None:
        self.species = species


def test_discounts_need_two_dodges_and_a_way_to_dodge():
    P = {**tuning.DEFAULTS, "dodge_adapt": 1.0}
    theirs = {1: _Foe("Kingambit"), 2: _Foe("Dondozo")}
    one = MatchMemory(dodges=[(1, "kingambit", "protect")], targeted_attacks=[(1, "kingambit"), (2, "dondozo")])
    assert battle.dodge_discounts(theirs, one, P, can_protect=lambda f: True, bench=2) == {}
    two = MatchMemory(dodges=[(1, "kingambit", "protect"), (2, "kingambit", "switch")],
                      targeted_attacks=[(1, "kingambit"), (2, "kingambit"), (3, "dondozo"), (4, "dondozo")])
    assert battle.dodge_discounts(theirs, two, P, can_protect=lambda f: True, bench=0) == {"kingambit": 0.5, "dondozo": 0.5}
    # a foe with no usable Protect and no bench to switch to cannot dodge
    only_kingambit = battle.dodge_discounts(theirs, two, P, can_protect=lambda f: f.species == "Kingambit", bench=0)
    assert only_kingambit == {"kingambit": 0.5}
    # q is capped at 0.8, and the key scales it
    all_dodged = MatchMemory(dodges=[(1, "kingambit", "protect"), (2, "kingambit", "switch")],
                             targeted_attacks=[(1, "kingambit"), (2, "kingambit")])
    assert battle.dodge_discounts(theirs, all_dodged, P, can_protect=lambda f: True, bench=1)["kingambit"] == pytest.approx(0.8)
    assert battle.dodge_discounts(theirs, all_dodged, {**P, "dodge_adapt": 0.5}, can_protect=lambda f: True, bench=1)["kingambit"] == pytest.approx(0.4)
    assert battle.dodge_discounts(theirs, all_dodged, tuning.DEFAULTS, can_protect=lambda f: True, bench=1) == {}


def _row(target: int, species: str, lo: float, hi: float) -> dict:
    return {"target": target, "species": species, "side": "theirs", "damage_pct_of_current_hp": [lo, hi]}


def test_only_heavy_single_target_hits_are_discounted():
    P = dict(tuning.DEFAULTS)
    heavy = {"move": "closecombat", "priority": 0, "targets": [_row(1, "Kingambit", 60, 70), _row(2, "Dondozo", 30, 36)]}
    spread = {"move": "rockslide", "priority": 0, "targets": [_row(0, "Kingambit", 40, 48), _row(0, "Dondozo", 15, 18)]}
    dodge = {"kingambit": 0.5, "dondozo": 0.5}
    plain = battle._best_attack({"options": [heavy]}, P=P)
    assert (plain[1]["target"], plain[2]) == (1, pytest.approx(65.0))
    # the heavy hit into the dodger counts half; the light hit into Dondozo (33%, no KO) is not discounted and now leads
    opt, row, value = battle._best_attack({"options": [heavy]}, P=P, dodge=dodge)
    assert (row["target"], value) == (2, pytest.approx(33.0))
    assert battle._best_attack({"options": [heavy]}, P=P, dodge={"kingambit": 0.5})[2] == pytest.approx(33.0)
    # a spread move is unchanged
    assert battle._best_attack({"options": [spread]}, P=P, dodge=dodge)[2] == battle._best_attack({"options": [spread]}, P=P)[2]
    # a KO chance counts as a heavy threat even under 50%
    assert battle.dodge_threatened(0.1, 30.0) and battle.dodge_threatened(0.0, 50.0) and not battle.dodge_threatened(0.0, 49.0)


@pytest.fixture(scope="module")
def t8():
    return _state(MATCH, 0, 8)


def _rows(agent, template, obs, key: float, memory=None) -> list[tuple]:
    sheet = battle.build_sheet(template, obs, memory or agent.memory, {**agent.params, "dodge_adapt": key})
    return [(c["name"], c["slot_0"], c["slot_1"], c["score"]) for c in sheet.candidates], sheet


def test_live_off_is_identical_and_on_discounts_the_dodgers(t8):
    agent, template, obs = t8
    assert agent.memory.dodge_tally() == (6, 10)
    off, off_sheet = _rows(agent, template, obs, 0.0)
    defaults = battle.build_sheet(template, obs, agent.memory, agent.params)
    assert off == [(c["name"], c["slot_0"], c["slot_1"], c["score"]) for c in defaults.candidates]
    assert not [n for n in off_sheet.notes if n.startswith("DODGE READ")]
    on, on_sheet = _rows(agent, template, obs, 1.0)
    assert on != off
    assert [n for n in on_sheet.notes if n.startswith("DODGE READ")]
    # the attack rows lose value (q = 0.6: x0.4 on the heavy single-target hits); a Protect row's own credit stays
    off_scores, on_scores = {(r[0], str(r[1]), str(r[2])): r[3] for r in off}, {(r[0], str(r[1]), str(r[2])): r[3] for r in on}
    assert all(on_scores[k] <= off_scores[k] for k in on_scores if k in off_scores)
    assert on_scores[("finish_the_other", "{'option': 2, 'target': 2}", "{'option': 0, 'target': 1}")] \
        < off_scores[("finish_the_other", "{'option': 2, 'target': 2}", "{'option': 0, 'target': 1}")]


def test_live_one_dodge_changes_nothing(t8):
    agent, template, obs = t8
    memory = copy.copy(agent.memory)
    memory.dodges = agent.memory.dodges[:1]
    assert _rows(agent, template, obs, 1.0, memory)[0] == _rows(agent, template, obs, 0.0, memory)[0]
