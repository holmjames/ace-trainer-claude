"""Opponent Choice locks read from the live server's protocol log, and pivots that would die on entry."""

from agent.pokemon.memory import MatchMemory

LOG = [
    "|turn|1",
    "|move|p2a: Urshifu|Surging Strikes|p1a: Garchomp",
    "|-damage|p1a: Garchomp|40/100",
    "|move|p2b: Miraidon|Draco Meteor|p1b: Salamence",
    "|faint|p1b: Salamence",
    "|switch|p2b: Lunala|Lunala, L50|100/100",
    "|turn|2",
]


def test_protocol_log_gives_each_opponents_last_move_since_it_entered():
    mem = MatchMemory()
    mem.observe_protocol({"protocol_log": LOG, "team": {"p1: Garchomp": {"species": "garchomp"}, "p1: Salamence": {"species": "salamence"}}})
    assert mem.opp_last_move["urshifu"] == "surgingstrikes"
    assert mem.opp_last_move["lunala"] is None  # just switched in: nothing to be locked into
    assert "garchomp" not in mem.opp_last_move  # our own moves are ignored


def test_protocol_log_is_ignored_without_a_log_or_a_side():
    mem = MatchMemory()
    mem.observe_protocol({"team": {"p1: Garchomp": {"species": "garchomp"}}})
    mem.observe_protocol({"protocol_log": LOG, "team": {}})
    assert mem.opp_last_move == {}
