"""The switch-heavy practice opponent (agent/versions/pivot.py) and the style statistics it is measured by
(sim/style_stats.py).

Positions are replayed from live fixtures battle request by battle request (a restarted agent: no draft or Team
Preview, the cards come from the catalog), so a test costs a second, not a deep Team Preview."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon import data, tuning
from agent.pokemon.memory import observation_dict
from agent.versions import pivot
from sim import style_stats

LIVE = Path(__file__).parent / "fixtures" / "live"
PROBABILITIES = ("switch_p", "double_switch_p", "protect_p", "unprotect_p", "fake_out_p", "pivot_p", "disrupt_p")
ALWAYS = {k: 1.0 for k in PROBABILITIES}
NEVER = {**{k: 0.0 for k in PROBABILITIES}, "draft_tilt": 0.0}


def _folder(prefix: str) -> Path:
    return next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))


def _battle(prefix: str, seat: int) -> list[tuple[GameState, DecisionContext]]:
    folder = _folder(prefix)
    files = sorted(folder.glob(f"moving-*-seat{seat}.json"), key=lambda p: int(re.search(r"-(\d+)-seat", p.name).group(1)))
    out = []
    for path in files:
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if state.legal_actions:
            out.append((state, DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                               agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat)))
    return out


def _quiet(agent):
    agent._log_dir = "/dev/null"
    agent._print = lambda *_: None
    return agent


def _play(agent, prefix: str, seat: int) -> list[tuple[int, dict, str]]:
    """(turn, payload, public summary) for every battle decision of one live seat."""
    rows = []
    for state, ctx in _battle(prefix, seat):
        answer = agent.choose_action(state, ctx)
        rows.append((observation_dict(state).get("turn"), answer.action, answer.reasoning_summary))
    return rows


def _at(prefix: str, seat: int, turn: int, style: dict | None = None, index: int = 0):
    """A pivot agent replayed up to decision ``index`` of ``turn`` (1: a replacement after a faint): (agent, state, context)."""
    agent = _quiet(pivot.PivotAgent(style=style))
    seen = 0
    for state, ctx in _battle(prefix, seat):
        if observation_dict(state).get("turn") == turn:
            if seen == index:
                return agent, state, ctx
            seen += 1
        agent.choose_action(state, ctx)
    raise AssertionError(f"turn {turn} not in {prefix}")


# -- determinism and the code brain underneath --------------------------------------------------------------------


def test_with_every_trigger_off_it_plays_the_code_brain():
    code = _play(_quiet(PokemonAgent(None, version="code-only")), "c63f19be", 0)
    bot = _play(_quiet(pivot.PivotAgent(style=NEVER)), "c63f19be", 0)
    assert [r[1] for r in bot] == [r[1] for r in code]
    assert not any(r[2].startswith("Pivot style") for r in bot)


def test_same_input_same_action():
    first = _play(_quiet(pivot.PivotAgent()), "c63f19be", 0)
    again = _play(_quiet(pivot.PivotAgent()), "c63f19be", 0)
    assert first == again


def test_rolls_come_from_the_stable_game_id():
    assert pivot.stable_key("sim-1791872003-4") == pivot.stable_key("sim-1791999999-4") == "sim-4"
    assert pivot.stable_key("eval-1234-1") == "eval-1234-1"
    a = pivot.roll("sim-4", 1, 7, "switch:0")
    assert a == pivot.roll("sim-4", 1, 7, "switch:0") and 0.0 <= a < 1.0
    assert len({pivot.roll("sim-4", seat, turn, tag) for seat in (0, 1) for turn in (1, 2) for tag in ("switch:0", "protect:1")}) == 8


def test_an_error_in_the_style_layer_plays_the_code_brain(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    code = _play(_quiet(PokemonAgent(None, version="code-only")), "c63f19be", 0)
    monkeypatch.setattr(pivot.PivotAgent, "style_turn", broken)
    bot = _play(_quiet(pivot.PivotAgent(style=ALWAYS)), "c63f19be", 0)
    assert [r[1] for r in bot] == [r[1] for r in code]


def test_unknown_knob_is_an_error():
    with pytest.raises(KeyError):
        pivot.style_params({"swich_p": 1.0})


# -- the triggers, on live positions ---------------------------------------------------------------------------------


def test_protects_a_slot_facing_a_ko_with_no_switch_in():
    """c63f19be T9: Arcanine (slot 0, 20% HP) under Incineroar's Knock Off with no bench left; the code brain attacks
    with both, the style Protects it (and only it: Porygon2 still attacks)."""
    agent, state, ctx = _at("c63f19be", 0, 9, ALWAYS)
    answer = agent.choose_action(state, ctx)
    assert answer.action["slot_0"]["move_id"] == "protect"
    assert answer.action["slot_1"]["type"] == "move" and answer.action["slot_1"]["move_id"] != "protect"
    assert "Protects (no good switch-in" in answer.reasoning_summary
    agent, state, ctx = _at("c63f19be", 0, 9, {**ALWAYS, "protect_p": 0.0})
    assert agent.choose_action(state, ctx).action["slot_0"]["move_id"] != "protect"


def test_reads_the_hits_aimed_at_each_slot():
    agent, state, ctx = _at("c63f19be", 0, 9, ALWAYS)
    obs = observation_dict(state)
    agent.memory.observe_battle(obs)
    template = state.legal_actions[0].input["action"]
    from agent.pokemon import battle as battle_rules

    sheet = battle_rules.build_sheet(template, obs, agent.memory, agent.params)
    ours = {m.position: m for m in sheet.field.actives if m.side == "mine"}
    reads = agent.read_threats(sheet, ours)
    assert reads[0]["threatened"] and reads[0]["ko"] >= 0.5
    assert any(move == "knockoff" for _, move, _, _ in reads[0]["aimed"])


def _decide(prefix: str, seat: int, turn: int, style: dict | None = None, index: int = 0):
    agent, state, ctx = _at(prefix, seat, turn, style, index)
    answer = agent.choose_action(state, ctx)
    return answer.action, answer.reasoning_summary


def test_switches_to_the_teammate_that_takes_the_aimed_hit():
    """8e3a639c (seat 1) T5: the read has Low Kick aimed at slot 0; Torkoal (140 base Defense) takes it far better, so
    slot 0 goes out; with switch_p 0 the code brain's move stands."""
    action, why = _decide("8e3a639c", 1, 5, ALWAYS)
    assert action["slot_0"] == {"type": "switch", "species": "torkoal"} and "lowkick" in why
    action, why = _decide("8e3a639c", 1, 5, {**ALWAYS, "switch_p": 0.0})
    assert action["slot_0"]["type"] == "move" and "switches to" not in why


def test_double_switch_when_both_slots_face_hits_the_bench_absorbs():
    """d74bfb88 (seat 1) T2: Sludge Bomb read at slot 0 and Wood Hammer at slot 1; Baxcalibur (Steel: immune to poison)
    and Arcanine (Fire: resists Grass) take them, so both slots go out in one turn."""
    action, why = _decide("d74bfb88", 1, 2, ALWAYS)
    assert action["slot_0"] == {"type": "switch", "species": "baxcalibur"}
    assert action["slot_1"] == {"type": "switch", "species": "arcanine"}
    assert why.startswith("Pivot style: double switch")
    action, _ = _decide("d74bfb88", 1, 2, {**ALWAYS, "double_switch_p": 0.0})
    assert [action[f"slot_{n}"]["type"] for n in (0, 1)].count("switch") == 1


def test_fake_out_on_the_entry_turn():
    action, why = _decide("8e3a639c", 1, 1, ALWAYS)
    assert action["slot_0"]["move_id"] == "fakeout" and "Fakes Out incineroar" in why


def test_pivot_move():
    action, why = _decide("66e2891d", 1, 2, ALWAYS)
    assert action["slot_0"]["move_id"] == "partingshot" and "pivots with partingshot" in why


@pytest.mark.parametrize("prefix,seat,turn,move", [("bcbaabe2", 0, 5, "leechseed"), ("51c86e5c", 0, 2, "taunt")])
def test_disruption(prefix, seat, turn, move):
    action, why = _decide(prefix, seat, turn, ALWAYS)
    assert any(action[f"slot_{n}"].get("move_id") == move for n in (0, 1)) and f"disrupts with {move}" in why


def test_replacement_prefers_an_intimidate_user():
    """da9c168e (seat 1) T3, a faint replacement: Arcanine (Intimidate) instead of the code brain's Kingambit."""
    action, why = _decide("da9c168e", 1, 3, ALWAYS, index=1)
    assert action["slot_0"] == {"type": "switch", "species": "arcanine"} and "cycling" in why


def test_the_code_brains_protect_of_a_slot_not_facing_a_ko_becomes_an_attack(monkeypatch):
    """4e6fa669 (seat 1) T2. The code brain's Protect here is Scarf Landorus's, which choice_protect_guard leaves out (it
    switches Landorus to Iron Hands instead), so the position is replayed with the guard off."""
    monkeypatch.setitem(tuning.DEFAULTS, "choice_protect_guard", 0.0)
    action, why = _decide("4e6fa669", 1, 2, ALWAYS)
    assert action["slot_0"]["move_id"] == "rockslide" and "instead of Protecting" in why
    action, _ = _decide("4e6fa669", 1, 2, {**ALWAYS, "unprotect_p": 0.0})
    assert action["slot_0"]["move_id"] == "protect"


# -- the draft tilt ----------------------------------------------------------------------------------------------------


def test_card_style_prefers_bulky_support():
    style = {s: pivot.card_style(data.catalog_card(species=s) or {}) for s in ("Incineroar", "Landorus-Therian", "Amoonguss", "Chien-Pao")}
    assert style["Incineroar"] > style["Landorus-Therian"] > style["Amoonguss"] > style["Chien-Pao"]
    assert all(0.0 <= v <= 1.0 for v in style.values())


# -- the style statistics ----------------------------------------------------------------------------------------------


def test_style_stats_on_a_live_log():
    """c63f19be, the opponent's side (p2): Fake Out T1 and T11, Parting Shot T2 (its switch is a pivot switch), the hard
    switch T6, Leech Seed T7, T8 and T10, Protect T11, Intimidate on the lead and on the T2 replacement; the log stops at
    the start of T12, which is left out."""
    lines = json.loads((_folder("c63f19be") / "moving-00014-seat0.json").read_text())["observation"]["protocol_log"]
    s = style_stats.side_stats(lines, "p2")
    assert (s["turns"], s["turn_blocks"], s["slot_turns"]) == (12, 11, 22)
    assert (s["hard_switches"], s["double_switch_turns"], s["pivot_switches"], s["pivot"]) == (1, 0, 1, 1)
    assert (s["fake_out"], s["disruption"], s["protect"], s["intimidate"]) == (2, 3, 1, 2)


def test_style_stats_classifies_switches():
    log = [
        "|start", "|switch|p1a: A|A|100/100", "|switch|p1b: B|B|100/100", "|switch|p2a: X|X|100/100", "|switch|p2b: Y|Y|100/100",
        "|-ability|p2a: X|Intimidate|boost",
        "|turn|1", "|", "|switch|p2a: Z|Z|100/100", "|switch|p2b: W|W|100/100", "|move|p1a: A|Tackle|p2a: Z", "|upkeep",
        "|turn|2", "|", "|move|p2a: Z|U-turn|p1a: A", "|switch|p2a: X|X|90/100", "|move|p2b: W|Spore|p1b: B",
        "|cant|p1b: B|slp", "|move|p1a: A|Tackle|p2b: W", "|faint|p2b: W", "|upkeep", "|switch|p2b: Y|Y|100/100",
        "|turn|3", "|", "|cant|p2a: X|move: Taunt|Will-O-Wisp", "|move|p2b: Y|Protect|p2b: Y", "|upkeep",
        "|turn|4", "|", "|move|p2b: Y|Fake Out|p1a: A", "|faint|p1a: A", "|win|p2",
    ]
    s = style_stats.side_stats(log, "p2")
    assert (s["turns"], s["turn_blocks"], s["slot_turns"]) == (4, 3, 6)
    assert (s["hard_switches"], s["double_switch_turns"], s["pivot_switches"]) == (2, 1, 1)
    assert (s["pivot"], s["disruption"], s["protect"], s["fake_out"], s["intimidate"]) == (1, 2, 1, 0, 1)
    assert style_stats.side_stats(log, "p1")["hard_switches"] == 0
    summary = style_stats.summarize([s, s])
    assert summary["games"] == 2 and summary["hard_switch_rate"] == round(4 / 12, 3) and summary["avg_turns"] == 4.0
