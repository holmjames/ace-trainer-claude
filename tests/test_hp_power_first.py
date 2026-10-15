"""hp_power_first: Eruption / Water Spout are credited at the HP their user likely has when it moves.

Live, Oct 14 (58c44401) T1: our Incineroar + Torkoal (sun) against Basculegion (speed 130) and Booster Energy Iron Bundle
(309); Torkoal 36. The code's row 0 was finish_the_other (243.4): Torkoal's Eruption "already KOs" Iron Bundle (161-191%
at full HP) while Incineroar Knocks Off Basculegion. Iron Bundle's Hydro Pump (65-79%) hit Torkoal first, the Eruption ran
at a quarter of its power and Iron Bundle stayed in at 60%."""
import copy
import json
import tempfile
from pathlib import Path

import pytest
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon import battle, tuning
from agent.pokemon.memory import observation_dict
from test_live_semantics import LIVE, _sequence
from test_reliability_f import _context

MATCH = "58c44401"


def _state(prefix: str, seat: int, turn: int):
    """(agent, template, obs) at a live battle decision, the agent fed everything before it (as _position)."""
    folder = next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))
    agent = PokemonAgent(None, version="e3", log_dir=Path(tempfile.mkdtemp()), capture_dir="", log=lambda *_: None)
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        obs = observation_dict(state)
        if state.phase == "moving" and obs.get("turn") == turn:
            agent.memory.observe_battle(obs)
            return agent, state.legal_actions[0].input.get("action") or {}, obs
        agent.choose_action(state, _context(folder, raw, seat))
    raise AssertionError(f"turn {turn} not in {prefix}")


@pytest.fixture(scope="module")
def t1():
    return _state(MATCH, 0, 1)


def _sheet(t1, key: float, obs: dict | None = None) -> "battle.TurnSheet":
    agent, template, live_obs = t1
    return battle.build_sheet(template, obs or live_obs, agent.memory, {**agent.params, "hp_power_first": key})


def _row(sheet, move: str, species: str) -> dict:
    return next(t for e in sheet.our_options for o in e["options"] if o["move"] == move for t in o["targets"] if t["species"] == species)


def _slowed(obs: dict, species: str, stages: int) -> dict:
    """The observation with one Pokémon's speed stage set (an Icy Wind / Electroweb read)."""
    out = copy.deepcopy(obs)
    for team in ("team", "opponent_team"):
        for entry in (out.get(team) or {}).values():
            if isinstance(entry, dict) and entry.get("species") == species:
                entry.setdefault("boosts", {})["spe"] = stages
    return out


def test_default_is_on():
    assert tuning.DEFAULTS["hp_power_first"] == 1.0


def test_live_t1_off_is_deploy_7(t1):
    """Off: deploy #7's rows, as logged live: the Eruption 'already KOs' Iron Bundle."""
    sheet = _sheet(t1, 0.0)
    assert not sheet.hp_first
    assert [(c["name"], c["score"]) for c in sheet.candidates] == [
        ("finish_the_other", 243.4), ("fake_out_setup", 103.1), ("protect_double_target", 101.8), ("protect_and_switch", 15.8)]
    assert "eruption already KOs ironbundle" in sheet.candidates[0]["why"]
    assert _row(sheet, "eruption", "ironbundle")["ko"] == "guaranteed"


def test_live_t1_on_no_guaranteed_eruption_ko(t1):
    """On: Iron Bundle's Hydro Pump (aimed at Torkoal 0.8, 80% accurate) and Basculegion's Wave Crash (0.2) land first,
    so the Eruption KOs Iron Bundle only when neither does (33%), and the row says so. No row says it 'already KOs';
    Heat Wave, which does not lose power, takes over the split; the Fake Out row keeps the Eruption, now at the HP
    Torkoal keeps with Iron Bundle flinched."""
    sheet = _sheet(t1, 1.0)
    row = _row(sheet, "eruption", "ironbundle")
    assert row["ko"] == "possible" and row["ko_chance_pct"] == 33 and row["full_hp_pct"] == [161.4, 190.9]
    assert "the KO depends on not being hit first (33%)" in row["if_hit_first"] and "ironbundle's hydropump" in row["if_hit_first"]
    assert battle.ko_probability(row) == pytest.approx(0.33, abs=0.01)
    assert not any("eruption already KOs" in c["why"] for c in sheet.candidates)
    assert [(c["name"], c["score"]) for c in sheet.candidates] == [
        ("finish_the_other", 226.4), ("protect_double_target", 101.8), ("fake_out_setup", 94.5), ("protect_and_switch", 15.8)]
    assert "heatwave already KOs ironbundle" in sheet.candidates[0]["why"]
    fake = sheet.candidates[2]
    assert "eruption at the HP it keeps: ironbundle cannot hit it first" in fake["why"]
    # the prompt carries the note and the odds, not the cases
    shown = json.dumps(sheet.as_prompt())
    assert "if_hit_first" in shown and "cases" not in shown


def _no_aqua_jet(t1, monkeypatch) -> None:
    """Basculegion's card without Aqua Jet, its +1 hit (the sheet reads their known moves from the card)."""
    card = t1[0].memory.opp_cards["basculegion"]
    monkeypatch.setitem(card, "moves", [m for m in card["moves"] if m != "Aqua Jet"])


def test_no_foe_first_is_off(t1, monkeypatch):
    """Nothing hits Torkoal before it moves (Torkoal at +3 speed, both foes at -6, Basculegion without Aqua Jet): the
    sheet is deploy #7's, rows and prompt alike."""
    _no_aqua_jet(t1, monkeypatch)
    slow = _slowed(_slowed(_slowed(t1[2], "basculegion", -6), "ironbundle", -6), "torkoal", 3)
    off, on = _sheet(t1, 0.0, slow), _sheet(t1, 1.0, slow)
    assert [r["species"] for r in on.speed_order][0] == "torkoal" and not on.hp_first
    assert on.candidates == off.candidates and on.candidates[0]["name"] == "finish_the_other"
    assert json.dumps(on.as_prompt(), sort_keys=True, default=str) == json.dumps(off.as_prompt(), sort_keys=True, default=str)


def test_partner_fake_out_removes_the_only_faster_threat(t1, monkeypatch):
    """Basculegion at -6 speed (32 against Torkoal's 36) without Aqua Jet: only Iron Bundle's Hydro Pump lands on Torkoal
    first. Every row but the Fake Out one sees it cut the Eruption (no row says it 'already KOs' Iron Bundle); the Fake
    Out on Iron Bundle stops it, so that row credits the full-power Eruption exactly as deploy #7 does."""
    _no_aqua_jet(t1, monkeypatch)
    slow = _slowed(t1[2], "basculegion", -6)
    off, on = _sheet(t1, 0.0, slow), _sheet(t1, 1.0, slow)
    assert [h["species"] for h in on.hp_first[(1, 0)]["hitters"]] == ["ironbundle"]
    assert _row(on, "eruption", "ironbundle")["ko"] == "possible" and _row(off, "eruption", "ironbundle")["ko"] == "guaranteed"
    assert "eruption already KOs ironbundle" in off.candidates[0]["why"]
    assert not any("eruption already KOs" in c["why"] for c in on.candidates)
    fake_on = next(c for c in on.candidates if c["name"] == "fake_out_setup")
    fake_off = next(c for c in off.candidates if c["name"] == "fake_out_setup")
    assert (fake_on["slot_0"], fake_on["slot_1"], fake_on["score"]) == (fake_off["slot_0"], fake_off["slot_1"], fake_off["score"])
    assert "eruption at the HP it keeps: ironbundle cannot hit it first" in fake_on["why"]
