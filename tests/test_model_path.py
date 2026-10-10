"""The model path (lane M1): what happens between the model's answer and the move, and what the model is shown.

1. A reply-mixture exception costs the mixture for that turn, never the match: the scorer's rows play (readiness 2B.1).
2. Our last Pokémon's Protect with nothing to wait for earns no override credit and never plays over a row (model
   audit R1/R7; d9405f02 T7 and T9).

Positions are replayed from the live captures (tests/fixtures/live/) through a code-only agent; the model is a scripted
fake or absent. No test calls a model.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState
from examples.llm.base import Choice, InvalidChoice

from agent.agent import PokemonAgent, override_value
from agent.pokemon import battle as battle_rules
from agent.pokemon import tuning
from agent.pokemon.memory import observation_dict
from test_live_semantics import LIVE, _sequence

FAST = {"lineup_sim_budget_s": 0.0}  # Team Preview's 20 s search is not what these tests are about


def _folder(prefix: str) -> Path:
    return next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))


def _context(folder: Path, raw: dict, seat: int) -> DecisionContext:
    return DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                           agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat)


def _rows(log_dir: Path) -> list[dict]:
    return [json.loads(line) for f in Path(log_dir).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]


def _position(prefix: str, seat: int, turn: int, params: dict | None = None, index: int = 0):
    """Replay a live session code-only up to a battle decision: (agent, sheet, template, adapter) for that position."""
    from examples.llm import pokemon as adapters

    folder = _folder(prefix)
    agent = PokemonAgent(None, version="mpath", log_dir=Path(tempfile.mkdtemp()), capture_dir="", log=lambda *_: None,
                         params=tuning.merged({**FAST, **(params or {})}))
    seen = 0
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        obs = observation_dict(state)
        if state.phase == "moving" and obs.get("turn") == turn:
            if seen == index:
                agent.memory.observe_battle(obs)
                action = state.legal_actions[0]
                template = action.input.get("action") or {}
                sheet = battle_rules.build_sheet(template, obs, agent.memory, agent.params)
                base = adapters.doubles_choice(action, state)
                choice = Choice(kind="doubles_turn", prompt={}, schema={}, build=base.build, fallback=lambda: None)
                sheet.candidates = [c for c in sheet.candidates if _valid(base.build, c)]
                for i, c in enumerate(sheet.candidates):
                    c["row"] = i
                return agent, sheet, template, choice
            seen += 1
        agent.choose_action(state, _context(folder, raw, seat))
    raise AssertionError(f"turn {turn} not in {prefix}")


def _valid(build, row: dict) -> bool:
    try:
        build({"slot_0": row["slot_0"], "slot_1": row["slot_1"]})
        return True
    except InvalidChoice:
        return False


def _index(template: dict, number: int, move_or_species: str) -> int:
    options = next(s for s in template["slots"] if s.get("slot") == number)["options"]
    return next(i for i, o in enumerate(options) if o.get("move_id") == move_or_species or o.get("species") == move_or_species)


def _answer(row: int = 0, *, override: dict | None = None, prediction: tuple | None = None) -> dict:
    out = {"row": row, "win_condition": "w", "reasoning_summary": "Test answer.",
           "override": {"use": override is not None, "slot_0": (override or {}).get("slot_0", {"option": 0, "target": 0}),
                        "slot_1": (override or {}).get("slot_1", {"option": 0, "target": 0})}}
    if prediction is not None:
        out["opponent_prediction"] = {"their_slot_1": prediction[0], "their_slot_2": prediction[1], "why": "test"}
    return out


# -- 1. a reply-mixture exception is contained (readiness 2B.1) ----------------------------------------------------------


def test_a_mixture_exception_is_contained_and_every_turn_plays_a_scorer_row(tmp_path, monkeypatch):
    """replies.evaluate raises on every call at bcbaabe2 (reply_mixture 1). Before the fix, 17 of its 18 battle decisions
    played the smoke agent's "Fallback:" move (readiness-1010b live-safety/mixture_raise.txt). Now each main turn notes the
    failure, carries no reply table, and plays the scorer's row 0; nothing reaches the guard."""
    from agent.pokemon import replies

    calls: list[int] = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise KeyError("injected mixture bug")

    monkeypatch.setattr(replies, "evaluate", boom)
    folder = _folder("bcbaabe2")
    agent = PokemonAgent(None, version="mpath", log_dir=tmp_path, capture_dir="", log=lambda *_: None,
                         params=tuning.merged({**FAST, "reply_mixture": 1.0, "reply_seed": 7.0}))
    played = []
    for path in _sequence(folder, 0):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        decision = agent.choose_action(state, _context(folder, raw, 0))
        if state.phase == "moving":
            played.append((state, decision))
    rows = _rows(tmp_path)
    assert not [r for r in rows if r["kind"] in ("error", "fallback")], "the exception reached the guard"
    turns = [r for r in rows if r["kind"] == "turn"]
    assert len(played) == len(turns) == 18
    main = [r for r in turns if not any(s.get("force_switch") for s in _template_of(played, r).get("slots") or [])]
    assert len(calls) == len(main) == 17  # the mixture ran (and raised) on every main turn
    for (state, decision), row in zip(played, turns):
        assert not decision.reasoning_summary.startswith("Fallback"), decision.reasoning_summary
        sheet = row["turn_sheet"]
        assert "reply_table" not in sheet and all("worst" not in c and "best_reply" not in c for c in sheet["candidate_turns"])
        top = sheet["candidate_turns"][0]
        from examples.llm import pokemon as adapters

        expected = adapters.doubles_choice(state.legal_actions[0], state).build({"slot_0": top["slot_0"], "slot_1": top["slot_1"]})
        assert decision.action == expected, (row["turn"], decision.action, expected)
    assert all(any("reply mixture failed this turn (KeyError" in n for n in r["turn_sheet"]["notes"]) for r in main)


def _template_of(played: list, row: dict) -> dict:
    state = next(s for s, _ in played if s.state_version == row["state_version"])
    return state.legal_actions[0].input.get("action") or {}


# -- 2. our last Pokémon's Protect with nothing to wait for (model audit R1 / R7) ------------------------------------------


@pytest.mark.parametrize("mixture", [0.0, 1.0])
@pytest.mark.parametrize("turn", [7, 9])
def test_a_lone_protect_with_nothing_to_wait_for_earns_nothing_and_never_plays_d9405f02(turn, mixture):
    """d9405f02 T7 and T9: Cresselia, our last Pokémon, under LETHAL Shadow Ball, nothing runs out. The live answer
    Protected (robust worse). The menu lists only the attack; the Protect override was credited 45 and accepted on both
    builds. Now it earns no credit, and since 0 still clears a 0.15 margin under a best row of 0.0 (scorer) or below 0
    (mixture), it is refused outright, read or no read: the attack row plays."""
    agent, sheet, template, choice = _position("d9405f02", 0, turn, {"reply_mixture": mixture, "reply_seed": 7.0})
    assert sheet.alone and not sheet.stall_reasons and any(w.startswith("LETHAL") and "slot 1 (cresselia)" in w for w in sheet.warnings)
    protect = {"slot_0": {"option": 0, "target": 0}, "slot_1": {"option": _index(template, 1, "protect"), "target": 0}}
    payload = choice.build(protect)
    assert override_value(payload, sheet, agent.params) == 0.0
    prediction = {"top_column": {1: "attack", 2: "attack"}}
    for read in (None, ("attack", "attack")):  # no read (the live answer's format) and a read matching the code's column
        meta: dict = {}
        value = agent._resolve_turn_answer(_answer(0, override=protect, prediction=read), sheet, choice, prediction, meta)
        assert meta["resolution"].startswith("override rejected: our last Pokémon's Protect has nothing to wait for"), meta
        assert value["slot_1"]["type"] == "move" and value["slot_1"]["move_id"] != "protect", value
        top = sheet.candidates[0]
        assert value == choice.build({"slot_0": top["slot_0"], "slot_1": top["slot_1"]})
    # with a row number that is not a row, the rejection is an invalid answer (retried), never the Protect
    with pytest.raises(InvalidChoice):
        agent._resolve_turn_answer(_answer(5, override=protect), sheet, choice, prediction, {})
    # a stall reason gives the Protect its credit back (the fix is that narrow)
    sheet.stall_reasons = ["their Tailwind runs out (then we move first)"]
    assert override_value(payload, sheet, agent.params) == agent.params["protect_bonus_guaranteed"]
