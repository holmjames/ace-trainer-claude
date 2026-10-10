"""The model path (lane M1): what happens between the model's answer and the move, and what the model is shown.

1. A reply-mixture exception costs the mixture for that turn, never the match: the scorer's rows play (readiness 2B.1).

Positions are replayed from the live captures (tests/fixtures/live/) through a code-only agent; the model is a scripted
fake or absent. No test calls a model.
"""

from __future__ import annotations

import json
from pathlib import Path

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon import tuning
from test_live_semantics import LIVE, _sequence

FAST = {"lineup_sim_budget_s": 0.0}  # Team Preview's 20 s search is not what these tests are about


def _folder(prefix: str) -> Path:
    return next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))


def _context(folder: Path, raw: dict, seat: int) -> DecisionContext:
    return DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                           agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat)


def _rows(log_dir: Path) -> list[dict]:
    return [json.loads(line) for f in Path(log_dir).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]


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
