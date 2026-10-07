"""Replay every state captured from the real platform (tests/fixtures/live/, AGENT_CAPTURE_DIR) through the
code-only agent. The first live self-play match on Oct 7 2026 proved the real observation parses: these fixtures
keep that true. Any new capture folder is picked up automatically."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon.battle import field_from_obs, normalize_weather
from agent.pokemon.memory import observation_dict

LIVE = Path(__file__).parent / "fixtures" / "live"
FILES = sorted(LIVE.glob("*/*.json"))

pytestmark = pytest.mark.skipif(not FILES, reason="no live captures yet (run a match with AGENT_CAPTURE_DIR set)")


def _log_rows(log_dir: Path) -> list[dict]:
    return [json.loads(line) for f in Path(log_dir).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]


def _ctx(seat: int = 0) -> DecisionContext:
    return DecisionContext(session_id="live-fixture", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="me", seat_position=seat)


@pytest.mark.parametrize("path", FILES, ids=[f"{p.parent.name[:8]}/{p.name}" for p in FILES])
def test_every_captured_live_state_gets_a_computed_decision(path, tmp_path):
    raw = json.loads(path.read_text())
    state = GameState.from_mcp_state(raw)
    if not state.legal_actions:
        pytest.skip("terminal or observe-only state")
    agent = PokemonAgent(None, version="live-fixture", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    decision = agent.choose_action(state, _ctx())
    summary = getattr(decision, "reasoning_summary", "")
    assert not summary.startswith("Fallback:"), f"decision logic raised on a real state: {summary}"
    assert not any(r.get("kind") in ("error", "fallback") for r in _log_rows(tmp_path)), "decision logic hit the exception path"
    action = getattr(decision, "action", decision)
    if state.phase == "team_preview":
        assert action["type"] == "select_lineup" and len(action["bring"]) == 4 and len(action["leads"]) == 2
    elif state.phase == "moving":
        assert action["type"] == "doubles_turn" and "slot_0" in action
        turns = [r for r in _log_rows(tmp_path) if r.get("kind") == "turn"]
        assert turns and (turns[-1].get("turn_sheet") or {}).get("candidate_turns"), "the brain produced no candidate on a real state (adapter default would play)"
    else:
        assert hasattr(action, "action_id")


def test_live_battle_observation_fields_are_read():
    moving = [json.loads(p.read_text()) for p in FILES]
    moving = [r for r in moving if isinstance(r.get("observation"), dict) and "turn" in r["observation"]]
    assert moving, "no battle state captured"
    obs = moving[0]["observation"]
    for key in ("clock", "weather", "fields", "side_conditions", "opponent_side_conditions", "team", "opponent_team", "available_moves"):
        assert key in obs, f"live observation lost the {key} field"
    assert "decision_seconds_left" in obs["clock"] and "bank_seconds_left" in obs["clock"]
    # weather and terrain arrive as dicts keyed by UPPER_SNAKE names
    assert normalize_weather({}) is None and normalize_weather({"SUNNYDAY": 0}) == "sun"
    fs = field_from_obs({"weather": {"RAINDANCE": 3}, "fields": {"PSYCHIC_TERRAIN": 2}, "side_conditions": {}, "opponent_side_conditions": {},
                         "team": {}, "opponent_team": {}}, {}, {}, None)
    assert fs.weather == "rain" and fs.terrain == "psychic"
