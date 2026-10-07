"""Tests for agent/arena.py (seat-based version routing) and scripts/tally.py (log summaries)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from altruagent import DecisionContext
from altruagent.models import GameState, LegalAction

from agent import arena
from agent.agent import PokemonAgent


def _state():
    return GameState.from_mcp_state({
        "session_id": "s", "status": "in_progress", "phase": "moving", "is_current_actor": True, "state_version": 1,
        "legal_actions": {"actions": [{"action_id": "draft_pick:c1", "label": "Draft c1", "input": {}}]},
    })


def test_arena_routes_each_seat_to_its_configured_version(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_SEAT0", "agent.agent")
    monkeypatch.setenv("AGENT_SEAT1", "examples.smoke_agent")
    monkeypatch.setenv("AGENT_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("agent.agent.load_dotenv", lambda *a, **k: None)

    seat0 = arena.create_agent()
    seat0.choose_action(_state(), DecisionContext("s", None, "pokemon_vgc_doubles_draft", "a", seat_position=0))
    assert isinstance(seat0._inner, PokemonAgent) and seat0.spec == "agent.agent"
    assert seat0._inner._version == "agent.agent"  # the decision log carries the version label

    seat1 = arena.create_agent()
    decision = seat1.choose_action(_state(), DecisionContext("s", None, "pokemon_vgc_doubles_draft", "b", seat_position=1))
    assert seat1.spec == "examples.smoke_agent" and isinstance(decision, LegalAction)


def test_arena_defaults_to_the_champion_when_unset(monkeypatch):
    monkeypatch.delenv("AGENT_SEAT0", raising=False)
    monkeypatch.delenv("AGENT_SEAT1", raising=False)
    assert arena.spec_for_seat(1) == "agent.agent" and arena.spec_for_seat(None) == "agent.agent"


def _load_tally():
    path = Path(__file__).resolve().parent.parent / "scripts" / "tally.py"
    spec = importlib.util.spec_from_file_location("tally", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_decisions(dir_: Path, session: str, version: str, ts: str, *, latency=5000, fallback=False, model="claude-fable-5-1"):
    lines = [
        {"ts": ts, "session_id": session, "version": version, "kind": "draft"},
        {"ts": ts, "session_id": session, "version": version, "kind": "turn", "model": model, "latency_ms": latency, "fallback": fallback,
         "usage": {"input_tokens": 4000, "output_tokens": 100, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}},
    ]
    (dir_ / f"{session}.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")


def test_tally_joins_supervisor_and_claim_results_and_prices_usage(tmp_path, capsys):
    tally = _load_tally()
    _write_decisions(tmp_path, "sess-A", "agent.agent", "2026-10-01T10:00:00+00:00", latency=4000)
    _write_decisions(tmp_path, "sess-B", "agent.versions.v2", "2026-10-01T10:05:00+00:00", latency=9000, fallback=True, model="claude-sonnet-5-5")
    _write_decisions(tmp_path, "sess-C", "agent.agent", "2026-10-01T11:00:00+00:00")
    (tmp_path / "runtime-match-1.log").write_text(
        "[match pokemon_vgc_doubles_draft seat=seat-1] finished termination_reason=normal score=1.0\n"
        "[agent] worker for session_id=sess-A finished\n"
        "[match pokemon_vgc_doubles_draft seat=seat-2] finished termination_reason=normal score=0.0\n"
        "[agent] worker for session_id=sess-B finished\n"
    )
    claim_log = tmp_path / "runtime-match-2.log"
    claim_log.write_text("Claimed seat 1/2\nMatch finished (termination_reason=normal) after 12 decision(s).\nYour score: 1.0\n")

    assert tally.main(["--logs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    champion = next(l for l in out.splitlines() if l.startswith("agent.agent "))
    challenger = next(l for l in out.splitlines() if l.startswith("agent.versions.v2 "))
    assert " 2-0-0 " in champion  # two wins: one from the supervisor log, one paired from the claim-mode log
    assert " 0-1-0 " in challenger
    assert "claude-sonnet-5-5×1" in challenger and "claude-fable-5-1×2" in champion
    # Cost: 4000 in @ $10/M + 1000 cached @ $1/M + 100 out @ $50/M = 0.046 per Fable turn.
    assert "0.09" in champion
    assert "cache hit rate 20%" in out


def test_tally_handles_an_empty_directory(tmp_path, capsys):
    tally = _load_tally()
    assert tally.main(["--logs", str(tmp_path)]) == 0
    assert "no decision logs" in capsys.readouterr().out
