"""The ``fable`` player of sim/draft_eval.py: agent/agent.py's model-piloted agent, built by its own factory from the
.env of the checkout the script runs from. Here the factory is replaced by one with a fake provider (no key, no
network), and the real factory is checked to print no key."""

from __future__ import annotations

import json
import os
from pathlib import Path

import agent.agent as agent_module
from agent.agent import PokemonAgent
from altruagent import DecisionContext
from altruagent.models import GameState
from sim import draft_eval

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "fixtures" / "live"


class FakeProvider:
    model = "fake-model"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.last_model = "fake-model"
        self.last_latency_ms = 7
        self.last_usage = {"input_tokens": 1, "output_tokens": 1}

    def complete_structured(self, messages, schema_name, schema):
        self.calls.append({"schema_name": schema_name})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_fable_is_an_external_spec_built_by_the_agents_factory(monkeypatch, tmp_path):
    assert draft_eval.parse_spec("fable") == ("fable", None) and draft_eval.external("fable")
    fake = FakeProvider()
    monkeypatch.setattr(agent_module, "create_agent", lambda: PokemonAgent(fake, version="fable-test", log=lambda _l: None))
    monkeypatch.setattr(draft_eval, "LOG_DIR", str(tmp_path / "decisions"))
    player = draft_eval.make_player("fable", seed=1)
    assert isinstance(player, PokemonAgent) and player._provider is fake
    assert player._log_dir == str(tmp_path / "decisions") and os.environ["AGENT_LOG_DIR"] == str(tmp_path / "decisions")


def test_fable_answers_team_preview_through_the_provider(monkeypatch):
    raw = json.loads(next(iter(sorted(LIVE.glob("*/team_preview-*-seat0.json")))).read_text())
    state = GameState.from_mcp_state(raw)
    roster = state.legal_actions[0].input["action"]["roster"]
    fake = FakeProvider({"bring": roster[:4], "leads": roster[:2], "reasoning_summary": "Fast leads."})
    monkeypatch.setattr(agent_module, "create_agent", lambda: PokemonAgent(fake, version="fable-test", log=lambda _l: None))
    monkeypatch.setattr(draft_eval, "LOG_DIR", draft_eval.NO_LOG)
    player = draft_eval.make_player("fable", seed=1)
    ctx = DecisionContext(session_id=raw["session_id"], tournament_id=None, game_type=raw["game_type"],
                          agent_id=raw["current_actor"]["agent_id"], seat_position=0)
    decision = player.choose_action(state, ctx)
    payload = getattr(decision, "action", decision)
    assert payload["type"] == "select_lineup" and payload["bring"] == roster[:4] and payload["leads"] == roster[:2]
    assert [c["schema_name"] for c in fake.calls] == ["select_lineup"]


def test_the_real_factory_never_prints_the_key(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key-0123456789")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.setattr(draft_eval, "LOG_DIR", draft_eval.NO_LOG)
    player = draft_eval.make_player("fable", seed=1)
    printed = capsys.readouterr()
    assert player._provider is not None and "model chain" in printed.err
    assert "sk-ant-not-a-real-key" not in printed.out + printed.err and "sk-ant" not in repr(player._provider)
