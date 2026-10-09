"""E1 — the clock budget: one model deadline per decision from the server's clock (decision time left and the bank),
per-call deadline and effort through the provider chain, and attempt-level timing in the log.

Live facts (platform_skill_pokemon.txt:262-268): a random legal move is played for us at 50 s per battle decision
(85 s at Team Preview); each side has a 420 s bank; an empty bank forfeits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState
from examples.llm.base import Choice
from examples.llm.providers import ProviderError

import agent.agent as agent_mod
from agent.agent import (DECISION_SECONDS, MIN_MODEL_SECONDS, PREVIEW_DECISION_SECONDS, PokemonAgent, _clock_budget,
                         expected_remaining_turns)

CTX = DecisionContext(session_id="clock-1", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="agent-a", seat_position=0)
# A synthetic position has no cards and so no rows: the answer proposes the joint action as an override (legal here).
ANSWER = {"row": 0, "override": {"use": True, "slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": 1}},
          "opponent_prediction": {"their_slot_1": "attack", "their_slot_2": "attack", "why": "w"}, "win_condition": "w", "reasoning_summary": "attack."}


class RecordingProvider:
    """Takes the per-call parameters and remembers them, like the real chain."""

    model = "claude-primary"
    effort = "low"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.last_model = "claude-primary"
        self.last_latency_ms = 5
        self.last_usage = {"input_tokens": 10, "output_tokens": 3}
        self.last_errors: list[str] = []
        self.last_attempts: list[dict] = []

    def complete_structured(self, messages, schema_name, schema, *, deadline=None, effort=None):
        self.calls.append({"deadline": deadline, "effort": effort, "schema_name": schema_name})
        self.last_attempts = [{"model": self.model, "effort": effort or self.effort, "timeout_s": deadline, "latency_ms": 5, "outcome": "ok"}]
        answer = self.answers.pop(0) if self.answers else ANSWER
        if isinstance(answer, Exception):
            self.last_attempts[0]["outcome"] = "error"
            raise answer
        return answer


def _choice(build=None):
    return Choice(kind="doubles_turn", prompt={}, schema={"type": "object"}, build=build or (lambda a: dict(a)), fallback=lambda: {"computed": True})


def _agent(provider, tmp_path):
    return PokemonAgent(provider, version="clock", log_dir=tmp_path, capture_dir="", log=lambda *_: None)


# -- the budget ------------------------------------------------------------------------------------------------------


def test_the_live_constants_are_the_platforms():
    assert DECISION_SECONDS == 50.0 and PREVIEW_DECISION_SECONDS == 85.0 and agent_mod.BANK_SECONDS == 420.0
    assert not hasattr(agent_mod, "SECOND_CALL_CUTOFF_SECONDS") and not hasattr(agent_mod, "MIN_DECISION_SECONDS_FOR_MODEL")


def test_deadline_is_decision_left_minus_safety_capped_at_twenty_seconds():
    full = _clock_budget({"decision_seconds_left": 50, "bank_seconds_left": 420}, remaining_turns=14)
    assert full.deadline == 20.0 and full.reason == "cap"  # 42 from the decision, 23.6 from the bank, 20 is the cap
    tight = _clock_budget({"decision_seconds_left": 22, "bank_seconds_left": 420}, remaining_turns=14)
    assert tight.deadline == 14.0 and tight.reason == "decision"
    assert _clock_budget({"decision_seconds_left": 13.9, "bank_seconds_left": 420}).deadline < MIN_MODEL_SECONDS
    assert _clock_budget({"decision_seconds_left": 14.0, "bank_seconds_left": 420}).deadline == MIN_MODEL_SECONDS


def test_deadline_spends_only_the_bank_above_the_gate_spread_over_the_turns_left():
    mid = _clock_budget({"decision_seconds_left": 50, "bank_seconds_left": 150}, remaining_turns=8)
    assert mid.deadline == 7.5 and mid.reason == "bank"  # (150 - 90) / 8
    low = _clock_budget({"decision_seconds_left": 50, "bank_seconds_left": 100}, remaining_turns=2)
    assert low.deadline == 2.5 and low.remaining_turns == 4  # never fewer than 4 turns assumed
    assert _clock_budget({"decision_seconds_left": 50, "bank_seconds_left": 90}).deadline == 0.0
    assert _clock_budget({"decision_seconds_left": 50, "bank_seconds_left": 60}).deadline == 0.0
    # The floor on the average: a full bank with 14 turns to go allows at most 20 s; over a 20-turn game that is
    # 330 s of bank at worst, i.e. the 90 s gate is never crossed by model time alone.
    assert (420 - 90) / 20 >= 12.0 and _clock_budget({"decision_seconds_left": 50, "bank_seconds_left": 420}).deadline <= 20.0


def test_team_preview_gets_its_own_allowance_and_cap():
    preview = _clock_budget({"decision_seconds_left": 85, "bank_seconds_left": 420}, kind="lineup")
    assert preview.deadline == 60.0 and preview.reason == "cap"  # 77 from the decision, 66 (a fifth of the bank above the gate), 60 is the cap
    assert _clock_budget({"decision_seconds_left": 85, "bank_seconds_left": 300}, kind="lineup").deadline == 42.0
    assert _clock_budget(None, kind="lineup").deadline == 60.0
    no_clock = _clock_budget(None)
    assert no_clock.deadline == 20.0 and no_clock.decision_left is None  # live allowances assumed without a clock


def test_expected_remaining_turns_is_a_simple_documented_estimate():
    assert expected_remaining_turns(4, 4) == 14 and expected_remaining_turns(2, 3) == 9 and expected_remaining_turns(1, 1) == 4
    assert expected_remaining_turns(None, None) == 14 and expected_remaining_turns(0, 1) == 4


# -- the decision ----------------------------------------------------------------------------------------------------


def test_a_clock_too_short_for_any_model_plays_the_computed_move_without_asking(tmp_path):
    provider = RecordingProvider()
    agent = _agent(provider, tmp_path)
    for clock in ({"decision_seconds_left": 12, "bank_seconds_left": 400}, {"decision_seconds_left": 50, "bank_seconds_left": 95},
                  {"decision_seconds_left": 50, "bank_seconds_left": 100}):
        value, answer, info = agent._ask(_choice(), {}, kind="turn", clock=clock, remaining_turns=14)
        assert value == {"computed": True} and answer is None and info["fallback"] is True
        assert info["errors"][0].startswith("clock too short") and "deadline" in info["errors"][0]
    assert provider.calls == [] and agent.memory.llm_calls == 0


def test_every_call_gets_what_is_left_of_the_deadline_and_the_kinds_effort(tmp_path):
    first = {"slot_0": {"option": 9}}  # invalid: a retry follows with what is left of the deadline
    provider = RecordingProvider(first, ANSWER)
    agent = _agent(provider, tmp_path)
    agent._efforts["turn"] = "medium"

    def build(answer):
        if (answer.get("slot_0") or {}).get("option") == 9:
            raise agent_mod.InvalidChoice("no option 9")
        return dict(answer)

    value, answer, info = agent._ask(_choice(build), {}, kind="turn", clock={"decision_seconds_left": 50, "bank_seconds_left": 420}, remaining_turns=14)
    assert answer == ANSWER and info["attempts"] == 2
    assert abs(provider.calls[0]["deadline"] - 20.0) < 0.05 and provider.calls[0]["effort"] == "medium"
    assert 19.0 < provider.calls[1]["deadline"] <= 20.0  # the retry gets the remainder
    log = info["attempt_log"]
    assert [a["outcome"] for a in log] == ["invalid", "answered"] and [a["attempt"] for a in log] == [1, 2]
    assert all(a["effort"] == "medium" and a["deadline_s"] <= 20.0 and "latency_ms" in a and a["chain"] for a in log)
    assert log[0]["deadline_s"] == 20.0
    assert info["model_latency_ms"] == sum(a["latency_ms"] for a in log) and info["clock"]["deadline_s"] == 20.0
    assert info["effort"] == "medium" and provider.calls[0]["schema_name"] == "doubles_turn"


def test_a_provider_without_per_call_parameters_is_still_called(tmp_path):
    class Plain:
        model = "plain"

        def __init__(self):
            self.calls = 0
            self.last_model = "plain"
            self.last_latency_ms = 1

        def complete_structured(self, messages, schema_name, schema):
            self.calls += 1
            return ANSWER

    provider = Plain()
    value, answer, info = _agent(provider, tmp_path)._ask(_choice(), {}, kind="turn", clock={"decision_seconds_left": 50, "bank_seconds_left": 420})
    assert answer == ANSWER and provider.calls == 1 and info["attempt_log"][0]["outcome"] == "answered"


def test_the_turn_row_logs_wall_time_attempts_and_the_deadline(tmp_path):
    from test_reliability_f import GOOD_SLOTS, _doubles

    provider = RecordingProvider(ANSWER)
    agent = _agent(provider, tmp_path)
    state = _doubles(GOOD_SLOTS)
    agent.choose_action(state, CTX)
    rows = [json.loads(line) for f in Path(tmp_path).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    config, turn = rows[0], [r for r in rows if r["kind"] == "turn"][-1]
    assert config["kind"] == "config" and config["clock"]["decision_s"] == 50.0 and config["effort"]["provider_default"] == "low"
    assert config["game_config"]["present"] is False and "real-time" in config["game_config"]["note"]
    assert isinstance(turn["wall_ms"], int) and turn["wall_ms"] >= turn["model_latency_ms"] >= 0
    assert turn["clock"]["deadline_s"] == 20.0 and turn["clock"]["remaining_turns"] >= 4
    assert turn["attempt_log"][0]["outcome"] == "answered" and abs(turn["attempt_log"][0]["deadline_s"] - 20.0) < 0.2


def test_game_config_cards_feed_the_pool_and_are_described_not_dumped(tmp_path):
    cards = [{"card_id": "vgc-farigiraf", "species": "Farigiraf", "item": "Throat Spray", "ability": "Armor Tail", "nature": "Quiet",
              "evs": {"hp": 252}, "moves": ["Hyper Voice", "Psychic", "Trick Room", "Protect"]}]
    config = {"rules": {"format": "gen9vgc2025regi"}, "draft": {"pool": cards, "picks": 12}, "timeouts": [15, 50, 85]}
    lines: list[str] = []
    agent = PokemonAgent(None, version="clock", log_dir=tmp_path, capture_dir="", log=lines.append)
    described = agent._absorb_game_config(config)
    assert described == {"present": True, "keys": {"rules": 1, "draft": 2, "timeouts": 3}, "cards": 1, "cards_added_to_pool": 1}
    assert agent.memory.pool_cards["vgc-farigiraf"]["species"] == "Farigiraf" and any("game_config carries 1 cards" in line for line in lines)
    assert agent._absorb_game_config({"rules": {}}) == {"present": True, "keys": {"rules": 0}, "cards": 0}
    assert agent._absorb_game_config(None)["present"] is False


def test_agent_effort_env_is_read_once_at_startup_and_logged(monkeypatch, tmp_path):
    from agent.llm.anthropic_provider import provider_from_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setenv("AGENT_EFFORT", "medium")
    chain = provider_from_env()
    assert chain.effort == "medium" and all(p.effort == "medium" for p in chain._providers)
    monkeypatch.setenv("AGENT_EFFORT", "high")
    assert chain.effort == "medium"  # read once at startup, not per call
    agent = PokemonAgent(chain, version="clock", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    agent._log_for(CTX)
    config = json.loads(next(Path(tmp_path).glob("*.jsonl")).read_text().splitlines()[0])
    assert config["effort"] == {"provider_default": "medium", "turn": None, "lineup": "high", "AGENT_EFFORT": "high"}
    assert "sk-ant" not in json.dumps(config)
    monkeypatch.delenv("AGENT_EFFORT")
    assert provider_from_env().effort == "low"  # the code default stays low


# -- the chain splits a deadline ------------------------------------------------------------------------------------


def test_fallback_provider_splits_the_deadline_and_skips_a_model_with_no_time(monkeypatch):
    from agent.llm.anthropic_provider import FallbackProvider

    class Slot:
        def __init__(self, model, timeout, *results):
            self.model, self.timeout, self.effort = model, timeout, "low"
            self.results = list(results)
            self.calls: list[dict] = []
            self.last_latency_ms = 4
            self.last_usage = None

        def complete_structured(self, messages, schema_name, schema, *, timeout=None, effort=None):
            self.calls.append({"timeout": timeout, "effort": effort})
            result = self.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    primary, fallback = Slot("opus", 18.0, ProviderError("opus: timed out after 14s")), Slot("sonnet", 8.0, {"x": 1})
    chain = FallbackProvider([primary, fallback])
    assert chain.complete_structured([], "x", {}, deadline=20.0, effort="medium") == {"x": 1}
    assert abs(primary.calls[0]["timeout"] - 14.0) < 0.05 and primary.calls[0]["effort"] == "medium"  # 20 - the 6 s reserve for the fallback
    assert fallback.calls[0]["timeout"] is not None and 0 < fallback.calls[0]["timeout"] <= 8.0
    assert [a["outcome"] for a in chain.last_attempts] == ["error", "ok"] and chain.last_attempts[1]["model"] == "sonnet"
    assert chain.last_model == "sonnet"

    # A primary that really spends its slice (a fake clock advances 5 s inside the call) leaves the fallback under
    # MIN_CALL_SECONDS: it is not asked, and the chain reports that instead of starting a call that cannot finish.
    import agent.llm.anthropic_provider as provider_mod

    now = [100.0]
    monkeypatch.setattr(provider_mod.time, "monotonic", lambda: now[0])

    class SlowFailure(Slot):
        def complete_structured(self, messages, schema_name, schema, *, timeout=None, effort=None):
            now[0] += 5.0
            return super().complete_structured(messages, schema_name, schema, timeout=timeout, effort=effort)

    primary, fallback = SlowFailure("opus", 18.0, ProviderError("opus: timed out")), Slot("sonnet", 8.0, {"x": 1})
    chain = FallbackProvider([primary, fallback])
    with pytest.raises(ProviderError, match="not tried"):
        chain.complete_structured([], "x", {}, deadline=6.0)  # the primary gets 4.2 s (70%); 1.0 s is left: not enough for sonnet
    assert abs(primary.calls[0]["timeout"] - 4.2) < 0.01 and fallback.calls == []
    assert chain.last_attempts[-1]["outcome"] == "skipped" and chain.last_attempts[-1]["left_s"] == 1.0
    monkeypatch.undo()

    primary, fallback = Slot("opus", 18.0, {"x": 2}), Slot("sonnet", 8.0, {"x": 1})
    chain = FallbackProvider([primary, fallback])
    assert chain.complete_structured([], "x", {}) == {"x": 2}  # no deadline: the instance timeouts apply
    assert primary.calls[0] == {"timeout": None, "effort": None}


def test_anthropic_provider_takes_a_per_call_timeout_and_effort():
    from types import SimpleNamespace

    from agent.llm.anthropic_provider import AnthropicProvider

    requests: list[dict] = []

    def create(**request):
        requests.append(request)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"x": 1}')],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0))

    provider = AnthropicProvider("claude-test", client=SimpleNamespace(messages=SimpleNamespace(create=create)), timeout=18.0, effort="low")
    assert provider.complete_structured([{"role": "user", "content": "{}"}], "x", {"type": "object"}, timeout=7.5, effort="high") == {"x": 1}
    assert requests[0]["timeout"] == 7.5 and requests[0]["output_config"]["effort"] == "high"
    assert provider.last_timeout == 7.5 and provider.last_effort == "high"
    provider.complete_structured([{"role": "user", "content": "{}"}], "x", {"type": "object"})
    assert "timeout" not in requests[1] and requests[1]["output_config"]["effort"] == "low" and provider.last_timeout == 18.0
