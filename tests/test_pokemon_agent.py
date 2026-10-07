"""Tests for agent/agent.py (PokemonAgent), agent/pokemon/draft.py, memory and log.

The model is always a scripted ``FakeProvider``; no test touches the network
or needs a key. Payload shapes mirror tests/test_smoke_agent.py and
tests/test_llm_agent.py (Agent_ACP GameAPI's Pokémon adapter).
"""

from __future__ import annotations

import json

import pytest

from altruagent import DecisionContext, WithReasoning
from altruagent.models import GameState, LegalAction
from examples.llm.providers import ProviderError

from agent.agent import PokemonAgent, create_agent
from agent.pokemon import draft as draft_rules
from agent.pokemon.log import redact
from agent.pokemon.memory import MatchMemory, species_key

CONTEXT = DecisionContext(session_id="match-1", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                          agent_id="agent-a", seat_position=0)
ROSTER = ["incineroar", "rillaboom", "urshifu", "amoonguss", "tornadus", "fluttermane"]
PASS = {"type": "pass"}


class FakeProvider:
    model = "fake-model"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.last_model = "fake-model"
        self.last_latency_ms = 7
        self.last_usage = {"input_tokens": 1, "output_tokens": 1}

    def complete_structured(self, messages, schema_name, schema):
        self.calls.append({"messages": [dict(m) for m in messages], "schema_name": schema_name, "schema": schema})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def prompt(self, index=0) -> dict:
        return json.loads(self.calls[index]["messages"][1]["content"])


def _state(*legal: dict, observation=None, version=3) -> GameState:
    return GameState.from_mcp_state(
        {
            "session_id": "match-1",
            "status": "in_progress",
            "phase": "moving",
            "is_current_actor": True,
            "state_version": version,
            "observation": observation or {},
            "legal_actions": {"session_id": "match-1", "state_version": version, "actions": list(legal)},
        }
    )


def _templated(action_id: str, template) -> dict:
    return {"action_id": action_id, "label": action_id,
            "input": {"session_id": "match-1", "action_id": action_id, "state_version": 3, "action": template}}


def _lineup_action(roster=ROSTER) -> dict:
    return _templated("select_lineup", {"type": "select_lineup", "roster": roster, "bring_count": 4,
                                        "lead_count": 2, "instructions": "pick 4, lead 2"})


def _move(move_id: str, targets: list[int], target_options=None) -> dict:
    option = {"type": "move", "move_id": move_id, "base_power": 80, "category": "physical",
              "move_type": "normal", "current_pp": 10, "accuracy": 100, "targets": targets}
    if target_options is not None:
        option["target_options"] = target_options
    return option


def _doubles_action(slot_0: list[dict], slot_1: list[dict]) -> dict:
    slots = [
        {"slot": 0, "board_position": -1, "active": "incineroar", "force_switch": False, "options": slot_0},
        {"slot": 1, "board_position": -2, "active": "rillaboom", "force_switch": False, "options": slot_1},
    ]
    return _templated("doubles_turn", {"type": "doubles_turn", "slots": slots, "target_legend": {},
                                       "instructions": "pick per slot"})


def _draft_cards():
    return [
        {"card_id": "vgc-amoonguss", "species": "Amoonguss", "item": "Rocky Helmet", "ability": "Regenerator",
         "nature": "Calm", "moves": ["Spore", "Rage Powder", "Pollen Puff", "Protect"],
         "base_stats": {"hp": 114, "atk": 85, "def": 70, "spa": 85, "spd": 80, "spe": 30}},
        {"card_id": "vgc-dragapult", "species": "Dragapult", "item": "Choice Band", "ability": "Clear Body",
         "nature": "Jolly", "moves": ["Dragon Darts", "Phantom Force", "U-turn", "Sucker Punch"],
         "base_stats": {"hp": 88, "atk": 120, "def": 75, "spa": 100, "spd": 75, "spe": 142}},
        {"card_id": "vgc-pikachu", "species": "Pikachu", "item": "Light Ball", "ability": "Static",
         "nature": "Timid", "moves": ["Thunderbolt", "Volt Switch", "Grass Knot", "Protect"],
         "base_stats": {"hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90}},
    ]


def _draft_state(cards=None, *, picks=(), rosters=None):
    cards = cards if cards is not None else _draft_cards()
    obs = {
        "phase": "draft",
        "current_seat": "seat-a",
        "first_drafter": "seat-a",
        "available_card_ids": [c["card_id"] for c in cards],
        "available_cards": cards,
        "rosters": rosters or {"seat-a": [], "seat-b": []},
        "picks": list(picks),
        "decision_timeout_seconds": 15,
    }
    legal = [{"action_id": f"draft_pick:{c['card_id']}", "label": f"Draft {c['species']}",
              "input": {"action": {"type": "draft_pick", "card_id": c["card_id"]}}} for c in cards]
    return _state(*legal, observation=obs)


def _agent(provider=None, tmp_path=None, **kwargs) -> PokemonAgent:
    return PokemonAgent(provider, log_dir=tmp_path, capture_dir="", log=lambda line: None, **kwargs)


def _unwrap(decision):
    assert isinstance(decision, WithReasoning)
    return decision.action


# -- draft ------------------------------------------------------------------------------


def test_draft_is_code_only_and_prefers_the_strong_fast_card(tmp_path):
    provider = FakeProvider()
    agent = _agent(provider, tmp_path)

    decision = agent.choose_action(_draft_state(), CONTEXT)

    picked = _unwrap(decision)
    assert isinstance(picked, LegalAction)
    assert picked.action_id == "draft_pick:vgc-dragapult"
    assert provider.calls == []  # never a model call in the draft
    assert agent.memory.my_seat_key == "seat-a"
    assert len(agent.memory.pool_cards) == 3


def test_draft_remembers_both_sides_full_sets(tmp_path):
    agent = _agent(None, tmp_path)
    cards = _draft_cards()
    agent.choose_action(_draft_state(cards), CONTEXT)  # learns we are seat-a
    rosters = {"seat-a": [{"card_id": "vgc-dragapult", "species": "Dragapult"}],
               "seat-b": [{"card_id": "vgc-amoonguss", "species": "Amoonguss"}]}
    remaining = [cards[2]]
    agent.choose_action(_draft_state(remaining, rosters=rosters), CONTEXT)

    assert agent.memory.my_cards[species_key("Dragapult")]["moves"][0] == "Dragon Darts"
    assert agent.memory.opp_cards[species_key("Amoonguss")]["item"] == "Rocky Helmet"


def test_draft_ties_break_toward_the_servers_order():
    memory = MatchMemory()
    legal = [LegalAction("draft_pick:x", "x", {}), LegalAction("draft_pick:y", "y", {})]
    assert draft_rules.choose_pick(legal, memory).action is legal[0]


# -- team preview -------------------------------------------------------------------------


def test_lineup_uses_the_model_and_validates_the_answer(tmp_path):
    provider = FakeProvider({"bring": ROSTER[:4], "leads": ROSTER[:2], "reasoning_summary": "Fast leads."})
    agent = _agent(provider, tmp_path)
    obs = {"phase": "team_preview", "your_roster": [{"species": s, "types": ["FIRE"]} for s in ROSTER],
           "opponent_roster": [{"species": "garchomp", "types": ["DRAGON", "GROUND"]}]}

    decision = _unwrap(agent.choose_action(_state(_lineup_action(), observation=obs), CONTEXT))

    assert decision == {"type": "select_lineup", "bring": ROSTER[:4], "leads": ROSTER[:2]}
    prompt = provider.prompt()
    assert prompt["decision"] == "select_lineup"
    assert "known_sets" in prompt and "opponent" in prompt["known_sets"]
    assert agent.memory.my_lineup == ROSTER[:4]
    assert agent.memory.opp_cards[species_key("garchomp")]["types"] == ["DRAGON", "GROUND"]


def test_invalid_lineup_is_retried_once_then_accepted(tmp_path):
    provider = FakeProvider(
        {"bring": ["incineroar", "incineroar", "urshifu", "amoonguss"], "leads": ROSTER[:2], "reasoning_summary": "x"},
        {"bring": ROSTER[:4], "leads": ROSTER[:2], "reasoning_summary": "fixed"},
    )
    agent = _agent(provider, tmp_path)

    decision = _unwrap(agent.choose_action(_state(_lineup_action()), CONTEXT))

    assert decision["bring"] == ROSTER[:4]
    assert len(provider.calls) == 2
    assert "invalid" in provider.calls[1]["messages"][-1]["content"]
    assert agent.memory.fallbacks == 0


def test_every_model_failing_falls_back_to_the_deterministic_lineup(tmp_path):
    provider = FakeProvider(ProviderError("every model failed"))
    agent = _agent(provider, tmp_path)

    decision = _unwrap(agent.choose_action(_state(_lineup_action()), CONTEXT))

    # The fallback is the best COMPUTED lineup: a legal 4-of-6 with 2 leads from those 4.
    assert decision["type"] == "select_lineup"
    assert len(set(decision["bring"])) == 4 and set(decision["bring"]) <= set(ROSTER)
    assert len(set(decision["leads"])) == 2 and set(decision["leads"]) <= set(decision["bring"])
    assert agent.memory.fallbacks == 1
    assert len(provider.calls) == 1  # no retry after a provider error


# -- battle turn ----------------------------------------------------------------------------


def test_doubles_turn_builds_a_valid_payload_from_the_models_choice(tmp_path):
    slot_0 = [_move("fakeout", [1, 2], [{"target": 1, "side": "opponent", "species": "garchomp"},
                                        {"target": 2, "side": "opponent", "species": "amoonguss"}]),
              _move("protect", [])]
    slot_1 = [_move("woodhammer", [1, 2]), {"type": "switch", "species": "urshifu"}]
    provider = FakeProvider({"slot_0": {"option": 1, "target": 0}, "slot_1": {"option": 0, "target": 2},
                             "reasoning_summary": "Protect and hit Amoonguss."})
    agent = _agent(provider, tmp_path)
    obs = {"phase": "moving", "turn": 3, "weather": None, "team": {}, "opponent_team": {"p2: Garchomp": {"species": "garchomp"}}}

    decision = _unwrap(agent.choose_action(_state(_doubles_action(slot_0, slot_1), observation=obs), CONTEXT))

    assert decision == {"type": "doubles_turn",
                        "slot_0": {"type": "move", "move_id": "protect"},
                        "slot_1": {"type": "move", "move_id": "woodhammer", "target": 2}}
    schema = provider.calls[0]["schema"]
    assert schema["properties"]["slot_0"]["properties"]["target"] == {"type": "integer"}  # no nullable unions
    assert agent.memory.opp_lineup_seen == ["garchomp"]
    assert agent.memory.turns[0]["turn"] == 3


def test_bad_target_is_retried_then_falls_back_to_smoke_move(tmp_path):
    slot_0 = [_move("fakeout", [1, 2])]
    slot_1 = [PASS]
    bad = {"slot_0": {"option": 0, "target": 5}, "slot_1": {"option": 0, "target": 0}, "reasoning_summary": "x"}
    provider = FakeProvider(bad, bad)
    agent = _agent(provider, tmp_path)

    decision = _unwrap(agent.choose_action(_state(_doubles_action(slot_0, slot_1)), CONTEXT))

    assert decision == {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "fakeout", "target": 1},
                        "slot_1": {"type": "pass"}}
    assert len(provider.calls) == 2
    assert agent.memory.fallbacks == 1


def test_code_only_mode_plays_the_deterministic_turn(tmp_path):
    agent = _agent(None, tmp_path)
    decision = _unwrap(agent.choose_action(_state(_doubles_action([_move("tackle", [1, 2])], [PASS])), CONTEXT))
    assert decision["slot_0"] == {"type": "move", "move_id": "tackle", "target": 1}


# -- the outer guard ------------------------------------------------------------------------


def test_an_internal_bug_plays_the_safe_move_and_is_logged(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyError("simulated bug")

    monkeypatch.setattr(draft_rules, "choose_pick", boom)
    agent = _agent(None, tmp_path)

    decision = _unwrap(agent.choose_action(_draft_state(), CONTEXT))

    assert isinstance(decision, LegalAction) and decision.action_id.startswith("draft_pick:")
    assert agent.memory.fallbacks == 1
    lines = [json.loads(l) for l in (tmp_path / "match-1.jsonl").read_text().splitlines()]
    assert [l["kind"] for l in lines] == ["error", "fallback"]
    assert "simulated bug" in lines[0]["error"]


# -- log and secrets ---------------------------------------------------------------------------


def test_decision_log_records_each_decision_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TESTKEY-not-real")
    provider = FakeProvider({"bring": ROSTER[:4], "leads": ROSTER[:2], "reasoning_summary": "eak_live_leaked sk-ant-TESTKEY-not-real"})
    agent = _agent(provider, tmp_path)
    agent.choose_action(_draft_state(), CONTEXT)
    agent.choose_action(_state(_lineup_action()), CONTEXT)

    text = (tmp_path / "match-1.jsonl").read_text()
    lines = [json.loads(l) for l in text.splitlines()]
    assert [l["kind"] for l in lines] == ["draft", "lineup"]
    assert lines[1]["model"] == "fake-model" and lines[1]["latency_ms"] == 7
    assert lines[1]["provider_errors"] == []  # the first model answered; a Fable->Sonnet rescue would list Fable's error here
    assert "sk-ant-TESTKEY" not in text and "eak_live_leaked" not in text
    assert redact("key eak_live_abc123 and sk-ant-xyz") == "key [REDACTED] and [REDACTED]"


def test_create_agent_without_a_key_runs_code_only(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("agent.agent.load_dotenv", lambda *a, **k: None)
    agent = create_agent()
    assert isinstance(agent, PokemonAgent)
    assert "code-only" in capsys.readouterr().err


def test_draft_identifies_our_seat_from_the_agent_id_before_our_first_turn(tmp_path):
    memory = MatchMemory()
    obs = {"rosters": {"agent-a": [{"card_id": "vgc-garchomp", "species": "Garchomp"}], "agent-b": [{"card_id": "vgc-amoonguss", "species": "Amoonguss"}]},
           "current_seat": "agent-b", "available_cards": _draft_cards()}
    memory.observe_draft(obs, my_turn=False, agent_id="agent-a")
    assert memory.my_seat_key == "agent-a"
    assert species_key("Garchomp") in memory.my_cards and species_key("Amoonguss") in memory.opp_cards
