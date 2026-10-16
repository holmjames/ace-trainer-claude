"""The "reader" practice opponent (agent/versions/reader.py) and the simulator's oracle hook it reads us through
(sim/harness.py ``oracle_of``).

Live positions are replayed battle request by battle request (a restarted agent: no draft or Team Preview), with our
predicted turn written in by hand; the hook is checked on two real simulator games (Team Preview search off, tie-break
seeds pinned, so the code brain plays them the same way every time)."""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent, _payload
from agent.pokemon import battle as battle_rules
from agent.pokemon import tuning
from agent.pokemon.memory import observation_dict
from agent.versions import reader

ROOT = Path(__file__).resolve().parent.parent
LIVE = Path(__file__).parent / "fixtures" / "live"
ENGINE = ROOT / "sim" / "node_modules" / "pokemon-showdown"
needs_engine = pytest.mark.skipif(not shutil.which("node") or not ENGINE.exists(), reason="needs Node and `npm install` in sim/")


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


def _at(prefix: str, seat: int, turn: int):
    """A reader replayed (no predictions: the code brain) up to the first decision of ``turn``: (agent, state, context)."""
    agent = _quiet(reader.ReaderAgent())
    for state, ctx in _battle(prefix, seat):
        if observation_dict(state).get("turn") == turn:
            return agent, state, ctx
        agent.choose_action(state, ctx)
    raise AssertionError(f"turn {turn} not in {prefix}")


def _predict(agent, turn: int, slot_0: dict | None = None, slot_1: dict | None = None) -> None:
    """Write our predicted turn in by hand, as the shadow brain would have answered it (the built payload; our slot k
    is the reader's foe position k + 1, our target 1 / 2 the reader's slot 0 / 1)."""
    agent._prediction = {"turn": turn, "forced": False, "state_version": None, "template": {"slots": [{"slot": 0}, {"slot": 1}]},
                         "payload": {"type": "doubles_turn", "slot_0": slot_0 or {"type": "pass"}, "slot_1": slot_1 or {"type": "pass"}}}


def _hit(move: str, target: int) -> dict:
    return {"type": "move", "move_id": move, "target": target}


# -- the prediction, in the reader's frame -------------------------------------------------------------------------------


def test_predicted_play_maps_our_slots_and_targets_into_the_readers_frame():
    template = {"slots": [{"slot": 0, "options": [{"type": "move", "move_id": "protect", "targets": []},
                                                  {"type": "move", "move_id": "closecombat", "targets": [-2, 1, 2]}]},
                          {"slot": 1, "options": [{"type": "move", "move_id": "rockslide", "targets": []},
                                                  {"type": "switch", "species": "Incineroar"}]}]}
    by_index = reader.predicted_play(template, {"slot_0": {"option": 1, "target": 2}, "slot_1": {"option": 0, "target": 0}})
    assert by_index[1] == {"kind": "move", "move": "closecombat", "into": [1], "target": 2}
    assert by_index[2]["kind"] == "move" and by_index[2]["into"] == [0, 1]  # a spread move hits both of the reader's slots
    built = reader.predicted_play(template, {"type": "doubles_turn", "slot_0": _hit("protect", 0),
                                             "slot_1": {"type": "switch", "species": "Incineroar"}})
    assert built[1]["kind"] == "protect" and built[2]["kind"] == "switch"
    assert reader.predicted_play(template, None) == {}


# -- the best response, on live positions ----------------------------------------------------------------------------------


def test_protects_the_slot_our_predicted_play_would_ko():
    """c63f19be T9: Arcanine (slot 0, 20% HP) with no bench left. Predicted: their Incineroar Knocks it Off. The reader
    Protects it; with nothing predicted into it, it plays the code brain's attack."""
    agent, state, ctx = _at("c63f19be", 0, 9)
    obs = observation_dict(state)
    template = state.legal_actions[0].input["action"]
    agent.memory.observe_battle(obs)
    sheet = battle_rules.build_sheet(template, obs, agent.memory, agent.params)
    pos = next(t["position"] for t in sheet.threats if any(h["move"] == "knockoff" and h["into_slot"] == 0 for h in t["hits"]))
    _predict(agent, 9, **{f"slot_{pos - 1}": _hit("knockoff", 1)})
    answer = agent.choose_action(state, ctx)
    assert answer.action["slot_0"]["type"] == "move" and answer.action["slot_0"]["move_id"] == "protect"
    assert answer.action["slot_1"]["type"] == "move" and answer.action["slot_1"]["move_id"] != "protect"
    assert "slot 0 Protects" in answer.reasoning_summary
    assert agent.stats["dodge_protect"] == 1

    agent, state, ctx = _at("c63f19be", 0, 9)
    plain = agent.choose_action(state, ctx)
    assert plain.action["slot_0"].get("move_id") != "protect"  # the code brain alone attacks with it


def test_switches_the_targeted_slot_to_the_bench_pokemon_that_is_immune():
    """37a1ee8a T3: predicted Close Combat into slot 1; Gholdengo on the bench is immune to it: the reader sends it in."""
    agent, state, ctx = _at("37a1ee8a", 0, 3)
    _predict(agent, 3, slot_0=_hit("closecombat", 2))
    answer = agent.choose_action(state, ctx)
    assert answer.action["slot_1"] == {"type": "switch", "species": answer.action["slot_1"].get("species")}
    assert "gholdengo" in answer.action["slot_1"]["species"].lower()
    assert agent.stats["dodge_switch"] == 1


def test_does_not_attack_the_foe_we_are_predicted_to_protect():
    """The first turn of c63f19be on which the code brain hits one foe with a single-target move: we are predicted to
    Protect that foe, so no slot of the reader attacks it."""
    code = _quiet(PokemonAgent(None, version="code-only"))
    agent = _quiet(reader.ReaderAgent())
    for state, ctx in _battle("c63f19be", 0):
        obs = observation_dict(state)
        template = state.legal_actions[0].input["action"]
        if any(s.get("force_switch") for s in template["slots"]):
            code.choose_action(state, ctx)
            agent.choose_action(state, ctx)
            continue
        plain = code.choose_action(state, ctx).action
        aimed = [plain[k]["target"] for k in ("slot_0", "slot_1") if plain[k].get("type") == "move" and plain[k].get("target") in (1, 2)]
        if not aimed:
            assert agent.choose_action(state, ctx).action == plain
            continue
        foe = aimed[0]
        _predict(agent, obs.get("turn"), **{f"slot_{foe - 1}": _hit("protect", 0)})
        answer = agent.choose_action(state, ctx).action
        assert all(answer[k].get("target") != foe for k in ("slot_0", "slot_1") if answer[k].get("type") == "move")
        assert agent.stats["retarget"] + agent.stats["support"] >= 1
        return
    raise AssertionError("no single-target attack in c63f19be")


# -- fallbacks -----------------------------------------------------------------------------------------------------------


def test_without_a_prediction_it_plays_the_code_brain():
    code = []
    code_agent = _quiet(PokemonAgent(None, version="code-only"))
    bot = _quiet(reader.ReaderAgent())
    for state, ctx in _battle("c63f19be", 0):
        code.append(code_agent.choose_action(state, ctx).action)
        assert bot.choose_action(state, ctx).action == code[-1]


def test_an_error_in_the_read_plays_the_code_brain(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(reader.ReaderAgent, "respond", broken)
    code_agent = _quiet(PokemonAgent(None, version="code-only"))
    bot = _quiet(reader.ReaderAgent())
    for state, ctx in _battle("c63f19be", 0):
        _predict(bot, observation_dict(state).get("turn"), slot_0=_hit("knockoff", 1), slot_1=_hit("knockoff", 2))
        assert bot.choose_action(state, ctx).action == code_agent.choose_action(state, ctx).action
    assert bot.stats["read_errors"] >= 1


def test_a_broken_view_costs_only_the_prediction():
    bot = _quiet(reader.ReaderAgent())
    bot._prediction = {"turn": 1}
    bot.observe_opponent(object(), None)  # no legal_actions, no observation
    assert bot._prediction is None and bot.stats["shadow_errors"] == 1
    state, ctx = _battle("c63f19be", 0)[0]
    _predict(bot, observation_dict(state).get("turn"), slot_0={"type": "move", "move_id": "nosuchmove", "target": 7})
    answer = bot.choose_action(state, ctx)  # a prediction it cannot price: the code brain's turn
    assert answer.action == _quiet(PokemonAgent(None, version="code-only")).choose_action(state, ctx).action


def test_default_tuning_never_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_TUNE", json.dumps({"protect_bonus_guaranteed": 999}))
    bot = reader.create_agent()
    assert bot.params == dict(tuning.DEFAULTS) and bot.shadow.params == dict(tuning.DEFAULTS)
    assert bot.wants_opponent_view is True


# -- the oracle hook in the simulator ------------------------------------------------------------------------------------


class _Rec:
    """Records every answer of the player it wraps (the hook finds a reader through ``inner``)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.moves: list[tuple[str, str]] = []
        self.views: list[str] = []  # every state it was asked about, as JSON

    def choose_action(self, state, context):
        self.views.append(json.dumps(state.raw, sort_keys=True, default=str))
        out = self.inner.choose_action(state, context)
        self.moves.append((state.phase, json.dumps(_payload(getattr(out, "action", out)), sort_keys=True, default=str)))
        return out


class _Watching(reader.ReaderAgent):
    """The reader with its read off: the code brain's every move, while the shadow watches through the hook."""

    def __init__(self) -> None:
        super().__init__()
        self.timeline: list[tuple] = []
        self.views: list[str] = []  # every state the hook showed it, as JSON
        for agent in (self, self.shadow):
            agent.params["lineup_sim_budget_s"] = 0.0  # Team Preview off the wall clock: the same game every run

    def observe_opponent(self, state, context):
        obs = observation_dict(state)
        self.timeline.append(("seen", state.phase, obs.get("turn"), getattr(state, "state_version", None)))
        self.views.append(json.dumps(state.raw, sort_keys=True, default=str))
        super().observe_opponent(state, context)

    def choose_action(self, state, context):
        self.timeline.append(("own", state.phase, observation_dict(state).get("turn"), getattr(state, "state_version", None)))
        return super().choose_action(state, context)

    def respond(self, template, obs, sheet, answer, play):
        return answer, [], {}


def _code() -> PokemonAgent:
    return _quiet(PokemonAgent(None, version="code-only", params={**tuning.DEFAULTS, "lineup_sim_budget_s": 0.0}))


def _game(p1, p2, seed: int) -> dict:
    from sim import draft_eval, harness

    bridge = harness.Bridge()
    try:
        with draft_eval.pin_tiebreak(seed, False):
            result = harness.play_game(bridge, f"reader-hook-{seed}", {"p1": p1, "p2": p2}, {"p1": "a", "p2": "b"},
                                       random.Random(seed), verbose=False)
    finally:
        bridge.close()
    return {k: result[k] for k in ("winner_side", "turns", "teams", "rosters", "rejected")}


def _no_deadline(view: str) -> str:
    """A state's JSON without the draft's wall-clock deadline (translate.draft_state stamps it when the state is built)."""
    return re.sub(r'"decision_deadline_at": "[^"]*"', '"decision_deadline_at": ""', view)


@needs_engine
@pytest.mark.parametrize("seat", ["p2", "p1"])
def test_the_oracle_hook_changes_nothing_for_normal_players_and_shows_every_decision_first(seat):
    seed = 31
    other = "p1" if seat == "p2" else "p2"
    plain_players = {"p1": _Rec(_code()), "p2": _Rec(_code())}
    plain = _game(plain_players["p1"], plain_players["p2"], seed)
    watcher = _quiet(_Watching())
    players = {seat: _Rec(watcher), other: _Rec(_code())}
    watched = _game(players["p1"], players["p2"], seed)
    assert watched == plain  # the same game, move for move, with the hook and the shadow brain running
    assert players["p1"].moves == plain_players["p1"].moves and players["p2"].moves == plain_players["p2"].moves
    assert not hasattr(plain_players[seat].inner, "wants_opponent_view")
    # it was shown exactly what the other seat was asked about, every decision once, in order
    assert [_no_deadline(v) for v in watcher.views] == [_no_deadline(v) for v in players[other].views]
    seen = [t for t in watcher.timeline if t[0] == "seen"]
    assert len([t for t in seen if t[1] == "draft"]) == 6 and len([t for t in seen if t[1] == "team_preview"]) == 1
    for i, step in enumerate(watcher.timeline):  # each of its battle turns: the other seat's state of that turn came first
        if step[0] == "own" and step[1] in ("team_preview", "moving") and any(t[1] == step[1] and t[2] == step[2] for t in seen):
            assert any(t[0] == "seen" and t[1] == step[1] and t[2] == step[2] for t in watcher.timeline[:i])
    assert watcher.stats["predictions"] == len([m for m in players[other].moves if m[0] == "moving"])
