"""Workstream F (reliability): a restarted worker must not play blind, the guard must never raise, the pre-flight must
prove the model answers, the launcher must survive its terminal, and the versions we tested are the versions we run.

F1 — restart rebuild. The runtime spawns a fresh worker per match; a worker started mid-battle has no draft or Team
Preview memory. ``MatchMemory.observe_battle`` rebuilds the notebook from the observation: every visible card from the
live catalog, both rosters and who-entered-when from the protocol log. The test below replays every live session twice
(one agent fed everything in order, and a brand-new agent per battle state) and asserts the same turn sheet, roster and
known sets.

Best-effort by design (documented here, not asserted away silently):
- with a protocol log, "who just came in" (Fake Out is live), "who Protected last turn" and both sides' Protect
  streaks are read from the log, which is what Showdown actually counted (a clicked Protect that never ran, a flinch
  that spent Fake Out's first turn) and is the same whether the agent lived through the match or was just started;
  without a log (unit tests, an old simulator) the decision-based bookkeeping applies and a restart is still blind
  to the previous turns;
- an opponent Protect inferred from an unchanged HP bar needs our previous payload and the previous HP, which a
  fresh agent does not have; it is only used without a log;
- a species missing from ``data/cards.json`` cannot be rebuilt (the drill session's Iron Valiant): both replays are
  equally blind to it (the moves shown so far are kept), and ``test_catalog_covers_live_sessions`` reports the gap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent, battle_roster
from agent.pokemon import data
from agent.pokemon.memory import observation_dict, species_key
from test_live_semantics import LIVE, _sequence

# -- F1: restart rebuild -----------------------------------------------------------------------------------------


def _sessions() -> list[Path]:
    return sorted(p for p in LIVE.iterdir() if p.is_dir()) if LIVE.exists() else []


def _seats(folder: Path) -> list[int]:
    seats = set()
    for path in folder.glob("*.json"):
        m = re.search(r"-seat(\d+)\.json$", path.name)
        if m:
            seats.add(int(m.group(1)))
    return sorted(seats)


CASES = [(folder, seat) for folder in _sessions() for seat in _seats(folder)]


def _agent(log_dir: Path, name: str) -> PokemonAgent:
    return PokemonAgent(None, version=name, log_dir=log_dir, capture_dir="", log=lambda *_: None)


def _context(folder: Path, raw: dict, seat: int) -> DecisionContext:
    agent_id = (raw.get("current_actor") or {}).get("agent_id") or "me"
    return DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                           agent_id=agent_id, seat_position=seat)


def _last_turn_row(log_dir: Path) -> dict:
    rows = [json.loads(line) for f in log_dir.glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    turns = [r for r in rows if r.get("kind") == "turn"]
    assert turns, "no turn row was logged"
    assert not [r for r in rows if r.get("kind") in ("error", "fallback")], "decision logic raised"
    return turns[-1]


def _snapshot(agent: PokemonAgent, state: GameState, log_dir: Path) -> dict:
    """What the model would be shown for this state: the sheet, the roster and both known-set lists."""
    row = _last_turn_row(log_dir)
    obs = observation_dict(state)
    template = state.legal_actions[0].input.get("action") or {}
    roster = battle_roster(obs, template, agent.memory)
    mine = sorted(agent.memory.known_sets("mine", only=roster["ours"]["in_this_battle"]), key=lambda c: species_key(c["species"]))
    theirs = sorted(agent.memory.known_sets("opponent", status=roster["theirs"]["status"]), key=lambda c: species_key(c["species"]))
    return {"turn_sheet": row["turn_sheet"], "battle_roster": row["battle_roster"], "known_mine": mine, "known_opp": theirs}


def _first_difference(a, b, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                return f"{path}.{key}: only in {'full replay' if key in a else 'fresh agent'}"
            found = _first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} vs {len(b)} entries: {a!r:.300} != {b!r:.300}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r:.200} != {b!r:.200}"


REFRESH = "python scripts/pool_from_live.py (rewrites data/cards.json from the captured drafts)"


def _catalog_gaps(folder: Path, seat: int) -> set[str]:
    """Species of this session (both sides, from Team Preview's |poke| lines) the catalog cannot rebuild."""
    moving = [p for p in _sequence(folder, seat) if p.name.startswith("moving")]
    if not moving:
        return set()
    obs = observation_dict(GameState.from_mcp_state(json.loads(moving[0].read_text())))
    gaps = set()
    for line in obs.get("protocol_log") or []:
        parts = str(line).split("|")
        if len(parts) > 3 and parts[1] == "poke":
            name = parts[3].split(",")[0].strip()
            if name and not data.catalog_card(species=name):
                gaps.add(name)
    return gaps


@pytest.mark.skipif(not CASES, reason="no live captures")
@pytest.mark.parametrize("folder, seat", CASES, ids=[f"{f.name[:8]}/seat{s}" for f, s in CASES])
def test_a_fresh_agent_sees_the_same_sheet_as_a_full_replay(folder, seat, tmp_path):
    """Every battle state of every live session: an agent restarted on that state builds the same turn sheet, roster
    and known sets as the agent that lived through the draft and Team Preview."""
    from agent.pokemon.memory import _complete

    gaps = _catalog_gaps(folder, seat)
    full = _agent(tmp_path / "full", "full-replay")
    diffs: list[str] = []
    compared = 0
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        ctx = _context(folder, raw, seat)
        full.choose_action(state, ctx)
        if state.phase != "moving":
            continue
        # A species the draft taught the full replay but the catalog lacks: a restart IS blind to it, by data, not by
        # code. Sessions without draft captures (the drill) are equally blind both ways and still compared.
        blind = sorted(g for g in gaps if _complete(full.memory.my_cards.get(species_key(g)) or full.memory.opp_cards.get(species_key(g)) or {}))
        if blind:
            pytest.skip(f"the catalog lacks {blind}: a restart here is blind to them until {REFRESH}")
        expected = _snapshot(full, state, tmp_path / "full")
        fresh_dir = tmp_path / f"fresh-{path.stem}"
        fresh = _agent(fresh_dir, "fresh")
        fresh.choose_action(state, ctx)
        actual = _snapshot(fresh, state, fresh_dir)
        compared += 1
        found = _first_difference(expected, actual)
        if found:
            diffs.append(f"{path.name}: {found}")
    assert compared, "no battle state in this capture"
    assert not diffs, f"{len(diffs)}/{compared} states differ after a restart:\n  " + "\n  ".join(diffs[:12])


@pytest.mark.skipif(not CASES, reason="no live captures")
def test_a_fresh_agent_knows_every_drafted_set_from_the_catalog(tmp_path):
    """A restart on the first battle state of the newest session recovers full sets (moves, EVs, item) for everything
    on both sides, marked as catalog knowledge, and the opponent's unseen reserves."""
    folder, seat, moving = next((f, s, m) for f, s in reversed(CASES)
                                for m in [[p for p in _sequence(f, s) if p.name.startswith("moving")]] if m and not _catalog_gaps(f, s))
    raw = json.loads(moving[0].read_text())
    state = GameState.from_mcp_state(raw)
    agent = _agent(tmp_path, "fresh")
    agent.choose_action(state, _context(folder, raw, seat))
    memory = agent.memory
    obs = observation_dict(state)
    assert memory.roster_by_side["mine"] and memory.roster_by_side["theirs"], "no |poke| lines read"
    for side, cards in (("mine", memory.my_cards), ("theirs", memory.opp_cards)):
        for name in memory.roster_by_side[side]:
            key = species_key(name)
            if not data.catalog_card(species=name):
                continue  # documented gap: a species never offered to us live
            card = cards[key]
            assert len(card.get("moves") or []) == 4 and card.get("evs") and card.get("item"), (side, key, card)
            assert card.get("source") == "catalog", (side, key)
    assert len(memory.opp_cards) == 6 and len(memory.my_cards) == 6
    assert len(memory.my_lineup) == 4, memory.my_lineup
    assert set(memory.last_active) <= set(memory.my_lineup)
    assert memory.log_turn == obs.get("turn")


def test_catalog_knowledge_never_overwrites_a_drafted_card():
    from agent.pokemon.memory import MatchMemory

    memory = MatchMemory()
    drafted = {"card_id": "vgc-test", "species": "Incineroar", "item": "Sitrus Berry", "ability": "Intimidate", "nature": "Careful",
               "evs": {"hp": 252}, "moves": ["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"]}
    memory.opp_cards["incineroar"] = dict(drafted)
    memory.observe_battle({"turn": 3, "opponent_active_pokemon": [{"species": "incineroar", "item": "unknown_item", "moves": ["fakeout"]}],
                           "active_pokemon": [{"species": "garchomp"}]})
    assert memory.opp_cards["incineroar"]["item"] == "Sitrus Berry" and "source" not in memory.opp_cards["incineroar"]
    assert memory.opp_cards["incineroar"]["moves"] == drafted["moves"]
    # A species the catalog knows is filled in; one it does not know keeps what the battle showed, nothing invented.
    memory.observe_battle({"turn": 4, "opponent_active_pokemon": [{"species": "incineroar"}, {"species": "garchomp", "types": ["DRAGON", "GROUND"], "moves": ["earthquake"]}],
                           "active_pokemon": [{"species": "rillaboom"}]})
    chomp = memory.opp_cards["garchomp"]
    if data.catalog_card(species="Garchomp"):
        assert chomp.get("source") == "catalog" and len(chomp["moves"]) == 4
    else:
        assert chomp.get("moves") == ["earthquake"] and chomp.get("types") == ["DRAGON", "GROUND"]
    memory.observe_battle({"turn": 5, "opponent_active_pokemon": [{"species": "incineroar"}, {"species": "Not A Pokemon", "moves": ["tackle"]}],
                           "active_pokemon": [{"species": "rillaboom"}]})
    assert memory.opp_cards["notapokemon"].get("moves") == ["tackle"] and "source" not in memory.opp_cards["notapokemon"]


def test_catalog_covers_live_sessions():
    """Every species that has appeared live must be in data/cards.json, or a worker restarted mid-battle plays blind
    to it (its moves, item, EVs). The fix is one command, run from the production checkout before the freeze."""
    missing: dict[str, set[str]] = {}
    for folder, seat in CASES:
        for name in _catalog_gaps(folder, seat):
            missing.setdefault(name, set()).add(folder.name[:8])
    if missing:
        pytest.skip(f"data/cards.json lacks {sorted(missing)} (seen in {missing}): run {REFRESH}")


# -- F2: the guard's own guard ---------------------------------------------------------------------------------

CTX = DecisionContext(session_id="match-f", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="agent-a", seat_position=0)
GOOD_SLOTS = [
    {"slot": 0, "board_position": -1, "active": {"species": "incineroar"}, "force_switch": False,
     "options": [{"type": "move", "move_id": "fakeout", "targets": [1, 2]}, {"type": "switch", "species": "amoonguss"}]},
    {"slot": 1, "board_position": -2, "active": {"species": "rillaboom"}, "force_switch": False,
     "options": [{"type": "move", "move_id": "woodhammer", "targets": [1, 2]}, {"type": "switch", "species": "amoonguss"}]},
]


def _state(*legal: dict, observation=None, version: int = 3, phase: str = "moving") -> GameState:
    return GameState.from_mcp_state({"session_id": "match-f", "status": "in_progress", "phase": phase, "is_current_actor": True,
                                     "state_version": version, "observation": observation or {},
                                     "legal_actions": {"session_id": "match-f", "state_version": version, "actions": list(legal)}})


def _templated(action_id: str, template) -> dict:
    return {"action_id": action_id, "label": action_id,
            "input": {"session_id": "match-f", "action_id": action_id, "state_version": 3, "action": template}}


def _doubles(slots) -> GameState:
    return _state(_templated("doubles_turn", {"type": "doubles_turn", "slots": slots, "target_legend": {}, "instructions": "pick"}))


def _rows(log_dir: Path) -> list[dict]:
    return [json.loads(line) for f in Path(log_dir).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]


def _raise(exc):
    raise exc


def test_guard_plays_the_smoke_move_when_the_decision_logic_raises(tmp_path, monkeypatch):
    import agent.agent as agent_mod

    monkeypatch.setattr(agent_mod.battle_rules, "build_sheet", lambda *a, **k: _raise(RuntimeError("sheet bug")))
    agent = _agent(tmp_path, "f2")
    decision = agent.choose_action(_doubles(GOOD_SLOTS), CTX)
    assert decision.action == {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "fakeout", "target": 1},
                               "slot_1": {"type": "move", "move_id": "woodhammer", "target": 1}}
    assert decision.reasoning_summary.startswith("Fallback:") and agent.memory.fallbacks == 1
    rows = _rows(tmp_path)
    assert [r["kind"] for r in rows if r["kind"] in ("error", "fallback")] == ["error", "fallback"]
    assert "sheet bug" in next(r for r in rows if r["kind"] == "error")["error"]


def test_guard_builds_a_minimal_payload_when_even_the_smoke_agent_cannot_read_the_request(tmp_path):
    """A slot whose options the smoke agent rejects (it raises SmokeAgentError on anything unreadable) still leaves a
    legal first move in the other slot: the hand-built payload plays it and passes the broken slot."""
    broken = [GOOD_SLOTS[0], {"slot": 1, "board_position": -2, "active": None, "force_switch": False, "options": "garbage"}]
    agent = _agent(tmp_path, "f2")
    decision = agent.choose_action(_doubles(broken), CTX)
    assert decision.action == {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "fakeout", "target": 1}, "slot_1": {"type": "pass"}}
    rows = _rows(tmp_path)
    assert any("smoke agent" in (r.get("error") or "") for r in rows if r["kind"] == "error")
    assert "hand-built" in next(r for r in rows if r["kind"] == "fallback")["reason"]


def test_a_log_writer_and_a_console_that_raise_never_cost_a_move(tmp_path, monkeypatch):
    from agent.pokemon.log import DecisionLog

    monkeypatch.setattr(DecisionLog, "write", lambda self, kind, **fields: _raise(RuntimeError("disk on fire")))
    agent = _agent(tmp_path, "f2")
    agent._print = lambda line: _raise(OSError("stderr closed"))
    decision = agent.choose_action(_doubles(GOOD_SLOTS), CTX)  # the normal path: a computed turn, not a fallback
    assert decision.action["type"] == "doubles_turn" and not decision.reasoning_summary.startswith("Fallback:")
    import agent.agent as agent_mod

    monkeypatch.setattr(agent_mod.battle_rules, "build_sheet", lambda *a, **k: _raise(RuntimeError("sheet bug")))
    decision = agent.choose_action(_doubles(GOOD_SLOTS), CTX)  # the guard path, logging and printing both broken
    assert decision.reasoning_summary.startswith("Fallback:") and decision.action["slot_0"]["move_id"] == "fakeout"


def test_a_log_that_cannot_be_opened_never_costs_a_move(tmp_path, monkeypatch):
    import agent.agent as agent_mod

    monkeypatch.setattr(agent_mod, "DecisionLog", lambda *a, **k: _raise(PermissionError("logs/ is read-only")))
    agent = _agent(tmp_path, "f2")
    decision = agent.choose_action(_doubles(GOOD_SLOTS), CTX)
    assert decision.action["type"] == "doubles_turn" and not decision.reasoning_summary.startswith("Fallback:")


def test_minimal_legal_payload_covers_every_decision_kind():
    from agent.agent import minimal_legal_payload

    # doubles: first move into the first opposing target; a pass-only slot passes; two switches never pick the same one
    assert minimal_legal_payload(_doubles(GOOD_SLOTS)) == {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "fakeout", "target": 1},
                                                           "slot_1": {"type": "move", "move_id": "woodhammer", "target": 1}}
    switches = [{"slot": 0, "options": [{"type": "switch", "species": "amoonguss"}, {"type": "switch", "species": "garchomp"}]},
                {"slot": 1, "options": [{"type": "switch", "species": "amoonguss"}, {"type": "switch", "species": "garchomp"}]}]
    assert minimal_legal_payload(_doubles(switches)) == {"type": "doubles_turn", "slot_0": {"type": "switch", "species": "amoonguss"},
                                                         "slot_1": {"type": "switch", "species": "garchomp"}}
    odd = [{"slot": 0, "options": [{"type": "pass"}]}, {"options": [{"type": "move", "move_id": "tailwind", "targets": []}, {"type": "move"}]}]
    assert minimal_legal_payload(_doubles(odd)) == {"type": "doubles_turn", "slot_0": {"type": "pass"}, "slot_1": {"type": "move", "move_id": "tailwind"}}
    assert minimal_legal_payload(_doubles("not even a list")) == {"type": "doubles_turn", "slot_0": {"type": "pass"}, "slot_1": {"type": "pass"}}
    # Team Preview: the first four distinct of the roster, else the observation's roster
    lineup = _templated("select_lineup", {"type": "select_lineup", "roster": ["a", "b", "a", "c", "d", "e"]})
    assert minimal_legal_payload(_state(lineup, phase="team_preview")) == {"type": "select_lineup", "bring": ["a", "b", "c", "d"], "leads": ["a", "b"]}
    short = _templated("select_lineup", {"type": "select_lineup", "roster": None})
    obs = {"your_roster": [{"species": s} for s in ("w", "x", "y", "z", "v", "u")]}
    assert minimal_legal_payload(_state(short, observation=obs, phase="team_preview"))["bring"] == ["w", "x", "y", "z"]
    with pytest.raises(ValueError):
        minimal_legal_payload(_state(short, phase="team_preview"))
    # a draft pick (or any plain legal action): the first one
    pick = {"action_id": "draft_pick:vgc-garchomp", "label": "Garchomp", "input": {}}
    assert minimal_legal_payload(_state(pick, phase="draft")).action_id == "draft_pick:vgc-garchomp"
    with pytest.raises(ValueError):
        minimal_legal_payload(_state(phase="draft"))


def test_guard_re_raises_only_when_nothing_legal_can_be_built(tmp_path):
    """Team Preview with a two-species roster and no observation: the smoke agent and the builder both fail, so the
    original error reaches the runner (which reports it) instead of a guessed payload."""
    agent = _agent(tmp_path, "f2")
    state = _state(_templated("select_lineup", {"type": "select_lineup", "roster": ["a", "b"]}), phase="team_preview")
    with pytest.raises(Exception):
        agent.choose_action(state, CTX)
    assert [r["kind"] for r in _rows(tmp_path) if r["kind"] == "error"] == ["error", "error", "error"]


# -- F3: the pre-flight proves a model answers; loud when none does; one summary line per match -----------------


class _FakeProvider:
    """A scripted provider chain: answers in order; an exception is raised instead."""

    def __init__(self, *answers, model="claude-primary", last_model=None):
        self.answers = list(answers)
        self.model = model
        self.last_model = last_model or model
        self.last_latency_ms = 42
        self.last_usage = None
        self.last_errors: list[str] = []
        self.calls: list[tuple] = []

    def complete_structured(self, messages, schema_name, schema):
        self.calls.append((messages, schema_name, schema))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def __repr__(self):
        return f"FakeProvider({self.model!r})"


def test_probe_makes_one_tiny_structured_call_and_reports_who_answered():
    from agent.llm.anthropic_provider import PROBE_SCHEMA, probe

    provider = _FakeProvider({"ok": True}, last_model="claude-fallback")
    provider.last_errors = ["claude-primary: timed out after 18s"]
    result = probe(provider)
    assert result == {"model": "claude-fallback", "latency_ms": 42, "errors": ["claude-primary: timed out after 18s"]}
    messages, schema_name, schema = provider.calls[0]
    assert schema == PROBE_SCHEMA and schema_name == "preflight_probe" and len(messages) == 2


def test_probe_fails_on_a_wrong_answer_or_a_dead_chain():
    from examples.llm.providers import ProviderError

    from agent.llm.anthropic_provider import probe

    with pytest.raises(ProviderError, match="probe answer"):
        probe(_FakeProvider({"ok": False}))
    with pytest.raises(ProviderError, match="every model failed"):
        probe(_FakeProvider(ProviderError("every model failed: a: timed out | b: HTTP 429")))


def test_check_tournament_probe_line_through_the_agents_provider_chain(monkeypatch):
    import agent.__main__ as agent_main

    answered = _FakeProvider({"ok": True})
    line, failed = agent_main._probe_model(PokemonAgent(answered, log=lambda *_: None))
    assert (line, failed) == ("model probe: claude-primary 42 ms", False)

    rescued = _FakeProvider({"ok": True}, last_model="claude-fallback")
    rescued.last_errors = ["claude-primary: request failed (HTTP 529)"]
    line, failed = agent_main._probe_model(PokemonAgent(rescued, log=lambda *_: None))
    assert not failed and line.startswith("model probe: claude-fallback 42 ms (the fallback answered; the primary model failed: claude-primary: request failed (HTTP 529)")

    from examples.llm.providers import ProviderError

    dead = _FakeProvider(ProviderError("every model failed: claude-primary: request failed (HTTP 401) [authentication_error: invalid x-api-key]"))
    line, failed = agent_main._probe_model(PokemonAgent(dead, log=lambda *_: None))
    assert failed and line.startswith("model probe failed through FakeProvider('claude-primary'): every model failed")

    line, failed = agent_main._probe_model(PokemonAgent(None, log=lambda *_: None))  # a reverted .env: code-only
    assert failed and "NO MODEL PROVIDER" in line and "CODE-ONLY" in line

    from examples import smoke_agent

    line, failed = agent_main._probe_model(smoke_agent.create_agent())  # a plain function: nothing to probe
    assert not failed and line.startswith("model probe: skipped")

    crashing = _FakeProvider(RuntimeError("socket exploded"))
    line, failed = agent_main._probe_model(PokemonAgent(crashing, log=lambda *_: None))
    assert failed and "RuntimeError: socket exploded" in line


def test_every_model_failing_prints_the_all_models_failed_banner(tmp_path):
    from examples.llm.providers import ProviderError

    lines: list[str] = []
    provider = _FakeProvider(ProviderError("every model failed: claude-primary: timed out after 18s | claude-fallback: timed out after 8s"))
    agent = PokemonAgent(provider, version="f3", log_dir=tmp_path, capture_dir="", log=lines.append)
    decision = agent.choose_action(_doubles(GOOD_SLOTS), CTX)
    assert decision.action["type"] == "doubles_turn" and not decision.reasoning_summary.startswith("Fallback:")
    banner = [line for line in lines if "ALL MODELS FAILED" in line]
    assert len(banner) == 1 and "timed out after 18s" in banner[0] and "playing the computed move" in banner[0]
    assert agent._all_failed == 1


def test_match_summary_counts_turns_by_model_and_prints_once_when_the_match_ends(tmp_path):
    from examples.llm.providers import ProviderError

    lines: list[str] = []
    good = {"slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": 1}, "win_condition": "w", "reasoning_summary": "attack"}
    provider = _FakeProvider(good, good, ProviderError("every model failed: x"))
    agent = PokemonAgent(provider, version="f3", log_dir=tmp_path, capture_dir="", log=lines.append)
    agent.choose_action(_doubles(GOOD_SLOTS), CTX)  # primary answers
    provider.last_model = "claude-fallback"
    agent.choose_action(_doubles(GOOD_SLOTS), CTX)  # the fallback model answers
    agent.choose_action(_doubles(GOOD_SLOTS), CTX)  # nobody answers: code
    agent.on_action_result({"accepted": True, "status": "in_progress", "state_version": 4}, CTX)
    assert not [line for line in lines if "match summary" in line]
    agent.on_action_result({"accepted": True, "status": "completed", "state_version": 5}, CTX)
    agent.on_action_result({"accepted": True, "status": "completed", "state_version": 5}, CTX)
    agent.print_match_summary()
    summary = [line for line in lines if "match summary" in line]
    assert len(summary) == 1, summary
    assert "3 battle decisions: 1 by claude-primary, 1 by the fallback model, 1 code-only" in summary[0]
    assert "Team Preview by code" in summary[0] and "all-models-failed 1" in summary[0]
    assert [r["kind"] for r in _rows(tmp_path)][-1] == "summary"
    # a worker that never got a decision stays quiet
    quiet: list[str] = []
    PokemonAgent(None, version="f3", log_dir=tmp_path, capture_dir="", log=quiet.append).print_match_summary()
    assert quiet == []


def test_on_action_result_never_raises(tmp_path):
    agent = PokemonAgent(None, version="f3", log_dir=tmp_path, capture_dir="", log=lambda line: _raise(OSError("closed")))
    agent.memory.llm_calls = 1
    for result in (None, "x", {"is_terminal": True}, {"state": {"is_terminal": True}}, {"error": "RUNTIME_TEMPORARILY_UNAVAILABLE"}):
        agent.on_action_result(result, CTX)
    assert agent._summarized


def test_launcher_script_parses_and_stops_on_a_failed_preflight():
    import subprocess

    script = Path(__file__).parent.parent / "scripts" / "run_tournament.sh"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    text = script.read_text()
    assert "pipefail" in text.split("check-tournament")[0], "pipefail must be set before the pre-flight's tee pipeline"


# -- F4: circuit breaker -------------------------------------------------------------------------------------------


def _breaker_agent(tmp_path, *answers):
    from examples.llm.base import Choice

    lines: list[str] = []
    provider = _FakeProvider(*answers)
    agent = PokemonAgent(provider, version="f4", log_dir=tmp_path, capture_dir="", log=lines.append)
    choice = Choice(kind="doubles_turn", prompt={}, schema={"type": "object"}, build=lambda answer: dict(answer), fallback=lambda: {"computed": True})
    return agent, provider, choice, lines


def test_two_consecutive_all_model_failures_skip_the_model_once_then_retry(tmp_path):
    from examples.llm.providers import ProviderError

    dead = ProviderError("every model failed: a: timed out after 18s | b: timed out after 8s")
    answer = {"slot_0": {"option": 0}, "reasoning_summary": "ok"}
    agent, provider, choice, lines = _breaker_agent(tmp_path, dead, dead, answer, dead, dead, answer)

    value, _, info = agent._ask(choice, {}, kind="turn")  # failure 1
    assert value == {"computed": True} and "breaker" not in info and not agent._breaker_open
    value, _, info = agent._ask(choice, {}, kind="turn")  # failure 2: the breaker opens
    assert value == {"computed": True} and info["breaker"] == "open" and agent._breaker_open
    assert len(provider.calls) == 2
    value, _, info = agent._ask(choice, {}, kind="turn")  # skipped: no model call at all, code plays #1 at once
    assert value == {"computed": True} and info["breaker"] == "skipped" and info["fallback"] is True
    assert len(provider.calls) == 2 and agent._breaker_skips == 1 and not agent._breaker_open
    value, answered, info = agent._ask(choice, {}, kind="turn")  # retried: the model answers, the breaker closes
    assert answered == answer and info["breaker"] == "closed" and agent._consecutive_failures == 0
    assert len(provider.calls) == 3

    # an outage that goes on: fail, fail, skip, fail (3rd in a row) -> skip again, then the model comes back
    agent._ask(choice, {}, kind="turn")
    agent._ask(choice, {}, kind="turn")
    assert agent._breaker_open and agent._consecutive_failures == 2
    assert agent._ask(choice, {}, kind="turn")[2]["breaker"] == "skipped"
    value, answered, info = agent._ask(choice, {}, kind="turn")
    assert answered == answer and info["breaker"] == "closed" and len(provider.calls) == 6
    transitions = [line for line in lines if "circuit breaker" in line]
    assert [("OPEN" in t, "SKIPPING" in t, "CLOSED" in t) for t in transitions] == [
        (True, False, False), (False, True, False), (False, False, True), (True, False, False), (False, True, False), (False, False, True)]
    assert agent.memory.llm_calls == 6 and agent._breaker_skips == 2 and agent._all_failed == 4


def test_one_failure_followed_by_an_answer_never_trips_the_breaker(tmp_path):
    from examples.llm.providers import ProviderError

    answer = {"slot_0": {"option": 0}, "reasoning_summary": "ok"}
    agent, provider, choice, lines = _breaker_agent(tmp_path, ProviderError("every model failed: x"), answer,
                                                    ProviderError("every model failed: x"), answer)
    for _ in range(4):
        agent._ask(choice, {}, kind="turn")
    assert len(provider.calls) == 4 and agent._breaker_skips == 0
    assert not [line for line in lines if "OPEN" in line]


def test_the_breaker_skip_is_visible_in_the_decision_log(tmp_path):
    from examples.llm.providers import ProviderError

    dead = ProviderError("every model failed: x")
    lines: list[str] = []
    provider = _FakeProvider(dead, dead)
    agent = PokemonAgent(provider, version="f4", log_dir=tmp_path, capture_dir="", log=lines.append)
    for _ in range(3):
        agent.choose_action(_doubles(GOOD_SLOTS), CTX)
    turns = [r for r in _rows(tmp_path) if r["kind"] == "turn"]
    assert [t.get("breaker") for t in turns] == [None, "open", "skipped"]
    assert turns[2]["fallback"] is True and turns[2]["model"] is None and len(provider.calls) == 2
    assert "breaker skips 1" in agent.match_summary()


# -- F5: the launcher runs detached, can be stopped, and is testable without a tournament ---------------------------

import os
import signal
import subprocess
import sys
import time

SCRIPT = Path(__file__).parent.parent / "scripts" / "run_tournament.sh"
STUB = """#!/usr/bin/env bash
# Stands in for the venv python in launcher tests: `-c CODE ...` runs the real python (the setsid shim);
# `-m agent --check-tournament` exits with $FAKE_PREFLIGHT_RC; `-m agent --tournament ...` is a runtime that prints
# its arguments and waits for SIGINT/SIGTERM.
if [ "${1:-}" = "-c" ]; then exec "$FAKE_REAL_PYTHON" "$@"; fi
if [ "${3:-}" = "--check-tournament" ]; then echo "fake pre-flight rc=$FAKE_PREFLIGHT_RC"; exit "$FAKE_PREFLIGHT_RC"; fi
echo "fake runtime started: $*"
trap 'echo "fake runtime: signal, exiting"; exit 0' INT TERM
while true; do sleep 0.2; done
"""


def _launcher_env(tmp_path: Path, preflight_rc: int = 0) -> dict:
    stub = tmp_path / "fake_python.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    return {**os.environ, "TOURNAMENT_LOG_DIR": str(tmp_path / "logs"), "TOURNAMENT_PYTHON": str(stub),
            "AGENT_CAPTURE_DIR": str(tmp_path / "capture"), "FAKE_PREFLIGHT_RC": str(preflight_rc), "FAKE_REAL_PYTHON": sys.executable}


def _launcher(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_for(condition, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.1)


def _log_text(tmp_path: Path) -> str:
    return "".join(p.read_text() for p in sorted((tmp_path / "logs").glob("runtime-tournament-*.log")))


def test_launcher_dry_run_prints_the_plan_and_starts_nothing(tmp_path):
    env = _launcher_env(tmp_path)
    result = _launcher(env, "--dry-run", "--match")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "-m agent --tournament --match" in result.stdout and str(tmp_path / "logs" / "runtime-tournament.pid") in result.stdout
    assert "nothing starts" in result.stdout and "--stop" in result.stdout
    assert not (tmp_path / "logs" / "runtime-tournament.pid").exists()
    assert _launcher(env, "--status").returncode == 1 and _launcher(env, "--stop").returncode == 1
    loop = _launcher(env, "--loop")
    assert loop.returncode != 0 and "internal" in loop.stderr


def test_launcher_stops_when_the_preflight_fails(tmp_path):
    env = _launcher_env(tmp_path, preflight_rc=1)
    result = _launcher(env, "--match")
    assert result.returncode == 1, result.stdout
    assert "fake pre-flight rc=1" in result.stdout and "Pre-flight failed" in result.stdout
    assert "runtime started" not in result.stdout and not (tmp_path / "logs" / "runtime-tournament.pid").exists()
    assert "Pre-flight failed" in _log_text(tmp_path)


def test_launcher_runs_detached_survives_its_terminal_and_stops_on_request(tmp_path):
    env = _launcher_env(tmp_path)
    pid_file = tmp_path / "logs" / "runtime-tournament.pid"
    launcher = subprocess.Popen(["bash", str(SCRIPT), "--match"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, text=True)
    try:
        _wait_for(lambda: pid_file.exists() and "fake runtime started" in _log_text(tmp_path))
        loop_pid = int(pid_file.read_text())
        assert _alive(loop_pid)
        # The Terminal window closes: SIGHUP to the launcher's whole process group. The loop lives in its own session.
        os.killpg(launcher.pid, signal.SIGHUP)
        launcher.wait(timeout=10)
        time.sleep(0.5)
        assert _alive(loop_pid), "the detached loop died with its terminal"
        log = _log_text(tmp_path)
        assert "fake pre-flight rc=0" in log and "runtime started detached" in log
        assert "fake runtime started: -m agent --tournament --match" in log
        status = _launcher(env, "--status")
        assert status.returncode == 0 and f"running (pid {loop_pid}" in status.stdout
        second = _launcher(env, "--match")
        assert second.returncode == 1 and "already running" in second.stdout and "fake pre-flight" not in second.stdout
    finally:
        stop = _launcher(env, "--stop")
    assert stop.returncode == 0 and "stopped" in stop.stdout, stop.stdout
    _wait_for(lambda: not _alive(loop_pid))
    assert not pid_file.exists()
    assert "fake runtime: signal, exiting" in _log_text(tmp_path) and "Stopping." in _log_text(tmp_path)
    assert _launcher(env, "--stop").returncode == 1


# -- F6: the versions we tested are the versions that play ---------------------------------------------------------


def test_pyproject_pins_the_installed_model_stack_and_the_preflight_warns_on_a_mismatch(monkeypatch):
    import importlib.metadata

    import agent.__main__ as agent_main

    pins = agent_main._pinned_versions()
    assert pins == {"anthropic": "1.12.0", "httpx": "0.28.1"}
    assert agent_main._installed_versions() == {name: importlib.metadata.version(name) for name in ("anthropic", "httpx")}

    monkeypatch.setattr(agent_main, "_installed_versions", lambda: dict(pins))
    assert agent_main._versions_line() == ("Installed anthropic 1.12.0, httpx 0.28.1 (as pinned in pyproject.toml)", False)
    monkeypatch.setattr(agent_main, "_installed_versions", lambda: {"anthropic": "1.13.0", "httpx": None})
    line, mismatch = agent_main._versions_line()
    assert mismatch and "anthropic 1.13.0 installed, pyproject pins 1.12.0" in line and "httpx MISSING installed, pyproject pins 0.28.1" in line


def test_preflight_prints_the_version_warning_but_still_passes(env_free_check):
    code, out = env_free_check({"anthropic": "1.13.0", "httpx": "0.28.1"})
    assert code == 0
    assert any(line.startswith("! Installed anthropic 1.13.0, httpx 0.28.1: anthropic 1.13.0 installed, pyproject pins 1.12.0") for line in out.splitlines()), out


@pytest.fixture
def env_free_check(monkeypatch, capsys):
    """Run --check-tournament with a fake control plane, a canned probe, and the given installed versions."""
    import agent.__main__ as agent_main
    from test_main_tournament import KEY, FakeOfficial, _install

    def run(installed: dict):
        for name in ("ALTRUAGENT_CLAIM_TOKEN", "ALTRUAGENT_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ALTRUAGENT_OFFICIAL_AGENT_KEY", KEY)
        monkeypatch.setenv("ALTRUAGENT_CONTROL_URL", "https://control.example.test")
        _install(monkeypatch, FakeOfficial(assignments_result=[]))
        monkeypatch.setattr(agent_main, "_mark", lambda symbol, fallback: symbol)
        monkeypatch.setattr(agent_main, "_probe_model", lambda contestant: ("model probe: claude-test-model 1 ms", False))
        monkeypatch.setattr(agent_main, "_installed_versions", lambda: installed)
        code = agent_main.main(["--check-tournament"])
        return code, capsys.readouterr().out

    return run


# -- public reason: no JSON debris reaches the public reasoning_summary ------------------------------------------------


def test_clean_reason_strips_the_debris_seen_live_and_keeps_whole_sentences():
    from agent.agent import REASONING_CHAR_LIMIT, clean_reason

    # The four shapes from match bcbaabe2 (4 of 19 answers) and three more from the Oct 7 logs.
    assert clean_reason("Body Presses OPPONENT tinglu for a likely KO.\n") == "Body Presses OPPONENT tinglu for a likely KO."
    assert clean_reason("Moonblast for a likely KO while its Protect is unreliable.]") == "Moonblast for a likely KO while its Protect is unreliable."
    assert clean_reason("Archaludon Body Presses OPPONENT rillaboom.)") == "Archaludon Body Presses OPPONENT rillaboom."
    assert clean_reason("Protect is unreliable this turn.ewhere.") == "Protect is unreliable this turn."
    assert clean_reason('Lead Incineroar to Intimidate their physical attackers."}') == "Lead Incineroar to Intimidate their physical attackers."
    assert clean_reason("Moonblast also covers us if it Protects this turn.}") == "Moonblast also covers us if it Protects this turn."
    assert clean_reason("Tailwind lets Great Tusk move first.'] Hmm") == "Tailwind lets Great Tusk move first."
    # Untouched when clean; balanced parentheses and quotes inside stay; a genuine unfinished last sentence stays.
    assert clean_reason("Protect slot 0 (Fake Out is live) and Moonblast OPPONENT gyarados.") == "Protect slot 0 (Fake Out is live) and Moonblast OPPONENT gyarados."
    assert clean_reason('Taunt the "support" first.') == 'Taunt the "support" first.'
    assert clean_reason("Protect slot 0. Moonblast into OPPONENT gyarados for the KO this turn") == "Protect slot 0. Moonblast into OPPONENT gyarados for the KO this turn"
    assert clean_reason("Attack with Moonblast; Protect the other slot") == "Attack with Moonblast; Protect the other slot"
    assert clean_reason("") == "" and clean_reason(None) == ""
    # Over the limit: the last full sentence that fits, never a cut word.
    long = "Protect slot 0 because Kingambit's Sucker Punch is lethal. " * 6
    cleaned = clean_reason(long)
    assert len(cleaned) <= REASONING_CHAR_LIMIT and cleaned.endswith("lethal.") and cleaned.count("lethal.") == 4
    no_sentence = "a" * 10 + " " + "word " * 80
    cut = clean_reason(no_sentence.strip())
    assert len(cut) <= REASONING_CHAR_LIMIT and cut.endswith("…") and not cut.endswith("wor…")


def test_the_public_reason_the_runtime_sends_is_cleaned(tmp_path):
    answer = {"slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": 1}, "win_condition": "w",
              "reasoning_summary": "Fake Out OPPONENT a, Wood Hammer OPPONENT b.]\n"}
    agent = PokemonAgent(_FakeProvider(answer), version="reason", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    decision = agent.choose_action(_doubles(GOOD_SLOTS), CTX)
    assert decision.reasoning_summary == "Fake Out OPPONENT a, Wood Hammer OPPONENT b."
    assert decision.action["slot_0"] == {"type": "move", "move_id": "fakeout", "target": 1}
