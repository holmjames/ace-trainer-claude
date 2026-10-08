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


@pytest.mark.skipif(not CASES, reason="no live captures")
@pytest.mark.parametrize("folder, seat", CASES, ids=[f"{f.name[:8]}/seat{s}" for f, s in CASES])
def test_a_fresh_agent_sees_the_same_sheet_as_a_full_replay(folder, seat, tmp_path):
    """Every battle state of every live session: an agent restarted on that state builds the same turn sheet, roster
    and known sets as the agent that lived through the draft and Team Preview."""
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
                                for m in [[p for p in _sequence(f, s) if p.name.startswith("moving")]] if m)
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
    """Report (not fail on) species the catalog cannot rebuild: scripts/pool_from_live.py refreshes data/cards.json."""
    missing: dict[str, set[str]] = {}
    for folder, seat in CASES:
        for path in _sequence(folder, seat):
            if not path.name.startswith("moving"):
                continue
            obs = observation_dict(GameState.from_mcp_state(json.loads(path.read_text())))
            for line in obs.get("protocol_log") or []:
                parts = str(line).split("|")
                if len(parts) > 3 and parts[1] == "poke":
                    name = parts[3].split(",")[0].strip()
                    if name and not data.catalog_card(species=name):
                        missing.setdefault(name, set()).add(folder.name[:8])
            break
    known_gaps = {"Iron Valiant"}  # the drill session (4e6fa669) drew it; it was never offered to us in a captured draft
    assert set(missing) <= known_gaps, f"new catalog gaps: {missing}"
