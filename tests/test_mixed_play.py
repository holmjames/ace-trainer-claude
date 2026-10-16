"""mix_play: with probability mix_play the final play is swapped for a row drawn among the rows within mix_margin of
the best (weights by score, the chosen row included), never a row the gate refuses, never on a forced switch; the
draw is a random.Random seeded by sha256 of (session id, turn, our seat). Off (the default): the play is untouched."""
import json
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon import battle, tuning
from agent.pokemon.memory import observation_dict
from test_live_semantics import LIVE, _sequence
from test_reliability_f import _context

MOVES = {0: "closecombat", 1: "moonblast", 2: "protect", 3: "drainpunch", 4: "dazzlinggleam"}


def _payload(answers: dict) -> dict:
    return {f"slot_{n}": {"type": "move", "move_id": MOVES[answers[f"slot_{n}"]["option"]], "target": answers[f"slot_{n}"]["target"]}
            for n in (0, 1)}


BASE = SimpleNamespace(build=_payload)


def _sheet(scores: list[float], **kwargs) -> battle.TurnSheet:
    """Rows with these scores: row i is (option i % 5 on slot 0, option 1 on slot 1), so every row is a distinct play."""
    sheet = battle.TurnSheet(**kwargs)
    sheet.candidates = [{"name": f"r{i}", "slot_0": {"option": i % 5, "target": 1 + i // 5}, "slot_1": {"option": 1, "target": 2},
                         "score": s, "why": f"row {i}"} for i, s in enumerate(scores)]
    return sheet


def _agent(**params) -> PokemonAgent:
    return PokemonAgent(None, version="t", log_dir=Path(tempfile.mkdtemp()), capture_dir="", log=lambda *_: None,
                        params={**tuning.DEFAULTS, **params})


def _mix(agent: PokemonAgent, sheet, *, chosen: int = 0, session: str = "s", turn: int = 3, seat: int = 0, forced: bool = False):
    agent._mix_seed_key = (session, seat)
    meta: dict = {}
    value = BASE.build(sheet.candidates[chosen])
    return agent._mix_play(value, sheet, BASE, {"turn": turn}, meta, forced=forced), meta


def test_defaults_are_off():
    assert tuning.DEFAULTS["mix_play"] == 0.0
    assert tuning.DEFAULTS["mix_margin"] == 0.08


def test_mixing_swaps_only_within_the_margin():
    agent = _agent(mix_play=1.0, mix_margin=0.08)
    sheet = _sheet([100.0, 95.0, 92.5, 91.0, 50.0])  # 8% of 100: rows 0-2
    drawn = Counter()
    for k in range(300):
        new, meta = _mix(agent, sheet, session=f"game-{k}")
        assert meta["mix"]["pool"] == [0, 1, 2]
        drawn[0 if new is None else new] += 1
        if new is not None:
            assert (meta["mixed_from"], meta["mixed_to"]) == (0, new)
    assert set(drawn) == {0, 1, 2}
    # weights by score: row 0 (100) is drawn a little more often than row 2 (92.5), each about a third of the time
    assert all(70 <= drawn[i] <= 130 for i in (0, 1, 2))


def test_mixing_is_reproducible_for_the_same_seed():
    agent = _agent(mix_play=0.5, mix_margin=0.08)
    sheet = _sheet([100.0, 96.0, 94.0])
    runs = [[_mix(agent, sheet, session=f"game-{k}", turn=t)[0] for k in range(20) for t in range(1, 6)] for _ in range(2)]
    assert runs[0] == runs[1]
    assert any(x is not None for x in runs[0]) and any(x is None for x in runs[0])  # about half the turns swap at 0.5
    other_seat = [_mix(agent, sheet, session=f"game-{k}", turn=t, seat=1)[0] for k in range(20) for t in range(1, 6)]
    assert other_seat != runs[0]


def test_mix_play_zero_never_draws():
    sheet = _sheet([100.0, 99.0, 98.0])
    for p in (0.0,):
        agent = _agent(mix_play=p, mix_margin=0.5)
        assert _mix(agent, sheet) == (None, {})


def test_the_chosen_row_stays_eligible_and_the_margin_is_capped():
    # the model's row 3 (passed by the gate's 15%) stays in the pool beside rows within 8% of row 0
    agent = _agent(mix_play=1.0, mix_margin=0.08)
    new, meta = _mix(agent, _sheet([100.0, 93.0, 90.0, 86.0]), chosen=3)
    assert meta["mix"]["pool"] == [0, 1, 3]
    # a margin wider than override_margin (0.15) is capped there: never a row E2 would send back
    agent = _agent(mix_play=1.0, mix_margin=0.5)
    assert _mix(agent, _sheet([100.0, 90.0, 86.0, 60.0]))[1]["mix"]["pool"] == [0, 1, 2]


def test_refused_rows_are_never_drawn():
    agent = _agent(mix_play=1.0, mix_margin=0.08)
    # row 2 Protects with slot 0, which holds an unlocked Choice item: refused by the gate, never drawn
    sheet = _sheet([100.0, 97.0, 99.0], choice_protect={0})
    assert all(_mix(agent, sheet, session=f"g{k}")[1]["mix"]["pool"] == [0, 1] for k in range(5))
    # our last Pokémon alone with nothing to wait for: its Protect row is refused
    alone = _sheet([100.0, 97.0, 99.0], alone=True)
    assert _mix(agent, alone)[1]["mix"]["pool"] == [0, 1]


def test_no_mixing_on_a_forced_switch_an_override_or_a_single_row():
    agent = _agent(mix_play=1.0, mix_margin=0.08)
    sheet = _sheet([100.0, 99.0])
    assert _mix(agent, sheet, forced=True) == (None, {})
    assert _mix(agent, _sheet([100.0])) == (None, {})
    forced_row = _sheet([1000.0, 999.0])
    forced_row.candidates[1]["name"] = "forced_switch"
    assert _mix(agent, forced_row)[1]["mix"]["pool"] == [0]
    meta: dict = {}
    override = {"slot_0": {"type": "move", "move_id": "drainpunch", "target": 2}, "slot_1": {"type": "move", "move_id": "moonblast", "target": 1}}
    assert agent._mix_play(override, sheet, BASE, {"turn": 1}, meta) is None
    assert meta == {"mix": {"skipped": "the play is not a listed row"}}


def _replay(prefix: str, seat: int, turn: int, **params):
    """Play a live match code-only up to the battle decision at ``turn`` and decide it: (agent, decision, log rows)."""
    folder = next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))
    log_dir = Path(tempfile.mkdtemp())
    agent = PokemonAgent(None, version="t", log_dir=log_dir, capture_dir="", log=lambda *_: None, params={**tuning.DEFAULTS, **params})
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        decision = agent.choose_action(state, _context(folder, raw, seat))
        if state.phase == "moving" and observation_dict(state).get("turn") == turn:
            rows = [json.loads(line) for f in log_dir.glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
            return agent, decision, [r for r in rows if r.get("kind") == "turn"]
    raise AssertionError(f"turn {turn} not in {prefix}")


@pytest.fixture(scope="module")
def live_default():
    return _replay("3666fb1d", 0, 8)


def test_live_off_plays_exactly_as_now(live_default):
    _, decision, turns = live_default
    _, same, same_turns = _replay("3666fb1d", 0, 8, mix_play=0.0, mix_margin=0.5)
    assert [t["payload"] for t in same_turns] == [t["payload"] for t in turns]
    assert same.action == decision.action and same.reasoning_summary == decision.reasoning_summary
    assert not any("mix" in (t.get("resolution") or {}) or "mixed_from" in t for t in same_turns)


def test_live_on_plays_a_row_within_the_margin(live_default):
    _, _, base_turns = live_default
    _, _, turns = _replay("3666fb1d", 0, 8, mix_play=1.0, mix_margin=0.08)
    _, _, again = _replay("3666fb1d", 0, 8, mix_play=1.0, mix_margin=0.08)
    assert [t["payload"] for t in turns] == [t["payload"] for t in again]  # the same game, the same draws
    assert len(turns) == len(base_turns)
    pools = [t["resolution"]["mix"] for t in turns if "pool" in ((t.get("resolution") or {}).get("mix") or {})]
    assert pools and all(m["chosen"] in m["pool"] for m in pools)
    for t in turns:
        if "mixed_from" in t:
            mix = t["resolution"]["mix"]
            assert t["mixed_to"] in mix["pool"] and t["mixed_from"] == mix["chosen"] != t["mixed_to"]
            assert t["resolution"]["summary_from_row"] == t["mixed_to"]
