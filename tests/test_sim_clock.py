"""sim/clock.py and the --clock mode of sim/harness.py / sim/draft_eval.py: a slow player (one that advances an injected
time source instead of sleeping) loses a late Team Preview to a random lineup and forfeits when the bank runs dry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent.agent import PokemonAgent
from sim import draft_eval, harness
from sim.clock import ALLOWANCE, BANK, Clock

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "sim" / "node_modules" / "pokemon-showdown"
needs_engine = pytest.mark.skipif(not ENGINE.exists() or shutil.which("node") is None, reason="needs Node and npm install in sim/")


class FakeTime:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_the_clock_charges_decisions_and_calls_timeouts_and_forfeits():
    clock = Clock()
    assert clock.state("p1", "team_preview") == {"decision_seconds_left": 85.0, "bank_seconds_left": BANK}
    assert clock.state("p1", "moving") == {"decision_seconds_left": 50.0, "bank_seconds_left": BANK}
    assert clock.settle("p1", "team_preview", 10.0, turn=0) == "ok"
    assert clock.state("p1", "moving")["bank_seconds_left"] == 410.0 and clock.state("p2", "moving")["bank_seconds_left"] == BANK
    assert clock.settle("p1", "moving", 50.5, turn=1) == "timeout"
    assert clock.sides["p1"].timeouts == 1 and clock.sides["p1"].events[-1]["kind"] == "timeout"
    assert clock.settle("p1", "moving", 300.0, turn=2) == "timeout"  # bank 59.5 left: late but not out
    assert clock.state("p1", "moving") == {"decision_seconds_left": 50.0, "bank_seconds_left": 59.5}
    assert clock.settle("p1", "moving", 49.0, turn=3) == "ok"  # 10.5 s of bank left
    assert clock.state("p1", "moving") == {"decision_seconds_left": 10.5, "bank_seconds_left": 10.5}  # the allowance shrinks to the bank
    assert clock.settle("p1", "moving", 11.0, turn=4) == "forfeit"
    summary = clock.summary()
    assert summary["p1"]["forfeited"] and summary["p1"]["bank_left"] == 0.0 and summary["p1"]["timeouts"] == 2
    assert [e["kind"] for e in summary["p1"]["events"]] == ["timeout", "timeout", "forfeit"] and summary["p1"]["events"][-1]["turn"] == 4
    assert not summary["p2"]["events"] and summary["p2"]["bank_left"] == BANK
    assert Clock(bank=100, allowance={"moving": 5}).allowance == {**ALLOWANCE, "moving": 5}


class SlowPlayer:
    """A code-only agent whose decisions appear to take ``cost[phase]`` seconds (the harness reads ``clock.now``)."""

    def __init__(self, inner, fake: FakeTime, cost: dict[str, float]) -> None:
        self.inner, self.fake, self.cost = inner, fake, cost
        self.decisions: list[str] = []

    def choose_action(self, state, context):
        phase = state.phase or ""
        self.fake.t += self.cost.get(phase, 0.0)
        self.decisions.append(phase)
        return self.inner.choose_action(state, context)


@needs_engine
def test_a_slow_player_loses_the_preview_to_a_random_lineup_and_then_forfeits(tmp_path, monkeypatch):
    fake = FakeTime()
    slow_inner = PokemonAgent(None, version="slow", log=lambda _l: None)
    slow_inner._log_dir = draft_eval.NO_LOG
    slow = SlowPlayer(slow_inner, fake, {"team_preview": 90.0, "moving": 40.0})  # 90 > 85: timeout; 90 + 40 > 120: forfeit at turn 1
    original = draft_eval.make_player
    monkeypatch.setattr(draft_eval, "make_player", lambda spec, seed: slow if spec == "slow" else original(spec, seed))
    monkeypatch.setattr(draft_eval, "Clock", lambda: Clock(bank=120.0, now=fake))
    bridge = harness.Bridge()
    try:
        result = draft_eval.play(bridge, "eval-clock-0", 11, {"p1": "slow", "p2": "v2l"}, {"p1": "A", "p2": "B"}, clock=True)
    finally:
        bridge.close()
    assert result["forfeit"] == "p1" and result["winner_side"] == "p2" and result["winner"] == "B"
    p1 = result["clock"]["p1"]
    assert [e["kind"] for e in p1["events"]] == ["timeout", "forfeit"] and p1["events"][0]["phase"] == "team_preview"
    assert p1["events"][1] == {"kind": "forfeit", "phase": "moving", "turn": 1, "seconds": 40.0, "bank_left": 0.0}
    assert p1["forfeited"] and p1["timeouts"] == 1 and p1["bank_left"] == 0.0 and p1["used"] == 130.0
    assert result["clock"]["p2"]["events"] == [] and result["clock"]["p2"]["bank_left"] <= 120.0
    assert slow.decisions.count("team_preview") == 1 and slow.decisions.count("moving") == 1
    assert len(result["teams"]["p1"]) == 4 and len(result["teams"]["p2"]) == 4  # the random lineup still brought four
    assert result["decision_seconds"]["p1"] and result["turns"] == 1

    class _Out:
        text = ""

        def write(self, s):
            self.text += s

        def flush(self):
            pass

    out = _Out()
    draft_eval.record(out, result, 11, {"p1": "slow", "p2": "v2l"})
    line = json.loads(out.text)
    assert line["forfeit"] == "p1" and line["clock"]["p1"]["timeouts"] == 1 and line["decision_seconds"]["p2"]


def test_without_the_flag_nothing_is_timed(monkeypatch):
    seen = {}

    def fake_play_game(bridge, game_id, players, names, rng, *, verbose, fixed_rosters=None, clock=None):
        seen["clock"] = clock
        return {"game": game_id, "turns": 0, "winner_side": None, "winner": None, "first_drafter": "p1", "rejected": {}, "versions": {},
                "teams": {"p1": [], "p2": []}, "rosters": {"p1": [], "p2": []}, "picks": [], "forfeit": None, "clock": None}

    monkeypatch.setattr(harness, "play_game", fake_play_game)
    result = draft_eval.play(None, "eval-x", 3, {"p1": "v2l", "p2": "v1"}, {"p1": "A", "p2": "B"})
    assert seen["clock"] is None and result["clock"] is None and result["decision_seconds"] == {"p1": [], "p2": []}
    draft_eval.play(None, "eval-y", 3, {"p1": "v2l", "p2": "v1"}, {"p1": "A", "p2": "B"}, clock=True)
    assert isinstance(seen["clock"], Clock)
