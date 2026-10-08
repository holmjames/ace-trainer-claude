"""The human seat accepts typed choices, rejects bad ones, and plays a legal default on blank input."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("sim.translate")
from sim import harness  # noqa: E402
from sim.human import HumanPlayer  # noqa: E402


def _scripted(*lines):
    queue = list(lines)
    out: list[str] = []

    def read(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    return read, out


def test_human_seat_plays_a_whole_game_against_the_code_brain_on_blank_input():
    bridge = harness.Bridge()
    try:
        read, out = _scripted()  # every prompt hits EOF: first legal choice everywhere
        human = HumanPlayer(read=read, write=out.append)
        code, _ = harness.make_player("code", random.Random(1))
        result = harness.play_game(bridge, "human-test-1", {"p1": human, "p2": code}, {"p1": "human", "p2": "code"}, random.Random(1), verbose=False)
    finally:
        bridge.close()
    assert result["winner_side"] in ("p1", "p2", None)
    assert result["rejected"]["p1"] == 0
    text = "\n".join(out)
    # who drafts first depends on the seed and the pool; the human seat must see its picks, the preview and the battle
    assert "=== DRAFT pick " in text and "=== TEAM PREVIEW" in text and "=== TURN 1 ===" in text


def test_human_seat_validates_input_and_shows_a_hint():
    bridge = harness.Bridge()
    try:
        # 6 draft picks ("2" each), a lineup, then on turn 1: a bad option, a hint, then a real choice, then EOF.
        read, out = _scripted("2", "2", "2", "2", "2", "2", "9 9 9 9", "3 1 2 4", "99", "h", "0 1; 0 1")
        human = HumanPlayer(read=read, write=out.append)
        code, _ = harness.make_player("code", random.Random(2))
        result = harness.play_game(bridge, "human-test-2", {"p1": human, "p2": code}, {"p1": "human", "p2": "code"}, random.Random(2), verbose=False)
    finally:
        bridge.close()
    text = "\n".join(out)
    assert "need four different numbers" in text
    assert "--- code brain ---" in text and "speed:" in text
    assert result["rejected"]["p1"] == 0
