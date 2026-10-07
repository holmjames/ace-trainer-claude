"""Self-play arena: one runtime, two different agent versions, chosen by seat.

Plain language: to compare version A against version B we make them play
each other. The runtime calls ``create_agent()`` once per match with no
arguments, so this factory returns a small stand-in that waits for the first
decision, reads which seat it is sitting in (``context.seat_position``), and
only then builds the real agent for that seat:

    AGENT_SEAT0=agent.agent            # champion plays seat 0
    AGENT_SEAT1=agent.versions.v2      # challenger plays seat 1
    python -m agent --claim <token for seat 0> --agent agent.arena
    python -m agent --claim <token for seat 1> --agent agent.arena

(With one process per seat today, you could also pass ``--agent`` directly;
the arena matters once ``--match`` plays every seat from a single runtime.)

Each decision log line carries the version label, so ``scripts/tally.py`` can
report per-version win rates, latency and fallbacks. Swap the seats between
matches so neither version always drafts first.
"""

from __future__ import annotations

import os
import sys

from altruagent import DecisionContext, GameState
from altruagent.agent_loader import load_agent_factory

DEFAULT_SPEC = "agent.agent"


def spec_for_seat(seat: int | None) -> str:
    if seat is None:
        return os.environ.get("AGENT_SEAT0") or DEFAULT_SPEC
    return os.environ.get(f"AGENT_SEAT{seat}") or os.environ.get("AGENT_SEAT0") or DEFAULT_SPEC


class ArenaAgent:
    def __init__(self) -> None:
        self._inner = None
        self.spec: str | None = None

    def _resolve(self, context: DecisionContext):
        if self._inner is None:
            self.spec = spec_for_seat(context.seat_position)
            factory = load_agent_factory(self.spec)
            self._inner = factory()
            # Label this match's decision log with the version that played it.
            if hasattr(self._inner, "_version"):
                self._inner._version = self.spec
            print(f"[arena] seat {context.seat_position} -> {self.spec}", file=sys.stderr, flush=True)
        return self._inner

    def choose_action(self, state: GameState, context: DecisionContext):
        inner = self._resolve(context)
        choose = getattr(inner, "choose_action", inner)
        return choose(state, context)

    def choose_message(self, state: GameState, context: DecisionContext):
        inner = self._resolve(context)
        method = getattr(inner, "choose_message", None)
        if method is None:
            from altruagent import TERMINATE_MESSAGING

            return TERMINATE_MESSAGING
        return method(state, context)


def create_agent() -> ArenaAgent:
    return ArenaAgent()
