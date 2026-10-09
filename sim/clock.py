"""The live clock, for the simulator: a per-decision allowance (85 s at Team Preview, 50 s in battle) and a 420 s bank per
side. Every decision's measured wall time comes off the bank; a decision past its allowance is replaced by a random
legal move (what the live server does), and an empty bank forfeits. sim/harness.py turns it on with ``--clock`` and
hands each player the clock it would see live (``decision_seconds_left`` = the allowance or what is left of the bank,
whichever is smaller; ``bank_seconds_left``). The time source is injectable so tests can play a slow player without
waiting.

The draft has no clock live (the observation carries none), so draft picks are not timed here."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

ALLOWANCE = {"team_preview": 85.0, "moving": 50.0}
BANK = 420.0


@dataclass
class SideClock:
    bank: float = BANK
    used: float = 0.0
    timeouts: int = 0
    forfeited: bool = False
    events: list[dict] = field(default_factory=list)


class Clock:
    def __init__(self, *, bank: float = BANK, allowance: dict[str, float] | None = None, now: Callable[[], float] = time.monotonic) -> None:
        self.sides = {"p1": SideClock(bank), "p2": SideClock(bank)}
        self.allowance = {**ALLOWANCE, **(allowance or {})}
        self.now = now

    def state(self, side: str, phase: str) -> dict:
        """The clock the player sees before deciding (the observation's ``clock``)."""
        s = self.sides[side]
        return {"decision_seconds_left": round(max(min(self.allowance.get(phase, ALLOWANCE["moving"]), s.bank), 0.0), 1),
                "bank_seconds_left": round(max(s.bank, 0.0), 1)}

    def settle(self, side: str, phase: str, seconds: float, *, turn: int | None = None) -> str:
        """Charge a finished decision: 'ok', 'timeout' (past the allowance: a random move is played) or 'forfeit' (bank empty)."""
        s = self.sides[side]
        s.used += seconds
        s.bank -= seconds
        verdict = "ok"
        if seconds > self.allowance.get(phase, ALLOWANCE["moving"]):
            verdict = "timeout"
            s.timeouts += 1
        if s.bank <= 0:
            verdict = "forfeit"
            s.forfeited = True
        if verdict != "ok":
            s.events.append({"kind": verdict, "phase": phase, "turn": turn, "seconds": round(seconds, 2), "bank_left": round(max(s.bank, 0.0), 1)})
        return verdict

    def summary(self) -> dict:
        return {side: {"bank_left": round(max(s.bank, 0.0), 1), "used": round(s.used, 2), "timeouts": s.timeouts,
                       "forfeited": s.forfeited, "events": list(s.events)} for side, s in self.sides.items()}


__all__ = ["Clock", "SideClock", "ALLOWANCE", "BANK"]
