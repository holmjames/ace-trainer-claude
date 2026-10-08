"""Draft and Team Preview experiments on the local engine: code-only on both sides, $0, minimal logs.

    python sim/draft_eval.py match v2 v1 --pairs 500 --seed 1          # new drafter vs old drafter
    python sim/draft_eval.py match v2 counter --pairs 500 --seed 2     # new drafter vs a the search bot-style counter-picker
    python sim/draft_eval.py collect random random --games 10000 --out /tmp/x.jsonl   # training data for the draft model

Players (``PLAYERS`` below): ``v1`` is the old greedy scorer with the old Team Preview, ``v2`` the counter-aware
drafter (old Team Preview), ``v2l`` the counter-aware drafter with the new Team Preview, ``v1l`` the old drafter
with the new Team Preview, ``counter`` the counter-picker (agent/versions/counter_picker.py, old Team Preview),
``random`` a seeded random drafter. ``name{"key": value}`` adds tuning overrides, e.g. ``v2{"draft_lookahead": 0}``.
The battle itself is always the code brain (agent/pokemon/battle.py), so only drafting and Team Preview differ.

``match`` plays games in PAIRS: the same 18-card pool and the same battle seed twice, with the two players swapping
seats, so each drafts first exactly once per pool. That cancels most of the pool and first-pick luck. Output: win
rate of the first player with a 95% Wilson interval, overall and by who drafted first.

Decision logs are switched off (the agent's DecisionLog is pointed at /dev/null) and each game appends one short
JSON line to ``--out`` (default: none), so thousands of games cost a few megabytes at most. One engine process; do
not run two of these at once (two bridges on one machine have deadlocked before).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from agent.agent import PokemonAgent  # noqa: E402
from agent.pokemon import tuning  # noqa: E402
from sim import harness  # noqa: E402

NO_LOG = "/dev/null"  # DecisionLog cannot create a directory there, so it stays silent (by design it never raises)

# name -> tuning overrides on top of the defaults (and, for some, a different agent class)
PLAYERS: dict[str, dict] = {
    "v1": {"draft_v2": 0, "lineup_v2": 0},
    "v2": {"draft_v2": 1, "lineup_v2": 0},
    "v2l": {"draft_v2": 1, "lineup_v2": 1},
    "v1l": {"draft_v2": 0, "lineup_v2": 1},
    "counter": {"draft_v2": 0, "lineup_v2": 0},
    "counterl": {"draft_v2": 0, "lineup_v2": 1},
    "random": {"draft_v2": 0, "lineup_v2": 0},
}


def parse_spec(spec: str) -> tuple[str, dict]:
    """'v2{"draft_lookahead": 0}' -> ('v2', {...}). Unknown names are an error."""
    name, brace, rest = spec.partition("{")
    if name not in PLAYERS:
        raise SystemExit(f"unknown player {name!r}; known: {sorted(PLAYERS)}")
    extra = json.loads(brace + rest) if brace else {}
    return name, {**PLAYERS[name], **extra}


class _SeededRandomDraft(PokemonAgent):
    """Random draft picks from a seeded generator (reproducible training data); everything else is the code brain."""

    def __init__(self, seed: int, params: dict) -> None:
        super().__init__(None, version="random-draft", params=params, log=lambda _line: None)
        self._rng = random.Random(seed)

    def _draft(self, state, obs, log):
        pick = self._rng.choice(state.legal_actions)
        from agent.pokemon import draft as draft_rules

        self.memory.record_pick(draft_rules.card_id_of(pick))
        return pick, "Random draft pick."


def make_player(spec: str, seed: int):
    name, overrides = parse_spec(spec)
    params = tuning.merged(overrides)
    if name == "random":
        agent = _SeededRandomDraft(seed, params)
    elif name.startswith("counter"):
        from agent.versions.counter_picker import CounterPickerAgent

        agent = CounterPickerAgent(params=params)
        agent._print = lambda _line: None
    else:
        agent = PokemonAgent(None, version=spec, params=params, log=lambda _line: None)
    agent._log_dir = NO_LOG
    return agent


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Win rate and its 95% Wilson interval (better than the normal approximation near 0 or 1)."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def play(bridge, game_id: str, seed: int, specs: dict[str, str], names: dict[str, str]) -> dict:
    rng = random.Random(seed)
    players = {side: make_player(spec, seed * 2 + (side == "p2")) for side, spec in specs.items()}
    t0 = time.monotonic()
    result = harness.play_game(bridge, game_id, players, names, rng, verbose=False)
    result["seconds"] = round(time.monotonic() - t0, 3)
    return result


def record(out, result: dict, seed: int, specs: dict[str, str]) -> None:
    if out is None:
        return
    out.write(json.dumps({"seed": seed, "specs": specs, "first": result["first_drafter"], "winner": result["winner_side"],
                          "turns": result["turns"], "rosters": result["rosters"], "teams": result["teams"]}) + "\n")
    out.flush()


def cmd_match(args) -> int:
    a, b = args.a, args.b
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    bridge = harness.Bridge()
    wins = games = 0
    by_first = {"a_first": [0, 0], "b_first": [0, 0]}
    errors = 0
    started = time.monotonic()
    try:
        for k in range(args.pairs):
            seed = args.seed * 1_000_003 + k
            for swap in (False, True):
                specs = {"p1": b, "p2": a} if swap else {"p1": a, "p2": b}
                names = {"p1": "B" if swap else "A", "p2": "A" if swap else "B"}
                try:
                    result = play(bridge, f"eval-{seed}-{int(swap)}", seed, specs, names)
                except Exception as exc:  # noqa: BLE001 — one broken game must not end a long run
                    errors += 1
                    print(f"   pair {k} swap {swap}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    continue
                record(out, result, seed, specs)
                if result["winner"] is None:
                    continue  # unfinished (turn cap): counts for neither
                games += 1
                a_won = result["winner"] == "A"
                wins += a_won
                a_side = "p2" if swap else "p1"
                bucket = by_first["a_first" if result["first_drafter"] == a_side else "b_first"]
                bucket[0] += a_won
                bucket[1] += 1
            if args.progress and (k + 1) % args.progress == 0:
                p, lo, hi = wilson(wins, games)
                print(f"  {k + 1} pairs: {a} {100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] over {games} games "
                      f"({int(time.monotonic() - started)} s)", flush=True)
    finally:
        bridge.close()
        if out:
            out.close()
    p, lo, hi = wilson(wins, games)
    print(f"{a} vs {b}: {wins}/{games} = {100 * p:.1f}%  95% CI [{100 * lo:.1f}, {100 * hi:.1f}]  "
          f"({errors} errors, {int(time.monotonic() - started)} s)")
    for key, (w, n) in by_first.items():
        q, l, h = wilson(w, n)
        print(f"   {key:8s} {w}/{n} = {100 * q:.1f}% [{100 * l:.1f}, {100 * h:.1f}]")
    return 0


def cmd_collect(args) -> int:
    out = open(args.out, "a", encoding="utf-8")
    bridge = harness.Bridge()
    started = time.monotonic()
    errors = 0
    try:
        for g in range(args.games):
            seed = args.seed * 1_000_003 + g
            specs = {"p1": args.a, "p2": args.b}
            try:
                result = play(bridge, f"collect-{seed}", seed, specs, {"p1": "p1", "p2": "p2"})
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"   game {g}: ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            record(out, result, seed, specs)
            if args.progress and (g + 1) % args.progress == 0:
                print(f"  {g + 1} games ({int(time.monotonic() - started)} s, {errors} errors)", flush=True)
    finally:
        bridge.close()
        out.close()
    print(f"collected {args.games - errors} games in {int(time.monotonic() - started)} s ({errors} errors)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match", help="paired head-to-head between two players")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--pairs", type=int, default=250, help="pools to play; each pool is played twice with seats swapped")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--out", default=None, help="append one JSON line per game here (optional)")
    m.add_argument("--progress", type=int, default=50)
    c = sub.add_parser("collect", help="training data: rosters and results, one JSON line per game")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--games", type=int, default=1000)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    c.add_argument("--progress", type=int, default=500)
    args = parser.parse_args(argv)
    return cmd_match(args) if args.cmd == "match" else cmd_collect(args)


if __name__ == "__main__":
    sys.exit(main())
