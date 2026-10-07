"""Parameter sweep: play tuned variants of the code brain against the baseline and report win rates.

    python sim/sweep.py --games 300 '{"protect_bonus_guaranteed": 70}' '{"draft_offense_w": 14}'
    python sim/sweep.py --games 300 --preset knobs      # one-at-a-time nudges of every parameter

Each variant plays ``--games`` games (seats alternate) against the untuned code brain in one shared
engine process. Output: win rate with a 95% interval, so you can tell signal from noise (at 300 games
the interval is about ±5.5 points). Free and fast: ~20 games per second.
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


def variant_player(overrides: dict):
    params = tuning.merged(overrides)
    agent = PokemonAgent(None, version=tuning.label(params), params=params)
    agent._log_dir = str(ROOT / "logs" / "sim" / "sweep")
    return agent


def baseline_player():
    agent = PokemonAgent(None, version="baseline")
    agent._log_dir = str(ROOT / "logs" / "sim" / "sweep-baseline")
    return agent


def run_variant(bridge, overrides: dict, games: int, rng: random.Random) -> tuple[int, int]:
    wins = decided = 0
    for g in range(games):
        if g % 2 == 0:
            players = {"p1": variant_player(overrides), "p2": baseline_player()}
            names = {"p1": "variant", "p2": "baseline"}
        else:
            players = {"p1": baseline_player(), "p2": variant_player(overrides)}
            names = {"p1": "baseline", "p2": "variant"}
        try:
            result = harness.play_game(bridge, f"sweep-{int(time.time())}-{g}", players, names, rng, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"   game {g}: ERROR {type(exc).__name__}: {exc}")
            continue
        if result["winner"]:
            decided += 1
            wins += result["winner"] == "variant"
    return wins, decided


def knob_presets() -> list[dict]:
    out = []
    for key, value in tuning.DEFAULTS.items():
        if value == 0:
            continue
        out.append({key: round(value * 0.5, 4)})
        out.append({key: round(value * 1.5, 4)})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help="JSON objects of parameter overrides")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--preset", choices=["knobs"], help="generate variants automatically")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    variants = [json.loads(v) for v in args.variants]
    if args.preset == "knobs":
        variants += knob_presets()
    if not variants:
        print("nothing to sweep")
        return 2
    bridge = harness.Bridge()
    rng = random.Random(args.seed)
    rows = []
    try:
        for overrides in variants:
            started = time.monotonic()
            wins, decided = run_variant(bridge, overrides, args.games, rng)
            rate = wins / decided if decided else 0.0
            ci = 1.96 * math.sqrt(rate * (1 - rate) / decided) if decided else 0.0
            rows.append((rate, ci, decided, overrides))
            flag = "  <-- better" if rate - ci > 0.5 else ("  <-- worse" if rate + ci < 0.5 else "")
            print(f"{tuning.label(tuning.merged(overrides)):48s} {100 * rate:5.1f}% ±{100 * ci:4.1f}  (n={decided}, {int(time.monotonic() - started)} s){flag}", flush=True)
    finally:
        bridge.close()
    print("\nsorted:")
    for rate, ci, decided, overrides in sorted(rows, key=lambda r: -r[0]):
        print(f"  {100 * rate:5.1f}% ±{100 * ci:4.1f}  {json.dumps(overrides)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
