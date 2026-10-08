"""Every lineup our drafted six allowed, against the lineup the opponent actually brought: was the match winnable at
Team Preview? (Same agent on both sides; only our bring/leads vary.)

    python scripts/lineup_sweep.py 7ce71025 --games-per 24
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agent.pokemon import data  # noqa: E402
from draft_eval import LIVE_LINEUPS, ForcedLineup, teams  # noqa: E402
from sim import harness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session")
    ap.add_argument("--games-per", type=int, default=24)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)
    mine, theirs = teams(args.session)
    ours_ids = [data.to_id(c["species"]) for c in mine]
    live_ours, their_options = LIVE_LINEUPS[args.session]
    rng = random.Random(args.seed)
    bridge = harness.Bridge()
    results = []
    try:
        for bring in itertools.combinations(ours_ids, 4):
            for leads in itertools.combinations(bring, 2):
                wins = 0
                for g in range(args.games_per):
                    ours_side = "p1" if g % 2 == 0 else "p2"
                    other = "p2" if ours_side == "p1" else "p1"
                    p_ours, _ = harness.make_player("code", rng)
                    p_theirs, _ = harness.make_player("code", rng)
                    theirs_lu = their_options[g % len(their_options)]
                    players = {ours_side: ForcedLineup(p_ours, list(bring), list(leads)), other: ForcedLineup(p_theirs, *theirs_lu)}
                    result = harness.play_game(bridge, f"lineup-sweep-{args.session}", players, {ours_side: "ours", other: "theirs"},
                                               rng, verbose=False, fixed_rosters={ours_side: mine, other: theirs})
                    wins += result["winner_side"] == ours_side
                results.append((wins / args.games_per, list(bring), list(leads)))
    finally:
        bridge.close()
    results.sort(key=lambda r: -r[0])
    live = next((r for r in results if sorted(r[1]) == sorted(live_ours[0]) and sorted(r[2]) == sorted(live_ours[1])), None)
    n = args.games_per
    print(f"{args.session}: {len(results)} lineups x {n} games vs their live lineup {their_options[0][0]} (leads {their_options[0][1]})")
    for p, bring, leads in results[:6]:
        print(f"   {100 * p:5.1f}% (+-{196 * math.sqrt(max(p * (1 - p), 0.01) / n):.0f})  bring {bring}  lead {leads}")
    if live:
        print(f"   live lineup: {100 * live[0]:.1f}%  bring {live[1]}  lead {live[2]}  (rank {results.index(live) + 1} of {len(results)})")
    print(f"   median lineup: {100 * results[len(results) // 2][0]:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
