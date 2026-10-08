"""How winnable did the draft leave a real match? Replay its two six-card teams many times with the SAME agent on both
sides, so only the teams differ.

    python scripts/draft_eval.py                      # the four Oct 7 losses, 200 games each, code-only both sides
    python scripts/draft_eval.py --games 400 d9405f02

Teams come from the captured match (tests/fixtures/live/<session>/ + logs/<session>.jsonl): the live draft's picks,
completed from the decision log and the Team Preview roster; sets come from data/cards.json. Each team plays half its
games as Showdown's p1 and half as p2. Prints our drafted team's win rate with a 95% interval.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

from agent.pokemon import data  # noqa: E402
from agent.pokemon.memory import species_key  # noqa: E402
from sim import harness  # noqa: E402

LIVE = ROOT / "tests" / "fixtures" / "live"
LOSSES = ["d9405f02", "1aef9227", "7ce71025", "0a2b7c2a"]


def teams(prefix: str) -> tuple[list[dict], list[dict]]:
    """(our six cards, their six cards) as drafted live."""
    folder = next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))
    preview = json.loads(next(folder.glob("team_preview-*-seat0.json")).read_text())["observation"]

    def six(entries: list[dict]) -> list[dict]:
        out = []
        for e in entries:
            card = data.catalog_card(species=e.get("species") or e.get("name"))
            if card is None:
                raise SystemExit(f"{prefix}: {e.get('species')} is not in data/cards.json")
            out.append(card)
        return out

    return six(preview["your_roster"]), six(preview["opponent_roster"])


# The lineups actually played live (bring, leads). Their 4th in match 1 was never revealed: one of the three is used per game.
LIVE_LINEUPS = {
    "d9405f02": ((["sneasler", "cresselia", "landorustherian", "garchomp"], ["sneasler", "cresselia"]),
                 [(["dragonite", "gholdengo", "ironhands", x], ["dragonite", "gholdengo"]) for x in ("archaludon", "ironbundle", "indeedeef")]),
    "1aef9227": ((["ironhands", "whimsicott", "archaludon", "gyarados"], ["ironhands", "whimsicott"]),
                 [(["incineroar", "sylveon", "fluttermane", "kingambit"], ["incineroar", "sylveon"])]),
    "7ce71025": ((["greattusk", "tornadus", "hatterene", "indeedeef"], ["greattusk", "tornadus"]),
                 [(["ironhands", "sylveon", "whimsicott", "incineroar"], ["ironhands", "sylveon"])]),
    "0a2b7c2a": ((["landorustherian", "tornadus", "baxcalibur", "lucario"], ["landorustherian", "tornadus"]),
                 [(["dragonite", "gholdengo", "incineroar", "archaludon"], ["dragonite", "gholdengo"])]),
}


class ForcedLineup:
    """Plays an agent but submits a fixed Team Preview lineup (the agent still observes the preview, so its memory of
    both rosters is complete)."""

    def __init__(self, inner, bring: list[str], leads: list[str]) -> None:
        self.inner, self.bring, self.leads = inner, bring, leads

    def choose_action(self, state, context):
        decision = self.inner.choose_action(state, context)
        if state.phase == "team_preview":
            if hasattr(self.inner, "memory"):
                self.inner.memory.my_lineup = list(self.bring)
            return {"type": "select_lineup", "bring": list(self.bring), "leads": list(self.leads)}
        return decision


def evaluate(prefix: str, games: int, ours_spec: str, theirs_spec: str, seed: int, live_lineups: bool = False) -> tuple[int, int]:
    mine, theirs = teams(prefix)
    rng = random.Random(seed)
    bridge = harness.Bridge()
    wins = played = 0
    try:
        for g in range(games):
            ours_side = "p1" if g % 2 == 0 else "p2"
            other = "p2" if ours_side == "p1" else "p1"
            players, names = {}, {}
            players[ours_side], names[ours_side] = harness.make_player(ours_spec, rng)
            players[other], names[other] = harness.make_player(theirs_spec, rng)
            if live_lineups:
                ours_lu, their_options = LIVE_LINEUPS[prefix]
                theirs_lu = their_options[g % len(their_options)]
                players[ours_side] = ForcedLineup(players[ours_side], *ours_lu)
                players[other] = ForcedLineup(players[other], *theirs_lu)
            names = {ours_side: "ours", other: "theirs"}
            result = harness.play_game(bridge, f"draft-eval-{prefix}-{g}", players, names, rng, verbose=False,
                                       fixed_rosters={ours_side: mine, other: theirs})
            if result["winner_side"]:
                played += 1
                wins += result["winner_side"] == ours_side
    finally:
        bridge.close()
    return wins, played


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", default=LOSSES)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--ours", default="code")
    ap.add_argument("--theirs", default="code")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live-lineups", action="store_true", help="both sides bring and lead exactly what was played live")
    args = ap.parse_args(argv)
    for prefix in args.sessions:
        mine, theirs = teams(prefix)
        wins, played = evaluate(prefix, args.games, args.ours, args.theirs, args.seed, live_lineups=args.live_lineups)
        p = wins / played if played else 0.0
        half = 1.96 * math.sqrt(p * (1 - p) / played) if played else 0.0
        print(f"{prefix}: our six {[c['species'] for c in mine]}\n          their six {[c['species'] for c in theirs]}\n"
              f"          our drafted team wins {wins}/{played} = {100 * p:.0f}% (+-{100 * half:.0f}) with the same agent on both sides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
