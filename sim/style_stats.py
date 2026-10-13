"""Play-style statistics of one side of a doubles battle, read from its Showdown protocol log.

    python sim/style_stats.py live tests/fixtures/live/<session>/ [...]      # the opponent's style in live games
    python sim/style_stats.py play --games 20 --p1 code --p2 agent.versions.pivot --seed 7

``live`` reads each fixture folder's last battle request (its ``protocol_log``) and reports the side that is NOT the
seat the fixture was recorded from (``...-seat0.json``: we were Showdown's p1). ``play`` runs games exactly as
sim/harness.py does (draft, Team Preview, battle; seats swap every game), with no decision logs written, and reports
each player's wins and style from the final protocol log of every game.

Definitions, the same for live and simulated games:

- a *slot-turn* is one of the side's Pokemon standing on the field at the start of a turn that played to its end
  (the turn block reaches ``|upkeep``); the turn a game ends in is left out (live logs stop before it).
- a *hard switch* is a ``|switch|`` of the side inside a turn block before any Pokemon moved (switch actions resolve
  first); a *double switch* turn has two of them. A switch right after the same slot's pivot move (U-turn, Volt Switch,
  Flip Turn, Parting Shot, ...) is a *pivot switch*; faint replacements after ``|upkeep`` count as neither.
- moves count every ``|move|`` of the side in a complete turn block, plus a ``|cant|`` that names the move it stopped
  (Taunt stopping Spore: the choice was Spore). ``protect``, ``pivot``, ``fake_out`` and ``disruption`` are the move
  groups below; rates are per slot-turn.
- ``intimidate`` counts the side's Intimidate activations anywhere in the log (leads and replacements included).
- the game length is the last ``|turn|N`` the log reached.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PIVOT_MOVES = frozenset({"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception", "shedtail", "batonpass"})
PROTECT_MOVES = frozenset({"protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "kingsshield", "obstruct"})
DISRUPTION_MOVES = frozenset({"taunt", "spore", "sleeppowder", "hypnosis", "yawn", "sing", "leechseed", "willowisp", "thunderwave",
                              "glare", "stunspore", "toxic", "snarl", "encore", "disable"})
COUNTS = ("slot_turns", "turn_blocks", "moves", "hard_switches", "double_switch_turns", "pivot_switches", "other_switches",
          "protect", "pivot", "fake_out", "disruption", "intimidate")


def to_id(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def side_stats(lines: list[str], side: str) -> dict:
    """Counts for ``side`` ("p1" or "p2") over one battle's protocol log (see the module docstring)."""
    counts = {k: 0 for k in COUNTS}
    alive: dict[str, bool] = {}
    last_turn = 0
    block: dict | None = None  # the turn block in progress: its slot count and the side's events so far

    def close(complete: bool) -> None:
        nonlocal block
        if block is not None and complete:
            counts["turn_blocks"] += 1
            counts["slot_turns"] += block["slots"]
            for key in ("moves", "hard_switches", "pivot_switches", "other_switches", "protect", "pivot", "fake_out", "disruption"):
                counts[key] += block[key]
            counts["double_switch_turns"] += block["hard_switches"] >= 2
        block = None

    def count_move(move: str) -> None:
        block["moves"] += 1
        block["protect"] += move in PROTECT_MOVES
        block["pivot"] += move in PIVOT_MOVES
        block["fake_out"] += move == "fakeout"
        block["disruption"] += move in DISRUPTION_MOVES

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        pos = parts[2][:3] if len(parts) > 2 else ""
        mine = pos[:2] == side
        if tag == "turn":
            close(complete=False)  # a block never reached |upkeep| (should not happen before the last turn)
            try:
                last_turn = int(parts[2])
            except (IndexError, ValueError):
                pass
            block = {"slots": sum(1 for p, up in alive.items() if p.startswith(side) and up), "moved": False, "last_move": {},
                     "upkeep": False, **{k: 0 for k in ("moves", "hard_switches", "pivot_switches", "other_switches", "protect",
                                                       "pivot", "fake_out", "disruption")}}
        elif tag == "upkeep":
            if block is not None and not block["upkeep"]:
                block["upkeep"] = True
                close(complete=True)
        elif tag in ("win", "tie"):
            close(complete=False)
        elif tag in ("switch", "drag", "replace"):
            alive[pos] = True
            if block is not None and mine and tag == "switch":
                pivoted = block["last_move"].get(pos) in PIVOT_MOVES or "[from]" in line
                if not block["moved"] and not pivoted:
                    block["hard_switches"] += 1
                elif pivoted:
                    block["pivot_switches"] += 1
                else:
                    block["other_switches"] += 1  # Eject Button / Eject Pack / Emergency Exit
        elif tag == "faint":
            alive[pos] = False
        elif tag == "move":
            if block is not None:
                block["moved"] = True
                if mine and not any(p.startswith("[from]ability:") or p.startswith("[from] ability:") for p in parts[4:]):
                    move = to_id(parts[3] if len(parts) > 3 else "")
                    block["last_move"][pos] = move
                    count_move(move)
        elif tag == "cant":
            if block is not None:
                block["moved"] = True
                if mine and len(parts) > 4 and parts[4]:
                    count_move(to_id(parts[4]))
        elif tag == "-ability" and mine and len(parts) > 3 and to_id(parts[3]) == "intimidate":
            counts["intimidate"] += 1
    close(complete=False)
    counts["turns"] = last_turn
    return counts


def summarize(games: list[dict]) -> dict:
    """Totals and rates over several games' ``side_stats``."""
    total = {k: sum(g[k] for g in games) for k in (*COUNTS, "turns")}
    n, slots = len(games), max(1, total["slot_turns"])
    return {
        "games": n,
        "avg_turns": round(total["turns"] / n, 2) if n else 0.0,
        "slot_turns": total["slot_turns"],
        "hard_switch_rate": round(total["hard_switches"] / slots, 3),
        "double_switch_turns": total["double_switch_turns"],
        "double_switch_per_game": round(total["double_switch_turns"] / n, 2) if n else 0.0,
        "pivot_switch_rate": round(total["pivot_switches"] / slots, 3),
        "protect_rate": round(total["protect"] / slots, 3),
        "pivot_move_rate": round(total["pivot"] / slots, 3),
        "fake_out_per_game": round(total["fake_out"] / n, 2) if n else 0.0,
        "disruption_rate": round(total["disruption"] / slots, 3),
        "intimidate_per_game": round(total["intimidate"] / n, 2) if n else 0.0,
        "totals": total,
    }


def format_row(label: str, s: dict) -> str:
    return (f"{label:28s} games {s['games']:3d}  turns {s['avg_turns']:5.2f}  slot-turns {s['slot_turns']:4d}  "
            f"switch/slot-turn {s['hard_switch_rate']:.3f}  double-switch turns {s['double_switch_turns']:2d} "
            f"({s['double_switch_per_game']:.2f}/game)  protect {s['protect_rate']:.3f}  pivot moves {s['pivot_move_rate']:.3f} "
            f"(+switches {s['pivot_switch_rate']:.3f})  fake out {s['fake_out_per_game']:.2f}/game  "
            f"disruption {s['disruption_rate']:.3f}  intimidate {s['intimidate_per_game']:.2f}/game")


# -- live fixtures ---------------------------------------------------------------------------------------------------


def live_game(folder: Path) -> tuple[list[str], str]:
    """(the last battle request's protocol log, the opponent's Showdown side) of one live fixture folder."""
    moving = sorted(folder.glob("moving-*.json"))
    if not moving:
        raise SystemExit(f"{folder}: no moving-*.json requests")
    last = moving[-1]
    seat = re.search(r"seat(\d)", last.name)
    if not seat:
        raise SystemExit(f"{last}: the file name does not say which seat it was recorded from")
    raw = json.loads(last.read_text())
    lines = (raw.get("observation") or {}).get("protocol_log") or []
    return list(lines), ("p2" if seat.group(1) == "0" else "p1")


def cmd_live(args) -> int:
    games = []
    for name in args.folders:
        lines, opp = live_game(Path(name))
        stats = side_stats(lines, opp)
        games.append(stats)
        print(format_row(Path(name).name[:8], summarize([stats])))
    print(format_row("ALL (opponent side)", summarize(games)))
    if args.json:
        print(json.dumps(summarize(games)))
    return 0


# -- simulated games -------------------------------------------------------------------------------------------------


def cmd_play(args) -> int:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from sim import harness

    class Bridge(harness.Bridge):
        """Keeps every game's last snapshot, for its final protocol log."""

        def __init__(self) -> None:
            super().__init__()
            self.last: dict[str, dict] = {}

        def call(self, **msg) -> dict:
            reply = super().call(**msg)
            if msg.get("cmd") == "choose" and reply.get("ok"):
                self.last[msg["id"]] = reply["state"]
            return reply

    rng = random.Random(args.seed)
    bridge = Bridge()
    per_spec: dict[str, list[dict]] = {}
    wins: dict[str, int] = {}
    decided = 0
    started = time.monotonic()
    try:
        for g in range(args.games):
            specs = {"p1": args.p1, "p2": args.p2} if g % 2 == 0 else {"p1": args.p2, "p2": args.p1}
            players, names = {}, {}
            for side, spec in specs.items():
                players[side], _ = harness.make_player(spec, rng)
                if hasattr(players[side], "_log_dir"):
                    players[side]._log_dir = "/dev/null"  # DecisionLog cannot create it: no decision logs on a tight disk
                if hasattr(players[side], "_print"):  # only the error lines reach the console
                    players[side]._print = lambda line: print(line, file=sys.stderr, flush=True) if "ERROR" in line else None
                names[side] = spec
            game_id = f"style-{args.seed}-{g}"
            try:
                result = harness.play_game(bridge, game_id, players, names, rng, verbose=False)
            except Exception as exc:  # noqa: BLE001
                print(f"game {g + 1}: ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            snap = bridge.last.pop(game_id, None) or {}
            lines = ((snap.get("logs") or {}).get("p1")) or []
            for side, spec in specs.items():
                per_spec.setdefault(spec if args.p1 != args.p2 else f"{spec}:{side}", []).append(side_stats(lines, side))
            if result["winner"]:
                decided += 1
                wins[result["winner"]] = wins.get(result["winner"], 0) + 1
            print(f"game {g + 1}: p1 {specs['p1']} vs p2 {specs['p2']} -> {result['winner']} in {result['turns']} turns "
                  f"(rejected {result['rejected']})", flush=True)
    finally:
        bridge.close()
    print(f"\n{decided} decided games in {int(time.monotonic() - started)} s")
    for spec in dict.fromkeys((args.p1, args.p2)):
        print(f"  {spec:28s} {wins.get(spec, 0):3d} wins ({100 * wins.get(spec, 0) / decided if decided else 0:.0f}%)")
    for spec, games in per_spec.items():
        print(format_row(spec, summarize(games)))
    if args.json:
        print(json.dumps({spec: summarize(games) for spec, games in per_spec.items()} | {"wins": wins, "decided": decided}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    live = sub.add_parser("live", help="the opponent's style in live fixture folders")
    live.add_argument("folders", nargs="+")
    live.add_argument("--json", action="store_true")
    play = sub.add_parser("play", help="play simulated games (as sim/harness.py) and report each player's style")
    play.add_argument("--games", type=int, default=10)
    play.add_argument("--p1", default="code")
    play.add_argument("--p2", default="agent.versions.pivot")
    play.add_argument("--seed", type=int, default=7)
    play.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return cmd_live(args) if args.cmd == "live" else cmd_play(args)


if __name__ == "__main__":
    sys.exit(main())
