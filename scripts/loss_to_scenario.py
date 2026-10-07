"""Turn a logged battle turn into a scenario skeleton for scripts/scenarios.py (item 5).

    python scripts/loss_to_scenario.py logs/sim/fable/sim-1791349727-12.jsonl --turn 4

Prints: both sides' full sets as CARDS entries (if not already in scenarios.py), the Scenario(...) block with
actives, HP fractions, boosts, field/side conditions, and what the agent actually played vs the code's top
candidate, plus the turn sheet's warnings. You still write the acceptance rule and CHECK THE POSITION BY HAND
(two of the first six hard scenarios were wrong on first writing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.pokemon import data  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="a decision log (logs/sim/<player>/<game>.jsonl or logs/<session>.jsonl)")
    parser.add_argument("--turn", type=int, required=True, help="battle turn number to extract")
    args = parser.parse_args(argv)
    rows = [json.loads(l) for l in open(args.log)]
    turn = next((r for r in rows if r.get("kind") == "turn" and r.get("turn") == args.turn), None)
    if turn is None:
        print("no such turn; turns in this log:", [r.get("turn") for r in rows if r.get("kind") == "turn"])
        return 2
    sheet = turn.get("turn_sheet") or {}
    # Known sets are not in the log line itself; recover them from the lineup/draft lines when present.
    known = {}
    for r in rows:
        for side in ("mine", "opponent"):
            for c in ((r.get("known_sets") or {}).get(side) or []):
                known[data.to_id(c.get("species"))] = (side, c)
    actives_mine = [e.get("active") for e in sheet.get("our_damage_estimates") or []]
    actives_theirs = [t.get("species") for t in sheet.get("opponent_threats") or []]
    print("# --- scenario skeleton (verify every number by hand before trusting it) ---")
    print(f"# from {Path(args.log).name} turn {args.turn}: played {json.dumps(turn.get('payload'))}")
    top = (sheet.get("candidate_turns") or [{}])[0]
    print(f"# code top candidate: {top.get('name')} :: {top.get('why')}")
    print(f"# model said: {(turn.get('answer') or {}).get('reasoning_summary')}")
    for w in sheet.get("WARNINGS") or []:
        print(f"# WARN: {w}")
    print(f"# speed order: {[(s['species'], s['speed']) for s in sheet.get('speed_order') or []]}")
    print(f"# notes: {sheet.get('notes')}")
    hp_theirs = {t["species"]: t.get("hp_pct") for t in sheet.get("opponent_threats") or []}
    print()
    print("Scenario(")
    print(f'    "from_{Path(args.log).stem.replace("-", "_")}_t{args.turn}", "<describe the position>",')
    print(f"    ({', '.join(repr(data.to_id(a)) for a in actives_mine)}), ({', '.join(repr(data.to_id(a)) for a in actives_theirs)}), hard=True, turn={args.turn},")
    print(f"    their_hp={{{', '.join(f'{data.to_id(k)!r}: {v / 100:.2f}' for k, v in hp_theirs.items() if v is not None)}}},")
    print("    # my_hp={...}, their_boosts={...}, my_side=[...], their_side=[...], fields=[...]  <- fill from the observation if relevant")
    print("    accept=lambda p: ...,")
    print('    accept_text="...",')
    print(")")
    missing = [a for a in actives_mine + actives_theirs if a and data.to_id(a) not in known]
    if missing:
        print(f"\n# species not found in known_sets of this log (add CARDS entries by hand): {missing}")
    for key, (side, c) in known.items():
        if key in {data.to_id(a) for a in actives_mine + actives_theirs}:
            print(f'    "{key}": card("{c.get("species")}", {json.dumps(c.get("moves"))}, ability="{c.get("ability")}", nature="{c.get("nature")}", evs={json.dumps(c.get("evs") or {})}, item="{c.get("item")}"),  # {side}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
