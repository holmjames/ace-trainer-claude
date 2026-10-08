"""Rebuild the card catalog from every real card seen in captured live draft states (tests/fixtures/live/*/draft-*.json).

The platform's cards are fixed, complete sets (species, item, ability, nature, EVs, moves), drawn 18 at a time from a
catalog it does not publish. Everything we have ever been offered live is the best catalog we can have:

- data/cards.json is what the AGENT reads: when the opponent drafts first, their first pick is never offered to us, so
  its set is only knowable from a card seen in an earlier match (the Oct 7 losses: Incineroar and Dragonite were
  first picks and played the whole match as blank sets with no EVs).
- sim/cards.json is what the SIMULATOR drafts from. It used to mix in our own guessed sets for species never seen
  live (Choice Specs Gholdengo, Calyrex-Ice, Miraidon...), so offline tuning ran on a pool that does not exist.
  It is now the live catalog only.

    python scripts/pool_from_live.py            # report + write both files
    python scripts/pool_from_live.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("card_id", "species", "item", "ability", "nature", "level", "evs", "ivs", "moves")


def live_cards(root: Path = ROOT) -> dict[str, dict]:
    """card_id -> card, for every complete card in the captured draft observations (later captures win)."""
    real: dict[str, dict] = {}
    paths = sorted(glob.glob(str(root / "tests/fixtures/live/*/*.json")), key=lambda p: Path(p).stat().st_mtime)
    for path in paths:
        try:
            obs = json.load(open(path)).get("observation") or {}
        except (OSError, ValueError):
            continue
        if not isinstance(obs, dict):
            continue
        for k in ("pool", "available_cards", "cards"):
            for card in obs.get(k) or []:
                if isinstance(card, dict) and card.get("card_id") and card.get("species") and card.get("moves"):
                    real[card["card_id"]] = {f: card[f] for f in FIELDS if f in card}
    return real


def _committed_sim_pool() -> dict[str, dict]:
    """The simulator pool as last committed (what offline tuning actually ran on), for the report."""
    try:
        text = subprocess.run(["git", "show", "HEAD:sim/cards.json"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return {c["card_id"]: c for c in json.loads(text)}
    except (subprocess.CalledProcessError, ValueError, OSError):
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    real = live_cards()
    old = _committed_sim_pool()
    replaced = sorted(c["species"] for cid, c in real.items() if cid in old and {f: old[cid].get(f) for f in FIELDS} != {f: c.get(f) for f in FIELDS})
    added = sorted(c["species"] for cid, c in real.items() if cid not in old)
    dropped = sorted(c["species"] for cid, c in old.items() if cid not in real)
    print(f"live cards seen: {len(real)}")
    print(f"  sets that differ from the committed simulator pool: {len(replaced)} {replaced}")
    print(f"  new to the simulator: {len(added)} {added}")
    print(f"  guessed cards never seen live (dropped): {len(dropped)} {dropped}")
    if not args.dry_run:
        text = json.dumps(sorted(real.values(), key=lambda c: c["species"]), indent=1, ensure_ascii=False) + "\n"
        for rel in ("data/cards.json", "sim/cards.json"):
            (ROOT / rel).write_text(text)
            print("wrote", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
