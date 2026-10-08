"""Merge every real card seen in captured live draft states (tests/fixtures/live/*/draft-*.json) into sim/cards.json.

The platform's cards are complete sets (species, item, ability, nature, EVs, moves). A real set replaces our guess for the
same species; new species are appended. Run after any live match played with AGENT_CAPTURE_DIR set:

    python scripts/pool_from_live.py            # report + write
    python scripts/pool_from_live.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def key(species: str) -> str:
    return "".join(ch for ch in species.lower() if ch.isalnum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    real: dict[str, dict] = {}
    for path in sorted(glob.glob(str(ROOT / "tests/fixtures/live/*/draft-*.json"))):
        obs = json.load(open(path)).get("observation") or {}
        for k in ("pool", "available_cards", "cards"):
            for card in obs.get(k) or []:
                if card.get("species") and card.get("moves"):
                    real[key(card["species"])] = card
        for roster in (obs.get("rosters") or {}).values() if isinstance(obs.get("rosters"), dict) else []:
            for card in roster or []:
                if isinstance(card, dict) and card.get("species") and card.get("moves"):
                    real[key(card["species"])] = card
    pool_path = ROOT / "sim/cards.json"
    pool = json.load(open(pool_path))
    by_key = {key(c["species"]): c for c in pool}
    replaced = [c["species"] for k, c in real.items() if k in by_key and by_key[k] != c]
    added = [c["species"] for k, c in real.items() if k not in by_key]
    for k, c in real.items():
        by_key[k] = c
    print(f"real cards seen: {len(real)} | replaced our guess: {len(replaced)} {replaced} | new species: {len(added)} {added}")
    print(f"pool size: {len(pool)} -> {len(by_key)}")
    if not args.dry_run:
        pool_path.write_text(json.dumps(sorted(by_key.values(), key=lambda c: c['species']), indent=1, ensure_ascii=False) + "\n")
        print("wrote", pool_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
