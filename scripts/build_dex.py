"""Vendor species and move data from Pokémon Showdown into data/*.json.

Pokémon Showdown (MIT licensed, https://github.com/smogon/pokemon-showdown)
serves its data tables as JSON for its web client. This script downloads the
two we need, keeps only the fields the agent uses, and writes small files the
agent loads at startup with no network call:

    data/pokedex.json   {species_id: {name, types, base_stats, abilities, weightkg}}
    data/moves.json     {move_id: {name, type, category, base_power, accuracy, priority, target, flags, multihit}}

Ids are Showdown ids (lowercase, letters and digits only), the same
normalization the game server uses for species, so lookups are direct.

Run from the repo root:

    python scripts/build_dex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

POKEDEX_URL = "https://play.pokemonshowdown.com/data/pokedex.json"
MOVES_URL = "https://play.pokemonshowdown.com/data/moves.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
MOVE_FLAGS = ("contact", "protect", "sound", "bullet", "punch", "slicing", "wind", "charge", "recharge")


def fetch(url: str) -> dict:
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"{url} did not return a JSON object")
    return data


def trim_pokedex(raw: dict) -> dict:
    out = {}
    for species_id, entry in raw.items():
        if not isinstance(entry, dict) or "baseStats" not in entry or "types" not in entry:
            continue
        out[species_id] = {
            "name": entry.get("name"),
            "types": list(entry["types"]),
            "base_stats": dict(entry["baseStats"]),
            "abilities": sorted(str(a) for a in (entry.get("abilities") or {}).values()),
            "weightkg": entry.get("weightkg"),
        }
    return out


def trim_moves(raw: dict) -> dict:
    out = {}
    for move_id, entry in raw.items():
        if not isinstance(entry, dict) or "type" not in entry:
            continue
        flags = entry.get("flags") or {}
        out[move_id] = {
            "name": entry.get("name"),
            "type": entry["type"],
            "category": entry.get("category"),
            "base_power": entry.get("basePower", 0),
            "accuracy": entry.get("accuracy", True),
            "priority": entry.get("priority", 0),
            "target": entry.get("target"),
            "flags": [flag for flag in MOVE_FLAGS if flags.get(flag)],
            "multihit": entry.get("multihit"),  # int, [min, max], or absent
            "will_crit": bool(entry.get("willCrit")),  # Surging Strikes, Wicked Blow, Flower Trick
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pokedex = trim_pokedex(fetch(POKEDEX_URL))
    moves = trim_moves(fetch(MOVES_URL))
    for name, table in (("pokedex.json", pokedex), ("moves.json", moves)):
        path = OUT_DIR / name
        path.write_text(json.dumps(table, ensure_ascii=False, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        print(f"wrote {path.relative_to(OUT_DIR.parent)}: {len(table)} entries, {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
