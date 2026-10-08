"""Pokémon reference data and the small amount of battle math the agent does in code.

Plain language: LLMs are bad at arithmetic and at remembering 18x18 tables.
This module holds exactly that — the type chart, the stat formula, natures,
and lookups into the vendored Showdown tables (``data/pokedex.json``,
``data/moves.json``, built by ``scripts/build_dex.py``). Everything is a pure
function; nothing here talks to the network.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TYPES = (
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground",
    "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
)

# Attacking type -> {defending type: multiplier}; anything missing is 1.0.
TYPE_CHART: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5,
              "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2,
                 "ghost": 0.0, "dark": 2, "steel": 2, "fairy": 0.5},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0.0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5,
            "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5},
}

STATS = ("hp", "atk", "def", "spa", "spd", "spe")

# nature -> (boosted stat, lowered stat); neutral natures are absent.
NATURES: dict[str, tuple[str, str]] = {
    "lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
    "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
    "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
    "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
    "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa"),
}

LEVEL = 50  # VGC


def to_id(name: object) -> str:
    """Showdown id: lowercase letters and digits only (``"Flutter Mane"`` -> ``"fluttermane"``)."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


# -- type chart -----------------------------------------------------------------------------


def effectiveness(move_type: str, defender_types: list[str] | tuple[str, ...] | None) -> float:
    """Combined type multiplier of one move type against a defender's type(s)."""
    attack = to_id(move_type)
    if attack not in TYPE_CHART:
        return 1.0
    result = 1.0
    for defender_type in defender_types or ():
        result *= TYPE_CHART[attack].get(to_id(defender_type), 1.0)
    return result


def defensive_weaknesses(defender_types: list[str] | None) -> dict[str, float]:
    """Every attacking type that is not neutral against these defender types."""
    return {t: effectiveness(t, defender_types) for t in TYPES if effectiveness(t, defender_types) != 1.0}


# -- vendored tables ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def pokedex() -> dict:
    return _load("pokedex.json")


@lru_cache(maxsize=1)
def moves() -> dict:
    return _load("moves.json")


def _load(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _catalog() -> tuple[dict, dict]:
    """(card_id -> card, species id -> card) for every card ever seen live (data/cards.json, built by
    scripts/pool_from_live.py). The platform's cards are fixed sets, so a card seen once is known for good."""
    path = DATA_DIR / "cards.json"
    try:
        cards = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError):
        cards = []
    by_id = {c["card_id"]: c for c in cards if isinstance(c, dict) and c.get("card_id")}
    by_species = {to_id(c.get("species")): c for c in by_id.values()}
    return by_id, by_species


def catalog_card(card_id: object = None, species: object = None) -> dict | None:
    """The full set for a card we were never offered in this match (the opponent's first pick when they draft first,
    or a recovery after a restart). Looked up by card id first, then by species. Returns a copy, or None."""
    by_id, by_species = _catalog()
    card = by_id.get(str(card_id)) if card_id else None
    if card is None and species:
        card = by_species.get(to_id(species))
    return dict(card) if card else None


def species_info(name: object) -> dict | None:
    return pokedex().get(to_id(name))


def move_info(name: object) -> dict | None:
    return moves().get(to_id(name))


def expected_hits(move: dict | None, *, item: object = None) -> float:
    """How many times a move strikes: 1 for most, a fixed count (Surging Strikes 3, Dragon Darts 2),
    or the average of a range (2–5 hit moves average 3.1; 4–5 with Loaded Dice)."""
    multihit = (move or {}).get("multihit")
    if not multihit:
        return 1.0
    if isinstance(multihit, (int, float)):
        return float(multihit)
    if isinstance(multihit, list) and len(multihit) == 2:
        lo, hi = multihit
        if to_id(item) == "loadeddice" and lo <= 4 <= hi:
            return 4.5
        if (lo, hi) == (2, 5):
            return 3.1  # 2,3 at 35% each; 4,5 at 15% each
        return (lo + hi) / 2
    return 1.0


# -- stats -------------------------------------------------------------------------------


def nature_multiplier(nature: object, stat: str) -> float:
    boosted_lowered = NATURES.get(to_id(nature))
    if not boosted_lowered:
        return 1.0
    boosted, lowered = boosted_lowered
    if stat == boosted:
        return 1.1
    if stat == lowered:
        return 0.9
    return 1.0


def calc_hp(base: int, ev: int = 0, *, iv: int = 31, level: int = LEVEL) -> int:
    if base == 1:  # Shedinja
        return 1
    return math.floor((2 * base + iv + ev // 4) * level / 100) + level + 10


def calc_stat(base: int, ev: int = 0, *, nature_mult: float = 1.0, iv: int = 31, level: int = LEVEL) -> int:
    return math.floor((math.floor((2 * base + iv + ev // 4) * level / 100) + 5) * nature_mult)


def resolve_base_stats(card: dict) -> dict | None:
    """Base stats from the card if present, else from the vendored pokedex."""
    stats = card.get("base_stats") or card.get("baseStats")
    if isinstance(stats, dict) and stats:
        return {k: int(stats.get(k, 0)) for k in STATS if k in stats} or None
    info = species_info(card.get("species") or card.get("name"))
    return dict(info["base_stats"]) if info else None


def resolve_types(card: dict) -> list[str]:
    types = card.get("types")
    if isinstance(types, list) and types:
        return [str(t).lower() for t in types]
    info = species_info(card.get("species") or card.get("name"))
    return [t.lower() for t in info["types"]] if info else []


def parse_evs(evs: object) -> dict[str, int]:
    """EVs as a dict (``{"hp": 252, ...}``) or Showdown text (``"252 HP / 4 Def / 252 Spe"``)."""
    out = {stat: 0 for stat in STATS}
    if isinstance(evs, dict):
        for key, value in evs.items():
            stat = _stat_alias(key)
            if stat and isinstance(value, (int, float)):
                out[stat] = int(value)
    elif isinstance(evs, str):
        for part in evs.split("/"):
            tokens = part.strip().split()
            if len(tokens) == 2 and tokens[0].isdigit():
                stat = _stat_alias(tokens[1])
                if stat:
                    out[stat] = int(tokens[0])
    return out


def _stat_alias(name: object) -> str | None:
    key = to_id(name)
    aliases = {"hp": "hp", "atk": "atk", "attack": "atk", "def": "def", "defense": "def", "spa": "spa",
               "spatk": "spa", "specialattack": "spa", "spd": "spd", "spdef": "spd", "specialdefense": "spd",
               "spe": "spe", "speed": "spe"}
    return aliases.get(key)


def actual_stats(card: dict) -> dict[str, int] | None:
    """Level-50 stats for a drafted card (base stats + its EVs + its nature, 31 IVs)."""
    base = resolve_base_stats(card)
    if not base:
        return None
    evs = parse_evs(card.get("evs") or card.get("EVs"))
    nature = card.get("nature")
    stats = {"hp": calc_hp(base.get("hp", 0), evs["hp"])}
    for stat in ("atk", "def", "spa", "spd", "spe"):
        stats[stat] = calc_stat(base.get(stat, 0), evs[stat], nature_mult=nature_multiplier(nature, stat))
    return stats
