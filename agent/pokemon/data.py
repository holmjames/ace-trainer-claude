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


def expected_hits(move: dict | None, *, item: object = None, ability: object = None) -> float:
    """How many times a move strikes on average: 1 for most, a fixed count (Surging Strikes 3, Dragon Darts 2),
    or the mean of ``hit_count_distribution`` (2–5 hit moves average 3.1; 4–5 with Loaded Dice; 5 with Skill Link)."""
    return sum(n * p for n, p in hit_count_distribution(move, {"item": item, "ability": ability}).items())


# -- what the move table alone cannot say: weight, accuracy, hit counts, Intimidate -------------------------------

# Weights in kg of every live-catalog species (data/cards.json), copied once from the Showdown dex in
# sim/node_modules/pokemon-showdown (0.11.11; Iron Treads, Iron Valiant, Talonflame and Toxapex joined on Oct 8). Low Kick / Grass Knot take their power from the target's weight and
# Heavy Slam / Heat Crash from the user:target ratio; the move table lists them at 0 power, which read as "no damage"
# (Oct 8 review: Tyranitar's Low Kick, 81-97% into Kingambit, had no row). ``species_weight`` falls back to the
# vendored pokedex for anything else.
SPECIES_WEIGHT_KG: dict[str, float] = {
    "amoonguss": 10.5, "arcanine": 155.0, "archaludon": 60.0, "basculegion": 110.0, "baxcalibur": 210.0,
    "chienpao": 152.2, "cresselia": 85.6, "dondozo": 220.0, "dragonite": 210.0, "fluttermane": 4.0,
    "garchomp": 95.0, "gholdengo": 30.0, "greattusk": 320.0, "grimmsnarl": 61.0, "gyarados": 235.0,
    "hatterene": 5.1, "incineroar": 83.0, "indeedeef": 28.0, "ironbundle": 11.0, "ironhands": 380.7, "irontreads": 240.0,
    "ironvaliant": 35.0,
    "kingambit": 120.0, "klefki": 3.0, "landorustherian": 68.0, "lucario": 54.0, "milotic": 162.0,
    "pelipper": 28.0, "porygon2": 32.5, "rillaboom": 90.0, "sableye": 11.0, "salamence": 102.6,
    "screamtail": 8.0, "sneasler": 43.0, "sylveon": 23.5, "talonflame": 24.5, "thundurus": 61.0, "tinglu": 699.7,
    "torkoal": 80.4, "tornadus": 63.0, "toxapex": 14.5, "tyranitar": 202.0, "ursaluna": 290.0, "whimsicott": 6.6, "wochien": 74.2,
}
WEIGHT_MOVES = frozenset({"lowkick", "grassknot"})          # power from the target's weight
WEIGHT_RATIO_MOVES = frozenset({"heavyslam", "heatcrash"})  # power from user weight / target weight
FIXED_DAMAGE_MOVES = frozenset({"ruination", "beatup"})     # damage with no base power of their own
DAMAGING_WITHOUT_POWER = WEIGHT_MOVES | WEIGHT_RATIO_MOVES | FIXED_DAMAGE_MOVES
# Low Kick / Grass Knot: (at least this many kg, power); Heavy Slam / Heat Crash: (at least this user:target ratio, power).
WEIGHT_BANDS = ((200, 120), (100, 100), (50, 80), (25, 60), (10, 40), (0, 20))
WEIGHT_RATIO_BANDS = ((5, 120), (4, 100), (3, 80), (2, 60), (0, 40))


def species_weight(name: object) -> float:
    """A species' weight in kg (0.0 when unknown): the catalog table first, then the vendored pokedex."""
    key = to_id(name)
    if key in SPECIES_WEIGHT_KG:
        return SPECIES_WEIGHT_KG[key]
    info = pokedex().get(key) or {}
    try:
        return float(info.get("weightkg") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _kg(value: object) -> float:
    """Kilograms from a number, a species name, or a card / Mon-like object with a species."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return species_weight(value.get("species") or value.get("name"))
    species = getattr(value, "species", None)
    return species_weight(species if species is not None else value)


def _move_id(move: object) -> str:
    if isinstance(move, dict):
        return to_id(move.get("id") or move.get("name"))
    return to_id(move)


def weight_power(move: object, target: object, user: object = None) -> int | None:
    """Base power of a weight-based move: Low Kick / Grass Knot from the target's weight, Heavy Slam / Heat Crash from
    the user:target weight ratio. ``target`` / ``user`` may be kg, a species name, a card or a Mon. None for any other
    move; an unknown weight gives the lowest band."""
    mid = _move_id(move)
    if mid in WEIGHT_MOVES:
        kg = _kg(target)
        return next(power for limit, power in WEIGHT_BANDS if kg >= limit)
    if mid in WEIGHT_RATIO_MOVES:
        user_kg, target_kg = _kg(user), _kg(target)
        ratio = user_kg / target_kg if user_kg and target_kg else 0.0
        return next(power for limit, power in WEIGHT_RATIO_BANDS if ratio >= limit)
    return None


def is_damaging(move: object) -> bool:
    """Does this move deal direct damage: a Physical / Special move with base power, or one whose power the table
    cannot hold (weight-based moves, Ruination, Beat Up). The one gate every "skip status moves" loop should use."""
    info = move if isinstance(move, dict) else (move_info(move) or {})
    mid = _move_id(move) or _move_id(info)
    if info.get("category") not in ("Physical", "Special"):
        return False
    return (info.get("base_power") or 0) > 0 or mid in DAMAGING_WITHOUT_POWER


RAIN_SURE_HIT = frozenset({"hurricane", "thunder", "bleakwindstorm", "wildboltstorm", "sandsearstorm"})
SNOW_SURE_HIT = frozenset({"blizzard"})
SUN_HALVED = frozenset({"hurricane", "thunder"})


def hit_chance(move: object, weather: object = None) -> float:
    """The move's chance to land as a probability: 1.0 for never-miss moves (accuracy ``true``) and for Hurricane /
    Thunder / the storm moves in rain (Blizzard in snow), 0.5 for Hurricane / Thunder in sun, else accuracy / 100.
    No evasion or accuracy stages: nothing in the catalog changes them."""
    info = move if isinstance(move, dict) else (move_info(move) or {})
    mid = _move_id(move) or _move_id(info)
    acc = info.get("accuracy", True)
    if acc is True or acc is None:
        return 1.0
    w = to_id(weather)
    if "rain" in w and mid in RAIN_SURE_HIT:
        return 1.0
    if ("snow" in w or "hail" in w) and mid in SNOW_SURE_HIT:
        return 1.0
    if ("sun" in w or "drought" in w) and mid in SUN_HALVED:
        return 0.5
    try:
        return max(0.0, min(1.0, float(acc) / 100.0))
    except (TypeError, ValueError):
        return 1.0


MULTI_HIT_2_5 = {2: 0.35, 3: 0.35, 4: 0.15, 5: 0.15}


def hit_count_distribution(move: object, mon: object = None) -> dict[int, float]:
    """How many times the move strikes, as {hits: probability}: {1: 1.0} for ordinary moves, a fixed count for
    Surging Strikes / Dragon Darts / Population Bomb, 2-5 hits at 35/35/15/15% for the variable moves, 4-5 at 50/50
    with Loaded Dice and always 5 with Skill Link. ``mon`` (the user: a Mon, or a dict with item / ability) supplies
    the item and ability."""
    info = move if isinstance(move, dict) else (move_info(move) or {})
    multihit = info.get("multihit")
    if not multihit:
        return {1: 1.0}
    if isinstance(multihit, (int, float)):
        return {int(multihit): 1.0}
    if not (isinstance(multihit, (list, tuple)) and len(multihit) == 2):
        return {1: 1.0}
    lo, hi = int(multihit[0]), int(multihit[1])
    if lo >= hi:
        return {lo: 1.0}
    if isinstance(mon, dict):
        item, ability = mon.get("item"), mon.get("ability")
    else:
        item, ability = getattr(mon, "item", None), getattr(mon, "ability", None)
    if (lo, hi) == (2, 5):
        if to_id(ability) == "skilllink":
            return {5: 1.0}
        if to_id(item) == "loadeddice":
            return {4: 0.5, 5: 0.5}
        return dict(MULTI_HIT_2_5)
    return {n: 1.0 / (hi - lo + 1) for n in range(lo, hi + 1)}


def rolls_reach(hits: int, per_hit_lo: float, per_hit_hi: float, need: float) -> float:
    """Chance that ``hits`` strikes, each rolling uniformly between ``per_hit_lo`` and ``per_hit_hi``, add up to at
    least ``need`` (the Irwin-Hall distribution: the sum of several rolls bunches around its mean, so five strikes
    that reach a KO only when all roll high are a long shot, not a coin flip)."""
    n = int(hits)
    if n <= 0:
        return 0.0
    if n * per_hit_lo >= need:
        return 1.0
    if n * per_hit_hi < need:
        return 0.0
    width = per_hit_hi - per_hit_lo
    if width <= 0:
        return 0.0
    s = (need - n * per_hit_lo) / width  # the sum of n uniform(0, 1) rolls must reach s
    cdf = sum((-1) ** k * math.comb(n, k) * (s - k) ** n for k in range(int(math.floor(s)) + 1)) / math.factorial(n)
    return max(0.0, min(1.0, 1.0 - cdf))


def multi_hit_ko_chance(per_hit_lo: float, per_hit_hi: float, hp_frac: float, distribution: dict, *,
                        first_hit: tuple[float, float] | list | None = None) -> float:
    """Chance a multi-hit move KOs: ``per_hit_lo`` / ``per_hit_hi`` are one strike's damage as a percentage of the
    target's MAX HP and ``hp_frac`` the fraction it has left (or the strike as % of CURRENT HP with ``hp_frac`` 1.0,
    the way the sheet reports it); ``distribution`` is {hits: probability}. Each strike rolls on its own, so a Loaded
    Dice Scale Shot that needs all five strikes high is far from a 50% KO. ``first_hit`` is the first strike's own
    range when it differs (Multiscale / Shadow Shield halve only that one; a resist berry is eaten by it)."""
    need = 100.0 * hp_frac
    total = 0.0
    for hits, p in distribution.items():
        n = int(hits)
        if first_hit is not None and n >= 1:
            lo, hi = first_hit[0] + per_hit_lo * (n - 1), first_hit[1] + per_hit_hi * (n - 1)
            total += float(p) * rolls_reach(n, lo / n, hi / n, need)
        else:
            total += float(p) * rolls_reach(n, per_hit_lo, per_hit_hi, need)
    return max(0.0, min(1.0, total))


# Abilities that stop Intimidate (Mirror Armor bounces it back; the foe itself is untouched).
INTIMIDATE_BLOCKERS = frozenset({"clearbody", "whitesmoke", "hypercutter", "fullmetalbody", "innerfocus", "owntempo",
                                 "oblivious", "scrappy", "mirrorarmor"})


def intimidate_stages(ability: object, item: object = None) -> dict[str, int]:
    """What one Intimidate does to a Pokémon with this ability and item: -1 Attack, nothing through a blocking
    ability or a Clear Amulet, +1 Attack for Defiant / Guard Dog / Contrary, -2 for Simple, and Competitive takes
    the drop but gains +2 Special Attack (Rattled also gains +1 Speed)."""
    a, i = to_id(ability), to_id(item)
    if a in INTIMIDATE_BLOCKERS or i == "clearamulet":
        return {}
    if a in ("defiant", "guarddog", "contrary"):
        return {"atk": 1}
    if a == "simple":
        return {"atk": -2}
    out = {"atk": -1}
    if a == "competitive":
        out["spa"] = 2
    if a == "rattled":
        out["spe"] = 1
    return out


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
