"""The battle turn sheet: the arithmetic a strong player does before every move.

Plain language: before the model sees a doubles turn, this module works out
the things LLMs get wrong when left to intuition — who moves first, roughly
how much each of our moves does to each target (as a percentage of the
target's remaining HP), what the opponent's known sets can do to us, and a
short ranked list of sensible complete turns. The model then judges among
good options instead of inventing them. The top candidate doubles as the
deterministic fallback.

Everything degrades gracefully: when a number can't be computed (unknown
species, missing field), that line is simply omitted from the sheet.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import data
from .memory import MatchMemory, species_key
from .tuning import DEFAULTS

LEVEL = 50
SPREAD_TARGETS = {"allAdjacentFoes", "allAdjacent"}
# Items that boost one type's moves by 20%.
TYPE_BOOST_ITEMS = {
    "silkscarf": "normal", "charcoal": "fire", "mysticwater": "water", "magnet": "electric", "miracleseed": "grass",
    "nevermeltice": "ice", "blackbelt": "fighting", "poisonbarb": "poison", "softsand": "ground", "sharpbeak": "flying",
    "twistedspoon": "psychic", "silverpowder": "bug", "hardstone": "rock", "spelltag": "ghost", "dragonfang": "dragon",
    "blackglasses": "dark", "metalcoat": "steel", "fairyfeather": "fairy",
}
RANDOM_MIN, RANDOM_MAX = 0.85, 1.0


# -- stat helpers -----------------------------------------------------------------------------


def stage_multiplier(stage: int) -> float:
    stage = max(-6, min(6, int(stage or 0)))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def staged_stat(base: float, stage: int) -> float:
    """A stat at a stage the way the engine does it: floor(stat x numerator / denominator) in integers (Kingambit's
    205 Attack at -1 is 136, not 136.67; Oct 8 live match da9c168e: Waterfall did 108 HP against a sheet minimum of
    109.7 built on the fraction). Items and abilities multiply afterwards."""
    stage = max(-6, min(6, int(stage or 0)))
    if not stage:
        return base
    numerator, denominator = (2 + stage, 2) if stage > 0 else (2, 2 - stage)
    return math.floor(base * numerator / denominator)


@dataclass
class Mon:
    """One Pokémon on the field, merged from the live summary and its drafted card."""

    species: str
    side: str  # "mine" | "theirs"
    position: int | None  # our slot 0/1; their board position 1/2
    types: list[str]
    stats: dict[str, int]
    hp_fraction: float
    status: str | None
    item: str | None
    ability: str | None
    boosts: dict[str, int]
    moves: list[dict]  # known moves (move_info dicts with "id")
    fainted: bool = False

    tailwind: bool = False
    boosted_stat: str | None = None  # Protosynthesis / Quark Drive: the stat the paradox boost raises (set by the sheet)
    paradox_stat: str | None = None  # the same, read from the battle log (-start protosynthesisatk): exact, and survives Booster Energy being used up
    unburden: bool = False  # Unburden active: its item was used up during this stint on the field (speed x2), from the log

    def stat(self, name: str, *, min_stage: int | None = None, max_stage: int | None = None) -> float:
        """One stat with stages, status, item and paradox boosts applied. ``min_stage``/``max_stage`` clamp the
        stage (a critical hit ignores the attacker's drops and the defender's raises)."""
        base = self.stats.get(name, 0)
        stage = self.boosts.get(name, 0)
        if min_stage is not None:
            stage = max(stage, min_stage)
        if max_stage is not None:
            stage = min(stage, max_stage)
        value = staged_stat(base, stage)
        if name == "spe":
            if self.status == "par":
                value *= 0.5
            if data.to_id(self.item) == "choicescarf":
                value *= 1.5
            if self.tailwind:
                value *= 2
            if self.unburden and data.to_id(self.ability) == "unburden":
                value *= 2  # Oct 7: Sneasler after its Aguav Berry (378) was listed as slower than Gholdengo (298)
        if name == "atk" and data.to_id(self.item) == "choiceband":
            value *= 1.5
        if name == "spa" and data.to_id(self.item) == "choicespecs":
            value *= 1.5
        if name == "spd" and data.to_id(self.item) == "assaultvest":
            value *= 1.5
        if name in ("def", "spd") and data.to_id(self.item) == "eviolite":
            value *= 1.5
        if name == "atk" and data.to_id(self.ability) == "guts" and self.status:
            value *= 1.5
        if self.boosted_stat == name:
            value *= 1.5 if name == "spe" else 1.3
        return value

    @property
    def grounded(self) -> bool:
        return "flying" not in self.types and data.to_id(self.ability) != "levitate" and data.to_id(self.item) != "airballoon"


UNKNOWN_ITEMS = ("unknown_item", "unknownitem")


def build_mon(summary: dict, card: dict | None, *, side: str, position: int | None, item_lost: bool = False) -> Mon | None:
    """Merge the live summary (HP, boosts, status, what the battle has revealed) over the drafted card (the full set).

    The live server marks what it has not revealed as unknown (opponent item "unknown_item", ability null, moves =
    only the ones used so far); those placeholders never hide the card. Items: our own summary is authoritative (""
    once consumed); an opponent's revealed item wins over the card, and ``item_lost`` (the protocol log showed
    -enditem: popped Air Balloon, eaten berry, Knock Off) removes it."""
    summary = summary or {}
    card = card or {}
    merged = {**card, **{k: v for k, v in summary.items() if v not in (None, "", [], {}) and v not in UNKNOWN_ITEMS}}
    species = merged.get("species") or merged.get("name")
    if not species:
        return None
    if side == "mine" and "item" in summary and summary.get("item") in (None, ""):
        merged["item"] = None  # consumed or knocked off: our own summary knows
    if item_lost:
        merged["item"] = None
    stats = data.actual_stats({**merged, "evs": card.get("evs") or merged.get("evs"), "nature": card.get("nature") or merged.get("nature")}) or {}
    moves: list[dict] = []
    names = list(card.get("moves") or [])
    for revealed in summary.get("moves") or []:  # a used move the card does not list (shouldn't happen) still counts
        rid = data.to_id(revealed.get("id") if isinstance(revealed, dict) else revealed)
        if rid and rid not in {data.to_id(n.get("id") if isinstance(n, dict) else n) for n in names}:
            names.append(revealed)
    for name in names:
        move_name = name.get("id") if isinstance(name, dict) else name
        info = data.move_info(move_name)
        moves.append({"id": data.to_id(move_name), **(info or {"type": None, "category": None, "base_power": 0, "priority": 0, "target": None})})
    frac = summary.get("current_hp_fraction") if isinstance(summary, dict) else None
    if frac is None and isinstance(summary, dict) and summary.get("max_hp"):
        frac = (summary.get("current_hp") or 0) / summary["max_hp"]
    mon = Mon(
        species=str(species),
        side=side,
        position=position,
        types=data.resolve_types(merged),
        stats=stats,
        hp_fraction=float(frac) if frac is not None else 1.0,
        status=(summary or {}).get("status") or None,
        item=merged.get("item"),
        ability=merged.get("ability"),
        boosts={k: int(v) for k, v in ((summary or {}).get("boosts") or {}).items() if isinstance(v, (int, float))},
        moves=moves,
        fainted=bool((summary or {}).get("fainted")),
    )
    mon.boosted_stat = paradox_boosted_stat(mon, None)
    return mon


# -- damage ------------------------------------------------------------------------------------


PROTECT_LIKE = {"protect", "detect", "wideguard", "quickguard", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "kingsshield", "obstruct"}
CHOICE_ITEMS = {"choicescarf", "choiceband", "choicespecs"}
RUIN_ABILITIES = {"swordofruin": ("def", "Physical"), "beadsofruin": ("spd", "Special"),
                  "vesselofruin": ("spa", "Special"), "tabletsofruin": ("atk", "Physical")}
PARADOX_BOOST = 5461 / 4096  # Hadron Engine / Orichalcum Pulse offensive boost (x1.333, @smogon/calc; the terrain base-power boost is 5325 = x1.3)


def _poke_round(x: float) -> int:
    """The engine's rounding of 4096-based modifiers: halves round DOWN."""
    whole = math.floor(x)
    return whole + 1 if x - whole > 0.5 else whole


def _chain(mods: "Iterable[int]") -> int:
    """Chain 4096-based modifiers the way the engine does (rounding after each one)."""
    total = 4096
    for mod in mods:
        if mod != 4096:
            total = (total * mod + 2048) >> 12
    return total
# -ate abilities: Normal moves become this type and get x1.2 (Oct 7: Pixilate Hyper Voice was scored as a Normal move,
# 14% into Iron Hands; it is a super-effective Fairy hit that took ~55%).
ATE_ABILITIES = {"pixilate": "Fairy", "aerilate": "Flying", "refrigerate": "Ice", "galvanize": "Electric"}
ATE_BOOST = 4915 / 4096
# Type-resist berries: halve one super-effective hit of their type (Chilan: any Normal hit), then they are eaten.
RESIST_BERRIES = {
    "occaberry": "fire", "passhoberry": "water", "wacanberry": "electric", "rindoberry": "grass", "yacheberry": "ice",
    "chopleberry": "fighting", "kebiaberry": "poison", "shucaberry": "ground", "cobaberry": "flying", "payapaberry": "psychic",
    "tangaberry": "bug", "chartiberry": "rock", "kasibberry": "ghost", "habanberry": "dragon", "colburberry": "dark",
    "babiriberry": "steel", "roseliberry": "fairy", "chilanberry": "normal",
}


@dataclass
class FieldState:
    """What is on the field that changes damage: weather, terrain, screens, who is active (Ruin abilities),
    and how many of each side have fainted (Last Respects). Built once per turn by ``field_from_obs``."""

    weather: str | None = None            # "rain" | "sun" | "sand" | "snow" | None
    terrain: str | None = None            # "electric" | "psychic" | "grassy" | "misty" | None
    trick_room: bool = False
    my_side: set[str] = field(default_factory=set)      # ids: reflect, lightscreen, auroraveil, tailwind, ...
    their_side: set[str] = field(default_factory=set)
    actives: list["Mon"] = field(default_factory=list)   # everyone on the field
    fainted: dict[str, int] = field(default_factory=dict)  # "mine"/"theirs" -> fainted count
    party_base_atk: dict[str, list[int]] = field(default_factory=dict)  # healthy party members' base Atk (Beat Up)

    def side_conditions(self, side: str) -> set[str]:
        return self.my_side if side == "mine" else self.their_side


def normalize_weather(text: object) -> str | None:
    w = data.to_id(text)
    if not w or w == "none":
        return None
    if "rain" in w:
        return "rain"
    if "sun" in w or "drought" in w or "harsh" in w:
        return "sun"
    if "sand" in w:
        return "sand"
    if "snow" in w or "hail" in w:
        return "snow"
    return None


def normalize_terrain(text: object) -> str | None:
    t = data.to_id(text)
    for name in ("electric", "psychic", "grassy", "misty"):
        if name in t and "terrain" in t or t == name:
            return name
    return None


def paradox_boosted_stat(mon: "Mon", fs: "FieldState | None") -> str | None:
    """Protosynthesis (sun or Booster Energy) / Quark Drive (Electric Terrain or Booster Energy) raise the holder's
    highest stat: 1.3x, or 1.5x speed. Booster Energy is consumed on entry, so a holder is always boosted."""
    if mon.paradox_stat:
        return mon.paradox_stat
    ability = data.to_id(mon.ability)
    if ability not in ("protosynthesis", "quarkdrive"):
        return None
    active = data.to_id(mon.item) == "boosterenergy"
    if fs is not None:
        active = active or (ability == "protosynthesis" and fs.weather == "sun") or (ability == "quarkdrive" and fs.terrain == "electric")
    if not active:
        return None
    stats = {k: mon.stats.get(k, 0) * stage_multiplier(mon.boosts.get(k, 0)) for k in ("atk", "def", "spa", "spd", "spe")}
    return max(stats, key=lambda k: (stats[k], ["atk", "def", "spa", "spd", "spe"].index(k) * -1)) if any(stats.values()) else None


def _weight(mon: "Mon") -> float:
    return data.species_weight(mon.species)


SUPREME_OVERLORD_MODS = (4096, 4506, 4915, 5325, 5734, 6144)  # x1.0 .. x1.5 for 0..5 fallen allies (@smogon/calc's table)


def effective_move(attacker: "Mon", move: dict, defender: "Mon", fs: "FieldState | None", *, fallen: int | None = None) -> dict:
    """The move as it will actually resolve: type/power/category changes from weather, terrain, HP, status, weight,
    and the Supreme Overlord count. ``fallen`` is how many allies had fainted when the attacker came in (the protocol
    log's ``fallenN``, via the memory); None reads the attacker's side's fainted count from the field instead."""
    mid = move.get("id")
    out = dict(move)
    weather = fs.weather if fs else None
    terrain = fs.terrain if fs else None
    if mid == "weatherball" and weather:
        out["type"] = {"rain": "Water", "sun": "Fire", "sand": "Rock", "snow": "Ice"}[weather]
        out["base_power"] = 100
    elif mid in ("eruption", "waterspout"):
        out["base_power"] = max(1, math.floor(150 * attacker.hp_fraction))
    elif mid == "lastrespects" and fs is not None:
        out["base_power"] = 50 + 50 * fs.fainted.get(attacker.side, 0)
    elif mid == "facade" and attacker.status in ("brn", "psn", "tox", "par"):
        out["base_power"] = 140
    elif mid in data.WEIGHT_MOVES or mid in data.WEIGHT_RATIO_MOVES:
        # Low Kick / Grass Knot from the target's weight, Heavy Slam / Heat Crash from the user:target ratio (the move
        # table says 0, which every "no base power" gate read as "no damage": Tyranitar's Low Kick had no row, Oct 8).
        out["base_power"] = data.weight_power(mid, _weight(defender), _weight(attacker))
    elif mid == "expandingforce" and terrain == "psychic" and attacker.grounded:
        out["base_power"] = 120
    elif mid == "beatup" and fs is not None:
        hits = fs.party_base_atk.get(attacker.side) or []
        if hits:
            out["beatup_hits"] = [math.floor(b / 10) + 5 for b in hits]
    if mid == "grassyglide" and terrain == "grassy" and attacker.grounded:
        out["priority"] = 1
    ate = ATE_ABILITIES.get(data.to_id(attacker.ability))
    if ate and data.to_id(out.get("type")) == "normal" and mid not in ("struggle", "weatherball", "judgment", "naturalgift", "technoblast", "multiattack", "terrainpulse"):
        out["type"] = ate
        out["ate_boost"] = True
    if data.to_id(attacker.ability) == "supremeoverlord":
        # +10% power per ally that had fainted when Kingambit came in, up to +50% (Oct 8 review: Kowtow Cleave into
        # Cresselia at fallen 3 is 77-91%, the sheet said 61-71%).
        count = fallen if fallen is not None else (fs.fainted.get(attacker.side, 0) if fs is not None else 0)
        if count:
            out["supreme_overlord"] = min(5, int(count))
    return out


def _field_arg(weather: str | None, field: "FieldState | None") -> "FieldState | None":
    if field is not None:
        return field
    return FieldState(weather=normalize_weather(weather)) if weather is not None else None


def _strike(attacker: Mon, move: dict, defender: Mon, fs: "FieldState | None", *, spread: bool = False,
            fallen: int | None = None) -> tuple[float, float, dict] | None:
    """One strike of ``move`` into ``defender`` as (min%, max%, ctx) of the defender's CURRENT HP: before the hit count
    and the Focus Sash cap, which ``damage_percent`` / ``hit_profile`` add on top. ``ctx`` carries what they need:
    ``fixed`` (Ruination: half the HP, no rolls), ``beatup`` (every party member's hit is already summed), ``stamina``
    (Defense rises after every strike), ``sash`` (an intact Focus Sash at full HP) and ``def_stage``. None when the
    move does no damage; an immune or blocked hit is (0.0, 0.0, ctx)."""
    move = effective_move(attacker, move, defender, fs, fallen=fallen)
    mid = move.get("id")
    ctx = {"fixed": False, "beatup": False, "stamina": False, "sash": False, "def_stage": defender.boosts.get("def", 0), "later": (0.0, 0.0)}
    if mid == "ruination":
        ctx["fixed"] = True
        return (0.0, 0.0, ctx) if data.effectiveness("dark", defender.types) == 0 else (50.0, 50.0, ctx)
    power = move.get("base_power") or 0
    hits_list = move.get("beatup_hits")
    if power <= 0 and not hits_list:
        return None
    category = move.get("category")
    if category not in ("Physical", "Special"):
        return None
    move_type = data.to_id(move.get("type"))
    attacker_ability = data.to_id(attacker.ability)
    defender_ability = data.to_id(defender.ability)
    defender_types = defender.types
    if attacker_ability in ("scrappy", "mindseye") and move_type in ("normal", "fighting"):
        defender_types = [t for t in defender.types if t != "ghost"]  # these abilities hit Ghosts with Normal/Fighting moves
    type_mult = data.effectiveness(move.get("type") or "", defender_types)
    if mid == "freezedry" and "water" in defender_types:
        type_mult *= 4  # Freeze-Dry is super effective on Water (Ice would be resisted): x0.5 becomes x2
    if type_mult == 0:
        return (0.0, 0.0, ctx)
    if move_type == "ground" and not defender.grounded and mid != "thousandarrows":
        return (0.0, 0.0, ctx)  # Levitate / Air Balloon (Oct 7: Earthquake showed 91-107% into an Air Balloon Gholdengo)
    crit = bool(move.get("will_crit"))
    weather_now = fs.weather if fs else None
    terrain = fs.terrain if fs else None
    if (move.get("priority") or 0) > 0 and (defender_ability in ("armortail", "dazzling", "queenlymajesty")
                                             or (terrain == "psychic" and defender.grounded)):
        return (0.0, 0.0, ctx)  # priority attacks cannot target this Pokémon

    # From here on the engine's order of operations and rounding (4096 = x1; halves round down), so that ONE strike
    # of a multi-hit move is exact: a 2 HP error per strike is 10 HP over five, and that decides "KO on five hits".
    item = data.to_id(attacker.item)
    defender_item = data.to_id(defender.item)
    seed_used = defender_item.endswith("seed") and terrain and defender_item.startswith(terrain) and defender.grounded
    knock_bonus = mid == "knockoff" and defender_item and defender_item not in ("rustedshield", "rustedsword", "boosterenergy") and not seed_used

    def powered(bp: float) -> int:
        """Base power after its own modifiers: Technician, terrain, Supreme Overlord, -ate, Knock Off, type-boost items."""
        mods: list[int] = []
        if attacker_ability == "technician" and bp <= 60:
            mods.append(6144)
        if move.get("supreme_overlord"):
            mods.append(SUPREME_OVERLORD_MODS[move["supreme_overlord"]])
        if terrain == "electric" and move_type == "electric" and attacker.grounded:
            mods.append(5325)
        if terrain == "psychic" and move_type == "psychic" and attacker.grounded:
            mods.append(5325)
        if terrain == "grassy" and move_type == "grass" and attacker.grounded:
            mods.append(5325)
        if terrain == "grassy" and mid in ("earthquake", "bulldoze", "magnitude") and defender.grounded:
            mods.append(2048)
        if terrain == "misty" and move_type == "dragon" and defender.grounded:
            mods.append(2048)
        if mid in ("collisioncourse", "electrodrift") and type_mult > 1:
            mods.append(5461)
        if move.get("ate_boost"):
            mods.append(4915)
        if knock_bonus:
            mods.append(6144)  # Knock Off hits harder when the target holds an item it can lose
        if TYPE_BOOST_ITEMS.get(item) == move_type:
            mods.append(4915)
        return max(1, _poke_round(bp * _chain(mods) / 4096))

    # Stats: crits ignore the attacker's drops and the defender's raises; Body Press / Foul Play borrow stats.
    atk_stat, def_stat = ("atk", "def") if category == "Physical" else ("spa", "spd")
    # Unaware (Dondozo): a defender with it ignores the attacker's stat stages; an attacker with it ignores the defender's.
    a_lo, a_hi = (0, 0) if defender_ability == "unaware" else ((0 if crit else None), None)
    if mid == "bodypress":
        a = attacker.stat("def", min_stage=a_lo, max_stage=a_hi)
    elif mid == "foulplay":
        a = defender.stat("atk", min_stage=0 if crit else None)
    elif mid in ("meteorbeam", "electroshot"):
        # The charge turn raises Special Attack by one stage before the hit lands (Power Herb / rain: same turn).
        boosted = Mon(**{**attacker.__dict__, "boosts": {**attacker.boosts, "spa": min(6, attacker.boosts.get("spa", 0) + 1)}})
        a = boosted.stat("spa", min_stage=a_lo, max_stage=a_hi)
    else:
        a = attacker.stat(atk_stat, min_stage=a_lo, max_stage=a_hi)
    if mid in ("sacredsword", "chipaway", "darkestlariat") or attacker_ability == "unaware":
        d = defender.stat(def_stat, min_stage=0, max_stage=0)  # ignores the target's stat stages
    else:
        d = defender.stat(def_stat, max_stage=0 if crit else None)
    if not a or not d or not defender.stats.get("hp"):
        return None
    if fs is not None:
        applied: set[str] = set()  # the Ruin abilities do not stack: two Sword of Ruin holders still lower Def once
        for other in fs.actives:
            ability = data.to_id(other.ability)
            ruin = RUIN_ABILITIES.get(ability)
            if not ruin or ruin[1] != category or ability in applied:
                continue
            stat_name, _ = ruin
            if stat_name in ("def", "spd") and other is not defender and defender_ability != ability:
                d *= 0.75
                applied.add(ability)
            if stat_name in ("atk", "spa") and other is not attacker and attacker_ability != ability:
                a *= 0.75
                applied.add(ability)
    if attacker_ability == "hadronengine" and terrain == "electric" and category == "Special":
        a *= PARADOX_BOOST
    if attacker_ability == "orichalcumpulse" and weather_now == "sun" and category == "Physical":
        a *= PARADOX_BOOST
    if weather_now == "sand" and "rock" in defender.types and category == "Special":
        d *= 1.5
    if weather_now == "snow" and "ice" in defender.types and category == "Physical":
        d *= 1.5
    a_i, d_i = max(1, _poke_round(a)), max(1, _poke_round(d))

    weather_mod = 4096
    if weather_now == "rain" and move_type == "water" or weather_now == "sun" and move_type == "fire":
        weather_mod = 6144
    if weather_now == "rain" and move_type == "fire" or weather_now == "sun" and move_type == "water":
        weather_mod = 2048
    stab = 4096
    if move_type in attacker.types:
        stab = 8192 if attacker_ability == "adaptability" else 6144
    burned = attacker.status == "brn" and category == "Physical" and attacker_ability != "guts" and mid != "facade"
    full_hp = defender.hp_fraction >= 0.999
    # Final modifiers in the engine's order: screens, then what only the FIRST strike of a multi-hit move meets
    # (Multiscale / Shadow Shield break on it; a resist berry is eaten by it), then the attacker's items.
    screen_mods: list[int] = []
    if not crit:
        screens = fs.side_conditions(defender.side) if fs else set()
        if "auroraveil" in screens or (category == "Physical" and "reflect" in screens) or (category == "Special" and "lightscreen" in screens):
            screen_mods.append(2732)  # doubles: x2/3
    first_only: list[int] = []
    if full_hp and defender_ability in ("multiscale", "shadowshield"):
        first_only.append(2048)
    berry = RESIST_BERRIES.get(defender_item)
    if berry == move_type and (type_mult > 1 or berry == "normal"):
        first_only.append(2048)  # its resist berry takes the first such hit (Tyranitar's Chople Berry vs Fighting)
    item_mods: list[int] = []
    if item == "expertbelt" and type_mult > 1:
        item_mods.append(4915)
    if item == "lifeorb":
        item_mods.append(5324)
    final_first = _chain(screen_mods + first_only + item_mods)
    final_rest = _chain(screen_mods + item_mods)
    current_hp = max(1.0, defender.stats["hp"] * defender.hp_fraction)

    def strike(bp: float, roll: int, final: int) -> int:
        """One strike at one random roll (85..100 percent): base -> spread -> weather -> crit -> roll -> STAB -> type
        -> burn -> the final modifiers, rounded at every step like the engine."""
        base = math.floor(math.floor(math.floor(2 * LEVEL / 5 + 2) * powered(bp) * a_i / d_i) / 50) + 2
        if spread:
            base = _poke_round(base * 3072 / 4096)
        if weather_mod != 4096:
            base = _poke_round(base * weather_mod / 4096)
        if crit:
            base = math.floor(base * 1.5)  # always a critical hit (Surging Strikes, Wicked Blow, Flower Trick)
        dmg = math.floor(base * roll / 100)
        if stab != 4096:
            dmg = _poke_round(dmg * stab / 4096)
        dmg = math.floor(dmg * type_mult)
        if burned:
            dmg = math.floor(dmg / 2)
        return max(1, _poke_round(dmg * final / 4096))

    if hits_list:
        finals = [final_first] + [final_rest] * (len(hits_list) - 1)
        lo, hi = sum(strike(bp, 85, f) for bp, f in zip(hits_list, finals)), sum(strike(bp, 100, f) for bp, f in zip(hits_list, finals))
        later = (lo, hi)
        ctx["beatup"] = True
    else:
        lo, hi = strike(power, 85, final_first), strike(power, 100, final_first)
        later = (strike(power, 85, final_rest), strike(power, 100, final_rest)) if final_rest != final_first else (lo, hi)
    ctx["stamina"] = defender_ability == "stamina" and category == "Physical" and not crit
    ctx["sash"] = full_hp and defender_item == "focussash"
    ctx["later"] = (later[0] / current_hp * 100, later[1] / current_hp * 100)  # strikes after the first (Multiscale gone, berry eaten)
    return (lo / current_hp * 100, hi / current_hp * 100, ctx)


def _total(first_lo: float, first_hi: float, ctx: dict, eff_hits: float) -> tuple[float, float]:
    """Damage of ``eff_hits`` strikes: the first at its own range, the rest at the later-strike range (Multiscale and a
    resist berry go with the first strike; Stamina's rising Defense is already folded into ``eff_hits``)."""
    if eff_hits <= 1:
        return first_lo * eff_hits, first_hi * eff_hits
    later_lo, later_hi = ctx["later"]
    return first_lo + later_lo * (eff_hits - 1), first_hi + later_hi * (eff_hits - 1)


def _stamina_hits(hits: float, ctx: dict) -> float:
    """Stamina raises Defense after every strike, so strikes 2..n land on +1, +2, ...: scale the hit count down."""
    if hits > 1 and ctx["stamina"]:
        stage = ctx["def_stage"]
        return sum(stage_multiplier(stage) / stage_multiplier(min(6, stage + k)) for k in range(int(round(hits))))
    return hits


def _clamp(low: float, high: float) -> tuple[float, float]:
    return (round(min(low, 999.0), 1), round(min(high, 999.0), 1))  # anything past a KO is just "a KO"


def damage_percent(attacker: Mon, move: dict, defender: Mon, *, weather: str | None = None, spread: bool = False,
                   field: FieldState | None = None, fallen: int | None = None) -> tuple[float, float] | None:
    """(min%, max%) of the defender's CURRENT HP, or None when the move does no damage / is immune. Multi-hit moves
    are at their EXPECTED hit count (the mean); ``hit_profile`` gives the fewest-to-most-strikes range and the KO odds.

    Models what the drafted sets make knowable: stats, stages, STAB, type chart, items, weather, terrain, screens,
    Technician, the Ruin abilities, Guts, Protosynthesis/Quark Drive, Hadron Engine/Orichalcum Pulse, Supreme
    Overlord (``fallen`` allies, else the field's fainted count), always-crit moves, multi-hit, Multiscale, Focus Sash,
    and the variable-power moves in the pool (weight-based ones included). Checked against @smogon/calc in
    tests/test_damage_vs_smogon*.py and tests/test_mech_d.py."""
    strike = _strike(attacker, move, defender, _field_arg(weather, field), spread=spread, fallen=fallen)
    if strike is None:
        return None
    low, high, ctx = strike
    if ctx["fixed"]:
        return (low, high)
    hits = 1.0 if ctx["beatup"] else _stamina_hits(data.expected_hits(move, item=attacker.item, ability=attacker.ability), ctx)
    low, high = _total(low, high, ctx, hits)
    if ctx["sash"] and hits <= 1 and not ctx["beatup"]:
        # The sash leaves the holder at 1 HP from a single hit at full health: no KO this turn.
        low, high = min(low, 99.0), min(high, 99.0)
    return _clamp(low, high)


def damage_per_hit(attacker: Mon, move: dict, defender: Mon, *, weather: str | None = None, spread: bool = False,
                   field: FieldState | None = None, fallen: int | None = None) -> tuple[float, float] | None:
    """ONE strike's (min%, max%) of the defender's CURRENT HP, with no Focus Sash cap (a multi-hit breaks the sash):
    the FIRST strike (Multiscale and a resist berry act on that one; ``hit_profile`` reports the later strikes as
    ``per_hit``). None when the move does no damage. Pair it with ``data.hit_count_distribution`` for the hit counts."""
    strike = _strike(attacker, move, defender, _field_arg(weather, field), spread=spread, fallen=fallen)
    if strike is None:
        return None
    low, high, _ = strike
    return _clamp(low, high)


def _ko_label(low: float, high: float) -> str:
    return "guaranteed" if low >= 100 else ("possible" if high >= 100 else "no")


def hit_profile(attacker: Mon, move: dict, defender: Mon, *, weather: str | None = None, spread: bool = False,
                field: FieldState | None = None, fallen: int | None = None) -> dict | None:
    """What the sheet says about one attack into one target. ``damage_pct_of_current_hp`` runs from the FEWEST strikes
    at the minimum roll to the MOST at the maximum (a Loaded Dice Scale Shot shows 4 hits low to 5 hits high, not the
    4.5-hit mean that printed "no KO" for a KO the fifth strike delivers half the time); ``ko`` is the label on that
    range. Multi-hit rows also carry ``per_hit`` (one strike), ``hits`` ({count: probability}) and ``ko_chance`` (each
    strike rolls on its own). ``hit_chance`` is the move's chance to land when it is below 1. None when the move does
    no damage."""
    fs = _field_arg(weather, field)
    strike = _strike(attacker, move, defender, fs, spread=spread, fallen=fallen)
    if strike is None:
        return None
    low1, high1, ctx = strike
    if ctx["fixed"] or ctx["beatup"]:
        dist: dict[int, float] = {1: 1.0}
        low, high = low1, high1
    else:
        dist = data.hit_count_distribution(move, attacker)
        low = _total(low1, high1, ctx, _stamina_hits(float(min(dist)), ctx))[0]
        high = _total(low1, high1, ctx, _stamina_hits(float(max(dist)), ctx))[1]
    if ctx["sash"] and max(dist) <= 1 and not ctx["beatup"]:
        low, high = min(low, 99.0), min(high, 99.0)
    low, high = _clamp(low, high)
    out: dict[str, Any] = {"damage_pct_of_current_hp": [low, high], "ko": _ko_label(low, high)}
    acc = data.hit_chance(move, fs.weather if fs else None)
    if acc < 1.0:
        out["hit_chance"] = round(acc, 3)
    if max(dist) > 1:
        later = _clamp(*ctx["later"])
        first = _clamp(low1, high1)
        out["per_hit"] = list(later)  # a strike after the first; ko_probability() folds these into the KO odds
        if first != later:
            out["first_hit"] = list(first)  # Multiscale / Shadow Shield or a resist berry: the first strike is the halved one
        out["hits"] = {int(n): p for n, p in dist.items()}
    return out


def field_from_obs(obs: dict, ours: dict[int, "Mon"], theirs: dict[int, "Mon"], memory: "MatchMemory | None" = None) -> FieldState:
    """Read weather, terrain, screens and fainted counts out of the server's observation."""
    fs = FieldState()
    fs.weather = normalize_weather(obs.get("weather"))  # the live server sends a dict ({} or {"SUNNYDAY": turns}); strings also work
    field_blob = json.dumps([obs.get("fields"), obs.get("field")], default=str)
    fs.terrain = normalize_terrain(field_blob)
    fs.trick_room = "trickroom" in data.to_id(field_blob)

    def conditions(raw: Any) -> set[str]:
        out: set[str] = set()
        if isinstance(raw, dict):
            raw = [k for k, v in raw.items() if v] or list(raw)
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                out.add(data.to_id(item if not isinstance(item, dict) else (item.get("id") or item.get("name") or json.dumps(item))))
        elif isinstance(raw, str):
            out.add(data.to_id(raw))
        return out

    fs.my_side = conditions(obs.get("side_conditions"))
    fs.their_side = conditions(obs.get("opponent_side_conditions"))
    fs.actives = list(ours.values()) + list(theirs.values())

    def fainted_count(team: Any) -> int:
        items = team.values() if isinstance(team, dict) else (team if isinstance(team, list) else [])
        return sum(1 for s in items if isinstance(s, dict) and s.get("fainted"))

    fs.fainted = {"mine": fainted_count(obs.get("team")), "theirs": fainted_count(obs.get("opponent_team"))}

    def party_atk(team: Any, cards: dict) -> list[int]:
        items = team.values() if isinstance(team, dict) else (team if isinstance(team, list) else [])
        out = []
        for s in items:
            if not isinstance(s, dict) or s.get("fainted") or s.get("status"):
                continue
            card = cards.get(species_key(s.get("species"))) if cards else None
            base = data.resolve_base_stats({**(card or {}), **{k: v for k, v in s.items() if v not in (None, "", [], {})}}) or {}
            if base.get("atk"):
                out.append(int(base["atk"]))
        return out

    fs.party_base_atk = {"mine": party_atk(obs.get("team"), memory.my_cards if memory else {}),
                         "theirs": party_atk(obs.get("opponent_team"), memory.opp_cards if memory else {})}
    for mon in fs.actives:
        mon.boosted_stat = paradox_boosted_stat(mon, fs)
    return fs


# -- the sheet -----------------------------------------------------------------------------------


@dataclass
class TurnSheet:
    speed_order: list[dict] = field(default_factory=list)
    trick_room: bool = False
    our_options: list[dict] = field(default_factory=list)  # per slot: option index -> damage per target
    threats: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # lethal threats that act before our slot can
    choice_locks: dict = field(default_factory=dict)  # slot -> {move_id: [opponents it does nothing to]}
    bench_hp: dict = field(default_factory=dict)  # species_key -> hp fraction of our benched Pokémon (for switch-in checks)
    flinch_risk: dict = field(default_factory=dict)  # our slot -> chance a fresh opposing Fake Out flinches it this turn
    focus_threats: list = field(default_factory=list)  # our slots that BOTH foes together can KO this turn (double-targeting)
    alone: bool = False  # our last Pokémon stands alone (the other slot can only pass)
    switch_ins: dict = field(default_factory=dict)  # bench species_key -> what it takes / deals / outspeeds if it comes in now
    stall_reasons: list = field(default_factory=list)  # what a Protect would actually wait out (their Tailwind, screens, residual...)
    field: "FieldState | None" = None  # keep last: this name shadows dataclasses.field for anything declared after it

    def as_prompt(self) -> dict:
        return {
            "WARNINGS": self.warnings,
            "speed_order": self.speed_order,
            "trick_room": self.trick_room,
            "our_damage_estimates": self.our_options,
            "opponent_threats": self.threats,
            "candidate_turns": self.candidates,
            "notes": self.notes,
            **({"switch_ins": {k: {kk: vv for kk, vv in v.items() if kk != "score"} for k, v in self.switch_ins.items()}} if self.switch_ins else {}),
        }


def type_multiplier(attacker: "Mon", move: dict, defender: "Mon", fs: "FieldState | None") -> float:
    """The type effectiveness this move will actually have (after -ate abilities, Weather Ball, Scrappy, Freeze-Dry,
    and Ground into Levitate/Air Balloon): what the model needs to stop calling neutral hits 'resisted' (Oct 7)."""
    eff = effective_move(attacker, move, defender, fs)
    move_type = data.to_id(eff.get("type"))
    types = defender.types
    if data.to_id(attacker.ability) in ("scrappy", "mindseye") and move_type in ("normal", "fighting"):
        types = [t for t in types if t != "ghost"]
    mult = data.effectiveness(eff.get("type") or "", types)
    if eff.get("id") == "freezedry" and "water" in types:
        mult *= 4
    if move_type == "ground" and not defender.grounded and eff.get("id") != "thousandarrows":
        mult = 0.0
    return mult


def effect_label(mult: float) -> str:
    return {0.0: "x0 immune", 0.25: "x0.25 resisted", 0.5: "x0.5 resisted", 1.0: "x1", 2.0: "x2 super effective", 4.0: "x4 super effective"}.get(mult, f"x{mult:g}")


PROFILE_ROW_KEYS = ("per_hit", "first_hit", "hits", "hit_chance")  # what a hit_profile adds to a row beyond the range and label


def _row(target: int, defender: "Mon", est: tuple[float, float] | list, mult: float | None = None, profile: dict | None = None) -> dict:
    row = {"target": target, "species": defender.species, "side": defender.side, "damage_pct_of_current_hp": list(est),
           "ko": _ko_label(est[0], est[1]), "hp_frac": round(defender.hp_fraction, 3)}
    if mult is not None:
        row["effect"] = effect_label(mult)
    if profile:
        row.update({k: profile[k] for k in PROFILE_ROW_KEYS if k in profile})
    return row


def _find_summary(team: Any, species: str) -> dict:
    if isinstance(team, dict):
        for summary in team.values():
            if isinstance(summary, dict) and species_key(summary.get("species")) == species_key(species):
                return summary
    elif isinstance(team, list):
        for summary in team:
            if isinstance(summary, dict) and species_key(summary.get("species")) == species_key(species):
                return summary
    return {}


def _active_summaries(team: Any) -> list[dict]:
    items = team.values() if isinstance(team, dict) else (team if isinstance(team, list) else [])
    return [s for s in items if isinstance(s, dict) and s.get("active") and not s.get("fainted")]


DURATIONS = {"tailwind": 4, "trickroom": 5, "reflect": 5, "lightscreen": 5, "auroraveil": 5}


def turns_left(conditions: Any, name: str, turn: Any, *, extended: bool = False) -> int | None:
    """Turns a timed condition still covers, counting this one. The live server reports the turn it STARTED
    ({"TAILWIND": 4} on turn 6 = this turn and next); Light Clay stretches screens to 8. None when unknown."""
    if not isinstance(conditions, dict) or not isinstance(turn, int):
        return None
    for key, started in conditions.items():
        if data.to_id(key) == name and isinstance(started, int) and started > 0:
            duration = 8 if extended and name in ("reflect", "lightscreen", "auroraveil") else DURATIONS.get(name)
            if duration:
                return max(0, started + duration - turn)
    return None


def _left_text(n: int | None) -> str:
    if n is None:
        return ""
    return " Last turn of it." if n <= 1 else f" {n} turns left, counting this one."


def build_sheet(template: dict, obs: dict, memory: MatchMemory, params: dict | None = None) -> TurnSheet:
    P = params or DEFAULTS
    sheet = TurnSheet()
    slots = sorted(template.get("slots") or [], key=lambda s: s.get("slot", 0))
    weather = normalize_weather(obs.get("weather"))  # dict from the live server, string from the simulator, None when clear
    field_text = data.to_id(json.dumps([obs.get("fields"), obs.get("field")], default=str))
    turn = obs.get("turn")
    sheet.trick_room = "trickroom" in field_text
    if sheet.trick_room:
        sheet.notes.append("Trick Room is up: slower Pokémon move first." + _left_text(turns_left(obs.get("fields"), "trickroom", turn)))
    my_tailwind = "tailwind" in data.to_id(json.dumps(obs.get("side_conditions"), default=str))
    their_tailwind = "tailwind" in data.to_id(json.dumps(obs.get("opponent_side_conditions"), default=str))
    if my_tailwind:
        sheet.notes.append("Our Tailwind is up (speed doubled)." + _left_text(turns_left(obs.get("side_conditions"), "tailwind", turn)))
    if their_tailwind:
        sheet.notes.append("Opponent's Tailwind is up (their speed doubled)." + _left_text(turns_left(obs.get("opponent_side_conditions"), "tailwind", turn)))
    # Our active Pokémon, one per slot.
    ours: dict[int, Mon] = {}
    for slot in slots:
        species = slot.get("active")
        if isinstance(species, dict):
            species = species.get("species")
        if not species:
            continue
        key = species_key(species)
        summary = _find_summary(obs.get("team"), species)
        if summary.get("fainted") or (summary.get("current_hp_fraction") == 0 and "current_hp_fraction" in summary):
            continue  # a fainted Pokémon can still be named in its slot (the simulator does this): it is not an ally
        mon = build_mon(summary, memory.my_cards.get(key), side="mine", position=slot.get("slot"))
        if mon:
            mon.tailwind = my_tailwind
            mon.paradox_stat = memory.paradox_active.get("mine:" + key)
            mon.unburden = "mine:" + key in memory.unburden_active
            ours[slot.get("slot", 0)] = mon

    # Their active Pokémon by board position (from target_options when present).
    theirs: dict[int, Mon] = {}
    for slot in slots:
        for option in slot.get("options") or []:
            for t in option.get("target_options") or []:
                if t.get("side") == "opponent" and t.get("species") and t.get("target") not in theirs:
                    key = species_key(t["species"])
                    mon = build_mon(_find_summary(obs.get("opponent_team"), t["species"]), memory.opp_cards.get(key), side="theirs",
                                    position=t.get("target"), item_lost=key in memory.opp_items_lost)
                    if mon:
                        mon.tailwind = their_tailwind
                        mon.paradox_stat = memory.paradox_active.get("theirs:" + key)
                        mon.unburden = "theirs:" + key in memory.unburden_active
                        theirs[int(t["target"])] = mon
    if not theirs:
        for index, summary in enumerate(_active_summaries(obs.get("opponent_team"))[:2], start=1):
            key = species_key(summary.get("species"))
            mon = build_mon(summary, memory.opp_cards.get(key), side="theirs", position=index, item_lost=key in memory.opp_items_lost)
            if mon:
                mon.tailwind = their_tailwind
                mon.paradox_stat = memory.paradox_active.get("theirs:" + key)
                theirs[index] = mon

    fs = field_from_obs(obs, ours, theirs, memory)
    sheet.field = fs
    if fs.weather:
        sheet.notes.append(f"Weather: {fs.weather}" + (" (Water moves x1.5, Fire x0.5)." if fs.weather == "rain" else " (Fire moves x1.5, Water x0.5)." if fs.weather == "sun" else "."))
    if fs.terrain:
        sheet.notes.append(f"{fs.terrain.capitalize()} Terrain is up" + {
            "electric": " (grounded Electric moves x1.3; Hadron Engine users hit harder).",
            "psychic": " (grounded Psychic moves x1.3; Expanding Force becomes a 120-power spread move; priority moves cannot target grounded Pokémon).",
            "grassy": " (grounded Grass moves x1.3, Earthquake/Bulldoze halved, Grassy Glide gets +1 priority).",
            "misty": " (Dragon moves into grounded targets halved; no new status).",
        }.get(fs.terrain, "."))
    for side_name, conds in (("Our", fs.my_side), ("Their", fs.their_side)):
        screens = [c for c in ("reflect", "lightscreen", "auroraveil") if c in conds]
        if screens:
            sheet.notes.append(f"{side_name} side has {', '.join(screens)} up (damage into that side x2/3; the numbers below include it).")
    for mon in fs.actives:
        if mon.boosted_stat:
            sheet.notes.append(f"{'Our' if mon.side == 'mine' else 'Their'} {mon.species}'s {data.to_id(mon.ability)} is boosting its {mon.boosted_stat} (included in the numbers).")

    # Speed order.
    everyone = list(ours.values()) + list(theirs.values())
    speeds = [(m, m.stat("spe")) for m in everyone if m.stats.get("spe")]
    speeds.sort(key=lambda pair: pair[1], reverse=not sheet.trick_room)
    sheet.speed_order = [
        {"species": m.species, "side": m.side, "position": m.position, "speed": int(s)} for m, s in speeds
    ]

    for mon in list(ours.values()) + list(theirs.values()):
        if data.to_id(mon.ability) == "prankster":
            sheet.notes.append(f"{'Our' if mon.side == 'mine' else 'Their'} {mon.species} has Prankster: its status moves (Tailwind, "
                               "Encore, Taunt, screens, Thunder Wave...) go first, +1 priority, but fail on Dark-type targets.")
        if mon.unburden and data.to_id(mon.ability) == "unburden":
            sheet.notes.append(f"{'Our' if mon.side == 'mine' else 'Their'} {mon.species}'s Unburden is active (item used up): speed doubled (in the speed order).")
    for number, me in sorted(ours.items()):
        if data.to_id(me.item) == "focussash":
            sheet.notes.append(f"Our {me.species} holds Focus Sash" + (" (intact at full HP: survives ONE single hit, not multi-hit or two hits)." if me.hp_fraction >= 0.999 else " but is not at full HP: the sash will not save it."))
    for pos, opp in sorted(theirs.items()):
        if data.to_id(opp.item) == "focussash" and opp.hp_fraction >= 0.999:
            sheet.notes.append(f"Their {opp.species} holds an intact Focus Sash: it needs two hits or a multi-hit move to KO this turn.")

    for number, me in sorted(ours.items()):
        streak = memory.protect_streak(me.species, number)
        if streak:
            odds = 3 ** streak
            sheet.notes.append(f"Our {me.species} used Protect {'LAST turn' if streak == 1 else f'{streak} turns in a row'}: another one works only 1 time in {odds}. "
                               "Switch it out or attack instead of protecting again.")
    for key in memory.opp_protected_last_turn:
        mon = next((o for o in theirs.values() if species_key(o.species) == key), None)
        name = mon.species if mon is not None else key
        if mon is not None and not any(m["id"] in PROTECT_LIKE for m in mon.moves):
            sheet.notes.append(f"Their {name} took no damage from the attack we aimed at it last turn, and its set has no Protect: "
                               "a flinch, a miss or an immunity, not a Protect.")
            continue
        sheet.notes.append(f"Their {name} probably Protected last turn (we aimed at it and its HP did not move): a second Protect in a row fails 2/3 of the time, so this is the turn to attack it.")

    # Choice items lock the user into its first move until it switches: warn when a move we might pick
    # would be useless against something the opponent still has (immunity / heavy resistance).
    bench_opp = []
    opp_team = obs.get("opponent_team")
    for summary in (opp_team.values() if isinstance(opp_team, dict) else []):
        if isinstance(summary, dict) and not summary.get("active") and not summary.get("fainted"):
            bkey = species_key(summary.get("species"))
            bm = build_mon(summary, memory.opp_cards.get(bkey), side="theirs", position=None, item_lost=bkey in memory.opp_items_lost)
            if bm:
                bench_opp.append(bm)
    choice_targets = list(theirs.values()) + bench_opp
    for number, me in sorted(ours.items()):
        if data.to_id(me.item) not in ("choicescarf", "choiceband", "choicespecs"):
            continue
        slot = next((sl for sl in slots if sl.get("slot", 0) == number), None)
        if not slot:
            continue
        for option in slot.get("options") or []:
            if option.get("type") != "move":
                continue
            info = data.move_info(option.get("move_id")) or {}
            if not data.is_damaging(info):
                continue
            dead_against = [t.species for t in choice_targets if t.types and data.effectiveness(info.get("type") or "", t.types) <= 0.25]
            if dead_against:
                mid = data.to_id(option.get("move_id"))
                sheet.notes.append(f"Our {me.species} holds a Choice item: picking {mid} LOCKS it into that move, which does nothing to their {dead_against}. Prefer a move that still works next turn, or plan the switch.")
                sheet.choice_locks.setdefault(number, {})[mid] = dead_against

    # Our damage estimates per slot option.
    for slot in slots:
        number = slot.get("slot", 0)
        me = ours.get(number)
        entry = {"slot": number, "active": me.species if me else slot.get("active"), "options": []}
        for index, option in enumerate(slot.get("options") or []):
            if option.get("type") != "move" or not me:
                continue
            move = data.move_info(option.get("move_id")) or {}
            move = {**move, "id": data.to_id(option.get("move_id")), "type": move.get("type") or option.get("move_type"),
                    "category": move.get("category") or str(option.get("category") or "").capitalize() or None,
                    "base_power": move.get("base_power") or option.get("base_power") or 0}
            spread = is_spread(move, foes=len(theirs), allies=len(ours) - 1)  # Earthquake with one foe and our ally is still 0.75x (batch 2)
            eff = effective_move(me, move, me, fs)
            psychic_spread = move["id"] == "expandingforce" and fs.terrain == "psychic" and me.grounded and len(theirs) > 1
            per_target = []
            note = None
            # Target 0 means "no target needed" (self, field or SPREAD move): the live server sends [0], the old simulator [].
            # Treating [0] as a target list gave every spread move an empty damage row on Oct 7 (Dazzling Gleam, Earthquake,
            # Rock Slide and Icy Wind were invisible to the ranking and to the model).
            targets = [t for t in option.get("targets") or [] if t != 0]
            fresh_now = species_key(me.species) in set(memory.fresh_active) or obs.get("turn") in (None, 1)
            priority_blocked = (eff.get("priority") or 0) > 0 and data.is_damaging(move) and (
                any(data.to_id(o.ability) in ("armortail", "dazzling", "queenlymajesty") for o in theirs.values())
                or (fs.terrain == "psychic" and any(o.grounded for o in theirs.values())))
            if move["id"] in ("fakeout", "firstimpression") and not fresh_now:
                note = f"{move['id']} FAILS: {me.species} is not on its first turn out."
                targets = []
            elif priority_blocked:
                note = f"{move['id']} is BLOCKED: their Armor Tail/Dazzling/Queenly Majesty (or Psychic Terrain) stops priority moves aimed at them."
                targets = []
            if targets:
                for target in targets:
                    defender = theirs.get(target) if target > 0 else (ours.get(-target - 1) if target < 0 else None)
                    if defender is None:
                        continue
                    prof = hit_profile(me, move, defender, field=fs, spread=spread or psychic_spread)
                    if prof is None:
                        continue
                    per_target.append(_row(target, defender, prof["damage_pct_of_current_hp"], type_multiplier(me, move, defender, fs), prof))
                if psychic_spread:
                    note = "Expanding Force in Psychic Terrain hits BOTH opponents (120 power, spread): the numbers are per target."
            elif note is None and data.is_damaging(move):
                # No target to choose: a spread move (or a fixed-target move). It hits every
                # opposing active (and, for allAdjacent moves like Earthquake, our own ally too).
                victims = list(theirs.values())
                if move.get("target") == "allAdjacent":
                    victims += [ally for n, ally in ours.items() if n != number]
                for defender in victims:
                    prof = hit_profile(me, move, defender, field=fs, spread=spread)
                    if prof is not None:
                        est = prof["damage_pct_of_current_hp"]
                        per_target.append({**_row(0, defender, est, type_multiplier(me, move, defender, fs), prof), "spread": True})
                        if defender.side == "mine" and est[1] > 0:
                            sheet.notes.append(f"{move['id']} from slot {number} also hits our own {defender.species} for {est[0]}-{est[1]}%.")
            if per_target or data.is_damaging(move):
                row = {"option": index, "move": move["id"], "priority": eff.get("priority", 0), "spread": spread or psychic_spread, "targets": per_target}
                if note:
                    row["note"] = note
                    if "FAILS" in note or "BLOCKED" in note:
                        row["fails"] = True  # never a candidate: it does nothing this turn (first live match, Oct 7: a stale Fake Out
                        # fell through to the spread branch, kept its +3 priority bonus and won "best attack" for seven turns)
                entry["options"].append(row)
        sheet.our_options.append(entry)

    # Our bench's HP, for "would the switch-in survive its entry?" checks.
    team = obs.get("team")
    if isinstance(team, dict):
        for summary in team.values():
            if isinstance(summary, dict) and summary.get("species") and not summary.get("active"):
                frac = summary.get("current_hp_fraction")
                if frac is None and summary.get("max_hp"):
                    frac = (summary.get("current_hp") or 0) / summary["max_hp"]
                sheet.bench_hp[species_key(summary["species"])] = float(frac if frac is not None else 1.0)

    # Opponent threats from their KNOWN sets.
    for position, opp in sorted(theirs.items()):
        threat = {"position": position, "species": opp.species, "hp_pct": round(opp.hp_fraction * 100), "status": opp.status,
                  "item": opp.item, "ability": opp.ability, "known_moves": [m["id"] for m in opp.moves], "hits": []}
        opp_fresh = species_key(opp.species) in set(memory.opp_fresh_active) or obs.get("turn") in (None, 1)
        # A Choice item locks its holder into the last move it used (live server: read from the protocol log).
        locked = memory.opp_last_move.get(species_key(opp.species)) if data.to_id(opp.item) in CHOICE_ITEMS else None
        if locked:
            threat["locked_move"] = locked
            threat["note"] = f"Choice-locked into {locked} until it switches out: its other moves cannot be used this turn."
        our_priority_block = any(data.to_id(m.ability) in ("armortail", "dazzling", "queenlymajesty") for m in ours.values()) \
            or (fs.terrain == "psychic" and any(m.grounded for m in ours.values()))
        for move in opp.moves:
            if locked and move["id"] != locked:
                continue  # Choice-locked: only the locked move can come out
            if not data.is_damaging(move):
                continue
            if move["id"] in ("fakeout", "firstimpression") and not opp_fresh:
                continue  # only works on the user's first turn out
            eff = effective_move(opp, move, opp, fs)
            if (eff.get("priority") or 0) > 0 and our_priority_block:
                continue  # our Armor Tail (or Psychic Terrain) blocks their priority attacks
            for number, me in sorted(ours.items()):
                prof = hit_profile(opp, move, me, field=fs, spread=is_spread(move, foes=len(ours), allies=len(theirs) - 1))
                if prof is None:
                    continue
                if me.fainted or me.hp_fraction <= 0:
                    continue
                est = prof["damage_pct_of_current_hp"]
                hit = {"move": move["id"], "into_slot": number, "species": me.species, "damage_pct_of_current_hp": list(est),
                       "ko": prof["ko"], "priority": eff.get("priority", 0), "effect": effect_label(type_multiplier(opp, move, me, fs)),
                       **{k: prof[k] for k in PROFILE_ROW_KEYS if k in prof}}
                if hit["ko"] == "possible":
                    hit["ko_chance_pct"] = int(round(100 * ko_probability(hit)))  # how often the rolls (and hit count) reach a KO, once it lands
                if max(prof.get("hits") or {1: 1.0}) > 1:
                    hit["multi_hit"] = True  # breaks Focus Sash
                threat["hits"].append(hit)
        if any(m["id"] in PROTECT_LIKE for m in opp.moves):
            streak = memory.opp_protect_streak.get(species_key(opp.species), 0)
            threat["protect"] = ("can Protect this turn" if not streak else
                                 f"Protected last turn: another works only 1 in {3 ** streak} (good turn to attack it)")
        if "fakeout" in threat["known_moves"] and not locked:
            if opp_fresh and not our_priority_block:
                threat["note"] = "FRESH: its Fake Out works THIS turn (priority +3): expect a flinch on one of our slots, so a setup move (Trick Room/Tailwind) from the slot it targets fails."
            elif opp_fresh:
                threat["note"] = "has Fake Out but our Armor Tail / Psychic Terrain blocks it."
            else:
                threat["note"] = "has Fake Out but is past its first turn out: Fake Out fails now."
        sheet.threats.append(threat)

    _glaive_rush_carry(sheet, ours, memory)
    _flinch_risk(sheet, ours, theirs, fs, slots=slots, P=P)
    _double_target_threats(sheet, ours, P)
    _endgame(sheet, slots, ours, theirs, fs)
    _switch_previews(sheet, slots, theirs, obs, memory, fs, hazards=_memory_hazards(memory, "mine"))
    sheet.candidates = rank_candidates(slots, ours, theirs, sheet, fresh=set(memory.fresh_active), memory=memory, params=P)
    return sheet


def _memory_hazards(memory: "MatchMemory | None", side: str) -> object:
    """The memory's hazard layer count for ``side`` (ops lane: ``memory.hazards(side)``); None until it exists, which
    makes the previews read the side conditions as one layer each."""
    fn = getattr(memory, "hazards", None)
    if not callable(fn):
        return None
    try:
        record = fn(side)
    except Exception:  # a memory from another version: fall back to the side conditions
        return None
    if isinstance(record, dict) and not any(record.values()):
        return None  # an all-zero record means the memory knows of nothing: the field's own side conditions decide
    return record

def _glaive_rush_carry(sheet: "TurnSheet", ours: dict[int, "Mon"], memory: "MatchMemory") -> None:
    """Glaive Rush: its user takes double damage until it next moves. A slot that used it LAST turn is still doubled
    against everything that lands before it moves this turn; relabel those threat rows (A3 / D1h)."""
    payload = getattr(memory, "last_payload", None) or {}
    actives = list(getattr(memory, "last_active", None) or [])
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    for number, me in ours.items():
        choice = payload.get(f"slot_{number}") or {}
        if choice.get("type") != "move" or data.to_id(choice.get("move_id")) != GLAIVE_RUSH:
            continue
        if len(actives) <= number or actives[number] != species_key(me.species):
            continue
        for threat in sheet.threats:
            for hit in threat["hits"]:
                if hit["into_slot"] != number:
                    continue
                before = (hit.get("priority") or 0) > 0 or rank.get(("theirs", threat["position"]), 99) < rank.get(("mine", number), 99)
                if not before:
                    continue
                lo, hi = hit["damage_pct_of_current_hp"]
                lo, hi = min(lo * 2, 999.0), min(hi * 2, 999.0)
                hit["damage_pct_of_current_hp"] = [round(lo, 1), round(hi, 1)]
                hit["ko"] = _ko_label(lo, hi)
                if hit.get("per_hit"):
                    hit["per_hit"] = [hit["per_hit"][0] * 2, hit["per_hit"][1] * 2]
                    if hit.get("first_hit"):
                        hit["first_hit"] = [hit["first_hit"][0] * 2, hit["first_hit"][1] * 2]
                hit["glaive_rush_doubled"] = True
                if hit["ko"] == "possible":
                    hit["ko_chance_pct"] = int(round(100 * ko_probability(hit)))
                elif "ko_chance_pct" in hit:
                    del hit["ko_chance_pct"]
        sheet.notes.append(f"Our {me.species} used Glaive Rush last turn: until it moves, every hit on it does DOUBLE damage (the rows above include it).")


def _avg(est: list | tuple) -> float:
    return (est[0] + est[1]) / 2


def is_spread(move: dict, *, foes: int, allies: int = 0) -> bool:
    """Does this move get the doubles 0.75x spread modifier: it targets every adjacent foe (or everyone adjacent, like
    Earthquake) and more than one Pokémon is there to be hit. ``foes`` / ``allies`` are counted from the USER's side."""
    target = move.get("target")
    if target == "allAdjacentFoes":
        return foes >= 2
    if target == "allAdjacent":
        return foes + allies >= 2
    return False


def intimidated_foes(incoming: "Mon", foes: "Iterable[Mon]") -> list["Mon"]:
    """Copies of ``foes`` at the stat stages an entering ``incoming`` with Intimidate leaves them (-1 Attack unless an
    ability or Clear Amulet blocks it; Defiant / Guard Dog end up +1, Competitive gains +2 Special Attack). The same
    objects when it has no Intimidate. Until Oct 8 Intimidate was priced only for turn-1 leads."""
    if data.to_id(incoming.ability) != "intimidate":
        return list(foes)
    out = []
    for foe in foes:
        stages = data.intimidate_stages(foe.ability, foe.item)
        if not stages:
            out.append(foe)
            continue
        boosts = dict(foe.boosts)
        for stat, delta in stages.items():
            boosts[stat] = max(-6, min(6, boosts.get(stat, 0) + delta))
        out.append(Mon(**{**foe.__dict__, "boosts": boosts}))
    return out


def intimidate_notes(incoming: "Mon", foes: "Iterable[Mon]") -> list[str]:
    """One line per foe on what the incoming Pokémon's Intimidate does to it (empty without Intimidate)."""
    if data.to_id(incoming.ability) != "intimidate":
        return []
    out = []
    for foe in foes:
        stages = data.intimidate_stages(foe.ability, foe.item)
        ability = data.to_id(foe.ability)
        if not stages:
            out.append(f"{foe.species}: unaffected ({'Clear Amulet' if data.to_id(foe.item) == 'clearamulet' else ability})")
        elif stages.get("atk", 0) > 0:
            out.append(f"{foe.species}: +{stages['atk']} Atk INSTEAD ({ability} turns Intimidate around)")
        elif stages.get("spa", 0) > 0:
            out.append(f"{foe.species}: -1 Atk but +{stages['spa']} SpA ({ability})")
        else:
            out.append(f"{foe.species}: {stages['atk']} Atk")
    return out


def entry_field(fs: "FieldState | None", *, add: "Iterable[Mon]", drop: "Iterable[Mon | None]" = ()) -> "FieldState":
    """The field as it stands once the Pokémon in ``add`` are out and those in ``drop`` have left: the same weather,
    terrain, screens and fainted counts, the actives updated, so a switch-in's own Ruin ability and the Ruin abilities
    still out price every hit (Oct 8 live match: our Wo-Chien's Tablets of Ruin cut Body Press into Ting-Lu by a
    quarter the turn it came in, its leaving gave the foes their Attack back, and their Ting-Lu's Vessel of Ruin cut
    Draco Meteor), and each newcomer's paradox boost read with the real field."""
    base = fs if fs is not None else FieldState()
    dropped = {id(m) for m in drop if m is not None}
    added = list(add)
    actives = [m for m in base.actives if id(m) not in dropped and not any(m is a for a in added)] + added
    new = FieldState(weather=base.weather, terrain=base.terrain, trick_room=base.trick_room, my_side=set(base.my_side),
                     their_side=set(base.their_side), actives=actives, fainted=dict(base.fainted), party_base_atk=dict(base.party_base_atk))
    for mon in added:
        mon.boosted_stat = paradox_boosted_stat(mon, new)
    return new


def entry_hits(attacker: "Mon", incoming: "Mon", fs: "FieldState | None", *, our_actives: int = 2, their_actives: int = 2,
               locked: str | None = None, outgoing: "Mon | None" = None) -> list[dict]:
    """What ``attacker`` can do to ``incoming`` on the turn it comes in: one hit profile per damaging move it can use,
    worst first, each with ``move``, ``effect`` and ``ko_chance`` (hit count, damage rolls AND accuracy: a 70%-accurate
    guaranteed KO is a 70% KO). The incoming Pokémon's Intimidate is applied to the attacker; the field is the one AFTER
    the switch (``entry_field``: the incoming Pokémon's Ruin ability counts, ``outgoing``'s no longer does); a spread
    move gets the 0.75x when our side has two Pokémon out (``our_actives``, the switch-in included) or, for
    Earthquake-likes, when the attacker's partner is there too; a Choice lock limits the moves. Fake Out / First
    Impression are left out (a fresh attacker's Fake Out is the threat list's business)."""
    original = attacker
    attacker = intimidated_foes(incoming, [attacker])[0]
    after = entry_field(fs, add=[attacker, incoming], drop=[outgoing, original])
    out = []
    for mv in attacker.moves:
        if not data.is_damaging(mv) or mv["id"] in ("fakeout", "firstimpression") or (locked and mv["id"] != locked):
            continue
        prof = hit_profile(attacker, mv, incoming, field=after, spread=is_spread(mv, foes=our_actives, allies=their_actives - 1))
        if prof is None:
            continue
        hit = {"move": mv["id"], **prof, "effect": effect_label(type_multiplier(attacker, mv, incoming, after))}
        hit["ko_chance"] = round(ko_probability(hit, with_accuracy=True), 3)
        out.append(hit)
    out.sort(key=lambda h: -_danger(_avg(h["damage_pct_of_current_hp"]), h.get("hit_chance", 1.0), h["ko_chance"]))
    return out


HAZARD_IDS = ("spikes", "stealthrock", "stickyweb", "toxicspikes")
SPIKES_CHIP = (0.0, 1 / 8, 1 / 6, 1 / 4)  # by layer count


def normalize_hazards(hazards: object, fs: "FieldState | None" = None) -> dict[str, int]:
    """Entry hazards on the switch-in's side as {id: layers}. ``hazards`` is a dict ({"spikes": 2}), a list / set of ids
    (one layer each, repeats add up), or None: then whatever the field's side conditions name, one layer each (the
    exact layer count from the protocol log's -sidestart lines arrives through the ops lane's memory.hazards(side))."""
    out: dict[str, int] = {}
    if hazards is None:
        hazards = [h for h in HAZARD_IDS if fs is not None and h in fs.my_side]
    if isinstance(hazards, dict):
        for name, value in hazards.items():
            key = data.to_id(name)
            if key in HAZARD_IDS and value:
                out[key] = max(1, int(value)) if isinstance(value, (int, float)) else 1
    else:
        for name in hazards or ():
            key = data.to_id(name)
            if key in HAZARD_IDS:
                out[key] = out.get(key, 0) + 1
    return out


def hazard_chip(incoming: "Mon", hazards: dict[str, int]) -> tuple[float, list[str]]:
    """What the hazards on its side do to a Pokémon as it comes in: the fraction of its MAX HP lost (Spikes 1/8, 1/6,
    1/4 by layer, grounded only; Stealth Rock 1/8 times the Rock effectiveness) and notes (Sticky Web: -1 Speed on a
    grounded switch-in; Toxic Spikes: poisoned, badly with two layers, absorbed by a grounded Poison type, nothing to
    Steel types). Heavy-Duty Boots ignore all of it."""
    if not hazards:
        return 0.0, []
    if data.to_id(incoming.item) == "heavydutyboots":
        return 0.0, ["Heavy-Duty Boots: hazards do nothing"]
    chip, notes = 0.0, []
    spikes = hazards.get("spikes", 0)
    if spikes and incoming.grounded:
        frac = SPIKES_CHIP[min(3, spikes)]
        chip += frac
        notes.append(f"Spikes x{min(3, spikes)}: -{frac * 100:.1f}% on entry")
    if hazards.get("stealthrock"):
        frac = data.effectiveness("rock", incoming.types) / 8
        chip += frac
        notes.append(f"Stealth Rock: -{frac * 100:.1f}% on entry")
    if hazards.get("stickyweb") and incoming.grounded:
        notes.append("Sticky Web: -1 Speed on entry")
    if hazards.get("toxicspikes") and incoming.grounded and "steel" not in incoming.types:
        if "poison" in incoming.types:
            notes.append("Toxic Spikes: absorbed (Poison type)")
        else:
            notes.append("Toxic Spikes: " + ("badly poisoned" if hazards["toxicspikes"] >= 2 else "poisoned") + " on entry")
    return min(1.0, chip), notes


def entry_verdict(incoming: "Mon", foes: "Iterable[Mon]", fs: "FieldState | None", *, outgoing: "Mon | None" = None,
                  our_actives: int = 2, hazards: object = None, locked: dict[str, str] | None = None) -> dict:
    """Is this switch-in safe against BOTH foes? HP after the hazard chip, each foe's worst entry hit (``entry_hits``:
    Intimidate, the post-switch field, spread, Choice locks, accuracy), the two foes' double-target (the faster foe's hit
    first; the slower one's meets +1 Defense on a Stamina holder) and ``ko_on_entry``: the KO odds when they pick their
    best line (one hit or the double). ``verdict`` is "ko_on_entry" from 50% up, else "safe". Oct 8 live match T12: the
    sheet checked Milotic only against the foe whose threat triggered the switch and offered it into Rillaboom's Wood
    Hammer (152-181% once Wo-Chien's Tablets left, more after Spikes); it died on entry."""
    foes = list(foes)
    chip, notes = hazard_chip(incoming, normalize_hazards(hazards, fs))
    x = incoming if chip <= 0 else Mon(**{**incoming.__dict__, "hp_fraction": max(0.01, incoming.hp_fraction - chip)})
    takes: list[dict] = []
    for foe in foes:
        hits = entry_hits(foe, x, fs, our_actives=our_actives, their_actives=len(foes),
                          locked=(locked or {}).get(species_key(foe.species)), outgoing=outgoing)
        if hits:
            takes.append({"_foe": foe, "_hit": hits[0]})
    double = None
    if len(takes) == 2:
        first, second = sorted(takes, key=lambda t: -t["_foe"].stat("spe"))  # the faster foe's hit lands first
        a, b = first["_hit"], second["_hit"]
        braced = False
        if data.to_id(x.ability) == "stamina" and (data.move_info(b["move"]) or {}).get("category") == "Physical":
            # Stamina: the first hit raises Defense, so the second hit of a double lands on +1.
            up = Mon(**{**x.__dict__, "boosts": {**x.boosts, "def": min(6, x.boosts.get("def", 0) + 1)}})
            again = [h for h in entry_hits(second["_foe"], up, fs, our_actives=our_actives, their_actives=len(foes),
                                           locked=(locked or {}).get(species_key(second["_foe"].species)), outgoing=outgoing)
                     if h["move"] == b["move"]]
            if again:
                b, braced = again[0], True
        lo = a["damage_pct_of_current_hp"][0] + b["damage_pct_of_current_hp"][0]
        hi = a["damage_pct_of_current_hp"][1] + b["damage_pct_of_current_hp"][1]
        acc_a, acc_b = a.get("hit_chance", 1.0), b.get("hit_chance", 1.0)
        both = acc_a * acc_b * data.rolls_reach(2, lo / 2, hi / 2, 100.0)
        p_double = both + acc_a * (1 - acc_b) * ko_probability(a) + (1 - acc_a) * acc_b * ko_probability(b)
        double = {"moves": [f"{first['_foe'].species} {a['move']}", f"{second['_foe'].species} {b['move']}"],
                  "damage_pct": [round(lo, 1), round(hi, 1)], "ko_chance": round(min(1.0, p_double), 3),
                  **({"note": f"Stamina: {second['_foe'].species}'s hit lands on +1 Defense"} if braced else {})}
    entries = []
    for t in takes:
        hit = t["_hit"]
        entries.append({"from": t["_foe"].species, "move": hit["move"], "damage_pct": list(hit["damage_pct_of_current_hp"]),
                        "effect": hit["effect"], "ko_chance": hit["ko_chance"],
                        **{k: hit[k] for k in ("hit_chance", "per_hit", "first_hit", "hits") if k in hit}})
    p_single = max((e["ko_chance"] for e in entries), default=0.0)
    ko = max(p_single, double["ko_chance"] if double else 0.0)
    worst = max(entries, key=lambda e: _danger(_avg(e["damage_pct"]), e.get("hit_chance", 1.0), e["ko_chance"]), default=None)
    return {"hp_pct": int(round(100 * x.hp_fraction)), "hazards": notes, "takes": entries, "takes_worst": worst,
            "double_target": double, "ko_on_entry": round(ko, 3), "verdict": "ko_on_entry" if ko >= 0.5 else "safe"}


def _danger(avg_pct: float, hit_chance: float, ko_chance: float) -> float:
    """How bad one incoming hit is: its expected damage (capped at a KO, accuracy included) plus a premium for the KO
    odds, so a 70%-accurate OHKO outranks a sure 75% hit."""
    return min(avg_pct, 100.0) * hit_chance + 30.0 * ko_chance


def _switch_previews(sheet: "TurnSheet", slots: list[dict], theirs: dict[int, "Mon"], obs: dict, memory: MatchMemory,
                     fs: "FieldState", *, hazards: object = None) -> None:
    """For every Pokémon we could bring in now: ``entry_verdict`` against BOTH foes (its own Intimidate on them, the
    post-switch field with its own Ruin ability and without the outgoing Pokémon's, the spread modifier when our partner
    stays out, the hazard chip, accuracy and hit counts, their double-target), its best hit back with that field, who it
    outspeeds, and a score that ``best_switch`` reads. A switch-in both slots can make is priced for each (the outgoing
    Pokémon differs) and reported at its worse slot, with ``by_slot`` when the two differ. ``hazards`` is the memory's
    layer count for our side (ops lane); None reads the side conditions as one layer each. Replaces 'type chart only'
    switch judgement (Oct 7: Garchomp came in because it 'resists Make It Rain' (it is neutral) and died before moving)."""
    offered: dict[str, list[int]] = {}
    labels: dict[str, str] = {}
    for slot in slots:
        for o in slot.get("options") or []:
            if o.get("type") == "switch" and o.get("species"):
                key = species_key(o["species"])
                labels.setdefault(key, o["species"])
                if slot.get("slot", 0) not in offered.setdefault(key, []):
                    offered[key].append(slot.get("slot", 0))
    if not offered or not theirs:
        return
    ours_by_pos = {m.position: m for m in fs.actives if m.side == "mine"}
    our_live = max(1, len(_active_summaries(obs.get("team"))))  # the switch replaces one of them: the count stays
    locks = {species_key(opp.species): memory.opp_last_move[species_key(opp.species)] for opp in theirs.values()
             if data.to_id(opp.item) in CHOICE_ITEMS and memory.opp_last_move.get(species_key(opp.species))}
    for key, slot_numbers in offered.items():
        summary = _find_summary(obs.get("team"), labels[key])
        x = build_mon({**summary, "boosts": {}}, memory.my_cards.get(key), side="mine", position=None)  # boosts reset on switch
        if x is None:
            continue
        x.tailwind = "tailwind" in fs.my_side
        fake_out = any(m["id"] == "fakeout" for m in x.moves)
        per_slot: dict[int, dict] = {}
        for number in sorted(slot_numbers):
            outgoing = ours_by_pos.get(number)
            verdict = entry_verdict(x, theirs.values(), fs, outgoing=outgoing, our_actives=our_live, hazards=hazards, locked=locks)
            after = entry_field(fs, add=[x], drop=[outgoing])
            deals = []
            for opp in theirs.values():
                best = None
                for mv in x.moves:
                    if not data.is_damaging(mv) or mv["id"] in ("fakeout", "firstimpression"):
                        continue
                    prof = hit_profile(x, mv, opp, field=after, spread=is_spread(mv, foes=len(theirs), allies=our_live - 1))
                    if prof is None:
                        continue
                    est = prof["damage_pct_of_current_hp"]
                    acc = prof.get("hit_chance", 1.0)
                    if best is None or min(_avg(est), 100) * acc > min(_avg(best[1]), 100) * best[3]:
                        best = (mv["id"], est, type_multiplier(x, mv, opp, after), acc, round(ko_probability(prof, with_accuracy=True), 3))
                if best:
                    deals.append({"to": opp.species, "move": best[0], "damage_pct": list(best[1]), "effect": effect_label(best[2]),
                                  "ko_chance": best[4], **({"hit_chance": best[3]} if best[3] < 1 else {})})
            my_speed = x.stat("spe")
            outspeeds = [o.species for o in theirs.values() if (my_speed > o.stat("spe")) != fs.trick_room]
            takes, worst_in = verdict["takes"], verdict["takes_worst"]
            offense = sum(min(_avg(d["damage_pct"]), 100) * d.get("hit_chance", 1.0) + 25 * d["ko_chance"] for d in deals) / max(len(theirs), 1)
            danger = 0.0
            if worst_in:
                danger = _danger(_avg(worst_in["damage_pct"]), worst_in.get("hit_chance", 1.0), worst_in["ko_chance"])
                others = sorted((min(_avg(t["damage_pct"]), 100) * t.get("hit_chance", 1.0) for t in takes if t is not worst_in), reverse=True)
                danger += 0.35 * others[0] if others else 0.0  # they may double into the newcomer
            score = offense - 0.8 * danger + 8 * len(outspeeds) + (10 if fake_out else 0) - 100 * verdict["ko_on_entry"]
            per_slot[number] = {**verdict, "deals_best": deals, "outspeeds": outspeeds, "score": round(score, 1)}
        head = min(per_slot.values(), key=lambda v: v["score"])
        notes = intimidate_notes(x, theirs.values())
        entry = {
            "hp_pct": head["hp_pct"],
            **({"hazards": head["hazards"]} if head["hazards"] else {}),
            "takes_worst": head["takes_worst"],
            "takes": head["takes"],
            **({"double_target": head["double_target"]} if head["double_target"] else {}),
            "ko_on_entry": head["ko_on_entry"],
            "verdict": head["verdict"],
            "deals_best": head["deals_best"],
            "outspeeds": head["outspeeds"],
            **({"intimidate": notes} if notes else {}),
            **({"fake_out_next_turn": True} if fake_out else {}),
            "score": head["score"],
        }
        if len(per_slot) > 1 and len({v["score"] for v in per_slot.values()}) > 1:
            entry["by_slot"] = {n: {"replacing": ours_by_pos[n].species if n in ours_by_pos else None, "takes_worst": v["takes_worst"],
                                    "ko_on_entry": v["ko_on_entry"], "verdict": v["verdict"], "score": v["score"]} for n, v in per_slot.items()}
        sheet.switch_ins[key] = entry


def _flinchable(me: "Mon", ours: dict[int, "Mon"], fs: "FieldState") -> bool:
    """Can a Fake Out make this Pokémon flinch? Not a Ghost (Fake Out is Normal), not Inner Focus / Shield Dust /
    Covert Cloak, not grounded in Psychic Terrain, and no Armor Tail / Dazzling / Queenly Majesty on our side."""
    if "ghost" in me.types or data.to_id(me.ability) in ("innerfocus", "shielddust") or data.to_id(me.item) == "covertcloak":
        return False
    if fs.terrain == "psychic" and me.grounded:
        return False
    return not any(data.to_id(m.ability) in ("armortail", "dazzling", "queenlymajesty") for m in ours.values())


def _flinch_risk(sheet: "TurnSheet", ours: dict[int, "Mon"], theirs: dict[int, "Mon"], fs: "FieldState",
                 slots: list[dict] | None = None, P: dict | None = None) -> None:
    """A FRESH opposing Fake Out (+3 priority) will usually hit one of our slots, and that slot does not move this turn.
    Oct 7, match 3: Great Tusk 'outspeeds and KOs Whimsicott', but a just-switched-in Incineroar Faked Out Tusk and
    Whimsicott's Moonblast KO'd it. A plan that needs a slot to move first is only as good as that slot's flinch odds.
    The Fake Out is aimed: the search bot used its fresh Fake Outs on the slot whose action it could still stop every
    time (Oct 8, 2 of 2), so the slot with more to deny (its best attack, a KO, a setup move) carries 0.8 of the risk
    and the other 0.2, instead of an even split that under-priced the Thundurus it flinched and then KO'd."""
    users = [t for t in sheet.threats if "fakeout" in t["known_moves"] and "FRESH" in (t.get("note") or "")]
    if not users:
        return
    eligible = [n for n, me in ours.items() if _flinchable(me, ours, fs)]
    if not eligible:
        return
    P = P or DEFAULTS
    by_slot = {e["slot"]: e for e in sheet.our_options}

    def stoppable(n: int) -> float:
        value = max(_best_attack(by_slot.get(n, {}), P=P)[2], 0.0)
        for s in slots or []:
            if s.get("slot", 0) == n:
                for o in s.get("options") or []:
                    if o.get("type") == "move":
                        value = max(value, float(SETUP_MOVES.get(data.to_id(o.get("move_id")), 0)))
        return value

    if len(eligible) == 1:
        aim = {eligible[0]: 1.0}
    else:
        weight = {n: stoppable(n) for n in eligible}
        top = max(weight, key=weight.get)
        rest = max((v for n, v in weight.items() if n != top), default=0.0)
        aim = {n: (0.5 if weight[top] <= 1.1 * rest else (0.8 if n == top else 0.2)) for n in eligible}
    for n in eligible:
        sheet.flinch_risk[n] = round(min(0.9, 1 - (1 - aim[n]) ** len(users)), 2)
    names = " and ".join(t["species"] for t in users)
    slots_txt = ", ".join(f"slot {n} ({ours[n].species}): {int(sheet.flinch_risk[n] * 100)}%" for n in eligible)
    sheet.warnings.append(
        f"FAKE OUT: {names} can Fake Out this turn (+3 priority), most likely at the slot whose action it can stop. Chance "
        f"to flinch and not move at all: {slots_txt}. Any plan that needs that slot to move first (to KO a threat before it "
        "acts, or to set Tailwind/Trick Room) can fail; Protect blocks Fake Out, and a Ghost type, Inner Focus or Covert Cloak ignores it.")


def _double_target_threats(sheet: "TurnSheet", ours: dict[int, "Mon"], P: dict) -> None:
    """Both foes into ONE of our slots: the threat list scores each attacker alone, but good players double-target, and
    two hits that are survivable alone can KO together (Oct 7: Landorus fell to Extreme Speed + Shadow Ball, and to
    Scale Shot + Draco Meteor, with no warning). Recorded per slot with the combined range and whether both land first."""
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    for number, me in sorted(ours.items()):

        def first(t: dict, h: dict) -> bool:
            if (h.get("priority") or 0) > 0:
                return True
            return rank.get(("theirs", t["position"]), 99) < rank.get(("mine", number), 99)

        foes = [(t, [h for h in t["hits"] if h["into_slot"] == number]) for t in sheet.threats]
        foes = [(t, hits) for t, hits in foes if hits]
        if len(foes) != 2:
            continue
        if any(h["ko"] == "guaranteed" for _, hits in foes for h in hits):
            continue  # one of them already KOs alone: the single-threat warning covers it
        pairs = [((foes[0][0], a), (foes[1][0], b)) for a in foes[0][1] for b in foes[1][1]]

        def order_key(t: dict, h: dict) -> tuple[int, int]:
            return (-(h.get("priority") or 0), rank.get(("theirs", t["position"]), 99))

        def combined(pair) -> tuple[float, float, float, bool]:
            """The two hits in the order they land (priority, then speed), a fraction move (Ruination) taking its share
            of what is LEFT, a Sitrus Berry between them, accuracy in the odds; a double that needs their spread move
            to hit their own partner is discounted (the search bot never did it in 6 chances, Oct 8)."""
            (ta, a), (tb, b) = sorted(pair, key=lambda th: order_key(*th))
            lo, hi, chance = compose_double(a, b, me)
            if any((data.move_info(h["move"]) or {}).get("target") == "allAdjacent" for h in (a, b)):
                chance *= 0.5
            return lo, hi, chance, first(ta, a) and first(tb, b)

        scored = [(combined(pair), pair) for pair in pairs]
        lethal_first = [x for x in scored if x[0][3] and x[0][2] >= 0.5]  # Oct 7: Extreme Speed + Shadow Ball KO'd Landorus before it moved
        pick = max(lethal_first or scored, key=lambda x: (x[0][2], x[0][1]))
        (lo, hi, chance, before), ((t1, h1), (t2, h2)) = pick
        if hi < 100 or chance <= 0:
            continue
        sheet.focus_threats.append({"slot": number, "species": me.species, "moves": [f"{t1['species']} {h1['move']}", f"{t2['species']} {h2['move']}"],
                                    "damage_pct_of_current_hp": [round(lo, 1), round(hi, 1)], "ko_chance": round(chance, 2), "before_we_move": before})
        if chance >= 0.5:
            sheet.warnings.append(
                f"DOUBLE-TARGET: if both foes go into slot {number} ({me.species}) — {t1['species']}'s {h1['move']} + {t2['species']}'s "
                f"{h2['move']} = {lo:.0f}-{hi:.0f}% — it goes down" + (" BEFORE it moves" if before else "")
                + ". Opponents double the target that threatens them most; Protect or switch if that is this slot.")


def _endgame(sheet: "TurnSheet", slots: list[dict], ours: dict[int, "Mon"], theirs: dict[int, "Mon"], fs: "FieldState") -> None:
    """With our last Pokémon alone on the field, Protect only delays — unless waiting changes something. Oct 7, match 1:
    Cresselia, our last Pokémon, Protected four turns running against a +3 Gholdengo (the model believed benched
    teammates were coming; none were). Record what a stall could actually wait out."""
    filled = [s for s in slots if s.get("slot", 0) in ours]
    empty = [s for s in slots if s.get("slot", 0) not in ours]
    if len(filled) != 1 or not empty or any(o.get("type") == "switch" for s in slots for o in s.get("options") or []):
        return
    sheet.alone = True
    me = next(iter(ours.values()))
    reasons = []
    my_speed = me.stat("spe")
    if "tailwind" in fs.their_side and not fs.trick_room and any(
            opp.stat("spe") > my_speed >= opp.stat("spe") / 2 for opp in theirs.values()):
        reasons.append("their Tailwind runs out (then we move first)")
    for screen in ("reflect", "lightscreen", "auroraveil"):
        if screen in fs.their_side:
            reasons.append(f"their {screen} runs out")
    if fs.trick_room and any(opp.stat("spe") < my_speed for opp in theirs.values()):
        reasons.append("Trick Room runs out (then we move first)")
    for opp in theirs.values():
        if opp.status in ("brn", "psn", "tox"):
            reasons.append(f"their {opp.species} takes {opp.status} damage")
        if fs.weather == "sand" and not set(opp.types) & {"rock", "ground", "steel"} and data.to_id(opp.ability) not in ("overcoat", "sandveil", "sandrush", "sandforce", "magicguard"):
            reasons.append(f"their {opp.species} takes sandstorm damage")
    if data.to_id(me.item) == "leftovers" or (fs.terrain == "grassy" and me.grounded):
        reasons.append(f"our {me.species} heals a little")
    sheet.stall_reasons = reasons
    sheet.notes.append(
        f"ENDGAME: {me.species} is our last Pokémon. " + (
            f"A Protect would wait out: {'; '.join(reasons)}." if reasons else
            "Nothing runs out or ticks in our favour, so Protect only delays the same position by a turn: deal the most damage you can."))


# -- candidates --------------------------------------------------------------------------------------


def _expected(row: dict, priority: int, P: dict | None = None, threat: float = 0.0) -> float:
    """Expected value of a hit in 'damage points': the average roll, CAPPED at the target's remaining HP (a 999% hit
    takes one Pokémon, exactly like a 101% one — uncapped, both slots piled into a 9% Whimsicott on Oct 7), plus a KO
    bonus and a little for priority. ``threat`` is what the target would do to us (``threat_weight``): it scales the
    KO bonus once the KO-credit item lands."""
    P = P or DEFAULTS
    lo, hi = row["damage_pct_of_current_hp"]
    if hi <= 0:
        return 0.0  # an immune or blocked hit is worth nothing, priority or not (Fake Out into a Ghost, Oct 7)
    acc = float(row.get("hit_chance", 1.0))
    damage = min((lo + hi) / 2, 100.0) * acc
    if P.get("hp_units", 0) and "hp_frac" in row:
        damage *= row["hp_frac"]  # in % of MAX HP: finishing a 9% Pokémon is 9 points of damage plus the KO, not 100
    bonus = 0.0
    if hi >= 100:
        # The KO bonus by the odds (rolls, hit count, accuracy: a 1% chance is not a 95% one) and by the threat removed.
        bonus = ko_probability(row, with_accuracy=True) * (P["ko_bonus_guaranteed"] + P.get("ko_threat_w", 0.0) * threat)
    return damage + bonus + P["priority_w"] * priority


def _best_attack(entry: dict, *, avoid_target: int | None = None, P: dict | None = None,
                 threats: dict[str, float] | None = None) -> tuple[dict | None, dict | None, float]:
    """Highest expected damage (option, target_row, expected%) among a slot's move options. A spread move (rows with
    target 0) counts the damage to every opposing Pokémon it hits (its summary row carries the per-target ``rows``, so
    a KO is read per foe, never from the summed range) and is charged for damage to our own ally. With
    ``avoid_target`` only single-target rows into the other foes count: spread rows carry target 0 and slipped
    through the filter, so focus_fire / finish_the_other vanished whenever a slot held Rock Slide / Dazzling Gleam /
    Bleakwind Storm (Oct 7 match 3 turn 3; Oct 8 practice turn 6). ``threats`` (species_key -> weight) scales each
    KO bonus by the threat removed."""
    P = P or DEFAULTS
    threats = threats or {}
    best: tuple[dict | None, dict | None, float] = (None, None, -1.0)
    for option in entry.get("options") or []:
        if option.get("fails"):
            continue
        priority = option.get("priority") or 0
        rows = option.get("targets") or []
        foe_rows = [r for r in rows if r["side"] == "theirs"]
        spread_rows = [r for r in foe_rows if r["target"] == 0]
        if spread_rows and avoid_target is None:
            ally_hit = sum(_expected(r, 0, P) for r in rows if r["side"] == "mine")
            expected = sum(_expected(r, priority, P, threats.get(species_key(r["species"]), 0.0)) for r in spread_rows) - P["ally_damage_w"] * ally_hit
            if expected > best[2]:
                summary = {"target": 0, "species": " + ".join(r["species"] for r in spread_rows), "side": "theirs",
                           "damage_pct_of_current_hp": [round(sum(r["damage_pct_of_current_hp"][0] for r in spread_rows), 1),
                                                        round(sum(r["damage_pct_of_current_hp"][1] for r in spread_rows), 1)],
                           "ally_damage_pct": round(ally_hit, 1) if ally_hit else 0, "spread": True, "rows": spread_rows}
                best = (option, summary, expected)
        for row in foe_rows:
            if row["target"] == 0 or row["target"] == avoid_target:
                continue
            expected = _expected(row, priority, P, threats.get(species_key(row["species"]), 0.0))
            if expected > best[2]:
                best = (option, row, expected)
    return best


def kos_a_foe(row: dict | None, *, min_chance: float = 1.0) -> bool:
    """Does this chosen row take a foe out: a single-target row at a guaranteed KO, or a spread summary with ANY per-
    target row at one. The summary's own range is the SUM over its targets, which read as a single-target KO (Oct 8:
    fake_out_setup dropped to its 'a KO is already there' base on a 60% + 50% spread)."""
    if not row:
        return False
    rows = row.get("rows") or [row]
    return any(r.get("side") == "theirs" and ko_probability(r) >= min_chance for r in rows if r.get("damage_pct_of_current_hp"))


def _option_index(slot: dict, kind: str, **match: Any) -> int | None:
    for index, option in enumerate(slot.get("options") or []):
        if option.get("type") == kind and all(option.get(k) == v for k, v in match.items()):
            return index
    return None


SLEEP_MOVES = {"spore", "sleeppowder", "hypnosis", "darkvoid", "lovelykiss", "sing", "grasswhistle", "yawn"}
REDIRECT_MOVES = {"ragepowder", "followme"}
SETUP_MOVES = {"tailwind": 55, "trickroom": 50, "icywind": 35, "electroweb": 30, "snarl": 20}
STATUS_MOVES = {"willowisp": 35, "thunderwave": 35, "stunspore": 25, "glare": 30, "nuzzle": 25}
MISC_VALUE = {"helpinghand": 20, "partingshot": 25, "encore": 20, "nastyplot": 30, "swordsdance": 30, "dragondance": 30,
              "calmmind": 25, "bulkup": 25, "coaching": 20, "lifedew": 20, "wideguard": 20, "substitute": 10, "haze": 10}


ALLY_ONLY = {"helpinghand", "coaching", "followme", "ragepowder", "spotlight", "afteryou", "decorate", "allyswitch"}


def status_value(move_id: str, me: "Mon", theirs: dict[int, "Mon"], sheet: "TurnSheet", slot_number: int,
                 has_ally: bool = True) -> tuple[float, int]:
    """(value in 'expected damage points', target int) for a non-damaging move; 0 when pointless. Moves that only help
    (or shield) a partner are worth nothing without one (Oct 7: Helping Hand into an empty slot was our 'best' move)."""
    if move_id in ALLY_ONLY and not has_ally:
        return (0.0, 0)
    if move_id in SLEEP_MOVES:
        best = (0.0, 0)
        for pos, opp in theirs.items():
            if opp.status or "grass" in opp.types and move_id in ("spore", "sleeppowder", "stunspore") or data.to_id(opp.item) in ("safetygoggles", "covertcloak") and move_id in ("spore", "sleeppowder"):
                continue
            danger = sum(_expected(h, 0) for t in sheet.threats if t["position"] == pos for h in t["hits"])
            value = 50 + min(danger, 60) / 3
            if value > best[0]:
                best = (value, pos)
        return best
    if move_id in REDIRECT_MOVES:
        partner_hits = [h for t in sheet.threats for h in t["hits"] if h["into_slot"] != slot_number]
        return ((45 if any(h["ko"] != "no" for h in partner_hits) else 15), 0)
    if move_id in SETUP_MOVES:
        up = {"tailwind": any("Our Tailwind" in n for n in sheet.notes), "trickroom": sheet.trick_room}.get(move_id, False)
        return ((0 if up else SETUP_MOVES[move_id]), 0)
    if move_id in STATUS_MOVES:
        best = (0.0, 0)
        for pos, opp in theirs.items():
            if opp.status:
                continue
            immune = ("fire" in opp.types and move_id == "willowisp") or ("electric" in opp.types and move_id in ("thunderwave", "nuzzle")) \
                or ("ground" in opp.types and move_id in ("thunderwave", "nuzzle")) or ("grass" in opp.types and move_id == "stunspore")
            if immune:
                continue
            if STATUS_MOVES[move_id] > best[0]:
                best = (float(STATUS_MOVES[move_id]), pos)
        return best
    if move_id in MISC_VALUE:
        value = MISC_VALUE[move_id]
        if move_id in ("nastyplot", "swordsdance", "dragondance", "calmmind", "bulkup") and me.hp_fraction < 0.6:
            value = 5
        return (float(value), 0)
    return (0.0, 0)


def ko_probability(hit: dict, *, with_accuracy: bool = False) -> float:
    """Chance a hit KOs once it lands: the damage roll is uniform between its min and max for a single strike; a
    multi-hit row (``per_hit`` + ``hits`` from ``hit_profile``) sums the hit-count odds with each strike rolling on its
    own. ``with_accuracy`` also multiplies by the move's chance to land (``hit_chance`` on the row; 1.0 when absent),
    so a 70%-accurate guaranteed KO reads as a 70% KO."""
    acc = float(hit.get("hit_chance", 1.0)) if with_accuracy else 1.0
    per_hit, hits = hit.get("per_hit"), hit.get("hits")
    if per_hit and hits:
        return acc * data.multi_hit_ko_chance(per_hit[0], per_hit[1], 1.0, hits, first_hit=hit.get("first_hit"))
    lo, hi = hit["damage_pct_of_current_hp"]
    if lo >= 100:
        return acc
    if hi < 100:
        return 0.0
    return acc * (hi - 100.0) / max(hi - lo, 0.1)


def linear_ko(lo: float, hi: float) -> float:
    """Chance a roll uniform between ``lo`` and ``hi`` (% of the target's current HP) reaches a KO."""
    if lo >= 100:
        return 1.0
    if hi < 100:
        return 0.0
    return (hi - 100.0) / max(hi - lo, 0.1)


def material(mon: "Mon", P: dict | None = None) -> float:
    """What one of our Pokémon on the board is worth, in the damage points the scorer uses: its HP in % of max (the
    unit under hp_units) plus the KO bonus, so losing a full-HP Pokémon costs what taking one is worth."""
    P = P or DEFAULTS
    return 100.0 * max(0.0, min(1.0, mon.hp_fraction)) + P["ko_bonus_guaranteed"]


def threat_weight(pos: int, sheet: "TurnSheet", ours: dict[int, "Mon"], P: dict | None = None) -> float:
    """The most one action of the foe at ``pos`` does to us this turn, in damage points: a spread move counts both its
    targets, a single-target move only its best one, each capped at what that slot has left (in % of max HP), with
    accuracy and the KO bonus by its KO odds. The KO bonus scales by this: taking out the Pokémon that OHKOs us is worth
    more than the one with more HP left (the Oct 8 LLM-agent game, turn 5: Ice Shard into the 27% Flutter Mane that
    KOs Baxcalibur before it moves, not into the 34% Rillaboom)."""
    P = P or DEFAULTS
    by_move: dict[str, float] = {}
    for t in sheet.threats:
        if t["position"] != pos:
            continue
        for h in t["hits"]:
            me = ours.get(h["into_slot"])
            hp = me.hp_fraction if me is not None else 1.0
            v = min(_avg(h["damage_pct_of_current_hp"]), 100.0) * hp * float(h.get("hit_chance", 1.0)) \
                + P["ko_bonus_guaranteed"] * ko_probability(h, with_accuracy=True)
            spread = (data.move_info(h["move"]) or {}).get("target") in SPREAD_TARGETS
            by_move[h["move"]] = by_move.get(h["move"], 0.0) + v if spread else max(by_move.get(h["move"], 0.0), v)
    return max(by_move.values(), default=0.0)


def threat_weights(sheet: "TurnSheet", ours: dict[int, "Mon"], theirs: dict[int, "Mon"], P: dict | None = None) -> dict[str, float]:
    """species_key -> threat_weight for every foe on the field (what ``_best_attack`` takes as ``threats``)."""
    return {species_key(foe.species): threat_weight(pos, sheet, ours, P) for pos, foe in theirs.items()}


def row_into(opt: dict | None, row: dict | None, pos: int, species: str) -> dict | None:
    """The chosen option's damage row into the foe at ``pos``: the single-target row when that is the chosen target, or
    the spread move's per-target row for that species (spread rows carry target 0). None when the hit misses that foe."""
    if not row:
        return None
    chosen = row.get("target", 0)
    if chosen > 0:
        return row if chosen == pos else None
    if not row.get("rows") and species_key(row.get("species")) == species_key(species) and row.get("side") == "theirs":
        return row  # a per-target spread row
    for r in row.get("rows") or (opt.get("targets") if opt else None) or []:
        if r.get("side") == "theirs" and r.get("target", 0) == 0 and species_key(r.get("species")) == species_key(species):
            return r
    return None


def their_priority_into(sheet: "TurnSheet", pos: int, slot: int | None = None) -> int:
    """The priority of the foe's most dangerous KO hit (into ``slot``, or any slot): removing it 'before it moves' means
    beating that priority (an Extreme Speed KO lands before anything slower than +2)."""
    return max((h.get("priority", 0) or 0 for t in sheet.threats if t["position"] == pos for h in t["hits"]
                if h["ko"] != "no" and (slot is None or h["into_slot"] == slot)), default=0)


def _outcome(parts: list[tuple[float, dict]]) -> tuple[float, float]:
    """(expected damage capped at a KO, KO chance) of up to a few hits into ONE foe, each landing with its own
    probability (the attacker's chance to act and the move's accuracy) and rolling on its own. Enumerates which hits
    land; a single hit keeps the exact multi-hit odds, two or more use the summed range. Damage is in % of the foe's
    current HP, each hit capped at a KO BEFORE it is weighted, so an overkill hit at a 50% flinch is worth half a KO,
    not a full one (the Oct 8 loss, turn 3: best_attacks scored its no-flinch 125 under a fresh Fake Out)."""
    parts = [(max(0.0, min(1.0, p)), r) for p, r in parts if p > 0 and r.get("damage_pct_of_current_hp", [0, 0])[1] > 0]
    if not parts:
        return 0.0, 0.0
    damage = 0.0
    ko = 0.0
    n = len(parts)
    for mask in range(1, 1 << n):
        prob = 1.0
        lo = hi = 0.0
        landed = []
        for i, (p, r) in enumerate(parts):
            if mask >> i & 1:
                prob *= p
                landed.append(r)
                lo += r["damage_pct_of_current_hp"][0]
                hi += r["damage_pct_of_current_hp"][1]
            else:
                prob *= 1.0 - p
        if prob <= 0:
            continue
        damage += prob * min((lo + hi) / 2, 100.0)
        ko += prob * (ko_probability(landed[0]) if len(landed) == 1 else linear_ko(lo, hi))
    return damage, min(1.0, ko)


def survival_factor(slot_number: int, sheet: "TurnSheet", my_priority: int, speed_rank: dict,
                    my_attack_row: dict | None = None, P: dict | None = None, removed: dict[int, float] | None = None,
                    ignore_pos: int | None = None) -> float:
    """Chance this slot gets its action to count: 1.0 if nothing faster can KO it, 0.0 for a sure KO first, in between
    by the KO odds (rolls, hit count and accuracy). ``removed`` is each foe's chance of being taken out before it moves
    by the PARTNER (``joint_value`` fills it in): a threat the partner's faster or priority hit removes does not count
    (the Oct 8 LLM-agent game, turn 7: Baxcalibur's +1 Ice Shard KOs Rillaboom before its Wood Hammer, so Gyarados
    survives to Waterfall; it was scored 0.00). Our own attack (``my_attack_row``) removes the foe it KOs when it acts
    first. A slot that acts first ONLY through a priority move still lands its hit; what it loses right afterwards is
    ``doomed_after_priority``, charged as material by the ranking (D1i, the Oct 8 practice game turn 13: Extreme Speed
    for 39-46% into Dragonite was 'an escape' from a Scale Shot that then KO'd Arcanine). Slower threats are the
    ranking's business."""
    P = P or DEFAULTS
    removed = removed or {}
    factor = 1.0
    ignored = next((t["species"] for t in sheet.threats if t["position"] == ignore_pos), None)
    for threat in sheet.threats:
        pos = threat["position"]
        if pos == ignore_pos:
            continue  # the question is whether we hit THIS foe before it moves: its own threat cannot stop that
        for hit in threat["hits"]:
            if hit["into_slot"] != slot_number or hit["ko"] == "no":
                continue
            their_priority = hit.get("priority", 0) or 0
            first_by_speed = speed_rank.get(("mine", slot_number), 99) < speed_rank.get(("theirs", pos), 99)
            we_first = my_priority > their_priority or (my_priority == their_priority and first_by_speed)
            p_self = 0.0
            if we_first and my_attack_row:
                sub = row_into(None, my_attack_row, pos, threat["species"])
                if sub is not None:
                    p_self = ko_probability(sub, with_accuracy=True)
            p_partner = removed.get(pos, 0.0)
            p_ko = ko_probability(hit, with_accuracy=True) * (1.0 if hit["ko"] == "guaranteed" else 2 * P["survival_possible"])
            if not we_first:
                factor = min(factor, 1.0 - p_ko * (1.0 - p_partner))
    for focus in sheet.focus_threats:
        if ignored and any(m.startswith(ignored + " ") for m in focus["moves"]):
            continue
        if focus["slot"] == slot_number and focus["before_we_move"] and my_priority <= 0:
            factor *= 1.0 - P.get("focus_risk", 0.35) * focus["ko_chance"]
    if my_priority < 3:
        factor *= 1.0 - sheet.flinch_risk.get(slot_number, 0.0)  # a fresh Fake Out (+3) stops anything slower than it
    return max(0.0, factor)


def doomed_after_priority(slot_number: int, sheet: "TurnSheet", my_priority: int, speed_rank: dict, my_attack_row: dict | None,
                          removed: dict[int, float] | None = None) -> float:
    """Chance this slot is KO'd right after acting first ONLY through its priority move: the threat is faster, our hit
    lands, and unless that hit (or the partner) takes the threat out first, the KO follows (D1i). A foe uses one move,
    so each foe counts at its best such hit; the foes combine."""
    removed = removed or {}
    survive = 1.0
    for threat in sheet.threats:
        pos = threat["position"]
        worst = 0.0
        for hit in threat["hits"]:
            if hit["into_slot"] != slot_number or hit["ko"] == "no":
                continue
            their_priority = hit.get("priority", 0) or 0
            first_by_speed = speed_rank.get(("mine", slot_number), 99) < speed_rank.get(("theirs", pos), 99)
            we_first = my_priority > their_priority or (my_priority == their_priority and first_by_speed)
            if not we_first or first_by_speed:
                continue
            p_self = 0.0
            if my_attack_row:
                sub = row_into(None, my_attack_row, pos, threat["species"])
                if sub is not None:
                    p_self = ko_probability(sub, with_accuracy=True)
            p_gone = 1.0 - (1.0 - p_self) * (1.0 - removed.get(pos, 0.0))
            worst = max(worst, ko_probability(hit, with_accuracy=True) * (1.0 - p_gone))
        survive *= 1.0 - worst
    return 1.0 - survive


# -- what our own move does to us before the slower threats land (A3 / D1h) ---------------------------------------

POST_MOVE_STAGES = {"dracometeor": {"spa": -2}, "overheat": {"spa": -2}, "leafstorm": {"spa": -2}, "fleurcannon": {"spa": -2},
                    "makeitrain": {"spa": -1}, "closecombat": {"def": -1, "spd": -1}, "headlongrush": {"def": -1, "spd": -1},
                    "armorcannon": {"def": -1, "spd": -1}, "superpower": {"atk": -1, "def": -1}, "hammerarm": {"spe": -1},
                    "scaleshot": {"def": -1, "spe": 1}, "vcreate": {"def": -1, "spd": -1, "spe": -1}}
RECOIL_MOVES = {"flareblitz": 1 / 3, "wavecrash": 1 / 3, "woodhammer": 1 / 3, "bravebird": 1 / 3, "doubleedge": 1 / 3,
                "volttackle": 1 / 3, "headsmash": 1 / 2, "wildcharge": 1 / 4, "takedown": 1 / 4, "submission": 1 / 4,
                "headcharge": 1 / 4, "supercellslam": 0.0}
GLAIVE_RUSH = "glaiverush"  # the user takes double damage until it next moves


def self_cost_pct(me: "Mon", move_id: str, dealt: tuple[float, float] | None, foe: "Mon | None") -> tuple[float, float]:
    """What the move costs its user, as (min%, max%) of the user's CURRENT HP: recoil from the damage dealt (``dealt``
    is the row's range in % of the foe's current HP), Life Orb, the target's Rough Skin / Iron Barbs / Rocky Helmet,
    through mech's ``data.self_damage`` when it exists (the local table otherwise). (0, 0) when nothing."""
    cur = max(1.0, me.stats.get("hp", 1) * me.hp_fraction)
    foe_cur = foe.stats["hp"] * foe.hp_fraction if (foe is not None and foe.stats.get("hp")) else 0.0
    helper = getattr(data, "self_damage", None)
    if callable(helper) and foe is not None:
        out = []
        for pct in (dealt or (0.0, 0.0)):
            hp = min(pct / 100 * foe_cur, foe_cur) if foe_cur else 0.0
            try:
                cost = helper({"id": move_id, **(data.move_info(move_id) or {})}, me, foe, hp)
            except Exception:  # a helper from another version: fall back to the table below
                break
            out.append(float(cost.get("pct_current", 0.0) or 0.0))
        if len(out) == 2:
            return round(out[0], 1), round(out[1], 1)
    lo = hi = 0.0
    frac = RECOIL_MOVES.get(move_id, 0.0)
    if frac and dealt and foe_cur:
        lo += frac * min(dealt[0] / 100 * foe_cur, foe_cur) / cur * 100
        hi += frac * min(dealt[1] / 100 * foe_cur, foe_cur) / cur * 100
    if data.to_id(me.item) == "lifeorb" and dealt and dealt[1] > 0 and data.to_id(me.ability) not in ("magicguard", "sheerforce"):
        orb = 0.1 * me.stats.get("hp", 1) / cur * 100
        lo += orb
        hi += orb
    return round(lo, 1), round(hi, 1)


def post_move_mon(me: "Mon", move_id: str, *, cost: tuple[float, float] = (0.0, 0.0)) -> "Mon":
    """The Pokémon as it stands right after using ``move_id``: its self-inflicted stat stages applied, its recoil /
    Life Orb cost (``cost`` in % of current HP, the average taken) off its HP."""
    boosts = dict(me.boosts)
    for stat, delta in POST_MOVE_STAGES.get(move_id, {}).items():
        boosts[stat] = max(-6, min(6, boosts.get(stat, 0) + delta))
    hp = me.hp_fraction * max(0.0, 1.0 - _avg(cost) / 100)
    return Mon(**{**me.__dict__, "boosts": boosts, "hp_fraction": max(0.0, hp)})


def post_move_threat_hits(sheet: "TurnSheet", slot_number: int, me_after: "Mon", my_priority: int, theirs: dict[int, "Mon"],
                          fs: "FieldState | None", speed_rank: dict, *, doubled: bool = False, ours_count: int = 2) -> list[dict]:
    """The threat rows that land on this slot AFTER it moves (a slower foe, or an equal-priority foe we outspeed),
    recomputed against ``me_after`` (post-move stages and HP); ``doubled`` doubles them (Glaive Rush's drawback). Each
    hit carries ``from``, ``move``, ``damage_pct_of_current_hp``, ``ko``, ``ko_chance`` (accuracy included)."""
    out: list[dict] = []
    for threat in sheet.threats:
        pos = threat["position"]
        foe = theirs.get(pos)
        if foe is None:
            continue
        seen: set[str] = set()
        for hit in threat["hits"]:
            if hit["into_slot"] != slot_number or hit["move"] in seen:
                continue
            seen.add(hit["move"])
            their_priority = hit.get("priority", 0) or 0
            first_by_speed = speed_rank.get(("mine", slot_number), 99) < speed_rank.get(("theirs", pos), 99)
            we_first = my_priority > their_priority or (my_priority == their_priority and first_by_speed)
            if not we_first:
                continue
            info = data.move_info(hit["move"]) or {}
            prof = hit_profile(foe, {**info, "id": hit["move"]}, me_after, field=fs, spread=is_spread(info, foes=ours_count, allies=len(theirs) - 1))
            if prof is None:
                continue
            lo, hi = prof["damage_pct_of_current_hp"]
            if doubled:
                lo, hi = min(lo * 2, 999.0), min(hi * 2, 999.0)
                prof = {**prof, "damage_pct_of_current_hp": [round(lo, 1), round(hi, 1)], "ko": _ko_label(lo, hi)}
                if prof.get("per_hit"):
                    prof["per_hit"] = [prof["per_hit"][0] * 2, prof["per_hit"][1] * 2]
                    if prof.get("first_hit"):
                        prof["first_hit"] = [prof["first_hit"][0] * 2, prof["first_hit"][1] * 2]
            out.append({"from": foe.species, "position": pos, "move": hit["move"], **prof,
                        "ko_chance": round(ko_probability(prof, with_accuracy=True), 3)})
    return out


def _ko_after(hits: list[dict], gone: dict[int, float] | None = None) -> float:
    """Chance at least one foe's best after-move hit KOs the slot (each foe picks its best; both may go for it); a foe
    already taken out before it moves (``gone``, by position) does not swing."""
    by_foe: dict[int, float] = {}
    for h in hits:
        by_foe[h["position"]] = max(by_foe.get(h["position"], 0.0), h.get("ko_chance", 0.0))
    p = 1.0
    for pos, v in by_foe.items():
        p *= 1.0 - v * (1.0 - (gone or {}).get(pos, 0.0))
    return 1.0 - p


def extra_exposure(sheet: "TurnSheet", slot_number: int, me: "Mon", opt: dict, row: dict | None, theirs: dict[int, "Mon"],
                   fs: "FieldState | None", speed_rank: dict, gone: dict[int, float] | None = None) -> tuple[float, list[dict]]:
    """How much MORE likely the slot is to be KO'd after it moves because of the move it uses — its stat drops, its
    recoil / Life Orb, Glaive Rush's doubled damage — than it would be with a move without those effects. (The KO
    odds of the plain after-move rows are the ranking's position-value business.) Returns (extra chance, the post-move
    rows that turned into KOs). L4 T7 (Oct 7): Close Combat put Lucario at -2 Def and Scale Shot went from 85-100% to
    108-129%; the Oct 8 LLM-agent game turn 7: Glaive Rush doubled Flare Blitz into Baxcalibur to 171-201%."""
    move_id = opt.get("move") or ""
    doubled = move_id == GLAIVE_RUSH
    stages = POST_MOVE_STAGES.get(move_id)
    foe = None
    dealt = None
    if row is not None and row.get("target", 0) > 0:
        foe = theirs.get(row["target"])
        dealt = tuple(row.get("damage_pct_of_current_hp") or (0.0, 0.0))
    cost = self_cost_pct(me, move_id, dealt, foe)
    if not doubled and not stages and cost == (0.0, 0.0):
        return 0.0, []
    prio = opt.get("priority", 0) or 0
    before = post_move_threat_hits(sheet, slot_number, me, prio, theirs, fs, speed_rank)
    after = post_move_threat_hits(sheet, slot_number, post_move_mon(me, move_id, cost=cost), prio, theirs, fs, speed_rank, doubled=doubled)
    extra = max(0.0, _ko_after(after, gone) - _ko_after(before, gone))
    was = {(b["position"], b["move"]): b["ko_chance"] for b in before}
    turned = [h for h in after if h["ko_chance"] - was.get((h["position"], h["move"]), 0.0) >= 0.25]  # a KO the plain rows understate
    return extra, turned


# -- the two foes into one of our slots, in move order --------------------------------------------------------------

FRACTION_MOVES = {"ruination", "superfang", "naturesmadness", "guardianofalola"}  # a share of the CURRENT HP, never a KO alone


def compose_double(first: dict, second: dict, me: "Mon") -> tuple[float, float, float]:
    """Two hits into ``me`` in the order they land, as (min%, max%, KO chance) of its HP at the start of the turn. A
    fraction move (Ruination: half of what is LEFT) is applied to the HP remaining after the earlier hit, never added as
    a flat 50% (the Oct 8 practice game: six impossible 'goes down' warnings from Scale Shot + Ruination); a Sitrus
    Berry heals a quarter of max HP between the hits once the first drops it to half; accuracy enters the KO odds."""
    sitrus = data.to_id(me.item) == "sitrusberry"
    max_in_current = 100.0 / max(me.hp_fraction, 0.01)  # max HP in % of current

    def after(a: float, b: float, second_is_fraction: bool) -> float:
        """Total damage of the two hits (uncapped, so an overkill keeps its margin), in % of the HP at the start."""
        remaining = 100.0 - a
        heal = 0.0
        if 0 < remaining <= 0.5 * max_in_current and sitrus:
            heal = min(0.25 * max_in_current, max_in_current - remaining)
        remaining += heal
        taken = max(remaining, 0.0) / 2 if second_is_fraction else b
        return a - heal + taken

    a_lo, a_hi = first["damage_pct_of_current_hp"]
    b_lo, b_hi = second["damage_pct_of_current_hp"]
    second_frac = second["move"] in FRACTION_MOVES
    lo, hi = after(a_lo, b_lo, second_frac), after(a_hi, b_hi, second_frac)
    acc_a, acc_b = float(first.get("hit_chance", 1.0)), float(second.get("hit_chance", 1.0))
    # A fraction move never finishes: with it second, only the first hit's own KO counts.
    both = acc_a * acc_b * (ko_probability(first) if second_frac else linear_ko(lo, hi))
    p = both + acc_a * (1 - acc_b) * ko_probability(first) + (1 - acc_a) * acc_b * ko_probability(second)
    return round(lo, 1), round(hi, 1), round(min(1.0, p), 3)


def switch_value(species: str, theirs: dict[int, "Mon"], memory: "MatchMemory") -> float:
    """How good a bench Pokémon is to bring in against their current actives (offense + resists)."""
    from .draft import defense_vs, hit_quality, profile  # local import: draft imports nothing from here

    card = memory.my_cards.get(species_key(species)) or {"species": species}
    me = profile(card)
    foes = [profile(memory.opp_cards.get(species_key(opp.species)) or {"species": opp.species}) for opp in theirs.values()]
    if not foes:
        return me.bst / 100.0
    offense = sum(hit_quality(me, foe) for foe in foes) / len(foes)
    return 10 * offense + 5 * defense_vs(me, foes) + me.bst / 200.0


def _preview_for(sheet: "TurnSheet | None", species: str, slot_number: int) -> dict:
    preview = (sheet.switch_ins.get(species_key(species)) if sheet is not None else None) or {}
    if isinstance(preview.get("by_slot"), dict):
        preview = {**preview, **preview["by_slot"].get(slot_number, {})}
    return preview


def best_switch(slot: dict, theirs: dict[int, "Mon"], memory: "MatchMemory", *, exclude: str | None = None,
                sheet: "TurnSheet | None" = None) -> tuple[int | None, str | None]:
    """(option index, species) of the best switch option in this slot, skipping ``exclude``. Uses the sheet's
    switch-in preview (real damage both ways against both foes, hazards, entry effects, speed; the slot's own
    pricing when the two slots differ) when there is one, else the type-chart estimate."""
    best: tuple[int | None, str | None, float] = (None, None, float("-inf"))
    for index, option in enumerate(slot.get("options") or []):
        if option.get("type") != "switch" or option.get("species") == exclude:
            continue
        preview = _preview_for(sheet, option["species"], slot.get("slot", 0))
        value = preview["score"] if preview else switch_value(option["species"], theirs, memory)
        if value > best[2]:
            best = (index, option["species"], value)
    return best[0], best[1]


def safe_switch(slot: dict, theirs: dict[int, "Mon"], memory: "MatchMemory", *, sheet: "TurnSheet | None" = None,
                exclude: str | None = None) -> tuple[int | None, str | None, dict, bool]:
    """The switch option to offer for this slot, from mech's switch-in verdict (both foes, their double, hazards, entry
    effects): (index, species, preview, safe). The best 'safe' switch-in first; when every bench card is KO'd on entry
    half the time or more, the least bad one with ``safe`` False, so the row can say so. (Oct 8 practice game, turn 12:
    Milotic was sent into Spikes + Wood Hammer; the pivot check had looked at one attacker.)"""
    best_safe: tuple[int | None, str | None, dict, float] = (None, None, {}, float("-inf"))
    best_any: tuple[int | None, str | None, dict, float] = (None, None, {}, float("-inf"))
    for index, option in enumerate(slot.get("options") or []):
        if option.get("type") != "switch" or not option.get("species") or option.get("species") == exclude:
            continue
        preview = _preview_for(sheet, option["species"], slot.get("slot", 0))
        value = preview["score"] if preview else switch_value(option["species"], theirs, memory)
        if value > best_any[3]:
            best_any = (index, option["species"], preview, value)
        if (preview.get("verdict", "safe") == "safe") and value > best_safe[3]:
            best_safe = (index, option["species"], preview, value)
    if best_safe[0] is not None:
        return best_safe[0], best_safe[1], best_safe[2], True
    return best_any[0], best_any[1], best_any[2], False


def _switch_entry_cost(preview: dict, species: str, sheet: "TurnSheet", memory: "MatchMemory", P: dict) -> float:
    """What offering this switch-in risks, in damage points: its chance of a KO on entry times what it is worth."""
    odds = float(preview.get("ko_on_entry", 0.0) or 0.0)
    if odds <= 0:
        return 0.0
    hp = sheet.bench_hp.get(species_key(species), 1.0)
    return odds * (100.0 * hp + P["ko_bonus_guaranteed"])


PROTECT_OPTIONS = ("protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap")
COVER_ROWS = {"protect_threatened", "switch_threatened", "protect_double_target", "protect_and_switch", "redirect_to_protect_partner"}


def rank_candidates(slots: list[dict], ours: dict[int, "Mon"], theirs: dict[int, "Mon"], sheet: TurnSheet,
                    fresh: set[str] | None = None, memory: "MatchMemory | None" = None, params: dict | None = None) -> list[dict]:
    """A few complete turns worth considering, best first. Each is {name, slot_0, slot_1, why, score}."""
    P = params or DEFAULTS
    by_slot = {e["slot"]: e for e in sheet.our_options}
    fresh = fresh or set()
    speed_rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    candidates: list[dict] = []
    weights = threat_weights(sheet, ours, theirs, P)

    def slot_answer(option_index: int | None, target: int | None) -> dict | None:
        if option_index is None:
            return None
        return {"option": option_index, "target": target if target is not None else 0}

    def pass_only(number: int) -> dict | None:
        """The answer for a slot that can only pass (our last Pokémon stands alone, the other board position is empty).
        The live server offers such a slot every end-game; without this every candidate was dropped and the adapter's
        first-option default played (seen in the first live match, Oct 7)."""
        slot = slot_template(number)
        if not slot:
            return None
        options = slot.get("options") or []
        if options and all(o.get("type") == "pass" for o in options):
            return {"option": 0, "target": 0}
        return None

    lone: dict[int, dict | None] = {0: None, 1: None}  # filled once slot_template exists, below

    def add(name: str, s0: dict | None, s1: dict | None, why: str, score: float, **extra: Any) -> None:
        if s0 is None and lone[0]:
            s0 = lone[0]
        if s1 is None and lone[1]:
            s1 = lone[1]
        if s0 is None or s1 is None:
            return
        cand = {"name": name, "slot_0": s0, "slot_1": s1, "why": why, "score": round(score, 1), **extra}
        for existing in candidates:
            if existing["slot_0"] == s0 and existing["slot_1"] == s1:
                if cand["score"] > existing["score"]:
                    existing.clear()
                    existing.update(cand)  # the same turn, better explained and better scored: keep that one (D1j)
                return
        candidates.append(cand)

    def slot_template(number: int) -> dict | None:
        return next((s for s in slots if s.get("slot", 0) == number), None)

    lone.update({n: pass_only(n) for n in (0, 1)})

    def acts_before(me_slot: int, opp_pos: int, my_priority: int = 0, their_priority: int = 0) -> bool:
        if my_priority != their_priority:
            return my_priority > their_priority
        return speed_rank.get(("mine", me_slot), 99) < speed_rank.get(("theirs", opp_pos), 99)

    def rows_into(opt: dict, pos: int) -> list[dict]:
        """Every damage row of an option that reaches the foe at ``pos`` (its single-target row, or its spread row)."""
        if pos not in theirs:
            return []
        out = []
        for r in opt.get("targets") or []:
            if r.get("side") != "theirs":
                continue
            if r.get("target") == pos or (r.get("target", 0) == 0 and species_key(r.get("species")) == species_key(theirs[pos].species)):
                out.append(r)
        return out

    def removal_chance(number: int, opp_pos: int, their_priority: int) -> float:
        """Chance this slot takes out the foe at ``opp_pos`` before that foe moves, over ALL its options (not only its
        best attack: Oct 7, Tornadus was told to Protect from Iron Hands while Great Tusk's Headlong Rush KOs Iron
        Hands first ~9 times in 10; Oct 8: the partner's +1 Ice Shard), times the chance it survives to act and is not
        flinched (``partner_removes`` read only the best attack and never checked that the partner survives)."""
        best_p = 0.0
        for opt in by_slot.get(number, {}).get("options") or []:
            if opt.get("fails"):
                continue
            prio = opt.get("priority", 0) or 0
            if not acts_before(number, opp_pos, prio, their_priority):
                continue
            for r in rows_into(opt, opp_pos):
                p = ko_probability(r, with_accuracy=True)
                if p <= best_p:
                    continue
                p *= survival_factor(number, sheet, prio, speed_rank, r, P)
                best_p = max(best_p, p)
        return best_p

    def partner_removes(other: int, opp_pos: int, their_priority: int) -> float:
        if opp_pos not in theirs or other not in ours:
            return 0.0
        return removal_chance(other, opp_pos, their_priority)

    def best_action(number: int, *, allow_ally_only: bool = True) -> tuple[dict | None, dict | None, float]:
        """Best attack or status move, as (option, row, value): the ATTACK is the option that keeps the most value once
        its chance to act is counted (the strongest raw move, discounted afterwards, picked a Kowtow Cleave worth 78 x 0
        over a Sucker Punch worth 64 x 1 in the Oct 8 loss, turn 6); the raw best stands when every option is worth 0.
        ``allow_ally_only`` off leaves out Helping Hand / Coaching / Follow Me: next to a partner that Protects or
        switches they do nothing (Oct 7 match 1 turn 5: Protect + Helping Hand was code #1)."""
        entry = by_slot.get(number, {})
        raw = _best_attack(entry, P=P, threats=weights)
        choice: tuple[dict | None, dict | None, float] = (None, None, -1.0)
        if raw[0]:
            best_disc = (raw[0], raw[1], raw[2] * survival_factor(number, sheet, raw[0].get("priority", 0) or 0, speed_rank, raw[1], P))
            for opt in entry.get("options") or []:
                if opt.get("fails") or not opt.get("targets"):
                    continue
                prio = opt.get("priority", 0) or 0
                one = _best_attack({"options": [opt]}, P=P, threats=weights)
                if not one[0]:
                    continue
                value = one[2] * survival_factor(number, sheet, prio, speed_rank, one[1], P)
                if value > best_disc[2] + 1e-9:
                    best_disc = (one[0], one[1], value)
            choice = best_disc
        slot = slot_template(number)
        me = ours.get(number)
        if slot and me:
            for index, option in enumerate(slot.get("options") or []):
                if option.get("type") != "move":
                    continue
                move_id = data.to_id(option.get("move_id"))
                info = data.move_info(move_id) or {}
                if data.is_damaging(info) or move_id in ("protect", "detect", "fakeout"):
                    continue
                if not allow_ally_only and move_id in ALLY_ONLY:
                    continue
                prio = info.get("priority", 0) or 0
                if data.to_id(me.ability) == "prankster" and info.get("category") == "Status":
                    prio += 1  # Prankster: status moves get +1 priority (Tailwind before their attacks land)
                sv, target = status_value(move_id, me, theirs, sheet, number, has_ally=len(ours) > 1)
                if data.to_id(me.ability) == "prankster" and info.get("target") in ("normal", "any", "adjacentFoe") \
                        and target in theirs and "dark" in theirs[target].types:
                    sv = 0.0  # a Prankster-boosted status move fails on a Dark-type target
                sv *= survival_factor(number, sheet, prio, speed_rank, None, P) if prio <= 0 else 1.0
                if sv > 0 and sv > choice[2]:
                    legal = option.get("targets") or []
                    tgt = target if target in legal else (legal[0] if legal else 0)
                    pseudo = {"option": index, "move": move_id, "priority": prio}
                    choice = (pseudo, {"target": tgt, "species": theirs[tgt].species if tgt in theirs else "field", "side": "theirs",
                                       "damage_pct_of_current_hp": [0, 0], "status_value": round(sv, 1)}, sv)
        return choice

    best = {n: best_action(n) for n in (0, 1)}
    solo = {n: best_action(n, allow_ally_only=False) for n in (0, 1)}  # what a slot does while its partner Protects / switches
    for n in (0, 1):
        opt = best[n][0]
        if opt and P.get("choice_lock_rerank", 1) and sheet.choice_locks.get(n, {}).get(opt.get("move")):
            # A Choice-locked move with a dead target left: prefer the best option that stays useful, if it is close.
            usable = _best_attack({"options": [o for o in by_slot.get(n, {}).get("options") or [] if o.get("move") not in sheet.choice_locks[n]]}, P=P, threats=weights)
            if usable[0] and usable[2] >= 0.6 * max(best[n][2], 1e-6):
                best[n] = (usable[0], usable[1], usable[2] * survival_factor(n, sheet, usable[0].get("priority", 0), speed_rank, usable[1], P))

    # D1l: never a recoil / Life Orb self-KO with our last Pokémon when the KO ends the game (a tie scores 0 in Swiss).
    bench = {species_key(o.get("species")) for s in slots for o in s.get("options") or [] if o.get("type") == "switch" and o.get("species")}
    our_remaining = len(ours) + len(bench)
    fainted_theirs = sheet.field.fainted.get("theirs", 0) if sheet.field is not None else 0
    their_remaining = max(len(theirs), 4 - fainted_theirs)
    if our_remaining == 1 and their_remaining == len(theirs):
        for n in list(ours):
            opt, row, value = best[n]
            me = ours[n]
            if not opt or not row or "status_value" in row:
                continue
            kos = [pos for pos in theirs if any(ko_probability(r) > 0 for r in rows_into(opt, pos))]
            if set(kos) != set(theirs):
                continue
            foe = theirs.get(row.get("target", 0))
            cost = self_cost_pct(me, opt.get("move") or "", tuple(row.get("damage_pct_of_current_hp") or (0, 0)), foe)
            if cost[1] < 100:
                continue
            safer = None
            for other in by_slot.get(n, {}).get("options") or []:
                if other.get("fails") or other is opt:
                    continue
                one = _best_attack({"options": [other]}, P=P, threats=weights)
                if not one[0]:
                    continue
                c2 = self_cost_pct(me, other.get("move") or "", tuple(one[1].get("damage_pct_of_current_hp") or (0, 0)), theirs.get(one[1].get("target", 0)))
                if c2[1] < 100 and (safer is None or one[2] > safer[2]):
                    safer = one
            if safer:
                best[n] = (safer[0], safer[1], safer[2] * survival_factor(n, sheet, safer[0].get("priority", 0) or 0, speed_rank, safer[1], P))
                sheet.notes.append(f"TIE GUARD: {me.species} is our last Pokémon and {opt.get('move')} could knock it out with its own recoil "
                                   f"({cost[0]}-{cost[1]}% of its HP) while finishing their last Pokémon: a tie scores nothing, so {safer[0]['move']} instead.")

    # 0. Forced switches (a Pokémon fainted): bring in the best matchup; the other slot keeps its best action or passes.
    forced = [n for n in (0, 1) if (slot_template(n) or {}).get("force_switch")]
    if forced and memory is not None:
        answers: dict[int, dict] = {}
        used: str | None = None
        for n in forced:
            idx, species = best_switch(slot_template(n), theirs, memory, exclude=used, sheet=sheet)
            if idx is None:
                pass_idx = _option_index(slot_template(n), "pass")
                answers[n] = {"option": pass_idx, "target": 0} if pass_idx is not None else None
            else:
                answers[n] = {"option": idx, "target": 0}
                used = species
        for n in (0, 1):
            if n in answers:
                continue
            slot = slot_template(n)
            pass_idx = _option_index(slot, "pass")
            if best[n][0] is not None and best[n][1] is not None and pass_idx is None:
                answers[n] = slot_answer(best[n][0]["option"], best[n][1]["target"])
            elif pass_idx is not None:
                answers[n] = {"option": pass_idx, "target": 0}
        if all(answers.get(n) for n in (0, 1)):
            add("forced_switch", answers[0], answers[1], f"replace fainted slot(s) {forced} with the best matchup", 1000)

    # Spell out every KO threat that moves before the slot it targets (priority ignores speed and Tailwind). The speed
    # comparison uses the slot's ordinary speed, never the priority of whatever move the code happens to like: the model
    # may pick a different move, and a warning that silently assumed Fake Out or Follow Me has cost games. A priority
    # move is an escape only when it (or the partner, faster) takes the threat out first at least half the time: Helping
    # Hand and Follow Me never are, and a +2 Extreme Speed for 39-46% is not either (D1i, the Oct 8 practice game).
    endangered: set[int] = set()
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["ko"] == "no":
                continue
            number = hit["into_slot"]
            me = ours.get(number)
            slot = slot_template(number) or {}
            their_prio = hit.get("priority", 0) or 0
            first_normally = acts_before(number, threat["position"], 0, their_prio)
            if first_normally:
                # Slower than us: it only lands if we leave the slot in and fail to remove it. Say so (not a warning).
                if hit["ko"] == "guaranteed" or ko_probability(hit) >= 0.5:
                    endangered.add(number)
                    sheet.notes.append(
                        f"EXPOSED: slot {number} ({me.species if me else '?'}) moves before {threat['species']}, but {threat['species']}'s "
                        f"{hit['move']} KOs it ({hit['damage_pct_of_current_hp'][0]}-{hit['damage_pct_of_current_hp'][1]}%) afterwards unless "
                        f"{threat['species']} is removed, flinched, redirected or this slot Protects/switches.")
                continue
            escape = ""
            best_escape = (0.0, "")
            for option in by_slot.get(number, {}).get("options") or []:
                prio = option.get("priority", 0) or 0
                if option.get("fails") or prio <= 0 or not acts_before(number, threat["position"], prio, their_prio):
                    continue
                for r in rows_into(option, threat["position"]):
                    p = ko_probability(r, with_accuracy=True)
                    if p > best_escape[0]:
                        best_escape = (p, f"{option['move']} (+{prio})")
            partner_p = partner_removes(1 - number, threat["position"], their_prio) if (1 - number) in ours else 0.0
            if best_escape[0] >= 0.5:
                escape = f" (this slot's {best_escape[1]} goes first and KOs it {int(round(best_escape[0] * 100))}% of the time)"
            elif partner_p >= 0.5:
                escape = f" (slot {1 - number} can take it out first, {int(round(partner_p * 100))}% of the time)"
            prio_note = f" with priority +{hit['priority']}" if their_prio > 0 else ""
            chance = ko_probability(hit)
            where = f"slot {number} ({me.species if me else '?'})"
            dmg = f"{hit['damage_pct_of_current_hp'][0]}-{hit['damage_pct_of_current_hp'][1]}%"
            if hit["ko"] == "guaranteed" or chance >= 0.5:
                endangered.add(number)
                advice = ("This is our last Pokémon and nothing runs out in our favour: Protect only delays the same KO by a "
                          "turn, so attack." if sheet.alone and not sheet.stall_reasons else
                          "Protect/switch that slot unless trading it wins the game.")
                sheet.warnings.append(
                    f"LETHAL: {threat['species']}'s {hit['move']}{prio_note} hits {where} for {dmg} BEFORE it can move"
                    + (f" ({int(round(chance * 100))}% KO chance)" if hit["ko"] != "guaranteed" else "")
                    + f"{escape}. {advice}")
            elif chance >= 0.2:
                sheet.warnings.append(
                    f"HIGH RISK: {threat['species']}'s {hit['move']}{prio_note} hits {where} for {dmg} before it can move: a "
                    f"{int(round(chance * 100))}% chance of losing it on the roll{escape}. Protect/switch unless the attack is worth that risk.")
            else:
                sheet.warnings.append(
                    f"RISK: {threat['species']}'s {hit['move']}{prio_note} hits {where} for {dmg} before it can move: only a "
                    f"{int(round(chance * 100))}% chance of a KO on the roll{escape}. Usually worth playing through, not Protecting.")
    for focus in sheet.focus_threats:
        if focus["ko_chance"] >= 0.5:
            endangered.add(focus["slot"])

    # What our own moves do to us before the slower threats land (A3 / D1h): a hit that becomes a KO only because of the
    # move's stat drops, recoil or Glaive Rush gets an EXPOSED AFTER note (Oct 7 match 4 turn 7: Lucario at -2 Def).
    for number, me in sorted(ours.items()):
        for opt in by_slot.get(number, {}).get("options") or []:
            move_id = opt.get("move") or ""
            if opt.get("fails") or not (move_id in POST_MOVE_STAGES or move_id == GLAIVE_RUSH or move_id in RECOIL_MOVES or data.to_id(me.item) == "lifeorb"):
                continue
            one = _best_attack({"options": [opt]}, P=P, threats=weights)
            if not one[0]:
                continue
            _, turned = extra_exposure(sheet, number, me, opt, one[1], theirs, sheet.field, speed_rank)
            for h in turned[:2]:
                cause = ("Glaive Rush doubles the damage it takes" if move_id == GLAIVE_RUSH else
                         ", ".join(f"{d:+d} {s.upper()}" for s, d in POST_MOVE_STAGES.get(move_id, {}).items()) or "its recoil")
                sheet.notes.append(
                    f"EXPOSED AFTER {move_id}: slot {number} ({me.species}) would take {h['from']}'s {h['move']} for "
                    f"{h['damage_pct_of_current_hp'][0]}-{h['damage_pct_of_current_hp'][1]}% afterwards ({cause}): a KO "
                    f"{int(round(100 * h['ko_chance']))}% of the time that the plain rows do not show.")

    def can_protect(foe: "Mon") -> bool:
        return any(m["id"] in PROTECT_LIKE for m in foe.moves) and not (memory is not None and memory.opp_protect_streak.get(species_key(foe.species)))

    def joint_value(picks: list[tuple[int, dict | None, dict | None, float]], *, protect_aware: bool = False) -> float:
        """Both slots' actions valued TOGETHER. Damage into each foe is pooled across the slots, each hit capped at a KO
        before it is weighted by its slot's chance to act (an overkill at a 50% flinch is half a KO, not a whole one),
        one KO bonus per foe scaled by the threat it removes. A foe one slot takes out before it moves no longer
        threatens the other slot (the partner's priority or faster sure KO is credited). Each attacking slot is charged
        for what its own move does to it before the slower threats land (stat drops, recoil, Glaive Rush). With
        ``protect_aware`` a target that can Protect (and didn't last turn) and is in KO range is discounted by the
        chance it does (the Oct 7 opponent Protected the threatened Pokémon three times in four games)."""
        attacks = [(n, opt, row) for n, opt, row, _ in picks if opt and row and "status_value" not in row and opt.get("targets")]
        extra = sum(max(v, 0.0) for _, opt, row, v in picks if opt and row and ("status_value" in row or not opt.get("targets")))
        removed: dict[int, float] = {}
        own: dict[tuple[int, int], float] = {}  # (slot, foe) -> the slot's own chance of taking that foe out before it moves
        for pos, foe in theirs.items():
            parts = []
            for n, opt, row in attacks:
                sub = row_into(opt, row, pos, foe.species)
                if sub is None:
                    continue
                prio = opt.get("priority", 0) or 0
                if acts_before(n, pos, prio, their_priority_into(sheet, pos)):
                    # this hit lands before the foe moves: only the OTHER foe's faster threats and a flinch can stop it
                    p_act = survival_factor(n, sheet, prio, speed_rank, row, P, ignore_pos=pos) * float(sub.get("hit_chance", 1.0))
                    parts.append((p_act, sub))
                    own[(n, pos)] = p_act * ko_probability(sub)
            removed[pos] = _outcome(parts)[1] if parts else 0.0
        surv = {n: survival_factor(n, sheet, opt.get("priority", 0) or 0, speed_rank, row, P, removed=removed) for n, opt, row in attacks}
        total = 0.0
        for pos, foe in theirs.items():
            parts = []
            for n, opt, row in attacks:
                sub = row_into(opt, row, pos, foe.species)
                if sub is not None:
                    parts.append((surv[n] * float(sub.get("hit_chance", 1.0)), sub))
            if not parts:
                continue
            damage, p_ko = _outcome(parts)
            if P.get("hp_units", 0):
                damage *= foe.hp_fraction
            v = damage + p_ko * (P["ko_bonus_guaranteed"] + P.get("ko_threat_w", 0.0) * weights.get(species_key(foe.species), 0.0))
            if protect_aware and p_ko > 0 and can_protect(foe):
                v *= 1.0 - P.get("opp_protect_risk", 0.3)
            total += v
        for n, opt, row in attacks:
            prio = opt.get("priority", 0) or 0
            for r in opt.get("targets") or []:
                if r.get("side") == "mine" and r.get("target") == 0:
                    extra -= P["ally_damage_w"] * min(_avg(r["damage_pct_of_current_hp"]), 100) * surv[n]
            extra += P["priority_w"] * prio * surv[n]
            me = ours.get(n)
            if me is not None and P.get("exposure_w", 0):
                gone = {pos: 1.0 - (1.0 - removed.get(pos, 0.0)) * (1.0 - own.get((n, pos), 0.0)) for pos in theirs}
                more, _ = extra_exposure(sheet, n, me, opt, row, theirs, sheet.field, speed_rank, gone=gone)
                doomed = doomed_after_priority(n, sheet, prio, speed_rank, row, removed)  # D1i: the hit lands, the Pokémon is lost
                if more > 0 or doomed > 0:
                    extra -= P["exposure_w"] * min(1.0, more + doomed) * material(me, P) * surv[n]
        return total + extra

    def solo_value(n: int) -> float:
        """The slot's best action next to a partner that Protects or switches (no Helping Hand into it), valued on its
        own: its survival, its KO credit, what its move costs it."""
        return joint_value([(n, *solo[n])]) if solo[n][0] else 0.0

    def solo_answer(n: int) -> dict | None:
        return slot_answer(solo[n][0]["option"], solo[n][1]["target"]) if solo[n][0] and solo[n][1] else None

    # 1. Both slots use their own best attack.
    if best[0][0] and best[1][0]:
        add("best_attacks", slot_answer(best[0][0]["option"], best[0][1]["target"]), slot_answer(best[1][0]["option"], best[1][1]["target"]),
            f"slot 0 {best[0][0]['move']} -> {best[0][1]['species']} ({best[0][1]['damage_pct_of_current_hp']}%), "
            f"slot 1 {best[1][0]['move']} -> {best[1][1]['species']} ({best[1][1]['damage_pct_of_current_hp']}%)",
            joint_value([(0, *best[0]), (1, *best[1])]))
    elif any(lone.values()):
        n = 0 if lone[1] else 1  # the slot that still fights
        if best[n][0]:
            add("best_attack_alone", slot_answer(best[n][0]["option"], best[n][1]["target"]) if n == 0 else None,
                slot_answer(best[n][0]["option"], best[n][1]["target"]) if n == 1 else None,
                f"slot {n} {best[n][0]['move']} -> {best[n][1]['species']} ({best[n][1]['damage_pct_of_current_hp']}%); the other slot can only pass",
                joint_value([(n, *best[n])]))

    # 2. Focus fire: both into the same target when together they likely KO it.
    for target, opp in theirs.items():
        rows = []
        for number in (0, 1):
            other_target = next((t for t in theirs if t != target), None)
            opt, row, exp = _best_attack(by_slot.get(number, {}), avoid_target=other_target, P=P, threats=weights)
            if opt and row and row["target"] == target:
                exp *= survival_factor(number, sheet, opt.get("priority", 0), speed_rank, row, P)
                rows.append((number, opt, row, exp))
        if len(rows) == 2:
            # only count damage from slots that actually get to move
            combined = sum(r["damage_pct_of_current_hp"][0] * (1 if e > 0 else 0) for _, _, r, e in rows)
            if combined >= 100:
                guard = can_protect(opp)
                # The bonus is for what the SECOND hit adds: when one hit alone already takes the target out, the other is
                # overkill, and the split (finish_the_other) is the better turn (the Oct 8 LLM-agent game, turn 5).
                single = max(ko_probability(r, with_accuracy=True) for _, _, r, _ in rows)
                add("focus_fire", slot_answer(rows[0][1]["option"], target), slot_answer(rows[1][1]["option"], target),
                    f"both into {opp.species}: {combined:.0f}% minimum combined, likely KO" + (" (it can Protect)" if guard else ""),
                    joint_value([(n, o, r, e) for n, o, r, e in rows], protect_aware=True)
                    + P["focus_fire_bonus"] * (1.0 - single) * ((1.0 - P.get("opp_protect_risk", 0.3)) if guard else 1.0))

    # 2a. Priority focus: both slots' priority moves into a foe that KOs one of them before it moves, when together they
    #     take it out first (the Oct 8 loss, turn 6: Extreme Speed + Sucker Punch KO'd Lucario before its Close Combat;
    #     the code's row was a second Protect in a row). Sucker Punch counts only against a foe that is attacking.
    for threat in sheet.threats:
        pos = threat["position"]
        lethal = [h for h in threat["hits"] if h["ko"] != "no" and not acts_before(h["into_slot"], pos, 0, h.get("priority", 0) or 0)
                  and (h["ko"] == "guaranteed" or ko_probability(h) >= 0.5)]
        if not lethal or len(ours) < 2:
            continue
        their_prio = max(h.get("priority", 0) or 0 for h in lethal)
        picks = []
        for number in (0, 1):
            pick = None
            for opt in by_slot.get(number, {}).get("options") or []:
                prio = opt.get("priority", 0) or 0
                if opt.get("fails") or prio <= 0 or not acts_before(number, pos, prio, their_prio):
                    continue
                for r in rows_into(opt, pos):
                    value = _expected(r, prio, P, weights.get(species_key(r["species"]), 0.0))
                    if pick is None or value > pick[3]:
                        pick = (number, opt, r, value)
            if pick:
                picks.append(pick)
        if len(picks) < 2:
            continue
        combined_ko = _outcome([(float(r.get("hit_chance", 1.0)), r) for _, _, r, _ in picks])[1]
        if combined_ko < 0.5:
            continue
        answers = {n: slot_answer(o["option"], r["target"] if r.get("target", 0) > 0 else 0) for n, o, r, _ in picks}
        conditional = [o["move"] for _, o, _, _ in picks if getattr(data, "needs_target_attacking", lambda m: False)(o["move"])]
        add("priority_focus", answers.get(0), answers.get(1),
            f"{threat['species']} KOs slot {lethal[0]['into_slot']} before it moves; slot {picks[0][0]} {picks[0][1]['move']} (+{picks[0][1]['priority']}) + "
            f"slot {picks[1][0]} {picks[1][1]['move']} (+{picks[1][1]['priority']}) take it out first {int(round(combined_ko * 100))}% of the time"
            + (f" ({', '.join(conditional)} fails if it Protects, switches or uses a status move instead)" if conditional else ""),
            joint_value([(n, o, r, v) for n, o, r, v in picks]) + P["focus_fire_bonus"] * combined_ko, covers=lethal[0]["into_slot"])

    # 2b. Don't overkill: when one slot's best attack already guarantees a KO on a target, point the other slot at the
    # remaining foe. If that foe's Focus Sash gets broken by the first slot's spread hit (and it moves first), the second
    # hit is a full one, so recompute it against the HP that will actually be left.
    for killer, other in ((0, 1), (1, 0)):
        k_opt, k_row, k_val = best[killer]
        if not k_opt or not k_row or "status_value" in k_row:
            continue
        pos_by_species = {species_key(m.species): pos for pos, m in theirs.items()}
        killer_rows = []
        for r in k_opt.get("targets") or []:
            if r.get("side") != "theirs":
                continue
            pos = r["target"] if r["target"] > 0 else pos_by_species.get(species_key(r["species"]))
            if pos in theirs and (k_row.get("target", 0) <= 0 or pos == k_row["target"]):
                killer_rows.append({**r, "pos": pos})
        kos = [r["pos"] for r in killer_rows if r["damage_pct_of_current_hp"][0] >= 100]
        if not kos:
            continue
        remaining = [t for t in theirs if t not in kos]
        if not remaining or not slot_template(other) or not ours.get(other):
            continue
        victim, survivor = kos[0], remaining[0]
        alt_opt, alt_row, alt_val = _best_attack(by_slot.get(other, {}), avoid_target=victim, P=P, threats=weights)
        if not alt_opt or not alt_row or alt_row.get("target") != survivor:
            continue
        why_extra = ""
        chip = next((r for r in killer_rows if r["pos"] == survivor), None)
        foe = theirs[survivor]
        k_prio, o_prio = k_opt.get("priority", 0) or 0, alt_opt.get("priority", 0) or 0
        killer_first = k_prio > o_prio or (k_prio == o_prio and speed_rank.get(("mine", killer), 99) < speed_rank.get(("mine", other), 99))
        if chip and chip["damage_pct_of_current_hp"][0] > 0 and killer_first and data.to_id(foe.item) == "focussash" and foe.hp_fraction >= 0.999:
            after = Mon(**{**foe.__dict__, "hp_fraction": max(0.01, foe.hp_fraction * (1 - chip["damage_pct_of_current_hp"][0] / 100))})
            info = data.move_info(alt_opt["move"]) or {}
            est = damage_percent(ours[other], {**info, "id": alt_opt["move"]}, after, field=sheet.field, spread=bool(alt_opt.get("spread")))
            if est:
                alt_row = {**alt_row, "damage_pct_of_current_hp": [est[0], est[1]], "ko": "guaranteed" if est[0] >= 100 else ("possible" if est[1] >= 100 else "no")}
                alt_val = _expected(alt_row, alt_opt.get("priority", 0) or 0, P, weights.get(species_key(foe.species), 0.0))
                why_extra = f" (its Focus Sash is broken by slot {killer}'s {k_opt['move']} first, so this hit is a full {est[0]}-{est[1]}%)"
        answers = {killer: slot_answer(k_opt["option"], k_row["target"]), other: slot_answer(alt_opt["option"], survivor)}
        add("finish_the_other", answers[0], answers[1],
            f"slot {killer} {k_opt['move']} already KOs {theirs[victim].species}; slot {other} {alt_opt['move']} goes into {foe.species} ({alt_row['damage_pct_of_current_hp']}%){why_extra}",
            joint_value([(killer, k_opt, k_row, k_val), (other, alt_opt, alt_row, alt_val)]) + P["focus_fire_bonus"] * 0.5)

    def protect_index(slot: dict | None) -> int | None:
        return next((i for i, o in enumerate((slot or {}).get("options") or []) if o.get("type") == "move"
                     and data.to_id(o.get("move_id")) in PROTECT_OPTIONS), None)

    def switch_row(endangered: int, other: int, label: str, why: str, base_score: float, *, allow_unsafe: bool) -> None:
        """A switch for the endangered slot while the other slot attacks, from mech's verdict: the best safe switch-in;
        when none survives entry, the least bad one, priced for the KO it risks and saying so (never offered at all
        when ``allow_unsafe`` is off: switching just donates a Pokémon)."""
        slot = slot_template(endangered)
        if not slot or memory is None or not solo[other][0]:
            return
        idx, species, preview, safe = safe_switch(slot, theirs, memory, sheet=sheet)
        if idx is None or not species:
            return
        if not safe:
            if not allow_unsafe:
                return
            odds = int(round(100 * float(preview.get("ko_on_entry", 0.0) or 0.0)))
            why += f"; no bench card survives entry: {species} is KO'd on entry {odds}% of the time"
            base_score -= _switch_entry_cost(preview, species, sheet, memory, P)
        answers = {endangered: {"option": idx, "target": 0}, other: solo_answer(other)}
        add(label, answers[0], answers[1], why, base_score, covers=endangered)

    # 3. A slot faces a KO this turn from something that acts before it: Protect it while the other attacks.
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["ko"] not in ("guaranteed", "possible"):
                continue
            endangered_slot, other = hit["into_slot"], 1 - hit["into_slot"]
            my_best = best[endangered_slot]
            my_prio = my_best[0].get("priority", 0) if my_best[0] else 0
            can_remove_first = (my_best[0] and my_best[1] and my_best[1].get("target") == threat["position"]
                                and my_best[1]["damage_pct_of_current_hp"][0] >= 100
                                and acts_before(endangered_slot, threat["position"], my_prio, hit.get("priority", 0)))
            flinch = sheet.flinch_risk.get(endangered_slot, 0.0) if my_prio < 3 else 0.0
            if can_remove_first and not flinch:
                continue  # we KO it before it moves (and nothing can stop us moving)
            if sheet.alone and not sheet.stall_reasons:
                continue  # our last Pokémon: a Protect only delays the same position by a turn
            slot = slot_template(endangered_slot)
            protect = None
            attacker = theirs.get(threat["position"])
            move_info = data.move_info(hit["move"]) or {}
            pierces_protect = (attacker is not None and data.to_id(attacker.ability) == "unseenfist" and "contact" in (move_info.get("flags") or [])) \
                or hit["move"] in ("feint", "hyperspacefury", "hyperspacehole", "phantomforce", "shadowforce")
            if slot and not pierces_protect:
                protect = protect_index(slot)
            if pierces_protect:
                note = f"{threat['species']}'s {hit['move']} goes THROUGH Protect (Unseen Fist / Protect-piercing move): switch slot {endangered_slot} out instead of protecting."
                if note not in sheet.warnings:
                    sheet.warnings.append(note)
            me_e = ours.get(endangered_slot)
            streak = memory.protect_streak(me_e.species, endangered_slot) if (me_e is not None and memory is not None) else 0
            removed = partner_removes(other, threat["position"], hit.get("priority", 0))
            when = "before it moves" if not acts_before(endangered_slot, threat["position"], 0, hit.get("priority", 0)) else "after it moves"
            if protect is not None and solo[other][0]:
                answers = {endangered_slot: {"option": protect, "target": 0}, other: solo_answer(other)}
                bonus = P["protect_bonus_guaranteed"] if hit["ko"] == "guaranteed" else P["protect_bonus_possible"] * (0.5 + ko_probability(hit))
                bonus *= 0.33 ** streak  # each consecutive Protect succeeds 1/3 as often: 1/3, then 1/9 (seen: 3 in a row in self-play)
                if can_remove_first:
                    bonus *= flinch  # we'd KO it first, unless Fake Out stops us: Protect is insurance worth the flinch odds
                bonus *= 1.0 - removed
                add("protect_threatened", answers[0], answers[1],
                    f"{threat['species']}'s {hit['move']} can KO slot {endangered_slot} ({hit['damage_pct_of_current_hp']}%) {when}; Protect it, slot {other} attacks"
                    + (f" (slot {other} KOs {threat['species']} first {int(removed * 100)}% of the time)" if removed else "")
                    + (f" (Protect used {streak} turn(s) in a row already: only 1 in {3 ** streak} to work)" if streak else ""),
                    solo_value(other) + bonus, covers=endangered_slot)
            if slot and solo[other][0] and (protect is None or streak >= 1):
                # No Protect, or Protect is a coin we already flipped: offer the switch so the judge sees the alternative,
                # unless every switch-in is KO'd on entry (then say so and price it; without a Protect at all, offer it anyway).
                base_score = solo_value(other) + P["switch_threatened_bonus"] + (P["protect_bonus_guaranteed"] if pierces_protect else 0) * ko_probability(hit) \
                    + (P["protect_bonus_guaranteed"] * (1 - 0.33 ** streak) if streak else 0)
                base_score *= 1.0 - removed if removed else 1.0
                _, species_any, preview_any, safe_any = safe_switch(slot, theirs, memory, sheet=sheet)
                if not safe_any and preview_any:
                    worst, double = preview_any.get("takes_worst") or {}, preview_any.get("double_target")
                    killer = (" + ".join(double["moves"]) + " together") if double and double["ko_chance"] > worst.get("ko_chance", 0.0) else \
                        (f"{worst['from']}'s {worst['move']}" if worst else "their attacks")
                    note = (f"No safe pivot for slot {endangered_slot}: {species_any} would be KO'd on entry by {killer} "
                            f"({int(round(100 * float(preview_any.get('ko_on_entry', 0.0) or 0.0)))}% of the time"
                            + (", hazards included" if preview_any.get("hazards") else "")
                            + ("; the rest of the bench fares no better" if sum(1 for o in slot.get("options") or [] if o.get("type") == "switch") > 1 else "")
                            + "). Switching just donates a Pokémon; attack (or Protect, if it works) instead.")
                    if note not in sheet.warnings:
                        sheet.warnings.append(note)
                switch_row(endangered_slot, other, "switch_threatened",
                           f"{threat['species']}'s {hit['move']} can KO slot {endangered_slot}" + (" and pierces Protect" if pierces_protect else "")
                           + (f"; it already Protected {streak} turn(s) in a row" if streak else "") + f"; switch it out, slot {other} attacks",
                           base_score, allow_unsafe=False)  # a pivot KO'd on entry is never a row: the warning says why

    # 3a. Double-targeting: both foes together KO a slot before it moves. Protect it while the partner attacks; the
    #     bonus is the chance they actually double into it (they pick one target) times the KO chance.
    for focus in sheet.focus_threats:
        if not focus["before_we_move"] or focus["ko_chance"] < 0.5 or (sheet.alone and not sheet.stall_reasons):
            continue
        endangered_slot, other = focus["slot"], 1 - focus["slot"]
        slot = slot_template(endangered_slot)
        protect = protect_index(slot)
        me_f = ours.get(endangered_slot)
        streak = memory.protect_streak(me_f.species, endangered_slot) if (me_f is not None and memory is not None) else 0
        if protect is not None and solo[other][0]:
            answers = {endangered_slot: {"option": protect, "target": 0}, other: solo_answer(other)}
            add("protect_double_target", answers[0], answers[1],
                f"both foes together KO slot {endangered_slot} ({focus['moves'][0]} + {focus['moves'][1]} = {focus['damage_pct_of_current_hp']}%) before it moves; Protect it, slot {other} attacks",
                solo_value(other) + P["protect_bonus_guaranteed"] * P.get("focus_risk", 0.35) * focus["ko_chance"] * 0.33 ** streak, covers=endangered_slot)

    # 3b. Protect one slot, switch the other: the slot in danger Protects while its partner pivots to a safe bench card
    #     (the Oct 8 loss, turn 3: Thundurus Protect + Kingambit out to Landorus-T; the generator did not exist).
    if memory is not None and len(ours) == 2:
        for endangered_slot in sorted(endangered):
            other = 1 - endangered_slot
            protect = protect_index(slot_template(endangered_slot))
            other_slot = slot_template(other)
            if protect is None or not other_slot:
                continue
            idx, species, preview, safe = safe_switch(other_slot, theirs, memory, sheet=sheet)
            if idx is None or not safe:
                continue
            me_e = ours.get(endangered_slot)
            streak = memory.protect_streak(me_e.species, endangered_slot) if me_e is not None else 0
            worst = max((ko_probability(h, with_accuracy=True) for t in sheet.threats for h in t["hits"] if h["into_slot"] == endangered_slot), default=0.0)
            value = P["protect_bonus_guaranteed"] * worst * 0.33 ** streak + P["switch_weak_bonus"] + max(float(preview.get("score", 0.0) or 0.0), 0.0) * 0.25
            answers = {endangered_slot: {"option": protect, "target": 0}, other: {"option": idx, "target": 0}}
            add("protect_and_switch", answers[0], answers[1],
                f"slot {endangered_slot} is in KO range; Protect it while slot {other} pivots to {species}"
                + (f" (Intimidate: {', '.join(preview['intimidate'])})" if preview.get("intimidate") else ""), value, covers=endangered_slot)

    # 3c. Redirection: Follow Me / Rage Powder soaks a single-target lethal hit aimed at our partner, who then
    #     gets its turn (setup or attack). Only when the redirector is sturdier than the partner against that hit.
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["ko"] == "no":
                continue
            endangered_slot, redirector = hit["into_slot"], 1 - hit["into_slot"]
            slot_r = slot_template(redirector)
            if not slot_r:
                continue
            redirect = next((i for i, o in enumerate(slot_r.get("options") or []) if o.get("type") == "move"
                             and data.to_id(o.get("move_id")) in ("followme", "ragepowder")), None)
            if redirect is None:
                continue
            move_info = data.move_info(hit["move"]) or {}
            if move_info.get("target") not in ("normal", "any", "adjacentFoe"):
                continue  # spread moves can't be redirected
            me_r = ours.get(redirector)
            if me_r is None:
                continue
            # does the redirector survive that hit?
            attacker = theirs.get(threat["position"])
            est = damage_percent(attacker, {**move_info, "id": hit["move"]}, me_r, field=sheet.field) if attacker else None  # with the field: screens, weather, Ruin (D1k)
            if est is None or est[1] >= 100:
                continue  # it would just die instead
            partner_slot = slot_template(endangered_slot)
            partner_action = None
            for setup in ("tailwind", "trickroom", "icywind", "electroweb"):
                idx = _option_index(partner_slot, "move", move_id=setup) if partner_slot else None
                if idx is not None:
                    partner_action = ({"option": idx, "target": 0}, f"sets {setup}")
                    break
            raw_partner = _best_attack(by_slot.get(endangered_slot, {}), P=P, threats=weights)
            if partner_action is None and raw_partner[0]:
                partner_action = (slot_answer(raw_partner[0]["option"], raw_partner[1]["target"]), "attacks freely")
            if partner_action is None:
                continue
            answers = {redirector: {"option": redirect, "target": 0}, endangered_slot: partner_action[0]}
            add("redirect_to_protect_partner", answers[0], answers[1],
                f"{threat['species']}'s {hit['move']} would KO slot {endangered_slot}; slot {redirector} redirects it ({est[0]}-{est[1]}% to itself) while slot {endangered_slot} {partner_action[1]}",
                (P["protect_bonus_guaranteed"] if hit["ko"] == "guaranteed" else P["protect_bonus_possible"]) + 25 + (50 if "sets" in partner_action[1] else raw_partner[2] * 0.8),
                covers=endangered_slot)

    # 4. Fake Out on a Pokémon's first turn out: flinch the biggest threat while the partner sets up or attacks.
    for number in (0, 1):
        me = ours.get(number)
        slot = slot_template(number)
        if not me or not slot or species_key(me.species) not in fresh:
            continue
        fake = _option_index(slot, "move", move_id="fakeout")
        if fake is None:
            continue
        fake_option = (slot.get("options") or [])[fake]
        # Armor Tail / Dazzling / Queenly Majesty on EITHER opponent blocks priority moves into both of them.
        priority_blocked = any(data.to_id(o.ability) in ("armortail", "dazzling", "queenlymajesty") for o in theirs.values())
        legal_targets = [t for t in fake_option.get("targets") or [] if t > 0 and t in theirs and not priority_blocked
                         and (data.effectiveness("normal", theirs[t].types) > 0 or data.to_id(me.ability) in ("scrappy", "mindseye"))
                         and not (sheet.field is not None and sheet.field.terrain == "psychic" and theirs[t].grounded)
                         and data.to_id(theirs[t].item) != "covertcloak" and data.to_id(theirs[t].ability) not in ("innerfocus", "shielddust")]
        if not legal_targets:
            continue  # nothing flinchable: a Fake Out would be wasted
        # The threat whose known attacks do the most to us. A target that would KO one of our slots BEFORE that slot moves
        # comes first: flinching it is a Protect that also works through Unseen Fist, and the partner still gets its turn.
        def lands_first(t: dict, h: dict) -> bool:
            return not acts_before(h["into_slot"], t["position"], 0, h.get("priority", 0))
        lethal_by_target = {pos: max((ko_probability(h) if h["ko"] == "possible" else 1.0)
                                     for t in sheet.threats if t["position"] == pos for h in t["hits"] if h["ko"] != "no" and lands_first(t, h))
                            for pos in legal_targets if any(h["ko"] != "no" and lands_first(t, h) for t in sheet.threats if t["position"] == pos for h in t["hits"])}
        danger = {pos: threat_weight(pos, sheet, ours, P) for pos in legal_targets}
        target = max(lethal_by_target, key=lambda pos: (lethal_by_target[pos], danger[pos])) if lethal_by_target else max(danger, key=danger.get)
        other = 1 - number
        other_slot = slot_template(other)
        partner = None
        setup_name = None
        if other_slot:
            for setup in ("tailwind", "trickroom", "icywind", "electroweb"):
                idx = _option_index(other_slot, "move", move_id=setup)
                if idx is not None:
                    partner, setup_name = {"option": idx, "target": 0}, setup
                    break
        if partner is None and solo[other][0]:
            partner = solo_answer(other)
        if partner is None:
            continue
        answers = {number: {"option": fake, "target": target}, other: partner}
        why = f"Fake Out {theirs[target].species} (its attacks threaten us most) while slot {other} " + (f"sets {setup_name}" if setup_name else "attacks")
        attacks_ko = any(kos_a_foe(b[1]) for b in best.values())  # per foe: a spread's summed range is not a KO (D1j)
        lethal_first = lethal_by_target.get(target, 0.0)  # chance this target KOs one of our slots before it moves
        base = P["fakeout_base"] if (not attacks_ko or lethal_first) else 15
        # Flinching the Pokémon that threatens a KO on us this turn is worth extra: it can't act. When that KO would land
        # before we move, the flinch is worth a Protect (and unlike Protect it stops Unseen Fist). Lost sparring game, Oct 7:
        # the code preferred switching Incineroar out to Fake Out on a Scarf Urshifu while Tornadus set Tailwind.
        threatens_us = any(h["ko"] != "no" for t in sheet.threats if t["position"] == target for h in t["hits"])
        if lethal_first:
            pierces = data.to_id(theirs[target].ability) == "unseenfist"
            why += f"; its attack would KO one of our slots before it moves ({int(lethal_first * 100)}% chance), and the flinch stops that" + (" even though Unseen Fist goes through Protect" if pierces else "")
        # Fake Out vs Fake Out is decided by speed: a faster fresh Fake Out on their side can flinch ours first.
        faster_fakers = [t for t in sheet.threats if "fakeout" in t["known_moves"] and "FRESH" in (t.get("note") or "")
                         and speed_rank.get(("theirs", t["position"]), 99) < speed_rank.get(("mine", number), 99)]
        lands = 1.0 - (sheet.flinch_risk.get(number, 0.0) if faster_fakers else 0.0)
        if lands < 1.0:
            why += f"; their faster Fake Out may flinch ours first ({int((1 - lands) * 100)}%)"
        add("fake_out_setup", answers[0], answers[1], why,
            (base + (P["fakeout_threat_bonus"] if threatens_us else 0) + P["protect_bonus_guaranteed"] * lethal_first) * lands
            + (solo_value(other) * 0.6 if not setup_name else P["fakeout_setup_value"]))

    # 5. Switch a slot that has no worthwhile attack into a bench Pokémon (a safe one: a pivot that is KO'd on entry
    #    just donates a Pokémon, Oct 8 practice game turn 12).
    for number in (0, 1):
        slot = slot_template(number)
        if slot and (best[number][0] is None or best[number][2] < P["switch_weak_threshold"]):
            other = 1 - number
            if solo[other][0] is None:
                continue
            switch_row(number, other, "switch_weak_slot", f"slot {number} has no good attack; switch out while slot {other} attacks",
                       solo_value(other) + P["switch_weak_bonus"], allow_unsafe=False)

    candidates.sort(key=lambda c: -c["score"])
    # The cut: the best four, plus the Protect / switch rows that cover a slot in a KO range (the Oct 8 loss, turn 3:
    # the row that Protected the exposed Thundurus ranked 5th and was cut; it decided the game).
    kept = candidates[:4]
    for cand in candidates[4:]:
        if len(kept) >= 6:
            break
        if cand["name"] in COVER_ROWS and cand.get("covers") in endangered:
            kept.append(cand)
    return kept
