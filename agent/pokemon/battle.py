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
    overlord_fallen: int | None = None  # Supreme Overlord: allies fainted when it came in (memory.overlord_count); None: unknown

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


def _status_id(raw: object) -> str | None:
    """The status as the code compares it ("slp", "par", "brn", "psn", "tox", "frz"): the live observation sends it in
    capitals ("SLP"; Oct 8 evening loss: four of ours slept and the sheet never knew, and a "PAR" never halved a
    speed), and "FNT" is the fainted flag's business, not a status."""
    status = data.to_id(raw)
    return status if status and status != "fnt" else None


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
        status=_status_id((summary or {}).get("status")),
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
        # Cresselia at fallen 3 is 77-91%, the sheet said 61-71%). The count is fixed on entry: the sheet sets it for a
        # Kingambit on the field (6ce0b81e T5: Iron Head at fallen 1 did 95% to Baxcalibur, the side's 2 faints said a KO).
        count = fallen if fallen is not None else attacker.overlord_fallen if attacker.overlord_fallen is not None else \
            (fs.fainted.get(attacker.side, 0) if fs is not None else 0)
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
    reply: dict = field(default_factory=dict)  # the reply mixture's table (agent/pokemon/replies.py): columns, their prior, the top rows
    wake: dict = field(default_factory=dict)  # our slot -> chance an ASLEEP Pokémon there acts this turn (0, 1/3, 1/2, 1 by missed attempts)
    hampered: dict = field(default_factory=dict)  # our slot -> chance a paralysed (0.75) or frozen (0.2 thaw) Pokémon there acts this turn
    taunted: set = field(default_factory=set)  # their species_keys under Taunt now (memory.taunted, from the protocol log)
    bench_summary: dict = field(default_factory=dict)  # our bench species_key -> its observation entry (current item, status, HP)
    sleep_risk: dict = field(default_factory=dict)  # our slot -> (chance a faster foe's sleep move stops its action this turn, that move's priority)
    sleep_threat: dict = field(default_factory=dict)  # their position -> what one use of its sleep move costs us (damage points)
    bench_status: dict = field(default_factory=dict)  # our bench species_key -> its status ("slp": a Spore sink if it comes in)
    opp_team: dict = field(default_factory=dict)  # their species_key -> {"hp", "status", "fainted", "active"} from the observation
    turn: int | None = None  # the battle turn of this sheet (the reply mixture's priors depend on it)
    timers: dict = field(default_factory=dict)  # "trickroom", "tailwind" (ours) -> turns it still covers, this one counted (None: unknown)
    field: "FieldState | None" = None  # keep last: this name shadows dataclasses.field for anything declared after it

    def acts(self, slot: int) -> float:
        """Chance our Pokémon in ``slot`` gets to act at all this turn: asleep (wake odds), paralysed (3 in 4), frozen (1 in
        5 thaws). Its switch is never stopped."""
        return self.wake.get(slot, 1.0) * self.hampered.get(slot, 1.0)

    def reply_prompt(self) -> dict:
        """The mixture's table as the model sees it: its seed, class, lambda and the replies it finds plausible (prior or
        equilibrium weight >= 0.05); not its rows, whose score, worst case and best reply candidate_turns carry already
        (the payload stays inside the ops budget, tests/test_reliability_f.py)."""
        table = {k: v for k, v in self.reply.items() if k != "rows"}
        columns = table.get("columns") or []
        table["columns"] = [c for c in columns if max(c.get("prior", 0.0), c.get("eq", 0.0)) >= 0.05] or columns
        return table

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
            **({"asleep_acts_pct": {str(n): int(round(100 * w)) for n, w in sorted(self.wake.items())}} if self.wake else {}),
            **({"reply_table": self.reply_prompt()} if self.reply else {}),
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
    taunt_started = {k.split(":", 1)[1]: v for k, v in (getattr(memory, "taunted", None) or {}).items() if k.startswith("theirs:")}
    sheet.taunted = set(taunt_started)
    slots = sorted(template.get("slots") or [], key=lambda s: s.get("slot", 0))
    weather = normalize_weather(obs.get("weather"))  # dict from the live server, string from the simulator, None when clear
    field_text = data.to_id(json.dumps([obs.get("fields"), obs.get("field")], default=str))
    turn = obs.get("turn")
    sheet.trick_room = "trickroom" in field_text
    if sheet.trick_room:
        sheet.timers["trickroom"] = turns_left(obs.get("fields"), "trickroom", turn)
        sheet.notes.append("Trick Room is up: slower Pokémon move first." + _left_text(sheet.timers["trickroom"]))
    my_tailwind = "tailwind" in data.to_id(json.dumps(obs.get("side_conditions"), default=str))
    their_tailwind = "tailwind" in data.to_id(json.dumps(obs.get("opponent_side_conditions"), default=str))
    if my_tailwind:
        sheet.timers["tailwind"] = turns_left(obs.get("side_conditions"), "tailwind", turn)
        sheet.notes.append("Our Tailwind is up (speed doubled)." + _left_text(sheet.timers["tailwind"]))
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
            mon.overlord_fallen = memory.overlord_count("mine", key)
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
                        mon.overlord_fallen = memory.overlord_count("theirs", key)
                        theirs[int(t["target"])] = mon
    if not theirs:
        for index, summary in enumerate(_active_summaries(obs.get("opponent_team"))[:2], start=1):
            key = species_key(summary.get("species"))
            mon = build_mon(summary, memory.opp_cards.get(key), side="theirs", position=index, item_lost=key in memory.opp_items_lost)
            if mon:
                mon.tailwind = their_tailwind
                mon.paradox_stat = memory.paradox_active.get("theirs:" + key)
                mon.overlord_fallen = memory.overlord_count("theirs", key)
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
            # A priority attack fails into a foe on an Armor Tail / Dazzling / Queenly Majesty side (they shield the holder
            # and its ally) and, in Psychic Terrain, into a GROUNDED foe only: a Flying type, Levitate or an Air Balloon is
            # still hit (priority_move_blocked, per target). BLOCKED only when no foe is left to hit.
            shielded = {pos for pos, o in theirs.items() if priority_move_blocked(sheet, o)} \
                if (eff.get("priority") or 0) > 0 and data.is_damaging(move) else set()
            priority_blocked = bool(shielded) and not [t for t in targets if t in theirs and t not in shielded]
            if move["id"] in ("fakeout", "firstimpression") and not fresh_now:
                note = f"{move['id']} FAILS: {me.species} is not on its first turn out."
                targets = []
            elif priority_blocked:
                note = f"{move['id']} is BLOCKED: their Armor Tail/Dazzling/Queenly Majesty (or Psychic Terrain) stops priority moves aimed at them."
                targets = []
            elif shielded:
                targets = [t for t in targets if t not in shielded]
                note = (f"{move['id']} cannot hit {', '.join(theirs[p].species for p in sorted(shielded))}: Psychic Terrain stops "
                        "priority moves into grounded targets (a Flying type, Levitate or an Air Balloon is still hit).")
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

    # Opponent threats from their KNOWN sets. Their priority attack fails into one of ours on an Armor Tail / Dazzling /
    # Queenly Majesty side (both slots) and, in Psychic Terrain, into a GROUNDED one only: our Flying type, Levitate or
    # Air Balloon is still hit (priority_move_blocked, per target, as for our own priority attacks).
    shielded = {number for number, me in ours.items() if priority_move_blocked(sheet, me)}
    for position, opp in sorted(theirs.items()):
        threat = {"position": position, "species": opp.species, "hp_pct": round(opp.hp_fraction * 100), "status": opp.status,
                  "item": opp.item, "ability": opp.ability, "known_moves": [m["id"] for m in opp.moves], "hits": []}
        opp_fresh = species_key(opp.species) in set(memory.opp_fresh_active) or obs.get("turn") in (None, 1)
        # A Choice item locks its holder into the last move it used (live server: read from the protocol log).
        locked = memory.opp_last_move.get(species_key(opp.species)) if data.to_id(opp.item) in CHOICE_ITEMS else None
        if locked:
            threat["locked_move"] = locked
            threat["note"] = f"Choice-locked into {locked} until it switches out: its other moves cannot be used this turn."
        for move in opp.moves:
            if locked and move["id"] != locked:
                continue  # Choice-locked: only the locked move can come out
            if not data.is_damaging(move):
                continue
            if move["id"] in ("fakeout", "firstimpression") and not opp_fresh:
                continue  # only works on the user's first turn out
            eff = effective_move(opp, move, opp, fs)
            for number, me in sorted(ours.items()):
                if (eff.get("priority") or 0) > 0 and number in shielded:
                    continue  # our Armor Tail, or Psychic Terrain under this grounded one, blocks their priority attack
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
        if species_key(opp.species) in sheet.taunted:
            started = taunt_started.get(species_key(opp.species))
            threat["taunted"] = f"Taunted on turn {started}: no Protect and no status moves (Spore, Tailwind, Trick Room...) until it ends (3 turns)"
            sheet.notes.append(f"Their {opp.species} is Taunted (since turn {started}): it cannot Protect or use status moves for its "
                               "three turns; another Taunt on it fails.")
        elif any(m["id"] in PROTECT_LIKE for m in opp.moves):
            streak = memory.opp_protect_streak.get(species_key(opp.species), 0)
            threat["protect"] = ("can Protect this turn" if not streak else
                                 f"Protected last turn: another works only 1 in {3 ** streak} (good turn to attack it)")
        if "fakeout" in threat["known_moves"] and not locked:
            # FRESH (read as "expect a flinch") only with a slot of ours it can flinch: not shielded, and not a Ghost type,
            # Inner Focus / Shield Dust or Covert Cloak (_flinchable, the rule _flinch_risk prices).
            open_slots = [n for n in sorted(ours) if n not in shielded and _flinchable(ours[n], ours, fs)]
            if opp_fresh and (open_slots or not ours):
                threat["note"] = "FRESH: its Fake Out works THIS turn (priority +3): expect a flinch on one of our slots, so a setup move (Trick Room/Tailwind) from the slot it targets fails."
                if shielded and open_slots:
                    threat["note"] += (f" Only into slot {open_slots[0]} ({ours[open_slots[0]].species}): Psychic Terrain shields our "
                                       f"grounded {', '.join(ours[n].species for n in sorted(shielded))}.")
            elif opp_fresh and all(n in shielded for n in ours):
                threat["note"] = "has Fake Out but our Armor Tail / Psychic Terrain blocks it."
            elif opp_fresh:
                threat["note"] = ("has Fake Out, but it cannot flinch any of ours this turn: "
                                  + "; ".join(_no_flinch_reason(ours[n], fs) for n in sorted(ours)) + ".")
            else:
                threat["note"] = "has Fake Out but is past its first turn out: Fake Out fails now."
        sheet.threats.append(threat)

    sheet.turn = turn if isinstance(turn, int) else None
    _our_status(sheet, ours, obs)
    for slot in slots:
        me = ours.get(slot.get("slot", 0))
        if me is not None and data.to_id(me.item) in CHOICE_ITEMS and any(
                o.get("type") == "move" and data.to_id(o.get("move_id")) in PROTECT_OPTIONS for o in slot.get("options") or []) \
                and len([o for o in slot.get("options") or [] if o.get("type") == "move"]) > 1:
            sheet.notes.append(f"Our {me.species} holds a Choice item: Protect LOCKS it into Protect until it switches out (next turn "
                               "only a 1-in-3 Protect or a switch, no attack).")
    _glaive_rush_carry(sheet, ours, memory)
    _flinch_risk(sheet, ours, theirs, fs, slots=slots, P=P)
    _sleep_threats(sheet, ours, theirs, fs, slots=slots, P=P)
    _double_target_threats(sheet, ours, P)
    _endgame(sheet, slots, ours, theirs, fs)
    _switch_previews(sheet, slots, theirs, obs, memory, fs, hazards=_memory_hazards(memory, "mine"))
    _sleep_switch_notes(sheet, theirs, memory)
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


WEATHER_SETTERS = {"drizzle": "rain", "drought": "sun", "orichalcumpulse": "sun", "sandstream": "sand", "snowwarning": "snow"}
TERRAIN_SETTERS = {"grassysurge": "grassy", "psychicsurge": "psychic", "electricsurge": "electric", "mistysurge": "misty",
                   "hadronengine": "electric"}


def entry_field(fs: "FieldState | None", *, add: "Iterable[Mon]", drop: "Iterable[Mon | None]" = (),
                entering: "Iterable[Mon] | None" = None) -> "FieldState":
    """The field as it stands once the Pokémon in ``add`` are out and those in ``drop`` have left: the same screens and
    fainted counts, the actives updated, so a switch-in's own Ruin ability and the Ruin abilities still out price every
    hit (Oct 8 live match: our Wo-Chien's Tablets of Ruin cut Body Press into Ting-Lu by a quarter the turn it came in,
    its leaving gave the foes their Attack back, and their Ting-Lu's Vessel of Ruin cut Draco Meteor); the weather and
    terrain the ``entering`` Pokémon set on arrival (every one in ``add`` when None: Sand Stream, Drought, Drizzle, Snow
    Warning, the Surge terrains, Hadron Engine, Orichalcum Pulse; d74bfb88 T9: Moonblast into an incoming Tyranitar
    is 56-66% in its own sand, not 81-95%); and every paradox boost re-read under that field (newcomers in place, the
    Pokémon already out as copies when theirs change)."""
    base = fs if fs is not None else FieldState()
    dropped = {id(m) for m in drop if m is not None}
    added = list(add)
    arriving = added if entering is None else list(entering)
    actives = [m for m in base.actives if id(m) not in dropped and not any(m is a for a in added)] + added
    new = FieldState(weather=base.weather, terrain=base.terrain, trick_room=base.trick_room, my_side=set(base.my_side),
                     their_side=set(base.their_side), actives=actives, fainted=dict(base.fainted), party_base_atk=dict(base.party_base_atk))
    for mon in arriving:
        ability = data.to_id(mon.ability)
        if ability in WEATHER_SETTERS:
            new.weather = WEATHER_SETTERS[ability]
        if ability in TERRAIN_SETTERS:
            new.terrain = TERRAIN_SETTERS[ability]
    for mon in added:
        mon.boosted_stat = paradox_boosted_stat(mon, new)
    if new.weather != base.weather or new.terrain != base.terrain:
        refreshed = []
        for mon in new.actives:
            boosted = paradox_boosted_stat(mon, new)
            refreshed.append(mon if any(mon is a for a in added) or boosted == mon.boosted_stat else Mon(**{**mon.__dict__, "boosted_stat": boosted}))
        new.actives = refreshed
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
    after = entry_field(fs, add=[attacker, incoming], drop=[outgoing, original], entering=[incoming])  # the attacker is already out
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


def _no_flinch_reason(me: "Mon", fs: "FieldState") -> str:
    """Why a Fake Out cannot make this Pokémon of ours flinch (a case _flinchable rules out), for the threat note."""
    if fs.terrain == "psychic" and me.grounded:
        return f"Psychic Terrain shields our grounded {me.species}"
    if "ghost" in me.types:
        return f"{me.species} is a Ghost type (Fake Out cannot touch it)"
    if data.to_id(me.ability) in ("innerfocus", "shielddust"):
        return f"{me.species}'s {me.ability} stops the flinch"
    if data.to_id(me.item) == "covertcloak":
        return f"{me.species}'s Covert Cloak stops the flinch"
    return f"our Armor Tail / Dazzling / Queenly Majesty shields {me.species}"


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
    eligible = [n for n, me in ours.items() if _flinchable(me, ours, fs) and sheet.acts(n) > 0]
    if not eligible:
        return  # a sleeping slot that cannot act has nothing for a Fake Out to stop (Oct 8 evening loss, turn 4)
    P = P or DEFAULTS

    def stoppable(n: int) -> float:
        return _slot_action_value(sheet, n, slots, P) * sheet.acts(n)

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


# -- sleep: our own Pokémon asleep, and their sleep moves (the Oct 8 evening loss: Spore slept four of ours in seven turns) --

SLEEP_TURNS = 2.0  # move attempts a sleep move costs on average (Showdown: the counter starts at 2-4, so 1-3 attempts are lost)
FLINCH_CHANCE = {"iciclecrash": 0.3, "rockslide": 0.3, "ironhead": 0.3, "airslash": 0.3, "bite": 0.3, "zingzap": 0.3,
                 "darkpulse": 0.2, "waterfall": 0.2, "zenheadbutt": 0.2, "twister": 0.2, "extrasensory": 0.1}  # secondary flinches


def sleep_attempts(obs: dict) -> dict[str, tuple[int, bool]]:
    """Our Pokémon asleep right now, as species_key -> (move attempts missed since it fell asleep, slept by Rest), from
    the protocol log: its last |-status|..|slp, then one per |cant|..|slp; a |-curestatus|, |-cureteam| or a faint
    clears it. The counter does not reset on a switch (Gen 9: Archaludon, Spored T1, missed T2, sat out T3-T4 and woke
    on T5 of the Oct 8 evening loss)."""
    team = obs.get("team") if isinstance(obs.get("team"), dict) else {}
    side = next((str(k).split(":")[0] for k in team if str(k)[:2] in ("p1", "p2") and ":" in str(k)), None)
    if not side:
        return {}
    nicks = {str(k).split(":", 1)[1].strip(): species_key(v.get("species")) for k, v in team.items()
             if isinstance(v, dict) and ":" in str(k) and v.get("species")}
    out: dict[str, tuple[int, bool]] = {}
    for raw in obs.get("protocol_log") or []:
        parts = str(raw).split("|")
        if len(parts) < 3 or not parts[2].startswith(side):
            continue
        kind = parts[1]
        if kind == "-cureteam":
            out.clear()
            continue
        name = parts[2].split(":", 1)[1].strip() if ":" in parts[2] else parts[2]
        key = nicks.get(name, species_key(name))
        if kind == "-status" and len(parts) > 3 and parts[3] == "slp":
            out[key] = (0, "move: Rest" in str(raw))
        elif kind == "cant" and len(parts) > 3 and parts[3] == "slp" and key in out:
            out[key] = (out[key][0] + 1, out[key][1])
        elif kind in ("-curestatus", "faint") and key in out:
            del out[key]
    return out


def wake_chance(missed: int | None, rest: bool = False) -> float:
    """Chance a sleeping Pokémon acts this turn after ``missed`` move attempts asleep: 0, 1/3, 1/2, then 1 (the counter
    starts at 2, 3 or 4 and drops by one per attempt); Rest sleeps exactly two attempts. Unknown history: 1/3."""
    if missed is None:
        return 1 / 3
    if rest:
        return 1.0 if missed >= 2 else 0.0
    return (0.0, 1 / 3, 0.5)[missed] if missed < 3 else 1.0


def _our_status(sheet: "TurnSheet", ours: dict[int, "Mon"], obs: dict) -> None:
    """Our sleeping slots' chance to act (``sheet.wake``), our bench's statuses and their team's state, from the
    observation. A sleeping slot's attacks and its Protect count only if it wakes; its switch always works."""
    attempts = sleep_attempts(obs) if obs.get("protocol_log") else {}
    for n, me in sorted(ours.items()):
        if me.status != "slp":
            continue
        missed, rest = attempts.get(species_key(me.species), (None, False))
        sheet.wake[n] = round(wake_chance(missed, rest), 3)
        history = "" if missed is None else f" ({missed} turn{'s' if missed != 1 else ''} missed so far)"
        sheet.notes.append(f"Our {me.species} (slot {n}) is ASLEEP{history}: it acts this turn only {int(round(100 * sheet.wake[n]))}% "
                           "of the time, so its attacks and its Protect count only then (a switch always works). A Pokémon "
                           "already asleep cannot be put to sleep again: while it stays in, their sleep move has one target fewer.")
    for n, me in sorted(ours.items()):
        if me.status in ("par", "frz"):
            sheet.hampered[n] = 0.75 if me.status == "par" else 0.2  # full paralysis 1 in 4; a frozen Pokémon thaws 1 in 5
    team = obs.get("team")
    for s in (team.values() if isinstance(team, dict) else []):
        if isinstance(s, dict) and s.get("species") and not s.get("active") and not s.get("fainted"):
            sheet.bench_summary[species_key(s["species"])] = s
            status = _status_id(s.get("status"))
            if status:
                sheet.bench_status[species_key(s["species"])] = status
    opp = obs.get("opponent_team")
    for s in (opp.values() if isinstance(opp, dict) else []):
        if isinstance(s, dict) and s.get("species"):
            frac = s.get("current_hp_fraction")
            sheet.opp_team[species_key(s["species"])] = {"hp": float(frac) if frac is not None else 1.0, "status": _status_id(s.get("status")),
                                                        "fainted": bool(s.get("fainted")), "active": bool(s.get("active"))}


def _slot_action_value(sheet: "TurnSheet", n: int, slots: list[dict] | None, P: dict) -> float:
    """What one turn of this slot's action is worth, in damage points: its best attack's expected value or its best
    speed-control move, whichever is more."""
    by_slot = {e["slot"]: e for e in sheet.our_options}
    value = max(_best_attack(by_slot.get(n, {}), P=P)[2], 0.0)
    for s in slots or []:
        if s.get("slot", 0) == n:
            for o in s.get("options") or []:
                if o.get("type") == "move":
                    value = max(value, float(SETUP_MOVES.get(data.to_id(o.get("move_id")), 0)))
    return value


def _sleep_blocked(move: dict, target: "Mon", user: "Mon", ours: dict[int, "Mon"], fs: "FieldState | None") -> str | None:
    """Why a sleep move does nothing to ``target`` (mech's ``data.status_move_blocked``: Grass types / Overcoat / Safety
    Goggles for powders, Insomnia, Vital Spirit, Sweet Veil, Electric or Misty Terrain while grounded, already asleep or
    statused, Lum / Chesto Berry, Magic Bounce, Good as Gold, Prankster into a Dark type), plus an ally's Sweet Veil and
    Safeguard on that side."""
    side = ours.values() if target.side == "mine" else ()
    why = data.status_move_blocked(move, target, user=user, terrain=fs.terrain if fs else None)
    if why is None and any(data.to_id(m.ability) == "sweetveil" for m in side):
        why = "an ally's Sweet Veil prevents sleep"
    if why is None and fs is not None and "safeguard" in (fs.my_side if target.side == "mine" else fs.their_side):
        why = "Safeguard prevents status"
    return why


def sleep_users(sheet: "TurnSheet", theirs: dict[int, "Mon"]) -> dict[int, list[dict]]:
    """Their actives that can use a sleep move this turn: position -> the move dicts (not asleep or frozen themselves,
    not Taunted, not Choice-locked into something else)."""
    out: dict[int, list[dict]] = {}
    for t in sheet.threats:
        foe = theirs.get(t["position"])
        if foe is None or foe.status in ("slp", "frz") or species_key(foe.species) in sheet.taunted:
            continue
        locked = t.get("locked_move")
        moves = [{**(data.move_info(m["id"]) or {}), **m} for m in foe.moves if m["id"] in SLEEP_MOVES and (not locked or m["id"] == locked)]
        if moves:
            out[t["position"]] = moves
    return out


def _sleep_threats(sheet: "TurnSheet", ours: dict[int, "Mon"], theirs: dict[int, "Mon"], fs: "FieldState",
                   slots: list[dict] | None = None, P: dict | None = None) -> None:
    """A foe's sleep move (Spore, Sleep Powder, Hypnosis, Yawn...) into each of our slots it works on: a threat row
    "slot N slept" (``threat["sleep"]``) valued as the slot's lost actions over the expected sleep (1-3 turns: this
    turn's too when the sleep lands before the slot moves, later turns at ``sleep_future_w``), the most one use costs us
    (``sheet.sleep_threat``: the KO credit for removing the sleeper and the Taunt that blocks it read it), the chance it
    stops a slower slot's action THIS turn (``sheet.sleep_risk``, aimed 0.8 at the slot that loses most), and a SLEEP
    warning with the counters. The Oct 8 evening loss: Amoonguss's Spore landed five times on four of ours in seven
    turns, and no warning on any turn mentioned it."""
    P = P or DEFAULTS
    users = sleep_users(sheet, theirs)
    if not users:
        return
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    w = float(P.get("sleep_future_w", 0.5))
    by_slot = {e["slot"]: e for e in sheet.our_options}
    for t in sheet.threats:
        pos = t["position"]
        foe = theirs.get(pos)
        if pos not in users or foe is None:
            continue
        rows, immune = [], []
        for mv in users[pos]:
            prio = (mv.get("priority") or 0) + (1 if data.to_id(foe.ability) == "prankster" else 0)
            acc = data.hit_chance(mv, fs.weather if fs else None)
            for n, me in sorted(ours.items()):
                why = _sleep_blocked(mv, me, foe, ours, fs)
                if why:
                    immune.append(f"slot {n} ({me.species}): {why}")
                    continue
                first = mv["id"] != "yawn" and (prio > 0 or rank.get(("theirs", pos), 99) < rank.get(("mine", n), 99))
                if mv["id"] == "yawn":
                    lost = 0.5 * w * SLEEP_TURNS  # drowsy now, asleep at the end of next turn unless it switches out
                else:
                    lost = (1.0 + w * (SLEEP_TURNS - 1.0)) if first else w * SLEEP_TURNS
                value = acc * _slot_action_value(sheet, n, slots, P) * lost
                rows.append({"move": mv["id"], "into_slot": n, "species": me.species,
                             "effect": "falls asleep at the end of next turn (Yawn)" if mv["id"] == "yawn" else "sleeps it for 1-3 turns",
                             "before_we_move": first, "priority": prio, "value": round(value, 1),
                             **({"hit_chance": round(acc, 2)} if acc < 1 else {}),
                             "acts_next_turn_pct": 100 if mv["id"] == "yawn" else 0})
        if not rows:
            continue
        t["sleep"] = rows
        sheet.sleep_threat[pos] = max(r["value"] for r in rows)
        # this turn: a sleep move that lands before the slot moves takes this turn's action too (Spore our biggest threat)
        best_by_slot: dict[int, dict] = {}
        for r in rows:
            if r["into_slot"] not in best_by_slot or r["value"] > best_by_slot[r["into_slot"]]["value"]:
                best_by_slot[r["into_slot"]] = r
        top = max(best_by_slot, key=lambda n: best_by_slot[n]["value"])
        use = float(P.get("sleep_use_prior", 0.6))
        for n, r in best_by_slot.items():
            if not r["before_we_move"]:
                continue
            aim = 1.0 if len(best_by_slot) == 1 else (0.8 if n == top else 0.2)
            p = use * aim * float(r.get("hit_chance", 1.0))
            prev = sheet.sleep_risk.get(n, (0.0, 0))
            sheet.sleep_risk[n] = (round(1.0 - (1.0 - prev[0]) * (1.0 - p), 3), max(prev[1], r["priority"]))
        # the warning: who, when, and the counters
        names = ", ".join(sorted({m["id"] for m in users[pos]}))
        targets = " or ".join(f"slot {n} ({best_by_slot[n]['species']})" for n in sorted(best_by_slot))
        when = ("it moves BEFORE " + " and ".join(f"slot {n}" for n in sorted(best_by_slot) if best_by_slot[n]["before_we_move"])
                + ": that slot loses this turn's action too" if any(r["before_we_move"] for r in best_by_slot.values()) else
                "it moves after them: they act this turn and lose the next 1-3")
        counters = []
        for n, me in sorted(ours.items()):
            taunt = _option_index(next((s for s in slots or [] if s.get("slot", 0) == n), {}), "move", move_id="taunt")
            if taunt is None:
                continue
            t_prio = 1 if data.to_id(me.ability) == "prankster" else 0
            if data.status_move_blocked({"id": "taunt", **(data.move_info("taunt") or {})}, foe, user=me):
                continue
            if t_prio > 0 and priority_move_blocked(sheet, foe):
                continue  # Psychic Terrain under a grounded target, or Armor Tail / Dazzling / Queenly Majesty on its side
            if t_prio > 0 or rank.get(("mine", n), 99) < rank.get(("theirs", pos), 99):
                counters.append(f"slot {n} ({me.species})'s Taunt goes first and blocks it for three turns")
        best_ko = 0.0
        for n in ours:
            for opt in by_slot.get(n, {}).get("options") or []:
                prio = opt.get("priority", 0) or 0
                if opt.get("fails") or not (prio > 0 or rank.get(("mine", n), 99) < rank.get(("theirs", pos), 99)):
                    continue
                for r in opt.get("targets") or []:
                    if r.get("side") == "theirs" and (r.get("target") == pos or (r.get("target", 0) == 0 and species_key(r.get("species")) == species_key(foe.species))):
                        best_ko = max(best_ko, ko_probability(r, with_accuracy=True))
        counters.append(f"a KO before it moves ({int(round(100 * best_ko))}% with one hit)" if best_ko > 0 else "a KO before it moves (no single hit does it)")
        counters.append("keeping a Pokémon that is already asleep in (it cannot be slept again), or bringing a Grass type / "
                        "Overcoat / Safety Goggles / Insomnia / Vital Spirit holder; switching in a fresh target hands it a free sleep")
        sheet.warnings.append(
            f"SLEEP: their {foe.species} can use {names} on {targets}: the target sleeps for 1-3 turns (no Sleep Clause), {when}."
            + (f" Immune: {'; '.join(sorted(set(immune)))}." if immune else "")
            + " Counters: " + "; ".join(counters) + ".")


def _sleep_switch_notes(sheet: "TurnSheet", theirs: dict[int, "Mon"], memory: "MatchMemory | None") -> None:
    """Mark each switch-in preview that their active sleep move would hit on entry (``spore_target``)."""
    users = sleep_users(sheet, theirs)
    if not users or not sheet.switch_ins:
        return
    for key, entry in sheet.switch_ins.items():
        why = fresh_sleep_target(sheet, key, theirs, memory)
        if why:
            entry["sleep_target"] = why


def _bench_mon(sheet: "TurnSheet", species: str, memory: "MatchMemory | None") -> "Mon | None":
    """A bench Pokémon as it stands now: its observation entry (current item: an eaten Lum Berry is gone; status; HP)
    over its drafted card."""
    key = species_key(species)
    summary = dict(sheet.bench_summary.get(key) or {"species": species, "current_hp_fraction": sheet.bench_hp.get(key, 1.0)})
    summary["boosts"] = {}  # boosts reset on a switch
    card = memory.my_cards.get(key) if memory is not None else None
    return build_mon(summary, card, side="mine", position=None)


def fresh_sleep_target(sheet: "TurnSheet", species: str, theirs: dict[int, "Mon"], memory: "MatchMemory | None", *,
                       outgoing: "Mon | None" = None) -> str | None:
    """Would bringing this bench Pokémon in hand their active sleep user a target it would not otherwise have? The reason
    when it would; None when no sleep user can act (none out, asleep itself, Taunted, locked), the newcomer is already
    asleep or statused (a Spore sink, like the Archaludon that made Spore fail on T6), it is immune (with its CURRENT
    item: an eaten Lum Berry is no cover), or the Pokémon it replaces (``outgoing``) could be slept where it stands: an
    awake, sleepable Pokémon going out for another leaves Spore a target either way (the review of stage 5714330: the
    old rule vetoed every switch while Amoonguss was out and kept a Chi-Yu in front of a 128-151% Earthquake). The Oct
    8 evening loss: the sleeping Archaludon out for Chi-Yu (T3) and the switch into Indeedee-F (T4) were Spored on entry."""
    users = sleep_users(sheet, theirs)
    if not users:
        return None
    if sheet.bench_status.get(species_key(species)):
        return None  # asleep or otherwise statused: a sleep move cannot land on it
    incoming = _bench_mon(sheet, species, memory)
    if incoming is None:
        return None
    side = {i: m for i, m in enumerate(m for m in (sheet.field.actives if sheet.field else []) if m.side == "mine" and m is not outgoing)}
    for pos, moves in users.items():
        foe = theirs.get(pos)
        for mv in moves:
            if foe is None or _sleep_blocked(mv, incoming, foe, side, sheet.field) is not None:
                continue
            if outgoing is not None and _sleep_blocked(mv, outgoing, foe, side, sheet.field) is None:
                return None  # the Pokémon going out was a target already: the switch adds none
            return f"their {foe.species}'s {mv['id']} can put it to sleep on entry"
    return None


def sleep_sink(sheet: "TurnSheet", species: str, theirs: dict[int, "Mon"], memory: "MatchMemory | None") -> bool:
    """While their sleep user can act: is this bench Pokémon one its sleep move cannot touch (asleep already, statused,
    or immune)? The switch-in to prefer then."""
    users = sleep_users(sheet, theirs)
    if not users:
        return False
    if sheet.bench_status.get(species_key(species)):
        return True
    incoming = _bench_mon(sheet, species, memory)
    if incoming is None:
        return False
    side = {i: m for i, m in enumerate(m for m in (sheet.field.actives if sheet.field else []) if m.side == "mine")}
    return all(_sleep_blocked(mv, incoming, theirs[pos], side, sheet.field) is not None
               for pos, moves in users.items() if pos in theirs for mv in moves)


def secondary_flinch(sheet: "TurnSheet", number: int, ours: dict[int, "Mon"], speed_rank: dict, my_priority: int = 0) -> float:
    """Chance a faster foe's flinching attack (Icicle Crash / Rock Slide 30%, Waterfall / Dark Pulse 20%...) stops this
    slot this turn: each faster foe's best such move, aimed at this slot half the time (a spread one always), times its
    accuracy and flinch chance. Inner Focus, Shield Dust and Covert Cloak ignore it; a fresh Fake Out is
    ``sheet.flinch_risk``'s business."""
    me = ours.get(number)
    if me is None or data.secondary_blocked(me) or data.to_id(me.ability) == "innerfocus":
        return 0.0
    keep = 1.0
    for t in sheet.threats:
        best = 0.0
        for h in t["hits"]:
            chance = FLINCH_CHANCE.get(h["move"], 0.0)
            if h["into_slot"] != number or not chance:
                continue
            prio = h.get("priority", 0) or 0
            if not (prio > my_priority or (prio == my_priority and speed_rank.get(("theirs", t["position"]), 99) < speed_rank.get(("mine", number), 99))):
                continue
            aim = 1.0 if (data.move_info(h["move"]) or {}).get("target") in SPREAD_TARGETS else 0.5
            best = max(best, aim * float(h.get("hit_chance", 1.0)) * chance)
        keep *= 1.0 - best
    return 1.0 - keep


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
        if any(h["ko"] == "guaranteed" and first(t, h) for t, hits in foes for h in hits):
            continue  # one of them already KOs alone BEFORE we move: the LETHAL warning covers it (a sure KO that lands
            # after we move is no reason to skip the double: Oct 8, Fake Out + Extreme Speed did 93-110% with no warning)
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


def _expected(row: dict, priority: int, P: dict | None = None, threat: float = 0.0, protect: float = 0.0) -> float:
    """Expected value of a hit in 'damage points': the average roll, CAPPED at the target's remaining HP (a 999% hit
    takes one Pokémon, exactly like a 101% one — uncapped, both slots piled into a 9% Whimsicott on Oct 7), plus a KO
    bonus and a little for priority. ``threat`` is what the target would do to us (``threat_weight``): it scales the
    KO bonus once the KO-credit item lands. ``protect`` is the chance the target Protects (the hit then does nothing)."""
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
    return (damage + bonus) * (1.0 - protect) + P["priority_w"] * priority


def _best_attack(entry: dict, *, avoid_target: int | None = None, P: dict | None = None,
                 threats: dict[str, float] | None = None, protect: dict[str, float] | None = None) -> tuple[dict | None, dict | None, float]:
    """Highest expected damage (option, target_row, expected%) among a slot's move options. A spread move (rows with
    target 0) counts the damage to every opposing Pokémon it hits (its summary row carries the per-target ``rows``, so
    a KO is read per foe, never from the summed range) and is charged for damage to our own ally. With
    ``avoid_target`` only single-target rows into the other foes count: spread rows carry target 0 and slipped
    through the filter, so focus_fire / finish_the_other vanished whenever a slot held Rock Slide / Dazzling Gleam /
    Bleakwind Storm (Oct 7 match 3 turn 3; Oct 8 practice turn 6). ``threats`` (species_key -> weight) scales each
    KO bonus by the threat removed; ``protect`` (species_key -> chance it Protects) discounts the hits into it."""
    P = P or DEFAULTS
    threats = threats or {}
    protect = protect or {}
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
            expected = sum(_expected(r, priority, P, threats.get(species_key(r["species"]), 0.0), protect.get(species_key(r["species"]), 0.0))
                           for r in spread_rows) - P["ally_damage_w"] * ally_hit
            if expected > best[2]:
                summary = {"target": 0, "species": " + ".join(r["species"] for r in spread_rows), "side": "theirs",
                           "damage_pct_of_current_hp": [round(sum(r["damage_pct_of_current_hp"][0] for r in spread_rows), 1),
                                                        round(sum(r["damage_pct_of_current_hp"][1] for r in spread_rows), 1)],
                           "ally_damage_pct": round(ally_hit, 1) if ally_hit else 0, "spread": True, "rows": spread_rows}
                best = (option, summary, expected)
        for row in foe_rows:
            if row["target"] == 0 or row["target"] == avoid_target:
                continue
            expected = _expected(row, priority, P, threats.get(species_key(row["species"]), 0.0), protect.get(species_key(row["species"]), 0.0))
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
SINGLE_TARGETS = ("normal", "any", "adjacentFoe")  # the move targets redirection can move
LIKELY_KO = 0.5  # a redirector this likely to be KO'd (accuracy included) is not counted on


def _redirect_move(sheet: "TurnSheet", redirector: int) -> str | None:
    """The Follow Me / Rage Powder our Pokémon in slot ``redirector`` knows (for callers that do not name the move)."""
    me = next((m for m in (sheet.field.actives if sheet.field is not None else []) if m.side == "mine" and m.position == redirector), None)
    return next((mv["id"] for mv in (me.moves if me is not None else []) if mv.get("id") in REDIRECT_MOVES), None)


def redirect_cover(sheet: "TurnSheet", redirector: int, killer_pos: int, hit: dict, theirs: dict[int, "Mon"],
                   move: str | None = None) -> dict:
    """Does Follow Me / Rage Powder from our slot ``redirector`` keep ``hit`` (a threat row of the foe at ``killer_pos``
    into the partner) off the partner? Redirection lasts only while the redirector stands: a foe that moves before the
    killer (priority first, then the sheet's speed order, which has Trick Room, Tailwind, Scarf and paralysis in it) and
    likely KOs the redirector (its best hit into it: KO chance >= 50% with accuracy) ends it, and the killer then hits
    its original target. Spread moves are never redirected, nor a hit that lands before the +2 redirection is up (+3,
    or +2 from a faster foe). Returns {"covers": the hit lands on the redirector,
    "broken_by": the earlier hit that likely KOs it first (from, position, move, range, ko_chance) or None, "why": why
    it does not hold, "drawn_ko": the drawn hit's own KO chance on the redirector, "holds": it covers AND the
    redirector likely survives what it draws (one that just goes down instead buys nothing)}. ``redirect_holds`` sums
    it up for ``status_value``'s cover credit and the ``redirect_to_protect_partner`` row alike (07a3cb1f T2: Follow Me
    was worth 45 from an Indeedee that Wood Hammer KOs before Archaludon moves; Draco Meteor then KO'd the Basculegion
    its own Wave Crash recoil had weakened). ``move`` is the redirect (default: the one the redirector knows): an
    attacker that ignores it is not covered (mech's ``data.redirect_ignored``: Grass types, Overcoat and Safety Goggles
    ignore Rage Powder, a powder move; Stalwart / Propeller Tail ignore both)."""
    out: dict[str, Any] = {"covers": False, "broken_by": None, "why": "", "drawn_ko": 0.0, "holds": False}
    if (data.move_info(hit["move"]) or {}).get("target") not in SINGLE_TARGETS:
        out["why"] = f"{hit['move']} is a spread move, which redirection does not move"
        return out
    killer = theirs.get(killer_pos)
    ignored = data.redirect_ignored(move or _redirect_move(sheet, redirector), killer) if killer is not None else None
    if ignored:
        out["why"] = f"{ignored}, so {killer.species}'s {hit['move']} still goes into its target"
        return out
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    k_prio = hit.get("priority", 0) or 0
    if k_prio > 2 or (k_prio == 2 and rank.get(("theirs", killer_pos), 99) < rank.get(("mine", redirector), 99)):
        out["why"] = f"{hit['move']} (+{k_prio}) lands before the redirection (+2) is up"
        return out
    for t in sheet.threats:
        pos = t["position"]
        if pos == killer_pos:
            continue
        for h in t["hits"]:
            prio = h.get("priority", 0) or 0
            if h["into_slot"] != redirector or prio < k_prio or (prio == k_prio and rank.get(("theirs", pos), 99) >= rank.get(("theirs", killer_pos), 99)):
                continue  # not into the redirector, or it lands after the killer's hit
            p = ko_probability(h, with_accuracy=True)
            if p >= LIKELY_KO and (out["broken_by"] is None or p > out["broken_by"]["ko_chance"]):
                out["broken_by"] = {"from": t["species"], "position": pos, "move": h["move"], "ko_chance": round(p, 3),
                                    "damage_pct_of_current_hp": list(h["damage_pct_of_current_hp"])}
    if out["broken_by"] is not None:
        b = out["broken_by"]
        name = next((m.species for m in (sheet.field.actives if sheet.field is not None else []) if m.side == "mine" and m.position == redirector), "the redirector")
        out["why"] = (f"{b['from']}'s {b['move']} ({b['damage_pct_of_current_hp'][0]}-{b['damage_pct_of_current_hp'][1]}%, a KO "
                      f"{int(round(100 * b['ko_chance']))}% of the time) KOs {name} before {killer.species if killer is not None else 'the attacker'} "
                      "moves, and redirection ends when the redirector faints")
        return out
    out["covers"] = True
    drawn = [h for t in sheet.threats if t["position"] == killer_pos for h in t["hits"] if h["into_slot"] == redirector and h["move"] == hit["move"]]
    out["drawn_ko"] = round(max((ko_probability(h, with_accuracy=True) for h in drawn), default=0.0), 3)
    out["holds"] = out["drawn_ko"] < LIKELY_KO
    if not out["holds"]:
        out["why"] = f"the redirector would go down to the {hit['move']} it draws ({int(round(100 * out['drawn_ko']))}%)"
    return out


def redirect_holds(sheet: "TurnSheet", redirector: int, theirs: dict[int, "Mon"],
                   move: str | None = None) -> tuple[bool, list[tuple[dict, dict]]]:
    """Is Follow Me / Rage Powder from our slot ``redirector`` real cover for the partner this turn: ``redirect_cover``
    holds for at least one of the foes' KO hits into the partner, and none of the LIKELY ones (KO chance >= 50% with
    accuracy) is left uncovered (a spread move, an attacker that ignores the redirect, a redirector KO'd first or by what
    it draws: the partner goes down anyway). A likely SPREAD KO cancels the cover on purpose (kept on review, Oct 9):
    Protect, not a redirect, answers it, and crediting the redirect for the single-target KOs alone turned 7551b7da
    turns 4-5 from Psychic + Bleakwind Storm, which KO Salamence before it moves, into Follow Me beside a Make It Rain
    that still KOs the partner. Returns (that, [(hit, its verdict), ...]). ``status_value``'s cover credit and the
    ``redirect_to_protect_partner`` row both call this, so they agree (``move``: the redirect, as ``redirect_cover``)."""
    verdicts = [(h, redirect_cover(sheet, redirector, t["position"], h, theirs, move))
                for t in sheet.threats for h in t["hits"] if h["into_slot"] != redirector and h["ko"] != "no"]
    held = any(v["holds"] for _, v in verdicts)
    uncovered = any(not v["holds"] and ko_probability(h, with_accuracy=True) >= LIKELY_KO for h, v in verdicts)
    return held and not uncovered, verdicts


def _boosted_row(row: dict, factor: float) -> dict:
    """A damage row with every strike multiplied by ``factor`` (Helping Hand's 1.5)."""
    out = {**row, "damage_pct_of_current_hp": [round(v * factor, 1) for v in row["damage_pct_of_current_hp"]]}
    for key in ("per_hit", "first_hit"):
        if row.get(key):
            out[key] = [v * factor for v in row[key]]
    out["ko"] = _ko_label(*out["damage_pct_of_current_hp"])
    return out


def helping_hand_value(sheet: "TurnSheet", slot_number: int, theirs: dict[int, "Mon"], P: dict | None = None) -> float | None:
    """What Helping Hand from ``slot_number`` creates, in damage points: the partner's best attack (``_best_attack`` on
    its sheet rows) with its damage x1.5 (mech's ``data.helping_hand_boost``; tests/test_mech_d.py holds that against
    @smogon/calc), valued wherever the boosted hit BECOMES a likely KO (>= 50% with accuracy, above the unboosted
    odds) as the boosted row's ``_expected`` minus the plain one's: the damage and the KO bonus it adds, plus the threat
    that foe carries (``threat_weight``) when the partner lands before it moves (priority, then the speed order), and
    discounted by the partner's own chance to act (``survival_factor``). 0 when the boost creates no likely KO (an intact
    Focus Sash stops a single boosted hit too). None when there is nothing to boost: the partner has no attack the hand
    raises (Ruination is a fixed half), or one of its status moves outvalues its best attack (the turn's best_attacks row
    then pairs the hand with that status move, which it does not boost). 07a3cb1f T2: Wave Crash for 68-81% into
    Rillaboom becomes a KO before it moves with the hand (@smogon/calc 143-169 HP into 139); the flat 20 could not see it."""
    P = P or DEFAULTS
    partner = 1 - slot_number
    entry = next((e for e in sheet.our_options if e.get("slot") == partner), None)
    ours = {m.position: m for m in (sheet.field.actives if sheet.field is not None else []) if m.side == "mine"}
    if not entry or partner not in ours:
        return None
    weights = threat_weights(sheet, ours, theirs, P)
    opt, row, attack = _best_attack(entry, P=P, threats=weights)
    if not opt or not row or opt.get("move") in data.FIXED_DAMAGE_MOVES or opt.get("move") in FRACTION_MOVES:
        return None
    for mv in ours[partner].moves:
        mid = mv.get("id")
        if not data.is_damaging(mv) and mid not in PROTECT_LIKE and mid not in ("fakeout", "helpinghand") \
                and status_value(mid, ours[partner], theirs, sheet, partner)[0] > attack:
            return None
    factor = float(data.helping_hand_boost("helpinghand", opt.get("move")))
    if factor <= 1.0:
        return None
    prio = opt.get("priority", 0) or 0
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    by_species = {species_key(m.species): pos for pos, m in theirs.items()}
    value = 0.0
    for r in row.get("rows") or [row]:
        pos = r["target"] if r.get("target", 0) > 0 else by_species.get(species_key(r.get("species")))
        foe = theirs.get(pos) if r.get("side") == "theirs" else None
        if foe is None or (data.to_id(foe.item) == "focussash" and foe.hp_fraction >= 0.999 and not r.get("per_hit")):
            continue
        up = _boosted_row(r, factor)
        before, after = ko_probability(r, with_accuracy=True), ko_probability(up, with_accuracy=True)
        if after < LIKELY_KO or after <= before:
            continue
        their_prio = their_priority_into(sheet, pos)
        first = prio > their_prio or (prio == their_prio and rank.get(("mine", partner), 99) < rank.get(("theirs", pos), 99))
        threat = weights.get(species_key(foe.species), 0.0) if first else 0.0
        value += _expected(up, 0, P, threat) - _expected(r, 0, P, threat)
    return value * survival_factor(partner, sheet, prio, rank, row, P) if value > 0 else 0.0


def status_value(move_id: str, me: "Mon", theirs: dict[int, "Mon"], sheet: "TurnSheet", slot_number: int,
                 has_ally: bool = True) -> tuple[float, int]:
    """(value in 'expected damage points', target int) for a non-damaging move; 0 when pointless. Moves that only help
    (or shield) a partner are worth nothing without one (Oct 7: Helping Hand into an empty slot was our 'best' move)."""
    if move_id in ALLY_ONLY and not has_ally:
        return (0.0, 0)
    if move_id == "taunt":
        best = (0.0, 0)
        for pos, opp in theirs.items():
            value = taunt_value(me, slot_number, pos, opp, sheet)
            if value > best[0]:
                best = (value, pos)
        return best
    if move_id in SLEEP_MOVES:
        best = (0.0, 0)
        terrain = sheet.field.terrain if sheet.field is not None else None
        info = {"id": move_id, **(data.move_info(move_id) or {})}
        for pos, opp in theirs.items():
            if data.status_move_blocked(info, opp, user=me, terrain=terrain):
                continue  # mech's immunity list: Grass / Overcoat / Safety Goggles, Insomnia, Electric Terrain, statused...
            danger = sum(_expected(h, 0) for t in sheet.threats if t["position"] == pos for h in t["hits"])
            value = 50 + min(danger, 60) / 3
            if value > best[0]:
                best = (value, pos)
        return best
    if move_id in REDIRECT_MOVES:
        # The cover credit (45) only when the redirection holds for the partner (``redirect_holds``); otherwise it just
        # draws hits, like Follow Me with no KO to stop (15), and nothing when every foe ignores it (Rage Powder vs Grass).
        if theirs and all(data.redirect_ignored(move_id, foe) for foe in theirs.values()):
            return (0.0, 0)
        return ((45 if redirect_holds(sheet, slot_number, theirs, move_id)[0] else 15), 0)
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
    if move_id == "helpinghand":
        # by the KO it creates (helping_hand_value); the flat value stays the floor (the extra half of a hit), and
        # nothing when the partner has no hit for it to boost
        made = helping_hand_value(sheet, slot_number, theirs)
        return ((0.0 if made is None else max(float(MISC_VALUE[move_id]), round(made, 1))), 0)
    if move_id in MISC_VALUE:
        value = MISC_VALUE[move_id]
        if move_id in ("nastyplot", "swordsdance", "dragondance", "calmmind", "bulkup") and me.hp_fraction < 0.6:
            value = 5
        return (float(value), 0)
    return (0.0, 0)


def setup_spent(sheet: "TurnSheet", move_id: str, memory: "MatchMemory | None" = None) -> bool:
    """Would our ``move_id`` set nothing this turn: our Tailwind is already up, or the Trick Room that is up is ours, so a
    second one ENDS it (ours: the last Trick Room the protocol log shows is ours, or the log shows none). The redirect
    and Fake Out rows then pair their move with an attack, not with this "setup"."""
    if move_id == "tailwind":
        return any("Our Tailwind" in n for n in sheet.notes)
    if move_id != "trickroom" or not sheet.trick_room:
        return False
    used = {side: max((t for by in ((getattr(memory, "moves_by_turn", None) or {}).get(side) or {}).values()
                       for t, mv in by.items() if mv == "trickroom"), default=0) for side in ("mine", "theirs")}
    return used["mine"] >= used["theirs"]


def setup_backfires(sheet: "TurnSheet", move_id: str, memory: "MatchMemory | None" = None) -> bool:
    """Would our ``move_id`` turn the speed order against us: a Trick Room under our own Tailwind (the room reverses the
    order), or a Tailwind under our own Trick Room (doubled, a slow one of ours stops moving first), when one of our
    actives would then move after a foe it moves before now (the sheet's speed order). Never on the last turn of the
    condition it reverses (``sheet.timers``), the standard hand-offs: a Trick Room set then (-7, after everyone has
    moved) starts as our Tailwind ends, and a Tailwind set on our room's last turn runs its three more turns without the
    room. The redirect and Fake Out rows then pair their move with an attack; otherwise the setup stays (it can still
    help another pairing)."""
    ours = [e["speed"] for e in sheet.speed_order if e["side"] == "mine"]
    foes = [e["speed"] for e in sheet.speed_order if e["side"] == "theirs"]

    def ends_now(name: str) -> bool:
        left = sheet.timers.get(name)
        return left is not None and left <= 1

    if move_id == "trickroom" and not sheet.trick_room and setup_spent(sheet, "tailwind"):
        return not ends_now("tailwind") and any(o > f for o in ours for f in foes)
    if move_id == "tailwind" and setup_spent(sheet, "trickroom", memory) and not setup_spent(sheet, "tailwind"):
        return not ends_now("trickroom") and any(o < f < 2 * o for o in ours for f in foes)
    return False


def priority_move_blocked(sheet: "TurnSheet", target: "Mon") -> bool:
    """Does a priority move of ours (Prankster's +1 status moves included) fail into ``target``: Psychic Terrain with the
    target grounded, or Armor Tail / Dazzling / Queenly Majesty on the target's side (the check build_sheet makes for
    priority attacks)."""
    fs = sheet.field
    if fs is not None and fs.terrain == "psychic" and target.grounded:
        return True
    side = [m for m in (fs.actives if fs is not None else []) if m.side == target.side] or [target]
    return any(data.to_id(m.ability) in ("armortail", "dazzling", "queenlymajesty") for m in side)


TAUNT_USE = {"sleep": 0.6, "redirect": 0.5, "speed": 0.25, "setup": 0.3, "status": 0.4, "other": 0.2}  # how often it would use each
TAUNT_SETUP = {"nastyplot", "swordsdance", "dragondance", "calmmind", "bulkup", "quiverdance", "shellsmash", "bellydrum", "coil",
               "irondefense", "agility", "shiftgear", "victorydance", "tidyup", "filletaway"}


def taunt_value(me: "Mon", slot_number: int, pos: int, foe: "Mon", sheet: "TurnSheet", P: dict | None = None) -> float:
    """What our Taunt on the foe at ``pos`` blocks over its three turns, in damage points: its sleep move (one use costs
    us ``sheet.sleep_threat``), its redirection, Tailwind / Trick Room, setup, burns and paralysis, each weighted by how
    often it would use them (``TAUNT_USE``) and by the turns that matter (this turn only when our Taunt lands before
    that move: Prankster or faster, and not against Rage Powder's +2; later turns at ``sleep_future_w``). 0 when Taunt
    fails on it (mech's ``data.status_move_blocked``: Oblivious, Aroma Veil, Mental Herb, Magic Bounce, Good as Gold,
    Prankster into a Dark type). Replaces Taunt's flat 0 (the Oct 8 evening loss, turn 1: Gyarados's Taunt on Amoonguss
    stops Spore and Rage Powder for three turns; the code never offered it)."""
    P = P or DEFAULTS
    if data.status_move_blocked({"id": "taunt", **(data.move_info("taunt") or {})}, foe, user=me):
        return 0.0
    if species_key(foe.species) in sheet.taunted:
        return 0.0  # Taunted already: a second Taunt fails (the volatile has no restart)
    my_prio = 1 if data.to_id(me.ability) == "prankster" else 0
    if my_prio > 0 and priority_move_blocked(sheet, foe):
        return 0.0  # a Prankster Taunt is a +1 move: Psychic Terrain (grounded target) and Armor Tail stop it
    w = float(P.get("sleep_future_w", 0.5))
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    faster = rank.get(("mine", slot_number), 99) < rank.get(("theirs", pos), 99)
    their_side = sheet.field.their_side if sheet.field is not None else set()
    total = 0.0
    for mv in foe.moves:
        mid = mv["id"]
        info = data.move_info(mid) or {}
        if data.is_damaging(info) or mid in PROTECT_LIKE:
            continue
        prio = (info.get("priority") or 0) + (1 if data.to_id(foe.ability) == "prankster" else 0)
        now = my_prio > prio or (my_prio == prio and faster)
        if mid in SLEEP_MOVES:
            per_use, kind, uses = float(sheet.sleep_threat.get(pos, 0.0)), "sleep", 2.0
        elif mid in REDIRECT_MOVES:  # nothing to block when every one of ours ignores it (Rage Powder: Grass, Goggles...)
            ours = [m for m in (sheet.field.actives if sheet.field is not None else []) if m.side == "mine"] or [me]
            per_use, kind, uses = (0.0 if all(data.redirect_ignored(mid, m) for m in ours) else 20.0), "redirect", 3.0
        elif mid in ("tailwind", "trickroom"):
            up = (mid == "tailwind" and "tailwind" in their_side) or (mid == "trickroom" and sheet.trick_room)
            per_use, kind, uses = (0.0 if up else float(SETUP_MOVES[mid])), "speed", 1.0
        elif mid in TAUNT_SETUP:
            per_use, kind, uses = 30.0, "setup", 2.0
        elif mid in STATUS_MOVES or data.STATUS_CONDITION_MOVES.get(mid):
            per_use, kind, uses = float(STATUS_MOVES.get(mid, 30)), "status", 2.0
        else:
            per_use, kind, uses = 10.0, "other", 1.0
        turns = ([1.0] if now else []) + [w] * (2 if now else 3)
        total += per_use * TAUNT_USE[kind] * sum(turns[: int(uses)])
    if foe.status in ("slp", "frz"):
        total *= 0.5  # it acts only once it wakes or thaws
    return round(total, 1)


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
    # its sleep move is an action too: the turns of ours it takes (the KO on a Spore user removes that, Part 3)
    return max(max(by_move.values(), default=0.0), float(sheet.sleep_threat.get(pos, 0.0)))


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


def pinch_heal(defender: "Mon | None") -> tuple[float, float, float] | None:
    """The defender's pinch berry (Sitrus, Wiki, Aguav, Figy, Mago, Iapapa; mech's ``data.pinch_berry``, Gluttony and
    Ripen included) in % of its CURRENT HP: (it triggers once the HP left is at or below this, it heals this much, its max
    HP). None without one (or once it is eaten: the item is gone)."""
    if defender is None or defender.hp_fraction <= 0:
        return None
    berry = data.pinch_berry(defender.item, defender.ability)
    if not berry:
        return None
    hp = max(float(defender.hp_fraction), 0.01)
    threshold, heal = berry
    return threshold / hp * 100.0, heal / hp * 100.0, 100.0 / hp


def _reach(row: dict, need: float) -> float:
    """Chance one hit's roll (uniform over its range, % of the HP at the start of the turn) reaches ``need``."""
    lo, hi = row["damage_pct_of_current_hp"]
    if lo >= need:
        return 1.0
    if hi < need:
        return 0.0
    return (hi - need) / max(hi - lo, 0.1)


def ko_in_order(hits: list[dict], defender: "Mon | None" = None) -> float:
    """Chance hits that all land, in this order, KO the defender: one hit by its own odds (multi-hit rows exactly); two
    or more by their summed rolls, and when the defender holds a pinch berry the berry heals once, as soon as a hit
    leaves it at its threshold, so the later hits must cover the heal too (mech's ``data.pinch_berry``; the Oct 7
    review's m2 T3 Flash Cannon + Earthquake into the 75% Wiki Berry Sylveon is a 0.80 KO, not 0.975: a high Flash
    Cannon roll leaves it at a quarter and the berry puts a third back before the Earthquake)."""
    if not hits:
        return 0.0
    if len(hits) == 1:
        return ko_probability(hits[0])
    berry = pinch_heal(defender)
    if berry is None:
        return linear_ko(sum(h["damage_pct_of_current_hp"][0] for h in hits), sum(h["damage_pct_of_current_hp"][1] for h in hits))
    trigger, heal, cap = berry

    def go(i: int, left: float, held: bool) -> float:
        row = hits[i]
        if i == len(hits) - 1:
            return _reach(row, left)
        lo, hi = row["damage_pct_of_current_hp"]
        n = 16 if len(hits) - i == 2 else 8
        total = 0.0
        for k in range(n):
            rest = left - (lo + (hi - lo) * (k + 0.5) / n)
            if rest <= 0:
                total += 1.0
            elif held and rest <= trigger + 1e-9:
                total += go(i + 1, min(cap, rest + heal), False)
            else:
                total += go(i + 1, rest, held)
        return total / n

    return go(0, 100.0, True)


def _outcome(parts: list[tuple[float, dict]], defender: "Mon | None" = None) -> tuple[float, float]:
    """(expected damage capped at a KO, KO chance) of up to a few hits into ONE foe, each landing with its own
    probability (the attacker's chance to act and the move's accuracy) and rolling on its own. Enumerates which hits
    land; a single hit keeps the exact multi-hit odds, two or more use the summed range (``ko_in_order``: in the order
    given, which is the order they land, the ``defender``'s pinch berry healing between them). Damage is in % of the
    foe's current HP, each hit capped at a KO BEFORE it is weighted, so an overkill hit at a 50% flinch is worth half a
    KO, not a full one (the Oct 8 loss, turn 3: best_attacks scored its no-flinch 125 under a fresh Fake Out)."""
    parts = [(max(0.0, min(1.0, p)), r) for p, r in parts if p > 0 and r.get("damage_pct_of_current_hp", [0, 0])[1] > 0]
    if not parts:
        return 0.0, 0.0
    berry = defender if pinch_heal(defender) is not None else None
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
        if len(landed) == 1:
            ko += prob * ko_probability(landed[0])
        else:
            ko += prob * (ko_in_order(landed, berry) if berry is not None else linear_ko(lo, hi))
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
    factor *= sheet.acts(slot_number)  # asleep: it acts only if it wakes (0, 1/3, 1/2, 1 by missed attempts)
    risk = sheet.sleep_risk.get(slot_number)
    if risk and my_priority <= risk[1]:
        factor *= 1.0 - risk[0]  # a faster sleep move (Spore) aimed at this slot takes this turn's action too
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
    a flat 50% (the Oct 8 practice game: six impossible 'goes down' warnings from Scale Shot + Ruination); a pinch berry
    (Sitrus, Wiki and the rest: mech's ``data.heal_between_hits``) heals between the hits once the first leaves it at its
    threshold, in the range and in the KO odds (``ko_in_order``); accuracy enters the KO odds."""

    def after(a: float, b: float, second_is_fraction: bool) -> float:
        """Total damage of the two hits (uncapped, so an overkill keeps its margin), in % of the HP at the start."""
        heal = data.heal_between_hits(me, a) if a < 100 else 0.0
        remaining = 100.0 - a + heal
        taken = max(remaining, 0.0) / 2 if second_is_fraction else b
        return a - heal + taken

    a_lo, a_hi = first["damage_pct_of_current_hp"]
    b_lo, b_hi = second["damage_pct_of_current_hp"]
    second_frac = second["move"] in FRACTION_MOVES
    lo, hi = after(a_lo, b_lo, second_frac), after(a_hi, b_hi, second_frac)
    acc_a, acc_b = float(first.get("hit_chance", 1.0)), float(second.get("hit_chance", 1.0))
    # A fraction move never finishes: with it second, only the first hit's own KO counts.
    both = acc_a * acc_b * (ko_probability(first) if second_frac else ko_in_order([first, second], me))
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


def sleep_switch_cost(sheet: "TurnSheet", species: str, preview: dict, theirs: dict[int, "Mon"], P: dict | None = None) -> float:
    """What sending ``species`` into their sleep move costs, in damage points, when it is a target Spore would not
    otherwise have: the chance the sleeper uses it on the newcomer (``sleep_use_prior``, halved when our other active is a
    target too) times its accuracy, times the newcomer's next turns of action (its preview's best hit, at
    ``sleep_future_w`` per turn over the 1-3 turn sleep)."""
    P = P or DEFAULTS
    users = sleep_users(sheet, theirs)
    if not users:
        return 0.0
    acc = max(data.hit_chance(mv, sheet.field.weather if sheet.field else None) for moves in users.values() for mv in moves)
    aim = 0.5 if any(t.get("sleep") for t in sheet.threats) else 1.0  # our other active can be slept too: it may take the Spore instead
    action = max((min(_avg(d["damage_pct"]), 100.0) * float(d.get("hit_chance", 1.0)) for d in preview.get("deals_best") or []), default=0.0)
    return float(P.get("sleep_use_prior", 0.6)) * acc * aim * max(action, 30.0) * float(P.get("sleep_future_w", 0.5)) * SLEEP_TURNS


def safe_switch(slot: dict, theirs: dict[int, "Mon"], memory: "MatchMemory", *, sheet: "TurnSheet | None" = None,
                exclude: str | None = None, allow_sleep: bool = False) -> tuple[int | None, str | None, dict, bool]:
    """The switch option to offer for this slot, from mech's switch-in verdict (both foes, their double, hazards, entry
    effects): (index, species, preview, safe). The best 'safe' switch-in first; when every bench card is KO'd on entry
    half the time or more, the least bad one with ``safe`` False, so the row can say so. (Oct 8 practice game, turn 12:
    Milotic was sent into Spikes + Wood Hammer; the pivot check had looked at one attacker.) While their sleep user can
    act, a switch-in that hands it a target it would not otherwise have (the Pokémon going out is asleep or immune:
    ``fresh_sleep_target``) is not safe either, unless ``allow_sleep`` (a slot facing a KO: losing it is worse than a
    slept newcomer), when it is offered with its ``sleep_cost``; a sleep sink (asleep or immune) is preferred over a
    better-scoring sleepable card. The returned preview of an unsafe pick says why (``unsafe``: "ko_on_entry" or "sleep")."""
    best_safe: tuple[int | None, str | None, dict, tuple] = (None, None, {}, (False, float("-inf")))
    best_any: tuple[int | None, str | None, dict, float] = (None, None, {}, float("-inf"))
    reasons: dict[int, str] = {}
    outgoing = next((m for m in (sheet.field.actives if sheet is not None and sheet.field is not None else [])
                     if m.side == "mine" and m.position == slot.get("slot", 0)), None)
    for index, option in enumerate(slot.get("options") or []):
        if option.get("type") != "switch" or not option.get("species") or option.get("species") == exclude:
            continue
        preview = _preview_for(sheet, option["species"], slot.get("slot", 0))
        value = preview["score"] if preview else switch_value(option["species"], theirs, memory)
        if value > best_any[3]:
            best_any = (index, option["species"], preview, value)
        sleepy = fresh_sleep_target(sheet, option["species"], theirs, memory, outgoing=outgoing) if sheet is not None else None
        if preview.get("verdict", "safe") != "safe":
            reasons[index] = "ko_on_entry"
        elif sleepy and not allow_sleep:
            reasons[index] = "sleep"
        else:
            cost = sleep_switch_cost(sheet, option["species"], preview, theirs) if sleepy else 0.0
            key = (sheet is not None and sleep_sink(sheet, option["species"], theirs, memory), value - cost)
            if key > best_safe[3]:
                best_safe = (index, option["species"], {**preview, "sleep_cost": round(cost, 1)} if cost else preview, key)
    if best_safe[0] is not None:
        return best_safe[0], best_safe[1], best_safe[2], True
    if best_any[0] is None:
        return None, None, {}, False
    return best_any[0], best_any[1], {**best_any[2], "unsafe": reasons.get(best_any[0], "ko_on_entry")}, False


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
    protect_p: dict[str, float] = {}  # species_key -> chance it Protects (protect_prior_rows; filled below)

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

    def can_protect(foe: "Mon") -> bool:
        return any(m["id"] in PROTECT_LIKE for m in foe.moves) and not (memory is not None and memory.opp_protect_streak.get(species_key(foe.species))) \
            and species_key(foe.species) not in sheet.taunted  # a Taunted foe cannot Protect

    def aimed_last_turn(foe: "Mon") -> bool:
        """Did one of our single-target moves name this foe last turn (the protocol log's move targets)?"""
        if memory is None or not isinstance(sheet.turn, int):
            return False
        key = species_key(foe.species)
        last = (getattr(memory, "targets_by_turn", None) or {}).get("mine", {}).get(sheet.turn - 1, [])
        moves = (getattr(memory, "moves_by_turn", None) or {}).get("mine", {})
        for who, target in last:
            move = (moves.get(who) or {}).get(sheet.turn - 1)
            if species_key(target) == key and (data.move_info(move) or {}).get("target") not in SPREAD_TARGETS:
                return True
        return False

    protect_cache: dict[int, float] = {}

    def protect_risk(pos: int) -> float:
        """Chance the foe at ``pos`` Protects this turn: ``opp_protect_risk`` flat (the old focus-fire discount), or with
        ``protect_prior_rows`` the reply model's search-class prior for the threat it faces (``replies._p_protect``: a
        single-move KO 0.47, our two hits together 0.40, >= 30% with no KO 0.32, a lead on turn 1 0.57, from the Oct 7-8
        pooled tally), raised to ``protect_aimed_risk`` (0.62) when it faces a KO and one of our single-target moves aimed at it
        last turn (Protect 4 of 6 then, 2 of 7 otherwise)."""
        if not P.get("protect_prior_rows", 0):
            return P.get("opp_protect_risk", 0.3)
        if pos in protect_cache:
            return protect_cache[pos]
        from . import replies  # local import: replies imports this module

        foe = theirs[pos]
        best: dict[int, tuple[float, dict]] = {}
        facing = 0.0
        for n in ours:
            for opt in by_slot.get(n, {}).get("options") or []:
                if opt.get("fails"):
                    continue
                for r in rows_into(opt, pos):
                    p = ko_probability(r, with_accuracy=True)
                    facing = max(facing, _avg(r["damage_pct_of_current_hp"]))
                    key = min(_avg(r["damage_pct_of_current_hp"]), 100.0) + 100 * p
                    if n not in best or key > best[n][0]:
                        best[n] = (key, r)
        ko_one = max((ko_probability(r, with_accuracy=True) for _, r in best.values()), default=0.0)
        ko_two = _outcome([(float(r.get("hit_chance", 1.0)), r) for _, r in best.values()], foe)[1] if len(best) == 2 else 0.0
        # the search class's rates, whatever the class detector says: the default scorer stays a function of the battle
        # state (a restarted agent sees the same sheet), and it is the class that punishes a missed Protect read
        fresh_in = species_key(foe.species) in set(getattr(memory, "opp_fresh_active", None) or [])
        risk = replies._p_protect("search", streak=0, ko_one=ko_one, ko_two=ko_two if ko_one < 0.5 else 0.0, facing=facing,
                                  fresh_in=fresh_in, turn=sheet.turn)
        if max(ko_one, ko_two) >= 0.5 and aimed_last_turn(foe):
            risk = max(risk, float(P.get("protect_aimed_risk", 0.62)))
        protect_cache[pos] = risk
        return risk

    def best_action(number: int, *, allow_ally_only: bool = True) -> tuple[dict | None, dict | None, float]:
        """Best attack or status move, as (option, row, value): the ATTACK is the option that keeps the most value once
        its chance to act is counted (the strongest raw move, discounted afterwards, picked a Kowtow Cleave worth 78 x 0
        over a Sucker Punch worth 64 x 1 in the Oct 8 loss, turn 6); the raw best stands when every option is worth 0.
        ``allow_ally_only`` off leaves out Helping Hand / Coaching / Follow Me: next to a partner that Protects or
        switches they do nothing (Oct 7 match 1 turn 5: Protect + Helping Hand was code #1)."""
        entry = by_slot.get(number, {})
        raw = _best_attack(entry, P=P, threats=weights, protect=protect_p)
        choice: tuple[dict | None, dict | None, float] = (None, None, -1.0)
        if raw[0]:
            best_disc = (raw[0], raw[1], raw[2] * survival_factor(number, sheet, raw[0].get("priority", 0) or 0, speed_rank, raw[1], P))
            for opt in entry.get("options") or []:
                if opt.get("fails") or not opt.get("targets"):
                    continue
                prio = opt.get("priority", 0) or 0
                one = _best_attack({"options": [opt]}, P=P, threats=weights, protect=protect_p)
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
                if prio > 0 and info.get("target") in ("normal", "any", "adjacentFoe") and target in theirs and priority_move_blocked(sheet, theirs[target]):
                    sv = 0.0  # a +1 status move into Psychic Terrain (grounded target) or an Armor Tail side fails
                sv *= survival_factor(number, sheet, prio, speed_rank, None, P) if prio <= 0 else sheet.acts(number)
                if move_id == "taunt":
                    sv *= 1.0 - secondary_flinch(sheet, number, ours, speed_rank, prio)  # Icicle Crash first: no Taunt
                if sv > 0 and sv > choice[2]:
                    legal = option.get("targets") or []
                    tgt = target if target in legal else (legal[0] if legal else 0)
                    pseudo = {"option": index, "move": move_id, "priority": prio}
                    choice = (pseudo, {"target": tgt, "species": theirs[tgt].species if tgt in theirs else "field", "side": "theirs",
                                       "damage_pct_of_current_hp": [0, 0], "status_value": round(sv, 1)}, sv)
        return choice

    if P.get("protect_prior_rows", 0):
        # every row prices the foe's Protect (the search-class tally's rate for the threat it faces), not only focus fire
        protect_p.update({species_key(foe.species): protect_risk(pos) for pos, foe in theirs.items() if can_protect(foe)})
    best = {n: best_action(n) for n in (0, 1)}
    solo = {n: best_action(n, allow_ally_only=False) for n in (0, 1)}  # what a slot does while its partner Protects / switches
    for n in (0, 1):
        opt = best[n][0]
        if opt and P.get("choice_lock_rerank", 1) and sheet.choice_locks.get(n, {}).get(opt.get("move")):
            # A Choice-locked move with a dead target left: prefer the best option that stays useful, if it is close.
            usable = _best_attack({"options": [o for o in by_slot.get(n, {}).get("options") or [] if o.get("move") not in sheet.choice_locks[n]]}, P=P, threats=weights, protect=protect_p)
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
                one = _best_attack({"options": [other]}, P=P, threats=weights, protect=protect_p)
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
    def redirect_gap(number: int, killer_pos: int, hit: dict) -> tuple[dict, str] | None:
        """When the partner offers Follow Me / Rage Powder but it would NOT keep ``hit`` off slot ``number``
        (``redirect_cover``: a spread move, or a redirector a faster foe likely KOs first): (the verdict, "slot N's
        <move> does not cover it: <why>"). None when it would, or there is no redirect to offer."""
        mate = 1 - number
        move_id = next((data.to_id(o.get("move_id")) for o in (slot_template(mate) or {}).get("options") or []
                        if o.get("type") == "move" and data.to_id(o.get("move_id")) in REDIRECT_MOVES), None)
        if move_id is None or mate not in ours:
            return None
        verdict = redirect_cover(sheet, mate, killer_pos, hit, theirs, move_id)
        return None if verdict["covers"] else (verdict, f"slot {mate}'s {move_id} does not cover it: {verdict['why']}")

    endangered: set[int] = set()
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["ko"] == "no":
                continue
            number = hit["into_slot"]
            me = ours.get(number)
            slot = slot_template(number) or {}
            their_prio = hit.get("priority", 0) or 0
            wake = sheet.acts(number)
            state = "asleep" if number in sheet.wake else ("frozen" if me is not None and me.status == "frz" else "paralysed")
            first_normally = acts_before(number, threat["position"], 0, their_prio) and wake > 0  # a sleeper that cannot act moves before nobody
            if first_normally:
                # Slower than us: it only lands if we leave the slot in and fail to remove it. Say so (not a warning).
                if hit["ko"] == "guaranteed" or ko_probability(hit) >= 0.5:
                    endangered.add(number)
                    gap = redirect_gap(number, threat["position"], hit)  # 07a3cb1f T2: never an escape the redirector cannot give
                    sheet.notes.append(
                        f"EXPOSED: slot {number} ({me.species if me else '?'}) moves before {threat['species']}"
                        + (f" if it acts ({state}: {int(round(100 * wake))}%)" if wake < 1 else "") + f", but {threat['species']}'s "
                        f"{hit['move']} KOs it ({hit['damage_pct_of_current_hp'][0]}-{hit['damage_pct_of_current_hp'][1]}%) afterwards unless "
                        + (f"{threat['species']} is removed, flinched or this slot Protects/switches ({gap[1]})." if gap else
                           f"{threat['species']} is removed, flinched, redirected or this slot Protects/switches."))
                continue
            escape = ""
            best_escape = (0.0, "")
            for option in by_slot.get(number, {}).get("options") or []:
                prio = option.get("priority", 0) or 0
                if option.get("fails") or prio <= 0 or not acts_before(number, threat["position"], prio, their_prio):
                    continue
                for r in rows_into(option, threat["position"]):
                    p = ko_probability(r, with_accuracy=True) * wake
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
                          f"It is {state.upper()} ({int(round(100 * wake))}% to act): Protect works only if it acts, so switch it out "
                          "unless trading it wins the game." if wake < 1 else
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
            one = _best_attack({"options": [opt]}, P=P, threats=weights, protect=protect_p)
            if not one[0]:
                continue
            _, turned = extra_exposure(sheet, number, me, opt, one[1], theirs, sheet.field, speed_rank)
            for h in turned[:2]:
                cause = ("Glaive Rush doubles the damage it takes" if move_id == GLAIVE_RUSH else
                         ", ".join(f"{d:+d} {s.upper()}" for s, d in POST_MOVE_STAGES.get(move_id, {}).items()) or "its recoil")
                text = (f"EXPOSED AFTER {move_id}: slot {number} ({me.species}) would take {h['from']}'s {h['move']} for "
                        f"{h['damage_pct_of_current_hp'][0]}-{h['damage_pct_of_current_hp'][1]}% afterwards ({cause}): a KO "
                        f"{int(round(100 * h['ko_chance']))}% of the time that the plain rows do not show.")
                # The partner's redirect does not save it when a faster foe likely KOs the redirector first: a WARNING, not
                # a note (07a3cb1f T2: Wood Hammer KO'd Indeedee, then Draco Meteor KO'd the recoil-weakened Basculegion).
                row = next((x for t in sheet.threats if t["position"] == h["position"] for x in t["hits"]
                            if x["move"] == h["move"] and x["into_slot"] == number), {"move": h["move"]})
                gap = redirect_gap(number, h["position"], row)
                if gap and gap[0]["broken_by"] is not None:
                    sheet.warnings.append(f"{text} {gap[1][0].upper()}{gap[1][1:]}.")
                else:  # the redirector stands: a note, which still says when its redirect misses this hit (ignored, spread)
                    sheet.notes.append(f"{text} {gap[1][0].upper()}{gap[1][1:]}." if gap else text)

    def joint_value(picks: list[tuple[int, dict | None, dict | None, float]], *, protect_aware: bool = False,
                    partner: str | None = None) -> float:
        """Both slots' actions valued TOGETHER. Damage into each foe is pooled across the slots, each hit capped at a KO
        before it is weighted by its slot's chance to act (an overkill at a 50% flinch is half a KO, not a whole one),
        one KO bonus per foe scaled by the threat it removes, the hits taken in the order they land (a pinch berry heals
        between them: ``ko_in_order``). A foe one slot takes out before it moves no longer threatens the other slot (the
        partner's priority or faster sure KO is credited). Each attacking slot is charged for what its own move does to
        it before the slower threats land (stat drops, recoil, Glaive Rush), and for the spread hit its move puts into
        our own ally (Earthquake next to Archaludon: in the same units as damage into a foe, with that Pokémon's KO
        when it can fall to it; nothing when ``partner`` is "protect", the ally's Protect blocking it). With
        ``protect_aware`` a target that can Protect (and didn't last turn) and is in KO range is discounted by the
        chance it does (the Oct 7 opponent Protected the threatened Pokémon three times in four games)."""
        attacks = [(n, opt, row) for n, opt, row, _ in picks if opt and row and "status_value" not in row and opt.get("targets")]
        attacks.sort(key=lambda a: (-(a[1].get("priority", 0) or 0), speed_rank.get(("mine", a[0]), 99)))  # the order they land
        status_picks = [(opt, row, v) for _, opt, row, v in picks if opt and row and ("status_value" in row or not opt.get("targets"))]
        extra = 0.0
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
            removed[pos] = _outcome(parts, foe)[1] if parts else 0.0
        for opt, row, v in status_picks:
            # a Taunt (or any status move) aimed at a foe the partner removes before it moves blocks nothing: Heat Wave
            # KOs Amoonguss before it moves, so a Taunt on it next to the Heat Wave is worth the miss only (review 5714330)
            target = row.get("target", 0)
            extra += max(v, 0.0) * (1.0 - removed.get(target, 0.0)) if target in theirs else max(v, 0.0)
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
            damage, p_ko = _outcome(parts, foe)
            if P.get("hp_units", 0):
                damage *= foe.hp_fraction
            v = damage + p_ko * (P["ko_bonus_guaranteed"] + P.get("ko_threat_w", 0.0) * weights.get(species_key(foe.species), 0.0))
            if P.get("protect_prior_rows", 0):
                v *= 1.0 - protect_p.get(species_key(foe.species), 0.0)  # its Protect blocks the damage too, not only the KO
            elif protect_aware and p_ko > 0 and can_protect(foe):
                v *= 1.0 - protect_risk(pos)
            total += v
        for n, opt, row in attacks:
            prio = opt.get("priority", 0) or 0
            exposed = ally_exposed(n, partner)
            for r in opt.get("targets") or []:
                if r.get("side") == "mine" and r.get("target") == 0 and exposed > 0:
                    extra -= P["ally_damage_w"] * _expected(r, 0, P) * surv[n] * exposed  # friendly fire, like damage into a foe
            extra += P["priority_w"] * prio * surv[n]
            me = ours.get(n)
            if me is not None and P.get("exposure_w", 0):
                gone = {pos: 1.0 - (1.0 - removed.get(pos, 0.0)) * (1.0 - own.get((n, pos), 0.0)) for pos in theirs}
                more, _ = extra_exposure(sheet, n, me, opt, row, theirs, sheet.field, speed_rank, gone=gone)
                doomed = doomed_after_priority(n, sheet, prio, speed_rank, row, removed)  # D1i: the hit lands, the Pokémon is lost
                if more > 0 or doomed > 0:
                    extra -= P["exposure_w"] * min(1.0, more + doomed) * material(me, P) * surv[n]
        return total + extra

    def ally_exposed(n: int, partner: str | None) -> float:
        """How much of a spread hit from slot ``n`` reaches our own ally: all of it, none behind the ally's Protect (a
        repeat Protect only 1 time in 3), all of it on whatever comes in when the ally switches."""
        if partner != "protect":
            return 1.0
        mate = ours.get(1 - n)
        streak = memory.protect_streak(mate.species, 1 - n) if (mate is not None and memory is not None) else 0
        return 1.0 - 0.33 ** streak

    def solo_value(n: int, partner: str | None = None) -> float:
        """The slot's best action next to a partner that Protects or switches (no Helping Hand into it), valued on its
        own: its survival, its KO credit, what its move costs it (its spread hit on the partner only when the partner
        does not Protect)."""
        return joint_value([(n, *solo[n])], partner=partner) if solo[n][0] else 0.0

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
            opt, row, exp = _best_attack(by_slot.get(number, {}), avoid_target=other_target, P=P, threats=weights, protect=protect_p)
            if opt and row and row["target"] == target:
                exp *= survival_factor(number, sheet, opt.get("priority", 0), speed_rank, row, P)
                rows.append((number, opt, row, exp))
        if len(rows) == 2:
            # only count damage from slots that actually get to move
            combined = sum(r["damage_pct_of_current_hp"][0] * (1 if e > 0 else 0) for _, _, r, e in rows)
            in_order = [r for n, o, r, _ in sorted(rows, key=lambda x: (-(x[1].get("priority", 0) or 0), speed_rank.get(("mine", x[0]), 99)))]
            if combined >= 100 and (pinch_heal(opp) is None or ko_in_order(in_order, opp) >= 0.5):
                # a pinch berry between the hits can turn a 'guaranteed' double into a coin flip (Wiki Berry Sylveon: 0.80)
                guard = can_protect(opp)
                # The bonus is for what the SECOND hit adds: when one hit alone already takes the target out, the other is
                # overkill, and the split (finish_the_other) is the better turn (the Oct 8 LLM-agent game, turn 5).
                single = max(ko_probability(r, with_accuracy=True) for _, _, r, _ in rows)
                add("focus_fire", slot_answer(rows[0][1]["option"], target), slot_answer(rows[1][1]["option"], target),
                    f"both into {opp.species}: {combined:.0f}% minimum combined, likely KO" + (" (it can Protect)" if guard else ""),
                    joint_value([(n, o, r, e) for n, o, r, e in rows], protect_aware=True)
                    + P["focus_fire_bonus"] * (1.0 - single) * ((1.0 - protect_risk(target)) if guard else 1.0))

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
        combined_ko = _outcome([(float(r.get("hit_chance", 1.0)) * sheet.acts(n), r) for n, _, r, _ in picks], theirs.get(pos))[1]
        if combined_ko < 0.5:
            continue
        answers = {n: slot_answer(o["option"], r["target"] if r.get("target", 0) > 0 else 0) for n, o, r, _ in picks}
        conditional = [o["move"] for _, o, _, _ in picks if getattr(data, "needs_target_attacking", lambda m: False)(o["move"])]
        add("priority_focus", answers.get(0), answers.get(1),
            f"{threat['species']} KOs slot {lethal[0]['into_slot']} before it moves; slot {picks[0][0]} {picks[0][1]['move']} (+{picks[0][1]['priority']}) + "
            f"slot {picks[1][0]} {picks[1][1]['move']} (+{picks[1][1]['priority']}) take it out first {int(round(combined_ko * 100))}% of the time"
            + (f" ({', '.join(conditional)} fails if it Protects, switches or uses a status move instead)" if conditional else ""),
            joint_value([(n, o, r, v) for n, o, r, v in picks]) + P["focus_fire_bonus"] * combined_ko, covers=lethal[0]["into_slot"])

    def screened_first(killer: int, opt: dict, pos: int) -> bool:
        """Does a Prankster foe's Reflect / Light Screen (not up yet, the setter not Taunted) go up before our ``opt``
        lands (+1 against our priority 0 or less) and leave its hit on the foe at ``pos`` short of a sure KO? Then it
        does not 'already KO' it (07a3cb1f T3: Grimmsnarl's Light Screen held Heat Wave under the Archaludon it was said
        to KO)."""
        info = data.move_info(opt.get("move")) or {}
        screen = {"Physical": "reflect", "Special": "lightscreen"}.get(info.get("category"))
        fs = sheet.field
        if not screen or fs is None or (opt.get("priority", 0) or 0) > 0 or {screen, "auroraveil"} & fs.their_side:
            return False
        if not any(data.to_id(f.ability) == "prankster" and species_key(f.species) not in sheet.taunted
                   and any(m["id"] == screen for m in f.moves) for f in theirs.values()):
            return False
        screened = FieldState(**{**fs.__dict__, "their_side": set(fs.their_side) | {screen}})
        prof = hit_profile(ours[killer], {**info, "id": opt["move"]}, theirs[pos], field=screened, spread=bool(opt.get("spread")))
        return prof is not None and prof["damage_pct_of_current_hp"][0] < 100

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
        kos = [r["pos"] for r in killer_rows if r["damage_pct_of_current_hp"][0] >= 100 and not screened_first(killer, k_opt, r["pos"])]
        if not kos:
            continue
        remaining = [t for t in theirs if t not in kos]
        if not remaining or not slot_template(other) or not ours.get(other):
            continue
        victim, survivor = kos[0], remaining[0]
        alt_opt, alt_row, alt_val = _best_attack(by_slot.get(other, {}), avoid_target=victim, P=P, threats=weights, protect=protect_p)
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
        # the bonus is for a KO that happens: a killer that dies before it moves earns none of it (the Oct 8 win as
        # player 2, turn 2: +15 for a Rillaboom that never moved tied the winning switch)
        killer_acts = survival_factor(killer, sheet, k_prio, speed_rank, k_row, P)
        add("finish_the_other", answers[0], answers[1],
            f"slot {killer} {k_opt['move']} already KOs {theirs[victim].species}; slot {other} {alt_opt['move']} goes into {foe.species} ({alt_row['damage_pct_of_current_hp']}%){why_extra}",
            joint_value([(killer, k_opt, k_row, k_val), (other, alt_opt, alt_row, alt_val)]) + P["focus_fire_bonus"] * 0.5 * killer_acts)

    def protect_index(slot: dict | None) -> int | None:
        return next((i for i, o in enumerate((slot or {}).get("options") or []) if o.get("type") == "move"
                     and data.to_id(o.get("move_id")) in PROTECT_OPTIONS), None)

    def switch_row(endangered: int, other: int, label: str, why: str, base_score: float, *, allow_unsafe: bool,
                   allow_sleep: bool = False) -> None:
        """A switch for the endangered slot while the other slot attacks, from mech's verdict: the best safe switch-in;
        when none survives entry, the least bad one, priced for the KO it risks and saying so (never offered at all
        when ``allow_unsafe`` is off: switching just donates a Pokémon). With ``allow_sleep`` a newcomer their sleep move
        would gain as a target is offered at its sleep cost."""
        slot = slot_template(endangered)
        if not slot or memory is None or not solo[other][0]:
            return
        idx, species, preview, safe = safe_switch(slot, theirs, memory, sheet=sheet, allow_sleep=allow_sleep)
        if idx is None or not species:
            return
        if not safe:
            if not allow_unsafe:
                return
            odds = int(round(100 * float(preview.get("ko_on_entry", 0.0) or 0.0)))
            why += f"; no bench card survives entry: {species} is KO'd on entry {odds}% of the time"
            base_score -= _switch_entry_cost(preview, species, sheet, memory, P)
        elif preview.get("sleep_cost"):
            why += f"; {species} goes in where their sleep move had no target (priced at {preview['sleep_cost']})"
            base_score -= float(preview["sleep_cost"])
        elif preview.get("sleep_target"):
            why += f"; {species} goes in ({preview['sleep_target']}, as the Pokémon it replaces could be)"
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
                bonus *= sheet.acts(endangered_slot)  # asleep: the Protect happens only if it wakes
                if can_remove_first:
                    bonus *= flinch  # we'd KO it first, unless Fake Out stops us: Protect is insurance worth the flinch odds
                bonus *= 1.0 - removed
                add("protect_threatened", answers[0], answers[1],
                    f"{threat['species']}'s {hit['move']} can KO slot {endangered_slot} ({hit['damage_pct_of_current_hp']}%) {when}; Protect it, slot {other} attacks"
                    + (f" (slot {other} KOs {threat['species']} first {int(removed * 100)}% of the time)" if removed else "")
                    + (f" (Protect used {streak} turn(s) in a row already: only 1 in {3 ** streak} to work)" if streak else ""),
                    solo_value(other, "protect") + bonus, covers=endangered_slot)
            asleep = sheet.acts(endangered_slot) < 1
            if slot and solo[other][0] and (protect is None or streak >= 1 or asleep):
                # No Protect, Protect is a coin we already flipped, or the slot is asleep (its Protect needs it to wake): offer
                # the switch so the judge sees the alternative, unless every switch-in is KO'd on entry (then say so and price
                # it; without a Protect at all, offer it anyway).
                base_score = solo_value(other) + P["switch_threatened_bonus"] + (P["protect_bonus_guaranteed"] if pierces_protect else 0) * ko_probability(hit) \
                    + (P["protect_bonus_guaranteed"] * (1 - 0.33 ** streak) if streak else 0) \
                    + P["protect_bonus_guaranteed"] * (1.0 - sheet.acts(endangered_slot))
                base_score *= 1.0 - removed if removed else 1.0
                _, species_any, preview_any, safe_any = safe_switch(slot, theirs, memory, sheet=sheet, allow_sleep=True)
                if not safe_any and preview_any and preview_any.get("unsafe") == "sleep":
                    # the real reason: the newcomer would be slept, and the Pokémon in the slot is asleep or immune already
                    users = sleep_users(sheet, theirs)
                    pos_s, moves_s = next(iter(users.items()))
                    note = (f"No sleep-safe pivot for slot {endangered_slot}: {species_any}"
                            + (" (and the rest of the bench)" if sum(1 for o in slot.get("options") or [] if o.get("type") == "switch") > 1 else "")
                            + f" can be put to sleep on entry by their {theirs[pos_s].species}'s {moves_s[0]['id']}, while "
                            f"{me_e.species if me_e is not None else 'the Pokémon in it'} cannot be: switching hands their sleep "
                            "move a fresh target. Keep it in, Taunt the sleep user or take it out.")
                    if note not in sheet.warnings:
                        sheet.warnings.append(note)
                elif not safe_any and preview_any:
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
                           base_score, allow_unsafe=False, allow_sleep=True)  # a pivot KO'd on entry is never a row: the warning says why

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
                solo_value(other, "protect") + P["protect_bonus_guaranteed"] * P.get("focus_risk", 0.35) * focus["ko_chance"] * 0.33 ** streak
                * sheet.acts(endangered_slot), covers=endangered_slot)

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
            value = P["protect_bonus_guaranteed"] * worst * 0.33 ** streak * sheet.acts(endangered_slot) + P["switch_weak_bonus"] \
                + max(float(preview.get("score", 0.0) or 0.0), 0.0) * 0.25
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
            real, verdicts = redirect_holds(sheet, redirector, theirs, data.to_id(slot_r["options"][redirect].get("move_id")))
            if not real or not next((v["holds"] for h, v in verdicts if h is hit), False):
                continue  # status_value's check: the partner is not safe behind it, or this hit is not kept off it
            move_info = data.move_info(hit["move"]) or {}
            me_r = ours.get(redirector)
            if me_r is None:
                continue
            # does the redirector survive that hit? (this row's own stricter bar: no KO roll at all)
            attacker = theirs.get(threat["position"])
            est = damage_percent(attacker, {**move_info, "id": hit["move"]}, me_r, field=sheet.field) if attacker else None  # with the field: screens, weather, Ruin (D1k)
            if est is None or est[1] >= 100:
                continue  # it would just die instead
            partner_slot = slot_template(endangered_slot)
            partner_action = None
            for setup in ("tailwind", "trickroom", "icywind", "electroweb"):
                idx = _option_index(partner_slot, "move", move_id=setup) if partner_slot and not (
                    setup_spent(sheet, setup, memory) or setup_backfires(sheet, setup, memory)) else None
                if idx is not None:
                    partner_action = ({"option": idx, "target": 0}, f"sets {setup}")
                    break
            raw_partner = _best_attack(by_slot.get(endangered_slot, {}), P=P, threats=weights, protect=protect_p)
            if partner_action is None and raw_partner[0]:
                partner_action = (slot_answer(raw_partner[0]["option"], raw_partner[1]["target"]), "attacks freely")
            if partner_action is None:
                continue
            answers = {redirector: {"option": redirect, "target": 0}, endangered_slot: partner_action[0]}
            add("redirect_to_protect_partner", answers[0], answers[1],
                f"{threat['species']}'s {hit['move']} would KO slot {endangered_slot}; slot {redirector} redirects it ({est[0]}-{est[1]}% to itself) while slot {endangered_slot} {partner_action[1]}",
                ((P["protect_bonus_guaranteed"] if hit["ko"] == "guaranteed" else P["protect_bonus_possible"]) + 25 + (50 if "sets" in partner_action[1] else raw_partner[2] * 0.8))
                * sheet.acts(redirector),
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
                idx = None if setup_spent(sheet, setup, memory) or setup_backfires(sheet, setup, memory) else \
                    _option_index(other_slot, "move", move_id=setup)
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
        lands = (1.0 - (sheet.flinch_risk.get(number, 0.0) if faster_fakers else 0.0)) * sheet.acts(number)
        if lands < 1.0:
            why += f"; their faster Fake Out may flinch ours first ({int((1 - lands) * 100)}%)"
        add("fake_out_setup", answers[0], answers[1], why,
            (base + (P["fakeout_threat_bonus"] if threatens_us else 0) + P["protect_bonus_guaranteed"] * lethal_first) * lands
            + (solo_value(other) * 0.6 if not setup_name else P["fakeout_setup_value"]))

    # 4b. Taunt the support (Part 3; the Oct 8 evening loss, turn 1: Gyarados's Taunt on Amoonguss stops Spore and Rage
    #     Powder for three turns, and no generator built it): Taunt a foe whose status moves (sleep, redirection, Trick
    #     Room, Tailwind, setup) would cost us, valued by what it blocks (``taunt_value``), the slot's chance to act and a
    #     faster foe's flinch (Icicle Crash 30%), while the partner attacks.
    for number in (0, 1):
        me = ours.get(number)
        slot = slot_template(number)
        if not me or not slot:
            continue
        idx = _option_index(slot, "move", move_id="taunt")
        if idx is None:
            continue
        other = 1 - number
        partner = solo_answer(other) if other in ours else None
        if other in ours and partner is None:
            continue
        prio = 1 if data.to_id(me.ability) == "prankster" else 0
        legal = (slot.get("options") or [])[idx].get("targets") or []
        flinch = secondary_flinch(sheet, number, ours, speed_rank, prio)
        acts = survival_factor(number, sheet, prio, speed_rank, None, P) if prio <= 0 else sheet.acts(number)
        for pos, foe in sorted(theirs.items()):
            if legal and pos not in legal:
                continue
            blocked = taunt_value(me, number, pos, foe, sheet, P)
            if blocked <= 0:
                continue
            if other in ours and solo[other][0] and solo[other][1] is not None:
                o_opt, o_row = solo[other][0], solo[other][1]
                sub = row_into(o_opt, o_row, pos, foe.species)
                o_prio = o_opt.get("priority", 0) or 0
                if sub is not None and "status_value" not in o_row and acts_before(other, pos, o_prio, 1 if data.to_id(foe.ability) == "prankster" else 0):
                    blocked *= 1.0 - ko_probability(sub, with_accuracy=True) * survival_factor(other, sheet, o_prio, speed_rank, o_row, P)
            what = [m["id"] for m in foe.moves if not data.is_damaging(data.move_info(m["id"]) or {}) and m["id"] not in PROTECT_LIKE]
            answers = {number: {"option": idx, "target": pos}, other: partner}
            add("taunt_status", answers.get(0), answers.get(1),
                f"slot {number} Taunts {foe.species}: it cannot use {', '.join(what)} for three turns"
                + (f" ({int(round(100 * flinch))}% that a faster flinch stops the Taunt)" if flinch >= 0.05 else "")
                + (f"; slot {other} attacks" if other in ours else ""),
                blocked * acts * (1.0 - flinch) + (solo_value(other) if other in ours else 0.0))

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
    if P.get("reply_mixture", 0) and not forced and memory is not None:
        # Workstream A: value every row against the opponent's plausible replies (agent/pokemon/replies.py).
        from . import replies  # local import: replies imports this module

        ranked = replies.evaluate(sheet, memory, None, slots=slots, ours=ours, theirs=theirs, params=P, candidates=candidates,
                                  obs={"turn": sheet.turn})
        if ranked:
            candidates = ranked
            # the mixture's rows carry their worst case and best reply: the model sees five (the sixth was within the override
            # margin of the top in 1 of 41 replayed live turns; the payload stays inside the ops budget)
            return candidates[:5]
    # The cut: the best four, plus the Protect / switch rows that cover a slot in a KO range (the Oct 8 loss, turn 3:
    # the row that Protected the exposed Thundurus ranked 5th and was cut; it decided the game).
    kept = candidates[:4]
    for cand in candidates[4:]:
        if len(kept) >= 6:
            break
        if cand["name"] in COVER_ROWS and cand.get("covers") in endangered:
            kept.append(cand)
    return kept
