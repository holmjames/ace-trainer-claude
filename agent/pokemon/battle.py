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
from typing import Any

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

    def stat(self, name: str) -> float:
        base = self.stats.get(name, 0)
        value = base * stage_multiplier(self.boosts.get(name, 0))
        if name == "spe":
            if self.status == "par":
                value *= 0.5
            if data.to_id(self.item) == "choicescarf":
                value *= 1.5
            if self.tailwind:
                value *= 2
        if name == "atk" and data.to_id(self.item) == "choiceband":
            value *= 1.5
        if name == "spa" and data.to_id(self.item) == "choicespecs":
            value *= 1.5
        if name == "spd" and data.to_id(self.item) == "assaultvest":
            value *= 1.5
        if name in ("def", "spd") and data.to_id(self.item) == "eviolite":
            value *= 1.5
        return value


def build_mon(summary: dict, card: dict | None, *, side: str, position: int | None) -> Mon | None:
    merged = {**(card or {}), **{k: v for k, v in (summary or {}).items() if v not in (None, "", [], {})}}
    species = merged.get("species") or merged.get("name")
    if not species:
        return None
    stats = data.actual_stats({**merged, "evs": (card or {}).get("evs") or merged.get("evs"), "nature": (card or {}).get("nature") or merged.get("nature")}) or {}
    moves: list[dict] = []
    for name in (card or {}).get("moves") or merged.get("moves") or []:
        move_name = name.get("id") if isinstance(name, dict) else name
        info = data.move_info(move_name)
        moves.append({"id": data.to_id(move_name), **(info or {"type": None, "category": None, "base_power": 0, "priority": 0, "target": None})})
    frac = summary.get("current_hp_fraction") if isinstance(summary, dict) else None
    if frac is None and isinstance(summary, dict) and summary.get("max_hp"):
        frac = (summary.get("current_hp") or 0) / summary["max_hp"]
    return Mon(
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


# -- damage ------------------------------------------------------------------------------------


def damage_percent(attacker: Mon, move: dict, defender: Mon, *, weather: str | None = None, spread: bool = False) -> tuple[float, float] | None:
    """(min%, max%) of the defender's CURRENT HP, or None when the move does no damage / is immune."""
    power = move.get("base_power") or 0
    if power <= 0:
        return None
    category = move.get("category")
    if category not in ("Physical", "Special"):
        return None
    type_mult = data.effectiveness(move.get("type") or "", defender.types)
    if type_mult == 0:
        return (0.0, 0.0)
    atk_stat, def_stat = ("atk", "def") if category == "Physical" else ("spa", "spd")
    a, d = attacker.stat(atk_stat), defender.stat(def_stat)
    if not a or not d or not defender.stats.get("hp"):
        return None
    base = math.floor(math.floor(math.floor(2 * LEVEL / 5 + 2) * power * a / d) / 50) + 2
    mod = 1.0
    if spread:
        mod *= 0.75
    w = data.to_id(weather)
    move_type = data.to_id(move.get("type"))
    if w in ("rain", "raindance") and move_type == "water" or w in ("sun", "sunnyday") and move_type == "fire":
        mod *= 1.5
    if w in ("rain", "raindance") and move_type == "fire" or w in ("sun", "sunnyday") and move_type == "water":
        mod *= 0.5
    if move_type in attacker.types:
        mod *= 1.5
    mod *= type_mult
    if attacker.status == "brn" and category == "Physical":
        mod *= 0.5
    item = data.to_id(attacker.item)
    if item == "lifeorb":
        mod *= 1.3
    if TYPE_BOOST_ITEMS.get(item) == move_type:
        mod *= 1.2
    if item == "expertbelt" and type_mult > 1:
        mod *= 1.2
    if move.get("id") == "knockoff" and defender.item:
        mod *= 1.5  # Knock Off hits harder when the target holds an item
    if move.get("will_crit"):
        mod *= 1.5  # always a critical hit (Surging Strikes, Wicked Blow, Flower Trick)
    hits = data.expected_hits(move, item=attacker.item)
    full_hp = defender.hp_fraction >= 0.999
    if full_hp and data.to_id(defender.ability) in ("multiscale", "shadowshield"):
        mod *= 0.5
    current_hp = max(1.0, defender.stats["hp"] * defender.hp_fraction)
    low = base * RANDOM_MIN * mod * hits / current_hp * 100
    high = base * RANDOM_MAX * mod * hits / current_hp * 100
    if full_hp and data.to_id(defender.item) == "focussash" and hits <= 1:
        # The sash leaves the holder at 1 HP from a single hit at full health: no KO this turn.
        low, high = min(low, 99.0), min(high, 99.0)
    return (round(min(low, 999.0), 1), round(min(high, 999.0), 1))  # anything past a KO is just "a KO"


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

    def as_prompt(self) -> dict:
        return {
            "WARNINGS": self.warnings,
            "speed_order": self.speed_order,
            "trick_room": self.trick_room,
            "our_damage_estimates": self.our_options,
            "opponent_threats": self.threats,
            "candidate_turns": self.candidates,
            "notes": self.notes,
        }


def _row(target: int, defender: "Mon", est: tuple[float, float]) -> dict:
    return {"target": target, "species": defender.species, "side": defender.side, "damage_pct_of_current_hp": list(est),
            "ko": "guaranteed" if est[0] >= 100 else ("possible" if est[1] >= 100 else "no")}


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


def build_sheet(template: dict, obs: dict, memory: MatchMemory, params: dict | None = None) -> TurnSheet:
    P = params or DEFAULTS
    sheet = TurnSheet()
    slots = sorted(template.get("slots") or [], key=lambda s: s.get("slot", 0))
    weather = obs.get("weather") if isinstance(obs.get("weather"), str) else None
    field_text = data.to_id(json.dumps([obs.get("fields"), obs.get("field")], default=str))
    sheet.trick_room = "trickroom" in field_text
    if sheet.trick_room:
        sheet.notes.append("Trick Room is up: slower Pokémon move first.")
    my_tailwind = "tailwind" in data.to_id(json.dumps(obs.get("side_conditions"), default=str))
    their_tailwind = "tailwind" in data.to_id(json.dumps(obs.get("opponent_side_conditions"), default=str))
    if my_tailwind:
        sheet.notes.append("Our Tailwind is up (speed doubled).")
    if their_tailwind:
        sheet.notes.append("Opponent's Tailwind is up (their speed doubled).")
    if weather:
        sheet.notes.append(f"Weather: {weather}.")

    # Our active Pokémon, one per slot.
    ours: dict[int, Mon] = {}
    for slot in slots:
        species = slot.get("active")
        if isinstance(species, dict):
            species = species.get("species")
        if not species:
            continue
        key = species_key(species)
        mon = build_mon(_find_summary(obs.get("team"), species), memory.my_cards.get(key), side="mine", position=slot.get("slot"))
        if mon:
            mon.tailwind = my_tailwind
            ours[slot.get("slot", 0)] = mon

    # Their active Pokémon by board position (from target_options when present).
    theirs: dict[int, Mon] = {}
    for slot in slots:
        for option in slot.get("options") or []:
            for t in option.get("target_options") or []:
                if t.get("side") == "opponent" and t.get("species") and t.get("target") not in theirs:
                    key = species_key(t["species"])
                    mon = build_mon(_find_summary(obs.get("opponent_team"), t["species"]), memory.opp_cards.get(key), side="theirs", position=t.get("target"))
                    if mon:
                        mon.tailwind = their_tailwind
                        theirs[int(t["target"])] = mon
    if not theirs:
        for index, summary in enumerate(_active_summaries(obs.get("opponent_team"))[:2], start=1):
            mon = build_mon(summary, memory.opp_cards.get(species_key(summary.get("species"))), side="theirs", position=index)
            if mon:
                mon.tailwind = their_tailwind
                theirs[index] = mon

    # Speed order.
    everyone = list(ours.values()) + list(theirs.values())
    speeds = [(m, m.stat("spe")) for m in everyone if m.stats.get("spe")]
    speeds.sort(key=lambda pair: pair[1], reverse=not sheet.trick_room)
    sheet.speed_order = [
        {"species": m.species, "side": m.side, "position": m.position, "speed": int(s)} for m, s in speeds
    ]

    for number, me in sorted(ours.items()):
        if data.to_id(me.item) == "focussash":
            sheet.notes.append(f"Our {me.species} holds Focus Sash" + (" (intact at full HP: survives ONE single hit, not multi-hit or two hits)." if me.hp_fraction >= 0.999 else " but is not at full HP: the sash will not save it."))
    for pos, opp in sorted(theirs.items()):
        if data.to_id(opp.item) == "focussash" and opp.hp_fraction >= 0.999:
            sheet.notes.append(f"Their {opp.species} holds an intact Focus Sash: it needs two hits or a multi-hit move to KO this turn.")

    for number, me in sorted(ours.items()):
        if memory.our_protect_last_turn(me.species, number):
            sheet.notes.append(f"Our {me.species} used Protect LAST turn: a repeat only works 1 time in 3.")
    for key in memory.opp_protected_last_turn:
        name = next((o.species for o in theirs.values() if species_key(o.species) == key), key)
        sheet.notes.append(f"Their {name} probably Protected last turn (we hit it and its HP did not move): a second Protect in a row fails 2/3 of the time, so this is the turn to attack it.")

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
            spread = move.get("target") in SPREAD_TARGETS and len(theirs) > 1
            per_target = []
            targets = option.get("targets") or []
            if targets:
                for target in targets:
                    defender = theirs.get(target) if target > 0 else (ours.get(-target - 1) if target < 0 else None)
                    if defender is None:
                        continue
                    est = damage_percent(me, move, defender, weather=weather, spread=spread)
                    if est is None:
                        continue
                    per_target.append(_row(target, defender, est))
            elif (move.get("base_power") or 0) > 0:
                # No target to choose: a spread move (or a fixed-target move). It hits every
                # opposing active (and, for allAdjacent moves like Earthquake, our own ally too).
                victims = list(theirs.values())
                if move.get("target") == "allAdjacent":
                    victims += [ally for n, ally in ours.items() if n != number]
                for defender in victims:
                    est = damage_percent(me, move, defender, weather=weather, spread=spread)
                    if est is not None:
                        per_target.append({**_row(0, defender, est), "spread": True})
                        if defender.side == "mine" and est[1] > 0:
                            sheet.notes.append(f"{move['id']} from slot {number} also hits our own {defender.species} for {est[0]}-{est[1]}%.")
            if per_target or move.get("base_power"):
                entry["options"].append({"option": index, "move": move["id"], "priority": move.get("priority", 0), "spread": spread, "targets": per_target})
        sheet.our_options.append(entry)

    # Opponent threats from their KNOWN sets.
    for position, opp in sorted(theirs.items()):
        threat = {"position": position, "species": opp.species, "hp_pct": round(opp.hp_fraction * 100), "status": opp.status,
                  "item": opp.item, "ability": opp.ability, "known_moves": [m["id"] for m in opp.moves], "hits": []}
        opp_fresh = species_key(opp.species) in set(memory.opp_fresh_active) or obs.get("turn") in (None, 1)
        for move in opp.moves:
            if (move.get("base_power") or 0) <= 0:
                continue
            if move["id"] in ("fakeout", "firstimpression") and not opp_fresh:
                continue  # only works on the user's first turn out
            for number, me in sorted(ours.items()):
                est = damage_percent(opp, move, me, weather=weather, spread=move.get("target") in SPREAD_TARGETS and len(ours) > 1)
                if est is None:
                    continue
                if me.fainted or me.hp_fraction <= 0:
                    continue
                hit = {"move": move["id"], "into_slot": number, "species": me.species, "damage_pct_of_current_hp": list(est),
                       "ko": "guaranteed" if est[0] >= 100 else ("possible" if est[1] >= 100 else "no"), "priority": move.get("priority", 0)}
                if data.expected_hits(move, item=opp.item) > 1:
                    hit["multi_hit"] = True  # breaks Focus Sash
                threat["hits"].append(hit)
        if "fakeout" in threat["known_moves"]:
            threat["note"] = "has Fake Out (only works on its first turn out)"
        sheet.threats.append(threat)

    sheet.candidates = rank_candidates(slots, ours, theirs, sheet, fresh=set(memory.fresh_active), memory=memory, params=P)
    return sheet


# -- candidates --------------------------------------------------------------------------------------


def _expected(row: dict, priority: int, P: dict | None = None) -> float:
    P = P or DEFAULTS
    lo, hi = row["damage_pct_of_current_hp"]
    return (lo + hi) / 2 + (P["ko_bonus_guaranteed"] if lo >= 100 else (P["ko_bonus_possible"] if hi >= 100 else 0)) + P["priority_w"] * priority


def _best_attack(entry: dict, *, avoid_target: int | None = None, P: dict | None = None) -> tuple[dict | None, dict | None, float]:
    P = P or DEFAULTS
    """Highest expected damage (option, target_row, expected%) among a slot's move options.
    A spread move (rows with target 0) counts the damage to every opposing Pokémon it hits and is
    charged for damage to our own ally."""
    best: tuple[dict | None, dict | None, float] = (None, None, -1.0)
    for option in entry.get("options") or []:
        priority = option.get("priority") or 0
        rows = [r for r in option.get("targets") or [] if r["target"] != avoid_target]
        foe_rows = [r for r in rows if r["side"] == "theirs"]
        spread_rows = [r for r in foe_rows if r["target"] == 0]
        if spread_rows:
            ally_hit = sum(_expected(r, 0, P) for r in rows if r["side"] == "mine")
            expected = sum(_expected(r, priority, P) for r in spread_rows) - P["ally_damage_w"] * ally_hit
            if expected > best[2]:
                summary = {"target": 0, "species": " + ".join(r["species"] for r in spread_rows), "side": "theirs",
                           "damage_pct_of_current_hp": [sum(r["damage_pct_of_current_hp"][0] for r in spread_rows),
                                                        sum(r["damage_pct_of_current_hp"][1] for r in spread_rows)],
                           "ally_damage_pct": round(ally_hit, 1) if ally_hit else 0}
                best = (option, summary, expected)
        for row in foe_rows:
            if row["target"] == 0:
                continue
            expected = _expected(row, priority, P)
            if expected > best[2]:
                best = (option, row, expected)
    return best


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


def status_value(move_id: str, me: "Mon", theirs: dict[int, "Mon"], sheet: "TurnSheet", slot_number: int) -> tuple[float, int]:
    """(value in 'expected damage points', target int) for a non-damaging move; 0 when pointless."""
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


def ko_probability(hit: dict) -> float:
    """Chance a hit KOs, treating the damage roll as uniform between its min and max."""
    lo, hi = hit["damage_pct_of_current_hp"]
    if lo >= 100:
        return 1.0
    if hi < 100:
        return 0.0
    return (hi - 100.0) / max(hi - lo, 0.1)


def survival_factor(slot_number: int, sheet: "TurnSheet", my_priority: int, speed_rank: dict,
                    my_attack_row: dict | None = None, P: dict | None = None) -> float:
    """Chance this slot gets to act: 1.0 if nothing faster can KO it, 0.0 for a guaranteed KO, and in between by
    the KO roll probability (scaled by ``survival_possible``). A threat we act before AND knock out with our own
    attack (``my_attack_row``) doesn't count."""
    P = P or DEFAULTS
    factor = 1.0
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["into_slot"] != slot_number or hit["ko"] == "no":
                continue
            their_priority = hit.get("priority", 0)
            we_first = my_priority > their_priority or (my_priority == their_priority and
                       speed_rank.get(("mine", slot_number), 99) < speed_rank.get(("theirs", threat["position"]), 99))
            if we_first and my_attack_row and my_attack_row.get("target") == threat["position"] \
                    and my_attack_row.get("damage_pct_of_current_hp", [0, 0])[0] >= 100:
                continue  # we remove it before it moves
            if not we_first:
                p_ko = ko_probability(hit)
                factor = min(factor, 1.0 - p_ko * (1.0 if hit["ko"] == "guaranteed" else 2 * P["survival_possible"]))
    return max(0.0, factor)


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


def best_switch(slot: dict, theirs: dict[int, "Mon"], memory: "MatchMemory", *, exclude: str | None = None) -> tuple[int | None, str | None]:
    """(option index, species) of the best switch option in this slot, skipping ``exclude``."""
    best: tuple[int | None, str | None, float] = (None, None, float("-inf"))
    for index, option in enumerate(slot.get("options") or []):
        if option.get("type") != "switch" or option.get("species") == exclude:
            continue
        value = switch_value(option["species"], theirs, memory)
        if value > best[2]:
            best = (index, option["species"], value)
    return best[0], best[1]


def rank_candidates(slots: list[dict], ours: dict[int, "Mon"], theirs: dict[int, "Mon"], sheet: TurnSheet,
                    fresh: set[str] | None = None, memory: "MatchMemory | None" = None, params: dict | None = None) -> list[dict]:
    """A few complete turns worth considering, best first. Each is {name, slot_0, slot_1, why, score}."""
    P = params or DEFAULTS
    by_slot = {e["slot"]: e for e in sheet.our_options}
    fresh = fresh or set()
    speed_rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    candidates: list[dict] = []

    def slot_answer(option_index: int | None, target: int | None) -> dict | None:
        if option_index is None:
            return None
        return {"option": option_index, "target": target if target is not None else 0}

    def add(name: str, s0: dict | None, s1: dict | None, why: str, score: float) -> None:
        if s0 is None or s1 is None:
            return
        cand = {"name": name, "slot_0": s0, "slot_1": s1, "why": why, "score": round(score, 1)}
        if not any(c["slot_0"] == s0 and c["slot_1"] == s1 for c in candidates):
            candidates.append(cand)

    def slot_template(number: int) -> dict | None:
        return next((s for s in slots if s.get("slot", 0) == number), None)

    def acts_before(me_slot: int, opp_pos: int, my_priority: int = 0, their_priority: int = 0) -> bool:
        if my_priority != their_priority:
            return my_priority > their_priority
        return speed_rank.get(("mine", me_slot), 99) < speed_rank.get(("theirs", opp_pos), 99)

    def best_action(number: int) -> tuple[dict | None, dict | None, float]:
        """Best attack (discounted by the chance this slot dies first) or best status move, as (option, row, value)."""
        attack = _best_attack(by_slot.get(number, {}), P=P)
        prio = attack[0].get("priority", 0) if attack[0] else 0
        value = attack[2] * survival_factor(number, sheet, prio, speed_rank, attack[1], P) if attack[0] else -1.0
        choice = (attack[0], attack[1], value)
        slot = slot_template(number)
        me = ours.get(number)
        if slot and me:
            surv = survival_factor(number, sheet, 0, speed_rank, None, P)
            for index, option in enumerate(slot.get("options") or []):
                if option.get("type") != "move":
                    continue
                move_id = data.to_id(option.get("move_id"))
                info = data.move_info(move_id) or {}
                if (info.get("base_power") or 0) > 0 or move_id in ("protect", "detect", "fakeout"):
                    continue
                prio = info.get("priority", 0) or 0
                sv, target = status_value(move_id, me, theirs, sheet, number)
                sv *= survival_factor(number, sheet, prio, speed_rank, None, P) if prio <= 0 else 1.0
                if sv > choice[2]:
                    legal = option.get("targets") or []
                    tgt = target if target in legal else (legal[0] if legal else 0)
                    pseudo = {"option": index, "move": move_id, "priority": prio}
                    choice = (pseudo, {"target": tgt, "species": theirs[tgt].species if tgt in theirs else "field", "side": "theirs",
                                       "damage_pct_of_current_hp": [0, 0], "status_value": round(sv, 1)}, sv)
        return choice

    best = {n: best_action(n) for n in (0, 1)}

    # 0. Forced switches (a Pokémon fainted): bring in the best matchup; the other slot keeps its best action or passes.
    forced = [n for n in (0, 1) if (slot_template(n) or {}).get("force_switch")]
    if forced and memory is not None:
        answers: dict[int, dict] = {}
        used: str | None = None
        for n in forced:
            idx, species = best_switch(slot_template(n), theirs, memory, exclude=used)
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

    # Spell out every KO threat that moves before the slot it targets (priority ignores speed and Tailwind).
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["ko"] == "no":
                continue
            number = hit["into_slot"]
            me = ours.get(number)
            my_prio = best[number][0].get("priority", 0) if best[number][0] else 0
            if acts_before(number, threat["position"], my_prio, hit.get("priority", 0)):
                continue
            prio_note = f" with priority +{hit['priority']}" if hit.get("priority", 0) > 0 else ""
            sheet.warnings.append(
                f"LETHAL: {threat['species']}'s {hit['move']}{prio_note} hits slot {number} ({me.species if me else '?'}) for "
                f"{hit['damage_pct_of_current_hp'][0]}-{hit['damage_pct_of_current_hp'][1]}% BEFORE it can move. "
                f"Protect/switch that slot unless the game is won anyway."
            )

    # 1. Both slots use their own best attack.
    if best[0][0] and best[1][0]:
        add("best_attacks", slot_answer(best[0][0]["option"], best[0][1]["target"]), slot_answer(best[1][0]["option"], best[1][1]["target"]),
            f"slot 0 {best[0][0]['move']} -> {best[0][1]['species']} ({best[0][1]['damage_pct_of_current_hp']}%), "
            f"slot 1 {best[1][0]['move']} -> {best[1][1]['species']} ({best[1][1]['damage_pct_of_current_hp']}%)",
            best[0][2] + best[1][2])

    # 2. Focus fire: both into the same target when together they likely KO it.
    for target, opp in theirs.items():
        rows = []
        for number in (0, 1):
            other_target = next((t for t in theirs if t != target), None)
            opt, row, exp = _best_attack(by_slot.get(number, {}), avoid_target=other_target, P=P)
            if opt and row and row["target"] == target:
                exp *= survival_factor(number, sheet, opt.get("priority", 0), speed_rank, row, P)
                rows.append((number, opt, row, exp))
        if len(rows) == 2:
            # only count damage from slots that actually get to move
            combined = sum(r["damage_pct_of_current_hp"][0] * (1 if e > 0 else 0) for _, _, r, e in rows)
            if combined >= 100:
                add("focus_fire", slot_answer(rows[0][1]["option"], target), slot_answer(rows[1][1]["option"], target),
                    f"both into {opp.species}: {combined:.0f}% minimum combined, likely KO", sum(e for *_, e in rows) + P["focus_fire_bonus"])

    # 3. A slot faces a KO this turn from something that acts before it: Protect it while the other attacks.
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["ko"] not in ("guaranteed", "possible"):
                continue
            endangered, other = hit["into_slot"], 1 - hit["into_slot"]
            my_best = best[endangered]
            my_prio = my_best[0].get("priority", 0) if my_best[0] else 0
            can_remove_first = (my_best[0] and my_best[1] and my_best[1].get("target") == threat["position"]
                                and my_best[1]["damage_pct_of_current_hp"][0] >= 100
                                and acts_before(endangered, threat["position"], my_prio, hit.get("priority", 0)))
            if can_remove_first:
                continue
            slot = slot_template(endangered)
            protect = None
            attacker = theirs.get(threat["position"])
            move_info = data.move_info(hit["move"]) or {}
            pierces_protect = (attacker is not None and data.to_id(attacker.ability) == "unseenfist" and "contact" in (move_info.get("flags") or [])) \
                or hit["move"] in ("feint", "hyperspacefury", "hyperspacehole", "phantomforce", "shadowforce")
            if slot and not pierces_protect:
                protect = next((i for i, o in enumerate(slot.get("options") or []) if o.get("type") == "move"
                                and data.to_id(o.get("move_id")) in ("protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap")), None)
            if pierces_protect:
                note = f"{threat['species']}'s {hit['move']} goes THROUGH Protect (Unseen Fist / Protect-piercing move): switch slot {endangered} out instead of protecting."
                if note not in sheet.warnings:
                    sheet.warnings.append(note)
            if protect is not None and best[other][0]:
                answers = {endangered: {"option": protect, "target": 0}, other: slot_answer(best[other][0]["option"], best[other][1]["target"])}
                bonus = P["protect_bonus_guaranteed"] if hit["ko"] == "guaranteed" else P["protect_bonus_possible"] * (0.5 + ko_probability(hit))
                me_e = ours.get(endangered)
                if me_e is not None and memory is not None and memory.our_protect_last_turn(me_e.species, endangered):
                    bonus *= 0.33  # consecutive Protect succeeds 1/3 of the time
                add("protect_threatened", answers[0], answers[1],
                    f"{threat['species']}'s {hit['move']} can KO slot {endangered} ({hit['damage_pct_of_current_hp']}%) before it moves; Protect it, slot {other} attacks",
                    best[other][2] + bonus)
            elif slot and best[other][0]:
                switch = best_switch(slot, theirs, memory)[0] if memory is not None else _option_index(slot, "switch")
                if switch is not None:
                    answers = {endangered: {"option": switch, "target": 0}, other: slot_answer(best[other][0]["option"], best[other][1]["target"])}
                    add("switch_threatened", answers[0], answers[1],
                        f"{threat['species']}'s {hit['move']} can KO slot {endangered}" + (" and pierces Protect" if pierces_protect else "") + f"; switch it out, slot {other} attacks",
                        best[other][2] + P["switch_threatened_bonus"] + (P["protect_bonus_guaranteed"] if pierces_protect else 0) * ko_probability(hit))

    # 3b. Redirection: Follow Me / Rage Powder soaks a single-target lethal hit aimed at our partner, who then
    #     gets its turn (setup or attack). Only when the redirector is sturdier than the partner against that hit.
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["ko"] == "no":
                continue
            endangered, redirector = hit["into_slot"], 1 - hit["into_slot"]
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
            if data.to_id(move_info.get("id") or hit["move"]) == "ragepowder" or (data.to_id(me_r.item) == "safetygoggles"):
                pass
            # does the redirector survive that hit?
            attacker = theirs.get(threat["position"])
            est = damage_percent(attacker, {**move_info, "id": hit["move"]}, me_r) if attacker else None
            if est is None or est[1] >= 100:
                continue  # it would just die instead
            partner_slot = slot_template(endangered)
            partner_action = None
            for setup in ("tailwind", "trickroom", "icywind", "electroweb"):
                idx = _option_index(partner_slot, "move", move_id=setup) if partner_slot else None
                if idx is not None:
                    partner_action = ({"option": idx, "target": 0}, f"sets {setup}")
                    break
            raw_partner = _best_attack(by_slot.get(endangered, {}), P=P)
            if partner_action is None and raw_partner[0]:
                partner_action = (slot_answer(raw_partner[0]["option"], raw_partner[1]["target"]), "attacks freely")
            if partner_action is None:
                continue
            answers = {redirector: {"option": redirect, "target": 0}, endangered: partner_action[0]}
            add("redirect_to_protect_partner", answers[0], answers[1],
                f"{threat['species']}'s {hit['move']} would KO slot {endangered}; slot {redirector} redirects it ({est[0]}-{est[1]}% to itself) while slot {endangered} {partner_action[1]}",
                (P["protect_bonus_guaranteed"] if hit["ko"] == "guaranteed" else P["protect_bonus_possible"]) + 25 + (50 if "sets" in partner_action[1] else raw_partner[2] * 0.8))

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
                         and data.to_id(theirs[t].item) != "covertcloak" and data.to_id(theirs[t].ability) not in ("innerfocus", "shielddust")]
        if not legal_targets:
            continue  # nothing flinchable: a Fake Out would be wasted
        # The threat whose known attacks do the most to us.
        danger = {pos: sum(_expected(h, 0, P) for t in sheet.threats if t["position"] == pos for h in t["hits"]) for pos in legal_targets}
        target = max(danger, key=danger.get)
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
        if partner is None and best[other][0]:
            partner = slot_answer(best[other][0]["option"], best[other][1]["target"])
        if partner is None:
            continue
        answers = {number: {"option": fake, "target": target}, other: partner}
        why = f"Fake Out {theirs[target].species} (its attacks threaten us most) while slot {other} " + (f"sets {setup_name}" if setup_name else "attacks")
        attacks_ko = any(b[1] and b[1].get("damage_pct_of_current_hp", [0, 0])[0] >= 100 for b in best.values())
        base = P["fakeout_base"] if not attacks_ko else 15
        # Flinching the Pokémon that threatens a KO on us this turn is worth extra: it can't act.
        threatens_us = any(h["ko"] != "no" for t in sheet.threats if t["position"] == target for h in t["hits"])
        add("fake_out_setup", answers[0], answers[1], why, base + (P["fakeout_threat_bonus"] if threatens_us else 0) + (best[other][2] * 0.6 if not setup_name else P["fakeout_setup_value"]))

    # 5. Switch a slot that has no worthwhile attack into a bench Pokémon.
    for number in (0, 1):
        slot = slot_template(number)
        if slot and (best[number][0] is None or best[number][2] < P["switch_weak_threshold"]):
            switch = best_switch(slot, theirs, memory)[0] if memory is not None else _option_index(slot, "switch")
            other = 1 - number
            if switch is not None and best[other][0]:
                answers = {number: {"option": switch, "target": 0}, other: slot_answer(best[other][0]["option"], best[other][1]["target"])}
                add("switch_weak_slot", answers[0], answers[1], f"slot {number} has no good attack; switch out while slot {other} attacks", best[other][2] + P["switch_weak_bonus"])

    candidates.sort(key=lambda c: -c["score"])
    return candidates[:4]
