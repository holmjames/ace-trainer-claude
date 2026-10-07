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

LEVEL = 50
SPREAD_TARGETS = {"allAdjacentFoes", "allAdjacent"}
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

    def stat(self, name: str) -> float:
        base = self.stats.get(name, 0)
        value = base * stage_multiplier(self.boosts.get(name, 0))
        if name == "spe":
            if self.status == "par":
                value *= 0.5
            if data.to_id(self.item) == "choicescarf":
                value *= 1.5
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
    if data.to_id(attacker.item) == "lifeorb":
        mod *= 1.3
    current_hp = max(1.0, defender.stats["hp"] * defender.hp_fraction)
    low = base * RANDOM_MIN * mod / current_hp * 100
    high = base * RANDOM_MAX * mod / current_hp * 100
    return (round(low, 1), round(high, 1))


# -- the sheet -----------------------------------------------------------------------------------


@dataclass
class TurnSheet:
    speed_order: list[dict] = field(default_factory=list)
    trick_room: bool = False
    our_options: list[dict] = field(default_factory=list)  # per slot: option index -> damage per target
    threats: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_prompt(self) -> dict:
        return {
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


def build_sheet(template: dict, obs: dict, memory: MatchMemory) -> TurnSheet:
    sheet = TurnSheet()
    slots = sorted(template.get("slots") or [], key=lambda s: s.get("slot", 0))
    weather = obs.get("weather") if isinstance(obs.get("weather"), str) else None
    sheet.trick_room = "trickroom" in data.to_id(json.dumps(obs.get("field") or {}, default=str))
    if sheet.trick_room:
        sheet.notes.append("Trick Room is up: slower Pokémon move first.")

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
                        theirs[int(t["target"])] = mon
    if not theirs:
        for index, summary in enumerate(_active_summaries(obs.get("opponent_team"))[:2], start=1):
            mon = build_mon(summary, memory.opp_cards.get(species_key(summary.get("species"))), side="theirs", position=index)
            if mon:
                theirs[index] = mon

    # Speed order.
    everyone = list(ours.values()) + list(theirs.values())
    speeds = [(m, m.stat("spe")) for m in everyone if m.stats.get("spe")]
    speeds.sort(key=lambda pair: pair[1], reverse=not sheet.trick_room)
    sheet.speed_order = [
        {"species": m.species, "side": m.side, "position": m.position, "speed": int(s)} for m, s in speeds
    ]

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
            if per_target or move.get("base_power"):
                entry["options"].append({"option": index, "move": move["id"], "priority": move.get("priority", 0), "spread": spread, "targets": per_target})
        sheet.our_options.append(entry)

    # Opponent threats from their KNOWN sets.
    for position, opp in sorted(theirs.items()):
        threat = {"position": position, "species": opp.species, "hp_pct": round(opp.hp_fraction * 100), "status": opp.status,
                  "item": opp.item, "ability": opp.ability, "known_moves": [m["id"] for m in opp.moves], "hits": []}
        for move in opp.moves:
            if (move.get("base_power") or 0) <= 0:
                continue
            for number, me in sorted(ours.items()):
                est = damage_percent(opp, move, me, weather=weather, spread=move.get("target") in SPREAD_TARGETS and len(ours) > 1)
                if est is None:
                    continue
                threat["hits"].append({"move": move["id"], "into_slot": number, "species": me.species, "damage_pct_of_current_hp": list(est),
                                       "ko": "guaranteed" if est[0] >= 100 else ("possible" if est[1] >= 100 else "no"), "priority": move.get("priority", 0)})
        if "fakeout" in threat["known_moves"]:
            threat["note"] = "has Fake Out (only works on its first turn out)"
        sheet.threats.append(threat)

    sheet.candidates = rank_candidates(slots, ours, theirs, sheet)
    return sheet


# -- candidates --------------------------------------------------------------------------------------


def _expected(row: dict, priority: int) -> float:
    lo, hi = row["damage_pct_of_current_hp"]
    return (lo + hi) / 2 + (25 if lo >= 100 else (10 if hi >= 100 else 0)) + 3 * priority


def _best_attack(entry: dict, *, avoid_target: int | None = None) -> tuple[dict | None, dict | None, float]:
    """Highest expected damage (option, target_row, expected%) among a slot's move options.
    A spread move (rows with target 0) counts the damage to every opposing Pokémon it hits."""
    best: tuple[dict | None, dict | None, float] = (None, None, -1.0)
    for option in entry.get("options") or []:
        priority = option.get("priority") or 0
        rows = [r for r in option.get("targets") or [] if r["side"] == "theirs" and r["target"] != avoid_target]
        spread_rows = [r for r in rows if r["target"] == 0]
        if spread_rows:
            expected = sum(_expected(r, priority) for r in spread_rows)
            if expected > best[2]:
                summary = {"target": 0, "species": " + ".join(r["species"] for r in spread_rows), "side": "theirs",
                           "damage_pct_of_current_hp": [sum(r["damage_pct_of_current_hp"][0] for r in spread_rows),
                                                        sum(r["damage_pct_of_current_hp"][1] for r in spread_rows)]}
                best = (option, summary, expected)
        for row in rows:
            if row["target"] == 0:
                continue
            expected = _expected(row, priority)
            if expected > best[2]:
                best = (option, row, expected)
    return best


def _option_index(slot: dict, kind: str, **match: Any) -> int | None:
    for index, option in enumerate(slot.get("options") or []):
        if option.get("type") == kind and all(option.get(k) == v for k, v in match.items()):
            return index
    return None


def rank_candidates(slots: list[dict], ours: dict[int, "Mon"], theirs: dict[int, "Mon"], sheet: TurnSheet) -> list[dict]:
    """A few complete turns worth considering, best first. Each is {slot_0, slot_1, why}."""
    by_slot = {e["slot"]: e for e in sheet.our_options}
    candidates: list[dict] = []

    def slot_answer(number: int, option_index: int | None, target: int | None) -> dict | None:
        if option_index is None:
            return None
        return {"option": option_index, "target": target if target is not None else 0}

    def add(name: str, s0: dict | None, s1: dict | None, why: str) -> None:
        if s0 is None or s1 is None:
            return
        cand = {"name": name, "slot_0": s0, "slot_1": s1, "why": why}
        if cand not in candidates:
            candidates.append(cand)

    # Which opponent is in the most danger from our combined best hits?
    best0 = _best_attack(by_slot.get(0, {}))
    best1 = _best_attack(by_slot.get(1, {}))

    # 1. Both slots use their own best attack.
    if best0[0] and best1[0]:
        add("best_attacks",
            slot_answer(0, best0[0]["option"], best0[1]["target"]),
            slot_answer(1, best1[0]["option"], best1[1]["target"]),
            f"slot 0 {best0[0]['move']} -> {best0[1]['species']} ({best0[1]['damage_pct_of_current_hp']}%), "
            f"slot 1 {best1[0]['move']} -> {best1[1]['species']} ({best1[1]['damage_pct_of_current_hp']}%)")

    # 2. Focus fire: both into the same target when together they likely KO it.
    for target, opp in theirs.items():
        rows = []
        for number in (0, 1):
            opt, row, _ = _best_attack(by_slot.get(number, {}), avoid_target=next((t for t in theirs if t != target), None))
            if opt and row and row["target"] == target:
                rows.append((number, opt, row))
        if len(rows) == 2:
            combined = sum(r["damage_pct_of_current_hp"][0] for _, _, r in rows)
            if combined >= 100:
                add("focus_fire",
                    slot_answer(0, rows[0][1]["option"], target), slot_answer(1, rows[1][1]["option"], target),
                    f"both into {opp.species}: {combined:.0f}% minimum combined, likely KO")

    # 3. Protect a slot that can be KO'd this turn while the other attacks.
    for threat in sheet.threats:
        for hit in threat["hits"]:
            if hit["ko"] in ("guaranteed", "possible"):
                endangered = hit["into_slot"]
                other = 1 - endangered
                slot = next((s for s in slots if s.get("slot", 0) == endangered), None)
                protect = _option_index(slot, "move", move_id="protect") if slot else None
                if protect is None and slot:
                    protect = next((i for i, o in enumerate(slot.get("options") or []) if o.get("type") == "move" and data.to_id(o.get("move_id")) in ("detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap")), None)
                best_other = _best_attack(by_slot.get(other, {}))
                if protect is not None and best_other[0]:
                    answers = {endangered: {"option": protect, "target": 0}, other: slot_answer(other, best_other[0]["option"], best_other[1]["target"])}
                    add("protect_threatened", answers[0], answers[1],
                        f"{threat['species']}'s {hit['move']} can KO slot {endangered} ({hit['damage_pct_of_current_hp']}%); Protect it, slot {other} attacks")

    # 4. Switch a slot that has no worthwhile attack into a bench Pokémon.
    for number in (0, 1):
        slot = next((s for s in slots if s.get("slot", 0) == number), None)
        best = _best_attack(by_slot.get(number, {}))
        if slot and (best[0] is None or best[2] < 20):
            switch = _option_index(slot, "switch")
            other = 1 - number
            best_other = _best_attack(by_slot.get(other, {}))
            if switch is not None and best_other[0]:
                answers = {number: {"option": switch, "target": 0}, other: slot_answer(other, best_other[0]["option"], best_other[1]["target"])}
                add("switch_weak_slot", answers[0], answers[1], f"slot {number} has no good attack; switch out while slot {other} attacks")

    return candidates[:4]
