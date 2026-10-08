"""Matchup model for the draft and Team Preview: how each drafted set fares against each other set.

Plain language: every card is a complete set and both rosters are public, so before a single turn is played we
can ask, for any two cards, "if these two met, who would come out ahead?" This module answers with a short
1-on-1 damage race run on the repo's verified damage calculator (``battle.damage_percent``, checked against
@smogon/calc), never with hand-computed damage. The race knows who moves first (speed, Choice Scarf, Booster
Energy, priority moves, Psychic Terrain blocking priority), how many hits each side needs (Focus Sash,
Multiscale, Air Balloon, Levitate, recoil, Life Orb, Rocky Helmet, Leftovers, Sitrus Berry, Stamina, stat
drops from Draco Meteor or Close Combat), what the two abilities do on entry (Intimidate, Download, weather and
terrain setters) and whether a set-up move like Nasty Plot pays off.

Each race is run four ways: plain, with A's side under Tailwind, with B's side under Tailwind, and under Trick
Room, because speed control is what turns many doubles matchups around.

The result for an ordered pair is a ``Duel``: ``margin`` > 0 means A wins with that fraction of its HP left,
< 0 means B wins with that fraction left. ``margin(A, B) == -margin(B, A)`` by construction.

Team level (``team_features`` / ``evaluate``): the drafter and Team Preview combine those pairwise margins with
the doubles factors a 1-on-1 race cannot see (Fake Out, redirection, spread moves, Intimidate on both foes,
speed control, screens, sleep) into one number, the log-odds that our team beats theirs. The weights come from
``data/draft_model.json`` (fitted on simulator games by ``scripts/fit_draft_model.py``); built-in defaults are
used when the file is missing.

Everything is cached per card fingerprint, so a full 18-card pool costs well under a second the first time and
nothing after that.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import data
from .battle import FieldState, Mon, build_mon, damage_percent, effective_move, paradox_boosted_stat

SPREAD_TARGETS = {"allAdjacentFoes", "allAdjacent"}
MAX_TURNS = 8  # a 1-on-1 race that is still open after 8 turns is a stall; it is scored on HP left

# -- damage effects the shared calculator does not model yet ------------------------------------------------------
# Verified against @smogon/calc on the live catalog in tests/test_matchup.py (fixture: scripts/gen_matchup_fixture.js).
ATE_ABILITIES = {"pixilate": "Fairy", "aerilate": "Flying", "refrigerate": "Ice", "galvanize": "Electric"}
ATE_POWER = 4915 / 4096  # the -ate abilities also raise the converted move's power by about 1.2x
ABSORB_ABILITIES = {"flashfire": "fire", "wellbakedbody": "fire", "waterabsorb": "water", "stormdrain": "water",
                    "dryskin": "water", "voltabsorb": "electric", "lightningrod": "electric", "motordrive": "electric",
                    "sapsipper": "grass", "eartheater": "ground", "levitate": "ground"}
MOLD_BREAKERS = {"moldbreaker", "teravolt", "turboblaze", "myceliummight"}
RAIN_SURE_HIT = {"hurricane", "thunder", "bleakwindstorm", "wildboltstorm", "sandsearstorm"}
WEIGHT_MOVES = {"lowkick", "grassknot"}  # power from the target's weight; the shared calculator reads their 0 as "no damage"
RESIST_BERRIES = {"occaberry": "fire", "passhoberry": "water", "wacanberry": "electric", "rindoberry": "grass",
                  "yacheberry": "ice", "chopleberry": "fighting", "kebiaberry": "poison", "shucaberry": "ground",
                  "cobaberry": "flying", "payapaberry": "psychic", "tangaberry": "bug", "chartiberry": "rock",
                  "kasibberry": "ghost", "habanberry": "dragon", "colburberry": "dark", "babiriberry": "steel",
                  "roseliberry": "fairy", "chilanberry": "normal"}
DAMAGING_WITHOUT_POWER = WEIGHT_MOVES | {"ruination", "beatup", "heavyslam", "heatcrash"}

# -- what moves and abilities do in a race -----------------------------------------------------------------------
RECOIL = {"flareblitz": 1 / 3, "wavecrash": 1 / 3, "woodhammer": 1 / 3, "bravebird": 1 / 3, "doubleedge": 1 / 3,
          "volttackle": 1 / 3, "headsmash": 1 / 2, "wildcharge": 1 / 4, "takedown": 1 / 4, "submission": 1 / 4}
DRAIN = {"drainpunch": 1 / 2, "gigadrain": 1 / 2, "hornleech": 1 / 2, "leechlife": 1 / 2, "bitterblade": 1 / 2,
         "paraboliccharge": 1 / 2, "drainingkiss": 3 / 4, "oblivionwing": 3 / 4}
SELF_STAGES = {"dracometeor": {"spa": -2}, "overheat": {"spa": -2}, "leafstorm": {"spa": -2}, "makeitrain": {"spa": -1},
               "closecombat": {"def": -1, "spd": -1}, "headlongrush": {"def": -1, "spd": -1}, "armorcannon": {"def": -1, "spd": -1},
               "superpower": {"atk": -1, "def": -1}, "hammerarm": {"spe": -1}, "scaleshot": {"def": -1, "spe": 1},
               "flamecharge": {"spe": 1}, "aquastep": {"spe": 1}, "trailblaze": {"spe": 1}, "rapidspin": {"spe": 1}}
TARGET_STAGES = {"icywind": {"spe": -1}, "electroweb": {"spe": -1}, "bulldoze": {"spe": -1}, "rocktomb": {"spe": -1},
                 "lowsweep": {"spe": -1}, "mudshot": {"spe": -1}, "snarl": {"spa": -1}, "spiritbreak": {"spa": -1},
                 "strugglebug": {"spa": -1}, "skittersmack": {"spa": -1}, "breakingswipe": {"atk": -1}, "lunge": {"atk": -1},
                 "firelash": {"def": -1}}
SETUP_MOVES = {"nastyplot": {"spa": 2}, "calmmind": {"spa": 1, "spd": 1}, "swordsdance": {"atk": 2},
               "dragondance": {"atk": 1, "spe": 1}, "bulkup": {"atk": 1, "def": 1}, "quiverdance": {"spa": 1, "spd": 1, "spe": 1},
               "shellsmash": {"atk": 2, "spa": 2, "spe": 2, "def": -1, "spd": -1}, "coil": {"atk": 1, "def": 1}}
FIRST_TURN_ONLY = {"fakeout", "firstimpression"}  # Fake Out is a doubles tempo tool, valued at team level instead
PINCH_BERRIES = {"aguavberry", "wikiberry", "figyberry", "magoberry", "iapapaberry"}
INTIMIDATE_IMMUNE = {"innerfocus", "owntempo", "oblivious", "scrappy", "guarddog", "clearbody", "whitesmoke",
                     "fullmetalbody", "hypercutter", "mirrorarmor"}
WEATHER_SETTERS = {"drizzle": "rain", "drought": "sun", "orichalcumpulse": "sun", "sandstream": "sand", "snowwarning": "snow"}
TERRAIN_SETTERS = {"grassysurge": "grassy", "psychicsurge": "psychic", "electricsurge": "electric", "mistysurge": "misty",
                   "hadronengine": "electric"}

# Doubles roles a 1-on-1 race cannot see. The team level weighs them (see ``team_features``).
ROLE_MOVES = {
    "fakeout": {"fakeout"},
    "tailwind": {"tailwind"},
    "trickroom": {"trickroom"},
    "speed_drop": {"icywind", "electroweb", "thunderwave", "bleakwindstorm", "scaryface", "bulldoze", "glare", "nuzzle"},
    "redirect": {"followme", "ragepowder", "spotlight"},
    "screens": {"reflect", "lightscreen", "auroraveil"},
    "sleep": {"spore", "sleeppowder", "hypnosis", "yawn", "lovelykiss", "darkvoid"},
    "burn_para": {"willowisp", "thunderwave", "nuzzle", "glare", "stunspore"},
    "disrupt": {"taunt", "encore", "partingshot", "snarl", "spiritbreak", "imprison", "quash"},
    "helpinghand": {"helpinghand"},
    "wideguard": {"wideguard"},
    "pivot": {"uturn", "voltswitch", "flipturn", "partingshot"},
}


def fingerprint(card: dict) -> str:
    """A stable id for one complete set, so caches and fitted per-card values never mix up two different cards
    of the same species (the live catalog may one day hold two Incineroar sets)."""
    evs = data.parse_evs(card.get("evs") or card.get("EVs"))
    body = json.dumps([data.to_id(card.get("species") or card.get("name")), data.to_id(card.get("item")),
                       data.to_id(card.get("ability")), data.to_id(card.get("nature")), [evs[s] for s in data.STATS],
                       sorted(data.to_id(m) for m in card.get("moves") or [])])
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


@dataclass
class Unit:
    """One card, prepared once for matchup math."""

    key: str
    card: dict
    species: str
    mon: Mon
    item: str
    ability: str
    types: list[str]
    roles: frozenset[str]
    attacks: list[dict]  # damaging moves (Fake Out excluded: it is a team-level tempo tool)
    setup: dict | None  # the set-up move it carries, if any: {"id": ..., "stages": {...}}
    weather: str | None  # weather it sets on entry
    terrain: str | None  # terrain it sets on entry
    bst: int

    @property
    def maxhp(self) -> int:
        return int(self.mon.stats.get("hp") or 0)


_UNITS: dict[str, Unit] = {}


def unit(card: dict) -> Unit:
    """The ``Unit`` for a card (cached by fingerprint)."""
    key = fingerprint(card)
    cached = _UNITS.get(key)
    if cached is not None:
        return cached
    species = str(card.get("species") or card.get("name") or "?")
    mon = build_mon({"species": species, "current_hp_fraction": 1.0}, card, side="mine", position=0)
    if mon is None:  # no species at all: an inert placeholder that never wins or loses a race
        mon = Mon(species=species, side="mine", position=0, types=[], stats={}, hp_fraction=1.0, status=None, item=None,
                  ability=None, boosts={}, moves=[])
    ids = {m["id"] for m in mon.moves}
    roles = {name for name, moves in ROLE_MOVES.items() if ids & moves}
    ability = data.to_id(card.get("ability"))
    if ability == "intimidate":
        roles.add("intimidate")
    if ability == "prankster":
        roles.add("prankster")
    attacks = [m for m in mon.moves if ((m.get("base_power") or 0) > 0 or m["id"] in DAMAGING_WITHOUT_POWER)
               and m.get("category") in ("Physical", "Special") and m["id"] not in FIRST_TURN_ONLY]
    if any((m.get("priority") or 0) > 0 for m in attacks):
        roles.add("priority")
    if any(m.get("target") in SPREAD_TARGETS for m in attacks):
        roles.add("spread")
    setup = next(({"id": m["id"], "stages": SETUP_MOVES[m["id"]]} for m in mon.moves if m["id"] in SETUP_MOVES), None)
    base = data.resolve_base_stats(card) or {}
    built = Unit(key=key, card=dict(card), species=species, mon=mon, item=data.to_id(card.get("item")), ability=ability,
                 types=list(mon.types), roles=frozenset(roles), attacks=attacks, setup=setup,
                 weather=WEATHER_SETTERS.get(ability), terrain=TERRAIN_SETTERS.get(ability), bst=sum(base.values()) if base else 0)
    _UNITS[key] = built
    return built


# -- one hit, through the verified calculator ------------------------------------------------------------------


def calc_hit(attacker: Mon, move: dict, defender: Mon, fs: FieldState | None, *, spread: bool) -> tuple[float, float] | None:
    """``battle.damage_percent`` plus the effects it does not model yet, so the matchup model sees the live
    catalog as @smogon/calc does: Pixilate-style type changes (+20% power), Adaptability (STAB 2x), immunities from
    Levitate, Air Balloon and absorbing abilities, Low Kick's weight-based power, Freeze-Dry hitting Water types
    super-effectively, type-resist berries (Chople Berry) and Unaware (stat stages of the other side ignored; the
    attacker's Defense counts for Body Press). (min%, max%) of the defender's CURRENT HP, or None for a move that
    does no damage."""
    move = dict(move)
    ability = data.to_id(attacker.ability)
    target_ability = data.to_id(defender.ability)
    if move.get("id") in WEIGHT_MOVES:
        move["base_power"] = weight_power(defender)
    if target_ability == "unaware" and any(attacker.boosts.get(k) for k in ("atk", "spa", "def")):
        attacker = replace(attacker, boosts={k: v for k, v in attacker.boosts.items() if k not in ("atk", "spa", "def")})
    if ability == "unaware" and any(defender.boosts.get(k) for k in ("def", "spd")):
        defender = replace(defender, boosts={k: v for k, v in defender.boosts.items() if k not in ("def", "spd")})
    eff = effective_move(attacker, move, defender, fs)
    move_type = data.to_id(eff.get("type"))
    if ability in ATE_ABILITIES and move_type == "normal" and (eff.get("base_power") or 0) > 0:
        move["type"] = ATE_ABILITIES[ability]
        move["base_power"] = int(round((move.get("base_power") or 0) * ATE_POWER))
        move_type = data.to_id(move["type"])
    if ability not in MOLD_BREAKERS:
        if ABSORB_ABILITIES.get(target_ability) == move_type and move.get("category") in ("Physical", "Special"):
            return (0.0, 0.0)
        if move_type == "ground" and data.to_id(defender.item) == "airballoon" and move.get("id") != "thousandarrows":
            return (0.0, 0.0)
    if move.get("id") == "freezedry" and "water" in defender.types:
        # Freeze-Dry is 2x on Water instead of 0.5x: score it against a stand-in type Ice is 2x on.
        defender = replace(defender, types=["grass" if t == "water" else t for t in defender.types])
    result = damage_percent(attacker, move, defender, field=fs, spread=spread)
    if result is None:
        return None
    low, high = result
    berry = RESIST_BERRIES.get(data.to_id(defender.item))
    if berry and berry == move_type and (berry == "normal" or data.effectiveness(move_type, defender.types) > 1):
        low, high = low / 2, high / 2  # the berry halves this hit (and is used up; the race tracks that)
    if ability == "adaptability" and move_type in [data.to_id(t) for t in attacker.types] and move.get("id") != "ruination":
        low, high = low * 2 / 1.5, high * 2 / 1.5  # Adaptability makes STAB 2x instead of 1.5x
    return (round(min(low, 999.0), 1), round(min(high, 999.0), 1))


def weight_power(defender: Mon) -> int:
    """Low Kick / Grass Knot power from the target's weight (kg)."""
    info = data.species_info(defender.species) or {}
    try:
        kg = float(info.get("weightkg") or 0)
    except (TypeError, ValueError):
        kg = 0.0
    for limit, power in ((200, 120), (100, 100), (50, 80), (25, 60), (10, 40)):
        if kg >= limit:
            return power
    return 20


def accuracy(move: dict, weather: str | None) -> float:
    acc = move.get("accuracy")
    if acc is True or acc is None:
        return 1.0
    if weather == "rain" and move.get("id") in RAIN_SURE_HIT:
        return 1.0
    return max(0.0, min(1.0, float(acc) / 100.0))


def pair_field(a: Unit, b: Unit) -> tuple[str | None, str | None]:
    """Weather and terrain when these two meet: whoever sets one wins; if both do, the slower one's ability
    activates last and its weather/terrain stays."""
    def pick(x: str | None, y: str | None) -> str | None:
        if x and y and x != y:
            return x if a.mon.stat("spe") < b.mon.stat("spe") else y
        return x or y
    return pick(a.weather, b.weather), pick(a.terrain, b.terrain)


def entry_stages(me: Unit, foe: Unit) -> dict[str, int]:
    """Stat stages ``me`` starts a race with: the foe's Intimidate, and our own Download."""
    stages: dict[str, int] = {}
    if foe.ability == "intimidate" and me.ability not in INTIMIDATE_IMMUNE and me.item != "clearamulet":
        stages["atk"] = -1
        if me.ability == "defiant":
            stages["atk"] = 1  # -1, then +2
        elif me.ability == "competitive":
            stages["spa"] = 2
        elif me.ability == "guarddog":
            stages["atk"] = 1
    if me.ability == "download" and foe.mon.stats:
        # Download: +1 Attack when the foe's Defense is the lower of its two defenses, else +1 Special Attack.
        key = "atk" if foe.mon.stats.get("def", 0) < foe.mon.stats.get("spd", 0) else "spa"
        stages[key] = stages.get(key, 0) + 1
    return stages


# -- the 1-on-1 race -----------------------------------------------------------------------------------------------


@dataclass
class Duel:
    """How card A fares 1-on-1 against card B, from A's point of view.

    ``margin`` is A's HP fraction left when B falls (positive) or minus B's HP fraction left when A falls
    (negative); a race nobody finishes in MAX_TURNS scores half the HP difference. The other margins are the
    same race with A's side under Tailwind, B's side under Tailwind, and Trick Room up.
    """

    margin: float
    margin_tw_a: float
    margin_tw_b: float
    margin_tr: float
    a_ko_turn: int | None = None  # turn A KOs B in the plain race (None: not within the horizon)
    b_ko_turn: int | None = None
    a_move: str | None = None  # the attack A leans on in the plain race
    b_move: str | None = None
    first: int = 0  # who acts first on turn 1 of the plain race: 1 = A, -1 = B, 0 = speed tie
    spread_a: float = 0.0  # A's best spread move into B, % of B's max HP (doubles 0.75x applied)
    spread_b: float = 0.0
    ohko_a: bool = False  # A KOs B from full HP on turn 1 of the plain race, before B moves
    ohko_b: bool = False

    def flipped(self) -> "Duel":
        return Duel(margin=-self.margin, margin_tw_a=-self.margin_tw_b, margin_tw_b=-self.margin_tw_a, margin_tr=-self.margin_tr,
                    a_ko_turn=self.b_ko_turn, b_ko_turn=self.a_ko_turn, a_move=self.b_move, b_move=self.a_move,
                    first=-self.first, spread_a=self.spread_b, spread_b=self.spread_a,
                    ohko_a=self.ohko_b, ohko_b=self.ohko_a)


@dataclass
class _Side:
    unit: Unit
    hp: float = 100.0  # % of max HP
    stages: dict[str, int] = field(default_factory=dict)
    item: str = ""
    status: str | None = None
    moved_setup: bool = False


@dataclass
class _RaceResult:
    margin: float
    ko_turn: tuple[int | None, int | None]
    moves: tuple[str | None, str | None]
    first: int
    ohko: tuple[bool, bool]


class _Race:
    """Runs 1-on-1 races between two units, memoizing every damage number it asks the calculator for."""

    def __init__(self, a: Unit, b: Unit) -> None:
        self.units = (a, b)
        self.weather, self.terrain = pair_field(a, b)
        self.memo: dict[tuple, tuple[float, int, float]] = {}
        self.speed_memo: dict[tuple, float] = {}

    # one hit: (expected % of the defender's max HP, priority, expected hits)
    def hit(self, att: _Side, move: dict, dfn: _Side) -> tuple[float, int, float]:
        full = dfn.hp >= 99.95
        key = (att.unit.key, move["id"], tuple(sorted(att.stages.items())), tuple(sorted(dfn.stages.items())), full,
               dfn.item, att.item, att.status, att.hp >= 99.95 if move["id"] in ("eruption", "waterspout") else None)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        # The attacker's HP only matters for Eruption / Water Spout: full, or a typical "already hit" 60%.
        a_mon = replace(att.unit.mon, boosts=dict(att.stages), status=att.status, item=att.item or None,
                        hp_fraction=1.0 if att.hp >= 99.95 else 0.6)
        d_mon = replace(dfn.unit.mon, boosts=dict(dfn.stages), status=dfn.status, item=dfn.item or None,
                        hp_fraction=1.0 if full else 0.5)
        fs = FieldState(weather=self.weather, terrain=self.terrain, actives=[a_mon, d_mon])
        a_mon.boosted_stat = paradox_boosted_stat(a_mon, fs)
        d_mon.boosted_stat = paradox_boosted_stat(d_mon, fs)
        eff = effective_move(a_mon, move, d_mon, fs)
        priority = int(eff.get("priority") or 0)
        if move["id"] == "ruination":
            immune = data.effectiveness("dark", d_mon.types) == 0
            result = (0.0 if immune else -50.0, priority, 1.0)  # negative: "half of current HP", resolved by the race
        else:
            est = calc_hit(a_mon, move, d_mon, fs, spread=move.get("target") in SPREAD_TARGETS)
            if not est:
                result = (0.0, priority, 1.0)
            else:
                mean = (est[0] + est[1]) / 2 * d_mon.hp_fraction
                result = (mean * accuracy(move, self.weather), priority, data.expected_hits(move, item=a_mon.item))
        self.memo[key] = result
        return result

    def speed(self, side: _Side, mult: float) -> float:
        key = (side.unit.key, side.stages.get("spe", 0), side.item, side.status)
        value = self.speed_memo.get(key)
        if value is None:
            mon = replace(side.unit.mon, boosts=dict(side.stages), status=side.status, item=side.item or None)
            fs = FieldState(weather=self.weather, terrain=self.terrain, actives=[mon])
            mon.boosted_stat = paradox_boosted_stat(mon, fs)
            value = mon.stat("spe")
            if side.unit.ability == "unburden" and side.unit.item and not side.item:
                value *= 2  # its item is gone
            self.speed_memo[key] = value
        return value * mult

    def best_attack(self, att: _Side, dfn: _Side, *, priority_only: bool = False) -> tuple[dict | None, float, int]:
        best: tuple[dict | None, float, int] = (None, 0.0, 0)
        for move in att.unit.attacks:
            dmg, prio, _ = self.hit(att, move, dfn)
            if dmg < 0:
                dmg = dfn.hp / 2  # Ruination: half the target's current HP
            if priority_only and prio <= 0:
                continue
            if dmg > best[1] + 1e-9 or (abs(dmg - best[1]) <= 1e-9 and best[0] is not None and prio > best[2]):
                best = (move, dmg, prio)
        return best

    def run(self, plans: tuple[str, str], speed_mult: tuple[float, float], trick_room: bool) -> _RaceResult:
        a_unit, b_unit = self.units
        sides = [_Side(a_unit, stages=entry_stages(a_unit, b_unit), item=a_unit.item),
                 _Side(b_unit, stages=entry_stages(b_unit, a_unit), item=b_unit.item)]
        ko_turn: list[int | None] = [None, None]
        used: list[dict[str, float]] = [{}, {}]
        first_turn1 = 0
        ohko = [False, False]
        for turn in range(1, MAX_TURNS + 1):
            actions: list[tuple[str, dict | None, float, int]] = []  # (kind, move, damage, priority)
            for i in (0, 1):
                me, foe = sides[i], sides[1 - i]
                if plans[i] == "setup" and not me.moved_setup and me.unit.setup:
                    prio = 1 if me.unit.ability == "prankster" else 0
                    actions.append(("setup", None, 0.0, prio))
                    continue
                move, dmg, prio = self.best_attack(me, foe)
                actions.append(("attack", move, dmg, prio) if move else ("idle", None, 0.0, 0))
            speeds = [self.speed(sides[i], speed_mult[i]) for i in (0, 1)]

            def order(acts: list) -> list[int]:
                if acts[0][3] != acts[1][3]:
                    return [0, 1] if acts[0][3] > acts[1][3] else [1, 0]
                if abs(speeds[0] - speeds[1]) < 1e-9:
                    return []  # speed tie: treat as simultaneous
                faster = 0 if speeds[0] > speeds[1] else 1
                if trick_room:
                    faster = 1 - faster
                return [faster, 1 - faster]

            # A side that would be knocked out before it moves reaches for its priority attack, if it has one.
            seq = order(actions)
            for i in (0, 1):
                if actions[i][0] != "attack":
                    continue
                goes_second = seq == [1 - i, i]
                foe_kills_me = actions[1 - i][0] == "attack" and actions[1 - i][2] >= sides[i].hp
                if goes_second:
                    move, dmg, prio = self.best_attack(sides[i], sides[1 - i], priority_only=True)
                    if move is not None and dmg > 0 and (foe_kills_me or dmg >= sides[1 - i].hp):
                        actions[i] = ("attack", move, dmg, prio)
            seq = order(actions)
            if turn == 1:
                first_turn1 = (1 if seq[0] == 0 else -1) if seq else 0
            groups = [[seq[0]], [seq[1]]] if seq else [[0, 1]]
            for step, group in enumerate(groups):
                pending = [i for i in group if sides[i].hp > 0]
                for i in pending:
                    self._act(sides[i], sides[1 - i], actions[i], used[i])
                for i in (0, 1):
                    if sides[i].hp <= 0 and ko_turn[1 - i] is None:
                        ko_turn[1 - i] = turn
                        if turn == 1 and step == 0 and group == [1 - i]:
                            ohko[1 - i] = True  # knocked out from full HP before it could act
                if any(s.hp <= 0 for s in sides):
                    break
            if any(s.hp <= 0 for s in sides):
                break
            for side in sides:
                self._end_of_turn(side, turn)
            if any(s.hp <= 0 for s in sides):
                for i in (0, 1):
                    if sides[i].hp <= 0 and ko_turn[1 - i] is None:
                        ko_turn[1 - i] = turn
                break
        a, b = sides
        if a.hp > 0 and b.hp <= 0:
            margin = max(0.05, a.hp / 100.0)
        elif b.hp > 0 and a.hp <= 0:
            margin = -max(0.05, b.hp / 100.0)
        elif a.hp <= 0 and b.hp <= 0:
            margin = 0.0
        else:
            margin = (a.hp - b.hp) / 200.0
        main = [max(u, key=u.get) if u else None for u in used]
        return _RaceResult(margin=max(-1.0, min(1.0, margin)), ko_turn=(ko_turn[0], ko_turn[1]), moves=(main[0], main[1]),
                           first=first_turn1, ohko=(ohko[0], ohko[1]))

    def _act(self, me: _Side, foe: _Side, action: tuple, used: dict[str, float]) -> None:
        kind, move, _, _ = action
        if kind == "setup":
            me.moved_setup = True
            for stat, delta in me.unit.setup["stages"].items():
                me.stages[stat] = max(-6, min(6, me.stages.get(stat, 0) + delta))
            return
        if kind != "attack" or move is None:
            return
        dmg, _, hits = self.hit(me, move, foe)  # recomputed: the state may have changed since the choice
        if dmg < 0:
            dmg = foe.hp / 2
        if dmg <= 0:
            return
        dealt = min(dmg, foe.hp)
        foe.hp -= dmg
        used[move["id"]] = used.get(move["id"], 0.0) + dealt
        if foe.item == "airballoon":
            foe.item = ""  # popped
        if foe.item == "focussash" and 0 < foe.hp < 1.5:
            foe.item = ""  # it held on at 1 HP: the sash is spent
        if foe.item in RESIST_BERRIES:
            eff_type = data.to_id(effective_move(me.unit.mon, move, foe.unit.mon, None).get("type"))
            if me.unit.ability in ATE_ABILITIES and eff_type == "normal":
                eff_type = data.to_id(ATE_ABILITIES[me.unit.ability])
            berry = RESIST_BERRIES[foe.item]
            if berry == eff_type and (berry == "normal" or data.effectiveness(eff_type, foe.unit.types) > 1):
                foe.item = ""  # eaten
        if move["id"] == "knockoff" and foe.item and foe.item != "boosterenergy":
            foe.item = ""  # (Booster Energy stays: its boost lasts after the item is used up)
        if foe.unit.ability == "stamina" and move.get("category") == "Physical" and foe.hp > 0:
            foe.stages["def"] = min(6, foe.stages.get("def", 0) + int(round(hits)))
        for stat, delta in TARGET_STAGES.get(move["id"], {}).items():
            if foe.hp > 0 and foe.unit.ability not in ("clearbody", "whitesmoke", "fullmetalbody") and foe.item != "clearamulet":
                foe.stages[stat] = max(-6, min(6, foe.stages.get(stat, 0) + delta))
        for stat, delta in SELF_STAGES.get(move["id"], {}).items():
            me.stages[stat] = max(-6, min(6, me.stages.get(stat, 0) + delta))
        # What the attacker pays or gains, converted from the foe's HP scale to its own.
        ratio = (foe.unit.maxhp / me.unit.maxhp) if me.unit.maxhp else 1.0
        if move["id"] in RECOIL:
            me.hp -= RECOIL[move["id"]] * dealt * ratio
        if move["id"] in DRAIN:
            me.hp = min(100.0, me.hp + DRAIN[move["id"]] * dealt * ratio)
        if me.item == "lifeorb":
            me.hp -= 10.0
        if "contact" in (move.get("flags") or []):
            if foe.item == "rockyhelmet":
                me.hp -= 100 / 6 * hits  # every contact hit of a multi-hit move triggers it
            if foe.unit.ability in ("roughskin", "ironbarbs"):
                me.hp -= 12.5 * hits
        self._berry(foe)
        self._berry(me)

    @staticmethod
    def _berry(side: _Side) -> None:
        if side.hp <= 0:
            return
        if side.item == "sitrusberry" and side.hp <= 50:
            side.hp, side.item = min(100.0, side.hp + 25.0), ""
        elif side.item in PINCH_BERRIES and side.hp <= 25:
            side.hp, side.item = min(100.0, side.hp + 100 / 3), ""

    def _end_of_turn(self, side: _Side, turn: int) -> None:
        if side.hp <= 0:
            return
        if side.status == "brn":
            side.hp -= 6.25
        if side.item in ("flameorb", "toxicorb") and side.status is None and turn == 1:
            if not (side.item == "flameorb" and ("fire" in side.unit.types or side.unit.ability in ("thermalexchange", "waterveil"))):
                side.status = "brn" if side.item == "flameorb" else "tox"
        if self.weather == "sand" and not set(side.unit.types) & {"rock", "ground", "steel"}:
            side.hp -= 6.25
        if side.item == "leftovers":
            side.hp = min(100.0, side.hp + 6.25)
        if self.terrain == "grassy" and side.unit.mon.grounded:
            side.hp = min(100.0, side.hp + 6.25)


def _plans(u: Unit) -> list[str]:
    return ["attack", "setup"] if u.setup else ["attack"]


def _solve(race: _Race, speed_mult: tuple[float, float], trick_room: bool) -> tuple[float, _RaceResult]:
    """Each side picks its better plan (attack straight away, or set up first). Scored as the average of A's
    maximin and minimax so that margin(A, B) == -margin(B, A) holds exactly."""
    a_plans, b_plans = _plans(race.units[0]), _plans(race.units[1])
    grid = {(pa, pb): race.run((pa, pb), speed_mult, trick_room) for pa in a_plans for pb in b_plans}
    maximin = max(min(grid[(pa, pb)].margin for pb in b_plans) for pa in a_plans)
    minimax = min(max(grid[(pa, pb)].margin for pa in a_plans) for pb in b_plans)
    best_a = max(a_plans, key=lambda pa: min(grid[(pa, pb)].margin for pb in b_plans))
    best_b = min(b_plans, key=lambda pb: max(grid[(pa, pb)].margin for pa in a_plans))
    return (maximin + minimax) / 2, grid[(best_a, best_b)]


_DUELS: dict[tuple[str, str], Duel] = {}


def duel(a: Unit, b: Unit) -> Duel:
    """How ``a`` fares against ``b`` 1-on-1 (cached; the reverse pair is the same race, flipped)."""
    cached = _DUELS.get((a.key, b.key))
    if cached is not None:
        return cached
    if not a.mon.stats or not b.mon.stats or a.key == b.key:
        result = Duel(0.0, 0.0, 0.0, 0.0)
    else:
        race = _Race(a, b)
        base, plain = _solve(race, (1.0, 1.0), False)
        tw_a, _ = _solve(race, (2.0, 1.0), False)
        tw_b, _ = _solve(race, (1.0, 2.0), False)
        tr, _ = _solve(race, (1.0, 1.0), True)
        spreads = []
        for me, foe in ((a, b), (b, a)):
            side_me = _Side(me, stages=entry_stages(me, foe), item=me.item)
            side_foe = _Side(foe, stages=entry_stages(foe, me), item=foe.item)
            best = 0.0
            for move in me.attacks:
                if move.get("target") in SPREAD_TARGETS:
                    dmg, _, _ = race.hit(side_me, move, side_foe)
                    best = max(best, dmg)
            spreads.append(best)
        result = Duel(margin=round(base, 4), margin_tw_a=round(tw_a, 4), margin_tw_b=round(tw_b, 4), margin_tr=round(tr, 4),
                      a_ko_turn=plain.ko_turn[0], b_ko_turn=plain.ko_turn[1], a_move=plain.moves[0], b_move=plain.moves[1],
                      first=plain.first, spread_a=round(spreads[0], 1), spread_b=round(spreads[1], 1),
                      ohko_a=plain.ohko[0], ohko_b=plain.ohko[1])
    _DUELS[(a.key, b.key)] = result
    _DUELS[(b.key, a.key)] = result.flipped()
    return result


def beats(a: Unit, b: Unit, threshold: float = 0.2) -> bool:
    """``a`` wins the 1-on-1 race against ``b`` with a real cushion (the 'beats X' a counter-picker reasons with)."""
    return duel(a, b).margin >= threshold


# -- team level ----------------------------------------------------------------------------------------------------
#
# A doubles game is not a string of 1-on-1 races, but the races say who can deal with whom. The team score asks the
# questions a good drafter asks: for each card they have, do we hold something that beats it (and a backup)? For each
# card we have, do they? How strong are our best four on average? Then the doubles pieces a race cannot see: spread
# moves, Tailwind / Trick Room (which flip races; the margins under each are already computed), Fake Out, redirection,
# Intimidate, screens, sleep. The result is the log-odds that our six (or four) beat theirs.

MODEL_PATH = data.DATA_DIR / "draft_model.json"

# Role counts are capped: a second Fake Out user helps, a third does not.
ROLE_CAPS = {"fakeout": 2, "redirect": 1, "intimidate": 2, "tailwind": 1, "trickroom": 1, "speed_drop": 1, "screens": 1,
             "sleep": 1, "burn_para": 1, "disrupt": 2, "helpinghand": 1, "priority": 2, "prankster": 2, "pivot": 2,
             "wideguard": 1}
MATCHUP_FEATURES = ("answers", "backup", "mean", "top4", "spread", "ohko", "tw_gain", "tr_gain")
FEATURES = MATCHUP_FEATURES + tuple(f"role_{r}" for r in ROLE_CAPS)

# Used only when data/draft_model.json is missing: hand-set, in the spirit of the fitted values.
DEFAULT_MODEL: dict = {
    "alpha_tw": 0.5, "alpha_tr": 0.5, "answer_cap": 0.5,
    "weights": {"answers": 0.35, "backup": 0.1, "mean": 2.0, "top4": 1.0, "spread": 0.5, "ohko": 0.1, "tw_gain": 1.0, "tr_gain": 1.0,
                "role_fakeout": 0.15, "role_redirect": 0.1, "role_intimidate": 0.15, "role_tailwind": 0.1,
                "role_trickroom": 0.0, "role_speed_drop": 0.05, "role_screens": 0.0, "role_sleep": 0.05,
                "role_burn_para": 0.0, "role_disrupt": 0.0, "role_helpinghand": 0.0, "role_priority": 0.1,
                "role_prankster": 0.0, "role_pivot": 0.0, "role_wideguard": 0.0},
    "cards": {},
}

_MODEL_CACHE: dict[str, dict] = {}


def load_model(path: str | Path | None = None, *, section: str = "draft") -> dict:
    """The fitted team model (``data/draft_model.json``, written by scripts/fit_draft_model.py), or the built-in
    defaults. ``section`` is ``"draft"`` (six against six) or ``"preview"`` (four against four, with leads)."""
    target = Path(path) if path else MODEL_PATH
    cache_key = f"{target}:{section}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    model = DEFAULT_MODEL
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(blob.get(section), dict):
            model = {**DEFAULT_MODEL, **blob[section]}
    except (OSError, ValueError, AttributeError):
        pass
    _MODEL_CACHE[cache_key] = model
    return model


class Board:
    """Every pairwise race among a fixed set of cards, laid out by index so a draft search can score thousands of
    team pairs per second. ``score(us, them)`` takes index lists and returns the log-odds that ``us`` wins."""

    def __init__(self, units: list[Unit], model: dict | None = None) -> None:
        self.units = list(units)
        self.model = model or load_model()
        n = len(self.units)
        self.n = n
        self.base = [[0.0] * n for _ in range(n)]
        self.tw_a = [[0.0] * n for _ in range(n)]
        self.tw_b = [[0.0] * n for _ in range(n)]
        self.tr = [[0.0] * n for _ in range(n)]
        self.spread = [[0.0] * n for _ in range(n)]  # i's best spread hit into j, % of j's HP
        self.ohko = [[False] * n for _ in range(n)]  # i knocks j out from full HP on turn 1 before j moves
        for i, a in enumerate(self.units):
            for j, b in enumerate(self.units):
                if i == j:
                    continue
                d = duel(a, b)
                self.base[i][j], self.tw_a[i][j], self.tw_b[i][j], self.tr[i][j] = d.margin, d.margin_tw_a, d.margin_tw_b, d.margin_tr
                self.spread[i][j] = d.spread_a
                self.ohko[i][j] = d.ohko_a
        self.roles = [u.roles for u in self.units]
        self.items = [u.item for u in self.units]
        self.has_tw = [("tailwind" in r) for r in self.roles]
        self.has_tr = [("trickroom" in r) for r in self.roles]
        cards = self.model.get("cards") or {}
        self.resid = [float((cards.get(u.key) or {}).get("s", 0.0)) for u in self.units]
        w = self.model.get("weights") or {}
        self.w = [float(w.get(f, 0.0)) for f in FEATURES]
        self.alpha_tw = float(self.model.get("alpha_tw", 0.5))
        self.alpha_tr = float(self.model.get("alpha_tr", 0.5))
        self.cap = float(self.model.get("answer_cap", 0.5))

    def index(self, key: str) -> int:
        for i, u in enumerate(self.units):
            if u.key == key:
                return i
        raise KeyError(key)

    def margins(self, us: list[int], them: list[int]) -> tuple[list[list[float]], float, float]:
        """Effective race margins (rows: ours, columns: theirs) once each side's speed control is counted, plus the
        average gain each kind of speed control gives us (Tailwind, Trick Room)."""
        base, tw_a, tw_b, tr = self.base, self.tw_a, self.tw_b, self.tr
        tw_us = any(self.has_tw[i] for i in us)
        tw_them = any(self.has_tw[j] for j in them)
        tr_us = any(self.has_tr[i] for i in us)
        tr_them = any(self.has_tr[j] for j in them)
        pairs = len(us) * len(them)
        tw_gain = tr_gain = 0.0
        tr_weight = 0.0
        if pairs and (tw_us or tw_them):
            if tw_us:
                tw_gain += sum(tw_a[i][j] - base[i][j] for i in us for j in them) / pairs
            if tw_them:
                tw_gain += sum(tw_b[i][j] - base[i][j] for i in us for j in them) / pairs
        if pairs and (tr_us or tr_them):
            shift = sum(tr[i][j] - base[i][j] for i in us for j in them) / pairs
            # Trick Room goes up only for the side it helps (the setter will not set it against itself).
            if (tr_us and shift > 0) or (tr_them and shift < 0):
                tr_weight = self.alpha_tr * (0.5 if tr_us and tr_them else 1.0)
                tr_gain = shift
        a_tw = self.alpha_tw
        rows = []
        for i in us:
            bi, ai, ci, ti = base[i], tw_a[i], tw_b[i], tr[i]
            row = []
            for j in them:
                m = bi[j]
                if tw_us:
                    m += a_tw * (ai[j] - bi[j])
                if tw_them:
                    m += a_tw * (ci[j] - bi[j])
                if tr_weight:
                    m += tr_weight * (ti[j] - bi[j])
                row.append(m)
            rows.append(row)
        return rows, tw_gain, tr_gain

    def features(self, us: list[int], them: list[int]) -> list[float]:
        """The team features in ``FEATURES`` order (each one is 'ours minus theirs')."""
        rows, tw_gain, tr_gain = self.margins(us, them)
        cap = self.cap
        nu, nt = len(us), len(them)
        answers = backup = mean = top4 = 0.0
        if nu and nt:
            cols = [[rows[a][b] for a in range(nu)] for b in range(nt)]
            for col in cols:  # each of their cards: our best answer, and our second-best
                ranked = sorted(col, reverse=True)
                answers += min(cap, ranked[0])
                if len(ranked) > 1:
                    backup += min(cap, ranked[1])
            for row in rows:  # each of our cards: their best answer
                ranked = sorted((-m for m in row), reverse=True)
                answers -= min(cap, ranked[0])
                if len(ranked) > 1:
                    backup -= min(cap, ranked[1])
            mean = sum(sum(r) for r in rows) / (nu * nt)
            row_means = sorted((sum(r) / nt for r in rows), reverse=True)
            col_means = sorted((-sum(c) / nu for c in cols), reverse=True)
            top4 = sum(row_means[:4]) / min(4, nu) - sum(col_means[:4]) / min(4, nt)
        spread = 0.0
        if nu and nt:
            ours = sorted((sum(self.spread[i][j] for j in them) / nt for i in us), reverse=True)
            theirs = sorted((sum(self.spread[j][i] for i in us) / nu for j in them), reverse=True)
            spread = (sum(ours[:2]) - sum(theirs[:2])) / 100.0
        ohko = 0.0
        if nu and nt:  # their cards we can remove before they act, minus ours they can
            ohko = sum(1.0 for j in them if any(self.ohko[i][j] for i in us)) - sum(1.0 for i in us if any(self.ohko[j][i] for j in them))
        out = [answers, backup, mean, top4, spread, ohko, tw_gain, tr_gain]
        for role, cap_n in ROLE_CAPS.items():
            mine = min(cap_n, sum(1 for i in us if role in self.roles[i]))
            yours = min(cap_n, sum(1 for j in them if role in self.roles[j]))
            out.append((mine - yours) / cap_n)
        return out

    def score(self, us: list[int], them: list[int]) -> float:
        """Log-odds that ``us`` beats ``them`` (antisymmetric: score(a, b) == -score(b, a))."""
        feats = self.features(us, them)
        value = sum(w * f for w, f in zip(self.w, feats))
        value += sum(self.resid[i] for i in us) - sum(self.resid[j] for j in them)
        return value


def win_probability(log_odds: float) -> float:
    if log_odds >= 0:
        return 1.0 / (1.0 + pow(2.718281828459045, -log_odds))
    z = pow(2.718281828459045, log_odds)
    return z / (1.0 + z)
