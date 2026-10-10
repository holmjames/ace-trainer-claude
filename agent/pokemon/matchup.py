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

Leads (``opening``): turns 1-2 of two leads against two, a small doubles turn loop with Fake Out, Prankster
Tailwind, redirection, sleep and spread moves, so Team Preview can see a lead pair that gets destroyed.

Team level (``Board``): the drafter and Team Preview combine the pairwise margins with the doubles factors a
1-on-1 race cannot see (Fake Out, redirection, spread moves, Intimidate, speed control, screens, sleep) into one
number, the log-odds that our team beats theirs. The weights come from ``data/draft_model.json`` (fitted on
simulator games by ``scripts/fit_draft_model.py``); built-in defaults are used when the file is missing.

Everything is cached per card fingerprint, so a full 18-card pool costs about 0.05 s the first time and nothing
after that.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import data
from .battle import RESIST_BERRIES, FieldState, Mon, build_mon, damage_percent, effective_move, paradox_boosted_stat

SPREAD_TARGETS = {"allAdjacentFoes", "allAdjacent"}
MAX_TURNS = 8  # a 1-on-1 race that is still open after 8 turns is a stall; it is scored on HP left

# -- damage effects the shared calculator does not model yet ------------------------------------------------------
# (battle.damage_percent handles the -ate abilities, Adaptability, Unaware, resist berries, Freeze-Dry and Levitate /
# Air Balloon since the Oct 8 fixes; verified against @smogon/calc on the live catalog in tests/test_matchup.py.)
ABSORB_ABILITIES = {"flashfire": "fire", "wellbakedbody": "fire", "waterabsorb": "water", "stormdrain": "water",
                    "dryskin": "water", "voltabsorb": "electric", "lightningrod": "electric", "motordrive": "electric",
                    "sapsipper": "grass", "eartheater": "ground", "levitate": "ground"}
MOLD_BREAKERS = {"moldbreaker", "teravolt", "turboblaze", "myceliummight"}
RAIN_SURE_HIT = data.RAIN_SURE_HIT
WEIGHT_MOVES = set(data.WEIGHT_MOVES)  # power from the target's weight; the shared calculator reads their 0 as "no damage"
DAMAGING_WITHOUT_POWER = set(data.DAMAGING_WITHOUT_POWER) | {"ruination", "beatup"}

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
INTIMIDATE_IMMUNE = {"innerfocus", "owntempo", "oblivious", "scrappy", "clearbody", "whitesmoke", "fullmetalbody",
                     "hypercutter", "mirrorarmor"}
WEATHER_SETTERS = {"drizzle": "rain", "drought": "sun", "orichalcumpulse": "sun", "sandstream": "sand", "snowwarning": "snow"}
TERRAIN_SETTERS = {"grassysurge": "grassy", "psychicsurge": "psychic", "electricsurge": "electric", "mistysurge": "misty",
                   "hadronengine": "electric"}

# Doubles roles a 1-on-1 race cannot see. The team level weighs them (see ``Board.features``).
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
    """``battle.damage_percent`` (which models the -ate abilities, Adaptability, Unaware, resist berries, Freeze-Dry
    and Levitate / Air Balloon) plus the two things it still lacks for the live catalog: Low Kick / Grass Knot power
    from the target's weight (the move table says 0, which reads as "no damage") and immunities from absorbing
    abilities (Flash Fire, Water Absorb, Volt Absorb...). Checked against @smogon/calc in tests/test_matchup.py.
    (min%, max%) of the defender's CURRENT HP, or None for a move that does no damage."""
    power = data.weight_power(move, defender, attacker)  # Low Kick / Grass Knot, Heavy Slam / Heat Crash
    if power is not None:
        move = {**move, "base_power": power}
    if data.to_id(attacker.ability) not in MOLD_BREAKERS and move.get("category") in ("Physical", "Special"):
        absorbed = ABSORB_ABILITIES.get(data.to_id(defender.ability))
        if absorbed and data.to_id(effective_move(attacker, move, defender, fs).get("type")) == absorbed:
            return (0.0, 0.0)
    return damage_percent(attacker, move, defender, field=fs, spread=spread)


def weight_power(defender: Mon, move: dict | None = None, attacker: Mon | None = None) -> int:
    """Low Kick / Grass Knot power from the target's weight (``data.weight_power``, the mech lane's table)."""
    return int(data.weight_power(move or {"id": "lowkick"}, defender, attacker) or 20)


def accuracy(move: dict, weather: str | None) -> float:
    """The move's chance to land (``data.hit_chance``: never-miss moves, rain / snow sure hits, sun halving)."""
    return data.hit_chance(move, weather)


def pair_field(a: Unit, b: Unit) -> tuple[str | None, str | None]:
    """Weather and terrain when these two meet: whoever sets one wins; if both do, the slower one's ability
    activates last and its weather/terrain stays."""
    def pick(x: str | None, y: str | None) -> str | None:
        if x and y and x != y:
            return x if a.mon.stat("spe") < b.mon.stat("spe") else y
        return x or y
    return pick(a.weather, b.weather), pick(a.terrain, b.terrain)


def intimidated(me: Unit) -> dict[str, int]:
    """What one Intimidate does to ``me``: -1 Attack, unless an ability or Clear Amulet stops or turns it."""
    if me.ability in INTIMIDATE_IMMUNE or me.item == "clearamulet":
        return {}
    if me.ability in ("defiant", "guarddog"):
        return {"atk": 1}  # Defiant: -1 then +2; Guard Dog: raised instead
    if me.ability == "competitive":
        return {"atk": -1, "spa": 2}
    return {"atk": -1}


def entry_stages(me: Unit, foe: Unit) -> dict[str, int]:
    """Stat stages ``me`` starts a race with: the foe's Intimidate, and our own Download."""
    stages: dict[str, int] = dict(intimidated(me)) if foe.ability == "intimidate" else {}
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
    hit_a: float = 0.0  # A's best single hit into B at full HP, max roll, % of B's max HP (immunities, Air Balloon, multi-hit)
    hit_b: float = 0.0
    hit_a_move: str | None = None
    hit_b_move: str | None = None

    def flipped(self) -> "Duel":
        return Duel(margin=-self.margin, margin_tw_a=-self.margin_tw_b, margin_tw_b=-self.margin_tw_a, margin_tr=-self.margin_tr,
                    a_ko_turn=self.b_ko_turn, b_ko_turn=self.a_ko_turn, a_move=self.b_move, b_move=self.a_move,
                    first=-self.first, spread_a=self.spread_b, spread_b=self.spread_a,
                    ohko_a=self.ohko_b, ohko_b=self.ohko_a, hit_a=self.hit_b, hit_b=self.hit_a,
                    hit_a_move=self.hit_b_move, hit_b_move=self.hit_a_move)


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

    def max_hit(self, att: _Side, dfn: _Side) -> tuple[float, str | None]:
        """A's best single attack into B at full HP: the max roll as % of B's max HP (a multi-hit move's strikes are
        already summed by the calculator), with immunities and Air Balloon through ``calc_hit``; (0.0, None) when
        nothing lands."""
        best: tuple[float, str | None] = (0.0, None)
        a_mon = replace(att.unit.mon, boosts=dict(att.stages), item=att.item or None, hp_fraction=1.0)
        d_mon = replace(dfn.unit.mon, boosts=dict(dfn.stages), item=dfn.item or None, hp_fraction=1.0)
        fs = FieldState(weather=self.weather, terrain=self.terrain, actives=[a_mon, d_mon])
        a_mon.boosted_stat = paradox_boosted_stat(a_mon, fs)
        d_mon.boosted_stat = paradox_boosted_stat(d_mon, fs)
        for move in att.unit.attacks:
            if move["id"] == "ruination":
                value = 0.0 if data.effectiveness("dark", d_mon.types) == 0 else 50.0
            else:
                est = calc_hit(a_mon, move, d_mon, fs, spread=False)
                value = est[1] if est else 0.0
            if value > best[0]:
                best = (value, move["id"])
        return best

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
        if foe.item in RESIST_BERRIES:  # a resist berry halves one super-effective hit of its type, then it is eaten
            eff_type = data.to_id(effective_move(me.unit.mon, move, foe.unit.mon, None).get("type"))
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
        spreads, hits = [], []
        for me, foe in ((a, b), (b, a)):
            side_me = _Side(me, stages=entry_stages(me, foe), item=me.item)
            side_foe = _Side(foe, stages=entry_stages(foe, me), item=foe.item)
            best = 0.0
            for move in me.attacks:
                if move.get("target") in SPREAD_TARGETS:
                    dmg, _, _ = race.hit(side_me, move, side_foe)
                    best = max(best, dmg)
            spreads.append(best)
            hits.append(race.max_hit(side_me, side_foe))
        result = Duel(margin=round(base, 4), margin_tw_a=round(tw_a, 4), margin_tw_b=round(tw_b, 4), margin_tr=round(tr, 4),
                      a_ko_turn=plain.ko_turn[0], b_ko_turn=plain.ko_turn[1], a_move=plain.moves[0], b_move=plain.moves[1],
                      first=plain.first, spread_a=round(spreads[0], 1), spread_b=round(spreads[1], 1),
                      ohko_a=plain.ohko[0], ohko_b=plain.ohko[1], hit_a=round(hits[0][0], 1), hit_b=round(hits[1][0], 1),
                      hit_a_move=hits[0][1], hit_b_move=hits[1][1])
    _DUELS[(a.key, b.key)] = result
    _DUELS[(b.key, a.key)] = result.flipped()
    return result


def reset_match_state() -> None:
    """A new match starts (agent/pokemon/match_state.py): forget the races. On an exact speed tie ``_Race.run`` resolves
    slot 0 first, so duel(a, b) and duel(b, a).flipped() differ and the cache keeps whichever direction came first;
    emptied here, a match's races follow its own order of computation, as in the fresh process the runtime gives it."""
    _DUELS.clear()


def beats(a: Unit, b: Unit, threshold: float = 0.2) -> bool:
    """``a`` wins the 1-on-1 race against ``b`` with a real cushion (the 'beats X' a counter-picker reasons with)."""
    return duel(a, b).margin >= threshold


# -- the opening: two leads against two leads, turns 1-2 ---------------------------------------------------------
#
# Team Preview's sharpest question is "what happens on turns 1 and 2 if we lead these two into theirs?" A 1-on-1 race
# cannot see Fake Out on our best attacker, a Prankster Tailwind that lets their partner move first the same turn,
# Follow Me soaking the hit meant for its partner, Spore, or one spread move landing on both of our leads. This is a
# deliberately small doubles turn loop with one plain policy for both sides (what strong players and the Oct 7 opponent do
# with these leads): Fake Out the most dangerous foe it can flinch, set Tailwind / screens / Trick Room on turn 1,
# redirect when the partner is in danger, put the biggest threat to sleep, otherwise take the attack (and target)
# that does the most, preferring knock-outs. Every damage number goes through ``calc_hit``.

OPENING_TURNS = 2
STATUS_PRIORITY_MOVES = {"tailwind", "reflect", "lightscreen", "auroraveil", "thunderwave", "willowisp", "taunt",
                         "encore", "spore", "sleeppowder", "yawn", "followme", "ragepowder", "trickroom"}


@dataclass
class Opening:
    """Turns 1-2 when our two leads meet their two, from our side: ``margin`` > 0 means we come out ahead
    (HP dealt minus HP taken, with knock-outs counted extra)."""

    margin: float
    ko_for: list[str] = field(default_factory=list)  # their leads we knock out
    ko_against: list[str] = field(default_factory=list)  # ours they knock out
    events: list[str] = field(default_factory=list)


@dataclass
class _Active:
    unit: Unit
    side: int  # 0 ours, 1 theirs
    hp: float = 100.0
    stages: dict[str, int] = field(default_factory=dict)
    item: str = ""
    status: str | None = None
    asleep: bool = False
    flinched: bool = False
    acted: bool = False

    @property
    def alive(self) -> bool:
        return self.hp > 0


class _Opening:
    def __init__(self, ours: list[Unit], theirs: list[Unit]) -> None:
        self.mons = [_Active(u, 0, item=u.item) for u in ours] + [_Active(u, 1, item=u.item) for u in theirs]
        everyone = sorted(self.mons, key=lambda m: -m.unit.mon.stat("spe"))  # abilities activate fastest first
        self.weather = self.terrain = None
        for m in everyone:  # the slowest setter's weather / terrain is the one that stays
            self.weather = m.unit.weather or self.weather
            self.terrain = m.unit.terrain or self.terrain
        for m in self.mons:
            foes = [f.unit for f in self.mons if f.side != m.side]
            for foe in foes:  # every opposing Intimidate lands (two of them stack)
                if foe.ability == "intimidate":
                    for stat, delta in intimidated(m.unit).items():
                        m.stages[stat] = max(-6, min(6, m.stages.get(stat, 0) + delta))
            if m.unit.ability == "download" and foes:  # Download reads both foes' defenses together
                defense = sum(f.mon.stats.get("def", 0) for f in foes)
                special = sum(f.mon.stats.get("spd", 0) for f in foes)
                key = "atk" if defense < special else "spa"
                m.stages[key] = m.stages.get(key, 0) + 1
        self.tailwind = [False, False]
        self.screens: list[set[str]] = [set(), set()]
        self.trick_room = False
        self.redirect: list[_Active | None] = [None, None]
        self.memo: dict[tuple, tuple[float, int, float]] = {}
        self.events: list[str] = []
        self.tie_first = 0

    def foes(self, m: _Active) -> list[_Active]:
        return [f for f in self.mons if f.side != m.side and f.alive]

    def ally(self, m: _Active) -> _Active | None:
        return next((a for a in self.mons if a.side == m.side and a is not m and a.alive), None)

    def mon_view(self, m: _Active, *, full: bool | None = None) -> Mon:
        whole = m.hp >= 99.95 if full is None else full
        view = replace(m.unit.mon, boosts=dict(m.stages), status=m.status, item=m.item or None,
                       hp_fraction=1.0 if whole else 0.5, side="mine" if m.side == 0 else "theirs")
        return view

    def hit(self, att: _Active, move: dict, dfn: _Active, spread: bool) -> tuple[float, int]:
        """(expected % of the defender's max HP, priority)."""
        ruin = tuple(sorted(m.unit.ability for m in self.mons if m.alive and m.unit.ability.endswith("ofruin")))
        key = (att.unit.key, move["id"], dfn.unit.key, tuple(sorted(att.stages.items())), tuple(sorted(dfn.stages.items())),
               dfn.hp >= 99.95, dfn.item, att.item, att.status, spread, self.weather, self.terrain,
               tuple(sorted(self.screens[dfn.side])), ruin)
        cached = self.memo.get(key)
        if cached is None:
            a_mon, d_mon = self.mon_view(att), self.mon_view(dfn)
            others = [self.mon_view(m) for m in self.mons if m.alive and m is not att and m is not dfn]
            fs = FieldState(weather=self.weather, terrain=self.terrain, actives=[a_mon, d_mon] + others,
                            my_side=set(self.screens[0]), their_side=set(self.screens[1]))
            for view in fs.actives:
                view.boosted_stat = paradox_boosted_stat(view, fs)
            eff = effective_move(a_mon, move, d_mon, fs)
            if move["id"] == "ruination":
                dmg = 0.0 if data.effectiveness("dark", d_mon.types) == 0 else -50.0
            else:
                est = calc_hit(a_mon, move, d_mon, fs, spread=spread)
                dmg = ((est[0] + est[1]) / 2 * d_mon.hp_fraction * accuracy(move, self.weather)) if est else 0.0
            cached = (dmg, int(eff.get("priority") or 0), data.expected_hits(move, item=a_mon.item))
            self.memo[key] = cached
        dmg, priority, _ = cached
        if dmg < 0:
            dmg = dfn.hp / 2  # Ruination
        return dmg, priority

    def speed(self, m: _Active) -> float:
        view = replace(m.unit.mon, boosts=dict(m.stages), status=m.status, item=m.item or None)
        view.boosted_stat = paradox_boosted_stat(view, FieldState(weather=self.weather, terrain=self.terrain, actives=[view]))
        value = view.stat("spe")
        if m.unit.ability == "unburden" and m.unit.item and not m.item:
            value *= 2
        return value * (2 if self.tailwind[m.side] else 1)

    # -- what each Pokémon does --------------------------------------------------------------------------------------

    def targets_of(self, m: _Active, move: dict) -> list[_Active]:
        if move.get("target") in SPREAD_TARGETS:
            victims = self.foes(m)
            if move.get("target") == "allAdjacent" and self.ally(m):
                victims = victims + [self.ally(m)]
            return victims
        return self.foes(m)

    def attack_value(self, m: _Active, move: dict, target: _Active | None) -> float:
        spread_move = move.get("target") in SPREAD_TARGETS
        victims = self.targets_of(m, move) if spread_move else [target]
        spread = spread_move and len(victims) > 1
        value = 0.0
        for v in victims:
            if v is None:
                continue
            dmg, _ = self.hit(m, move, v, spread)
            gain = min(dmg, v.hp) + (40.0 if dmg >= v.hp else 0.0)
            value += -gain if v.side == m.side else gain
        return value

    def threat(self, foe: _Active, to_side: int) -> float:
        """How much damage ``foe`` can put into side ``to_side`` this turn (its best attack)."""
        best = 0.0
        for move in foe.unit.attacks:
            spread_move = move.get("target") in SPREAD_TARGETS
            victims = [v for v in self.mons if v.side == to_side and v.alive]
            if spread_move:
                best = max(best, sum(min(self.hit(foe, move, v, len(victims) > 1)[0], v.hp) for v in victims))
            else:
                best = max([best] + [min(self.hit(foe, move, v, False)[0], v.hp) + (40 if self.hit(foe, move, v, False)[0] >= v.hp else 0) for v in victims])
        return best

    def flinchable(self, target: _Active) -> bool:
        u = target.unit
        if "ghost" in u.types or u.ability in ("innerfocus", "shielddust") or target.item == "covertcloak":
            return False
        if self.terrain == "psychic" and replace(u.mon, item=target.item or None).grounded:
            return False
        return not any(f.unit.ability in ("armortail", "dazzling", "queenlymajesty") for f in self.mons if f.side == target.side and f.alive)

    def decide(self, m: _Active, turn: int) -> tuple[str, dict | None, _Active | None]:
        ids = {mv["id"]: mv for mv in m.unit.mon.moves}
        foes = self.foes(m)
        ally = self.ally(m)
        if turn == 1 and "fakeout" in ids:
            options = [f for f in foes if self.flinchable(f)]
            if options:
                return ("fakeout", ids["fakeout"], max(options, key=lambda f: (self.threat(f, m.side), -self.mons.index(f))))
        if turn == 1 and "tailwind" in ids and not self.tailwind[m.side]:
            return ("tailwind", ids["tailwind"], None)
        if "trickroom" in ids and not self.trick_room:
            mine = sum(self.speed(x) for x in self.mons if x.side == m.side and x.alive)
            theirs = sum(self.speed(x) for x in foes)
            if mine < theirs:
                return ("trickroom", ids["trickroom"], None)
        if turn == 1 and ids.keys() & {"reflect", "lightscreen"} and not self.screens[m.side]:
            physical = sum(1 for f in foes for mv in f.unit.attacks if mv.get("category") == "Physical")
            special = sum(1 for f in foes for mv in f.unit.attacks if mv.get("category") == "Special")
            pick = "reflect" if ("reflect" in ids and (physical >= special or "lightscreen" not in ids)) else "lightscreen"
            return ("screen", ids[pick], None)
        redirect = next((ids[k] for k in ("followme", "ragepowder") if k in ids), None)
        if redirect and ally and turn == 1 and any(self.threat(f, m.side) >= 50 for f in foes):
            return ("redirect", redirect, None)
        sleep = next((ids[k] for k in ("spore", "sleeppowder") if k in ids), None)
        if sleep is not None:
            # A Lum / Chesto Berry holder is a target too (the sleep spends its berry), but one that falls asleep comes first.
            options = [f for f in foes if not f.asleep and f.status is None and self.sleepable(f, sleep, m, berry=False)]
            if options:
                return ("sleep", sleep, max(options, key=lambda f: (self.sleepable(f, sleep, m), self.threat(f, m.side), -self.mons.index(f))))
        if "willowisp" in ids and turn == 1:
            options = [f for f in foes if f.status is None and "fire" not in f.unit.types and not f.asleep
                       and sum(1 for mv in f.unit.attacks if mv.get("category") == "Physical") * 2 > len(f.unit.attacks)]
            if options:
                return ("burn", ids["willowisp"], max(options, key=lambda f: self.threat(f, m.side)))
        best: tuple[float, dict | None, _Active | None] = (0.0, None, None)
        for move in m.unit.attacks:
            if move.get("target") in SPREAD_TARGETS:
                value = self.attack_value(m, move, None)
                if value > best[0]:
                    best = (value, move, None)
            else:
                for f in foes:
                    value = self.attack_value(m, move, f)
                    if value > best[0]:
                        best = (value, move, f)
        if best[1] is None:
            return ("idle", None, None)
        return ("attack", best[1], best[2])

    def sleepable(self, f: _Active, move: dict | str = "spore", user: _Active | None = None, *, berry: bool = True) -> bool:
        """Can ``move`` put ``f`` to sleep now: the mech lane's ``data.status_move_blocked`` on its current item and
        status under the field's terrain (Grass / Overcoat / Safety Goggles against powders, Insomnia, Good as Gold,
        Magic Bounce, Electric / Misty Terrain when grounded; a Lum / Chesto Berry still held wakes it at once).
        ``berry=False`` asks whether the move lands at all, leaving such a berry out: it wakes its holder once, and
        ``execute`` eats it then."""
        item = f.item or None
        if not berry and item in CURE_BERRIES:
            item = None
        view = replace(f.unit.mon, item=item, status=f.status)
        return data.status_move_blocked(move, view, user=user.unit.mon if user else None, terrain=self.terrain) is None

    def priority(self, m: _Active, action: tuple) -> int:
        kind, move, target = action
        if move is None:
            return 0
        if kind in ("tailwind", "screen", "sleep", "burn", "redirect", "trickroom") and move["id"] in STATUS_PRIORITY_MOVES:
            base = int(move.get("priority") or 0)
            return base + (1 if m.unit.ability == "prankster" and kind != "trickroom" else 0)
        if kind == "attack" and target is not None:
            return self.hit(m, move, target, False)[1]
        return int(move.get("priority") or 0)

    # -- one turn ------------------------------------------------------------------------------------------------------

    def run_turn(self, turn: int) -> None:
        plans = {id(m): self.decide(m, turn) for m in self.mons if m.alive and not m.asleep}
        for m in self.mons:
            m.flinched = m.acted = False
        self.redirect = [None, None]
        pending = [m for m in self.mons if id(m) in plans]
        while pending:
            def order_key(m: _Active) -> tuple:
                speed = self.speed(m)
                if self.trick_room:
                    speed = -speed
                tie = 1 if m.side == self.tie_first else 0
                return (self.priority(m, plans[id(m)]), speed, tie)
            m = max(pending, key=order_key)
            pending.remove(m)
            if m.alive and not m.flinched:
                self.execute(m, plans[id(m)], plans)
            m.acted = True
        for m in self.mons:
            self.end_of_turn(m, turn)

    def execute(self, m: _Active, action: tuple, plans: dict) -> None:
        kind, move, target = action
        if kind == "idle" or move is None:
            return
        if kind == "tailwind":
            self.tailwind[m.side] = True
            return
        if kind == "trickroom":
            self.trick_room = not self.trick_room
            return
        if kind == "screen":
            self.screens[m.side].add(move["id"])
            return
        if kind == "redirect":
            self.redirect[m.side] = m
            return
        if target is not None and not target.alive:
            alternatives = self.foes(m)
            target = alternatives[0] if alternatives else None
        if kind in ("sleep", "burn"):
            if target is None or target.status is not None or target.unit.ability == "goodasgold":
                return
            if m.unit.ability == "prankster" and "dark" in target.unit.types:
                return
            if kind == "sleep" and self.sleepable(target, move, m, berry=False):
                if target.item in CURE_BERRIES:  # it falls asleep and its berry wakes it at once: spent, awake
                    self.events.append(f"{target.unit.species}'s {target.item} wakes it from {m.unit.species}'s {move['id']}")
                    target.item = ""
                else:
                    target.asleep, target.status = True, "slp"
                    self.events.append(f"{m.unit.species} puts {target.unit.species} to sleep")
            elif kind == "burn" and "fire" not in target.unit.types and target.unit.ability not in ("thermalexchange", "waterveil"):
                target.status = "brn"
            return
        # attacks (Fake Out included)
        spread_move = move.get("target") in SPREAD_TARGETS
        if not spread_move:
            guard = self.redirect[1 - m.side]
            if guard is not None and guard.alive and not (move["id"] == "fakeout" and False):
                powder = any(mv["id"] == "ragepowder" for mv in guard.unit.mon.moves) and not any(mv["id"] == "followme" for mv in guard.unit.mon.moves)
                immune = powder and ("grass" in m.unit.types or m.item == "safetygoggles" or m.unit.ability == "overcoat")
                if not immune and m.unit.ability not in ("stalwart", "propellertail"):
                    target = guard
            if target is None:
                return
            if move["id"] == "suckerpunch":
                plan = plans.get(id(target))
                if target.acted or plan is None or plan[0] not in ("attack", "fakeout"):
                    return  # Sucker Punch fails unless the target is about to attack
            victims = [target]
        else:
            victims = self.targets_of(m, move)
        spread = spread_move and len(victims) > 1
        for v in victims:
            dmg, _ = self.hit(m, move, v, spread)
            if dmg <= 0:
                continue
            dealt = min(dmg, v.hp)
            v.hp -= dmg
            if v.item == "airballoon":
                v.item = ""
            if v.item == "focussash" and 0 < v.hp < 1.5:
                v.item = ""
            if v.item in RESIST_BERRIES:
                eff_type = data.to_id(effective_move(m.unit.mon, move, v.unit.mon, None).get("type"))
                berry = RESIST_BERRIES[v.item]
                if berry == eff_type and (berry == "normal" or data.effectiveness(eff_type, v.unit.types) > 1):
                    v.item = ""
            if move["id"] == "fakeout" and v.alive and not v.acted and self.flinchable(v):
                v.flinched = True
            for stat, delta in TARGET_STAGES.get(move["id"], {}).items():
                if v.alive and v.unit.ability not in ("clearbody", "whitesmoke", "fullmetalbody") and v.item != "clearamulet":
                    v.stages[stat] = max(-6, min(6, v.stages.get(stat, 0) + delta))
            if v.unit.ability == "stamina" and move.get("category") == "Physical" and v.alive:
                v.stages["def"] = min(6, v.stages.get("def", 0) + 1)
            ratio = (v.unit.maxhp / m.unit.maxhp) if m.unit.maxhp else 1.0
            if move["id"] in RECOIL:
                m.hp -= RECOIL[move["id"]] * dealt * ratio
            if move["id"] in DRAIN:
                m.hp = min(100.0, m.hp + DRAIN[move["id"]] * dealt * ratio)
            if "contact" in (move.get("flags") or []) and v.item == "rockyhelmet":
                m.hp -= 100 / 6
            if "contact" in (move.get("flags") or []) and v.unit.ability in ("roughskin", "ironbarbs"):
                m.hp -= 12.5
            if v.hp <= 0:
                self.events.append(f"{m.unit.species}'s {move['id']} KOs {v.unit.species}")
            _Race._berry(v)  # type: ignore[arg-type]  (same fields: hp, item)
        for stat, delta in SELF_STAGES.get(move["id"], {}).items():
            m.stages[stat] = max(-6, min(6, m.stages.get(stat, 0) + delta))
        if m.item == "lifeorb" and any(True for _ in victims):
            m.hp -= 10.0
        _Race._berry(m)  # type: ignore[arg-type]

    def end_of_turn(self, m: _Active, turn: int) -> None:
        if not m.alive:
            return
        if m.status == "brn":
            m.hp -= 6.25
        if m.item in ("flameorb", "toxicorb") and m.status is None and turn == 1 and "fire" not in m.unit.types:
            m.status = "brn" if m.item == "flameorb" else "tox"
        if self.weather == "sand" and not set(m.unit.types) & {"rock", "ground", "steel"}:
            m.hp -= 6.25
        if m.item == "leftovers":
            m.hp = min(100.0, m.hp + 6.25)
        if self.terrain == "grassy" and replace(m.unit.mon, item=m.item or None).grounded:
            m.hp = min(100.0, m.hp + 6.25)


def opening_plan(ours: list[Unit], theirs: list[Unit]) -> list[str]:
    """What the opening model has each lead do on turn 1 (our two first, then theirs), as 'Species: move -> target'
    lines: the turn-1 line Team Preview can show the model."""
    sim = _Opening(ours, theirs)
    lines = []
    for m in sim.mons:
        kind, move, target = sim.decide(m, 1)
        if move is None:
            lines.append(f"{m.unit.species}: nothing")
            continue
        what = move["id"]
        if kind == "attack" and target is None and move.get("target") in SPREAD_TARGETS:
            lines.append(f"{m.unit.species}: {what} (spread)")
        elif target is not None:
            lines.append(f"{m.unit.species}: {what} -> {target.unit.species}")
        else:
            lines.append(f"{m.unit.species}: {what}")
    return lines


_OPENINGS: dict[tuple, Opening] = {}


def opening(ours: list[Unit], theirs: list[Unit], *, turns: int = OPENING_TURNS) -> Opening:
    """Turns 1-2 of our two leads against their two (cached). Speed ties are played both ways and averaged."""
    key = (tuple(u.key for u in ours), tuple(u.key for u in theirs), turns)
    cached = _OPENINGS.get(key)
    if cached is not None:
        return cached
    results = []
    speeds = [u.mon.stat("spe") for u in ours], [u.mon.stat("spe") for u in theirs]
    tie_possible = any(abs(a * x - b * y) < 1e-9 for a in speeds[0] for b in speeds[1] for x in (1, 2) for y in (1, 2))
    for tie_first in ((0, 1) if tie_possible else (0,)):
        sim = _Opening(ours, theirs)
        sim.tie_first = tie_first
        for turn in range(1, turns + 1):
            if not any(m.alive for m in sim.mons if m.side == 0) or not any(m.alive for m in sim.mons if m.side == 1):
                break
            sim.run_turn(turn)
        dealt = sum(100.0 - max(0.0, m.hp) for m in sim.mons if m.side == 1)
        taken = sum(100.0 - max(0.0, m.hp) for m in sim.mons if m.side == 0)
        ko_for = [m.unit.species for m in sim.mons if m.side == 1 and not m.alive]
        ko_against = [m.unit.species for m in sim.mons if m.side == 0 and not m.alive]
        margin = (dealt - taken) / 200.0 + 0.25 * (len(ko_for) - len(ko_against))
        results.append(Opening(round(margin, 4), ko_for, ko_against, sim.events))
    result = results[0] if len(results) == 1 or results[0].margin == results[1].margin else Opening(
        round((results[0].margin + results[1].margin) / 2, 4), results[0].ko_for, results[0].ko_against, results[0].events)
    _OPENINGS[key] = result
    if len(_OPENINGS) > 20000:
        _OPENINGS.clear()
    return result


def fake_out_exposed(ours: list[Unit], theirs: list[Unit]) -> list[tuple[str, str]]:
    """Our speed-control setters (Tailwind / Trick Room) that a Fake Out user among ``theirs`` can flinch on turn 1,
    as (setter, their Fake Out user) pairs. A setter is safe when the flinch cannot land (Ghost type, Inner Focus,
    Shield Dust, Covert Cloak, Psychic Terrain under a grounded setter, an Armor Tail / Dazzling / Queenly Majesty
    ally), when its partner carries Quick Guard, or when its partner's own Fake Out is faster than theirs and flinches
    their user first. Redirection does not cover it: Fake Out (+3) moves before Follow Me / Rage Powder (+2)."""
    sim = _Opening(ours, theirs)
    fakers = [m for m in sim.mons if m.side == 1 and any(mv["id"] == "fakeout" for mv in m.unit.mon.moves)]
    out: list[tuple[str, str]] = []
    for setter in (m for m in sim.mons if m.side == 0 and m.unit.roles & {"tailwind", "trickroom"}):
        if not fakers or not sim.flinchable(setter):
            continue
        partner = sim.ally(setter)
        partner_moves = {mv["id"] for mv in partner.unit.mon.moves} if partner else set()
        if "quickguard" in partner_moves:
            continue
        for faker in fakers:
            if partner and "fakeout" in partner_moves and sim.flinchable(faker) and sim.speed(partner) > sim.speed(faker):
                continue  # our Fake Out lands first
            out.append((setter.unit.species, faker.unit.species))
    return out


# -- team level ----------------------------------------------------------------------------------------------------
#
# A doubles game is not a string of 1-on-1 races, but the races say who can deal with whom. The team score asks the
# questions a good drafter asks: for each card they have, do we hold something that beats it (and a backup)? For each
# card we have, do they? How strong are our best four on average? Then the doubles pieces a race cannot see: spread
# moves, Tailwind / Trick Room (which flip races; the margins under each are already computed), Fake Out, redirection,
# Intimidate, screens, sleep. The result is the log-odds that our six (or four) beat theirs.

MODEL_PATH = data.DATA_DIR / "draft_model.json"
TIERS_PATH = data.DATA_DIR / "card_tiers.json"
HOLE_THRESHOLD = 40.0  # % of max HP: below this from every card of ours, an opposing card is a coverage hole

# Role counts are capped: a second Fake Out user helps, a third does not.
ROLE_CAPS = {"fakeout": 2, "redirect": 1, "intimidate": 2, "tailwind": 1, "trickroom": 1, "speed_drop": 1, "screens": 1,
             "sleep": 1, "burn_para": 1, "disrupt": 2, "helpinghand": 1, "priority": 2, "prankster": 2, "pivot": 2,
             "wideguard": 1}
MATCHUP_FEATURES = ("answers", "backup", "mean", "top4", "spread", "ohko", "tw_gain", "tr_gain")
FEATURES = MATCHUP_FEATURES + tuple(f"role_{r}" for r in ROLE_CAPS)

# Used when data/draft_model.json is missing, and as the VGC prior for every doubles-role weight the fit got wrong
# (see ``load_model``): hand-set from how the roles are valued in real doubles play. (Oct 8: halving tw_gain / tr_gain
# and crediting Trick Room 0.1 moved the bcbaabe2 pick-3 state off Archaludon, the card that won that game; kept.)
DEFAULT_MODEL: dict = {
    "alpha_tw": 0.5, "alpha_tr": 0.5, "answer_cap": 0.5,
    "weights": {"answers": 0.35, "backup": 0.1, "mean": 2.0, "top4": 1.0, "spread": 0.5, "ohko": 0.1, "tw_gain": 1.0, "tr_gain": 1.0,
                "role_fakeout": 0.15, "role_redirect": 0.1, "role_intimidate": 0.15, "role_tailwind": 0.1,
                "role_trickroom": 0.0, "role_speed_drop": 0.05, "role_screens": 0.0, "role_sleep": 0.05,
                "role_burn_para": 0.0, "role_disrupt": 0.0, "role_helpinghand": 0.0, "role_priority": 0.1,
                "role_prankster": 0.0, "role_pivot": 0.0, "role_wideguard": 0.0},
    "cards": {},
}
# Weights a fit may never turn negative: a doubles role (Fake Out, Intimidate, speed control, redirection...) or the
# gain from our own speed control. The fitted file carries negative values for several of them because it was fitted
# on games piloted by an older battle brain that misplayed those roles (Fake Out -0.32, Tailwind -0.08, Trick Room
# -0.16 at Team Preview); until a refit from games piloted by a brain that uses them, a negative fitted value is
# replaced by the prior above.
PRIOR_CLAMPED = ("tw_gain", "tr_gain") + tuple(f"role_{r}" for r in ROLE_CAPS)

_MODEL_CACHE: dict[str, dict] = {}


def load_model(path: str | Path | None = None, *, section: str = "draft", clamp: bool = True) -> dict:
    """The fitted team model (``data/draft_model.json``, written by scripts/fit_draft_model.py), or the built-in
    defaults. ``section`` is ``"draft"`` (six against six) or ``"preview"`` (four against four, with leads).
    ``clamp`` replaces negative role / speed-control weights with the VGC priors (``clamp_roles``); Team Preview
    passes ``clamp=False`` when it predicts the opponent's bring (Oct 8, eight live sessions with their fitted
    per-card terms zeroed: the weights as fitted predicted their exact four 3 times, bcbaabe2 at 100%; the clamped
    weights never did, crediting a Fake Out user this opponent benched)."""
    target = Path(path) if path else MODEL_PATH
    cache_key = f"{target}:{section}:{'clamped' if clamp else 'fitted'}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    model = DEFAULT_MODEL
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(blob.get(section), dict):
            model = {**DEFAULT_MODEL, **blob[section]}
    except (OSError, ValueError, AttributeError):
        pass
    if clamp:
        model = clamp_roles(model)
    _MODEL_CACHE[cache_key] = model
    return model


def clamp_roles(model: dict) -> dict:
    """The model with every role / speed-control weight at >= 0: a negative fitted value becomes the VGC prior from
    ``DEFAULT_MODEL`` (never below 0). The replaced keys are listed under ``model["clamped"]``. Other weights
    (answers, mean, spread...) are left exactly as fitted."""
    weights = dict(model.get("weights") or {})
    prior = DEFAULT_MODEL["weights"]
    clamped = {}
    for key in PRIOR_CLAMPED:
        value = float(weights.get(key, prior.get(key, 0.0)))
        if value < 0:
            clamped[key] = value
            weights[key] = max(0.0, float(prior.get(key, 0.0)))
    if not clamped:
        return model
    return {**model, "weights": weights, "clamped": clamped}


_TIERS_CACHE: dict[str, dict] = {}


def load_tiers(path: str | Path | None = None) -> dict[str, dict]:
    """The card tier list (``data/card_tiers.json``, built by scripts/build_tiers.py), keyed by card fingerprint so a
    different set under a known card id is never mistaken for the rated one. Empty when the file is missing."""
    target = str(Path(path) if path else TIERS_PATH)
    if target not in _TIERS_CACHE:
        try:
            blob = json.loads(Path(target).read_text(encoding="utf-8"))
            _TIERS_CACHE[target] = {e["fingerprint"]: e for e in (blob.get("cards") or {}).values() if e.get("fingerprint")}
        except (OSError, ValueError, AttributeError, KeyError):
            _TIERS_CACHE[target] = {}
    return _TIERS_CACHE[target]


UNSEEN_PERCENTILE = 0.25


def unseen_strength(cards: dict) -> float:
    """The per-card strength a card outside the fitted table gets: the 25th percentile of the fitted values
    (-0.079 on the Oct 8 fit), so an unknown card is assumed below average rather than average."""
    values = sorted(float((entry or {}).get("s", 0.0)) for entry in cards.values())
    if not values:
        return 0.0
    pos = UNSEEN_PERCENTILE * (len(values) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (pos - lo) * (values[hi] - values[lo])


# -- sleep pressure: a sleep-move user against the other side's sleep-proof cards (Oct 8, 51c86e5c) -------------------
#
# A sleep-move user (Spore, Sleep Powder, Hypnosis, Yawn...) takes one Pokémon out of the game per turn it acts; the
# cards it cannot touch (Grass / Overcoat / Safety Goggles against powders, Insomnia / Vital Spirit / Sweet Veil, Good
# as Gold, Magic Bounce, Electric / Misty Terrain) keep acting. The 1-on-1 races never see sleep and the fitted role
# weight for holding a sleeper comes from games an engine played that misuses Spore (Amoonguss's tier flag), so on
# Oct 8 Team Preview gave every Amoonguss four of theirs 0% and Amoonguss slept four of ours in seven turns. Who is
# sleep-proof comes from the mech lane's ``data.status_move_blocked``; ``Board.sleep_reach`` / ``sleep_pressure`` turn it
# into a score, which Team Preview (``lineup_sleep_w``) and the finished-draft value (``draft_sleep_w``) add at a weight
# that is 0 until a simulator series measures it (Oct 9 review: unmeasured at 0.4, it moved live picks and previews).

SLEEP_MOVES = frozenset(m for m, cond in data.STATUS_CONDITION_MOVES.items() if cond == "slp")
YAWN_SHARE = 0.5  # Yawn puts the target to sleep only at the end of the next turn, and a switch escapes it
TERRAIN_SHARE = 0.5  # an Electric / Misty Terrain keeps a grounded card awake only while it is up (5 turns)
SLEEP_TERRAINS = frozenset({"electric", "misty"})
SLEEP_SLOTS = 2  # doubles: two of a side's Pokémon are on the field, so two sleep-proof cards keep both slots awake
CURE_BERRIES = frozenset({"lumberry", "chestoberry"})  # wake the holder once: not sleep-proof


def sleep_moves(u: Unit) -> list[dict]:
    """``u``'s moves that put their target to sleep."""
    return [m for m in u.mon.moves if m["id"] in SLEEP_MOVES]


def sleep_threat(u: Unit) -> float:
    """How surely ``u`` puts a target to sleep: its best sleep move's accuracy (``data.hit_chance``; Spore 1.0, Sleep
    Powder 0.75, Hypnosis 0.6), Yawn at YAWN_SHARE. 0 for a card with no sleep move."""
    return max((data.hit_chance(m) * (YAWN_SHARE if m["id"] == "yawn" else 1.0) for m in sleep_moves(u)), default=0.0)


def sleep_proof(target: Unit, sleeper: Unit, terrain: str | None = None) -> bool:
    """``target`` cannot be put to sleep by any of ``sleeper``'s sleep moves (``data.status_move_blocked``, with
    ``terrain`` in play). A Lum / Chesto Berry wakes its holder only once and is then eaten, so it never counts."""
    moves = sleep_moves(sleeper)
    if not moves:
        return False
    mon = target.mon
    if data.to_id(mon.item) in CURE_BERRIES:
        mon = replace(mon, item=None)
    return all(data.status_move_blocked(m, mon, user=sleeper.mon, terrain=terrain) for m in moves)


class Board:
    """Every pairwise race among a fixed set of cards, laid out by index so a draft search can score thousands of
    team pairs per second. ``score(us, them)`` takes index lists and returns the log-odds that ``us`` wins."""

    def __init__(self, units: list[Unit], model: dict | None = None, *, card_weight: float = 1.0,
                 tiers: dict[str, dict] | None = None, tier_adjust: float = 1.0, four_model: dict | None = None) -> None:
        self.units = list(units)
        self.model = model or load_model()
        self.four_model = four_model  # weights for four-against-four (``score4``); the preview section when None
        tiers = load_tiers() if tiers is None else tiers
        n = len(self.units)
        self.n = n
        self.base = [[0.0] * n for _ in range(n)]
        self.tw_a = [[0.0] * n for _ in range(n)]
        self.tw_b = [[0.0] * n for _ in range(n)]
        self.tr = [[0.0] * n for _ in range(n)]
        self.spread = [[0.0] * n for _ in range(n)]  # i's best spread hit into j, % of j's HP
        self.ohko = [[False] * n for _ in range(n)]  # i knocks j out from full HP on turn 1 before j moves
        self.hit = [[0.0] * n for _ in range(n)]  # i's best single hit into j (max roll, % of j's max HP; 0 = cannot damage)
        self.hit_move = [[None] * n for _ in range(n)]
        for i, a in enumerate(self.units):
            for j, b in enumerate(self.units):
                if i == j:
                    continue
                d = duel(a, b)
                self.base[i][j], self.tw_a[i][j], self.tw_b[i][j], self.tr[i][j] = d.margin, d.margin_tw_a, d.margin_tw_b, d.margin_tr
                self.spread[i][j] = d.spread_a
                self.ohko[i][j] = d.ohko_a
                self.hit[i][j], self.hit_move[i][j] = d.hit_a, d.hit_a_move
        self.roles = [u.roles for u in self.units]
        self.items = [u.item for u in self.units]
        self.has_tw = [("tailwind" in r) for r in self.roles]
        self.has_tr = [("trickroom" in r) for r in self.roles]
        cards = self.model.get("cards") or {}
        # Each card's base value is its tier rating (data/card_tiers.json): the fit's per-card term ("wins more than
        # its matchups explain") plus the tier list's written-down adjustment, while the matchup features below add
        # the reactions to this particular opponent. card_weight scales the fitted per-card part only (0 = matchups,
        # roles and the written-down adjustments); tier_adjust scales the adjustments only (0 = the fit as it is).
        # A card the fit never saw gets the 25th-percentile strength, never 0: at 0 (the median) an unseen Iron
        # Valiant was the first pick of da9c168e by 0.7 points of win chance, stayed home at Team Preview, and the game
        # was lost (Oct 8).
        unseen = unseen_strength(cards)
        self.resid = []
        for u in self.units:
            term = card_weight * float((cards.get(u.key) or {}).get("s", unseen))
            tier = tiers.get(u.key)
            if tier:
                term += tier_adjust * (float(tier.get("rating", 0.0)) - float(tier.get("sim_rating", 0.0))) / 100.0
            self.resid.append(term)
        w = self.model.get("weights") or {}
        self.w = [float(w.get(f, 0.0)) for f in FEATURES]
        self.alpha_tw = float(self.model.get("alpha_tw", 0.5))
        self.alpha_tr = float(self.model.get("alpha_tr", 0.5))
        self.cap = float(self.model.get("answer_cap", 0.5))
        # Sleep (``sleep_reach`` / ``sleep_pressure``, outside the fitted features; Team Preview and the finished-draft
        # value add it at their own weight, like ``holes``): per sleep-move user on the board, which cards it cannot put
        # to sleep, alone and under an Electric / Misty Terrain.
        self.sleep_r = [sleep_threat(u) for u in self.units]
        self.sleepers = [j for j in range(n) if self.sleep_r[j] > 0]
        self.sleep_setter = [u.terrain if u.terrain in SLEEP_TERRAINS else None for u in self.units]
        self.proof: dict[int, list[bool]] = {}
        self.terrain_proof: dict[int, list[frozenset[str]]] = {}
        for j in self.sleepers:
            sleeper = self.units[j]
            self.proof[j] = [i != j and sleep_proof(u, sleeper) for i, u in enumerate(self.units)]
            self.terrain_proof[j] = [frozenset(t for t in SLEEP_TERRAINS if i != j and sleep_proof(u, sleeper, t))
                                     for i, u in enumerate(self.units)]

    def index(self, key: str) -> int:
        for i, u in enumerate(self.units):
            if u.key == key:
                return i
        raise KeyError(key)

    def holes(self, us: list[int], them: list[int], threshold: float = HOLE_THRESHOLD) -> list[int]:
        """Their cards none of ``us`` hits for at least ``threshold`` % of max HP with its best single attack
        (immunities and Air Balloon included): the coverage holes a Protecting, switching opponent never lets a
        two-turn plan around (Oct 7 L1: six cards with <= 18% into Air Balloon Gholdengo but one)."""
        return [j for j in them if not any(self.hit[i][j] >= threshold for i in us)]

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

    def cover(self, side: list[int], j: int, field: list[int] = ()) -> float:
        """How much of ``side``'s two active slots sleeper ``j`` cannot put to sleep, 0..1: a slot per sleep-proof card
        (``sleep_proof``), TERRAIN_SHARE of one per grounded card an Electric / Misty Terrain keeps awake (set by any
        card of ``side`` or ``field``: terrain covers both sides), at most both slots."""
        terrains = {self.sleep_setter[k] for k in (*side, *field) if self.sleep_setter[k]}
        proof, under = self.proof[j], self.terrain_proof[j]
        slots = 0.0
        for i in side:
            if proof[i]:
                slots += 1.0
            elif terrains and under[i] & terrains:
                slots += TERRAIN_SHARE
        return min(slots, SLEEP_SLOTS) / SLEEP_SLOTS

    def sleep_reach(self, sleepers: list[int], targets: list[int]) -> float:
        """How much of ``targets``' two active slots the sleep-move users among ``sleepers`` can put to sleep: each
        one's reliability (``sleep_r``) x (1 - ``cover``), summed. A Spore user against no sleep-proof card is 1,
        against one 0.5, against two 0, so a sleep-proof card is worth up to half a Spore to the side holding it. The
        fitted features never see sleep (the races have no status moves; the role weight for a sleeper is flat). The
        finished-draft value charges their reach into our four (one-sided, ``draft_sleep_w``)."""
        if not self.sleepers:
            return 0.0
        return sum(self.sleep_r[i] * (1.0 - self.cover(targets, i, sleepers)) for i in sleepers if self.sleep_r[i])

    def sleep_pressure(self, us: list[int], them: list[int]) -> float:
        """Sleep, ours minus theirs: ``sleep_reach(us, them) - sleep_reach(them, us)``. Antisymmetric, as Team
        Preview's zero-sum pairing needs (``lineup_sleep_w``); 0 when neither side holds a sleeper."""
        return self.sleep_reach(us, them) - self.sleep_reach(them, us)

    def score(self, us: list[int], them: list[int]) -> float:
        """Log-odds that ``us`` beats ``them`` (antisymmetric: score(a, b) == -score(b, a))."""
        feats = self.features(us, them)
        value = sum(w * f for w, f in zip(self.w, feats))
        value += sum(self.resid[i] for i in us) - sum(self.resid[j] for j in them)
        return value

    def score4(self, us: list[int], them: list[int], *, resid_them: bool = True) -> float:
        """``score`` with the Team Preview weights (the ``preview`` section: four against four), so the drafter can
        value a finished draft by the four it would bring the way Team Preview values it. ``resid_them=False``
        models their side by roles and matchups only (no fitted per-card term), as Team Preview does."""
        w4 = getattr(self, "_w4", None)
        if w4 is None:
            weights = (self.four_model or load_model(section="preview")).get("weights") or {}
            w4 = self._w4 = [float(weights.get(f, 0.0)) for f in FEATURES]
        feats = self.features(us, them)
        value = sum(w * f for w, f in zip(w4, feats))
        value += sum(self.resid[i] for i in us)
        if resid_them:
            value -= sum(self.resid[j] for j in them)
        return value


def win_probability(log_odds: float) -> float:
    if log_odds >= 0:
        return 1.0 / (1.0 + pow(2.718281828459045, -log_odds))
    z = pow(2.718281828459045, log_odds)
    return z / (1.0 + z)
