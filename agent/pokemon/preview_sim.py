"""The deep Team Preview: a fast, seeded simulation of the first turns of a game, for every plan we could bring.

Plain language: Team Preview has 85 seconds and used to spend a third of one. A *plan* is the four we bring and the
two we lead. For a plan and one four the opponent is likely to bring (opening with the leads it is likely to pick),
this module plays the first three or four turns of the game, many times over, with dice: damage rolls, accuracy,
speed ties, secondary flinches, how long a Pokémon sleeps, and the opponent's habits as measured in the live games
(``SEARCH_BOT``: Protect about half the time when we threaten a KO and never twice in a row, out of a KO about a third
of the time once its Protect is spent, Fake Out on its first turn about half the time, Tailwind when nobody threatens
the setter...). Our side plays a cheap version of our own turn scorer: Fake Out their biggest threat, speed control
on the first turn, Protect a slot facing a likely KO, switch out of a sure one into the bench Pokémon that takes it
best, otherwise the attack and target that do the most (a knock-out counts extra, a hit the partner already lands
first does not). Both benches play: a fainted slot is refilled, a switch-in brings its Intimidate, weather and
terrain. The end of the horizon is scored as material (``Sim.margin``): each Pokémon of both fours, alive and by HP,
plus speed control still running and a Fake Out still fresh, ours minus theirs, in Pokémon.

``Search`` is the anytime part: successive halving over the plans under a deadline. Every plan gets a rollout
against every likely four of theirs, the better half twice as many, and so on until one plan is left or the time
(``lineup_sim_budget_s``) runs out; the best plan so far can be read at any moment. A plan's value is
``expected_w`` x its expected margin over their fours + (1 - ``expected_w``) x its worst four. Rollout k against
their four g is seeded from (match seed, g, k) for every plan alike, so plans are compared on the same dice.

Every damage number goes through the verified calculator (``matchup.calc_hit`` -> ``battle.damage_percent``,
checked against @smogon/calc); nothing here is a hand-computed damage figure.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from . import data, matchup
from .battle import FLINCH_CHANCE, RESIST_BERRIES, FieldState, effective_move, gale_wings, paradox_boosted_stat

SPREAD = matchup.SPREAD_TARGETS
PROTECT_MOVES = ("protect", "detect", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "kingsshield", "obstruct")
SLEEP_MOVES = ("spore", "sleeppowder", "hypnosis", "lovelykiss", "darkvoid")  # Yawn (asleep a turn later, a switch escapes it) is left out
SCREENS = ("reflect", "lightscreen", "auroraveil")
REDIRECTS = ("followme", "ragepowder")
PRIORITY_GUARDS = ("armortail", "dazzling", "queenlymajesty")
FLINCH_PROOF = ("innerfocus", "shielddust")
DROP_PROOF = ("clearbody", "whitesmoke", "fullmetalbody")
SAND_PROOF = ("overcoat", "sandveil", "sandrush", "sandforce")
KO_BONUS = 40.0  # a knock-out's worth over the damage it deals (% of max HP), as in matchup's opening model
TAILWIND_TURNS, TRICK_ROOM_TURNS, SCREEN_TURNS, LIGHT_CLAY_TURNS = 4, 5, 5, 8  # Tailwind and Trick Room count the turn they go up

# The search bot's habits, from its live decisions (ace-trainer-data/opponent-tallies-2026-10-08.json, pooled Oct 7-8;
# the event names are the tally's): the chance it makes each kind of play in the situation named.
SEARCH_BOT: dict[str, float] = {
    "fakeout": 0.5,             # fakeout_first_turn: a Fake Out user on its first turn out Fake Outs half the time
    "fakeout_threat": 0.5,      # ... and aims it at our biggest threat less than half the time (the other slot otherwise)
    "protect_lead_ko": 0.67,    # lead_turn1_protect: a lead we can KO (>= 50%) Protects on turn 1 two times in three
    "protect_lead": 0.12,       # ... a lead facing no KO one time in eight
    "protect_ko": 0.47,         # fresh_protect_ko50: facing our >= 50% KO, did not Protect last turn
    "protect_ko_aimed": 0.62,   # ... and one of our single-target moves aimed at it last turn
    "protect_hit": 0.32,        # fresh_protect_30_noko: no KO, but our best hit is >= 30% of its HP
    "protect_hit_aimed": 0.5,
    "protect_switchin": 0.43,   # switchin_first_turn_protect: a fresh switch-in we can hit hard
    "protect_idle": 0.05,       # nothing threatens it
    "switch_spent_ko": 0.38,    # spent_protect_ko_switch: facing a KO with its Protect spent (or none), a bench to go to
    "switch_ko": 0.15,          # facing a KO with Protect available, not Protecting
    "switch_double": 0.1,       # double_only_ko_stays: only our two attacks together KO it (it stays 0.88)
    "switch_idle": 0.03,        # voluntary_switch_rate when we threaten no KO
    "tailwind_free": 0.6,       # tailwind_when_available: 2 of 2 when its setter was unthreatened...
    "tailwind": 0.2,            # ... about one available turn in five otherwise
    "trickroom": 0.5,           # (no tally: a coin flip when Trick Room favours its side)
    "screens": 0.4,             # (no tally)
    "setup_free": 0.36,         # setup_unthreatened
    "sleep": 0.6,               # the battle sheet's sleep_use_prior: a sleep user with a target uses it
    "redirect": 0.5,            # (no tally) Follow Me / Rage Powder when its partner faces a KO
    "burn": 0.5,                # (no tally) Will-O-Wisp at a physical attacker
}

# Our side's cheap scorer: thresholds on the chance a slot is knocked out this turn.
OURS: dict[str, float] = {
    "protect": 0.5,       # Protect a slot this likely to fall (unless it removes the threat first)
    "switch": 0.9,        # switch out of a sure KO when Protect is not available
    "focus": 0.6,         # both_attack_same_target: the chance their two attacks go into one slot
    "protect_risk_ko": 0.4,   # what we expect a foe with a fresh Protect to do when we threaten its KO...
    "protect_risk": 0.15,     # ... and when we do not
}

# The end-of-horizon value, in Pokémon: alive is worth VALUE["alive"], plus VALUE["hp"] x its HP fraction.
VALUE: dict[str, float] = {"alive": 0.5, "hp": 0.5, "asleep": 0.8, "burned": 0.8, "paralysed": 0.9, "stage": 0.05,
                           "tailwind_turn": 0.05, "trick_room_turn": 0.05, "fresh_fake_out": 0.05}

IDLE = ("idle", None, None, None)


def ko_chance(lo: float, hi: float, hp: float) -> float:
    """Chance a damage roll uniform on [lo, hi] (% of max HP) reaches ``hp``."""
    if hp <= 0 or lo >= hp:
        return 1.0
    if hi < hp or hi <= lo:
        return 0.0
    return (hi - hp) / (hi - lo)


class _Info:
    """What a card can do, read once per card."""

    __slots__ = ("attacks", "protect", "fakeout", "tailwind", "trickroom", "screens", "redirect", "sleep", "burn", "setup",
                 "physical", "choice", "types", "grounded")

    def __init__(self, unit: matchup.Unit) -> None:
        ids = {m["id"]: m for m in unit.mon.moves}
        self.attacks = list(unit.attacks)
        self.protect = next((ids[k] for k in PROTECT_MOVES if k in ids), None)
        self.fakeout = ids.get("fakeout")
        self.tailwind = ids.get("tailwind")
        self.trickroom = ids.get("trickroom")
        self.screens = [ids[k] for k in SCREENS if k in ids]
        self.redirect = next((ids[k] for k in REDIRECTS if k in ids), None)
        self.sleep = next((ids[k] for k in SLEEP_MOVES if k in ids), None)
        self.burn = ids.get("willowisp")
        setup = next((k for k in matchup.SETUP_MOVES if k in ids), None)
        self.setup = (ids[setup], matchup.SETUP_MOVES[setup]) if setup else None
        self.physical = sum(1 for m in self.attacks if m.get("category") == "Physical") * 2 > len(self.attacks)
        self.choice = unit.item in data.CHOICE_ITEMS
        self.types = set(unit.types)
        self.grounded = "flying" not in self.types and unit.ability != "levitate"


class Calc:
    """Damage ranges, speeds and accuracies for the rollouts, memoized for a whole search and across searches (keys
    are card fingerprints and field state, so the next preview with the same cards starts warm)."""

    LIMIT = 400_000

    def __init__(self) -> None:
        self.hits: dict[tuple, tuple[float, float, int]] = {}
        self.speeds: dict[tuple, float] = {}
        self.accuracy: dict[tuple, float] = {}
        self.infos: dict[str, _Info] = {}
        self.misses = 0

    def info(self, unit: matchup.Unit) -> _Info:
        got = self.infos.get(unit.key)
        if got is None:
            got = self.infos[unit.key] = _Info(unit)
        return got

    def trim(self) -> None:
        if len(self.hits) > self.LIMIT:
            self.hits.clear()
        if len(self.speeds) > self.LIMIT:
            self.speeds.clear()


_CALC = Calc()


class _Mon:
    """One Pokémon of a rollout. ``hp`` is % of max HP; ``slot`` is 0/1 on the field, None on the bench."""

    __slots__ = ("unit", "info", "side", "hp", "stages", "skey", "item", "status", "sleep_left", "protect_last",
                 "used_protect", "protecting", "flinched", "acted", "actions", "locked", "aimed", "aimed_now", "slot", "tie")

    def __init__(self, unit: matchup.Unit, side: int, info: _Info) -> None:
        self.unit, self.info, self.side = unit, info, side
        self.hp = 100.0
        self.stages: dict[str, int] = {}
        self.skey: tuple = ()
        self.item = unit.item or ""
        self.status: str | None = None
        self.sleep_left = 0
        self.protect_last = self.used_protect = self.protecting = False
        self.flinched = self.acted = False
        self.actions = 0  # actions since it came in: 0 = Fake Out still works
        self.locked: str | None = None  # the move a Choice item holds it to
        self.aimed = self.aimed_now = False  # a single-target move of the other side aimed at it last turn / this turn
        self.slot: int | None = None
        self.tie = 0.0

    @property
    def name(self) -> str:
        return self.unit.species

    def boost(self, delta: dict[str, int]) -> None:
        for stat, d in delta.items():
            self.stages[stat] = max(-6, min(6, self.stages.get(stat, 0) + d))
        self.skey = tuple(sorted((k, v) for k, v in self.stages.items() if v))


class Sim:
    """One rollout: both fours, two of each on the field, the field, the turn. ``prior`` drives their side
    (``SEARCH_BOT``); ``ours`` is our cheap scorer's thresholds (``OURS``). ``log`` collects a line per event when
    it is a list (tests and post-mortems); None keeps rollouts fast."""

    def __init__(self, ours: list[matchup.Unit], our_leads: tuple[int, int], theirs: list[matchup.Unit],
                 their_leads: tuple[int, int], *, rng: random.Random, calc: Calc | None = None,
                 prior: dict | None = None, policy: dict | None = None, log: list | None = None) -> None:
        self.rng = rng
        self.calc = calc or _CALC
        self.prior = prior or SEARCH_BOT
        self.policy = policy or OURS
        self.log = log
        self.team = [[_Mon(u, 0, self.calc.info(u)) for u in ours], [_Mon(u, 1, self.calc.info(u)) for u in theirs]]
        self.active: list[list[_Mon | None]] = [[self.team[0][our_leads[0]], self.team[0][our_leads[1]]],
                                                [self.team[1][their_leads[0]], self.team[1][their_leads[1]]]]
        for side in (0, 1):
            for slot, m in enumerate(self.active[side]):
                m.slot = slot
        self.weather: str | None = None
        self.terrain: str | None = None
        self.tailwind = [0, 0]
        self.trick_room = 0
        self.screens: list[dict[str, int]] = [{}, {}]
        self.screen_key: list[tuple] = [(), ()]
        self.redirect: list[_Mon | None] = [None, None]
        self.turn = 0
        self._ruin: tuple | None = None
        self._best: dict[tuple, tuple] = {}
        self.planned: list[dict[int, float]] = [{}, {}]
        self.plans: dict[int, tuple] = {}
        self.reserved: set[int] = set()  # bench Pokémon a partner is already switching to this turn
        self._enter([m for side in self.active for m in side])

    # -- who is where ---------------------------------------------------------------------------------------------

    def field(self) -> list[_Mon]:
        return [m for side in self.active for m in side if m is not None and m.hp > 0]

    def foes(self, m: _Mon) -> list[_Mon]:
        return [f for f in self.active[1 - m.side] if f is not None and f.hp > 0]

    def ally(self, m: _Mon) -> _Mon | None:
        return next((a for a in self.active[m.side] if a is not None and a is not m and a.hp > 0), None)

    def bench(self, side: int) -> list[_Mon]:
        return [b for b in self.team[side] if b.slot is None and b.hp > 0]

    def over(self) -> bool:
        return not any(m.hp > 0 for m in self.team[0]) or not any(m.hp > 0 for m in self.team[1])

    def say(self, text: str, *args: object) -> None:
        """Log an event (``text`` formatted with ``args`` only when a log is kept: rollouts stay fast)."""
        if self.log is not None:
            self.log.append(f"T{self.turn}: " + (text.format(*args) if args else text))

    # -- the calculator, memoized ------------------------------------------------------------------------------------

    def ruin(self) -> tuple:
        if self._ruin is None:
            self._ruin = tuple(sorted(m.unit.ability for m in self.field() if m.unit.ability.endswith("ofruin")))
        return self._ruin

    def view(self, m: _Mon):
        return replace(m.unit.mon, boosts=dict(m.stages), status=m.status, item=m.item or None,
                       hp_fraction=1.0 if m.hp >= 99.95 else 0.5, side="mine" if m.side == 0 else "theirs")

    def hit(self, att: _Mon, move: dict, dfn: _Mon, spread: bool) -> tuple[float, float, int]:
        """(low, high) damage as % of ``dfn``'s max HP and the move's priority; (-1, -1) is Ruination (half its HP)."""
        key = (att.unit.key, move["id"], dfn.unit.key, att.skey, dfn.skey, dfn.hp >= 99.95, att.hp >= 99.95, dfn.item,
               att.item, att.status, spread, self.weather, self.terrain, self.screen_key[dfn.side], self.ruin())
        got = self.calc.hits.get(key)
        if got is None:
            got = self.calc.hits[key] = self._calc_hit(att, move, dfn, spread)
            self.calc.misses += 1
        return got

    def _calc_hit(self, att: _Mon, move: dict, dfn: _Mon, spread: bool) -> tuple[float, float, int]:
        a_mon, d_mon = self.view(att), self.view(dfn)
        others = [self.view(m) for m in self.field() if m is not att and m is not dfn]
        fs = FieldState(weather=self.weather, terrain=self.terrain, actives=[a_mon, d_mon] + others,
                        my_side=set(self.screen_key[0]), their_side=set(self.screen_key[1]))
        for v in fs.actives:
            v.boosted_stat = paradox_boosted_stat(v, fs)
        eff = effective_move(a_mon, move, d_mon, fs)
        priority = int(gale_wings(a_mon, eff).get("priority") or 0)  # the view is at full HP exactly when att.hp >= 99.95
        if move["id"] == "ruination":
            return (0.0, 0.0, priority) if data.effectiveness("dark", d_mon.types) == 0 else (-1.0, -1.0, priority)
        est = matchup.calc_hit(a_mon, move, d_mon, fs, spread=spread)
        if not est:
            return (0.0, 0.0, priority)
        return (est[0] * d_mon.hp_fraction, est[1] * d_mon.hp_fraction, priority)

    def acc(self, move: dict) -> float:
        key = (move["id"], self.weather)
        got = self.calc.accuracy.get(key)
        if got is None:
            got = self.calc.accuracy[key] = matchup.accuracy(move, self.weather)
        return got

    def speed(self, m: _Mon) -> float:
        key = (m.unit.key, m.skey, m.item, m.status, self.weather, self.terrain)
        got = self.calc.speeds.get(key)
        if got is None:
            view = replace(m.unit.mon, boosts=dict(m.stages), status=m.status, item=m.item or None)
            view.boosted_stat = paradox_boosted_stat(view, FieldState(weather=self.weather, terrain=self.terrain, actives=[view]))
            got = float(view.stat("spe"))
            if m.unit.ability == "unburden" and m.unit.item and not m.item:
                got *= 2
            self.calc.speeds[key] = got
        return got * 2 if self.tailwind[m.side] else got

    def grounded(self, m: _Mon) -> bool:
        return m.info.grounded and m.item != "airballoon"

    def usable(self, m: _Mon) -> list[dict]:
        """Its attacks, or only the one a Choice item locked it into."""
        if m.locked is not None:
            return [a for a in m.info.attacks if a["id"] == m.locked]
        return m.info.attacks

    def best(self, att: _Mon, dfn: _Mon) -> tuple:
        """``att``'s best attack into ``dfn`` right now: (expected % of max HP, KO chance, move, low, high, priority,
        accuracy). Cached for the decision phase of a turn."""
        key = (id(att), id(dfn))
        got = self._best.get(key)
        if got is None:
            got = (0.0, 0.0, None, 0.0, 0.0, 0, 1.0)
            n = len([f for f in self.active[1 - att.side] if f is not None and f.hp > 0])
            for move in self.usable(att):
                spread = move.get("target") in SPREAD and n > 1
                lo, hi, prio = self.hit(att, move, dfn, spread)
                if lo < 0:
                    lo = hi = dfn.hp / 2
                acc = self.acc(move)
                expected = (lo + hi) / 2 * acc
                if expected > got[0]:
                    got = (expected, acc * ko_chance(lo, hi, dfn.hp), move, lo, hi, prio, acc)
            self._best[key] = got
        return got

    # -- entering the field ------------------------------------------------------------------------------------------

    def _enter(self, mons: list[_Mon]) -> None:
        """Switch-in abilities, fastest first (so the slowest weather / terrain setter's stays): weather, terrain,
        Intimidate on every foe on the field (matchup.intimidated: the immunities, Defiant, Competitive, Clear Amulet)
        and Download."""
        for m in sorted(mons, key=lambda m: -self.speed(m)):
            if m.unit.weather:
                self.weather = m.unit.weather
            if m.unit.terrain:
                self.terrain = m.unit.terrain
        for m in mons:
            m.actions = 0
            foes = self.foes(m)
            if m.unit.ability == "intimidate":
                for f in foes:
                    f.boost(matchup.intimidated(f.unit))
            if m.unit.ability == "download" and foes:
                defense = sum(f.unit.mon.stats.get("def", 0) for f in foes)
                special = sum(f.unit.mon.stats.get("spd", 0) for f in foes)
                m.boost({"atk" if defense < special else "spa": 1})
        self._ruin = None

    def place(self, side: int, slot: int, m: _Mon) -> None:
        self.active[side][slot] = m
        m.slot = slot
        m.protect_last = m.used_protect = m.protecting = False
        self._ruin = None

    def switch(self, m: _Mon, sub: _Mon) -> None:
        if m.hp <= 0 or m.slot is None or sub.hp <= 0 or sub.slot is not None:
            return
        slot = m.slot
        if m.unit.ability == "regenerator":
            m.hp = min(100.0, m.hp + 100 / 3)
        m.slot, m.stages, m.skey, m.locked = None, {}, (), None
        m.protect_last = m.used_protect = m.protecting = False
        self.place(m.side, slot, sub)
        self.say("{} switches to {}", m.name, sub.name)
        self._enter([sub])

    # -- small rules ---------------------------------------------------------------------------------------------

    def flinchable(self, f: _Mon) -> bool:
        """Can a Fake Out flinch ``f``: not a Ghost, Inner Focus, Shield Dust or Covert Cloak, not grounded in Psychic
        Terrain, no Armor Tail / Dazzling / Queenly Majesty on its side."""
        if "ghost" in f.info.types or f.unit.ability in FLINCH_PROOF or f.item == "covertcloak":
            return False
        return not self.priority_blocked(f)

    def priority_blocked(self, target: _Mon) -> bool:
        """A priority move into ``target`` fails: Psychic Terrain with it grounded, or a priority guard on its side."""
        if self.terrain == "psychic" and self.grounded(target):
            return True
        return any(a is not None and a.hp > 0 and a.unit.ability in PRIORITY_GUARDS for a in self.active[target.side])

    def threat(self, f: _Mon) -> float:
        """The most ``f`` can put into one Pokémon of the other side this turn (expected % of its max HP)."""
        return max((self.best(f, o)[0] for o in self.foes(f)), default=0.0)

    def danger(self, m: _Mon) -> tuple[float, _Mon | None]:
        """The chance ``m`` is knocked out this turn: the likeliest single KO, or both foes' best hits together times
        the chance they both aim at it. Also returns the foe behind the single KO."""
        foes = self.foes(m)
        single, killer = 0.0, None
        for f in foes:
            ko = self.best(f, m)[1]
            if ko > single:
                single, killer = ko, f
        double = 0.0
        if len(foes) == 2:
            a, b = self.best(foes[0], m), self.best(foes[1], m)
            double = a[6] * b[6] * ko_chance(a[3] + b[3], a[4] + b[4], m.hp) * self.policy["focus"]
        return max(single, double), killer

    def first(self, m: _Mon, m_prio: int, f: _Mon, f_prio: int) -> bool:
        """``m`` moves before ``f`` (priority, then speed, Trick Room reversing it)."""
        if m_prio != f_prio:
            return m_prio > f_prio
        a, b = self.speed(m), self.speed(f)
        return (a < b) if self.trick_room else (a > b)

    def removes_first(self, m: _Mon, killer: _Mon | None) -> bool:
        """``m`` can knock ``killer`` out (80%+) before it moves."""
        if killer is None:
            return False
        hit = self.best(m, killer)
        return hit[1] >= 0.8 and self.first(m, hit[5], killer, self.best(killer, m)[5])

    def absorber(self, m: _Mon) -> _Mon | None:
        """The bench Pokémon that best takes ``m``'s place against the foes on the field: survives their best hits (no
        single one KOs it half the time) and hits back hardest for what it takes."""
        foes = self.foes(m)
        best, best_value = None, None
        for b in self.bench(m.side):
            if id(b) in self.reserved or any(self.best(f, b)[1] >= 0.5 for f in foes):
                continue
            value = max((self.best(b, f)[0] for f in foes), default=0.0) - sum(self.best(f, b)[0] for f in foes)
            if best is None or value > best_value:
                best, best_value = b, value
        return best

    def slower(self, side: int) -> bool:
        mine = [self.speed(m) for m in self.active[side] if m is not None and m.hp > 0]
        theirs = [self.speed(m) for m in self.active[1 - side] if m is not None and m.hp > 0]
        return bool(mine and theirs) and sum(mine) / len(mine) < sum(theirs) / len(theirs)

    def sleepable(self, f: _Mon, move: dict, user: _Mon) -> bool:
        """The sleep move lands on ``f`` (data.status_move_blocked; a Lum / Chesto Berry is left out: it wakes its holder
        once, and ``status_move`` eats it then)."""
        if f.status is not None:
            return False
        item = None if f.item in matchup.CURE_BERRIES else (f.item or None)
        view = replace(f.unit.mon, item=item, status=f.status)
        return data.status_move_blocked(move, view, user=user.unit.mon, terrain=self.terrain) is None

    def protect_risk(self, v: _Mon) -> float:
        """Our scorer's guess that foe ``v`` Protects this turn."""
        if v.info.protect is None or v.protect_last or v.locked is not None:
            return 0.0
        ko = max((self.best(o, v)[1] for o in self.foes(v)), default=0.0)
        return self.policy["protect_risk_ko"] if ko >= 0.5 else self.policy["protect_risk"]

    # -- choosing ------------------------------------------------------------------------------------------------------

    def gain(self, m: _Mon, move: dict, v: _Mon, spread: bool) -> float:
        lo, hi, _ = self.hit(m, move, v, spread)
        if lo < 0:
            lo = hi = v.hp / 2
        hp = v.hp - self.planned[m.side].get(id(v), 0.0)
        if hp <= 0:
            return 0.05 * (lo + hi) / 2  # the partner is expected to knock it out first
        value = self.acc(move) * (min((lo + hi) / 2, hp) + KO_BONUS * ko_chance(lo, hi, hp))
        if m.side == 0 and v.side == 1:
            value *= 1.0 - self.protect_risk(v)
        return value

    def attack(self, m: _Mon) -> tuple:
        """The attack and target that do the most (spread moves summed over their targets, our partner's damage
        charged), recorded so the partner deciding next sees it."""
        foes, ally = self.foes(m), self.ally(m)
        best: tuple[float, dict | None, _Mon | None] = (0.0, None, None)
        for move in self.usable(m):
            if move.get("target") in SPREAD:
                victims = foes + ([ally] if move["target"] == "allAdjacent" and ally is not None else [])
                spread = len(victims) > 1
                value = sum(-self.gain(m, move, v, spread) if v.side == m.side else self.gain(m, move, v, spread) for v in victims)
                if value > best[0]:
                    best = (value, move, None)
            else:
                for f in foes:
                    value = self.gain(m, move, f, False)
                    if value > best[0]:
                        best = (value, move, f)
        if best[1] is None:
            return IDLE
        move, target = best[1], best[2]
        for v in ([target] if target is not None else foes):
            lo, hi, _ = self.hit(m, move, v, target is None and len(foes) > 1)
            self.planned[m.side][id(v)] = self.planned[m.side].get(id(v), 0.0) + (v.hp / 2 if lo < 0 else (lo + hi) / 2) * self.acc(move)
        return ("attack", move, target.slot if target is not None else None, None)

    def decide(self, m: _Mon) -> tuple:
        return self._ours(m) if m.side == 0 else self._theirs(m)

    def _ours(self, m: _Mon) -> tuple:
        """Our cheap scorer: Fake Out their biggest threat on its first turn, speed control first, Protect a slot facing
        a likely KO it cannot prevent, switch out of a sure one, screens / redirection / sleep / burn, else attack."""
        info, foes = m.info, self.foes(m)
        if not foes:
            return IDLE
        free = m.locked is None
        if m.actions == 0 and info.fakeout is not None and free:
            options = [f for f in foes if self.flinchable(f)]
            if options:
                return ("fakeout", info.fakeout, max(options, key=lambda f: (self.threat(f), -f.slot)).slot, None)
        if free and info.tailwind is not None and not self.tailwind[0] and m.actions == 0:
            return ("tailwind", info.tailwind, None, None)
        if free and info.trickroom is not None and not self.trick_room and self.slower(0):
            return ("trickroom", info.trickroom, None, None)
        danger, killer = self.danger(m)
        can_protect = info.protect is not None and not m.protect_last and free and not info.choice
        if danger >= self.policy["protect"] and can_protect and not self.removes_first(m, killer):
            return ("protect", info.protect, None, None)
        if danger >= self.policy["switch"] and not can_protect and not self.removes_first(m, killer):
            sub = self.absorber(m)
            if sub is not None:
                return ("switch", None, None, sub)
        if free:
            if info.screens and self.turn == 1 and not self.screen_key[0]:
                physical = sum(1 for f in foes if f.info.physical)
                pick = next((s for s in info.screens if s["id"] == ("reflect" if physical * 2 >= len(foes) else "lightscreen")), info.screens[0])
                return ("screen", pick, None, None)
            ally = self.ally(m)
            if info.redirect is not None and ally is not None and self.danger(ally)[0] >= 0.5:
                return ("redirect", info.redirect, None, None)
            if info.sleep is not None:
                options = [f for f in foes if self.sleepable(f, info.sleep, m)]
                if options:
                    return ("sleep", info.sleep, max(options, key=lambda f: (self.threat(f), -f.slot)).slot, None)
            if info.burn is not None and self.turn == 1:
                options = [f for f in foes if f.info.physical and f.status is None and self.lands(f, info.burn, m)]
                if options:
                    return ("burn", info.burn, max(options, key=self.threat).slot, None)
        return self.attack(m)

    def lands(self, f: _Mon, move: dict, user: _Mon) -> bool:
        """A status move (Will-O-Wisp) works on ``f`` (data.status_move_blocked, its item and status as they are now)."""
        view = replace(f.unit.mon, item=f.item or None, status=f.status)
        return data.status_move_blocked(move, view, user=user.unit.mon, terrain=self.terrain) is None

    def _theirs(self, m: _Mon) -> tuple:
        """The search bot as its live decisions show it (``SEARCH_BOT``), each habit a coin with the measured odds."""
        P, roll = self.prior, self.rng.random
        info, foes = m.info, self.foes(m)
        if not foes:
            return IDLE
        free = m.locked is None
        ko = max((self.best(o, m)[1] for o in foes), default=0.0)
        top = max((self.best(o, m)[4] for o in foes), default=0.0)
        threatened = top >= 0.3 * m.hp
        fresh = m.actions == 0
        if fresh and info.fakeout is not None and free and roll() < P["fakeout"]:
            options = [f for f in foes if self.flinchable(f)]
            if options:
                options.sort(key=lambda f: (-self.threat(f), f.slot))
                pick = options[0] if len(options) == 1 or roll() < P["fakeout_threat"] else options[1]
                return ("fakeout", info.fakeout, pick.slot, None)
        if info.protect is not None and not m.protect_last and free:
            if self.turn == 1:
                p = P["protect_lead_ko"] if ko >= 0.5 else P["protect_lead"]
            elif fresh:
                p = P["protect_switchin"] if threatened else P["protect_idle"]
            elif ko >= 0.5:
                p = P["protect_ko_aimed"] if m.aimed else P["protect_ko"]
            elif threatened:
                p = P["protect_hit_aimed"] if m.aimed else P["protect_hit"]
            else:
                p = P["protect_idle"]
            if roll() < p:
                return ("protect", info.protect, None, None)
        if self.bench(m.side):
            spent = info.protect is None or m.protect_last or not free
            if ko >= 0.5:
                p = P["switch_spent_ko"] if spent else P["switch_ko"]
            elif len(foes) == 2 and self.danger(m)[0] >= 0.5 * self.policy["focus"]:
                p = P["switch_double"]
            else:
                p = P["switch_idle"]
            if roll() < p:
                sub = self.absorber(m)
                if sub is not None:
                    return ("switch", None, None, sub)
        if free:
            if info.tailwind is not None and not self.tailwind[1] and roll() < (P["tailwind"] if threatened else P["tailwind_free"]):
                return ("tailwind", info.tailwind, None, None)
            if info.trickroom is not None and not self.trick_room and self.slower(1) and roll() < P["trickroom"]:
                return ("trickroom", info.trickroom, None, None)
            if info.screens and not self.screen_key[1] and self.turn <= 2 and roll() < P["screens"]:
                physical = sum(1 for f in foes if f.info.physical)
                pick = next((s for s in info.screens if s["id"] == ("reflect" if physical * 2 >= len(foes) else "lightscreen")), info.screens[0])
                return ("screen", pick, None, None)
            if info.sleep is not None:
                options = [f for f in foes if self.sleepable(f, info.sleep, m)]
                if options and roll() < P["sleep"]:
                    return ("sleep", info.sleep, max(options, key=lambda f: (self.threat(f), -f.slot)).slot, None)
            ally = self.ally(m)
            if info.redirect is not None and ally is not None and self.danger(ally)[0] >= 0.5 and roll() < P["redirect"]:
                return ("redirect", info.redirect, None, None)
            if info.burn is not None:
                options = [f for f in foes if f.info.physical and self.lands(f, info.burn, m)]
                if options and roll() < P["burn"]:
                    return ("burn", info.burn, max(options, key=self.threat).slot, None)
            if info.setup is not None and not threatened and roll() < P["setup_free"]:
                return ("setup", info.setup[0], None, None)
        return self.attack(m)

    # -- one turn ------------------------------------------------------------------------------------------------------

    def priority(self, m: _Mon, action: tuple) -> int:
        kind, move, slot, _ = action
        if kind == "switch":
            return 7
        if move is None:
            return 0
        if kind == "attack" or kind == "fakeout":
            target = self.target(m, slot) or next(iter(self.foes(m)), None)
            if target is not None:
                return self.hit(m, move, target, False)[2]
        prio = int(move.get("priority") or 0)
        if m.unit.ability == "prankster" and move.get("category") == "Status":
            prio += 1
        return prio

    def target(self, m: _Mon, slot: int | None) -> _Mon | None:
        if slot is None:
            return None
        t = self.active[1 - m.side][slot]
        if t is None or t.hp <= 0:
            return next(iter(self.foes(m)), None)  # a fainted target: the move goes to the other foe
        return t

    def run_turn(self) -> None:
        self.turn += 1
        self._best.clear()
        self.planned = [{}, {}]
        self.reserved = set()
        self.redirect = [None, None]
        on = self.field()
        for m in on:
            m.flinched = m.acted = m.protecting = False
            m.tie = self.rng.random()
        plans: dict[int, tuple] = {}
        for side in (0, 1):
            mine = [m for m in on if m.side == side]
            mine.sort(key=lambda m: (self.speed(m) if self.trick_room else -self.speed(m), m.tie))
            for m in mine:
                plans[id(m)] = self.decide(m)
                if plans[id(m)][0] == "switch":
                    self.reserved.add(id(plans[id(m)][3]))
        self.plans = plans
        for m in sorted((m for m in on if plans[id(m)][0] == "switch"), key=lambda m: (-self.speed(m), m.tie)):
            self.switch(m, plans[id(m)][3])
        pending = [m for m in on if plans[id(m)][0] != "switch" and m.slot is not None]
        while pending:
            m = max(pending, key=lambda x: (self.priority(x, plans[id(x)]), -self.speed(x) if self.trick_room else self.speed(x), x.tie))
            pending.remove(m)
            if m.hp > 0 and m.slot is not None:
                m.actions += 1
                if not m.flinched:
                    self.execute(m, plans[id(m)])
            m.acted = True
        self.end_of_turn()
        self.replace_fainted()

    def execute(self, m: _Mon, action: tuple) -> None:
        kind, move, slot, _ = action
        if m.status == "slp":
            if m.sleep_left > 0:
                m.sleep_left -= 1
                self.say("{} is asleep", m.name)
                return
            m.status = None
            self.say("{} wakes up", m.name)
        if m.status == "par" and self.rng.random() < 0.25:
            return
        if kind == "idle" or move is None:
            return
        if self.log is not None:
            target = self.target(m, slot)
            self.say("{} uses {}{}", m.name, move["id"], f" -> {target.name}" if target is not None else "")
        if m.info.choice and m.locked is None:
            m.locked = move["id"]
        if kind == "protect":
            m.used_protect = True
            m.protecting = not m.protect_last or self.rng.random() < 1 / 3
            self.say("{} protects", m.name)
            return
        if kind == "tailwind":
            self.tailwind[m.side] = TAILWIND_TURNS
            self.say("{} sets Tailwind", m.name)
        elif kind == "trickroom":
            self.trick_room = 0 if self.trick_room else TRICK_ROOM_TURNS
            self.say("{} sets Trick Room", m.name)
        elif kind == "screen":
            self.screens[m.side][move["id"]] = LIGHT_CLAY_TURNS if m.item == "lightclay" else SCREEN_TURNS
            self.screen_key[m.side] = tuple(sorted(k for k, v in self.screens[m.side].items() if v > 0))
        elif kind == "redirect":
            self.redirect[m.side] = m
        elif kind == "setup":
            m.boost(m.info.setup[1])
            self.say("{} uses {}", m.name, move["id"])
        elif kind in ("sleep", "burn"):
            self.status_move(m, move, self.target(m, slot), kind)
        else:
            self.strike(m, move, self.target(m, slot), fake_out=(kind == "fakeout"))

    def status_move(self, m: _Mon, move: dict, t: _Mon | None, kind: str) -> None:
        if t is None or t.protecting or t.status is not None:
            return
        if m.unit.ability == "prankster" and self.priority_blocked(t):
            return
        if self.rng.random() >= self.acc(move):
            return
        if kind == "sleep":
            if not self.sleepable(t, move, m):
                return
            if t.item in matchup.CURE_BERRIES:
                t.item = ""  # it falls asleep and its berry wakes it at once
                self.say("{}'s berry wakes it from {}", t.name, move["id"])
                return
            t.status, t.sleep_left = "slp", self.rng.choice((1, 2, 3))
            self.say("{} puts {} to sleep", m.name, t.name)
        elif self.lands(t, move, m):
            t.status = "brn"
            self.say("{} burns {}", m.name, t.name)

    def strike(self, m: _Mon, move: dict, target: _Mon | None, *, fake_out: bool = False) -> None:
        if fake_out and m.actions > 1:
            return  # Fake Out works only on its first action after coming in
        foes = self.foes(m)
        if move.get("target") in SPREAD:
            ally = self.ally(m)
            victims = foes + ([ally] if move["target"] == "allAdjacent" and ally is not None else [])
        else:
            guard = self.redirect[1 - m.side]
            if guard is not None and guard.hp > 0 and guard.slot is not None and guard.side != m.side \
                    and not data.redirect_ignored(guard.info.redirect, m.unit.mon):
                target = guard
            if target is None:
                return
            if move["id"] in data.NEEDS_TARGET_ATTACKING:
                plan = self.plans.get(id(target))
                if target.acted or plan is None or plan[0] not in ("attack", "fakeout"):
                    return  # Sucker Punch fails unless its target is about to attack
            if target.side != m.side:
                target.aimed_now = True
            victims = [target]
        spread = move.get("target") in SPREAD and len(victims) > 1
        pierce = m.unit.ability == "unseenfist" and "contact" in (move.get("flags") or [])
        acc = self.acc(move)
        landed = False
        for v in victims:
            if v.hp <= 0:
                continue
            if v.protecting and not pierce:
                self.say("{} protects itself from {}'s {}", v.name, m.name, move["id"])
                continue
            lo, hi, prio = self.hit(m, move, v, spread)
            if prio > 0 and v.side != m.side and self.priority_blocked(v):
                continue
            if acc < 1.0 and self.rng.random() >= acc:
                self.say("{}'s {} misses {}", m.name, move["id"], v.name)
                continue
            dmg = v.hp / 2 if lo < 0 else lo + (hi - lo) * self.rng.random()
            if dmg <= 0:
                continue
            landed = True
            self.damage(m, move, v, dmg, fake_out)
        if landed:
            for stat, delta in matchup.SELF_STAGES.get(move["id"], {}).items():
                m.boost({stat: delta})
            if m.item == "lifeorb" and m.hp > 0:
                m.hp -= 10.0
            matchup._Race._berry(m)  # type: ignore[arg-type]  (same fields: hp, item)
            if m.hp <= 0:
                self._ruin = None

    def damage(self, m: _Mon, move: dict, v: _Mon, dmg: float, fake_out: bool) -> None:
        dealt = min(dmg, v.hp)
        v.hp -= dmg
        self.say("{} takes {:.0f}% ({:.0f}% left)", v.name, dmg, max(0.0, v.hp))
        if v.item == "airballoon":
            v.item = ""
        if v.item == "focussash" and 0 < v.hp < 1.5:
            v.item = ""
        if v.item in RESIST_BERRIES:
            eff_type = data.to_id(effective_move(m.unit.mon, move, v.unit.mon, None).get("type"))
            berry = RESIST_BERRIES[v.item]
            if berry == eff_type and (berry == "normal" or data.effectiveness(eff_type, v.unit.types) > 1):
                v.item = ""
        if v.hp > 0:
            if not v.acted and v.unit.ability not in FLINCH_PROOF and v.item != "covertcloak":
                chance = 1.0 if fake_out else FLINCH_CHANCE.get(move["id"], 0.0)
                if chance and (chance >= 1.0 or self.rng.random() < chance):
                    v.flinched = True
            drops = matchup.TARGET_STAGES.get(move["id"])
            if drops and v.unit.ability not in DROP_PROOF and v.item != "clearamulet":
                v.boost(drops)
                if v.unit.ability == "defiant":
                    v.boost({"atk": 2})
                elif v.unit.ability == "competitive":
                    v.boost({"spa": 2})
            if v.unit.ability == "stamina" and move.get("category") == "Physical":
                v.boost({"def": 1})
            if move["id"] == "knockoff" and v.item and v.item != "boosterenergy":
                v.item = ""
        ratio = (v.unit.maxhp / m.unit.maxhp) if m.unit.maxhp else 1.0
        if move["id"] in matchup.RECOIL:
            m.hp -= matchup.RECOIL[move["id"]] * dealt * ratio
        if move["id"] in matchup.DRAIN:
            m.hp = min(100.0, m.hp + matchup.DRAIN[move["id"]] * dealt * ratio)
        if "contact" in (move.get("flags") or []):
            hits = data.expected_hits(move, item=m.item)
            if v.item == "rockyhelmet":
                m.hp -= 100 / 6 * hits
            if v.unit.ability in ("roughskin", "ironbarbs"):
                m.hp -= 12.5 * hits
        matchup._Race._berry(v)  # type: ignore[arg-type]
        if v.hp <= 0:
            self._ruin = None
            self.say("{}'s {} KOs {}", m.name, move["id"], v.name)

    def end_of_turn(self) -> None:
        for m in self.field():
            if m.status == "brn":
                m.hp -= 6.25
            elif m.status in ("psn", "tox"):
                m.hp -= 6.25
            if m.item in ("flameorb", "toxicorb") and m.status is None and "fire" not in m.info.types:
                m.status = "brn" if m.item == "flameorb" else "tox"
            if self.weather == "sand" and not m.info.types & {"rock", "ground", "steel"} and m.item != "safetygoggles" \
                    and m.unit.ability not in SAND_PROOF:
                m.hp -= 6.25
            if m.item == "leftovers":
                m.hp = min(100.0, m.hp + 6.25)
            elif m.item == "blacksludge":
                m.hp = min(100.0, m.hp + 6.25) if "poison" in m.info.types else m.hp - 12.5
            if self.terrain == "grassy" and self.grounded(m):
                m.hp = min(100.0, m.hp + 6.25)
            matchup._Race._berry(m)  # type: ignore[arg-type]
        for side in (0, 1):
            self.tailwind[side] = max(0, self.tailwind[side] - 1)
            if self.screens[side]:
                self.screens[side] = {k: v - 1 for k, v in self.screens[side].items() if v > 1}
                self.screen_key[side] = tuple(sorted(self.screens[side]))
        self.trick_room = max(0, self.trick_room - 1)
        for m in self.field():
            m.protect_last, m.used_protect, m.protecting = m.used_protect, False, False
            m.aimed, m.aimed_now = m.aimed_now, False
        self._ruin = None

    def replace_fainted(self) -> None:
        """Fill every fainted slot from the bench with the Pokémon that best fits the foes on the field."""
        entering = []
        for side in (0, 1):
            for slot in (0, 1):
                m = self.active[side][slot]
                if m is not None and m.hp > 0:
                    continue
                if m is not None:
                    m.slot = None
                self.active[side][slot] = None
                bench = self.bench(side)
                if not bench:
                    continue
                self._best.clear()
                foes = [f for f in self.active[1 - side] if f is not None and f.hp > 0]

                def fit(b: _Mon) -> float:
                    if not foes:
                        return b.hp
                    return max(self.best(b, f)[0] for f in foes) - max(self.best(f, b)[0] for f in foes) + 0.3 * b.hp

                sub = max(bench, key=lambda b: (fit(b), -self.team[side].index(b)))
                self.place(side, slot, sub)
                self.say("{} comes in", sub.name)
                entering.append(sub)
        if entering:
            self._enter(entering)

    # -- the end of the horizon ----------------------------------------------------------------------------------------

    def margin(self) -> float:
        """Ours minus theirs, in Pokémon: each alive Pokémon VALUE["alive"] + VALUE["hp"] x its HP fraction (asleep,
        burned (a physical attacker) or paralysed: less; its attacking stage +-VALUE["stage"] a step), Tailwind turns
        still to run, a Fake Out still fresh (a bench Fake Out user re-enters with one), Trick Room for the slower side."""
        value = [0.0, 0.0]
        for side in (0, 1):
            for m in self.team[side]:
                if m.hp <= 0:
                    continue
                x = VALUE["alive"] + VALUE["hp"] * min(100.0, m.hp) / 100.0
                if m.status == "slp" and m.sleep_left > 0:
                    x *= VALUE["asleep"]
                elif m.status == "brn" and m.info.physical and m.unit.ability != "guts":
                    x *= VALUE["burned"]
                elif m.status == "par":
                    x *= VALUE["paralysed"]
                stage = m.stages.get("atk" if m.info.physical else "spa", 0)
                x *= 1.0 + VALUE["stage"] * max(-2, min(2, stage))
                if m.info.fakeout is not None and (m.slot is None or m.actions == 0):
                    x += VALUE["fresh_fake_out"]
                value[side] += x
            value[side] += VALUE["tailwind_turn"] * self.tailwind[side]
        if self.trick_room:
            side = 0 if self.slower(0) else (1 if self.slower(1) else None)
            if side is not None:
                value[side] += VALUE["trick_room_turn"] * self.trick_room
        return value[0] - value[1]

    def lost(self, side: int) -> list[str]:
        return [m.name for m in self.team[side] if m.hp <= 0]


# -- the anytime search over plans ------------------------------------------------------------------------------------

GOLDEN = 0.6180339887498949
NOMINAL_ROLLOUT_S = 0.0004  # what the schedule assumes a 4-turn rollout costs (measured 0.3-0.5 ms on the live sixes, Oct 9)
BUDGET_SHARE = 0.75  # the share of the budget the schedule plans to use; the rest absorbs a cold memo and a loaded machine
KEEP_SHARES = (0.5, 0.67, 0.75)  # the share of plans each successive-halving round may keep


@dataclass
class PlanStats:
    """One plan (board indices: the four we bring and the two we lead) and its rollout margins per four of theirs."""

    four: tuple[int, ...]
    leads: tuple[int, int]
    sums: list[float]
    counts: list[int]
    rounds: int = 0  # successive-halving rounds it survived

    def means(self) -> list[float | None]:
        return [s / c if c else None for s, c in zip(self.sums, self.counts)]

    @property
    def n(self) -> int:
        return sum(self.counts)


@dataclass
class Ranked:
    plan: PlanStats
    score: float
    expected: float
    worst: float
    worst_four: int  # index into Search.their


@dataclass
class SearchResult:
    ranked: list[Ranked]  # best first: plans that survived more rounds first, then by score
    rollouts: int
    rounds: int
    seconds: float
    complete: bool  # every plan got its first round against every four of theirs
    seed: int
    plans: int
    turns: int
    their: list[tuple[float, tuple[int, ...], list[tuple[tuple[int, int], float]]]] = field(default_factory=list)


class Search:
    """Successive halving over ``plans`` (each (four, leads) of board indices into ``units``) against ``their``:
    (weight, their four, [(their lead pair, weight), ...]) for each four they are likely to bring."""

    def __init__(self, units: list[matchup.Unit], plans: list[tuple[tuple[int, ...], tuple[int, int]]],
                 their: list[tuple[float, tuple[int, ...], list[tuple[tuple[int, int], float]]]], *, seed: int,
                 turns: int = 4, expected_w: float = 0.7, prior: dict | None = None, calc: Calc | None = None,
                 finalists: int = 3, four_value: dict[tuple, float] | None = None,
                 keep: tuple[tuple[int, ...], tuple[int, int]] | None = None) -> None:
        self.units = units
        self.four_value = dict(four_value or {})  # a constant added to every plan of a four (the bench's long-run worth)
        total = sum(w for w, _, _ in their) or 1.0
        self.their = [(w / total, four, sorted(pairs, key=lambda t: (-t[1], t[0]))) for w, four, pairs in their]
        self.plans = [PlanStats(tuple(f), tuple(l), [0.0] * len(self.their), [0] * len(self.their)) for f, l in plans]
        self.seed = int(seed) & ((1 << 64) - 1)
        self.turns = int(turns)
        self.expected_w = float(expected_w)
        self.prior = prior
        self.calc = calc or _CALC
        self.offsets = [(g * 0.7548776662466927) % 1.0 for g in range(len(self.their))]
        # the final round plays the best plan of each of this many different fours, on equal rollouts (the shortlist),
        # and ``keep`` (the plan the caller would play without the search) whatever its value, so the two meet on equal terms
        self.finalists = max(1, min(int(finalists), len({p.four for p in self.plans}) or 1))
        self.keep = next((i for i, p in enumerate(self.plans) if keep is not None and p.four == tuple(keep[0])
                          and set(p.leads) == set(keep[1])), None)

    def their_leads(self, g: int, k: int) -> tuple[int, int]:
        """Their lead pair for rollout ``k`` against four ``g``: a low-discrepancy draw from its lead distribution, so
        every prefix of rollouts covers their likely leads in proportion."""
        pairs = self.their[g][2]
        total = sum(w for _, w in pairs) or 1.0
        u = (k * GOLDEN + self.offsets[g]) % 1.0 * total
        for pair, w in pairs:
            u -= w
            if u < 0:
                return pair
        return pairs[-1][0]

    def rollout(self, plan: PlanStats, g: int, k: int, log: list | None = None) -> float:
        _, four, _ = self.their[g]
        pair = self.their_leads(g, k)
        rng = random.Random((self.seed * 0x9E3779B97F4A7C15 + g * 1_000_003 + k) & ((1 << 64) - 1))
        sim = Sim([self.units[i] for i in plan.four], (plan.four.index(plan.leads[0]), plan.four.index(plan.leads[1])),
                  [self.units[j] for j in four], (four.index(pair[0]), four.index(pair[1])), rng=rng, calc=self.calc,
                  prior=self.prior, log=log)
        for _ in range(self.turns):
            if sim.over():
                break
            sim.run_turn()
        return sim.margin()

    def value(self, plan: PlanStats) -> tuple[float, float, float, int] | None:
        """(score, expected, worst, worst four) from the rollouts so far; None until every four has one. The score is
        ``expected_w`` x expected + the rest x worst, plus the four's constant (``four_value``)."""
        means = plan.means()
        if any(m is None for m in means):
            return None
        expected = sum(w * m for (w, _, _), m in zip(self.their, means))
        worst_g = min(range(len(means)), key=lambda g: (means[g], g))
        worst = means[worst_g]
        score = self.expected_w * expected + (1 - self.expected_w) * worst + self.four_value.get(plan.four, 0.0)
        return score, expected, worst, worst_g

    def schedule(self, budget_s: float) -> tuple[int, float]:
        """(rollouts per (plan, four) in the first round, share of plans kept each round): the costliest schedule
        whose successive halving fits ``budget_s`` at NOMINAL_ROLLOUT_S a rollout (more rollouts first, then more
        survivors). A fixed cost, not a measured one, so a search that finishes in time is the same search on any
        machine (the deadline still stops a slow one)."""
        target = max(0.0, budget_s) * BUDGET_SHARE / NOMINAL_ROLLOUT_S
        best = (0, 1, 0.5)
        for keep in KEEP_SHARES:
            c0 = 1
            while self.cost(c0 + 1, keep) <= target and c0 < 4096:
                c0 += 1
            cost = self.cost(c0, keep)
            if cost <= target and (cost, c0) > (best[0], best[1]):
                best = (cost, c0, keep)
        return best[1], best[2]

    def cost(self, c0: int, keep: float) -> int:
        """Rollouts in a full run (``run``): ``c0`` per (plan, four) first, then the better ``keep`` share of the plans
        (never fewer than the protected ones) with their count per four doubled, until only the protected plans are
        left and have played their round."""
        floor = self.finalists + (1 if self.keep is not None else 0)
        n, done, total, per = len(self.plans), 0, 0, c0
        while n >= 1:
            total += n * len(self.their) * per
            done += per
            if n <= floor:
                break
            n, per = max(floor, min(n - 1, int(n * keep + 0.5))), done
        return total

    def protected(self, ranked: list[int]) -> list[int]:
        """Of ``ranked`` (best first): the best plan of each of the best ``finalists`` fours, and the kept plan."""
        out: list[int] = []
        seen: set[tuple] = set()
        for i in ranked:
            four = self.plans[i].four
            if four not in seen and len(seen) < self.finalists:
                seen.add(four)
                out.append(i)
        if self.keep is not None and self.keep in ranked and self.keep not in out:
            out.append(self.keep)
        return out

    def halve(self, alive: list[int], keep: float) -> list[int]:
        """The better ``keep`` share of ``alive`` by value, always with the protected plans."""
        ranked = sorted(alive, key=lambda i: (-self.value(self.plans[i])[0], i))
        protected = self.protected(ranked)
        n = max(len(protected), min(len(alive) - 1, int(len(alive) * keep + 0.5)))
        chosen = set(protected)
        for i in ranked:
            if len(chosen) >= n:
                break
            chosen.add(i)
        return [i for i in ranked if i in chosen]

    def run(self, budget_s: float, *, now: Callable[[], float] = time.monotonic, max_rollouts: int | None = None,
            first: int | None = None) -> SearchResult:
        """Successive halving until only the finalists are left (the best plan of each of the best ``finalists``
        fours, played on equal rollouts), ``budget_s`` seconds have passed or ``max_rollouts`` rollouts are done. The
        first round gives every plan ``first`` rollouts against each four of theirs (else ``schedule`` sizes the run
        to the budget); each later round keeps the better share of the plans and gives them as many new rollouts as
        all earlier rounds together, so the count per survivor doubles."""
        started = now()
        deadline = started + max(0.0, budget_s)
        alive = list(range(len(self.plans)))
        done = rounds = rollouts = 0
        per, keep = (first, 0.5) if first is not None else self.schedule(budget_s)
        stopped = complete = False
        while alive and not stopped:
            for k in range(done, done + per):
                for g in range(len(self.their)):
                    for i in alive:
                        if now() >= deadline or (max_rollouts is not None and rollouts >= max_rollouts):
                            stopped = True
                            break
                        plan = self.plans[i]
                        plan.sums[g] += self.rollout(plan, g, k)
                        plan.counts[g] += 1
                        rollouts += 1
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
            done += per
            rounds += 1
            complete = True
            for i in alive:
                self.plans[i].rounds = rounds
            ranked = sorted(alive, key=lambda i: (-self.value(self.plans[i])[0], i))
            if set(alive) <= set(self.protected(ranked)):
                break
            alive = self.halve(alive, keep)
            per = done
        self.calc.trim()
        ranked = []
        for plan in self.plans:
            got = self.value(plan)
            if got is not None:
                ranked.append(Ranked(plan, *got))
        ranked.sort(key=lambda r: (-r.plan.rounds, -r.score, r.plan.four, r.plan.leads))
        return SearchResult(ranked, rollouts, rounds, now() - started, complete, self.seed, len(self.plans), self.turns,
                            list(self.their))
