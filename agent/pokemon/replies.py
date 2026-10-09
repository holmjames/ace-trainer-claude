"""Workstream A: the reply mixture over the sheet's rows.

The old scorer values every turn as if the opponent stands still. This module values each of OUR plausible turns (a row)
against each of THEIR plausible replies (a column) from rows the sheet already has: our damage rows per target, their
threat rows per slot, the switch previews, the double-target threats, the flinch odds and the speed order, plus two
tables built here: their switch-ins (each bench card with its boosts reset and its entry effects, our moves' damage into
it) and post-move threat rows (our move's stat drops, recoil and Glaive Rush applied before the slower threats land).

A cell is resolved in move order: Protect blanks the hits into the protected Pokémon (a repeat works 1 time in 3), a
switch sends our hits into the newcomer, a fresh Fake Out flinches its target, a KO before a Pokémon moves takes its
action, a first hit breaks Multiscale or a Focus Sash for the next one. The cell's value is the damage landed (capped
per foe, one KO per foe, accuracy and hit counts included) plus KOs weighted by the threat removed, minus our material
lost, plus or minus tempo (speed control, setup, a free switch, a dead Choice lock).

Rows are ranked by a blend of the equilibrium value (regret matching over the matrix) and the value against the
opponent prior (the opponent class's tendencies, gated on what the memory has seen), a worst-case floor keeps out rows
whose worst reply is far below the best worst case, and rows that tie within epsilon are sampled at a low temperature
from a per-match secret seed so a fictitious-play opponent cannot lock onto a pattern. Nothing here is keyed to an
opponent's name: the classes are search bot / generic LLM / random, from behaviour.

Cost: 12 rows x 10 columns, each cell a few dozen row lookups; measured in milliseconds, see tests/test_replies_mix.py.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any

from . import battle, data
from .memory import MatchMemory, species_key
from .tuning import DEFAULTS

PROTECT_IDS = battle.PROTECT_OPTIONS
SETUP_IDS = {"nastyplot": 30, "calmmind": 25, "swordsdance": 30, "dragondance": 30, "bulkup": 25, "quiverdance": 35,
             "tailwind": 28, "trickroom": 28, "icywind": 20, "electroweb": 20, "snarl": 12}
FLINCH_IMMUNE_ABILITIES = ("innerfocus", "shielddust")

# The search-bot class's starting rates: (k+1)/(n+2) from the Oct 8 pooled tally of the class's 130 decisions over the
# eight live games (Oct 7-8; Oct 8 alone where the two days differ at Fisher p < 0.1). The in-game tallies (A4) move
# them; a tally file outside the repository overrides them at import time when it is present.
SEARCH_BOT_SEEDS = {
    "protect_after_protect": 0.05,       # 0/17: never twice in a row
    "fresh_protect_ko50": 0.47,          # 6/13: a fresh Protect against our one-move KO
    "fresh_protect_30_noko": 0.32,       # 6/20: taking >= 30% with no KO on the board
    "switchin_first_turn_protect": 0.43, # 5/12: a fresh switch-in on its first turn
    "lead_turn1_protect": 0.57,          # 3/5 on the current build: a lead on turn 1
    "spent_protect_ko_switch": 0.38,     # 3/6 like-for-like: Protect spent, our move KOs it, a better absorber behind
    "double_only_ko_stays": 0.88,        # 22/24: it stays when only our two attacks together KO it
    "double_only_ko_protect": 0.40,      # the no-Protect-on-t-1 branch of the same denominator
    "both_attack_same_target": 0.61,     # 10/16 turns
    "fakeout_first_turn": 0.50,          # 8/16, aimed at the slot whose action it can stop
    "fakeout_after_first_turn": 0.07,    # 0/16 since its stale-Fake-Out fix
    "setup_unthreatened": 0.36,          # 4/12
    "tailwind_when_available": 0.22,     # 5/25
    "voluntary_switch_rate": 0.16,       # 14/130 decisions (8/35 when facing our >= 0.5 KO with a bench)
}
TALLIES_PATH = os.environ.get("ACE_OPPONENT_TALLIES", os.path.expanduser("~/projects/ace-trainer-data/opponent-tallies-2026-10-08.json"))


def _load_seeds(path: str) -> dict[str, float]:
    """The tally file's suggested seeds over the dictionary above; the dictionary alone when the file is missing."""
    seeds = dict(SEARCH_BOT_SEEDS)
    try:
        import json

        with open(path) as f:
            doc = json.load(f)
        for event in doc.get("events") or []:
            name, seed = event.get("event"), event.get("suggested_seed")
            if name in seeds and seed is not None:
                seeds[name] = float(seed)
    except Exception:
        pass
    return seeds


SEEDS = _load_seeds(TALLIES_PATH)


# -- the opponent class and its priors ------------------------------------------------------------------------------


def opponent_class(memory: MatchMemory | None) -> str:
    """'search', 'llm' or 'random', from the ops lane's classifier when it exists (memory.opponent_class()), else from
    the draft profile (templated 30-40 character reasons = search bot), else 'search' (the field's strongest class)."""
    fn = getattr(memory, "opponent_class", None)
    if callable(fn):
        try:
            label = str(fn() or "").lower()
            if "random" in label or "dead" in label:
                return "random"
            if "llm" in label or "generic" in label:
                return "llm"
            if label:
                return "search"
        except Exception:
            pass
    profile = getattr(memory, "draft_profile", None)
    if callable(profile):
        try:
            info = profile() or {}
            style = str(info.get("reason_style") or info.get("style") or "").lower()
            if "free" in style or "llm" in style:
                return "llm"
            if "templated" in style or "search" in style:
                return "search"
        except Exception:
            pass
    return "search"


def _p_protect(klass: str, *, streak: int, ko_one: float, ko_two: float, facing: float, fresh_in: bool, turn: int | None) -> float:
    """Chance this foe Protects this turn. Search bot: the pooled tally's seeds (SEEDS): never twice in a row, a fresh
    Protect against our one-move KO about a coin flip, against a double-only KO 0.4, a lead on turn 1 and a fresh
    switch-in on its first turn a little above that, a third when it only takes >= 30%. Generic LLM: Protects in KO
    range about half the time, even again."""
    if klass == "random":
        return 0.15
    if klass == "llm":
        base = 0.5 if ko_one >= 0.5 else (0.3 if ko_two >= 0.5 else 0.1)
        return base * (0.5 if streak else 1.0)
    if streak:
        return SEEDS["protect_after_protect"]
    if ko_one >= 0.5:
        if (turn or 1) <= 1:
            return SEEDS["lead_turn1_protect"]
        if fresh_in:
            return max(SEEDS["switchin_first_turn_protect"], SEEDS["fresh_protect_ko50"])
        return SEEDS["fresh_protect_ko50"]
    if ko_two >= 0.5:
        return SEEDS["double_only_ko_protect"]
    if fresh_in:
        return SEEDS["switchin_first_turn_protect"]
    if facing >= 30:
        return SEEDS["fresh_protect_30_noko"]
    return 0.1


def _p_switch(klass: str, *, streak: int, ko_one: float, ko_two: float = 0.0, has_absorber: bool, has_protect: bool = True) -> float:
    """Chance this foe switches out. Search bot (SEEDS): Protect spent (or no Protect) + our move KOs it + a better
    absorber behind -> 0.38; facing our one-move KO with a fresh Protect 8/35 of the time; it STAYS when only our two
    attacks together KO it (0.88); otherwise the voluntary rate 0.16 over a bench. Generic LLM: 1 switch in 14."""
    if klass == "random":
        return 0.1 if has_absorber else 0.0
    if not has_absorber:
        return 0.0
    if klass == "llm":
        return 0.1
    if ko_one >= 0.5 and (streak or not has_protect):
        return SEEDS["spent_protect_ko_switch"]
    if ko_one >= 0.5:
        return 0.23  # 8/35 facing a >= 0.5 KO with a bench (voluntary_switch_rate's split)
    if ko_two >= 0.5:
        return max(0.02, 1.0 - SEEDS["double_only_ko_stays"] - SEEDS["double_only_ko_protect"] * 0.3)
    return SEEDS["voluntary_switch_rate"] * 0.5


def _p_setup(klass: str, *, threatened: bool, turn: int | None, move_id: str) -> float:
    if klass == "random":
        return 0.1
    if threatened:
        return 0.05
    if move_id == "tailwind":
        return SEEDS["tailwind_when_available"] if klass == "search" else 0.15
    return 0.15 if klass == "llm" else SEEDS["setup_unthreatened"]


def _p_fakeout(klass: str, *, turn: int | None) -> float:
    """A fresh Fake Out: 8/16 first turns on the field for the search bot (turn-1 leads 1/4), aimed at the stoppable slot."""
    if klass == "random":
        return 0.3
    if klass == "search":
        return SEEDS["fakeout_first_turn"]
    return 0.6 if (turn or 1) <= 1 else 0.2


# -- the matrix ----------------------------------------------------------------------------------------------------


class _Actor:
    """One Pokémon's action this turn, resolved in order: kind in {attack, protect, switch, fakeout, setup, status,
    pass}; ``rows`` the damage rows it lands (per target key); ``priority``; ``slot`` our slot or their position."""

    __slots__ = ("side", "slot", "kind", "move", "priority", "rows", "target", "species", "payload", "incoming", "name", "spread", "ally_rows")

    def __init__(self, side: str, slot: int, kind: str, *, move: str = "", priority: int = 0, rows: dict | None = None,
                 target: object = None, species: str = "", payload: dict | None = None, incoming: object = None,
                 name: str = "", spread: bool = False, ally_rows: dict | None = None) -> None:
        self.side, self.slot, self.kind, self.move, self.priority = side, slot, kind, move, priority
        self.rows, self.target, self.species, self.payload = rows or {}, target, species, payload
        self.incoming, self.name, self.spread, self.ally_rows = incoming, name, spread, ally_rows or {}


def _key(side: str, slot: int) -> tuple[str, int]:
    return (side, slot)


def evaluate(sheet: battle.TurnSheet, memory: MatchMemory | None, rng: random.Random | None = None, *, slots: list[dict],
             ours: dict[int, battle.Mon], theirs: dict[int, battle.Mon], params: dict | None = None,
             candidates: list[dict] | None = None, obs: dict | None = None) -> list[dict]:
    """Rank complete turns by the reply mixture. Returns candidate dicts in the sheet's shape ({name, slot_0, slot_1,
    why, score}) plus {value, worst, best_reply, if_protect, if_switch, what}; fills ``sheet.reply`` with the table."""
    P = params or DEFAULTS
    if not ours or not theirs:
        return list(candidates or [])
    klass = opponent_class(memory)
    turn = (obs or {}).get("turn") if obs else None
    if turn is None:
        turn = getattr(memory, "turn", None)
    seed = _match_seed(memory, P)
    rng = rng or random.Random(seed)
    ctx = _Context(sheet, memory, slots, ours, theirs, P, klass, turn, (obs or {}).get("opponent_team"))
    rows = ctx.our_menu(candidates or [])
    cols = ctx.their_menu()
    if not rows or not cols:
        return list(candidates or [])
    matrix = [[ctx.cell(r, c) for c in cols] for r in rows]
    prior = ctx.prior(cols)
    lam = _lambda(memory, ctx, cols, prior)
    floor_cols = [j for j in range(len(cols)) if prior[j] >= float(P.get("reply_floor_min_prior", 0.08))] or list(range(len(cols)))
    # the equilibrium is over the PLAUSIBLE columns: a reply the class never makes (a second Protect in a row) is not
    # a strategy the opponent can mix in
    sub = _regret_matching([[matrix[i][j] for j in floor_cols] for i in range(len(rows))], iterations=int(P.get("reply_iterations", 200)))
    eq_cols = [0.0] * len(cols)
    for k, j in enumerate(floor_cols):
        eq_cols[j] = sub[k]
    scored = []
    for i, r in enumerate(rows):
        v_pi = sum(matrix[i][j] * prior[j] for j in range(len(cols)))
        v_eq = sum(matrix[i][j] * eq_cols[j] for j in range(len(cols)))
        worst = min(matrix[i][j] for j in floor_cols)  # the worst PLAUSIBLE reply: a column nobody plays is not a floor
        value = (1.0 - lam) * v_eq + lam * v_pi + float(P.get("reply_worst_w", 0.15)) * worst  # a little risk aversion
        scored.append((value, worst, i))
    best_worst = max(w for _, w, _ in scored)
    margin = float(P.get("reply_floor_margin", 100.0))
    kept = [(v, w, i) for v, w, i in scored if w >= best_worst - margin] or scored
    kept.sort(key=lambda t: -t[0])
    # tie-break: rows within epsilon of the top are sampled at a low temperature from the per-match seed
    eps = float(P.get("reply_tie_eps", 2.0))
    top_value = kept[0][0]
    tied = [t for t in kept if t[0] >= top_value - eps]
    if len(tied) > 1:
        temperature = float(P.get("reply_temperature", 0.3))
        weights = [math.exp((t[0] - top_value) / max(temperature * eps, 1e-6)) for t in tied]
        pick = rng.choices(range(len(tied)), weights=weights)[0]
        chosen = tied[pick]
        kept = [chosen] + [t for t in kept if t is not chosen]
    out = []
    for value, worst, i in kept:
        r = rows[i]
        j_best = min(floor_cols, key=lambda j: matrix[i][j])
        cand = dict(r.payload)
        cand.update({"score": round(value, 1), "value": round(value, 1), "worst": round(worst, 1),
                     "best_reply": cols[j_best].name, "what": ctx.describe(r, cols[j_best])})
        target_pos = ctx.main_target(r)
        for tag, col in (("if_protect", ctx.protect_column(cols, target_pos)), ("if_switch", ctx.switch_column(cols, target_pos))):
            if col is not None:
                cand[tag] = round(matrix[i][cols.index(col)], 1)
        out.append(cand)
    sheet.reply = {"seed": seed, "class": klass, "lambda": round(lam, 2),
                   "columns": [{"name": c.name, "prior": round(prior[j], 2), "eq": round(eq_cols[j], 2)} for j, c in enumerate(cols)],
                   "rows": [{"name": c["name"], "value": c["value"], "worst": c["worst"], "best_reply": c["best_reply"]} for c in out[:8]]}
    _remember_prediction(memory, ctx, cols, prior, turn)
    return out


def _match_seed(memory: MatchMemory | None, P: dict | None = None) -> int:
    """A per-match secret seed (logged in the sheet's reply table), made once from os.urandom; ``reply_seed`` in the
    params pins it (tests, replays)."""
    pinned = int((P or {}).get("reply_seed", 0) or 0)
    if pinned:
        return pinned
    seed = getattr(memory, "_reply_seed", None)
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
        if memory is not None:
            try:
                setattr(memory, "_reply_seed", seed)
            except Exception:
                pass
    return int(seed)


def _lambda(memory: MatchMemory | None, ctx: "_Context", cols: list["_Actor"], prior: list[float]) -> float:
    """The weight on the opponent prior: 0.3 at the start, +0.15 per turn the predicted column matched what the
    protocol log shows the opponent did, halved per miss, capped at 0.7."""
    lam = float(getattr(memory, "_reply_lambda", 0.3) or 0.3)
    predicted = getattr(memory, "_reply_predicted", None)
    if predicted and memory is not None and isinstance(ctx.turn, int) and predicted.get("turn") is not None and ctx.turn > predicted["turn"]:
        matched = ctx.prediction_matched(predicted)
        if matched is not None:
            # both foes as predicted: +0.15; one of two: unchanged; neither: halved (floor 0.15)
            lam = min(0.7, lam + 0.15) if matched >= 1.0 else (lam if matched > 0 else max(0.15, lam / 2))
            try:
                setattr(memory, "_reply_lambda", lam)
                setattr(memory, "_reply_predicted", None)
            except Exception:
                pass
    return lam


def _remember_prediction(memory: MatchMemory | None, ctx: "_Context", cols: list["_JointColumn"], prior: list[float], turn: object) -> None:
    if memory is None or not cols:
        return
    j = max(range(len(cols)), key=lambda k: prior[k])
    col = cols[j]
    try:
        setattr(memory, "_reply_predicted", {"turn": turn, "actions": [(a.species, a.kind, a.move) for a in col.actions],
                                             "actives": [species_key(m.species) for m in ctx.theirs.values()]})
    except Exception:
        pass


def _regret_matching(matrix: list[list[float]], *, iterations: int = 200) -> list[float]:
    """The column player's (their) average strategy from regret matching on the zero-sum matrix (our value); returns
    the column mixture to value rows against."""
    n, m = len(matrix), len(matrix[0])
    row_regret = [0.0] * n
    col_regret = [0.0] * m
    col_sum = [0.0] * m
    for _ in range(max(1, iterations)):
        x = _strategy(row_regret)
        y = _strategy(col_regret)
        row_payoff = [sum(matrix[i][j] * y[j] for j in range(m)) for i in range(n)]
        col_payoff = [-sum(matrix[i][j] * x[i] for i in range(n)) for j in range(m)]
        u_row = sum(x[i] * row_payoff[i] for i in range(n))
        u_col = sum(y[j] * col_payoff[j] for j in range(m))
        for i in range(n):
            row_regret[i] += row_payoff[i] - u_row
        for j in range(m):
            col_regret[j] += col_payoff[j] - u_col
            col_sum[j] += y[j]
    total = sum(col_sum) or 1.0
    return [c / total for c in col_sum]


def _strategy(regret: list[float]) -> list[float]:
    positive = [max(r, 0.0) for r in regret]
    total = sum(positive)
    if total <= 0:
        return [1.0 / len(regret)] * len(regret)
    return [p / total for p in positive]


class _JointColumn:
    __slots__ = ("name", "actions", "weight")

    def __init__(self, name: str, actions: list[_Actor], weight: float) -> None:
        self.name, self.actions, self.weight = name, actions, weight


class _Context:
    """Everything a cell needs, built once per turn."""

    def __init__(self, sheet: battle.TurnSheet, memory: MatchMemory | None, slots: list[dict], ours: dict[int, battle.Mon],
                 theirs: dict[int, battle.Mon], P: dict, klass: str, turn: object, obs_team: dict | None = None) -> None:
        self.sheet, self.memory, self.slots, self.ours, self.theirs, self.P, self.klass, self.turn = sheet, memory, slots, ours, theirs, P, klass, turn
        self.speed_rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
        self.by_slot = {e["slot"]: e for e in sheet.our_options}
        self.weights = battle.threat_weights(sheet, ours, theirs, P)
        self.fs = sheet.field
        self.fresh = set(getattr(memory, "fresh_active", None) or [])
        self.their_fresh = set(getattr(memory, "opp_fresh_active", None) or [])
        self.their_streak = dict(getattr(memory, "opp_protect_streak", None) or {})
        self.our_streak = {n: (memory.protect_streak(m.species, n) if memory is not None else 0) for n, m in ours.items()}
        self.obs_team = obs_team
        self.bench_prob: dict[str, float] = {}
        self.revealed = self._revealed()
        self.their_bench = self._their_bench()
        self.switch_rows = self._their_switch_rows()
        self.our_switch_rows = self._our_switch_rows()
        self.broken_rows = self._broken_rows()
        # what each of ours is worth: its HP and the KO bonus (battle.material), plus what it threatens them with, the
        # mirror of the KO credit on their side (losing a sweeper in a 2-for-1 is not a bargain)
        self.material = {n: battle.material(m, P) + P.get("ko_threat_w", 0.0) * max(battle._best_attack(self.by_slot.get(n, {}), P=P, threats=self.weights)[2], 0.0)
                         for n, m in ours.items()}

    # -- tables ------------------------------------------------------------------------------------------------------

    def _revealed(self) -> set[str]:
        """Our Pokémon the opponent has seen on the field this battle (its search models switches to those)."""
        seen = {species_key(m.species) for m in self.ours.values()}
        for fields in getattr(self.memory, "turns", None) or []:
            for s in fields.get("actives") or []:
                if s:
                    seen.add(species_key(s))
        for s in getattr(self.memory, "last_active", None) or []:
            seen.add(species_key(s))
        return seen

    def _their_bench(self) -> list[battle.Mon]:
        """Their cards that can still come in: not on the field, not fainted (the observation's team, or the sheet's copy
        of it: battle.rank_candidates passes no observation, and m4 T7's worst reply was a switch into the fainted
        Gholdengo), priced by the chance an unseen card was brought at all, at the HP and status the battle shows."""
        memory = self.memory
        if memory is None:
            return []
        active = {species_key(m.species) for m in self.theirs.values()}
        seen = list(getattr(memory, "opp_lineup_seen", None) or [])
        fainted = {species_key(s.get("species")) for s in (self.obs_team or {}).values() if isinstance(s, dict) and s.get("fainted")}
        team = getattr(self.sheet, "opp_team", None) or {}
        fainted |= {k for k, v in team.items() if v.get("fainted")}
        unseen_slots = max(0, 4 - len(seen))
        unseen_pool = [k for k in memory.opp_cards if k not in seen]
        self.bench_prob = {}
        out = []
        for key, card in memory.opp_cards.items():
            if key in active or key in fainted:
                continue
            if seen and key not in seen:
                if unseen_slots <= 0:
                    continue
                self.bench_prob[key] = unseen_slots / max(1, len(unseen_pool))  # could be in the back: priced by that chance
            else:
                self.bench_prob[key] = 1.0
            known = team.get(key) or {}
            mon = battle.build_mon({"species": card.get("species") or key, "current_hp_fraction": known.get("hp", 1.0) or 1.0,
                                    "status": known.get("status")}, card, side="theirs", position=None,
                                   item_lost=key in (getattr(memory, "opp_items_lost", None) or []))
            if mon is not None and not mon.fainted:
                out.append(mon)
        return out[:4]

    def _their_switch_rows(self) -> dict:
        """(bench key, our slot, option index) -> our damage row into that bench card as it comes in (boosts reset, its
        Intimidate on our attacker, its Ruin ability on the field, hazards on their side)."""
        rows: dict = {}
        hazards = battle._memory_hazards(self.memory, "theirs")
        for incoming in self.their_bench:
            chip, _ = battle.hazard_chip(incoming, battle.normalize_hazards(hazards, None))
            x = incoming if chip <= 0 else battle.Mon(**{**incoming.__dict__, "hp_fraction": max(0.01, 1.0 - chip)})
            for number, me in self.ours.items():
                attacker = battle.intimidated_foes(x, [me])[0]
                after = battle.entry_field(self.fs, add=[x])
                for opt in self.by_slot.get(number, {}).get("options") or []:
                    if opt.get("fails"):
                        continue
                    info = data.move_info(opt["move"]) or {}
                    prof = battle.hit_profile(attacker, {**info, "id": opt["move"]}, x, field=after,
                                              spread=battle.is_spread(info, foes=len(self.theirs), allies=len(self.ours) - 1))
                    if prof is None:
                        continue
                    rows[(species_key(incoming.species), number, opt["option"])] = {
                        "target": 0, "species": incoming.species, "side": "theirs", "hp_frac": round(x.hp_fraction, 3), **prof}
        return rows

    def _our_switch_rows(self) -> dict:
        """(our bench key, their position, move id) -> that foe's hit into our bench card as it comes in (its Intimidate
        on the foe, the post-switch field, hazards on our side), in % of the newcomer's HP after the hazard chip."""
        rows: dict = {}
        if self.memory is None:
            return rows
        offered = {}
        for s in self.slots:
            for o in s.get("options") or []:
                if o.get("type") == "switch" and o.get("species"):
                    offered.setdefault(species_key(o["species"]), (o["species"], s.get("slot", 0)))
        hazards = battle._memory_hazards(self.memory, "mine")
        for key, (species, number) in offered.items():
            incoming = battle.build_mon({"species": species, "current_hp_fraction": self.sheet.bench_hp.get(key, 1.0), "boosts": {}},
                                        self.memory.my_cards.get(key), side="mine", position=number)
            if incoming is None:
                continue
            chip, _ = battle.hazard_chip(incoming, battle.normalize_hazards(hazards, self.fs))
            x = incoming if chip <= 0 else battle.Mon(**{**incoming.__dict__, "hp_fraction": max(0.01, incoming.hp_fraction - chip)})
            x.tailwind = bool(self.fs and "tailwind" in self.fs.my_side)
            for pos, foe in self.theirs.items():
                attacker = battle.intimidated_foes(x, [foe])[0]
                after = battle.entry_field(self.fs, add=[attacker, x], drop=[foe, self.ours.get(number)])
                for mv in foe.moves:
                    if not data.is_damaging(mv):
                        continue
                    prof = battle.hit_profile(attacker, mv, x, field=after, spread=battle.is_spread(mv, foes=len(self.ours), allies=len(self.theirs) - 1))
                    if prof is None:
                        continue
                    rows[(key, pos, mv["id"])] = {**prof, "move": mv["id"], "into_slot": number, "species": x.species, "hp_frac": round(x.hp_fraction, 3),
                                                   "priority": mv.get("priority", 0) or 0}
            rows[(key, "mon")] = x
        return rows

    def _broken_rows(self) -> dict:
        """(our slot, option index, their position) -> the row once the foe's Multiscale / Shadow Shield or Focus Sash
        has been broken by an earlier hit this turn."""
        rows: dict = {}
        for pos, foe in self.theirs.items():
            guarded = foe.hp_fraction >= 0.999 and (data.to_id(foe.ability) in ("multiscale", "shadowshield") or data.to_id(foe.item) == "focussash")
            if not guarded:
                continue
            broken = battle.Mon(**{**foe.__dict__, "hp_fraction": 0.99, "item": None if data.to_id(foe.item) == "focussash" else foe.item})
            for number, me in self.ours.items():
                for opt in self.by_slot.get(number, {}).get("options") or []:
                    if opt.get("fails"):
                        continue
                    info = data.move_info(opt["move"]) or {}
                    prof = battle.hit_profile(me, {**info, "id": opt["move"]}, broken, field=self.fs,
                                              spread=battle.is_spread(info, foes=len(self.theirs), allies=len(self.ours) - 1))
                    if prof is None:
                        continue
                    # in % of the foe's CURRENT (full) HP: the broken Mon's 99% re-bases the row, undo that
                    scale = 0.99
                    lo, hi = prof["damage_pct_of_current_hp"]
                    rows[(number, opt["option"], pos)] = {**prof, "damage_pct_of_current_hp": [round(lo * scale, 1), round(hi * scale, 1)],
                                                          "target": pos, "species": foe.species, "side": "theirs", "hp_frac": round(foe.hp_fraction, 3)}
        return rows

    # -- helpers -----------------------------------------------------------------------------------------------------

    def post_rows(self, number: int, option_index: int, prio: int) -> dict:
        """(their position, move) -> the threat row into our slot AFTER it used this option: its stat drops, recoil /
        Life Orb and Glaive Rush applied (A3). Empty for a move without such effects. Cached per (slot, option)."""
        key = (number, option_index)
        cache = self.__dict__.setdefault("_post_cache", {})
        if key in cache:
            return cache[key]
        out: dict = {}
        me = self.ours.get(number)
        opt = self.move_option(number, option_index)
        if me is not None and opt is not None:
            move_id = opt.get("move") or ""
            has_effect = move_id in battle.POST_MOVE_STAGES or move_id == battle.GLAIVE_RUSH or move_id in battle.RECOIL_MOVES \
                or data.to_id(me.item) == "lifeorb"
            if has_effect:
                row = next((r for r in opt.get("targets") or [] if r.get("side") == "theirs"), None)
                foe = self.theirs.get(row["target"]) if row and row.get("target", 0) > 0 else None
                dealt = tuple(row["damage_pct_of_current_hp"]) if row else None
                cost = battle.self_cost_pct(me, move_id, dealt, foe)
                after = battle.post_move_mon(me, move_id, cost=cost)
                for h in battle.post_move_threat_hits(self.sheet, number, after, prio, self.theirs, self.fs, self.speed_rank,
                                                      doubled=(move_id == battle.GLAIVE_RUSH), ours_count=len(self.ours)):
                    out[(h["position"], h["move"])] = {**h, "into_slot": number}
        cache[key] = out
        return out

    def slot_template(self, number: int) -> dict | None:
        return next((s for s in self.slots if s.get("slot", 0) == number), None)

    def option_index(self, number: int, move_id: str) -> int | None:
        slot = self.slot_template(number)
        for i, o in enumerate((slot or {}).get("options") or []):
            if o.get("type") == "move" and data.to_id(o.get("move_id")) == move_id:
                return i
        return None

    def move_option(self, number: int, option_index: int) -> dict | None:
        return next((o for o in self.by_slot.get(number, {}).get("options") or [] if o["option"] == option_index), None)

    def actor_for(self, number: int, answer: dict, name: str = "") -> _Actor | None:
        """Our slot's action from a candidate payload {option, target}."""
        slot = self.slot_template(number)
        if not slot or number not in self.ours:
            return None
        options = slot.get("options") or []
        index = answer.get("option")
        if index is None or index >= len(options):
            return None
        raw = options[index]
        kind = raw.get("type")
        me = self.ours[number]
        if kind == "switch":
            species = raw.get("species") or ""
            preview = battle._preview_for(self.sheet, species, number)
            incoming = battle.build_mon({"species": species, "current_hp_fraction": self.sheet.bench_hp.get(species_key(species), 1.0)},
                                        (self.memory.my_cards.get(species_key(species)) if self.memory else None), side="mine", position=number)
            return _Actor("mine", number, "switch", priority=6, species=species, payload=answer, incoming=(incoming, preview), name=name)
        if kind == "pass":
            return _Actor("mine", number, "pass", priority=-9, payload=answer, name=name)
        move_id = data.to_id(raw.get("move_id"))
        info = data.move_info(move_id) or {}
        prio = info.get("priority", 0) or 0
        if data.to_id(me.ability) == "prankster" and info.get("category") == "Status":
            prio += 1
        if move_id in PROTECT_IDS:
            return _Actor("mine", number, "protect", move=move_id, priority=4, payload=answer, name=name)
        opt = self.move_option(number, index)
        if opt is not None and opt.get("targets") and data.is_damaging(info):
            if opt.get("fails"):
                return _Actor("mine", number, "pass", priority=-9, payload=answer, name=name)
            target = answer.get("target", 0) or 0
            rows: dict = {}
            ally: dict = {}
            for r in opt["targets"]:
                if r.get("side") == "mine":
                    if r.get("target", 0) == 0:
                        ally[r["species"]] = r
                    continue
                pos = r["target"] if r.get("target", 0) > 0 else self.pos_of(r["species"])
                if pos is None:
                    continue
                if target > 0 and pos != target:
                    continue
                rows[pos] = r
            if move_id == "fakeout" and species_key(me.species) in self.fresh:
                return _Actor("mine", number, "fakeout", move=move_id, priority=3, rows=rows, target=target, payload=answer, name=name)
            return _Actor("mine", number, "attack", move=move_id, priority=opt.get("priority", prio) or 0, rows=rows, target=target,
                          payload=answer, name=name, spread=bool(opt.get("spread")), ally_rows=ally)
        # a status move
        if move_id in SETUP_IDS or move_id in battle.SETUP_MOVES:
            return _Actor("mine", number, "setup", move=move_id, priority=prio, payload=answer, name=name)
        return _Actor("mine", number, "status", move=move_id, priority=prio, payload=answer, target=answer.get("target", 0), name=name)

    def pos_of(self, species: str) -> int | None:
        return next((pos for pos, m in self.theirs.items() if species_key(m.species) == species_key(species)), None)

    def best_hit_into(self, pos: int, slot: int) -> dict | None:
        """The foe's most dangerous known hit into our slot (expected damage points with the KO premium)."""
        best, best_v = None, -1.0
        for t in self.sheet.threats:
            if t["position"] != pos:
                continue
            for h in t["hits"]:
                if h["into_slot"] != slot or h["move"] == "fakeout":
                    continue
                v = battle._danger(battle._avg(h["damage_pct_of_current_hp"]), h.get("hit_chance", 1.0), battle.ko_probability(h, with_accuracy=True))
                if v > best_v:
                    best, best_v = h, v
        return best

    def their_rows(self, pos: int, move_id: str) -> dict[int, dict]:
        out = {}
        for t in self.sheet.threats:
            if t["position"] != pos:
                continue
            for h in t["hits"]:
                if h["move"] == move_id:
                    out[h["into_slot"]] = h
        return out

    def acts_before(self, a: _Actor, b: _Actor) -> bool:
        if a.priority != b.priority:
            return a.priority > b.priority
        ra = self.speed_rank.get((a.side, a.slot if a.side == "mine" else a.slot), 99)
        rb = self.speed_rank.get((b.side, b.slot), 99)
        if ra == rb:
            return a.side == "mine"  # the search bot's engine breaks ties in our favour; a coin flip in truth
        return ra < rb

    # -- our menu -------------------------------------------------------------------------------------------------

    def our_menu(self, candidates: list[dict]) -> list[_Actor]:
        rows: list["_JointRow"] = []
        seen: set = set()

        def add(name: str, a0: _Actor | None, a1: _Actor | None, payload: dict | None = None) -> None:
            if a0 is None and 0 in self.ours:
                return
            if a1 is None and 1 in self.ours:
                return
            acts = [a for a in (a0, a1) if a is not None]
            key = tuple((a.slot, a.kind, a.move, a.target, a.species) for a in acts)
            if key in seen or not acts:
                return
            seen.add(key)
            rows.append(_JointRow(name, acts, payload or {}))

        def from_payload(name: str, p0: dict | None, p1: dict | None) -> None:
            a0 = self.actor_for(0, p0, name) if p0 else None
            a1 = self.actor_for(1, p1, name) if p1 else None
            if self.sheet.alone and not self.sheet.stall_reasons and any(a is not None and a.kind == "protect" for a in (a0, a1)):
                return  # our last Pokémon with nothing to wait for: a Protect only delays the same position (Oct 7, match 1)
            add(name, a0, a1, {"name": name, "slot_0": p0, "slot_1": p1})

        for c in candidates:
            from_payload(c["name"], c.get("slot_0"), c.get("slot_1"))
        if len(self.ours) == 2 and all(n in self.ours for n in (0, 1)):
            best = {n: self.best_attack_answer(n) for n in (0, 1)}
            protect = {n: self.option_index(n, "protect") if self.option_index(n, "protect") is not None else self.option_index(n, "detect") for n in (0, 1)}
            if best[0] and best[1]:
                from_payload("both_attack", best[0], best[1])
            if protect[0] is not None and protect[1] is not None and all(self.in_ko_range(n) for n in (0, 1)):
                from_payload("double_protect", {"option": protect[0], "target": 0}, {"option": protect[1], "target": 0})
            for n in (0, 1):
                other = 1 - n
                for move_id, answer in self.status_answers(n):
                    if best[other]:
                        from_payload(f"{move_id}_and_attack", *((answer, best[other]) if n == 0 else (best[other], answer)))
            for pos in sorted(self.theirs):
                p0, p1 = self.best_single_answer(0, pos), self.best_single_answer(1, pos)
                if p0 and p1:
                    from_payload(f"double_{pos}", p0, p1)
            for n in (0, 1):
                other = 1 - n
                if protect[n] is not None and best[other]:
                    from_payload(f"protect_{n}", *(({"option": protect[n], "target": 0}, best[other]) if n == 0 else (best[other], {"option": protect[n], "target": 0})))
                    for pos in sorted(self.theirs):
                        aimed = self.best_single_answer(other, pos)
                        if aimed and aimed != best[other]:
                            from_payload(f"protect_{n}_into_{pos}", *(({"option": protect[n], "target": 0}, aimed) if n == 0 else (aimed, {"option": protect[n], "target": 0})))
                idx, species, preview, safe = battle.safe_switch(self.slot_template(n) or {}, self.theirs, self.memory, sheet=self.sheet) if self.memory else (None, None, {}, False)
                if idx is not None and best[other]:
                    from_payload(f"switch_{n}" + ("" if safe else "_risky"), *(({"option": idx, "target": 0}, best[other]) if n == 0 else (best[other], {"option": idx, "target": 0})))
                    spread_other = self.best_spread_answer(other)
                    if spread_other and spread_other != best[other]:
                        from_payload(f"switch_{n}_spread", *(({"option": idx, "target": 0}, spread_other) if n == 0 else (spread_other, {"option": idx, "target": 0})))
                if idx is not None and safe and best[other] and self.memory is not None:
                    # the best pivot by the preview score may be one the verdict calls unsafe: let the matrix price it
                    b_idx, b_species = battle.best_switch(self.slot_template(n) or {}, self.theirs, self.memory, sheet=self.sheet)
                    if b_idx is not None and b_idx != idx:
                        from_payload(f"switch_{n}_risky", *(({"option": b_idx, "target": 0}, best[other]) if n == 0 else (best[other], {"option": b_idx, "target": 0})))
                # speed control behind the partner's Protect
                for setup in ("tailwind", "trickroom", "icywind", "electroweb"):
                    s_idx = self.option_index(n, setup)
                    if s_idx is not None and protect[other] is not None:
                        pair = ({"option": s_idx, "target": 0}, {"option": protect[other], "target": 0})
                        from_payload(f"{setup}_behind_protect", *(pair if n == 0 else pair[::-1]))
                        break
            # split: slot 0 -> A, slot 1 -> B and the reverse; spread variants; Fake Out on each target; the partner of
            # the obvious-KO target; break Multiscale / Sash then finish
            positions = sorted(self.theirs)
            if len(positions) == 2:
                a, b = positions
                for first, second in ((a, b), (b, a)):
                    p0 = self.best_single_answer(0, first)
                    p1 = self.best_single_answer(1, second)
                    if p0 and p1:
                        from_payload("split", p0, p1)
            for n in (0, 1):
                other = 1 - n
                spread = self.best_spread_answer(n)
                if spread and best[other]:
                    from_payload("spread", *((spread, best[other]) if n == 0 else (best[other], spread)))
                fake = self.option_index(n, "fakeout")
                if fake is not None and species_key(self.ours[n].species) in self.fresh:
                    for pos in positions:
                        if self.flinchable_foe(pos) and best[other]:
                            from_payload(f"fakeout_{pos}", *(({"option": fake, "target": pos}, best[other]) if n == 0 else (best[other], {"option": fake, "target": pos})))
                for pos in positions:
                    if self.one_move_ko(n, pos) and len(positions) == 2:
                        partner_pos = next(p for p in positions if p != pos)
                        p_other = self.best_single_answer(other, partner_pos) or best[other]
                        p_n = self.best_single_answer(n, partner_pos)
                        if p_n and p_other:
                            from_payload("hit_the_partner", *((p_n, p_other) if n == 0 else (p_other, p_n)))
            for pos, foe in self.theirs.items():
                if not any(k[2] == pos for k in self.broken_rows):
                    continue
                for breaker, finisher in ((0, 1), (1, 0)):
                    b_ans = self.best_single_answer(breaker, pos, prefer_priority=True)
                    f_ans = self.best_single_answer(finisher, pos, broken=True)
                    if b_ans and f_ans:
                        from_payload("break_then_finish", *((b_ans, f_ans) if breaker == 0 else (f_ans, b_ans)))
        elif len(self.ours) == 1:
            n = next(iter(self.ours))
            lone = {"option": 0, "target": 0}
            protect = self.option_index(n, "protect")
            if protect is not None:
                from_payload(f"protect_{n}", *(({"option": protect, "target": 0}, lone) if n == 0 else (lone, {"option": protect, "target": 0})))
        cap = int(self.P.get("reply_rows", 16))
        return rows[:cap]

    def in_ko_range(self, number: int) -> bool:
        """Does a KO threaten this slot this turn (a single hit at >= 0.5, or the double-target)?"""
        for t in self.sheet.threats:
            for h in t["hits"]:
                if h["into_slot"] == number and h["ko"] != "no" and battle.ko_probability(h) >= 0.5:
                    return True
        return any(f["slot"] == number and f["ko_chance"] >= 0.5 for f in self.sheet.focus_threats)

    def best_attack_answer(self, number: int) -> dict | None:
        opt, row, _ = battle._best_attack(self.by_slot.get(number, {}), P=self.P, threats=self.weights)
        if not opt or not row:
            return None
        return {"option": opt["option"], "target": row.get("target", 0) if row.get("target", 0) > 0 else 0}

    def best_single_answer(self, number: int, pos: int, *, prefer_priority: bool = False, broken: bool = False) -> dict | None:
        best, best_v = None, -1.0
        for opt in self.by_slot.get(number, {}).get("options") or []:
            if opt.get("fails") or (opt.get("spread") and len(self.theirs) > 1):
                continue
            for r in opt.get("targets") or []:
                if r.get("side") != "theirs":
                    continue
                if not (r.get("target") == pos or (r.get("target", 0) == 0 and self.pos_of(r["species"]) == pos)):
                    continue
                row = self.broken_rows.get((number, opt["option"], pos), r) if broken else r
                v = battle._expected(row, opt.get("priority", 0) or 0, self.P, self.weights.get(species_key(r["species"]), 0.0))
                if prefer_priority:
                    v = v * (1.0 if (opt.get("priority", 0) or 0) > 0 else 0.3)
                if v > best_v:
                    best, best_v = {"option": opt["option"], "target": pos if r.get("target", 0) > 0 else 0}, v
        return best

    def status_answers(self, number: int) -> list[tuple[str, dict]]:
        """Every non-damaging move the slot can use that is worth something next to an attacking partner."""
        out = []
        slot = self.slot_template(number)
        me = self.ours.get(number)
        if not slot or me is None:
            return out
        for i, o in enumerate(slot.get("options") or []):
            if o.get("type") != "move":
                continue
            move_id = data.to_id(o.get("move_id"))
            info = data.move_info(move_id) or {}
            if data.is_damaging(info) or move_id in PROTECT_IDS or move_id == "fakeout":
                continue
            if move_id not in SETUP_IDS and move_id not in battle.SETUP_MOVES and move_id not in battle.ALLY_ONLY \
                    and battle.status_value(move_id, me, self.theirs, self.sheet, number, has_ally=len(self.ours) > 1)[0] <= 0:
                continue
            legal = o.get("targets") or []
            target = next((t for t in legal if t < 0), legal[0] if legal else 0) if move_id in battle.ALLY_ONLY else 0
            out.append((move_id, {"option": i, "target": target}))
        return out

    def best_spread_answer(self, number: int) -> dict | None:
        best, best_v = None, -1.0
        for opt in self.by_slot.get(number, {}).get("options") or []:
            if opt.get("fails") or not opt.get("spread"):
                continue
            one = battle._best_attack({"options": [opt]}, P=self.P, threats=self.weights)
            if one[0] and one[2] > best_v:
                best, best_v = {"option": opt["option"], "target": 0}, one[2]
        return best

    def one_move_ko(self, number: int, pos: int) -> bool:
        for opt in self.by_slot.get(number, {}).get("options") or []:
            if opt.get("fails"):
                continue
            for r in opt.get("targets") or []:
                if r.get("side") == "theirs" and (r.get("target") == pos or (r.get("target", 0) == 0 and self.pos_of(r["species"]) == pos)) \
                        and battle.ko_probability(r, with_accuracy=True) >= 0.5:
                    return True
        return False

    def flinchable_foe(self, pos: int) -> bool:
        foe = self.theirs.get(pos)
        if foe is None:
            return False
        if "ghost" in foe.types or data.to_id(foe.ability) in FLINCH_IMMUNE_ABILITIES or data.to_id(foe.item) == "covertcloak":
            return False
        if any(data.to_id(o.ability) in ("armortail", "dazzling", "queenlymajesty") for o in self.theirs.values()):
            return False
        return not (self.fs is not None and self.fs.terrain == "psychic" and foe.grounded)

    # -- their menu -------------------------------------------------------------------------------------------------

    def their_menu(self) -> list[_JointColumn]:
        cols: list[_JointColumn] = []
        seen: set = set()
        positions = sorted(self.theirs)

        def add(name: str, acts: list[_Actor]) -> None:
            key = tuple((a.slot, a.kind, a.move, a.target, a.species) for a in acts)
            if key in seen:
                return
            seen.add(key)
            cols.append(_JointColumn(name, acts, 1.0))

        attacks: dict[int, list[_Actor]] = {}
        for pos, foe in self.theirs.items():
            attacks[pos] = []
            for slot in sorted(self.ours):
                hit = self.best_hit_into(pos, slot)
                if hit is None:
                    continue
                info = data.move_info(hit["move"]) or {}
                spread = info.get("target") in battle.SPREAD_TARGETS
                rows = self.their_rows(pos, hit["move"]) if spread else {slot: hit}
                attacks[pos].append(_Actor("theirs", pos, "attack", move=hit["move"], priority=hit.get("priority", 0) or 0, rows=rows,
                                           target=(0 if spread else slot), species=foe.species, spread=spread))
            if not attacks[pos]:
                attacks[pos].append(_Actor("theirs", pos, "pass", species=foe.species, priority=-9))
        if len(positions) == 2:
            a, b = positions
            for ia in attacks[a]:
                for ib in attacks[b]:
                    label = f"{self.theirs[a].species} {ia.move or 'pass'}->{self._tgt(ia)} + {self.theirs[b].species} {ib.move or 'pass'}->{self._tgt(ib)}"
                    add(label, [ia, ib])
        else:
            for ia in attacks[positions[0]]:
                add(f"{self.theirs[positions[0]].species} {ia.move}->{self._tgt(ia)}", [ia])
        for pos, foe in self.theirs.items():
            partner_pos = next((p for p in positions if p != pos), None)
            partner_best = attacks[partner_pos][0] if partner_pos is not None and attacks.get(partner_pos) else None
            partner = self._most_dangerous(attacks.get(partner_pos, [])) if partner_pos is not None else None
            others = [partner] if partner is not None else []
            if any(m["id"] in battle.PROTECT_LIKE for m in foe.moves):
                add(f"{foe.species} Protect" + (f" + {self.theirs[partner_pos].species} {partner.move}->{self._tgt(partner)}" if partner else ""),
                    [_Actor("theirs", pos, "protect", move="protect", priority=4, species=foe.species)] + others)
            absorber = self.best_absorber(pos)
            if absorber is not None:
                add(f"{foe.species} -> {absorber.species}" + (f" + {self.theirs[partner_pos].species} {partner.move}->{self._tgt(partner)}" if partner else ""),
                    [_Actor("theirs", pos, "switch", priority=6, species=foe.species, incoming=absorber)] + others)
            if "fakeout" in {m["id"] for m in foe.moves} and species_key(foe.species) in self.their_fresh:
                stop = self.stoppable_slot()
                if stop is not None and battle._flinchable(self.ours[stop], self.ours, self.fs):
                    rows = self.their_rows(pos, "fakeout")
                    add(f"{foe.species} Fake Out->slot {stop}" + (f" + {self.theirs[partner_pos].species} {partner.move}->{self._tgt(partner)}" if partner else ""),
                        [_Actor("theirs", pos, "fakeout", move="fakeout", priority=3, rows={stop: rows[stop]} if stop in rows else {}, target=stop, species=foe.species)] + others)
            setup = next((m["id"] for m in foe.moves if m["id"] in SETUP_IDS and not data.is_damaging(m)), None)
            if setup:
                prio = 1 if data.to_id(foe.ability) == "prankster" else 0
                add(f"{foe.species} {setup}" + (f" + {self.theirs[partner_pos].species} {partner.move}->{self._tgt(partner)}" if partner else ""),
                    [_Actor("theirs", pos, "setup", move=setup, priority=prio, species=foe.species)] + others)
            spread_hit = next((h for t in self.sheet.threats if t["position"] == pos for h in t["hits"]
                               if (data.move_info(h["move"]) or {}).get("target") in battle.SPREAD_TARGETS), None)
            if spread_hit and partner is not None:
                rows = self.their_rows(pos, spread_hit["move"])
                add(f"{foe.species} {spread_hit['move']} (spread) + {self.theirs[partner_pos].species} {partner.move}->{self._tgt(partner)}",
                    [_Actor("theirs", pos, "attack", move=spread_hit["move"], priority=spread_hit.get("priority", 0) or 0, rows=rows, target=0,
                            species=foe.species, spread=True), partner])
        cap = int(self.P.get("reply_columns", 12))
        if len(cols) > cap:
            prior = self.prior(cols)
            order = sorted(range(len(cols)), key=lambda j: -prior[j])[:cap]
            cols = [cols[j] for j in sorted(order)]
        return cols

    def _tgt(self, a: _Actor) -> str:
        if a.kind == "pass":
            return "-"
        return "both" if a.spread else f"slot {a.target}"

    def _most_dangerous(self, acts: list[_Actor]) -> _Actor | None:
        best, best_v = None, -1.0
        for a in acts:
            v = sum(battle._danger(battle._avg(h["damage_pct_of_current_hp"]), h.get("hit_chance", 1.0), battle.ko_probability(h, with_accuracy=True))
                    for h in a.rows.values())
            if v > best_v:
                best, best_v = a, v
        return best

    def best_absorber(self, pos: int) -> battle.Mon | None:
        """Their bench card that takes the least from our best hits aimed at the foe at ``pos`` (what the search bot
        switches to once its Protect is spent)."""
        foe = self.theirs.get(pos)
        if foe is None or not self.their_bench:
            return None
        aimed = []
        for number in self.ours:
            ans = self.best_single_answer(number, pos)
            if ans:
                aimed.append((number, ans["option"]))
        if not aimed:
            return None
        best, best_taken = None, float("inf")
        for incoming in self.their_bench:
            taken = sum(min(battle._avg(self.switch_rows[(species_key(incoming.species), n, o)]["damage_pct_of_current_hp"]), 100.0)
                        for n, o in aimed if (species_key(incoming.species), n, o) in self.switch_rows)
            if taken < best_taken:
                best, best_taken = incoming, taken
        current = sum(min(battle._avg(r["damage_pct_of_current_hp"]), 100.0) for number, o in aimed
                      for r in (self.move_option(number, o) or {}).get("targets") or [] if r.get("side") == "theirs" and r.get("target") == pos)
        return best if best is not None and best_taken < current else None

    def stoppable_slot(self) -> int | None:
        """The slot whose action a fresh Fake Out would stop: the one with more to deny."""
        best, best_v = None, -1.0
        for n in self.ours:
            v = max(battle._best_attack(self.by_slot.get(n, {}), P=self.P, threats=self.weights)[2], 0.0)
            for setup in ("tailwind", "trickroom"):
                if self.option_index(n, setup) is not None:
                    v = max(v, 40.0)
            if v > best_v:
                best, best_v = n, v
        return best

    # -- the prior over columns ----------------------------------------------------------------------------------------

    def prior(self, cols: list[_JointColumn]) -> list[float]:
        weights = []
        for col in cols:
            w = 1.0
            for a in col.actions:
                w *= self._action_prob(a, col)
            weights.append(max(w, 1e-6))
        total = sum(weights) or 1.0
        return [w / total for w in weights]

    def _action_prob(self, a: _Actor, col: _JointColumn) -> float:
        foe = self.theirs.get(a.slot)
        if foe is None:
            return 1.0
        key = species_key(foe.species)
        streak = int(self.their_streak.get(key, 0))
        ko_one = max((battle.ko_probability(r, with_accuracy=True) for n in self.ours for opt in self.by_slot.get(n, {}).get("options") or []
                      if not opt.get("fails") for r in opt.get("targets") or []
                      if r.get("side") == "theirs" and (r.get("target") == a.slot or (r.get("target", 0) == 0 and self.pos_of(r["species"]) == a.slot))), default=0.0)
        facing = max((battle._avg(r["damage_pct_of_current_hp"]) for n in self.ours for opt in self.by_slot.get(n, {}).get("options") or []
                      if not opt.get("fails") for r in opt.get("targets") or []
                      if r.get("side") == "theirs" and (r.get("target") == a.slot or (r.get("target", 0) == 0 and self.pos_of(r["species"]) == a.slot))), default=0.0)
        ko_two = 1.0 if (ko_one < 0.5 and self._two_move_ko(a.slot)) else 0.0
        fresh_in = key in self.their_fresh
        p_protect = _p_protect(self.klass, streak=streak, ko_one=ko_one, ko_two=ko_two, facing=facing, fresh_in=fresh_in, turn=self.turn) \
            if any(m["id"] in battle.PROTECT_LIKE for m in foe.moves) else 0.0
        absorber = self.best_absorber(a.slot)
        has_protect = any(m["id"] in battle.PROTECT_LIKE for m in foe.moves)
        p_switch = _p_switch(self.klass, streak=streak, ko_one=ko_one, ko_two=ko_two, has_absorber=absorber is not None, has_protect=has_protect) \
            * (self.bench_prob.get(species_key(absorber.species), 1.0) if absorber is not None else 1.0)
        p_fake = _p_fakeout(self.klass, turn=self.turn) if ("fakeout" in {m["id"] for m in foe.moves} and fresh_in) else 0.0
        setup_move = next((m["id"] for m in foe.moves if m["id"] in SETUP_IDS and not data.is_damaging(m)), None)
        p_setup = _p_setup(self.klass, threatened=ko_one >= 0.3 or facing >= 30, turn=self.turn, move_id=setup_move) if setup_move else 0.0
        rest = max(0.05, 1.0 - min(0.95, p_protect + p_switch + p_fake + p_setup))
        if a.kind == "protect":
            return p_protect
        if a.kind == "switch":
            return p_switch
        if a.kind == "fakeout":
            return p_fake
        if a.kind == "setup":
            return p_setup
        if a.kind == "pass":
            return rest
        # an attack: the share of the rest by target, favouring the slot it can KO (the search bot KOs the slot that
        # cannot escape in its model) and the slot the partner is also aiming at (both on one target 4/6)
        targets = [h for h in (a.rows or {}).values()]
        ko_here = max((battle.ko_probability(h, with_accuracy=True) for h in targets), default=0.0)
        share = 0.5 + 0.5 * ko_here
        if a.spread:
            share = 0.6
        partner = next((b for b in col.actions if b is not a), None)
        if partner is not None and partner.kind == "attack" and not partner.spread and not a.spread and partner.target == a.target:
            share *= 1.0 + SEEDS["both_attack_same_target"]  # both on one target 10/16 turns
        return rest * share

    def _two_move_ko(self, pos: int) -> bool:
        parts = []
        for n in self.ours:
            ans = self.best_single_answer(n, pos)
            if not ans:
                continue
            opt = self.move_option(n, ans["option"])
            row = next((r for r in (opt or {}).get("targets") or [] if r.get("target") == pos), None)
            if row:
                parts.append((float(row.get("hit_chance", 1.0)), row))
        return len(parts) == 2 and battle._outcome(parts, self.theirs.get(pos))[1] >= 0.5

    def protect_column(self, cols: list[_JointColumn], pos: int | None = None) -> _JointColumn | None:
        hit = next((c for c in cols if any(a.kind == "protect" and a.slot == pos for a in c.actions)), None)
        return hit or next((c for c in cols if any(a.kind == "protect" for a in c.actions)), None)

    def switch_column(self, cols: list[_JointColumn], pos: int | None = None) -> _JointColumn | None:
        hit = next((c for c in cols if any(a.kind == "switch" and a.slot == pos for a in c.actions)), None)
        return hit or next((c for c in cols if any(a.kind == "switch" for a in c.actions)), None)

    def main_target(self, row: "_JointRow") -> int | None:
        """The foe our row aims the most damage at."""
        totals: dict[int, float] = {}
        for a in row.actions:
            for pos, r in (a.rows or {}).items():
                totals[pos] = totals.get(pos, 0.0) + min(battle._avg(r["damage_pct_of_current_hp"]), 100.0)
        return max(totals, key=totals.get) if totals else None

    def prediction_matched(self, predicted: dict) -> float | None:
        """Did the opponent do what the top-prior column said last turn? Read from the memory's protocol-log tables."""
        memory = self.memory
        if memory is None:
            return None
        last = getattr(memory, "opp_last_move", None) or {}
        actives_now = [species_key(m.species) for m in self.theirs.values()]
        checks = []
        for species, kind, move in predicted.get("actions", []):
            key = species_key(species)
            used = str(last.get(key) or "")
            if kind == "switch":
                checks.append(key not in actives_now)
            elif kind == "protect":
                checks.append(used in battle.PROTECT_LIKE)
            elif kind == "fakeout":
                checks.append(used == "fakeout")
            elif kind == "setup":
                checks.append(used == move)
            elif kind == "attack":
                # the log gives the move, not its target: an attack is matched when the foe attacked at all
                info = data.move_info(used) or {}
                checks.append(bool(used) and key in actives_now and data.is_damaging(info) and used not in battle.PROTECT_LIKE)
        if not checks:
            return None
        return sum(1.0 for c in checks if c) / len(checks)

    # -- the cell ----------------------------------------------------------------------------------------------------

    def cell(self, row: "_JointRow", col: _JointColumn) -> float:
        P = self.P
        actors = list(row.actions) + list(col.actions)
        order = sorted(actors, key=lambda a: (-a.priority, self.speed_rank.get((a.side, a.slot), 99), 0 if a.side == "mine" else 1))
        acts: dict[tuple[str, int], float] = {}        # chance each actor gets its action off
        protected: dict[tuple[str, int], float] = {}   # chance a Pokémon is behind a working Protect
        gone: dict[tuple[str, int], float] = {}        # chance a Pokémon has left (switched) before a hit
        flinched: dict[tuple[str, int], float] = {}
        incoming: dict[tuple[str, int], tuple] = {}    # the switch-in that replaced it
        parts: dict[tuple[str, int], list] = {}        # hits landing on each Pokémon, (land prob, row)
        hit_first: dict[tuple[str, int], float] = {}   # chance a guarded foe took an earlier hit this turn
        boosted: dict[tuple[str, int], float] = {}     # chance a slot's attack carries a Helping Hand
        hh_boost = getattr(data, "helping_hand_boost", None)
        moved: dict[tuple[str, int], tuple[float, dict]] = {}  # our slot -> (chance it has acted, its post-move rows)
        tempo = 0.0
        for a in order:
            me_key = (a.side, a.slot)
            p_alive = self._alive_before(me_key, parts, a, order)
            p_act = p_alive * (1.0 - flinched.get(me_key, 0.0))
            if a.side == "mine" and a.kind != "switch":
                p_act *= self.sheet.wake.get(a.slot, 1.0)  # asleep: its move (and its Protect) happens only if it wakes
            if a.kind == "pass":
                acts[me_key] = 0.0
                continue
            acts[me_key] = p_act
            if a.side == "mine" and a.kind == "attack":
                moved[me_key] = (p_act, self.post_rows(a.slot, a.payload.get("option"), a.priority))
                if a.move in SETUP_IDS:  # Icy Wind / Electroweb / Snarl: the drop is worth its flips, on top of the damage
                    tempo += SETUP_IDS[a.move] * p_act * (self._speed_control_gain(a.move) if a.move != "snarl" else 0.5)
            if a.kind == "protect":
                streak = self.our_streak.get(a.slot, 0) if a.side == "mine" else int(self.their_streak.get(species_key(a.species), 0))
                protected[me_key] = p_act * (1.0 / 3 ** streak)
                if a.side == "mine":
                    tempo -= P.get("reply_protect_tempo", 8.0)  # the turn it gives away (their free switch, setup, a spent Protect)
                    if self.sheet.alone and not self.sheet.stall_reasons:
                        tempo -= P.get("reply_stall_w", 30.0)  # our last Pokémon with nothing to wait for: a Protect only delays
                continue
            if a.kind == "switch":
                gone[me_key] = p_act
                incoming[me_key] = a.incoming
                if a.side == "theirs":
                    tempo -= P.get("reply_free_switch_w", 8.0) * p_act  # a free switch is tempo for them
                continue
            if a.kind == "setup":
                value = SETUP_IDS.get(a.move, 20)
                if a.side == "mine":
                    tempo += value * p_act * self._speed_control_gain(a.move)
                else:
                    tempo -= value * p_act * (1.0 if a.move not in ("tailwind", "trickroom") else self._their_speed_control_gain(a.move))
                continue
            if a.kind == "status":
                if a.move in ("helpinghand", "coaching") and a.side == "mine":
                    boosted[("mine", 1 - a.slot)] = p_act  # the partner's attack this turn is x1.5
                elif a.side == "mine":
                    blocked = protected.get(("theirs", a.target), 0.0) if isinstance(a.target, int) and a.target in self.theirs else 0.0
                    tempo += battle.status_value(a.move, self.ours[a.slot], self.theirs, self.sheet, a.slot, has_ally=len(self.ours) > 1)[0] \
                        * p_act * 0.5 * (1.0 - blocked)  # a Taunt or a Spore into a working Protect does nothing
                continue
            # attacks and Fake Outs: each row lands on its target unless it Protected (and the Protect worked) or left
            for tgt_slot, hit in a.rows.items():
                tgt_side = "theirs" if a.side == "mine" else "mine"
                tgt_key = (tgt_side, tgt_slot)
                p_land = p_act * float(hit.get("hit_chance", 1.0)) * (1.0 - protected.get(tgt_key, 0.0))
                if gone.get(tgt_key, 0.0) > 0 and incoming.get(tgt_key):
                    p_in = p_land * gone[tgt_key]
                    p_land *= 1.0 - gone[tgt_key]
                    sub = self._row_into_switch_in(a, tgt_key, incoming[tgt_key])
                    if sub is not None and p_in > 0:
                        parts.setdefault(("in",) + tgt_key, []).append((p_in, sub))
                if p_land <= 0:
                    continue
                row_used = hit
                if a.side == "theirs" and tgt_key in moved and moved[tgt_key][0] > 0:
                    p_moved, post = moved[tgt_key]
                    later = post.get((a.slot, hit["move"]))
                    if later is not None:
                        # the slot already used a move with a drawback: its own stat drops / recoil / Glaive Rush meet this hit
                        lo = (1 - p_moved) * hit["damage_pct_of_current_hp"][0] + p_moved * later["damage_pct_of_current_hp"][0]
                        hi = (1 - p_moved) * hit["damage_pct_of_current_hp"][1] + p_moved * later["damage_pct_of_current_hp"][1]
                        row_used = {**hit, "damage_pct_of_current_hp": [round(lo, 1), round(hi, 1)], "ko": battle._ko_label(lo, hi)}
                        for k in ("per_hit", "first_hit", "hits"):
                            row_used.pop(k, None)
                if a.side == "theirs" and hit["move"] in battle.FRACTION_MOVES:
                    # Ruination takes half of what is LEFT: scale by the HP the earlier hits leave, and it never finishes
                    so_far = battle._outcome(parts.get(tgt_key, []))[0]
                    share = 0.5 * max(0.0, 100.0 - so_far)
                    row_used = {**hit, "damage_pct_of_current_hp": [round(min(share, 99.0), 1), round(min(share, 99.0), 1)], "ko": "no"}
                    for k in ("per_hit", "first_hit", "hits"):
                        row_used.pop(k, None)
                if a.side == "mine" and (a.slot, a.payload.get("option"), tgt_slot) in self.broken_rows:
                    p_prev = hit_first.get(tgt_key, 0.0)
                    if p_prev > 0:
                        broken = self.broken_rows[(a.slot, a.payload["option"], tgt_slot)]
                        row_used = {**hit, "damage_pct_of_current_hp": [(1 - p_prev) * hit["damage_pct_of_current_hp"][0] + p_prev * broken["damage_pct_of_current_hp"][0],
                                                                          (1 - p_prev) * hit["damage_pct_of_current_hp"][1] + p_prev * broken["damage_pct_of_current_hp"][1]]}
                        for k in ("per_hit", "first_hit", "hits"):
                            row_used.pop(k, None)
                if a.side == "mine" and boosted.get(me_key, 0.0) > 0 and a.kind == "attack":
                    factor = 1.0 + 0.5 * boosted[me_key] * (float(hh_boost("helpinghand", a.move)) if callable(hh_boost) else 1.0)
                    lo, hi = row_used["damage_pct_of_current_hp"]
                    row_used = {**row_used, "damage_pct_of_current_hp": [round(lo * factor, 1), round(hi * factor, 1)]}
                    for k in ("per_hit", "first_hit", "hits"):
                        row_used.pop(k, None)
                parts.setdefault(tgt_key, []).append((p_land, row_used))
                hit_first[tgt_key] = 1.0 - (1.0 - hit_first.get(tgt_key, 0.0)) * (1.0 - p_land)
                if a.kind == "fakeout" and tgt_side == "mine" and battle._flinchable(self.ours[tgt_slot], self.ours, self.fs):
                    flinched[tgt_key] = 1.0 - (1.0 - flinched.get(tgt_key, 0.0)) * (1.0 - p_land)
                if a.kind == "fakeout" and tgt_side == "theirs" and self.flinchable_foe(tgt_slot):
                    flinched[tgt_key] = 1.0 - (1.0 - flinched.get(tgt_key, 0.0)) * (1.0 - p_land)
            for ally_species, r in (a.ally_rows or {}).items():
                ally = next(((("mine", n)) for n, m in self.ours.items() if m.species == ally_species), None)
                if not ally:
                    continue
                p_hit = p_act * (1.0 - protected.get(ally, 0.0))  # its Protect blocks our own Earthquake too
                p_in = p_hit * gone.get(ally, 0.0)
                if p_hit - p_in > 0:
                    parts.setdefault(ally, []).append((p_hit - p_in, r))
                if p_in > 0:
                    parts.setdefault(("in",) + ally, []).append((p_in, r))  # the newcomer takes it (same share, an estimate)
            if a.side == "mine":
                tempo += P["priority_w"] * a.priority * p_act
        # value: damage into each foe, KOs by the threat removed; our losses as material
        value = 0.0
        for pos, foe in self.theirs.items():
            dmg, p_ko = battle._outcome(parts.get(("theirs", pos), []), foe)
            if P.get("hp_units", 0):
                dmg *= foe.hp_fraction
            p_ko *= 1.0 - gone.get(("theirs", pos), 0.0)
            value += dmg * (1.0 - gone.get(("theirs", pos), 0.0)) + p_ko * (P["ko_bonus_guaranteed"] + P.get("ko_threat_w", 0.0) * self.weights.get(species_key(foe.species), 0.0))
            inc = parts.get(("in", "theirs", pos), [])
            if inc:
                dmg_in, ko_in = battle._outcome(inc)
                value += dmg_in * (inc[0][1].get("hp_frac", 1.0) if P.get("hp_units", 0) else 1.0) + ko_in * P["ko_bonus_guaranteed"]
        for n, me in self.ours.items():
            key = ("mine", n)
            dmg, p_ko = battle._outcome(parts.get(key, []), me)
            p_ko *= 1.0 - gone.get(key, 0.0)
            value -= P.get("reply_loss_w", 1.0) * p_ko * self.material[n]
            value -= P.get("reply_chip_w", 0.35) * dmg * me.hp_fraction * (1.0 - gone.get(key, 0.0))
            if key in incoming and incoming[key]:
                mon, preview = incoming[key]
                hits_in = parts.get(("in",) + key, [])
                dmg_in, ko_in = battle._outcome(hits_in)
                hp_in = float(mon.hp_fraction if mon else 1.0)
                value -= P.get("reply_loss_w", 1.0) * ko_in * (100.0 * hp_in + P["ko_bonus_guaranteed"])
                value -= P.get("reply_chip_w", 0.35) * dmg_in * hp_in
                if mon is not None and species_key(mon.species) not in self.revealed:
                    value += P.get("reply_pivot_w", 6.0) * gone[key]  # an unrevealed newcomer: its search cannot aim at it (a seen one it can)
        # choice lock into a dead move
        for a in row.actions:
            if a.side == "mine" and a.kind == "attack" and self.sheet.choice_locks.get(a.slot, {}).get(a.move):
                value -= P.get("reply_dead_lock_w", 15.0) * acts.get(("mine", a.slot), 1.0)
        return value + tempo + self._next_turn(parts, gone, incoming, order)

    def _next_turn(self, parts: dict, gone: dict, incoming: dict, order: list[_Actor]) -> float:
        """One ply ahead, discounted: what is left after this turn and who threatens whom next turn. For each Pokémon
        still standing (its chance and its remaining HP from this turn's hits), the KO each side can take next turn
        with its best hit at that HP, in speed order (a foe we take out first does not swing), weighted by the
        threat removed / the material lost. All-negative turns then rank by what they leave (L3 T1 type), and a pivot
        that sets up a KO next turn (L4 T2: Landorus's Earthquake into a Balloon-less Gholdengo) gets its credit."""
        P = self.P
        w = float(P.get("reply_next_w", 0.4))
        if w <= 0:
            return 0.0

        def state(key: tuple[str, int]) -> tuple[float, float]:
            """(chance it is still standing, remaining HP as a fraction of its HP at the start of the turn)."""
            dmg, p_ko = battle._outcome(parts.get(key, []), self._mon_at(key))
            return (1.0 - p_ko), max(0.05, 1.0 - dmg / 100.0)

        protected_now = {a.slot for a in order if a.side == "mine" and a.kind == "protect"}
        their_protected_now = {a.slot for a in order if a.side == "theirs" and a.kind == "protect"}

        def protect_answer(side: str, slot: int, mon: battle.Mon) -> float:
            """How much of a KO threat next turn a Protect answers: a fresh one all but the tempo (0.6), a spent one 1 in 3."""
            if not any(m["id"] in battle.PROTECT_LIKE for m in mon.moves):
                return 0.0
            spent = slot in (protected_now if side == "mine" else their_protected_now)
            return 0.6 / 3 if spent else 0.6

        theirs_alive: dict[int, tuple[float, float, battle.Mon]] = {}
        for pos, foe in self.theirs.items():
            left = 1.0 - gone.get(("theirs", pos), 0.0)
            alive, rem = state(("theirs", pos))
            if left > 0:
                theirs_alive[pos] = (left * alive, rem, foe)
        ours_alive: dict[int, tuple[float, float, battle.Mon, float]] = {}
        for n, me in self.ours.items():
            key = ("mine", n)
            left = 1.0 - gone.get(key, 0.0)
            alive, rem = state(key)
            mat = self.material[n]
            if left > 0:
                ours_alive[n] = (left * alive, rem, me, mat)
            if key in incoming and incoming[key]:
                mon = incoming[key][0] if isinstance(incoming[key], tuple) else incoming[key]
                if isinstance(mon, battle.Mon):
                    alive_in, rem_in = state(("in",) + key)
                    ours_alive[n] = (gone[key] * alive_in, rem_in * mon.hp_fraction / max(me.hp_fraction, 0.05), mon,
                                     battle.material(mon, P))
        total = 0.0
        for pos, (q, rem_f, foe) in theirs_alive.items():
            # our best KO on it next turn, over our standing Pokémon, rescaled to what it has left
            best_p, best_first = 0.0, False
            for n, (p, _, me, _) in ours_alive.items():
                ans = self.best_single_answer(n, pos) if me is self.ours.get(n) else None
                if not ans:
                    continue
                opt = self.move_option(n, ans["option"])
                row = next((r for r in (opt or {}).get("targets") or [] if r.get("side") == "theirs"
                            and (r.get("target") == pos or (r.get("target", 0) == 0 and self.pos_of(r["species"]) == pos))), None)
                if row is None:
                    continue
                lo, hi = row["damage_pct_of_current_hp"]
                p_ko = battle.linear_ko(lo / rem_f, hi / rem_f) * float(row.get("hit_chance", 1.0)) * p
                if p_ko > best_p:
                    best_p = p_ko
                    best_first = (opt.get("priority", 0) or 0) > 0 or self.speed_rank.get(("mine", n), 99) < self.speed_rank.get(("theirs", pos), 99)
            if best_p > 0:
                total += w * q * best_p * (1.0 - protect_answer("theirs", pos, foe)) \
                    * (P["ko_bonus_guaranteed"] + P.get("ko_threat_w", 0.0) * self.weights.get(species_key(foe.species), 0.0))
            # its best KO on one of ours next turn, unless we take it out first
            for n, (p, rem_s, me, mat) in ours_alive.items():
                hit = self.best_hit_into(pos, n) if me is self.ours.get(n) else None
                if hit is None:
                    continue
                lo, hi = hit["damage_pct_of_current_hp"]
                p_ko = battle.linear_ko(lo / rem_s, hi / rem_s) * float(hit.get("hit_chance", 1.0))
                if p_ko <= 0:
                    continue
                removed_first = best_p if best_first else 0.0
                total -= w * q * p * p_ko * (1.0 - removed_first) * (1.0 - protect_answer("mine", n, me)) * mat
        return total

    def _alive_before(self, key: tuple[str, int], parts: dict, actor: _Actor, order: list[_Actor]) -> float:
        """Chance this Pokémon is still standing when its turn comes: the hits already recorded on it."""
        if key[0] == "mine" and key[1] not in self.ours:
            return 0.0
        hits = parts.get(key, [])
        if not hits:
            return 1.0
        _, p_ko = battle._outcome(hits, self._mon_at(key))
        return 1.0 - p_ko

    def _mon_at(self, key: tuple) -> battle.Mon | None:
        """The Pokémon a parts key stands for (ours or theirs on the field; None for a newcomer)."""
        if len(key) != 2:
            return None
        return (self.ours if key[0] == "mine" else self.theirs).get(key[1])

    def _row_into_switch_in(self, a: _Actor, tgt_key: tuple[str, int], incoming: object) -> dict | None:
        """The row of this hit into whatever replaced its target: our move into their switch-in, or their move into ours."""
        if a.side == "mine":
            mon = incoming if isinstance(incoming, battle.Mon) else None
            if mon is None:
                return None
            return self.switch_rows.get((species_key(mon.species), a.slot, a.payload.get("option")))
        mon = incoming[0] if isinstance(incoming, tuple) else incoming
        if not isinstance(mon, battle.Mon):
            return None
        return self.our_switch_rows.get((species_key(mon.species), a.slot, a.move))

    TURNS_LEFT = {"tailwind": 3, "trickroom": 4, "icywind": 3, "electroweb": 3}  # turns the effect still covers after this one

    def _speed_control_gain(self, move: str) -> float:
        """How much a Tailwind / Trick Room / Icy Wind of ours changes who moves first: the share of (our attacker,
        their foe) pairs it flips, times the turns it still covers after this one (over a three-turn yardstick)."""
        pairs = [(m, f) for m in self.ours.values() for f in self.theirs.values()]
        if not pairs:
            return 0.0
        tr = self.fs.trick_room if self.fs else False
        flips = 0
        for m, f in pairs:
            before = (m.stat("spe") > f.stat("spe")) != tr
            if move == "tailwind":
                after = (m.stat("spe") * 2 > f.stat("spe")) != tr
            elif move == "trickroom":
                after = not before
            else:  # Icy Wind / Electroweb: -1 Speed on the foes
                after = (m.stat("spe") > f.stat("spe") * 2 / 3) != tr
            flips += int(after and not before) - int(before and not after)
        if flips <= 0:
            return 0.1
        return min(1.5, (0.4 + flips / len(pairs)) * self.TURNS_LEFT.get(move, 3) / 3)

    def _their_speed_control_gain(self, move: str) -> float:
        pairs = [(m, f) for m in self.ours.values() for f in self.theirs.values()]
        if not pairs:
            return 0.0
        tr = self.fs.trick_room if self.fs else False
        flips = 0
        for m, f in pairs:
            before = (f.stat("spe") > m.stat("spe")) != tr
            after = ((f.stat("spe") * 2 > m.stat("spe")) != tr) if move == "tailwind" else (not before)
            flips += int(after and not before)
        return min(1.5, 0.3 + flips / len(pairs)) if flips else 0.1

    def describe(self, row: "_JointRow", col: _JointColumn) -> str:
        ours = "; ".join(self._actor_text(a) for a in row.actions)
        return f"{ours} | worst reply: {col.name}"

    def _actor_text(self, a: _Actor) -> str:
        if a.kind == "switch":
            return f"slot {a.slot} -> {a.species}"
        if a.kind == "protect":
            return f"slot {a.slot} Protect"
        if a.kind in ("setup", "status"):
            return f"slot {a.slot} {a.move}"
        if a.kind == "pass":
            return f"slot {a.slot} passes"
        if a.spread:
            tgt = "both"
        elif a.target in self.theirs:
            tgt = self.theirs[a.target].species
        else:
            tgt = ", ".join(self.theirs[p].species for p in a.rows if p in self.theirs) or "?"
        return f"slot {a.slot} {a.move} -> {tgt}"


class _JointRow:
    __slots__ = ("name", "actions", "payload")

    def __init__(self, name: str, actions: list[_Actor], payload: dict) -> None:
        self.name, self.actions, self.payload = name, actions, payload
        if "name" not in self.payload:
            self.payload["name"] = name
        for a in actions:
            self.payload[f"slot_{a.slot}"] = a.payload
        self.payload.setdefault("why", name)
