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
    "fresh_protect_ko_switch": 0.11,     # 1/16 like-for-like: a fresh Protect, our one-move KO, a bench (the Oct 8 verification of m4 T2; 8/35 mixed in spent Protects)
    "fresh_protect_ko_covered": 0.25,    # 0/2: of 7/9 fresh Protects against our one-move KO, both misses had a reason: it could KO first, or its partner's fresh Fake Out covered it (the da9c168e post-mortem, section 4; T6 Lucario attacked into our double priority)
    "sleep_unthreatened": 0.6,           # a sleep user with an awake target uses its sleep move (start value; Oct 8 evening loss: 6 of 7 turns)
    "sleep_threatened": 0.4,             # the same while our one-move KO threatens it
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


def _p_protect(klass: str, *, streak: int, ko_one: float, ko_two: float, facing: float, fresh_in: bool, turn: int | None,
               covered: bool = False) -> float:
    """Chance this foe Protects this turn. Search bot: the pooled tally's seeds (SEEDS): never twice in a row, a fresh
    Protect against our one-move KO about a coin flip, but a quarter when it is ``covered`` (it can KO one of ours before
    that slot moves, or its partner's fresh Fake Out covers it), against a double-only KO 0.4, a lead on turn 1 and a
    fresh switch-in on its first turn a little above that, a third when it only takes >= 30%. Generic LLM: Protects in KO
    range about half the time, even again."""
    if klass == "random":
        return 0.15
    if klass == "llm":
        base = 0.5 if ko_one >= 0.5 else (0.3 if ko_two >= 0.5 else 0.1)
        return base * (0.5 if streak else 1.0)
    if streak:
        return SEEDS["protect_after_protect"]
    if ko_one >= 0.5:
        if covered:
            return SEEDS["fresh_protect_ko_covered"]
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
        return SEEDS["fresh_protect_ko_switch"]  # 1/16 like-for-like with a fresh Protect (8/35 counted spent Protects too)
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


def _p_sleep(klass: str, *, threatened: bool, used: int = 0, chances: int = 0) -> float:
    """Chance a sleep user with an awake, sleepable target uses its sleep move this turn: the start value (0.6, 0.4
    while our one-move KO threatens it) updated by this game's own count (``used`` of ``chances`` turns it moved), two
    pseudo-counts at the start value. Part 3: the Oct 8 evening loss's Amoonguss used Spore on 6 of its 7 turns."""
    base = 0.25 if klass == "random" else SEEDS["sleep_threatened" if threatened else "sleep_unthreatened"]
    return (used + 2.0 * base) / (chances + 2.0)


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
    # one draw per TURN from the match's secret seed (a draw rebuilt from the bare seed repeated the same u every turn of
    # a match: m2_t1 failed at every seed with u >= 0.623, review 5714330)
    rng = rng or random.Random(seed * 1_000_003 + (turn if isinstance(turn, int) else 0))
    ctx = _Context(sheet, memory, slots, ours, theirs, P, klass, turn, (obs or {}).get("opponent_team"))
    rows = ctx.our_menu(candidates or [])
    cols = ctx.their_menu()
    if not rows or not cols:
        return list(candidates or [])
    matrix = [[ctx.cell(r, c) for c in cols] for r in rows]
    prior, tally = _blended_prior(ctx, cols, matrix, P)
    lam = _lambda(memory, ctx, cols, prior)
    # a column is plausible when either prior says so: an overconfident read (0.93-0.97 on one reply) must not shrink the
    # worst case to one column (review 5714330: 7 of 64 positions kept a single column at reply_search_qr 1)
    floor_cols = [j for j in range(len(cols)) if max(prior[j], tally[j]) >= float(P.get("reply_floor_min_prior", 0.08))] or list(range(len(cols)))
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
    tied = [t for t in kept if t[0] >= top_value - eps and t[1] >= kept[0][1]]  # only a row at least as safe as the top one ties
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


def _blended_prior(ctx: "_Context", cols: list["_JointColumn"], matrix: list[list[float]], P: dict) -> tuple[list[float], list[float]]:
    """(the prior over their columns, the class tally it starts from): for the search class, ``reply_search_qr`` of its
    own read of the turn (``_search_reply``) and the rest the tally."""
    prior = ctx.prior(cols)
    tally = list(prior)
    kappa = float(P.get("reply_search_qr", 0.0))
    if kappa > 0 and ctx.klass == "search":
        searched = _search_reply(matrix, prior, P)
        # a one-turn search sees a sleep move as one lost action, our cells as the turns it takes: such a column keeps at
        # least the class's own rate for it (the Oct 8 evening loss: Spore on 6 of 7 turns, a read of 0.05 on turn 1)
        searched = [max(s, t) if any(a.kind == "sleep" for a in c.actions) else s for s, t, c in zip(searched, prior, cols)]
        # the mirror: a foe facing our KO that can KO one of ours first (or whose partner's fresh Fake Out covers it) did not
        # Protect in either such case (the da9c168e post-mortem), but the read prices our likely KO as if it would wait
        # (da9c168e T6 Lucario 0.64, d9405f02 T6 Dragonite 0.73): such a Protect column keeps at most the class's own rate
        capped = [j for j, c in enumerate(cols) if any(a.kind == "protect" and ctx.covered_protect(a.slot) for a in c.actions)]
        excess = sum(max(0.0, searched[j] - prior[j]) for j in capped)
        others = [j for j in range(len(cols)) if j not in capped]
        mass = sum(searched[j] for j in others)
        if excess > 0 and mass > 0:  # the excess goes to the other replies by their read, so a capped column stays capped
            searched = [min(v, prior[j]) if j in capped else v * (1.0 + excess / mass) for j, v in enumerate(searched)]
        total = sum(searched) or 1.0
        searched = [s / total for s in searched]
        prior = [(1.0 - kappa) * a + kappa * b for a, b in zip(prior, searched)]
    return prior, tally


def _softmax(values: list[float], beta: float) -> list[float]:
    top = max(values)
    weights = [math.exp(beta * (v - top)) for v in values]
    total = sum(weights) or 1.0
    return [w / total for w in weights]


def _search_reply(matrix: list[list[float]], prior: list[float], P: dict) -> list[float]:
    """The search class's own read of the turn, the way such a bot decides: it expects our rows by a softmax over their
    value against its current mix (``reply_qr_beta``), scores each of its replies by 0.7 of our expected value plus 0.3
    of our best likely row (it minimises ours: robust), and moves its mix halfway to a softmax at twice that sharpness;
    a few rounds from the class prior. It is a model of a search, not of a table of past choices: the Oct 7 search bot
    Protected the threatened Sylveon in every version (m2 T3), where the pooled tally says 0.40."""
    beta = float(P.get("reply_qr_beta", 0.06))
    n, m = len(matrix), len(matrix[0])
    mix = list(prior)
    for _ in range(int(P.get("reply_qr_rounds", 4))):
        ours = _softmax([sum(matrix[i][j] * mix[j] for j in range(m)) for i in range(n)], beta)
        ranked = sorted(range(n), key=lambda i: -ours[i])
        likely, mass = [], 0.0
        for i in ranked:
            likely.append(i)
            mass += ours[i]
            if mass >= 0.8:
                break
        scores = [-(0.7 * sum(ours[i] * matrix[i][j] for i in range(n)) + 0.3 * max(matrix[i][j] for i in likely)) for j in range(m)]
        best = _softmax(scores, 2.0 * beta)
        mix = [0.5 * a + 0.5 * b for a, b in zip(mix, best)]
    return mix


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

    def choice_locked_by(self, a: _Actor) -> bool:
        """Is this our Protect from a Choice item holder (it locks the holder into Protect until it switches)?"""
        me = self.ours.get(a.slot) if a.side == "mine" else None
        return a.kind == "protect" and me is not None and data.to_id(me.item) in battle.CHOICE_ITEMS

    def pos_of(self, species: str) -> int | None:
        return next((pos for pos, m in self.theirs.items() if species_key(m.species) == species_key(species)), None)

    def best_hit_into(self, pos: int, slot: int) -> dict | None:
        """The foe's most dangerous known hit into our slot (expected damage points with the KO premium); a likely KO that
        lands before our slot moves (priority, or the foe is faster) comes first: two KO moves tie on the capped premium,
        and the first listed won (protect_vs_ko: Surging Strikes, which the faster Flutter Mane outspeeds, over the +1
        Aqua Jet the LETHAL warning names)."""
        best, best_v = None, (False, -1.0)
        for t in self.sheet.threats:
            if t["position"] != pos:
                continue
            for h in t["hits"]:
                if h["into_slot"] != slot or h["move"] == "fakeout":
                    continue
                p_ko = battle.ko_probability(h, with_accuracy=True)
                first = (h.get("priority", 0) or 0) > 0 or self.speed_rank.get(("theirs", pos), 99) < self.speed_rank.get(("mine", slot), 99)
                v = (p_ko >= 0.5 and first, battle._danger(battle._avg(h["damage_pct_of_current_hp"]), h.get("hit_chance", 1.0), p_ko))
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

        def add(name: str, a0: _Actor | None, a1: _Actor | None, payload: dict | None = None, cost: float = 0.0) -> None:
            if a0 is None and 0 in self.ours:
                return
            if a1 is None and 1 in self.ours:
                return
            acts = [a for a in (a0, a1) if a is not None]
            key = tuple((a.slot, a.kind, a.move, a.target, a.species) for a in acts)
            if key in seen or not acts:
                return
            seen.add(key)
            rows.append(_JointRow(name, acts, payload or {}, cost))

        def from_payload(name: str, p0: dict | None, p1: dict | None) -> None:
            a0 = self.actor_for(0, p0, name) if p0 else None
            a1 = self.actor_for(1, p1, name) if p1 else None
            if self.sheet.alone and not self.sheet.stall_reasons and any(a is not None and a.kind == "protect" for a in (a0, a1)):
                return  # our last Pokémon with nothing to wait for: a Protect only delays the same position (Oct 7, match 1)
            cost = 0.0
            for a in (a0, a1):
                if a is not None and a.kind == "switch" and self.memory is not None:
                    fresh = battle.fresh_sleep_target(self.sheet, a.species, self.theirs, self.memory, outgoing=self.ours.get(a.slot))
                    if fresh:
                        # safe_switch's rule (review 5714330): a sleeper or an immune Pokémon going out hands their sleep move
                        # a new target; only a slot facing a KO goes, into a newcomer that survives its entry, at the sleep
                        # cost (51c86e5c T5-T6: the sleeping Indeedee-F out to a Gyarados that Spore and Wild Charge both had)
                        mon, preview = a.incoming
                        if not self.in_ko_range(a.slot) or (preview or {}).get("verdict", "safe") != "safe":
                            return
                        cost += battle.sleep_switch_cost(self.sheet, a.species, preview or {}, self.theirs, self.P)
            add(name, a0, a1, {"name": name, "slot_0": p0, "slot_1": p1}, cost)

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
            value, aim = battle.status_value(move_id, me, self.theirs, self.sheet, number, has_ally=len(self.ours) > 1)
            if move_id not in SETUP_IDS and move_id not in battle.SETUP_MOVES and move_id not in battle.ALLY_ONLY and value <= 0:
                continue
            legal = o.get("targets") or []
            if move_id in battle.SLEEP_MOVES and any(t > 0 for t in legal):
                # one row per foe it can put to sleep: the matrix picks the target (status_value aims by a flat danger sum)
                for t in sorted(t for t in legal if t > 0 and t in self.theirs):
                    if not battle._sleep_blocked({"id": move_id, **info}, self.theirs[t], me, self.theirs, self.fs):
                        out.append((f"{move_id}_{t}", {"option": i, "target": t}))
                continue
            if move_id in battle.ALLY_ONLY:
                target = next((t for t in legal if t < 0), legal[0] if legal else 0)
            else:  # a move aimed at a foe goes where it is worth most (Taunt on the Spore user), not at target 0
                target = aim if aim in legal else (next((t for t in legal if t > 0), 0) if legal and legal != [0] else 0)
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
        """Their joint replies: each foe's best attack into each of our slots, paired; then for each foe its Protect, its
        switch to the best absorber, a fresh Fake Out on the slot it can stop, its setup, its spread move and its sleep
        move on our biggest threat, each next to EVERY attack of its partner (one 'most dangerous' partner put the whole
        weight of those columns on one move: m1 T6 priced Extreme Speed at 0.55 against 6 of 29 observed). A column is
        a set of actions: the same joint action reached twice (a spread column repeating an attack pair) is one column
        (m4 T2 counted Scale Shot + Make It Rain twice, 0.36 for 0.18)."""
        cols: list[_JointColumn] = []
        seen: set = set()
        positions = sorted(self.theirs)

        def add(name: str, acts: list[_Actor]) -> None:
            key = tuple(sorted((a.slot, a.kind, a.move, str(a.target), a.species) for a in acts))
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
                actor = _Actor("theirs", pos, "attack", move=hit["move"], priority=hit.get("priority", 0) or 0, rows=rows,
                               target=(0 if spread else slot), species=foe.species, spread=spread)
                if not any((x.move, x.target) == (actor.move, actor.target) for x in attacks[pos]):
                    attacks[pos].append(actor)
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
        sleepers = battle.sleep_users(self.sheet, self.theirs)
        for pos, foe in self.theirs.items():
            partner_pos = next((p for p in positions if p != pos), None)
            partners = [q for q in attacks.get(partner_pos, []) if q.kind != "pass"] if partner_pos is not None else []
            mate_sleep = self.sleep_target(partner_pos) if partner_pos in sleepers else None
            if mate_sleep is not None:
                # the partner's sleep move next to this foe's Protect or switch (the Oct 8 evening loss: Protect + Spore on
                # turn 1, a switch + Spore on turn 3)
                mv, slot = mate_sleep
                prio = (mv.get("priority") or 0) + (1 if data.to_id(self.theirs[partner_pos].ability) == "prankster" else 0)
                partners.append(_Actor("theirs", partner_pos, "sleep", move=mv["id"], priority=prio, target=slot, species=self.theirs[partner_pos].species))
            variants = [[q] for q in partners] or [[]]

            def text(acts: list[_Actor]) -> str:
                return "".join(f" + {self.theirs[partner_pos].species} {q.move}->{self._tgt(q)}" for q in acts)

            specials: list[tuple[str, _Actor]] = []
            taunted = species_key(foe.species) in self.sheet.taunted  # no Protect, no status moves while the Taunt runs
            if any(m["id"] in battle.PROTECT_LIKE for m in foe.moves) and not taunted:
                specials.append((f"{foe.species} Protect", _Actor("theirs", pos, "protect", move="protect", priority=4, species=foe.species)))
            absorber = self.best_absorber(pos)
            if absorber is not None:
                specials.append((f"{foe.species} -> {absorber.species}", _Actor("theirs", pos, "switch", priority=6, species=foe.species, incoming=absorber)))
            if "fakeout" in {m["id"] for m in foe.moves} and species_key(foe.species) in self.their_fresh:
                stop = self.stoppable_slot()
                if stop is not None and battle._flinchable(self.ours[stop], self.ours, self.fs):
                    rows = self.their_rows(pos, "fakeout")
                    specials.append((f"{foe.species} Fake Out->slot {stop}", _Actor("theirs", pos, "fakeout", move="fakeout", priority=3,
                                                                                   rows={stop: rows[stop]} if stop in rows else {}, target=stop, species=foe.species)))
            setup = next((m["id"] for m in foe.moves if m["id"] in SETUP_IDS and not data.is_damaging(m)), None) if not taunted else None
            if setup:
                prio = 1 if data.to_id(foe.ability) == "prankster" else 0
                specials.append((f"{foe.species} {setup}", _Actor("theirs", pos, "setup", move=setup, priority=prio, species=foe.species)))
            spread_hit = next((h for t in self.sheet.threats if t["position"] == pos for h in t["hits"]
                               if (data.move_info(h["move"]) or {}).get("target") in battle.SPREAD_TARGETS), None)
            if spread_hit and partners:
                specials.append((f"{foe.species} {spread_hit['move']} (spread)",
                                 _Actor("theirs", pos, "attack", move=spread_hit["move"], priority=spread_hit.get("priority", 0) or 0,
                                        rows=self.their_rows(pos, spread_hit["move"]), target=0, species=foe.species, spread=True)))
            sleep = self.sleep_target(pos) if pos in sleepers else None
            if sleep is not None:
                mv, slot = sleep
                prio = (mv.get("priority") or 0) + (1 if data.to_id(foe.ability) == "prankster" else 0)
                specials.append((f"{foe.species} {mv['id']}->slot {slot}", _Actor("theirs", pos, "sleep", move=mv["id"], priority=prio,
                                                                                 target=slot, species=foe.species)))
            for label, actor in specials:
                for others in variants:
                    add(label + text(others), [actor] + others)
        cap = int(self.P.get("reply_columns", 12))
        if len(cols) > cap:
            prior = self.prior(cols)
            order = sorted(range(len(cols)), key=lambda j: -prior[j])[:cap]
            cols = [cols[j] for j in sorted(order)]
        return cols

    def sleep_target(self, pos: int) -> tuple[dict, int] | None:
        """(sleep move, our slot) for the foe's 'sleep our biggest threat' column: the slot whose sleep costs us most by
        the sheet's sleep rows (Part 3), or None when no slot of ours can be slept."""
        cache = self.__dict__.setdefault("_sleep_cache", {})
        if pos in cache:
            return cache[pos]
        threat = next((t for t in self.sheet.threats if t["position"] == pos), None)
        rows = [r for r in (threat or {}).get("sleep") or [] if r["into_slot"] in self.ours]
        found = None
        if rows:
            best = max(rows, key=lambda r: r["value"])
            mv = next((m for m in battle.sleep_users(self.sheet, self.theirs).get(pos, []) if m["id"] == best["move"]), None)
            found = (mv, best["into_slot"]) if mv is not None else None
        cache[pos] = found
        return found

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
        """The slot whose action a fresh Fake Out would stop: the one with more to deny among those it can flinch (an
        Inner Focus Lucario left the Fake Out column out of m4 T7, where the bot's Fake Out went into Tornadus)."""
        best, best_v = None, -1.0
        for n in self.ours:
            if not battle._flinchable(self.ours[n], self.ours, self.fs) or self.sheet.acts(n) <= 0:
                continue
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
        taunted = key in self.sheet.taunted
        p_protect = _p_protect(self.klass, streak=streak, ko_one=ko_one, ko_two=ko_two, facing=facing, fresh_in=fresh_in, turn=self.turn,
                               covered=self.covered(a.slot)) \
            if any(m["id"] in battle.PROTECT_LIKE for m in foe.moves) and not taunted else 0.0
        absorber = self.best_absorber(a.slot)
        has_protect = any(m["id"] in battle.PROTECT_LIKE for m in foe.moves)
        p_switch = _p_switch(self.klass, streak=streak, ko_one=ko_one, ko_two=ko_two, has_absorber=absorber is not None, has_protect=has_protect) \
            * (self.bench_prob.get(species_key(absorber.species), 1.0) if absorber is not None else 1.0)
        p_fake = _p_fakeout(self.klass, turn=self.turn) if ("fakeout" in {m["id"] for m in foe.moves} and fresh_in) else 0.0
        setup_move = next((m["id"] for m in foe.moves if m["id"] in SETUP_IDS and not data.is_damaging(m)), None) if not taunted else None
        p_setup = _p_setup(self.klass, threatened=ko_one >= 0.3 or facing >= 30, turn=self.turn, move_id=setup_move) if setup_move else 0.0
        p_sleep = self.p_sleep(a.slot, threatened=ko_one >= 0.5) if self.sleep_target(a.slot) is not None else 0.0
        rest = max(0.05, 1.0 - min(0.95, p_protect + p_switch + p_fake + p_setup + p_sleep))
        if a.kind == "sleep":
            return p_sleep
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

    def covered_protect(self, pos: int) -> bool:
        """Is this foe's fresh Protect the 'covered' case: our one-move KO threatens it and it is ``covered``?"""
        return self.covered(pos) and self.ko_one(pos) >= 0.5

    def ko_one(self, pos: int) -> float:
        """Our best one-move KO chance on the foe at ``pos`` (accuracy included)."""
        cache = self.__dict__.setdefault("_ko_one_cache", {})
        if pos not in cache:
            cache[pos] = max((battle.ko_probability(r, with_accuracy=True) for n in self.ours for opt in self.by_slot.get(n, {}).get("options") or []
                              if not opt.get("fails") for r in opt.get("targets") or []
                              if r.get("side") == "theirs" and (r.get("target") == pos or (r.get("target", 0) == 0 and self.pos_of(r["species"]) == pos))), default=0.0)
        return cache[pos]

    def covered(self, pos: int) -> bool:
        """Does this foe have a reason not to Protect against our KO: a likely KO (>= 0.5) on one of our slots that lands
        before that slot moves, or a fresh Fake Out on its partner (the da9c168e post-mortem's two exceptions)?"""
        cache = self.__dict__.setdefault("_covered_cache", {})
        if pos not in cache:
            first_ko = any(h["into_slot"] in self.ours and battle.ko_probability(h, with_accuracy=True) >= 0.5
                           and ((h.get("priority", 0) or 0) > 0 or self.speed_rank.get(("theirs", pos), 99) < self.speed_rank.get(("mine", h["into_slot"]), 99))
                           for t in self.sheet.threats if t["position"] == pos for h in t["hits"] if h["move"] != "fakeout")
            partner = next((m for p, m in self.theirs.items() if p != pos), None)
            fake_out = partner is not None and species_key(partner.species) in self.their_fresh and any(m["id"] == "fakeout" for m in partner.moves)
            cache[pos] = bool(first_ko or fake_out)
        return cache[pos]

    def p_sleep(self, pos: int, *, threatened: bool) -> float:
        """This foe's sleep-move rate: the class start value updated by its own moves this game (memory's protocol-log
        table: turns it used a sleep move / turns it moved)."""
        foe = self.theirs.get(pos)
        moves = ((getattr(self.memory, "moves_by_turn", None) or {}).get("theirs") or {}).get(species_key(foe.species) if foe else "", {})
        used = sum(1 for m in moves.values() if m in battle.SLEEP_MOVES)
        return _p_sleep(self.klass, threatened=threatened, used=used, chances=len(moves))

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
            elif kind in ("setup", "sleep"):
                checks.append(used == move)  # a Spore call is scored too (review 5714330: its misses halved lambda)
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
        taunted: dict[tuple[str, int], float] = {}     # chance our Taunt landed on a foe before its status move
        slept: dict[tuple[str, int], float] = {}       # chance a sleep move put a Pokémon to sleep before it acted
        redirect: dict | None = None                    # our Rage Powder / Follow Me this turn: {"slot", "p", "move"}
        tempo = 0.0
        for a in order:
            me_key = (a.side, a.slot)
            p_alive = self._alive_before(me_key, parts, a, order)
            p_act = p_alive * (1.0 - flinched.get(me_key, 0.0)) * (1.0 - slept.get(me_key, 0.0))
            if a.side == "mine" and a.kind != "switch":
                p_act *= self.sheet.acts(a.slot)  # asleep, paralysed or frozen: its move (and its Protect) happens only if it acts
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
                    if self.choice_locked_by(a):
                        # a Choice item locks it into Protect: next turn it Protects again (1 in 3) or switches, no attack
                        # (review 5714330: the mixture topped a Scarf Landorus-T Protect in 4 positions)
                        tempo -= float(P.get("sleep_future_w", 0.5)) * battle._slot_action_value(self.sheet, a.slot, self.slots, P) * p_act
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
                    tempo -= value * p_act * (1.0 - taunted.get(me_key, 0.0)) \
                        * (1.0 if a.move not in ("tailwind", "trickroom") else self._their_speed_control_gain(a.move))
                continue
            if a.kind == "sleep":
                tempo -= self._sleep_loss(a, p_act * (1.0 - taunted.get(me_key, 0.0)), protected, gone, incoming, acts, slept)
                continue
            if a.kind == "status":
                if a.side == "mine" and a.move in battle.REDIRECT_MOVES:
                    # worth the hits it draws (below), and only while the redirector stands: no flat cover credit (the Oct 9
                    # loss 07a3cb1f: Follow Me ended when the faster Wood Hammer KO'd Indeedee, and Draco Meteor went home)
                    redirect = {"slot": a.slot, "p": p_act, "move": a.move}
                    continue
                if a.side == "mine" and a.move in battle.SLEEP_MOVES:
                    tempo += self._our_sleep_gain(a, p_act, protected, gone, acts, slept)
                    continue
                if a.move in ("helpinghand", "coaching") and a.side == "mine":
                    boosted[("mine", 1 - a.slot)] = p_act  # the partner's attack this turn is x1.5
                elif a.side == "mine":
                    blocked = protected.get(("theirs", a.target), 0.0) if isinstance(a.target, int) and a.target in self.theirs else 0.0
                    if a.move == "taunt" and isinstance(a.target, int) and a.target in self.theirs:
                        key_t = ("theirs", a.target)
                        worth = battle.taunt_value(self.ours[a.slot], a.slot, a.target, self.theirs[a.target], self.sheet, P)  # its target's, 0 when it fails
                        if worth > 0:
                            taunted[key_t] = 1.0 - (1.0 - taunted.get(key_t, 0.0)) * (1.0 - p_act * (1.0 - blocked) * (1.0 - gone.get(key_t, 0.0)))
                    else:
                        worth = battle.status_value(a.move, self.ours[a.slot], self.theirs, self.sheet, a.slot, has_ally=len(self.ours) > 1)[0]
                    tempo += worth * p_act * 0.5 * (1.0 - blocked)  # a Taunt or a Spore into a working Protect does nothing
                continue
            # attacks and Fake Outs: each row lands on its target unless it Protected (and the Protect worked) or left
            for tgt_slot, hit, share in self._aimed(a, redirect, parts, order):
                tgt_side = "theirs" if a.side == "mine" else "mine"
                tgt_key = (tgt_side, tgt_slot)
                if self._fails_into(a, hit, tgt_key, order, acts):
                    continue
                p_land = p_act * share * float(hit.get("hit_chance", 1.0)) * (1.0 - protected.get(tgt_key, 0.0))
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
                if a.side == "theirs":
                    row_used = self._under_intimidate(a, row_used, tgt_slot, gone, incoming)
                parts.setdefault(tgt_key, []).append((p_land, row_used))
                hit_first[tgt_key] = 1.0 - (1.0 - hit_first.get(tgt_key, 0.0)) * (1.0 - p_land)
                if a.kind == "attack" and tgt_key not in acts:
                    chance = self._flinch_chance(a, tgt_key)  # a faster Rock Slide / Icicle Crash: it may not move at all
                    if chance > 0:
                        flinched[tgt_key] = 1.0 - (1.0 - flinched.get(tgt_key, 0.0)) * (1.0 - p_land * chance)
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
            if a.side == "mine" and a.rows:
                # the priority bonus is for a hit that lands first: none for a Fake Out into a Protect (m1 T6)
                landed = sum(1.0 - protected.get(("theirs", t), 0.0) for t in a.rows) / len(a.rows)
                tempo += P["priority_w"] * a.priority * p_act * landed
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
        return value + tempo + self._next_turn(parts, gone, incoming, order) - row.cost

    def _aimed(self, a: _Actor, redirect: dict | None, parts: dict, order: list[_Actor]) -> list[tuple[int, dict, float]]:
        """(target, row, share) for each hit of this actor. A single-target hit of theirs that comes after our Rage Powder
        or Follow Me goes into the redirector while it stands (the redirection ends when it faints: a faster hit that KOs
        it sends the slower ones home); mech's ``data.redirect_ignored``: Grass types, Overcoat and Safety Goggles ignore
        Rage Powder, Stalwart / Propeller Tail both. A Sucker Punch drawn into the redirector fails: it is not attacking."""
        rows = list(a.rows.items())
        if not redirect or a.side != "theirs" or a.spread or redirect["p"] <= 0:
            return [(t, h, 1.0) for t, h in rows]
        r_slot = redirect["slot"]
        foe = self.theirs.get(a.slot)
        if foe is None or data.redirect_ignored(redirect["move"], foe):
            return [(t, h, 1.0) for t, h in rows]
        q = redirect["p"] * self._alive_before(("mine", r_slot), parts, a, order)
        out: list[tuple[int, dict, float]] = []
        for t, h in rows:
            if t == r_slot or q <= 0:
                out.append((t, h, 1.0))
                continue
            out.append((t, h, 1.0 - q))
            move = h.get("move") or a.move
            drawn = self.their_rows(a.slot, move).get(r_slot)
            if drawn is not None and not data.needs_target_attacking(move):
                out.append((r_slot, drawn, q))
        return out

    def _fails_into(self, a: _Actor, hit: dict, tgt_key: tuple[str, int], order: list[_Actor], acts: dict) -> bool:
        """Sucker Punch / Thunderclap (mech's ``data.needs_target_attacking``) fail unless the target is about to use a
        damaging move: a status move, a Protect, a switch or a target that has already moved make them fail."""
        if not data.needs_target_attacking(a.move or hit.get("move")):
            return False
        target = next((x for x in order if (x.side, x.slot) == tgt_key), None)
        return target is None or target.kind not in ("attack", "fakeout") or tgt_key in acts

    def _flinch_chance(self, a: _Actor, tgt_key: tuple[str, int]) -> float:
        """A secondary flinch (Rock Slide / Icicle Crash 30%, Waterfall 20%: battle.FLINCH_CHANCE) on a target that has
        not moved yet: Inner Focus, Shield Dust and Covert Cloak ignore it, Sheer Force removes it, Serene Grace doubles it."""
        chance = battle.FLINCH_CHANCE.get(a.move, 0.0)
        if not chance:
            return 0.0
        user = (self.ours if a.side == "mine" else self.theirs).get(a.slot)
        target = (self.theirs if tgt_key[0] == "theirs" else self.ours).get(tgt_key[1])
        if user is None or target is None or data.to_id(user.ability) == "sheerforce":
            return 0.0
        if data.secondary_blocked(target) or data.to_id(target.ability) in FLINCH_IMMUNE_ABILITIES:
            return 0.0
        return min(1.0, chance * (2.0 if data.to_id(user.ability) == "serenegrace" else 1.0))

    def _under_intimidate(self, a: _Actor, row: dict, slot: int, gone: dict, incoming: dict) -> dict:
        """Their physical hit into our slot after our OTHER slot brought in an Intimidate user this turn (switches go
        first): the hit at the foe's Intimidated Attack (mech's ``intimidated_foes``: Clear Amulet, Defiant... respected),
        weighted by the chance the switch happened. Intimidate cycling covers the partner, not only the newcomer."""
        other = ("mine", 1 - slot)
        p = gone.get(other, 0.0)
        new = incoming.get(other)
        mon = new[0] if isinstance(new, tuple) else new
        if p <= 0 or not isinstance(mon, battle.Mon) or data.to_id(mon.ability) != "intimidate":
            return row
        if (data.move_info(a.move) or {}).get("category") != "Physical":
            return row
        weak = self._intimidated_row(a.slot, a.move, slot, mon)
        lo, hi = row["damage_pct_of_current_hp"]
        if weak is None or lo + hi <= 0:
            return row
        scale = (1.0 - p) + p * (sum(weak["damage_pct_of_current_hp"]) / max(lo + hi, 0.1))
        out = {**row, "damage_pct_of_current_hp": [round(lo * scale, 1), round(hi * scale, 1)]}
        for k in ("per_hit", "first_hit", "hits"):
            out.pop(k, None)
        out["ko"] = battle._ko_label(*out["damage_pct_of_current_hp"])
        return out

    def _intimidated_row(self, pos: int, move: str, slot: int, newcomer: battle.Mon) -> dict | None:
        cache = self.__dict__.setdefault("_intimidated_cache", {})
        key = (pos, move, slot, species_key(newcomer.species))
        if key not in cache:
            foe, me = self.theirs.get(pos), self.ours.get(slot)
            mv = next((m for m in (foe.moves if foe else []) if m["id"] == move), None)
            weakened = battle.intimidated_foes(newcomer, [foe])[0] if foe is not None else None
            prof = None
            if mv is not None and me is not None and weakened is not None and weakened is not foe:
                prof = battle.hit_profile(weakened, mv, me, field=self.fs,
                                          spread=battle.is_spread(mv, foes=len(self.ours), allies=len(self.theirs) - 1))
            cache[key] = prof
        return cache[key]

    def _our_sleep_gain(self, a: _Actor, p_act: float, protected: dict, gone: dict, acts: dict, slept: dict) -> float:
        """What our sleep move on a foe is worth in this cell, the mirror of ``_sleep_loss``: its later actions lost over
        the expected sleep (``sleep_future_w`` each), this turn's too when it lands before the foe moves (``slept`` stops
        that action), each action worth what the foe does to us with one (battle.threat_weight). Nothing through a
        working Protect, into a foe that left, or into an immune or statused one (mech's status_move_blocked). A flat
        credit undervalued it: Spore on the +1 Gyarados lost to two chip attacks (the intimidate_the_dancer scenario)."""
        pos = a.target
        foe = self.theirs.get(pos) if isinstance(pos, int) else None
        me = self.ours.get(a.slot)
        if foe is None or me is None or p_act <= 0:
            return 0.0
        mv = {"id": a.move, **(data.move_info(a.move) or {})}
        if battle._sleep_blocked(mv, foe, me, self.theirs, self.fs):
            return 0.0
        key = ("theirs", pos)
        p_land = p_act * data.hit_chance(mv, self.fs.weather if self.fs else None) * (1.0 - protected.get(key, 0.0)) * (1.0 - gone.get(key, 0.0))
        if p_land <= 0:
            return 0.0
        first = key not in acts
        if first:
            slept[key] = 1.0 - (1.0 - slept.get(key, 0.0)) * (1.0 - p_land)
        w = float(self.P.get("sleep_future_w", 0.5))
        return p_land * self.weights.get(species_key(foe.species), 0.0) * w * (battle.SLEEP_TURNS - (1.0 if first else 0.0))

    def _sleep_loss(self, a: _Actor, p_use: float, protected: dict, gone: dict, incoming: dict, acts: dict, slept: dict) -> float:
        """What their sleep move costs us in this cell, in damage points: the target slot's later actions lost (this
        turn's too when it lands before the slot moves: ``slept`` stops it), nothing through a working Protect, and the
        newcomer's actions when our slot switched into it (a fresh target; a sleeping or immune newcomer is none)."""
        tgt = ("mine", a.target)
        mon = self.ours.get(a.target)
        if mon is None or p_use <= 0:
            return 0.0
        foe = self.theirs.get(a.slot)
        mv = next((m for m in battle.sleep_users(self.sheet, self.theirs).get(a.slot, []) if m["id"] == a.move), {"id": a.move})
        acc = data.hit_chance(mv, self.fs.weather if self.fs else None)
        p_land = p_use * acc * (1.0 - protected.get(tgt, 0.0))
        p_in = p_land * gone.get(tgt, 0.0)
        p_stay = p_land - p_in
        w = float(self.P.get("sleep_future_w", 0.5))
        value = battle._slot_action_value(self.sheet, a.target, self.slots, self.P)
        first = tgt not in acts  # it has not acted yet this turn: the sleep takes this turn's action too
        if first:
            slept[tgt] = 1.0 - (1.0 - slept.get(tgt, 0.0)) * (1.0 - p_stay)
        loss = p_stay * value * w * (battle.SLEEP_TURNS - (1.0 if first else 0.0))  # this turn's own action is lost through ``slept``
        if p_in > 0 and incoming.get(tgt):
            newcomer = incoming[tgt][0] if isinstance(incoming[tgt], tuple) else incoming[tgt]
            if isinstance(newcomer, battle.Mon) and foe is not None and not battle._sleep_blocked(mv, newcomer, foe, self.ours, self.fs) \
                    and not self.sheet.bench_status.get(species_key(newcomer.species)):
                preview = incoming[tgt][1] if isinstance(incoming[tgt], tuple) else {}
                deals = max((min(battle._avg(d["damage_pct"]), 100.0) for d in (preview or {}).get("deals_best") or []), default=30.0)
                loss += p_in * deals * w * battle.SLEEP_TURNS
        return loss

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
        locked_now = {a.slot for a in order if a.side == "mine" and self.choice_locked_by(a)}  # locked into Protect: no hit next turn
        their_protected_now = {a.slot for a in order if a.side == "theirs" and a.kind == "protect"}

        def protect_answer(side: str, slot: int, mon: battle.Mon, newcomer: bool = False) -> float:
            """How much of a KO threat next turn a Protect answers: a fresh one all but the tempo (0.6), a spent one 1 in 3
            (a newcomer's is fresh)."""
            if not any(m["id"] in battle.PROTECT_LIKE for m in mon.moves):
                return 0.0
            spent = not newcomer and slot in (protected_now if side == "mine" else their_protected_now)
            return 0.6 / 3 if spent else 0.6

        # who stands next turn: (chance, HP left as a share of its HP at the start of this turn, the Pokémon, its rows)
        theirs_alive: list[tuple[int, float, float, battle.Mon, bool]] = []
        for pos, foe in self.theirs.items():
            key = ("theirs", pos)
            left = 1.0 - gone.get(key, 0.0)
            alive, rem = state(key)
            if left > 0:
                theirs_alive.append((pos, left * alive, rem, foe, False))
            newcomer = incoming.get(key)
            if gone.get(key, 0.0) > 0 and isinstance(newcomer, battle.Mon):
                alive_in, rem_in = state(("in",) + key)
                theirs_alive.append((pos, gone[key] * alive_in, rem_in, newcomer, True))  # their switch-in fights next turn too
        ours_alive: list[tuple[int, float, float, battle.Mon, float, dict | None]] = []
        for n, me in self.ours.items():
            key = ("mine", n)
            left = 1.0 - gone.get(key, 0.0)
            alive, rem = state(key)
            if left > 0:
                ours_alive.append((n, left * alive, rem, me, self.material[n], None))
            if key in incoming and incoming[key]:
                mon = incoming[key][0] if isinstance(incoming[key], tuple) else incoming[key]
                preview = incoming[key][1] if isinstance(incoming[key], tuple) else {}
                if isinstance(mon, battle.Mon):
                    alive_in, rem_in = state(("in",) + key)
                    # our switch-in acts next turn: its hits and its exposure come from its own preview rows (Part 3b: the
                    # Landorus pivot's Earthquake into a Balloon-less Gholdengo was never counted)
                    ours_alive.append((n, gone[key] * alive_in, rem_in, mon, battle.material(mon, P), preview or {}))
        total = 0.0
        # a slot of ours facing a KO next turn can also switch out (the newcomer takes the hit): a bench card answers part
        # of that threat, like a Protect does (the hand scenarios: Incineroar's Fake Out turn was charged its full
        # material for a Surging Strikes it can simply leave next turn, and every Protect row for the threats after it)
        switch_answer = float(P.get("reply_next_switch", 0.5)) if self._bench_ready() else 0.0
        keep: dict[int, float] = {}  # ours_alive index -> chance it is still standing after their hits next turn
        for pos, q, rem_f, foe, fresh in theirs_alive:
            # our best KO on it next turn, over our standing Pokémon, rescaled to what it has left
            best_p, best_first = 0.0, False
            for n, p, _, me, _, preview in ours_alive:
                if preview is None and n in locked_now:
                    continue
                for row, first in self._our_next_hits(n, me, preview, pos, foe, fresh):
                    lo, hi = row["damage_pct_of_current_hp"]
                    p_ko = battle.linear_ko(lo / rem_f, hi / rem_f) * float(row.get("hit_chance", 1.0)) * p
                    if p_ko > best_p:
                        best_p, best_first = p_ko, first
            if best_p > 0:
                total += w * q * best_p * (1.0 - protect_answer("theirs", pos, foe, fresh)) \
                    * (P["ko_bonus_guaranteed"] + P.get("ko_threat_w", 0.0) * self.weights.get(species_key(foe.species), 0.0))
            # its KO on ours next turn, unless we take it out first: ONE target (the slot whose KO is worth most to it; a
            # spread hit reaches every slot it hits), and each of ours falls at most once (it was charged once per foe and
            # per slot: a Pokémon two foes could KO paid twice, a foe with two KO targets took both)
            removed_first = best_p if best_first else 0.0
            by_slot: dict[int, list[tuple[int, float]]] = {}
            by_spread: dict[str, list[tuple[int, float]]] = {}
            for i, (n, p, rem_s, me, mat, preview) in enumerate(ours_alive):
                hit = self._their_next_hit(pos, foe, fresh, n, me, preview)
                if hit is None:
                    continue
                lo, hi = hit["damage_pct_of_current_hp"]
                p_ko = battle.linear_ko(lo / rem_s, hi / rem_s) * float(hit.get("hit_chance", 1.0))
                if p_ko <= 0:
                    continue
                escape = max(protect_answer("mine", n, me, preview is not None), switch_answer,
                             self._redirect_answer(n, ours_alive, hit, foe))
                x = q * p_ko * (1.0 - removed_first) * (1.0 - escape)
                by_slot.setdefault(n, []).append((i, x))
                if (data.move_info(hit.get("move") or "") or {}).get("target") in battle.SPREAD_TARGETS:
                    by_spread.setdefault(hit["move"], []).append((i, x))
            if not by_slot:
                continue

            def worth(entries: list[tuple[int, float]]) -> float:
                return sum(x * ours_alive[i][1] * ours_alive[i][4] for i, x in entries)

            aimed = max(list(by_slot.values()) + list(by_spread.values()), key=worth)
            for i, x in aimed:
                keep[i] = keep.get(i, 1.0) * (1.0 - x)
        for i, k in keep.items():
            n, p, rem_s, me, mat, preview = ours_alive[i]
            total -= w * p * (1.0 - k) * mat
        return total

    def _redirect_answer(self, n: int, ours_alive: list, hit: dict, foe: battle.Mon) -> float:
        """How much of a single-target KO threat next turn on our slot ``n`` our partner's Rage Powder / Follow Me answers:
        all but the tempo (0.6, as a fresh Protect) times the chance the redirector stands next turn, nothing against a
        spread hit or a foe that ignores it (mech's ``data.redirect_ignored``). The partner keeps attacking behind it."""
        if (data.move_info(hit.get("move") or "") or {}).get("target") in battle.SPREAD_TARGETS:
            return 0.0
        standing = 0.0
        for m, p, _, mon, _, _ in ours_alive:
            if m == n:
                continue
            move = next((mv["id"] for mv in mon.moves if mv["id"] in battle.REDIRECT_MOVES), None)
            if move and not data.redirect_ignored(move, foe):
                standing += p
        return 0.6 * min(1.0, standing)

    def _bench_ready(self) -> bool:
        """Do we have a Pokémon to switch in next turn (any switch option this turn: what leaves this turn joins the bench)?"""
        return any(o.get("type") == "switch" for s in self.slots for o in s.get("options") or [])

    def _our_next_hits(self, n: int, me: battle.Mon, preview: dict | None, pos: int, foe: battle.Mon, fresh: bool) -> list[tuple[dict, bool]]:
        """Our slot's best hit next turn into the foe at ``pos`` (or into their switch-in when ``fresh``), with whether it
        lands first: the sheet's rows for a Pokémon already out, the switch-in preview's best hit for our newcomer, the
        switch rows for theirs."""
        if preview is not None:  # our newcomer: the preview's best hit into that foe
            if fresh:
                return []
            deal = next((d for d in preview.get("deals_best") or [] if species_key(d.get("to")) == species_key(foe.species)), None)
            if deal is None:
                return []
            row = {"damage_pct_of_current_hp": deal["damage_pct"], **({"hit_chance": deal["hit_chance"]} if "hit_chance" in deal else {})}
            return [(row, foe.species in (preview.get("outspeeds") or []))]
        if fresh:  # their newcomer: our moves into it as it came in (switch rows), at our speed against its
            out = []
            for opt in self.by_slot.get(n, {}).get("options") or []:
                row = self.switch_rows.get((species_key(foe.species), n, opt["option"]))
                if row is not None and not opt.get("fails"):
                    first = (opt.get("priority", 0) or 0) > 0 or (me.stat("spe") > foe.stat("spe")) != bool(self.fs and self.fs.trick_room)
                    out.append((row, first))
            return sorted(out, key=lambda x: -battle.ko_probability(x[0], with_accuracy=True))[:1]
        ans = self.best_single_answer(n, pos) if me is self.ours.get(n) else None
        if not ans:
            return []
        opt = self.move_option(n, ans["option"])
        row = next((r for r in (opt or {}).get("targets") or [] if r.get("side") == "theirs"
                    and (r.get("target") == pos or (r.get("target", 0) == 0 and self.pos_of(r["species"]) == pos))), None)
        if row is None:
            return []
        first = (opt.get("priority", 0) or 0) > 0 or self.speed_rank.get(("mine", n), 99) < self.speed_rank.get(("theirs", pos), 99)
        return [(row, first)]

    def _their_next_hit(self, pos: int, foe: battle.Mon, fresh: bool, n: int, me: battle.Mon, preview: dict | None) -> dict | None:
        """The foe's most dangerous hit next turn into our slot: the sheet's threat rows, our switch-in's entry rows, or
        (their switch-in) its own moves into ours, computed once per pair."""
        if fresh:
            cache = self.__dict__.setdefault("_newcomer_hits", {})
            key = (species_key(foe.species), species_key(me.species))
            if key not in cache:
                best, best_v = None, -1.0
                for mv in foe.moves:
                    if not data.is_damaging(mv) or mv["id"] in ("fakeout", "firstimpression"):
                        continue
                    prof = battle.hit_profile(foe, mv, me, field=self.fs, spread=battle.is_spread(mv, foes=len(self.ours), allies=len(self.theirs) - 1))
                    if prof is None:
                        continue
                    v = battle._danger(battle._avg(prof["damage_pct_of_current_hp"]), prof.get("hit_chance", 1.0), battle.ko_probability(prof, with_accuracy=True))
                    if v > best_v:
                        best, best_v = prof, v
                cache[key] = best
            return cache[key]
        if preview is not None:
            best, best_v = None, -1.0
            for (k, p, _), row in ((k, r) for k, r in self.our_switch_rows.items() if len(k) == 3):
                if k == species_key(me.species) and p == pos:
                    v = battle._danger(battle._avg(row["damage_pct_of_current_hp"]), row.get("hit_chance", 1.0), battle.ko_probability(row, with_accuracy=True))
                    if v > best_v:
                        best, best_v = row, v
            return best
        return self.best_hit_into(pos, n) if me is self.ours.get(n) else None

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
    __slots__ = ("name", "actions", "payload", "cost")

    def __init__(self, name: str, actions: list[_Actor], payload: dict, cost: float = 0.0) -> None:
        self.name, self.actions, self.payload, self.cost = name, actions, payload, cost
        if "name" not in self.payload:
            self.payload["name"] = name
        for a in actions:
            self.payload[f"slot_{a.slot}"] = a.payload
        self.payload.setdefault("why", name)
