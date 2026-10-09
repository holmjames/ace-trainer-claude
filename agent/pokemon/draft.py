"""Draft picks: pure code, always well under the 15-second clock.

Plain language: each offered card gets a score, the best score wins, ties go
to the server's order so the same pool always drafts the same way. No model
call ever happens here (Fable thinks for several seconds; the draft clock is
15 s including our call).

What the score rewards, roughly in order of weight:

1. Raw quality: base stat total and real speed (EVs and nature applied).
2. Offense against the opponent's drafted cards: can this card hit what
   they already have, using its actual moves, type chart and STAB?
3. Defense against the opponent's drafted cards: does it resist the attacks
   their sets actually carry, or does it get hit super-effectively?
4. Team fit with our own picks: new offensive types are good, stacking a
   shared weakness is bad, and support (speed control, redirection, Fake
   Out) matters more once we have attackers.
5. Denial: when our top two cards are close, take the one the opponent
   needs more.

Both players' picks are public and every card is a complete set, which is why
the opponent-aware parts work at all. ``choose_pick`` is the only entry point
the agent uses; everything else is a helper you can unit-test on its own.

``draft_v2`` = 1 switches to the counter-aware search at the bottom of this file
(``choose_pick_v2``): every offered card is valued by how our FINISHED team would
fare against theirs once the rest of the snake draft is played out, with their
best replies tried at their next pick. That is the answer to the Oct 7 losses to
a counter-picking opponent (the Oct 7 opponent), which the score above never asked about.
"""

from __future__ import annotations

import json
import math
import os
import random
import secrets
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from altruagent import LegalAction

from . import data
from .memory import MatchMemory, species_key
from .tuning import DEFAULTS

PREFIX = "draft_pick:"

# Moves that make a doubles team work (speed control, redirection, disruption).
SUPPORT_MOVES = {
    "fakeout": 25, "tailwind": 25, "trickroom": 20, "icywind": 15, "electroweb": 12, "followme": 20,
    "ragepowder": 20, "spore": 15, "protect": 5, "helpinghand": 8, "willowisp": 6, "thunderwave": 6,
    "snarl": 6, "partingshot": 8, "coaching": 6, "lifedew": 6, "pollenpuff": 4, "wideguard": 8,
}
SUPPORT_ABILITIES = {"intimidate": 8, "friendguard": 6, "prankster": 5, "regenerator": 3}
TOTAL_PICKS = 6


@dataclass
class Profile:
    """Everything the scorer needs about one card, computed once."""

    species: str
    types: list[str]
    stats: dict[str, int]  # level-50 actual stats
    bst: int
    moves: list[dict] = field(default_factory=list)  # move_info dicts with an "id" key
    support: int = 0
    priority: bool = False
    spread: bool = False
    protect: bool = False
    ability: str = ""
    card_item: str = ""
    card: dict = field(default_factory=dict)  # the full drafted set, for damage math at Team Preview

    @property
    def attacks(self) -> list[dict]:
        return [m for m in self.moves if (m.get("base_power") or 0) > 0]


@dataclass
class ScoredCard:
    action: LegalAction
    card_id: str
    species: str
    score: float
    notes: list[str]
    exposure: int = 0  # v2 (rank_picks_v2): undrafted pool cards that beat this card by >= BEATS_MARGIN, the counters left to take


def card_id_of(action: LegalAction) -> str:
    if action.action_id.startswith(PREFIX):
        return action.action_id[len(PREFIX):]
    template = action.input.get("action") if isinstance(action.input, dict) else None
    if isinstance(template, dict) and template.get("card_id"):
        return str(template["card_id"])
    return action.action_id


def profile(card: dict) -> Profile:
    base = data.resolve_base_stats(card) or {}
    stats = data.actual_stats(card) or {}
    moves = []
    for name in card.get("moves") or []:
        info = data.move_info(name)
        moves.append({"id": data.to_id(name), **(info or {"type": None, "category": None, "base_power": 0, "priority": 0, "target": None})})
    ids = [m["id"] for m in moves]
    return Profile(
        species=str(card.get("species") or card.get("name") or "?"),
        types=data.resolve_types(card),
        stats=stats,
        bst=sum(base.values()) if base else 0,
        moves=moves,
        support=min(sum(SUPPORT_MOVES.get(i, 0) for i in ids), 30),
        priority=any((m.get("priority") or 0) > 0 and (m.get("base_power") or 0) > 0 for m in moves),
        spread=any(m.get("target") in ("allAdjacentFoes", "allAdjacent") for m in moves),
        protect="protect" in ids or "wideguard" in ids,
        ability=data.to_id(card.get("ability")),
        card_item=data.to_id(card.get("item")),
        card=dict(card),
    )


# -- matchup helpers ---------------------------------------------------------------------


def hit_quality(attacker: Profile, defender: Profile) -> float:
    """Best single attack from attacker into defender, as a rough 'how hard' number:
    effectiveness * STAB * base_power, scaled so a neutral 80 BP STAB hit is ~1.0."""
    best = 0.0
    for move in attacker.attacks:
        mult = data.effectiveness(move.get("type") or "", defender.types)
        if mult == 0:
            continue
        stab = 1.5 if data.to_id(move.get("type")) in attacker.types else 1.0
        power = (move.get("base_power") or 0) * data.expected_hits(move)
        best = max(best, mult * stab * power / 120.0)
    return best


def offense_vs(attacker: Profile, defenders: list[Profile]) -> float:
    if not defenders:
        return 0.0
    return sum(hit_quality(attacker, d) for d in defenders) / len(defenders)


def defense_vs(defender: Profile, attackers: list[Profile]) -> float:
    """Positive when the opponent's actual attacks struggle against this card, negative when they
    hit it super-effectively. Each attacker contributes its best multiplier only."""
    if not attackers or not defender.types:
        return 0.0
    total = 0.0
    for attacker in attackers:
        mults = [data.effectiveness(m.get("type") or "", defender.types) for m in attacker.attacks]
        if not mults:
            continue
        worst = max(mults)
        if worst >= 4:
            total -= 2.0
        elif worst >= 2:
            total -= 1.0
        elif worst == 0:
            total += 1.0
        elif worst <= 0.5:
            total += 0.6
    return total / len(attackers)


def shared_weaknesses(candidate: Profile, team: list[Profile]) -> int:
    """How many of our existing picks share a weakness with this card."""
    mine = {t for t, m in data.defensive_weaknesses(candidate.types).items() if m > 1}
    count = 0
    for member in team:
        theirs = {t for t, m in data.defensive_weaknesses(member.types).items() if m > 1}
        count += len(mine & theirs) > 0
    return count


def new_offensive_types(candidate: Profile, team: list[Profile]) -> list[str]:
    have = {data.to_id(m.get("type")) for member in team for m in member.attacks}
    return sorted({data.to_id(m.get("type")) for m in candidate.attacks if m.get("type")} - have)


# -- modes and roles (item 2) --------------------------------------------------------------------

MODE_SETTERS = {"trickroom": "trickroom", "tailwind": "tailwind", "raindance": "rain", "sunnyday": "sun"}
MODE_ABILITIES = {"drizzle": "rain", "drought": "sun", "orichalcumpulse": "sun", "hadronengine": "electricterrain"}


def team_modes(team: list[Profile]) -> set[str]:
    modes: set[str] = set()
    for p in team:
        for m in p.moves:
            if m["id"] in MODE_SETTERS:
                modes.add(MODE_SETTERS[m["id"]])
        if p.ability in MODE_ABILITIES:
            modes.add(MODE_ABILITIES[p.ability])
    return modes


def mode_fit(me: Profile, modes: set[str]) -> tuple[float, str]:
    """How well a card rides a mode the team already has (0..1) and why."""
    speed = me.stats.get("spe", 0)
    best, why = 0.0, ""
    if "trickroom" in modes and speed and speed <= 80:
        v = 1.0 if speed <= 60 else 0.6
        if v > best: best, why = v, f"Trick Room partner (speed {speed})"
    if "tailwind" in modes and speed >= 100 and me.attacks:
        v = 0.8
        if v > best: best, why = v, "Tailwind attacker"
    if "rain" in modes and ("water" in {data.to_id(m.get("type")) for m in me.attacks} or me.ability == "swiftswim"):
        v = 1.0 if me.ability == "swiftswim" else 0.7
        if v > best: best, why = v, "rain abuser"
    if "sun" in modes and ("fire" in {data.to_id(m.get("type")) for m in me.attacks} or me.ability in ("chlorophyll", "protosynthesis")):
        v = 1.0 if me.ability in ("chlorophyll", "protosynthesis") else 0.7
        if v > best: best, why = v, "sun abuser"
    return best, why


def missing_roles(team: list[Profile], theirs: list[Profile]) -> set[str]:
    roles = {"speed_control", "disruption", "answer"}
    ids = {m["id"] for p in team for m in p.moves}
    if ids & {"tailwind", "trickroom", "icywind", "electroweb", "thunderwave"}:
        roles.discard("speed_control")
    if ids & {"fakeout", "followme", "ragepowder", "spore"}:
        roles.discard("disruption")
    if theirs:
        best_opp = max(theirs, key=lambda o: max((hit_quality(o, p) for p in team), default=0.0) if team else o.bst)
        stabs = [m for m in best_opp.attacks if data.to_id(m.get("type")) in best_opp.types] or best_opp.attacks
        if stabs:
            stab = max(stabs, key=lambda m: m.get("base_power") or 0)
            if any(data.effectiveness(stab.get("type") or "", p.types) <= 0.5 for p in team if p.types):
                roles.discard("answer")
    else:
        roles.discard("answer")
    return roles


def fills_role(me: Profile, roles: set[str], theirs: list[Profile]) -> list[str]:
    ids = {m["id"] for m in me.moves}
    filled = []
    if "speed_control" in roles and ids & {"tailwind", "trickroom", "icywind", "electroweb", "thunderwave"}:
        filled.append("speed_control")
    if "disruption" in roles and ids & {"fakeout", "followme", "ragepowder", "spore"}:
        filled.append("disruption")
    if "answer" in roles and theirs:
        best_opp = max(theirs, key=lambda o: o.bst)
        stabs = [m for m in best_opp.attacks if data.to_id(m.get("type")) in best_opp.types] or best_opp.attacks
        if stabs and me.types:
            stab = max(stabs, key=lambda m: m.get("base_power") or 0)
            if data.effectiveness(stab.get("type") or "", me.types) <= 0.5:
                filled.append(f"answer to {best_opp.species}")
    return filled


# -- the score -----------------------------------------------------------------------------


def score_card(card: dict, memory: MatchMemory, *, my_team: list[Profile] | None = None,
               opp_team: list[Profile] | None = None, params: dict | None = None) -> tuple[float, list[str]]:
    P = params or DEFAULTS
    me = profile(card)
    mine = my_team if my_team is not None else [profile(c) for c in memory.my_cards.values()]
    theirs = opp_team if opp_team is not None else [profile(c) for c in memory.opp_cards.values()]
    notes: list[str] = []
    score = 0.0

    # 1. raw quality
    if me.bst:
        score += me.bst / 10.0
        notes.append(f"bst {me.bst}")
    speed = me.stats.get("spe", 0)
    if speed >= 150:
        score += P["draft_veryfast_bonus"]; notes.append(f"very fast {speed}")
    elif speed >= 120:
        score += P["draft_fast_bonus"]; notes.append(f"fast {speed}")
    elif speed and speed <= 60 and "trickroom" not in {m["id"] for m in me.moves}:
        score -= 2; notes.append(f"slow {speed}")

    # 2/3. matchup against what they already have
    if theirs:
        off = offense_vs(me, theirs)
        score += P["draft_offense_w"] * off
        notes.append(f"offense vs opp {off:.2f}")
        de = defense_vs(me, theirs)
        score += P["draft_defense_w"] * de
        notes.append(f"defense vs opp {de:+.2f}")
    elif not me.attacks:
        score -= 4
        notes.append("no attacks")

    # 4. team fit
    new_types = new_offensive_types(me, mine)
    if mine:
        if new_types:
            score += P["draft_coverage_w"] * min(len(new_types), 3)
            notes.append(f"new coverage {new_types[:3]}")
        shared = shared_weaknesses(me, mine)
        if shared:
            score -= P["draft_shared_weakness_w"] * shared
            notes.append(f"shares weakness with {shared}")
    # item 2: ride our mode, fill missing roles, deny their mode piece
    if P.get("draft_mode_w", 0) and mine:
        fit, why = mode_fit(me, team_modes(mine))
        if fit:
            score += P["draft_mode_w"] * fit
            notes.append(why)
    if P.get("draft_role_w", 0) and len(mine) >= 3:
        filled = fills_role(me, missing_roles(mine, theirs), theirs)
        if filled:
            score += P["draft_role_w"] * len(filled)
            notes.append(f"fills {filled}")
    if P.get("draft_deny_mode_w", 0) and theirs:
        their_fit, their_why = mode_fit(me, team_modes(theirs))
        if their_fit >= 0.7:
            score += P["draft_deny_mode_w"] * their_fit
            notes.append(f"denies their {their_why}")

    weight = min(len(mine), 3) / 3  # support counts once we have attackers
    have_support = any(m.support >= 15 for m in mine)
    if me.support and weight:
        bonus = min(me.support, P["draft_support_cap"]) * weight * (0.5 if have_support else 1.0)
        score += bonus
        notes.append(f"support +{bonus:.0f}")
    if me.ability in SUPPORT_ABILITIES:
        score += SUPPORT_ABILITIES[me.ability]
        notes.append(me.ability)
    if me.priority:
        score += 3; notes.append("priority")
    if me.spread:
        score += 2; notes.append("spread")
    if me.protect:
        score += 1

    return round(score, 3), notes


def choose_pick(legal_actions: list[LegalAction], memory: MatchMemory, params: dict | None = None) -> ScoredCard:
    P = params or DEFAULTS
    if P.get("draft_v2", 0):
        try:
            return choose_pick_v2(legal_actions, memory, P)
        except Exception as exc:  # noqa: BLE001 — a bug in the search must not cost a pick: use the old scorer
            pick = choose_pick(legal_actions, memory, {**P, "draft_v2": 0.0})
            pick.notes.append(f"v2 search failed ({type(exc).__name__}: {exc}); old scorer used")
            return pick
    offered = [(a, memory.pool_cards.get(card_id_of(a), {})) for a in legal_actions
               if a.action_id.startswith(PREFIX) or "draft" in a.action_id]
    if not offered:
        first = legal_actions[0]
        return ScoredCard(first, card_id_of(first), first.label or first.action_id, 0.0, ["not a draft list"])

    mine = [profile(c) for c in memory.my_cards.values()]
    theirs = [profile(c) for c in memory.opp_cards.values()]
    scored: list[ScoredCard] = []
    for action, card in offered:
        score, notes = score_card(card, memory, my_team=mine, opp_team=theirs, params=P)
        scored.append(ScoredCard(action, card_id_of(action), str(card.get("species") or card_id_of(action)), score, notes))

    order = sorted(scored, key=lambda s: (-s.score, legal_actions.index(s.action)))
    best = order[0]

    # 5. denial: when it's close, take what the opponent needs more.
    if len(order) > 1 and theirs and len(theirs) < TOTAL_PICKS:
        runner = order[1]
        if best.score - runner.score < P["draft_denial_margin"]:
            def value_to_them(entry: ScoredCard) -> float:
                card = memory.pool_cards.get(entry.card_id, {})
                s, _ = score_card(card, memory, my_team=theirs, opp_team=mine, params=P)
                return s
            if value_to_them(runner) > value_to_them(best) + P["draft_denial_gap"]:
                runner.notes.append(f"denial over {best.species}")
                best = runner
    return best


# -- v2: counter-aware draft search (``draft_v2`` = 1) ----------------------------------------------------------------
#
# Plain language: the score above judges a card against the picks ALREADY made. A counter-picking opponent wins the
# draft on the picks still to come: if we pass Gholdengo, they take it next and nothing we hold beats it. So v2 asks,
# for every card we could take now: "if I take this, what will they take back, and how do the two finished teams
# compare?" It plays the rest of the draft out in snake order, both sides picking greedily, and for the opponent's
# very next pick it tries their few best options and leans toward the one that hurts us most. The card whose
# finished drafts look best wins. Denial falls out on its own: a card they would use against us scores well for us
# because the rollout no longer hands it to them.
#
# Their replies come from two readings of the opponent (``DraftSearch._replies``): our own team model in
# agent/pokemon/matchup.py (which counts answers to each opposing card, so a greedy opponent IS a counter-picker) and
# the search-bot-style counter-picker in agent/pokemon/counterpick.py, which reproduces the Oct 7 opponent's public
# reasons ("Incineroar: Fake Out pressure, Intimidate, beats Tornadus"). The team model alone held their actual next
# pick in 8 of 20 live positions and never ranked Incineroar first where they took it; each reply is played out to
# the end with the opponent that proposed it.
#
# Cost: the pairwise races for an 18-card pool take ~0.05 s (cached for the whole match); the search scores a few
# thousand team pairs at ~30 microseconds each, so a pick takes well under a second.

FIRST_SLOTS = (0, 3, 4, 7, 8, 11)  # 0-based pick indices of the first drafter (picks 1, 4, 5, 8, 9, 12)
_BOARDS: dict[tuple, object] = {}  # (card fingerprints, knobs) -> matchup.Board, reused across a match's picks

# -- secret tie-breaks ------------------------------------------------------------------------------------------------
#
# The draft and Team Preview are deterministic from public state, so a reader of this public repository could
# precompute our choices. Among candidates the model cannot tell apart (within ``draft_tiebreak_eps`` /
# ``lineup_tiebreak_eps`` of the best, about the model's own noise), the choice follows a seed drawn from ``secrets``
# at the match's first decision, never from public state. The seed is written into the decision notes (so the
# decision log carries it and a post-mortem can replay the choice); it is kept on the match memory and, for Team
# Preview, found again through the match's card dict.

_SEEDS: dict[int, int] = {}  # id(memory.my_cards) -> this match's seed


def tiebreak_seed(my_cards: dict, memory: MatchMemory | None = None) -> int:
    """This match's secret tie-break seed: the one the memory already carries, else a fresh ``secrets`` draw."""
    key = id(my_cards)
    seed = getattr(memory, "tiebreak_seed", None) if memory is not None else None
    if seed is None:
        seed = _SEEDS.get(key)
    if seed is None:
        seed = secrets.randbits(64)
    if len(_SEEDS) > 256:
        _SEEDS.clear()
    _SEEDS[key] = seed
    if memory is not None and getattr(memory, "tiebreak_seed", None) != seed:
        try:
            memory.tiebreak_seed = seed
        except Exception:  # noqa: BLE001 — a memory that refuses the attribute still gets the registry
            pass
    return seed


def known_seed(my_cards: dict) -> int | None:
    """The match's seed if a draft decision registered one for these cards, else None (a Team Preview outside a
    match, as in the simulator's lineup replays, stays deterministic)."""
    return _SEEDS.get(id(my_cards))


def tiebreak(values: list[float], eps: float, seed: int | None, context: str) -> int:
    """The index of the chosen candidate: the best, unless others are within ``eps`` win chance of it, in which case
    one of those is drawn with ``random.Random(f"{seed}:{context}")`` (the same answer for the same seed and
    context; a different seed may choose differently only inside the window). ``values`` are log-odds."""
    from . import matchup

    if not values:
        raise ValueError("no candidates")
    best = max(values)
    if seed is None or eps <= 0:
        return values.index(best)
    top = matchup.win_probability(best)
    tied = [i for i, v in enumerate(values) if top - matchup.win_probability(v) <= eps]
    if len(tied) == 1:
        return tied[0]
    return random.Random(f"{seed}:{context}").choice(tied)


def pick_order(picks_made: int, we_are_first: bool) -> list[str]:
    """Who picks next, from pick index ``picks_made`` to the end: 'us' / 'them'."""
    return ["us" if (q in FIRST_SLOTS) == we_are_first else "them" for q in range(picks_made, 2 * TOTAL_PICKS)]


@dataclass
class _Plan:
    value: float  # log-odds that our finished team beats theirs, aggregated over their likely responses
    responses: list[tuple[str, float]]  # their best replies to this pick and the value after each
    final_us: list[int]
    final_them: list[int]


GREEDY, COUNTER = "greedy", "counter"  # the two readings of the opponent a reply can come from

# -- the first-pick policy (B1) -----------------------------------------------------------------------------------------
#
# The lookahead values a first pick by the six it imagines around it, which is where the Oct 7 first picks went wrong
# (Tornadus, Landorus-Therian: both "beaten" by most of the pool, both counter-picked at once). On our first two picks
# the search value also carries what a strong drafter reads off the pool itself: the card's duel margin over the pool
# it will face, how scarce its doubles role is in that pool (Fake Out, Intimidate, speed control, redirection, a Ghost
# for Fake Out immunity), how many pool cards beat it that the opponent can still take, and a hand guard list for the
# live catalog (what the Oct 7 opponent took first against what we took first).

FIRST_PICK_GUARD: dict[str, int] = {"Gholdengo": 1, "Incineroar": 1, "Archaludon": 1, "Iron Hands": 1, "Sylveon": 1, "Flutter Mane": 1,
                                    "Rillaboom": 1, "Amoonguss": 1, "Tornadus": -1, "Whimsicott": -1, "Baxcalibur": -1, "Iron Bundle": -1}
SPEED_CONTROL_ROLES = frozenset({"tailwind", "trickroom", "speed_drop"})
BEATS_MARGIN = 0.2

# A first-pick table (log-odds, the search value's unit), for calibrating the policy from simulator series without
# touching the search: data/first_pick_adjust.json ships all zeros, which is the policy exactly as before. Sections:
# "adjust" (species -> log-odds, either draft order), "adjust_we_first" / "adjust_they_first" (added in that order
# only), and "pairs" ("A + B" -> log-odds, added to B at our second pick when A is our first pick, and vice versa:
# the Oct 9 series analysis found a pair of picks losing where each card alone did not). Every entry follows the
# policy's schedule (full at our first pick, half at our second, nothing after) and draft_first_adjust_w.
# DRAFT_FIRST_PICK_ADJUST=<path to a file of the same shape> swaps in another table for one process (a simulator A/B
# arm); that file must read cleanly or the pick raises (so a mistyped path cannot quietly turn an arm into the
# control). A missing or unreadable shipped table counts as all zeros.
FIRST_PICK_ADJUST_FILE = data.DATA_DIR / "first_pick_adjust.json"
FIRST_PICK_ADJUST_ENV = "DRAFT_FIRST_PICK_ADJUST"


@dataclass(frozen=True)
class FirstPickTable:
    adjust: dict[str, float] = field(default_factory=dict)  # species id -> log-odds, either draft order
    we_first: dict[str, float] = field(default_factory=dict)  # ... added when we draft first
    they_first: dict[str, float] = field(default_factory=dict)  # ... added when they draft first
    pairs: dict[frozenset, float] = field(default_factory=dict)  # {species id, species id} -> log-odds

    def value(self, species: str, *, we_first: bool | None = None, team: tuple[str, ...] | list[str] = ()) -> float:
        """The table's log-odds for taking ``species`` now, with ``team`` (species) already ours."""
        sid = data.to_id(species)
        v = self.adjust.get(sid, 0.0)
        if we_first is not None:
            v += (self.we_first if we_first else self.they_first).get(sid, 0.0)
        for mate in team:
            v += self.pairs.get(frozenset((sid, data.to_id(mate))), 0.0)
        return v

    def entries(self) -> int:
        """How many non-zero entries the table holds (0 for the shipped table)."""
        return sum(1 for section in (self.adjust, self.we_first, self.they_first, self.pairs) for v in section.values() if v)


_ADJUST_TABLES: dict[str, FirstPickTable] = {}


def _table_section(raw: dict, key: str) -> dict[str, float]:
    entries = raw.get(key) or {}
    if not isinstance(entries, dict):
        raise ValueError(f"{key!r} must map species to log-odds")
    out = {data.to_id(k): float(v) for k, v in entries.items() if not str(k).startswith("_")}
    bad = [k for k, v in out.items() if not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite values in {key!r}: {bad}")
    return out


def first_pick_adjust(path: str | None = None) -> FirstPickTable:
    """The first-pick table in use. ``path`` (else $DRAFT_FIRST_PICK_ADJUST, else the shipped file) is read once per
    process; keys are matched as species ids (data.to_id), so "Iron Hands" and "iron-hands" are the same card."""
    chosen = path or os.environ.get(FIRST_PICK_ADJUST_ENV) or ""
    source = chosen or str(FIRST_PICK_ADJUST_FILE)
    table = _ADJUST_TABLES.get(source)
    if table is None:
        try:
            raw = json.loads(Path(source).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("expected a JSON object with \"adjust\" / \"adjust_we_first\" / \"adjust_they_first\" / \"pairs\"")
            raw_pairs = raw.get("pairs") or {}
            if not isinstance(raw_pairs, dict):
                raise ValueError("'pairs' must map \"A + B\" to log-odds")
            pairs: dict[frozenset, float] = {}
            for key, v in raw_pairs.items():
                if str(key).startswith("_"):
                    continue
                names = [data.to_id(part) for part in str(key).split("+")]
                if len(names) != 2 or len(set(names)) != 2 or not all(names):
                    raise ValueError(f"pair {key!r} must name two different species as \"A + B\"")
                if not math.isfinite(float(v)):
                    raise ValueError(f"non-finite value for pair {key!r}")
                pairs[frozenset(names)] = float(v)
            table = FirstPickTable(_table_section(raw, "adjust"), _table_section(raw, "adjust_we_first"),
                                   _table_section(raw, "adjust_they_first"), pairs)
        except (OSError, ValueError, TypeError) as exc:
            if chosen:
                raise ValueError(f"first-pick table {source!r} ({FIRST_PICK_ADJUST_ENV} or path) unusable: {exc}") from exc
            table = FirstPickTable()
        _ADJUST_TABLES[source] = table
    return table


def first_pick_terms(board, c: int, pool: list[int], *, we_first: bool | None = None,
                     team: tuple[str, ...] | list[str] = ()) -> dict[str, float]:
    """The policy's readings of candidate ``c`` against the remaining ``pool`` (which still holds ``c``): four from the
    pool and the guard list, plus the first-pick table's entry for it (``first_pick_adjust``: its draft-order section
    when ``we_first`` is known, its pair entries with the species in ``team``, our picks so far)."""
    units = board.units
    rest = [j for j in pool if j != c]
    me = units[c]

    def has(j: int, role: str) -> bool:
        u = units[j]
        if role == "ghost":
            return "ghost" in u.types
        if role == "speed":
            return bool(u.roles & SPEED_CONTROL_ROLES)
        return role in u.roles

    scarcity = 0.0
    for role in ("fakeout", "intimidate", "speed", "redirect", "ghost"):
        if has(c, role):
            scarcity += 1.0 / sum(1 for j in pool if has(j, role))
    return {
        "strength": sum(board.base[c][j] for j in rest) / len(rest) if rest else 0.0,  # duel margin over the pool
        "scarcity": scarcity,
        "exposure": sum(1 for j in rest if board.base[j][c] >= BEATS_MARGIN),  # pool cards that beat it, theirs to take
        "guard": float(FIRST_PICK_GUARD.get(me.species, 0)),
        "adjust": first_pick_adjust().value(me.species, we_first=we_first, team=team),
    }


def first_pick_bonus(board, c: int, pool: list[int], params: dict, *, we_first: bool | None = None,
                     team: tuple[str, ...] | list[str] = ()) -> tuple[float, dict[str, float]]:
    """(log-odds to add to the search value, the terms) for a first-round pick."""
    t = first_pick_terms(board, c, pool, we_first=we_first, team=team)
    bonus = (float(params.get("draft_first_strength_w", 0.0)) * t["strength"]
             + float(params.get("draft_first_scarcity_w", 0.0)) * t["scarcity"]
             - float(params.get("draft_first_exposure_w", 0.0)) * t["exposure"] / 6.0
             + float(params.get("draft_first_guard_w", 0.0)) * t["guard"])
    if t["adjust"]:  # the shipped table is all zeros: the sum above is then the bonus, bit for bit
        bonus += float(params.get("draft_first_adjust_w", 0.0)) * t["adjust"]
    return bonus, t


class DraftSearch:
    """Snake-draft lookahead on a ``matchup.Board`` (indices into one list of cards)."""

    def __init__(self, board, *, lookahead: int = 3, minimax: float = 0.5, reply_depth: int = 1, counter_replies: int = 3,
                 counter_prior: float = 0.0, four_eval: bool = True, their_fours: int = 3, robust: float = 0.25,
                 hole_w: float = 0.0, sleep_w: float = 0.0) -> None:
        self.board = board
        self.lookahead = max(0, int(lookahead))
        self.minimax = max(0.0, min(1.0, float(minimax)))
        self.reply_depth = max(1, int(reply_depth))  # how many of their consecutive picks to branch on (snake doubles: 2)
        self.counter_replies = max(0, int(counter_replies))  # their best replies by the counter-picker's scorer to try too
        self.counter_prior = float(counter_prior)  # x our tier rating: the counter-picker's strength prior
        self.four_eval = bool(four_eval)  # value a finished draft by the four we would bring (Team Preview's criterion)
        self.their_fours = max(1, int(their_fours))  # ... against this many of their likeliest fours
        self.robust = max(0.0, min(1.0, float(robust)))  # weight on the worst of those fours (lineup_v2_robust)
        self.hole_w = float(hole_w)  # log-odds per coverage hole (Board.holes): their card none of our four can hit for 40%
        self.sleep_w = float(sleep_w)  # x Board.sleep_reach of their four into ours, charged in the four-against-four value
        self.items = board.items
        self._counter_base: list[float] | None = None  # per card: role points + raw strength (counterpick.counter_value)
        self._beats: dict[tuple[int, int], float | None] = {}
        self._finished: dict[tuple, tuple[float, tuple[int, ...]]] = {}

    def legal(self, team: list[int], pool: list[int]) -> list[int]:
        held = {self.items[i] for i in team if self.items[i]}
        return [c for c in pool if not self.items[c] or self.items[c] not in held]

    def greedy(self, side: str, us: list[int], them: list[int], pool: list[int]) -> list[tuple[float, int]]:
        """Every legal pick for ``side``, best first, valued by the team model right now."""
        mine = us if side == "us" else them
        options = []
        for c in self.legal(mine, pool):
            if side == "us":
                options.append((self.board.score(us + [c], them), c))
            else:
                options.append((-self.board.score(us, them + [c]), c))
        options.sort(key=lambda t: (-t[0], pool.index(t[1])))
        return options

    def counter(self, us: list[int], them: list[int], pool: list[int]) -> list[tuple[float, int]]:
        """Every legal pick for THEM, best first, as a search-bot-style counter-picker values it against our team
        (``counterpick.counter_value``: role points, "beats X" for each of our cards, raw strength)."""
        from . import counterpick

        units = self.board.units
        if self._counter_base is None:
            from . import matchup

            tiers = matchup.load_tiers() if self.counter_prior else {}
            self._counter_base = [counterpick.counter_value(u, [], strength=self.counter_prior * float((tiers.get(u.key) or {}).get("rating", 0.0)))[0]
                                  for u in units]
        options = []
        for c in self.legal(them, pool):
            score = self._counter_base[c]
            for i in us:
                value = self._beats.get((c, i))
                if (c, i) not in self._beats:
                    value = self._beats[(c, i)] = counterpick.counter_beats(units[c], units[i])
                if value is not None:
                    score += counterpick.BEAT_POINTS + value
            options.append((score, c))
        options.sort(key=lambda t: (-t[0], pool.index(t[1])))
        return options

    def rollout(self, us: list[int], them: list[int], pool: list[int], order: list[str],
                opponent: str = GREEDY, include: int | None = None) -> tuple[float, list[int], list[int]]:
        """The rest of the draft played out greedily: our picks by the team model, theirs by ``opponent`` (the team
        model, or the counter-picker's scorer). The value is ``finished``: the four we would bring (one holding
        ``include``, the pick under test), not all six."""
        us, them, pool = list(us), list(them), list(pool)
        for side in order:
            if side == "them" and opponent == COUNTER:
                options = self.counter(us, them, pool)
            else:
                options = self.greedy(side, us, them, pool)
            if not options:
                continue  # Item Clause left this side nothing (the server would auto-pick); skip it here
            c = options[0][1]
            (us if side == "us" else them).append(c)
            pool.remove(c)
        return self.finished(us, them, include)[0], us, them

    def finished(self, us: list[int], them: list[int], include: int | None = None) -> tuple[float, tuple[int, ...]]:
        """Value of a finished draft and the four it rests on. Team Preview brings four, so a card outside the four we
        would bring is worth nothing here beyond having kept it from them (Oct 8: in four live games an early pick
        stayed home). Mirrors lineup.shortlist_v2: our best four by (1 - robust) x the mean + robust x the worst of
        Board.score4 (the preview weights) against their ``their_fours`` likeliest fours (the strongest against our
        six); the 6-against-6 team score when ``four_eval`` is off. With ``include`` (the pick being valued) only
        fours holding that card count, so a pick the finished six would bench is worth the best four it still fits
        in, never the six built around it. (Their fitted per-card terms stay in here: taking them out, as Team
        Preview does for their bring, moved the bcbaabe2 pick-3 state off Archaludon.) With ``sleep_w`` each
        four-against-four value is charged their sleepers' reach into our four (Board.sleep_reach: their Spore against
        our sleep-proof cards) and their likeliest fours are picked with their reach into our six. One-sided: our own
        sleepers earn nothing (the brief's scope; the Oct 9 review measured that half moving 10 live picks toward our
        own Amoonguss / Torkoal)."""
        key = (tuple(sorted(us)), tuple(sorted(them)), include)
        cached = self._finished.get(key)
        if cached is not None:
            return cached
        if not self.four_eval or len(us) < 4 or len(them) < 4:
            result = (self.board.score(us, them), tuple(us))
        else:
            score4, holes, hole_w = self.board.score4, self.board.holes, self.hole_w
            sleep_w, reach = self.sleep_w if self.board.sleepers else 0.0, self.board.sleep_reach
            theirs = sorted(combinations(them, 4), key=lambda g: -(score4(list(g), us) + sleep_w * reach(list(g), us)))[: self.their_fours]
            theirs = [list(g) for g in theirs]
            best: tuple[float, tuple[int, ...]] | None = None
            for four in combinations(us, 4):
                if include is not None and include not in four:
                    continue
                f = list(four)
                values = [score4(f, g) - sleep_w * reach(g, f) - hole_w * (len(holes(f, g)) - len(holes(g, f))) for g in theirs]
                value = (1 - self.robust) * sum(values) / len(values) + self.robust * min(values)
                if best is None or value > best[0]:
                    best = (value, four)
            result = best
        self._finished[key] = result
        return result

    def reply_set(self, us: list[int], them: list[int], pool: list[int]) -> list[tuple[int, str]]:
        """Their candidate next picks: the team model's best ``lookahead`` and the counter-picker's best
        ``counter_replies``, interleaved (model first), each tagged with the opponent that proposed it."""
        by_model = [c for _, c in self.greedy("them", us, them, pool)[: self.lookahead]]
        by_counter = [c for _, c in self.counter(us, them, pool)[: self.counter_replies]] if self.counter_replies else []
        out: list[tuple[int, str]] = []
        seen: set[int] = set()
        for k in range(max(len(by_model), len(by_counter))):
            for source, picks in ((GREEDY, by_model), (COUNTER, by_counter)):
                if k < len(picks) and picks[k] not in seen:
                    seen.add(picks[k])
                    out.append((picks[k], source))
        return out

    def plan(self, candidate: int, us: list[int], them: list[int], pool: list[int], order: list[str]) -> _Plan:
        """Value of taking ``candidate`` now (``order[0]`` is this pick, ours)."""
        us1 = us + [candidate]
        pool1 = [c for c in pool if c != candidate]
        rest = order[1:]
        # Our own picks before their next one are filled greedily (snake double picks).
        while rest and rest[0] == "us":
            options = self.greedy("us", us1, them, pool1)
            if not options:
                rest = rest[1:]
                continue
            c = options[0][1]
            us1, pool1, rest = us1 + [c], [x for x in pool1 if x != c], rest[1:]
        if not rest or self.lookahead == 0:
            value, fu, ft = self.rollout(us1, them, pool1, rest, include=candidate)
            return _Plan(value, [], fu, ft)
        outcomes = self._replies(us1, them, pool1, rest, self.reply_depth, include=candidate)
        if not outcomes:
            value, fu, ft = self.rollout(us1, them, pool1, rest[1:], include=candidate)
            return _Plan(value, [], fu, ft)
        worst = min(outcomes, key=lambda o: o[0])
        mean = sum(o[0] for o in outcomes) / len(outcomes)
        value = self.minimax * worst[0] + (1 - self.minimax) * mean
        firsts: dict[int, float] = {}
        for o in outcomes:  # per first reply, the worst finish it leads to
            firsts[o[1]] = min(firsts.get(o[1], o[0]), o[0])
        return _Plan(value, [(self.board.units[r].species, round(v, 3)) for r, v in firsts.items()], worst[2], worst[3])

    def _replies(self, us: list[int], them: list[int], pool: list[int], rest: list[str], depth: int,
                 include: int | None = None) -> list[tuple]:
        """Their best few picks at ``rest[0]`` by both readings of the opponent (``reply_set``; with depth 2, at
        their second consecutive pick too), each played out to the end with the opponent that proposed it:
        [(value, first reply, our final team, their final team)]."""
        outcomes = []
        for reply, source in self.reply_set(us, them, pool):
            them1, pool1 = them + [reply], [x for x in pool if x != reply]
            if depth > 1 and len(rest) > 1 and rest[1] == "them":
                for value, _, fu, ft in self._replies(us, them1, pool1, rest[1:], depth - 1, include):
                    outcomes.append((value, reply, fu, ft))
            else:
                value, fu, ft = self.rollout(us, them1, pool1, rest[1:], opponent=source, include=include)
                outcomes.append((value, reply, fu, ft))
        return outcomes


def _board_for(cards: list[dict], card_weight: float = 1.0, tier_adjust: float = 0.0):
    """One Board per (match pool): built on the first pick, reused for the rest of the draft."""
    from . import matchup

    units = [matchup.unit(c) for c in cards]
    key = (tuple(sorted(u.key for u in units)), card_weight, tier_adjust)
    board = _BOARDS.get(key)
    if board is None:
        if len(_BOARDS) > 64:
            _BOARDS.clear()
        board = matchup.Board(units, card_weight=card_weight, tier_adjust=tier_adjust)
        _BOARDS[key] = board
    return board


def _drafted_ids(memory: MatchMemory) -> set[str]:
    return {str(c.get("card_id")) for c in (*memory.my_cards.values(), *memory.opp_cards.values()) if c.get("card_id")}


def rank_picks_v2(legal_actions: list[LegalAction], memory: MatchMemory, params: dict | None = None) -> list[ScoredCard]:
    """Every offered card, best first, valued by the rest-of-draft lookahead (what ``choose_pick_v2`` chooses from;
    draft-only replays and tests read the whole ranking). Empty when nothing offered is a known card."""
    from . import matchup

    P = params or DEFAULTS
    offered = [(a, memory.pool_cards.get(card_id_of(a))) for a in legal_actions
               if a.action_id.startswith(PREFIX) or "draft" in a.action_id]
    offered = [(a, c) for a, c in offered if c]
    if not offered:
        return []
    drafted = _drafted_ids(memory)
    mine_cards = [c for c in memory.my_cards.values() if c.get("moves")]
    their_cards = [c for c in memory.opp_cards.values() if c.get("moves")]
    pool_cards = {cid: c for cid, c in memory.pool_cards.items() if cid not in drafted and c.get("moves")}
    for _, c in offered:
        pool_cards.setdefault(str(c.get("card_id")), c)
    everyone = mine_cards + their_cards + list(pool_cards.values())
    board = _board_for(everyone, float(P.get("draft_card_w", 1.0)), float(P.get("draft_tier_adjust", 1.0)))
    index = {u.key: i for i, u in enumerate(board.units)}
    us = [index[matchup.fingerprint(c)] for c in mine_cards]
    them = [index[matchup.fingerprint(c)] for c in their_cards]
    held = set(us) | set(them)
    pool = [i for i in dict.fromkeys(index[matchup.fingerprint(c)] for c in pool_cards.values()) if i not in held]
    offered_idx = [(a, index[matchup.fingerprint(c)]) for a, c in offered]
    picks_made = len(memory.my_cards) + len(memory.opp_cards)
    we_first = picks_made in FIRST_SLOTS  # it is our turn, so the slot says which drafter we are
    order = pick_order(picks_made, we_first)
    search = DraftSearch(board, lookahead=int(P.get("draft_lookahead", 3)), minimax=P.get("draft_minimax", 0.5),
                         reply_depth=int(P.get("draft_reply_depth", 1)), counter_replies=int(P.get("draft_counter_replies", 3)),
                         counter_prior=float(P.get("draft_counter_prior", 0.0)),
                         four_eval=bool(P.get("draft_four_eval", 1)), their_fours=int(P.get("draft_their_fours", 3)),
                         robust=float(P.get("lineup_v2_robust", 0.25)), hole_w=float(P.get("draft_hole_w", 0.0)),
                         sleep_w=float(P.get("draft_sleep_w", 0.0)))
    # Our first two picks also read the pool itself (first_pick_bonus): full weight on the first, half on the second.
    policy_w = float(P.get("draft_first_pick_w", 0.0)) * max(0.0, 1.0 - len(us) / 2.0)
    team = tuple(board.units[k].species for k in us)  # for the first-pick table's pair entries
    scored: list[tuple[_Plan, LegalAction, int, float, dict]] = []
    for action, i in offered_idx:
        if i in us:
            continue
        plan = search.plan(i, us, them, pool, order)
        bonus, terms = first_pick_bonus(board, i, pool, P, we_first=we_first, team=team) if policy_w else (0.0, {})
        plan.value += policy_w * bonus
        scored.append((plan, action, i, policy_w * bonus, terms))
    scored.sort(key=lambda t: (-t[0].value, legal_actions.index(t[1])))
    ranked: list[ScoredCard] = []
    for rank, (plan, action, i, bonus, terms) in enumerate(scored):
        me = board.units[i]
        notes = [f"v2: finished-draft win chance {100 * matchup.win_probability(plan.value):.0f}%"]
        if terms:
            notes.append(f"first-pick policy {bonus:+.2f} (duel margin over the pool {terms['strength']:+.2f}, role scarcity "
                         f"{terms['scarcity']:.2f}, beaten by {int(terms['exposure'])} pool cards, guard {terms['guard']:+.0f}"
                         + (f", table {terms['adjust']:+.2f}" if terms.get("adjust") else "") + ")")
        if plan.responses:
            notes.append("their likely replies: " + ", ".join(f"{s} ({100 * matchup.win_probability(v):.0f}%)" for s, v in plan.responses))
        answered = [board.units[j].species for j in them if board.base[i][j] >= 0.2]
        if answered:
            notes.append(f"beats {answered[:3]}")
        # Sleep (Board.sleep_reach; draft_sleep_w): their sleep-move users this card cannot be slept by, and ours.
        proof_vs = [board.units[j].species for j in them if board.sleep_r[j] and board.proof[j][i]]
        if proof_vs:
            notes.append(f"sleep-proof against their {', '.join(proof_vs)}")
        if board.sleep_r[i] and them:
            safe = [board.units[j].species for j in them if board.proof[i][j]]
            notes.append(f"its sleep move cannot touch their {', '.join(safe)}" if safe else "its sleep move reaches all their picks")
        threats = sorted(((board.base[j][i], j) for j in pool if j != i and board.base[j][i] >= BEATS_MARGIN), key=lambda t: -t[0])
        if rank == 0 and len(scored) > 1:
            runner = scored[1]
            notes.append(f"next best {board.units[runner[2]].species} ({100 * matchup.win_probability(runner[0].value):.0f}%)")
        if threats:  # biggest margin first: the one-hit KO they will take next is the first name, not the fourth
            notes.append("loses to " + ", ".join(f"{board.units[j].species} {m:+.2f}" for m, j in threats[:3]) + " still in the pool")
        final_them = [board.units[j].species for j in plan.final_them if j not in them]
        if final_them:
            notes.append(f"expected rest of their draft: {final_them}")
        if search.four_eval and len(plan.final_us) >= 4:
            _, four = search.finished(plan.final_us, plan.final_them, i)
            _, best4 = search.finished(plan.final_us, plan.final_them)
            notes.append(f"would bring {[board.units[k].species for k in four]} of {[board.units[k].species for k in plan.final_us]}"
                         + (f" (its best four benches {me.species}: {[board.units[k].species for k in best4]})" if i not in best4 else ""))
            gaps = board.holes(list(four), plan.final_them)
            if gaps:
                notes.append(f"coverage hole: nothing in that four hits {[board.units[k].species for k in gaps]} for {matchup.HOLE_THRESHOLD:.0f}%")
        ranked.append(ScoredCard(action, card_id_of(action), me.species, round(plan.value, 3), notes, exposure=len(threats)))
    return ranked


# -- exposure-first near-ties (``draft_tiebreak_exposure``) --------------------------------------------------------------
#
# The Oct 9 draft review's natural experiment: among picks inside the tie window, the card fewer undrafted pool cards
# beat (its exposure: the counters the opponent can still take) won 61 % of 171 discordant pairs, while the search's
# own value inside the window picked the winner 44 %. With the key on, the window tiebreak() builds is cut to its
# lowest-exposure cards and the secret seed draws among those. Search values are untouched, so by the model's measure
# a decision gives up at most eps. tiebreak() itself is unchanged: Team Preview shares it.


def tiebreak_exposure(ranked: list[ScoredCard], eps: float, seed: int | None, context: str) -> tuple[int, list[ScoredCard]]:
    """``tiebreak`` over the scores of ``ranked`` with its window cut to the cards of lowest ``exposure`` before the
    draw: the same window (same values, same ``eps`` test) and the same ``random.Random(f"{seed}:{context}")`` draw,
    made among the cards kept. Returns the chosen index and the window's cards the cut set aside: none when the window
    holds one card or all its cards have the same exposure, and then the answer is ``tiebreak``'s."""
    from . import matchup

    values = [s.score for s in ranked]
    if not values:
        raise ValueError("no candidates")
    best = max(values)
    if seed is None or eps <= 0:
        return values.index(best), []
    top = matchup.win_probability(best)
    tied = [i for i, v in enumerate(values) if top - matchup.win_probability(v) <= eps]
    if len(tied) == 1:
        return tied[0], []
    fewest = min(ranked[i].exposure for i in tied)
    kept = [i for i in tied if ranked[i].exposure == fewest]
    set_aside = [ranked[i] for i in tied if ranked[i].exposure != fewest]
    if len(kept) == 1:
        return kept[0], set_aside
    return random.Random(f"{seed}:{context}").choice(kept), set_aside


def choose_pick_v2(legal_actions: list[LegalAction], memory: MatchMemory, params: dict | None = None) -> ScoredCard:
    """Counter-aware pick: the offered card whose rest-of-draft lookahead gives our finished team the best odds;
    among cards within ``draft_tiebreak_eps`` of the best, the match's secret seed decides (``tiebreak``), among the
    window's fewest-counters cards only when ``draft_tiebreak_exposure`` > 0 (``tiebreak_exposure``)."""
    from . import matchup

    P = params or DEFAULTS
    ranked = rank_picks_v2(legal_actions, memory, P)
    if not ranked:
        return choose_pick(legal_actions, memory, {**P, "draft_v2": 0.0})
    seed = tiebreak_seed(memory.my_cards, memory)
    picks_made = len(memory.my_cards) + len(memory.opp_cards)
    eps = float(P.get("draft_tiebreak_eps", 0.0))
    set_aside: list[ScoredCard] = []
    if float(P.get("draft_tiebreak_exposure", 0.0)) > 0:
        index, set_aside = tiebreak_exposure(ranked, eps, seed, f"pick:{picks_made}")
        chosen = ranked[index]
    else:
        chosen = ranked[tiebreak([s.score for s in ranked], eps, seed, f"pick:{picks_made}")]
    best = ranked[0]
    tied = [s.species for s in ranked if s is not chosen and matchup.win_probability(best.score) - matchup.win_probability(s.score) <= eps]
    chosen.notes.append(f"tie-break seed {seed:#x}" + (f": {chosen.species} over {tied} (within {100 * eps:.0f}% win chance)" if tied else ""))
    if set_aside:  # exposure cut the window: it decided the pick, or narrowed the seed's draw
        by_seed = ranked[tiebreak([s.score for s in ranked], eps, seed, f"pick:{picks_made}")]
        n = chosen.exposure
        chosen.notes.append(f"exposure tie-break: {chosen.species} is beaten by {n} undrafted pool card{'' if n == 1 else 's'}, the fewest in the "
                            "window; set aside " + ", ".join(f"{s.species} ({s.exposure})" for s in set_aside)
                            + ("" if by_seed is chosen else f"; the seed alone would take {by_seed.species}"))
    return chosen
