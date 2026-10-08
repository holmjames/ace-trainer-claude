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

from dataclasses import dataclass, field

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
# compare?" It plays the rest of the draft out in snake order, both sides picking greedily by the team model in
# agent/pokemon/matchup.py (which counts answers to each opposing card, so a greedy opponent IS a counter-picker),
# and for the opponent's very next pick it tries their few best options and leans toward the one that hurts us most.
# The card whose finished drafts look best wins. Denial falls out on its own: a card they would use against us
# scores well for us because the rollout no longer hands it to them.
#
# Cost: the pairwise races for an 18-card pool take ~0.05 s (cached for the whole match); the search scores a few
# thousand team pairs at ~30 microseconds each, so a pick takes well under a second.

FIRST_SLOTS = (0, 3, 4, 7, 8, 11)  # 0-based pick indices of the first drafter (picks 1, 4, 5, 8, 9, 12)
_BOARDS: dict[tuple, object] = {}  # (card fingerprints, knobs) -> matchup.Board, reused across a match's picks


def pick_order(picks_made: int, we_are_first: bool) -> list[str]:
    """Who picks next, from pick index ``picks_made`` to the end: 'us' / 'them'."""
    return ["us" if (q in FIRST_SLOTS) == we_are_first else "them" for q in range(picks_made, 2 * TOTAL_PICKS)]


@dataclass
class _Plan:
    value: float  # log-odds that our finished team beats theirs, aggregated over their likely responses
    responses: list[tuple[str, float]]  # their best replies to this pick and the value after each
    final_us: list[int]
    final_them: list[int]


class DraftSearch:
    """Snake-draft lookahead on a ``matchup.Board`` (indices into one list of cards)."""

    def __init__(self, board, *, lookahead: int = 3, minimax: float = 0.5, reply_depth: int = 1) -> None:
        self.board = board
        self.lookahead = max(0, int(lookahead))
        self.minimax = max(0.0, min(1.0, float(minimax)))
        self.reply_depth = max(1, int(reply_depth))  # how many of their consecutive picks to branch on (snake doubles: 2)
        self.items = board.items

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

    def rollout(self, us: list[int], them: list[int], pool: list[int], order: list[str]) -> tuple[float, list[int], list[int]]:
        us, them, pool = list(us), list(them), list(pool)
        for side in order:
            options = self.greedy(side, us, them, pool)
            if not options:
                continue  # Item Clause left this side nothing (the server would auto-pick); skip it here
            c = options[0][1]
            (us if side == "us" else them).append(c)
            pool.remove(c)
        return self.board.score(us, them), us, them

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
            value, fu, ft = self.rollout(us1, them, pool1, rest)
            return _Plan(value, [], fu, ft)
        outcomes = self._replies(us1, them, pool1, rest, self.reply_depth)
        if not outcomes:
            value, fu, ft = self.rollout(us1, them, pool1, rest[1:])
            return _Plan(value, [], fu, ft)
        worst = min(outcomes, key=lambda o: o[0])
        mean = sum(o[0] for o in outcomes) / len(outcomes)
        value = self.minimax * worst[0] + (1 - self.minimax) * mean
        firsts: dict[int, float] = {}
        for o in outcomes:  # per first reply, the worst finish it leads to
            firsts[o[1]] = min(firsts.get(o[1], o[0]), o[0])
        return _Plan(value, [(self.board.units[r].species, round(v, 3)) for r, v in firsts.items()], worst[2], worst[3])

    def _replies(self, us: list[int], them: list[int], pool: list[int], rest: list[str], depth: int) -> list[tuple]:
        """Their best few picks at ``rest[0]`` (and, with depth 2, at their second consecutive pick too), each played
        out to the end: [(value, first reply, our final team, their final team)]."""
        outcomes = []
        for _, reply in self.greedy("them", us, them, pool)[: self.lookahead]:
            them1, pool1 = them + [reply], [x for x in pool if x != reply]
            if depth > 1 and len(rest) > 1 and rest[1] == "them":
                for value, _, fu, ft in self._replies(us, them1, pool1, rest[1:], depth - 1):
                    outcomes.append((value, reply, fu, ft))
            else:
                value, fu, ft = self.rollout(us, them1, pool1, rest[1:])
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
                         reply_depth=int(P.get("draft_reply_depth", 1)))
    scored: list[tuple[_Plan, LegalAction, int]] = []
    for action, i in offered_idx:
        if i in us:
            continue
        scored.append((search.plan(i, us, them, pool, order), action, i))
    scored.sort(key=lambda t: (-t[0].value, legal_actions.index(t[1])))
    ranked: list[ScoredCard] = []
    for rank, (plan, action, i) in enumerate(scored):
        me = board.units[i]
        notes = [f"v2: finished-draft win chance {100 * matchup.win_probability(plan.value):.0f}%"]
        if plan.responses:
            notes.append("their likely replies: " + ", ".join(f"{s} ({100 * matchup.win_probability(v):.0f}%)" for s, v in plan.responses))
        answered = [board.units[j].species for j in them if board.base[i][j] >= 0.2]
        if answered:
            notes.append(f"beats {answered[:3]}")
        threats = [board.units[j].species for j in pool if j != i and board.base[j][i] >= 0.2]
        if rank == 0 and len(scored) > 1:
            runner = scored[1]
            notes.append(f"next best {board.units[runner[2]].species} ({100 * matchup.win_probability(runner[0].value):.0f}%)")
        if threats:
            notes.append(f"loses to {threats[:3]} still in the pool")
        final_them = [board.units[j].species for j in plan.final_them if j not in them]
        if final_them:
            notes.append(f"expected rest of their draft: {final_them}")
        ranked.append(ScoredCard(action, card_id_of(action), me.species, round(plan.value, 3), notes))
    return ranked


def choose_pick_v2(legal_actions: list[LegalAction], memory: MatchMemory, params: dict | None = None) -> ScoredCard:
    """Counter-aware pick: the offered card whose rest-of-draft lookahead gives our finished team the best odds."""
    P = params or DEFAULTS
    ranked = rank_picks_v2(legal_actions, memory, P)
    if not ranked:
        return choose_pick(legal_actions, memory, {**P, "draft_v2": 0.0})
    return ranked[0]
