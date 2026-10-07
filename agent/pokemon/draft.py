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
