"""A the search bot-style counter-picking drafter, for the simulator.

Live, the contestant "the search bot" beat us 0-4 with a drafter that counter-picks in the open. Its public pick
reasons read like "Incineroar: Fake Out pressure, Intimidate, beats Landorus-Therian", "Gholdengo: spread damage,
beats Baxcalibur, Landorus-Therian", "Amoonguss: redirection, beats Tornadus, Hatterene", "Kingambit: strong
attacker." This agent drafts the same way so our drafters can be measured against it for free:

- every offered card earns role points (Tailwind / Trick Room, Fake Out, Intimidate, redirection, spread damage),
  points for each card the OTHER side has already drafted that it "beats", and a little raw strength;
- "beats X", as inferred from their reasons (several of which no type chart explains, e.g. Incineroar "beats"
  Landorus-Therian and Tornadus): it wins the 1-on-1 race against X with a cushion (``matchup.duel``), or it
  carries the tool that shuts X's job down: Fake Out / Taunt / Encore / Spore against a Tailwind, Trick Room,
  screens or redirection user, and Intimidate against a physical attacker that Intimidate works on;
- greedy: it reacts to picks already made and never looks ahead (their first picks were always role picks:
  Incineroar twice, Dragonite once).

Team Preview and battle are our own code brain, so a match against it isolates the draft. Its public reason is the
same shape as the search bot's, which makes simulator logs easy to compare with the live transcripts.
"""

from __future__ import annotations

from agent.agent import PokemonAgent
from agent.pokemon import draft as draft_rules
from agent.pokemon import matchup

ROLE_POINTS = {"tailwind": 3.0, "trickroom": 2.0, "fakeout": 2.5, "intimidate": 1.5, "redirect": 2.0, "spread": 1.0}
ROLE_TAGS = [("fakeout", "Fake Out pressure"), ("intimidate", "Intimidate"), ("tailwind", "Tailwind"), ("trickroom", "Trick Room"),
             ("redirect", "redirection"), ("spread", "spread damage")]
SHUTDOWN_TOOLS = {"fakeout", "taunt", "encore", "spore", "sleeppowder"}
SUPPORT_JOBS = {"tailwind", "trickroom", "screens", "redirect"}
BEAT_MARGIN = 0.2
INTIMIDATE_PROOF = matchup.INTIMIDATE_IMMUNE | {"defiant", "competitive", "guarddog"}
BEAT_POINTS = 2.0


def tool_works(tool: str, user: matchup.Unit, target: matchup.Unit) -> bool:
    """Whether one shutdown tool does its job on ``target``."""
    if target.ability in ("goodasgold", "magicbounce") and tool != "fakeout":
        return False  # status moves bounce off or fail
    if tool == "fakeout":
        return not ("ghost" in target.types or target.ability in ("innerfocus", "shielddust") or target.item == "covertcloak")
    if tool in ("taunt", "encore"):
        if target.item == "mentalherb" or target.ability in ("oblivious", "aromaveil"):
            return False
        return not (user.ability == "prankster" and "dark" in target.types)
    if tool in ("spore", "sleeppowder"):
        return not ("grass" in target.types or target.item == "safetygoggles"
                    or target.ability in ("overcoat", "insomnia", "vitalspirit", "sweetveil"))
    return False


def counter_beats(a: matchup.Unit, b: matchup.Unit) -> float | None:
    """How convincingly ``a`` 'beats' ``b`` (None if it does not): the race margin, or a flat 0.3 when the win
    comes from a tool that shuts ``b``'s job down rather than from the race."""
    margin = matchup.duel(a, b).margin
    if margin >= BEAT_MARGIN:
        return margin
    tools = {m["id"] for m in a.mon.moves} & SHUTDOWN_TOOLS
    if b.roles & SUPPORT_JOBS and any(tool_works(t, a, b) for t in tools):
        return 0.3
    if "intimidate" in a.roles and b.ability not in INTIMIDATE_PROOF and b.item != "clearamulet":
        # Their reasons credit Intimidate with "beating" a physical attacker even where the race says otherwise
        # (Incineroar "beats" Landorus-Therian).
        physical = [m for m in b.attacks if m.get("category") == "Physical"]
        if physical and len(physical) * 2 >= len(b.attacks):
            return 0.3
    return None


def counter_score(card: dict, theirs: list[dict], *, strength: float = 0.0) -> tuple[float, str]:
    """(score, public reason) for drafting ``card`` against the other side's picks ``theirs``."""
    me = matchup.unit(card)
    score = sum(points for role, points in ROLE_POINTS.items() if role in me.roles)
    beaten = []
    for other in theirs:
        value = counter_beats(me, matchup.unit(other))
        if value is not None:
            beaten.append((value, matchup.unit(other).species))
    beaten.sort(key=lambda t: -t[0])
    score += BEAT_POINTS * len(beaten) + sum(v for v, _ in beaten)
    score += (me.bst - 500) / 50.0 + strength
    tags = [label for role, label in ROLE_TAGS if role in me.roles]
    if beaten:
        tags.append("beats " + ", ".join(s for _, s in beaten[:2]))
    reason = f"{me.species}: {', '.join(tags)}." if tags else f"{me.species}: strong attacker."
    return round(score, 3), reason


class CounterPickerAgent(PokemonAgent):
    """The code brain with a greedy counter-picking draft (see the module docstring)."""

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(None, version="counter-picker", params=params)

    def _draft(self, state, obs, log):
        theirs = [c for c in self.memory.opp_cards.values() if c.get("moves")]
        best = None
        for index, action in enumerate(state.legal_actions):
            card = self.memory.pool_cards.get(draft_rules.card_id_of(action))
            if not card:
                continue
            score, reason = counter_score(card, theirs)
            if best is None or score > best[0]:
                best = (score, index, action, reason)
        if best is None:
            pick = state.legal_actions[0]
            reason = "first legal card"
        else:
            _, _, pick, reason = best
        card_id = draft_rules.card_id_of(pick)
        self.memory.record_pick(card_id)
        log.write("draft", state_version=state.state_version, card_id=card_id, reason=reason, counter_picker=True)
        return pick, reason


def create_agent() -> CounterPickerAgent:
    return CounterPickerAgent()
