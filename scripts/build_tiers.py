"""Build the card tier list the drafter uses as each card's base value.

    python scripts/build_tiers.py          # writes data/card_tiers.json and docs/card_tiers.md

Plain language: a card's rating is what it is worth on an average team, in hundredths of a log-odds point of
winning (+25 ~ "turns a 50% game into a 56% game"). It is measured, not guessed, from the fitted team model
(data/draft_model.json, fitted on 16,000 simulator games by scripts/fit_draft_model.py):

    sim_rating = matchup part + card part
      matchup part: the card's average share of the team score's matchup terms (its 1-on-1 races, spread
                    pressure, roles) over 30,000 random team pairs from the catalog;
      card part:    what the fit says it wins beyond those matchups.

``rating`` = sim_rating + ``adjust``: a small, written-down prior where the simulator's code-only pilot is known
to misplay a card (Intimidate cycling, Prankster support, Tailwind turns) and live play disagrees (the Oct 7 opponent,
4-0 against us on Oct 7, took Incineroar 3/3 and Dragonite 2/2). ``draft_tier_adjust`` scales the adjustments
(default 1.25 since Oct 8; 0 turns them off). Tiers: S >= 40, A >= 12, B >= -2, C >= -16, D below.

The drafter (agent/pokemon/matchup.py, ``Board``) adds ``draft_tier_adjust`` x ``adjust`` to the card's base value
(with ``draft_card_w`` = 0, the Oct 8 default, the fitted card part is not used: it came from games piloted by an
older battle brain) and the matchup reactions on top. Since Oct 8 ``matchup.load_model`` clamps the fitted role
weights at >= 0, so a rebuild also refreshes every sim rating under the clamped weights. Rebuild after refitting.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.pokemon import data, matchup  # noqa: E402

TIERS = (("S", 40), ("A", 12), ("B", -2), ("C", -16))
SAMPLES = 30000

# Where the simulator's code-only pilot is known to misplay a card, and what live play says. ``adjust`` is in the
# rating's units (1/100 log-odds) and is kept small: enough to break near-ties toward the live evidence.
PRIORS: dict[str, dict] = {
    "Incineroar": {"adjust": 25, "flag": "engine rarely pivots, so Parting Shot / repeat Intimidates go unused",
                   "live": "the Oct 7 opponent took it 3/3 at pick 1-2"},
    "Dragonite": {"adjust": 17, "flag": "engine spends Tailwind as a flat-value move; Multiscale + Extreme Speed under Tailwind is the real plan",
                  "live": "the Oct 7 opponent took it 2/2"},
    # Tornadus / Whimsicott carried +10 while the fitted card part offset it; with draft_card_w = 0 the +10 stood alone
    # and made a Prankster Tailwind user the first pick of the Oct 7 pool L3 (the pick that lost that game): 0 since Oct 8.
    "Tornadus": {"adjust": 0, "flag": "Prankster Tailwind / Taunt valued by fixed constants in the engine"},
    "Whimsicott": {"adjust": 0, "flag": "Prankster Tailwind / Encore valued by fixed constants in the engine"},
    "Amoonguss": {"adjust": 10, "flag": "engine redirects only to save a partner from a KO; Spore + Rage Powder do more in good hands"},
    "Hatterene": {"adjust": 0, "flag": "engine sets Trick Room whether or not its own team wants it"},
    "Porygon2": {"adjust": 0, "flag": "engine sets Trick Room whether or not its own team wants it"},
    "Klefki": {"adjust": 0, "flag": "screens valued by fixed constants in the engine"},
    "Grimmsnarl": {"adjust": 0, "flag": "screens valued by fixed constants in the engine"},
    "Sableye": {"adjust": 0, "flag": "Prankster Will-O-Wisp / Taunt valued by fixed constants in the engine"},
    "Dondozo": {"adjust": 0, "flag": "no Commander (no Tatsugiri in the catalog); this set is Unaware, which the race model does credit"},
    "Gholdengo": {"live": "the Oct 7 opponent took it 2/2"},
    "Sylveon": {"live": "the Oct 7 opponent took it 2/2"},
    "Archaludon": {"live": "the Oct 7 opponent took it 3/4"},
    "Iron Hands": {"live": "the Oct 7 opponent took it 2/3"},
    "Baxcalibur": {"live": "our drafter's specialist pick in two losses"},
    "Lucario": {"live": "our drafter's specialist pick in two losses"},
    "Ursaluna": {"live": "our drafter's specialist pick in a loss"},
}


def tier_of(rating: float) -> str:
    for name, floor in TIERS:
        if rating >= floor:
            return name
    return "D"


def matchup_parts(board: matchup.Board, rng: random.Random) -> list[float]:
    """Each card's average signed share of the team score's matchup terms over random team pairs."""
    n = len(board.units)
    sums, counts = [0.0] * n, [0] * n

    def team(avoid: set[int]) -> list[int]:
        while True:
            picks = rng.sample([i for i in range(n) if i not in avoid], 6)
            if len({board.items[i] for i in picks}) == 6:  # Item Clause
                return picks

    for _ in range(SAMPLES):
        us = team(set())
        them = team(set(us))
        value = sum(w * f for w, f in zip(board.w, board.features(us, them)))
        for k in us:
            sums[k] += value
            counts[k] += 1
        for k in them:
            sums[k] -= value
            counts[k] += 1
    return [s / c if c else 0.0 for s, c in zip(sums, counts)]


def why(board: matchup.Board, k: int, card_part: float, ratings: list[float]) -> str:
    """One line from the numbers: the 1-on-1 record with its most valuable wins and counters, roles, and what the
    simulated games add on top of the matchups."""
    unit = board.units[k]
    others = [j for j in range(len(board.units)) if j != k]
    by_rating = sorted(others, key=lambda j: -ratings[j])
    beats = [j for j in by_rating if board.base[k][j] >= 0.2]
    loses = [j for j in by_rating if board.base[k][j] <= -0.2]
    roles = [r for r in ("fakeout", "intimidate", "tailwind", "trickroom", "redirect", "spread", "priority", "screens", "sleep")
             if r in unit.roles]
    names = {"fakeout": "Fake Out", "intimidate": "Intimidate", "tailwind": "Tailwind", "trickroom": "Trick Room",
             "redirect": "redirection", "spread": "spread", "priority": "priority", "screens": "screens", "sleep": "sleep move"}
    text = f"wins {len(beats)}/{len(others)} 1-on-1s"
    if beats:
        text += f" (best: {', '.join(board.units[j].species for j in beats[:2])})"
    if loses:
        text += f"; countered by {', '.join(board.units[j].species for j in loses[:2])}"
    if roles:
        text += "; " + ", ".join(names[r] for r in roles)
    if card_part >= 0.10:
        text += "; wins more in simulated games than its matchups explain"
    elif card_part <= -0.10:
        text += "; wins less in simulated games than its matchups suggest"
    return text


def main() -> int:
    cards = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))
    model = matchup.load_model(section="draft")
    units = [matchup.unit(c) for c in cards]
    board = matchup.Board(units, model, tiers={})
    parts = matchup_parts(board, random.Random(1))
    card_parts = [float((model.get("cards", {}).get(u.key) or {}).get("s", 0.0)) for u in units]
    ratings = [100 * (parts[k] + card_parts[k]) + PRIORS.get(u.species, {}).get("adjust", 0) for k, u in enumerate(units)]
    out: dict[str, dict] = {}
    for k, (card, unit) in enumerate(zip(cards, units)):
        sim = 100 * (parts[k] + card_parts[k])
        prior = PRIORS.get(unit.species, {})
        out[card["card_id"]] = {
            "species": unit.species, "fingerprint": unit.key, "tier": tier_of(ratings[k]), "rating": round(ratings[k], 1),
            "sim_rating": round(sim, 1), "adjust": prior.get("adjust", 0), "matchup_part": round(100 * parts[k], 1),
            "card_part": round(100 * card_parts[k], 1), "why": why(board, k, card_parts[k], ratings),
            **({"flag": prior["flag"]} if prior.get("flag") else {}), **({"live": prior["live"]} if prior.get("live") else {}),
        }
    ordered = dict(sorted(out.items(), key=lambda kv: -kv[1]["rating"]))
    blob = {"note": "Built by scripts/build_tiers.py. rating = sim_rating + adjust; the drafter's base value per card. "
                    "sim_rating = matchup_part + card_part from the fitted team model (data/draft_model.json); adjust is a "
                    "written-down prior where the code-only simulator misplays the card (see flag / live).",
            "units": "1/100 log-odds of winning", "tiers": {name: f">= {floor}" for name, floor in TIERS},
            "cards": ordered}
    (ROOT / "data" / "card_tiers.json").write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Card tiers (live catalog)", "",
             "Built by `scripts/build_tiers.py` from the fitted team model; the drafter's base value per card. Rating = "
             "average worth on a team in 1/100 log-odds of winning (sim rating + a small prior where the code-only "
             "simulator is known to misplay the card). Rebuild after refitting `data/draft_model.json`.", "",
             "| Tier | Card | Rating | Sim | Why | Flag / live |", "|---|---|---:|---:|---|---|"]
    for entry in ordered.values():
        note = "; ".join(x for x in (entry.get("flag"), entry.get("live")) if x)
        lines.append(f"| {entry['tier']} | {entry['species']} | {entry['rating']:+.0f} | {entry['sim_rating']:+.0f} | {entry['why']} | {note} |")
    (ROOT / "docs" / "card_tiers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for entry in ordered.values():
        print(f"{entry['tier']}  {entry['species']:18s} {entry['rating']:+6.1f} (sim {entry['sim_rating']:+6.1f})  {entry['why']}")
    print("wrote data/card_tiers.json and docs/card_tiers.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
