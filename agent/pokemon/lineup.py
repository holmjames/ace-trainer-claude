"""Team Preview: score every way to bring 4 of our 6 against the opponent's known 6.

Plain language: there are only 15 ways to choose 4 Pokémon from 6, so code can
look at all of them. For each one we ask: can these four hit everything the
opponent brought, do they resist what the opponent's sets actually throw, are
they fast enough or do they carry speed control, is there a support piece, and
do too many of them share a weakness? The best few go to the model with the
numbers attached; the best one is also the fallback if the model fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from . import data
from .battle import FieldState, build_mon, damage_percent
from .draft import Profile, hit_quality, profile
from .tuning import DEFAULTS

BRING = 4
LEADS = 2
SPEED_CONTROL = {"tailwind", "trickroom", "icywind", "electroweb", "thunderwave", "bulldoze"}
REDIRECT = {"followme", "ragepowder"}


@dataclass
class Candidate:
    bring: list[str]
    leads: list[str]
    score: float
    notes: list[str] = field(default_factory=list)

    def as_payload(self) -> dict:
        return {"type": "select_lineup", "bring": list(self.bring), "leads": list(self.leads)}


def _moves(p: Profile) -> set[str]:
    return {m["id"] for m in p.moves}


def score_four(four: list[Profile], theirs: list[Profile], params: dict | None = None) -> tuple[float, list[str]]:
    P = params or DEFAULTS
    notes: list[str] = []
    score = 0.0

    # Offense: for each opposing Pokémon, how hard can our best member hit it?
    if theirs:
        hits = [max((hit_quality(me, opp) for me in four), default=0.0) for opp in theirs]
        covered = sum(1 for h in hits if h >= 1.5)
        weak_spots = [theirs[i].species for i, h in enumerate(hits) if h < 0.9]
        score += P["lineup_offense_w"] * sum(hits) / len(theirs)
        notes.append(f"strong hits on {covered}/{len(theirs)}")
        if weak_spots:
            score -= 2 * len(weak_spots)
            notes.append(f"struggles vs {weak_spots[:3]}")

        # Defense: for each opposing attacker, does someone on our four resist its best attack?
        resisted = 0
        for opp in theirs:
            if not opp.attacks:
                continue
            best_mult = min(
                max(data.effectiveness(m.get("type") or "", me.types) for m in opp.attacks) for me in four if me.types
            ) if any(me.types for me in four) else 1.0
            if best_mult <= 0.5:
                resisted += 1
        score += P["lineup_resist_w"] * resisted
        notes.append(f"resists {resisted}/{len(theirs)} attackers")

        # Speed: how many of ours outspeed the median opposing speed?
        opp_speeds = sorted(o.stats.get("spe", 0) for o in theirs if o.stats)
        if opp_speeds:
            median = opp_speeds[len(opp_speeds) // 2]
            faster = sum(1 for me in four if me.stats.get("spe", 0) > median)
            score += P["lineup_speed_w"] * faster
            notes.append(f"{faster}/4 outspeed their median {median}")

    # Defensive answers (item 1): for each of their two most dangerous attackers, do we bring something that
    # resists or is immune to its strongest STAB attack? If not, that attacker runs through our four.
    if theirs and P.get("lineup_answer_w", 0):
        def threat_level(opp: Profile) -> float:
            return max((hit_quality(opp, me) for me in four), default=0.0) * (1 + opp.stats.get("spe", 0) / 300.0)
        top = sorted(theirs, key=threat_level, reverse=True)[:2]
        unanswered = []
        for opp in top:
            stabs = [m for m in opp.attacks if data.to_id(m.get("type")) in opp.types] or opp.attacks
            if not stabs:
                continue
            best_stab = max(stabs, key=lambda m: (m.get("base_power") or 0) * data.expected_hits(m))
            if not any(data.effectiveness(best_stab.get("type") or "", me.types) <= 0.5 for me in four if me.types):
                unanswered.append(opp.species)
        if unanswered:
            score -= P["lineup_answer_w"] * len(unanswered)
            notes.append(f"no resist/immunity for {unanswered}")

    # Roles.
    moves_all = [_moves(p) for p in four]
    if any(mv & SPEED_CONTROL for mv in moves_all):
        score += P["lineup_speed_control"]
        notes.append("speed control")
    if any("fakeout" in mv for mv in moves_all):
        score += P["lineup_fake_out"]
        notes.append("fake out")
    if any(mv & REDIRECT for mv in moves_all):
        score += 2
        notes.append("redirection")
    if any(p.ability == "intimidate" for p in four):
        score += 2
    attackers = [p for p in four if p.attacks and p.bst >= 470]
    physical = any(m.get("category") == "Physical" for p in attackers for m in p.attacks)
    special = any(m.get("category") == "Special" for p in attackers for m in p.attacks)
    if physical and special:
        score += 2
        notes.append("mixed attackers")
    if len(attackers) < 2:
        score -= 6
        notes.append("too few attackers")
    score += sum(p.bst for p in four) / 200.0

    # Known 4x weaknesses: a brought Pokémon that takes 4x from one of their actual moves is a liability.
    if theirs:
        quad = sum(1 for me in four for opp in theirs for mv in opp.attacks if data.effectiveness(mv.get("type") or "", me.types) >= 4)
        if quad:
            score -= P["lineup_quad_penalty"] * quad
            notes.append(f"{quad} known 4x hit(s) into our four")

    # Shared weaknesses: three or more of the four weak to one type is a liability.
    for attack_type in data.TYPES:
        weak = sum(1 for p in four if data.effectiveness(attack_type, p.types) > 1)
        if weak >= 3:
            score -= 4
            notes.append(f"{weak} weak to {attack_type}")

    return round(score, 3), notes


INTIMIDATE_IMMUNE = {"innerfocus", "owntempo", "oblivious", "scrappy", "guarddog", "clearbody", "whitesmoke", "fullmetalbody"}


def lead_ohkos(pair: list[Profile], theirs: list[Profile]) -> list[str]:
    """Which opposing sets can knock out one of our leads at full HP before it moves (faster, or a priority move),
    using the real damage model with both leads' Intimidate applied. Returns notes like 'Maushold OHKOs Hatterene'."""
    out: list[str] = []
    leads = [build_mon({"species": p.species, "current_hp_fraction": 1.0}, p.card, side="mine", position=i) for i, p in enumerate(pair)]
    leads = [m for m in leads if m]
    if len(leads) < 2:
        return out
    intimidate = any(p.ability == "intimidate" for p in pair)
    for opp in theirs:
        boosts = {}
        if intimidate and opp.ability not in INTIMIDATE_IMMUNE and opp.card_item != "clearamulet":
            boosts = {"atk": -1}
        mon = build_mon({"species": opp.species, "current_hp_fraction": 1.0, "boosts": boosts}, opp.card, side="theirs", position=1)
        if not mon:
            continue
        fs = FieldState(actives=leads + [mon])
        for lead in leads:
            faster = mon.stat("spe") > lead.stat("spe")
            for move in mon.moves:
                if (move.get("base_power") or 0) <= 0:
                    continue
                if not (faster or (move.get("priority") or 0) > 0):
                    continue
                if move["id"] in ("fakeout", "firstimpression"):
                    continue
                spread = move.get("target") in ("allAdjacentFoes", "allAdjacent")
                est = damage_percent(mon, move, lead, field=fs, spread=spread)
                if est and est[0] >= 100:
                    out.append(f"{opp.species} OHKOs {lead.species} with {move['id']} before it moves")
                    break
    return out


def pick_leads(four: list[Profile], theirs: list[Profile], params: dict | None = None) -> tuple[list[str], str]:
    """The best pair to start with: speed, Fake Out, Intimidate, and not both frail to their attacks."""
    P = params or DEFAULTS
    best_pair, best_score, best_note = four[:2], float("-inf"), ""
    for a, b in combinations(four, 2):
        s = 0.0
        notes = []
        if theirs and P.get("lead_ohko_w", 0):
            ohkos = lead_ohkos([a, b], theirs)
            if ohkos:
                # Two leads lost to the same opponent is the turn-1 disaster; charge extra for it.
                by_opp: dict[str, int] = {}
                for line in ohkos:
                    by_opp[line.split(" OHKOs ")[0]] = by_opp.get(line.split(" OHKOs ")[0], 0) + 1
                s -= P["lead_ohko_w"] * (len(ohkos) + sum(1 for n in by_opp.values() if n >= 2))
                notes.append("; ".join(ohkos[:2]))
        mv = _moves(a) | _moves(b)
        if "fakeout" in mv:
            s += 4; notes.append("fake out")
        if mv & SPEED_CONTROL:
            s += 3; notes.append("speed control")
        if any(p.ability == "intimidate" for p in (a, b)):
            s += 2; notes.append("intimidate")
        s += (a.stats.get("spe", 0) + b.stats.get("spe", 0)) / 100.0
        if a.attacks and b.attacks:
            s += 1
        if theirs:
            # Known attacks into our leads: 4x is a disaster, spread 2x into both is nearly as bad.
            for opp in theirs:
                for move in opp.attacks:
                    mults = [data.effectiveness(move.get("type") or "", p.types) for p in (a, b)]
                    spread = move.get("target") in ("allAdjacentFoes", "allAdjacent")
                    for m in mults:
                        if m >= 4:
                            s -= P["lead_quad_penalty"]
                        elif m >= 2:
                            s -= P["lead_se_penalty"]
                    if spread and all(m >= 2 for m in mults):
                        s -= P["lead_spread_penalty"]
                if "fakeout" in {m["id"] for m in opp.moves}:
                    s -= 0.5  # their Fake Out costs us tempo on turn 1
                if P.get("lead_priority_multihit_w", 0):
                    # A priority move or a multi-hit (sash-breaking) move that is super-effective into one of our leads
                    # is the classic turn-1 disaster; Focus Sash does not help against multi-hit.
                    for move in opp.attacks:
                        prio = (move.get("priority") or 0) > 0
                        multi = data.expected_hits(move) > 1
                        if not (prio or multi):
                            continue
                        for lead in (a, b):
                            mult = data.effectiveness(move.get("type") or "", lead.types)
                            if mult >= 2 or (multi and data.to_id(lead.card_item) == "focussash"):
                                s -= P["lead_priority_multihit_w"]
            s += sum(max(hit_quality(p, opp) for p in (a, b)) for opp in theirs) / len(theirs)
            # Our leads should threaten their likely leads back: reward a lead pair that outspeeds and hits hard.
            fastest_opp = max((o.stats.get("spe", 0) for o in theirs), default=0)
            s += 1.5 * sum(1 for p in (a, b) if p.stats.get("spe", 0) > fastest_opp or p.priority)
        if s > best_score:
            best_pair, best_score, best_note = [a, b], s, ", ".join(notes)
    return [p.species for p in best_pair], best_note


def shortlist(my_cards: dict[str, dict], opp_cards: dict[str, dict], roster: list[str], *, top: int = 3,
              params: dict | None = None) -> list[Candidate]:
    """Rank all 4-of-6 lineups. ``roster`` is the server's spelling of our species ids;
    the result uses those spellings so it can be sent as-is."""
    by_key = {data.to_id(r): r for r in roster}
    mine: list[Profile] = []
    for key, card in my_cards.items():
        if key in by_key:
            p = profile({**card, "species": card.get("species") or key})
            p.species = by_key[key]  # server spelling
            mine.append(p)
    # Roster entries we never saw a card for (shouldn't happen): bare profiles from the dex.
    seen = {data.to_id(p.species) for p in mine}
    for key, spelling in by_key.items():
        if key not in seen:
            p = profile({"species": spelling})
            p.species = spelling
            mine.append(p)
    theirs = [profile(c) for c in opp_cards.values()]
    P = params or DEFAULTS

    # Mirror-model the opponent: which four would THEY bring against our six, and which two lead?
    # Their drafted sets are public, so we can run our own lineup scorer from their side.
    predicted: list[Profile] = []
    predicted_leads: list[Profile] = []
    if P.get("lineup_predict_opp", 0) and len(theirs) >= BRING and mine:
        best_score, best_four = float("-inf"), None
        for four in combinations(sorted(theirs, key=lambda p: p.species), BRING):
            s, _ = score_four(list(four), mine, params)
            if s > best_score:
                best_score, best_four = s, list(four)
        if best_four:
            predicted = best_four
            lead_names, _ = pick_leads(best_four, mine, params)
            predicted_leads = [p for p in best_four if p.species in lead_names]

    candidates: list[Candidate] = []
    for four in combinations(sorted(mine, key=lambda p: p.species), BRING):
        four = list(four)
        score, notes = score_four(four, theirs, params)
        if predicted:
            focused, _ = score_four(four, predicted, params)
            score = (score + P["lineup_predict_opp"] * focused) / (1 + P["lineup_predict_opp"])
            notes.append(f"vs predicted {[p.species for p in predicted]}")
        leads, lead_note = pick_leads(four, predicted_leads or theirs, params) if predicted_leads else pick_leads(four, theirs, params)
        candidates.append(Candidate([p.species for p in four], leads, score, notes + ([f"leads: {lead_note}"] if lead_note else [])))
    candidates.sort(key=lambda c: (-c.score, c.bring))
    return candidates[:top]
