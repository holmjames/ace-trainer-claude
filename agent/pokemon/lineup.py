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
    if P.get("lineup_v2", 0) and len(mine) >= BRING and len([t for t in theirs if t.moves]) >= LEADS \
            and all(p.moves for p in mine):
        return shortlist_v2(mine, [t for t in theirs if t.moves], top=top, params=P)

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


# -- v2: Team Preview from the matchup model (``lineup_v2`` = 1) ----------------------------------------------------
#
# Plain language: the scorer above rates each of our 15 fours against all six of theirs. But they bring four too,
# chosen against OUR six, and the two leads meet first. v2 rates every (our four, their four) pairing with the team
# model in agent/pokemon/matchup.py ('preview' section: four against four, plus a separate term for the two leads on
# each side), treats Team Preview as the simultaneous choice it is (a small zero-sum game, solved by fictitious
# play), and brings the four that does best against the mix of fours they are likely to bring. Leads are chosen the
# same way, our 6 possible pairs against the pairs their likely fours open with, so a lead their likely leads
# destroy (a Fighting type into Fake Out + Pixilate Hyper Voice) scores badly and is avoided.


def solve_game(payoff: list[list[float]], iterations: int = 200) -> tuple[list[float], list[float], float]:
    """Mixed strategies (row, column) and value of a zero-sum matrix game where the row player maximises, by
    fictitious play: each side keeps best-responding to the other's history so far."""
    rows, cols = len(payoff), len(payoff[0]) if payoff else 0
    if not rows or not cols:
        return [], [], 0.0
    row_counts, col_counts = [0] * rows, [0] * cols
    row_total, col_total = [0.0] * rows, [0.0] * cols
    r = max(range(rows), key=lambda i: (min(payoff[i]), -i))  # open with the safest row
    for _ in range(iterations):
        row_counts[r] += 1
        line = payoff[r]
        for j in range(cols):
            col_total[j] += line[j]
        c = min(range(cols), key=lambda j: (col_total[j], j))
        col_counts[c] += 1
        for i in range(rows):
            row_total[i] += payoff[i][c]
        r = max(range(rows), key=lambda i: (row_total[i], -i))
    p = [n / iterations for n in row_counts]
    q = [n / iterations for n in col_counts]
    value = sum(p[i] * q[j] * payoff[i][j] for i in range(rows) for j in range(cols))
    return p, q, value


class PreviewModel:
    """Scores four-against-four pairings and lead pairs on one ``matchup.Board`` (our six first, then theirs)."""

    def __init__(self, mine: list[dict], theirs: list[dict]) -> None:
        from . import matchup

        self.matchup = matchup
        model = matchup.load_model(section="preview")
        self.board = matchup.Board([matchup.unit(c) for c in mine + theirs], model)
        lead_w = model.get("lead_weights") or {}
        self.lead_w = [float(lead_w.get(f, 0.0)) for f in matchup.FEATURES]
        self.ours = list(range(len(mine)))
        self.theirs = list(range(len(mine), len(mine) + len(theirs)))
        self._lead_cache: dict[tuple, float] = {}

    def lead_score(self, ours: tuple[int, int], theirs: tuple[int, int]) -> float:
        key = (ours, theirs)
        value = self._lead_cache.get(key)
        if value is None:
            feats = self.board.features(list(ours), list(theirs))
            value = sum(w * f for w, f in zip(self.lead_w, feats))
            self._lead_cache[key] = value
        return value

    def lead_game(self, four: tuple[int, ...], other: tuple[int, ...]) -> tuple[list, list, list[float], list[float], float]:
        our_pairs = list(combinations(four, LEADS))
        their_pairs = list(combinations(other, LEADS))
        payoff = [[self.lead_score(a, b) for b in their_pairs] for a in our_pairs]
        p, q, value = solve_game(payoff, iterations=60)
        return our_pairs, their_pairs, p, q, value

    def pairing(self, four: tuple[int, ...], other: tuple[int, ...]) -> float:
        """Log-odds that ``four`` beats ``other``, with both sides leading as well as they can."""
        _, _, _, _, lead_value = self.lead_game(four, other)
        return self.board.score(list(four), list(other)) + lead_value


def shortlist_v2(mine: list[Profile], theirs: list[Profile], *, top: int = 3, params: dict | None = None) -> list[Candidate]:
    P = params or DEFAULTS
    model = PreviewModel([p.card for p in mine], [p.card for p in theirs])
    win = model.matchup.win_probability
    names = {i: p.species for i, p in enumerate(mine)}
    names.update({len(mine) + j: p.species for j, p in enumerate(theirs)})
    our_fours = list(combinations(model.ours, BRING))
    their_fours = list(combinations(model.theirs, min(BRING, len(model.theirs))))
    payoff = [[model.pairing(f, g) for g in their_fours] for f in our_fours]
    _, q, game_value = solve_game(payoff)
    robust = float(P.get("lineup_v2_robust", 0.25))
    ranked = []
    for i, four in enumerate(our_fours):
        expected = sum(q[j] * payoff[i][j] for j in range(len(their_fours)))
        worst = min(payoff[i])
        ranked.append(((1 - robust) * expected + robust * worst, expected, i))
    ranked.sort(key=lambda t: (-t[0], t[2]))
    likely = sorted(((q[j], j) for j in range(len(their_fours)) if q[j] > 0), reverse=True)
    likely_note = "; ".join(f"{[names[k] for k in their_fours[j]]} {100 * w:.0f}%" for w, j in likely[:2])
    out: list[Candidate] = []
    for value, expected, i in ranked[:top]:
        four = our_fours[i]
        # Leads: our pair that does best against the lead pairs their likely fours open with.
        lead_values: dict[tuple, float] = {}
        their_leads: dict[tuple, float] = {}
        for weight, j in likely:
            pairs, opp_pairs, _, opp_mix, _ = model.lead_game(four, their_fours[j])
            for pair in pairs:
                lead_values[pair] = lead_values.get(pair, 0.0) + weight * sum(
                    opp_mix[k] * model.lead_score(pair, opp_pairs[k]) for k in range(len(opp_pairs)))
            for k, opp in enumerate(opp_pairs):
                their_leads[opp] = their_leads.get(opp, 0.0) + weight * opp_mix[k]
        best_pair = max(lead_values, key=lambda pr: (lead_values[pr], [-x for x in pr])) if lead_values else four[:LEADS]
        expected_leads = sorted(their_leads.items(), key=lambda t: -t[1])[:2]
        notes = [f"v2: {100 * win(expected):.0f}% vs their likely fours ({likely_note})",
                 f"worst case {100 * win(min(payoff[i])):.0f}%"]
        if expected_leads:
            notes.append("their likely leads: " + "; ".join(f"{names[a]} + {names[b]} {100 * w:.0f}%" for (a, b), w in expected_leads))
        danger = [f"{names[j]} OHKOs {names[k]} before it moves" for k in best_pair for (a, b), _ in expected_leads[:1]
                  for j in (a, b) if model.board.ohko[j][k]]
        if danger:
            notes.append("lead risk: " + "; ".join(danger[:2]))
        out.append(Candidate([names[k] for k in four], [names[k] for k in best_pair], round(value, 3), notes))
    if out:
        out[0].notes.append(f"game value {100 * win(game_value):.0f}%")
    return out
