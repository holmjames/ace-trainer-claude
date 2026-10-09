"""Workstream C (Team Preview) regression tests on the live catalog: the v2 failure note (I8) and the lead veto (C2)."""

from __future__ import annotations

import json
import logging

import pytest

from agent.pokemon import data, lineup, tuning
from agent.pokemon.memory import species_key

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")
V2 = tuning.merged({"lineup_v2": 1})


def rosters(mine: list[str], theirs: list[str]):
    return ({species_key(s): CATALOG[s] for s in mine}, {species_key(s): CATALOG[s] for s in theirs}, [data.to_id(s) for s in mine])


# -- I8: a v2 failure is logged and noted, never silent -----------------------------------------------------------

MINE_M2 = ["Iron Bundle", "Iron Hands", "Archaludon", "Gyarados", "Whimsicott", "Hatterene"]
THEIRS_M2 = ["Incineroar", "Flutter Mane", "Sylveon", "Rillaboom", "Kingambit", "Dondozo"]


def test_a_failing_v2_leaves_a_one_line_note_and_a_log_record(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(lineup, "shortlist_v2", broken)
    with caplog.at_level(logging.WARNING, logger="agent.pokemon.lineup"):
        cands = lineup.shortlist(*rosters(MINE_M2, THEIRS_M2), params=V2)
    assert cands and len(cands[0].bring) == 4  # the old scorer still answers
    assert any(n == "Team Preview v2 failed (RuntimeError: boom); old scorer used" for n in cands[0].notes), cands[0].notes
    assert [r for r in caplog.records if "Team Preview v2 failed (RuntimeError: boom)" in r.getMessage()]
    assert not any(n.startswith("v2:") for n in cands[0].notes)


# -- C2: the lead veto (choose_leads_v2) on the fours we actually brought in the live sessions ----------------------
#
# Each entry: our six, their six, the four we brought live (our live leads first), the leads they actually opened with.
LIVE = {
    "L1 d9405f02": (["Baxcalibur", "Landorus-Therian", "Sneasler", "Garchomp", "Cresselia", "Lucario"],
                    ["Dragonite", "Gholdengo", "Iron Hands", "Archaludon", "Iron Bundle", "Indeedee-F"],
                    ["Sneasler", "Cresselia", "Landorus-Therian", "Garchomp"], ["Dragonite", "Gholdengo"]),
    "L2 1aef9227": (MINE_M2, THEIRS_M2, ["Iron Hands", "Whimsicott", "Archaludon", "Gyarados"], ["Incineroar", "Sylveon"]),
    "L3 7ce71025": (["Tornadus", "Garchomp", "Great Tusk", "Hatterene", "Ursaluna", "Indeedee-F"],
                    ["Incineroar", "Iron Hands", "Whimsicott", "Archaludon", "Sylveon", "Amoonguss"],
                    ["Great Tusk", "Tornadus", "Hatterene", "Indeedee-F"], ["Iron Hands", "Sylveon"]),
    "L4 0a2b7c2a": (["Landorus-Therian", "Baxcalibur", "Tornadus", "Lucario", "Great Tusk", "Hatterene"],
                    ["Incineroar", "Dragonite", "Archaludon", "Gholdengo", "Gyarados", "Indeedee-F"],
                    ["Landorus-Therian", "Tornadus", "Baxcalibur", "Lucario"], ["Dragonite", "Gholdengo"]),
    "W bcbaabe2": (["Iron Hands", "Archaludon", "Arcanine", "Wo-Chien", "Milotic", "Scream Tail"],
                   ["Incineroar", "Dragonite", "Salamence", "Ting-Lu", "Basculegion", "Rillaboom"],
                   ["Archaludon", "Arcanine", "Wo-Chien", "Milotic"], ["Dragonite", "Salamence"]),
}


class Session:
    """The preview model for one live session, with the four we brought as board indices."""

    def __init__(self, name: str, params: dict | None = None):
        from itertools import combinations

        self.P = params or V2
        our_six, their_six, four, their_live = LIVE[name]
        self.model = lineup.PreviewModel([CATALOG[s] for s in our_six], [CATALOG[s] for s in their_six],
                                         tier_adjust=self.P["draft_tier_adjust"])
        self.names = {i: s for i, s in enumerate(our_six)}
        self.names.update({6 + j: s for j, s in enumerate(their_six)})
        self.idx = {s: i for i, s in self.names.items()}
        self.four = tuple(sorted(self.idx[s] for s in four))
        self.live_pair = tuple(self.idx[s] for s in four[:2])
        self.live_opp = tuple(self.idx[s] for s in their_live)
        their_fours = list(combinations(self.model.theirs, lineup.BRING))
        payoff = [[self.model.pairing(f, g) for g in their_fours] for f in combinations(self.model.ours, lineup.BRING)]
        _, q, _ = lineup.solve_game(payoff)
        likely = sorted(((q[j], j) for j in range(len(their_fours)) if q[j] > 0), reverse=True)
        self.their_leads = self.model.likely_leads(self.four, likely, their_fours, self.P["lineup_v2_template"])
        self.top2 = [opp for opp, _ in sorted(self.their_leads.items(), key=lambda t: -t[1])[:2]]

    def choose(self, their_leads=None):
        pair, notes = lineup.choose_leads_v2(self.model, self.four, their_leads or self.their_leads, self.P)
        return tuple(self.names[i] for i in pair), notes

    def lost(self, pair) -> int:
        """Leads lost in the worst opening against their two likeliest leads (what the veto counts)."""
        return max(len(self.model.opening(pair, opp).ko_against) for opp in self.top2)

    def pairs(self):
        from itertools import combinations

        return list(combinations(self.four, lineup.LEADS))

    def pair(self, *species):
        return tuple(self.idx[s] for s in species)


def test_fake_out_exposure_knows_immunity_terrain_and_a_faster_fake_out_of_our_own():
    from agent.pokemon import matchup

    u = lambda s: matchup.unit(CATALOG[s])  # noqa: E731
    incin_sylveon = [u("Incineroar"), u("Sylveon")]
    assert matchup.fake_out_exposed([u("Iron Hands"), u("Whimsicott")], incin_sylveon) == [("Whimsicott", "Incineroar")]
    # Hatterene is grounded under Indeedee's Psychic Terrain: Fake Out fails. Tornadus flies above it, and Follow Me
    # (+2) does not move before Fake Out (+3), so it is exposed next to the same Indeedee.
    assert matchup.fake_out_exposed([u("Hatterene"), u("Indeedee-F")], incin_sylveon) == []
    assert matchup.fake_out_exposed([u("Tornadus"), u("Indeedee-F")], incin_sylveon) == [("Tornadus", "Incineroar")]
    assert matchup.fake_out_exposed([u("Tornadus"), u("Great Tusk")], [u("Gholdengo"), u("Dragonite")]) == []  # no Fake Out there
    # Sneasler's Fake Out is faster than Rillaboom's: our flinch lands first, Whimsicott's Tailwind goes up.
    assert matchup.fake_out_exposed([u("Whimsicott"), u("Sneasler")], [u("Rillaboom"), u("Sylveon")]) == []
    cloaked = matchup.unit({**CATALOG["Whimsicott"], "item": "Covert Cloak"})
    assert matchup.fake_out_exposed([cloaked, u("Iron Hands")], incin_sylveon) == []
    ghost = matchup.unit({**CATALOG["Whimsicott"], "species": "Gholdengo"})  # a Ghost-type setter cannot be flinched
    assert matchup.fake_out_exposed([ghost, u("Iron Hands")], incin_sylveon) == []


def test_the_tusk_tornadus_lead_of_the_third_loss_is_rejected_for_a_flinch_proof_pair():
    s = Session("L3 7ce71025")
    live = s.pair("Great Tusk", "Tornadus")
    assert s.lost(live) >= 1  # the model sees a lead lost in the opening (live: Tornadus fell to Wild Charge on turn 1)
    assert any(x[0] == "Tornadus" for opp in s.top2 for x in s.model.fake_out_exposed(live, opp))
    chosen, notes = s.choose()
    assert set(chosen) != {"Great Tusk", "Tornadus"}, (chosen, notes)
    assert "Tornadus" not in chosen  # every Tornadus pair leads a Tailwind setter into Fake Out
    assert not any(s.model.fake_out_exposed(s.pair(*chosen), opp) for opp in s.top2)
    assert s.lost(s.pair(*chosen)) == min(s.lost(p) for p in s.pairs())
    assert any(n.startswith("no lead pair of this four is safe") for n in notes), notes


def test_the_iron_hands_whimsicott_lead_of_the_second_loss_is_rejected_for_its_exposed_tailwind_setter():
    s = Session("L2 1aef9227")
    live = s.pair("Iron Hands", "Whimsicott")
    assert s.model.fake_out_exposed(live, s.pair("Incineroar", "Sylveon")) == [("Whimsicott", "Incineroar")]
    assert s.model.opening(live, s.live_opp).ko_against == ["Iron Hands", "Whimsicott"]  # what happened live by turn 2
    chosen, notes = s.choose()
    assert "Whimsicott" not in chosen, (chosen, notes)
    assert s.lost(s.pair(*chosen)) <= 1 and s.lost(s.pair(*chosen)) < s.lost(live)
    assert chosen == ("Iron Hands", "Gyarados")  # loses nothing against Sylveon + Rillaboom, one lead against Incineroar + Sylveon


def test_todays_winning_lead_passes_against_the_lead_they_played_and_the_veto_keeps_a_safe_pair():
    s = Session("W bcbaabe2")
    live = s.pair("Archaludon", "Arcanine")
    # Against the leads they actually opened with (Dragonite + Salamence, both Protecting on turn 1) it loses nothing
    # and leads no exposed setter: the veto has no case against it, and it is chosen when that lead is expected.
    assert s.model.opening(live, s.live_opp).ko_against == []
    assert s.model.fake_out_exposed(live, s.live_opp) == []
    chosen_live, notes_live = s.choose({s.live_opp: 1.0})
    assert not any("not Archaludon + Arcanine" in n for n in notes_live) and "Archaludon" in chosen_live, notes_live
    # Against the model's two likeliest leads (Ting-Lu next to Dragonite or Rillaboom) the opening says Arcanine
    # falls, and Archaludon + Wo-Chien loses nothing, so the veto takes that pair; it also loses nothing against the
    # lead they played (+1.27 by the opening model).
    chosen, notes = s.choose()
    assert s.lost(live) == 1 and s.lost(s.pair(*chosen)) == 0, (chosen, notes)
    assert any(n.startswith("lead veto: not Archaludon + Arcanine") for n in notes), notes
    assert s.model.opening(s.pair(*chosen), s.live_opp).ko_against == []


def test_the_hard_veto_keeps_a_pair_that_loses_a_lead_out_while_a_safe_pair_exists():
    s = Session("W bcbaabe2")
    live_leads = {s.live_opp: 1.0}  # suppose we knew they would open Dragonite + Salamence
    losers = [p for p in s.pairs() if s.model.opening(p, s.live_opp).ko_against]
    assert losers and len(losers) < len(s.pairs())  # Arcanine + Wo-Chien loses Arcanine; five pairs lose nothing
    chosen, notes = s.choose(live_leads)
    assert s.pair(*chosen) not in losers
    assert any(n.startswith("lead veto: not ") for n in notes) and any("lose nothing" in n for n in notes), notes
    off = tuning.merged({"lineup_v2": 1, "lead_veto_w": 0.0})
    pair_off, notes_off = lineup.choose_leads_v2(s.model, s.four, live_leads, off)
    assert notes_off == [] and len(pair_off) == 2


@pytest.mark.parametrize("name", list(LIVE))
def test_with_no_safe_pair_the_lead_loses_the_fewest_in_the_opening(name):
    s = Session(name)
    chosen, notes = s.choose()
    if any(n.startswith("no lead pair of this four is safe") for n in notes):
        assert s.lost(s.pair(*chosen)) == min(s.lost(p) for p in s.pairs()), (name, chosen, notes)
    else:
        assert s.lost(s.pair(*chosen)) == 0


def test_shortlist_v2_carries_the_veto_notes_and_still_answers_fast():
    import time

    our_six, their_six, _, _ = LIVE["L3 7ce71025"]
    started = time.perf_counter()
    cands = lineup.shortlist(*rosters(our_six, their_six), params=V2)
    assert time.perf_counter() - started < 2.0
    assert cands and all(len(c.leads) == 2 and set(c.leads) <= set(c.bring) for c in cands)
    assert any(("lead veto" in n) or n.startswith("no lead pair of this four is safe") or ("lose nothing" in n)
               for n in cands[0].notes), cands[0].notes


# -- C: their bring and leads are modelled by roles and matchups only ----------------------------------------------


def test_neither_side_carries_fitted_card_terms_and_todays_actual_four_is_predicted_first():
    from itertools import combinations

    our_six, their_six, _, their_live = LIVE["W bcbaabe2"]
    model = lineup.PreviewModel([CATALOG[s] for s in our_six], [CATALOG[s] for s in their_six])
    assert all(r == 0.0 for r in model.board.resid)  # roles and matchups only, both sides
    assert any(float((model.board.model.get("cards") or {}).get(u.key, {}).get("s", 0.0)) != 0.0 for u in model.board.units)
    names = {6 + j: s for j, s in enumerate(their_six)}
    their_fours = list(combinations(model.theirs, lineup.BRING))
    payoff = [[model.pairing(f, g) for g in their_fours] for f in combinations(model.ours, lineup.BRING)]
    _, q, _ = lineup.solve_game(payoff)
    predicted = max(range(len(their_fours)), key=lambda j: q[j])
    assert {names[k] for k in their_fours[predicted]} == {"Dragonite", "Salamence", "Ting-Lu", "Rillaboom"}, \
        [(sorted(names[k] for k in their_fours[j]), round(q[j], 2)) for j in range(len(their_fours)) if q[j] > 0]
    assert q[predicted] > 0.99
    s = Session("W bcbaabe2")
    assert s.their_leads.get(s.live_opp, 0.0) > 0.0  # Dragonite + Salamence, the lead they opened with (two support users)
    # With the fitted terms on their side (the Oct 7 behaviour) Dragonite's -0.355 dropped it from their four.
    with_terms = lineup.PreviewModel([CATALOG[s] for s in our_six], [CATALOG[s] for s in their_six])
    with_terms.board.resid = [float((with_terms.board.model.get("cards") or {}).get(u.key, {}).get("s", 0.0)) for u in with_terms.board.units]
    with_terms._lead_cache.clear()
    payoff = [[with_terms.pairing(f, g) for g in their_fours] for f in combinations(with_terms.ours, lineup.BRING)]
    _, q2, _ = lineup.solve_game(payoff)
    assert "Dragonite" not in {names[k] for k in their_fours[max(range(len(their_fours)), key=lambda j: q2[j])]}


def test_the_lead_template_gives_two_support_users_a_share():
    s = Session("W bcbaabe2")
    four = tuple(sorted(s.idx[x] for x in ("Dragonite", "Salamence", "Ting-Lu", "Rillaboom")))
    prior = s.model.template_leads(four, list(s.four))
    double = {p for p in prior if all(s.model.board.roles[i] & {"fakeout", "tailwind", "trickroom", "speed_drop"} for i in p)
              and s.idx["Ting-Lu"] not in p}
    assert double and all(prior[p] > 0 for p in double) and abs(sum(prior[p] for p in double) - 0.25) < 1e-9
    assert abs(sum(prior.values()) - 1.0) < 1e-9


# -- Secret tie-breaks at Team Preview --------------------------------------------------------------------------------


def test_the_preview_follows_the_match_seed_and_varies_only_inside_the_window():
    from agent.pokemon import draft, matchup

    our_six, their_six, _, _ = LIVE["W bcbaabe2"]
    mine, theirs, roster = rosters(our_six, their_six)
    assert not any(n.startswith("tie-break seed") for n in lineup.shortlist(mine, theirs, roster, params=V2)[0].notes)  # no match: no seed
    seed = draft.tiebreak_seed(mine)  # what the match's first draft decision registers
    a = lineup.shortlist(mine, theirs, roster, params=V2)
    b = lineup.shortlist(mine, theirs, roster, params=V2)  # the same card dict: the same seed, the same answer
    assert [c.as_payload() for c in a] == [c.as_payload() for c in b]
    assert any(n.startswith(f"tie-break seed {seed:#x}") for n in a[0].notes), a[0].notes
    profiles_m = [lineup.profile({**c, "species": c.get("species")}) for c in mine.values()]
    profiles_t = [lineup.profile(c) for c in theirs.values()]
    fixed = [lineup.shortlist_v2(profiles_m, profiles_t, params=V2, seed=s)[0] for s in range(10)]
    win = matchup.win_probability
    scores = sorted({c.score for c in lineup.shortlist_v2(profiles_m, profiles_t, params=V2, seed=None)}, reverse=True)
    within = [s for s in scores if win(scores[0]) - win(s) <= V2["lineup_tiebreak_eps"]]
    assert all(c.score in within for c in fixed)  # never a four outside the window
    assert lineup.shortlist_v2(profiles_m, profiles_t, params=V2, seed=3)[0].as_payload() == \
        lineup.shortlist_v2(profiles_m, profiles_t, params=V2, seed=3)[0].as_payload()
    s = Session("W bcbaabe2")
    leads = {lineup.choose_leads_v2(s.model, s.four, s.their_leads, V2, seed=k)[0] for k in range(6)}
    assert all(not s.lost(p) for p in leads)  # the seed never picks outside the safe set


OURS_W2 = ["Gyarados", "Iron Hands", "Iron Treads", "Salamence", "Sylveon", "Baxcalibur"]
THEIRS_W2 = ["Incineroar", "Rillaboom", "Flutter Mane", "Grimmsnarl", "Amoonguss", "Toxapex"]


def fours_ranked(our_six, their_six):
    """All 15 fours of our six in shortlist_v2's order (the blended value), as species sets."""
    from itertools import combinations

    model = lineup.PreviewModel([CATALOG[s] for s in our_six], [CATALOG[s] for s in their_six])
    our_fours = list(combinations(model.ours, lineup.BRING)); their_fours = list(combinations(model.theirs, lineup.BRING))
    payoff = [[model.pairing(f, g) for g in their_fours] for f in our_fours]
    _, q, _ = lineup.solve_game(payoff)
    robust = V2["lineup_v2_robust"]
    ranked = sorted(((1 - robust) * sum(q[j] * payoff[i][j] for j in range(len(their_fours))) + robust * min(payoff[i]), i)
                    for i in range(len(our_fours)))[::-1]
    return [({our_six[k] for k in our_fours[i]}, value) for value, i in ranked]


def test_without_the_preview_card_terms_an_iron_hands_four_ranks_first_on_both_oct_8_sixes():
    # 93a98b6f post-mortem: the preview table's -0.143 for Iron Hands (the draft table says +0.091) benched our pick-2
    # Iron Hands on near-ties of 0.037 and 0.001 log-odds in bcbaabe2 and 93a98b6f. With the terms off, an Iron Hands
    # four is first on both sixes, and the four we actually brought in bcbaabe2 is no longer the choice (8th of 15,
    # 65% against 75%).
    for our_six, their_six in ((LIVE["W bcbaabe2"][0], LIVE["W bcbaabe2"][1]), (OURS_W2, THEIRS_W2)):
        ranked = fours_ranked(our_six, their_six)
        assert "Iron Hands" in ranked[0][0], ranked[:3]
    ranked = fours_ranked(LIVE["W bcbaabe2"][0], LIVE["W bcbaabe2"][1])
    brought = {"Archaludon", "Arcanine", "Wo-Chien", "Milotic"}
    assert ranked[0][0] != brought and next(k for k, (four, _) in enumerate(ranked, 1) if four == brought) == 8, ranked[:3]
    best = lineup.shortlist(*rosters(LIVE["W bcbaabe2"][0], LIVE["W bcbaabe2"][1]), top=1, params=V2)[0]
    assert "ironhands" in best.bring


# -- C4: the Team Preview payload table ---------------------------------------------------------------------------


def preview_for(our_six, their_six):
    mine, theirs, roster = rosters(our_six, their_six)
    cands = lineup.shortlist(mine, theirs, roster, params=V2)
    return mine, theirs, cands, lineup.preview_table(mine, theirs, cands, V2)


def test_the_preview_table_fits_and_carries_the_numbers_for_todays_first_win():
    our_six, their_six, _, _ = LIVE["W bcbaabe2"]
    mine, theirs, cands, table = preview_for(our_six, their_six)
    assert 0 < len(table) <= 2500, len(table)
    lines = table.splitlines()
    assert lines[1].startswith("OURS->THEIRS") and lines[8].startswith("THEIRS->OURS") and len(lines) >= 17
    assert sum(1 for line in lines[2:8]) == 6 and sum(1 for line in lines[9:15]) == 6  # 6x6 both ways
    hands = next(line for line in lines[2:8] if line.startswith("Iron Hands"))
    assert "102!" in hands  # Close Combat one-shots Incineroar
    basc = next(line for line in lines[9:15] if line.startswith("Basculegion"))
    assert "172!>" in basc  # Adaptability Wave Crash one-shots Arcanine, and Basculegion moves first
    assert "Fake Out: Incineroar, Rillaboom" in table and "Intimidate: Incineroar, Salamence" in table
    assert "speed control: Dragonite (tailwind), Salamence (tailwind)" in table
    assert "Fake Out-immune leads: Scream Tail" in table  # Covert Cloak
    assert "their likely leads:" in table and "Predicted: v2:" in table
    best = cands[0]
    bring_line = next(line for line in lines if line.startswith("We bring "))
    for b in best.bring:
        assert any(data.to_id(s) == data.to_id(b) for s in bring_line.split(";")[0][len("We bring "):].split(", "))
    assert any(line.startswith("Turn 1 vs ") and " || they: " in line for line in lines)
    assert lines[-1].startswith("Cannot damage: ")


def test_the_preview_table_on_the_second_oct_8_match_and_the_immune_cell():
    mine, theirs, cands, table = preview_for(OURS_W2, THEIRS_W2)
    assert 0 < len(table) <= 2500, len(table)
    assert "Fake Out: Incineroar, Rillaboom" in table and "Flutter Mane (speed_drop)" in table
    assert "lead veto" in table or "leads lose nothing" in table or "no lead pair" in table
    # L1: Sneasler cannot touch Air Balloon Gholdengo (Close Combat into a Ghost): an 'x' cell and a named gap
    our_six = ["Baxcalibur", "Landorus-Therian", "Sneasler", "Garchomp", "Cresselia", "Lucario"]
    their_six = ["Dragonite", "Gholdengo", "Iron Hands", "Archaludon", "Iron Bundle", "Indeedee-F"]
    mine, theirs, cands, table = preview_for(our_six, their_six)
    sneasler = next(line for line in table.splitlines() if line.startswith("Sneasler"))
    cols = sneasler.split()
    assert cols[2] == "x"  # the Gholdengo column
    if any(data.to_id(b) == "sneasler" for b in cands[0].bring):
        assert "Sneasler: Gholdengo" in table.split("Cannot damage: ")[1]
    assert lineup.preview_table({}, {}, cands, V2) == ""


def test_the_opening_plan_reads_the_turn_one_line():
    from agent.pokemon import matchup

    u = lambda s: matchup.unit(CATALOG[s])  # noqa: E731
    plan = matchup.opening_plan([u("Iron Hands"), u("Whimsicott")], [u("Incineroar"), u("Sylveon")])
    assert len(plan) == 4 and plan[0].startswith("Iron Hands: fakeout -> ") and plan[1] == "Whimsicott: tailwind"
    assert plan[2].startswith("Incineroar: fakeout -> ") and plan[3].startswith("Sylveon: ")


# -- Sleep in the four-against-four score (Oct 8 evening, 51c86e5c: Amoonguss slept four of ours in seven turns) -------

OURS_E3 = ["Chi-Yu", "Gyarados", "Archaludon", "Tornadus", "Indeedee-F", "Rillaboom"]
THEIRS_E3 = ["Iron Hands", "Chien-Pao", "Landorus-Therian", "Ursaluna", "Hatterene", "Amoonguss"]


def test_spore_is_priced_four_against_four_and_their_spore_four_gets_weight():
    # The deployed Team Preview scored four against four with nothing for Spore (only the two-turn opening looked at
    # it) and gave every four of theirs holding Amoonguss 0%: their actual four (Chien-Pao, Amoonguss, Iron Hands,
    # Hatterene) included. The pairing now carries lineup_sleep_w x Board.sleep_pressure: their Spore against a four
    # of ours with no sleep-proof card costs the full weight, against one (Rillaboom, Grass) half of it.
    from itertools import combinations

    w = V2["lineup_sleep_w"]
    assert w > 0
    def model(sleep_w):
        return lineup.PreviewModel([CATALOG[s] for s in OURS_E3], [CATALOG[s] for s in THEIRS_E3], tier_adjust=V2["draft_tier_adjust"],
                                   hole_w=V2["lineup_hole_w"], sleep_w=sleep_w)
    priced, plain = model(w), model(0.0)
    idx = {s: i for i, s in enumerate(OURS_E3 + THEIRS_E3)}
    four = lambda *names: tuple(sorted(idx[s] for s in names))  # noqa: E731
    brought = four("Chi-Yu", "Gyarados", "Archaludon", "Indeedee-F")
    with_rillaboom = four("Gyarados", "Archaludon", "Indeedee-F", "Rillaboom")
    actual = four("Chien-Pao", "Amoonguss", "Iron Hands", "Hatterene")
    no_spore = four("Iron Hands", "Chien-Pao", "Landorus-Therian", "Ursaluna")
    assert priced.pairing(brought, actual) - plain.pairing(brought, actual) == pytest.approx(-w)
    assert priced.pairing(with_rillaboom, actual) - plain.pairing(with_rillaboom, actual) == pytest.approx(-w / 2)
    assert priced.pairing(brought, no_spore) == pytest.approx(plain.pairing(brought, no_spore))  # no sleeper, no change

    def spore_weight(m):
        our_fours, their_fours = list(combinations(m.ours, lineup.BRING)), list(combinations(m.theirs, lineup.BRING))
        _, q, _ = lineup.solve_game([[m.pairing(f, g) for g in their_fours] for f in our_fours])
        return sum(q[j] for j, g in enumerate(their_fours) if idx["Amoonguss"] in g), q[their_fours.index(actual)]
    before, actual_before = spore_weight(plain)
    after, actual_after = spore_weight(priced)
    assert before < 0.01 and actual_before < 0.01  # what the deployed build predicted
    assert after >= 0.3, after  # a four holding their Spore user is now a real part of the prediction
    # (Their exact four stays near 0: the fitted preview weights also charge their Hatterene's Trick Room, and Landorus-T's
    # spread outscores Chien-Pao; that is outside sleep and not changed here.)
    assert actual_after < 0.05
    # The notes and the payload table say who Spore cannot touch, for the model to read.
    mine, theirs, roster = rosters(OURS_E3, THEIRS_E3)
    cands = lineup.shortlist(mine, theirs, roster, params=V2)
    assert any(n.startswith("v2:") and "Amoonguss" in n for n in cands[0].notes), cands[0].notes[0]
    sleep_notes = [next(n for n in c.notes if n.startswith("sleep: ")) for c in cands]
    assert all(("cannot sleep Rillaboom" in n) == ("rillaboom" in c.bring) for n, c in zip(sleep_notes, cands)), sleep_notes
    table = lineup.preview_table(mine, theirs, cands, V2)
    assert "Sleep: their Amoonguss (spore) cannot sleep our Rillaboom" in table and len(table) <= 2500
