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


def test_todays_winning_lead_still_passes():
    s = Session("W bcbaabe2")
    live = s.pair("Archaludon", "Arcanine")
    # Against the leads they actually opened with it loses nothing and leads no exposed setter: nothing to veto.
    assert s.model.opening(live, s.live_opp).ko_against == []
    assert s.model.fake_out_exposed(live, s.live_opp) == []
    # Against the model's predicted leads no pair of that four is safe (a Basculegion lead would take Arcanine or
    # Milotic), so the hard veto does not apply and the live pair stays in the same one-lead-at-risk class as the pick.
    chosen, notes = s.choose()
    assert s.lost(live) == 1 and s.lost(s.pair(*chosen)) == 1
    assert not any(n.startswith("lead veto") for n in notes), notes
    assert "Archaludon" in chosen


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
