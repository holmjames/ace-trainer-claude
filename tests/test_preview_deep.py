"""The deep Team Preview in the shortlist and in the agent: ``lineup_sim_budget_s`` 0 is the stage2 preview exactly;
above 0 the simulated plan leads the shortlist with its margins in the notes; the search keeps to its budget; and the
Team Preview model call gets only the clock the search left (the agent's _lineup), code #1 when under
``lineup_min_model_s``."""

from __future__ import annotations

import json
import time

import pytest

from agent.pokemon import data, lineup, preview_sim, tuning
from agent.pokemon.memory import species_key

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")

# Oct 7 L2 (the review's better lead from this six: Gyarados + Archaludon) and Oct 7 L3.
L2 = (["Iron Bundle", "Iron Hands", "Archaludon", "Gyarados", "Whimsicott", "Hatterene"],
      ["Incineroar", "Flutter Mane", "Sylveon", "Rillaboom", "Kingambit", "Dondozo"])
L3 = (["Tornadus", "Garchomp", "Great Tusk", "Hatterene", "Ursaluna", "Indeedee-F"],
      ["Incineroar", "Iron Hands", "Whimsicott", "Archaludon", "Sylveon", "Amoonguss"])


def rosters(mine, theirs):
    return ({species_key(s): CATALOG[s] for s in mine}, {species_key(s): CATALOG[s] for s in theirs}, [data.to_id(s) for s in mine])


def profiles(mine, theirs):
    return [lineup.profile({**CATALOG[s]}) for s in mine], [lineup.profile(CATALOG[s]) for s in theirs]


@pytest.fixture
def small_schedule(monkeypatch):
    """A fixed, small search (1 rollout per plan and four first): it always finishes well inside the budget, so the
    result is the same on any machine."""
    monkeypatch.setattr(preview_sim.Search, "schedule", lambda self, budget_s: (1, 0.5))


def test_budget_zero_is_the_stage2_preview_exactly():
    P = tuning.merged()
    assert P["lineup_sim_budget_s"] == 20.0  # live default
    for six in (L2, L3):
        stage2 = lineup.shortlist(*rosters(*six), params=P)
        off = lineup.shortlist(*rosters(*six), params=P, sim_budget_s=0.0)
        assert [(c.bring, c.leads, c.score, c.notes) for c in off] == [(c.bring, c.leads, c.score, c.notes) for c in stage2]
        assert not any(n.startswith("sim") for c in off for n in c.notes)


def test_the_deep_preview_leads_with_its_best_plan_and_shows_the_margins(small_schedule):
    mine, theirs = profiles(*L2)
    a = lineup.shortlist_v2(mine, theirs, params=tuning.merged(), seed=11, sim_budget_s=30.0)
    b = lineup.shortlist_v2(mine, theirs, params=tuning.merged(), seed=11, sim_budget_s=30.0)
    assert [(c.bring, c.leads, c.score) for c in a] == [(c.bring, c.leads, c.score) for c in b]  # the same seed, the same plan
    assert len(a) == 3 and len({tuple(sorted(c.bring)) for c in a}) == 3  # three different fours, best plan first
    assert all(len(c.bring) == 4 and len(c.leads) == 2 and set(c.leads) <= set(c.bring) for c in a)
    assert a[0].score >= a[1].score >= a[2].score
    first = a[0].notes
    assert first[0].startswith(f"sim: lead {a[0].leads[0]} + {a[0].leads[1]}: margin ") and "worst" in first[0]
    assert any(n.startswith("sim, other leads of this four") for n in first)
    assert any(n.startswith("sim search: ") and "seed 0xb" in n for n in first)
    assert any(n.startswith("v2: ") for n in first) and any(n.startswith("game value") for n in first)
    assert all(c.notes[0].startswith("sim: lead ") for c in a)


def test_a_budget_too_short_for_one_round_or_a_failure_leaves_v2_s_lineup(monkeypatch):
    P = tuning.merged()
    mine, theirs = profiles(*L3)
    stage2 = lineup.shortlist_v2(mine, theirs, params=P, seed=3)
    short = lineup.shortlist_v2(mine, theirs, params=P, seed=3, sim_budget_s=0.01)
    assert [(c.bring, c.leads) for c in short] == [(c.bring, c.leads) for c in stage2]
    assert any(n.startswith("sim search: ") and n.endswith("v2's lineup stands") for n in short[0].notes)

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(lineup, "deep_candidates", broken)
    failed = lineup.shortlist_v2(mine, theirs, params=P, seed=3, sim_budget_s=5.0)
    assert [(c.bring, c.leads) for c in failed] == [(c.bring, c.leads) for c in stage2]
    assert "sim: failed (RuntimeError: boom); v2's lineup stands" in failed[0].notes


def test_the_whole_preview_call_keeps_to_its_budget():
    for budget in (1.0, 3.0):
        started = time.perf_counter()
        cands = lineup.shortlist(*rosters(*L3), params=tuning.merged(), sim_budget_s=budget)
        took = time.perf_counter() - started
        assert took <= budget + 0.5, (budget, took)
        assert cands and len(cands[0].bring) == 4


def test_their_side_spreads_weight_over_their_likely_fours_and_every_lead_pair():
    from itertools import combinations

    P = tuning.merged()
    mine, theirs = profiles(*L3)
    model = lineup.PreviewModel([p.card for p in mine], [p.card for p in theirs])
    our_fours, their_fours = list(combinations(model.ours, 4)), list(combinations(model.theirs, 4))
    payoff = [[model.pairing(f, g) for g in their_fours] for f in our_fours]
    our_mix, q, _ = lineup.solve_game(payoff)
    side = lineup.their_side(model, our_fours, their_fours, payoff, our_mix, q, P)
    assert int(P["lineup_sim_their_fours"]) <= len(side) <= int(P["lineup_sim_their_fours"]) + 1
    assert sum(w for w, _, _ in side) == pytest.approx(1.0)
    assert min(w for w, _, _ in side) >= P["lineup_sim_flat"] / len(side) - 1e-9
    assert len({four for _, four, _ in side}) == len(side)
    for _, four, pairs in side:
        assert len(pairs) == 6 and {p for p, _ in pairs} == set(combinations(four, 2))
        assert sum(w for _, w in pairs) == pytest.approx(1.0)
        assert min(w for _, w in pairs) >= P["lineup_sim_lead_floor"] / 6 - 1e-9
    # the four the Oct 7 opponent actually brought in L3 is among them
    names = {6 + j: s for j, s in enumerate(L3[1])}
    assert any({names[k] for k in four} == {"Incineroar", "Iron Hands", "Whimsicott", "Sylveon"} for _, four, _ in side)
