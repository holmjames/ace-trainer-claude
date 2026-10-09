"""The deep Team Preview in the shortlist and in the agent: ``lineup_sim_budget_s`` 0 is the stage2 preview exactly;
above 0 the simulated plan leads the shortlist with its margins in the notes; the search keeps to its budget; and the
Team Preview model call gets only the clock the search left (the agent's _lineup), code #1 when under
``lineup_min_model_s``. The clock tests fake the time spent by moving the decision's start back, not by waiting."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent.pokemon import data, lineup, preview_sim, tuning
from agent.pokemon.memory import species_key

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")
LIVE = Path(__file__).parent / "fixtures" / "live"
REAL_SHORTLIST = lineup.shortlist

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


def test_v2_s_plan_stays_code_one_unless_the_search_beats_it_by_more_than_the_band(small_schedule):
    mine, theirs = profiles(*L3)
    v2 = lineup.shortlist_v2(mine, theirs, params=tuning.merged(), seed=4)[0]
    wide = lineup.shortlist_v2(mine, theirs, params=tuning.merged({"lineup_sim_keep_v2": 100.0}), seed=4, sim_budget_s=30.0)[0]
    assert (set(wide.bring), set(wide.leads)) == (set(v2.bring), set(v2.leads))
    assert wide.notes[0].startswith(f"sim: lead {v2.leads[0]} + {v2.leads[1]}") or wide.notes[0].startswith(f"sim: lead {v2.leads[1]} + {v2.leads[0]}")
    narrow = lineup.shortlist_v2(mine, theirs, params=tuning.merged({"lineup_sim_keep_v2": 0.0}), seed=4, sim_budget_s=30.0)
    if (set(narrow[0].bring), set(narrow[0].leads)) != (set(v2.bring), set(v2.leads)):
        assert any(n.startswith("sim changes v2's lineup") for n in narrow[0].notes)
        assert any(n.startswith("sim keeps v2's lead") for n in wide.notes)
    assert narrow[0].score >= max(c.score for c in narrow)


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


# -- the agent: the model gets only the clock the search left ---------------------------------------------------------


class RecordingProvider:
    model = "claude-primary"
    effort = "high"

    def __init__(self):
        self.calls: list[dict] = []
        self.last_model, self.last_latency_ms, self.last_usage = "claude-primary", 5, {}
        self.last_errors: list[str] = []
        self.last_attempts: list[dict] = []

    def complete_structured(self, messages, schema_name, schema, *, deadline=None, effort=None):
        self.calls.append({"deadline": deadline, "schema_name": schema_name})
        payload = json.loads(messages[-1]["content"])
        first = payload["computed_candidates"][0]
        return {"bring": first["bring"], "leads": first["leads"], "reasoning_summary": "code #1."}


def preview_decision(tmp_path, monkeypatch, *, budget: float, spent: float, clock: dict | None):
    """Team Preview of the Oct 8 win (bcbaabe2) through the agent, with the deep preview replaced by v2 plus ``spent``
    seconds taken off the decision's clock (the decision's start moved back, as if the search had run that long)."""
    from altruagent import DecisionContext
    from altruagent.models import GameState

    from agent.agent import PokemonAgent

    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = json.loads(next(LIVE.glob("bcbaabe2-*/team_preview-00000-seat0.json")).read_text())
    if clock is None:
        raw["observation"].pop("clock", None)
    else:
        raw["observation"]["clock"] = clock
    provider = RecordingProvider()
    agent = PokemonAgent(provider, version="preview-clock", log_dir=tmp_path, capture_dir="", log=lambda *_: None,
                         params=tuning.merged({"lineup_sim_budget_s": budget}))
    seen: dict = {}

    def deep(*args, sim_budget_s=0.0, **kwargs):
        seen["budget"] = sim_budget_s
        agent._decision_started -= spent
        return REAL_SHORTLIST(*args, **kwargs)

    monkeypatch.setattr(lineup, "shortlist", deep)
    ctx = DecisionContext(session_id="preview-clock", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                          agent_id="me", seat_position=0)
    decision = agent.choose_action(GameState.from_mcp_state(raw), ctx)
    rows = [json.loads(line) for f in Path(tmp_path).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    row = next(r for r in reversed(rows) if "candidates" in r)
    return getattr(decision, "action", decision), provider, row, seen


def test_the_preview_model_gets_the_decision_time_the_search_left(tmp_path, monkeypatch):
    payload, provider, row, seen = preview_decision(tmp_path, monkeypatch, budget=20.0, spent=25.0,
                                                    clock={"decision_seconds_left": 85, "bank_seconds_left": 420})
    assert seen["budget"] == 20.0
    assert len(provider.calls) == 1
    # min(60 cap, 85 - 25 - 8 = 52, bank (395 - 90) x 0.2 = 61): the decision term binds, less the real seconds the
    # preview's own code took (v2 and the table: about one, three at most here)
    assert 49.0 <= provider.calls[0]["deadline"] <= 52.0
    assert row["clock"]["rule"] == "decision" and 57.0 <= row["clock"]["decision_seconds_left"] <= 60.0
    assert payload["bring"] == row["candidates"][0]["bring"]


def test_with_under_ten_seconds_left_code_one_is_played_without_the_model(tmp_path, monkeypatch):
    payload, provider, row, _ = preview_decision(tmp_path, monkeypatch, budget=20.0, spent=70.0,
                                                 clock={"decision_seconds_left": 85, "bank_seconds_left": 420})
    assert provider.calls == []  # 85 - 70 - 8 = 7 s < lineup_min_model_s
    assert row["fallback"] is True and row["model"] is None
    assert any("clock too short for a Team Preview model call" in e for e in row["errors"])
    first = row["candidates"][0]
    assert sorted(payload["bring"]) == sorted(first["bring"]) and sorted(payload["leads"]) == sorted(first["leads"])
    # no server clock at all (an old simulator): the 85 s Team Preview allowance is charged the same way
    payload, provider, row, _ = preview_decision(tmp_path / "noclock", monkeypatch, budget=20.0, spent=70.0, clock=None)
    assert provider.calls == [] and row["fallback"] is True


def test_with_the_deep_preview_off_the_clock_goes_to_the_model_untouched(tmp_path, monkeypatch):
    payload, provider, row, seen = preview_decision(tmp_path, monkeypatch, budget=0.0, spent=0.0,
                                                    clock={"decision_seconds_left": 85, "bank_seconds_left": 420})
    assert seen["budget"] == 0.0
    assert provider.calls and provider.calls[0]["deadline"] == pytest.approx(60.0, abs=0.6)  # the stage2 cap
    assert row["clock"] == {"decision_seconds_left": 85.0, "bank_seconds_left": 420.0, "deadline_s": 60.0,
                            "remaining_turns": 14, "rule": "cap"}
