"""Exposure-first near-ties (``draft_tiebreak_exposure``; the Oct 9 draft review's P1): with the key on, the window the
secret tie-break draws from (``draft.tiebreak``) is first cut to the cards the fewest undrafted pool cards beat
(``ScoredCard.exposure``), and the same seeded draw is made among those. At 0 the draw is the shipped one, bit for bit:
the cut is never consulted and the notes are unchanged."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from altruagent.models import GameState, LegalAction

from agent.pokemon import data, draft, matchup, tuning
from agent.pokemon.memory import MatchMemory, observation_dict

EPS = tuning.DEFAULTS["draft_tiebreak_eps"]
ON = tuning.merged({"draft_tiebreak_exposure": 1})
LIVE = Path(__file__).parent / "fixtures" / "live"
CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
needs_data = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")

# Oct 7 L3 (sim/draft_eval.py LIVE_POOLS 7ce71025), our first pick
L3 = ["Amoonguss", "Archaludon", "Dondozo", "Garchomp", "Great Tusk", "Grimmsnarl", "Hatterene", "Incineroar", "Indeedee-F",
      "Iron Hands", "Rillaboom", "Sableye", "Sylveon", "Ting-Lu", "Tornadus", "Ursaluna", "Whimsicott", "Wo-Chien"]


@pytest.fixture(autouse=True)
def _shipped_table(monkeypatch):
    """The shipped first-pick table, whatever the environment says (the table is read once per process)."""
    monkeypatch.delenv(draft.FIRST_PICK_ADJUST_ENV, raising=False)
    monkeypatch.setattr(draft, "_ADJUST_TABLES", {})


def card(species: str, score: float, exposure: int) -> draft.ScoredCard:
    return draft.ScoredCard(LegalAction(f"draft_pick:{species}", f"Draft {species}", {}), species, species, score, [], exposure=exposure)


def window(ranked: list[draft.ScoredCard], eps: float = EPS) -> list[str]:
    top = matchup.win_probability(max(s.score for s in ranked))
    return [s.species for s in ranked if top - matchup.win_probability(s.score) <= eps]


def note(pick: draft.ScoredCard) -> str | None:
    return next((n for n in pick.notes if n.startswith("exposure tie-break")), None)


def test_the_key_ships_off_and_scored_cards_default_to_no_exposure():
    assert tuning.DEFAULTS["draft_tiebreak_exposure"] == 0.0
    assert tuning.merged({"draft_tiebreak_exposure": 1})["draft_tiebreak_exposure"] == 1.0
    old = draft.ScoredCard(LegalAction("draft_pick:x", "Draft x", {}), "x", "X", 1.0, [])  # every existing constructor still works
    assert old.exposure == 0


# -- the cut itself (draft.tiebreak_exposure) ------------------------------------------------------------------------------


def test_the_window_is_cut_to_its_fewest_counters_cards_before_the_draw():
    ranked = [card("A", 1.00, 3), card("B", 0.99, 1), card("C", 0.98, 2), card("D", 0.50, 0)]
    assert window(ranked) == ["A", "B", "C"]  # D is far behind: beaten by nothing, but outside the window it never counts
    for seed in range(40):
        index, aside = draft.tiebreak_exposure(ranked, EPS, seed, "pick:4")
        assert ranked[index].species == "B"
        assert [s.species for s in aside] == ["A", "C"]
    assert {draft.tiebreak([s.score for s in ranked], EPS, seed, "pick:4") for seed in range(40)} == {0, 1, 2}  # the seed alone: any of three


def test_ties_in_exposure_go_to_the_same_seeded_draw_among_the_cards_kept():
    ranked = [card("A", 1.00, 2), card("B", 0.99, 1), card("C", 0.985, 1)]
    chosen = set()
    for seed in range(40):
        index, aside = draft.tiebreak_exposure(ranked, EPS, seed, "pick:4")
        assert index == random.Random(f"{seed}:pick:4").choice([1, 2])  # tiebreak()'s draw, over the two kept
        assert [s.species for s in aside] == ["A"]
        chosen.add(index)
    assert chosen == {1, 2}
    same = [card("A", 1.00, 2), card("B", 0.99, 2), card("C", 0.985, 2)]  # one exposure across the window: nothing is cut
    for seed in range(40):
        assert draft.tiebreak_exposure(same, EPS, seed, "pick:4") == (draft.tiebreak([s.score for s in same], EPS, seed, "pick:4"), [])


def test_a_single_candidate_window_or_no_window_is_the_best_card():
    apart = [card("A", 1.00, 6), card("B", 0.80, 0)]  # 4 points of win chance apart: not a tie, however few cards beat B
    assert window(apart) == ["A"]
    for seed in (0, 1, 7):
        assert draft.tiebreak_exposure(apart, EPS, seed, "pick:2") == (0, [])
    close = [card("A", 1.00, 6), card("B", 0.995, 0)]
    assert draft.tiebreak_exposure(close, 0.0, 7, "pick:2") == (0, [])  # eps 0: no window
    assert draft.tiebreak_exposure(close, EPS, None, "pick:2") == (0, [])  # no seed: always the best
    assert draft.tiebreak_exposure(close, EPS, 7, "pick:2") == (1, [close[0]])
    assert draft.tiebreak_exposure([card("A", 0.3, 4)], EPS, 7, "pick:2") == (0, [])
    with pytest.raises(ValueError):
        draft.tiebreak_exposure([], EPS, 7, "pick:0")


# -- choose_pick_v2 with the key on and off ----------------------------------------------------------------------------


def test_the_note_names_the_cards_set_aside_and_what_the_seed_alone_would_take(monkeypatch):
    def ranking(legal, memory, params):
        return [card("Incineroar", 1.00, 6), card("Archaludon", 0.99, 3), card("Chi-Yu", 0.985, 3), card("Tornadus", 0.40, 0)]

    monkeypatch.setattr(draft, "rank_picks_v2", ranking)
    alone = set()
    for seed in range(30):
        memory = MatchMemory()
        memory.tiebreak_seed = seed
        off = draft.choose_pick_v2([], memory, tuning.DEFAULTS)
        on = draft.choose_pick_v2([], memory, ON)
        assert note(off) is None
        assert on.species in ("Archaludon", "Chi-Yu")
        others = [s for s in ("Incineroar", "Archaludon", "Chi-Yu") if s != on.species]
        assert on.notes[0] == f"tie-break seed {seed:#x}: {on.species} over {others} (within 1% win chance)"
        assert on.notes[1] == (f"exposure tie-break: {on.species} is beaten by 3 undrafted pool cards, the fewest in the window; set aside "
                               f"Incineroar (6)" + ("" if off.species == on.species else f"; the seed alone would take {off.species}"))
        alone.add(off.species)
    assert alone == {"Incineroar", "Archaludon", "Chi-Yu"}

    def single(legal, memory, params):
        return [card("Iron Hands", 1.00, 1), card("Indeedee-F", 0.99, 2)]

    monkeypatch.setattr(draft, "rank_picks_v2", single)
    memory = MatchMemory()
    memory.tiebreak_seed = 5
    on = draft.choose_pick_v2([], memory, ON)
    assert on.species == "Iron Hands" and note(on).startswith("exposure tie-break: Iron Hands is beaten by 1 undrafted pool card, ")


def live_decisions(session: str, seat: int):
    """One live session's captured draft decisions for a seat, each with the memory that observed it (as the agent)."""
    folder = next(LIVE.glob(f"{session}*"))
    memory = MatchMemory()
    for path in sorted(folder.glob(f"draft-*-seat{seat}.json"), key=lambda p: int(re.search(r"-(\d+)-", p.name).group(1))):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        memory.observe_draft(observation_dict(state), my_turn=True, agent_id=(raw.get("current_actor") or {}).get("agent_id"))
        yield state, memory


@needs_data
def test_key_zero_never_consults_the_cut_and_keeps_the_shipped_pick_and_notes(monkeypatch):
    def cut(*args, **kwargs):
        raise AssertionError("tiebreak_exposure consulted at draft_tiebreak_exposure 0")

    monkeypatch.setattr(draft, "tiebreak_exposure", cut)
    for state, memory in live_decisions("07a3cb1f", 1):  # Oct 9, they drafted first: our six picks
        ranked = draft.rank_picks_v2(state.legal_actions, memory, tuning.DEFAULTS)
        n = len(memory.my_cards) + len(memory.opp_cards)
        for seed in (0, 0x5EED):
            memory.tiebreak_seed = seed
            pick = draft.choose_pick_v2(state.legal_actions, memory, tuning.DEFAULTS)
            shipped = ranked[draft.tiebreak([s.score for s in ranked], EPS, seed, f"pick:{n}")]
            notes = [x for x in pick.notes if not x.startswith("first-pick table")]  # our first two picks also log the table
            assert (pick.card_id, pick.score, notes[:-1]) == (shipped.card_id, shipped.score, shipped.notes)
            assert notes[-1].startswith(f"tie-break seed {seed:#x}") and note(pick) is None


@needs_data
def test_on_a_live_tie_the_card_fewer_pool_cards_beat_is_taken():
    # 07a3cb1f our pick 10: the only tie window of that draft (the Oct 9 review), whichever card the seed alone takes.
    state, memory = next((s, m) for s, m in live_decisions("07a3cb1f", 1) if len(m.my_cards) + len(m.opp_cards) == 9)
    ranked = draft.rank_picks_v2(state.legal_actions, memory, tuning.DEFAULTS)
    tied = [s for s in ranked if s.species in window(ranked)]
    assert len({s.exposure for s in tied}) > 1, [(s.species, s.exposure) for s in tied]
    fewest = {s.species for s in tied if s.exposure == min(t.exposure for t in tied)}
    alone = set()
    for seed in range(6):
        memory.tiebreak_seed = seed
        off = draft.choose_pick_v2(state.legal_actions, memory, tuning.DEFAULTS)
        on = draft.choose_pick_v2(state.legal_actions, memory, ON)
        assert on.species in fewest and on.score == next(s.score for s in ranked if s.species == on.species)  # values untouched
        assert note(on) is not None and note(off) is None
        alone.add(off.species)
    assert alone == {s.species for s in tied}  # the seed alone takes either card


@needs_data
def test_exposure_is_the_count_of_undrafted_pool_cards_that_beat_the_card():
    memory = MatchMemory()
    memory.my_seat_key = "me"
    for species in L3:
        memory.pool_cards[CATALOG[species]["card_id"]] = CATALOG[species]
    legal = [LegalAction(f"draft_pick:{c['card_id']}", f"Draft {c['species']}", {}) for c in memory.pool_cards.values()]
    ranked = draft.rank_picks_v2(legal, memory, tuning.DEFAULTS)
    assert len(ranked) == len(L3)
    P = tuning.DEFAULTS
    board = draft._board_for([CATALOG[s] for s in L3], float(P["draft_card_w"]), float(P["draft_tier_adjust"]))
    index = {u.species: i for i, u in enumerate(board.units)}
    for s in ranked:
        i = index[s.species]
        assert s.exposure == sum(1 for j in index.values() if j != i and board.base[j][i] >= draft.BEATS_MARGIN)
        # the first-pick policy counts the same thing on its own (first_pick_terms) and prints it in its note
        policy = next(n for n in s.notes if n.startswith("first-pick policy"))
        assert f"beaten by {s.exposure} pool cards" in policy
    assert len({s.exposure for s in ranked}) > 1
