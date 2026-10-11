"""Support-blunted answers in the draft (``draft_support_w``; the Oct 9 support-counter analysis, pre-registration #2):
an answer Y to a card X is blunted by an Intimidate partner of X (a physical Y one Attack stage down needs another hit)
or by a partner that resists Y's attack and that Y does not beat (a safe switch-in), when Y cannot afford a free hit
from X. With the key on, the draft's search sees each blunted answer's effective margin scaled by 1 - min(1, w * n),
n = how many of the two kinds the holder supplies, in both directions. At 0 the search gets the board itself. Deploy #4
ships it on at 0.25 (pre-registration #2's w), so the tests below set the key both ways explicitly."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from altruagent.models import GameState, LegalAction

from agent.pokemon import data, draft, lineup, matchup, tuning
from agent.pokemon.memory import MatchMemory, observation_dict

NO_TABLE = {"draft_card_adjust_w": 0.0, "draft_card_adjust_them_w": 0.0}  # the per-card table off (deploy #5 ships it on)
ON = tuning.merged({"draft_support_w": 0.25, **NO_TABLE})  # deploy #4's draft (09a95b0)
OFF = tuning.merged({"draft_support_w": 0.0, **NO_TABLE})  # the Oct 10 deploy's draft (81c4591)
LIVE = Path(__file__).parent / "fixtures" / "live"
CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
needs_data = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")


@pytest.fixture(autouse=True)
def _shipped_table(monkeypatch):
    monkeypatch.delenv(draft.FIRST_PICK_ADJUST_ENV, raising=False)
    monkeypatch.setattr(draft, "_ADJUST_TABLES", {})


def board_of(*species: str):
    board = matchup.Board([matchup.unit(CATALOG[s]) for s in species])
    return board, {u.species: i for i, u in enumerate(board.units)}


def blunters(board, at: dict, y: str, x: str) -> tuple[set[str], set[str]]:
    """(safe switch-ins, Intimidate users) on the board that blunt y's answer to x."""
    pair = draft.support_table(board)[at[y]][at[x]]
    if pair is None:
        return set(), set()
    return ({u.species for k, u in enumerate(board.units) if pair[0] >> k & 1},
            {u.species for k, u in enumerate(board.units) if pair[1] >> k & 1})


def test_the_key_ships_on_at_the_pre_registered_w():
    assert tuning.DEFAULTS["draft_support_w"] == 0.25 == ON["draft_support_w"]
    assert OFF["draft_support_w"] == 0.0


# -- the two kinds of support (pure functions of the sets) ------------------------------------------------------------


@needs_data
def test_intimidate_blunts_a_physical_answer_that_needs_another_hit():
    board, at = board_of("Iron Hands", "Flutter Mane", "Incineroar", "Gyarados", "Amoonguss", "Kingambit", "Garchomp")
    assert board.base[at["Iron Hands"]][at["Flutter Mane"]] >= draft.BEATS_MARGIN
    switch, intimidate = blunters(board, at, "Iron Hands", "Flutter Mane")
    assert intimidate == {"Incineroar", "Gyarados"}  # Wild Charge one stage down no longer KOs in one hit
    assert switch == {"Amoonguss", "Garchomp"}  # Grass resists Wild Charge, Ground is immune, and Iron Hands beats neither
    # Kingambit's Sucker Punch into Flutter Mane: a one-hit KO with room to spare, so nothing blunts it
    assert blunters(board, at, "Kingambit", "Flutter Mane") == (set(), set())


@needs_data
def test_a_partner_that_resists_the_answer_and_is_not_beaten_by_it_blunts_it():
    board, at = board_of("Indeedee-F", "Iron Hands", "Incineroar", "Amoonguss", "Thundurus", "Gyarados", "Rillaboom")
    switch, intimidate = blunters(board, at, "Indeedee-F", "Iron Hands")
    assert "Incineroar" in switch  # Dark: immune to Indeedee-F's Psychic
    assert "Amoonguss" not in switch  # Poison: hit super-effectively
    switch, intimidate = blunters(board, at, "Thundurus", "Gyarados")
    assert {"Rillaboom", "Iron Hands"} <= switch and "Incineroar" not in switch and not intimidate  # Thunderbolt: special


# -- the draft's view of the board -------------------------------------------------------------------------------------


@needs_data
def test_the_view_scales_blunted_answers_in_both_directions():
    board, at = board_of("Iron Hands", "Flutter Mane", "Incineroar", "Indeedee-F", "Garchomp")
    for w in (0.25, 0.5):
        view = draft.SupportView(board, w)
        # ours blunted by theirs: our Iron Hands into their Flutter Mane beside their Incineroar (Intimidate)
        us, them = [at["Iron Hands"]], [at["Flutter Mane"], at["Incineroar"]]
        raw, scaled = board.margins(us, them)[0], view.margins(us, them)[0]
        assert raw[0][0] > 0 and scaled[0][0] == pytest.approx(raw[0][0] * (1 - w))
        # theirs blunted by ours: their Indeedee-F into our Iron Hands beside our Incineroar (a safe switch-in)
        us, them = [at["Iron Hands"], at["Incineroar"]], [at["Indeedee-F"]]
        raw, scaled = board.margins(us, them)[0], view.margins(us, them)[0]
        assert raw[0][0] < 0 and scaled[0][0] == pytest.approx(raw[0][0] * (1 - w))
        assert scaled[1] == raw[1]  # nothing blunts the other pair
        # both kinds at once: n = 2
        switch, intimidate = blunters(board, at, "Iron Hands", "Flutter Mane")
        assert "Garchomp" in switch and "Incineroar" in intimidate
        us, them = [at["Iron Hands"]], [at["Flutter Mane"], at["Incineroar"], at["Garchomp"]]
        raw, scaled = board.margins(us, them)[0], view.margins(us, them)[0]
        assert scaled[0][0] == pytest.approx(raw[0][0] * max(0.0, 1 - 2 * w))


@needs_data
def test_a_kind_counts_once_however_many_partners_supply_it():
    board, at = board_of("Thundurus", "Gyarados", "Rillaboom", "Iron Hands")
    view = draft.SupportView(board, 0.25)
    us, them = [at["Thundurus"]], [at["Gyarados"], at["Rillaboom"], at["Iron Hands"]]  # two safe switch-ins
    raw, scaled = board.margins(us, them)[0], view.margins(us, them)[0]
    assert raw[0][0] > 0 and scaled[0][0] == pytest.approx(raw[0][0] * 0.75)


@needs_data
def test_the_values_move_the_way_the_support_says():
    board, at = board_of("Iron Hands", "Flutter Mane", "Incineroar", "Indeedee-F")
    view = draft.SupportView(board, 0.25)
    blunted = ([at["Iron Hands"]], [at["Flutter Mane"], at["Incineroar"]])
    shielded = ([at["Iron Hands"], at["Incineroar"]], [at["Indeedee-F"]])
    assert view.score(*blunted) < board.score(*blunted)
    assert view.score(*shielded) > board.score(*shielded)
    assert view.score4(*blunted) < board.score4(*blunted)
    # everything else is the board's: races, roles, holes, sleep
    assert view.base is board.base and view.holes(*blunted) == board.holes(*blunted)


# -- the draft with the key off and on ---------------------------------------------------------------------------------


def live_decisions(session: str, seat: int):
    folder = next(LIVE.glob(f"{session}*"))
    memory = MatchMemory()
    for path in sorted(folder.glob(f"draft-*-seat{seat}.json"), key=lambda p: int(re.search(r"-(\d+)-", p.name).group(1))):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        memory.observe_draft(observation_dict(state), my_turn=True, agent_id=(raw.get("current_actor") or {}).get("agent_id"))
        yield state, memory


@needs_data
def test_key_zero_hands_the_search_the_board_itself(monkeypatch):
    def view(*args, **kwargs):
        raise AssertionError("SupportView built at draft_support_w 0")

    monkeypatch.setattr(draft, "SupportView", view)
    for state, memory in live_decisions("07a3cb1f", 1):
        for pick in draft.rank_picks_v2(state.legal_actions, memory, OFF):
            assert not any(n.startswith("support:") for n in pick.notes)


@needs_data
def test_with_the_key_on_the_search_values_change_and_say_why():
    changed = noted = 0
    for state, memory in live_decisions("07a3cb1f", 1):
        off = {s.species: s.score for s in draft.rank_picks_v2(state.legal_actions, memory, OFF)}
        on = draft.rank_picks_v2(state.legal_actions, memory, ON)
        defaults = draft.rank_picks_v2(state.legal_actions, memory, tuning.DEFAULTS)  # deploy #5: the key on, with the card table
        with_table = draft.rank_picks_v2(state.legal_actions, memory, dict(ON, draft_card_adjust_w=0.5, draft_card_adjust_them_w=0.5))
        assert [(s.species, s.score, s.notes) for s in defaults] == [(s.species, s.score, s.notes) for s in with_table]
        assert {s.species for s in on} == set(off)
        changed += sum(1 for s in on if s.score != off[s.species])
        noted += sum(1 for s in on if any(n.startswith("support:") for n in s.notes))
    assert changed and noted


@needs_data
def test_team_preview_never_sees_the_key():
    ours = ["Incineroar", "Iron Hands", "Flutter Mane", "Garchomp", "Amoonguss", "Gyarados"]
    theirs = ["Indeedee-F", "Kingambit", "Thundurus", "Rillaboom", "Archaludon", "Chi-Yu"]
    mine = [lineup.profile(CATALOG[s]) for s in ours]
    them = [lineup.profile(CATALOG[s]) for s in theirs]
    for seed in (None, 3):
        off = lineup.shortlist_v2(mine, them, params=OFF, seed=seed)
        on = lineup.shortlist_v2(mine, them, params=ON, seed=seed)
        assert [(c.as_payload(), c.score, c.notes) for c in on] == [(c.as_payload(), c.score, c.notes) for c in off]


def test_a_legal_action_list_without_known_cards_is_untouched_by_the_key():
    assert draft.rank_picks_v2([LegalAction("draft_pick:x", "Draft x", {})], MatchMemory(), ON) == []
