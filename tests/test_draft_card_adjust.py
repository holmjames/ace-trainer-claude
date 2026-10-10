"""Per-card corrections in the draft's finished-draft value (``draft_card_adjust_w``; draft lane 8, Oct 10): the search's
own value of a finished draft is overconfident, and on 13,841 of our simulator games some cards won more, or less, than
it credits. data/draft_card_adjust.json holds those corrections in log-odds: "us" when the card is in the four we would
bring, "them" for each of their considered fours that holds it. With the key above 0, DraftSearch.finished adds key x
table to every four-against-four value before the mean / worst blend; their fours are still chosen without it. At 0
the table is never read and the search runs its old code: the 132 live draft and 22 Team Preview decisions replay
identically to 39f50af (full-precision values, notes, picks)."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import pytest

from altruagent.models import GameState, LegalAction

from agent.pokemon import data, draft, lineup, matchup, tuning
from agent.pokemon.memory import MatchMemory, observation_dict

W = 0.5  # the gate's weight
ON = tuning.merged({"draft_card_adjust_w": W})
SUPPORT = tuning.merged({"draft_support_w": 0.25})  # the deploy #4 draft
SUPPORT_ON = tuning.merged({"draft_support_w": 0.25, "draft_card_adjust_w": W})
LIVE = Path(__file__).parent / "fixtures" / "live"
CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
needs_data = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")


@pytest.fixture(autouse=True)
def _fresh_tables(monkeypatch):
    monkeypatch.delenv(draft.FIRST_PICK_ADJUST_ENV, raising=False)
    monkeypatch.setattr(draft, "_ADJUST_TABLES", {})
    monkeypatch.setattr(draft, "_CARD_ADJUST_TABLES", {})
    monkeypatch.setattr(draft, "_CARD_ADJUST_ERRORS", {})


def board_of(*species: str):
    board = matchup.Board([matchup.unit(CATALOG[s]) for s in species])
    return board, {u.species: i for i, u in enumerate(board.units)}


def live_decisions(session: str, seat: int):
    folder = next(LIVE.glob(f"{session}*"))
    memory = MatchMemory()
    for path in sorted(folder.glob(f"draft-*-seat{seat}.json"), key=lambda p: int(re.search(r"-(\d+)-", p.name).group(1))):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        memory.observe_draft(observation_dict(state), my_turn=True, agent_id=(raw.get("current_actor") or {}).get("agent_id"))
        yield path.name, state, memory


def write(tmp_path: Path, body) -> Path:
    path = tmp_path / "card_adjust.json"
    path.write_text(body if isinstance(body, str) else json.dumps(body))
    return path


def test_the_key_ships_off():
    assert tuning.DEFAULTS["draft_card_adjust_w"] == 0.0
    assert ON["draft_card_adjust_w"] == W


# -- the data file ------------------------------------------------------------------------------------------------------


@needs_data
def test_the_shipped_table_loads_and_names_only_catalog_cards_within_its_limit():
    raw = json.loads(draft.CARD_ADJUST_FILE.read_text())
    assert raw["_doc"] and set(raw) >= {"us", "them"}
    for side in ("us", "them"):
        for species, value in raw[side].items():
            assert species in CATALOG, species  # the catalog's own spelling
            assert value and abs(value) <= draft.CARD_ADJUST_LIMIT
    assert "Archaludon" not in raw["us"] and "Archaludon" not in raw["them"]
    table = draft.card_adjust_table()
    assert not draft._CARD_ADJUST_ERRORS
    assert (len(table.us), len(table.them), table.entries()) == (16, 21, 37)
    # pinned from the gate's frozen table (skeptic work/table.json, sha256 01e6cf6c...): the largest entries each side
    assert table.us["irontreads"] == 0.5 and table.us["torkoal"] == -0.5 and table.us["kingambit"] == 0.436
    assert table.them["urshifurapidstrike"] == -0.5 and table.them["whimsicott"] == 0.464


@pytest.mark.parametrize("body, why", [
    ("{not json", "JSONDecodeError"),
    (["us"], "expected a JSON object"),
    ({"us": {"Archaludon": 0.1}, "them": {}}, "archaludon"),
    ({"us": {}, "them": {"Archaludon": -0.1}}, "archaludon"),
    ({"us": {"Kingambit": 0.6}, "them": {}}, "beyond"),
    ({"us": {}, "them": {"Missingno": 0.1}}, "not a catalog card"),
    ({"us": {"Kingambit": "lots"}, "them": {}}, "ValueError"),
    ({"us": {"Kingambit": float("nan")}, "them": {}}, "non-finite"),
    ({"us": ["Kingambit"], "them": {}}, "must map species"),
])
@needs_data
def test_a_table_that_breaks_a_rule_counts_as_empty_with_one_warning(tmp_path, capsys, body, why):
    path = write(tmp_path, body)
    table = draft.card_adjust_table(path)
    assert table == draft.CardAdjustTable() and table.entries() == 0
    assert why in draft._CARD_ADJUST_ERRORS[str(path)]
    assert capsys.readouterr().err.count("WARNING: draft card table") == 1
    assert draft.card_adjust_table(path) is table  # read once per process
    assert capsys.readouterr().err == ""


@needs_data
def test_a_missing_shipped_table_says_so_on_every_ranked_card(tmp_path, monkeypatch):
    monkeypatch.setattr(draft, "CARD_ADJUST_FILE", tmp_path / "missing.json")
    name, state, memory = next(live_decisions("07a3cb1f", 1))
    for pick in draft.rank_picks_v2(state.legal_actions, memory, ON):
        lines = [n for n in pick.notes if n.startswith("card table")]
        assert len(lines) == 1 and "0 entries" in lines[0] and "WARNING" in lines[0]


# -- the key at 0: nothing changes ----------------------------------------------------------------------------------------


@needs_data
def test_key_zero_never_reads_the_table_and_the_search_gets_no_adjustment(monkeypatch):
    def unread(*args, **kwargs):
        raise AssertionError("card table read at draft_card_adjust_w 0")

    monkeypatch.setattr(draft, "card_adjust_table", unread)
    seen = []
    original = draft.DraftSearch.__init__

    def init(self, board, **kwargs):
        seen.append(kwargs.get("card_adjust"))
        original(self, board, **kwargs)

    monkeypatch.setattr(draft.DraftSearch, "__init__", init)
    for params in (tuning.DEFAULTS, SUPPORT):
        for _, state, memory in live_decisions("07a3cb1f", 1):
            for pick in draft.rank_picks_v2(state.legal_actions, memory, params):
                assert not any(n.startswith("card table") for n in pick.notes)
    assert seen and all(a is None for a in seen)


@needs_data
@pytest.mark.parametrize("off, on", [(tuning.DEFAULTS, ON), (SUPPORT, SUPPORT_ON)])
def test_an_empty_table_at_the_key_values_every_pick_exactly_as_key_zero(monkeypatch, off, on):
    """The adjusted path is the old path plus the table: with nothing in the table it gives the same full-precision
    values, order and notes (but the card table line) as the key at 0."""
    monkeypatch.setattr(draft, "card_adjust_table", lambda path=None: draft.CardAdjustTable())
    values: list[float] = []
    original = draft.DraftSearch.plan

    def plan(self, *args, **kwargs):
        p = original(self, *args, **kwargs)
        values.append(p)
        return p

    monkeypatch.setattr(draft.DraftSearch, "plan", plan)
    for _, state, memory in live_decisions("bcbaabe2", 0):
        values.clear()
        a = draft.rank_picks_v2(state.legal_actions, memory, off)
        va = [p.value for p in values]
        values.clear()
        b = draft.rank_picks_v2(state.legal_actions, memory, on)
        vb = [p.value for p in values]
        assert va == vb  # bit for bit
        assert [(s.species, s.score, s.exposure, s.notes) for s in a] == \
               [(s.species, s.score, s.exposure, [n for n in s.notes if not n.startswith("card table")]) for s in b]


# -- what the table does -------------------------------------------------------------------------------------------------

US = ["Kingambit", "Incineroar", "Flutter Mane", "Rillaboom", "Amoonguss", "Garchomp"]
THEM = ["Gholdengo", "Iron Hands", "Dragonite", "Indeedee-F", "Ursaluna", "Whimsicott"]


def search_pair(table: draft.CardAdjustTable, *, four_eval: bool = True, hole_w: float = 0.5):
    board, at = board_of(*US, *THEM)
    plain = draft.DraftSearch(board, four_eval=four_eval, hole_w=hole_w)
    adjusted = draft.DraftSearch(board, four_eval=four_eval, hole_w=hole_w, card_adjust=table.vectors(board, W))
    return board, at, plain, adjusted, [at[s] for s in US], [at[s] for s in THEM]


@needs_data
def test_one_card_in_our_four_moves_the_value_by_exactly_w_times_its_entry():
    t = 0.436
    board, at, plain, adjusted, us, them = search_pair(draft.CardAdjustTable(us={"kingambit": t}))
    k = at["Kingambit"]
    v0, four0 = plain.finished(us, them, include=k)  # every four counted holds Kingambit
    v1, four1 = adjusted.finished(us, them, include=k)
    assert four1 == four0 and v1 == pytest.approx(v0 + W * t, abs=1e-12)
    # a six without the card gains nothing, bit for bit
    board, at = board_of(*US, *THEM, "Sylveon")
    plain = draft.DraftSearch(board, hole_w=0.5)
    adjusted = draft.DraftSearch(board, hole_w=0.5, card_adjust=draft.CardAdjustTable(us={"kingambit": t}).vectors(board, W))
    six = [at["Sylveon"]] + [at[s] for s in US[1:]]
    assert adjusted.finished(six, [at[s] for s in THEM]) == plain.finished(six, [at[s] for s in THEM])


@needs_data
def test_their_fours_are_chosen_without_the_table():
    """A huge "them" entry (straight into the search, past the loader's limit) adds to the fours of theirs that hold
    the card, but the fours considered are still their likeliest by Board.score4 against our six."""
    board, at = board_of(*US, *THEM)
    us, them = [at[s] for s in US], [at[s] for s in THEM]
    x = at["Whimsicott"]
    a_us, a_them = [0.0] * board.n, [0.0] * board.n
    a_them[x] = 100.0
    search = draft.DraftSearch(board, hole_w=0.0, card_adjust=(a_us, a_them))
    likeliest = sorted(combinations(them, 4), key=lambda g: -board.score4(list(g), us))[:3]
    assert any(x in g for g in likeliest) and not all(x in g for g in likeliest)  # the case means something
    expected = max(
        0.75 * sum(v) / len(v) + 0.25 * min(v)
        for v in ([board.score4(list(f), list(g)) + (100.0 if x in g else 0.0) for g in likeliest] for f in combinations(us, 4)))
    assert search.finished(us, them)[0] == pytest.approx(expected, abs=1e-9)


@needs_data
def test_the_six_against_six_fallback_adds_both_six_sums():
    table = draft.CardAdjustTable(us={"kingambit": 0.4, "rillaboom": -0.2}, them={"whimsicott": 0.3, "gholdengo": -0.1})
    board, at, plain, adjusted, us, them = search_pair(table, four_eval=False)
    assert adjusted.finished(us, them)[0] == pytest.approx(plain.finished(us, them)[0] + W * (0.4 - 0.2 + 0.3 - 0.1), abs=1e-12)


@needs_data
def test_the_correction_flips_a_near_tie_toward_the_card_the_table_rates_higher():
    """Live 6ce0b81e seat 0, our first pick (they drafted first): at the key's 0 Iron Hands leads Kingambit by 0.017
    log-odds, inside the tie window. The table rates Kingambit +0.436 in our four and Iron Hands -0.166, and at the
    gate's weight Kingambit is the clear pick."""
    name, state, memory = next(live_decisions("6ce0b81e", 0))
    assert not memory.my_cards and memory.opp_cards  # our first pick, after theirs
    table = draft.card_adjust_table()
    assert table.us["kingambit"] > 0 > table.us["ironhands"]
    off = draft.rank_picks_v2(state.legal_actions, memory, tuning.DEFAULTS)
    assert [s.species for s in off[:2]] == ["Iron Hands", "Kingambit"] and 0 < off[0].score - off[1].score < 0.02
    on = {s.species: s for s in draft.rank_picks_v2(state.legal_actions, memory, ON)}
    best = max(on.values(), key=lambda s: s.score)
    assert best.species == "Kingambit" and on["Kingambit"].score - on["Iron Hands"].score > 0.2
    assert "card table (weight 0.5, 37 entries): Kingambit +0.44 in our four" in best.notes


# -- what the table never reaches -------------------------------------------------------------------------------------------


@needs_data
def test_the_table_never_reaches_archaludon_the_rollouts_or_the_counter_picker():
    table = draft.card_adjust_table()
    board, at = board_of("Archaludon", *US, *THEM[:5])
    a_us, a_them = table.vectors(board, W)
    assert a_us[at["Archaludon"]] == 0.0 and a_them[at["Archaludon"]] == 0.0
    plain = draft.DraftSearch(board, hole_w=0.5)
    adjusted = draft.DraftSearch(board, hole_w=0.5, card_adjust=(a_us, a_them))
    us, them = [at["Archaludon"], at["Kingambit"]], [at["Gholdengo"]]
    pool = [i for i in range(board.n) if i not in us + them]
    for side in ("us", "them"):  # the greedy rollouts value picks by Board.score, which the table never touches
        assert adjusted.greedy(side, us, them, pool) == plain.greedy(side, us, them, pool)
    assert adjusted.counter(us, them, pool) == plain.counter(us, them, pool)
    assert adjusted.reply_set(us, them, pool) == plain.reply_set(us, them, pool)


@needs_data
def test_the_board_the_first_pick_policy_and_exposure_are_untouched():
    for name, state, memory in live_decisions("07a3cb1f", 1):
        off = draft.rank_picks_v2(state.legal_actions, memory, SUPPORT)
        boards = {key: (board, list(board.resid)) for key, board in draft._BOARDS.items()}
        on = draft.rank_picks_v2(state.legal_actions, memory, SUPPORT_ON)
        assert {key: (board, list(board.resid)) for key, board in draft._BOARDS.items()} == boards  # same board, same fitted terms
        assert {s.species: s.exposure for s in on} == {s.species: s.exposure for s in off}
        policy = lambda picks: {s.species: [n for n in s.notes if n.startswith("first-pick policy")] for s in picks}  # noqa: E731
        assert policy(on) == policy(off)
        for pick in on:
            assert sum(n.startswith("card table") for n in pick.notes) == 1


@needs_data
def test_team_preview_never_sees_the_key():
    ours = ["Kingambit", "Iron Hands", "Flutter Mane", "Torkoal", "Amoonguss", "Porygon2"]
    theirs = ["Whimsicott", "Urshifu-Rapid-Strike", "Thundurus", "Rillaboom", "Archaludon", "Chi-Yu"]
    mine = [lineup.profile(CATALOG[s]) for s in ours]
    them = [lineup.profile(CATALOG[s]) for s in theirs]
    for seed in (None, 3):
        off = lineup.shortlist_v2(mine, them, params=tuning.DEFAULTS, seed=seed)
        on = lineup.shortlist_v2(mine, them, params=ON, seed=seed)
        assert [(c.as_payload(), c.score, c.notes) for c in on] == [(c.as_payload(), c.score, c.notes) for c in off]


@needs_data
def test_archaludon_keeps_its_place_at_our_first_pick_on_the_live_drafts():
    """Where Archaludon tops our first pick's ranking on the live drafts, it still does with the key on (with and without
    the support key; four of the 14 such live decisions, both draft orders). The table holds no Archaludon entry. (One
    of the 14, 7551b7da seat 1 with the support key, keeps Archaludon on top by 0.001 and so shares the tie window
    with Gholdengo: the pinned simulator A/B's Archaludon rule is the check that counts.)"""
    for session, seat in (("0a2b7c2a", 0), ("51c86e5c", 0), ("07a3cb1f", 1), ("bcbaabe2", 0)):  # we first x2, they first x2
        name, state, memory = next(live_decisions(session, seat))  # our first pick
        for off_p, on_p in ((tuning.DEFAULTS, ON), (SUPPORT, SUPPORT_ON)):
            assert draft.rank_picks_v2(state.legal_actions, memory, off_p)[0].species == "Archaludon"
            assert draft.rank_picks_v2(state.legal_actions, memory, on_p)[0].species == "Archaludon", (session, seat)


def test_a_legal_action_list_without_known_cards_is_untouched_by_the_key():
    assert draft.rank_picks_v2([LegalAction("draft_pick:x", "Draft x", {})], MatchMemory(), ON) == []
