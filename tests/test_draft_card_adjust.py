"""Per-card corrections in the draft's finished-draft value (``draft_card_adjust_w``; draft lane 8, Oct 10): the search's
own value of a finished draft is overconfident, and on 13,841 of our simulator games some cards won more, or less, than
it credits. data/draft_card_adjust.json holds those corrections in log-odds: "us" when the card is in the four we would
bring, "them" for each of their considered fours that holds it. With the key above 0, DraftSearch.finished adds key x
table to every four-against-four value before the mean / worst blend; their fours are still chosen without it. At 0
the table is never read and the search runs its old code: the 132 live draft and 22 Team Preview decisions replay
identically to 39f50af (full-precision values, notes, picks). Deploy #5 ships both keys at 0.5 (the pinned simulator
A/Bs), so the tests below set them both ways explicitly."""

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
BOTH = {"draft_card_adjust_w": W, "draft_card_adjust_them_w": W}  # the whole table: "us" and "them" at the gate's weight
NO_TABLE = {"draft_card_adjust_w": 0.0, "draft_card_adjust_them_w": 0.0}  # the table off (deploy #5 ships both at 0.5)
PLAIN = tuning.merged({"draft_support_w": 0.0, **NO_TABLE})  # the draft without the support key (39f50af's defaults; deploy #4 ships 0.25)
ON = dict(PLAIN, **BOTH)
SUPPORT = tuning.merged({"draft_support_w": 0.25, **NO_TABLE})  # the deploy #4 draft (09a95b0)
SUPPORT_ON = dict(SUPPORT, **BOTH)  # the deploy #5 draft: the defaults
US_ONLY = dict(SUPPORT, draft_card_adjust_w=W)  # "us" alone (draft_card_adjust_them_w 0)
THEM_ONLY = dict(SUPPORT, draft_card_adjust_them_w=W)  # "them" alone (draft_card_adjust_w 0)
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


def test_the_key_ships_on_at_the_gates_weight():
    """Deploy #5 ships the whole table at the gate's weight (the pinned simulator A/Bs, with draft_support_w 0.25): the
    defaults are SUPPORT_ON, the A/B's arm; the off cases here set both keys to 0 explicitly."""
    assert tuning.DEFAULTS["draft_card_adjust_w"] == W == 0.5
    assert ON["draft_card_adjust_w"] == W
    # conftest pins the preview budget (which the draft never reads) after these module constants were built
    assert tuning.merged() == dict(SUPPORT_ON, lineup_sim_budget_s=tuning.DEFAULTS["lineup_sim_budget_s"])
    assert PLAIN["draft_card_adjust_w"] == SUPPORT["draft_card_adjust_w"] == 0.0


@needs_data
def test_the_defaults_rank_every_live_pick_as_the_ab_arm():
    """The shipped defaults are the A/B's arm (support 0.25, both halves 0.5): on a live draft whose first pick offers
    Archaludon (the floor runs there), every card's score, exposure and notes are SUPPORT_ON's, card table line included."""
    for _, state, memory in live_decisions("7551b7da", 1):
        shipped = draft.rank_picks_v2(state.legal_actions, memory)
        arm = draft.rank_picks_v2(state.legal_actions, memory, SUPPORT_ON)
        assert shipped and [(s.species, s.score, s.exposure, s.notes) for s in shipped] == \
               [(s.species, s.score, s.exposure, s.notes) for s in arm]
        assert all(sum(n.startswith("card table") for n in s.notes) == 1 for s in shipped)


def test_the_them_key_ships_on_and_is_a_tuning_key():
    """draft_card_adjust_them_w scales the "them" section on its own (draft lane 8 review: those entries say how
    well the search bot uses each card, so another opponent may need a different weight); on at 0.5 by default, with
    draft_card_adjust_w (deploy #5)."""
    assert tuning.DEFAULTS["draft_card_adjust_them_w"] == W == 0.5
    assert tuning.merged(BOTH)["draft_card_adjust_them_w"] == W
    assert PLAIN["draft_card_adjust_them_w"] == SUPPORT["draft_card_adjust_them_w"] == 0.0


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
    assert (len(table.us), len(table.them), table.entries()) == (13, 19, 32)
    # pinned from the gate's frozen table (skeptic work/table.json, sha256 01e6cf6c...): the largest entries each side
    assert table.us["irontreads"] == 0.5 and table.us["torkoal"] == -0.5 and table.us["kingambit"] == 0.436
    assert table.them["urshifurapidstrike"] == -0.5 and table.them["whimsicott"] == 0.464


@needs_data
def test_the_shipped_table_is_the_gate_table_less_five_noise_entries_and_matches_its_frozen_hash():
    """The review fix-up zeroed five of the gate's 37 entries before the simulator A/B, so the tested table is the
    shipped one: Farigiraf us, Gyarados us and Gholdengo them (their 90 % interval holds 0 once the bootstrap clusters
    by seed, the gate's own drop rule), Sylveon us (fails at 95 %, and the natural experiment puts it at -0.30) and
    Gyarados them (a selection artifact: we draft Gyarados 97 % of the times it is drafted). Every other entry is the
    gate's value. _table_sha256 freezes the entries: sha256 of json.dumps({"them": ..., "us": ...}, sort_keys=True)."""
    import hashlib

    raw = json.loads(draft.CARD_ADJUST_FILE.read_text())
    dropped = {"us": {"Farigiraf", "Gyarados", "Sylveon"}, "them": {"Gholdengo", "Gyarados"}}
    for side, names in dropped.items():
        assert not names & set(raw[side]), side
        assert names <= set(raw["_dropped"][side])
    entries = json.dumps({"them": raw["them"], "us": raw["us"]}, sort_keys=True)
    assert hashlib.sha256(entries.encode()).hexdigest() == raw["_table_sha256"]
    assert raw["_source_sha256"].startswith("01e6cf6cfd7fad54")  # the gate's frozen table it came from


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
    for params in (PLAIN, SUPPORT):
        for _, state, memory in live_decisions("07a3cb1f", 1):
            for pick in draft.rank_picks_v2(state.legal_actions, memory, params):
                assert not any(n.startswith("card table") for n in pick.notes)
    assert seen and all(a is None for a in seen)


@needs_data
@pytest.mark.parametrize("off, on", [(PLAIN, ON), (SUPPORT, SUPPORT_ON)])
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
        offered = {(memory.pool_cards.get(draft.card_id_of(x)) or {}).get("species") for x in state.legal_actions}
        assert vb == (va + va if "Archaludon" in offered else va)  # bit for bit (the Archaludon floor ranks twice)
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
    off = draft.rank_picks_v2(state.legal_actions, memory, PLAIN)
    assert [s.species for s in off[:2]] == ["Iron Hands", "Kingambit"] and 0 < off[0].score - off[1].score < 0.02
    on = {s.species: s for s in draft.rank_picks_v2(state.legal_actions, memory, ON)}
    best = max(on.values(), key=lambda s: s.score)
    assert best.species == "Kingambit" and on["Kingambit"].score - on["Iron Hands"].score > 0.2
    assert "card table (weight 0.5, 32 entries): Kingambit +0.44 in our four" in best.notes


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
        off = lineup.shortlist_v2(mine, them, params=PLAIN, seed=seed)
        on = lineup.shortlist_v2(mine, them, params=ON, seed=seed)
        assert [(c.as_payload(), c.score, c.notes) for c in on] == [(c.as_payload(), c.score, c.notes) for c in off]


@needs_data
def test_archaludon_keeps_its_place_at_our_first_pick_on_the_live_drafts():
    """Where Archaludon tops our first pick's ranking on the live drafts, it still does with the key on (with and without
    the support key; four of the 14 such live decisions, both draft orders). The table holds no Archaludon entry. The
    Archaludon floor tests below cover every live decision that offers it, at every pick."""
    for session, seat in (("0a2b7c2a", 0), ("51c86e5c", 0), ("07a3cb1f", 1), ("bcbaabe2", 0)):  # we first x2, they first x2
        name, state, memory = next(live_decisions(session, seat))  # our first pick
        for off_p, on_p in ((PLAIN, ON), (SUPPORT, SUPPORT_ON)):
            assert draft.rank_picks_v2(state.legal_actions, memory, off_p)[0].species == "Archaludon"
            assert draft.rank_picks_v2(state.legal_actions, memory, on_p)[0].species == "Archaludon", (session, seat)


def test_a_legal_action_list_without_known_cards_is_untouched_by_the_key():
    assert draft.rank_picks_v2([LegalAction("draft_pick:x", "Draft x", {})], MatchMemory(), ON) == []


# -- the Archaludon floor ---------------------------------------------------------------------------------------------------
#
# The table holds no Archaludon entry, yet the cards it raises (and the denial it adds) can still outbid Archaludon where
# the search bot never leaves it to us (they drafted first, or our second pick on) and so where the simulator A/B cannot
# see it, but live opponents can (draft lane 8 review: 34 of the 132 live decisions offer it, 27 of them at such picks).


def seed_chance(ranked, P, species="Archaludon"):
    """The chance that choose_pick_v2's secret seed takes ``species`` from ``ranked``: its share of the tie window
    (cards within draft_tiebreak_eps win chance of the best score; written out here, independent of draft.py)."""
    eps = float(P.get("draft_tiebreak_eps", 0.0))
    top = matchup.win_probability(ranked[0].score)
    window = [s.species for s in ranked if top - matchup.win_probability(s.score) <= eps] if eps > 0 else [ranked[0].species]
    return window.count(species) / len(window)


def archaludon_decisions():
    """Every live draft decision that offers Archaludon (seat by seat, in pick order, one memory per seat)."""
    for folder in sorted(p for p in LIVE.iterdir() if p.is_dir()):
        for seat in (0, 1):
            if not list(folder.glob(f"draft-*-seat{seat}.json")):
                continue
            for name, state, memory in live_decisions(folder.name, seat):
                offered = {(memory.pool_cards.get(draft.card_id_of(a)) or {}).get("species") for a in state.legal_actions}
                if "Archaludon" in offered:
                    yield f"{folder.name[:8]} s{seat} our pick {len(memory.my_cards) + 1}", state, memory


@needs_data
@pytest.mark.parametrize("off, on", [(PLAIN, ON), (SUPPORT, SUPPORT_ON)], ids=["support 0", "support 0.25"])
def test_the_table_never_lowers_archaludons_chance_on_the_live_drafts(off, on):
    """On every live decision that offers Archaludon, both draft orders and every pick: the chance the pick is
    Archaludon with the table on is at least its chance without it. A sole leader stays the sole leader (no
    table-made tie window), a window member stays in the window, and the table may still raise it (1aef9227 below).
    Before the floor: 07a3cb1f s1 and 48f23a42 s0 lost it outright (support 0), and 7551b7da s1 shared its window
    (support 0.25)."""
    seen, lowered = 0, []
    for where, state, memory in archaludon_decisions():
        a = draft.rank_picks_v2(state.legal_actions, memory, off)
        b = draft.rank_picks_v2(state.legal_actions, memory, on)
        pa, pb = seed_chance(a, off), seed_chance(b, on)
        if pb < pa:
            lowered.append((where, pa, pb, [(s.species, s.score) for s in b[:3]]))
        seen += 1
    assert seen >= 34  # the 20 live drafts' Archaludon decisions
    assert lowered == []


def gate_table(monkeypatch):
    """The gate's whole 37-entry table (the shipped entries plus the five the fix-up dropped), as the loader's answer:
    the table these floor cases were found with."""
    raw = json.loads(draft.CARD_ADJUST_FILE.read_text())
    whole = {side: {**raw[side], **raw["_dropped"][side]} for side in ("us", "them")}
    table = draft.CardAdjustTable(*({data.to_id(k): v for k, v in whole[side].items()} for side in ("us", "them")))
    assert table.entries() == 37
    monkeypatch.setattr(draft, "card_adjust_table", lambda path=None: table)


@needs_data
def test_where_the_floor_holds_the_pick_is_ranked_without_the_table(monkeypatch):
    """Live 7551b7da seat 1, our first pick after they drafted first, with the support key and the gate's whole table:
    without the table Archaludon leads alone; with it, Gholdengo comes within the tie window (the seed would pass
    Archaludon half the time). The floor ranks this decision without the table, bit for bit, and every card says why."""
    gate_table(monkeypatch)
    name, state, memory = next(live_decisions("7551b7da", 1))
    assert not memory.my_cards and memory.opp_cards
    off = draft.rank_picks_v2(state.legal_actions, memory, SUPPORT)
    assert off[0].species == "Archaludon" and seed_chance(off, SUPPORT) == 1.0
    on = draft.rank_picks_v2(state.legal_actions, memory, SUPPORT_ON)
    assert [(s.species, s.score, s.exposure) for s in on] == [(s.species, s.score, s.exposure) for s in off]
    for s_on, s_off in zip(on, off):
        lines = [n for n in s_on.notes if n.startswith("card table")]
        assert len(lines) == 1 and "Archaludon floor" in lines[0] and "100% to 50%" in lines[0], lines
        assert [n for n in s_on.notes if not n.startswith("card table")] == s_off.notes


@needs_data
def test_the_floor_keeps_what_the_table_gives_archaludon(monkeypatch):
    """Live 1aef9227 seat 0, our second pick (they drafted first), with the gate's whole table: without the table the
    search takes Sylveon and Archaludon is outside the window; with it Archaludon is the pick. The floor stops losses
    only. (The shipped table, without Sylveon's entry, puts Archaludon in a window with Sylveon: still a gain.)"""
    gate_table(monkeypatch)
    name, state, memory = next(d for d in live_decisions("1aef9227", 0) if len(d[2].my_cards) == 1)
    off = draft.rank_picks_v2(state.legal_actions, memory, PLAIN)
    assert off[0].species == "Sylveon" and seed_chance(off, PLAIN) == 0.0
    on = draft.rank_picks_v2(state.legal_actions, memory, ON)
    assert on[0].species == "Archaludon" and seed_chance(on, ON) == 1.0
    assert not any("Archaludon floor" in n for s in on for n in s.notes)


@needs_data
def test_the_floor_ranks_a_second_time_only_where_archaludon_is_offered(monkeypatch):
    calls = []
    original = draft.DraftSearch.__init__

    def init(self, board, **kwargs):
        calls.append(kwargs.get("card_adjust") is not None)
        original(self, board, **kwargs)

    monkeypatch.setattr(draft.DraftSearch, "__init__", init)
    kinds = set()
    for where, state, memory in live_decisions("7551b7da", 1):  # Archaludon is offered at our first pick only
        offered = "Archaludon" in {(memory.pool_cards.get(draft.card_id_of(a)) or {}).get("species") for a in state.legal_actions}
        calls.clear()
        draft.rank_picks_v2(state.legal_actions, memory, ON)
        assert calls == ([True, False] if offered else [True]), (where, calls)  # one search with the table, one without
        kinds.add(offered)
    assert kinds == {True, False}


# -- the two halves: draft_card_adjust_w ("us") and draft_card_adjust_them_w ("them") ---------------------------------------


@needs_data
def test_each_weight_scales_only_its_own_section():
    table = draft.CardAdjustTable(us={"kingambit": 0.4}, them={"whimsicott": 0.3})
    board, at = board_of(*US, *THEM)
    a_us, a_them = table.vectors(board, 0.5, 0.25)
    assert a_us[at["Kingambit"]] == 0.2 and a_them[at["Whimsicott"]] == 0.075
    assert a_us[at["Whimsicott"]] == 0.0 and a_them[at["Kingambit"]] == 0.0
    assert table.vectors(board, 0.5, 0.0)[1] == [0.0] * board.n and table.vectors(board, 0.0, 0.5)[0] == [0.0] * board.n
    assert table.vectors(board, 0.5) == table.vectors(board, 0.5, 0.5)  # one weight: both sections, as before


def spy_card_adjust(monkeypatch):
    seen = []
    original = draft.DraftSearch.__init__

    def init(self, board, **kwargs):
        seen.append((board, kwargs.get("card_adjust")))
        original(self, board, **kwargs)

    monkeypatch.setattr(draft.DraftSearch, "__init__", init)
    return seen


@needs_data
@pytest.mark.parametrize("params, w_us, w_them", [(US_ONLY, W, 0.0), (THEM_ONLY, 0.0, W), (SUPPORT_ON, W, W)],
                         ids=["us only", "them only", "both"])
def test_the_search_gets_each_section_at_its_own_weight(monkeypatch, params, w_us, w_them):
    """Live 6ce0b81e seat 0 (no Archaludon offered, so one search per decision): the vectors handed to the search are
    the table's sections at their own weights; equal weights give the single-weight vectors, float for float (the
    first build's arm B)."""
    table = draft.card_adjust_table()
    seen = spy_card_adjust(monkeypatch)
    for _, state, memory in live_decisions("6ce0b81e", 0):
        seen.clear()
        ranked = draft.rank_picks_v2(state.legal_actions, memory, params)
        (board, card_adjust), = seen
        ids = [data.to_id(u.species) for u in board.units]
        assert card_adjust == ([w_us * table.us.get(s, 0.0) for s in ids], [w_them * table.them.get(s, 0.0) for s in ids])
        if w_us == w_them:
            assert card_adjust == table.vectors(board, W)
        for pick in ranked:
            line, = [n for n in pick.notes if n.startswith("card table")]
            head = f"weight {W:g}" if w_us == w_them else f"weights us {w_us:g}, them {w_them:g}"
            assert line.startswith(f"card table ({head}, ")
            sid = data.to_id(pick.species)
            assert ("in our four" in line) == bool(w_us and table.us.get(sid))
            assert ("in theirs" in line) == bool(w_them and table.them.get(sid))


@needs_data
def test_the_halves_add_up_to_the_whole_table():
    """On a fixed six-against-six, the finished value moves by the "us" half plus the "them" half, each alone."""
    table = draft.card_adjust_table()
    board, at = board_of(*US, *THEM)
    us, them = [at[s] for s in US], [at[s] for s in THEM]

    def value(w_us, w_them):
        return draft.DraftSearch(board, hole_w=0.5, card_adjust=table.vectors(board, w_us, w_them)).finished(us, them)[0]

    base = draft.DraftSearch(board, hole_w=0.5).finished(us, them)[0]
    d_us, d_them, d_both = value(W, 0.0) - base, value(0.0, W) - base, value(W, W) - base
    assert d_us and d_them
    # the best four can differ between the three, so the halves add up only when they all bring the same four
    fours = {draft.DraftSearch(board, hole_w=0.5, card_adjust=table.vectors(board, *w)).finished(us, them)[1]
             for w in ((W, 0.0), (0.0, W), (W, W))}
    if len(fours) == 1:
        assert d_both == pytest.approx(d_us + d_them, abs=1e-9)


@needs_data
def test_the_floor_reads_both_halves(monkeypatch):
    """The Archaludon floor's second ranking turns both halves off, and the "them" half alone also reaches it."""
    seen = spy_card_adjust(monkeypatch)
    name, state, memory = next(live_decisions("7551b7da", 1))
    draft.rank_picks_v2(state.legal_actions, memory, THEM_ONLY)
    assert [c is not None for _, c in seen] == [True, False]
