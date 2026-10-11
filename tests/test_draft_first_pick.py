"""B1, the first-pick policy, made calibratable (Oct 9): each of its terms has its own tuning weight, and a per-card
table (data/first_pick_adjust.json, shipped all zeros, weighted by draft_first_adjust_w) can shift single cards at our
first two picks. With the zero table the draft is stage2's exactly: the 120 captured live draft decisions replayed
through rank_picks_v2 give identical rankings, full-precision scores and notes before and after this change."""

from __future__ import annotations

import json

import pytest

from altruagent.models import LegalAction

from agent.pokemon import data, draft, tuning
from agent.pokemon.memory import MatchMemory, species_key

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")

# Oct 7 L3 (we drafted first; sim/draft_eval.py LIVE_POOLS 7ce71025): the pool the scarcity term was tuned on.
L3 = ["Amoonguss", "Archaludon", "Dondozo", "Garchomp", "Great Tusk", "Grimmsnarl", "Hatterene", "Incineroar", "Indeedee-F",
      "Iron Hands", "Rillaboom", "Sableye", "Sylveon", "Ting-Lu", "Tornadus", "Ursaluna", "Whimsicott", "Wo-Chien"]


SHIPPED_FILE = draft.FIRST_PICK_ADJUST_FILE


@pytest.fixture(autouse=True)
def _zero_table(monkeypatch, tmp_path):
    """Every test starts from the all-zeros table (the stage2 draft) and an empty cache (the table is read once per
    process). The shipped table's own entries (non-zero since Oct 10) are pinned by the first test below."""
    monkeypatch.delenv(draft.FIRST_PICK_ADJUST_ENV, raising=False)
    zero = tmp_path / "zero_first_pick_adjust.json"
    zero.write_text(json.dumps({"adjust": {}, "adjust_we_first": {}, "adjust_they_first": {}, "pairs": {}}))
    monkeypatch.setattr(draft, "FIRST_PICK_ADJUST_FILE", zero)
    monkeypatch.setattr(draft, "_ADJUST_TABLES", {})


def memory_for(pool: list[str], mine: list[str] = (), theirs: list[str] = ()) -> MatchMemory:
    memory = MatchMemory()
    memory.my_seat_key = "me"
    for species in pool:
        memory.pool_cards[CATALOG[species]["card_id"]] = CATALOG[species]
    for species in mine:
        memory.my_cards[species_key(species)] = CATALOG[species]
    for species in theirs:
        memory.opp_cards[species_key(species)] = CATALOG[species]
    return memory


def offered(memory: MatchMemory) -> list[LegalAction]:
    drafted = {c["card_id"] for c in (*memory.my_cards.values(), *memory.opp_cards.values())}
    held = {data.to_id(c["item"]) for c in memory.my_cards.values()}
    return [LegalAction(f"draft_pick:{c['card_id']}", f"Draft {c['species']}", {}) for c in memory.pool_cards.values()
            if c["card_id"] not in drafted and data.to_id(c["item"]) not in held]


def ranked(mine: list[str] = (), theirs: list[str] = (), params: dict | None = None) -> dict[str, draft.ScoredCard]:
    memory = memory_for(L3, mine, theirs)
    return {s.species: s for s in draft.rank_picks_v2(offered(memory), memory, params or tuning.DEFAULTS)}


def l3_board():
    P = tuning.DEFAULTS
    board = draft._board_for([CATALOG[s] for s in L3], float(P["draft_card_w"]), float(P["draft_tier_adjust"]))
    index = {u.species: i for i, u in enumerate(board.units)}
    return board, list(range(len(board.units))), index


def write_table(tmp_path, entries: dict | None = None, **sections) -> str:
    path = tmp_path / "first_pick_adjust.json"
    path.write_text(json.dumps({"adjust": entries or {}, **sections}))
    return str(path)


def test_the_shipped_table_names_only_catalog_cards_and_holds_the_archaludon_entries(monkeypatch):
    raw = json.loads(SHIPPED_FILE.read_text())
    assert raw["adjust"] and set(raw["adjust"]) <= set(CATALOG)  # a misspelt card would silently adjust nothing
    assert all(v == 0.0 for v in raw["adjust"].values()) and raw["pairs"] == {}
    # Oct 10, from paired simulator A/Bs: Archaludon when we draft first (+1.5) and when they draft first (+0.7)
    assert raw["adjust_we_first"] == {"Archaludon": 1.5} and raw["adjust_they_first"] == {"Archaludon": 0.7}
    monkeypatch.setattr(draft, "FIRST_PICK_ADJUST_FILE", SHIPPED_FILE)
    table = draft.first_pick_adjust()
    assert table.entries() == 2 and set(table.adjust.values()) == {0.0}
    assert table.value("Archaludon", we_first=True) == 1.5 and table.value("Archaludon", we_first=False) == 0.7
    assert tuning.DEFAULTS["draft_first_adjust_w"] == 1.0  # a filled-in table counts as written


def test_with_the_zero_table_the_bonus_is_the_four_term_sum_bit_for_bit():
    board, pool, index = l3_board()
    P = tuning.DEFAULTS
    for species in ("Incineroar", "Archaludon", "Iron Hands", "Tornadus"):
        bonus, t = draft.first_pick_bonus(board, index[species], pool, P)
        assert t["adjust"] == 0.0
        assert bonus == (P["draft_first_strength_w"] * t["strength"] + P["draft_first_scarcity_w"] * t["scarcity"]
                         - P["draft_first_exposure_w"] * t["exposure"] / 6.0 + P["draft_first_guard_w"] * t["guard"])
    no_card_table = tuning.merged({"draft_card_adjust_w": 0.0, "draft_card_adjust_them_w": 0.0})
    assert not any("table" in n for s in ranked(params=no_card_table).values() for n in s.notes)  # the policy note reads as before
    # the defaults add the draft's per-card table (deploy #5): its one line per card is the only other "table" note
    assert all([n for n in s.notes if "table" in n] == [n for n in s.notes if n.startswith("card table")] and
               sum(n.startswith("card table") for n in s.notes) == 1 for s in ranked().values())


@pytest.mark.parametrize("key, term, scale", [("draft_first_strength_w", "strength", 1.0), ("draft_first_scarcity_w", "scarcity", 1.0),
                                              ("draft_first_exposure_w", "exposure", -1 / 6), ("draft_first_guard_w", "guard", 1.0)])
def test_each_policy_term_has_its_own_weight(key, term, scale):
    board, pool, index = l3_board()
    inc = index["Incineroar"]
    base, t = draft.first_pick_bonus(board, inc, pool, tuning.DEFAULTS)
    bumped, _ = draft.first_pick_bonus(board, inc, pool, tuning.merged({key: tuning.DEFAULTS[key] + 1.0}))
    assert bumped - base == pytest.approx(scale * t[term])
    off, _ = draft.first_pick_bonus(board, inc, pool, tuning.merged({key: 0.0}))
    assert base - off == pytest.approx(tuning.DEFAULTS[key] * scale * t[term])


def test_role_scarcity_zero_takes_its_share_off_incineroar_at_pick_one():
    # Arm (a) of the first-pick A/B: Incineroar is L3's only Intimidate and one of four Fake Out users (scarcity 1.25).
    plain, none = ranked(), ranked(params=tuning.merged({"draft_first_scarcity_w": 0.0}))
    assert plain["Incineroar"].score - none["Incineroar"].score == pytest.approx(0.5 * 1.25, abs=2e-3)


def test_a_table_entry_moves_only_its_card_and_fades_like_the_policy(tmp_path, monkeypatch):
    before = [ranked(), ranked(["Archaludon"], ["Iron Hands", "Sylveon"]), ranked(["Archaludon", "Rillaboom"], ["Iron Hands", "Sylveon"])]
    monkeypatch.setenv(draft.FIRST_PICK_ADJUST_ENV, write_table(tmp_path, {"incineroar": -0.4}))  # keys are matched as species ids
    after = [ranked(), ranked(["Archaludon"], ["Iron Hands", "Sylveon"]), ranked(["Archaludon", "Rillaboom"], ["Iron Hands", "Sylveon"])]
    for weight, b, a in zip((1.0, 0.5, 0.0), before, after):  # full at our first pick, half at our second, none after
        assert a["Incineroar"].score - b["Incineroar"].score == pytest.approx(-0.4 * weight, abs=2e-3)
        assert all(a[s].score == b[s].score for s in b if s != "Incineroar")
    assert any(n.startswith("first-pick policy") and n.endswith("table -0.40)") for n in after[0]["Incineroar"].notes)
    monkeypatch.setattr(draft, "_ADJUST_TABLES", {})
    half = ranked(params=tuning.merged({"draft_first_adjust_w": 0.5}))
    assert half["Incineroar"].score - before[0]["Incineroar"].score == pytest.approx(-0.2, abs=2e-3)


def test_a_mistyped_table_path_fails_loudly_and_a_missing_shipped_table_is_zeros(tmp_path, monkeypatch):
    # A simulator A/B arm names its table through the environment: a typo must not quietly turn it into the control.
    monkeypatch.setenv(draft.FIRST_PICK_ADJUST_ENV, str(tmp_path / "no-such-table.json"))
    with pytest.raises(ValueError, match="unusable"):
        draft.first_pick_adjust()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"adjust": {"Incineroar": "minus a bit"}}))
    with pytest.raises(ValueError):
        draft.first_pick_adjust(str(bad))
    # The live agent never sets the variable: a shipped table that is gone or broken is the zero table, never a crash.
    monkeypatch.delenv(draft.FIRST_PICK_ADJUST_ENV)
    monkeypatch.setattr(draft, "FIRST_PICK_ADJUST_FILE", tmp_path / "missing.json")
    assert draft.first_pick_adjust() == draft.FirstPickTable()
    assert ranked()["Incineroar"].score == pytest.approx(ranked(params=tuning.merged({"draft_first_adjust_w": 0.0}))["Incineroar"].score)


def test_draft_order_sections_apply_only_in_that_order(tmp_path, monkeypatch):
    # The Oct 9 analysis: a Gyarados / Porygon2 / Dragonite first pick lost when they drafted first (our pick 2), not
    # when we did, so the table can say "only when they draft first".
    path = write_table(tmp_path, adjust_they_first={"Incineroar": -0.4}, adjust_we_first={"Archaludon": 0.3})
    table = draft.first_pick_adjust(path)
    assert table.value("Incineroar", we_first=False) == -0.4 and table.value("Incineroar", we_first=True) == 0.0
    assert table.value("Archaludon", we_first=True) == 0.3 and table.value("Archaludon", we_first=False) == 0.0
    assert table.value("Incineroar") == 0.0  # order unknown: the order sections stay out
    us_first = ranked()  # L3 at pick 1: we draft first
    them_first = ranked(theirs=["Iron Hands"])  # pick 2: they drafted first
    monkeypatch.setenv(draft.FIRST_PICK_ADJUST_ENV, path)
    assert ranked()["Incineroar"].score == us_first["Incineroar"].score
    assert ranked()["Archaludon"].score == pytest.approx(us_first["Archaludon"].score + 0.3, abs=2e-3)
    assert ranked(theirs=["Iron Hands"])["Incineroar"].score == pytest.approx(them_first["Incineroar"].score - 0.4, abs=2e-3)
    assert ranked(theirs=["Iron Hands"])["Archaludon"].score == them_first["Archaludon"].score


def test_a_pair_entry_charges_the_second_card_of_the_pair_at_our_second_pick(tmp_path, monkeypatch):
    # Two of Gyarados / Porygon2 / Dragonite in our first two picks won about 1 in 5 against the search bot's code in
    # every series since B1, while each alone did not lose: a pair entry prices the second one, only next to the first.
    path = write_table(tmp_path, pairs={"Incineroar + Archaludon": -0.6})
    table = draft.first_pick_adjust(path)
    assert table.value("Incineroar", team=["Archaludon"]) == table.value("Archaludon", team=("Incineroar",)) == -0.6
    assert table.value("Incineroar") == table.value("Incineroar", team=["Sylveon"]) == 0.0
    first, second = ranked(), ranked(["Archaludon"], ["Iron Hands", "Sylveon"])
    other = ranked(["Rillaboom"], ["Iron Hands", "Sylveon"])
    monkeypatch.setenv(draft.FIRST_PICK_ADJUST_ENV, path)
    assert ranked()["Incineroar"].score == first["Incineroar"].score  # no partner yet at our first pick
    assert ranked(["Archaludon"], ["Iron Hands", "Sylveon"])["Incineroar"].score == pytest.approx(second["Incineroar"].score - 0.3, abs=2e-3)
    assert ranked(["Rillaboom"], ["Iron Hands", "Sylveon"])["Incineroar"].score == other["Incineroar"].score
    monkeypatch.setattr(draft, "_ADJUST_TABLES", {})  # the table is read once per path and process
    with pytest.raises(ValueError, match="two different species"):
        draft.first_pick_adjust(write_table(tmp_path, pairs={"Incineroar": -1.0}))


def test_our_first_two_picks_name_the_table_and_warn_when_the_shipped_file_is_unreadable(tmp_path, monkeypatch):
    """A missing or broken shipped table still drafts (as all zeros), but no longer silently: at our first two picks,
    where the table acts, the pick's notes (the decision log's draft row) name the table and its entry count."""
    monkeypatch.setattr(draft, "_ADJUST_ERRORS", {})

    def table_notes(mine: list[str] = (), theirs: list[str] = ()) -> list[str]:
        memory = memory_for(L3, mine, theirs)
        return [n for n in draft.choose_pick(offered(memory), memory, tuning.DEFAULTS).notes if n.startswith("first-pick table")]

    monkeypatch.setattr(draft, "FIRST_PICK_ADJUST_FILE", SHIPPED_FILE)
    assert table_notes() == ["first-pick table: 2 entries (shipped first_pick_adjust.json, weight 1.0)"]
    # our second pick: the table acts at half weight there (was pinned as "weight 1.0", what the line said before Oct 10)
    assert table_notes(["Archaludon"], ["Iron Hands", "Sylveon"]) == ["first-pick table: 2 entries (shipped first_pick_adjust.json, weight 0.5)"]
    assert table_notes(["Archaludon", "Rillaboom"], ["Iron Hands", "Sylveon"]) == []  # the table acts no more
    for name, text in (("gone.json", None), ("broken.json", "{not json")):
        path = tmp_path / name
        if text is not None:
            path.write_text(text)
        monkeypatch.setattr(draft, "FIRST_PICK_ADJUST_FILE", path)
        monkeypatch.setattr(draft, "_ADJUST_TABLES", {})
        [line] = table_notes()
        assert line.startswith(f"first-pick table: 0 entries (shipped {name}, weight 1.0); WARNING: ") and "could not be read" in line, line


def test_the_table_line_follows_the_policys_own_count_of_our_picks(monkeypatch):
    """The line goes where the table acts, at the weight it acts with: the policy counts our picks as the drafted cards
    with a known set, so a roster entry without one (a server auto-pick) leaves it at half weight, line included."""
    monkeypatch.setattr(draft, "FIRST_PICK_ADJUST_FILE", SHIPPED_FILE)
    memory = memory_for(L3, ["Archaludon"], ["Iron Hands", "Sylveon"])
    actions = offered(memory)
    memory.my_cards["kingambit"] = {"species": "Kingambit", "card_id": "auto-pick-without-a-set"}
    notes = draft.choose_pick(actions, memory, tuning.DEFAULTS).notes
    assert [n for n in notes if n.startswith("first-pick table")] == ["first-pick table: 2 entries (shipped first_pick_adjust.json, weight 0.5)"]
    assert draft.first_pick_policy_w(memory, tuning.DEFAULTS) == 0.5
    assert draft.first_pick_policy_w(memory_for(L3, ["Archaludon", "Rillaboom"]), tuning.DEFAULTS) == 0.0


def test_the_decision_log_draft_row_carries_the_table_line(tmp_path, monkeypatch):
    from altruagent import DecisionContext
    from altruagent.models import GameState

    from agent.agent import PokemonAgent

    monkeypatch.setattr(draft, "FIRST_PICK_ADJUST_FILE", SHIPPED_FILE)
    live = data.DATA_DIR.parent / "tests" / "fixtures" / "live"
    path = next(live.glob("d74bfb88*/draft-00000-seat1.json"))  # our first pick: we drafted first
    raw = json.loads(path.read_text())
    agent = PokemonAgent(None, version="table-line", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    agent.choose_action(GameState.from_mcp_state(raw), DecisionContext(
        session_id=path.parent.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
        agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=1))
    rows = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    assert "first-pick table: 2 entries (shipped first_pick_adjust.json, weight 1.0)" in next(r for r in rows if r.get("kind") == "draft")["notes"]
