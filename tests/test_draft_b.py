"""Workstream B (draft) regression tests on the live catalog: the fitted-model fix (B2) measured on the four Oct 7
pools by draft-only replay (no battles)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from altruagent.models import LegalAction

from agent.pokemon import data, draft, matchup, tuning
from agent.pokemon.memory import MatchMemory, species_key

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")

# The four Oct 7 pools (sim/draft_eval.py LIVE_POOLS), with who drafted first and their first pick.
POOLS = {
    "L1": ("them", "Dragonite", ["Amoonguss", "Archaludon", "Baxcalibur", "Cresselia", "Garchomp", "Gholdengo", "Indeedee-F", "Iron Bundle",
                                 "Iron Hands", "Klefki", "Landorus-Therian", "Lucario", "Rillaboom", "Sneasler", "Thundurus", "Ting-Lu",
                                 "Wo-Chien", "Dragonite"]),
    "L2": ("them", "Incineroar", ["Archaludon", "Chien-Pao", "Cresselia", "Dondozo", "Flutter Mane", "Great Tusk", "Gyarados", "Hatterene",
                                  "Indeedee-F", "Iron Bundle", "Iron Hands", "Kingambit", "Klefki", "Rillaboom", "Sylveon", "Whimsicott",
                                  "Wo-Chien", "Incineroar"]),
    "L3": ("us", None, ["Amoonguss", "Archaludon", "Dondozo", "Garchomp", "Great Tusk", "Grimmsnarl", "Hatterene", "Incineroar",
                        "Indeedee-F", "Iron Hands", "Rillaboom", "Sableye", "Sylveon", "Ting-Lu", "Tornadus", "Ursaluna",
                        "Whimsicott", "Wo-Chien"]),
    "L4": ("us", None, ["Amoonguss", "Archaludon", "Baxcalibur", "Dragonite", "Gholdengo", "Great Tusk", "Gyarados", "Hatterene",
                        "Incineroar", "Indeedee-F", "Landorus-Therian", "Lucario", "Sableye", "Thundurus", "Ting-Lu", "Tornadus",
                        "Tyranitar", "Wo-Chien"]),
}


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


def ranking(pool: list[str], mine: list[str] = (), theirs: list[str] = (), params: dict | None = None) -> list[str]:
    """Draft-only replay of one position: every offered card, best first, by the default drafter."""
    memory = memory_for(pool, mine, theirs)
    return [s.species for s in draft.rank_picks_v2(offered(memory), memory, params or tuning.DEFAULTS)]


# -- B2: the fitted model's doubles-role weights ----------------------------------------------------------------------


@pytest.mark.parametrize("section", ["draft", "preview"])
def test_no_doubles_role_weight_is_negative(section):
    weights = matchup.load_model(section=section)["weights"]
    for key in matchup.PRIOR_CLAMPED:
        assert weights.get(key, 0.0) >= 0.0, (section, key, weights.get(key))
    # A Fake Out, Tailwind, Trick Room or Intimidate user is credited, never charged, for its role.
    for key in ("role_fakeout", "role_tailwind", "role_trickroom", "role_intimidate"):
        assert weights[key] > 0.0, (section, key)


def test_a_negative_fitted_role_weight_becomes_the_vgc_prior():
    fitted = {**matchup.DEFAULT_MODEL, "weights": {**matchup.DEFAULT_MODEL["weights"], "role_fakeout": -0.32, "tw_gain": -0.08,
                                                     "role_intimidate": 0.05, "mean": -1.0}}
    model = matchup.clamp_roles(fitted)
    assert model["weights"]["role_fakeout"] == matchup.DEFAULT_MODEL["weights"]["role_fakeout"]
    assert model["weights"]["tw_gain"] == matchup.DEFAULT_MODEL["weights"]["tw_gain"]
    assert model["weights"]["role_intimidate"] == 0.05  # a positive fitted value stands
    assert model["weights"]["mean"] == -1.0  # only role and speed-control weights are clamped
    assert set(model["clamped"]) == {"role_fakeout", "tw_gain"}
    assert matchup.clamp_roles(matchup.DEFAULT_MODEL) is matchup.DEFAULT_MODEL  # nothing to clamp: untouched


def test_the_role_credit_reaches_the_team_score():
    units = [matchup.unit(CATALOG[s]) for s in ("Incineroar", "Garchomp", "Gholdengo", "Archaludon")]
    board = matchup.Board(units, card_weight=0.0, tier_adjust=0.0)
    with_fake_out = board.score([0, 1], [2, 3])  # Incineroar (Fake Out, Intimidate) + Garchomp
    without = board.score([1], [2, 3])
    role = {f: w for f, w in zip(matchup.FEATURES, board.w)}
    assert role["role_fakeout"] > 0 and role["role_intimidate"] > 0
    feats = dict(zip(matchup.FEATURES, board.features([0, 1], [2, 3])))
    assert feats["role_fakeout"] > 0 and feats["role_intimidate"] > 0
    assert with_fake_out == pytest.approx(sum(w * f for w, f in zip(board.w, board.features([0, 1], [2, 3]))))
    assert isinstance(without, float)


def test_card_weight_zero_keeps_the_written_down_tier_adjustments():
    incineroar = CATALOG["Incineroar"]
    entry = matchup.load_tiers()[matchup.fingerprint(incineroar)]
    assert entry["adjust"] == 25
    units = [matchup.unit(incineroar), matchup.unit(CATALOG["Garchomp"])]
    board = matchup.Board(units, card_weight=0.0, tier_adjust=1.25)
    assert board.resid[0] == pytest.approx(1.25 * 25 / 100)
    assert matchup.Board(units, card_weight=0.0, tier_adjust=0.0).resid == [0.0, 0.0]
    assert tuning.DEFAULTS["draft_card_w"] == 0.0 and tuning.DEFAULTS["draft_tier_adjust"] == 1.25


# -- B2: measured on the four Oct 7 pools (draft-only replay) ---------------------------------------------------------


def test_incineroar_is_a_top_five_first_pick_on_every_live_pool_that_offered_it():
    # Live, the Oct 7 opponent took Incineroar first whenever it could; our drafter had it 8th-18th (card_w 1, tier 0).
    assert "Incineroar" not in POOLS["L1"][2]  # pool L1 never offered it (they opened with Dragonite)
    for name in ("L2", "L3", "L4"):
        order = ranking(POOLS[name][2])
        assert "Incineroar" in order[:5], (name, order[:8])


def test_the_pick_two_answers_to_their_first_pick_hold():
    first, theirs, pool = POOLS["L1"]
    assert ranking(pool, theirs=[theirs])[0] == "Gholdengo"  # L1: Gholdengo at pick 2, not Baxcalibur
    first, theirs, pool = POOLS["L2"]
    assert ranking(pool, theirs=[theirs])[0] != "Iron Bundle"  # L2: not the Iron Bundle we took live


def test_the_ranking_is_the_pick_order_and_stays_fast():
    memory = memory_for(POOLS["L3"][2])
    started = time.perf_counter()
    ranked = draft.rank_picks_v2(offered(memory), memory, tuning.DEFAULTS)
    assert time.perf_counter() - started < 2.0
    assert len(ranked) == 18 and len({s.card_id for s in ranked}) == 18
    assert [s.score for s in ranked] == sorted((s.score for s in ranked), reverse=True)
    assert draft.choose_pick(offered(memory), memory, tuning.DEFAULTS).card_id == ranked[0].card_id


# -- B3: the opponent's replies in the lookahead come from two readings of the opponent -----------------------------

LIVE_DIR = Path(__file__).parent / "fixtures" / "live"


def live_first_position(prefix: str) -> tuple[list[str], list[str], list[str], str]:
    """From the captured draft states of one live session: the 18-card pool, our and their picks before their first
    reply to our first pick(s), and the card they actually took then. (pool, ours, theirs, their actual next pick)"""
    folder = next((p for p in LIVE_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)), None)
    if folder is None:
        pytest.skip(f"live capture {prefix} not present")
    drafts = sorted(folder.glob("draft-*-seat0.json"), key=lambda p: int(re.search(r"-(\d+)-", p.name).group(1)))
    first, last = json.loads(drafts[0].read_text()), json.loads(drafts[-1].read_text())
    me = last["current_actor"]["agent_id"]
    picks = last["observation"]["picks"]
    pool = sorted({c["species"] for c in first["observation"]["available_cards"]} | {p["species"] for p in picks})
    before = 1 if first["observation"]["first_drafter"] == me else 3  # our one pick, or their one and our two
    assert picks[before]["seat_id"] != me
    ours = [p["species"] for p in picks[:before] if p["seat_id"] == me]
    theirs = [p["species"] for p in picks[:before] if p["seat_id"] != me]
    return pool, ours, theirs, picks[before]["species"]


def search_for(pool: list[str], ours: list[str], theirs: list[str]):
    P = tuning.DEFAULTS
    board = draft._board_for([CATALOG[s] for s in pool], float(P["draft_card_w"]), float(P["draft_tier_adjust"]))
    index = {u.key: i for i, u in enumerate(board.units)}
    us = [index[matchup.fingerprint(CATALOG[s])] for s in ours]
    them = [index[matchup.fingerprint(CATALOG[s])] for s in theirs]
    rest = [i for i in range(len(board.units)) if i not in us and i not in them]
    search = draft.DraftSearch(board, lookahead=int(P["draft_lookahead"]), counter_replies=int(P["draft_counter_replies"]))
    return board, search, us, them, rest


@pytest.mark.parametrize("prefix", ["7ce71025", "0a2b7c2a", "d9405f02"])
def test_the_reply_set_holds_their_actual_next_pick_in_the_live_first_positions(prefix):
    # The Oct 7 opponent's actual replies: Incineroar to our Tornadus (L3) and to our Landorus-Therian (L4), Gholdengo
    # to our Baxcalibur + Landorus-Therian after its Dragonite (L1). The team model alone held only the last one.
    pool, ours, theirs, actual = live_first_position(prefix)
    board, search, us, them, rest = search_for(pool, ours, theirs)
    replies = {board.units[c].species: source for c, source in search.reply_set(us, them, rest)}
    assert actual in replies, (prefix, replies)
    by_model = {board.units[c].species for _, c in search.greedy("them", us, them, rest)[:3]}
    by_counter = {board.units[c].species for _, c in search.counter(us, them, rest)[:3]}
    assert set(replies) == by_model | by_counter
    assert replies[actual] == (draft.GREEDY if actual in by_model else draft.COUNTER)


def test_the_one_live_reply_neither_reading_holds_is_known():
    # L2: after its Incineroar and our Iron Bundle + Iron Hands the Oct 7 opponent took Flutter Mane ("spread damage,
    # beats Iron Bundle"); both scorers rank it 6th-7th there, so a 3 + 3 reply set misses it. Pinned so a change shows.
    pool, ours, theirs, actual = live_first_position("1aef9227")
    assert actual == "Flutter Mane"
    board, search, us, them, rest = search_for(pool, ours, theirs)
    replies = {board.units[c].species for c, _ in search.reply_set(us, them, rest)}
    assert len(replies) >= 4 and "Sylveon" in replies and "Cresselia" in replies
    assert "Flutter Mane" not in replies


def test_counter_replies_follow_the_counter_pickers_scorer_and_roll_out_with_it():
    from agent.pokemon import counterpick

    pool, ours, theirs, _ = live_first_position("7ce71025")
    board, search, us, them, rest = search_for(pool, ours, theirs)
    ranked = search.counter(us, them, rest)
    expected = sorted((counterpick.counter_score(board.units[c].card, [board.units[i].card for i in us])[0], -rest.index(c))
                      for c in search.legal(them, rest))[::-1]
    assert [round(v, 3) for v, _ in ranked] == [round(v, 3) for v, _ in expected]
    assert board.units[ranked[0][1]].species == "Incineroar"
    order = draft.pick_order(1, True)  # picks 2-12: theirs at 2-3, 6-7, 10-11
    _, _, final_them = search.rollout(us, them, rest, order, opponent=draft.COUNTER)
    _, _, final_model = search.rollout(us, them, rest, order, opponent=draft.GREEDY)
    assert len(final_them) == 6 and board.units[final_them[0]].species == "Incineroar"
    assert final_them != final_model  # the two readings of the opponent draft differently
    no_counter = draft.DraftSearch(board, counter_replies=0)
    assert all(source == draft.GREEDY for _, source in no_counter.reply_set(us, them, rest))
