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
    # A Fake Out, Tailwind or Intimidate user is credited, never charged, for its role.
    for key in ("role_fakeout", "role_tailwind", "role_intimidate"):
        assert weights[key] > 0.0, (section, key)


def test_the_drafter_uses_the_clamped_model_and_team_preview_the_fitted_one():
    from agent.pokemon import lineup

    board = draft._board_for([CATALOG[s] for s in POOLS["L3"][2]], tuning.DEFAULTS["draft_card_w"], tuning.DEFAULTS["draft_tier_adjust"])
    assert board.model is matchup.load_model(section="draft") and "clamped" in board.model
    fitted = matchup.load_model(section="preview", clamp=False)
    assert "clamped" not in fitted and fitted["weights"]["role_fakeout"] < 0  # the fit charged Fake Out; Team Preview keeps it
    model = lineup.PreviewModel([CATALOG[s] for s in POOLS["L3"][2][:6]], [CATALOG[s] for s in POOLS["L3"][2][6:12]])
    assert model.board.model is fitted


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


def test_card_weight_and_tier_adjust_are_independent_knobs():
    # Before Oct 8 card_weight also scaled the written-down tier adjustment, so card_w 0 switched both off.
    incineroar = CATALOG["Incineroar"]
    entry = matchup.load_tiers()[matchup.fingerprint(incineroar)]
    assert entry["adjust"] == 25
    units = [matchup.unit(incineroar), matchup.unit(CATALOG["Garchomp"])]
    board = matchup.Board(units, card_weight=0.0, tier_adjust=1.25)
    assert board.resid[0] == pytest.approx(1.25 * 25 / 100)
    assert matchup.Board(units, card_weight=0.0, tier_adjust=0.0).resid == [0.0, 0.0]
    fitted = float((matchup.load_model()["cards"].get(units[0].key) or {}).get("s", 0.0))
    assert matchup.Board(units, card_weight=1.0, tier_adjust=0.0).resid[0] == pytest.approx(fitted)
    # Shipped: the clamp only. Flipping these showed no gain over the eight live drafts (Oct 8 post-mortem of bcbaabe2).
    assert tuning.DEFAULTS["draft_card_w"] == 1.0 and tuning.DEFAULTS["draft_tier_adjust"] == 0.0


# -- B2: measured on the four Oct 7 pools (draft-only replay) ---------------------------------------------------------


# Live, the Oct 7 opponent took Incineroar first whenever it could; before the clamp our drafter had it 18th (L2),
# 11th (L3) and 8th (L4) at pick 1. The clamp alone lifts it to 8th / 7th / 6th (on the mech lane's Oct 8 calculator);
# the first-pick policy (B1: role scarcity, mostly) earns 5th / 4th on L2 / L3. The variant that lifts it to the top
# five everywhere (draft_card_w 0 + draft_tier_adjust 1.25, Oct 8 draft-only replay: 4th / 2nd / 1st) showed no gain
# over the eight live drafts and is not shipped (see test_card_weight_and_tier_adjust_are_independent_knobs), so L4
# stays pinned as the known tension: the xfail is strict and will flag the day a change lifts it.
@pytest.mark.parametrize("name", [
    "L2",
    "L3",
    pytest.param("L4", marks=pytest.mark.xfail(strict=True, reason="Incineroar is 6th at pick 1 on the L4 pool (Gholdengo first)")),
])
def test_incineroar_is_a_top_five_first_pick_on_the_live_pools_that_offered_it(name):
    assert "Incineroar" not in POOLS["L1"][2]  # pool L1 never offered it (they opened with Dragonite)
    order = ranking(POOLS[name][2])
    assert "Incineroar" in order[:5], (name, order[:8])


POOL_W1 = ["Arcanine", "Archaludon", "Basculegion", "Baxcalibur", "Cresselia", "Dragonite", "Garchomp", "Gyarados", "Iron Hands",
           "Milotic", "Rillaboom", "Salamence", "Scream Tail", "Thundurus", "Ting-Lu", "Whimsicott", "Wo-Chien", "Incineroar"]


def test_the_pick_that_won_todays_first_match_stays_first():
    # bcbaabe2 (Oct 8, won 4-1): after their Incineroar and our Iron Hands we took Archaludon (79% live), which won
    # the game. A draft package that assumed Archaludon would still be there at pick 6 moved this pick to Cresselia.
    order = ranking(POOL_W1, mine=["Iron Hands"], theirs=["Incineroar"])
    assert order[0] == "Archaludon", order[:5]


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
    chosen = draft.choose_pick(offered(memory), memory, tuning.DEFAULTS)  # the best, or a tie within the secret-seed window
    assert matchup.win_probability(ranked[0].score) - matchup.win_probability(chosen.score) <= tuning.DEFAULTS["draft_tiebreak_eps"]


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
    search = draft.DraftSearch(board, lookahead=int(P["draft_lookahead"]), counter_replies=int(P["draft_counter_replies"]),
                               counter_prior=float(P["draft_counter_prior"]))
    return board, search, us, them, rest


def test_the_tier_prior_keeps_the_counter_pickers_first_picks_role_picks():
    # The prior (our tier rating x draft_counter_prior) sharpens the replies without turning the counter-picker into a
    # strength drafter: on L2's and L1's pools its own first pick stays Incineroar / Dragonite (what the Oct 7 opponent did).
    for pool, expected in ((POOLS["L2"][2], "Incineroar"), (POOLS["L1"][2], "Dragonite")):
        board, search, us, them, rest = search_for(pool, [], [])
        assert board.units[search.counter(us, them, rest)[0][1]].species == expected
    # ... and with it the L1 position's actual reply (Gholdengo) enters the counter-picker's top three as well.
    pool, ours, theirs, actual = live_first_position("d9405f02")
    board, search, us, them, rest = search_for(pool, ours, theirs)
    assert actual == "Gholdengo" and actual in {board.units[c].species for _, c in search.counter(us, them, rest)[:3]}


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
    tiers = matchup.load_tiers()
    prior = tuning.DEFAULTS["draft_counter_prior"]
    expected = sorted((counterpick.counter_value(board.units[c], [board.units[i] for i in us],
                                                 strength=prior * tiers[board.units[c].key]["rating"])[0], -rest.index(c))
                      for c in search.legal(them, rest))[::-1]
    assert [v for v, _ in ranked] == pytest.approx([v for v, _ in expected])
    assert counterpick.counter_score(board.units[ranked[0][1]].card, [board.units[i].card for i in us])[1].startswith("Incineroar:")
    assert board.units[ranked[0][1]].species == "Incineroar"
    order = draft.pick_order(1, True)  # picks 2-12: theirs at 2-3, 6-7, 10-11
    _, _, final_them = search.rollout(us, them, rest, order, opponent=draft.COUNTER)
    _, _, final_model = search.rollout(us, them, rest, order, opponent=draft.GREEDY)
    assert len(final_them) == 6 and board.units[final_them[0]].species == "Incineroar"
    assert final_them != final_model  # the two readings of the opponent draft differently
    no_counter = draft.DraftSearch(board, counter_replies=0)
    assert all(source == draft.GREEDY for _, source in no_counter.reply_set(us, them, rest))


# -- Draft / Team Preview coherence (Oct 8): a pick is worth the four we would bring ----------------------------------


def live_draft(prefix: str):
    """One live session from the fixtures: every full set the pool showed (today's pools hold cards the catalog lacks),
    the pick sequence as (ours?, species) and both final sixes in catalog spelling."""
    folder = next((p for p in LIVE_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)), None)
    if folder is None:
        pytest.skip(f"live capture {prefix} not present")
    seat = sorted({re.search(r"seat(\d)", p.name).group(1) for p in folder.glob("draft-*-seat*.json")})[0]
    drafts = sorted(folder.glob(f"draft-*-seat{seat}.json"), key=lambda p: int(re.search(r"-(\d+)-", p.name).group(1)))
    cards = dict(CATALOG)
    for d in drafts:
        for c in json.loads(d.read_text())["observation"]["available_cards"]:
            cards.setdefault(c["species"], c)
    by_id = {data.to_id(sp): sp for sp in cards}
    last = json.loads(drafts[-1].read_text())
    me = last["current_actor"]["agent_id"]
    picks = last["observation"]["picks"]
    preview = json.loads((folder / f"team_preview-00000-seat{seat}.json").read_text())["observation"]
    ours6 = [by_id[m["species"]] for m in preview["your_roster"]]
    theirs6 = [by_id[m["species"]] for m in preview["opponent_roster"]]
    seq = [(p["seat_id"] == me, p["species"]) for p in picks]
    seen = {sp for _, sp in seq}
    seq += [(True, sp) for sp in ours6 if sp not in seen] + [(False, sp) for sp in theirs6 if sp not in seen]
    return cards, seq, ours6, theirs6


def ranking_with(cards: dict, pool: list[str], mine: list[str], theirs: list[str]) -> list[str]:
    memory = MatchMemory()
    memory.my_seat_key = "me"
    for species in pool:
        memory.pool_cards[cards[species]["card_id"]] = cards[species]
    for species in mine:
        memory.my_cards[species_key(species)] = cards[species]
    for species in theirs:
        memory.opp_cards[species_key(species)] = cards[species]
    return [s.species for s in draft.rank_picks_v2(offered(memory), memory, tuning.DEFAULTS)]


@pytest.mark.parametrize("prefix", ["d9405f02", "1aef9227", "7ce71025", "0a2b7c2a", "da9c168e"])
def test_the_early_picks_of_the_losses_are_brought_or_not_picked_that_early(prefix):
    # Oct 7-8: in every loss an early pick stayed home at Team Preview (Iron Valiant at pick 1 in da9c168e). Now a
    # pick is valued by the four the finished six would bring, so what the drafter still takes that early, Team
    # Preview v2 brings on the final six; what it would not bring, it no longer takes there.
    from agent.pokemon import lineup

    cards, seq, ours6, theirs6 = live_draft(prefix)
    pool = sorted({sp for _, sp in seq} | set(ours6) | set(theirs6))
    cands = lineup.shortlist({species_key(s): cards[s] for s in ours6}, {species_key(s): cards[s] for s in theirs6},
                             [data.to_id(s) for s in ours6], top=1, params=tuning.DEFAULTS)
    brought = {data.to_id(b) for b in cands[0].bring}
    mine, theirs, checked = [], [], 0
    for is_me, species in seq:
        if is_me and len(mine) < 2:
            now = ranking_with(cards, pool, mine, theirs)[0]
            if now == species:
                assert data.to_id(species) in brought, (prefix, species, cands[0].bring, cands[0].notes[:2])
            checked += 1
        (mine if is_me else theirs).append(species)
    assert checked == 2


def test_a_finished_draft_is_worth_the_four_it_would_bring():
    from itertools import combinations

    pool = POOLS["L3"][2]
    board, search, us, them, rest = search_for(pool, [], [])
    ours = [board.index(matchup.unit(CATALOG[s]).key) for s in ("Tornadus", "Garchomp", "Great Tusk", "Hatterene", "Ursaluna", "Indeedee-F")]
    theirs = [board.index(matchup.unit(CATALOG[s]).key) for s in ("Incineroar", "Iron Hands", "Whimsicott", "Archaludon", "Sylveon", "Amoonguss")]
    value, four = search.finished(ours, theirs)
    assert len(four) == 4 and set(four) <= set(ours)
    their_fours = sorted(combinations(theirs, 4), key=lambda g: -board.score4(list(g), ours))[:search.their_fours]
    def blended(f):
        vals = [board.score4(list(f), list(g)) for g in their_fours]
        return (1 - search.robust) * sum(vals) / len(vals) + search.robust * min(vals)
    assert value == pytest.approx(max(blended(f) for f in combinations(ours, 4)))
    assert value == pytest.approx(blended(four))
    assert value != pytest.approx(board.score(ours, theirs))  # not the six-against-six team score
    plain = draft.DraftSearch(board, four_eval=False)
    assert plain.finished(ours, theirs) == (board.score(ours, theirs), tuple(ours))
    # the card outside the four we would bring adds nothing beyond denial: swapping it for a weaker one leaves the value
    bench = [i for i in ours if i not in four][0]
    weaker = board.index(matchup.unit(CATALOG["Sableye"]).key)
    swapped = [weaker if i == bench else i for i in ours]
    assert search.finished(swapped, theirs)[0] >= value - 0.3  # same best four, only their likeliest fours may shift


# -- Secret tie-breaks (Oct 8): a reader of the public repo cannot precompute our choices --------------------------


def test_tiebreak_is_deterministic_per_seed_and_varies_only_inside_the_window():
    values = [1.00, 0.98, 0.50]  # log-odds: the first two are 0.4 points of win chance apart, the third far behind
    assert draft.tiebreak(values, 0.01, None, "x") == 0  # no seed: always the best
    assert draft.tiebreak(values, 0.0, 7, "x") == 0  # eps 0: always the best
    assert all(draft.tiebreak(values, 0.01, 7, "pick:3") == draft.tiebreak(values, 0.01, 7, "pick:3") for _ in range(5))
    choices = {draft.tiebreak(values, 0.01, seed, "pick:3") for seed in range(40)}
    assert choices == {0, 1}  # both tied candidates get chosen across seeds, the distant third never
    assert {draft.tiebreak(values, 0.001, seed, "pick:3") for seed in range(40)} == {0}  # a tighter window: no tie
    assert {draft.tiebreak(values, 0.01, 7, f"pick:{n}") for n in range(40)} == {0, 1}  # the context matters too


def test_the_seed_is_secret_per_match_kept_on_the_memory_and_written_into_the_notes():
    memory = memory_for(POOLS["L3"][2])
    other = memory_for(POOLS["L3"][2])
    first = draft.choose_pick(offered(memory), memory, tuning.DEFAULTS)
    seed = memory.tiebreak_seed
    assert isinstance(seed, int) and seed.bit_length() <= 64
    assert any(n.startswith(f"tie-break seed {seed:#x}") for n in first.notes), first.notes
    assert draft.tiebreak_seed(memory.my_cards) == seed  # Team Preview finds the same seed through the match's cards
    assert draft.choose_pick(offered(memory), memory, tuning.DEFAULTS).card_id == first.card_id  # same match: same answer
    assert draft.tiebreak_seed(other.my_cards, other) != seed  # another match: another draw


def test_a_fixed_seed_reproduces_the_pick_and_seeds_differ_only_when_the_gap_is_inside_eps():
    from agent.pokemon import matchup

    pool = POOL_W1
    mine, theirs = ["Iron Hands", "Archaludon", "Arcanine", "Wo-Chien"], ["Incineroar", "Dragonite", "Salamence", "Ting-Lu", "Basculegion"]
    memory = memory_for(pool, mine, theirs)
    ranked = draft.rank_picks_v2(offered(memory), memory, tuning.DEFAULTS)
    gap = matchup.win_probability(ranked[0].score) - matchup.win_probability(ranked[1].score)
    picks = set()
    for seed in range(12):
        m, m2 = memory_for(pool, mine, theirs), memory_for(pool, mine, theirs)
        m.tiebreak_seed = m2.tiebreak_seed = seed
        pick = draft.choose_pick(offered(m), m, tuning.DEFAULTS).species
        assert draft.choose_pick(offered(m2), m2, tuning.DEFAULTS).species == pick  # a fixed seed reproduces the pick
        picks.add(pick)
    eps = tuning.DEFAULTS["draft_tiebreak_eps"]
    if gap <= eps:
        assert len(picks) >= 2 and picks <= {s.species for s in ranked if matchup.win_probability(ranked[0].score) - matchup.win_probability(s.score) <= eps}
    else:
        assert picks == {ranked[0].species}
    assert ranked[0].species in {s.species for s in ranked[:3]}


# -- B1: the first-pick policy ----------------------------------------------------------------------------------------


def test_first_round_picks_follow_the_review_targets():
    # The review's first-pick targets from the live drafts, plus today's loss. The lookahead alone put Sylveon first
    # on the L3 pool; the policy (duel margin over the pool, role scarcity, counter-exposure, the guard list) makes
    # the first pick one the Oct 7 opponent would take, and stays out of the way where the search was already right.
    strong = {"Incineroar", "Gholdengo", "Archaludon", "Iron Hands"}
    assert ranking(POOLS["L3"][2])[0] in strong
    assert ranking(POOLS["L4"][2])[0] in strong
    assert ranking(POOLS["L1"][2], theirs=["Dragonite"])[0] == "Gholdengo"
    assert ranking(POOLS["L2"][2], theirs=["Incineroar"])[0] != "Iron Bundle"
    assert ranking(POOL_W1, mine=["Iron Hands"], theirs=["Incineroar"])[0] == "Archaludon"
    cards, seq, ours6, theirs6 = live_draft("da9c168e")
    pool = sorted({sp for _, sp in seq} | set(ours6) | set(theirs6))
    assert ranking_with(cards, pool, [], [])[0] != "Iron Valiant"  # benched at Team Preview, and the game was lost
    off = tuning.merged({"draft_first_pick_w": 0.0})
    assert ranking(POOLS["L3"][2], params=off)[0] not in strong  # the search alone does not get L3 right


def test_the_policy_reads_the_pool_and_fades_after_our_second_pick():
    board, search, us, them, rest = search_for(POOLS["L3"][2], [], [])
    inc = board.index(matchup.unit(CATALOG["Incineroar"]).key)
    terms = draft.first_pick_terms(board, inc, rest)
    assert terms["scarcity"] == pytest.approx(1.0 + 0.25)  # the only Intimidate in that pool; one of four Fake Out users
    assert terms["guard"] == 1.0 and terms["exposure"] >= 1 and -1.0 <= terms["strength"] <= 1.0
    tornadus = board.index(matchup.unit(CATALOG["Tornadus"]).key)
    assert draft.first_pick_terms(board, tornadus, rest)["guard"] == -1.0
    bonus, _ = draft.first_pick_bonus(board, inc, rest, tuning.DEFAULTS)
    assert bonus == pytest.approx(tuning.DEFAULTS["draft_first_strength_w"] * terms["strength"] + tuning.DEFAULTS["draft_first_scarcity_w"] * terms["scarcity"]
                                  - tuning.DEFAULTS["draft_first_exposure_w"] * terms["exposure"] / 6 + tuning.DEFAULTS["draft_first_guard_w"])
    memory = memory_for(POOLS["L3"][2], mine=["Incineroar", "Archaludon"], theirs=["Iron Hands", "Sylveon"])
    assert not any(n.startswith("first-pick policy") for s in draft.rank_picks_v2(offered(memory), memory, tuning.DEFAULTS) for n in s.notes)
    first = draft.rank_picks_v2(offered(memory_for(POOLS["L3"][2])), memory_for(POOLS["L3"][2]), tuning.DEFAULTS)
    assert all(any(n.startswith("first-pick policy") for n in s.notes) for s in first)


# -- B4: coverage holes --------------------------------------------------------------------------------------------


L1_OURS = ["Baxcalibur", "Landorus-Therian", "Sneasler", "Garchomp", "Cresselia", "Lucario"]
L1_THEIRS = ["Dragonite", "Gholdengo", "Iron Hands", "Archaludon", "Iron Bundle", "Indeedee-F"]


def test_the_l1_six_had_one_answer_to_air_balloon_gholdengo():
    from agent.pokemon import lineup

    u = lambda s: matchup.unit(CATALOG[s])  # noqa: E731
    hits = {s: matchup.duel(u(s), u("Gholdengo")).hit_a for s in L1_OURS}
    assert sum(1 for h in hits.values() if h <= 18.0) == 5 and hits["Sneasler"] == 0.0  # Close Combat into a Ghost
    assert 40.0 <= hits["Baxcalibur"] <= 43.0 and matchup.duel(u("Baxcalibur"), u("Gholdengo")).hit_a_move == "glaiverush"
    board = matchup.Board([u(s) for s in L1_OURS + L1_THEIRS])
    us, them = list(range(6)), list(range(6, 12))
    assert board.holes(us, them) == []
    without_bax = [1, 2, 3, 4]  # Landorus-Therian, Sneasler, Garchomp, Cresselia: the four we actually brought
    assert "Gholdengo" in {board.units[j].species for j in board.holes(without_bax, them)}
    assert board.score4(without_bax, them[:4]) == board.score4(without_bax, them[:4])  # the matchup score ignores holes ...
    search = draft.DraftSearch(board, hole_w=0.5)
    plain = draft.DraftSearch(board, hole_w=0.0)
    assert search.finished(us, them)[0] <= plain.finished(us, them)[0]  # ... the finished-draft value charges them
    assert "Baxcalibur" in {board.units[k].species for k in search.finished(us, them)[1]}  # and keeps the answer in the four
    # the old scorer and the preview carry the same rule
    mine = [lineup.profile(CATALOG[s]) for s in L1_OURS]
    theirs = [lineup.profile(CATALOG[s]) for s in L1_THEIRS]
    assert lineup.coverage_holes(mine[1:5], theirs) and "Gholdengo" in lineup.coverage_holes(mine[1:5], theirs)
    with_rule, notes = lineup.score_four(mine[1:5], theirs, tuning.merged({"lineup_hole_w": 5.0}))
    without, _ = lineup.score_four(mine[1:5], theirs, tuning.merged({"lineup_hole_w": 0.0}))
    assert with_rule < without and any(n.startswith("coverage hole") for n in notes)
    model = lineup.PreviewModel([CATALOG[s] for s in L1_OURS], [CATALOG[s] for s in L1_THEIRS], hole_w=0.5)
    plain_model = lineup.PreviewModel([CATALOG[s] for s in L1_OURS], [CATALOG[s] for s in L1_THEIRS], hole_w=0.0)
    four, other = (1, 2, 3, 4), (6, 7, 8, 9)
    assert model.pairing(four, other) < plain_model.pairing(four, other)


# -- the da9c168e post-mortem additions ----------------------------------------------------------------------------


def test_a_card_the_fit_never_saw_is_assumed_below_average():
    cards = matchup.load_model()["cards"]
    unseen = matchup.unseen_strength(cards)
    assert unseen == pytest.approx(-0.079, abs=0.01)
    values = sorted(float(e["s"]) for e in cards.values())
    assert values[len(values) // 4 - 1] <= unseen <= values[len(values) // 4 + 1]
    valiant = matchup.unit(CATALOG["Iron Valiant"])  # in the catalog since Oct 8, not in the fit
    assert valiant.key not in cards
    mystery = matchup.unit({"card_id": "x-1", "species": "Totally Made Up", "item": "Leftovers", "ability": "Pressure", "nature": "Serious",
                            "evs": {}, "moves": ["Tackle", "Protect"]})
    board = matchup.Board([valiant, mystery, matchup.unit(CATALOG["Gholdengo"])], card_weight=1.0, tier_adjust=0.0)
    assert board.resid[0] == pytest.approx(unseen) and board.resid[1] == pytest.approx(unseen)
    assert board.resid[2] == pytest.approx(float(cards[board.units[2].key]["s"]))
    assert matchup.Board([valiant], card_weight=0.0).resid == [0.0]


def test_the_loses_to_note_names_the_biggest_counter_first():
    cards, seq, ours6, theirs6 = live_draft("da9c168e")
    pool = sorted({sp for _, sp in seq} | set(ours6) | set(theirs6))
    memory = MatchMemory()
    memory.my_seat_key = "me"
    for species in pool:
        memory.pool_cards[cards[species]["card_id"]] = cards[species]
    valiant = next(s for s in draft.rank_picks_v2(offered(memory), memory, tuning.DEFAULTS) if s.species == "Iron Valiant")
    note = next(n for n in valiant.notes if n.startswith("loses to "))
    assert note.startswith("loses to Flutter Mane +1.00"), note  # the one-hit KO they took next, first
    margins = [float(part.rsplit(" ", 1)[1]) for part in note[len("loses to "):].split(" still in the pool")[0].split(", ")]
    assert margins == sorted(margins, reverse=True)


# -- Sleep in the finished-draft value (Oct 8 evening, 51c86e5c) ------------------------------------------------------


def test_the_finished_draft_value_charges_their_spore_when_the_weight_is_on():
    # 51c86e5c, our last pick: their six holds Amoonguss (Spore). With draft_sleep_w on, every four-against-four value
    # is charged their sleepers' reach into our four (Board.sleep_reach), so a sleep-proof Grass type loses half as much
    # to their Spore as Toxapex does; its notes say why at any weight. The default is 0, the deployed draft, until a
    # simulator series sets it (unmeasured at 0.4 it changed 21 of 120 live picks, Oct 9 review).
    folder = next((p for p in LIVE_DIR.iterdir() if p.is_dir() and p.name.startswith("51c86e5c")), None)
    if folder is None:
        pytest.skip("live capture 51c86e5c not present")
    drafts = sorted(folder.glob("draft-*-seat0.json"))
    first, last = (json.loads(d.read_text())["observation"] for d in (drafts[0], drafts[-1]))
    pool = sorted({c["species"] for c in first["available_cards"]} | {p["species"] for p in last["picks"]})
    ours = ["Chi-Yu", "Gyarados", "Archaludon", "Tornadus", "Indeedee-F"]
    theirs = ["Iron Hands", "Chien-Pao", "Landorus-Therian", "Ursaluna", "Hatterene", "Amoonguss"]
    assert tuning.DEFAULTS["draft_sleep_w"] == 0.0
    w = 0.4

    def values(params):
        memory = memory_for(pool, ours, theirs)
        return {s.species: s for s in draft.rank_picks_v2(offered(memory), memory, params)}
    priced, plain = values(tuning.merged({"draft_sleep_w": w})), values(tuning.DEFAULTS)
    assert {"Rillaboom", "Whimsicott", "Toxapex"} <= set(priced)
    gain = {s: priced[s].score - plain[s].score for s in ("Rillaboom", "Whimsicott", "Toxapex")}
    assert gain["Toxapex"] <= -0.9 * w  # their Spore against a four with no sleep-proof card: the full weight
    assert gain["Rillaboom"] - gain["Toxapex"] >= 0.4 * w and gain["Whimsicott"] - gain["Toxapex"] >= 0.4 * w
    assert next(iter(priced)) == next(iter(plain)) == "Rillaboom"  # what we took live
    assert "sleep-proof against their Amoonguss" in plain["Rillaboom"].notes
    assert not any("sleep-proof" in n for n in plain["Toxapex"].notes)


def test_the_draft_sleep_term_is_one_sided():
    # Only THEIR sleepers count, against our sleep-proof cards; our own Amoonguss / Torkoal earn nothing (the brief's
    # scope; the Oct 9 review measured that half moving 10 live picks toward them). 46fd3d95's sixes: our Torkoal
    # (Yawn) against a six with no sleeper, where the two-sided term had swapped our last pick Rillaboom for Cresselia.
    ours = ["Archaludon", "Torkoal", "Iron Valiant", "Arcanine", "Milotic", "Rillaboom"]
    theirs = ["Iron Hands", "Landorus-Therian", "Dragonite", "Basculegion", "Ursaluna", "Hatterene"]
    units = [matchup.unit(CATALOG[s]) for s in ours + theirs + ["Amoonguss"]]
    board = matchup.Board(units)
    us, them, spore = list(range(6)), list(range(6, 12)), 12
    assert board.sleep_reach(us, them) == pytest.approx(0.5 * 0.5)  # our Yawn into their six (Hatterene bounces it)
    assert board.sleep_reach(them, us) == 0.0
    on, off = draft.DraftSearch(board, sleep_w=0.4), draft.DraftSearch(board, sleep_w=0.0)
    assert on.finished(us, them) == off.finished(us, them)  # no credit for our own sleeper
    assert on.finished(us, them, include=5) == off.finished(us, them, include=5)
    their_six = [6, 7, 8, 9, 10, spore]  # with their Amoonguss in place of Hatterene, the charge applies
    assert on.finished(us, their_six)[0] < off.finished(us, their_six)[0]
