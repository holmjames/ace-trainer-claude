"""Each match decides as a freshly started process would, however many matches the process has played before it.

The live runtime plays every match in its own spawned worker, so every module cache is empty when a match starts
there. A long-lived process (the simulator's arms, scripts/live_scenarios.py, the replay tools, this suite) plays many
matches one after another. The Oct 10 state audit found what an earlier match left behind that a later match in the
same process could read:

- draft._SEEDS (the match's secret tie-break seed, registered under id(memory.my_cards)): a later match whose card dict
  took a freed dict's address inherited the earlier match's seed. The registry now answers only for the dict that
  registered;
- matchup._DUELS caches a 1-on-1 race in whichever direction was computed first, and on an exact speed tie the two
  directions differ (Grimmsnarl / Sylveon at 80 Speed: -0.18 one way, -0.57 the other), so an earlier match's
  direction changed a later match's draft and Team Preview values (93a98b6f after 6ce0b81e); draft._BOARDS and
  draft._SUPPORT_PAIRS are built from those races;
- preview_sim._CALC: pure values, but a warm memo lets the deadline-bound deep Team Preview run more rollouts than the
  fresh process the runtime gives the match.

agent/pokemon/match_state.py empties the last three where a match's memory is created (PokemonAgent); the caches that
are pure functions of their keys stay. Every test here fails on 81c4591 except one guard: a match starting while
another is live leaves that match's seed alone.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.agent import PokemonAgent
from agent.pokemon import data, draft, matchup, preview_sim, tuning
from agent.pokemon.memory import MatchMemory


def _agent(**kw) -> PokemonAgent:
    return PokemonAgent(None, capture_dir="", log=lambda *_: None, **kw)


def _card(species: str) -> dict:
    card = data.catalog_card(species=species)
    assert card, f"{species} is not in the catalog"
    return card


# -- the secret tie-break seed --------------------------------------------------------------------------------------


def test_a_recycled_card_dict_address_never_hands_a_new_match_the_old_matchs_seed(monkeypatch):
    """CPython gives a freed dict's address to the next dict it creates (4 and 6 matches of 30 in a tight loop in one
    process took an earlier match's seed, Oct 10 audit). Modelled here with an id() that puts every dict at one
    address: the registry must answer only for the very dict it registered."""
    monkeypatch.setattr(draft, "id", lambda obj: 0xADD, raising=False)
    draws = iter([111, 222])
    monkeypatch.setattr(draft, "secrets", SimpleNamespace(randbits=lambda bits: next(draws)))
    old = MatchMemory()
    assert draft.tiebreak_seed(old.my_cards, old) == 111
    new = MatchMemory()  # the next match: its card dict lands where the old one was
    assert draft.known_seed(new.my_cards) is None  # its Team Preview before any draft decision: no seed, as in a fresh process
    assert draft.tiebreak_seed(new.my_cards, new) == 222  # its first draft decision draws its own
    assert new.tiebreak_seed == 222 and draft.known_seed(new.my_cards) == 222
    assert draft.known_seed(old.my_cards) is None and old.tiebreak_seed == 111  # each match keeps its own on its memory


def test_a_match_that_starts_while_another_is_live_leaves_that_matchs_seed_alone():
    """Two seats of one simulator game share a process: building the second seat's agent must not cost the first its
    seed between its draft and its Team Preview."""
    first = _agent()
    seed = draft.tiebreak_seed(first.memory.my_cards, first.memory)
    second = _agent()  # another match (the other seat) starts in this process
    assert draft.known_seed(first.memory.my_cards) == seed
    assert draft.known_seed(second.memory.my_cards) is None


# -- races computed by an earlier match -----------------------------------------------------------------------------


def test_an_earlier_match_never_decides_which_way_a_speed_tied_race_was_cached():
    """Grimmsnarl and Sylveon both run 80 Speed; the race treats the tie as simultaneous but resolves slot 0 first, so
    duel(a, b) != -duel(b, a) there. Whatever an earlier match computed, this match's races (and the Board its draft
    builds from them) are the ones a fresh process computes."""
    G, S = matchup.unit(_card("Grimmsnarl")), matchup.unit(_card("Sylveon"))
    assert G.mon.stat("spe") == S.mon.stat("spe")
    pool = [_card(s) for s in ("Grimmsnarl", "Sylveon", "Incineroar", "Iron Hands")]
    knobs = (float(tuning.DEFAULTS["draft_card_w"]), float(tuning.DEFAULTS["draft_tier_adjust"]))
    matchup._DUELS.clear()  # a fresh process: nothing computed yet
    draft._BOARDS.clear()
    fresh = matchup.duel(G, S)
    fresh_board = draft._board_for(pool, *knobs)
    fresh_row = list(fresh_board.base[fresh_board.index(G.key)])
    # an earlier match in the same process ran the race the other way round first
    matchup._DUELS.clear()
    draft._BOARDS.clear()
    earlier = matchup.duel(S, G)
    assert earlier.flipped().margin != fresh.margin  # the tie really is direction-dependent
    draft._board_for(list(reversed(pool)), *knobs)  # and built that pool's Board from it
    _agent()  # the next match starts
    assert matchup.duel(G, S) == fresh
    board = draft._board_for(pool, *knobs)
    assert board is not fresh_board and list(board.base[board.index(G.key)]) == fresh_row


# -- the preview calculator -----------------------------------------------------------------------------------------


def test_the_preview_calculator_starts_every_match_as_cold_as_a_fresh_process():
    """Timing only (its values are pure): a warm memo lets the deadline-bound deep Team Preview finish more rollouts."""
    for species in ("Grimmsnarl", "Sylveon"):
        preview_sim._CALC.info(matchup.unit(_card(species)))
    preview_sim._CALC.misses += 1
    assert preview_sim._CALC.infos  # the module memo is warm now
    _agent()  # the next match starts
    calc = preview_sim._CALC
    assert not (calc.hits or calc.speeds or calc.accuracy or calc.infos) and calc.misses == 0


# -- the whole inventory at a match boundary ------------------------------------------------------------------------


def test_a_new_match_empties_the_per_match_caches_and_keeps_the_pure_ones():
    pool = [_card(s) for s in ("Grimmsnarl", "Sylveon", "Incineroar", "Iron Hands", "Gholdengo", "Rillaboom")]
    board = draft._board_for(pool, 1.0, 1.0)
    draft.support_table(board)  # the support facts (draft_support_w > 0) read off that board
    units = [board.units[i] for i in range(4)]
    matchup.opening(units[:2], units[2:])
    preview_sim._CALC.info(units[0])
    assert draft._BOARDS and draft._SUPPORT_PAIRS and matchup._DUELS and preview_sim._CALC.infos
    kept = {"units": dict(matchup._UNITS), "openings": dict(matchup._OPENINGS), "resists": dict(draft._SUPPORT_RESIST)}
    _agent()  # the next match starts
    assert not (draft._BOARDS or draft._SUPPORT_PAIRS or matchup._DUELS or preview_sim._CALC.infos)
    assert all(matchup._UNITS.get(k) is v for k, v in kept["units"].items())  # pure, kept: they only save time
    assert all(matchup._OPENINGS.get(k) is v for k, v in kept["openings"].items())
    assert all(draft._SUPPORT_RESIST.get(k) is v for k, v in kept["resists"].items())


