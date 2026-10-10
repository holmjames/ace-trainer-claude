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

agent/pokemon/match_state.py empties _DUELS, _BOARDS, _SUPPORT_PAIRS and _CALC where a match's memory is created
(PokemonAgent); the caches that are pure functions of their keys stay. Every test here fails on 81c4591 except two guards: a match starting while
another is live leaves that match's seed alone, and the caches kept across matches come out of a match unchanged.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon import data, draft, matchup, preview_sim, tuning
from agent.pokemon.memory import MatchMemory
from test_live_semantics import LIVE, _sequence

ROOT = Path(__file__).resolve().parent.parent
SEED = 0x5EED5EED5EED  # pinned on the match memory, as the replay tools do: every run draws the same tie-breaks
TIMING = re.compile(r"in \d+(\.\d+)?s|\d+ rollouts|\d+ rounds")
# What the decision log keeps of each decision: the play and everything a reader or the model sees around it.
# Left out: timestamps, latencies, the clock, and the opponent's measured think time (our own wall clock feeds it).
KEEP = {
    "draft": ("card_id", "species", "score", "notes", "offered"),
    "lineup": ("payload", "candidates", "preview_table"),
    "turn": ("payload", "turn_sheet", "battle_roster", "code_prediction", "opponent_prediction", "win_condition", "last_turns"),
}
FIRST, SECOND = ("6ce0b81e", 0), ("93a98b6f", 0)  # the Oct 10 audit's pair: the first caches Grimmsnarl / Sylveon the other way


def _agent(**kw) -> PokemonAgent:
    return PokemonAgent(None, capture_dir="", log=lambda *_: None, **kw)


def _untimed(value):
    if isinstance(value, str):
        return TIMING.sub("<t>", value)
    if isinstance(value, list):
        return [_untimed(v) for v in value]
    if isinstance(value, dict):
        return {k: _untimed(v) for k, v in value.items()}
    return value


def replay_match(session8: str, seat: int, log_dir: str) -> list[dict]:
    """Every decision of one captured live match, played through a brand-new agent (the runtime's create_agent(), code
    only, the deep Team Preview off), as the decision log records it."""
    folder = next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(session8))
    agent = _agent(version="match-state", log_dir=log_dir, params=tuning.merged({"lineup_sim_budget_s": 0.0}))
    agent.memory.tiebreak_seed = SEED
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        agent.choose_action(state, DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                                   agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat))
    rows = [json.loads(line) for f in sorted(Path(log_dir).glob("*.jsonl")) for line in f.read_text().splitlines() if line.strip()]
    return [{"kind": r["kind"], **{k: _untimed(r.get(k)) for k in KEEP[r["kind"]]}} for r in rows if r.get("kind") in KEEP]


def _new_process(matches: list[tuple[str, int]], log_root: Path) -> list[list[dict]]:
    """``replay_match`` for each of ``matches``, one after another, in ONE new Python process. Several matches: a
    long-lived process meeting them in order; one match: the runtime's spawn-per-match worker."""
    code = ("import json, sys; sys.path[:0] = sys.argv[1:3]; import agent, test_match_state as t; "
            "games = [(a.split(':')[0], int(a.split(':')[1])) for a in sys.argv[4:]]; "
            "print(json.dumps({'agent': agent.__file__, 'matches': [t.replay_match(s, seat, f'{sys.argv[3]}/{s}-{seat}') for s, seat in games]}))")
    proc = subprocess.run([sys.executable, "-c", code, str(ROOT), str(ROOT / "tests"), str(log_root), *[f"{s}:{seat}" for s, seat in matches]],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(out["agent"]).resolve().is_relative_to(ROOT), out["agent"]  # the new process ran this tree's code
    return out["matches"]


def _canon(row: dict) -> str:
    return json.dumps(row, sort_keys=True, default=str)


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


# -- captured live matches: one process against fresh processes ----------------------------------------------------


def test_two_live_matches_in_one_process_decide_as_two_fresh_processes(tmp_path):
    """6ce0b81e then 93a98b6f (seat 0 each) in one process, against each alone in a new process: every draft, Team
    Preview and battle decision, its scores, notes, candidates, rows and the model's view are the same. On 81c4591 the
    second match's picks 4-6 came out with other values and notes in the long-lived process."""
    long_lived = _new_process([FIRST, SECOND], tmp_path / "one-process")
    fresh = [_new_process([match], tmp_path / f"fresh-{match[0]}")[0] for match in (FIRST, SECOND)]
    for (session8, seat), mine, theirs in zip((FIRST, SECOND), long_lived, fresh):
        assert mine and len(mine) == len(theirs), (session8, len(mine), len(theirs))
        assert {r["kind"] for r in mine} == {"draft", "lineup", "turn"}
        differ = [f"{a['kind']} #{i}: {[k for k in a if a[k] != b.get(k)]}" for i, (a, b) in enumerate(zip(mine, theirs))
                  if _canon(a) != _canon(b)]
        assert not differ, f"{session8} seat {seat}: {len(differ)} of {len(mine)} decisions differ from a fresh process: {differ[:6]}"


def test_the_caches_kept_across_matches_come_out_of_a_match_unchanged(tmp_path):
    """Guard (passes on 81c4591 too): the caches a new match keeps are pure only while no caller changes a cached
    object, so a whole live match must leave every entry an earlier match left exactly as it was."""
    replay_match(*FIRST, str(tmp_path / "first"))

    def kept() -> dict[str, dict]:
        return {"units": matchup._UNITS, "openings": matchup._OPENINGS, "models": matchup._MODEL_CACHE,
                "tiers": matchup._TIERS_CACHE, "first-pick tables": draft._ADJUST_TABLES, "resists": draft._SUPPORT_RESIST,
                "pokedex": data.pokedex(), "moves": data.moves(), "defaults": tuning.DEFAULTS}

    before = {name: copy.deepcopy(dict(cache)) for name, cache in kept().items()}
    catalog = copy.deepcopy(data._catalog())
    assert before["units"] and before["openings"]
    replay_match(*SECOND, str(tmp_path / "second"))
    after = kept()
    for name, entries in before.items():
        changed = [k for k, v in entries.items() if k in after[name] and after[name][k] != v]
        assert not changed, f"{name}: {len(changed)} cached entries changed during a match, e.g. {str(changed[0])[:120]}"
    assert data._catalog() == catalog
