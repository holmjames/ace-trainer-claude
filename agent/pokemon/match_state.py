"""Where a match starts: the module state one match fills, emptied before the next match is played.

The live runtime plays every match in a freshly spawned worker process, so every module cache is empty when a match
starts there. A long-lived process (the simulator's arms, scripts/live_scenarios.py, the replay tools, the test suite)
plays many matches one after another, and a cache an earlier match filled could make a later match decide differently
from the fresh process the runtime gives it (Oct 10 state audit):

- ``matchup._DUELS`` caches a 1-on-1 race in whichever direction was computed first, and on an exact speed tie the two
  directions differ (Grimmsnarl / Sylveon at 80 Speed: margin -0.18 one way, -0.57 the other; 4 of the catalog's 1,128
  pairs). In one process 93a98b6f's draft values at picks 4-6 and its deep Team Preview values followed the direction
  6ce0b81e had cached before it;
- ``draft._BOARDS`` and ``draft._SUPPORT_PAIRS`` are built from those races;
- ``preview_sim._CALC``: its values are pure, so it changes no value a decision reads; it is emptied so the deep Team
  Preview's rollouts start as cold as in the runtime's worker (a warm memo runs them faster, which only the wall clock
  turns into a different search: see below).

``new_match()`` empties them; ``PokemonAgent`` calls it where it creates a match's memory, the runtime's match boundary
(``create_agent()`` once per worker). In a fresh process they are empty already, so live play is unchanged.

Kept across matches, because each is a pure function of its key (no caller changes a cached object;
tests/test_match_state.py checks a whole live match leaves them as it found them): ``matchup._UNITS``,
``matchup._OPENINGS`` (speed ties played both ways and averaged), ``draft._SUPPORT_RESIST``, and the shipped-file
caches (``matchup._MODEL_CACHE``, ``matchup._TIERS_CACHE``, ``draft._ADJUST_TABLES``, ``draft._CARD_ADJUST_TABLES``
and ``_CARD_ADJUST_ERRORS``, ``data.pokedex`` / ``moves`` / ``_catalog``). The secret tie-break seed lives on the
match memory; ``draft._SEEDS``, the registry Team Preview finds it through, answers only for the card dict that
registered it, so it needs no reset and a reset never costs a match that is still being played (the other seat of a
simulator game) its seed.

What "decides as a fresh process" covers. Kept or emptied, none of these caches changes a value a decision reads; a warm
one only makes the code faster. So a long-lived process decides each match as a fresh process does wherever a decision
does not read the wall clock: the code's draft picks, its Team Preview at lineup_sim_budget_s 0 and its battle rows, and
the deep Team Preview on a fixed clock (Oct 10 review: 0 of 420 live decisions differ at budget 0 and 20 on a fake
clock). With the real clock, two reads let speed reach a decision or what the model is shown, in a long-lived process
and between two fresh processes on a machine under different load alike:

- the deep Team Preview (``lineup_sim_budget_s`` > 0) is handed the budget v2's shortlist left on the wall clock
  (``lineup.shortlist_v2``), and ``preview_sim.Search.schedule`` sizes the search by a step function of it (steps
  0.15-0.9 s apart near the live 20 s budget); its deadline also stops a search that runs slow. The kept Units and
  openings make v2 some 20-50 ms faster in a warm process (median, the 22 live previews, Oct 10), so a match whose
  budget sits that close to a step can be searched on another schedule than in a fresh process. None of the 22 was
  (the nearest sat 0.14 s from a step); run-to-run noise under load is larger (about 200 ms);
- the opponent's seconds per draft pick (``MatchMemory.opponent_evidence``) subtract our own think time, measured on
  the wall clock, from the server's clock; they break ties in the opponent's class, which the code's battle prediction
  and the model's Team Preview and battle messages carry (a real-clock replay of the live matches, one process against
  fresh ones: no decision differed; the seconds moved by 0.01).

Sizing the search from the nominal budget, with a rollout-count stop, would take the first off the clock; that waits
until after the Oct 16 tournament.
"""

from __future__ import annotations

import sys

# The modules holding per-match caches, by import name. A module not imported yet holds nothing and is not imported
# here: a fresh process loads it at first use, as before.
MODULES = ("draft", "matchup", "preview_sim")


def new_match() -> None:
    """Empty every per-match cache (see the module docstring), so the match about to start reads the caches a freshly
    started process would (it decides as that process does wherever no decision reads the wall clock)."""
    for name in MODULES:
        module = sys.modules.get(f"{__package__}.{name}")
        if module is not None:
            module.reset_match_state()
