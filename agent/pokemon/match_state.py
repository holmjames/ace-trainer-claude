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
- ``preview_sim._CALC``: its values are pure, but a warm memo lets the deadline-bound deep Team Preview finish more
  rollouts than the cold one a fresh process starts with.

``new_match()`` empties them; ``PokemonAgent`` calls it where it creates a match's memory, the runtime's match boundary
(``create_agent()`` once per worker). In a fresh process they are empty already, so live play is unchanged.

Kept across matches, because each is a pure function of its key and only saves time (no caller changes a cached
object; tests/test_match_state.py checks a whole live match leaves them as it found them): ``matchup._UNITS``,
``matchup._OPENINGS`` (speed ties played both ways and averaged), ``draft._SUPPORT_RESIST``, and the shipped-file
caches (``matchup._MODEL_CACHE``, ``matchup._TIERS_CACHE``, ``draft._ADJUST_TABLES``, ``data.pokedex`` / ``moves`` /
``_catalog``). The secret tie-break seed lives on the match memory; ``draft._SEEDS``, the registry Team Preview finds
it through, answers only for the card dict that registered it, so it needs no reset and a reset never costs a match
that is still being played (the other seat of a simulator game) its seed.
"""

from __future__ import annotations

import sys

# The modules holding per-match caches, by import name. A module not imported yet holds nothing and is not imported
# here: a fresh process loads it at first use, as before.
MODULES = ("draft", "matchup", "preview_sim")


def new_match() -> None:
    """Empty every per-match cache (see the module docstring), so the match about to start sees what a freshly
    started process would."""
    for name in MODULES:
        module = sys.modules.get(f"{__package__}.{name}")
        if module is not None:
            module.reset_match_state()
