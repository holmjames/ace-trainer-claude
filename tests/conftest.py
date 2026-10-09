"""Shared test configuration.

The deep Team Preview (``lineup_sim_budget_s``, 20 s live) simulates openings for that many seconds per preview. Tests
outside ``test_preview_*`` replay previews through ``PokemonAgent`` and the simulator's in-process players, so with the
live default every such preview would spend ~15 s and the suite would take half an hour. Outside the preview tests the
budget is pinned to 0, which is the stage2 preview exactly (``tests/test_preview_deep.py`` proves the equivalence).
"""

import pytest

from agent.pokemon import tuning


@pytest.fixture(scope="module", autouse=True)
def _fast_preview_outside_the_preview_tests(request):
    name = request.module.__name__.rsplit(".", 1)[-1]
    if name.startswith("test_preview_"):
        yield
        return
    old = tuning.DEFAULTS.get("lineup_sim_budget_s")
    tuning.DEFAULTS["lineup_sim_budget_s"] = 0.0
    try:
        yield
    finally:
        tuning.DEFAULTS["lineup_sim_budget_s"] = old
