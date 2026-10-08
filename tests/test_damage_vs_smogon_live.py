"""Our damage model against Smogon's official calculator on the LIVE card catalog (data/cards.json = sim/cards.json).

tests/fixtures/smogon_calc_live.json comes from `node scripts/gen_smogon_fixture.js` (every live card attacking every
other with its real set, sampled weather/terrain/screens/stages/burn, plus forced cases for Pixilate, sand, Air Balloon,
Unaware and resist berries). The first run on Oct 7 found four mechanics the model lacked, all in the live pool:
Pixilate (Sylveon), Adaptability (Basculegion), Unaware (Dondozo) and resist berries (Tyranitar's Chople Berry).
Regenerate it whenever scripts/pool_from_live.py changes the catalog.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from agent.pokemon import data

from test_damage_vs_smogon_full import _errors

LIVE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smogon_calc_live.json"
pytestmark = pytest.mark.skipif(not LIVE_FIXTURE.exists() or not (data.DATA_DIR / "pokedex.json").exists(),
                                reason="needs tests/fixtures/smogon_calc_live.json and data/ tables")


def test_live_catalog_damage_tracks_the_official_calculator():
    errors = _errors(LIVE_FIXTURE)
    assert len(errors) > 900
    values = sorted(e for e, _ in errors)
    assert statistics.median(values) < 1.0
    assert values[int(len(values) * 0.95)] < 3.0
    worst, case = max(errors, key=lambda pair: pair[0])
    assert worst < 8.0, (worst, case["attacker"]["species"], case["move"], case["defender"]["species"], case["field"])


@pytest.mark.parametrize("attacker,move,defender", [
    ("Sylveon", "Hyper Voice", "Iron Hands"),        # Pixilate
    ("Basculegion", "Wave Crash", "Cresselia"),     # Adaptability
    ("Rillaboom", "Grassy Glide", "Dondozo"),       # Unaware
    ("Iron Hands", "Close Combat", "Tyranitar"),    # Chople Berry
])
def test_live_mechanics_land_within_a_few_percent(attacker, move, defender):
    errors = [(e, c) for e, c in _errors(LIVE_FIXTURE)
              if c["attacker"]["species"] == attacker and c["move"] == move and c["defender"]["species"] == defender]
    if not errors:
        pytest.skip(f"no fixture case for {attacker} {move} -> {defender}")
    assert max(e for e, _ in errors) < 4.0, errors[0]
