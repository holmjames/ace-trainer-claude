"""Workstream C (Team Preview) regression tests on the live catalog: the v2 failure note (I8) and the lead veto (C2)."""

from __future__ import annotations

import json
import logging

import pytest

from agent.pokemon import data, lineup, tuning
from agent.pokemon.memory import species_key

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")
V2 = tuning.merged({"lineup_v2": 1})


def rosters(mine: list[str], theirs: list[str]):
    return ({species_key(s): CATALOG[s] for s in mine}, {species_key(s): CATALOG[s] for s in theirs}, [data.to_id(s) for s in mine])


# -- I8: a v2 failure is logged and noted, never silent -----------------------------------------------------------

MINE_M2 = ["Iron Bundle", "Iron Hands", "Archaludon", "Gyarados", "Whimsicott", "Hatterene"]
THEIRS_M2 = ["Incineroar", "Flutter Mane", "Sylveon", "Rillaboom", "Kingambit", "Dondozo"]


def test_a_failing_v2_leaves_a_one_line_note_and_a_log_record(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(lineup, "shortlist_v2", broken)
    with caplog.at_level(logging.WARNING, logger="agent.pokemon.lineup"):
        cands = lineup.shortlist(*rosters(MINE_M2, THEIRS_M2), params=V2)
    assert cands and len(cands[0].bring) == 4  # the old scorer still answers
    assert any(n == "Team Preview v2 failed (RuntimeError: boom); old scorer used" for n in cands[0].notes), cands[0].notes
    assert [r for r in caplog.records if "Team Preview v2 failed (RuntimeError: boom)" in r.getMessage()]
    assert not any(n.startswith("v2:") for n in cands[0].notes)
