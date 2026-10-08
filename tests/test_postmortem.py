"""scripts/postmortem.py reads our protocol side from the observation, so player-2 matches name the right opponent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "fixtures" / "live"


def _postmortem():
    spec = importlib.util.spec_from_file_location("postmortem", ROOT / "scripts" / "postmortem.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("session, side, brought", [
    ("bcbaabe2", "p1", ["Dragonite", "Salamence", "Ting-Lu", "Rillaboom"]),
    ("da9c168e", "p2", ["Flutter Mane", "Incineroar", "Gyarados", "Lucario"]),
    ("46fd3d95", "p2", ["Iron Hands", "Dragonite", "Hatterene", "Ursaluna"]),
])
def test_opponent_seen_from_our_side(session, side, brought, tmp_path, capsys):
    if not any(p.name.startswith(session) for p in LIVE.iterdir()):
        pytest.skip(f"fixture {session} not present")
    _postmortem().main([session, "--fixtures", str(LIVE), "--logs", str(tmp_path)])
    out = capsys.readouterr().out
    assert f"We are {side} in the protocol log" in out
    assert f"Opponent brought (seen in battle): {brought}" in out


def test_our_side_reads_team_keys():
    pm = _postmortem()
    assert pm.our_side({"team": {"p2: Archaludon": {}, "p2: Torkoal": {}}}) == ("p2", "p1")
    assert pm.our_side({"team": {"p1: Gyarados": {}}}) == ("p1", "p2")
    assert pm.our_side({}) == ("p1", "p2")
