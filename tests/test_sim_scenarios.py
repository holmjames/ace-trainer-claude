"""scripts/scenarios.py and scripts/live_scenarios.py as catalogues: every hand scenario's card is the live card from the
catalog (the hand-written sets had drifted), the hand-written stand-ins exist only for species the catalog has never
seen, and every live position names a captured turn that exists."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from agent.pokemon import data

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "tests" / "fixtures" / "live"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scenarios = _load("scenarios")
live_scenarios = _load("live_scenarios")


def test_every_scenario_card_is_the_live_card_or_a_declared_stand_in():
    assert set(scenarios.HAND_SETS) < set(scenarios.CARDS)
    for key, card in scenarios.CARDS.items():
        live = data.catalog_card(species=card["species"])
        if key in scenarios.HAND_SETS:
            assert live is None, f"{card['species']} is in the live catalog now: drop it from HAND_SETS and add it to LIVE_SPECIES"
        else:
            assert live is not None and card == live, f"{key} is not its live card"
            assert data.to_id(card["species"]) == key
    assert {data.to_id(s) for s in scenarios.LIVE_SPECIES} == set(scenarios.CARDS) - set(scenarios.HAND_SETS)


def test_every_scenario_uses_catalogued_cards_and_legal_moves():
    for sc in scenarios.SCENARIOS + scenarios.HARD:
        for key in [*sc.my_active, *sc.their_active, *sc.my_bench, *sc.mine, *sc.theirs]:
            assert key in scenarios.CARDS, (sc.name, key)
        for key, dropped in sc.drop_moves.items():
            assert dropped <= {data.to_id(m) for m in scenarios.CARDS[key]["moves"]}, (sc.name, key, dropped)
        mine_bring = list(sc.my_active) + [b for b in sc.my_bench if b not in sc.my_active]
        template = scenarios.build_template(sc, mine_bring)
        assert [s["slot"] for s in template["slots"]] == [0, 1]
        obs = scenarios.build_observation(sc, mine_bring)
        alive = [k for k in sc.their_active if sc.their_hp.get(k, 1.0) > 0]
        assert [a["species"] for a in obs["opponent_active_pokemon"]] == [scenarios.CARDS[k]["species"] for k in alive]
        assert all(o["target"] <= len(alive) for s in template["slots"] for opt in s["options"] for o in opt.get("target_options") or [] if o["side"] == "opponent")


def _turns(folder: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for path in folder.glob("moving-*-seat0.json"):
        turn = json.loads(path.read_text())["observation"].get("turn")
        counts[turn] = counts.get(turn, 0) + 1
    return counts


def test_every_live_position_names_a_captured_decision():
    names = [sc.name for sc in live_scenarios.SCENARIOS]
    assert len(names) == len(set(names)), "duplicate scenario names"
    for sc in live_scenarios.SCENARIOS:
        folders = [p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(sc.session)]
        assert len(folders) == 1, sc.name
        turns = _turns(folders[0])
        assert turns.get(sc.turn, 0) > sc.index, f"{sc.name}: no decision #{sc.index} on turn {sc.turn} of {sc.session}"
        assert callable(sc.accept) and sc.accept_text and sc.what
