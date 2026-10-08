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
    for path in folder.glob(f"moving-*-seat{live_scenarios.seat_of(folder)}.json"):
        turn = json.loads(path.read_text())["observation"].get("turn")
        counts[turn] = counts.get(turn, 0) + 1
    return counts


def test_every_live_position_names_a_captured_decision():
    names = [sc.name for sc in live_scenarios.SCENARIOS]
    assert len(names) == len(set(names)), "duplicate scenario names"
    for sc in live_scenarios.SCENARIOS:
        folders = [p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(sc.session)]
        assert len(folders) == 1, sc.name
        if sc.phase == "moving":
            turns = _turns(folders[0])
            assert turns.get(sc.turn, 0) > sc.index, f"{sc.name}: no decision #{sc.index} on turn {sc.turn} of {sc.session}"
        else:
            seat = live_scenarios.seat_of(folders[0])
            assert len(list(folders[0].glob(f"{sc.phase}-*-seat{seat}.json"))) > sc.index, f"{sc.name}: no {sc.phase} decision #{sc.index}"
        assert callable(sc.accept) and sc.accept_text and sc.what


def test_the_seat_we_sat_in_comes_from_the_observation():
    by_prefix = {p.name[:8]: p for p in LIVE.iterdir() if p.is_dir()}
    assert live_scenarios.seat_of(by_prefix["bcbaabe2"]) == 0  # player 1
    for prefix in ("46fd3d95", "da9c168e", "d74bfb88"):  # the Oct 8 matches played as player 2
        if prefix in by_prefix:
            assert live_scenarios.seat_of(by_prefix[prefix]) == 1 and not live_scenarios._sequence(by_prefix[prefix], 0)
    assert live_scenarios.our_side({"team": {"p2: Iron Valiant": {}}}) == "p2" and live_scenarios.our_side({}) is None


def test_a_player_two_session_replays_from_seat_one():
    folders = [p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith("da9c168e")]
    if not folders:
        import pytest

        pytest.skip("no player-2 capture present")
    sc = live_scenarios.LiveScenario("p2_smoke", "da9c168e", 1, "the first battle decision of a player-2 match",
                                     lambda p, t: p.get("type") == "doubles_turn" and {"slot_0", "slot_1"} <= set(p), "a doubles turn")
    result = live_scenarios.run(sc, code_only=True)
    assert result["ok"] and result["seconds"] >= 0


# ---- sheet-level checks on the Oct 8 player-2 loss (ace-trainer-data/scenarios/da9c168e.md): the engine's numbers at the
# decisive positions, converted to HP with the sheet's own Mon stats (no hand arithmetic).
import tempfile

import pytest

from agent.agent import PokemonAgent
from agent.pokemon import battle
from agent.pokemon.memory import species_key
from altruagent import DecisionContext
from altruagent.models import GameState


@pytest.fixture(scope="module")
def p2_loss():
    """Replay da9c168e from seat 1 once: the agent (its memory holds the cards), the decision-log rows per capture
    file, and the raw captures."""
    folders = [p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith("da9c168e")]
    if not folders:
        pytest.skip("no player-2 capture present")
    folder = folders[0]
    seat = live_scenarios.seat_of(folder)
    rows_by_file: dict[str, dict] = {}
    raws: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        agent = PokemonAgent(None, version="sheet-check", log_dir=tmp, capture_dir="", log=lambda *_: None)
        for path in live_scenarios._sequence(folder, seat):
            raw = json.loads(path.read_text())
            state = GameState.from_mcp_state(raw)
            if not state.legal_actions:
                continue
            ctx = DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                  agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat)
            agent.choose_action(state, ctx)
            rows = [json.loads(line) for f in Path(tmp).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
            rows_by_file[path.name] = rows[-1]
            raws[path.name] = raw
    return agent, rows_by_file, raws


def _mon(agent, raw: dict, species: str, *, side: str, position: int | None, boosts: dict | None = None):
    obs = raw["observation"]
    team = obs["team"] if side == "mine" else obs["opponent_team"]
    summary = next(v for k, v in team.items() if species_key(v.get("species") or k.split(":")[-1]) == species)
    cards = agent.memory.my_cards if side == "mine" else agent.memory.opp_cards
    if boosts:
        summary = {**summary, "boosts": {**(summary.get("boosts") or {}), **boosts}}
    mon = battle.build_mon(summary, cards.get(species), side=side, position=position)
    assert mon is not None, species
    return mon


def _hp_range(pct: list[float], mon) -> tuple[float, float]:
    current = mon.stats["hp"] * mon.hp_fraction
    return pct[0] * current / 100, pct[1] * current / 100


def _option(row: dict, slot: int, move: str, target: str) -> dict:
    for entry in row["turn_sheet"]["our_damage_estimates"]:
        if entry["slot"] == slot:
            for o in entry["options"]:
                if o["move"] == move:
                    return next(t for t in o["targets"] if t["species"] == target)
    raise AssertionError(f"slot {slot} {move} -> {target} not in the sheet")


def _threat(row: dict, attacker: str, move: str, target: str) -> dict:
    for th in row["turn_sheet"]["opponent_threats"]:
        if th["species"] == attacker:
            return next(h for h in th["hits"] if h["move"] == move and h["species"] == target)
    raise AssertionError(f"{attacker} {move} -> {target} not in the sheet")


def test_p2_pick_one_is_not_iron_valiant_and_the_reply_set_names_incineroar(p2_loss):
    _agent, rows, _raws = p2_loss
    row = rows["draft-00000-seat1.json"]
    assert row["pick_number"] == 1 and row["card_id"] != "vgc-iron-valiant"
    notes = row.get("notes") or []
    assert any("likely replies" in n and "Incineroar" in n for n in notes), notes
    assert any(n.startswith("loses to") for n in notes), notes


def test_p2_team_preview_predicts_their_four_with_lucario(p2_loss):
    _agent, rows, _raws = p2_loss
    row = rows["team_preview-00000-seat1.json"]
    top = row["candidates"][0]
    assert any("likely fours" in n and "Lucario" in n for n in top["notes"]), top["notes"]


def test_p2_t3_switch_preview_applies_the_entry_intimidate(p2_loss):
    """moving-00003: the preview for Kingambit -> Landorus-T carries Landorus's Intimidate, and the calc at -1 Atk puts
    Incineroar's Fake Out into the 57-HP Thundurus at 16-19 HP (the sheet's own pre-switch figure is 23-28 HP)."""
    agent, rows, raws = p2_loss
    row, raw = rows["moving-00003-seat1.json"], raws["moving-00003-seat1.json"]
    preview = row["turn_sheet"]["switch_ins"]["landorustherian"]
    assert any(e.startswith("incineroar: -1") for e in preview.get("intimidate") or []), preview.get("intimidate")
    thundurus = _mon(agent, raw, "thundurus", side="mine", position=0)
    assert round(thundurus.stats["hp"] * thundurus.hp_fraction) == 57
    before = _hp_range(_threat(row, "incineroar", "fakeout", "thundurus")["damage_pct_of_current_hp"], thundurus)
    assert 23 <= before[0] <= 24 and 27 <= before[1] <= 28.5, before
    incineroar = _mon(agent, raw, "incineroar", side="theirs", position=1, boosts={"atk": -1})
    result = battle.damage_percent(incineroar, data.move_info("fakeout"), thundurus)
    assert result is not None
    after = _hp_range(list(result[:2]), thundurus)
    assert after[0] <= 16.5 and after[1] >= 18.5 and after[1] < before[0], (after, before)


@pytest.mark.xfail(strict=False, reason="ops lane (memory): the logged move must override the HP-inferred 'probably Protected' note")
def test_p2_t3_replacement_has_no_inferred_protect_note(p2_loss):
    _agent, rows, _raws = p2_loss
    notes = rows["moving-00004-seat1.json"]["turn_sheet"].get("notes") or []
    assert not any("probably Protected" in n for n in notes), notes


def test_p2_t4_ranges_include_the_exact_engine_rolls(p2_loss):
    """moving-00005: Kowtow Cleave into Gyarados reaches 145 HP (Supreme Overlord, one fallen, x1.1; the old sheet topped
    out at 142.2) and Gyarados's Waterfall into Landorus-T reaches down to 108 HP (stats rounded down after the drop;
    the old sheet's minimum was 109.7)."""
    agent, rows, raws = p2_loss
    row, raw = rows["moving-00005-seat1.json"], raws["moving-00005-seat1.json"]
    gyarados = _mon(agent, raw, "gyarados", side="theirs", position=2)
    cleave = _hp_range(_option(row, 0, "kowtowcleave", "gyarados")["damage_pct_of_current_hp"], gyarados)
    assert cleave[0] <= 145 <= cleave[1] + 0.5, cleave
    landorus = _mon(agent, raw, "landorustherian", side="mine", position=1)
    assert landorus.stats["hp"] == 164
    waterfall = _hp_range(_threat(row, "gyarados", "waterfall", "landorustherian")["damage_pct_of_current_hp"], landorus)
    assert waterfall[0] - 0.5 <= 108 <= waterfall[1], waterfall
