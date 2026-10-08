"""Tests for the matchup model (agent/pokemon/matchup.py): the damage layer against @smogon/calc on the live
catalog, the 1-on-1 races on known matchups, and the team score's basic contracts."""

from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from pathlib import Path

import pytest

from agent.pokemon import battle, data, matchup

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smogon_calc_matchup.json"
CATALOG = {c["card_id"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")


def u(species: str) -> matchup.Unit:
    return matchup.unit(next(c for c in CATALOG.values() if c["species"] == species))


def move_of(unit: matchup.Unit, name: str) -> dict:
    return next(m for m in unit.mon.moves if m["id"] == data.to_id(name))


def hit(attacker: str, move: str, defender: str) -> tuple[float, float]:
    a, d = u(attacker), u(defender)
    fs = battle.FieldState(actives=[a.mon, d.mon])
    return matchup.calc_hit(a.mon, move_of(a, move), d.mon, fs, spread=move_of(a, move).get("target") in matchup.SPREAD_TARGETS) or (0.0, 0.0)


# -- the damage layer vs @smogon/calc -------------------------------------------------------------------------------


def _errors() -> list[tuple[float, dict]]:
    out = []
    for case in json.loads(FIXTURE.read_text()):
        a, d = matchup.unit(CATALOG[case["a"]]), matchup.unit(CATALOG[case["d"]])
        if d.item == "focussash":
            continue  # we cap a single hit at 99% through a full-HP sash on purpose; the calculator does not
        attacker = a.mon
        if a.ability == "download":  # the calculator applies Download on its own; our race does it on entry
            attacker = replace(a.mon, boosts={k: v for k, v in matchup.entry_stages(a, d).items() if k in ("atk", "spa")})
        move = move_of(a, case["move"])
        est = matchup.calc_hit(attacker, move, d.mon, battle.FieldState(actives=[attacker, d.mon]), spread=case["spread"]) or (0.0, 0.0)
        max_hp, ref = case["max_hp"], case["damage"]
        ours = (min(est[0] * max_hp / 100, max_hp), min(est[1] * max_hp / 100, max_hp))
        ref = (min(ref[0], max_hp), min(ref[1], max_hp))
        out.append((max(abs(ours[0] - ref[0]), abs(ours[1] - ref[1])) / max_hp * 100, case))
    return out


@pytest.mark.skipif(not FIXTURE.exists(), reason="run node scripts/gen_matchup_fixture.js")
def test_damage_layer_tracks_the_official_calculator_on_the_live_catalog():
    errors = _errors()
    assert len(errors) > 500
    values = sorted(e for e, _ in errors)
    median, p90, worst = values[len(values) // 2], values[int(len(values) * 0.9)], values[-1]
    assert median < 1.0 and p90 < 2.5 and worst < 6.0, (median, p90, worst, sorted(errors, key=lambda t: -t[0])[:3])
    forced = [e for e, case in errors if case["forced"]]
    assert len(forced) > 100 and max(forced) < 6.0  # Pixilate, Adaptability, Levitate, Air Balloon, Low Kick


def test_effects_the_shared_calculator_misses():
    # Pixilate turns Hyper Voice into a Fairy move: it hits Ghosts and is 2x on Dragon/Fighting/Dark.
    assert hit("Sylveon", "Hyper Voice", "Gholdengo")[0] > 0
    assert hit("Sylveon", "Hyper Voice", "Iron Hands")[0] > hit("Sylveon", "Hyper Voice", "Incineroar")[0]
    # Levitate and Air Balloon: Ground moves do nothing.
    assert hit("Garchomp", "Earthquake", "Cresselia") == (0.0, 0.0)
    assert hit("Landorus-Therian", "Earthquake", "Gholdengo") == (0.0, 0.0)
    # Low Kick's power comes from the target's weight (the shared table says 0).
    assert hit("Tyranitar", "Low Kick", "Kingambit")[0] > 50
    # Freeze-Dry is super-effective on Water types.
    assert hit("Iron Bundle", "Freeze-Dry", "Dondozo")[0] > 50
    # Chople Berry halves the first super-effective Fighting hit.
    plain_tyranitar = replace(u("Tyranitar").mon, item=None)
    archaludon = u("Archaludon")
    body_press = move_of(archaludon, "Body Press")
    with_berry = hit("Archaludon", "Body Press", "Tyranitar")
    without = matchup.calc_hit(archaludon.mon, body_press, plain_tyranitar, battle.FieldState(actives=[archaludon.mon, plain_tyranitar]), spread=False)
    assert with_berry[1] == pytest.approx(without[1] / 2, rel=0.02)


# -- 1-on-1 races ---------------------------------------------------------------------------------------------------


def test_races_are_antisymmetric():
    units = [matchup.unit(c) for c in CATALOG.values()]
    rng = random.Random(3)
    for _ in range(60):
        a, b = rng.sample(units, 2)
        ab, ba = matchup.duel(a, b), matchup.duel(b, a)
        assert ab.margin == pytest.approx(-ba.margin)
        assert ab.margin_tw_a == pytest.approx(-ba.margin_tw_b)
        assert ab.margin_tr == pytest.approx(-ba.margin_tr)
        assert -1.0 <= ab.margin <= 1.0


def test_known_matchups():
    # Sucker Punch (priority) knocks out the faster Flutter Mane before it moves.
    d = matchup.duel(u("Kingambit"), u("Flutter Mane"))
    assert d.margin > 0.5 and d.ohko_a and d.a_move == "suckerpunch"
    # Booster Energy Iron Bundle outspeeds Choice Scarf Landorus and Freeze-Dry is 4x.
    assert matchup.duel(u("Iron Bundle"), u("Landorus-Therian")).ohko_a
    # Air Balloon Gholdengo: Landorus has to pop the balloon before Earthquake works, and still wins the race...
    plain = matchup.duel(u("Gholdengo"), u("Landorus-Therian"))
    assert plain.margin < 0
    # ...unless Gholdengo's side has Tailwind: then it moves first and wins.
    assert plain.margin_tw_a > 0.5
    # Lucario has nothing that touches Gholdengo hard (Close Combat and Poison Jab are immune).
    assert matchup.duel(u("Lucario"), u("Gholdengo")).margin < -0.5


def test_beats_needs_a_cushion():
    assert matchup.beats(u("Iron Hands"), u("Tornadus"))
    assert not matchup.beats(u("Tornadus"), u("Iron Hands"))


def test_unknown_species_is_inert():
    ghost = matchup.unit({"species": "Totally Made Up", "moves": ["Tackle"], "item": "Leftovers", "ability": "Pressure"})
    assert matchup.duel(ghost, u("Garchomp")).margin == 0.0


# -- the team score -----------------------------------------------------------------------------------------------


def test_board_scores_are_antisymmetric_and_fast():
    cards = list(CATALOG.values())[:18]
    started = time.perf_counter()
    board = matchup.Board([matchup.unit(c) for c in cards])
    built = time.perf_counter() - started
    assert built < 2.0
    us, them = [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]
    assert board.score(us, them) == pytest.approx(-board.score(them, us))
    assert len(board.features(us, them)) == len(matchup.FEATURES)
    assert board.score([], []) == 0.0
    started = time.perf_counter()
    for _ in range(200):
        board.score(us, them)
    assert (time.perf_counter() - started) / 200 < 0.005


def test_answer_feature_rewards_holding_a_counter():
    # Their Gholdengo: our team with Garchomp (Earthquake once the balloon pops) answers it better than Lucario.
    board = matchup.Board([u("Gholdengo"), u("Garchomp"), u("Lucario"), u("Kingambit")])
    with_answer = board.features([1, 3], [0])[matchup.FEATURES.index("answers")]
    without = board.features([2, 3], [0])[matchup.FEATURES.index("answers")]
    assert with_answer >= without


def test_model_file_loads_or_defaults():
    model = matchup.load_model()
    assert set(model["weights"]) >= set(matchup.MATCHUP_FEATURES)
    assert 0.0 < matchup.win_probability(0.5) < 1.0 and matchup.win_probability(0.0) == 0.5


# -- the opening (turns 1-2 of two leads against two) -------------------------------------------------------------


def lead(*species: str) -> list[matchup.Unit]:
    return [u(s) for s in species]


def test_openings_are_antisymmetric():
    units = [matchup.unit(c) for c in CATALOG.values()]
    rng = random.Random(5)
    for _ in range(25):
        a, b, c, d = rng.sample(units, 4)
        assert matchup.opening([a, b], [c, d]).margin == pytest.approx(-matchup.opening([c, d], [a, b]).margin)


def test_opening_sees_the_lead_that_gets_destroyed():
    # Live match 1aef9227: Iron Hands into Fake Out + Pixilate Hyper Voice. Sylveon's Hyper Voice is a Fairy move now.
    result = matchup.opening(lead("Iron Hands", "Archaludon"), lead("Incineroar", "Sylveon"))
    assert "Iron Hands" in result.ko_against
    # Tornadus + Iron Hands into Iron Hands + Sylveon loses both leads (match 7ce71025's opponent leads).
    assert set(matchup.opening(lead("Tornadus", "Iron Hands"), lead("Iron Hands", "Sylveon")).ko_against) == {"Tornadus", "Iron Hands"}


def test_fake_out_and_its_immunities_change_the_opening():
    incineroar = next(c for c in CATALOG.values() if c["species"] == "Incineroar")
    no_fake_out = matchup.unit({**incineroar, "moves": [m for m in incineroar["moves"] if m != "Fake Out"] + ["Protect"]})
    with_it = matchup.opening(lead("Incineroar", "Kingambit"), lead("Flutter Mane", "Chien-Pao")).margin
    without = matchup.opening([no_fake_out, u("Kingambit")], lead("Flutter Mane", "Chien-Pao")).margin
    assert with_it >= without
    sim = matchup._Opening(lead("Indeedee-F", "Iron Hands"), lead("Incineroar", "Rillaboom"))
    assert sim.terrain == "psychic"
    assert not sim.flinchable(sim.mons[1])  # grounded in Psychic Terrain: Fake Out cannot touch it
    ghost = matchup._Opening(lead("Gholdengo", "Iron Hands"), lead("Incineroar", "Rillaboom"))
    assert not ghost.flinchable(ghost.mons[0])  # Fake Out is a Normal move


def test_prankster_tailwind_helps_the_partner_the_same_turn():
    tornadus = next(c for c in CATALOG.values() if c["species"] == "Tornadus")
    slow_setter = matchup.unit({**tornadus, "ability": "Defiant"})
    prankster = matchup.opening(lead("Tornadus", "Great Tusk"), lead("Flutter Mane", "Sylveon")).margin
    plain = matchup.opening([slow_setter, u("Great Tusk")], lead("Flutter Mane", "Sylveon")).margin
    assert prankster >= plain


# -- the tier list (data/card_tiers.json, scripts/build_tiers.py) --------------------------------------------------


TIERS_FILE = data.DATA_DIR / "card_tiers.json"


@pytest.mark.skipif(not TIERS_FILE.exists(), reason="run scripts/build_tiers.py")
def test_tier_list_is_consistent_and_covers_the_catalog():
    blob = json.loads(TIERS_FILE.read_text())
    entries = blob["cards"]
    assert set(entries) == set(CATALOG)
    floors = [("S", 40), ("A", 12), ("B", -2), ("C", -16)]
    for card_id, e in entries.items():
        assert e["fingerprint"] == matchup.fingerprint(CATALOG[card_id])
        assert e["rating"] == pytest.approx(e["sim_rating"] + e["adjust"], abs=0.11)
        assert e["sim_rating"] == pytest.approx(e["matchup_part"] + e["card_part"], abs=0.11)
        assert e["tier"] == next((name for name, floor in floors if e["rating"] >= floor), "D")
        assert e["why"]


@pytest.mark.skipif(not TIERS_FILE.exists(), reason="run scripts/build_tiers.py")
def test_the_drafter_base_value_is_the_tier_rating():
    incineroar = next(c for c in CATALOG.values() if c["species"] == "Incineroar")
    entry = matchup.load_tiers()[matchup.fingerprint(incineroar)]
    units = [u("Incineroar"), u("Garchomp")]
    plain = matchup.Board(units, tier_adjust=0.0)
    adjusted = matchup.Board(units, tier_adjust=1.0)
    shift = adjusted.resid[0] - plain.resid[0]
    assert shift == pytest.approx((entry["rating"] - entry["sim_rating"]) / 100, abs=1e-9)
    # Editing a rating moves the drafter by exactly that much: a unit of rating is 1/100 log-odds per card.
    edited = {**matchup.load_tiers(), entry["fingerprint"]: {**entry, "rating": entry["rating"] + 10}}
    assert matchup.Board(units, tiers=edited).resid[0] == pytest.approx(adjusted.resid[0] + 0.10)


# -- the shared helpers (Oct 8: data.weight_power / data.hit_chance replace matchup's private copies) ----------------


def test_openings_and_duels_are_unchanged_for_moves_without_weight_or_weather():
    # Pinned from the Oct 8 board before the switch; none of these sets carries Low Kick / Grass Knot / Heavy Slam /
    # Heat Crash (only Tyranitar does in the catalog) or sets weather.
    def lead(a, b):
        return [u(a), u(b)]

    assert matchup.opening(lead("Archaludon", "Arcanine"), lead("Dragonite", "Salamence")).margin == pytest.approx(0.7647, abs=1e-4)
    assert matchup.opening(lead("Iron Hands", "Whimsicott"), lead("Incineroar", "Sylveon")).margin == pytest.approx(-0.6242, abs=1e-4)
    assert matchup.opening(lead("Great Tusk", "Tornadus"), lead("Iron Hands", "Sylveon")).margin == pytest.approx(-1.3344, abs=1e-4)
    assert matchup.opening(lead("Hatterene", "Indeedee-F"), lead("Incineroar", "Sylveon")).margin == pytest.approx(-1.0961, abs=1e-4)
    for a, b, margin, tw_a, tr in (("Incineroar", "Gholdengo", 0.5673, 0.7313, 0.7313), ("Sylveon", "Iron Hands", 0.3110, 0.3110, -0.2887),
                                   ("Dragonite", "Archaludon", -0.7576, -0.7576, -0.8293), ("Flutter Mane", "Iron Valiant", 1.0, 1.0, 0.4920),
                                   ("Talonflame", "Rillaboom", 0.2729, 0.2729, 0.1655)):
        d = matchup.duel(u(a), u(b))
        assert (d.margin, d.margin_tw_a, d.margin_tr) == pytest.approx((margin, tw_a, tr), abs=1e-4), (a, b)
    # The helpers are the mech lane's: a Low Kick into Ting-Lu reads its weight band, Hurricane lands 70% outside rain.
    tyranitar = u("Tyranitar")
    assert matchup.weight_power(u("Ting-Lu").mon, move_of(tyranitar, "Low Kick"), tyranitar.mon) == data.weight_power("lowkick", "Ting-Lu")
    assert matchup.accuracy({"id": "hurricane", "accuracy": 70}, None) == pytest.approx(0.7)
    assert matchup.accuracy({"id": "hurricane", "accuracy": 70}, "rain") == 1.0
