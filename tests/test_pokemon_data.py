"""Tests for agent/pokemon/data.py: type chart, stat formula, parsing. No network."""

from __future__ import annotations

import pytest

from agent.pokemon import data


@pytest.mark.parametrize(
    "move_type, defender, expected",
    [
        ("fire", ["grass"], 2.0),
        ("fire", ["water"], 0.5),
        ("electric", ["ground"], 0.0),
        ("electric", ["water", "flying"], 4.0),
        ("fighting", ["ghost"], 0.0),
        ("ice", ["dragon", "flying"], 4.0),
        ("grass", ["fire", "steel"], 0.25),
        ("dragon", ["fairy"], 0.0),
        ("ghost", ["normal"], 0.0),
        ("Fairy", ["DRAGON", "Dark"], 4.0),  # case-insensitive
        ("normal", [], 1.0),
        ("mystery", ["fire"], 1.0),
    ],
)
def test_type_effectiveness(move_type, defender, expected):
    assert data.effectiveness(move_type, defender) == expected


def test_type_chart_is_complete_and_only_uses_real_types():
    assert set(data.TYPE_CHART) == set(data.TYPES)
    for attack, row in data.TYPE_CHART.items():
        assert set(row) <= set(data.TYPES), attack
        assert set(row.values()) <= {0.0, 0.5, 2}, attack


def test_level_50_stat_formula_matches_known_values():
    # Incineroar: base HP 95 / Atk 115, 252 HP / 252 Atk, Adamant.
    assert data.calc_hp(95, 252) == 202
    assert data.calc_stat(115, 252, nature_mult=1.1) == 183
    assert data.calc_stat(115, 252, nature_mult=1.0) == 167
    # Flutter Mane base Spe 135, 252 Spe Timid -> 205.
    assert data.calc_stat(135, 252, nature_mult=1.1) == 205
    assert data.calc_hp(1) == 1  # Shedinja


def test_nature_multipliers():
    assert data.nature_multiplier("Adamant", "atk") == 1.1
    assert data.nature_multiplier("Adamant", "spa") == 0.9
    assert data.nature_multiplier("Adamant", "spe") == 1.0
    assert data.nature_multiplier("Hardy", "atk") == 1.0
    assert data.nature_multiplier(None, "atk") == 1.0


def test_parse_evs_accepts_dict_and_showdown_text():
    assert data.parse_evs({"hp": 252, "Atk": 4, "spe": 252})["atk"] == 4
    parsed = data.parse_evs("252 HP / 4 Def / 252 Spe")
    assert parsed == {"hp": 252, "atk": 0, "def": 4, "spa": 0, "spd": 0, "spe": 252}
    assert data.parse_evs(None)["hp"] == 0


def test_actual_stats_from_a_card_with_base_stats():
    card = {"species": "Incineroar", "nature": "Adamant", "evs": {"hp": 252, "atk": 252, "spd": 4},
            "base_stats": {"hp": 95, "atk": 115, "def": 90, "spa": 80, "spd": 90, "spe": 60}}
    stats = data.actual_stats(card)
    assert stats["hp"] == 202 and stats["atk"] == 183 and stats["spe"] == 80
    assert stats["spa"] == 90  # 0.9 nature on 100


def test_to_id_normalizes_like_the_server():
    assert data.to_id("Urshifu-Rapid-Strike") == "urshifurapidstrike"
    assert data.to_id("Flutter Mane") == "fluttermane"
    assert data.to_id(None) == ""


@pytest.mark.skipif(not (data.DATA_DIR / "pokedex.json").exists(), reason="run scripts/build_dex.py first")
def test_vendored_tables_answer_real_lookups():
    incineroar = data.species_info("Incineroar")
    assert incineroar["types"] == ["Fire", "Dark"] and incineroar["base_stats"]["atk"] == 115
    assert data.resolve_types({"species": "Urshifu-Rapid-Strike"}) == ["fighting", "water"]
    fake_out = data.move_info("Fake Out")
    assert fake_out["priority"] == 3 and fake_out["category"] == "Physical"
    assert data.move_info("Rock Slide")["target"] == "allAdjacentFoes"
    assert data.actual_stats({"species": "Flutter Mane", "nature": "Timid", "evs": "252 SpA / 252 Spe"})["spe"] == 205
