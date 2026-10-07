"""Tuning parameters reach the code brain and change its decisions; two agents in one process keep their own."""

from __future__ import annotations

import json

import pytest

from agent.agent import PokemonAgent
from agent.pokemon import tuning


def test_merged_rejects_unknown_keys_and_labels_differences():
    params = tuning.merged({"protect_bonus_guaranteed": 70})
    assert params["protect_bonus_guaranteed"] == 70 and params["focus_fire_bonus"] == tuning.DEFAULTS["focus_fire_bonus"]
    assert tuning.label(params) == "tuned:protect_bonus_guaranteed=70"
    assert tuning.label(tuning.merged()) == "tuned:defaults"
    with pytest.raises(KeyError):
        tuning.merged({"not_a_knob": 1})


def test_from_env_reads_json(monkeypatch):
    monkeypatch.setenv("AGENT_TUNE", json.dumps({"draft_offense_w": 14}))
    assert tuning.from_env()["draft_offense_w"] == 14
    monkeypatch.setenv("AGENT_TUNE", "")
    assert tuning.from_env() == tuning.DEFAULTS


def test_two_agents_keep_separate_params():
    a = PokemonAgent(None, params=tuning.merged({"fakeout_base": 1}))
    b = PokemonAgent(None)
    assert a.params["fakeout_base"] == 1 and b.params["fakeout_base"] == tuning.DEFAULTS["fakeout_base"]
    assert tuning.DEFAULTS["fakeout_base"] == 60.0  # defaults untouched


def test_params_change_candidate_ranking():
    from agent.pokemon import battle
    from tests.test_lineup_battle import _obs, _template, memory

    normal = battle.build_sheet(_template(), _obs(), memory())
    # Make Fake Out worthless and Protect enormous: the top candidate must change.
    tuned = battle.build_sheet(_template(), _obs(), memory(), tuning.merged({"fakeout_base": 0, "fakeout_setup_value": 0,
                                                                            "fakeout_threat_bonus": 0, "protect_bonus_guaranteed": 500,
                                                                            "protect_bonus_possible": 500}))
    assert normal.candidates and tuned.candidates
    assert normal.candidates[0]["name"] != tuned.candidates[0]["name"] or normal.candidates[0]["score"] != tuned.candidates[0]["score"]
