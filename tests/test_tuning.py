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
    assert tuning.DEFAULTS["fakeout_base"] == 90.0  # defaults untouched


def test_params_change_candidate_ranking(monkeypatch):
    monkeypatch.setitem(tuning.DEFAULTS, "reply_mixture", 0.0)  # core-5 pin: the default scorer's KO bonus
    from agent.pokemon import battle
    from tests.test_lineup_battle import _obs, _template, memory

    normal = battle.build_sheet(_template(), _obs(), memory())
    # Wood Hammer into the half-HP Urshifu is a guaranteed KO, so the KO bonus feeds the top candidate's score.
    tuned = battle.build_sheet(_template(), _obs(), memory(), tuning.merged({"ko_bonus_guaranteed": 250}))
    assert normal.candidates and tuned.candidates
    assert tuned.candidates[0]["score"] > normal.candidates[0]["score"] + 100


def test_the_deploy_4_defaults_are_pinned():
    """Deploy #4 (cand/1010d) turns on recheck_v2 and prompt_v2 (the Oct 10 model-on test on 121 live positions),
    prompt_kb (the same test, on top of those two) and draft_support_w 0.25 (pre-registration #2, confirmed on fresh
    simulated games); answer_rules_v2, the reply mixture, the exposure tie-break and the deep Team Preview budget stay
    as deploy #3 (81c4591) shipped them.
    Read from the file as shipped too: tests/conftest.py pins lineup_sim_budget_s to 0 outside the preview tests."""
    import importlib.util

    pinned = {"recheck_v2": 1.0, "prompt_v2": 1.0, "draft_support_w": 0.25, "prompt_kb": 1.0, "answer_rules_v2": 0.0,
              "reply_mixture": 0.0, "draft_tiebreak_exposure": 0.0, "lineup_sim_budget_s": 20.0}
    spec = importlib.util.spec_from_file_location("tuning_as_shipped", tuning.__file__)
    shipped = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shipped)
    assert {key: shipped.DEFAULTS[key] for key in pinned} == pinned
    assert {key: tuning.DEFAULTS[key] for key in pinned if key != "lineup_sim_budget_s"} == {
        key: value for key, value in pinned.items() if key != "lineup_sim_budget_s"}
    assert tuning.label(tuning.merged()) == "tuned:defaults" and shipped.DEFAULTS.keys() == tuning.DEFAULTS.keys()
