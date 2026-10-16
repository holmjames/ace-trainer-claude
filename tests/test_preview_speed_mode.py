"""``preview_speed_mode``: Team Preview's four fits one speed plan (lineup.speed_plan).

Live, Oct 15 (f73663d8): we brought Iron Hands + Hatterene (Trick Room) as leads with Archaludon and Baxcalibur in the
back, into Dragonite (Extreme Speed), Rillaboom (Grassy Surge + Grassy Glide) and Urshifu-Rapid-Strike (Aqua Jet).
Priority KO'd Hatterene on turn 2, and under our own Trick Room our Baxcalibur (faster than four of their six) moved
last and was KO'd before its Icicle Spear. With the key on, a Trick Room four pays for each fast Pokémon the bench
could replace and for priority users beyond the second; the model sees a "speed plan:" line. Key 0 = the preview as it was.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon import lineup, matchup, tuning
from agent.pokemon.memory import observation_dict

LIVE = Path(__file__).parent / "fixtures" / "live"
OFF = tuning.merged()
ON = tuning.merged({"preview_speed_mode": 1.0})


def live_inputs(session: str, tmp_path: Path) -> tuple[dict, dict, list[str]]:
    """(our cards, their cards, roster) at a live Team Preview, with the draft states replayed into memory first."""
    folder = next(LIVE.glob(f"{session}-*"))
    preview = next(folder.glob("team_preview-*.json"))
    seat = preview.name.split("-")[-1]
    agent = PokemonAgent(None, version="speed-mode", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    for path in sorted(folder.glob(f"draft-*-{seat}"), key=lambda p: int(p.name.split("-")[1])):
        raw = json.loads(path.read_text())
        agent.memory.observe_draft(observation_dict(GameState.from_mcp_state(raw)), my_turn=True,
                                   agent_id=(raw.get("current_actor") or {}).get("agent_id"))
    state = GameState.from_mcp_state(json.loads(preview.read_text()))
    agent.memory.observe_team_preview(observation_dict(state))
    roster = list(state.legal_actions[0].input["action"]["roster"])
    return agent.memory.my_cards, agent.memory.opp_cards, roster


def speed_notes(cands: list[lineup.Candidate]) -> list[str]:
    return [n for c in cands for n in c.notes if n.startswith("speed plan")]


def test_the_live_trick_room_four_pays_for_baxcalibur_and_three_priority_users(tmp_path):
    mine, theirs, roster = live_inputs("f73663d8", tmp_path)
    units = [matchup.unit(c) for c in mine.values() if c.get("moves")] + [matchup.unit(c) for c in theirs.values() if c.get("moves")]
    names = [u.species for u in units]
    ours, others = list(range(6)), list(range(6, len(units)))
    k = {n: i for i, n in enumerate(names)}
    assert [s for s, _ in lineup.priority_users(units, others)] == ["Dragonite", "Rillaboom", "Urshifu-Rapid-Strike"]
    bax = tuple(sorted(k[n] for n in ("Archaludon", "Iron Hands", "Baxcalibur", "Hatterene")))
    ttar = tuple(sorted(k[n] for n in ("Archaludon", "Iron Hands", "Tyranitar", "Hatterene")))
    no_tr = tuple(sorted(k[n] for n in ("Archaludon", "Iron Hands", "Tyranitar", "Baxcalibur")))
    pen, note = lineup.speed_plan(units, bax, ours, others, 1.0)
    assert pen == pytest.approx(0.3 + 0.3)  # Baxcalibur (Tyranitar on the bench is not fast) + the priority user past the second
    assert "vs 3 priority users: Dragonite (Extreme Speed), Rillaboom (Grassy Glide), Urshifu-Rapid-Strike (Aqua Jet)" in note
    assert "Baxcalibur outspeeds most of their six" in note
    assert lineup.speed_plan(units, ttar, ours, others, 1.0)[0] == pytest.approx(0.3)  # priority only
    assert lineup.speed_plan(units, no_tr, ours, others, 1.0) == (0.0, None)  # no Trick Room, no Tailwind setter
    assert lineup.speed_plan(units, bax, ours, others, 0.0) == (0.0, None)


def test_the_live_preview_off_brings_baxcalibur_with_hatterene_and_on_does_not(tmp_path):
    mine, theirs, roster = live_inputs("f73663d8", tmp_path)
    off = lineup.shortlist(mine, theirs, roster, params=OFF)
    on = lineup.shortlist(mine, theirs, roster, params=ON)
    # off (v2, no simulation): the live four, Archaludon / Iron Hands / Baxcalibur / Hatterene, and no speed note
    assert set(off[0].bring) == {"archaludon", "ironhands", "baxcalibur", "hatterene"}
    assert not speed_notes(off)
    # on: every Hatterene four pays 0.3 for the priority, the Baxcalibur one 0.3 more; the best four with no speed
    # control (Archaludon / Iron Hands / Tyranitar / Baxcalibur, 45% against 49%) is v2's pick, then the Tailwind
    # four (Iron Hands at double speed still outspeeds three of their six: no charge), then Tyranitar's Trick Room four
    assert set(on[0].bring) == {"archaludon", "ironhands", "tyranitar", "baxcalibur"}
    assert set(on[1].bring) == {"archaludon", "ironhands", "tornadus", "baxcalibur"}
    assert set(on[2].bring) == {"archaludon", "ironhands", "tyranitar", "hatterene"}
    assert [c.bring for c in on if any(n.startswith("speed plan") for n in c.notes)] == [on[2].bring]
    table = lineup.preview_table(mine, theirs, on, params=ON)
    assert "Speed plan: #3 Trick Room plan (Hatterene); vs 3 priority users" in table
    assert "Speed plan" not in lineup.preview_table(mine, theirs, off, params=OFF)


def test_the_live_deep_preview_with_the_key_on_leaves_baxcalibur_out_of_the_trick_room_four(tmp_path):
    """The deployed path (the 20 s simulated preview). Off it brought the live four (sim: lead Iron Hands + Hatterene,
    +1.40 expected); on, every plan of a Hatterene four pays the term in full and Tornadus (Prankster Tailwind, a
    second mode, not a mismatch) replaces Baxcalibur behind the same lead. If the search cannot finish in time,
    v2's lineup stands: with the key on that is the four with no Trick Room at all."""
    mine, theirs, roster = live_inputs("f73663d8", tmp_path)
    on = lineup.shortlist(mine, theirs, roster, params=ON, sim_budget_s=ON["lineup_sim_budget_s"])
    assert not {"baxcalibur", "hatterene"} <= set(on[0].bring)
    if "hatterene" in on[0].bring:
        assert any(n.startswith("speed plan: Trick Room plan (Hatterene); vs 3 priority users") for n in on[0].notes)


def test_a_six_with_no_trick_room_or_tailwind_setter_is_unchanged(tmp_path):
    mine, theirs, roster = live_inputs("bcbaabe2", tmp_path)
    off = lineup.shortlist(mine, theirs, roster, params=OFF)
    on = lineup.shortlist(mine, theirs, roster, params=ON)
    assert [(c.bring, c.leads, c.score, c.notes) for c in on] == [(c.bring, c.leads, c.score, c.notes) for c in off]


def test_key_off_never_runs_the_term_and_is_identical(tmp_path, monkeypatch):
    mine, theirs, roster = live_inputs("f73663d8", tmp_path)
    before = lineup.shortlist(mine, theirs, roster, params=OFF)

    def boom(*args, **kwargs):
        raise AssertionError("speed_plan ran with preview_speed_mode 0")

    monkeypatch.setattr(lineup, "speed_plan", boom)
    after = lineup.shortlist(mine, theirs, roster, params=tuning.merged({"preview_speed_mode": 0}))
    assert [(c.bring, c.leads, c.score, c.notes) for c in after] == [(c.bring, c.leads, c.score, c.notes) for c in before]
    assert OFF["preview_speed_mode"] == 0.0
