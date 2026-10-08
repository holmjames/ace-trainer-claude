"""Tests for the the search bot-style counter-picker (agent/versions/counter_picker.py) and the paired evaluation
helpers in sim/draft_eval.py."""

from __future__ import annotations

import json

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.pokemon import data, matchup
from agent.versions.counter_picker import CounterPickerAgent, counter_beats, counter_score, tool_works

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")


def u(species: str) -> matchup.Unit:
    return matchup.unit(CATALOG[species])


def best(pool: list[str], ours: list[str]) -> tuple[float, str]:
    return max(counter_score(CATALOG[s], [CATALOG[o] for o in ours]) for s in pool if s not in ours)


def test_reproduces_the search bots_live_reasons():
    # 7ce71025: we opened with Tornadus; the search bot answered "Incineroar: Fake Out pressure, Intimidate, beats
    # Tornadus." then "Iron Hands: Fake Out pressure, beats Tornadus."
    pool = ["Amoonguss", "Archaludon", "Dondozo", "Garchomp", "Great Tusk", "Grimmsnarl", "Hatterene", "Incineroar",
            "Indeedee-F", "Iron Hands", "Rillaboom", "Sableye", "Sylveon", "Ting-Lu", "Tornadus", "Ursaluna", "Whimsicott", "Wo-Chien"]
    ranked = sorted((counter_score(CATALOG[s], [CATALOG["Tornadus"]]) for s in pool if s != "Tornadus"), reverse=True)
    assert ranked[0][1] == "Incineroar: Fake Out pressure, Intimidate, beats Tornadus."
    assert ranked[1][1] == "Iron Hands: Fake Out pressure, beats Tornadus."


def test_first_picks_are_role_picks():
    # 1aef9227 and d9405f02: the search bot drafted first and opened with Incineroar, then Dragonite.
    pool_1aef = ["Archaludon", "Chien-Pao", "Cresselia", "Dondozo", "Flutter Mane", "Great Tusk", "Gyarados", "Hatterene", "Indeedee-F",
                 "Iron Bundle", "Iron Hands", "Kingambit", "Klefki", "Rillaboom", "Sylveon", "Whimsicott", "Wo-Chien", "Incineroar"]
    assert best(pool_1aef, [])[1].startswith("Incineroar:")
    pool_d940 = ["Amoonguss", "Archaludon", "Baxcalibur", "Cresselia", "Garchomp", "Gholdengo", "Indeedee-F", "Iron Bundle", "Iron Hands",
                 "Klefki", "Landorus-Therian", "Lucario", "Rillaboom", "Sneasler", "Thundurus", "Ting-Lu", "Wo-Chien", "Dragonite"]
    assert best(pool_d940, [])[1] == "Dragonite: Tailwind."


def test_beats_notion():
    assert counter_beats(u("Iron Hands"), u("Tornadus")) is not None  # wins the race
    assert counter_beats(u("Incineroar"), u("Landorus-Therian")) == 0.3  # Intimidate vs a physical attacker
    assert counter_beats(u("Incineroar"), u("Kingambit")) is None or matchup.duel(u("Incineroar"), u("Kingambit")).margin >= 0.2
    # Shutdown tools respect their immunities: Mental Herb shrugs off Taunt; Good as Gold blocks status moves.
    assert not tool_works("taunt", u("Gyarados"), u("Tornadus"))
    assert not tool_works("spore", u("Amoonguss"), u("Gholdengo"))
    assert tool_works("fakeout", u("Incineroar"), u("Tornadus"))
    assert not tool_works("fakeout", u("Incineroar"), u("Gholdengo"))  # a Ghost


def test_agent_drafts_a_legal_card_with_a_public_reason():
    pool = [CATALOG[s] for s in ("Tornadus", "Incineroar", "Iron Hands", "Garchomp", "Amoonguss")]
    rosters = {"me": [], "you": [{"card_id": CATALOG["Tornadus"]["card_id"], "species": "Tornadus"}]}
    observation = {"phase": "draft", "current_seat": "me", "first_drafter": "you", "rosters": rosters,
                   "available_cards": pool[1:], "available_card_ids": [c["card_id"] for c in pool[1:]], "picks": []}
    actions = [{"action_id": f"draft_pick:{c['card_id']}", "label": c["species"],
                "input": {"action": {"type": "draft_pick", "card_id": c["card_id"]}}} for c in pool[1:]]
    state = GameState.from_mcp_state({"session_id": "t", "game_type": "pokemon_vgc_doubles_draft", "status": "in_progress",
                                      "state_version": 1, "phase": "draft", "observation": observation, "is_terminal": False,
                                      "is_current_actor": True, "legal_actions": {"session_id": "t", "state_version": 1, "actions": actions}})
    agent = CounterPickerAgent()
    agent._log_dir = "/dev/null"
    agent._print = lambda _line: None
    decision = agent.choose_action(state, DecisionContext(session_id="t", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                                         agent_id="me", seat_position=1))
    assert decision.action.action_id == f"draft_pick:{CATALOG['Incineroar']['card_id']}"
    assert decision.reasoning_summary.startswith("Incineroar: Fake Out pressure")


def test_eval_helpers():
    import sim.draft_eval as ev

    p, lo, hi = ev.wilson(60, 100)
    assert p == 0.6 and lo < 0.6 < hi and 0.49 < lo < 0.51
    assert ev.wilson(0, 0) == (0.0, 0.0, 0.0)
    name, overrides = ev.parse_spec('v2{"draft_lookahead": 0}')
    assert name == "v2" and overrides["draft_v2"] == 1 and overrides["draft_lookahead"] == 0
    with pytest.raises(SystemExit):
        ev.parse_spec("nonsense")


def test_forced_player_drafts_its_roster_and_fills_memory():
    import sim.draft_eval as ev
    from sim import translate

    from agent.agent import PokemonAgent
    from agent.pokemon import tuning

    ours = [CATALOG[s] for s in ("Tornadus", "Garchomp", "Great Tusk", "Hatterene", "Ursaluna", "Indeedee-F")]
    theirs = [CATALOG[s] for s in ("Incineroar", "Iron Hands", "Whimsicott", "Archaludon", "Sylveon", "Amoonguss")]
    inner = PokemonAgent(None, params=tuning.merged({}), log=lambda _line: None)
    inner._log_dir = "/dev/null"
    player = ev._Forced(inner, ours, ["Tornadus", "Garchomp", "Ursaluna", "Indeedee-F"])
    ctx = DecisionContext(session_id="t", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="p1", seat_position=0)
    state = translate.draft_state("t", me="p1", them="p2", pool=ours + theirs, rosters={"p1": [], "p2": []}, picks=[],
                                  first_drafter="p1", current_seat="p1", version=0)
    pick = player.choose_action(state, ctx)
    assert pick.action_id.split(":", 1)[1] in {c["card_id"] for c in ours}
    assert len(inner.memory.pool_cards) == 12 and len(inner.memory.my_cards) == 1
    from sim import harness

    bridge = harness.Bridge()  # Team Preview states come from a real Showdown request now (live format)
    try:
        reply = bridge.call(cmd="new", id="t", format="gen9vgc2025regi", seed=[1, 2, 3, 4],
                            p1={"name": "A-p0", "team": translate.team_text(ours)}, p2={"name": "B-p1", "team": translate.team_text(theirs)})
        preview = translate.preview_state("t", reply["state"], "p1", version=0, agent_id="p1")
    finally:
        bridge.close()
    answer = player.choose_action(preview, ctx)
    assert answer == {"type": "select_lineup", "bring": ["tornadus", "garchomp", "ursaluna", "indeedeef"], "leads": ["tornadus", "garchomp"]}
    assert len(inner.memory.opp_cards) == 6  # the battle sheet will know their sets


def test_live_pools_and_preview_choices_are_well_formed():
    import sim.draft_eval as ev
    from sim import harness

    from agent.pokemon import tuning

    species = {c["species"] for c in harness.CARDS}
    for first, pool in ev.LIVE_POOLS.values():
        assert first in ("us", "them") and len(set(pool)) == 18 and set(pool) <= species
    by = {c["species"]: c for c in harness.CARDS}
    spec = ev.LIVE_PREVIEWS["7ce71025"]
    choice = ev.preview_choices([by[s] for s in spec["ours"]], [by[s] for s in spec["theirs"]], tuning.merged({"lineup_v2": 1}))
    assert len(choice) == 4 and set(choice) <= set(spec["ours"])
