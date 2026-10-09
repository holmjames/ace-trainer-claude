"""Memory items of Workstream D (ops lane): D1g Protect-streak fixes, D1d fallen allies, entry hazards, and the
opponent's draft profile (A4/B5). Live cases come from the Oct 8 Testing match bcbaabe2 (we were seat 0, p1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState

from agent.agent import PokemonAgent
from agent.pokemon.memory import MatchMemory, observation_dict, species_key

LIVE = Path(__file__).parent / "fixtures" / "live"
MATCH = "bcbaabe2"


def _folder(prefix: str) -> Path:
    hits = [p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix)] if LIVE.exists() else []
    if len(hits) != 1:
        pytest.skip(f"live capture {prefix} not present")
    return hits[0]


def _obs(prefix: str, name: str) -> dict:
    return observation_dict(GameState.from_mcp_state(json.loads((_folder(prefix) / name).read_text())))


def _log(*lines: str, team_side: str = "p1") -> dict:
    """A battle observation carrying only a protocol log (and the team key that tells memory which side is ours)."""
    return {"turn": 0, "team": {f"{team_side}: Someone": {"species": "someone"}}, "protocol_log": list(lines)}


# -- D1g: Protect streaks ------------------------------------------------------------------------------------------


def test_a_failed_protect_resets_the_opponents_streak_and_the_protected_last_turn_flag():
    first = _log("|switch|p2a: Dragonite|Dragonite, L50, F|100/100", "|turn|1", "|move|p2a: Dragonite|Protect|p2a: Dragonite",
                 "|-singleturn|p2a: Dragonite|Protect", "|turn|2")
    memory = MatchMemory()
    memory.observe_battle({**first, "turn": 2})
    assert memory.opp_protect_streak == {"dragonite": 1} and memory.opp_protected_last_turn == ["dragonite"]
    # Turn 2: it tries again and fails ("[still]" + "-fail"). Showdown drops its counter: the next Protect is guaranteed.
    second = _log(*first["protocol_log"], "|move|p2a: Dragonite|Protect||[still]", "|-fail|p2a: Dragonite", "|turn|3")
    memory.observe_battle({**second, "turn": 3})
    assert memory.opp_protect_streak == {} and memory.opp_protected_last_turn == []
    assert memory.logged_streak("theirs", "dragonite") == 0
    # Turn 3: a fresh Protect works again -> streak 1, not 3.
    third = _log(*second["protocol_log"], "|move|p2a: Dragonite|Protect|p2a: Dragonite", "|-singleturn|p2a: Dragonite|Protect", "|turn|4")
    memory.observe_battle({**third, "turn": 4})
    assert memory.opp_protect_streak == {"dragonite": 1}


def test_our_own_streak_comes_from_the_log_and_resets_on_a_failed_protect():
    lines = ["|switch|p1a: Cresselia|Cresselia, L50, F|100/100", "|turn|1", "|move|p1a: Cresselia|Protect|p1a: Cresselia",
             "|-singleturn|p1a: Cresselia|Protect", "|turn|2", "|move|p1a: Cresselia|Protect|p1a: Cresselia",
             "|-singleturn|p1a: Cresselia|Protect", "|turn|3"]
    memory = MatchMemory()
    memory.observe_battle({**_log(*lines), "turn": 3, "active_pokemon": [{"species": "cresselia"}, None]})
    assert memory.protect_streak("cresselia", 0) == 2
    memory.observe_battle({**_log(*lines, "|move|p1a: Cresselia|Protect||[still]", "|-fail|p1a: Cresselia", "|turn|4"),
                           "turn": 4, "active_pokemon": [{"species": "cresselia"}, None]})
    assert memory.protect_streak("cresselia", 0) == 0


def test_slot_one_keeps_its_streak_when_slot_zero_is_empty_without_a_log():
    """Our last Pokémon standing in slot 1: the actives list used to be compacted, so actives[1] did not exist and
    every streak and last-turn flag read 0 (a second Protect in a row was recommended as fresh)."""
    memory = MatchMemory()
    memory.observe_battle({"turn": 7, "active_pokemon": [None, {"species": "cresselia"}]})
    memory.record_turn(turn=7, payload={"type": "doubles_turn", "slot_0": {"type": "pass"}, "slot_1": {"type": "move", "move_id": "protect"}})
    memory.observe_battle({"turn": 8, "active_pokemon": [None, {"species": "cresselia"}]})
    assert memory.last_active == [None, "cresselia"]
    assert memory.protect_streak("cresselia", 1) == 1
    assert memory.our_protect_last_turn("cresselia", 1) is True
    assert memory.protect_streak("cresselia", 0) == 0


def test_hp_inference_needs_a_damaging_move_from_us_without_a_log():
    memory = MatchMemory()
    memory.opp_cards["amoonguss"] = {"card_id": "x", "species": "Amoonguss", "moves": ["Spore", "Rage Powder", "Pollen Puff", "Protect"]}
    obs1 = {"turn": 1, "active_pokemon": [{"species": "Garchomp"}, {"species": "Rillaboom"}],
            "opponent_active_pokemon": [{"species": "Amoonguss", "current_hp_fraction": 1.0}, None]}
    memory.observe_battle(obs1)
    memory.record_turn(turn=1, payload={"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "taunt", "target": 1},
                                          "slot_1": {"type": "move", "move_id": "protect"}})
    memory.observe_battle({**obs1, "turn": 2})
    assert memory.opp_protected_last_turn == []  # Taunt never moves an HP bar


def test_live_bcbaabe2_flinch_and_a_set_without_protect_never_read_as_a_protect(tmp_path):
    """T11: Rillaboom's Fake Out flinched Archaludon (`|cant|p1a: Archaludon|flinch`): our move never ran, and
    Rillaboom's set (Fake Out / Wood Hammer / Grassy Glide / U-turn) has no Protect. The live agent said "Rillaboom
    probably Protected last turn". The log now decides: nothing Protected."""
    folder = _folder(MATCH)
    agent = PokemonAgent(None, version="d1g", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    ctx = DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="me", seat_position=0)
    for name in sorted(p.name for p in folder.glob("moving-*-seat0.json")):
        state = GameState.from_mcp_state(json.loads((folder / name).read_text()))
        obs = observation_dict(state)
        if obs.get("turn") not in (12, 13):
            continue
        agent.choose_action(state, ctx)
        assert "rillaboom" not in agent.memory.opp_protected_last_turn, (name, agent.memory.opp_protected_last_turn)
        assert "cant" in " ".join(obs["protocol_log"]) or obs.get("turn") == 13
    rows = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    notes = " ".join(n for r in rows if r.get("kind") == "turn" for n in r["turn_sheet"]["notes"])
    assert "rillaboom probably Protected" not in notes


# -- D1d: fallen allies ---------------------------------------------------------------------------------------------


def test_fallen_counts_faints_per_side_and_reads_supreme_overlord():
    memory = MatchMemory()
    memory.observe_battle({**_log("|switch|p1a: Baxcalibur|Baxcalibur, L50, F|191/191", "|switch|p2a: Gholdengo|Gholdengo, L50|100/100",
                                  "|turn|1", "|faint|p1a: Baxcalibur", "|faint|p2a: Gholdengo", "|turn|2", "|faint|p1b: Lucario",
                                  "|switch|p2b: Kingambit|Kingambit, L50, M|100/100", "|-ability|p2b: Kingambit|Supreme Overlord",
                                  "|-start|p2b: Kingambit|fallen1|[silent]", "|turn|3"), "turn": 3})
    assert memory.fallen("mine") == 2 and memory.fallen("theirs") == 1
    assert memory.overlord_fallen == {"theirs:kingambit": 1}
    assert MatchMemory().fallen("mine") == 0


def test_fallen_matches_the_live_match():
    obs = _obs(MATCH, "moving-00017-seat0.json")
    memory = MatchMemory()
    memory.observe_battle(obs)
    faints = [line for line in obs["protocol_log"] if line.startswith("|faint|")]
    assert memory.fallen("mine") == sum(1 for l in faints if l.startswith("|faint|p1")) and memory.fallen("theirs") == sum(1 for l in faints if l.startswith("|faint|p2"))
    assert memory.fallen("mine") + memory.fallen("theirs") == len(faints) > 0


# -- entry hazards ----------------------------------------------------------------------------------------------


def test_hazards_track_layers_per_side_and_clear_on_sideend():
    memory = MatchMemory()
    assert memory.hazards("mine") == {"spikes": 0, "stealthrock": False, "stickyweb": False, "toxicspikes": 0}
    memory.observe_battle({**_log("|-sidestart|p1: able-p0-x|Spikes", "|-sidestart|p1: able-p0-x|Spikes", "|-sidestart|p1: able-p0-x|Spikes",
                                  "|-sidestart|p1: able-p0-x|Spikes", "|-sidestart|p1: able-p0-x|move: Stealth Rock",
                                  "|-sidestart|p2: able-p1-x|move: Sticky Web", "|-sidestart|p2: able-p1-x|move: Toxic Spikes",
                                  "|-sidestart|p2: able-p1-x|move: Tailwind", "|turn|2"), "turn": 2})
    assert memory.hazards("mine") == {"spikes": 3, "stealthrock": True, "stickyweb": False, "toxicspikes": 0}
    assert memory.hazards("theirs") == {"spikes": 0, "stealthrock": False, "stickyweb": True, "toxicspikes": 1}
    memory.observe_battle({**_log("|-sidestart|p1: able-p0-x|Spikes", "|-sidestart|p1: able-p0-x|move: Stealth Rock",
                                  "|-sideend|p1: able-p0-x|Spikes|[from] move: Rapid Spin|[of] p1a: Great Tusk",
                                  "|-sideend|p1: able-p0-x|move: Stealth Rock|[from] move: Rapid Spin|[of] p1a: Great Tusk", "|turn|3"), "turn": 3})
    assert memory.hazards("mine") == {"spikes": 0, "stealthrock": False, "stickyweb": False, "toxicspikes": 0}


def test_hazards_see_the_spikes_ting_lu_set_on_our_side_in_the_live_match():
    before, after = _obs(MATCH, "moving-00004-seat0.json"), _obs(MATCH, "moving-00014-seat0.json")
    memory = MatchMemory()
    memory.observe_battle(before)
    assert memory.hazards("mine")["spikes"] == 0
    memory.observe_battle(after)
    assert memory.hazards("mine") == {"spikes": 1, "stealthrock": False, "stickyweb": False, "toxicspikes": 0}
    assert memory.hazards("theirs")["spikes"] == 0
    assert any("[from] Spikes" in line for line in after["protocol_log"])  # Milotic paid for them


# -- A4/B5: the opponent's draft profile --------------------------------------------------------------------------


def test_draft_memory_keeps_every_public_reason_and_times_the_opponents_picks():
    picks = lambda n: [{"pick_number": i + 1, "seat_id": "them" if i % 2 == 0 else "us", "card_id": f"vgc-{i}", "species": f"Mon{i}",
                        "public_reason": f"Mon{i}: beats Mon{max(i - 1, 0)}." if i % 2 == 0 else "Drafted.", "auto": False} for i in range(n)]
    base = {"phase": "draft", "current_seat": "us", "first_drafter": "them", "decision_timeout_seconds": 15.0, "available_cards": []}
    memory = MatchMemory()
    memory.observe_draft({**base, "picks": picks(1), "decision_deadline_at": "2026-10-08T19:00:15+00:00"}, my_turn=True)
    memory.record_pick(None)
    memory._draft_think_seconds = 2.0  # we answered in 2 s (measured locally in record_pick)
    memory.observe_draft({**base, "picks": picks(3), "decision_deadline_at": "2026-10-08T19:00:27+00:00"}, my_turn=True)
    theirs = [p for p in memory.draft_picks if not p["mine"]]
    assert [p["pick_number"] for p in theirs] == [1, 3] and all(p["public_reason"].endswith(".") for p in theirs)
    assert [p["mine"] for p in memory.draft_picks] == [False, True, False]
    assert theirs[0]["seconds"] is None  # before our first turn: nothing to measure against
    assert theirs[1]["seconds"] == 10.0  # 19:00:00 -> 19:00:12 is 12 s, minus our 2 s
    profile = memory.draft_profile()
    assert profile["opponent_picks"] == 2 and profile["mean_seconds_per_pick"] == 10.0 and profile["timed_picks"] == 1
    assert profile["reasons"] == {"templated": 2, "free_text": 0, "empty": 0}


def test_draft_profile_classifies_reasons():
    memory = MatchMemory()
    memory.my_seat_key = "us"
    memory.pool_cards = {"a": {"card_id": "a", "species": "Garchomp"}, "b": {"card_id": "b", "species": "Incineroar"}}
    reasons = ["Garchomp: beats Incineroar.", "Incineroar: beats Garchomp.", "", None,
               "I want a bulky pivot that can threaten their Water types and keep Fake Out pressure up early."]
    memory.observe_draft({"phase": "draft", "current_seat": "us", "available_cards": [],
                          "picks": [{"pick_number": i + 1, "seat_id": "them", "species": "Garchomp", "public_reason": r} for i, r in enumerate(reasons)]},
                         my_turn=True)
    profile = memory.draft_profile()
    assert profile["reasons"] == {"templated": 2, "free_text": 1, "empty": 2} and profile["mean_seconds_per_pick"] is None
    assert [p["reason_kind"] for p in memory.draft_picks] == ["templated", "templated", "empty", "empty", "free_text"]


def test_draft_profile_on_the_live_match(tmp_path):
    folder = _folder(MATCH)
    agent = PokemonAgent(None, version="b5", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    for name in sorted(p.name for p in folder.glob("draft-*-seat0.json")):
        raw = json.loads((folder / name).read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        agent.choose_action(state, DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                                   agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=0))
    profile = agent.memory.draft_profile()
    theirs = [p for p in agent.memory.draft_picks if not p["mine"]]
    assert profile["opponent_picks"] == len(theirs) >= 5
    assert all(isinstance(p["public_reason"], str) and p["public_reason"] for p in theirs)
    assert profile["reasons"]["templated"] >= 4, profile  # "<Species>: beats <Species>." every time
    assert profile["timed_picks"] >= 4 and 0 < profile["mean_seconds_per_pick"] < 15, profile


def test_93a98b6f_failed_protect_streak_is_zero_at_t4_and_one_at_t5(tmp_path):
    """Flutter Mane Protected at T2, failed at T3 ("[still]" + "-fail"), Protected again at T4. The live sheet said
    "1 in 9" at T4 and "1 in 27" at T5; the truth is certain (0) then 1 in 3 (streak 1)."""
    at_t4 = _obs("93a98b6f", "moving-00004-seat0.json")
    at_t5 = _obs("93a98b6f", "moving-00005-seat0.json")
    assert at_t4["turn"] == 4 and at_t5["turn"] == 5
    memory = MatchMemory()
    memory.observe_battle(at_t4)
    assert memory.opp_protect_streak.get("fluttermane", 0) == 0 and "fluttermane" not in memory.opp_protected_last_turn
    assert memory.protect_failed["theirs"]["fluttermane"] == {3}
    memory.observe_battle(at_t5)
    assert memory.opp_protect_streak.get("fluttermane") == 1 and memory.opp_protected_last_turn == ["fluttermane"]
    folder = _folder("93a98b6f")
    agent = PokemonAgent(None, version="d1g", log_dir=tmp_path, capture_dir="", log=lambda *_: None)
    ctx = DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="me", seat_position=0)
    agent.choose_action(GameState.from_mcp_state(json.loads((folder / "moving-00005-seat0.json").read_text())), ctx)
    rows = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]
    notes = " ".join(n for r in rows if r.get("kind") == "turn" for n in r["turn_sheet"]["notes"])
    assert "fluttermane probably protected last turn" in notes.lower(), notes
    assert "1 in 27" not in notes and "1 in 9" not in notes


# -- class signals, their clock, the digest and the tendency tally (E4 / A4) ----------------------------------------


def _replay_memory(prefix: str) -> MatchMemory:
    folder = _folder(prefix)
    seat = sorted({p.name.split("-seat")[1][:1] for p in folder.glob("moving-*.json")})[0]
    memory = MatchMemory()
    for path in sorted(folder.glob(f"draft-*-seat{seat}.json")):
        memory.observe_draft(observation_dict(GameState.from_mcp_state(json.loads(path.read_text()))), my_turn=True)
    last = sorted(folder.glob(f"moving-*-seat{seat}.json"))[-1]
    memory.observe_battle(observation_dict(GameState.from_mcp_state(json.loads(last.read_text()))))
    return memory


def test_opponent_class_from_name_free_signals_on_the_live_sessions():
    """93a98b6f: 190-char free-text reasons, slow on 6 of 7 turns, two repeated Protects -> generic_llm.
    bcbaabe2 and da9c168e: 36-48-char templates, never slow, 10 and 4 Protects with no repeat -> search_bot."""
    llm = _replay_memory("93a98b6f")
    label, evidence = llm.opponent_class()
    assert label == "generic_llm" and evidence["draft_reason_mean_chars"] > 150 and evidence["slow_turns"] >= 5 and evidence["repeat_protects"] >= 1
    for prefix in ("bcbaabe2", "da9c168e"):
        bot = _replay_memory(prefix)
        label, evidence = bot.opponent_class()
        assert label == "search_bot", (prefix, evidence)
        assert evidence["draft_reason_mean_chars"] < 60 and evidence["slow_turns"] == 0 and evidence["repeat_protects"] == 0
    assert MatchMemory().opponent_class()[0] == "search_bot"  # nothing seen yet: the strongest class is assumed
    # the core lane's reply mixture reads the same label through its own adapter
    from agent.pokemon.opponent import class_of

    assert class_of(llm) == "llm" and class_of(_replay_memory("bcbaabe2")) == "search"


def test_their_bank_is_estimated_from_showdowns_warnings_about_their_clock():
    llm = _replay_memory("93a98b6f")
    bank = llm.opponent_bank_estimate()
    assert bank["slow_turns"] == [2, 3, 4, 5, 6, 7] and 150 <= bank["bank_estimate_s"] <= 260  # ~230-250 s used in 7 turns live
    bot = _replay_memory("bcbaabe2")
    assert bot.opponent_bank_estimate()["slow_turns"] == [] and bot.opponent_bank_estimate()["bank_estimate_s"] > 300
    memory = MatchMemory()
    memory.observe_battle({**_log("|player|p1|able-p0-x 1|102|", "|player|p2|able-p1-x 1|102|", "|turn|1",
                                  "|inactive|Time left: 55 sec this turn | 420 sec total", "|inactive|able-p1-x 1 has 30 seconds left.",
                                  "|inactive|able-p1-x 1 has 20 seconds left.", "|inactive|able-p1-x 1 has 10 seconds left.", "|turn|2",
                                  "|inactive|Time left: 55 sec this turn | 410 sec total"), "turn": 2})
    assert memory.players == {"mine": "able-p0-x 1", "theirs": "able-p1-x 1"} and memory.their_slow_turns == {1: 10}
    assert memory.opponent_bank_estimate() == {"bank_estimate_s": 375.0, "used_estimate_s": 45.0, "slow_turns": [1], "turns": 1, "our_bank_s": 410.0}


def test_last_turns_digest_reads_both_sides_entries_kos_and_hp_from_a_live_session():
    memory = _replay_memory("bcbaabe2")
    digest = memory.last_turns(3)
    assert digest.startswith("T14:") and "T16:" in digest and len(digest) <= 600  # the last capture is turn 17
    assert "we:" in digest and "they:" in digest and "HP after:" in digest
    memory = _replay_memory("93a98b6f")
    digest = memory.last_turns(2)
    full = memory.last_turns(6, limit=4000)
    assert "[failed]" in full  # the failed Protect at T3 is marked
    assert "KO: theirs:grimmsnarl" in full and len(memory.last_turns(6)) <= 600  # the default cap trims the oldest detail
    assert MatchMemory().last_turns() == ""


def test_tendency_tally_counts_protects_switches_doubles_and_setups_against_what_we_threatened():
    memory = MatchMemory()
    lines = ["|player|p1|me 1|1|", "|player|p2|them 1|1|", "|switch|p1a: Garchomp|Garchomp, L50|100/100", "|switch|p1b: Gyarados|Gyarados, L50|100/100",
             "|switch|p2a: Dragonite|Dragonite, L50|100/100", "|switch|p2b: Amoonguss|Amoonguss, L50|100/100", "|turn|1",
             "|move|p2a: Dragonite|Protect|p2a: Dragonite", "|-singleturn|p2a: Dragonite|Protect", "|move|p2b: Amoonguss|Spore|p1a: Garchomp",
             "|move|p1a: Garchomp|Earthquake|p2a: Dragonite|[spread] p2a,p2b", "|turn|2",
             "|switch|p2a: Kingambit|Kingambit, L50|100/100", "|move|p2b: Amoonguss|Pollen Puff|p1b: Gyarados", "|move|p1b: Gyarados|Waterfall|p2b: Amoonguss", "|turn|3",
             "|move|p2a: Kingambit|Swords Dance|p2a: Kingambit", "|move|p2b: Amoonguss|Pollen Puff|p1b: Gyarados", "|turn|4"]
    for t in (1, 2, 3):
        memory.observe_battle({**_log(*lines[:lines.index(f"|turn|{t}") + 1]), "turn": t, "active_pokemon": [{"species": "garchomp"}, {"species": "gyarados"}]})
        memory.record_turn(turn=t, payload={"type": "doubles_turn"}, threatened=["dragonite"] if t <= 2 else [], doubled=[])
    memory.observe_battle({**_log(*lines), "turn": 4, "active_pokemon": [{"species": "garchomp"}, {"species": "gyarados"}]})
    tally = memory.opp_tendencies()
    assert tally["protects_when_threatened"] == "1/2"  # T1 Protect; T2 it left instead
    assert tally["switches_out_of_ko"] == "1/2" and memory.their_voluntary_switches == [2]
    assert tally["doubles_into_one_slot"] == "0/2"  # two turns with two attacks from them, never both into one of ours
    assert tally["setups_on_a_free_turn"].startswith("1/") and tally["repeat_protects"] == 0
