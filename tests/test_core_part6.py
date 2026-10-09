"""Core lane 6: no cover from a dying redirector (the Oct 9 Testing loss as player 2, 07a3cb1f, turn 2; verified
post-mortem: ace-trainer-data/postmortem-2026-10-09-07a3cb1f.md). Follow Me / Rage Powder cover counts only while the
redirector stands (battle.redirect_cover / redirect_holds, read by status_value and the redirect row alike), an EXPOSED
AFTER hit the partner's redirect cannot keep off is a WARNING, and Helping Hand is valued by the KO it creates. Every
damage number comes from the sheet (battle.damage_percent / hit_profile, whose Helping Hand x1.5 tests/test_mech_d.py
holds against @smogon/calc), never hand arithmetic."""

from __future__ import annotations

import json

import pytest

from agent.agent import PokemonAgent
from agent.pokemon import battle, data
from agent.pokemon.memory import observation_dict, species_key
from altruagent import DecisionContext
from altruagent.models import GameState
from tests.test_live_semantics import _folder, _sequence, replay, turn_row
from tests.test_mech_d import build

M9 = "07a3cb1f"  # seat 1: we were player 2


def move_of(payload: dict, slot: int) -> str | None:
    s = payload.get(f"slot_{slot}") or {}
    return s.get("move_id") if s.get("type") == "move" else None


def sides(sheet: battle.TurnSheet) -> tuple[dict, dict]:
    ours = {m.position: m for m in sheet.field.actives if m.side == "mine"}
    theirs = {m.position: m for m in sheet.field.actives if m.side == "theirs"}
    return ours, theirs


def hit(sheet: battle.TurnSheet, species: str, move: str, into_slot: int) -> tuple[int, dict]:
    threat = next(t for t in sheet.threats if species_key(t["species"]) == species)
    return threat["position"], next(h for h in threat["hits"] if h["move"] == move and h["into_slot"] == into_slot)


def sheet_at(name: str, log_dir) -> battle.TurnSheet:
    """The sheet a code-only agent builds at 07a3cb1f's capture ``name``, after playing the captures before it."""
    folder = _folder(M9)
    agent = PokemonAgent(None, version="core6", log_dir=log_dir, capture_dir="", log=lambda *_: None)
    for path in _sequence(folder, 1):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        if path.name == name:
            obs = observation_dict(state)
            agent.memory.observe_battle(obs)
            return battle.build_sheet(state.legal_actions[0].input.get("action") or {}, obs, agent.memory, agent.params)
        agent.choose_action(state, DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                                                   agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=1))
    raise AssertionError(f"{name} not captured")


@pytest.fixture(scope="module")
def t2(tmp_path_factory) -> battle.TurnSheet:
    """Turn 2: Basculegion 158/196 + Indeedee-F 78/177 vs Rillaboom + Archaludon, Psychic Terrain."""
    return sheet_at("moving-00002-seat1.json", tmp_path_factory.mktemp("m9t2"))


# -- the live position ------------------------------------------------------------------------------------------------


def test_t2_code_no_longer_plays_follow_me_from_the_doomed_indeedee(tmp_path):
    payload = turn_row(replay(M9, tmp_path, seat=1)[1], 2)["payload"]
    assert move_of(payload, 1) != "followme" or move_of(payload, 0) == "protect", payload


def test_t2_follow_me_is_no_cover_when_the_redirector_dies_to_what_it_draws(t2):
    ours, theirs = sides(t2)
    pos, wood_hammer = hit(t2, "rillaboom", "woodhammer", 0)
    _, drawn = hit(t2, "rillaboom", "woodhammer", 1)
    assert wood_hammer["ko"] != "no" and battle.ko_probability(drawn, with_accuracy=True) >= battle.LIKELY_KO
    verdict = battle.redirect_cover(t2, 1, pos, wood_hammer, theirs)
    assert verdict["covers"] and not verdict["holds"] and verdict["drawn_ko"] == pytest.approx(battle.ko_probability(drawn, with_accuracy=True), abs=1e-3)
    assert not battle.redirect_holds(t2, 1, theirs)[0]
    assert battle.status_value("followme", ours[1], theirs, t2, 1) == (15, 0)  # was 45: the live code #1 Wave Crash + Follow Me
    assert not [c for c in t2.candidates if c["name"] == "redirect_to_protect_partner"]


def test_t2_draco_meteor_after_the_recoil_is_a_warning_naming_the_faster_wood_hammer(t2):
    _, theirs = sides(t2)
    pos, draco = hit(t2, "archaludon", "dracometeor", 0)
    breaker = battle.redirect_cover(t2, 1, pos, draco, theirs)["broken_by"]
    assert breaker["move"] == "woodhammer" and species_key(breaker["from"]) == "rillaboom"
    exposed = [w for w in t2.warnings if w.startswith("EXPOSED AFTER wavecrash") and "dracometeor" in w]
    assert len(exposed) == 1 and "followme does not cover it" in exposed[0] and "redirection ends when the redirector faints" in exposed[0]
    assert not [n for n in t2.notes if n.startswith("EXPOSED AFTER wavecrash") and "dracometeor" in n]
    # Rillaboom moves first: Follow Me would draw its Grassy Glide, which Indeedee survives, so that one stays a note
    assert [n for n in t2.notes if n.startswith("EXPOSED AFTER wavecrash") and "grassyglide" in n]


def test_t2_helping_hand_is_worth_the_ko_it_creates_before_rillaboom_moves(t2):
    ours, theirs = sides(t2)
    pos = next(p for p, m in theirs.items() if species_key(m.species) == "rillaboom")
    wave_crash = next(o for e in t2.our_options if e["slot"] == 0 for o in e["options"] if o["move"] == "wavecrash")
    row = next(r for r in wave_crash["targets"] if r["target"] == pos)
    boosted = battle._boosted_row(row, data.helping_hand_boost("helpinghand", "wavecrash"))
    assert battle.ko_probability(row, with_accuracy=True) < battle.LIKELY_KO <= battle.ko_probability(boosted, with_accuracy=True)
    rank = {(r["side"], r["position"]): i for i, r in enumerate(t2.speed_order)}
    assert rank[("mine", 0)] < rank[("theirs", pos)]  # Basculegion lands it before Rillaboom moves
    value = battle.helping_hand_value(t2, 1, theirs)
    assert value > battle.MISC_VALUE["helpinghand"]
    assert battle.status_value("helpinghand", ours[1], theirs, t2, 1) == (round(value, 1), 0)
    best = t2.candidates[0]  # code #1: Helping Hand Wave Crash KOs Rillaboom before it moves
    assert "wavecrash -> rillaboom" in best["why"] and "helpinghand" in best["why"], t2.candidates[:2]


# -- the shared redirect check ---------------------------------------------------------------------------------------


def test_the_redirector_must_stand_when_the_killer_moves_speed_order_trick_room_and_priority():
    # Basculegion at half HP falls to Archaludon's Draco Meteor; Indeedee-F at 44% falls to the faster Rillaboom first.
    kwargs = dict(hp={"Basculegion": 0.5, "Indeedee-F": 0.44})
    sheet, _ = build(["Basculegion", "Indeedee-F"], ["Rillaboom", "Archaludon"], **kwargs)
    _, theirs = sides(sheet)
    pos, draco = hit(sheet, "archaludon", "dracometeor", 0)
    _, wood_hammer = hit(sheet, "rillaboom", "woodhammer", 1)
    assert draco["ko"] != "no" and battle.ko_probability(wood_hammer, with_accuracy=True) >= battle.LIKELY_KO
    verdict = battle.redirect_cover(sheet, 1, pos, draco, theirs)
    assert not verdict["covers"] and verdict["broken_by"]["move"] == "woodhammer"
    assert battle.status_value("followme", sides(sheet)[0][1], theirs, sheet, 1) == (15, 0)
    exposed = [n for n in sheet.notes if n.startswith("EXPOSED: slot 0") and "dracometeor" in n]
    assert exposed and "redirected" not in exposed[0] and "followme does not cover it" in exposed[0]
    # Trick Room: Archaludon (slower) now moves before Rillaboom, so nothing KOs Indeedee before the Draco Meteor
    room, _ = build(["Basculegion", "Indeedee-F"], ["Rillaboom", "Archaludon"], obs_extra={"fields": ["Trick Room"]}, **kwargs)
    pos, draco = hit(room, "archaludon", "dracometeor", 0)
    assert battle.redirect_cover(room, 1, pos, draco, sides(room)[1])["broken_by"] is None
    # Priority: the slowest foe's Extreme Speed (+2) lands before Archaludon's Draco Meteor
    fast, _ = build(["Basculegion", "Indeedee-F"], ["Dragonite", "Archaludon"], hp={"Basculegion": 0.5, "Indeedee-F": 0.15})
    rank = {(r["side"], r["position"]): i for i, r in enumerate(fast.speed_order)}
    dragonite = next(p for p, m in sides(fast)[1].items() if species_key(m.species) == "dragonite")
    pos, draco = hit(fast, "archaludon", "dracometeor", 0)
    assert rank[("theirs", dragonite)] > rank[("theirs", pos)] and draco["ko"] != "no"
    assert battle.redirect_cover(fast, 1, pos, draco, sides(fast)[1])["broken_by"]["move"] == "extremespeed"


def test_spread_moves_are_never_covered_and_a_faster_spread_hit_can_break_the_cover():
    sheet, _ = build(["Amoonguss", "Flutter Mane"], ["Chi-Yu", "Rillaboom"], hp={"Flutter Mane": 0.3})
    _, theirs = sides(sheet)
    pos, heat_wave = hit(sheet, "chiyu", "heatwave", 1)
    assert heat_wave["ko"] != "no"
    spread = battle.redirect_cover(sheet, 0, pos, heat_wave, theirs)
    assert not spread["covers"] and "spread" in spread["why"]
    # Rillaboom's Wood Hammer would be drawn, but Chi-Yu's faster Heat Wave (it hits Amoonguss too) KOs the redirector first
    pos, wood_hammer = hit(sheet, "rillaboom", "woodhammer", 1)
    assert wood_hammer["ko"] != "no"
    assert battle.redirect_cover(sheet, 0, pos, wood_hammer, theirs)["broken_by"]["move"] == "heatwave"


def test_a_faster_plus_two_hit_lands_before_the_redirection_is_up():
    # Extreme Speed and Follow Me are both +2: the faster Arcanine's lands first, unless Trick Room turns the order
    for room, covered in ((None, False), ({"fields": ["Trick Room"]}, True)):
        sheet, _ = build(["Chi-Yu", "Indeedee-F"], ["Arcanine", "Amoonguss"], hp={"Chi-Yu": 0.3}, obs_extra=room)
        pos, extreme_speed = hit(sheet, "arcanine", "extremespeed", 0)
        rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
        assert extreme_speed["ko"] != "no" and extreme_speed["priority"] == 2
        assert (rank[("theirs", pos)] > rank[("mine", 1)]) is covered
        assert battle.redirect_cover(sheet, 1, pos, extreme_speed, sides(sheet)[1])["covers"] is covered


def test_status_value_and_the_redirect_row_read_the_same_check():
    # tests/test_mech_d.py's position: behind our Reflect, Rage Powder Amoonguss survives the Flare Blitz it draws
    from tests.test_mech_d import CATALOG

    arcanine = {**CATALOG["Arcanine"], "moves": ["Flare Blitz", "Snarl", "Will-O-Wisp", "Protect"]}
    sheet, _ = build(["Flutter Mane", "Amoonguss"], ["Arcanine", "Scream Tail"], bench=[], hp={"Flutter Mane": 0.2},
                     cards={"Arcanine": arcanine}, obs_extra={"side_conditions": ["Reflect"]})
    ours, theirs = sides(sheet)
    assert battle.redirect_holds(sheet, 1, theirs)[0]
    assert battle.status_value("ragepowder", ours[1], theirs, sheet, 1) == (45, 0)
    assert [c for c in sheet.candidates if c["name"] == "redirect_to_protect_partner"]


# -- Helping Hand by what it enables ---------------------------------------------------------------------------------


def _hand(sheet: battle.TurnSheet, slot: int = 0) -> tuple[float, tuple[float, int]]:
    ours, theirs = sides(sheet)
    return battle.helping_hand_value(sheet, slot, theirs), battle.status_value("helpinghand", ours[slot], theirs, sheet, slot)


def test_helping_hand_counts_the_target_threat_only_when_the_ko_lands_before_it_moves():
    # Basculegion's Wave Crash into the 50% Amoonguss becomes a likely KO with the hand (the sheet's rows x1.5); under
    # Trick Room the Amoonguss moves first, so the same KO no longer removes its action
    kwargs = dict(hp={"Amoonguss": 0.5}, drop_moves={"Aqua Jet"})
    first, _ = build(["Indeedee-F", "Basculegion"], ["Amoonguss", "Archaludon"], **kwargs)
    room, _ = build(["Indeedee-F", "Basculegion"], ["Amoonguss", "Archaludon"], obs_extra={"fields": ["Trick Room"]}, **kwargs)
    pos = next(p for p, m in sides(first)[1].items() if species_key(m.species) == "amoonguss")
    for sheet, ahead in ((first, True), (room, False)):
        rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
        assert (rank[("mine", 1)] < rank[("theirs", pos)]) is ahead
        _, row, _ = battle._best_attack(next(e for e in sheet.our_options if e["slot"] == 1), threats=battle.threat_weights(sheet, *sides(sheet)))
        assert row["target"] == pos and battle.ko_probability(row, with_accuracy=True) < battle.LIKELY_KO \
            <= battle.ko_probability(battle._boosted_row(row, 1.5), with_accuracy=True)
    fast, slow = _hand(first)[0], _hand(room)[0]
    assert fast > slow > 0
    assert _hand(first)[1] == (round(fast, 1), 0)


def test_helping_hand_keeps_the_flat_value_when_it_creates_no_ko():
    # the partner's best hit already KOs: nothing to create, the flat value stays
    sheet, _ = build(["Indeedee-F", "Basculegion"], ["Torkoal", "Amoonguss"], hp={"Torkoal": 0.2})
    value, status = _hand(sheet)
    assert value == 0 and status == (battle.MISC_VALUE["helpinghand"], 0)
    # an intact Focus Sash stops a single boosted hit: the x1.5 row would read as a KO, but it is not one
    sash, _ = build(["Indeedee-F", "Basculegion"], ["Chien-Pao", "Amoonguss"])
    entry = next(e for e in sash.our_options if e["slot"] == 1)
    _, row, _ = battle._best_attack(entry, threats=battle.threat_weights(sash, *sides(sash)))
    assert species_key(row["species"]) == "chienpao" and data.to_id(sides(sash)[1][row["target"]].item) == "focussash"
    assert battle.ko_probability(battle._boosted_row(row, 1.5)) >= battle.LIKELY_KO
    assert _hand(sash) == (0.0, (battle.MISC_VALUE["helpinghand"], 0))


def test_no_value_next_to_a_partner_whose_status_move_outvalues_its_attack():
    # Whimsicott's Moonblast would KO the 50% Chi-Yu with the hand, but its Tailwind is worth more than that Moonblast:
    # the best_attacks row pairs the hand with Tailwind, which it does not boost (not even the flat value)
    sheet, _ = build(["Indeedee-F", "Whimsicott"], ["Chi-Yu", "Torkoal"], hp={"Chi-Yu": 0.5, "Torkoal": 0.5})
    ours, theirs = sides(sheet)
    _, row, attack = battle._best_attack(next(e for e in sheet.our_options if e["slot"] == 1), threats=battle.threat_weights(sheet, ours, theirs))
    assert battle.ko_probability(row, with_accuracy=True) < battle.LIKELY_KO <= battle.ko_probability(battle._boosted_row(row, 1.5), with_accuracy=True)
    assert battle.status_value("tailwind", ours[1], theirs, sheet, 1)[0] > attack
    assert _hand(sheet) == (None, (0.0, 0))


def test_the_prompt_says_redirection_ends_when_the_redirector_faints():
    from agent.pokemon.prompts import SYSTEM_PROMPT

    assert "Redirection ends when the redirector faints: if a faster foe KOs it, the slower foes hit their original targets." in SYSTEM_PROMPT
