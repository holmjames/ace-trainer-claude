"""Core lane, Part 1: the candidate generators and scorer fixes (D1f, D1i, D1j, D1l, the Ruination composition, unsafe
switch rows, the KO credit). Live positions replay a captured session through ONE code-only agent (the
tests/test_live_semantics.py pattern) and read the sheet and the played turn from its log; hand-built positions use the
catalog sets. Every damage number is the sheet's own (battle.damage_percent / hit_profile), never hand arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.pokemon import battle, data
from agent.pokemon.memory import MatchMemory, species_key
from agent.pokemon.tuning import DEFAULTS
from tests.test_live_semantics import replay, turn_row

# Sessions: the four Oct 7 losses (M1-M4), the Oct 8 practice win vs the search bot (O8) and vs a generic-LLM agent (O8B).
M1, M2, M3, M4, O8, O8B = "d9405f02", "1aef9227", "7ce71025", "0a2b7c2a", "bcbaabe2", "93a98b6f"
_ROWS: dict[str, list[dict]] = {}


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    """Replay a session once per module: ``rows(prefix)`` -> its log rows."""
    def get(prefix: str) -> list[dict]:
        if prefix not in _ROWS:
            _ROWS[prefix] = replay(prefix, tmp_path_factory.mktemp(prefix))[1]
        return _ROWS[prefix]
    return get


def candidates(row: dict) -> list[dict]:
    return row["turn_sheet"]["candidate_turns"]


def named(row: dict, name: str) -> list[dict]:
    return [c for c in candidates(row) if c["name"] == name]


def played(row: dict) -> dict:
    return row["payload"]


def move_of(payload: dict, slot: int) -> str | None:
    s = payload.get(f"slot_{slot}") or {}
    return s.get("move_id") if s.get("type") == "move" else None


# -- D1f: spread rows slipped past avoid_target --------------------------------------------------------------------


def test_spread_rows_no_longer_hide_finish_the_other(scorer_rows):
    # M3 turn 3: Hatterene's Dazzling Gleam KOs the 9% Whimsicott and Great Tusk holds Rock Slide (spread). The spread
    # row carried target 0, passed the avoid_target filter as Tusk's "best attack into the other foe", and the
    # "Gleam KOs Whimsicott, Headlong Rush into Incineroar" turn was never generated (REVIEW 4A).
    row = turn_row(scorer_rows(M3), 3)
    finish = named(row, "finish_the_other")
    assert finish, [c["name"] for c in candidates(row)]
    assert "dazzlinggleam already KOs whimsicott" in finish[0]["why"] and "headlongrush goes into incineroar" in finish[0]["why"]


def test_spread_rows_no_longer_hide_focus_fire(scorer_rows):
    # The Oct 8 LLM-agent game, turn 6: Waterfall + Icicle Spear into the 54% Grimmsnarl is the double that wins the
    # turn; Gyarados' Earthquake (spread) hid it and the sheet had one candidate.
    row = turn_row(scorer_rows(O8B), 6)
    focus = [c for c in named(row, "focus_fire") if "grimmsnarl" in c["why"]]
    assert focus, [c["why"] for c in candidates(row)]
    assert len(candidates(row)) >= 2


def test_best_attack_with_avoid_target_ignores_spread_rows():
    entry = {"options": [
        {"option": 0, "move": "rockslide", "priority": 0, "spread": True, "targets": [
            {"target": 0, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [300.0, 350.0], "ko": "guaranteed", "hp_frac": 0.09, "spread": True},
            {"target": 0, "species": "B", "side": "theirs", "damage_pct_of_current_hp": [30.0, 36.0], "ko": "no", "hp_frac": 1.0, "spread": True}]},
        {"option": 1, "move": "headlongrush", "priority": 0, "spread": False, "targets": [
            {"target": 1, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [400.0, 490.0], "ko": "guaranteed", "hp_frac": 0.09},
            {"target": 2, "species": "B", "side": "theirs", "damage_pct_of_current_hp": [40.0, 50.0], "ko": "no", "hp_frac": 1.0}]},
    ]}
    opt, row, _ = battle._best_attack(entry, avoid_target=1)
    assert opt["move"] == "headlongrush" and row["target"] == 2
    opt, row, _ = battle._best_attack(entry)  # no constraint: the spread summary carries its per-target rows
    assert row["target"] == 0 and row.get("rows") and {r["species"] for r in row["rows"]} == {"A", "B"}


# -- D1j: fake_out_setup's KO read and add()'s dedup ---------------------------------------------------------------


def test_a_spread_summary_is_not_a_single_target_ko():
    summary = {"target": 0, "species": "A + B", "side": "theirs", "damage_pct_of_current_hp": [110.0, 130.0], "rows": [
        {"target": 0, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [60.0, 70.0]},
        {"target": 0, "species": "B", "side": "theirs", "damage_pct_of_current_hp": [50.0, 60.0]}]}
    assert not battle.kos_a_foe(summary)
    summary["rows"][0]["damage_pct_of_current_hp"] = [100.0, 120.0]
    assert battle.kos_a_foe(summary)
    assert battle.kos_a_foe({"target": 1, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [101.0, 120.0]})
    assert not battle.kos_a_foe({"target": 1, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [90.0, 120.0]})
    assert battle.kos_a_foe({"target": 1, "species": "A", "side": "theirs", "damage_pct_of_current_hp": [90.0, 120.0]}, min_chance=0.5)


def _catalog() -> dict[str, dict]:
    return {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())}


def test_duplicate_turns_keep_the_better_scored_name():
    """best_attacks and finish_the_other produce the same payload when one slot's best attack already KOs a foe and the
    other slot's best single-target hit is into the survivor: keep the higher-scoring (and better explained) row."""
    cards = _catalog()
    m = MatchMemory()
    m.my_cards["gyarados"] = cards["Gyarados"]
    m.my_cards["lucario"] = cards["Lucario"]
    m.opp_cards["fluttermane"] = cards["Flutter Mane"]
    m.opp_cards["incineroar"] = cards["Incineroar"]
    opp = [{"target": 1, "side": "opponent", "species": "Flutter Mane"}, {"target": 2, "side": "opponent", "species": "Incineroar"}]
    slots = [{"slot": 0, "active": "Gyarados", "options": [{"type": "move", "move_id": "waterfall", "targets": [1, 2], "target_options": opp}]},
             {"slot": 1, "active": "Lucario", "options": [{"type": "move", "move_id": "closecombat", "targets": [1, 2], "target_options": opp}]}]
    obs = {"turn": 4, "team": {"a": {"species": "Gyarados", "active": True, "current_hp_fraction": 1.0}, "b": {"species": "Lucario", "active": True, "current_hp_fraction": 1.0}},
           "opponent_team": {"c": {"species": "Flutter Mane", "active": True, "current_hp_fraction": 0.2}, "d": {"species": "Incineroar", "active": True, "current_hp_fraction": 1.0}},
           "active_pokemon": [{"species": "Gyarados"}, {"species": "Lucario"}], "opponent_active_pokemon": [{"species": "Flutter Mane"}, {"species": "Incineroar"}]}
    sheet = battle.build_sheet({"type": "doubles_turn", "slots": slots}, obs, m)
    payloads = [(c["slot_0"]["option"], c["slot_0"]["target"], c["slot_1"]["option"], c["slot_1"]["target"]) for c in sheet.candidates]
    assert len(payloads) == len(set(payloads)), payloads
    waterfall_fm = next(c for c in sheet.candidates if c["slot_0"] == {"option": 0, "target": 1} and c["slot_1"] == {"option": 0, "target": 2})
    assert waterfall_fm["name"] == "finish_the_other", sheet.candidates


# -- shared fixtures for the Oct 8 positions ------------------------------------------------------------------------

M3D = "da9c168e"  # the Oct 8 loss to the search bot: we were player 2 (seat 1)


@pytest.fixture(scope="module")
def rows_seat1(tmp_path_factory):
    def get(prefix: str) -> list[dict]:
        key = prefix + ":1"
        if key not in _ROWS:
            _ROWS[key] = replay(prefix, tmp_path_factory.mktemp(prefix), seat=1)[1]
        return _ROWS[key]
    return get


def _scorer_replay(prefix: str, tmp_path: Path, seat: int = 0) -> list[dict]:
    """The same replay with the default scorer ranking the rows (reply_mixture 0): the tests below are about its
    generators, warnings and row names (core-5 pins; the mixture's own gate is tests/test_replies_live.py)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(DEFAULTS, "reply_mixture", 0.0)
        return replay(prefix, tmp_path, seat=seat)[1]


@pytest.fixture(scope="module")
def scorer_rows(tmp_path_factory):
    def get(prefix: str) -> list[dict]:
        key = prefix + ":scorer"
        if key not in _ROWS:
            _ROWS[key] = _scorer_replay(prefix, tmp_path_factory.mktemp(prefix))
        return _ROWS[key]
    return get


@pytest.fixture(scope="module")
def scorer_rows_seat1(tmp_path_factory):
    def get(prefix: str) -> list[dict]:
        key = prefix + ":1:scorer"
        if key not in _ROWS:
            _ROWS[key] = _scorer_replay(prefix, tmp_path_factory.mktemp(prefix), seat=1)
        return _ROWS[key]
    return get


def top(row: dict) -> dict:
    return candidates(row)[0]


def warnings_of(row: dict) -> list[str]:
    return row["turn_sheet"]["WARNINGS"]


# -- D1i: a priority move is an escape only when it (or the partner) removes the threat ------------------------------


def test_a_priority_move_that_cannot_ko_is_no_escape(scorer_rows):
    # O8 T13: Scale Shot KOs Arcanine before it moves; Extreme Speed does 39-46% to Dragonite. The old sheet printed
    # "(a priority move from this slot, +2, would go first)" and ranked finish_the_other with that Extreme Speed first.
    row = turn_row(scorer_rows(O8), 13)
    lethal = [w for w in warnings_of(row) if w.startswith("LETHAL") and "slot 1 (arcanine)" in w]
    assert lethal and "goes first" not in lethal[0] and "+2" not in lethal[0], lethal
    best = top(row)
    assert best["name"] == "protect_threatened" and best.get("covers") == 1, best
    assert move_of(played(row), 1) == "protect"


def test_a_priority_move_that_kos_is_an_escape(rows):
    # O8 T16: Dragonite at 28%: Extreme Speed KOs it first, so Scale Shot's 49% KO on Arcanine is no reason to Protect.
    row = turn_row(rows(O8), 16)
    risk = [w for w in warnings_of(row) if "scaleshot" in w and "slot 1 (arcanine)" in w]
    assert risk and "extremespeed (+2) goes first and KOs it 100%" in risk[0], risk
    assert move_of(played(row), 1) == "extremespeed"


def test_the_turn_after_keeps_protect_when_priority_still_cannot_ko(scorer_rows):
    row = turn_row(scorer_rows(O8), 14)
    assert move_of(played(row), 1) == "protect"


# -- #3 Ruination composition --------------------------------------------------------------------------------------


def test_ruination_is_half_of_what_is_left_not_a_flat_fifty(rows):
    # O8 T5: Scale Shot (43.7-67.2%) then Ruination into the 62% Wo-Chien (Sitrus Berry). The old sheet added a flat 50
    # and warned 'it goes down' with a 94-117% double; the real total cannot reach a KO.
    for turn in (5, 9, 10):
        row = turn_row(rows(O8), turn)
        assert not any(w.startswith("DOUBLE-TARGET") and "ruination" in w for w in warnings_of(row)), (turn, warnings_of(row))
        assert not named(row, "protect_double_target"), (turn, candidates(row))
    assert move_of(played(turn_row(rows(O8), 5)), 1) != "protect"


def test_compose_double_orders_by_speed_and_heals_with_sitrus():
    cards = _catalog()
    wochien = battle.build_mon({"species": "Wo-Chien", "current_hp_fraction": 0.62}, cards["Wo-Chien"], side="mine", position=1)
    assert data.to_id(wochien.item) == "sitrusberry"
    scale = {"move": "scaleshot", "damage_pct_of_current_hp": [43.7, 67.2], "ko": "no", "hit_chance": 0.9}
    ruin = {"move": "ruination", "damage_pct_of_current_hp": [50.0, 50.0], "ko": "no", "hit_chance": 0.9}
    lo, hi, p = battle.compose_double(scale, ruin, wochien)
    # Scale Shot first (faster): 56.3% / 32.8% is left, the berry heals 25% of max (40.3% of the HP at the start), then
    # Ruination takes half of what remains: nowhere near a KO.
    assert lo < hi < 100 and p == 0.0, (lo, hi, p)
    assert abs(lo - (100 - (56.3 + 40.3) / 2)) < 0.2 and abs(hi - (100 - (32.8 + 40.3) / 2)) < 0.2, (lo, hi)
    plain = battle.build_mon({"species": "Wo-Chien", "current_hp_fraction": 0.62}, {**cards["Wo-Chien"], "item": "Leftovers"}, side="mine", position=1)
    lo2, hi2, _ = battle.compose_double(scale, ruin, plain)
    assert lo2 > lo and hi2 > hi  # without the berry the second hit bites deeper
    flat_a = {"move": "moonblast", "damage_pct_of_current_hp": [60.0, 71.0], "ko": "no"}
    flat_b = {"move": "hypervoice", "damage_pct_of_current_hp": [49.0, 57.0], "ko": "no"}
    lo3, hi3, p3 = battle.compose_double(flat_a, flat_b, plain)
    assert (lo3, hi3) == (109.0, 128.0) and p3 == 1.0  # two flat hits still add up, uncapped


def test_a_double_that_needs_their_spread_move_to_hit_their_partner_is_discounted(rows_seat1):
    # M3D T3: Flare Blitz + Earthquake into Kingambit; Gyarados's Earthquake would hit its own Incineroar (never done, 0/6).
    row = turn_row(rows_seat1(M3D), 3)
    double = [w for w in warnings_of(row) if w.startswith("DOUBLE-TARGET") and "earthquake" in w]
    assert not double, double


# -- unsafe switch rows ---------------------------------------------------------------------------------------------


def test_no_switch_row_when_every_bench_card_dies_on_entry(scorer_rows):
    # O8 T12: Wo-Chien (43%, Protect spent) vs Scale Shot and U-turn. Arcanine is KO'd on entry 90% of the time and
    # Milotic 100% (Spikes + Wood Hammer): no switch row, the warning names the pivot, Protect (1 in 3) leads.
    row = turn_row(scorer_rows(O8), 12)
    assert not [c for c in candidates(row) if c["slot_1"].get("option") in (4, 5) and c["name"].startswith("switch")], candidates(row)
    pivot = [w for w in warnings_of(row) if w.startswith("No safe pivot for slot 1")]
    assert pivot and "arcanine would be KO'd on entry" in pivot[0] and "hazards included" in pivot[0], pivot
    assert move_of(played(row), 1) == "protect"


def test_safe_switch_prefers_the_verdict():
    cards = _catalog()
    m = MatchMemory()
    sheet = battle.TurnSheet()
    sheet.switch_ins = {"arcanine": {"verdict": "ko_on_entry", "ko_on_entry": 0.9, "score": -50.0},
                        "milotic": {"verdict": "safe", "ko_on_entry": 0.1, "score": -80.0}}
    slot = {"slot": 1, "options": [{"type": "move", "move_id": "protect"}, {"type": "switch", "species": "arcanine"}, {"type": "switch", "species": "milotic"}]}
    idx, species, preview, safe = battle.safe_switch(slot, {}, m, sheet=sheet)
    assert (idx, species, safe) == (2, "milotic", True)  # the safe one, although it scores lower
    sheet.switch_ins["milotic"] = {"verdict": "ko_on_entry", "ko_on_entry": 1.0, "score": -80.0}
    idx, species, preview, safe = battle.safe_switch(slot, {}, m, sheet=sheet)
    assert (idx, species, safe) == (1, "arcanine", False)  # none survives: the least bad, flagged


# -- the Protect note's wording ---------------------------------------------------------------------------------------


def test_protect_note_says_we_aimed_not_we_hit(rows):
    notes = [n for r in rows(O8) if r.get("kind") == "turn" for n in r["turn_sheet"]["notes"] if "probably Protected" in n]
    assert notes and all("we aimed at it" in n and "we hit it" not in n for n in notes), notes[:3]


# -- the KO credit (threat removed, partner credit, post-move rows) ------------------------------------------------


def test_ko_bonus_scales_with_the_threat_removed(rows):
    # O8B T5: Ice Shard belongs on the 27% Flutter Mane (its Moonblast KOs Baxcalibur before it moves), not on the 34%
    # Rillaboom that the HP-only bonus preferred. Code #1 sends Ice Shard into Flutter Mane and Waterfall elsewhere.
    row = turn_row(rows(O8B), 5)
    best = top(row)
    assert best["slot_1"] == {"option": 2, "target": 1}, best  # Ice Shard -> Flutter Mane
    assert best["slot_0"]["target"] != 1, best  # Waterfall is not wasted on the same target
    assert best["name"] == "finish_the_other"


def test_partner_priority_ko_is_credited_and_glaive_rush_drawback_is_charged(rows):
    # O8B T7: Rillaboom's Wood Hammer KOs Gyarados before it moves unless Baxcalibur's +1 Ice Shard removes it first;
    # Glaive Rush into Incineroar would double Flare Blitz into Baxcalibur (171-201%). Code #1 was Protect + Glaive Rush
    # with no Ice Shard row; it must now send Ice Shard into Rillaboom and never Glaive Rush.
    row = turn_row(rows(O8B), 7)
    best = top(row)
    assert best["slot_1"] == {"option": 2, "target": 2}, best  # Ice Shard -> Rillaboom
    assert not any(c["slot_1"] == {"option": 1, "target": 1} for c in candidates(row)[:3]), candidates(row)  # Glaive Rush -> Incineroar
    notes = [n for n in row["turn_sheet"]["notes"] if n.startswith("EXPOSED AFTER glaiverush")]
    assert notes and "doubles the damage" in notes[0], row["turn_sheet"]["notes"]


def test_outcome_caps_each_hit_before_the_survival_weight():
    overkill = {"damage_pct_of_current_hp": [195.0, 231.0], "ko": "guaranteed"}
    damage, ko = battle._outcome([(0.5, overkill)])
    assert damage == pytest.approx(50.0) and ko == pytest.approx(0.5)  # half a KO, not a whole one at 50% flinch
    a = {"damage_pct_of_current_hp": [60.0, 70.0], "ko": "no"}
    b = {"damage_pct_of_current_hp": [60.0, 70.0], "ko": "no"}
    damage, ko = battle._outcome([(1.0, a), (1.0, b)])
    assert damage == pytest.approx(100.0) and ko == pytest.approx(1.0)
    damage, ko = battle._outcome([(1.0, a), (0.5, b)])
    assert damage == pytest.approx(0.5 * 65 + 0.5 * 100) and ko == pytest.approx(0.5)


def test_exposed_after_note_for_a_self_dropping_move(rows):
    # M4 T7: Close Combat puts Lucario at -2 Def and Scale Shot goes from a 29% KO to a sure one.
    row = turn_row(rows(M4), 7)
    notes = [n for n in row["turn_sheet"]["notes"] if n.startswith("EXPOSED AFTER closecombat") and "scaleshot" in n]
    assert notes and "-1 DEF" in notes[0], row["turn_sheet"]["notes"]


# -- Part 1b: the Oct 8 loss ------------------------------------------------------------------------------------------


def test_fresh_fake_out_is_aimed_and_the_exposed_slot_protects(scorer_rows_seat1):
    # M3D T3: the fresh Fake Out goes to Thundurus (the slot whose Thunderbolt it can stop) 0.8 / Kingambit 0.2; the row
    # that Protects Thundurus was generated, ranked 5th and cut at the old top-4; now it is code #1.
    row = turn_row(scorer_rows_seat1(M3D), 3)
    fake = [w for w in warnings_of(row) if w.startswith("FAKE OUT")]
    assert fake and "slot 0 (thundurus): 80%" in fake[0] and "slot 1 (kingambit): 20%" in fake[0], fake
    best = top(row)
    assert best["name"] == "protect_threatened" and best["slot_0"] == {"option": 3, "target": 0}, best
    assert move_of(played(row), 0) == "protect"
    assert named(row, "protect_and_switch"), [c["name"] for c in candidates(row)]
    assert len(candidates(row)) > 4  # the cut keeps the rows that cover a slot in KO range


def test_priority_focus_row_beats_a_second_protect(rows_seat1):
    # M3D T6: Extreme Speed (+2) + Sucker Punch (+1) KO Lucario before its Close Combat; the code's row was a second
    # Protect in a row (1 in 3). The row did not exist (best_action took Kowtow Cleave x 0 over Sucker Punch x 1).
    row = turn_row(rows_seat1(M3D), 6)
    best = top(row)
    assert best["name"] == "priority_focus", best
    assert best["slot_0"] == {"option": 0, "target": 1} and best["slot_1"] == {"option": 1, "target": 1}, best
    p = played(row)
    assert move_of(p, 0) == "suckerpunch" and move_of(p, 1) == "extremespeed"


# -- Glaive Rush carry-over and D1l -----------------------------------------------------------------------------------


def _doubles(m: MatchMemory, mine: list[str], opp: list[str], hp: dict[str, float], *, extra_options=None, bench=()) -> tuple[dict, dict]:
    targets = [{"target": i + 1, "side": "opponent", "species": s} for i, s in enumerate(opp)]
    slots = []
    for i, s in enumerate(mine):
        options = []
        for mv in m.my_cards[species_key(s)]["moves"]:
            info = data.move_info(mv) or {}
            single = info.get("target") in ("normal", "any", "adjacentFoe")
            options.append({"type": "move", "move_id": data.to_id(mv), "targets": [t["target"] for t in targets] if single else [0],
                            "target_options": targets if single else []})
        options += [{"type": "switch", "species": data.to_id(b)} for b in bench]
        slots.append({"slot": i, "active": s, "force_switch": False, "options": options})
    if len(mine) == 1:
        slots.append({"slot": 1, "active": None, "options": [{"type": "pass"}]})
    team = {f"p1: {s}": {"species": s, "active": s in mine, "fainted": False, "current_hp_fraction": hp.get(s, 1.0)} for s in list(mine) + list(bench)}
    opp_team = {f"p2: {s}": {"species": s, "active": True, "fainted": False, "current_hp_fraction": hp.get(s, 1.0)} for s in opp}
    obs = {"turn": 9, "weather": None, "field": {}, "team": team, "opponent_team": opp_team,
           "active_pokemon": [{"species": s} for s in mine], "opponent_active_pokemon": [{"species": s} for s in opp]}
    return {"type": "doubles_turn", "slots": slots}, obs


def test_glaive_rush_used_last_turn_doubles_the_faster_threats():
    cards = _catalog()
    m = MatchMemory()
    m.my_cards["baxcalibur"] = cards["Baxcalibur"]
    m.my_cards["gyarados"] = cards["Gyarados"]
    m.opp_cards["fluttermane"] = cards["Flutter Mane"]
    m.opp_cards["rillaboom"] = cards["Rillaboom"]
    template, obs = _doubles(m, ["Baxcalibur", "Gyarados"], ["Flutter Mane", "Rillaboom"], {})
    plain = battle.build_sheet(template, obs, m)
    moon = next(h for t in plain.threats if t["species"] == "Flutter Mane" for h in t["hits"] if h["move"] == "moonblast" and h["into_slot"] == 0)
    m.last_active = ["baxcalibur", "gyarados"]
    m.last_payload = {"slot_0": {"type": "move", "move_id": "glaiverush", "target": 2}, "slot_1": {"type": "move", "move_id": "waterfall", "target": 1}}
    doubled = battle.build_sheet(template, obs, m)
    moon2 = next(h for t in doubled.threats if t["species"] == "Flutter Mane" for h in t["hits"] if h["move"] == "moonblast" and h["into_slot"] == 0)
    assert moon2.get("glaive_rush_doubled") and moon2["damage_pct_of_current_hp"][0] == pytest.approx(2 * moon["damage_pct_of_current_hp"][0], abs=0.2)
    glide = next(h for t in doubled.threats if t["species"] == "Rillaboom" for h in t["hits"] if h["move"] == "woodhammer" and h["into_slot"] == 0)
    assert not glide.get("glaive_rush_doubled")  # Rillaboom is slower than Baxcalibur: that hit lands after it moves
    assert any("used Glaive Rush last turn" in n for n in doubled.notes)


def test_tie_guard_avoids_a_recoil_self_ko_with_our_last_pokemon():
    """Arcanine, our last Pokémon at 8% HP and faster than their last Pokémon (a 40% Amoonguss), can finish it with
    Flare Blitz, but the recoil (a third of the damage dealt, plus the Rocky Helmet) would knock Arcanine out too: a
    tie scores nothing, so Extreme Speed (no recoil) is the row."""
    cards = _catalog()
    m = MatchMemory()
    m.my_cards["arcanine"] = cards["Arcanine"]
    m.opp_cards["amoonguss"] = cards["Amoonguss"]
    template, obs = _doubles(m, ["Arcanine"], ["Amoonguss"], {"Arcanine": 0.08, "Amoonguss": 0.4})
    for key in list(obs["opponent_team"]):
        obs["opponent_team"][key]["current_hp_fraction"] = 0.4
    obs["opponent_team"].update({f"p2: gone{i}": {"species": s, "active": False, "fainted": True, "current_hp_fraction": 0.0}
                                 for i, s in enumerate(["Incineroar", "Flutter Mane", "Grimmsnarl"])})
    sheet = battle.build_sheet(template, obs, m)
    arcanine = next(mon for mon in sheet.field.actives if mon.side == "mine")
    rilla = next(mon for mon in sheet.field.actives if mon.side == "theirs")
    blitz = next(r for e in sheet.our_options for o in e["options"] if o["move"] == "flareblitz" for r in o["targets"])
    cost = battle.self_cost_pct(arcanine, "flareblitz", tuple(blitz["damage_pct_of_current_hp"]), rilla)
    assert blitz["damage_pct_of_current_hp"][0] >= 100 and cost[1] >= 100, (blitz, cost)  # the KO would also KO Arcanine
    assert any(n.startswith("TIE GUARD") for n in sheet.notes), sheet.notes
    fb_index = next(i for i, o in enumerate(template["slots"][0]["options"]) if o.get("move_id") == "flareblitz")
    assert sheet.candidates and sheet.candidates[0]["slot_0"]["option"] != fb_index, sheet.candidates


# -- the two player-2 wins' row bugs (Oct 8 post-mortem) and the mixture's new rows -------------------------------------


def _two_vs_two(m: MatchMemory, mine: list[str], opp: list[str], hp: dict[str, float], *, fresh: list[str] = ()) -> tuple[dict, dict]:
    template, obs = _doubles(m, mine, opp, hp)
    obs["active_pokemon"] = [{"species": s} for s in mine]
    m.fresh_active = [species_key(s) for s in fresh]
    m.opp_fresh_active = []
    return template, obs


def test_finish_the_other_bonus_needs_a_killer_that_moves():
    """The +15 'one slot already KOs' bonus is for a KO that happens: a killer that is KO'd before it moves earns none
    of it (the Oct 8 win as player 2, turn 2: the bonus tied a Rillaboom that never moved with the winning switch)."""
    cards = _catalog()
    m = MatchMemory()
    m.my_cards["rillaboom"] = cards["Rillaboom"]
    m.my_cards["archaludon"] = cards["Archaludon"]
    m.opp_cards["dragonite"] = cards["Dragonite"]
    m.opp_cards["ursaluna"] = cards["Ursaluna"] if "Ursaluna" in cards else cards["Ting-Lu"]
    template, obs = _two_vs_two(m, ["Rillaboom", "Archaludon"], ["Dragonite", list(m.opp_cards.values())[1]["species"]], {"Rillaboom": 0.35, "Dragonite": 0.3})
    sheet = battle.build_sheet(template, obs, m)
    rows = [c for c in sheet.candidates if c["name"] == "finish_the_other"]
    speed_rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    rilla_first = speed_rank.get(("mine", 0), 99) < speed_rank.get(("theirs", 1), 99)
    lethal_on_rilla = [h for t in sheet.threats for h in t["hits"] if h["into_slot"] == 0 and h["ko"] != "no"]
    if rows and lethal_on_rilla and not rilla_first:
        # a killer in slot 0 that dies first: its finish_the_other row carries no bonus beyond its joint value
        for c in rows:
            if c["slot_0"]["option"] == 0:
                assert c["score"] <= max(x["score"] for x in sheet.candidates if x["name"] != "finish_the_other") + 1e-6, sheet.candidates


def test_double_target_warning_survives_a_sure_hit_that_lands_after_we_move():
    """A guaranteed single hit that lands AFTER our slot moves is no reason to skip the double-target check (the Oct 8
    win as player 2, turn 4: Fake Out + Extreme Speed did 93-110% into Iron Valiant with no warning)."""
    sheet = battle.TurnSheet()
    sheet.speed_order = [{"species": "Us", "side": "mine", "position": 0, "speed": 300}, {"species": "A", "side": "theirs", "position": 1, "speed": 200},
                         {"species": "B", "side": "theirs", "position": 2, "speed": 100}]
    sheet.threats = [
        {"position": 1, "species": "A", "hits": [{"move": "fakeout", "into_slot": 0, "species": "Us", "damage_pct_of_current_hp": [40.0, 48.0], "ko": "no", "priority": 3}]},
        {"position": 2, "species": "B", "hits": [{"move": "extremespeed", "into_slot": 0, "species": "Us", "damage_pct_of_current_hp": [53.0, 62.0], "ko": "no", "priority": 2},
                                                  {"move": "bodyslam", "into_slot": 0, "species": "Us", "damage_pct_of_current_hp": [100.0, 118.0], "ko": "guaranteed", "priority": 0}]},
    ]
    cards = _catalog()
    me = battle.build_mon({"species": "Flutter Mane", "current_hp_fraction": 1.0}, cards["Flutter Mane"], side="mine", position=0)
    battle._double_target_threats(sheet, {0: me}, DEFAULTS)
    assert sheet.focus_threats and sheet.focus_threats[0]["moves"] == ["A fakeout", "B extremespeed"], sheet.focus_threats
    assert any(w.startswith("DOUBLE-TARGET") for w in sheet.warnings), sheet.warnings
    # ... but a sure hit that lands BEFORE we move still defers to the LETHAL warning
    sheet2 = battle.TurnSheet()
    sheet2.speed_order = list(sheet.speed_order)
    sheet2.threats = [sheet.threats[0], {"position": 2, "species": "B", "hits": [
        {"move": "extremespeed", "into_slot": 0, "species": "Us", "damage_pct_of_current_hp": [100.0, 118.0], "ko": "guaranteed", "priority": 2}]}]
    battle._double_target_threats(sheet2, {0: me}, DEFAULTS)
    assert not sheet2.focus_threats


def test_the_mixture_offers_a_double_protect_only_when_both_slots_face_a_ko(rows):
    # M3 T1: Wild Charge KOs Tornadus and Hyper Voice KOs Great Tusk; both slots carry Protect.
    import tempfile
    from tests.test_replies_live import replay_mix, decision
    _, _, row = decision(M3, 1)
    names = [c["name"] for c in row["turn_sheet"]["candidate_turns"]]
    assert "double_protect" in names, names
    # O8 T6: only Wo-Chien is in KO range (a double), Archaludon is not: no double Protect row
    _, _, row6 = decision(O8, 6)
    assert "double_protect" not in [c["name"] for c in row6["turn_sheet"]["candidate_turns"]]


def test_the_search_bot_priors_come_from_the_pooled_tally():
    from agent.pokemon import replies
    assert replies.SEEDS["protect_after_protect"] <= 0.1 and 0.4 <= replies.SEEDS["fresh_protect_ko50"] <= 0.55
    assert replies._p_protect("search", streak=1, ko_one=1.0, ko_two=0.0, facing=100.0, fresh_in=False, turn=5) == replies.SEEDS["protect_after_protect"]
    assert replies._p_protect("search", streak=0, ko_one=1.0, ko_two=0.0, facing=100.0, fresh_in=False, turn=5) == replies.SEEDS["fresh_protect_ko50"]
    assert replies._p_protect("search", streak=0, ko_one=0.0, ko_two=1.0, facing=60.0, fresh_in=False, turn=5) == replies.SEEDS["double_only_ko_protect"]
    assert replies._p_switch("search", streak=1, ko_one=1.0, has_absorber=True, has_protect=True) == replies.SEEDS["spent_protect_ko_switch"]
    assert replies._p_switch("search", streak=0, ko_one=0.0, ko_two=1.0, has_absorber=True) < 0.1  # it stays when only a double KOs it
    assert replies._p_fakeout("search", turn=1) == replies.SEEDS["fakeout_first_turn"]


_SEEDS_PROBE = """
import json, os, sys
opened = []
sys.addaudithook(lambda e, a: opened.append(os.fsdecode(a[0])) if e == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)) else None)
from agent.pokemon import replies
home = os.path.realpath(os.environ["HOME"])
print(json.dumps({"file": replies.__file__, "equal": replies.SEEDS == replies.SEARCH_BOT_SEEDS, "copy": replies.SEEDS is not replies.SEARCH_BOT_SEEDS,
                  "under_home": [p for p in opened if os.path.realpath(p).startswith(home + os.sep)]}))
"""


def test_the_search_bot_seeds_are_the_repositorys_own_and_nothing_outside_it_is_read(tmp_path):
    """SEEDS is SEARCH_BOT_SEEDS alone. Until deploy #4 replies laid a tally file's suggested seeds over them at import:
    ACE_OPPONENT_TALLIES, else a file under the home directory (~/projects/ace-trainer-data/). A fresh interpreter whose
    HOME holds such a file with other seeds, and whose ACE_OPPONENT_TALLIES names another, imports replies with the
    repository's seeds and opens nothing under that HOME (on the old code both checks fail)."""
    import os
    import subprocess
    import sys

    from agent.pokemon import replies

    home = tmp_path / "home"
    planted = home / "projects" / "ace-trainer-data" / "opponent-tallies-2026-10-08.json"
    planted.parent.mkdir(parents=True)
    other = {"events": [{"event": "protect_after_protect", "suggested_seed": 0.99}, {"event": "fresh_protect_ko50", "suggested_seed": 0.01}]}
    planted.write_text(json.dumps(other))
    named = home / "named-tallies.json"
    named.write_text(json.dumps(other))
    root = Path(__file__).resolve().parent.parent
    for env in ({"HOME": str(home), "ACE_OPPONENT_TALLIES": str(named)}, {"HOME": str(home)}):
        child_env = {k: v for k, v in os.environ.items() if k != "ACE_OPPONENT_TALLIES"}
        out = subprocess.run([sys.executable, "-c", _SEEDS_PROBE], cwd=root, env={**child_env, **env}, capture_output=True, text=True,
                             timeout=120)
        assert out.returncode == 0, out.stderr[-2000:]
        result = json.loads(out.stdout.strip().splitlines()[-1])
        assert Path(result.pop("file")).resolve().is_relative_to(root)  # the repository's replies, not another checkout's
        assert result == {"equal": True, "copy": True, "under_home": []}, (env, result)
    assert replies.SEEDS == replies.SEARCH_BOT_SEEDS and not hasattr(replies, "TALLIES_PATH") and not hasattr(replies, "_load_seeds")
