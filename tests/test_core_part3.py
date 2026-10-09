"""Core lane, Part 3 / 3b, merge request 1 (the default path): our own sleep and its wake odds, their sleep move as a
threat, the Taunt row, no fresh Spore target, pinch berries in the KO odds, friendly fire in the right units, and the
fainted bench. Live positions replay a captured session through ONE code-only agent (tests/test_live_semantics.py's
pattern); the Oct 8 evening loss is 51c86e5c (spec: ace-trainer-data/scenarios/51c86e5c.md, verified post-mortem). Every
damage number comes from the sheet (battle.damage_percent / hit_profile), never hand arithmetic; the expected KO odds
are the Oct 8 verification's @smogon/calc numbers."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent.pokemon import battle, data, replies, tuning
from agent.pokemon.memory import MatchMemory, species_key
from tests.test_live_semantics import LIVE, replay, turn_row
from tests.test_replies import _catalog, _doubles

S51, M2, M4 = "51c86e5c", "1aef9227", "0a2b7c2a"
_ROWS: dict[str, list[dict]] = {}


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    def get(prefix: str) -> list[dict]:
        if prefix not in _ROWS:
            _ROWS[prefix] = replay(prefix, tmp_path_factory.mktemp(prefix))[1]
        return _ROWS[prefix]
    return get


def _obs(prefix: str, name: str) -> dict:
    folder = next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))
    return json.loads((folder / name).read_text())["observation"]


def cands(row: dict) -> list[dict]:
    return row["turn_sheet"]["candidate_turns"]


def move_of(payload: dict, slot: int) -> str | None:
    s = payload.get(f"slot_{slot}") or {}
    return s.get("move_id") if s.get("type") == "move" else None


def threat(row: dict, species: str) -> dict:
    return next(t for t in row["turn_sheet"]["opponent_threats"] if species_key(t["species"]) == species)


# -- (a) our own sleep: status in the roster and the sheet, Showdown's wake odds -------------------------------------


def test_wake_odds_follow_the_sleep_counter():
    # The counter starts at 2, 3 or 4 and drops by one per attempt: 0, 1/3, 1/2, 1 after 0-3 missed attempts.
    assert [battle.wake_chance(n) for n in range(5)] == [0.0, pytest.approx(1 / 3), 0.5, 1.0, 1.0]
    assert [battle.wake_chance(n, rest=True) for n in range(3)] == [0.0, 0.0, 1.0]  # Rest: exactly two turns
    assert battle.wake_chance(None) == pytest.approx(1 / 3)  # no log: the middle of the range


def test_missed_attempts_are_read_from_the_protocol_log():
    assert battle.sleep_attempts(_obs(S51, "moving-00003-seat0.json")) == {"archaludon": (1, False)}
    assert battle.sleep_attempts(_obs(S51, "moving-00010-seat0.json")) == {"gyarados": (0, False), "archaludon": (2, False)}
    assert battle.sleep_attempts(_obs(S51, "moving-00011-seat0.json")) == {"archaludon": (3, False)}


def test_status_build_mon_reads_the_live_capitals():
    mon = battle.build_mon({"species": "Archaludon", "status": "SLP"}, _catalog()["Archaludon"], side="mine", position=1)
    assert mon.status == "slp"
    assert battle.build_mon({"species": "Archaludon", "status": "FNT"}, _catalog()["Archaludon"], side="mine", position=1).status is None
    para = battle.build_mon({"species": "Archaludon", "status": "PAR"}, _catalog()["Archaludon"], side="mine", position=1)
    awake = battle.build_mon({"species": "Archaludon"}, _catalog()["Archaludon"], side="mine", position=1)
    assert para.stat("spe") == awake.stat("spe") * 0.5  # a live "PAR" never halved a speed before


@pytest.mark.parametrize("turn,index,acts", [(2, 0, {"1": 0}), (3, 0, {"1": 33}), (4, 0, {"1": 0}), (5, 0, {"0": 0, "1": 33}),
                                             (6, 0, {"0": 33, "1": 0}), (7, 0, {"1": 33}), (8, 0, {"0": 0, "1": 50}),
                                             (9, 0, {"1": 100})])
def test_the_sheet_carries_our_sleepers_chance_to_act(rows, turn, index, acts):
    # The spec's values per fixture (moving-00002 .. 00011), and the roster's on_field rows carry the status.
    row = turn_row(rows(S51), turn, index)
    assert row["turn_sheet"].get("asleep_acts_pct") == acts, row["turn_sheet"].get("asleep_acts_pct")
    sleepers = {m["slot"] for m in row["battle_roster"]["ours"]["on_field"] if m.get("status") == "slp"}
    assert sleepers == {int(n) for n in acts}


def test_a_sleeper_that_cannot_act_is_not_exposed_and_draws_no_fake_out(rows):
    # T4: Chi-Yu was Spored on entry (0% to act): no "EXPOSED: slot 1 (chiyu) moves before ironhands", the fresh Fake
    # Out is aimed at Gyarados alone, and no row counts Chi-Yu's Heat Wave.
    row = turn_row(rows(S51), 4)
    assert not [n for n in row["turn_sheet"]["notes"] if n.startswith("EXPOSED: slot 1 (chiyu)")]
    fake = [w for w in row["turn_sheet"]["WARNINGS"] if w.startswith("FAKE OUT")]
    assert fake and "slot 0 (gyarados)" in fake[0] and "chiyu" not in fake[0], fake
    assert not [c for c in cands(row) if "heatwave" in c["why"] and ("KOs" in c["why"] or "->" in c["why"])], cands(row)


# -- (b) their sleep move: a threat row per awake, sleepable slot and a SLEEP warning --------------------------------


def test_spore_is_a_threat_row_with_a_warning_that_names_the_counters(rows):
    row = turn_row(rows(S51), 1)
    sleep = threat(row, "amoonguss").get("sleep") or []
    assert {r["into_slot"] for r in sleep} == {0, 1} and all(r["move"] == "spore" and r["value"] > 0 for r in sleep), sleep
    warning = [w for w in row["turn_sheet"]["WARNINGS"] if w.startswith("SLEEP")]
    assert warning and "spore" in warning[0] and "slot 0 (gyarados)'s Taunt goes first" in warning[0], warning
    # T2: Archaludon is asleep, so only Gyarados can still be slept; the warning says why Archaludon is out of reach
    row2 = turn_row(rows(S51), 2)
    assert {r["into_slot"] for r in threat(row2, "amoonguss").get("sleep") or []} == {0}
    assert any("archaludon" in w and "already has a status" in w for w in row2["turn_sheet"]["WARNINGS"] if w.startswith("SLEEP"))


def test_sleep_immunities_come_from_mechs_list():
    cards = _catalog()
    m = MatchMemory()
    for s in ("Rillaboom", "Hatterene"):
        m.my_cards[species_key(s)] = cards[s]
    for s in ("Amoonguss", "Chien-Pao"):
        m.opp_cards[species_key(s)] = cards[s]
    template, obs = _doubles(m, ["Rillaboom", "Hatterene"], ["Amoonguss", "Chien-Pao"], {})
    sheet = battle.build_sheet(template, obs, m)
    # a Grass type ignores powder; Magic Bounce reflects it (and Hatterene's Safety Goggles): no row, no threat value
    assert not any(t.get("sleep") for t in sheet.threats) and not sheet.sleep_threat
    assert not [w for w in sheet.warnings if w.startswith("SLEEP")]


def test_a_sleep_user_is_a_bigger_ko_target():
    # The KO credit scales by the threat removed: the Spore user's threat is the sleep it inflicts.
    sheet = battle.TurnSheet()
    sheet.threats = [{"position": 2, "species": "Amoonguss", "hits": []}]
    sheet.sleep_threat = {2: 80.0}
    assert battle.threat_weight(2, sheet, {}) == 80.0


# -- (c) the Taunt row: valued by what it blocks, generated, and code #1 on the decisive turn -------------------------


def test_taunt_is_worth_what_it_blocks_and_nothing_into_its_blockers(rows):
    cards = _catalog()
    m = MatchMemory()
    for s in ("Gyarados", "Archaludon"):
        m.my_cards[species_key(s)] = cards[s]
    for s in ("Chien-Pao", "Amoonguss"):
        m.opp_cards[species_key(s)] = cards[s]
    template, obs = _doubles(m, ["Gyarados", "Archaludon"], ["Chien-Pao", "Amoonguss"], {})
    sheet = battle.build_sheet(template, obs, m)
    gyarados = next(x for x in sheet.field.actives if x.side == "mine" and x.species == "Gyarados")
    amoonguss = next(x for x in sheet.field.actives if x.side == "theirs" and x.species == "Amoonguss")
    value = battle.taunt_value(gyarados, 0, 2, amoonguss, sheet)
    assert value > 0 and battle.status_value("taunt", gyarados, {2: amoonguss}, sheet, 0)[0] == value
    for item, ability in (("Mental Herb", amoonguss.ability), (amoonguss.item, "Oblivious"), (amoonguss.item, "Magic Bounce")):
        blocked = battle.Mon(**{**amoonguss.__dict__, "item": item, "ability": ability})
        assert battle.taunt_value(gyarados, 0, 2, blocked, sheet) == 0.0, (item, ability)


def test_taunt_on_the_spore_user_is_code_one_on_turn_one(rows):
    # s51_t1_taunt_the_spore: Gyarados (102) Taunts Amoonguss (50) before its Spore; the only candidate used to be both
    # attacks into the Chien-Pao that Protected.
    row = turn_row(rows(S51), 1)
    assert move_of(row["payload"], 0) == "taunt" and row["payload"]["slot_0"]["target"] == 2, row["payload"]
    assert any(c["slot_0"] == {"option": 2, "target": 2} for c in cands(row)), cands(row)


def test_turn_three_taunts_or_protects_with_the_sleeper_in(rows):
    # s51_t3_no_fresh_sleeper_into_spore: Taunt Amoonguss, or Protect while the sleeping Archaludon stays in; never a
    # switch into Chi-Yu or Indeedee-F (Spored on entry live). No row offers that switch, and the previews say why.
    row = turn_row(rows(S51), 3)
    p = row["payload"]
    assert (move_of(p, 0) == "taunt" and p["slot_0"]["target"] == 2) or \
        (move_of(p, 0) == "protect" and (p.get("slot_1") or {}).get("type") != "switch"), p
    switch_rows = [c for c in cands(row) for k in ("slot_0", "slot_1") if c[k]["option"] in (4, 5)]
    assert not switch_rows, switch_rows
    ins = row["turn_sheet"]["switch_ins"]
    assert all("sleep_target" in ins[k] for k in ("chiyu", "indeedeef")), {k: v.get("sleep_target") for k, v in ins.items()}


# -- (d) no fresh target for an active sleep user --------------------------------------------------------------------


def test_a_sleeping_or_immune_bench_card_is_no_fresh_target():
    cards = _catalog()
    m = MatchMemory()
    for s in ("Gyarados", "Archaludon", "Rillaboom", "Indeedee-F"):
        m.my_cards[species_key(s)] = cards[s]
    for s in ("Chien-Pao", "Amoonguss"):
        m.opp_cards[species_key(s)] = cards[s]
    template, obs = _doubles(m, ["Gyarados", "Archaludon"], ["Chien-Pao", "Amoonguss"], {}, bench=("Rillaboom", "Indeedee-F"))
    sheet = battle.build_sheet(template, obs, m)
    theirs = {x.position: x for x in sheet.field.actives if x.side == "theirs"}
    assert battle.fresh_sleep_target(sheet, "indeedeef", theirs, m)  # awake and not immune: Spore takes it on entry
    assert battle.fresh_sleep_target(sheet, "rillaboom", theirs, m) is None  # Grass: powder does nothing
    sheet.bench_status["indeedeef"] = "slp"
    assert battle.fresh_sleep_target(sheet, "indeedeef", theirs, m) is None  # already asleep: a Spore sink
    slot = next(s for s in template["slots"] if s["slot"] == 1)
    idx, species, _, safe = battle.safe_switch(slot, theirs, m, sheet=sheet)
    assert safe and species in ("rillaboom", "indeedeef")


def test_turn_four_does_not_bring_indeedee_into_spore(rows):
    # s51_t4 (provisional): Gyarados -> Indeedee-F handed Spore a fresh target live.
    p = turn_row(rows(S51), 4)["payload"]
    assert not any((p.get(f"slot_{s}") or {}).get("type") == "switch" and species_key(p[f"slot_{s}"].get("species") or "") == "indeedeef"
                   for s in (0, 1)), p


# -- pinch berries in the KO odds (the Oct 8 verification: m2 T3 0.795, m2 T4 91.0%) ---------------------------------


def _rows_into(sheet: battle.TurnSheet, slot: int, move: str, species: str) -> dict:
    entry = next(e for e in sheet.our_options if e["slot"] == slot)
    opt = next(o for o in entry["options"] if o["move"] == move)
    return next(r for r in opt["targets"] if r["side"] == "theirs" and species_key(r["species"]) == species)


def _sheet_at(prefix: str, turn: int, overrides: dict | None = None):
    """Rebuild one turn's sheet from the replayed agent's own inputs."""
    captured = []
    original = battle.build_sheet

    def spy(template, obs, memory, params=None):
        sheet = original(template, obs, memory, params)
        if obs.get("turn") == turn and not captured:
            captured.append((copy.deepcopy(template), copy.deepcopy(obs), copy.deepcopy(memory), sheet))  # the replay goes on
        return sheet

    import agent.agent as agent_module
    agent_module.battle_rules.build_sheet = spy
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            replay(prefix, Path(tmp))
    finally:
        agent_module.battle_rules.build_sheet = original
    template, obs, memory, sheet = captured[0]
    return (battle.build_sheet(template, obs, memory, tuning.merged(overrides)) if overrides else sheet), template, obs, memory


def test_wiki_berry_between_two_hits_matches_the_verification():
    sheet, *_ = _sheet_at(M2, 3)
    sylveon = next(x for x in sheet.field.actives if x.side == "theirs" and species_key(x.species) == "sylveon")
    assert data.to_id(sylveon.item) == "wikiberry"
    flash, quake = _rows_into(sheet, 0, "flashcannon", "sylveon"), _rows_into(sheet, 1, "earthquake", "sylveon")
    # Archaludon (210) lands Flash Cannon before Gyarados (204) lands Earthquake: 0.795 by @smogon/calc
    assert battle.ko_in_order([flash, quake], sylveon) == pytest.approx(0.795, abs=0.03)
    plain = battle.Mon(**{**sylveon.__dict__, "item": None})
    assert battle.ko_in_order([flash, quake], plain) > 0.95  # the old summed-range number
    # m2 T4: Flash Cannon then Waterfall into the same Sylveon, 91.0% by @smogon/calc
    sheet4, *_ = _sheet_at(M2, 4)
    syl4 = next(x for x in sheet4.field.actives if x.side == "theirs" and species_key(x.species) == "sylveon")
    double = [_rows_into(sheet4, 0, "flashcannon", "sylveon"), _rows_into(sheet4, 1, "waterfall", "sylveon")]
    assert battle.ko_in_order(double, syl4) == pytest.approx(0.91, abs=0.04)
    assert battle._outcome([(1.0, double[0]), (1.0, double[1])], syl4)[1] == pytest.approx(battle.ko_in_order(double, syl4))


def test_compose_double_heals_with_any_pinch_berry():
    cards = _catalog()
    sylveon = battle.build_mon({"species": "Sylveon", "current_hp_fraction": 0.75}, cards["Sylveon"], side="mine", position=0)
    assert data.to_id(sylveon.item) == "wikiberry"
    first = {"move": "flashcannon", "damage_pct_of_current_hp": [57.0, 69.0], "ko": "no"}
    second = {"move": "earthquake", "damage_pct_of_current_hp": [42.5, 50.4], "ko": "no"}
    _, _, p_berry = battle.compose_double(first, second, sylveon)
    _, _, p_plain = battle.compose_double(first, second, battle.Mon(**{**sylveon.__dict__, "item": None}))
    assert p_berry < p_plain - 0.1


# -- friendly fire: charged like damage into a foe, blocked by the ally's Protect -----------------------------------


def test_friendly_fire_is_charged_in_hp_units():
    # M2 T3: Earthquake also hits our own Archaludon (28.4-33.5%); the row that uses it pays ally_damage_w x that hit.
    with_ff, *_ = _sheet_at(M2, 3, {})
    without, *_ = _sheet_at(M2, 3, {"ally_damage_w": 0.0})
    quake = next(o for e in with_ff.our_options if e["slot"] == 1 for o in e["options"] if o["move"] == "earthquake")
    ally = next(r for r in quake["targets"] if r["side"] == "mine")
    charge = tuning.DEFAULTS["ally_damage_w"] * battle._expected(ally, 0, tuning.DEFAULTS)

    def eq_line(sheet):
        return next(c for c in sheet.candidates if c["slot_1"] == {"option": 1, "target": 0} and c["slot_0"]["option"] == 2)

    assert eq_line(without)["score"] - eq_line(with_ff)["score"] == pytest.approx(charge, abs=0.2)


def test_the_reply_cell_spares_a_protecting_ally_from_our_earthquake():
    sheet, template, obs, memory = _sheet_at(M2, 3)
    P = tuning.merged({"reply_mixture": 1, "reply_seed": 7})
    ours = {x.position: x for x in sheet.field.actives if x.side == "mine"}
    theirs = {x.position: x for x in sheet.field.actives if x.side == "theirs"}
    ctx = replies._Context(sheet, memory, sorted(template["slots"], key=lambda s: s["slot"]), ours, theirs, P,
                           replies.opponent_class(memory), sheet.turn, obs.get("opponent_team"))
    rows = ctx.our_menu(sheet.candidates)
    col = ctx.their_menu()[0]
    shielded = next(r for r in rows if [a.kind for a in r.actions] == ["protect", "attack"] and r.actions[1].move == "earthquake")
    exposed = next(r for r in rows if [a.kind for a in r.actions] == ["attack", "attack"] and r.actions[1].move == "earthquake")
    for row, blocked in ((shielded, True), (exposed, False)):
        quake = row.actions[1]
        kept = dict(quake.ally_rows)
        full = ctx.cell(row, col)
        quake.ally_rows = {}
        bare = ctx.cell(row, col)
        quake.ally_rows = kept
        assert (full == pytest.approx(bare)) is blocked, (row.name, full, bare)


# -- the fainted bench (Part 3b bug 1) --------------------------------------------------------------------------------


def test_the_reply_model_never_switches_into_a_fainted_card():
    # M4 T7: Gholdengo fainted; the mixture's worst reply was "incineroar -> Gholdengo" (rank_candidates passes no
    # observation, so _their_bench kept the fainted card). The sheet now carries their team.
    sheet, template, obs, memory = _sheet_at(M4, 7, {"reply_mixture": 1, "reply_seed": 7})
    assert sheet.opp_team["gholdengo"]["fainted"]
    assert not [c for c in sheet.reply["columns"] if "-> Gholdengo" in c["name"]], sheet.reply["columns"]
    assert all(c.get("best_reply", "").find("Gholdengo") < 0 for c in sheet.candidates)


# -- merge request 2: the reply model (behind reply_mixture) -----------------------------------------------------------


def _context(prefix: str, turn: int):
    sheet, template, obs, memory = _sheet_at(prefix, turn)
    P = tuning.merged({"reply_mixture": 1, "reply_seed": 7})
    ours = {x.position: x for x in sheet.field.actives if x.side == "mine"}
    theirs = {x.position: x for x in sheet.field.actives if x.side == "theirs"}
    ctx = replies._Context(sheet, memory, sorted(template["slots"], key=lambda s: s["slot"]), ours, theirs, P,
                           replies.opponent_class(memory), sheet.turn, obs.get("opponent_team"))
    return ctx, ctx.our_menu(sheet.candidates), ctx.their_menu()


def test_a_joint_reply_is_one_column_however_it_is_reached():
    # m4 T2: "Gholdengo Make It Rain (spread) + Dragonite Scale Shot" repeated the attack pair, doubling its prior.
    _, _, cols = _context(M4, 2)
    keys = [tuple(sorted((a.slot, a.kind, a.move, str(a.target)) for a in c.actions)) for c in cols]
    assert len(keys) == len(set(keys)), [c.name for c in cols]


def test_a_fresh_protect_facing_our_ko_switches_one_time_in_nine():
    # The verification's like-for-like count (1 of 16, Laplace 0.11); a spent Protect keeps the 0.38.
    assert replies._p_switch("search", streak=0, ko_one=0.9, has_absorber=True) == pytest.approx(0.11)
    assert replies._p_switch("search", streak=1, ko_one=0.9, has_absorber=True) == replies.SEEDS["spent_protect_ko_switch"]


def test_their_special_columns_pair_with_every_partner_attack():
    # m1 T6: "Gholdengo Protect / Nasty Plot / switch" came only with Dragonite's Extreme Speed, so its prior added up
    # to 0.55 (6 of 29 observed); each now comes with Scale Shot too.
    ctx, _, cols = _context("d9405f02", 6)
    prior = ctx.prior(cols)
    es = sum(p for c, p in zip(cols, prior) if any(a.move == "extremespeed" for a in c.actions))
    plays = {a.move for c in cols if any(a.kind in ("protect", "setup") and a.species.lower() == "gholdengo" for a in c.actions)
             for a in c.actions if a.species.lower() == "dragonite"}
    assert {"extremespeed", "scaleshot"} <= plays, plays
    assert es < 0.45, es


def test_no_priority_bonus_for_a_hit_into_a_protect():
    ctx, rows, cols = _context("d9405f02", 6)
    fake = next(r for r in rows if r.actions[0].kind == "fakeout")
    protect = next(c for c in cols if any(a.kind == "protect" and a.species.lower() == "dragonite" for a in c.actions))
    base = ctx.cell(fake, protect)
    ctx.P = {**ctx.P, "priority_w": 0.0}
    assert ctx.cell(fake, protect) == pytest.approx(base)  # Fake Out into Dragonite's Protect earns nothing for its +3


def test_the_fake_out_column_aims_at_a_slot_it_can_flinch():
    # m4 T7: the bot's Fake Out went into Tornadus; Lucario (Inner Focus) was the 'stoppable' slot, so the column was missing.
    _, _, cols = _context(M4, 7)
    fakes = [a for c in cols for a in c.actions if a.kind == "fakeout"]
    assert fakes and all(a.target == 1 for a in fakes), [c.name for c in cols]


def test_the_spore_column_and_the_taunt_that_blocks_it():
    # 51c86e5c T1: "Spore our biggest threat" is a column (Archaludon, whose sleep costs most), and the Taunt row keeps
    # its value against it while the attack row pays for the sleep.
    ctx, rows, cols = _context(S51, 1)
    spore = [c for c in cols if any(a.kind == "sleep" for a in c.actions)]
    assert spore and all(next(a for a in c.actions if a.kind == "sleep").target == 1 for c in spore), [c.name for c in cols]
    taunt = next(r for r in rows if any(a.move == "taunt" for a in r.actions))
    attack = next(r for r in rows if all(a.kind == "attack" for a in r.actions))
    asleep = spore[0]
    awake = next(c for c in cols if not any(a.kind == "sleep" for a in c.actions) and
                 [a.move for a in c.actions if a.species.lower() == "chienpao"] == [a.move for a in asleep.actions if a.species.lower() == "chienpao"])
    # Against the Spore the Taunt row gains on the attack row by what the sleep costs (Archaludon's next 1-3 turns),
    # beyond whatever it gains against Amoonguss's attack
    gap_spore = ctx.cell(taunt, asleep) - ctx.cell(attack, asleep)
    gap_awake = ctx.cell(taunt, awake) - ctx.cell(attack, awake)
    assert gap_spore > gap_awake + 20, (gap_spore, gap_awake)
    lost = ctx._sleep_loss(next(a for a in asleep.actions if a.kind == "sleep"), 1.0, {}, {}, {}, {}, {})
    assert lost > 20


def test_the_next_turn_counts_our_switch_in():
    # m4 T2: the Landorus pivot's next turn (its Earthquake into the foes) was never counted; the newcomer's best hit
    # comes from its switch preview.
    ctx, rows, cols = _context(M4, 2)
    pivot = next((r for r in rows if any(a.kind == "switch" for a in r.actions)), None)
    assert pivot is not None
    switch = next(a for a in pivot.actions if a.kind == "switch")
    mon, preview = switch.incoming
    foe = next(iter(ctx.theirs.values()))
    hits = ctx._our_next_hits(switch.slot, mon, preview, foe.position, foe, False)
    deal = next(d for d in preview["deals_best"] if species_key(d["to"]) == species_key(foe.species))
    assert hits and hits[0][0]["damage_pct_of_current_hp"] == deal["damage_pct"]


def test_the_search_read_expects_the_protect_the_bot_made():
    # m2 T3: every version of the bot Protects Sylveon (the verification); the pooled tally gives those columns under
    # 0.5, the search read (a quantal response to our likely rows, the way a search scores its replies) raises them (0.64),
    # and the mixture then plays Body Press / Waterfall into Incineroar instead of Earthquake next to Archaludon. The
    # replay is the mixture's own (our earlier choices set our Protect streaks, so they shape the read).
    from tests.test_replies_live import decision

    payload, template, row = decision(M2, 3)
    table = row["turn_sheet"]["reply_table"]
    protect = sum(c["prior"] for c in table["columns"] if c["name"].startswith("sylveon Protect"))
    assert protect > 0.55, table["columns"]
    assert (payload.get("slot_1") or {}).get("move_id") != "earthquake", payload
    sheet, *_ = _sheet_at(M2, 3, {"reply_mixture": 1, "reply_seed": 7, "reply_search_qr": 0.0})
    assert sum(c["prior"] for c in sheet.reply["columns"] if c["name"].startswith("sylveon Protect")) < 0.5  # the tally alone


def test_the_mixture_taunts_the_spore_user_and_reads_protect_or_switch_with_spore():
    # 51c86e5c T1/T3 with the mixture on: the bot's replies were Protect + Spore and a switch + Spore; both are columns,
    # the Taunt row aims at Amoonguss (a status row used to go at target 0, nobody), and it is code #1 both turns.
    for turn, combo in ((1, "chienpao Protect + amoonguss spore"), (3, "chienpao -> Iron Hands + amoonguss spore")):
        ctx, rows, cols = _context(S51, turn)
        assert any(c.name.startswith(combo) for c in cols), (turn, [c.name for c in cols])
        taunts = [a for r in rows for a in r.actions if a.move == "taunt"]
        assert taunts and all(a.target == 2 for a in taunts), [(a.target, a.name) for a in taunts]
    from tests.test_replies_live import decision

    for turn in (1, 3):
        payload, template, row = decision(S51, turn)
        assert move_of(payload, 0) == "taunt" and payload["slot_0"]["target"] == 2, (turn, payload)
