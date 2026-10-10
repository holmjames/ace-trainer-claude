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


def test_friendly_fire_is_charged_in_hp_units(monkeypatch):
    # M2 T3: Earthquake also hits our own Archaludon (28.4-33.5%); the row that uses it pays ally_damage_w x that hit.
    monkeypatch.setitem(tuning.DEFAULTS, "reply_mixture", 0.0)  # core-5 pin: the default scorer
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


# -- merge request 3: the review of stage 5714330 (default path) ------------------------------------------------------


def _synthetic(mine, opp, bench=(), *, status=None, fields=None, hp=None, items=None, turn=9):
    cards = _catalog()
    m = MatchMemory()
    for s in list(mine) + list(bench):
        m.my_cards[species_key(s)] = cards[s]
    for s in opp:
        m.opp_cards[species_key(s)] = cards[s]
    template, obs = _doubles(m, list(mine), list(opp), hp or {}, bench=bench)
    obs["turn"] = turn
    for key, entry in list(obs["team"].items()) + list(obs["opponent_team"].items()):
        name = entry["species"]
        if status and name in status:
            entry["status"] = status[name]
        if items and name in items:
            entry["item"] = items[name]
    if fields:
        obs["fields"] = fields
    return battle.build_sheet(template, obs, m), template, obs, m


def _labels(template: dict, cand: dict) -> tuple[str, str]:
    opts = {s["slot"]: s["options"] for s in template["slots"]}
    out = []
    for n in (0, 1):
        o = opts[n][cand[f"slot_{n}"]["option"]]
        out.append(o.get("move_id") or "->" + o.get("species", "?"))
    return tuple(out)


def test_an_awake_pokemon_may_switch_while_a_spore_user_is_out():
    # Review finding 1: Chi-Yu + Gyarados vs Scarf Landorus-T + Amoonguss. Earthquake KOs Chi-Yu (128-151%) before it
    # moves; Gyarados is an awake Spore target anyway, so the switch adds none and must stay the top row.
    sheet, template, *_ = _synthetic(("Chi-Yu", "Gyarados"), ("Landorus-Therian", "Amoonguss"), ("Kingambit", "Garchomp"))
    top = _labels(template, sheet.candidates[0])
    assert top[0].startswith("->"), (top, [c["why"] for c in sheet.candidates])
    assert not [w for w in sheet.warnings if w.startswith("No safe pivot") or w.startswith("No sleep-safe pivot")], sheet.warnings
    # a sleeping Chi-Yu absorbs Spore where it stands: switching it hands Spore a target, priced, but the KO is worse
    sheet, template, *_ = _synthetic(("Chi-Yu", "Gyarados"), ("Landorus-Therian", "Amoonguss"), ("Kingambit", "Garchomp"),
                                     status={"Chi-Yu": "SLP"})
    top = sheet.candidates[0]
    assert _labels(template, top)[0].startswith("->") and "where their sleep move had no target" in top["why"], top


def test_a_sleeper_does_not_pivot_out_into_spore_when_nothing_threatens_it():
    # 51c86e5c T3 / T4: the sleeping Archaludon (T3) and Chi-Yu (T4) are Spore sinks; no row pivots them out into a fresh
    # target unless it is facing a KO (T4: Close Combat into Chi-Yu, so its switch goes to the sleeping Archaludon).
    cards = _catalog()
    sheet, template, obs, m = _synthetic(("Gyarados", "Archaludon"), ("Chien-Pao", "Amoonguss"), ("Chi-Yu", "Indeedee-F"),
                                         status={"Archaludon": "SLP"})
    theirs = {x.position: x for x in sheet.field.actives if x.side == "theirs"}
    slot1 = next(s for s in template["slots"] if s["slot"] == 1)
    idx, species, preview, safe = battle.safe_switch(slot1, theirs, m, sheet=sheet)
    assert not safe and preview.get("unsafe") == "sleep", (species, preview.get("unsafe"))
    assert not [c for c in sheet.candidates if c["name"] in ("protect_and_switch", "switch_weak_slot")], [c["name"] for c in sheet.candidates]


def test_taunt_is_tracked_from_the_protocol_log():
    m = MatchMemory()
    log = ["|switch|p1a: Gyarados|Gyarados, L50, F|202/202", "|switch|p2b: Amoonguss|Amoonguss, L50, F|100/100", "|turn|1",
           "|move|p1a: Gyarados|Taunt|p2b: Amoonguss", "|-start|p2b: Amoonguss|move: Taunt", "|cant|p2b: Amoonguss|move: Taunt|Spore", "|turn|2"]
    obs = {"team": {"p1: Gyarados": {"species": "gyarados"}}, "protocol_log": log}
    m.observe_protocol(obs)
    assert m.taunted == {"theirs:amoonguss": 1}
    m.observe_protocol({**obs, "protocol_log": log + ["|-end|p2b: Amoonguss|move: Taunt", "|turn|3"]})
    assert m.taunted == {}
    m.observe_protocol({**obs, "protocol_log": log + ["|switch|p2b: Hatterene|Hatterene, L50, F|100/100", "|turn|3"]})
    assert m.taunted == {}  # it left the field
    m.observe_protocol({**obs, "protocol_log": log + ["|faint|p2b: Amoonguss", "|turn|3"]})
    assert m.taunted == {}


def test_a_taunted_spore_user_is_no_sleep_threat_and_draws_no_second_taunt():
    # Review finding 2: after a turn-1 Taunt lands, turn 2 must not Taunt again (it fails) nor warn about Spore.
    sheet, template, obs, m = _synthetic(("Gyarados", "Archaludon"), ("Chien-Pao", "Amoonguss"), turn=2)
    m.taunted = {"theirs:amoonguss": 1}
    sheet = battle.build_sheet(template, obs, m)
    gyarados = next(x for x in sheet.field.actives if x.side == "mine" and x.species == "Gyarados")
    amoonguss = next(x for x in sheet.field.actives if x.side == "theirs" and x.species == "Amoonguss")
    assert battle.taunt_value(gyarados, 0, 2, amoonguss, sheet) == 0.0
    assert not battle.sleep_users(sheet, {x.position: x for x in sheet.field.actives if x.side == "theirs"})
    assert not [w for w in sheet.warnings if w.startswith("SLEEP")]
    assert not [c for c in sheet.candidates if _labels(template, c)[0] == "taunt"], [c["why"] for c in sheet.candidates]


def test_a_taunt_next_to_a_ko_on_its_target_is_worth_the_miss_only(monkeypatch):
    # Review finding 3: Heat Wave (90% accurate) KOs Amoonguss before it moves; the Taunt on it beside the Heat Wave is
    # credited only for the miss, and Gyarados Protects from Wild Charge instead (the deployed build's top row).
    monkeypatch.setitem(tuning.DEFAULTS, "reply_mixture", 0.0)  # core-5 pin: the default scorer
    sheet, template, *_ = _synthetic(("Chi-Yu", "Gyarados"), ("Iron Hands", "Amoonguss"))
    assert _labels(template, sheet.candidates[0]) != ("heatwave", "taunt"), [c["why"] for c in sheet.candidates[:3]]
    assert sheet.candidates[0]["name"] == "protect_threatened"


def test_a_prankster_taunt_fails_into_psychic_terrain_and_armor_tail():
    # Review finding 4: a +1 Taunt does nothing to a grounded target in Psychic Terrain or beside Armor Tail.
    for opp, fields in ((("Chien-Pao", "Amoonguss"), {"PSYCHIC_TERRAIN": 3}), (("Farigiraf", "Amoonguss"), None)):
        sheet, template, *_ = _synthetic(("Tornadus", "Indeedee-F"), opp, fields=fields)
        tornadus = next(x for x in sheet.field.actives if x.side == "mine" and x.species == "Tornadus")
        amoonguss = next(x for x in sheet.field.actives if x.side == "theirs" and x.species == "Amoonguss")
        assert battle.taunt_value(tornadus, 0, 2, amoonguss, sheet) == 0.0, opp
        assert not any("Taunt goes first" in w for w in sheet.warnings), sheet.warnings
        assert _labels(template, sheet.candidates[0])[0] != "taunt", opp


def test_an_eaten_lum_berry_is_no_sleep_cover_and_paralysis_or_freeze_cost_actions():
    sheet, template, obs, m = _synthetic(("Gyarados", "Archaludon"), ("Chien-Pao", "Amoonguss"), ("Salamence", "Kingambit"),
                                         items={"Salamence": ""})
    theirs = {x.position: x for x in sheet.field.actives if x.side == "theirs"}
    assert data.to_id(_catalog()["Salamence"]["item"]) == "lumberry"
    assert battle.fresh_sleep_target(sheet, "salamence", theirs, m)  # the berry is gone: Spore lands
    sheet, *_ = _synthetic(("Gyarados", "Archaludon"), ("Iron Hands", "Kingambit"), status={"Gyarados": "PAR", "Archaludon": "FRZ"})
    assert sheet.acts(0) == pytest.approx(0.75) and sheet.acts(1) == pytest.approx(0.2)
    rank = {(r["side"], r["position"]): i for i, r in enumerate(sheet.speed_order)}
    assert battle.survival_factor(1, sheet, 0, rank) <= 0.2 + 1e-9


# -- merge request 4: the review of stage 5714330 (reply model, behind reply_mixture) ----------------------------------


def test_a_choice_holders_protect_is_charged_the_locked_turn():
    # d9405f02 T4: Scarf Landorus-T's Protect locks it into Protect; the cell charges one lost action and no hit next turn.
    ctx, rows, cols = _context("d9405f02", 4)
    lando = next(n for n, m in ctx.ours.items() if species_key(m.species) == "landorustherian")
    assert data.to_id(ctx.ours[lando].item) == "choicescarf"
    row = next(r for r in rows if any(a.kind == "protect" and a.slot == lando for a in r.actions))
    col = cols[0]
    locked = ctx.cell(row, col)
    scarf = ctx.ours[lando].item
    ctx.ours[lando].item = None
    try:
        free = ctx.cell(row, col)
    finally:
        ctx.ours[lando].item = scarf
    assert locked < free - 5, (locked, free)


def test_the_worst_case_keeps_the_columns_the_tally_finds_plausible():
    # 0a2b7c2a T3: the read put 0.95 on one column; the floor now keeps every column either prior gives >= 0.08.
    ctx, rows, cols = _context(M4, 3)
    matrix = [[ctx.cell(r, c) for c in cols] for r in rows]
    tally = ctx.prior(cols)
    read = replies._search_reply(matrix, tally, ctx.P)
    kappa = ctx.P["reply_search_qr"]
    blended = [(1 - kappa) * a + kappa * b for a, b in zip(tally, read)]
    floor = [j for j in range(len(cols)) if max(blended[j], tally[j]) >= float(ctx.P.get("reply_floor_min_prior", 0.08))]
    assert len(floor) >= 2 and max(blended) < 0.9, (floor, max(blended))


def test_the_mixture_keeps_a_sleeper_in_rather_than_feed_spore_a_doomed_newcomer():
    # 51c86e5c T5/T6: the sleeping Indeedee-F out to a Gyarados that Spore and Wild Charge both had was the top row at
    # all 40 seeds; the menu now mirrors safe_switch.
    for turn in (5, 6):
        ctx, rows, cols = _context(S51, turn)
        assert not [r for r in rows if any(a.kind == "switch" and species_key(a.species) == "gyarados" for a in r.actions)], turn


def test_a_taunted_foe_has_no_protect_or_setup_column_and_spore_calls_are_scored():
    sheet, template, obs, m = _synthetic(("Gyarados", "Archaludon"), ("Chien-Pao", "Amoonguss"), turn=3)
    m.taunted = {"theirs:chienpao": 2, "theirs:amoonguss": 2}
    sheet = battle.build_sheet(template, obs, m)
    ours = {x.position: x for x in sheet.field.actives if x.side == "mine"}
    theirs = {x.position: x for x in sheet.field.actives if x.side == "theirs"}
    ctx = replies._Context(sheet, m, sorted(template["slots"], key=lambda s: s["slot"]), ours, theirs,
                           tuning.merged({"reply_mixture": 1}), "search", 3, obs.get("opponent_team"))
    cols = ctx.their_menu()
    assert not [c for c in cols for a in c.actions if a.kind in ("protect", "setup", "sleep")], [c.name for c in cols]
    m.opp_last_move = {"amoonguss": "spore", "chienpao": "iciclecrash"}
    assert ctx.prediction_matched({"actions": [("Amoonguss", "sleep", "spore")]}) == 1.0


# -- core-5: the mixture's tactical errors (the five hand scenarios it failed with reply_mixture on by default) --------


def _mix(mine, opp, bench=(), **kw):
    """A reply-model context for a synthetic doubles turn (catalog sets), the mixture on and its seed pinned."""
    sheet, template, obs, m = _synthetic(mine, opp, bench, **kw)
    ours = {x.position: x for x in sheet.field.actives if x.side == "mine"}
    theirs = {x.position: x for x in sheet.field.actives if x.side == "theirs"}
    ctx = replies._Context(sheet, m, sorted(template["slots"], key=lambda s: s["slot"]), ours, theirs,
                           tuning.merged({"reply_mixture": 1, "reply_seed": 7}), replies.opponent_class(m), sheet.turn, obs.get("opponent_team"))
    return ctx, template


def _act(ctx, template, slot: int, move_or_species: str, target: int = 0):
    options = next(s for s in template["slots"] if s["slot"] == slot)["options"]
    index = next(i for i, o in enumerate(options) if move_or_species in (o.get("move_id"), o.get("species")))
    return ctx.actor_for(slot, {"option": index, "target": target})


def _their(ctx, pos: int, move: str, slot: int):
    hit = ctx.their_rows(pos, move)[slot]
    return replies._Actor("theirs", pos, "attack", move=move, priority=hit.get("priority", 0) or 0, rows={slot: hit}, target=slot,
                          species=ctx.theirs[pos].species)


def test_rage_powder_draws_both_single_target_hits_while_the_redirector_stands():
    # redirect_to_free_specs_attacker: Surging Strikes (159.2-189.2%) and the +1 Sucker Punch (100.0-118.5%) each KO Flutter
    # Mane. Behind Rage Powder (+2) the Surging Strikes goes into Amoonguss (28.6-34.1%), the Sucker Punch fails (Amoonguss
    # is not attacking), and the faster Flutter Mane's Moonblast KOs Urshifu (143.2-168.2%) before it moves. The cell had no
    # redirection at all: both hits landed on Flutter Mane.
    ctx, template = _mix(("Amoonguss", "Flutter Mane"), ("Urshifu-Rapid-Strike", "Kingambit"))
    strikes, punch = _their(ctx, 1, "surgingstrikes", 1), _their(ctx, 2, "suckerpunch", 1)
    col = replies._JointColumn("both into Flutter Mane", [strikes, punch], 1.0)
    rage = replies._JointRow("rage", [_act(ctx, template, 0, "ragepowder"), _act(ctx, template, 1, "moonblast", 1)], {})
    sludge = replies._JointRow("sludge", [_act(ctx, template, 0, "sludgebomb", 1), _act(ctx, template, 1, "moonblast", 1)], {})
    assert ctx.cell(rage, col) > ctx.cell(sludge, col) + 100, (ctx.cell(rage, col), ctx.cell(sludge, col))
    redirect = {"slot": 0, "p": 1.0, "move": "ragepowder"}
    assert [(t, share) for t, _, share in ctx._aimed(strikes, redirect, {}, [strikes])] == [(1, 0.0), (0, 1.0)]
    assert [(t, share) for t, _, share in ctx._aimed(punch, redirect, {}, [punch])] == [(1, 0.0)]  # drawn in, and it fails
    # the redirection ends when the redirector faints (the Oct 9 loss 07a3cb1f): a KO already on it sends the hit home
    down = {("mine", 0): [(1.0, {"damage_pct_of_current_hp": [150.0, 160.0], "ko": "guaranteed"})]}
    assert [(t, share) for t, _, share in ctx._aimed(strikes, redirect, down, [strikes])] == [(1, 1.0)]


def test_sucker_punch_needs_a_target_that_is_about_to_attack():
    ctx, template = _mix(("Amoonguss", "Flutter Mane"), ("Urshifu-Rapid-Strike", "Kingambit"), ("Incineroar",))
    punch = _their(ctx, 2, "suckerpunch", 0)
    hit = punch.rows[0]
    for ours, fails in (("sludgebomb", False), ("spore", True), ("incineroar", True)):
        me = _act(ctx, template, 0, ours, 1 if ours != "incineroar" else 0)
        assert ctx._fails_into(punch, hit, ("mine", 0), [punch, me], {}) is fails, ours
    attack = _act(ctx, template, 0, "sludgebomb", 1)
    assert ctx._fails_into(punch, hit, ("mine", 0), [attack, punch], {("mine", 0): 1.0})  # it has already moved


def test_our_sleep_move_is_priced_like_theirs():
    # intimidate_the_dancer: Spore was a flat status credit (35 points) against two chip attacks. Now, like their sleep move
    # on us: the expected sleep's actions at sleep_future_w each, worth what the foe does with one (battle.threat_weight);
    # a sleep move that lands before the foe moves stops this turn's action too.
    ctx, template = _mix(("Amoonguss", "Whimsicott"), ("Gyarados", "Pelipper"))
    spore = _act(ctx, template, 0, "spore", 1)
    w, weight = tuning.DEFAULTS["sleep_future_w"], ctx.weights["gyarados"]
    slept: dict = {}
    after = ctx._our_sleep_gain(spore, 1.0, {}, {}, {("theirs", 1): 1.0}, slept)  # the faster Gyarados has moved
    assert after == pytest.approx(weight * w * battle.SLEEP_TURNS) and not slept
    before = ctx._our_sleep_gain(spore, 1.0, {}, {}, {}, slept)
    assert before == pytest.approx(weight * w * (battle.SLEEP_TURNS - 1)) and slept == {("theirs", 1): 1.0}
    assert ctx._our_sleep_gain(spore, 1.0, {("theirs", 1): 1.0}, {}, {}, {}) == 0.0  # into a working Protect
    grass, template = _mix(("Amoonguss", "Whimsicott"), ("Rillaboom", "Pelipper"))
    assert grass._our_sleep_gain(_act(grass, template, 0, "spore", 1), 1.0, {}, {}, {}, {}) == 0.0  # Grass ignores powders
    names = [r.name for r in ctx.our_menu([])]
    assert "spore_1_and_attack" in names and "spore_2_and_attack" in names  # one row per foe it can put to sleep


def test_the_next_turn_charges_a_pokemon_once():
    # Both foes KO our 5% Amoonguss next turn (Moonblast, Weather Ball); it has no Protect and there is no bench, so nothing
    # answers it: it falls once, reply_next_w x its material. The one-ply term charged it once per foe (twice).
    ctx, template = _mix(("Amoonguss", "Archaludon"), ("Whimsicott", "Pelipper"), hp={"Amoonguss": 0.05})
    ctx._our_next_hits = lambda *a, **k: []  # no KO credit of ours: the charges alone
    assert ctx._next_turn({}, {}, {}, []) == pytest.approx(-tuning.DEFAULTS["reply_next_w"] * ctx.material[0])


def test_a_bench_and_a_redirector_answer_a_ko_next_turn():
    ctx, template = _mix(("Amoonguss", "Flutter Mane"), ("Urshifu-Rapid-Strike", "Kingambit"), ("Incineroar",))
    assert ctx._bench_ready()
    alive = [(0, 1.0, 1.0, ctx.ours[0], ctx.material[0], None), (1, 1.0, 1.0, ctx.ours[1], ctx.material[1], None)]
    strikes = ctx.their_rows(1, "surgingstrikes")[1]
    assert ctx._redirect_answer(1, alive, strikes, ctx.theirs[1]) == pytest.approx(0.6)  # Amoonguss can Rage Powder again
    assert ctx._redirect_answer(1, [alive[0][:1] + (0.5,) + alive[0][2:], alive[1]], strikes, ctx.theirs[1]) == pytest.approx(0.3)
    assert ctx._redirect_answer(0, alive, ctx.their_rows(1, "surgingstrikes")[0], ctx.theirs[1]) == 0.0  # no redirector beside it


def test_their_column_carries_the_ko_that_lands_before_our_slot_moves():
    # protect_vs_ko: Surging Strikes (454.9-540.7%) and the +1 Aqua Jet (160.4-191.2%) both KO the 35% Flutter Mane; only Aqua
    # Jet lands before it moves (307 vs 149). The column took Surging Strikes (two capped KO premiums tied, the first listed
    # won), so the matrix let Flutter Mane's Moonblast KO Urshifu first.
    ctx, template = _mix(("Flutter Mane", "Rillaboom"), ("Urshifu-Rapid-Strike", "Gyarados"), hp={"Flutter Mane": 0.35})
    assert ctx.best_hit_into(1, 0)["move"] == "aquajet"


def test_a_foe_that_can_ko_first_is_not_read_as_protecting():
    # da9c168e T6: Lucario's Close Combat KOs the slower Kingambit first; the bot attacked into our double priority. Of its
    # 7/9 fresh Protects against our one-move KO, both misses had that reason or a partner's fresh Fake Out.
    seed = replies.SEEDS["fresh_protect_ko_covered"]
    assert replies._p_protect("search", streak=0, ko_one=0.9, ko_two=0.0, facing=95.0, fresh_in=False, turn=6, covered=True) == seed
    assert replies._p_protect("search", streak=0, ko_one=0.9, ko_two=0.0, facing=95.0, fresh_in=False, turn=6) == replies.SEEDS["fresh_protect_ko50"]
    ctx, template = _mix(("Flutter Mane", "Rillaboom"), ("Urshifu-Rapid-Strike", "Gyarados"), hp={"Flutter Mane": 0.35})
    assert ctx.covered(1) and ctx.covered_protect(1)  # Aqua Jet KOs Flutter Mane first; Moonblast KOs Urshifu
    cols = ctx.their_menu()
    rows = ctx.our_menu([])
    matrix = [[ctx.cell(r, c) for c in cols] for r in rows]
    prior, tally = replies._blended_prior(ctx, cols, matrix, ctx.P)
    for j, c in enumerate(cols):
        if any(a.kind == "protect" and a.slot == 1 for a in c.actions):
            assert prior[j] <= tally[j] + 1e-9, (c.name, prior[j], tally[j])


def test_an_intimidate_switch_in_weakens_the_hits_into_its_partner():
    ctx, template = _mix(("Amoonguss", "Whimsicott"), ("Gyarados", "Pelipper"), ("Incineroar",))
    switch = _act(ctx, template, 1, "incineroar")
    gone, incoming = {("mine", 1): 1.0}, {("mine", 1): switch.incoming}
    waterfall = _their(ctx, 1, "waterfall", 0)
    row = waterfall.rows[0]
    weak = ctx._under_intimidate(waterfall, row, 0, gone, incoming)
    assert weak["damage_pct_of_current_hp"][1] < 0.8 * row["damage_pct_of_current_hp"][1]  # -1 Attack: about two thirds
    hurricane = _their(ctx, 2, "hurricane", 0)
    assert ctx._under_intimidate(hurricane, hurricane.rows[0], 0, gone, incoming) == hurricane.rows[0]  # special: untouched


def test_a_faster_rock_slide_can_flinch():
    ctx, template = _mix(("Garchomp", "Flutter Mane"), ("Gyarados", "Pelipper"))
    rock = _act(ctx, template, 0, "rockslide")
    assert ctx._flinch_chance(rock, ("theirs", 1)) == pytest.approx(0.3)
    cloaked, template = _mix(("Garchomp", "Flutter Mane"), ("Gyarados", "Pelipper"), items={"Gyarados": "Covert Cloak"})
    assert cloaked._flinch_chance(_act(cloaked, template, 0, "rockslide"), ("theirs", 1)) == 0.0
