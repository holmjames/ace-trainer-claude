"""The deep Team Preview's opening simulation (agent/pokemon/preview_sim.py) on live-catalog cards: Protect, switching,
Fake Out, sleep, Intimidate, Tailwind, the end-of-horizon value and the anytime search. Every damage outcome a test
relies on is read from the simulation's own calculator (``Sim.best`` / ``Sim.hit``: battle.damage_percent through
matchup.calc_hit, checked against @smogon/calc), never from hand arithmetic."""

from __future__ import annotations

import json
import random
from itertools import combinations

import pytest

from agent.pokemon import data, matchup, preview_sim as ps

CATALOG = {c["species"]: c for c in json.loads((data.DATA_DIR / "cards.json").read_text())} if (data.DATA_DIR / "cards.json").exists() else {}
pytestmark = pytest.mark.skipif(not CATALOG or not (data.DATA_DIR / "pokedex.json").exists(), reason="needs data/ tables")

# Their side with every habit off: a test turns on the one it is about.
QUIET = {k: 0.0 for k in ps.SEARCH_BOT}


def u(name: str) -> matchup.Unit:
    return matchup.unit(CATALOG[name])


def make(ours: list[str], theirs: list[str], *, seed: int = 0, policy: dict | None = None, **prior: float) -> tuple[ps.Sim, list[str]]:
    """Ours and theirs as fours (the first two lead), their habits QUIET except ``prior``, our scorer ``policy``
    (ps.OURS by default), with an event log."""
    log: list[str] = []
    sim = ps.Sim([u(s) for s in ours], (0, 1), [u(s) for s in theirs], (0, 1), rng=random.Random(seed),
                 prior={**QUIET, **prior}, policy=policy, log=log)
    return sim, log


NEVER_PROTECT = {**ps.OURS, "protect": 2.0, "switch": 2.0}  # our side neither Protects nor switches


def mon(sim: ps.Sim, name: str) -> ps._Mon:
    return next(m for side in sim.team for m in side if m.name == name)


def turn_lines(log: list[str], turn: int) -> list[str]:
    return [line for line in log if line.startswith(f"T{turn}: ")]


# -- Protect --------------------------------------------------------------------------------------------------------


def test_a_protecting_lead_takes_nothing_and_never_protects_twice_in_a_row():
    sim, log = make(["Garchomp", "Archaludon", "Gyarados", "Kingambit"], ["Sylveon", "Arcanine", "Incineroar", "Dragonite"],
                    protect_lead_ko=1.0, protect_lead=1.0, protect_ko=1.0, protect_ko_aimed=1.0, protect_hit=1.0,
                    protect_hit_aimed=1.0, protect_idle=1.0)
    sylveon, arcanine = mon(sim, "Sylveon"), mon(sim, "Arcanine")
    assert sylveon.info.protect is not None and arcanine.info.protect is not None
    sim.run_turn()
    assert sylveon.hp == 100.0 and arcanine.hp == 100.0  # both Protected on turn 1: nothing of ours landed
    assert any("protects itself from" in line for line in turn_lines(log, 1))
    assert sylveon.protect_last and arcanine.protect_last
    sim.run_turn()
    assert not any(line.endswith(("Sylveon protects", "Arcanine protects")) for line in turn_lines(log, 2))  # never twice in a row
    assert sylveon.hp < 100.0 or arcanine.hp < 100.0  # and on turn 2 our attacks land


def test_our_slot_facing_a_likely_ko_protects_unless_it_removes_the_threat_first():
    # Iron Hands' Close Combat into Kingambit: the calc says a sure KO, and Kingambit cannot knock Iron Hands out first.
    sim, _ = make(["Kingambit", "Archaludon", "Gyarados", "Garchomp"], ["Iron Hands", "Farigiraf", "Incineroar", "Sylveon"])
    kingambit, hands = mon(sim, "Kingambit"), mon(sim, "Iron Hands")
    threat = sim.best(hands, kingambit)
    assert threat[2]["id"] == "closecombat" and threat[3] >= 100.0
    assert sim.danger(kingambit) == (pytest.approx(1.0), hands) and not sim.removes_first(kingambit, hands)
    assert sim.decide(kingambit)[0] == "protect"
    # Tyranitar's Knock Off KOs Basculegion, but Basculegion's Wave Crash KOs Tyranitar first: it attacks instead.
    sim, _ = make(["Basculegion", "Archaludon", "Gyarados", "Kingambit"], ["Tyranitar", "Farigiraf", "Incineroar", "Sylveon"])
    basculegion, tyranitar = mon(sim, "Basculegion"), mon(sim, "Tyranitar")
    assert sim.best(tyranitar, basculegion)[3] >= 100.0
    wave = sim.best(basculegion, tyranitar)
    assert wave[2]["id"] == "wavecrash" and wave[3] >= 100.0 and sim.first(basculegion, wave[5], tyranitar, sim.best(tyranitar, basculegion)[5])
    assert sim.danger(basculegion)[0] >= ps.OURS["protect"] and sim.removes_first(basculegion, tyranitar)
    assert sim.decide(basculegion)[0] == "attack"  # (aimed where a fresh Protect of theirs costs it least: protect_risk)


# -- switching ------------------------------------------------------------------------------------------------------


def test_a_sure_ko_without_protect_switches_into_the_bench_pokemon_that_takes_the_hit():
    sim, log = make(["Amoonguss", "Gyarados", "Garchomp", "Archaludon"], ["Torkoal", "Arcanine", "Incineroar", "Sylveon"])
    amoonguss, torkoal, garchomp = mon(sim, "Amoonguss"), mon(sim, "Torkoal"), mon(sim, "Garchomp")
    eruption = sim.best(torkoal, amoonguss)
    assert eruption[2]["id"] == "eruption" and eruption[3] >= 100.0  # the calc: a sure KO from full HP in Drought's sun
    assert amoonguss.info.protect is None and not sim.removes_first(amoonguss, torkoal)
    assert all(sim.best(f, garchomp)[1] < 0.5 for f in sim.foes(amoonguss))  # Garchomp survives either foe's best hit
    action = sim.decide(amoonguss)
    assert action[0] == "switch" and action[3] is garchomp
    sim.run_turn()
    assert sim.active[0][0] is garchomp and amoonguss.slot is None and amoonguss.hp == 100.0
    assert 0.0 < garchomp.hp < 100.0  # it took the Eruption meant for Amoonguss and lived
    assert "T1: Amoonguss switches to Garchomp" in log


def test_two_slots_never_switch_into_the_same_bench_pokemon():
    sim, _ = make(["Iron Hands", "Whimsicott", "Archaludon", "Gyarados"], ["Incineroar", "Sylveon", "Rillaboom", "Flutter Mane"])
    sim.run_turn()
    hands, whimsicott = mon(sim, "Iron Hands"), mon(sim, "Whimsicott")
    hands.hp, whimsicott.hp = 30.0, 20.0  # both in KO range of Sylveon's spread Hyper Voice
    sim._best.clear()
    sim.run_turn()
    on_field = [m for m in sim.active[0] if m is not None]
    assert len({id(m) for m in on_field}) == len(on_field)


# -- Fake Out -------------------------------------------------------------------------------------------------------


def test_our_fake_out_flinches_their_biggest_threat_on_turn_one_and_never_again():
    sim, log = make(["Iron Hands", "Archaludon", "Garchomp", "Gyarados"], ["Torkoal", "Arcanine", "Incineroar", "Sylveon"])
    hands = mon(sim, "Iron Hands")
    action = sim.decide(hands)
    assert action[0] == "fakeout"
    target = sim.active[1][action[2]]
    assert target is max(sim.foes(hands), key=sim.threat)
    sim.run_turn()
    assert f"T1: Iron Hands uses fakeout -> {target.name}" in log
    assert not any(line.startswith(f"T1: {target.name} uses ") for line in log)  # flinched: it never moved
    sim.run_turn()
    assert not any(line.startswith("T2: Iron Hands uses fakeout") for line in log)


def test_their_fake_out_takes_our_slot_s_turn_and_psychic_terrain_or_a_ghost_cannot_be_flinched():
    sim, log = make(["Garchomp", "Archaludon", "Gyarados", "Kingambit"], ["Incineroar", "Sylveon", "Arcanine", "Dragonite"],
                    fakeout=1.0, fakeout_threat=1.0)
    incineroar = mon(sim, "Incineroar")
    action = sim.decide(incineroar)
    assert action[0] == "fakeout"
    victim = sim.active[0][action[2]]
    assert victim is max(sim.foes(incineroar), key=sim.threat)
    sim.run_turn()
    assert not any(line.startswith(f"T1: {victim.name} uses ") for line in log)
    # Indeedee-F's Psychic Surge: priority moves fail into grounded targets, so neither lead can be Faked Out; Tornadus
    # flies above the terrain and can (matchup.fake_out_exposed says the same).
    sim, log = make(["Indeedee-F", "Hatterene", "Tornadus", "Ursaluna"], ["Incineroar", "Sylveon", "Arcanine", "Dragonite"],
                    fakeout=1.0, fakeout_threat=1.0)
    assert sim.terrain == "psychic"
    assert not sim.flinchable(mon(sim, "Hatterene")) and not sim.flinchable(mon(sim, "Indeedee-F"))
    sim.run_turn()
    assert not any("fakeout" in line for line in turn_lines(log, 1))
    tornadus = mon(sim, "Tornadus")
    sim.switch(mon(sim, "Hatterene"), tornadus)
    assert sim.flinchable(tornadus)
    gholdengo = matchup.unit(CATALOG["Gholdengo"])
    ghost = ps.Sim([gholdengo, u("Garchomp"), u("Archaludon"), u("Gyarados")], (0, 1), [u("Incineroar"), u("Sylveon"),
                   u("Arcanine"), u("Dragonite")], (0, 1), rng=random.Random(0), prior=QUIET)
    assert not ghost.flinchable(ghost.active[0][0])  # a Ghost type: Fake Out is Normal


def test_gale_wings_is_counted_once():
    """The simulation reads Gale Wings from battle.gale_wings, the turn sheet's rule: Talonflame's Brave Bird is +1 at
    full HP (never +2) and +0 below it."""
    sim, _ = make(["Talonflame", "Garchomp", "Gyarados", "Kingambit"], ["Indeedee-F", "Dragonite", "Arcanine", "Sylveon"])
    talonflame = mon(sim, "Talonflame")
    brave_bird = next(a for a in talonflame.info.attacks if a["id"] == "bravebird")
    assert sim.hit(talonflame, brave_bird, mon(sim, "Dragonite"), False)[2] == 1
    talonflame.hp = 60.0
    assert sim.hit(talonflame, brave_bird, mon(sim, "Dragonite"), False)[2] == 0


# -- sleep ----------------------------------------------------------------------------------------------------------


def test_spore_sleeps_its_target_which_misses_its_next_turns_and_grass_or_goggles_are_immune():
    sim, log = make(["Garchomp", "Archaludon", "Gyarados", "Kingambit"], ["Amoonguss", "Torkoal", "Incineroar", "Sylveon"],
                    policy=NEVER_PROTECT, sleep=1.0)
    amoonguss = mon(sim, "Amoonguss")
    spore = amoonguss.info.sleep
    assert spore["id"] == "spore"
    action = sim.decide(amoonguss)
    assert action[0] == "sleep"
    target = sim.active[0][action[2]]
    sim.run_turn()
    assert f"T1: Amoonguss puts {target.name} to sleep" in log
    assert target.status == "slp" and target.sleep_left in (1, 2, 3)
    left = target.sleep_left
    sim.run_turn()
    assert f"T2: {target.name} is asleep" in log and target.sleep_left == left - 1
    # who Spore cannot touch (data.status_move_blocked): Grass types, Safety Goggles; a Chesto Berry wakes its holder once
    sim, log = make(["Rillaboom", "Hatterene", "Milotic", "Garchomp"], ["Amoonguss", "Torkoal", "Incineroar", "Sylveon"], sleep=1.0)
    amoonguss = mon(sim, "Amoonguss")
    assert not sim.sleepable(mon(sim, "Rillaboom"), spore, amoonguss)
    assert not sim.sleepable(mon(sim, "Hatterene"), spore, amoonguss)
    assert sim.decide(amoonguss)[0] != "sleep"
    milotic = mon(sim, "Milotic")
    assert milotic.item == "chestoberry" and sim.sleepable(milotic, spore, amoonguss)
    sim.status_move(amoonguss, spore, milotic, "sleep")
    assert milotic.status is None and milotic.item == ""  # asleep and woken at once: the berry is spent
    sim.status_move(amoonguss, spore, milotic, "sleep")
    assert milotic.status == "slp"


# -- Intimidate, Tailwind, the value --------------------------------------------------------------------------------


def test_intimidate_lands_on_entry_and_on_a_switch_in():
    sim, _ = make(["Garchomp", "Archaludon", "Gyarados", "Kingambit"], ["Incineroar", "Sylveon", "Arcanine", "Dragonite"])
    garchomp, kingambit = mon(sim, "Garchomp"), mon(sim, "Kingambit")
    assert garchomp.stages == {"atk": -1}  # their lead Incineroar's Intimidate
    sim.switch(mon(sim, "Sylveon"), mon(sim, "Arcanine"))  # a second Intimidate comes in
    assert garchomp.stages == {"atk": -2}
    assert kingambit.stages == {}  # on the bench: untouched
    sim.switch(garchomp, kingambit)  # Defiant: -1 then +2 (matchup.intimidated)? only on entry of an Intimidate user
    assert kingambit.stages == {} and garchomp.stages == {}  # switching out clears its drops


def test_tailwind_doubles_its_side_s_speed_for_four_turns_counting_the_one_it_is_set():
    sim, log = make(["Whimsicott", "Garchomp", "Gyarados", "Archaludon"], ["Incineroar", "Sylveon", "Arcanine", "Dragonite"])
    whimsicott = mon(sim, "Whimsicott")
    before = sim.speed(whimsicott)
    sim.run_turn()
    assert "T1: Whimsicott sets Tailwind" in log
    assert sim.speed(whimsicott) == pytest.approx(2 * before) and sim.tailwind[0] == ps.TAILWIND_TURNS - 1
    for _ in range(3):
        sim.run_turn()
    assert sim.tailwind[0] == 0


def test_the_margin_is_zero_for_a_mirror_and_counts_knock_outs_and_speed_control():
    four = ["Garchomp", "Archaludon", "Gyarados", "Kingambit"]
    sim, _ = make(four, four)
    assert sim.margin() == pytest.approx(0.0)
    mon_theirs = sim.team[1][0]
    mon_theirs.hp = 0.0
    assert sim.margin() == pytest.approx(ps.VALUE["alive"] + ps.VALUE["hp"])
    sim.tailwind[0] = 3
    assert sim.margin() == pytest.approx(ps.VALUE["alive"] + ps.VALUE["hp"] + 3 * ps.VALUE["tailwind_turn"])


# -- the anytime search ---------------------------------------------------------------------------------------------


def small_search(seed: int, **kwargs) -> ps.Search:
    ours = [u(s) for s in ("Iron Hands", "Whimsicott", "Archaludon", "Gyarados", "Iron Bundle")]
    theirs = [u(s) for s in ("Incineroar", "Sylveon", "Rillaboom", "Flutter Mane", "Kingambit")]
    units = ours + theirs
    fours = [(0, 1, 2, 3), (0, 2, 3, 4)]
    plans = [(f, pair) for f in fours for pair in combinations(f, 2)]
    their = [(0.7, (5, 6, 7, 8), [((5, 6), 0.6), ((7, 8), 0.4)]), (0.3, (5, 6, 7, 9), [((6, 9), 1.0)])]
    return ps.Search(units, plans, their, seed=seed, **kwargs)


def test_the_search_is_deterministic_under_a_seed_and_follows_the_seed():
    a = small_search(7).run(60.0, max_rollouts=300)
    b = small_search(7).run(60.0, max_rollouts=300)
    assert a.rollouts == b.rollouts == 300
    assert [(r.plan.four, r.plan.leads, r.plan.sums, r.plan.counts) for r in a.ranked] == \
        [(r.plan.four, r.plan.leads, r.plan.sums, r.plan.counts) for r in b.ranked]
    c = small_search(8).run(60.0, max_rollouts=300)
    assert [r.plan.sums for r in c.ranked] != [r.plan.sums for r in a.ranked]
    full = small_search(7).run(60.0, first=2)  # a search that finishes before its deadline is the same search every time
    again = small_search(7).run(60.0, first=2)
    assert full.complete and [(r.plan.four, r.plan.leads, r.score) for r in full.ranked] == \
        [(r.plan.four, r.plan.leads, r.score) for r in again.ranked]
    assert full.ranked[0].plan.rounds == full.rounds and all(r.plan.rounds <= full.rounds for r in full.ranked)


def test_the_search_respects_its_deadline_on_a_fake_clock_and_a_zero_budget_plays_nothing():
    ticks = [0.0]

    def clock() -> float:
        ticks[0] += 0.01  # every look at the clock costs 10 ms
        return ticks[0]

    result = small_search(1).run(0.5, now=clock)
    assert 0 < result.rollouts <= 50 and result.seconds <= 0.52
    nothing = small_search(1).run(0.0)
    assert nothing.rollouts == 0 and not nothing.complete and nothing.ranked == []


def test_the_schedule_grows_with_the_budget_and_their_leads_are_drawn_in_proportion():
    search = small_search(3)
    sizes = [search.cost(*search.schedule(b)) for b in (0.5, 2.0, 8.0)]
    assert sizes[0] <= sizes[1] <= sizes[2] and sizes[2] > sizes[0]
    assert all(search.cost(*search.schedule(b)) * ps.NOMINAL_ROLLOUT_S <= b * ps.BUDGET_SHARE + 1e-9 or search.schedule(b)[0] == 1
               for b in (0.5, 2.0, 8.0))
    draws = [search.their_leads(0, k) for k in range(100)]
    assert 55 <= draws.count((5, 6)) <= 65 and 35 <= draws.count((7, 8)) <= 45
    assert all(search.their_leads(1, k) == (6, 9) for k in range(10))


def test_the_last_round_holds_the_best_plan_of_each_finalist_four_and_the_kept_plan_on_equal_rollouts():
    plain = small_search(5).run(60.0, first=2)
    last = [r for r in plain.ranked if r.plan.rounds == plain.rounds]
    assert {r.plan.four for r in last} == {(0, 1, 2, 3), (0, 2, 3, 4)}  # both fours reach the last round...
    assert len({r.plan.n for r in last}) == 1  # ...on equal rollouts
    worst = plain.ranked[-1].plan  # a plan the first round knocked out
    kept = small_search(5, keep=(worst.four, worst.leads)).run(60.0, first=2)
    survivor = next(r for r in kept.ranked if r.plan.four == worst.four and r.plan.leads == worst.leads)
    assert survivor.plan.rounds == kept.rounds and survivor.plan.n == kept.ranked[0].plan.n
