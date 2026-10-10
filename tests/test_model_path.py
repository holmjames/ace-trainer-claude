"""The model path (lane M1): what happens between the model's answer and the move, and what the model is shown.

1. A reply-mixture exception costs the mixture for that turn, never the match: the scorer's rows play (readiness 2B.1).
2. Our last Pokémon's Protect with nothing to wait for earns no override credit and never plays over a row (model
   audit R1/R7; d9405f02 T8-T10). "Nothing to wait for" also checks what the sheet's stall reasons miss (a fresh Fake
   Out, Black Sludge, a Wish, Leech Seed, Yawn, a Guts holder's orb): d9405f02 T7's fresh Fake Out keeps it allowed.
3. answer_rules_v2 (off by default): on the reply mixture the model chooses rows only; a lower row stands within
   max(row_margin_points, 15% of |row 0|); no read bypass; the prompt and the payload say so (model audit R1).
4. recheck_v2 (off by default): no DOUBLE-TARGET triggers; LETHAL / EXPOSED fire only where row 0 covers that slot;
   the message sends the model back to row 0 (model audit R3, readiness item 10).
5. prompt_v2 and prompt_kb (off by default): the audit's R2 and R5 prompt text and its R4 knowledge base, checked
   against the mechanics data; the defaults send the Oct 10 deploy's prompt and payload byte for byte.
6. scripts/live_positions.py: the live answers replayed through the agent (code only), a model-on mode that is a dry
   run without --max-calls and never spends more than it.

Positions are replayed from the live captures (tests/fixtures/live/) through a code-only agent; the model is a scripted
fake or absent. No test calls a model.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from altruagent import DecisionContext
from altruagent.models import GameState
from examples.llm.base import Choice, InvalidChoice

from agent.agent import PokemonAgent, _waits_out, override_value
from agent.pokemon import battle as battle_rules
from agent.pokemon import tuning
from agent.pokemon.memory import MatchMemory, observation_dict
from test_live_semantics import LIVE, _sequence

FAST = {"lineup_sim_budget_s": 0.0}  # Team Preview's 20 s search is not what these tests are about


def _folder(prefix: str) -> Path:
    return next(p for p in LIVE.iterdir() if p.is_dir() and p.name.startswith(prefix))


def _context(folder: Path, raw: dict, seat: int) -> DecisionContext:
    return DecisionContext(session_id=folder.name, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                           agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=seat)


def _rows(log_dir: Path) -> list[dict]:
    return [json.loads(line) for f in Path(log_dir).glob("*.jsonl") for line in f.read_text().splitlines() if line.strip()]


def _position(prefix: str, seat: int, turn: int, params: dict | None = None, index: int = 0):
    """Replay a live session code-only up to a battle decision: (agent, sheet, template, adapter) for that position."""
    from examples.llm import pokemon as adapters

    folder = _folder(prefix)
    agent = PokemonAgent(None, version="mpath", log_dir=Path(tempfile.mkdtemp()), capture_dir="", log=lambda *_: None,
                         params=tuning.merged({**FAST, **(params or {})}))
    seen = 0
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        obs = observation_dict(state)
        if state.phase == "moving" and obs.get("turn") == turn:
            if seen == index:
                agent.memory.observe_battle(obs)
                action = state.legal_actions[0]
                template = action.input.get("action") or {}
                sheet = battle_rules.build_sheet(template, obs, agent.memory, agent.params)
                base = adapters.doubles_choice(action, state)
                choice = Choice(kind="doubles_turn", prompt={}, schema={}, build=base.build, fallback=lambda: None)
                sheet.candidates = [c for c in sheet.candidates if _valid(base.build, c)]
                for i, c in enumerate(sheet.candidates):
                    c["row"] = i
                return agent, sheet, template, choice
            seen += 1
        agent.choose_action(state, _context(folder, raw, seat))
    raise AssertionError(f"turn {turn} not in {prefix}")


def _valid(build, row: dict) -> bool:
    try:
        build({"slot_0": row["slot_0"], "slot_1": row["slot_1"]})
        return True
    except InvalidChoice:
        return False


def _index(template: dict, number: int, move_or_species: str) -> int:
    options = next(s for s in template["slots"] if s.get("slot") == number)["options"]
    return next(i for i, o in enumerate(options) if o.get("move_id") == move_or_species or o.get("species") == move_or_species)


def _answer(row: int = 0, *, override: dict | None = None, prediction: tuple | None = None) -> dict:
    out = {"row": row, "win_condition": "w", "reasoning_summary": "Test answer.",
           "override": {"use": override is not None, "slot_0": (override or {}).get("slot_0", {"option": 0, "target": 0}),
                        "slot_1": (override or {}).get("slot_1", {"option": 0, "target": 0})}}
    if prediction is not None:
        out["opponent_prediction"] = {"their_slot_1": prediction[0], "their_slot_2": prediction[1], "why": "test"}
    return out


# -- 1. a reply-mixture exception is contained (readiness 2B.1) ----------------------------------------------------------


def test_a_mixture_exception_is_contained_and_every_turn_plays_a_scorer_row(tmp_path, monkeypatch):
    """replies.evaluate raises on every call at bcbaabe2 (reply_mixture 1). Before the fix, 17 of its 18 battle decisions
    played the smoke agent's "Fallback:" move (readiness-1010b live-safety/mixture_raise.txt). Now each main turn notes the
    failure, carries no reply table, and plays the scorer's row 0; nothing reaches the guard."""
    from agent.pokemon import replies

    calls: list[int] = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise KeyError("injected mixture bug")

    monkeypatch.setattr(replies, "evaluate", boom)
    folder = _folder("bcbaabe2")
    agent = PokemonAgent(None, version="mpath", log_dir=tmp_path, capture_dir="", log=lambda *_: None,
                         params=tuning.merged({**FAST, "reply_mixture": 1.0, "reply_seed": 7.0}))
    played = []
    for path in _sequence(folder, 0):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        decision = agent.choose_action(state, _context(folder, raw, 0))
        if state.phase == "moving":
            played.append((state, decision))
    rows = _rows(tmp_path)
    assert not [r for r in rows if r["kind"] in ("error", "fallback")], "the exception reached the guard"
    turns = [r for r in rows if r["kind"] == "turn"]
    assert len(played) == len(turns) == 18
    main = [r for r in turns if not any(s.get("force_switch") for s in _template_of(played, r).get("slots") or [])]
    assert len(calls) == len(main) == 17  # the mixture ran (and raised) on every main turn
    for (state, decision), row in zip(played, turns):
        assert not decision.reasoning_summary.startswith("Fallback"), decision.reasoning_summary
        sheet = row["turn_sheet"]
        assert "reply_table" not in sheet and all("worst" not in c and "best_reply" not in c for c in sheet["candidate_turns"])
        top = sheet["candidate_turns"][0]
        from examples.llm import pokemon as adapters

        expected = adapters.doubles_choice(state.legal_actions[0], state).build({"slot_0": top["slot_0"], "slot_1": top["slot_1"]})
        assert decision.action == expected, (row["turn"], decision.action, expected)
    assert all(any("reply mixture failed this turn (KeyError" in n for n in r["turn_sheet"]["notes"]) for r in main)


def _template_of(played: list, row: dict) -> dict:
    state = next(s for s, _ in played if s.state_version == row["state_version"])
    return state.legal_actions[0].input.get("action") or {}


# -- 2. our last Pokémon's Protect with nothing to wait for (model audit R1 / R7) ------------------------------------------


LONE_PROTECT_PREDICTION = {"top_column": {1: "attack", 2: "attack"}}


def _lone_protect(template: dict) -> dict:
    return {"slot_0": {"option": 0, "target": 0}, "slot_1": {"option": _index(template, 1, "protect"), "target": 0}}


@pytest.mark.parametrize("mixture", [0.0, 1.0])
@pytest.mark.parametrize("turn", [8, 9, 10])
def test_a_lone_protect_with_nothing_to_wait_for_earns_nothing_and_never_plays_d9405f02(turn, mixture):
    """d9405f02 T8-T10: Cresselia, our last Pokémon, under LETHAL Shadow Ball, nothing runs out, and none of the reasons
    the sheet's stall list misses applies (Iron Hands' Fake Out is spent, Sitrus Berry, Levitate, no Wish, Leech Seed or
    Yawn). The live answer Protected (T9 robust worse). The menu lists only the attack; the Protect override was credited
    45 and accepted on both builds. Now it earns no credit, and since 0 still clears a 0.15 margin under a best row of
    0.0 (scorer) or below 0 (mixture), it is refused outright, read or no read: the attack row plays."""
    agent, sheet, template, choice = _position("d9405f02", 0, turn, {"reply_mixture": mixture, "reply_seed": 7.0})
    assert sheet.alone and not sheet.stall_reasons and any(w.startswith("LETHAL") and "slot 1 (cresselia)" in w for w in sheet.warnings)
    assert _waits_out(sheet, agent.memory) == [] and not sheet.flinch_risk
    protect = _lone_protect(template)
    payload = choice.build(protect)
    assert override_value(payload, sheet, agent.params, memory=agent.memory) == 0.0
    for read in (None, ("attack", "attack")):  # no read (the live answer's format) and a read matching the code's column
        meta: dict = {}
        value = agent._resolve_turn_answer(_answer(0, override=protect, prediction=read), sheet, choice, LONE_PROTECT_PREDICTION, meta)
        assert meta["resolution"].startswith("override rejected: our last Pokémon's Protect has nothing to wait for"), meta
        assert value["slot_1"]["type"] == "move" and value["slot_1"]["move_id"] != "protect", value
        top = sheet.candidates[0]
        assert value == choice.build({"slot_0": top["slot_0"], "slot_1": top["slot_1"]}) and "lone_protect_waits" not in meta
    # with a row number that is not a row, the rejection is an invalid answer (retried), never the Protect
    with pytest.raises(InvalidChoice):
        agent._resolve_turn_answer(_answer(5, override=protect), sheet, choice, LONE_PROTECT_PREDICTION, {})
    # a stall reason gives the Protect its credit back (the fix is that narrow)
    sheet.stall_reasons = ["their Tailwind runs out (then we move first)"]
    assert override_value(payload, sheet, agent.params, memory=agent.memory) == agent.params["protect_bonus_guaranteed"]


@pytest.mark.parametrize("mixture", [0.0, 1.0])
def test_d9405f02_t7s_lone_protect_waits_out_a_fresh_fake_out_and_stays_allowed(mixture):
    """d9405f02 T7: Iron Hands has just come in, and its Fake Out (+3) flinches Cresselia 90% of the time (the sheet's
    FAKE OUT warning). A Protect blocks it, and from T8 on that Fake Out fails (live it did, twice): this Protect waits
    something out, so the rule does not refuse it. The sheet's stall reasons stay empty (battle._endgame is unchanged);
    the reason is logged, and the margin judges the Protect as on 81c4591: credited 45 under LETHAL, it is accepted.
    (Cresselia's T6 Protect had worked, so this one worked 1 time in 3; live it failed.)"""
    agent, sheet, template, choice = _position("d9405f02", 0, 7, {"reply_mixture": mixture, "reply_seed": 7.0})
    assert sheet.alone and not sheet.stall_reasons and sheet.flinch_risk == {1: 0.9}
    assert any(w.startswith("FAKE OUT: ironhands can Fake Out this turn") for w in sheet.warnings)
    waits = _waits_out(sheet, agent.memory)
    assert waits == ["a fresh Fake Out can flinch our cresselia this turn (90%): Protect blocks it, and from next turn it fails"]
    protect = _lone_protect(template)
    payload = choice.build(protect)
    assert override_value(payload, sheet, agent.params, memory=agent.memory) == agent.params["protect_bonus_guaranteed"]
    meta: dict = {}
    value = agent._resolve_turn_answer(_answer(0, override=protect), sheet, choice, LONE_PROTECT_PREDICTION, meta)
    assert value == payload and meta["resolution"] == "override accepted" and meta["lone_protect_waits"] == waits
    assert agent.memory.protect_streak("cresselia", 1) == 1


def _observed(lines: list[str]) -> MatchMemory:
    """A memory that has read this protocol log (we are p1)."""
    memory = MatchMemory()
    memory.observe_protocol({"protocol_log": ["|player|p1|us", "|player|p2|them"] + lines, "team": {"p1: Wo-Chien": {}}})
    return memory


def test_the_memory_tracks_leech_seed_yawn_and_wish_from_the_protocol_log():
    turn1 = ["|switch|p1a: Wo-Chien|Wo-Chien, L50|100/100", "|switch|p1b: Scream Tail|Scream Tail, L50|100/100",
             "|switch|p2a: Gholdengo|Gholdengo, L50|100/100", "|switch|p2b: Incineroar|Incineroar, L50, M|100/100", "|turn|1",
             "|move|p1a: Wo-Chien|Leech Seed|p2a: Gholdengo", "|-start|p2a: Gholdengo|move: Leech Seed",
             "|move|p1b: Scream Tail|Wish|p1b: Scream Tail", "|move|p2b: Incineroar|Leech Seed|p1a: Wo-Chien", "|-immune|p1a: Wo-Chien",
             "|turn|2"]
    memory = _observed(turn1)
    assert memory.seeded == {"theirs:gholdengo": "mine:a"} and memory.wishes == {"mine:b": 1} and memory.drowsy == {}
    turn2 = ["|switch|p1b: Torkoal|Torkoal, L50|100/100", "|switch|p2a: Iron Hands|Iron Hands, L50|100/100",
             "|move|p1b: Torkoal|Yawn|p2b: Incineroar", "|-start|p2b: Incineroar|move: Yawn|[of] p1b: Torkoal",
             "|move|p1a: Wo-Chien|Leech Seed|p2a: Iron Hands", "|-start|p2a: Iron Hands|move: Leech Seed", "|turn|3"]
    memory = _observed(turn1 + turn2)
    # Gholdengo left the field (its seed with it); the Wish stays with its position; Yawn makes Incineroar drowsy
    assert memory.seeded == {"theirs:ironhands": "mine:a"} and memory.wishes == {"mine:b": 1}
    assert memory.drowsy == {"theirs:incineroar": 2} and memory.log_turn == 3
    turn3 = ["|move|p2a: Iron Hands|Drain Punch|p1a: Wo-Chien", "|-end|p2b: Incineroar|move: Yawn|[silent]",
             "|-status|p2b: Incineroar|slp|[from] move: Yawn", "|-damage|p2a: Iron Hands|0 fnt", "|faint|p2a: Iron Hands", "|turn|4"]
    memory = _observed(turn1 + turn2 + turn3)
    assert memory.seeded == {} and memory.drowsy == {}


def _mon(species: str, side: str, position: int, types: list[str], *, item=None, ability=None, status=None):
    return battle_rules.Mon(species=species, side=side, position=position, types=types, stats={"spe": 100}, hp_fraction=0.5,
                            status=status, item=item, ability=ability, boosts={}, moves=[])


def _alone(me, *foes, turn: int = 6, terrain: str | None = None, flinch: dict | None = None):
    sheet = battle_rules.TurnSheet()
    sheet.alone, sheet.turn, sheet.flinch_risk = True, turn, dict(flinch or {})
    sheet.field = battle_rules.FieldState(actives=[me, *foes], terrain=terrain)
    return sheet


def test_waits_out_names_each_reason_the_stall_list_misses_and_nothing_else():
    foe = _mon("Gholdengo", "theirs", 1, ["steel", "ghost"], item="Air Balloon")
    plain = _mon("Cresselia", "mine", 1, ["psychic"], item="Sitrus Berry", ability="Levitate")
    assert _waits_out(_alone(plain, foe)) == [] and _waits_out(_alone(plain, foe), MatchMemory()) == []
    sheet = _alone(plain, foe)
    sheet.alone = False
    assert _waits_out(sheet, MatchMemory()) == []  # only our last Pokémon, alone
    # the sheet: a fresh Fake Out, Black Sludge on a Poison type, a Guts holder's unactivated orb
    assert _waits_out(_alone(plain, foe, flinch={1: 0.9}))[0].startswith("a fresh Fake Out can flinch our Cresselia this turn (90%)")
    assert _waits_out(_alone(_mon("Toxapex", "mine", 0, ["poison", "water"], item="Black Sludge"), foe)) == ["our Toxapex heals a little (Black Sludge)"]
    assert _waits_out(_alone(_mon("Arcanine", "mine", 0, ["fire"], item="Black Sludge"), foe)) == []  # it hurts a non-Poison holder
    guts = _mon("Ursaluna", "mine", 0, ["ground", "normal"], item="Flame Orb", ability="Guts")
    assert _waits_out(_alone(guts, foe)) == ["our Ursaluna's Flame Orb activates at the end of this turn (Guts boosts its Attack from then on)"]
    for spent in (_mon("Ursaluna", "mine", 0, ["ground", "normal"], item="Flame Orb", ability="Guts", status="brn"),
                  _mon("Ursaluna", "mine", 0, ["ground", "normal"], item=None, ability="Guts"),
                  _mon("Ursaluna", "mine", 0, ["ground", "normal"], item="Flame Orb", ability="Bulletproof")):
        assert _waits_out(_alone(spent, foe)) == []
    # the memory: our Wish from last turn at this position, our Leech Seed draining into it, a foe drowsy since last turn
    memory = MatchMemory()
    memory.log_turn = 6
    memory.wishes = {"mine:b": 5}
    assert _waits_out(_alone(plain, foe), memory) == ["our Wish from last turn heals Cresselia at the end of this turn"]
    memory.wishes = {"mine:b": 4, "mine:a": 5}  # two turns ago (it landed already), or the other position
    assert _waits_out(_alone(plain, foe), memory) == []
    memory.wishes, memory.seeded = {}, {"theirs:gholdengo": "mine:b"}
    assert _waits_out(_alone(plain, foe), memory) == ["Leech Seed drains their Gholdengo into our Cresselia every turn"]
    memory.seeded = {"theirs:gholdengo": "mine:a"}  # the seeder's position is empty now: the seed drains nothing
    assert _waits_out(_alone(plain, foe), memory) == []
    memory.seeded, memory.drowsy = {}, {"theirs:gholdengo": 5}
    assert _waits_out(_alone(plain, foe), memory) == ["their Gholdengo falls asleep at the end of this turn (Yawn)"]
    grounded = _mon("Iron Hands", "theirs", 2, ["fighting", "electric"])
    memory.drowsy = {"theirs:ironhands": 5}
    assert _waits_out(_alone(plain, grounded, terrain="electric"), memory) == []  # Electric Terrain keeps it awake
    assert _waits_out(_alone(plain, _mon("Iron Hands", "theirs", 2, ["fighting", "electric"], status="par")), memory) == []
    memory.drowsy = {"theirs:gholdengo": 4}  # stale: that Yawn resolved at the end of last turn
    assert _waits_out(_alone(plain, foe), memory) == []


def test_each_reason_to_wait_lets_a_lone_protect_be_judged_by_the_margin_d9405f02_t9():
    """At T9 the lone Protect is refused; give the position one reason to wait at a time and the margin judges it again
    (credited 45 under LETHAL: accepted). Remove it and the refusal is back."""
    agent, sheet, template, choice = _position("d9405f02", 0, 9)
    protect = _lone_protect(template)
    me = next(m for m in sheet.field.actives if m.side == "mine")
    assert agent.memory.log_turn == 9 and me.position == 1

    def resolution() -> dict:
        meta: dict = {}
        agent._resolve_turn_answer(_answer(0, override=protect), sheet, choice, LONE_PROTECT_PREDICTION, meta)
        return meta

    assert resolution()["resolution"].startswith("override rejected: our last Pokémon's Protect has nothing to wait for")
    for name, value in (("wishes", {"mine:b": 8}), ("seeded", {"theirs:gholdengo": "mine:b"}), ("drowsy", {"theirs:ironhands": 8})):
        setattr(agent.memory, name, value)
        meta = resolution()
        assert meta["resolution"] == "override accepted" and len(meta["lone_protect_waits"]) == 1, (name, meta)
        setattr(agent.memory, name, {})
    me.item, me.types = "Black Sludge", ["poison"]
    meta = resolution()
    assert meta["resolution"] == "override accepted" and meta["lone_protect_waits"] == ["our cresselia heals a little (Black Sludge)"]
    me.item, me.types = "Sitrus Berry", ["psychic"]
    assert resolution()["resolution"].startswith("override rejected: our last Pokémon's Protect has nothing to wait for")


# -- 3. answer_rules_v2: on the mixture the model chooses rows only (model audit R1) ----------------------------------------

MIX = {"reply_mixture": 1.0, "reply_seed": 7.0}
RULES = {**MIX, "answer_rules_v2": 1.0}
TOP = {"top_column": {1: "attack", 2: "attack"}}


def _unlisted(template: dict, rows: list[dict], build) -> dict:
    """A legal joint action that is not one of the rows (the first found, in option order)."""
    listed = {(json.dumps(r["slot_0"], sort_keys=True), json.dumps(r["slot_1"], sort_keys=True)) for r in rows}
    per_slot = []
    for number in (0, 1):
        options = next(s for s in template["slots"] if s.get("slot") == number)["options"]
        per_slot.append([{"option": i, "target": t} for i, o in enumerate(options) for t in (o.get("targets") or [0]) if t >= 0])
    for a in per_slot[0]:
        for b in per_slot[1]:
            if (json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)) in listed:
                continue
            try:
                build({"slot_0": a, "slot_1": b})
            except InvalidChoice:
                continue
            return {"slot_0": a, "slot_1": b}
    raise AssertionError("no unlisted legal joint action")


def _plays(choice, row: dict) -> dict:
    return choice.build({"slot_0": row["slot_0"], "slot_1": row["slot_1"]})


@pytest.fixture(scope="module")
def t6_rules():
    """bcbaabe2 T6 on the mixture (seed 7): rows double_1 61.5, finish_the_other 28.2, switch_1_risky 10.8, switch_0 -2.6,
    focus_fire -6.2. The live play was focus_fire: both attacks into a fresh Protect (robust worse)."""
    return _position("bcbaabe2", 0, 6, RULES)


def test_rows_only_lower_rows_stand_within_50_points_where_the_old_rule_sent_them_back(t6_rules):
    agent, sheet, template, choice = t6_rules
    rows = sheet.candidates
    assert agent._rows_only() and sheet.reply and [r["score"] for r in rows] == [61.5, 28.2, 10.8, -2.6, -6.2]
    old = PokemonAgent(None, version="mpath", log_dir=None, capture_dir="", log=lambda *_: None, params=tuning.merged({**FAST, **MIX}))
    # row 1 trails by 33.3: the old margin (15% of 61.5 = 9.2) sent it back to row 0; 50 points lets it stand
    meta: dict = {}
    assert agent._resolve_turn_answer(_answer(1), sheet, choice, TOP, meta) == _plays(choice, rows[1]) and meta["resolution"] == "row"
    assert meta["margin"] == 50.0 and meta["row_value"] == 28.2 and "row_played" not in meta
    meta = {}
    assert old._resolve_turn_answer(_answer(1), sheet, choice, TOP, meta) == _plays(choice, rows[0]) and meta["row_played"] == 0
    # row 2 trails by 50.7 and the live play (row 4) by 67.7: row 0 plays
    for row in (2, 4):
        meta = {}
        assert agent._resolve_turn_answer(_answer(row), sheet, choice, TOP, meta) == _plays(choice, rows[0])
        assert meta["row_played"] == 0 and meta["resolution"].startswith(f"row {row} rejected") and "by more than 50.0" in meta["resolution"]


def test_rows_only_margin_is_the_larger_of_the_points_and_15_percent_of_row_0():
    """07a3cb1f T4 on the mixture: rows -401.8 and -403.7. 15% of 401.8 is 60.3, more than 50: row 1 stands (the old
    margin was 0.15 points); a row trailing by 55.2 stands, by 61.2 does not; row_margin_points sets the floor."""
    agent, sheet, template, choice = _position("07a3cb1f", 1, 4, RULES)
    rows = sheet.candidates
    assert [r["score"] for r in rows] == [-401.8, -403.7]
    meta: dict = {}
    assert agent._resolve_turn_answer(_answer(1), sheet, choice, TOP, meta) == _plays(choice, rows[1]) and meta["margin"] == 60.3
    for score, stands in ((-457.0, True), (-463.0, False)):
        rows[1]["score"] = score
        meta = {}
        played = agent._resolve_turn_answer(_answer(1), sheet, choice, TOP, meta)
        assert (played == _plays(choice, rows[1])) is stands and (meta.get("row_played") == 0) is (not stands), (score, meta)
    rows[0]["score"], rows[1]["score"] = 61.5, 30.0
    agent.params["row_margin_points"] = 25.0  # the audit's cautious alternative: 31.5 behind is too far
    meta = {}
    assert agent._resolve_turn_answer(_answer(1), sheet, choice, TOP, meta) == _plays(choice, rows[0]) and meta["margin"] == 25.0


def test_rows_only_ignores_an_override_logs_it_and_plays_the_chosen_row(t6_rules):
    agent, sheet, template, choice = t6_rules
    rows = sheet.candidates
    proposed = _unlisted(template, rows, choice.build)
    meta: dict = {}
    played = agent._resolve_turn_answer(_answer(1, override=proposed, prediction=("attack", "attack")), sheet, choice, TOP, meta)
    assert played == _plays(choice, rows[1]) and meta["resolution"].startswith("override ignored")
    assert meta["override_ignored"] == {**proposed, "same_as_row": None, "legal": True}
    # an illegal override is ignored as well: no retry is spent on it
    meta = {}
    illegal = {"slot_0": {"option": 0, "target": 9}, "slot_1": rows[0]["slot_1"]}
    assert agent._resolve_turn_answer(_answer(0, override=illegal), sheet, choice, TOP, meta) == _plays(choice, rows[0])
    assert meta["override_ignored"]["legal"] is False
    # an override with a row that trails too far: ignored, then the row goes back to row 0
    meta = {}
    assert agent._resolve_turn_answer(_answer(4, override=proposed), sheet, choice, TOP, meta) == _plays(choice, rows[0])
    assert meta["resolution"].startswith("override ignored") and "; row 4 rejected" in meta["resolution"] and meta["row_played"] == 0
    # with a row number that is not a row, the answer is invalid (retried); the override never plays
    with pytest.raises(InvalidChoice, match="plays only listed rows"):
        agent._resolve_turn_answer(_answer(7, override=proposed), sheet, choice, TOP, {})
    # no rows at all: a proposed joint action is the only answer, validated and played
    saved, sheet.candidates = sheet.candidates, []
    try:
        meta = {}
        assert agent._resolve_turn_answer(_answer(0, override=proposed), sheet, choice, TOP, meta) == choice.build(proposed)
        assert meta["resolution"] == "override accepted (no rows)"
        with pytest.raises(InvalidChoice):
            agent._resolve_turn_answer(_answer(0), sheet, choice, TOP, {})
    finally:
        sheet.candidates = saved


def test_rows_only_has_no_read_bypass(t6_rules):
    """The old hook let a trailing row stand when the read matched the code's column and the reply table (keyed by row
    name, which the mixture repeats) rated it at least as well. Under answer_rules_v2 the margin alone decides."""
    agent, sheet, template, choice = t6_rules
    rows = sheet.candidates
    names = [r["name"] for r in rows]
    table = sheet.reply
    try:
        rows[0]["name"], rows[4]["name"] = "a_row", "b_row"
        sheet.reply = {**table, "rows": [{"name": "b_row", "value": 99.0}, {"name": "a_row", "value": 1.0}]}
        old = PokemonAgent(None, version="mpath", log_dir=None, capture_dir="", log=lambda *_: None, params=tuning.merged({**FAST, **MIX}))
        meta: dict = {}
        assert old._resolve_turn_answer(_answer(4, prediction=("attack", "attack")), sheet, choice, TOP, meta) == _plays(choice, rows[4])
        assert "stands on the opponent read" in meta["resolution"]
        meta = {}
        assert agent._resolve_turn_answer(_answer(4, prediction=("attack", "attack")), sheet, choice, TOP, meta) == _plays(choice, rows[0])
        assert meta["row_played"] == 0
    finally:
        for r, n in zip(rows, names):
            r["name"] = n
        sheet.reply = table


def test_answer_rules_v2_leaves_the_scorer_path_alone():
    """With reply_mixture 0 the key changes nothing: an override is scored and judged by the E2 rules as before."""
    agent, sheet, template, choice = _position("bcbaabe2", 0, 13, {"reply_mixture": 0.0, "answer_rules_v2": 1.0})
    assert not agent._rows_only() and not sheet.reply
    rows = sheet.candidates
    meta: dict = {}
    same = {"slot_0": rows[0]["slot_0"], "slot_1": rows[0]["slot_1"]}
    agent._resolve_turn_answer(_answer(1, override=same), sheet, choice, TOP, meta)
    assert meta["resolution"] == "override accepted (same as a row)" and "override_ignored" not in meta
    weak = {"slot_0": {"option": _index(template, 0, "flashcannon"), "target": 2}, "slot_1": {"option": _index(template, 1, "snarl"), "target": 0}}
    meta = {}
    agent._resolve_turn_answer(_answer(0, override=weak, prediction=("protect", "switch")), sheet, choice, TOP, meta)
    assert meta["resolution"].startswith("override rejected: value") and "override_ignored" not in meta


# -- the prompt and the payload say the rule in force; the defaults are byte for byte the Oct 10 deploy's ----------------

SYSTEM_PROMPT_SHA256 = "ac6102af4c7648bc20ccbe6400b69ee875de6586f11cd63c41c5bbea904d4763"  # 81c4591's SYSTEM_PROMPT (8,593 chars)
ANSWER_FORMAT_81C4591 = (
    "Answer with row (a candidate_turns row number) plus opponent_prediction and win_condition. To propose a joint "
    "action that is not a row, set override.use=true and give both slots as {option: <index from that slot's "
    "options>, target: <integer>}; the code scores it against the rows before it may play. If the chosen option's "
    "targets list is non-empty, target must be one of those integers; otherwise send target 0."
)
INSTRUCTIONS_81C4591 = "See answer_format: answer with a row, an opponent_prediction and a win_condition (override only with a reason)."


class _Recorder:
    """A scripted model that records every message list it is sent."""

    model = "fake-model"

    def __init__(self, answer: dict) -> None:
        self.answer, self.calls = answer, []
        self.last_model, self.last_latency_ms, self.last_usage = "fake-model", 1, None

    def complete_structured(self, messages, schema_name, schema):
        self.calls.append([dict(m) for m in messages])
        return dict(self.answer)


def _ask_at(prefix: str, seat: int, turn: int, params: dict, answer: dict) -> tuple[_Recorder, dict]:
    """Replay code-only up to a battle decision, then let a recording fake model answer it: (recorder, decision)."""
    folder = _folder(prefix)
    agent = PokemonAgent(None, version="mpath", log_dir=Path(tempfile.mkdtemp()), capture_dir="", log=lambda *_: None,
                         params=tuning.merged({**FAST, **params}))
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        if state.phase == "moving" and observation_dict(state).get("turn") == turn:
            recorder = _Recorder(answer)
            agent._provider = recorder
            return recorder, agent.choose_action(state, _context(folder, raw, seat))
        agent.choose_action(state, _context(folder, raw, seat))
    raise AssertionError(f"turn {turn} not in {prefix}")


def test_the_defaults_send_the_deploys_prompt_and_payload_texts_byte_for_byte():
    import hashlib

    from agent.pokemon.prompts import SYSTEM_PROMPT, system_prompt

    assert system_prompt(dict(tuning.DEFAULTS)) is SYSTEM_PROMPT or system_prompt(dict(tuning.DEFAULTS)) == SYSTEM_PROMPT
    assert hashlib.sha256(system_prompt(dict(tuning.DEFAULTS)).encode()).hexdigest() == SYSTEM_PROMPT_SHA256
    # the rows-only bullets need the mixture: on the scorer path the key leaves the prompt alone
    assert system_prompt({**tuning.DEFAULTS, "answer_rules_v2": 1.0}) == SYSTEM_PROMPT
    recorder, _ = _ask_at("bcbaabe2", 0, 13, {}, _answer(0))
    messages = recorder.calls[0]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    payload = json.loads(messages[-1]["content"])
    assert payload["answer_format"] == ANSWER_FORMAT_81C4591 and payload["instructions"] == INSTRUCTIONS_81C4591


def test_rows_only_prompt_and_payload_say_the_rule_in_force():
    from agent.pokemon.prompts import SYSTEM_PROMPT, system_prompt

    text = system_prompt(tuning.merged(RULES))
    assert "- override: leave use=false. This build plays only listed rows" in text
    assert "A row whose value trails row 0 by more than 50 points plays row 0 instead." in text
    assert "The code scores it with the same scorer as the rows" not in text
    assert text.replace(text[text.index("- row:"):text.index("- opponent_prediction")], "") == \
        SYSTEM_PROMPT.replace(SYSTEM_PROMPT[SYSTEM_PROMPT.index("- row:"):SYSTEM_PROMPT.index("- opponent_prediction")], "")
    assert "by more than 25 points" in system_prompt(tuning.merged({**RULES, "row_margin_points": 25}))
    recorder, decision = _ask_at("bcbaabe2", 0, 13, RULES, _answer(0))
    messages = recorder.calls[0]
    assert messages[0] == {"role": "system", "content": text}
    payload = json.loads(messages[-1]["content"])
    assert payload["answer_format"].startswith("Answer with row") and "Leave override.use=false" in payload["answer_format"]
    assert payload["instructions"].endswith("(override only when there are no rows).")
    assert payload["turn_sheet"]["reply_table"] and decision.action["type"] == "doubles_turn"


# -- 4. recheck_v2: a recheck only where the answer leaves row 0's cover (model audit R3, readiness item 10) ---------------


class _Warned:
    """A sheet with an EXPOSED note on slot 0 and a DOUBLE-TARGET warning on slot 1 (the shape of the live texts)."""

    warnings = ["DOUBLE-TARGET: if both foes go into slot 1 (tornadus) — incineroar's flareblitz + gholdengo's makeitrain = 147-172% — "
                "it goes down. Opponents double the target that threatens them most; Protect or switch if that is this slot."]
    notes = ["EXPOSED: slot 0 (lucario) moves before basculegion, but basculegion's wavecrash KOs it (303.2-356.7%) afterwards unless "
             "basculegion is removed, flinched, redirected or this slot Protects/switches."]
    threats: list = []
    our_options: list = []
    speed_order: list = []
    switch_ins: dict = {}
    alone = False
    stall_reasons: list = []

    def __init__(self, candidates: list | None = None) -> None:
        self.candidates = candidates or []


_WARNED_TEMPLATE = {"slots": [
    {"slot": 0, "options": [{"type": "move", "move_id": "closecombat", "targets": [1, 2]}, {"type": "move", "move_id": "protect", "targets": []}]},
    {"slot": 1, "options": [{"type": "move", "move_id": "bleakwindstorm", "targets": []}, {"type": "move", "move_id": "protect", "targets": []}]},
]}
_ATTACK = {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": "closecombat", "target": 1},
           "slot_1": {"type": "move", "move_id": "bleakwindstorm"}}
V2 = {"recheck_v2": 1.0}


def test_recheck_v2_drops_double_target_and_needs_row_0_to_cover_the_slot():
    from agent.agent import _lethal_recheck

    old = _lethal_recheck(_ATTACK, _Warned(), template=_WARNED_TEMPLATE)
    assert "slot 0 attacks with closecombat although: EXPOSED" in old and "although: DOUBLE-TARGET" in old
    assert old.endswith("public description of the move itself.") and "otherwise take the alternative row." in old
    # recheck_v2 with row 0 attacking with both slots: the code priced both warnings, nothing fires
    attacking_row = {"name": "best_attacks", "slot_0": {"option": 0, "target": 1}, "slot_1": {"option": 0, "target": 0}, "why": "attack"}
    info: dict = {}
    assert _lethal_recheck(_ATTACK, _Warned([attacking_row]), template=_WARNED_TEMPLATE, info=info, params=V2) is None
    assert info["recheck_skips"] == ["slot 0: row 0 does not Protect or switch this slot either (the code priced the warning)"]
    # row 0 Protects slot 0: the EXPOSED slot fires, the DOUBLE-TARGET one does not, and the model is sent back to row 0
    cover = {"name": "protect_threatened", "slot_0": {"option": 1, "target": 0}, "slot_1": {"option": 0, "target": 0},
             "why": "basculegion's wavecrash can KO slot 0 after it moves; Protect it, slot 1 attacks"}
    info = {}
    message = _lethal_recheck(_ATTACK, _Warned([cover, attacking_row]), template=_WARNED_TEMPLATE, info=info, params=V2)
    assert message.startswith("Recheck before this is final. slot 0 attacks with closecombat although: EXPOSED") and "DOUBLE-TARGET" not in message
    assert ("otherwise choose row 0. Row 0 takes that slot out of the line of fire: row 0 protect_threatened: slot 0 option 1 (protect), "
            "slot 1 option 0 (bleakwindstorm) — basculegion's wavecrash can KO slot 0") in message and "alternative" not in message
    assert info["recheck_problems"] and info["recheck_outcome"].startswith("fired: slot 0")
    # answering row 0 (or any cover of that slot) never fires
    protect = {**_ATTACK, "slot_0": {"type": "move", "move_id": "protect"}}
    assert _lethal_recheck(protect, _Warned([cover, attacking_row]), template=_WARNED_TEMPLATE, params=V2) is None


def _recheck_at(prefix: str, seat: int, turn: int, answer_row: int, params: dict) -> tuple[str | None, dict, list]:
    from agent.agent import _lethal_recheck

    agent, sheet, template, choice = _position(prefix, seat, turn)
    row = sheet.candidates[answer_row]
    value = choice.build({"slot_0": row["slot_0"], "slot_1": row["slot_1"]})
    info: dict = {}
    return _lethal_recheck(value, sheet, memory=agent.memory, template=template, info=info, params=params), info, sheet.candidates


def test_recheck_v2_spends_no_second_call_on_a_row_0_answer_07a3cb1f_t1():
    """07a3cb1f T1 (the scorer, the deploy): row 0 is Wave Crash + Follow Me, attacking with Basculegion under a
    DOUBLE-TARGET warning, and the live answer was row 0. The old rule sent it back (a second call that cannot change the
    move: the quoted cover trails row 0 far beyond the margin); recheck_v2 does not fire. On the replayed live answers at
    reply_mixture 0, 26 of the 43 firings were row-0 answers; recheck_v2 leaves 2 firings."""
    message, info, rows = _recheck_at("07a3cb1f", 1, 1, 0, {})
    assert rows[0]["name"] == "best_attacks" and message and "although: DOUBLE-TARGET" in message
    quoted = next(i for i, r in enumerate(rows) if f"row {i} {r['name']}: slot 0" in message)
    assert rows[quoted]["score"] < rows[0]["score"] * (1 - tuning.DEFAULTS["override_margin"])  # the margin sends it back to row 0
    message, info, rows = _recheck_at("07a3cb1f", 1, 1, 0, V2)
    assert message is None


def test_recheck_v2_still_sends_back_an_attack_that_leaves_row_0s_protect_7ce71025_t1():
    """7ce71025 T1: row 0 Protects the EXPOSED Great Tusk (Hyper Voice KOs it after it moves) and sets Tailwind; the live
    answer was row 1, Headlong Rush + Tailwind (within the scorer's margin). Both rules fire; recheck_v2 names row 0."""
    old, _, rows = _recheck_at("7ce71025", 0, 1, 1, {})
    assert rows[0]["name"] == "protect_threatened" and rows[1]["name"] == "best_attacks"
    assert old and "slot 0 attacks with headlongrush although: EXPOSED" in old
    message, info, _ = _recheck_at("7ce71025", 0, 1, 1, V2)
    assert message and "slot 0 attacks with headlongrush although: EXPOSED" in message
    assert "otherwise choose row 0. Row 0 takes that slot out of the line of fire: row 0 protect_threatened: slot 0 option 3 (protect)" in message


# -- 5. prompt_v2 and prompt_kb: the audit's prompt text and knowledge base (model audit R2, R4, R5) -----------------------


def test_prompt_v2_makes_double_target_a_risk_leads_with_the_live_lessons_and_defines_the_prediction_words():
    from agent.pokemon.prompts import SYSTEM_PROMPT, system_prompt

    text = system_prompt(tuning.merged({"prompt_v2": 1}))
    assert ("The Protect procedure, in this order, for a slot under a LETHAL or EXPOSED warning. DOUBLE-TARGET alone is a risk, "
            "not a forecast (both foes must pick that slot): cover it only when row 0 does.\n1. A priority escape THAT KOS") in text
    assert "for a slot under a LETHAL, EXPOSED or DOUBLE-TARGET warning" not in text
    lessons = text[text.index("Lessons from live games:\n"):text.index("\n\nAnswer only in the required JSON.")].splitlines()
    assert lessons[1].startswith("- Leaving row 0 to attack where it Protects or switches paid off")
    assert lessons[2].startswith("- Adding a Protect or a switch that row 0 does not make lost more than it saved")
    assert lessons[3] == "- Moving an attack from row 0's target to a foe that can Protect or switch out lost more than it gained."
    old = SYSTEM_PROMPT[SYSTEM_PROMPT.index("Lessons from overrides that lost (Oct 8):\n"):SYSTEM_PROMPT.index("\n\nAnswer only")].splitlines()
    assert lessons[4:] == old[1:] and "Lessons from overrides that lost" not in text  # the existing lessons follow, unchanged
    assert ("- opponent_prediction: what you expect each of their slots to do this turn, in the code's words, and why in one clause. "
            "attack = a damaging move, spread moves included. setup = Tailwind, Trick Room, Icy Wind, Electroweb, Snarl, Reflect") in text
    assert "Predict switch when that Pokémon faces our likely KO, cannot Protect (it just did) and has a Pokémon seen in back" in text
    assert "(their_slot_1 / their_slot_2: attack, protect, switch, setup, fakeout, none)" not in text
    # the setup words are the code's own (the code predicts setup for exactly these, plus the stat boosts named)
    from agent.pokemon.opponent import SETUP_MOVES

    named = {"tailwind", "trickroom", "icywind", "electroweb", "snarl", "reflect", "lightscreen", "nastyplot", "calmmind",
             "swordsdance", "dragondance", "bulkup", "quiverdance"}
    assert named == set(SETUP_MOVES)


def test_prompt_kb_is_the_audits_ten_facts_checked_against_the_mechanics_data():
    from agent.pokemon import data, preview_sim
    from agent.pokemon.prompts import KNOWLEDGE_BASE, SYSTEM_PROMPT, system_prompt

    text = system_prompt(tuning.merged({"prompt_kb": 1}))
    assert text.endswith("\n\n" + KNOWLEDGE_BASE + "\n\nAnswer only in the required JSON.")
    assert text.replace("\n\n" + KNOWLEDGE_BASE, "") == SYSTEM_PROMPT  # nothing else moves
    assert len(KNOWLEDGE_BASE.split()) < 450 and [line[:3] for line in KNOWLEDGE_BASE.splitlines()[1:]] == [f"{i}. " for i in range(1, 10)] + ["10."]
    # the facts against the repo's mechanics data
    priority = {m: (data.move_info(m) or {}).get("priority") for m in ("fakeout", "extremespeed", "suckerpunch", "iceshard", "aquajet")}
    assert priority == {"fakeout": 3, "extremespeed": 2, "suckerpunch": 1, "iceshard": 1, "aquajet": 1}
    assert "Fake Out (+3 priority)" in KNOWLEDGE_BASE and "Extreme Speed (+2)" in KNOWLEDGE_BASE and "Ice Shard and Aqua Jet (+1)" in KNOWLEDGE_BASE
    assert (data.move_info("earthquake") or {}).get("target") == "allAdjacent"  # "Earthquake also hits your ally"
    assert {"spore", "ragepowder"} <= set(data.POWDER_MOVES)  # Grass types, Overcoat and Safety Goggles ignore both
    assert (preview_sim.SCREEN_TURNS, preview_sim.LIGHT_CLAY_TURNS) == (5, 8) and "eight from a Light Clay holder" in KNOWLEDGE_BASE
    by_id, _ = data._catalog()
    assert any(c.get("item") == "Light Clay" and {"Reflect", "Light Screen"} & set(c.get("moves") or []) for c in by_id.values())
    # general mechanics and behaviour only: no opponent, player or class is named
    import re

    assert not re.findall(r"\b(search\w*|bots?|llm|entrants?|players?|opponent_model)\b", KNOWLEDGE_BASE.lower())


def test_the_arms_prompts_are_identical_on_every_call_and_the_defaults_stay_the_deploys():
    """Arm C's system prompt (all four model-path keys on the mixture) is one text for the whole match: the first system
    message of two decisions is byte-identical, and it carries all three changes."""
    from agent.pokemon.prompts import KNOWLEDGE_BASE, system_prompt

    arm_c = {**RULES, "recheck_v2": 1.0, "prompt_v2": 1.0, "prompt_kb": 1.0}
    first, _ = _ask_at("bcbaabe2", 0, 13, arm_c, _answer(0))
    second, _ = _ask_at("bcbaabe2", 0, 14, arm_c, _answer(0))
    text = system_prompt(tuning.merged(arm_c))
    assert first.calls[0][0] == second.calls[0][0] == {"role": "system", "content": text}
    assert KNOWLEDGE_BASE in text and "Lessons from live games:" in text and "This build plays only listed rows" in text


# -- 6. the live-position harness (scripts/live_positions.py): no model call unless asked, on a budget ---------------------


def _harness():
    import importlib.util
    import sys

    module = sys.modules.get("live_positions")
    if module is None:
        spec = importlib.util.spec_from_file_location("live_positions", Path(__file__).resolve().parent.parent / "scripts" / "live_positions.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["live_positions"] = module
        spec.loader.exec_module(module)
    return module


_TEMPLATE = {"slots": [
    {"slot": 0, "options": [{"type": "move", "move_id": "wavecrash", "targets": [1, 2]}, {"type": "move", "move_id": "protect", "targets": [0]},
                            {"type": "switch", "species": "ironhands"}]},
    {"slot": 1, "options": [{"type": "move", "move_id": "followme", "targets": [0]}, {"type": "move", "move_id": "dazzlinggleam", "targets": [0]}]},
]}


def test_the_harness_translates_a_live_play_into_a_row_or_a_proposed_joint_action():
    lp = _harness()
    rows = [{"row": 0, "slot_0": {"option": 1, "target": 0}, "slot_1": {"option": 0, "target": 0}},
            {"row": 1, "slot_0": {"option": 0, "target": 2}, "slot_1": {"option": 1, "target": 0}}]
    played = {"slot_0": {"type": "move", "move_id": "wavecrash", "target": 2}, "slot_1": {"type": "move", "move_id": "dazzlinggleam"}}
    assert lp.payload_to_answers(_TEMPLATE, played) == {"slot_0": {"option": 0, "target": 2}, "slot_1": {"option": 1, "target": 0}}
    assert lp.joint_key(_TEMPLATE, rows[1]) == "0:2|1:0" and lp.same_key("0:2|-", "0:2|1:0") and not lp.same_key("0:1|1:0", "0:2|1:0")
    assert lp.posture(_TEMPLATE, "1:0|0:0") == ["protect", "attack"] and lp.posture(_TEMPLATE, "2:0|1:0") == ["switch", "attack"]
    old_format = {"payload": {"type": "doubles_turn", **played}, "answer": {"win_condition": "w", "reasoning_summary": "r",
                                                                         "slot_0": {"option": 0, "target": 2}}}
    assert lp.translate(old_format, _TEMPLATE, rows) == {"win_condition": "w", "reasoning_summary": "r", "row": 1,
                                                         "override": {"use": False, "slot_0": {"option": 0, "target": 2}, "slot_1": {"option": 1, "target": 0}}}
    switch = {"payload": {"slot_0": {"type": "switch", "species": "Iron Hands"}, "slot_1": {"type": "move", "move_id": "followme", "target": 0}},
              "answer": {"row": 1, "override": {"use": True}, "opponent_prediction": {"their_slot_1": "attack"},
                         "turn_sheet": {}}}
    switch["turn_sheet"] = {"candidate_turns": [rows[0], rows[1]]}
    out = lp.translate(switch, _TEMPLATE, rows)
    # not listed: a proposed joint action; today's format keeps its own row (row 1's joint action is listed as row 1)
    assert out["override"] == {"use": True, "slot_0": {"option": 2, "target": 0}, "slot_1": {"option": 0, "target": 0}} and out["row"] == 1
    assert out["opponent_prediction"] == {"their_slot_1": "attack"}


def _own_log(tmp_path: Path, prefix: str = "07a3cb1f", seat: int = 1) -> Path:
    """A code-only agent plays a captured match and writes its own decision log: the 'live' answers to replay."""
    folder = _folder(prefix)
    logs = tmp_path / "logs"
    agent = PokemonAgent(None, version="mpath", log_dir=logs, capture_dir="", log=lambda *_: None, params=tuning.merged(FAST))
    for path in _sequence(folder, seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if state.legal_actions:
            agent.choose_action(state, _context(folder, raw, seat))
    return logs


def test_the_harness_replays_a_match_exactly_and_never_calls_a_model(tmp_path, monkeypatch, capsys):
    """Replaying a code-only match's own decision log rebuilds every sheet: row 0 is the played move at every main
    position, nothing falls back. Without --model-on, and with --model-on but no --max-calls, no provider is built."""
    import agent.llm.anthropic_provider as anthropic_provider

    monkeypatch.setattr(anthropic_provider, "provider_from_env", lambda **_: pytest.fail("a provider was built"))
    lp = _harness()
    logs = _own_log(tmp_path)
    out = tmp_path / "replay.jsonl"
    assert lp.main(["--logs", str(logs), "--audit", str(tmp_path / "no-audit"), "--out", str(out)]) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    records, summary = lines[:-1], lines[-1]["summary"]
    assert len(records) == 5 and summary["main"] == 4 and summary["replacements"] == 1  # 07a3cb1f: four turns, one replacement
    assert summary["row0_is_live_play"] == 4 and summary["departures"] == 0
    assert summary["fallbacks"] == 0 and summary["model_calls"] == 0 and summary["model_answers"] == 0
    assert all(r["source"] == "replay" and r["stands"] and r["plays_row0"] and r["calls"] for r in records)
    assert summary["verified_rules"]["of"] >= 1  # this match's documented positions are judged on the replayed plays
    assert lp.main(["--logs", str(logs), "--audit", str(tmp_path / "no-audit"), "--model-on", "--out", str(tmp_path / "dry.jsonl")]) == 0
    printed = capsys.readouterr().out
    assert "MODEL ON: 4 main positions x 1 call = 4 calls planned" in printed and "estimated $0.60" in printed
    assert "dry run: nothing was called" in printed and not (tmp_path / "dry.jsonl").exists()


def test_the_harness_spends_at_most_its_budget_and_logs_each_model_answer(tmp_path, monkeypatch):
    """--model-on --max-calls 2 with a scripted model: two calls are spent, the rest of the positions keep the live
    answer, and each model answer is logged with the keys the audit's evaluator reads."""
    import dotenv

    import agent.llm.anthropic_provider as anthropic_provider

    class Scripted:
        model = "scripted"
        last_model, last_latency_ms, last_usage, last_errors, last_attempts = "scripted", 5, {"input_tokens": 1}, [], []

        def __init__(self) -> None:
            self.calls = 0

        def complete_structured(self, messages, schema_name, schema, *, deadline=None, effort=None):
            self.calls += 1
            return {"row": 0, "override": {"use": False, "slot_0": {"option": 0, "target": 0}, "slot_1": {"option": 0, "target": 0}},
                    "opponent_prediction": {"their_slot_1": "attack", "their_slot_2": "protect", "why": "w"},
                    "win_condition": "w", "reasoning_summary": "Scripted."}

    scripted = Scripted()
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(anthropic_provider, "provider_from_env", lambda **_: scripted)
    lp = _harness()
    logs = _own_log(tmp_path)
    out = tmp_path / "model.jsonl"
    assert lp.main(["--logs", str(logs), "--audit", str(tmp_path / "no-audit"), "--model-on", "--max-calls", "2", "--out", str(out)]) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    records, summary = lines[:-1], lines[-1]["summary"]
    assert scripted.calls == 2 and summary["model_calls"] == 2
    model = [r for r in records if r["source"] == "model"]
    assert len(model) >= 1 and all(r["source"] == "replay" for r in records if r not in model)
    for r in model:
        assert r["answers"][0]["opponent_prediction"]["their_slot_2"] == "protect" and r["row0_key"] and r["final_key"]
        assert r["rows"] and all("key" in row for row in r["rows"]) and r["calls"][0]["source"] == "model"
    assert summary["fallbacks"] == 0
