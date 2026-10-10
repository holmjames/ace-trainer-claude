"""The model path (lane M1): what happens between the model's answer and the move, and what the model is shown.

1. A reply-mixture exception costs the mixture for that turn, never the match: the scorer's rows play (readiness 2B.1).
2. Our last Pokémon's Protect with nothing to wait for earns no override credit and never plays over a row (model
   audit R1/R7; d9405f02 T7-T10). "Nothing to wait for" also checks what the sheet's stall reasons miss (a fresh Fake
   Out, Black Sludge, a Wish, Leech Seed, Yawn on a foe without a Lum or Chesto Berry, a Guts holder's orb), for a
   Protect that is not a repeat; the Fake Out and the orb only when no sure KO lands first. d9405f02 T7's fresh Fake Out
   is neither: refused, as T8-T10.
3. answer_rules_v2 (off by default): on the reply mixture the model chooses rows only; a lower row stands within
   max(row_margin_points, 15% of |row 0|); no read bypass; the prompt and the payload say so (model audit R1). Only on
   rows the mixture priced (a turn it did not price keeps E2); an override that repeats a listed row is that row. When
   the proposal does not play, the public summary is the played row's why, on the deploy's scorer path as well.
4. recheck_v2 (off by default): no DOUBLE-TARGET triggers; LETHAL / EXPOSED fire only where row 0 covers that slot;
   the message sends the model back to row 0 (model audit R3, readiness item 10).
5. prompt_v2 and prompt_kb (off by default): the audit's R2 and R5 prompt text (with the sheet's DOUBLE-TARGET warnings
   ending neutrally) and its R4 knowledge base, general mechanics checked against the mechanics data; every variant is
   built at import and system_prompt never raises; the defaults send the Oct 10 deploy's prompt and payload byte for byte.
6. scripts/live_positions.py: the live answers replayed through the agent (code only), a model-on mode that is a dry
   run without --max-calls and never spends more than it (N calls, up to 2N requests with fallbacks); an ignored
   override is recorded with its key and the evaluator's value.

Positions are replayed from the live captures (tests/fixtures/live/) through a code-only agent; the model is a scripted
fake or absent. No test calls a model.
"""

from __future__ import annotations

import json
import re
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
def test_d9405f02_t7s_lone_protect_is_a_repeat_under_a_ko_that_lands_first_and_is_refused(mixture, monkeypatch):
    """d9405f02 T7: Iron Hands has just come in, and its Fake Out (+3) can flinch Cresselia (90%, the sheet's FAKE OUT
    warning), but that is no reason to wait here (the M1 review). Cresselia Protected on T6, so this one works 1 time in 3
    (the sheet's note; the prompt allows a Protect on a streak only with a listed stall reason), and Gholdengo's Shadow
    Ball KOs Cresselia before it moves (116.9-137.7%), Fake Out or not: the Protect only delays the same KO. The audit
    lists T7 robust worse (the harness evaluator: -53.6 against row 0). So it is refused, read or no read, and the attack
    row plays, as on 3eaaada. Either condition alone drops the Fake Out reason; with neither, it is one."""
    agent, sheet, template, choice = _position("d9405f02", 0, 7, {"reply_mixture": mixture, "reply_seed": 7.0})
    assert sheet.alone and not sheet.stall_reasons and sheet.flinch_risk == {1: 0.9}
    assert any(w.startswith("FAKE OUT: ironhands can Fake Out this turn") for w in sheet.warnings)
    lethal = [w for w in sheet.warnings if w.startswith("LETHAL: gholdengo's shadowball hits slot 1 (cresselia) for 116.9-137.7% BEFORE it can move.")]
    assert len(lethal) == 1 and agent.memory.protect_streak("cresselia", 1) == 1
    assert any(n.startswith("Our cresselia used Protect LAST turn: another one works only 1 time in 3") for n in sheet.notes)
    assert _waits_out(sheet, agent.memory) == []
    protect = _lone_protect(template)
    payload = choice.build(protect)
    assert override_value(payload, sheet, agent.params, memory=agent.memory) == 0.0
    top = sheet.candidates[0]
    for read in (None, ("attack", "attack")):  # no read (the live answer's format) and a read matching the code's column
        meta: dict = {}
        value = agent._resolve_turn_answer(_answer(0, override=protect, prediction=read), sheet, choice, LONE_PROTECT_PREDICTION, meta)
        assert meta["resolution"].startswith("override rejected: our last Pokémon's Protect has nothing to wait for"), meta
        assert value == choice.build({"slot_0": top["slot_0"], "slot_1": top["slot_1"]}) and "lone_protect_waits" not in meta
    # a fresh Protect still meets the Shadow Ball first ...
    monkeypatch.setattr(agent.memory, "protect_streak", lambda species, slot: 0)
    assert _waits_out(sheet, agent.memory) == []
    # ... without that KO, a fresh Protect waits out the Fake Out (logged; the margin judges it again) ...
    fake_out = ["a fresh Fake Out can flinch our cresselia this turn (90%): Protect blocks it, and from next turn it fails"]
    sheet.warnings = [w for w in sheet.warnings if w not in lethal]
    assert _waits_out(sheet, agent.memory) == fake_out
    meta = {}
    assert agent._resolve_turn_answer(_answer(0, override=protect), sheet, choice, LONE_PROTECT_PREDICTION, meta) == payload
    assert meta["resolution"] == "override accepted" and meta["lone_protect_waits"] == fake_out
    # ... and a repeat does not
    monkeypatch.setattr(agent.memory, "protect_streak", lambda species, slot: 1)
    assert _waits_out(sheet, agent.memory) == []


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
    # a Lum or Chesto Berry cures the sleep the moment it starts (Gen 9): the foe loses no turn, so nothing to wait for
    # (the catalog's holders, Salamence and Milotic); once the berry is gone (eaten, knocked off) the reason is back
    memory.drowsy = {"theirs:salamence": 5, "theirs:milotic": 5}
    lum = _mon("Salamence", "theirs", 1, ["dragon", "flying"], item="Lum Berry", ability="Intimidate")
    chesto = _mon("Milotic", "theirs", 2, ["water"], item="Chesto Berry", ability="Competitive")
    assert _waits_out(_alone(plain, lum), memory) == [] and _waits_out(_alone(plain, chesto), memory) == []
    assert _waits_out(_alone(plain, lum, chesto), memory) == []
    lum.item = None
    assert _waits_out(_alone(plain, lum, chesto), memory) == ["their Salamence falls asleep at the end of this turn (Yawn)"]
    chesto.item = "Sitrus Berry"  # a berry that does not cure sleep
    assert _waits_out(_alone(plain, chesto), memory) == ["their Milotic falls asleep at the end of this turn (Yawn)"]


def test_the_catalog_has_no_other_card_that_stays_awake_through_a_yawn_already_in_place():
    """_waits_out's Yawn reason knows the Gen 9 ways a drowsy foe does not fall asleep (the Showdown engine,
    sim/node_modules/pokemon-showdown): a status, Electric / Misty Terrain under a grounded foe, a Lum or Chesto Berry.
    The others need a card the 48-card catalog does not have: Insomnia, Vital Spirit, Comatose, Purifying Salt, Sweet
    Veil, Leaf Guard, Shields Down, an Uproar user, or a setter of either terrain. (Safeguard and Flower Veil only keep a
    Yawn from landing: the engine lets the sleep of one already in place through.) A card added with any of them needs a
    rule in _waits_out first; Unnerve would stop the berries."""
    cards = json.loads((Path(__file__).resolve().parent.parent / "data" / "cards.json").read_text())
    assert len(cards) == 48
    abilities = {c["ability"] for c in cards}
    moves = {m for c in cards for m in c["moves"]}
    assert not abilities & {"Insomnia", "Vital Spirit", "Comatose", "Purifying Salt", "Sweet Veil", "Leaf Guard", "Shields Down",
                            "Electric Surge", "Misty Surge", "Hadron Engine", "Unnerve", "As One"}
    assert not moves & {"Uproar", "Electric Terrain", "Misty Terrain"}
    assert {(c["species"], c["item"]) for c in cards if c["item"] in ("Lum Berry", "Chesto Berry")} == {("Salamence", "Lum Berry"),
                                                                                                    ("Milotic", "Chesto Berry")}
    assert [c["species"] for c in cards if "Yawn" in c["moves"]] == ["Torkoal"]


def test_a_repeat_protect_waits_nothing_out_and_a_sure_ko_first_makes_fake_out_and_the_orb_moot():
    """The M1 review on d9405f02 T7. A reason to wait counts only for a Protect that works: on a streak (the memory's)
    none does, as the prompt allows a repeat only with a listed stall reason. The two that pay off only if we act next
    turn (a fresh Fake Out, a Guts holder's orb) do not count under a sure KO on our slot that lands before we move,
    this turn and the next; a heal (the Wish) still does. A KO that is not sure, one we can escape, a first-turn-only
    hit, a warning on another slot or a HIGH RISK one leaves them standing."""
    foe = _mon("Gholdengo", "theirs", 1, ["steel", "ghost"], item="Air Balloon")
    me = _mon("Ursaluna", "mine", 1, ["ground", "normal"], item="Flame Orb", ability="Guts")
    memory = MatchMemory()
    memory.log_turn, memory.wishes = 6, {"mine:b": 5}
    fake_out = "a fresh Fake Out can flinch our Ursaluna this turn (90%): Protect blocks it, and from next turn it fails"
    orb = "our Ursaluna's Flame Orb activates at the end of this turn (Guts boosts its Attack from then on)"
    wish = "our Wish from last turn heals Ursaluna at the end of this turn"
    sheet = _alone(me, foe, flinch={1: 0.9})
    assert memory.protect_streak("Ursaluna", 1) == 0 and _waits_out(sheet, memory) == [fake_out, orb, wish]
    repeat = MatchMemory()
    repeat.log_turn, repeat.wishes = 6, {"mine:b": 5}
    repeat.protect_streak = lambda species, slot: 1
    assert _waits_out(sheet, repeat) == []
    assert _waits_out(sheet) == [fake_out, orb]  # no memory: no streak to read, the sheet's reasons only
    lethal = ("LETHAL: gholdengo's shadowball hits slot 1 (ursaluna) for 116.9-137.7% BEFORE it can move. This is our last "
              "Pokémon and nothing runs out in our favour: Protect only delays the same KO by a turn, so attack.")
    sheet.warnings = [lethal]
    assert _waits_out(sheet, memory) == [wish]
    for text in (lethal.replace("BEFORE it can move.", "BEFORE it can move (62% KO chance)."),
                 lethal.replace("BEFORE it can move.", "BEFORE it can move (this slot's extremespeed (+2) goes first and KOs it 75% of the time)."),
                 lethal.replace("gholdengo's shadowball", "ironhands's fakeout with priority +3"),
                 lethal.replace("slot 1 (ursaluna)", "slot 0 (ursaluna)"),
                 lethal.replace("LETHAL:", "HIGH RISK:")):
        sheet.warnings = [text]
        assert _waits_out(sheet, memory) == [fake_out, orb, wish], text


def test_each_reason_to_wait_lets_a_fresh_lone_protect_be_judged_by_the_margin_d9405f02_t8():
    """At T8 Cresselia's Protect would be fresh (T7's failed) and it is refused: nothing to wait for. Give the position
    one reason at a time and the margin judges it again (credited 45 under LETHAL: accepted); remove it and the refusal
    is back. A fresh Fake Out or a Guts holder's orb stays moot there: Gholdengo's sure KOs land before Cresselia moves.
    At T9 Cresselia had Protected again (live): a repeat, and no reason counts."""
    agent, sheet, template, choice = _position("d9405f02", 0, 8)
    protect = _lone_protect(template)
    me = next(m for m in sheet.field.actives if m.side == "mine")
    assert agent.memory.log_turn == 8 and me.position == 1 and agent.memory.protect_streak("cresselia", 1) == 0

    def resolution() -> dict:
        meta: dict = {}
        agent._resolve_turn_answer(_answer(0, override=protect), sheet, choice, LONE_PROTECT_PREDICTION, meta)
        return meta

    refused = "override rejected: our last Pokémon's Protect has nothing to wait for"
    assert resolution()["resolution"].startswith(refused)
    reasons = (("wishes", {"mine:b": 7}), ("seeded", {"theirs:gholdengo": "mine:b"}), ("drowsy", {"theirs:ironhands": 7}))
    for name, value in reasons:
        setattr(agent.memory, name, value)
        meta = resolution()
        assert meta["resolution"] == "override accepted" and len(meta["lone_protect_waits"]) == 1, (name, meta)
        setattr(agent.memory, name, {})
    me.item, me.types = "Black Sludge", ["poison"]
    meta = resolution()
    assert meta["resolution"] == "override accepted" and meta["lone_protect_waits"] == ["our cresselia heals a little (Black Sludge)"]
    me.item, me.types = "Sitrus Berry", ["psychic"]
    assert resolution()["resolution"].startswith(refused)
    sheet.flinch_risk = {1: 0.9}
    assert resolution()["resolution"].startswith(refused)
    sheet.flinch_risk, me.item, me.ability = {}, "Flame Orb", "Guts"
    assert resolution()["resolution"].startswith(refused)
    me.item, me.ability = "Sitrus Berry", "Levitate"

    agent, sheet, template, choice = _position("d9405f02", 0, 9)
    protect = _lone_protect(template)
    assert agent.memory.log_turn == 9 and agent.memory.protect_streak("cresselia", 1) == 1
    for name, value in (("wishes", {"mine:b": 8}), ("seeded", {"theirs:gholdengo": "mine:b"}), ("drowsy", {"theirs:ironhands": 8})):
        setattr(agent.memory, name, value)
        meta = resolution()
        assert meta["resolution"].startswith(refused) and "lone_protect_waits" not in meta, (name, meta)
        setattr(agent.memory, name, {})


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
    assert agent._rows_only(sheet) and sheet.reply and [r["score"] for r in rows] == [61.5, 28.2, 10.8, -2.6, -6.2]
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


def test_rows_only_an_override_that_repeats_a_listed_row_is_that_row_with_its_margin(t6_rules):
    """bcbaabe2 T6: an override whose joint action is listed row k plays as row k (not ignored for the answer's own row),
    judged by the same margin; both values are logged (the M1 review)."""
    agent, sheet, template, choice = t6_rules
    rows = sheet.candidates

    def as_override(k: int) -> dict:
        return {"slot_0": dict(rows[k]["slot_0"]), "slot_1": dict(rows[k]["slot_1"])}

    meta: dict = {}
    assert agent._resolve_turn_answer(_answer(0, override=as_override(1)), sheet, choice, TOP, meta) == _plays(choice, rows[1])
    assert meta["override_as_row"] == {"row": 1, "value": 28.2, "answer_row": 0, "answer_row_value": 61.5}
    assert meta["resolution"] == "override = row 1 (a listed row); row 1 plays" and "override_ignored" not in meta
    assert meta["row"] == 1 and meta["row_value"] == 28.2 and meta["margin"] == 50.0 and "row_played" not in meta
    # row 4 (the live play, -6.2) trails by 67.7: the same margin sends it back to row 0
    meta = {}
    assert agent._resolve_turn_answer(_answer(1, override=as_override(4)), sheet, choice, TOP, meta) == _plays(choice, rows[0])
    assert meta["override_as_row"] == {"row": 4, "value": -6.2, "answer_row": 1, "answer_row_value": 28.2} and meta["row_played"] == 0
    assert meta["resolution"].startswith("override = row 4 (a listed row)") and "; row 4 rejected: -6.2 trails" in meta["resolution"]
    # the answer's own row need not be valid: the override names the row
    meta = {}
    assert agent._resolve_turn_answer(_answer(9, override=as_override(1)), sheet, choice, TOP, meta) == _plays(choice, rows[1])
    assert meta["override_as_row"]["answer_row"] is None and meta["override_as_row"]["answer_row_value"] is None
    assert _harness().classify_override(meta) == "as_row"


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
    assert not agent._rows_only(sheet) and not agent._answer_rules_v2() and not sheet.reply
    rows = sheet.candidates
    meta: dict = {}
    same = {"slot_0": rows[0]["slot_0"], "slot_1": rows[0]["slot_1"]}
    agent._resolve_turn_answer(_answer(1, override=same), sheet, choice, TOP, meta)
    assert meta["resolution"] == "override accepted (same as a row)" and "override_ignored" not in meta
    weak = {"slot_0": {"option": _index(template, 0, "flashcannon"), "target": 2}, "slot_1": {"option": _index(template, 1, "snarl"), "target": 0}}
    meta = {}
    agent._resolve_turn_answer(_answer(0, override=weak, prediction=("protect", "switch")), sheet, choice, TOP, meta)
    assert meta["resolution"].startswith("override rejected: value") and "override_ignored" not in meta and "rules" not in meta


def _raising(*args, **kwargs):
    raise KeyError("injected mixture bug")


def test_rows_only_needs_rows_the_mixture_priced_else_the_e2_rules_decide_that_turn(monkeypatch):
    """answer_rules_v2 on the mixture applies its rows-only rules only to rows the mixture priced (sheet.reply set). On a
    turn it did not price (here it raised and the scorer's rows play; a replacement or a position it cannot read is the
    same) the rows are the scorer's, so the E2 rules decide (model audit R7), logged as such. bcbaabe2 T13."""
    from agent.pokemon import replies

    agent, sheet, template, choice = _position("bcbaabe2", 0, 13, RULES)
    assert agent._answer_rules_v2() and sheet.reply and agent._rows_only(sheet)
    weak = {"slot_0": {"option": _index(template, 0, "flashcannon"), "target": 2}, "slot_1": {"option": _index(template, 1, "snarl"), "target": 0}}
    meta: dict = {}
    agent._resolve_turn_answer(_answer(0, override=weak, prediction=("protect", "switch")), sheet, choice, TOP, meta)
    assert meta["resolution"].startswith("override ignored") and "rules" not in meta
    monkeypatch.setattr(replies, "evaluate", _raising)
    agent, sheet, template, choice = _position("bcbaabe2", 0, 13, RULES)
    assert agent._answer_rules_v2() and not sheet.reply and not agent._rows_only(sheet)
    assert any("reply mixture failed this turn" in n for n in sheet.notes)
    rows = sheet.candidates
    meta = {}
    agent._resolve_turn_answer(_answer(1, override={"slot_0": rows[0]["slot_0"], "slot_1": rows[0]["slot_1"]}), sheet, choice, TOP, meta)
    assert meta["resolution"] == "override accepted (same as a row)" and meta["rules"].startswith("E2: the reply mixture did not price")
    meta = {}
    agent._resolve_turn_answer(_answer(0, override=weak, prediction=("protect", "switch")), sheet, choice, TOP, meta)
    assert meta["resolution"].startswith("override rejected: value") and "override_ignored" not in meta and meta["rules"]


def test_a_turn_the_mixture_did_not_price_says_the_scorers_rule_in_its_answer_format(monkeypatch):
    """The system prompt (one text per match, cached) keeps the rows-only bullets; that turn's answer format and
    instructions give the scorer's rule instead, and name the reason (no reply_table)."""
    from agent.pokemon import replies
    from agent.pokemon.prompts import system_prompt

    monkeypatch.setattr(replies, "evaluate", _raising)
    recorder, decision = _ask_at("bcbaabe2", 0, 13, RULES, _answer(0))
    messages = recorder.calls[0]
    assert messages[0] == {"role": "system", "content": system_prompt(tuning.merged(RULES))}
    payload = json.loads(messages[-1]["content"])
    assert "reply_table" not in payload["turn_sheet"] and payload["instructions"] == INSTRUCTIONS_81C4591
    assert payload["answer_format"].startswith("Answer with row (a candidate_turns row number) plus opponent_prediction and win_condition. "
                                               "This turn's rows come from the code's scorer, not the reply mixture (there is no reply_table)")
    tail = ANSWER_FORMAT_81C4591[ANSWER_FORMAT_81C4591.index("set override.use=true"):]  # the scorer's own override rule, verbatim
    assert payload["answer_format"].endswith("so the scorer's rule applies this turn, not rows only: to propose a joint action that is "
                                             "not a row, " + tail)
    assert decision.action["type"] == "doubles_turn"


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


def _ask_at(prefix: str, seat: int, turn: int, params: dict, answer: dict, log_dir: Path | None = None) -> tuple[_Recorder, dict]:
    """Replay code-only up to a battle decision, then let a recording fake model answer it: (recorder, decision)."""
    folder = _folder(prefix)
    agent = PokemonAgent(None, version="mpath", log_dir=log_dir or Path(tempfile.mkdtemp()), capture_dir="", log=lambda *_: None,
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


def test_the_public_summary_describes_the_row_that_played_when_the_proposal_did_not(tmp_path):
    """answer_rules_v2 on the mixture, bcbaabe2 T6 (rows 61.5 / 28.2 / 10.8 / -2.6 / -6.2). The model's
    reasoning_summary describes what it proposed; when that does not play (an override ignored, a row sent back to row
    0) the public summary is the played row's why, and the log names the row. A proposal that plays keeps the model's
    summary. The same on the deploy's scorer path at the defaults (the M1b review): bcbaabe2 T13 (rows 131.2 / 94.2 /
    86.2, E2 margin 19.7) and d9405f02 T8 below."""
    from agent.agent import clean_reason

    _, sheet, template, choice = _position("bcbaabe2", 0, 6, RULES)
    unlisted = _unlisted(template, sheet.candidates, choice.build)
    listed_1 = {"slot_0": dict(sheet.candidates[1]["slot_0"]), "slot_1": dict(sheet.candidates[1]["slot_1"])}
    cases = (("ignored", _answer(1, override=unlisted), 1), ("sent back", _answer(4), 0), ("stands", _answer(1), None),
             ("override = row 1", _answer(0, override=listed_1), None))
    for name, answer, row in cases:
        logs = tmp_path / name.replace(" ", "_")
        recorder, decision = _ask_at("bcbaabe2", 0, 6, RULES, answer, log_dir=logs)
        shown = json.loads(recorder.calls[0][-1]["content"])["turn_sheet"]["candidate_turns"]
        turn = [r for r in _rows(logs) if r["kind"] == "turn"][-1]
        if row is None:
            assert decision.reasoning_summary == "Test answer." and "summary_from_row" not in turn["resolution"], name
        else:
            assert decision.reasoning_summary == clean_reason(shown[row]["why"]) != "Test answer.", name
            assert turn["resolution"]["summary_from_row"] == row, name

    # the scorer path at the defaults (the deploy, reply_mixture 0): on 81c4591 and up to M1b every case below published
    # "Test answer.", the model's own summary, whatever played
    _, sheet, template, choice = _position("bcbaabe2", 0, 13)
    weak = {"slot_0": {"option": _index(template, 0, "flashcannon"), "target": 2}, "slot_1": {"option": _index(template, 1, "snarl"), "target": 0}}
    row0_why = clean_reason(sheet.candidates[0]["why"])
    cases = (("E2 rejects the override; its row 0 plays", _answer(0, override=weak, prediction=("protect", "switch")), 0, "override rejected: value"),
             ("E2 rejects the override, the margin its row 2", _answer(2, override=weak, prediction=("protect", "switch")), 0, "override rejected: value"),
             ("the margin sends row 1 back", _answer(1), 0, "row 1 rejected: 94.2 trails the best row 131.2"),
             ("the override stands on the read", _answer(0, override=weak, prediction=("attack", "attack")), None, "override accepted on the opponent read"),
             ("row 0 stands", _answer(0), None, "row"))
    for name, answer, row, resolution in cases:
        logs = tmp_path / ("scorer " + name).replace(" ", "_").replace(",", "").replace(";", "")
        recorder, decision = _ask_at("bcbaabe2", 0, 13, {}, answer, log_dir=logs)
        turn = [r for r in _rows(logs) if r["kind"] == "turn"][-1]
        assert turn["resolution"]["resolution"].startswith(resolution), (name, turn["resolution"])
        if row is None:
            assert decision.reasoning_summary == "Test answer." and "summary_from_row" not in turn["resolution"], name
        else:
            assert decision.reasoning_summary == row0_why != "Test answer." and turn["resolution"]["summary_from_row"] == row, name
            assert decision.action == choice.build({"slot_0": sheet.candidates[0]["slot_0"], "slot_1": sheet.candidates[0]["slot_1"]})
    # our last Pokémon's Protect refused (d9405f02 T8, nothing to wait for): the row the answer named plays, and is described
    _, sheet, template, choice = _position("d9405f02", 0, 8)
    logs = tmp_path / "scorer_lone_protect"
    recorder, decision = _ask_at("d9405f02", 0, 8, {}, _answer(0, override=_lone_protect(template), prediction=("attack", "attack")),
                                 log_dir=logs)
    turn = [r for r in _rows(logs) if r["kind"] == "turn"][-1]
    assert turn["resolution"]["resolution"].startswith("override rejected: our last Pokémon's Protect has nothing to wait for")
    assert decision.reasoning_summary == clean_reason(sheet.candidates[0]["why"]) and turn["resolution"]["summary_from_row"] == 0


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
    # the search bot's doubles are no forecast for one slot any more (the M1 review); at prompt_v2 0 the line is unchanged
    assert "it doubles our biggest threat about half the time;" in SYSTEM_PROMPT and "it doubles our biggest threat" not in text
    search_line = next(line for line in text.splitlines() if line.startswith("- search_bot"))
    assert ("; it sometimes sends both attacks into one of our Pokémon, but not into a predictable slot (the rows already price "
            "doubles); it sets up rarely") in search_line
    old_line = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith("- search_bot"))
    assert search_line == old_line.replace("it doubles our biggest threat about half the time;", "it sometimes sends both attacks into "
                                           "one of our Pokémon, but not into a predictable slot (the rows already price doubles);")
    # the code's prediction gives one kind per slot and no target, so it is not what prices doubles (the M1b review); at
    # prompt_v2 0 the line is the deploy's, with neither clause
    assert "prediction already prices" not in text and "price doubles" not in SYSTEM_PROMPT and "prices doubles" not in SYSTEM_PROMPT
    assert system_prompt(tuning.merged({"prompt_v2": 0})) == SYSTEM_PROMPT
    from agent.pokemon.opponent import predict

    agent, sheet, _, _ = _position("07a3cb1f", 1, 1)
    per_slot = predict(sheet, agent.memory, turn=1)["per_slot"]
    assert per_slot and all(set(entry) == {"species", "p", "top"} for entry in per_slot.values())  # a kind per slot, no target
    # the setup words are the code's own (the code predicts setup for exactly these, plus the stat boosts named)
    from agent.pokemon.opponent import SETUP_MOVES

    named = {"tailwind", "trickroom", "icywind", "electroweb", "snarl", "reflect", "lightscreen", "nastyplot", "calmmind",
             "swordsdance", "dragondance", "bulkup", "quiverdance"}
    assert named == set(SETUP_MOVES)


def test_prompt_v2_ends_the_sheets_double_target_warnings_neutrally_and_changes_nothing_else_07a3cb1f_t1():
    """07a3cb1f T1: Basculegion under DOUBLE-TARGET (Wood Hammer + Spirit Break). Every DOUBLE-TARGET warning ended
    "Opponents double the target that threatens them most; Protect or switch if that is this slot.", a forecast that
    contradicts prompt_v2's own line. With prompt_v2 it ends "Both foes can reach this slot; it falls only if both choose
    it."; at 0 it is the Oct 10 deploy's text. Nothing else on the sheet moves, and the recheck still reads the warning."""
    from agent.agent import _SheetFacts

    old = "Opponents double the target that threatens them most; Protect or switch if that is this slot."
    new = "Both foes can reach this slot; it falls only if both choose it."
    sheets = {v2: _position("07a3cb1f", 1, 1, {"prompt_v2": v2})[1] for v2 in (0.0, 1.0)}
    for v2, sheet in sheets.items():
        warning = next(w for w in sheet.warnings if w.startswith("DOUBLE-TARGET: if both foes go into slot 0 (basculegion)"))
        assert warning.endswith("— it goes down. " + (new if v2 else old)), warning
        assert [t["kind"] for t in _SheetFacts(sheet).triggers() if t["slot"] == 0].count("double") == 1
    off, on = sheets[0.0].as_prompt(), sheets[1.0].as_prompt()
    assert on["WARNINGS"] != off["WARNINGS"] and on["WARNINGS"] == [w.replace(old, new) for w in off["WARNINGS"]]
    assert {k: v for k, v in on.items() if k != "WARNINGS"} == {k: v for k, v in off.items() if k != "WARNINGS"}


def test_prompt_kb_is_the_audits_mechanics_facts_checked_against_the_mechanics_data():
    from agent.pokemon import data, preview_sim
    from agent.pokemon.prompts import KNOWLEDGE_BASE, SYSTEM_PROMPT, system_prompt

    text = system_prompt(tuning.merged({"prompt_kb": 1}))
    assert text.endswith("\n\n" + KNOWLEDGE_BASE + "\n\nAnswer only in the required JSON.")
    assert text.replace("\n\n" + KNOWLEDGE_BASE, "") == SYSTEM_PROMPT  # nothing else moves
    assert len(KNOWLEDGE_BASE.split()) == 335 and [line[:3] for line in KNOWLEDGE_BASE.splitlines()[1:]] == [f"{i}. " for i in range(1, 10)]
    # general mechanics only: the audit's tenth item, a base rate learned from reviewed matches, is out (the M1 review)
    assert not re.search(r"reviewed|in \d+|times in|LETHAL|DOUBLE-TARGET", KNOWLEDGE_BASE)
    # ... and so is item 1's closing sentence, how a fresh Fake Out user tends to aim (the M1b review): no tendencies
    assert KNOWLEDGE_BASE.splitlines()[1].endswith("Psychic Terrain blocks it into a grounded target.")
    assert not re.search(r"\b(tends?|tendency|usually|often|likely|aims?)\b", KNOWLEDGE_BASE, re.IGNORECASE)
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


def test_every_prompt_variant_is_built_at_import_and_a_broken_anchor_fails_the_import():
    """The eight variants (rows only x prompt_v2 x prompt_kb) exist from import on; an anchor that stops matching
    SYSTEM_PROMPT raises while the module loads (here: SYSTEM_PROMPT's class line edited, the prompt_v2 anchor left)."""
    import importlib.util

    from agent.pokemon import prompts

    assert set(prompts.VARIANTS) == {(r, v, k) for r in (False, True) for v in (False, True) for k in (False, True)}
    assert prompts.VARIANTS[(False, False, False)] is prompts.SYSTEM_PROMPT
    assert len({text for key, text in prompts.VARIANTS.items()}) == 8
    for anchor in (prompts._ROW_BULLETS, prompts._PROTECT_LINE, prompts._DOUBLES_CLAUSE, prompts._LESSONS_HEADER,
                   prompts._PREDICTION_BULLET, prompts._LAST_LINE):
        with pytest.raises(ValueError, match="anchor not found exactly once"):
            prompts.build_variants(prompts.SYSTEM_PROMPT.replace(anchor, "", 1))
    source = Path(prompts.__file__).read_text(encoding="utf-8")
    assert source.count(prompts._DOUBLES_CLAUSE) == 2  # once in SYSTEM_PROMPT, once as the anchor
    spec = importlib.util.spec_from_loader("prompts_broken", loader=None)
    broken = importlib.util.module_from_spec(spec)
    with pytest.raises(ValueError, match="anchor not found exactly once"):
        exec(compile(source.replace(prompts._DOUBLES_CLAUSE, "it doubles a lot;", 1), "prompts_broken.py", "exec"), broken.__dict__)


def test_system_prompt_never_raises_at_decision_time_it_sends_the_deploys_prompt_and_says_so(monkeypatch):
    from agent.pokemon import prompts

    lines: list[str] = []
    assert prompts.system_prompt({**tuning.DEFAULTS, **RULES, "row_margin_points": "fifty"}, log=lines.append) is prompts.SYSTEM_PROMPT
    assert lines and lines[0].startswith("[prompts] system prompt variant failed (ValueError")
    monkeypatch.setattr(prompts, "VARIANTS", {})  # whatever goes wrong, a decision is never lost to it
    lines.clear()
    assert prompts.system_prompt(tuning.merged({"prompt_v2": 1, "prompt_kb": 1}), log=lines.append) is prompts.SYSTEM_PROMPT
    assert len(lines) == 1 and "KeyError" in lines[0]
    assert prompts.system_prompt(tuning.merged({"prompt_v2": 1}), log=_raising) is prompts.SYSTEM_PROMPT  # nor does a broken log
    recorder, decision = _ask_at("bcbaabe2", 0, 13, {"prompt_v2": 1.0, "prompt_kb": 1.0}, _answer(0))
    assert recorder.calls[0][0] == {"role": "system", "content": prompts.SYSTEM_PROMPT} and decision.action["type"] == "doubles_turn"


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
    assert "N calls, up to 2N requests with fallbacks" in printed and "at most $1.20 for 8 calls, $2.40 if every one falls back" in printed
    assert "dry run: nothing was called" in printed and not (tmp_path / "dry.jsonl").exists()
    # a negative cap prints a cost cap of 0, not a negative one, and spends nothing
    assert lp.main(["--logs", str(logs), "--audit", str(tmp_path / "no-audit"), "--model-on", "--max-calls", "-3",
                    "--out", str(tmp_path / "dry.jsonl")]) == 0
    printed = capsys.readouterr().out
    assert "at most $0.00 for 0 calls, $0.00 if every one falls back" in printed and "$-" not in printed
    assert "dry run: nothing was called" in printed and not (tmp_path / "dry.jsonl").exists()


def test_the_harness_records_an_ignored_overrides_key_and_evaluator_value(tmp_path):
    """Under answer_rules_v2 on the mixture a live play that is not a row becomes a proposed override, which is ignored:
    the record carries its option:target key and the evaluator's realised value of it, against the play made and row 0,
    and the summary adds them up (what playing the ignored overrides would have changed)."""
    lp = _harness()
    logs = _own_log(tmp_path)  # 07a3cb1f from seat 1, code only: every live play is row 0
    log_file = next(logs.glob("*.jsonl"))
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    first = next(r for r in lines if r["kind"] == "turn" and r["turn"] == 1)
    agent, sheet, template, choice = _position("07a3cb1f", 1, 1, RULES)
    unlisted = _unlisted(template, sheet.candidates, choice.build)
    first["payload"] = choice.build(unlisted)  # the "live" play: a joint action no row lists
    log_file.write_text("".join(json.dumps(r) + "\n" for r in lines))
    key = lp.joint_key(template, unlisted)
    row0 = lp.joint_key(template, sheet.candidates[0])
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text(json.dumps({"decisions": [{"match": "07a3cb1f", "state_version": first["state_version"], "forced": False,
                                                     "values": {key: 12.5, row0: 40.0}, "rows": [{"key": row0}]}]}))
    out = tmp_path / "rules.jsonl"
    assert lp.main(["--logs", str(logs), "--audit", str(tmp_path / "no-audit"), "--evaluator", str(evaluator),
                    "--tune", json.dumps(RULES), "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    records, summary = rows[:-1], rows[-1]["summary"]
    record = next(r for r in records if r["turn"] == 1)
    assert record["override"] == "ignored" and record["final_key"] == row0 and record["live_key"] == key
    assert record["override_ignored"] == {"key": key, "legal": True, "value": 12.5, "vs_final": -27.5, "vs_row0": -27.5}
    assert all(r["override_ignored"] is None for r in records if r is not record)
    assert summary["ignored_overrides"] == {"n": 1, "valued": 1, "vs_final": -27.5, "vs_row0": -27.5}


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
            # the first call falls back (the primary errs, the fallback answers; a skipped model sends nothing)
            self.last_attempts = ([{"model": "primary", "outcome": "error"}, {"model": "fallback", "outcome": "ok"}] if self.calls == 1
                                  else [{"model": "primary", "outcome": "ok"}, {"model": "fallback", "outcome": "skipped"}])
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
    assert scripted.calls == 2 and summary["model_calls"] == 2 and summary["model_requests"] == 3  # N calls, up to 2N requests
    assert [c["requests"] for r in records for c in r["calls"] if c["source"] == "model"] == [2, 1]
    assert all("override_ignored" in r and r["override_ignored"] is None for r in records)
    model = [r for r in records if r["source"] == "model"]
    assert len(model) >= 1 and all(r["source"] == "replay" for r in records if r not in model)
    for r in model:
        assert r["answers"][0]["opponent_prediction"]["their_slot_2"] == "protect" and r["row0_key"] and r["final_key"]
        assert r["rows"] and all("key" in row for row in r["rows"]) and r["calls"][0]["source"] == "model"
    assert summary["fallbacks"] == 0
