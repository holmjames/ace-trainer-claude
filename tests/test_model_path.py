"""The model path (lane M1): what happens between the model's answer and the move, and what the model is shown.

1. A reply-mixture exception costs the mixture for that turn, never the match: the scorer's rows play (readiness 2B.1).
2. Our last Pokémon's Protect with nothing to wait for earns no override credit and never plays over a row (model
   audit R1/R7; d9405f02 T7 and T9).
3. answer_rules_v2 (off by default): on the reply mixture the model chooses rows only; a lower row stands within
   max(row_margin_points, 15% of |row 0|); no read bypass; the prompt and the payload say so (model audit R1).

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

from agent.agent import PokemonAgent, override_value
from agent.pokemon import battle as battle_rules
from agent.pokemon import tuning
from agent.pokemon.memory import observation_dict
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


@pytest.mark.parametrize("mixture", [0.0, 1.0])
@pytest.mark.parametrize("turn", [7, 9])
def test_a_lone_protect_with_nothing_to_wait_for_earns_nothing_and_never_plays_d9405f02(turn, mixture):
    """d9405f02 T7 and T9: Cresselia, our last Pokémon, under LETHAL Shadow Ball, nothing runs out. The live answer
    Protected (robust worse). The menu lists only the attack; the Protect override was credited 45 and accepted on both
    builds. Now it earns no credit, and since 0 still clears a 0.15 margin under a best row of 0.0 (scorer) or below 0
    (mixture), it is refused outright, read or no read: the attack row plays."""
    agent, sheet, template, choice = _position("d9405f02", 0, turn, {"reply_mixture": mixture, "reply_seed": 7.0})
    assert sheet.alone and not sheet.stall_reasons and any(w.startswith("LETHAL") and "slot 1 (cresselia)" in w for w in sheet.warnings)
    protect = {"slot_0": {"option": 0, "target": 0}, "slot_1": {"option": _index(template, 1, "protect"), "target": 0}}
    payload = choice.build(protect)
    assert override_value(payload, sheet, agent.params) == 0.0
    prediction = {"top_column": {1: "attack", 2: "attack"}}
    for read in (None, ("attack", "attack")):  # no read (the live answer's format) and a read matching the code's column
        meta: dict = {}
        value = agent._resolve_turn_answer(_answer(0, override=protect, prediction=read), sheet, choice, prediction, meta)
        assert meta["resolution"].startswith("override rejected: our last Pokémon's Protect has nothing to wait for"), meta
        assert value["slot_1"]["type"] == "move" and value["slot_1"]["move_id"] != "protect", value
        top = sheet.candidates[0]
        assert value == choice.build({"slot_0": top["slot_0"], "slot_1": top["slot_1"]})
    # with a row number that is not a row, the rejection is an invalid answer (retried), never the Protect
    with pytest.raises(InvalidChoice):
        agent._resolve_turn_answer(_answer(5, override=protect), sheet, choice, prediction, {})
    # a stall reason gives the Protect its credit back (the fix is that narrow)
    sheet.stall_reasons = ["their Tailwind runs out (then we move first)"]
    assert override_value(payload, sheet, agent.params) == agent.params["protect_bonus_guaranteed"]


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
