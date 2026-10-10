"""Every captured live main position, replayed through the real agent: with the answer the model gave live (code only,
the default), or answered by the model again on an explicit call budget.

The positions: the live matches whose captures are in tests/fixtures/live/<session>/ and whose decision logs (the
agent's own logs/<session>.jsonl from the match) are in --logs. With the model audit's copies (Oct 7-9, 14 matches)
that is 121 main battle decisions and 17 replacements. Each match is replayed through one PokemonAgent whose memory
is fed what the live agent saw and did: the live draft picks, the live Team Preview bring and, after each battle
decision, the payload played live. So every position's turn sheet is the one this code builds for the real game,
whatever this run plays at the position before it.

Replay (the default; no model and no API call): ReplayProvider answers each battle decision with the live answer,
translated to today's answer format against the rebuilt sheet. The live play becomes {row: k} when the rebuilt sheet
lists it as row k. Otherwise it becomes a proposed joint action {override: {use: true}} carrying row 0; an answer in
today's format keeps its own row when that row's joint action is listed. The answer then goes through everything a
model answer goes through (_resolve_turn_answer, the lethal recheck, the retry). So a run shows what the answer rules
and the recheck do with the real answers: plays that stand, overrides accepted / rejected / ignored, rows sent back
to row 0, recheck firings. A recheck is answered with the same live answer (the first live answer before a recheck
was never logged).

Model on (--model-on): the model answers the main positions instead (all of them, or --only), one call per position
plus a second call when the recheck fires or the answer is invalid, up to --max-calls in all; once the budget is spent
the remaining positions keep the live answer. Replacements always keep it. A call goes through the agent's model chain:
the primary model, then the fallback model when the primary fails or times out, so N calls are up to 2N requests with
fallbacks (each call record counts its requests, and the summary both). Without --max-calls it is a dry run: it prints
the plan and the estimated cost and exits. Every answer is logged with the inputs the model audit's evaluator uses: the
session, turn and state_version, the play's and row 0's option:target keys, the rows shown, the reply table, both
predictions, the first and the rechecked answer, latency and usage. With the audit's skeptic replay (reply_mixture 1,
seed 7, ruled-out variants dropped; --evaluator) each play's realised value against the opponent's actual action is
looked up, against row 0's, and so is an ignored override's (its key and value are in the record, override_ignored).

The summary counts what the audit's measurement plan asks for: plays that stand, overrides proposed / ignored, rows
sent back to row 0, added covers (a Protect or switch row 0 does not make) and those on DOUBLE-TARGET-only slots,
attacks where row 0 covers, recheck firings and flips, the verified rules of scripts/live_scenarios.py met,
prediction accuracy on the observable foe slots (the model's and the code's top column), latency p50 / p95 and
timeouts, and the realised value against always playing row 0, split into the documented positions and the others.

    python scripts/live_positions.py                          # the deploy's rules (reply_mixture 0), live answers
    python scripts/live_positions.py --tune '{"reply_mixture": 1, "reply_seed": 7}'
    python scripts/live_positions.py --tune '{"reply_mixture": 1, "reply_seed": 7, "answer_rules_v2": 1, "recheck_v2": 1}'
    python scripts/live_positions.py --only d9405f02,93a98b6f:T4   # sessions (prefixes) or single turns
    python scripts/live_positions.py --model-on --tune '...'       # dry run: the plan and its cost, no call

The audit's two arms (model-audit.md, measurement plan step 2), on the candidate's mixture with its replay seed. One
call per main position plus a second sample on the 26 robust-labelled positions (--only robust) is about 150 first
calls per arm, about $22 at the $0.15 per Opus call estimate of scripts/live_scenarios.py; rechecks and retries are
second calls, and --max-calls caps them all (N calls, up to 2N requests with fallbacks):

    # arm B: the R1 and R3 code (answer_rules_v2 brings the R1 row/override prompt lines with it on the mixture)
    B='{"reply_mixture": 1, "reply_seed": 7, "answer_rules_v2": 1, "recheck_v2": 1}'
    python scripts/live_positions.py --model-on --max-calls 170 --label B --tune "$B"
    python scripts/live_positions.py --model-on --max-calls 40 --label B2 --only robust --tune "$B"
    # arm C: B plus the R2 and R5 prompt text and the R4 knowledge base
    C='{"reply_mixture": 1, "reply_seed": 7, "answer_rules_v2": 1, "recheck_v2": 1, "prompt_v2": 1, "prompt_kb": 1}'
    python scripts/live_positions.py --model-on --max-calls 170 --label C --tune "$C"
    python scripts/live_positions.py --model-on --max-calls 40 --label C2 --only robust --tune "$C"

Run each arm's replay first (the same command without --model-on: free) to check the positions rebuild; with the
audit's data it reproduces the audit's numbers (41 of 77 valued departures stand under answer_rules_v2, +247 realised
/ +519 material). Score an arm by its summary's realised value against row 0 on the other positions and its verified
rules met. The audit's decision rule: adopt C over B only if, on the other positions, its realised value is not lower
and it meets at least as many verified rules; with about 50 positions a difference under about 300 points is noise,
and then the simpler arm wins. On the deploy's scorer path (reply_mixture 0) the same keys apply without R1, which acts
only on the mixture's rows; the evaluator's cells are the candidate's, so values there are indicative.

Records go to logs/live_positions/<time>-<label>.jsonl: one line per battle decision, then a summary line.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from altruagent import DecisionContext  # noqa: E402
from altruagent.models import GameState  # noqa: E402
from examples.llm.providers import ProviderError  # noqa: E402

from agent.agent import PokemonAgent  # noqa: E402
from agent.pokemon import data, tuning  # noqa: E402
from agent.pokemon import draft as draft_rules  # noqa: E402
from agent.pokemon.memory import observation_dict, species_key  # noqa: E402

LIVE = ROOT / "tests" / "fixtures" / "live"
AUDIT = Path(os.path.expanduser("~/projects/ace-trainer-data/buildout/model-audit"))
PHASES = {"draft": 0, "team_preview": 1, "moving": 2}
COST_PER_CALL = 0.15  # scripts/live_scenarios.py's own estimate for one Opus call
PROTECT_LIKE = {"protect", "detect", "wideguard", "quickguard", "spikyshield", "banefulbunker", "burningbulwark", "silktrap",
                "kingsshield", "obstruct"}
PREDICTION_KINDS = ("attack", "protect", "switch", "setup", "fakeout")  # the observable kinds (none: nothing to observe)


# -- the captures and the live decision logs --------------------------------------------------------------------------


def sequence(folder: Path, seat: int) -> list[Path]:
    def order(p: Path) -> tuple[int, int]:
        m = re.match(r"(draft|team_preview|moving)-(\d+)-seat", p.name)
        return (PHASES[m.group(1)], int(m.group(2))) if m else (9, 0)

    return sorted(folder.glob(f"*-seat{seat}.json"), key=order)


def seat_of(folder: Path) -> int:
    """The seat we sat in: from our team's protocol side in a battle observation, else from the file names."""
    for seat in (0, 1):
        for path in sequence(folder, seat):
            if path.name.startswith("moving-"):
                team = (json.loads(path.read_text()).get("observation") or {}).get("team") or {}
                for key in team if isinstance(team, dict) else []:
                    m = re.match(r"(p[12]):", key or "")
                    if m:
                        return 0 if m.group(1) == "p1" else 1
    return 1 if sequence(folder, 1) and not sequence(folder, 0) else 0


@dataclass
class LiveMatch:
    session: str  # the fixture folder name
    folder: Path
    seat: int
    picks: dict[int, str] = field(default_factory=dict)  # draft state_version -> our live card_id
    pick_order: list[str] = field(default_factory=list)
    lineup: dict | None = None  # the live Team Preview payload
    turns: dict[int, dict] = field(default_factory=dict)  # battle state_version -> the live decision-log row


def load_matches(logs: Path) -> list[LiveMatch]:
    """Every fixture session with a decision log in ``logs`` (matched on the session id's first 8 characters)."""
    out = []
    by_prefix = {p.name[:8]: p for p in sorted(logs.glob("*.jsonl"))} if logs.is_dir() else {}
    for folder in sorted(p for p in LIVE.iterdir() if p.is_dir()):
        log = by_prefix.get(folder.name[:8])
        if log is None:
            continue
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        match = LiveMatch(folder.name, folder, seat_of(folder))
        for row in rows:
            if row.get("kind") == "draft" and row.get("card_id"):
                match.picks[row.get("state_version")] = row["card_id"]
                match.pick_order.append(row["card_id"])
            elif row.get("kind") == "lineup" and isinstance(row.get("payload"), dict):
                match.lineup = row["payload"]
            elif row.get("kind") == "turn" and isinstance(row.get("state_version"), int):
                match.turns[row["state_version"]] = row
        out.append(match)
    return out


# -- joint actions as option:target keys (the audit evaluator's format) -----------------------------------------------


def slot_options(template: dict, number: int) -> list[dict]:
    slot = next((s for s in template.get("slots") or [] if s.get("slot", 0) == number), None)
    return (slot or {}).get("options") or []


def norm_target(option: dict, target: Any) -> int:
    targets = option.get("targets") or []
    if not targets or targets == [0]:
        return 0
    return int(target) if isinstance(target, int) and not isinstance(target, bool) else 0


def slot_key(template: dict, number: int, answer: dict | None) -> str:
    if not answer:
        return "-"
    options = slot_options(template, number)
    index = answer.get("option")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(options):
        return "?"
    return f"{index}:{norm_target(options[index], answer.get('target'))}"


def joint_key(template: dict, joint: dict) -> str:
    return slot_key(template, 0, joint.get("slot_0")) + "|" + slot_key(template, 1, joint.get("slot_1"))


def same_key(a: str | None, b: str | None) -> bool:
    """Equal on every slot both sides fill (an empty slot on one side is ignored, as the audit compares)."""
    if not a or not b:
        return False
    return all(x == y for x, y in zip(a.split("|"), b.split("|")) if x != "-" and y != "-")


def payload_to_answers(template: dict, payload: dict) -> dict:
    """A server payload (slot_n: {type, move_id / species, target}) as option-index answers (None: not offered)."""
    out: dict[str, dict | None] = {}
    for number in (0, 1):
        p = (payload or {}).get(f"slot_{number}") or {}
        options = slot_options(template, number)
        index = None
        for i, o in enumerate(options):
            if p.get("type") == "move" and o.get("type") == "move" and data.to_id(o.get("move_id")) == data.to_id(p.get("move_id")):
                index = i
            elif p.get("type") == "switch" and o.get("type") == "switch" and species_key(o.get("species")) == species_key(p.get("species")):
                index = i
            elif p.get("type") == "pass" and o.get("type") == "pass":
                index = i
            if index is not None:
                break
        out[f"slot_{number}"] = None if index is None else {"option": index, "target": norm_target(options[index], p.get("target"))}
    return out


def posture(template: dict, key: str | None) -> list[str]:
    """What each slot of a joint action does, coarsely: protect / switch / attack / pass (- for no action)."""
    out = []
    for number, part in enumerate((key or "-|-").split("|")[:2]):
        if part in ("-", "?"):
            out.append("-")
            continue
        option = (slot_options(template, number) or [{}] * 99)[int(part.split(":")[0])]
        kind = option.get("type")
        if kind == "switch":
            out.append("switch")
        elif kind == "move":
            out.append("protect" if data.to_id(option.get("move_id")) in PROTECT_LIKE else "attack")
        else:
            out.append("pass")
    return out


def warned_slots(sheet: dict) -> dict[int, list[str]]:
    """Our slots under a warning, by kind (LETHAL and DOUBLE-TARGET warnings, EXPOSED notes)."""
    out: dict[int, list[str]] = {0: [], 1: []}
    for text in sheet.get("WARNINGS") or []:
        m = re.match(r"LETHAL: .+? hits slot (\d) \(", str(text)) or re.match(r"DOUBLE-TARGET: if both foes go into slot (\d)", str(text))
        if m:
            out[int(m.group(1))].append(str(text).split(":")[0])
    for text in sheet.get("notes") or []:
        m = re.match(r"EXPOSED: slot (\d) \(", str(text))
        if m:
            out[int(m.group(1))].append("EXPOSED")
    return out


def translate(live: dict, template: dict, rows: list[dict]) -> dict:
    """The live answer in today's format against the rebuilt rows (see the module docstring)."""
    answers = payload_to_answers(template, live.get("payload") or {})
    key = joint_key(template, answers)
    old = live.get("answer") or {}
    out: dict[str, Any] = {"win_condition": str(old.get("win_condition") or ""),
                           "reasoning_summary": str(old.get("reasoning_summary") or "Live replay.")}
    if isinstance(old.get("opponent_prediction"), dict):
        out["opponent_prediction"] = old["opponent_prediction"]
    slots = {n: answers.get(f"slot_{n}") or {"option": 0, "target": 0} for n in (0, 1)}
    listed = next((r.get("row", i) for i, r in enumerate(rows) if same_key(joint_key(template, r), key)), None)
    if listed is not None:
        out["row"] = listed
        out["override"] = {"use": False, "slot_0": slots[0], "slot_1": slots[1]}
        return out
    row = 0
    live_row = old.get("row")
    if isinstance(live_row, int) and not isinstance(live_row, bool) and live_row > 0:
        live_rows = (live.get("turn_sheet") or {}).get("candidate_turns") or []
        chosen = next((r for i, r in enumerate(live_rows) if r.get("row", i) == live_row), None)
        if chosen is not None:
            mapped = next((r.get("row", i) for i, r in enumerate(rows) if same_key(joint_key(template, r), joint_key(template, chosen))), None)
            row = mapped if mapped is not None else 0
    out["row"] = row
    out["override"] = {"use": True, "slot_0": answers.get("slot_0"), "slot_1": answers.get("slot_1")}
    return out


def _user_payload(messages: list[dict]) -> dict:
    """The decision's JSON payload: the first user message (a recheck or a retry appends plain-text user messages)."""
    for m in messages:
        if m.get("role") == "user":
            try:
                value = json.loads(m.get("content") or "")
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def digest(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


# -- providers ----------------------------------------------------------------------------------------------------------


class ReplayProvider:
    """Answers every call of a decision with that decision's live answer (translated on the first call)."""

    model = "live-replay"

    def __init__(self) -> None:
        self.live: dict | None = None
        self.template: dict = {}
        self.answer: dict | None = None
        self.calls: list[dict] = []
        self.last_model, self.last_latency_ms, self.last_usage = "live-replay", 0, None
        self.last_errors: list[str] = []
        self.last_attempts: list = []

    def prepare(self, live: dict | None, template: dict, *, use_model: bool = False) -> None:
        self.live, self.template, self.answer, self.calls = live, template, None, []

    def complete_structured(self, messages, schema_name, schema, **kwargs):
        if self.live is None:
            raise ProviderError("no live answer recorded for this decision")
        if self.answer is None:
            rows = ((_user_payload(messages).get("turn_sheet") or {}).get("candidate_turns")) or []
            self.answer = translate(self.live, self.template, rows)
        self.calls.append({"source": "replay", "digest": digest(messages), "answer": copy.deepcopy(self.answer)})
        return copy.deepcopy(self.answer)


class BudgetProvider:
    """The model for the decisions marked ``use_model`` while the call budget lasts; the live answer otherwise."""

    def __init__(self, model: Any, replay: ReplayProvider, max_calls: int) -> None:
        self.inner, self.replay, self.max_calls = model, replay, int(max_calls)
        self.spent = 0
        self.requests = 0  # what the calls sent: a call is up to 2 requests (the fallback model after a failed primary)
        self.use_model = False
        self.model = getattr(model, "model", "model")
        self._last: Any = replay

    def prepare(self, live: dict | None, template: dict, *, use_model: bool = False) -> None:
        self.replay.prepare(live, template)
        self.use_model = use_model and self.spent < self.max_calls

    @property
    def calls(self) -> list[dict]:
        return self.replay.calls

    def complete_structured(self, messages, schema_name, schema, **kwargs):
        if not self.use_model:
            self._last = self.replay
            return self.replay.complete_structured(messages, schema_name, schema)
        if self.spent >= self.max_calls:
            raise ProviderError(f"call budget spent ({self.spent} of {self.max_calls})")
        self.spent += 1
        self._last = self.inner
        started = time.monotonic()
        try:
            answer = self.inner.complete_structured(messages, schema_name, schema, **kwargs)
        except ProviderError:
            self.requests += self.sent()
            raise
        requests = self.sent()
        self.requests += requests
        self.replay.calls.append({"source": "model", "digest": digest(messages), "answer": copy.deepcopy(answer),
                                  "latency_ms": int((time.monotonic() - started) * 1000), "requests": requests,
                                  "usage": getattr(self.inner, "last_usage", None), "model": getattr(self.inner, "last_model", None)})
        return answer

    def sent(self) -> int:
        """Requests the last call sent: the chain's attempts that were not skipped (1 for a provider without a chain)."""
        attempts = getattr(self.inner, "last_attempts", None) or []
        return sum(1 for a in attempts if isinstance(a, dict) and a.get("outcome") != "skipped") or 1

    def __getattr__(self, name: str) -> Any:  # last_model, last_latency_ms, last_usage, last_errors, last_attempts...
        if name.startswith("last_"):
            return getattr(self.__dict__.get("_last"), name, None)
        raise AttributeError(name)


class CaptureLog:
    """The agent's decision log, kept in memory (no file, no git read)."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def write(self, kind: str, **fields: Any) -> None:
        self.rows.append({"kind": kind, **fields})


# -- the audit's data (optional) ----------------------------------------------------------------------------------------


def load_evaluator(path: Path | None) -> dict:
    """(session prefix, state_version) -> the audit's skeptic record: realised values of every legal joint action."""
    if not path or not path.exists():
        return {}
    blob = json.loads(path.read_text())
    return {(d["match"], d["state_version"]): d for d in blob.get("decisions") or [] if not d.get("forced")}


def load_documented(path: Path | None) -> set:
    if not path or not path.exists():
        return set()
    return {(d["match"], d["turn"]) for d in json.loads(path.read_text())}


def load_actual(path: Path | None) -> dict:
    """(session prefix, state_version) -> {their position: what that foe did, in the code's words}."""
    if not path or not path.exists():
        return {}
    out = {}
    for d in json.loads(path.read_text()):
        if d.get("forced") or not d.get("outcome_captured"):
            continue
        out[(d["match"], d["state_version"])] = {str(p): (a or {}).get("kind") for p, a in (d.get("opp_actual") or {}).items()}
    return out


def verified_rules() -> dict[tuple, Any]:
    """(session prefix, turn, index) -> scripts/live_scenarios.py's battle scenario at that decision."""
    path = ROOT / "scripts" / "live_scenarios.py"
    module = sys.modules.get("live_scenarios")
    if module is None:
        spec = importlib.util.spec_from_file_location("live_scenarios", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["live_scenarios"] = module  # its dataclass looks itself up there
        spec.loader.exec_module(module)  # importing it reads no .env (its main() does)
    return {(sc.session[:8], sc.turn, sc.index): sc for sc in module.SCENARIOS if sc.phase == "moving"}


def material(rec: dict, key: str) -> float | None:
    det = (rec.get("detail") or {}).get(key)
    if not det or not rec.get("variant_w_m"):
        return None
    return sum(w * v for w, v in zip(rec["variant_w_m"], det.get("per_variant_m") or []))


def eval_value(ev: dict | None, key: str | None) -> float | None:
    """The evaluator's realised value of the joint action ``key`` at this position (None: not there)."""
    values = (ev or {}).get("values") or {}
    return next((v for kk, v in values.items() if same_key(kk, key)), None) if key else None


def value_block(ev: dict | None, final: str, row0: str, live: str) -> dict | None:
    if not (ev or {}).get("values"):
        return None
    out: dict[str, Any] = {"final": eval_value(ev, final), "row0": eval_value(ev, row0), "live": eval_value(ev, live),
                           "eval_row1_key": ((ev.get("rows") or [{}])[0]).get("key")}
    if out["final"] is not None and out["row0"] is not None:
        out["final_vs_row0"] = round(out["final"] - out["row0"], 3)
    if out["live"] is not None and out["row0"] is not None:
        out["live_vs_row0"] = round(out["live"] - out["row0"], 3)
    mats = {k: material(ev, kk) for k, kk in (("final", final), ("row0", row0))}
    if mats["final"] is not None and mats["row0"] is not None:
        out["final_vs_row0_material"] = round(mats["final"] - mats["row0"], 3)
    return out


# -- the replay ---------------------------------------------------------------------------------------------------------


def wanted(only: set[str], session: str, turn: Any) -> bool:
    if not only:
        return True
    return session[:8] in only or f"{session[:8]}:T{turn}" in only


def replay_match(match: LiveMatch, params: dict, provider: Any, *, model_on: bool, only: set[str], evaluator: dict,
                 documented: set, actual: dict | None = None, rules: dict | None = None) -> list[dict]:
    agent = PokemonAgent(provider, version="live-positions", log_dir=None, capture_dir="", log=lambda *_: None, params=params)
    log = CaptureLog()
    agent.decision_log = log  # type: ignore[assignment]  (a log object the agent writes through _SafeLog)
    records: list[dict] = []
    seen_turn: dict[Any, int] = {}
    picks_done = 0
    for path in sequence(match.folder, match.seat):
        raw = json.loads(path.read_text())
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        ctx = DecisionContext(session_id=match.session, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                              agent_id=(raw.get("current_actor") or {}).get("agent_id") or "me", seat_position=match.seat)
        obs = observation_dict(state)
        first = state.legal_actions[0]
        sv = raw.get("state_version")
        if first.action_id.startswith(draft_rules.PREFIX):
            # the live pick, not the code's: the memory follows the real draft (the draft is code only and not measured here)
            agent.memory.observe_draft(obs, my_turn=True, agent_id=ctx.agent_id)
            offered = {draft_rules.card_id_of(a) for a in state.legal_actions}
            pick = match.picks.get(sv) or (match.pick_order[picks_done] if picks_done < len(match.pick_order) else None)
            if pick not in offered:
                pick = draft_rules.choose_pick(state.legal_actions, agent.memory, agent.params).card_id
            agent.memory.record_pick(pick)
            picks_done += 1
            continue
        if first.action_id == "select_lineup":
            agent.memory.observe_team_preview(obs)
            agent.memory.my_lineup = list((match.lineup or {}).get("bring") or [])
            continue
        if first.action_id != "doubles_turn":
            continue
        template = first.input.get("action") or {}
        forced = any(s.get("force_switch") for s in template.get("slots") or [])
        turn = obs.get("turn")
        index = seen_turn.get(turn, 0)
        seen_turn[turn] = index + 1
        live = match.turns.get(sv)
        provider.prepare(live, template, use_model=model_on and not forced and wanted(only, match.session, turn))
        before = len(log.rows)
        started = time.monotonic()
        decision = agent.choose_action(state, ctx)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        row = next((r for r in reversed(log.rows[before:]) if r.get("kind") == "turn"), {})
        errors = [r for r in log.rows[before:] if r.get("kind") in ("error", "fallback")]
        payload = getattr(decision, "action", decision)
        live_payload = (live or {}).get("payload")
        if isinstance(live_payload, dict) and agent.memory.turns:
            # the memory follows the game that was played: the live payload, whatever this run played
            agent.memory.turns[-1]["payload"] = live_payload
            agent.memory.last_payload = live_payload
        if only and not wanted(only, match.session, turn):
            continue
        sheet = row.get("turn_sheet") or {}
        rows = sheet.get("candidate_turns") or []
        keys = [joint_key(template, r) for r in rows]
        live_key = joint_key(template, payload_to_answers(template, live_payload or {}))
        final_key = joint_key(template, payload_to_answers(template, payload if isinstance(payload, dict) else {}))
        listed = next((i for i, k in enumerate(keys) if same_key(k, live_key)), None)
        resolution = row.get("resolution") or {}
        calls = list(getattr(provider, "calls", []) or [])
        rec = {
            "session": match.session[:8], "turn": turn, "index": index, "state_version": sv, "forced": forced,
            "documented": (match.session[:8], turn) in documented and not forced,
            "source": "model" if any(c.get("source") == "model" for c in calls) else "replay",
            "live_key": live_key, "row0_key": keys[0] if keys else None, "final_key": final_key, "listed": listed,
            "departure": bool(keys) and not same_key(keys[0], live_key),
            "stands": same_key(final_key, live_key), "plays_row0": bool(keys) and same_key(final_key, keys[0]),
            "posture": {"final": posture(template, final_key), "row0": posture(template, keys[0] if keys else None),
                        "live": posture(template, live_key)},
            "warned": warned_slots(sheet),
            "rows": [{"row": r.get("row", i), "key": keys[i], "name": r.get("name"), "score": r.get("score"), "worst": r.get("worst"),
                      "best_reply": r.get("best_reply")} for i, r in enumerate(rows)],
            "reply": {k: v for k, v in (sheet.get("reply_table") or {}).items() if k in ("seed", "class", "lambda", "columns")},
            "warnings": sheet.get("WARNINGS") or [],
            "code_top_column": {str(k): v for k, v in ((row.get("code_prediction") or {}).get("top_column") or {}).items()},
            "actual": (actual or {}).get((match.session[:8], sv)),
            "answers": [c.get("answer") for c in calls],
            "resolution": resolution,
            "override": classify_override(resolution),
            "row_sent_back": resolution.get("row_played") == 0,
            "recheck": bool(row.get("recheck")), "recheck_outcome": row.get("recheck_outcome"),
            "recheck_result": row.get("recheck_result"), "recheck_message": row.get("recheck"),
            "fallback": bool(row.get("fallback")) or bool(errors), "errors": row.get("errors") or [str(e.get("error") or e.get("reason")) for e in errors],
            "model": row.get("model"), "latency_ms": row.get("latency_ms"), "model_latency_ms": row.get("model_latency_ms"),
            "wall_ms": elapsed_ms, "usage": row.get("usage"), "attempts": row.get("attempts"), "payload_chars": row.get("payload_chars"),
            "calls": [{k: v for k, v in c.items() if k != "answer"} for c in calls],
        }
        scenario = (rules or {}).get((match.session[:8], turn, index))
        if scenario is not None and isinstance(payload, dict):
            rec["verified_rule"] = {"name": scenario.name, "provisional": scenario.provisional, "ok": bool(scenario.accept(payload, template))}
        if not forced:
            rec["value"] = value_block(evaluator.get((match.session[:8], sv)), final_key, rec["row0_key"], live_key)
        rec["override_ignored"] = ignored_override(resolution, template, None if forced else evaluator.get((match.session[:8], sv)),
                                                   rec.get("value"))
        records.append(rec)
    return records


def ignored_override(resolution: dict, template: dict, ev: dict | None, value: dict | None) -> dict | None:
    """answer_rules_v2: the override the rows-only rules ignored, as an option:target key, with the evaluator's realised
    value of it and its difference from the play made and from row 0 (what playing it would have changed); None when no
    override was ignored."""
    ignored = resolution.get("override_ignored")
    if not isinstance(ignored, dict):
        return None
    key = joint_key(template, {"slot_0": ignored.get("slot_0"), "slot_1": ignored.get("slot_1")})
    out: dict[str, Any] = {"key": key, "legal": ignored.get("legal"), "value": eval_value(ev, key)}
    for name in ("final", "row0"):
        base = (value or {}).get(name)
        out[f"vs_{name}"] = round(out["value"] - base, 3) if out["value"] is not None and base is not None else None
    return out


def classify_override(resolution: dict) -> str | None:
    text = str(resolution.get("resolution") or "")
    if resolution.get("override_ignored") is not None:
        return "ignored"
    if resolution.get("override_as_row") is not None:
        return "as_row"  # answer_rules_v2: a proposed joint action that repeats a listed row plays as that row
    if text.startswith("override accepted"):
        return "accepted"
    if text.startswith("override rejected"):
        return "rejected"
    return None


# -- the summary --------------------------------------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarize(records: list[dict]) -> dict:
    main = [r for r in records if not r["forced"]]
    deps = [r for r in main if r["departure"]]
    listed = [r for r in deps if r["listed"] is not None]
    unlisted = [r for r in deps if r["listed"] is None]
    valued = [r for r in deps if (r.get("value") or {}).get("final_vs_row0") is not None and (r.get("value") or {}).get("live_vs_row0") is not None]

    def total(group: list[dict], field_: str = "final_vs_row0") -> float:
        return round(sum((r.get("value") or {}).get(field_) or 0.0 for r in group), 1)

    first = [r["answers"][0] for r in main if r["answers"]]
    added = dt_only = attacks_where_row0_covers = 0
    for r in main:
        for n in (0, 1):
            final, row0 = r["posture"]["final"][n], r["posture"]["row0"][n]
            if final in ("protect", "switch") and row0 not in ("protect", "switch", "-"):
                added += 1
                kinds = r["warned"].get(n) or r["warned"].get(str(n)) or []
                dt_only += bool(kinds) and all(k == "DOUBLE-TARGET" for k in kinds)
            if final == "attack" and row0 in ("protect", "switch"):
                attacks_where_row0_covers += 1
    out: dict[str, Any] = {
        "main": len(main), "replacements": len(records) - len(main),
        "row0_is_live_play": sum(1 for r in main if not r["departure"]),
        "departures": len(deps), "departures_listed": len(listed), "departures_unlisted": len(unlisted),
        "stand": sum(r["stands"] for r in deps), "stand_listed": sum(r["stands"] for r in listed),
        "stand_unlisted": sum(r["stands"] for r in unlisted),
        "answers": {"row0": sum(1 for a in first if not (a.get("override") or {}).get("use") and a.get("row") == 0),
                    "lower_row": sum(1 for a in first if not (a.get("override") or {}).get("use") and isinstance(a.get("row"), int) and a.get("row") > 0),
                    "override": sum(1 for a in first if (a.get("override") or {}).get("use"))},
        "overrides": {k: sum(1 for r in main if r["override"] == k) for k in ("accepted", "rejected", "ignored", "as_row")},
        "rows_sent_back_to_row0": sum(1 for r in main if r["row_sent_back"]),
        "final_plays_row0": sum(1 for r in main if r["plays_row0"]),
        "added_covers": added, "added_covers_double_target_only": dt_only, "attacks_where_row0_covers": attacks_where_row0_covers,
        "rechecks_main": sum(1 for r in main if r["recheck"]), "rechecks_all": sum(1 for r in records if r["recheck"]),
        "recheck_flips": sum(1 for r in records if r["recheck"] and len(r["answers"]) > 1 and r["answers"][0] != r["answers"][1]),
        "fallbacks": sum(1 for r in records if r["fallback"]),
        "timeouts": sum(1 for r in records if any("timed out" in str(e).lower() or "timeout" in str(e).lower() for e in r["errors"] or [])),
        "model_answers": sum(1 for r in records if r["source"] == "model"),
    }
    ignored = [r["override_ignored"] for r in main if r.get("override_ignored")]
    valued_ignored = [i for i in ignored if i.get("vs_final") is not None]
    out["ignored_overrides"] = {"n": len(ignored), "valued": len(valued_ignored),
                                "vs_final": round(sum(i["vs_final"] for i in valued_ignored), 1),
                                "vs_row0": round(sum(i["vs_row0"] for i in valued_ignored if i.get("vs_row0") is not None), 1)}
    rules = [r for r in records if r.get("verified_rule")]
    counted = [r for r in rules if not r["verified_rule"]["provisional"]]
    out["verified_rules"] = {"met": sum(r["verified_rule"]["ok"] for r in counted), "of": len(counted),
                             "provisional_met": sum(r["verified_rule"]["ok"] for r in rules if r["verified_rule"]["provisional"]),
                             "provisional_of": len(rules) - len(counted)}
    model_rows = [r for r in records if r["source"] == "model"]
    if model_rows:
        lat = [float(c.get("latency_ms") or 0) for r in model_rows for c in r["calls"] if c.get("source") == "model"]
        out["latency_ms"] = {"calls": len(lat), "p50": _percentile(lat, 0.5), "p95": _percentile(lat, 0.95), "max": max(lat) if lat else None}
    predicted = observed = code_ok = model_ok = 0
    for r in main:
        if not r.get("actual"):
            continue
        guess = ((r["answers"][0] if r["answers"] else {}) or {}).get("opponent_prediction") or {}
        for pos, kind in r["actual"].items():
            if kind not in PREDICTION_KINDS:
                continue
            observed += 1
            code_ok += str((r["code_top_column"] or {}).get(str(pos)) or "") == kind
            said = str(guess.get(f"their_slot_{pos}") or "").lower()
            if said:
                predicted += 1
                model_ok += said == kind
    out["predictions"] = {"observable_foe_slots": observed, "code_top_column_right": code_ok, "model_predicted": predicted, "model_right": model_ok}
    if valued:
        out["valued_departures"] = len(valued)
        out["valued_stand"] = sum(r["stands"] for r in valued)
        for name, group in (("all", valued), ("documented", [r for r in valued if r["documented"]]),
                            ("other", [r for r in valued if not r["documented"]])):
            out[f"realised_vs_row0_{name}"] = {"n": len(group), "final": total(group), "model_always": total(group, "live_vs_row0"),
                                               "final_material": total(group, "final_vs_row0_material")}
    return out


def print_summary(label: str, s: dict) -> None:
    print(f"\n[{label}] main positions {s['main']} (+{s['replacements']} replacements); row 0 = the live play at {s['row0_is_live_play']}")
    print(f"  departures from row 0: {s['departures']} (listed lower rows {s['departures_listed']}, not listed {s['departures_unlisted']})")
    print(f"  plays that stand: {s['stand']} of {s['departures']} (listed {s['stand_listed']}/{s['departures_listed']}, "
          f"not listed {s['stand_unlisted']}/{s['departures_unlisted']}); final play is row 0 at {s['final_plays_row0']} of {s['main']}")
    a, o = s["answers"], s["overrides"]
    print(f"  first answers: row 0 {a['row0']}, a lower row {a['lower_row']}, a proposed override {a['override']}; overrides accepted "
          f"{o['accepted']}, rejected {o['rejected']}, ignored {o['ignored']}, played as the listed row they repeat {o['as_row']}; "
          f"rows sent back to row 0 {s['rows_sent_back_to_row0']}")
    print(f"  added covers {s['added_covers']} (DOUBLE-TARGET-only slots {s['added_covers_double_target_only']}); attacks where row 0 "
          f"covers {s['attacks_where_row0_covers']}")
    print(f"  recheck firings: {s['rechecks_main']} on main positions ({s['rechecks_all']} in all), flips {s['recheck_flips']}; "
          f"fallbacks {s['fallbacks']}, timeouts {s['timeouts']}; model answers {s['model_answers']}")
    v = s["verified_rules"]
    p = s["predictions"]
    print(f"  verified rules met: {v['met']}/{v['of']} (provisional {v['provisional_met']}/{v['provisional_of']}); predictions on "
          f"{p['observable_foe_slots']} observable foe slots: code's top column right {p['code_top_column_right']}, model right "
          f"{p['model_right']} of {p['model_predicted']} predicted")
    ig = s.get("ignored_overrides") or {}
    if ig.get("n"):
        print(f"  ignored overrides {ig['n']} ({ig['valued']} valued): playing them instead would have changed the realised value by "
              f"{ig['vs_final']:.1f} against the plays made ({ig['vs_row0']:.1f} against row 0)")
    if s.get("latency_ms"):
        lat = s["latency_ms"]
        print(f"  model latency over {lat['calls']} calls: p50 {lat['p50']} ms, p95 {lat['p95']} ms, max {lat['max']} ms")
    if "valued_departures" in s:
        print(f"  valued departures {s['valued_departures']}, standing {s['valued_stand']}; realised vs always row 0 "
              f"(final play / model always / final, material only):")
        for name in ("all", "documented", "other"):
            v = s[f"realised_vs_row0_{name}"]
            print(f"    {name:10s} n={v['n']:3d}  {v['final']:8.1f} / {v['model_always']:8.1f} / {v['final_material']:8.1f}")


def plan(matches: list[LiveMatch], only: set[str], repeat: int) -> int:
    """Main positions the model would answer (replacements keep the live answer), times the passes."""
    n = 0
    for m in matches:
        for path in sequence(m.folder, m.seat):
            if not path.name.startswith("moving-"):
                continue
            raw = json.loads(path.read_text())
            actions = ((raw.get("legal_actions") or {}).get("actions") or [{}])
            template = ((actions[0].get("input") or {}).get("action")) or {}
            if any(s.get("force_switch") for s in template.get("slots") or []):
                continue
            if wanted(only, m.session, (raw.get("observation") or {}).get("turn")):
                n += 1
    return n * repeat


def robust_positions(audit: Path) -> set[str]:
    stats = json.loads((audit / "synth" / "synth_stats.json").read_text())
    keys = list(stats.get("robust better list") or []) + list(stats.get("robust worse list") or [])
    return {k.replace(" ", ":") for k in keys}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", default="", help="tuning overrides as a JSON object (agent/pokemon/tuning.py keys)")
    ap.add_argument("--label", default="", help="a name for this run (the output file and the summary)")
    ap.add_argument("--only", default="", help="comma-separated session prefixes or <prefix>:T<turn> keys; 'robust' = the "
                    "audit's 26 robust-labelled positions")
    ap.add_argument("--audit", default=str(AUDIT), help="the model audit folder (decision logs, evaluator, documented positions, "
                    "the opponent's actual actions)")
    ap.add_argument("--logs", default="", help="decision logs (default <audit>/synth/logs)")
    ap.add_argument("--evaluator", default="", help="the skeptic replay JSON (default <audit>/skeptic/sk_eval_cand_filtered.json)")
    ap.add_argument("--model-on", action="store_true", help="the model answers the main positions (needs --max-calls to spend)")
    ap.add_argument("--max-calls", type=int, default=0, help="hard cap on model calls, rechecks and retries included (N calls, "
                    "up to 2N requests with fallbacks)")
    ap.add_argument("--repeat", type=int, default=1, help="model on: passes over the positions")
    ap.add_argument("--out", default="", help="output JSONL (default logs/live_positions/<time>-<label>.jsonl)")
    args = ap.parse_args(argv)

    params = tuning.merged(json.loads(args.tune)) if args.tune else dict(tuning.DEFAULTS)
    label = args.label or tuning.label(params)
    audit = Path(os.path.expanduser(args.audit))
    logs = Path(os.path.expanduser(args.logs)) if args.logs else audit / "synth" / "logs"
    matches = load_matches(logs)
    if not matches:
        print(f"no decision logs for the captured sessions in {logs}")
        return 2
    only = {k.strip() for k in args.only.split(",") if k.strip()}
    if "robust" in only:
        only = (only - {"robust"}) | robust_positions(audit)
    evaluator = load_evaluator(Path(os.path.expanduser(args.evaluator)) if args.evaluator else audit / "skeptic" / "sk_eval_cand_filtered.json")
    documented = load_documented(audit / "counterfactual" / "doc_check.json")
    actual = load_actual(audit / "choices" / "decisions_classified.json")
    rules = verified_rules()
    print(f"{label}: {len(matches)} matches, {sum(len(m.turns) for m in matches)} logged battle decisions; evaluator "
          f"{'on' if evaluator else 'off'} ({len(evaluator)} positions); documented positions {len(documented)}; actual actions "
          f"{len(actual)} positions; verified rules {sum(not sc.provisional for sc in rules.values())}")

    replay = ReplayProvider()
    provider: Any = replay
    repeat = 1
    if args.model_on:
        repeat = max(1, args.repeat)
        planned = plan(matches, only, repeat)
        cap = max(0, min(args.max_calls or 2 * planned, 2 * planned))  # the calls this run may make (never below 0)
        print(f"MODEL ON: {planned} main positions x 1 call = {planned} calls planned (up to {2 * planned} with rechecks and "
              f"retries); cap --max-calls {args.max_calls or 0}: N calls, up to 2N requests with fallbacks (the fallback model "
              f"answers a call the primary fails or times out)")
        print(f"  estimated ${COST_PER_CALL * planned:.2f}, at most ${COST_PER_CALL * cap:.2f} for {cap} calls, "
              f"${2 * COST_PER_CALL * cap:.2f} if every one falls back (${COST_PER_CALL:.2f} per request)")
        if args.max_calls <= 0:
            print("dry run: nothing was called. Add --max-calls N to spend at most N calls.")
            return 0
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        from agent.llm.anthropic_provider import provider_from_env

        provider = BudgetProvider(provider_from_env(log=lambda *_: None), replay, args.max_calls)

    out_path = Path(args.out) if args.out else ROOT / "logs" / "live_positions" / f"{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^A-Za-z0-9_.=,-]+', '_', label)[:80]}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = time.monotonic()
    for _ in range(repeat):
        for match in matches:
            if only and not any(k == match.session[:8] or k.startswith(match.session[:8] + ":") for k in only):
                continue
            records += replay_match(match, params, provider, model_on=args.model_on, only=only, evaluator=evaluator,
                                    documented=documented, actual=actual, rules=rules)
    summary = summarize(records)
    summary.update({"label": label, "tune": args.tune or None, "seconds": round(time.monotonic() - started, 1),
                    "model_calls": getattr(provider, "spent", 0), "model_requests": getattr(provider, "requests", 0)})
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({"label": label, **rec}, ensure_ascii=False, default=str) + "\n")
        fh.write(json.dumps({"summary": summary}, default=str) + "\n")
    print_summary(label, summary)
    print(f"  wrote {out_path} ({len(records)} decisions, {summary['seconds']} s, model calls {summary['model_calls']}, "
          f"requests {summary['model_requests']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
