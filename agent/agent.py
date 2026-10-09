"""The tournament agent: Pokémon VGC doubles draft, code first, Claude on top.

How the pieces fit (plain language):

- The runtime calls ``create_agent()`` once per match and then calls
  ``choose_action(state, context)`` every time it is our turn.
- ``PokemonAgent`` looks at which kind of decision the server is asking for
  and routes it:
    * draft pick      -> pure code (15-second clock; no model call ever)
    * team preview    -> code builds the options, Claude picks, code validates
    * doubles turn    -> code builds the options, Claude picks, code validates
- Every answer from the model goes through the validators in
  ``examples/llm/pokemon.py`` before it is sent. An invalid answer is retried
  once with the error; a second failure plays the deterministic fallback.
- If anything in here raises, the outer guard plays ``examples.smoke_agent``'s
  always-legal move and logs the traceback. A bug can cost a turn, never a match.
- Every decision is written to ``logs/<session_id>.jsonl`` with what the model
  saw, what it answered, which model answered, and how long it took. The first
  line of each log is a ``config`` record: git commit, model chain, tuning
  parameters (the Official Rules ask for records tying the submitted version
  to competitive play).

Run it (after keys are in ``.env``):

    python -m agent --check-tournament
    python -m agent --tournament
"""

from __future__ import annotations

import atexit
import inspect
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from altruagent import DecisionContext, GameState, LegalAction, WithReasoning
from examples import smoke_agent
from examples.llm import pokemon as pokemon_adapters
from examples.llm.base import Choice, InvalidChoice, object_schema
from examples.llm.providers import LLMProvider, ProviderError

from agent.pokemon import battle as battle_rules
from agent.pokemon import data
from agent.pokemon import draft as draft_rules
from agent.pokemon import lineup as lineup_rules
from agent.pokemon.log import DecisionLog, git_commit
from agent.pokemon.memory import MatchMemory, observation_dict, species_key
from agent.pokemon.prompts import LINEUP_INSTRUCTIONS, SYSTEM_PROMPT
from agent.pokemon.tuning import DEFAULTS

VERSION = os.environ.get("AGENT_VERSION") or "m3"
MAX_ATTEMPTS = 2  # first answer + one retry carrying the validation error
BREAKER_FAILURES = 2  # consecutive decisions on which every model failed before one decision skips the model entirely

# The live clock (platform_skill_pokemon.txt:262-268): the platform plays a random legal move for us at 50 s per battle
# decision and 85 s at Team Preview; both come out of a 420 s bank per side, and an empty bank forfeits the battle.
DECISION_SECONDS = 50.0
PREVIEW_DECISION_SECONDS = 85.0
BANK_SECONDS = 420.0
DECISION_SAFETY_SECONDS = 8.0  # network, validation and the send: the model's deadline is the decision's time left minus this
MIN_MODEL_SECONDS = 6.0  # a deadline below this cannot be met by any model in the chain: play the computed move at once
MIN_BANK_SECONDS_FOR_MODEL = 90.0  # the bank below which no model is asked at all (it forfeits at 0)
MODEL_DEADLINE_CAP = {"turn": 20.0, "lineup": 60.0}  # even with a full bank: 20 s a turn keeps a 20-turn game inside the bank
DEFAULT_REMAINING_TURNS = 14  # a fresh 4v4 (see expected_remaining_turns)


class ClockBudget:
    """What this decision may spend on the model, from the server's clock (``observation["clock"]``)."""

    __slots__ = ("decision_left", "bank_left", "deadline", "remaining_turns", "reason")

    def __init__(self, decision_left: float | None, bank_left: float | None, deadline: float, remaining_turns: int, reason: str) -> None:
        self.decision_left, self.bank_left, self.deadline, self.remaining_turns, self.reason = decision_left, bank_left, deadline, remaining_turns, reason

    def as_log(self) -> dict:
        return {"decision_seconds_left": self.decision_left, "bank_seconds_left": self.bank_left,
                "deadline_s": round(self.deadline, 1), "remaining_turns": self.remaining_turns, "rule": self.reason}


def expected_remaining_turns(ours_remaining: int | None, theirs_remaining: int | None) -> int:
    """A simple estimate of the turns left from the Pokémon left: about 1.75 turns per Pokémon still standing on
    either side (a fresh 4v4 is ~14 turns; 2v3 about 9), never under 4. It only spreads the bank: a position that
    is wrong by a few turns moves a deadline by a second or two."""
    ours = int(ours_remaining) if isinstance(ours_remaining, (int, float)) else 4
    theirs = int(theirs_remaining) if isinstance(theirs_remaining, (int, float)) else 4
    return max(4, int(round(1.75 * (ours + theirs))))


def _clock_budget(clock: dict | None, *, remaining_turns: int | None = None, kind: str = "turn") -> ClockBudget:
    """The model deadline for this decision: ``min(decision_left - 8, (bank_left - 90) / max(remaining_turns, 4))``,
    capped at 20 s for a turn (60 s at Team Preview). The bank term spends only what is above the 90 s gate, spread
    over the turns the position still needs, so a long game never runs the bank out (a 30 s deadline every turn
    would reach the gate in ~11 turns). Without a clock (unit tests, an old simulator) the live allowances apply."""
    decision_left = bank_left = None
    if isinstance(clock, dict):
        try:
            decision_left = float(clock["decision_seconds_left"]) if clock.get("decision_seconds_left") is not None else None
            bank_left = float(clock["bank_seconds_left"]) if clock.get("bank_seconds_left") is not None else None
        except (TypeError, ValueError):
            decision_left = bank_left = None
    turns = max(4, int(remaining_turns)) if isinstance(remaining_turns, (int, float)) else DEFAULT_REMAINING_TURNS
    allowance = decision_left if decision_left is not None else (PREVIEW_DECISION_SECONDS if kind == "lineup" else DECISION_SECONDS)
    deadline = allowance - DECISION_SAFETY_SECONDS
    reason = "decision"
    if bank_left is not None:
        per_turn = (bank_left - MIN_BANK_SECONDS_FOR_MODEL) / turns
        if per_turn < deadline:
            deadline, reason = per_turn, "bank"
    cap = MODEL_DEADLINE_CAP.get(kind, MODEL_DEADLINE_CAP["turn"])
    if deadline > cap:
        deadline, reason = cap, "cap"
    return ClockBudget(decision_left, bank_left, max(0.0, deadline), turns, reason)
OBSERVATION_CHAR_LIMIT = 12_000
REASONING_CHAR_LIMIT = 280

# Battle observation keys worth showing the model (the per-slot options are
# passed separately, already annotated with who each target is).
_BATTLE_KEYS = ("turn", "weather", "fields", "field", "side_conditions", "opponent_side_conditions", "active_pokemon",
                "opponent_active_pokemon", "force_switch", "team", "opponent_team")

# The adapter's own slot schema uses a nullable target; Claude's structured
# output is happiest with plain types, so we ask for an integer and tell the
# model to send 0 when a move takes no target (build() ignores it then).
_SLOT_SCHEMA = object_schema({"option": {"type": "integer"}, "target": {"type": "integer"}}, reasoning=False)
_TURN_SCHEMA = object_schema({"win_condition": {"type": "string"}, "slot_0": _SLOT_SCHEMA, "slot_1": _SLOT_SCHEMA})


_PROTECT_MOVES = {"protect", "detect", "wideguard", "quickguard", "spikyshield", "banefulbunker", "burningbulwark", "silktrap", "kingsshield", "obstruct"}
_PIERCES_PROTECT = {"feint", "hyperspacefury", "hyperspacehole", "phantomforce", "shadowforce"}
_FLINCH_PROOF_ABILITIES = {"innerfocus", "shielddust"}
RECHECK_KO_PROBABILITY = 0.5  # a hit that lands first "removes" a threat when it KOs at least this often


def _lethal_recheck(value: Any, sheet: Any, *, memory: Any = None, template: dict | None = None, info: dict | None = None) -> str | None:
    """A valid answer that leaves a threatened slot attacking gets one explicit second look, unless the sheet shows
    the attack is right. The model keeps the final say (trades can be right), but it has to make that call with the
    warning and the concrete alternative row in front of it.

    Triggers (per attacking slot): a LETHAL warning (the threat moves first and KOs), an EXPOSED note (we move first
    but its reply KOs us unless it is removed), a DOUBLE-TARGET warning (both foes together KO the slot).
    Skips: the chosen move's priority beats the threat and it KOs it (ko_probability >= 0.5) - a priority hit that
    cannot KO is no escape (bcbaabe2 T13/T14 lost Arcanine to exactly that); the partner's hit (or the two hits
    together) removes the threat before it moves with p >= 0.5; a Fake Out that goes first flinches it; Protect is
    on a streak and no switch-in is safe; the threat pierces Protect and no pivot is safe; our last Pokémon with
    nothing to wait for. ``info`` (optional) gets the outcome for the decision log."""
    payload = value if isinstance(value, dict) else getattr(value, "payload", None)
    if not isinstance(payload, dict):
        return None
    facts = _SheetFacts(sheet, memory, template)
    triggers = facts.triggers()
    if not triggers:
        return None
    if getattr(sheet, "alone", False) and not getattr(sheet, "stall_reasons", None):
        _note(info, "skipped: our last Pokémon with nothing to wait for")
        return None  # our last Pokémon with nothing to wait for: Protect only delays, so attacking is right (Oct 7)
    problems: list[str] = []
    skips: list[str] = []
    for number in (0, 1):
        slot = payload.get(f"slot_{number}") or {}
        if slot.get("type") != "move" or _move_key(slot.get("move_id")) in _PROTECT_MOVES:
            continue
        for trigger in [t for t in triggers if t["slot"] == number]:
            skip = facts.escape(number, trigger, payload)
            if skip:
                skips.append(f"slot {number}: {skip}")
                continue
            problems.append(f"slot {number} attacks with {slot.get('move_id')} although: {trigger['text']}")
            break
    if isinstance(info, dict):
        info["recheck_skips"] = list(skips)
        info["recheck_problems"] = [p[:120] for p in problems]
    if not problems:
        _note(info, "skipped: " + "; ".join(skips) if skips else "no attacking slot under a trigger")
        return None
    _note(info, "fired: " + " | ".join(p[:80] for p in problems) + (" (skipped: " + "; ".join(skips) + ")" if skips else ""))
    alternatives = [facts.alternative(number) for number in (0, 1) if any(p.startswith(f"slot {number} ") for p in problems)]
    alternatives = [a for a in alternatives if a]
    return ("Recheck before this is final. " + " | ".join(problems) +
            " A Pokémon that is knocked out before it moves never gets its attack off, and one that is knocked out after it moves "
            "is gone next turn. Keep your answer only if the other slot removes that threat first (priority, or faster with a "
            "likely KO) or the trade wins the game; otherwise take the alternative row."
            + (" The sheet's alternative: " + " / ".join(alternatives) + "." if alternatives else "")
            + " Answer again; reasoning_summary stays a one-sentence public description of the move itself.")


def _note(info: dict | None, outcome: str) -> None:
    if isinstance(info, dict):
        info["recheck_outcome"] = outcome


def _move_key(move_id: Any) -> str:
    return str(move_id or "").replace("-", "").replace(" ", "").lower()


class _SheetFacts:
    """What the recheck needs from a TurnSheet: who moves before whom, what each of our options does to each foe,
    which threats the warnings name, and the alternative rows."""

    def __init__(self, sheet: Any, memory: Any = None, template: dict | None = None) -> None:
        self.sheet = sheet
        self.memory = memory
        self.template_options: dict[int, list] = {}
        for index, slot in enumerate((template or {}).get("slots") or []):
            if isinstance(slot, dict):
                number = slot.get("slot", index)
                self.template_options[number if number in (0, 1) else index] = [o for o in slot.get("options") or [] if isinstance(o, dict)]
        self.order = {(e.get("side"), e.get("position")): i for i, e in enumerate(getattr(sheet, "speed_order", None) or [])}
        self.ours = {e.get("position"): e.get("species") for e in getattr(sheet, "speed_order", None) or [] if e.get("side") == "mine"}
        self.theirs = {e.get("position"): e.get("species") for e in getattr(sheet, "speed_order", None) or [] if e.get("side") == "theirs"}
        self.threats = {species_key(t.get("species")): t for t in getattr(sheet, "threats", None) or []}

    # -- triggers --------------------------------------------------------------------------------------------------

    def triggers(self) -> list[dict]:
        out = []
        for text in getattr(self.sheet, "warnings", None) or []:
            text = str(text)
            m = re.match(r"LETHAL: (.+?)'s (\S+?)( with priority \+\d)? hits slot (\d) \(", text)
            if m:
                out.append({"kind": "lethal", "slot": int(m.group(4)), "threat": species_key(m.group(1)), "move": m.group(2), "text": text})
                continue
            m = re.match(r"DOUBLE-TARGET: if both foes go into slot (\d) \((.+?)\) — (.+?)'s (\S+) \+ (.+?)'s (\S+) =", text)
            if m:
                out.append({"kind": "double", "slot": int(m.group(1)), "threat": species_key(m.group(3)), "move": m.group(4),
                            "threat2": species_key(m.group(5)), "move2": m.group(6), "text": text})
        for text in getattr(self.sheet, "notes", None) or []:
            text = str(text)
            m = re.match(r"EXPOSED: slot (\d) \((.+?)\) moves before (.+?), but .+?'s (\S+) KOs it", text)
            if m:
                out.append({"kind": "exposed", "slot": int(m.group(1)), "threat": species_key(m.group(3)), "move": m.group(4), "text": text})
        return out

    # -- our chosen moves ------------------------------------------------------------------------------------------

    def option_row(self, number: int, slot_payload: dict) -> dict | None:
        """The sheet's row for the option this slot chose (matched by move id)."""
        for entry in getattr(self.sheet, "our_options", None) or []:
            if entry.get("slot") != number:
                continue
            for option in entry.get("options") or []:
                if _move_key(option.get("move")) == _move_key(slot_payload.get("move_id")):
                    return option
        return None

    @staticmethod
    def damage_into(option: dict, threat_key: str, target: Any) -> list | None:
        for row in option.get("targets") or []:
            if row.get("side") != "theirs" or species_key(row.get("species")) != threat_key:
                continue
            if row.get("target") in (0, None) or target in (0, None) or row.get("target") == target:
                return row.get("damage_pct_of_current_hp")
        return None

    def acts_before(self, number: int, our_priority: int, threat_key: str, their_priority: int) -> bool:
        position = next((p for p, s in self.theirs.items() if species_key(s) == threat_key), None)
        if our_priority != their_priority:
            return our_priority > their_priority
        return self.order.get(("mine", number), 99) < self.order.get(("theirs", position), 99)

    def escape(self, number: int, trigger: dict, payload: dict) -> str | None:
        """Why the attack is right despite the trigger, or None."""
        threat_key = trigger["threat"]
        threat = self.threats.get(threat_key) or {}
        their_hit = next((h for h in threat.get("hits") or [] if _move_key(h.get("move")) == _move_key(trigger["move"])), {})
        their_priority = int(their_hit.get("priority") or 0) if trigger["kind"] != "exposed" else 0
        removal: list[tuple[float, str]] = []  # (probability the threat is gone before it moves, how)
        combined: list[list] = []
        for n in (0, 1):
            choice = payload.get(f"slot_{n}") or {}
            if choice.get("type") != "move" or _move_key(choice.get("move_id")) in _PROTECT_MOVES:
                continue
            option = self.option_row(n, choice)
            if not option:
                continue
            our_priority = int(option.get("priority") or 0)
            first = self.acts_before(n, our_priority, threat_key, their_priority) if trigger["kind"] != "exposed" else True
            if not first:
                continue
            damage = self.damage_into(option, threat_key, choice.get("target"))
            if damage:
                p = battle_rules.ko_probability({"damage_pct_of_current_hp": damage})
                combined.append(list(damage))
                if p >= RECHECK_KO_PROBABILITY:
                    removal.append((p, f"slot {n}'s {option.get('move')} ({damage[0]}-{damage[1]}%) lands first and KOs {threat_key} {int(p * 100)}% of the time"))
            if _move_key(option.get("move")) == "fakeout" and our_priority > their_priority and self.flinchable(threat, number):
                targets = [r for r in option.get("targets") or [] if species_key(r.get("species")) == threat_key]
                if targets and (choice.get("target") in (0, None) or any(r.get("target") == choice.get("target") for r in targets)):
                    removal.append((1.0, f"slot {n}'s Fake Out goes first and flinches {threat_key}"))
        if len(combined) == 2 and not removal:
            lo, hi = combined[0][0] + combined[1][0], combined[0][1] + combined[1][1]
            p = battle_rules.ko_probability({"damage_pct_of_current_hp": [lo, hi]})
            if p >= RECHECK_KO_PROBABILITY:
                removal.append((p, f"both hits land before {threat_key} moves and together KO it ({lo:.1f}-{hi:.1f}%, {int(p * 100)}%)"))
        if removal:
            return max(removal, key=lambda r: r[0])[1]
        species = self.ours.get(number)
        streak = self.memory.protect_streak(species, number) if (self.memory is not None and species) else 0
        safe = self.safe_switches(number)
        if streak and not safe:
            return f"Protect is on a streak of {streak} (1 in {3 ** streak}) and no switch-in is safe: attacking is what is left"
        if self.pierces_protect(threat, their_hit) and not safe:
            return f"{threat_key}'s {trigger['move']} goes through Protect and no switch-in is safe: attacking is what is left"
        return None

    def flinchable(self, threat: dict, number: int) -> bool:
        ability = data.to_id(threat.get("ability"))
        item = data.to_id(threat.get("item"))
        types = data.resolve_types({"species": threat.get("species")}) if threat.get("species") else []
        return "ghost" not in types and ability not in _FLINCH_PROOF_ABILITIES and item != "covertcloak"

    @staticmethod
    def pierces_protect(threat: dict, hit: dict) -> bool:
        move = _move_key(hit.get("move"))
        if move in _PIERCES_PROTECT:
            return True
        info = data.move_info(move) or {}
        return data.to_id(threat.get("ability")) == "unseenfist" and "contact" in (info.get("flags") or [])

    def safe_switches(self, number: int) -> list[str]:
        out = []
        for key, entry in (getattr(self.sheet, "switch_ins", None) or {}).items():
            by_slot = entry.get("by_slot") or {}
            verdict = (by_slot.get(number) or by_slot.get(str(number)) or {}).get("verdict") or entry.get("verdict")
            if verdict == "safe":
                out.append(key)
        return out

    # -- the alternative row -----------------------------------------------------------------------------------------

    def alternative(self, number: int) -> str | None:
        """The top protect_threatened / switch_threatened candidate that takes this slot out of the line of fire."""
        for cand in getattr(self.sheet, "candidates", None) or []:
            if cand.get("name") not in ("protect_threatened", "switch_threatened", "switch_weak_slot"):
                continue
            mine = cand.get(f"slot_{number}") or {}
            option = self.option_by_index(number, mine.get("option"))
            if option is None:
                continue
            is_protect = option.get("type") == "move" and _move_key(option.get("move_id")) in _PROTECT_MOVES
            is_switch = option.get("type") == "switch"
            if not (is_protect or is_switch):
                continue
            other = 1 - number
            theirs = cand.get(f"slot_{other}") or {}
            what = f"option {mine.get('option')} ({'switch to ' + str(option.get('species')) if is_switch else option.get('move_id')})"
            return (f"{cand.get('name')}: slot {number} {what}, slot {other} option {theirs.get('option')} target {theirs.get('target')} "
                    f"— {str(cand.get('why'))[:160]}")
        return None

    def option_by_index(self, number: int, index: Any) -> dict | None:
        """The server's option at that index for the slot (moves, switches and pass alike: the sheet's own rows only
        cover damaging moves)."""
        options = self.template_options.get(number) or []
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(options):
            return options[index]
        return None


class PokemonAgent:
    def __init__(
        self,
        provider: LLMProvider | None,
        *,
        version: str = VERSION,
        log_dir: str | os.PathLike | None = None,
        capture_dir: str | os.PathLike | None = None,
        log=None,
        params: dict | None = None,
    ) -> None:
        self._provider = provider  # None = code-only mode (no key, or tests)
        self.params = dict(params) if params else dict(DEFAULTS)
        self._version = version
        self._log_dir = log_dir
        self._capture_dir = capture_dir if capture_dir is not None else os.environ.get("AGENT_CAPTURE_DIR")
        self._print = log if log is not None else _stderr
        self.memory = MatchMemory()
        self.decision_log: DecisionLog | None = None
        self._all_failed = 0  # decisions on which every model failed (the ALL MODELS FAILED banner)
        self._efforts: dict[str, str | None] = {"turn": None, "lineup": None}  # per decision kind; None = the provider's default
        self._provider_kwargs: set[str] | None = None  # which per-call parameters the provider accepts (looked up once)
        self._decision_started = time.monotonic()
        self._breaker_skips = 0  # decisions the circuit breaker kept away from the model
        self._consecutive_failures = 0  # all-model failures in a row (reset by any model answer)
        self._breaker_open = False  # the next decision skips the model
        self._lineup_model: str | None = None  # who answered Team Preview (None: code)
        self._summarized = False

    # -- end of match: one summary line -------------------------------------------------

    def on_action_result(self, result: Any, context: DecisionContext) -> None:
        """The runner reports every play_action result; the one that ends the match (or a result whose state is
        terminal) triggers the per-match summary line. Never raises (the runner would turn that into an error)."""
        try:
            if not isinstance(result, dict):
                return
            state = result.get("state") if isinstance(result.get("state"), dict) else {}
            if result.get("is_terminal") is True or result.get("status") == "completed" or state.get("is_terminal") is True:
                self.print_match_summary(context)
        except Exception:  # noqa: BLE001
            pass

    def match_summary(self) -> str:
        """Turns decided by the primary model / the fallback model / code only, with the failure counters."""
        primary = getattr(self._provider, "model", None)
        turns = list(self.memory.turns)
        by_primary = sum(1 for t in turns if t.get("model") and t.get("model") == primary)
        by_fallback = sum(1 for t in turns if t.get("model") and t.get("model") != primary)
        code_only = len(turns) - by_primary - by_fallback
        preview = "model" if self._lineup_model == primary and primary else ("fallback model" if self._lineup_model else "code")
        return (f"[agent] match summary: {len(turns)} battle decisions: {by_primary} by {primary or 'no model'}, "
                f"{by_fallback} by the fallback model, {code_only} code-only; Team Preview by {preview}; "
                f"model calls {self.memory.llm_calls}, all-models-failed {self._all_failed}, "
                f"breaker skips {self._breaker_skips}, fallbacks {self.memory.fallbacks}")

    def print_match_summary(self, context: DecisionContext | None = None) -> None:
        """Once per match, and only if something was decided (a worker that never got a game stays quiet)."""
        if self._summarized or not (self.memory.turns or self.memory.llm_calls or self._lineup_model):
            return
        self._summarized = True
        self._say(self.match_summary())
        if context is not None:
            self._safe_log_for(context).write("summary", line=self.match_summary(), memory=self.memory.summary())

    # -- the one method the runtime calls ----------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        started = time.monotonic()
        self._decision_started = started
        log = self._safe_log_for(context)
        try:
            self._capture(state, context)
            decision, summary = self._decide(state, context, log)
        except Exception as exc:  # noqa: BLE001 — a bug must never forfeit the match
            return self._last_resort(state, context, log, exc)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self.memory.latencies_ms.append(elapsed_ms)
        self._say(f"[agent] {state.phase}: {summary} ({elapsed_ms} ms)")
        return WithReasoning(decision, clean_reason(summary))

    def _last_resort(self, state: GameState, context: DecisionContext, log: "_SafeLog", exc: BaseException):
        """The guard's own guard. A crash here would hand the runner a DecisionError: the worker dies, the seat sits
        in a 60 s cooldown, the server auto-moves at random and the match memory is gone. So nothing in here may
        raise: the log and the console are wrapped, the smoke agent is tried first, and if even that cannot read
        the request a minimal legal payload is built by hand from the raw request."""
        self.memory.fallbacks += 1
        phase = getattr(state, "phase", None)
        log.write("error", phase=phase, state_version=getattr(state, "state_version", None),
                  error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-3000:])
        self._say(f"[agent] ERROR in decision logic ({type(exc).__name__}); playing the safe fallback move")
        reason = "exception in decision logic"
        try:
            fallback = smoke_agent.choose_action(state, context)
        except Exception as smoke_exc:  # noqa: BLE001
            log.write("error", phase=phase, error=f"smoke agent: {type(smoke_exc).__name__}: {smoke_exc}")
            try:
                fallback = minimal_legal_payload(state)
                reason = "exception in decision logic and in the smoke agent; hand-built minimal payload"
            except Exception as minimal_exc:  # noqa: BLE001 — nothing legal can be built: the runner must know
                log.write("error", phase=phase, error=f"minimal payload: {type(minimal_exc).__name__}: {minimal_exc}")
                self._say(f"[agent] no legal payload could be built from the request ({minimal_exc})")
                raise exc
        log.write("fallback", phase=phase, payload=_payload(fallback), reason=reason)
        return WithReasoning(fallback, "Fallback: played a safe legal move after an internal error.")

    def _say(self, line: str) -> None:
        """Console chatter that cannot cost a move (a closed stderr raises on write)."""
        try:
            self._print(line)
        except Exception:  # noqa: BLE001
            pass

    # -- routing -------------------------------------------------------------------------

    def _decide(self, state: GameState, context: DecisionContext, log: DecisionLog):
        if not state.legal_actions:
            raise InvalidChoice("choose_action was called with no legal actions.")
        obs = observation_dict(state)
        first = state.legal_actions[0]

        if first.action_id.startswith(draft_rules.PREFIX):
            self.memory.observe_draft(obs, my_turn=True, agent_id=context.agent_id)
            return self._draft(state, obs, log)
        if first.action_id == "select_lineup":
            self.memory.observe_team_preview(obs)
            return self._lineup(first, state, obs, log)
        if first.action_id == "doubles_turn":
            self.memory.observe_battle(obs)
            return self._turn(first, state, obs, log)

        log.write("other", phase=state.phase, action_id=first.action_id, label=first.label)
        return first, f"Played {first.label or first.action_id}."

    # -- draft: pure code ---------------------------------------------------------------

    def _draft(self, state: GameState, obs: dict, log: DecisionLog):
        pick = draft_rules.choose_pick(state.legal_actions, self.memory, self.params)
        self.memory.record_pick(pick.card_id)
        log.write(
            "draft",
            state_version=state.state_version,
            pick_number=len(obs.get("picks") or []) + 1,
            card_id=pick.card_id,
            species=pick.species,
            score=round(pick.score, 2),
            notes=pick.notes,
            offered=[draft_rules.card_id_of(a) for a in state.legal_actions],
            memory=self.memory.summary(),
        )
        return pick.action, f"Drafted {pick.species}."

    # -- team preview: code shortlist (M3), Claude picks -----------------------------------

    def _lineup(self, action: LegalAction, state: GameState, obs: dict, log: DecisionLog):
        choice = pokemon_adapters.lineup_choice(action, state)
        roster = list(choice.prompt.get("roster") or [])
        candidates = lineup_rules.shortlist(self.memory.my_cards, self.memory.opp_cards, roster, params=self.params)
        # The best computed lineup is the fallback (validated like any answer); the adapter's
        # own fallback (first four of the roster) only if even that fails.
        if candidates:
            try:
                computed = choice.build({"bring": candidates[0].bring, "leads": candidates[0].leads})
                adapter_fallback = choice.fallback
                choice = Choice(choice.kind, choice.prompt, choice.schema, choice.build, lambda: computed)
            except InvalidChoice:
                pass
        payload = {
            "decision": "select_lineup",
            **choice.prompt,
            "instructions": f"{LINEUP_INSTRUCTIONS} Server instructions: {choice.prompt.get('instructions')}",
            "computed_candidates": [
                {"rank": i + 1, "bring": c.bring, "leads": c.leads, "score": c.score, "notes": c.notes}
                for i, c in enumerate(candidates)
            ],
            "known_sets": {"mine": self.memory.known_sets("mine"), "opponent": self.memory.known_sets("opponent")},
            "observation": _compact(_preview_view(obs), ("your_roster", "opponent_roster", "battle_format")),
        }
        value, answer, info = self._ask(choice, payload, kind="lineup", clock=obs.get("clock"), remaining_turns=DEFAULT_REMAINING_TURNS)
        self._lineup_model = info.get("model")
        if isinstance(value, dict):
            self.memory.my_lineup = list(value.get("bring") or [])
        log.write("lineup", state_version=state.state_version, payload=_payload(value),
                  candidates=payload["computed_candidates"], known_sets=payload["known_sets"], wall_ms=self._wall_ms(), **info)
        summary = (answer or {}).get("reasoning_summary") or f"Bringing {value.get('bring')}, leading {value.get('leads')}."
        return value, str(summary)

    # -- battle turn: code computes (M3), Claude judges, adapter validates ---------------

    def _turn(self, action: LegalAction, state: GameState, obs: dict, log: DecisionLog):
        base = pokemon_adapters.doubles_choice(action, state)
        template = action.input.get("action") or {}
        sheet = battle_rules.build_sheet(template, obs, self.memory, self.params)
        # Fallback order: first computed candidate that validates, else the adapter's deterministic move.
        fallback = base.fallback
        for candidate in sheet.candidates:
            try:
                computed = base.build({"slot_0": candidate["slot_0"], "slot_1": candidate["slot_1"]})
            except InvalidChoice as exc:
                candidate["invalid"] = str(exc)
                continue
            fallback = (lambda value=computed: value)
            break
        sheet.candidates = [c for c in sheet.candidates if "invalid" not in c]
        choice = Choice(kind=base.kind, prompt=base.prompt, schema=_TURN_SCHEMA, build=base.build, fallback=fallback)
        roster = battle_roster(obs, template, self.memory)
        payload = {
            "decision": "doubles_turn",
            **choice.prompt,
            "answer_format": (
                "slot_0 and slot_1 each need {option: <index from that slot's options>, target: <integer>}. "
                "If the chosen option's targets list is non-empty, target must be one of those integers; "
                "if the list is empty (or the option is a switch/pass), send target 0."
            ),
            "battle_roster": roster,
            "turn_sheet": sheet.as_prompt(),
            "known_sets": {"mine": self.memory.known_sets("mine", only=roster["ours"]["in_this_battle"]),
                           "opponent": self.memory.known_sets("opponent", status=roster["theirs"]["status"])},
            "observation": _compact(_battle_view(obs, roster, self.memory), _BATTLE_KEYS),
        }
        remaining = expected_remaining_turns(roster["ours"].get("remaining"), roster["theirs"].get("remaining"))
        recheck_info: dict[str, Any] = {}
        value, answer, info = self._ask(choice, payload, kind="turn",
                                        recheck=lambda v: _lethal_recheck(v, sheet, memory=self.memory, template=template, info=recheck_info),
                                        clock=obs.get("clock"), remaining_turns=remaining)
        forced = any(slot.get("force_switch") for slot in template.get("slots") or [])
        self.memory.record_turn(turn=obs.get("turn"), payload=_payload(value), model=info.get("model"), forced=forced)
        log.write("turn", state_version=state.state_version, turn=obs.get("turn"), payload=_payload(value),
                  win_condition=(answer or {}).get("win_condition"), battle_roster=roster, turn_sheet=sheet.as_prompt(),
                  wall_ms=self._wall_ms(), recheck_outcome=recheck_info.get("recheck_outcome"), **info)
        summary = (answer or {}).get("reasoning_summary") or (sheet.candidates[0]["why"] if sheet.candidates else "Played the computed default turn.")
        return value, str(summary)

    # -- the model call, with retry-once and deterministic fallback ---------------------

    def _ask(self, choice: Choice, payload: dict, *, kind: str, recheck: Any = None,
             clock: dict | None = None, remaining_turns: int | None = None) -> tuple[Any, dict | None, dict]:
        """Ask the model, retry once on an invalid answer, fall back to the computed move if it still fails.
        ``recheck(value)`` may return a message for a valid-but-suspicious first answer (e.g. attacking with a slot the
        sheet marks LETHAL); the model is then asked once more with that message and its second answer stands.
        Clock: ``_clock_budget`` turns the server's clock into one deadline for the whole decision; every call gets
        what is left of it, a second call (retry or recheck) only when at least MIN_MODEL_SECONDS remain, and a
        deadline under MIN_MODEL_SECONDS plays the computed move without asking."""
        info: dict[str, Any] = {"model": None, "latency_ms": None, "attempts": 0, "errors": [], "fallback": False}
        started = time.monotonic()
        # The server's own clock (observation["clock"]): seconds left for this decision and in the battle bank.
        budget = _clock_budget(clock, remaining_turns=remaining_turns, kind=kind)
        deadline = budget.deadline
        info["clock"] = budget.as_log()
        info["effort"] = self.effort_for(kind)
        if self._provider is not None and deadline < MIN_MODEL_SECONDS:
            info["errors"].append(f"clock too short for a model call (deadline {deadline:.1f}s from {budget.reason}: "
                                  f"{budget.decision_left}s left, bank {budget.bank_left}s); playing the computed move")
            info["fallback"] = True
            self.memory.fallbacks += 1
            return choice.fallback(), None, info
        if self._provider is None:
            info["errors"].append("no provider configured (code-only mode)")
            info["fallback"] = True
            self.memory.fallbacks += 1
            return choice.fallback(), None, info
        if self._breaker_open:
            # Two decisions in a row on which every model failed (timeouts, 5xx, a dead key): each costs the whole
            # chain's timeouts (~30 s of a 50 s decision). Skip the model once, play the computed move at once, then
            # try the model again on the next decision.
            self._breaker_open = False
            self._breaker_skips += 1
            info["errors"].append(f"circuit breaker: {self._consecutive_failures} consecutive all-model failures; "
                                  "skipping the model this decision, retrying on the next")
            info["fallback"] = True
            info["breaker"] = "skipped"
            self.memory.fallbacks += 1
            self._say(f"[agent] circuit breaker: SKIPPING the model on this {kind} after {self._consecutive_failures} "
                      "consecutive failures (code plays #1); retrying on the next decision")
            return choice.fallback(), None, info

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        info["attempt_log"] = []
        info["model_latency_ms"] = 0
        first_valid: tuple[Any, dict] | None = None  # a valid first answer under recheck: never discarded for a failed second call
        for attempt in range(1, MAX_ATTEMPTS + 1):
            info["attempts"] = attempt
            self.memory.llm_calls += 1
            call_deadline = max(0.0, deadline - (time.monotonic() - started))
            call_started = time.monotonic()
            record = {"attempt": attempt, "deadline_s": round(call_deadline, 1), "effort": info["effort"]}
            try:
                answer = self._call_model(messages, choice.kind, choice.schema, deadline=call_deadline, effort=info["effort"])
            except ProviderError as exc:
                record.update(latency_ms=int((time.monotonic() - call_started) * 1000), outcome="all models failed",
                              chain=list(getattr(self._provider, "last_attempts", None) or []))
                info["attempt_log"].append(record)
                info["model_latency_ms"] += record["latency_ms"]
                if first_valid is not None:
                    info["errors"].append(f"recheck call failed ({exc}); the first valid answer stands")
                    info["recheck_result"] = "first answer kept: the recheck call failed"
                    info["answer"] = first_valid[1]
                    return first_valid[0], first_valid[1], info
                info["errors"].append(str(exc))
                self._all_failed += 1
                self._consecutive_failures += 1
                # Loud on purpose: a dead key or a dead API plays the whole match code-only, and the runtime log is
                # the only place anyone would see it (the provider's own lines say "trying next provider").
                self._say(f"[agent] !!! ALL MODELS FAILED on this {kind} ({self._all_failed} so far this match): {exc} "
                          "— playing the computed move !!!")
                if self._consecutive_failures >= BREAKER_FAILURES:
                    self._breaker_open = True
                    info["breaker"] = "open"
                    self._say(f"[agent] circuit breaker OPEN after {self._consecutive_failures} consecutive all-model failures: "
                              "the next decision skips the model")
                break  # every model failed; retrying the same question won't help
            if self._consecutive_failures:
                self._say(f"[agent] circuit breaker CLOSED: the model answered again after {self._consecutive_failures} failure(s)")
                info["breaker"] = "closed"
                self._consecutive_failures = 0
            info["model"] = getattr(self._provider, "last_model", None) or getattr(self._provider, "model", None)
            info["latency_ms"] = getattr(self._provider, "last_latency_ms", None)
            info["usage"] = getattr(self._provider, "last_usage", None)
            # Why earlier models in the chain failed (e.g. Fable timed out and Sonnet answered). Empty when the first model answered.
            info["provider_errors"] = list(getattr(self._provider, "last_errors", None) or [])
            record.update(model=info["model"], latency_ms=int((time.monotonic() - call_started) * 1000), outcome="answered",
                          usage=info["usage"], chain=list(getattr(self._provider, "last_attempts", None) or []))
            info["attempt_log"].append(record)
            info["model_latency_ms"] += record["latency_ms"]
            time_left = deadline - (time.monotonic() - started)
            try:
                value = choice.build(answer)
            except InvalidChoice as exc:
                record["outcome"] = "invalid"
                info["errors"].append(f"invalid answer: {exc}")
                if first_valid is not None:
                    info["recheck_result"] = "first answer kept: the rechecked answer was invalid"
                    info["answer"] = first_valid[1]
                    return first_valid[0], first_valid[1], info
                if time_left < MIN_MODEL_SECONDS:
                    info["errors"].append(f"no time left for a second model call ({time_left:.1f}s of the {deadline:.1f}s deadline); playing the computed move")
                    break
                messages.append({"role": "assistant", "content": json.dumps(answer, default=str)})
                messages.append({"role": "user", "content": f"That answer was invalid: {exc}. Answer again using only the given options."})
                continue
            if recheck is not None and attempt == 1:
                message = recheck(value)
                if message and time_left < MIN_MODEL_SECONDS:
                    info["recheck_skipped"] = f"no time left for a second model call ({time_left:.1f}s of the {deadline:.1f}s deadline); first answer stands"
                    message = None
                if message:
                    info["recheck"] = message
                    first_valid = (value, answer)
                    messages.append({"role": "assistant", "content": json.dumps(answer, default=str)})
                    messages.append({"role": "user", "content": message})
                    continue
            if first_valid is not None:
                info["recheck_result"] = "second answer stands" if answer != first_valid[1] else "the model kept its answer"
            info["answer"] = answer
            return value, answer, info

        info["fallback"] = True
        self.memory.fallbacks += 1
        self._say(f"[agent] {kind}: FALLBACK to the deterministic move ({'; '.join(info['errors'])[:200]})")
        return choice.fallback(), None, info

    def effort_for(self, kind: str) -> str | None:
        """The effort for this decision kind, or None for the provider's own default (``AGENT_EFFORT`` at startup,
        else the code default). Team Preview may differ from a battle turn (per-call parameter, one provider)."""
        return self._efforts.get(kind)

    def _call_model(self, messages: list[dict], kind: str, schema: dict, *, deadline: float, effort: str | None) -> dict:
        """One chain call with the per-call clock and effort. A provider that does not take them (a test fake, the
        starter's OpenAI provider) is called the old way."""
        provider = self._provider
        accepted = self._provider_kwargs
        if accepted is None:
            try:
                params = inspect.signature(provider.complete_structured).parameters
                accepted = {"*"} if any(p.kind == p.VAR_KEYWORD for p in params.values()) else set(params)
            except (TypeError, ValueError):
                accepted = set()
            self._provider_kwargs = accepted
        kwargs: dict[str, Any] = {}
        if "*" in accepted or "deadline" in accepted:
            kwargs["deadline"] = deadline
        elif "timeout" in accepted:
            kwargs["timeout"] = deadline
        if ("*" in accepted or "effort" in accepted) and effort is not None:
            kwargs["effort"] = effort
        return provider.complete_structured(messages, kind, schema, **kwargs)

    # -- bookkeeping ------------------------------------------------------------------------

    def _safe_log_for(self, context: DecisionContext) -> "_SafeLog":
        try:
            return _SafeLog(self._log_for(context))
        except Exception as exc:  # noqa: BLE001 — a log that cannot be opened must never cost a move
            self._say(f"[agent] decision log unavailable ({type(exc).__name__}: {exc}); playing on without it")
            return _SafeLog(None)

    def _wall_ms(self) -> int:
        """Wall time of the decision in progress, start to the log row: what the clock actually sees (two-call turns
        used to log only the second call's latency)."""
        return int((time.monotonic() - self._decision_started) * 1000)

    def _log_for(self, context: DecisionContext) -> DecisionLog:
        if self.decision_log is None:
            self.decision_log = DecisionLog(context.session_id, directory=self._log_dir, version=self._version)
            # First line of every match log: what exactly played this match (Official Rules §9 asks for records
            # that tie the submitted commit and configuration to competitive play). No secrets: the provider's
            # repr names models, timeouts and effort only.
            self.decision_log.write("config", commit=git_commit(), agent_version=self._version,
                                    models=repr(self._provider) if self._provider is not None else "code-only",
                                    effort={"provider_default": getattr(self._provider, "effort", None), **self._efforts,
                                            "AGENT_EFFORT": os.environ.get("AGENT_EFFORT") or None},
                                    clock={"decision_s": DECISION_SECONDS, "preview_s": PREVIEW_DECISION_SECONDS, "bank_s": BANK_SECONDS,
                                           "safety_s": DECISION_SAFETY_SECONDS, "min_model_s": MIN_MODEL_SECONDS, "cap": MODEL_DEADLINE_CAP},
                                    params=self.params, python=sys.version.split()[0],
                                    anthropic_sdk=_anthropic_sdk_version(),
                                    game_config=self._absorb_game_config(getattr(context, "game_config", None)))
        return self.decision_log

    def _absorb_game_config(self, config: Any) -> dict:
        """What ``get_game_config`` gave us, described (top-level keys and sizes, never the payload), and any card
        list in it fed to the match pool: the authorized interface would close the "card never seen before" gap. The
        runner fetches it for real-time games only, so for a turn-based match it is normally absent: say so."""
        if not isinstance(config, dict):
            return {"present": False, "note": "absent (the runner fetches get_game_config for real-time games only)"}
        described = {"present": True, "keys": {}}
        try:
            for key, value in config.items():
                size = len(value) if isinstance(value, (dict, list, str)) else 1
                described["keys"][str(key)] = size
            cards = _find_card_list(config)
            if cards:
                added = 0
                for card in cards:
                    if card.get("card_id") and card["card_id"] not in self.memory.pool_cards:
                        self.memory.pool_cards[card["card_id"]] = dict(card)
                        added += 1
                described["cards"] = len(cards)
                described["cards_added_to_pool"] = added
                self._say(f"[agent] game_config carries {len(cards)} cards; {added} new ones feed this match's pool")
            else:
                described["cards"] = 0
        except Exception as exc:  # noqa: BLE001 — a description must never cost the first decision
            described["error"] = f"{type(exc).__name__}: {exc}"
        return described

    def _capture(self, state: GameState, context: DecisionContext) -> None:
        """Save the raw server state (for test fixtures) when AGENT_CAPTURE_DIR is set."""
        if not self._capture_dir:
            return
        try:
            folder = Path(self._capture_dir) / context.session_id
            folder.mkdir(parents=True, exist_ok=True)
            phase = str(getattr(state, "phase", None) or "state").replace("/", "_")
            (folder / f"{phase}-{state.state_version:05d}-seat{context.seat_position}.json").write_text(
                json.dumps(state.raw, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
            )
        except (OSError, TypeError, ValueError):
            pass


_REASON_GLUED_JUNK = re.compile(r"(?<=[.!?])[a-z]{1,10}\.$")  # "...this turn.ewhere." -> "...this turn."
# A sentence end: . ! or ? followed by whitespace, the end, or stray JSON/quote debris. A decimal point ("53.3]%",
# "92.1%", "v2.1") is followed by a digit and never matches: code-only reasons are full of damage ranges.
_REASON_SENTENCE_END = re.compile(r"""[.!?](?=[\s"'\]\[}{)]|$)""")
_REASON_FRAGMENT_LIMIT = 24  # a tail this short after the last full sentence is debris ("'] Hmm", "}"), not a sentence
_BRACKET_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))


def _balanced(text: str) -> bool:
    return all(text.count(open_) == text.count(close) for open_, close in _BRACKET_PAIRS)


def _strip_debris(text: str) -> str:
    """Trailing whitespace, unmatched closing brackets/braces and odd quotes: the tail of the model's JSON, never a
    balanced "(KO at 92.1%)" or a closing quote that has its opening one."""
    s = text.rstrip()
    while s:
        last = s[-1]
        if last in (")", "]", "}") and any(last == close and s.count(open_) < s.count(close) for open_, close in _BRACKET_PAIRS):
            s = s[:-1].rstrip()
        elif last in ('"', "'") and s.count(last) % 2 == 1:
            s = s[:-1].rstrip()
        else:
            break
    return s


def clean_reason(text: Any, limit: int = REASONING_CHAR_LIMIT) -> str:
    """The public reasoning_summary, cleaned: the model's JSON is parsed by the provider but its text sometimes carries
    the tail of the structured answer (4 of 19 answers in match bcbaabe2: ".ewhere.", ".]", ".)", a trailing newline).
    Strips trailing non-sentence debris and whitespace, and ends at the last full sentence within ``limit`` chars,
    keeping brackets balanced. Only the public text changes; the action is untouched."""
    s = _strip_debris(re.sub(r"\s+", " ", str(text or "")).strip())
    s = _REASON_GLUED_JUNK.sub("", s)
    if len(s) > limit:
        cut = s[:limit]
        ends = [m.end() for m in _REASON_SENTENCE_END.finditer(cut) if _balanced(cut[:m.end()])]
        if ends:
            s = cut[:ends[-1]]
        else:
            words = cut.rsplit(" ", 1)[0].rstrip(",;:")
            while words and " " in words and not _balanced(words):  # never cut inside a bracket
                words = words.rsplit(" ", 1)[0].rstrip(",;:")
            s = (words if words and _balanced(words) else cut.rsplit(" ", 1)[0]) + "…"
    if s and s[-1] not in ".!?":
        ends = [m.end() for m in _REASON_SENTENCE_END.finditer(s)]
        if ends and len(s) - ends[-1] <= _REASON_FRAGMENT_LIMIT and _balanced(s[:ends[-1]]):
            s = s[:ends[-1]]
    return _strip_debris(s)


class _SafeLog:
    """A decision log that cannot raise. ``DecisionLog`` already swallows OSError; this covers everything else
    (a log object a harness supplies, a serializer bug) and the absence of a log altogether."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def write(self, kind: str, **fields: Any) -> None:
        if self.inner is None:
            return
        try:
            self.inner.write(kind, **fields)
        except Exception:  # noqa: BLE001
            pass


def _minimal_slot(options: Any, taken: str | None) -> dict:
    """The simplest legal choice for one doubles slot from its raw options: the first move (into the first opposing
    target it lists), else the first switch the other slot did not take, else pass."""
    options = [o for o in options if isinstance(o, dict)] if isinstance(options, list) else []
    for option in options:
        if option.get("type") == "move" and isinstance(option.get("move_id"), str) and option["move_id"]:
            choice: dict[str, Any] = {"type": "move", "move_id": option["move_id"]}
            targets = [t for t in (option.get("targets") or []) if isinstance(t, int) and not isinstance(t, bool)] \
                if isinstance(option.get("targets"), list) else []
            if targets:
                choice["target"] = next((t for t in targets if t > 0), targets[0])
            return choice
    for option in options:
        if option.get("type") == "switch" and isinstance(option.get("species"), str) and option["species"] and option["species"] != taken:
            return {"type": "switch", "species": option["species"]}
    return {"type": "pass"}


def minimal_legal_payload(state: GameState) -> Any:
    """A legal answer built by hand from the raw request, for when even the smoke agent cannot read it: a template
    with a slot the smoke agent rejects still has a first legal move in the other slot. Doubles: per slot the first
    move (first opposing target), else a switch, else pass. Team Preview: the first four of the roster, leading with
    the first two. Anything else (a draft pick): the first legal action. Raises only when nothing legal exists."""
    legal = list(getattr(state, "legal_actions", None) or [])
    if not legal:
        raise ValueError("the request offers no legal action")
    first = legal[0]
    action_id = getattr(first, "action_id", None)
    template = (getattr(first, "input", None) or {}).get("action") if isinstance(getattr(first, "input", None), dict) else None
    template = template if isinstance(template, dict) else {}
    if action_id == "doubles_turn":
        by_number: dict[int, Any] = {}
        for index, slot in enumerate(template.get("slots") if isinstance(template.get("slots"), list) else []):
            if isinstance(slot, dict):
                number = slot.get("slot")
                by_number.setdefault(number if number in (0, 1) else index, slot.get("options"))
        payload: dict[str, Any] = {"type": "doubles_turn"}
        taken = None
        for number in (0, 1):
            choice = _minimal_slot(by_number.get(number), taken)
            if choice["type"] == "switch":
                taken = choice["species"]
            payload[f"slot_{number}"] = choice
        return payload
    if action_id == "select_lineup":
        roster = [s for s in dict.fromkeys(template.get("roster") or []) if isinstance(s, str) and s] \
            if isinstance(template.get("roster"), list) else []
        if len(roster) < 4:
            obs = observation_dict(state)
            for entry in obs.get("your_roster") or []:
                name = entry.get("species") if isinstance(entry, dict) else entry
                if isinstance(name, str) and name and name not in roster:
                    roster.append(name)
        if len(roster) < 4:
            raise ValueError(f"no roster of four to choose from: {roster!r}")
        return {"type": "select_lineup", "bring": roster[:4], "leads": roster[:2]}
    return first


def _find_card_list(config: Any, depth: int = 0) -> list[dict]:
    """The first list of card-shaped dicts (card_id + species + moves) anywhere in a game config, two levels deep."""
    if depth > 3:
        return []
    if isinstance(config, list):
        if config and all(isinstance(c, dict) and c.get("card_id") and c.get("species") and c.get("moves") for c in config[:3]):
            return [c for c in config if isinstance(c, dict)]
        for item in config:
            found = _find_card_list(item, depth + 1)
            if found:
                return found
    elif isinstance(config, dict):
        for value in config.values():
            found = _find_card_list(value, depth + 1)
            if found:
                return found
    return []


def _anthropic_sdk_version() -> str | None:
    try:
        import anthropic

        return getattr(anthropic, "__version__", None)
    except ImportError:
        return None


def _stderr(line: str) -> None:
    """Agent chatter goes to stderr so the runtime's own stdout stays clean."""
    print(line, file=sys.stderr, flush=True)


def _hp_pct(summary: dict) -> int | None:
    frac = summary.get("current_hp_fraction")
    if frac is None and summary.get("max_hp"):
        frac = (summary.get("current_hp") or 0) / summary["max_hp"]
    return int(round(100 * float(frac))) if frac is not None else None


def battle_roster(obs: dict, template: dict, memory: MatchMemory) -> dict:
    """Who is where, in plain words, for both sides.

    The live ``team`` lists all SIX of our drafted Pokémon, including the two left at Team Preview, and can even mark
    one of those ``active`` (Oct 7: the model planned around "Baxcalibur/Lucario can come in" — neither was brought).
    Ours: on the field (by slot), in the back (can switch in), fainted, and NOT in this battle. Theirs: on the field,
    seen in the back, fainted, and how many of their four we have not seen yet (and which species they could be)."""
    key = species_key
    team = obs.get("team") if isinstance(obs.get("team"), dict) else {}
    by_key = {key(s.get("species")): s for s in team.values() if isinstance(s, dict) and s.get("species")}
    on_field: list[dict] = []
    for slot in sorted(template.get("slots") or [], key=lambda s: s.get("slot", 0)):
        active = slot.get("active")
        species = active.get("species") if isinstance(active, dict) else active
        if species and not (by_key.get(key(species)) or {}).get("fainted"):
            on_field.append({"slot": slot.get("slot", 0), "species": key(species), "hp_pct": _hp_pct(by_key.get(key(species)) or {})})
    field_keys = {m["species"] for m in on_field}
    fainted = sorted(k for k, s in by_key.items() if s.get("fainted"))
    switchable = {key(o.get("species")) for slot in template.get("slots") or [] for o in slot.get("options") or []
                  if o.get("type") == "switch" and o.get("species")}
    for options in obs.get("available_switches") or []:
        for o in options or []:
            if isinstance(o, dict) and o.get("species"):
                switchable.add(key(o["species"]))
    # The battle state is the authority: only brought Pokémon can be on the field, offered as a switch, or faint. Our
    # remembered lineup only fills in a bench the server is not offering this turn (a trapped slot), and only when it
    # agrees with what the battle shows (a restart, or a lineup the server replaced, would otherwise mislead).
    brought = {key(s) for s in memory.my_lineup}
    if brought and (field_keys | switchable | set(fainted)) <= brought:
        bench = sorted(brought - field_keys - set(fainted))
    else:
        bench = sorted(switchable - field_keys)
    not_here = sorted(set(by_key) - field_keys - set(bench) - set(fainted))
    ours = {
        "on_field": on_field,
        "in_back_can_switch_in": [{"species": k, "hp_pct": _hp_pct(by_key.get(k) or {})} for k in bench],
        "fainted": fainted,
        "NOT_IN_THIS_BATTLE": not_here,
        "remaining": len(on_field) + len(bench),
        "in_this_battle": sorted(field_keys | set(bench) | set(fainted)),
    }
    opp_team = obs.get("opponent_team") if isinstance(obs.get("opponent_team"), dict) else {}
    seen = {key(s.get("species")): s for s in opp_team.values() if isinstance(s, dict) and s.get("species")}
    opp_field = []
    for summary in obs.get("opponent_active_pokemon") or []:
        if isinstance(summary, dict) and summary.get("species") and not summary.get("fainted"):
            opp_field.append({"species": key(summary["species"]), "hp_pct": _hp_pct(summary)})
    opp_field_keys = {m["species"] for m in opp_field}
    opp_fainted = sorted(k for k, s in seen.items() if s.get("fainted"))
    opp_back = sorted(k for k in seen if k not in opp_field_keys and k not in opp_fainted)
    unseen = max(0, 4 - len(seen))
    could_be = sorted(k for k in memory.opp_cards if k not in seen) if unseen else []
    status = {k: "on_field" for k in opp_field_keys}
    status.update({k: "in_back" for k in opp_back})
    status.update({k: "fainted" for k in opp_fainted})
    for k in memory.opp_cards:
        status.setdefault(k, "not_seen_yet (may be in their back)" if unseen else "NOT_IN_THIS_BATTLE")
    theirs = {
        "on_field": opp_field,
        "seen_in_back": [{"species": k, "hp_pct": _hp_pct(seen[k])} for k in opp_back],
        "fainted": opp_fainted,
        "unseen_count": unseen,
        "unseen_could_be": could_be,
        "remaining": len(opp_field) + len(opp_back) + unseen,
        "status": status,
    }
    return {"ours": ours, "theirs": theirs}


def _battle_view(obs: dict, roster: dict, memory: MatchMemory | None = None) -> dict:
    """The observation as the model sees it.

    - Our ``team`` without the two Pokémon left at Team Preview, with ``active`` matching who is actually on the field
      (the live flag can be wrong for benched entries).
    - Opponent entries carry the server's partial view (moves used so far, "unknown_item", ability null). Left as is,
      "Dragonite moves: [tailwind]" reads like Dragonite only has Tailwind. Fill them from the drafted card and keep
      what the battle has shown as ``revealed_moves``."""
    out = dict(obs)
    team = obs.get("team")
    if isinstance(team, dict):
        out_keys = set(roster["ours"]["NOT_IN_THIS_BATTLE"])
        field_keys = {m["species"] for m in roster["ours"]["on_field"]}
        view = {}
        for name, summary in team.items():
            if not isinstance(summary, dict) or species_key(summary.get("species")) in out_keys:
                continue
            view[name] = {**summary, "active": species_key(summary.get("species")) in field_keys}
        out["team"] = view
    if memory is not None:
        lost = set(memory.opp_items_lost)

        def fill(summary: dict) -> dict:
            if not isinstance(summary, dict):
                return summary
            key = species_key(summary.get("species"))
            card = memory.opp_cards.get(key) or {}
            if not card.get("moves"):
                return summary
            item = summary.get("item")
            if key in lost or (item is None and "item" in summary and key in lost):
                item_view = "none (used up or knocked off)"
            elif item in (None, "", "unknown_item"):
                item_view = f"{card.get('item')} (from its card; not revealed yet)"
            else:
                item_view = item
            return {**summary, "item": item_view, "ability": summary.get("ability") or card.get("ability"),
                    "moves": list(card.get("moves") or []), "revealed_moves": list(summary.get("moves") or [])}

        if isinstance(obs.get("opponent_team"), dict):
            out["opponent_team"] = {k: fill(v) for k, v in obs["opponent_team"].items()}
        if isinstance(obs.get("opponent_active_pokemon"), list):
            out["opponent_active_pokemon"] = [fill(v) for v in obs["opponent_active_pokemon"]]
    return out


def _preview_view(obs: dict) -> dict:
    """Team Preview: the opponent roster entries are placeholders (moves [], item "unknown_item", HP 0/0). Their real
    sets are in known_sets; show only species, types and base stats here so the two never contradict."""
    out = dict(obs)
    if isinstance(obs.get("opponent_roster"), list):
        out["opponent_roster"] = [{k: e.get(k) for k in ("species", "name", "types", "base_stats") if k in e}
                                  for e in obs["opponent_roster"] if isinstance(e, dict)]
    return out


def _compact(obs: dict, keys: tuple[str, ...]) -> Any:
    data = {k: obs[k] for k in keys if k in obs}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) > OBSERVATION_CHAR_LIMIT:
        return text[:OBSERVATION_CHAR_LIMIT] + "...(truncated)"
    return data


def _payload(decision: Any) -> Any:
    if isinstance(decision, LegalAction):
        return {"action_id": decision.action_id, "label": decision.label}
    return decision


def create_agent() -> PokemonAgent:
    load_dotenv()  # reads .env; never overrides variables already set in the shell
    provider: LLMProvider | None = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from agent.llm.anthropic_provider import provider_from_env

        provider = provider_from_env(log=_stderr)
        _stderr(f"[agent] model chain: {provider!r}")
    else:
        _stderr("[agent] WARNING: ANTHROPIC_API_KEY is not set — running in code-only mode (no model calls).")
    agent = PokemonAgent(provider)
    # The runtime runs one worker process per match: when it exits, the match is over. The summary line also comes
    # from on_action_result when our own move ended the game; this catches a game the opponent's move ended.
    atexit.register(agent.print_match_summary)
    return agent
