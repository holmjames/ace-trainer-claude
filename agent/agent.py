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

import json
import os
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
from agent.pokemon import draft as draft_rules
from agent.pokemon import lineup as lineup_rules
from agent.pokemon.log import DecisionLog, git_commit
from agent.pokemon.memory import MatchMemory, observation_dict, species_key
from agent.pokemon.prompts import LINEUP_INSTRUCTIONS, SYSTEM_PROMPT
from agent.pokemon.tuning import DEFAULTS

VERSION = os.environ.get("AGENT_VERSION") or "m3"
MAX_ATTEMPTS = 2  # first answer + one retry carrying the validation error
SECOND_CALL_CUTOFF_SECONDS = 10.0  # a retry/recheck after this many seconds could push a decision past Showdown's 55 s clock
MIN_DECISION_SECONDS_FOR_MODEL = 25.0  # below this the model chain (worst case ~30 s) could run the decision out: play the computed move
MIN_BANK_SECONDS_FOR_MODEL = 90.0      # a nearly empty bank forfeits the battle; stop spending it on model calls


def _clock_budget(clock: dict | None) -> tuple[float | None, float | None, float]:
    """(decision seconds left, bank seconds left, cutoff for a second model call) from the server's clock, if any."""
    decision_left = bank_left = None
    if isinstance(clock, dict):
        try:
            decision_left = float(clock["decision_seconds_left"]) if clock.get("decision_seconds_left") is not None else None
            bank_left = float(clock["bank_seconds_left"]) if clock.get("bank_seconds_left") is not None else None
        except (TypeError, ValueError):
            decision_left = bank_left = None
    cutoff = SECOND_CALL_CUTOFF_SECONDS
    if decision_left is not None:
        # a second call must still finish inside the decision: leave the chain's worst case (~30 s) after it starts
        cutoff = max(0.0, min(cutoff, decision_left - MIN_DECISION_SECONDS_FOR_MODEL - 5.0))
    return decision_left, bank_left, cutoff
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


def _lethal_recheck(value: Any, sheet: Any) -> str | None:
    """A valid answer that leaves a LETHAL-flagged slot attacking gets one explicit second look. The model keeps the
    final say (trades can be right), but it has to make that call with the warning in front of it."""
    payload = value if isinstance(value, dict) else getattr(value, "payload", None)
    if not isinstance(payload, dict):
        return None
    lethal = [w for w in getattr(sheet, "warnings", []) if str(w).startswith("LETHAL:")]
    if not lethal:
        return None
    if getattr(sheet, "alone", False) and not getattr(sheet, "stall_reasons", None):
        return None  # our last Pokémon with nothing to wait for: Protect only delays, so attacking is right (Oct 7)
    problems = []
    for number in (0, 1):
        slot = payload.get(f"slot_{number}") or {}
        if slot.get("type") != "move" or str(slot.get("move_id") or "").replace("-", "").replace(" ", "").lower() in _PROTECT_MOVES:
            continue
        hits = [w for w in lethal if f"slot {number} (" in w]
        if hits:
            problems.append(f"slot {number} attacks with {slot.get('move_id')} although: {hits[0]}")
    if not problems:
        return None
    return ("Recheck before this is final. " + " | ".join(problems) +
            " A Pokémon that is knocked out before it moves never gets its attack off. Keep your answer only if the other slot "
            "removes that threat first (priority, or faster with a guaranteed KO) or the trade wins the game; otherwise Protect "
            "or switch that slot. Answer again; reasoning_summary stays a one-sentence public description of the move itself.")


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

    # -- the one method the runtime calls ----------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        log = self._log_for(context)
        started = time.monotonic()
        try:
            self._capture(state, context)
            decision, summary = self._decide(state, context, log)
        except Exception as exc:  # noqa: BLE001 — a bug must never forfeit the match
            self.memory.fallbacks += 1
            log.write(
                "error",
                phase=state.phase,
                state_version=state.state_version,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-3000:],
            )
            self._print(f"[agent] ERROR in decision logic ({type(exc).__name__}); playing the safe fallback move")
            fallback = smoke_agent.choose_action(state, context)
            log.write("fallback", phase=state.phase, payload=_payload(fallback), reason="exception in decision logic")
            return WithReasoning(fallback, "Fallback: played a safe legal move after an internal error.")
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self.memory.latencies_ms.append(elapsed_ms)
        self._print(f"[agent] {state.phase}: {summary} ({elapsed_ms} ms)")
        return WithReasoning(decision, summary[:REASONING_CHAR_LIMIT])

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
        value, answer, info = self._ask(choice, payload, kind="lineup", clock=obs.get("clock"))
        if isinstance(value, dict):
            self.memory.my_lineup = list(value.get("bring") or [])
        log.write("lineup", state_version=state.state_version, payload=_payload(value),
                  candidates=payload["computed_candidates"], known_sets=payload["known_sets"], **info)
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
        value, answer, info = self._ask(choice, payload, kind="turn", recheck=lambda v: _lethal_recheck(v, sheet),
                                        clock=obs.get("clock"))
        forced = any(slot.get("force_switch") for slot in template.get("slots") or [])
        self.memory.record_turn(turn=obs.get("turn"), payload=_payload(value), model=info.get("model"), forced=forced)
        log.write("turn", state_version=state.state_version, turn=obs.get("turn"), payload=_payload(value),
                  win_condition=(answer or {}).get("win_condition"), battle_roster=roster, turn_sheet=sheet.as_prompt(), **info)
        summary = (answer or {}).get("reasoning_summary") or (sheet.candidates[0]["why"] if sheet.candidates else "Played the computed default turn.")
        return value, str(summary)

    # -- the model call, with retry-once and deterministic fallback ---------------------

    def _ask(self, choice: Choice, payload: dict, *, kind: str, recheck: Any = None,
             clock: dict | None = None) -> tuple[Any, dict | None, dict]:
        """Ask the model, retry once on an invalid answer, fall back to the computed move if it still fails.
        ``recheck(value)`` may return a message for a valid-but-suspicious first answer (e.g. attacking with a slot the
        sheet marks LETHAL); the model is then asked once more with that message and its second answer stands.
        Clock: Showdown gives 55 s per battle decision from a 420 s bank, so a second model call (retry or recheck) is
        only made while fewer than SECOND_CALL_CUTOFF_SECONDS have passed; past that the first valid answer stands or
        the computed move is played."""
        info: dict[str, Any] = {"model": None, "latency_ms": None, "attempts": 0, "errors": [], "fallback": False}
        started = time.monotonic()
        # The server's own clock (observation["clock"]): seconds left for this decision and in the battle bank.
        decision_left, bank_left, cutoff = _clock_budget(clock)
        if clock:
            info["clock"] = {"decision_seconds_left": decision_left, "bank_seconds_left": bank_left}
        if self._provider is not None and (decision_left is not None and decision_left < MIN_DECISION_SECONDS_FOR_MODEL
                                           or bank_left is not None and bank_left < MIN_BANK_SECONDS_FOR_MODEL):
            info["errors"].append(f"clock too short for a model call ({decision_left}s left, bank {bank_left}s); playing the computed move")
            info["fallback"] = True
            self.memory.fallbacks += 1
            return choice.fallback(), None, info
        if self._provider is None:
            info["errors"].append("no provider configured (code-only mode)")
            info["fallback"] = True
            self.memory.fallbacks += 1
            return choice.fallback(), None, info

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            info["attempts"] = attempt
            self.memory.llm_calls += 1
            try:
                answer = self._provider.complete_structured(messages, choice.kind, choice.schema)
            except ProviderError as exc:
                info["errors"].append(str(exc))
                break  # every model failed; retrying the same question won't help
            info["model"] = getattr(self._provider, "last_model", None) or getattr(self._provider, "model", None)
            info["latency_ms"] = getattr(self._provider, "last_latency_ms", None)
            info["usage"] = getattr(self._provider, "last_usage", None)
            # Why earlier models in the chain failed (e.g. Fable timed out and Sonnet answered). Empty when the first model answered.
            info["provider_errors"] = list(getattr(self._provider, "last_errors", None) or [])
            try:
                value = choice.build(answer)
            except InvalidChoice as exc:
                info["errors"].append(f"invalid answer: {exc}")
                if time.monotonic() - started > cutoff:
                    info["errors"].append("no time left for a second model call; playing the computed move")
                    break
                messages.append({"role": "assistant", "content": json.dumps(answer, default=str)})
                messages.append({"role": "user", "content": f"That answer was invalid: {exc}. Answer again using only the given options."})
                continue
            if recheck is not None and attempt == 1:
                message = recheck(value)
                if message and time.monotonic() - started > cutoff:
                    info["recheck_skipped"] = "no time left for a second model call; first answer stands"
                    message = None
                if message:
                    info["recheck"] = message
                    messages.append({"role": "assistant", "content": json.dumps(answer, default=str)})
                    messages.append({"role": "user", "content": message})
                    continue
            info["answer"] = answer
            return value, answer, info

        info["fallback"] = True
        self.memory.fallbacks += 1
        self._print(f"[agent] {kind}: FALLBACK to the deterministic move ({'; '.join(info['errors'])[:200]})")
        return choice.fallback(), None, info

    # -- bookkeeping ------------------------------------------------------------------------

    def _log_for(self, context: DecisionContext) -> DecisionLog:
        if self.decision_log is None:
            self.decision_log = DecisionLog(context.session_id, directory=self._log_dir, version=self._version)
            # First line of every match log: what exactly played this match (Official Rules §9 asks for records
            # that tie the submitted commit and configuration to competitive play). No secrets: the provider's
            # repr names models, timeouts and effort only.
            self.decision_log.write("config", commit=git_commit(), agent_version=self._version,
                                    models=repr(self._provider) if self._provider is not None else "code-only",
                                    params=self.params, python=sys.version.split()[0],
                                    anthropic_sdk=_anthropic_sdk_version())
        return self.decision_log

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
    return PokemonAgent(provider)
