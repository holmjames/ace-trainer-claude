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
  saw, what it answered, which model answered, and how long it took.

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
from agent.pokemon.log import DecisionLog
from agent.pokemon.memory import MatchMemory, observation_dict
from agent.pokemon.prompts import LINEUP_INSTRUCTIONS, SYSTEM_PROMPT

VERSION = os.environ.get("AGENT_VERSION") or "m3"
MAX_ATTEMPTS = 2  # first answer + one retry carrying the validation error
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
_TURN_SCHEMA = object_schema({"slot_0": _SLOT_SCHEMA, "slot_1": _SLOT_SCHEMA})


class PokemonAgent:
    def __init__(
        self,
        provider: LLMProvider | None,
        *,
        version: str = VERSION,
        log_dir: str | os.PathLike | None = None,
        capture_dir: str | os.PathLike | None = None,
        log=None,
    ) -> None:
        self._provider = provider  # None = code-only mode (no key, or tests)
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
        pick = draft_rules.choose_pick(state.legal_actions, self.memory)
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
        candidates = lineup_rules.shortlist(self.memory.my_cards, self.memory.opp_cards, roster)
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
            "observation": _compact(obs, ("your_roster", "opponent_roster", "battle_format")),
        }
        value, answer, info = self._ask(choice, payload, kind="lineup")
        if isinstance(value, dict):
            self.memory.my_lineup = list(value.get("bring") or [])
        log.write("lineup", state_version=state.state_version, payload=_payload(value),
                  candidates=payload["computed_candidates"], **info)
        summary = (answer or {}).get("reasoning_summary") or f"Bringing {value.get('bring')}, leading {value.get('leads')}."
        return value, str(summary)

    # -- battle turn: code computes (M3), Claude judges, adapter validates ---------------

    def _turn(self, action: LegalAction, state: GameState, obs: dict, log: DecisionLog):
        base = pokemon_adapters.doubles_choice(action, state)
        template = action.input.get("action") or {}
        sheet = battle_rules.build_sheet(template, obs, self.memory)
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
        payload = {
            "decision": "doubles_turn",
            **choice.prompt,
            "answer_format": (
                "slot_0 and slot_1 each need {option: <index from that slot's options>, target: <integer>}. "
                "If the chosen option's targets list is non-empty, target must be one of those integers; "
                "if the list is empty (or the option is a switch/pass), send target 0."
            ),
            "turn_sheet": sheet.as_prompt(),
            "known_sets": {"mine": self.memory.known_sets("mine"), "opponent": self.memory.known_sets("opponent")},
            "observation": _compact(obs, _BATTLE_KEYS),
        }
        value, answer, info = self._ask(choice, payload, kind="turn")
        self.memory.record_turn(turn=obs.get("turn"), payload=_payload(value), model=info.get("model"))
        log.write("turn", state_version=state.state_version, turn=obs.get("turn"), payload=_payload(value),
                  turn_sheet=sheet.as_prompt(), **info)
        summary = (answer or {}).get("reasoning_summary") or (sheet.candidates[0]["why"] if sheet.candidates else "Played the computed default turn.")
        return value, str(summary)

    # -- the model call, with retry-once and deterministic fallback ---------------------

    def _ask(self, choice: Choice, payload: dict, *, kind: str) -> tuple[Any, dict | None, dict]:
        info: dict[str, Any] = {"model": None, "latency_ms": None, "attempts": 0, "errors": [], "fallback": False}
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
            try:
                value = choice.build(answer)
            except InvalidChoice as exc:
                info["errors"].append(f"invalid answer: {exc}")
                messages.append({"role": "assistant", "content": json.dumps(answer, default=str)})
                messages.append({"role": "user", "content": f"That answer was invalid: {exc}. Answer again using only the given options."})
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
        return self.decision_log

    def _capture(self, state: GameState, context: DecisionContext) -> None:
        """Save the raw server state (for test fixtures) when AGENT_CAPTURE_DIR is set."""
        if not self._capture_dir:
            return
        try:
            folder = Path(self._capture_dir) / context.session_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{state.state_version:05d}.json").write_text(
                json.dumps(state.raw, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
            )
        except (OSError, TypeError, ValueError):
            pass


def _stderr(line: str) -> None:
    """Agent chatter goes to stderr so the runtime's own stdout stays clean."""
    print(line, file=sys.stderr, flush=True)


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
