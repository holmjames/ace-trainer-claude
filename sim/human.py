"""A human seat for the local simulator: you play against the agent from the terminal.

    python sim/harness.py --games 1 --p1 human --p2 fable --verbose      # you vs the Opus-judged agent
    python sim/harness.py --games 1 --p1 human --p2 code                 # you vs the code brain (free)

At every decision it prints the position and the numbered options and waits for input. Type ``h`` on a battle
turn to see what the code brain would do (its warnings and top candidates), so you can learn the sheet the model
reads. Blank input or end-of-input (no terminal attached) plays the first legal choice, which keeps automated
tests from hanging.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from agent.pokemon import battle, data
from agent.pokemon.memory import MatchMemory, observation_dict, species_key


def _set_line(card: dict) -> str:
    return f"{card.get('species')} @ {card.get('item')} | {card.get('ability')} | {card.get('nature')} {card.get('evs') or ''} | {', '.join(card.get('moves') or [])}"


class HumanPlayer:
    name = "human"
    _version = "human"

    def __init__(self, read: Callable[[str], str] | None = None, write: Callable[[str], None] | None = None) -> None:
        self.read = read or (lambda prompt: input(prompt))
        self.write = write or (lambda text: print(text, flush=True))
        self.memory = MatchMemory()

    # -- helpers --------------------------------------------------------------------------------

    def _ask(self, prompt: str) -> str:
        try:
            return (self.read(prompt) or "").strip()
        except EOFError:
            return ""

    def _ints(self, text: str) -> list[int]:
        out = []
        for tok in text.replace(",", " ").split():
            try:
                out.append(int(tok))
            except ValueError:
                pass
        return out

    # -- phases ----------------------------------------------------------------------------------

    def choose_action(self, state, context):
        obs = observation_dict(state)
        first = state.legal_actions[0]
        if first.action_id.startswith("draft_pick:"):
            self.memory.observe_draft(obs, my_turn=True, agent_id=context.agent_id)
            return self._draft(state, obs)
        if first.action_id == "select_lineup":
            self.memory.observe_team_preview(obs)
            return self._lineup(state, obs)
        self.memory.observe_battle(obs)
        return self._turn(state, obs, context)

    def _draft(self, state, obs: dict):
        legal = {a.action_id.split(":", 1)[1]: a for a in state.legal_actions}
        cards = [c for c in obs.get("available_cards") or [] if c.get("card_id") in legal]
        self.write(f"\n=== DRAFT pick {len(obs.get('picks') or []) + 1} of 12 ===")
        for seat, roster in (obs.get("rosters") or {}).items():
            names = [c.get("species") for c in roster]
            self.write(f"  {'YOU ' if seat == obs.get('current_seat') else 'THEM'}: {names}")
        for i, c in enumerate(cards, 1):
            self.write(f"  {i:2d}. {_set_line(c)}")
        while True:
            picked = self._ints(self._ask("pick # (blank = first): "))
            if not picked:
                self.memory.record_pick(cards[0]["card_id"])
                return legal[cards[0]["card_id"]]
            if 1 <= picked[0] <= len(cards):
                self.memory.record_pick(cards[picked[0] - 1]["card_id"])
                return legal[cards[picked[0] - 1]["card_id"]]
            self.write("  not a listed number")

    def _lineup(self, state, obs: dict):
        roster = list(state.legal_actions[0].input["action"]["roster"])
        self.write("\n=== TEAM PREVIEW: bring 4, the first two you list lead ===")
        self.write("  YOUR six:")
        for i, sp in enumerate(roster, 1):
            card = self.memory.my_cards.get(species_key(sp)) or {"species": sp}
            self.write(f"  {i}. {_set_line(card)}")
        self.write("  THEIR six:")
        for c in self.memory.opp_cards.values():
            self.write(f"     {_set_line(c)}")
        while True:
            nums = self._ints(self._ask("four numbers, leads first (e.g. 2 5 1 4): "))
            if not nums:
                nums = [1, 2, 3, 4]
            if len(nums) == 4 and len(set(nums)) == 4 and all(1 <= n <= len(roster) for n in nums):
                bring = [roster[n - 1] for n in nums]
                return {"type": "select_lineup", "bring": bring, "leads": bring[:2]}
            self.write("  need four different numbers from the list")

    def _turn(self, state, obs: dict, context):
        from examples.llm.pokemon import doubles_choice

        first = state.legal_actions[0]
        template = first.input["action"]
        base = doubles_choice(first, state)
        self.write(f"\n=== TURN {obs.get('turn')} ===  weather={obs.get('weather')} fields={obs.get('fields')} "
                   f"our side={obs.get('side_conditions')} their side={obs.get('opponent_side_conditions')}")

        def mon_line(m: dict) -> str:
            hp = int(round(100 * (m.get("current_hp_fraction") or 0)))
            boosts = {k: v for k, v in (m.get("boosts") or {}).items() if v}
            return f"{m.get('name') or m.get('species')} {hp}%{' ' + str(m.get('status')) if m.get('status') else ''}{' ' + str(boosts) if boosts else ''}"

        self.write("  THEM: " + " | ".join(mon_line(m) for m in obs.get("opponent_active_pokemon") or []))
        bench_them = [m for m in (obs.get("opponent_team") or {}).values() if not m.get("active") and not m.get("fainted")]
        if bench_them:
            self.write("        bench: " + ", ".join(mon_line(m) for m in bench_them))
        self.write("  YOU:  " + " | ".join(mon_line(m) for m in obs.get("active_pokemon") or []))
        for slot in template.get("slots") or []:
            self.write(f"  slot {slot.get('slot')} ({slot.get('active')}){' FORCED SWITCH' if slot.get('force_switch') else ''}:")
            for i, opt in enumerate(slot.get("options") or []):
                if opt.get("type") == "move":
                    tgts = ", ".join(f"{t['target']}={t['species']}" for t in opt.get("target_options") or []) or "no target"
                    self.write(f"     {i}. {opt['move_id']} ({opt.get('move_type', '').lower()} {opt.get('category', '').lower()} {opt.get('base_power') or '-'} bp) -> {tgts}")
                elif opt.get("type") == "switch":
                    self.write(f"     {i}. switch -> {opt.get('species')}")
                else:
                    self.write(f"     {i}. pass")
        while True:
            raw = self._ask("per slot 'option [target]', separated by ';'  (h = hint, blank = first legal): ")
            if raw.lower() == "h":
                self._hint(template, obs)
                continue
            answer: dict[str, Any] = {}
            parts = [p for p in raw.split(";")] if raw else []
            for idx, slot in enumerate(template.get("slots") or []):
                nums = self._ints(parts[idx]) if idx < len(parts) else []
                opts = slot.get("options") or []
                option = nums[0] if nums and 0 <= nums[0] < len(opts) else 0
                targets = opts[option].get("targets") or [] if opts else []
                target = nums[1] if len(nums) > 1 else (targets[0] if targets else 0)
                answer[f"slot_{slot.get('slot')}"] = {"option": option, "target": target}
            try:
                return base.build(answer)
            except Exception as exc:  # noqa: BLE001 - show the validator's message and ask again
                self.write(f"  invalid: {exc}")
                if not raw:
                    return base.fallback()

    def _hint(self, template: dict, obs: dict) -> None:
        sheet = battle.build_sheet(template, obs, self.memory)
        self.write("  --- code brain ---")
        self.write("  speed: " + ", ".join(f"{s['species']} {s['speed']}" for s in sheet.speed_order))
        for w in sheet.warnings:
            self.write("  " + w)
        for n in sheet.notes[:6]:
            self.write("  note: " + n)
        for c in sheet.candidates[:3]:
            self.write(f"  {c['score']:7.1f}  {c['name']}: {c['why']}")
